# Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing 

**Authors**: Tong Zheng, Chengsong Huang, Runpeng Dai, Yun He, Rui Liu, Xin Ni, Huiwen Bao, Kaishen Wang, Hongtu Zhu, Jiaxin Huang, Furong Huang, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03845)  

**Abstract**: Parallel thinking has emerged as a promising paradigm for reasoning, yet it imposes significant computational burdens. Existing efficiency methods primarily rely on local, per-trajectory signals and lack principled mechanisms to exploit global dynamics across parallel branches. We introduce 2D probing, an interface that exposes the width-depth dynamics of parallel thinking by periodically eliciting intermediate answers from all branches. Our analysis reveals three key insights: non-monotonic scaling across width-depth allocations, heterogeneous reasoning branch lengths, and early stabilization of global consensus. Guided by these insights, we introduce $\textbf{Parallel-Probe}$, a training-free controller designed to optimize online parallel thinking. Parallel-Probe employs consensus-based early stopping to regulate reasoning depth and deviation-based branch pruning to dynamically adjust width. Extensive experiments across three benchmarks and multiple models demonstrate that Parallel-Probe establishes a superior Pareto frontier for test-time scaling. Compared to standard majority voting, it reduces sequential tokens by up to $\textbf{35.8}$% and total token cost by over $\textbf{25.8}$% while maintaining competitive accuracy. 

---
# Accelerating Scientific Research with Gemini: Case Studies and Common Techniques 

**Authors**: David P. Woodruff, Vincent Cohen-Addad, Lalit Jain, Jieming Mao, Song Zuo, MohammadHossein Bateni, Simina Branzei, Michael P. Brenner, Lin Chen, Ying Feng, Lance Fortnow, Gang Fu, Ziyi Guan, Zahra Hadizadeh, Mohammad T. Hajiaghayi, Mahdi JafariRaviz, Adel Javanmard, Karthik C. S., Ken-ichi Kawarabayashi, Ravi Kumar, Silvio Lattanzi, Euiwoong Lee, Yi Li, Ioannis Panageas, Dimitris Paparas, Benjamin Przybocki, Bernardo Subercaseaux, Ola Svensson, Shayan Taherijam, Xuan Wu, Eylon Yogev, Morteza Zadimoghaddam, Samson Zhou, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2602.03837)  

**Abstract**: Recent advances in large language models (LLMs) have opened new avenues for accelerating scientific research. While models are increasingly capable of assisting with routine tasks, their ability to contribute to novel, expert-level mathematical discovery is less understood. We present a collection of case studies demonstrating how researchers have successfully collaborated with advanced AI models, specifically Google's Gemini-based models (in particular Gemini Deep Think and its advanced variants), to solve open problems, refute conjectures, and generate new proofs across diverse areas in theoretical computer science, as well as other areas such as economics, optimization, and physics. Based on these experiences, we extract common techniques for effective human-AI collaboration in theoretical research, such as iterative refinement, problem decomposition, and cross-disciplinary knowledge transfer. While the majority of our results stem from this interactive, conversational methodology, we also highlight specific instances that push beyond standard chat interfaces. These include deploying the model as a rigorous adversarial reviewer to detect subtle flaws in existing proofs, and embedding it within a "neuro-symbolic" loop that autonomously writes and executes code to verify complex derivations. Together, these examples highlight the potential of AI not just as a tool for automation, but as a versatile, genuine partner in the creative process of scientific discovery. 

---
# They Said Memes Were Harmless-We Found the Ones That Hurt: Decoding Jokes, Symbols, and Cultural References 

**Authors**: Sahil Tripathi, Gautam Siddharth Kashyap, Mehwish Nasim, Jian Yang, Jiechao Gao, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2602.03822)  

**Abstract**: Meme-based social abuse detection is challenging because harmful intent often relies on implicit cultural symbolism and subtle cross-modal incongruence. Prior approaches, from fusion-based methods to in-context learning with Large Vision-Language Models (LVLMs), have made progress but remain limited by three factors: i) cultural blindness (missing symbolic context), ii) boundary ambiguity (satire vs. abuse confusion), and iii) lack of interpretability (opaque model reasoning). We introduce CROSS-ALIGN+, a three-stage framework that systematically addresses these limitations: (1) Stage I mitigates cultural blindness by enriching multimodal representations with structured knowledge from ConceptNet, Wikidata, and Hatebase; (2) Stage II reduces boundary ambiguity through parameter-efficient LoRA adapters that sharpen decision boundaries; and (3) Stage III enhances interpretability by generating cascaded explanations. Extensive experiments on five benchmarks and eight LVLMs demonstrate that CROSS-ALIGN+ consistently outperforms state-of-the-art methods, achieving up to 17% relative F1 improvement while providing interpretable justifications for each decision. 

---
# Context Compression via Explicit Information Transmission 

**Authors**: Jiangnan Ye, Hanqi Yan, Zhenyi Shen, Heng Chang, Ye Mao, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2602.03784)  

**Abstract**: Long-context inference with Large Language Models (LLMs) is costly due to quadratic attention and growing key-value caches, motivating context compression. In this work, we study soft context compression, where a long context is condensed into a small set of continuous representations. Existing methods typically re-purpose the LLM itself as a trainable compressor, relying on layer-by-layer self-attention to iteratively aggregate information. We argue that this paradigm suffers from two structural limitations: (i) progressive representation overwriting across layers (ii) uncoordinated allocation of compression capacity across tokens. We propose ComprExIT (Context Compression via Explicit Information Transmission), a lightweight framework that formulates soft compression into a new paradigm: explicit information transmission over frozen LLM hidden states. This decouples compression from the model's internal self-attention dynamics. ComprExIT performs (i) depth-wise transmission to selectively transmit multi-layer information into token anchors, mitigating progressive overwriting, and (ii) width-wise transmission to aggregate anchors into a small number of slots via a globally optimized transmission plan, ensuring coordinated allocation of information. Across six question-answering benchmarks, ComprExIT consistently outperforms state-of-the-art context compression methods while introducing only ~1% additional parameters, demonstrating that explicit and coordinated information transmission enables more effective and robust long-context compression. 

---
# CUBO: Self-Contained Retrieval-Augmented Generation on Consumer Laptops 10 GB Corpora, 16 GB RAM, Single-Device Deployment 

**Authors**: Paolo Astrino  

**Link**: [PDF](https://arxiv.org/pdf/2602.03731)  

**Abstract**: Organizations handling sensitive documents face a tension: cloud-based AI risks GDPR violations, while local systems typically require 18-32 GB RAM. This paper presents CUBO, a systems-oriented RAG platform for consumer laptops with 16 GB shared memory. CUBO's novelty lies in engineering integration of streaming ingestion (O(1) buffer overhead), tiered hybrid retrieval, and hardware-aware orchestration that enables competitive Recall@10 (0.48-0.97 across BEIR domains) within a hard 15.5 GB RAM ceiling. The 37,000-line codebase achieves retrieval latencies of 185 ms (p50) on C1,300 laptops while maintaining data minimization through local-only processing aligned with GDPR Art. 5(1)(c). Evaluation on BEIR benchmarks validates practical deployability for small-to-medium professional archives. The codebase is publicly available at this https URL. 

---
# Training Multi-Turn Search Agent via Contrastive Dynamic Branch Sampling 

**Authors**: Yubao Zhao, Weiquan Huang, Sudong Wang, Ruochen Zhao, Chen Chen, Yao Shu, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2602.03719)  

**Abstract**: Agentic reinforcement learning has enabled large language models to perform complex multi-turn planning and tool use. However, learning in long-horizon settings remains challenging due to sparse, trajectory-level outcome rewards. While prior tree-based methods attempt to mitigate this issue, they often suffer from high variance and computational inefficiency. Through empirical analysis of search agents, We identify a common pattern: performance diverges mainly due to decisions near the tail. Motivated by this observation, we propose Branching Relative Policy Optimization (BranPO), a value-free method that provides step-level contrastive supervision without dense rewards. BranPO truncates trajectories near the tail and resamples alternative continuations to construct contrastive suffixes over shared prefixes, reducing credit ambiguity in long-horizon rollouts. To further boost efficiency and stabilize training, we introduce difficulty-aware branch sampling to adapt branching frequency across tasks, and redundant step masking to suppress uninformative actions. Extensive experiments on various question answering benchmarks demonstrate that BranPO consistently outperforms strong baselines, achieving significant accuracy gains on long-horizon tasks without increasing the overall training budget. Our code is available at \href{this https URL}{code}. 

---
# No Shortcuts to Culture: Indonesian Multi-hop Question Answering for Complex Cultural Understanding 

**Authors**: Vynska Amalia Permadi, Xingwei Tan, Nafise Sadat Moosavi, Nikos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2602.03709)  

**Abstract**: Understanding culture requires reasoning across context, tradition, and implicit social knowledge, far beyond recalling isolated facts. Yet most culturally focused question answering (QA) benchmarks rely on single-hop questions, which may allow models to exploit shallow cues rather than demonstrate genuine cultural reasoning. In this work, we introduce ID-MoCQA, the first large-scale multi-hop QA dataset for assessing the cultural understanding of large language models (LLMs), grounded in Indonesian traditions and available in both English and Indonesian. We present a new framework that systematically transforms single-hop cultural questions into multi-hop reasoning chains spanning six clue types (e.g., commonsense, temporal, geographical). Our multi-stage validation pipeline, combining expert review and LLM-as-a-judge filtering, ensures high-quality question-answer pairs. Our evaluation across state-of-the-art models reveals substantial gaps in cultural reasoning, particularly in tasks requiring nuanced inference. ID-MoCQA provides a challenging and essential benchmark for advancing the cultural competency of LLMs. 

---
# Beyond Tokens: Semantic-Aware Speculative Decoding for Efficient Inference by Probing Internal States 

**Authors**: Ximing Dong, Shaowei Wang, Dayi Lin, Boyuan Chen, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2602.03708)  

**Abstract**: Large Language Models (LLMs) achieve strong performance across many tasks but suffer from high inference latency due to autoregressive decoding. The issue is exacerbated in Large Reasoning Models (LRMs), which generate lengthy chains of thought. While speculative decoding accelerates inference by drafting and verifying multiple tokens in parallel, existing methods operate at the token level and ignore semantic equivalence (i.e., different token sequences expressing the same meaning), leading to inefficient rejections. We propose SemanticSpec, a semantic-aware speculative decoding framework that verifies entire semantic sequences instead of tokens. SemanticSpec introduces a semantic probability estimation mechanism that probes the model's internal hidden states to assess the likelihood of generating sequences with specific this http URL on four benchmarks show that SemanticSpec achieves up to 2.7x speedup on DeepSeekR1-32B and 2.1x on QwQ-32B, consistently outperforming token-level and sequence-level baselines in both efficiency and effectiveness. 

---
# OmniRAG-Agent: Agentic Omnimodal Reasoning for Low-Resource Long Audio-Video Question Answering 

**Authors**: Yifan Zhu, Xinyu Mu, Tao Feng, Zhonghong Ou, Yuning Gong, Haoran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.03707)  

**Abstract**: Long-horizon omnimodal question answering answers questions by reasoning over text, images, audio, and video. Despite recent progress on OmniLLMs, low-resource long audio-video QA still suffers from costly dense encoding, weak fine-grained retrieval, limited proactive planning, and no clear end-to-end this http URL address these issues, we propose OmniRAG-Agent, an agentic omnimodal QA method for budgeted long audio-video reasoning. It builds an image-audio retrieval-augmented generation module that lets an OmniLLM fetch short, relevant frames and audio snippets from external banks. Moreover, it uses an agent loop that plans, calls tools across turns, and merges retrieved evidence to answer complex queries. Furthermore, we apply group relative policy optimization to jointly improve tool use and answer quality over time. Experiments on OmniVideoBench, WorldSense, and Daily-Omni show that OmniRAG-Agent consistently outperforms prior methods under low-resource settings and achieves strong results, with ablations validating each component. 

---
# Cognitively Diverse Multiple-Choice Question Generation: A Hybrid Multi-Agent Framework with Large Language Models 

**Authors**: Yu Tian, Linh Huynh, Katerina Christhilf, Shubham Chakraborty, Micah Watanabe, Tracy Arner, Danielle McNamara  

**Link**: [PDF](https://arxiv.org/pdf/2602.03704)  

**Abstract**: Recent advances in large language models (LLMs) have made automated multiple-choice question (MCQ) generation increasingly feasible; however, reliably producing items that satisfy controlled cognitive demands remains a challenge. To address this gap, we introduce ReQUESTA, a hybrid, multi-agent framework for generating cognitively diverse MCQs that systematically target text-based, inferential, and main idea comprehension. ReQUESTA decomposes MCQ authoring into specialized subtasks and coordinates LLM-powered agents with rule-based components to support planning, controlled generation, iterative evaluation, and post-processing. We evaluated the framework in a large-scale reading comprehension study using academic expository texts, comparing ReQUESTA-generated MCQs with those produced by a single-pass GPT-5 zero-shot baseline. Psychometric analyses of learner responses assessed item difficulty and discrimination, while expert raters evaluated question quality across multiple dimensions, including topic relevance and distractor quality. Results showed that ReQUESTA-generated items were consistently more challenging, more discriminative, and more strongly aligned with overall reading comprehension performance. Expert evaluations further indicated stronger alignment with central concepts and superior distractor linguistic consistency and semantic plausibility, particularly for inferential questions. These findings demonstrate that hybrid, agentic orchestration can systematically improve the reliability and controllability of LLM-based generation, highlighting workflow design as a key lever for structured artifact generation beyond single-pass prompting. 

---
# OCRTurk: A Comprehensive OCR Benchmark for Turkish 

**Authors**: Deniz Yılmaz, Evren Ayberk Munis, Çağrı Toraman, Süha Kağan Köse, Burak Aktaş, Mehmet Can Baytekin, Bilge Kaan Görür  

**Link**: [PDF](https://arxiv.org/pdf/2602.03693)  

**Abstract**: Document parsing is now widely used in applications, such as large-scale document digitization, retrieval-augmented generation, and domain-specific pipelines in healthcare and education. Benchmarking these models is crucial for assessing their reliability and practical robustness. Existing benchmarks mostly target high-resource languages and provide limited coverage for low-resource settings, such as Turkish. Moreover, existing studies on Turkish document parsing lack a standardized benchmark that reflects real-world scenarios and document diversity. To address this gap, we introduce OCRTurk, a Turkish document parsing benchmark covering multiple layout elements and document categories at three difficulty levels. OCRTurk consists of 180 Turkish documents drawn from academic articles, theses, slide decks, and non-academic articles. We evaluate seven OCR models on OCRTurk using element-wise metrics. Across difficulty levels, PaddleOCR achieves the strongest overall results, leading most element-wise metrics except figures and attaining high Normalized Edit Distance scores in easy, medium, and hard subsets. We also observe performance variation by document type. Models perform well on non-academic documents, while slideshows become the most challenging. 

---
# Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation 

**Authors**: Jiashuo Sun, Pengcheng Jiang, Saizhuo Wang, Jiajun Fan, Heng Wang, Siru Ouyang, Ming Zhong, Yizhu Jiao, Chengsong Huang, Xueqiang Xu, Pengrui Han, Peiran Li, Jiaxin Huang, Ge Liu, Heng Ji, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.03689)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems remain brittle under realistic retrieval noise, even when the required evidence appears in the top-K results. A key reason is that retrievers and rerankers optimize solely for relevance, often selecting either trivial, answer-revealing passages or evidence that lacks the critical information required to answer the question, without considering whether the evidence is suitable for the generator. We propose BAR-RAG, which reframes the reranker as a boundary-aware evidence selector that targets the generator's Goldilocks Zone -- evidence that is neither trivially easy nor fundamentally unanswerable for the generator, but is challenging yet sufficient for inference and thus provides the strongest learning signal. BAR-RAG trains the selector with reinforcement learning using generator feedback, and adopts a two-stage pipeline that fine-tunes the generator under the induced evidence distribution to mitigate the distribution mismatch between training and inference. Experiments on knowledge-intensive question answering benchmarks show that BAR-RAG consistently improves end-to-end performance under noisy retrieval, achieving an average gain of 10.3 percent over strong RAG and reranking baselines while substantially improving robustness. Code is publicly avaliable at this https URL. 

---
# Neural Attention Search Linear: Towards Adaptive Token-Level Hybrid Attention Models 

**Authors**: Difan Deng, Andreas Bentzen Winje, Lukas Fehring, Marius Lindauer  

**Link**: [PDF](https://arxiv.org/pdf/2602.03681)  

**Abstract**: The quadratic computational complexity of softmax transformers has become a bottleneck in long-context scenarios. In contrast, linear attention model families provide a promising direction towards a more efficient sequential model. These linear attention models compress past KV values into a single hidden state, thereby efficiently reducing complexity during both training and inference. However, their expressivity remains limited by the size of their hidden state. Previous work proposed interleaving softmax and linear attention layers to reduce computational complexity while preserving expressivity. Nevertheless, the efficiency of these models remains bottlenecked by their softmax attention layers. In this paper, we propose Neural Attention Search Linear (NAtS-L), a framework that applies both linear attention and softmax attention operations within the same layer on different tokens. NAtS-L automatically determines whether a token can be handled by a linear attention model, i.e., tokens that have only short-term impact and can be encoded into fixed-size hidden states, or require softmax attention, i.e., tokens that contain information related to long-term retrieval and need to be preserved for future queries. By searching for optimal Gated DeltaNet and softmax attention combinations across tokens, we show that NAtS-L provides a strong yet efficient token-level hybrid architecture. 

---
# Instruction Anchors: Dissecting the Causal Dynamics of Modality Arbitration 

**Authors**: Yu Zhang, Mufan Xu, Xuefeng Bai, Kehai chen, Pengfei Zhang, Yang Xiang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03677)  

**Abstract**: Modality following serves as the capacity of multimodal large language models (MLLMs) to selectively utilize multimodal contexts based on user instructions. It is fundamental to ensuring safety and reliability in real-world deployments. However, the underlying mechanisms governing this decision-making process remain poorly understood. In this paper, we investigate its working mechanism through an information flow lens. Our findings reveal that instruction tokens function as structural anchors for modality arbitration: Shallow attention layers perform non-selective information transfer, routing multimodal cues to these anchors as a latent buffer; Modality competition is resolved within deep attention layers guided by the instruction intent, while MLP layers exhibit semantic inertia, acting as an adversarial force. Furthermore, we identify a sparse set of specialized attention heads that drive this arbitration. Causal interventions demonstrate that manipulating a mere $5\%$ of these critical heads can decrease the modality-following ratio by $60\%$ through blocking, or increase it by $60\%$ through targeted amplification of failed samples. Our work provides a substantial step toward model transparency and offers a principled framework for the orchestration of multimodal information in MLLMs. 

---
# RAGTurk: Best Practices for Retrieval Augmented Generation in Turkish 

**Authors**: Süha Kağan Köse, Mehmet Can Baytekin, Burak Aktaş, Bilge Kaan Görür, Evren Ayberk Munis, Deniz Yılmaz, Muhammed Yusuf Kartal, Çağrı Toraman  

**Link**: [PDF](https://arxiv.org/pdf/2602.03652)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances LLM factuality, yet design guidance remains English-centric, limiting insights for morphologically rich languages like Turkish. We address this by constructing a comprehensive Turkish RAG dataset derived from Turkish Wikipedia and CulturaX, comprising question-answer pairs and relevant passage chunks. We benchmark seven stages of the RAG pipeline, from query transformation and reranking to answer refinement, without task-specific fine-tuning. Our results show that complex methods like HyDE maximize accuracy (85%) that is considerably higher than the baseline (78.70%). Also a Pareto-optimal configuration using Cross-encoder Reranking and Context Augmentation achieves comparable performance (84.60%) with much lower cost. We further demonstrate that over-stacking generative modules can degrade performance by distorting morphological cues, whereas simple query clarification with robust reranking offers an effective solution. 

---
# TRE: Encouraging Exploration in the Trust Region 

**Authors**: Chao Huang, Yujing Lu, Quangang Li, Shenghe Wang, Yan Wang, Yueyang Zhang, Long Xia, Jiashu Zhao, Zhiyuan Sun, Daiting Shi, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03635)  

**Abstract**: Entropy regularization is a standard technique in reinforcement learning (RL) to enhance exploration, yet it yields negligible effects or even degrades performance in Large Language Models (LLMs). We attribute this failure to the cumulative tail risk inherent to LLMs with massive vocabularies and long generation horizons. In such environments, standard global entropy maximization indiscriminately dilutes probability mass into the vast tail of invalid tokens rather than focusing on plausible candidates, thereby disrupting coherent reasoning. To address this, we propose Trust Region Entropy (TRE), a method that encourages exploration strictly within the model's trust region. Extensive experiments across mathematical reasoning (MATH), combinatorial search (Countdown), and preference alignment (HH) tasks demonstrate that TRE consistently outperforms vanilla PPO, standard entropy regularization, and other exploration baselines. Our code is available at this https URL. 

---
# BIRDTurk: Adaptation of the BIRD Text-to-SQL Dataset to Turkish 

**Authors**: Burak Aktaş, Mehmet Can Baytekin, Süha Kağan Köse, Ömer İlbilgi, Elif Özge Yılmaz, Çağrı Toraman, Bilge Kaan Görür  

**Link**: [PDF](https://arxiv.org/pdf/2602.03633)  

**Abstract**: Text-to-SQL systems have achieved strong performance on English benchmarks, yet their behavior in morphologically rich, low-resource languages remains largely unexplored. We introduce BIRDTurk, the first Turkish adaptation of the BIRD benchmark, constructed through a controlled translation pipeline that adapts schema identifiers to Turkish while strictly preserving the logical structure and execution semantics of SQL queries and databases. Translation quality is validated on a sample size determined by the Central Limit Theorem to ensure 95% confidence, achieving 98.15% accuracy on human-evaluated samples. Using BIRDTurk, we evaluate inference-based prompting, agentic multi-stage reasoning, and supervised fine-tuning. Our results reveal that Turkish introduces consistent performance degradation, driven by both structural linguistic divergence and underrepresentation in LLM pretraining, while agentic reasoning demonstrates stronger cross-lingual robustness. Supervised fine-tuning remains challenging for standard multilingual baselines but scales effectively with modern instruction-tuned models. BIRDTurk provides a controlled testbed for cross-lingual Text-to-SQL evaluation under realistic database conditions. We release the training and development splits to support future research. 

---
# Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation 

**Authors**: Changze Lv, Jie Zhou, Wentao Zhao, Jingwen Xu, Zisu Huang, Muzhao Tian, Shihan Dou, Tao Gui, Le Tian, Xiao Zhou, Xiaoqing Zheng, Xuanjing Huang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.03619)  

**Abstract**: Nowadays, training and evaluating DeepResearch-generated reports remain challenging due to the lack of verifiable reward signals. Accordingly, rubric-based evaluation has become a common practice. However, existing approaches either rely on coarse, pre-defined rubrics that lack sufficient granularity, or depend on manually constructed query-specific rubrics that are costly and difficult to scale. In this paper, we propose a pipeline to train human-preference-aligned query-specific rubric generators tailored for DeepResearch report generation. We first construct a dataset of DeepResearch-style queries annotated with human preferences over paired reports, and train rubric generators via reinforcement learning with a hybrid reward combining human preference supervision and LLM-based rubric evaluation. To better handle long-horizon reasoning, we further introduce a Multi-agent Markov-state (MaMs) workflow for report generation. We empirically show that our proposed rubric generators deliver more discriminative and better human-aligned supervision than existing rubric design strategies. Moreover, when integrated into the MaMs training framework, DeepResearch systems equipped with our rubric generators consistently outperform all open-source baselines on the DeepResearch Bench and achieve performance comparable to that of leading closed-source models. 

---
# Controlling Output Rankings in Generative Engines for LLM-based Search 

**Authors**: Haibo Jin, Ruoxi Chen, Peiyan Zhang, Yifeng Luo, Huimin Zeng, Man Luo, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03608)  

**Abstract**: The way customers search for and choose products is changing with the rise of large language models (LLMs). LLM-based search, or generative engines, provides direct product recommendations to users, rather than traditional online search results that require users to explore options themselves. However, these recommendations are strongly influenced by the initial retrieval order of LLMs, which disadvantages small businesses and independent creators by limiting their visibility.
In this work, we propose CORE, an optimization method that \textbf{C}ontrols \textbf{O}utput \textbf{R}ankings in g\textbf{E}nerative Engines for LLM-based search. Since the LLM's interactions with the search engine are black-box, CORE targets the content returned by search engines as the primary means of influencing output rankings. Specifically, CORE optimizes retrieved content by appending strategically designed optimization content to steer the ranking of outputs. We introduce three types of optimization content: string-based, reasoning-based, and review-based, demonstrating their effectiveness in shaping output rankings. To evaluate CORE in realistic settings, we introduce ProductBench, a large-scale benchmark with 15 product categories and 200 products per category, where each product is associated with its top-10 recommendations collected from Amazon's search interface.
Extensive experiments on four LLMs with search capabilities (GPT-4o, Gemini-2.5, Claude-4, and Grok-3) demonstrate that CORE achieves an average Promotion Success Rate of \textbf{91.4\% @Top-5}, \textbf{86.6\% @Top-3}, and \textbf{80.3\% @Top-1}, across 15 product categories, outperforming existing ranking manipulation methods while preserving the fluency of optimized content. 

---
# Efficient Algorithms for Partial Constraint Satisfaction Problems over Control-flow Graphs 

**Authors**: Xuran Cai, Amir Goharshady  

**Link**: [PDF](https://arxiv.org/pdf/2602.03588)  

**Abstract**: In this work, we focus on the Partial Constraint Satisfaction Problem (PCSP) over control-flow graphs (CFGs) of programs. PCSP serves as a generalization of the well-known Constraint Satisfaction Problem (CSP). In the CSP framework, we define a set of variables, a set of constraints, and a finite domain $D$ that encompasses all possible values for each variable. The objective is to assign a value to each variable in such a way that all constraints are satisfied. In the graph variant of CSP, an underlying graph is considered and we have one variable corresponding to each vertex of the graph and one or several constraints corresponding to each edge. In PCSPs, we allow for certain constraints to be violated at a specified cost, aiming to find a solution that minimizes the total cost. Numerous classical compiler optimization tasks can be framed as PCSPs over control-flow graphs. Examples include Register Allocation, Lifetime-optimal Speculative Partial Redundancy Elimination (LOSPRE), and Optimal Placement of Bank Selection Instructions. On the other hand, it is well-known that control-flow graphs of structured programs are sparse and decomposable in a variety of ways. In this work, we rely on the Series-Parallel-Loop (SPL) decompositions as introduced by~\cite{RegisterAllocation}. Our main contribution is a general algorithm for PCSPs over SPL graphs with a time complexity of \(O(|G| \cdot |D|^6)\), where \(|G|\) represents the size of the control-flow graph. Note that for any fixed domain $D,$ this yields a linear-time solution. Our algorithm can be seen as a generalization and unification of previous SPL-based approaches for register allocation and LOSPRE. In addition, we provide experimental results over another classical PCSP task, i.e. Optimal Bank Selection, achieving runtimes four times better than the previous state of the art. 

---
# CL-bench: A Benchmark for Context Learning 

**Authors**: Shihan Dou, Ming Zhang, Zhangyue Yin, Chenhao Huang, Yujiong Shen, Junzhe Wang, Jiayi Chen, Yuchen Ni, Junjie Ye, Cheng Zhang, Huaibing Xie, Jianglu Hu, Shaolei Wang, Weichao Wang, Yanling Xiao, Yiting Liu, Zenan Xu, Zhen Guo, Pluto Zhou, Tao Gui, Zuxuan Wu, Xipeng Qiu, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang, Di Wang, Shunyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2602.03587)  

**Abstract**: Current language models (LMs) excel at reasoning over prompts using pre-trained knowledge. However, real-world tasks are far more complex and context-dependent: models must learn from task-specific context and leverage new knowledge beyond what is learned during pre-training to reason and resolve tasks. We term this capability context learning, a crucial ability that humans naturally possess but has been largely overlooked. To this end, we introduce CL-bench, a real-world benchmark consisting of 500 complex contexts, 1,899 tasks, and 31,607 verification rubrics, all crafted by experienced domain experts. Each task is designed such that the new content required to resolve it is contained within the corresponding context. Resolving tasks in CL-bench requires models to learn from the context, ranging from new domain-specific knowledge, rule systems, and complex procedures to laws derived from empirical data, all of which are absent from pre-training. This goes far beyond long-context tasks that primarily test retrieval or reading comprehension, and in-context learning tasks, where models learn simple task patterns via instructions and demonstrations. Our evaluations of ten frontier LMs find that models solve only 17.2% of tasks on average. Even the best-performing model, GPT-5.1, solves only 23.7%, revealing that LMs have yet to achieve effective context learning, which poses a critical bottleneck for tackling real-world, complex context-dependent tasks. CL-bench represents a step towards building LMs with this fundamental capability, making them more intelligent and advancing their deployment in real-world scenarios. 

---
# $V_0$: A Generalist Value Model for Any Policy at State Zero 

**Authors**: Yi-Kai Zhang, Zhiyuan Yao, Hongyan Hao, Yueqing Sun, Qi Gu, Hui Su, Xunliang Cai, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2602.03584)  

**Abstract**: Policy gradient methods rely on a baseline to measure the relative advantage of an action, ensuring the model reinforces behaviors that outperform its current average capability. In the training of Large Language Models (LLMs) using Actor-Critic methods (e.g., PPO), this baseline is typically estimated by a Value Model (Critic) often as large as the policy model itself. However, as the policy continuously evolves, the value model requires expensive, synchronous incremental training to accurately track the shifting capabilities of the policy. To avoid this overhead, Group Relative Policy Optimization (GRPO) eliminates the coupled value model by using the average reward of a group of rollouts as the baseline; yet, this approach necessitates extensive sampling to maintain estimation stability. In this paper, we propose $V_0$, a Generalist Value Model capable of estimating the expected performance of any model on unseen prompts without requiring parameter updates. We reframe value estimation by treating the policy's dynamic capability as an explicit context input; specifically, we leverage a history of instruction-performance pairs to dynamically profile the model, departing from the traditional paradigm that relies on parameter fitting to perceive capability shifts. Focusing on value estimation at State Zero (i.e., the initial prompt, hence $V_0$), our model serves as a critical resource scheduler. During GRPO training, $V_0$ predicts success rates prior to rollout, allowing for efficient sampling budget allocation; during deployment, it functions as a router, dispatching instructions to the most cost-effective and suitable model. Empirical results demonstrate that $V_0$ significantly outperforms heuristic budget allocation and achieves a Pareto-optimal trade-off between performance and cost in LLM routing tasks. 

---
# Use Graph When It Needs: Efficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs 

**Authors**: Su Dong, Qinggang Zhang, Yilin Xiao, Shengyuan Chen, Chuang Zhou, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03578)  

**Abstract**: Large language models (LLMs) often struggle with knowledge-intensive tasks due to hallucinations and outdated parametric knowledge. While Retrieval-Augmented Generation (RAG) addresses this by integrating external corpora, its effectiveness is limited by fragmented information in unstructured domain documents. Graph-augmented RAG (GraphRAG) emerged to enhance contextual reasoning through structured knowledge graphs, yet paradoxically underperforms vanilla RAG in real-world scenarios, exhibiting significant accuracy drops and prohibitive latency despite gains on complex queries. We identify the rigid application of GraphRAG to all queries, regardless of complexity, as the root cause. To resolve this, we propose an efficient and adaptive GraphRAG framework called EA-GraphRAG that dynamically integrates RAG and GraphRAG paradigms through syntax-aware complexity analysis. Our approach introduces: (i) a syntactic feature constructor that parses each query and extracts a set of structural features; (ii) a lightweight complexity scorer that maps these features to a continuous complexity score; and (iii) a score-driven routing policy that selects dense RAG for low-score queries, invokes graph-based retrieval for high-score queries, and applies complexity-aware reciprocal rank fusion to handle borderline cases. Extensive experiments on a comprehensive benchmark, consisting of two single-hop and two multi-hop QA benchmarks, demonstrate that our EA-GraphRAG significantly improves accuracy, reduces latency, and achieves state-of-the-art performance in handling mixed scenarios involving both simple and complex queries. 

---
# ACL: Aligned Contrastive Learning Improves BERT and Multi-exit BERT Fine-tuning 

**Authors**: Wei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03563)  

**Abstract**: Despite its success in self-supervised learning, contrastive learning is less studied in the supervised setting. In this work, we first use a set of pilot experiments to show that in the supervised setting, the cross-entropy loss objective (CE) and the contrastive learning objective often conflict with each other, thus hindering the applications of CL in supervised settings. To resolve this problem, we introduce a novel \underline{A}ligned \underline{C}ontrastive \underline{L}earning (ACL) framework. First, ACL-Embed regards label embeddings as extra augmented samples with different labels and employs contrastive learning to align the label embeddings with its samples' representations. Second, to facilitate the optimization of ACL-Embed objective combined with the CE loss, we propose ACL-Grad, which will discard the ACL-Embed term if the two objectives are in conflict. To further enhance the performances of intermediate exits of multi-exit BERT, we further propose cross-layer ACL (ACL-CL), which is to ask the teacher exit to guide the optimization of student shallow exits. Extensive experiments on the GLUE benchmark results in the following takeaways: (a) ACL-BRT outperforms or performs comparably with CE and CE+SCL on the GLUE tasks; (b) ACL, especially CL-ACL, significantly surpasses the baseline methods on the fine-tuning of multi-exit BERT, thus providing better quality-speed tradeoffs for low-latency applications. 

---
# HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing 

**Authors**: Yizhao Gao, Jianyu Wei, Qihao Zhang, Yu Cheng, Shimao Chen, Zhengju Tang, Zihan Jiang, Yifan Song, Hailin Zhang, Liang Zhao, Bo Yang, Gang Wang, Shijie Cao, Fuli Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.03560)  

**Abstract**: This work introduces Hybrid Sparse Attention (HySparse), a new architecture that interleaves each full attention layer with several sparse attention layers. While conceptually simple, HySparse strategically derives each sparse layer's token selection and KV caches directly from the preceding full attention layer. This architecture resolves two fundamental limitations of prior sparse attention methods. First, conventional approaches typically rely on additional proxies to predict token importance, introducing extra complexity and potentially suboptimal performance. In contrast, HySparse uses the full attention layer as a precise oracle to identify important tokens. Second, existing sparse attention designs often reduce computation without saving KV cache. HySparse enables sparse attention layers to reuse the full attention KV cache, thereby reducing both computation and memory. We evaluate HySparse on both 7B dense and 80B MoE models. Across all settings, HySparse consistently outperforms both full attention and hybrid SWA baselines. Notably, in the 80B MoE model with 49 total layers, only 5 layers employ full attention, yet HySparse achieves substantial performance gains while reducing KV cache storage by nearly 10x. 

---
# Assessing the Impact of Typological Features on Multilingual Machine Translation in the Age of Large Language Models 

**Authors**: Vitalii Hirak, Jaap Jumelet, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2602.03551)  

**Abstract**: Despite major advances in multilingual modeling, large quality disparities persist across languages. Besides the obvious impact of uneven training resources, typological properties have also been proposed to determine the intrinsic difficulty of modeling a language. The existing evidence, however, is mostly based on small monolingual language models or bilingual translation models trained from scratch. We expand on this line of work by analyzing two large pre-trained multilingual translation models, NLLB-200 and Tower+, which are state-of-the-art representatives of encoder-decoder and decoder-only machine translation, respectively. Based on a broad set of languages, we find that target language typology drives translation quality of both models, even after controlling for more trivial factors, such as data resourcedness and writing script. Additionally, languages with certain typological properties benefit more from a wider search of the output space, suggesting that such languages could profit from alternative decoding strategies beyond the standard left-to-right beam search. To facilitate further research in this area, we release a set of fine-grained typological properties for 212 languages of the FLORES+ MT evaluation benchmark. 

---
# SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue 

**Authors**: Yuqin Dai, Ning Gao, Wei Zhang, Jie Wang, Zichen Luo, Jinpeng Wang, Yujie Wang, Ruiyuan Wu, Chaozheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03548)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities in open-domain dialogues. However, current methods exhibit suboptimal performance in service dialogues, as they rely on noisy, low-quality human conversation data. This limitation arises from data scarcity and the difficulty of simulating authentic, goal-oriented user behaviors. To address these issues, we propose SEAD (Self-Evolving Agent for Service Dialogue), a framework that enables agents to learn effective strategies without large-scale human annotations. SEAD decouples user modeling into two components: a Profile Controller that generates diverse user states to manage training curriculum, and a User Role-play Model that focuses on realistic role-playing. This design ensures the environment provides adaptive training scenarios rather than acting as an unfair adversary. Experiments demonstrate that SEAD significantly outperforms Open-source Foundation Models and Closed-source Commercial Models, improving task completion rate by 17.6% and dialogue efficiency by 11.1%. Code is available at: this https URL. 

---
# Can Large Language Models Generalize Procedures Across Representations? 

**Authors**: Fangru Lin, Valentin Hofmann, Xingchen Wan, Weixing Wang, Zifeng Ding, Anthony G. Cohn, Janet B. Pierrehumbert  

**Link**: [PDF](https://arxiv.org/pdf/2602.03542)  

**Abstract**: Large language models (LLMs) are trained and tested extensively on symbolic representations such as code and graphs, yet real-world user tasks are often specified in natural language. To what extent can LLMs generalize across these representations? Here, we approach this question by studying isomorphic tasks involving procedures represented in code, graphs, and natural language (e.g., scheduling steps in planning). We find that training LLMs with popular post-training methods on graphs or code data alone does not reliably generalize to corresponding natural language tasks, while training solely on natural language can lead to inefficient performance gains. To address this gap, we propose a two-stage data curriculum that first trains on symbolic, then natural language data. The curriculum substantially improves model performance across model families and tasks. Remarkably, a 1.5B Qwen model trained by our method can closely match zero-shot GPT-4o in naturalistic planning. Finally, our analysis suggests that successful cross-representation generalization can be interpreted as a form of generative analogy, which our curriculum effectively encourages. 

---
# Learning to Reason Faithfully through Step-Level Faithfulness Maximization 

**Authors**: Runquan Gui, Yafu Li, Xiaoye Qu, Ziyan Liu, Yeqiu Cheng, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.03507)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has markedly improved the performance of Large Language Models (LLMs) on tasks requiring multi-step reasoning. However, most RLVR pipelines rely on sparse outcome-based rewards, providing little supervision over intermediate steps and thus encouraging over-confidence and spurious reasoning, which in turn increases hallucinations. To address this, we propose FaithRL, a general reinforcement learning framework that directly optimizes reasoning faithfulness. We formalize a faithfulness-maximization objective and theoretically show that optimizing it mitigates over-confidence. To instantiate this objective, we introduce a geometric reward design and a faithfulness-aware advantage modulation mechanism that assigns step-level credit by penalizing unsupported steps while preserving valid partial derivations. Across diverse backbones and benchmarks, FaithRL consistently reduces hallucination rates while maintaining (and often improving) answer correctness. Further analysis confirms that FaithRL increases step-wise reasoning faithfulness and generalizes robustly. Our code is available at this https URL. 

---
# Self-Verification Dilemma: Experience-Driven Suppression of Overused Checking in LLM Reasoning 

**Authors**: Quanyu Long, Kai Jie Jiang, Jianda Chen, Xu Guo, Leilei Gan, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03485)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance by generating long reasoning traces with reflection. Through a large-scale empirical analysis, we find that a substantial fraction of reflective steps consist of self-verification (recheck) that repeatedly confirm intermediate results. These rechecks occur frequently across models and benchmarks, yet the vast majority are confirmatory rather than corrective, rarely identifying errors and altering reasoning outcomes. This reveals a mismatch between how often self-verification is activated and how often it is actually useful. Motivated by this, we propose a novel, experience-driven test-time framework that reduces the overused verification. Our method detects the activation of recheck behavior, consults an offline experience pool of past verification outcomes, and estimates whether a recheck is likely unnecessary via efficient retrieval. When historical experience suggests unnecessary, a suppression signal redirects the model to proceed. Across multiple model and benchmarks, our approach reduces token usage up to 20.3% while maintaining the accuracy, and in some datasets even yields accuracy improvements. 

---
# Preferences for Idiomatic Language are Acquired Slowly -- and Forgotten Quickly: A Case Study on Swedish 

**Authors**: Jenny Kunz  

**Link**: [PDF](https://arxiv.org/pdf/2602.03484)  

**Abstract**: In this study, we investigate how language models develop preferences for \textit{idiomatic} as compared to \textit{linguistically acceptable} Swedish, both during pretraining and when adapting a model from English to Swedish. To do so, we train models on Swedish from scratch and by fine-tuning English-pretrained models, probing their preferences at various checkpoints using minimal pairs that differ in linguistic acceptability or idiomaticity. For linguistic acceptability, we adapt existing benchmarks into a minimal-pair format. To assess idiomaticity, we introduce two novel datasets: one contrasting conventionalized idioms with plausible variants, and another contrasting idiomatic Swedish with Translationese. Our findings suggest that idiomatic competence emerges more slowly than other linguistic abilities, including grammatical and lexical correctness. While longer training yields diminishing returns for most tasks, idiom-related performance continues to improve, particularly in the largest model tested (8B). However, instruction tuning on data machine-translated from English -- the common approach for languages with little or no native instruction data -- causes models to rapidly lose their preference for idiomatic language. 

---
# A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces 

**Authors**: Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan Wang, Pengyu Wang, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2602.03442)  

**Abstract**: Frontier language models have demonstrated strong reasoning and long-horizon tool-use capabilities. However, existing RAG systems fail to leverage these capabilities. They still rely on two paradigms: (1) designing an algorithm that retrieves passages in a single shot and concatenates them into the model's input, or (2) predefining a workflow and prompting the model to execute it step-by-step. Neither paradigm allows the model to participate in retrieval decisions, preventing efficient scaling with model improvements. In this paper, we introduce A-RAG, an Agentic RAG framework that exposes hierarchical retrieval interfaces directly to the model. A-RAG provides three retrieval tools: keyword search, semantic search, and chunk read, enabling the agent to adaptively search and retrieve information across multiple granularities. Experiments on multiple open-domain QA benchmarks show that A-RAG consistently outperforms existing approaches with comparable or lower retrieved tokens, demonstrating that A-RAG effectively leverages model capabilities and dynamically adapts to different RAG tasks. We further systematically study how A-RAG scales with model size and test-time compute. We will release our code and evaluation suite to facilitate future research. Code and evaluation suite are available at this https URL. 

---
# FactNet: A Billion-Scale Knowledge Graph for Multilingual Factual Grounding 

**Authors**: Yingli Shen, Wen Lai, Jie Zhou, Xueren Zhang, Yudong Wang, Kangyang Luo, Shuo Wang, Ge Gao, Alexander Fraser, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.03417)  

**Abstract**: While LLMs exhibit remarkable fluency, their utility is often compromised by factual hallucinations and a lack of traceable provenance. Existing resources for grounding mitigate this but typically enforce a dichotomy: they offer either structured knowledge without textual context (e.g., knowledge bases) or grounded text with limited scale and linguistic coverage. To bridge this gap, we introduce FactNet, a massive, open-source resource designed to unify 1.7 billion atomic assertions with 3.01 billion auditable evidence pointers derived exclusively from 316 Wikipedia editions. Unlike recent synthetic approaches, FactNet employs a strictly deterministic construction pipeline, ensuring that every evidence unit is recoverable with byte-level precision. Extensive auditing confirms a high grounding precision of 92.1%, even in long-tail languages. Furthermore, we establish FactNet-Bench, a comprehensive evaluation suite for Knowledge Graph Completion, Question Answering, and Fact Checking. FactNet provides the community with a foundational, reproducible resource for training and evaluating trustworthy, verifiable multilingual systems. 

---
# Verified Critical Step Optimization for LLM Agents 

**Authors**: Mukai Li, Qingcheng Zeng, Tianqing Fang, Zhenwen Liang, Linfeng Song, Qi Liu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03412)  

**Abstract**: As large language model agents tackle increasingly complex long-horizon tasks, effective post-training becomes critical. Prior work faces fundamental challenges: outcome-only rewards fail to precisely attribute credit to intermediate steps, estimated step-level rewards introduce systematic noise, and Monte Carlo sampling approaches for step reward estimation incur prohibitive computational cost. Inspired by findings that only a small fraction of high-entropy tokens drive effective RL for reasoning, we propose Critical Step Optimization (CSO), which focuses preference learning on verified critical steps, decision points where alternate actions demonstrably flip task outcomes from failure to success. Crucially, our method starts from failed policy trajectories rather than expert demonstrations, directly targeting the policy model's weaknesses. We use a process reward model (PRM) to identify candidate critical steps, leverage expert models to propose high-quality alternatives, then continue execution from these alternatives using the policy model itself until task completion. Only alternatives that the policy successfully executes to correct outcomes are verified and used as DPO training data, ensuring both quality and policy reachability. This yields fine-grained, verifiable supervision at critical decisions while avoiding trajectory-level coarseness and step-level noise. Experiments on GAIA-Text-103 and XBench-DeepSearch show that CSO achieves 37% and 26% relative improvement over the SFT baseline and substantially outperforms other post-training methods, while requiring supervision at only 16% of trajectory steps. This demonstrates the effectiveness of selective verification-based learning for agent post-training. 

---
# Towards Distillation-Resistant Large Language Models: An Information-Theoretic Perspective 

**Authors**: Hao Fang, Tianyi Zhang, Tianqu Zhuang, Jiawei Kong, Kuofeng Gao, Bin Chen, Leqi Liang, Shu-Tao Xia, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03396)  

**Abstract**: Proprietary large language models (LLMs) embody substantial economic value and are generally exposed only as black-box APIs, yet adversaries can still exploit their outputs to extract knowledge via distillation. Existing defenses focus exclusively on text-based distillation, leaving the important logit-based distillation largely unexplored. In this work, we analyze this problem and present an effective solution from an information-theoretic perspective. We characterize distillation-relevant information in teacher outputs using the conditional mutual information (CMI) between teacher logits and input queries conditioned on ground-truth labels. This quantity captures contextual information beneficial for model extraction, motivating us to defend distillation via CMI minimization. Guided by our theoretical analysis, we propose learning a transformation matrix that purifies the original outputs to enhance distillation resistance. We further derive a CMI-inspired anti-distillation objective to optimize this transformation, which effectively removes distillation-relevant information while preserving output utility. Extensive experiments across multiple LLMs and strong distillation algorithms demonstrate that the proposed method significantly degrades distillation performance while preserving task accuracy, effectively protecting models' intellectual property. 

---
# Pursuing Best Industrial Practices for Retrieval-Augmented Generation in the Medical Domain 

**Authors**: Wei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03368)  

**Abstract**: While retrieval augmented generation (RAG) has been swiftly adopted in industrial applications based on large language models (LLMs), there is no consensus on what are the best practices for building a RAG system in terms of what are the components, how to organize these components and how to implement each component for the industrial applications, especially in the medical domain. In this work, we first carefully analyze each component of the RAG system and propose practical alternatives for each component. Then, we conduct systematic evaluations on three types of tasks, revealing the best practices for improving the RAG system and how LLM-based RAG systems make trade-offs between performance and efficiency. 

---
# PEGRL: Improving Machine Translation by Post-Editing Guided Reinforcement Learning 

**Authors**: Yunzhi Shen, Hao Zhou, Xin Huang, Xue Han, Junlan Feng, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03352)  

**Abstract**: Reinforcement learning (RL) has shown strong promise for LLM-based machine translation, with recent methods such as GRPO demonstrating notable gains; nevertheless, translation-oriented RL remains challenged by noisy learning signals arising from Monte Carlo return estimation, as well as a large trajectory space that favors global exploration over fine-grained local optimization. We introduce \textbf{PEGRL}, a \textit{two-stage} RL framework that uses post-editing as an auxiliary task to stabilize training and guide overall optimization. At each iteration, translation outputs are sampled to construct post-editing inputs, allowing return estimation in the post-editing stage to benefit from conditioning on the current translation behavior, while jointly supporting both global exploration and fine-grained local optimization. A task-specific weighting scheme further balances the contributions of translation and post-editing objectives, yielding a biased yet more sample-efficient estimator. Experiments on English$\to$Finnish, English$\to$Turkish, and English$\leftrightarrow$Chinese show consistent gains over RL baselines, and for English$\to$Turkish, performance on COMET-KIWI is comparable to advanced LLM-based systems (DeepSeek-V3.2). 

---
# Accurate Failure Prediction in Agents Does Not Imply Effective Failure Prevention 

**Authors**: Rakshith Vasudev, Melisa Russak, Dan Bikel, Waseem Alshikh  

**Link**: [PDF](https://arxiv.org/pdf/2602.03338)  

**Abstract**: Proactive interventions by LLM critic models are often assumed to improve reliability, yet their effects at deployment time are poorly understood. We show that a binary LLM critic with strong offline accuracy (AUROC 0.94) can nevertheless cause severe performance degradation, inducing a 26 percentage point (pp) collapse on one model while affecting another by near zero pp. This variability demonstrates that LLM critic accuracy alone is insufficient to determine whether intervention is safe.
We identify a disruption-recovery tradeoff: interventions may recover failing trajectories but also disrupt trajectories that would have succeeded. Based on this insight, we propose a pre-deployment test that uses a small pilot of 50 tasks to estimate whether intervention is likely to help or harm, without requiring full deployment. Across benchmarks, the test correctly anticipates outcomes: intervention degrades performance on high-success tasks (0 to -26 pp), while yielding a modest improvement on the high-failure ALFWorld benchmark (+2.8 pp, p=0.014). The primary value of our framework is therefore identifying when not to intervene, preventing severe regressions before deployment. 

---
# MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research 

**Authors**: Yifan Shi, Jialong Shi, Jiayi Wang, Ye Fan, Jianyong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.03318)  

**Abstract**: Operations Research (OR) relies on expert-driven modeling-a slow and fragile process ill-suited to novel scenarios. While large language models (LLMs) can automatically translate natural language into optimization models, existing approaches either rely on costly post-training or employ multi-agent frameworks, yet most still lack reliable collaborative error correction and task-specific retrieval, often leading to incorrect outputs. We propose MIRROR, a fine-tuning-free, end-to-end multi-agent framework that directly translates natural language optimization problems into mathematical models and solver code. MIRROR integrates two core mechanisms: (1) execution-driven iterative adaptive revision for automatic error correction, and (2) hierarchical retrieval to fetch relevant modeling and coding exemplars from a carefully curated exemplar library. Experiments show that MIRROR outperforms existing methods on standard OR benchmarks, with notable results on complex industrial datasets such as IndustryOR and Mamo-ComplexLP. By combining precise external knowledge infusion with systematic error correction, MIRROR provides non-expert users with an efficient and reliable OR modeling solution, overcoming the fundamental limitations of general-purpose LLMs in expert optimization tasks. 

---
# POP: Prefill-Only Pruning for Efficient Large Model Inference 

**Authors**: Junhui He, Zhihui Fu, Jun Wang, Qingan Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.03295)  

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated remarkable capabilities. However, their deployment is hindered by significant computational costs. Existing structured pruning methods, while hardware-efficient, often suffer from significant accuracy degradation. In this paper, we argue that this failure stems from a stage-agnostic pruning approach that overlooks the asymmetric roles between the prefill and decode stages. By introducing a virtual gate mechanism, our importance analysis reveals that deep layers are critical for next-token prediction (decode) but largely redundant for context encoding (prefill). Leveraging this insight, we propose Prefill-Only Pruning (POP), a stage-aware inference strategy that safely omits deep layers during the computationally intensive prefill stage while retaining the full model for the sensitive decode stage. To enable the transition between stages, we introduce independent Key-Value (KV) projections to maintain cache integrity, and a boundary handling strategy to ensure the accuracy of the first generated token. Extensive experiments on Llama-3.1, Qwen3-VL, and Gemma-3 across diverse modalities demonstrate that POP achieves up to 1.37$\times$ speedup in prefill latency with minimal performance loss, effectively overcoming the accuracy-efficiency trade-off limitations of existing structured pruning methods. 

---
# ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLMs 

**Authors**: Xuancheng Li, Haitao Li, Yujia Zhou, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03226)  

**Abstract**: Long-context inputs in large language models (LLMs) often suffer from the "lost in the middle" problem, where critical information becomes diluted or ignored due to excessive length. Context compression methods aim to address this by reducing input size, but existing approaches struggle with balancing information preservation and compression efficiency. We propose Adaptive Task-Aware Compressor (ATACompressor), which dynamically adjusts compression based on the specific requirements of the task. ATACompressor employs a selective encoder that compresses only the task-relevant portions of long contexts, ensuring that essential information is preserved while reducing unnecessary content. Its adaptive allocation controller perceives the length of relevant content and adjusts the compression rate accordingly, optimizing resource utilization. We evaluate ATACompressor on three QA datasets: HotpotQA, MSMARCO, and SQUAD-showing that it outperforms existing methods in terms of both compression efficiency and task performance. Our approach provides a scalable solution for long-context processing in LLMs. Furthermore, we perform a range of ablation studies and analysis experiments to gain deeper insights into the key components of ATACompressor. 

---
# Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection 

**Authors**: Dongwon Jo, Beomseok Kang, Jiwon Song, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.03216)  

**Abstract**: The quadratic complexity of attention remains the central bottleneck in long-context inference for large language models. Prior acceleration methods either sparsify the attention map with structured patterns or permanently evict tokens at specific layers, which can retain irrelevant tokens or rely on irreversible early decisions despite the layer-/head-wise dynamics of token importance. In this paper, we propose Token Sparse Attention, a lightweight and dynamic token-level sparsification mechanism that compresses per-head $Q$, $K$, $V$ to a reduced token set during attention and then decompresses the output back to the original sequence, enabling token information to be reconsidered in subsequent layers. Furthermore, Token Sparse Attention exposes a new design point at the intersection of token selection and sparse attention. Our approach is fully compatible with dense attention implementations, including Flash Attention, and can be seamlessly composed with existing sparse attention kernels. Experimental results show that Token Sparse Attention consistently improves accuracy-latency trade-off, achieving up to $\times$3.23 attention speedup at 128K context with less than 1% accuracy degradation. These results demonstrate that dynamic and interleaved token-level sparsification is a complementary and effective strategy for scalable long-context inference. 

---
# ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution 

**Authors**: Zican Dong, Peiyu Liu, Junyi Li, Zhipeng Chen, Han Peng, Shuo Wang, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.03203)  

**Abstract**: Recently, large language models (LLMs) have shown remarkable reasoning abilities by producing long reasoning traces. However, as the sequence length grows, the key-value (KV) cache expands linearly, incurring significant memory and computation costs. Existing KV cache eviction methods mitigate this issue by discarding less important KV pairs, but often fail to capture complex KV dependencies, resulting in performance degradation. To better balance efficiency and performance, we introduce ForesightKV, a training-based KV cache eviction framework that learns to predict which KV pairs to evict during long-text generations. We first design the Golden Eviction algorithm, which identifies the optimal eviction KV pairs at each step using future attention scores. These traces and the scores at each step are then distilled via supervised training with a Pairwise Ranking Loss. Furthermore, we formulate cache eviction as a Markov Decision Process and apply the GRPO algorithm to mitigate the significant language modeling loss increase on low-entropy tokens. Experiments on AIME2024 and AIME2025 benchmarks of three reasoning models demonstrate that ForesightKV consistently outperforms prior methods under only half the cache budget, while benefiting synergistically from both supervised and reinforcement learning approaches. 

---
# Privasis: Synthesizing the Largest "Public" Private Dataset from Scratch 

**Authors**: Hyunwoo Kim, Niloofar Mireshghallah, Michael Duan, Rui Xin, Shuyue Stella Li, Jaehun Jung, David Acuna, Qi Pang, Hanshen Xiao, G. Edward Suh, Sewoong Oh, Yulia Tsvetkov, Pang Wei Koh, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.03183)  

**Abstract**: Research involving privacy-sensitive data has always been constrained by data scarcity, standing in sharp contrast to other areas that have benefited from data scaling. This challenge is becoming increasingly urgent as modern AI agents--such as OpenClaw and Gemini Agent--are granted persistent access to highly sensitive personal information. To tackle this longstanding bottleneck and the rising risks, we present Privasis (i.e., privacy oasis), the first million-scale fully synthetic dataset entirely built from scratch--an expansive reservoir of texts with rich and diverse private information--designed to broaden and accelerate research in areas where processing sensitive social data is inevitable. Compared to existing datasets, Privasis, comprising 1.4 million records, offers orders-of-magnitude larger scale with quality, and far greater diversity across various document types, including medical history, legal documents, financial records, calendars, and text messages with a total of 55.1 million annotated attributes such as ethnicity, date of birth, workplace, etc. We leverage Privasis to construct a parallel corpus for text sanitization with our pipeline that decomposes texts and applies targeted sanitization. Our compact sanitization models (<=4B) trained on this dataset outperform state-of-the-art large language models, such as GPT-5 and Qwen-3 235B. We plan to release data, models, and code to accelerate future research on privacy-sensitive domains and agents. 

---
# FASA: Frequency-aware Sparse Attention 

**Authors**: Yifei Wang, Yueqi Wang, Zhenrui Yue, Huimin Zeng, Yong Wang, Ismini Lourentzou, Zhengzhong Tu, Xiangxiang Chu, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2602.03152)  

**Abstract**: The deployment of Large Language Models (LLMs) faces a critical bottleneck when handling lengthy inputs: the prohibitive memory footprint of the Key Value (KV) cache. To address this bottleneck, the token pruning paradigm leverages attention sparsity to selectively retain a small, critical subset of tokens. However, existing approaches fall short, with static methods risking irreversible information loss and dynamic strategies employing heuristics that insufficiently capture the query-dependent nature of token importance. We propose FASA, a novel framework that achieves query-aware token eviction by dynamically predicting token importance. FASA stems from a novel insight into RoPE: the discovery of functional sparsity at the frequency-chunk (FC) level. Our key finding is that a small, identifiable subset of "dominant" FCs consistently exhibits high contextual agreement with the full attention head. This provides a robust and computationally free proxy for identifying salient tokens. %making them a powerful and efficient proxy for token importance. Building on this insight, FASA first identifies a critical set of tokens using dominant FCs, and then performs focused attention computation solely on this pruned subset. % Since accessing only a small fraction of the KV cache, FASA drastically lowers memory bandwidth requirements and computational cost. Across a spectrum of long-context tasks, from sequence modeling to complex CoT reasoning, FASA consistently outperforms all token-eviction baselines and achieves near-oracle accuracy, demonstrating remarkable robustness even under constraint budgets. Notably, on LongBench-V1, FASA reaches nearly 100\% of full-KV performance when only keeping 256 tokens, and achieves 2.56$\times$ speedup using just 18.9\% of the cache on AIME24. 

---
# Short Chains, Deep Thoughts: Balancing Reasoning Efficiency and Intra-Segment Capability via Split-Merge Optimization 

**Authors**: Runquan Gui, Jie Wang, Zhihai Wang, Chi Ma, Jianye Hao, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03141)  

**Abstract**: While Large Reasoning Models (LRMs) have demonstrated impressive capabilities in solving complex tasks through the generation of long reasoning chains, this reliance on verbose generation results in significant latency and computational overhead. To address these challenges, we propose \textbf{CoSMo} (\textbf{Co}nsistency-Guided \textbf{S}plit-\textbf{M}erge \textbf{O}ptimization), a framework designed to eliminate structural redundancy rather than indiscriminately restricting token volume. Specifically, CoSMo utilizes a split-merge algorithm that dynamically refines reasoning chains by merging redundant segments and splitting logical gaps to ensure coherence. We then employ structure-aligned reinforcement learning with a novel segment-level budget to supervise the model in maintaining efficient reasoning structures throughout training. Extensive experiments across multiple benchmarks and backbones demonstrate that CoSMo achieves superior performance, improving accuracy by \textbf{3.3} points while reducing segment usage by \textbf{28.7\%} on average compared to reasoning efficiency baselines. 

---
# One Model, All Roles: Multi-Turn, Multi-Agent Self-Play Reinforcement Learning for Conversational Social Intelligence 

**Authors**: Bowen Jiang, Taiwei Shi, Ryo Kamoi, Yuan Yuan, Camillo J. Taylor, Longqi Yang, Pei Zhou, Sihao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03109)  

**Abstract**: This paper introduces OMAR: One Model, All Roles, a reinforcement learning framework that enables AI to develop social intelligence through multi-turn, multi-agent conversational self-play. Unlike traditional paradigms that rely on static, single-turn optimizations, OMAR allows a single model to role-play all participants in a conversation simultaneously, learning to achieve long-term goals and complex social norms directly from dynamic social interaction. To ensure training stability across long dialogues, we implement a hierarchical advantage estimation that calculates turn-level and token-level advantages. Evaluations in the SOTOPIA social environment and Werewolf strategy games show that our trained models develop fine-grained, emergent social intelligence, such as empathy, persuasion, and compromise seeking, demonstrating the effectiveness of learning collaboration even under competitive scenarios. While we identify practical challenges like reward hacking, our results show that rich social intelligence can emerge without human supervision. We hope this work incentivizes further research on AI social intelligence in group conversations. 

---
# ChemPro: A Progressive Chemistry Benchmark for Large Language Models 

**Authors**: Aaditya Baranwal, Shruti Vyas  

**Link**: [PDF](https://arxiv.org/pdf/2602.03108)  

**Abstract**: We introduce ChemPro, a progressive benchmark with 4100 natural language question-answer pairs in Chemistry, across 4 coherent sections of difficulty designed to assess the proficiency of Large Language Models (LLMs) in a broad spectrum of general chemistry topics. We include Multiple Choice Questions and Numerical Questions spread across fine-grained information recall, long-horizon reasoning, multi-concept questions, problem-solving with nuanced articulation, and straightforward questions in a balanced ratio, effectively covering Bio-Chemistry, Inorganic-Chemistry, Organic-Chemistry and Physical-Chemistry. ChemPro is carefully designed analogous to a student's academic evaluation for basic to high-school chemistry. A gradual increase in the question difficulty rigorously tests the ability of LLMs to progress from solving basic problems to solving more sophisticated challenges.
We evaluate 45+7 state-of-the-art LLMs, spanning both open-source and proprietary variants, and our analysis reveals that while LLMs perform well on basic chemistry questions, their accuracy declines with different types and levels of complexity. These findings highlight the critical limitations of LLMs in general scientific reasoning and understanding and point towards understudied dimensions of difficulty, emphasizing the need for more robust methodologies to improve LLMs. 

---
# The Mask of Civility: Benchmarking Chinese Mock Politeness Comprehension in Large Language Models 

**Authors**: Yitong Zhang, Yuhan Xiang, Mingxuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03107)  

**Abstract**: From a pragmatic perspective, this study systematically evaluates the differences in performance among representative large language models (LLMs) in recognizing politeness, impoliteness, and mock politeness phenomena in Chinese. Addressing the existing gaps in pragmatic comprehension, the research adopts the frameworks of Rapport Management Theory and the Model of Mock Politeness to construct a three-category dataset combining authentic and simulated Chinese discourse. Six representative models, including GPT-5.1 and DeepSeek, were selected as test subjects and evaluated under four prompting conditions: zero-shot, few-shot, knowledge-enhanced, and hybrid strategies. This study serves as a meaningful attempt within the paradigm of ``Great Linguistics,'' offering a novel approach to applying pragmatic theory in the age of technological transformation. It also responds to the contemporary question of how technology and the humanities may coexist, representing an interdisciplinary endeavor that bridges linguistic technology and humanistic reflection. 

---
# Task--Specificity Score: Measuring How Much Instructions Really Matter for Supervision 

**Authors**: Pritam Kadasi, Abhishek Upperwal, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2602.03103)  

**Abstract**: Instruction tuning is now the default way to train and adapt large language models, but many instruction--input--output pairs are only weakly specified: for a given input, the same output can remain plausible under several alternative instructions. This raises a simple question: \emph{does the instruction uniquely determine the target output?}
We propose the \textbf{Task--Specificity Score (TSS)} to quantify how much an instruction matters for predicting its output, by contrasting the true instruction against plausible alternatives for the same input. We further introduce \textbf{TSS++}, which uses hard alternatives and a small quality term to mitigate easy-negative effects. Across three instruction datasets (\textsc{Alpaca}, \textsc{Dolly-15k}, \textsc{NI-20}) and three open LLMs (Gemma, Llama, Qwen), we show that selecting task-specific examples improves downstream performance under tight token budgets and complements quality-based filters such as perplexity and IFD. 

---
# Test-time Recursive Thinking: Self-Improvement without External Feedback 

**Authors**: Yufan Zhuang, Chandan Singh, Liyuan Liu, Yelong Shen, Dinghuai Zhang, Jingbo Shang, Jianfeng Gao, Weizhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03094)  

**Abstract**: Modern Large Language Models (LLMs) have shown rapid improvements in reasoning capabilities, driven largely by reinforcement learning (RL) with verifiable rewards. Here, we ask whether these LLMs can self-improve without the need for additional training. We identify two core challenges for such systems: (i) efficiently generating diverse, high-quality candidate solutions, and (ii) reliably selecting correct answers in the absence of ground-truth supervision. To address these challenges, we propose Test-time Recursive Thinking (TRT), an iterative self-improvement framework that conditions generation on rollout-specific strategies, accumulated knowledge, and self-generated verification signals. Using TRT, open-source models reach 100% accuracy on AIME-25/24, and on LiveCodeBench's most difficult problems, closed-source models improve by 10.4-14.8 percentage points without external feedback. 

---
# AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback 

**Authors**: Zhitao Gao, Jie Ma, Xuhong Li, Pengyu Li, Ning Qu, Yaqiang Wu, Hui Liu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03084)  

**Abstract**: Large Language Models (LLMs) have achieved significant success in complex reasoning but remain bottlenecked by reliance on expert-annotated data and external verifiers. While existing self-evolution paradigms aim to bypass these constraints, they often fail to identify the optimal learning zone and risk reinforcing collective hallucinations and incorrect priors through flawed internal feedback. To address these challenges, we propose \underline{A}utonomous \underline{E}volutionary \underline{R}easoning \underline{O}ptimization (AERO), an unsupervised framework that achieves autonomous reasoning evolution by internalizing self-questioning, answering, and criticism within a synergistic dual-loop system. Inspired by the \textit{Zone of Proximal Development (ZPD)} theory, AERO utilizes entropy-based positioning to target the ``solvability gap'' and employs Independent Counterfactual Correction for robust verification. Furthermore, we introduce a Staggered Training Strategy to synchronize capability growth across functional roles and prevent curriculum collapse. Extensive evaluations across nine benchmarks spanning three domains demonstrate that AERO achieves average performance improvements of 4.57\% on Qwen3-4B-Base and 5.10\% on Qwen3-8B-Base, outperforming competitive baselines. Code is available at this https URL. 

---
# ReMiT: RL-Guided Mid-Training for Iterative LLM Evolution 

**Authors**: Junjie Huang, Jiarui Qin, Di Yin, Weiwen Liu, Yong Yu, Xing Sun, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03075)  

**Abstract**: Standard training pipelines for large language models (LLMs) are typically unidirectional, progressing from pre-training to post-training. However, the potential for a bidirectional process--where insights from post-training retroactively improve the pre-trained foundation--remains unexplored. We aim to establish a self-reinforcing flywheel: a cycle in which reinforcement learning (RL)-tuned model strengthens the base model, which in turn enhances subsequent post-training performance, requiring no specially trained teacher or reference model. To realize this, we analyze training dynamics and identify the mid-training (annealing) phase as a critical turning point for model capabilities. This phase typically occurs at the end of pre-training, utilizing high-quality corpora under a rapidly decaying learning rate. Building upon this insight, we introduce ReMiT (Reinforcement Learning-Guided Mid-Training). Specifically, ReMiT leverages the reasoning priors of RL-tuned models to dynamically reweight tokens during the mid-training phase, prioritizing those pivotal for reasoning. Empirically, ReMiT achieves an average improvement of 3\% on 10 pre-training benchmarks, spanning math, code, and general reasoning, and sustains these gains by over 2\% throughout the post-training pipeline. These results validate an iterative feedback loop, enabling continuous and self-reinforcing evolution of LLMs. 

---
# SAES-SVD: Self-Adaptive Suppression of Accumulated and Local Errors for SVD-based LLM Compression 

**Authors**: Xing Hu, Dawei Yang, Yuan Cheng, Zhixuan Chen, Zukang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.03051)  

**Abstract**: The rapid growth in the parameter scale of large language models (LLMs) has created a high demand for efficient compression techniques. As a hardware-agnostic and highly compatible technique, low-rank compression has been widely adopted. However, existing methods typically compress each layer independently by minimizing per-layer reconstruction error, overlooking a critical limitation: the reconstruction error propagates and accumulates through the network, which leads to amplified global deviations from the full-precision baseline. To address this, we propose Self-Adaptive Error Suppression SVD (SAES-SVD), a LLMs compression framework that jointly optimizes intra-layer reconstruction and inter-layer error compensation. SAES-SVD is composed of two novel components: (1) Cumulative Error-Aware Layer Compression (CEALC), which formulates the compression objective as a combination of local reconstruction and weighted cumulative error compensation. Based on it, we derive a closed-form low-rank solution relied on second-order activation statistics, which explicitly aligns each layer's output with its full-precision counterpart to compensate for accumulated errors. (2) Adaptive Collaborative Error Suppression (ACES), which automatically adjusts the weighting coefficient to enhance the low-rank structure of the compression objective in CEALC. Specifically, the coefficient is optimized to maximize the ratio between the Frobenius norm of the compressed layer's output and that of the compression objective under a fixed rank, thus ensuring that the rank budget is utilized effectively. Extensive experiments across multiple LLM architectures and tasks show that, without fine-tuning or mixed-rank strategies, SAES-SVD consistently improves post-compression performance. 

---
# LatentMem: Customizing Latent Memory for Multi-Agent Systems 

**Authors**: Muxin Fu, Guibin Zhang, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03036)  

**Abstract**: Large language model (LLM)-powered multi-agent systems (MAS) demonstrate remarkable collective intelligence, wherein multi-agent memory serves as a pivotal mechanism for continual adaptation. However, existing multi-agent memory designs remain constrained by two fundamental bottlenecks: (i) memory homogenization arising from the absence of role-aware customization, and (ii) information overload induced by excessively fine-grained memory entries. To address these limitations, we propose LatentMem, a learnable multi-agent memory framework designed to customize agent-specific memories in a token-efficient manner. Specifically, LatentMem comprises an experience bank that stores raw interaction trajectories in a lightweight form, and a memory composer that synthesizes compact latent memories conditioned on retrieved experience and agent-specific contexts. Further, we introduce Latent Memory Policy Optimization (LMPO), which propagates task-level optimization signals through latent memories to the composer, encouraging it to produce compact and high-utility representations. Extensive experiments across diverse benchmarks and mainstream MAS frameworks show that LatentMem achieves a performance gain of up to $19.36$% over vanilla settings and consistently outperforms existing memory architectures, without requiring any modifications to the underlying frameworks. 

---
# CPMobius: Iterative Coach-Player Reasoning for Data-Free Reinforcement Learning 

**Authors**: Ran Li, Zeyuan Liu, Yinghao chen, Bingxiang He, Jiarui Yuan, Zixuan Fu, Weize Chen, Jinyi Hu, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.02979)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong potential in complex reasoning, yet their progress remains fundamentally constrained by reliance on massive high-quality human-curated tasks and labels, either through supervised fine-tuning (SFT) or reinforcement learning (RL) on reasoning-specific data. This dependence renders supervision-heavy training paradigms increasingly unsustainable, with signs of diminishing scalability already evident in practice. To overcome this limitation, we introduce CPMöbius (CPMobius), a collaborative Coach-Player paradigm for data-free reinforcement learning of reasoning models. Unlike traditional adversarial self-play, CPMöbius, inspired by real world human sports collaboration and multi-agent collaboration, treats the Coach and Player as independent but cooperative roles. The Coach proposes instructions targeted at the Player's capability and receives rewards based on changes in the Player's performance, while the Player is rewarded for solving the increasingly instructive tasks generated by the Coach. This cooperative optimization loop is designed to directly enhance the Player's mathematical reasoning ability. Remarkably, CPMöbius achieves substantial improvement without relying on any external training data, outperforming existing unsupervised approaches. For example, on Qwen2.5-Math-7B-Instruct, our method improves accuracy by an overall average of +4.9 and an out-of-distribution average of +5.4, exceeding RENT by +1.5 on overall accuracy and R-zero by +4.2 on OOD accuracy. 

---
# Where Norms and References Collide: Evaluating LLMs on Normative Reasoning 

**Authors**: Mitchell Abrams, Kaveh Eskandari Miandoab, Felix Gervits, Vasanth Sarathy, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2602.02975)  

**Abstract**: Embodied agents, such as robots, will need to interact in situated environments where successful communication often depends on reasoning over social norms: shared expectations that constrain what actions are appropriate in context. A key capability in such settings is norm-based reference resolution (NBRR), where interpreting referential expressions requires inferring implicit normative expectations grounded in physical and social context. Yet it remains unclear whether Large Language Models (LLMs) can support this kind of reasoning. In this work, we introduce SNIC (Situated Norms in Context), a human-validated diagnostic testbed designed to probe how well state-of-the-art LLMs can extract and utilize normative principles relevant to NBRR. SNIC emphasizes physically grounded norms that arise in everyday tasks such as cleaning, tidying, and serving. Across a range of controlled evaluations, we find that even the strongest LLMs struggle to consistently identify and apply social norms, particularly when norms are implicit, underspecified, or in conflict. These findings reveal a blind spot in current LLMs and highlight a key challenge for deploying language-based systems in socially situated, embodied settings. 

---
# Equal Access, Unequal Interaction: A Counterfactual Audit of LLM Fairness 

**Authors**: Alireza Amiri-Margavi, Arshia Gharagozlou, Amin Gholami Davodi, Seyed Pouyan Mousavi Davoudi, Hamidreza Hasani Balyani  

**Link**: [PDF](https://arxiv.org/pdf/2602.02932)  

**Abstract**: Prior work on fairness in large language models (LLMs) has primarily focused on access-level behaviors such as refusals and safety filtering. However, equitable access does not ensure equitable interaction quality once a response is provided. In this paper, we conduct a controlled fairness audit examining how LLMs differ in tone, uncertainty, and linguistic framing across demographic identities after access is granted. Using a counterfactual prompt design, we evaluate GPT-4 and LLaMA-3.1-70B on career advice tasks while varying identity attributes along age, gender, and nationality. We assess access fairness through refusal analysis and measure interaction quality using automated linguistic metrics, including sentiment, politeness, and hedging. Identity-conditioned differences are evaluated using paired statistical tests. Both models exhibit zero refusal rates across all identities, indicating uniform access. Nevertheless, we observe systematic, model-specific disparities in interaction quality: GPT-4 expresses significantly higher hedging toward younger male users, while LLaMA exhibits broader sentiment variation across identity groups. These results show that fairness disparities can persist at the interaction level even when access is equal, motivating evaluation beyond refusal-based audits. 

---
# HALT: Hallucination Assessment via Log-probs as Time series 

**Authors**: Ahmad Shapiro, Karan Taneja, Ashok Goel  

**Link**: [PDF](https://arxiv.org/pdf/2602.02888)  

**Abstract**: Hallucinations remain a major obstacle for large language models (LLMs), especially in safety-critical domains. We present HALT (Hallucination Assessment via Log-probs as Time series), a lightweight hallucination detector that leverages only the top-20 token log-probabilities from LLM generations as a time series. HALT uses a gated recurrent unit model combined with entropy-based features to learn model calibration bias, providing an extremely efficient alternative to large encoders. Unlike white-box approaches, HALT does not require access to hidden states or attention maps, relying only on output log-probabilities. Unlike black-box approaches, it operates on log-probs rather than surface-form text, which enables stronger domain generalization and compatibility with proprietary LLMs without requiring access to internal weights. To benchmark performance, we introduce HUB (Hallucination detection Unified Benchmark), which consolidates prior datasets into ten capabilities covering both reasoning tasks (Algorithmic, Commonsense, Mathematical, Symbolic, Code Generation) and general purpose skills (Chat, Data-to-Text, Question Answering, Summarization, World Knowledge). While being 30x smaller, HALT outperforms Lettuce, a fine-tuned modernBERT-base encoder, achieving a 60x speedup gain on HUB. HALT and HUB together establish an effective framework for hallucination detection across diverse LLM capabilities. 

---
# Which course? Discourse! Teaching Discourse and Generation in the Era of LLMs 

**Authors**: Junyi Jessy Li, Yang Janet Liu, Kanishka Misra, Valentina Pyatkin, William Sheffield  

**Link**: [PDF](https://arxiv.org/pdf/2602.02878)  

**Abstract**: The field of NLP has undergone vast, continuous transformations over the past few years, sparking debates going beyond discipline boundaries. This begs important questions in education: how do we design courses that bridge sub-disciplines in this shifting landscape? This paper explores this question from the angle of discourse processing, an area with rich linguistic insights and computational models for the intentional, attentional, and coherence structure of language. Discourse is highly relevant for open-ended or long-form text generation, yet this connection is under-explored in existing undergraduate curricula. We present a new course, "Computational Discourse and Natural Language Generation". The course is collaboratively designed by a team with complementary expertise and was offered for the first time in Fall 2025 as an upper-level undergraduate course, cross-listed between Linguistics and Computer Science. Our philosophy is to deeply integrate the theoretical and empirical aspects, and create an exploratory mindset inside the classroom and in the assignments. This paper describes the course in detail and concludes with takeaways from an independent survey as well as our vision for future directions. 

---
# Act or Clarify? Modeling Sensitivity to Uncertainty and Cost in Communication 

**Authors**: Polina Tsvilodub, Karl Mulligan, Todd Snider, Robert D. Hawkins, Michael Franke  

**Link**: [PDF](https://arxiv.org/pdf/2602.02843)  

**Abstract**: When deciding how to act under uncertainty, agents may choose to act to reduce uncertainty or they may act despite that this http URL communicative settings, an important way of reducing uncertainty is by asking clarification questions (CQs). We predict that the decision to ask a CQ depends on both contextual uncertainty and the cost of alternative actions, and that these factors interact: uncertainty should matter most when acting incorrectly is costly. We formalize this interaction in a computational model based on expected regret: how much an agent stands to lose by acting now rather than with full information. We test these predictions in two experiments, one examining purely linguistic responses to questions and another extending to choices between clarification and non-linguistic action. Taken together, our results suggest a rational tradeoff: humans tend to seek clarification proportional to the risk of substantial loss when acting under uncertainty. 

---
# CATNIP: LLM Unlearning via Calibrated and Tokenized Negative Preference Alignment 

**Authors**: Zhengbang Yang, Yisheng Zhong, Junyuan Hong, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02824)  

**Abstract**: Pretrained knowledge memorized in LLMs raises critical concerns over safety and privacy, which has motivated LLM Unlearning as a technique for selectively removing the influences of undesirable knowledge. Existing approaches, rooted in Gradient Ascent (GA), often degrade general domain knowledge while relying on retention data or curated contrastive pairs, which can be either impractical or data and computationally prohibitive. Negative Preference Alignment has been explored for unlearning to tackle the limitations of GA, which, however, remains confined by its choice of reference model and shows undermined performance in realistic data settings. These limitations raise two key questions: i) Can we achieve effective unlearning that quantifies model confidence in undesirable knowledge and uses it to calibrate gradient updates more precisely, thus reducing catastrophic forgetting? ii) Can we make unlearning robust to data scarcity and length variation? We answer both questions affirmatively with CATNIP (Calibrated and Tokenized Negative Preference Alignment), a principled method that rescales unlearning effects in proportion to the model's token-level confidence, thus ensuring fine-grained control over forgetting. Extensive evaluations on MUSE and WMDP benchmarks demonstrated that our work enables effective unlearning without requiring retention data or contrastive unlearning response pairs, with stronger knowledge forgetting and preservation tradeoffs than state-of-the-art methods. 

---
# R2-Router: A New Paradigm for LLM Routing with Reasoning 

**Authors**: Jiaqi Xue, Qian Lou, Jiarong Xing, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02823)  

**Abstract**: As LLMs proliferate with diverse capabilities and costs, LLM routing has emerged by learning to predict each LLM's quality and cost for a given query, then selecting the one with high quality and low cost. However, existing routers implicitly assume a single fixed quality and cost per LLM for each query, ignoring that the same LLM's quality varies with its output length. This causes routers to exclude powerful LLMs when their estimated cost exceeds the budget, missing the opportunity that these LLMs could still deliver high quality at reduced cost with shorter outputs. To address this, we introduce R2-Router, which treats output length budget as a controllable variable and jointly selects the best LLM and length budget, enforcing the budget via length-constrained instructions. This enables R2-Router to discover that a powerful LLM with constrained output can outperform a weaker LLM at comparable cost-efficient configurations invisible to prior methods. Together with the router framework, we construct R2-Bench, the first routing dataset capturing LLM behavior across diverse output length budgets. Experiments show that R2-Router achieves state-of-the-art performance at 4-5x lower cost compared with existing routers. This work opens a new direction: routing as reasoning, where routers evolve from reactive selectors to deliberate reasoners that explore which LLM to use and at what cost budget. 

---
# When Efficient Communication Explains Convexity 

**Authors**: Ashvin Ranjan, Shane Steinert-Threlkeld  

**Link**: [PDF](https://arxiv.org/pdf/2602.02821)  

**Abstract**: Much recent work has argued that the variation in the languages of the world can be explained from the perspective of efficient communication; in particular, languages can be seen as optimally balancing competing pressures to be simple and to be informative. Focusing on the expression of meaning -- semantic typology -- the present paper asks what factors are responsible for successful explanations in terms of efficient communication. Using the Information Bottleneck (IB) approach to formalizing this trade-off, we first demonstrate and analyze a correlation between optimality in the IB sense and a novel generalization of convexity to this setting. In a second experiment, we manipulate various modeling parameters in the IB framework to determine which factors drive the correlation between convexity and optimality. We find that the convexity of the communicative need distribution plays an especially important role. These results move beyond showing that efficient communication can explain aspects of semantic typology into explanations for why that is the case by identifying which underlying factors are responsible. 

---
# AmharicStoryQA: A Multicultural Story Question Answering Benchmark in Amharic 

**Authors**: Israel Abebe Azime, Abenezer Kebede Angamo, Hana Mekonen Tamiru, Dagnachew Mekonnen Marilign, Philipp Slusallek, Seid Muhie Yimam, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2602.02774)  

**Abstract**: With the growing emphasis on multilingual and cultural evaluation benchmarks for large language models, language and culture are often treated as synonymous, and performance is commonly used as a proxy for a models understanding of a given language. In this work, we argue that such evaluations overlook meaningful cultural variation that exists within a single language. We address this gap by focusing on narratives from different regions of Ethiopia and demonstrate that, despite shared linguistic characteristics, region-specific and domain-specific content substantially influences language evaluation outcomes. To this end, we introduce \textbf{\textit{AmharicStoryQA}}, a long-sequence story question answering benchmark grounded in culturally diverse narratives from Amharic-speaking regions. Using this benchmark, we reveal a significant narrative understanding gap in existing LLMs, highlight pronounced regional differences in evaluation results, and show that supervised fine-tuning yields uneven improvements across regions and evaluation settings. Our findings emphasize the need for culturally grounded benchmarks that go beyond language-level evaluation to more accurately assess and improve narrative understanding in low-resource languages. 

---
# From Task Solving to Robust Real-World Adaptation in LLM Agents 

**Authors**: Pouya Pezeshkpour, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2602.02760)  

**Abstract**: Large language models are increasingly deployed as specialized agents that plan, call tools, and take actions over extended horizons. Yet many existing evaluations assume a "clean interface" where dynamics are specified and stable, tools and sensors are reliable, and success is captured by a single explicit objective-often overestimating real-world readiness. In practice, agents face underspecified rules, unreliable signals, shifting environments, and implicit, multi-stakeholder goals. The challenge is therefore not just solving tasks, but adapting while solving: deciding what to trust, what is wanted, when to verify, and when to fall back or escalate. We stress-test deployment-relevant robustness under four operational circumstances: partial observability, dynamic environments, noisy signals, and dynamic agent state. We benchmark agentic LLMs in a grid-based game with a simple goal but long-horizon execution. Episodes violate clean-interface assumptions yet remain solvable, forcing agents to infer rules, pay for information, adapt to environmental and internal shifts, and act cautiously under noise. Across five state-of-the-art LLM agents, we find large gaps between nominal task-solving and deployment-like robustness. Performance generally degrades as grid size and horizon increase, but rankings are unstable: weaker models can beat stronger ones when strategy matches the uncertainty regime. Despite no explicit instruction, agents trade off completion, efficiency, and penalty avoidance, suggesting partial objective inference. Ablations and feature analyses reveal model-specific sensitivities and failure drivers, motivating work on verification, safe action selection, and objective inference under partial observability, noise, and non-stationarity. 

---
# Time-Critical Multimodal Medical Transportation: Organs, Patients, and Medical Supplies 

**Authors**: Elaheh Sabziyan Varnousfaderani, Syed A. M. Shihab, Mohammad Taghizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2602.02736)  

**Abstract**: Timely transportation of organs, patients, and medical supplies is critical to modern healthcare, particularly in emergencies and transplant scenarios where even short delays can severely impact outcomes. Traditional ground-based vehicles such as ambulances are often hindered by traffic congestion; while air vehicles such as helicopters are faster but costly. Emerging air vehicles -- Unmanned Aerial Vehicles and electric vertical take-off and landing aircraft -- have lower operating costs, but remain limited by range and susceptibility to weather conditions. A multimodal transportation system that integrates both air and ground vehicles can leverage the strengths of each to enhance overall transportation efficiency. This study introduces a constructive greedy heuristic algorithm for multimodal vehicle dispatching for medical transportation. Four different fleet configurations were tested: (i) ambulances only, (ii) ambulances with Unmanned Aerial Vehicles, (iii) ambulances with electric vertical take-off and landing aircraft, and (iv) a fully integrated fleet of ambulances, Unmanned Aerial Vehicles, and electric vertical take-off and landing aircraft. The algorithm incorporates payload consolidation across compatible routes, accounts for traffic congestion in ground operations and weather conditions in aerial operations, while enabling rapid vehicle dispatching compared to computationally intensive optimization models. Using a common set of conditions, we evaluate all four fleet types to identify the most effective configurations for fulfilling medical transportation needs while minimizing operating costs, recharging/fuel costs, and total transportation time. 

---
# Predicting first-episode homelessness among US Veterans using longitudinal EHR data: time-varying models and social risk factors 

**Authors**: Rohan Pandey, Haijuan Yan, Hong Yu, Jack Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2602.02731)  

**Abstract**: Homelessness among US veterans remains a critical public health challenge, yet risk prediction offers a pathway for proactive intervention. In this retrospective prognostic study, we analyzed electronic health record (EHR) data from 4,276,403 Veterans Affairs patients during a 2016 observation period to predict first-episode homelessness occurring 3-12 months later in 2017 (prevalence: 0.32-1.19%). We constructed static and time-varying EHR representations, utilizing clinician-informed logic to model the persistence of clinical conditions and social risks over time. We then compared the performance of classical machine learning, transformer-based masked language models, and fine-tuned large language models (LLMs). We demonstrate that incorporating social and behavioral factors into longitudinal models improved precision-recall area under the curve (PR-AUC) by 15-30%. In the top 1% risk tier, models yielded positive predictive values ranging from 3.93-4.72% at 3 months, 7.39-8.30% at 6 months, 9.84-11.41% at 9 months, and 11.65-13.80% at 12 months across model architectures. Large language models underperformed encoder-based models on discrimination but showed smaller performance disparities across racial groups. These results demonstrate that longitudinal, socially informed EHR modeling concentrates homelessness risk into actionable strata, enabling targeted and data-informed prevention strategies for at-risk veterans. 

---
# InfMem: Learning System-2 Memory Control for Long-Context Agent 

**Authors**: Xinyu Wang, Mingze Li, Peng Lu, Xiao-Wen Chang, Lifeng Shang, Jinping Li, Fei Mi, Prasanna Parthasarathi, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2602.02704)  

**Abstract**: Reasoning over ultra-long documents requires synthesizing sparse evidence scattered across distant segments under strict memory constraints. While streaming agents enable scalable processing, their passive memory update strategy often fails to preserve low-salience bridging evidence required for multi-hop reasoning. We propose InfMem, a control-centric agent that instantiates System-2-style control via a PreThink-Retrieve-Write protocol. InfMem actively monitors evidence sufficiency, performs targeted in-document retrieval, and applies evidence-aware joint compression to update a bounded memory. To ensure reliable control, we introduce a practical SFT-to-RL training recipe that aligns retrieval, writing, and stopping decisions with end-task correctness. On ultra-long QA benchmarks from 32k to 1M tokens, InfMem consistently outperforms MemAgent across backbones. Specifically, InfMem improves average absolute accuracy by +10.17, +11.84, and +8.23 points on Qwen3-1.7B, Qwen3-4B, and Qwen2.5-7B, respectively, while reducing inference time by $3.9\times$ on average (up to $5.1\times$) via adaptive early stopping. 

---
# Monotonicity as an Architectural Bias for Robust Language Models 

**Authors**: Patrick Cooper, Alireza Nadali, Ashutosh Trivedi, Alvaro Velasquez  

**Link**: [PDF](https://arxiv.org/pdf/2602.02686)  

**Abstract**: Large language models (LLMs) are known to exhibit brittle behavior under adversarial prompts and jailbreak attacks, even after extensive alignment and fine-tuning. This fragility reflects a broader challenge of modern neural language models: small, carefully structured perturbations in high-dimensional input spaces can induce large and unpredictable changes in internal semantic representations and output.
We investigate monotonicity as an architectural inductive bias for improving the robustness of Transformer-based language models. Monotonicity constrains semantic transformations so that strengthening information, evidence, or constraints cannot lead to regressions in the corresponding internal representations. Such order-preserving behavior has long been exploited in control and safety-critical systems to simplify reasoning and improve robustness, but has traditionally been viewed as incompatible with the expressivity required by neural language models.
We show that this trade-off is not inherent. By enforcing monotonicity selectively in the feed-forward sublayers of sequence-to-sequence Transformers -- while leaving attention mechanisms unconstrained -- we obtain monotone language models that preserve the performance of their pretrained counterparts. This architectural separation allows negation, contradiction, and contextual interactions to be introduced explicitly through attention, while ensuring that subsequent semantic refinement is order-preserving. Empirically, monotonicity substantially improves robustness: adversarial attack success rates drop from approximately 69% to 19%, while standard summarization performance degrades only marginally. 

---
# WideSeek: Advancing Wide Research via Multi-Agent Scaling 

**Authors**: Ziyang Huang, Haolin Ren, Xiaowei Yuan, Jiawei Wang, Zhongtao Jiang, Kun Xu, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02636)  

**Abstract**: Search intelligence is evolving from Deep Research to Wide Research, a paradigm essential for retrieving and synthesizing comprehensive information under complex constraints in parallel. However, progress in this field is impeded by the lack of dedicated benchmarks and optimization methodologies for search breadth. To address these challenges, we take a deep dive into Wide Research from two perspectives: Data Pipeline and Agent Optimization. First, we produce WideSeekBench, a General Broad Information Seeking (GBIS) benchmark constructed via a rigorous multi-phase data pipeline to ensure diversity across the target information volume, logical constraints, and domains. Second, we introduce WideSeek, a dynamic hierarchical multi-agent architecture that can autonomously fork parallel sub-agents based on task requirements. Furthermore, we design a unified training framework that linearizes multi-agent trajectories and optimizes the system using end-to-end RL. Experimental results demonstrate the effectiveness of WideSeek and multi-agent RL, highlighting that scaling the number of agents is a promising direction for advancing the Wide Research paradigm. 

---
# Graph-Augmented Reasoning with Large Language Models for Tobacco Pest and Disease Management 

**Authors**: Siyu Li, Chenwei Song, Qi Zhou, Wan Zhou, Xinyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02635)  

**Abstract**: This paper proposes a graph-augmented reasoning framework for tobacco pest and disease management that integrates structured domain knowledge into large language models. Building on GraphRAG, we construct a domain-specific knowledge graph and retrieve query-relevant subgraphs to provide relational evidence during answer generation. The framework adopts ChatGLM as the Transformer backbone with LoRA-based parameter-efficient fine-tuning, and employs a graph neural network to learn node representations that capture symptom-disease-treatment dependencies. By explicitly modeling diseases, symptoms, pesticides, and control measures as linked entities, the system supports evidence-aware retrieval beyond surface-level text similarity. Retrieved graph evidence is incorporated into the LLM input to guide generation toward domain-consistent recommendations and to mitigate hallucinated or inappropriate treatments. Experimental results show consistent improvements over text-only baselines, with the largest gains observed on multi-hop and comparative reasoning questions that require chaining multiple relations. 

---
# ROSA-Tuning: Enhancing Long-Context Modeling via Suffix Matching 

**Authors**: Yunao Zheng, Xiaojie Wang, Lei Ren, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.02499)  

**Abstract**: Long-context capability and computational efficiency are among the central challenges facing today's large language models. Existing efficient attention methods reduce computational complexity, but they typically suffer from a limited coverage of the model state. This paper proposes ROSA-Tuning, a retrieval-and-recall mechanism for enhancing the long-context modeling ability of pretrained models. Beyond the standard attention mechanism, ROSA-Tuning introduces in parallel a CPU-based ROSA (RWKV Online Suffix Automaton) retrieval module, which efficiently locates historical positions in long contexts that are relevant to the current query, and injects the retrieved information into the model state in a trainable manner; subsequent weighted fusion can then be handled by range-restricted attention. To enable end-to-end training, we design a binary discretization strategy and a counterfactual gradient algorithm, and further optimize overall execution efficiency via an asynchronous CPU-GPU pipeline. Systematic evaluations on Qwen3-Base-1.7B show that ROSA-Tuning substantially restores the long-context modeling ability of windowed-attention models, achieving performance close to and in some cases matching global attention on benchmarks such as LongBench, while maintaining computational efficiency and GPU memory usage that are nearly comparable to windowed-attention methods, offering a new technical path for efficient long-context processing. The example code can be found at this https URL. 

---
# Test-Time Detoxification without Training or Learning Anything 

**Authors**: Baturay Saglam, Dionysis Kalogerias  

**Link**: [PDF](https://arxiv.org/pdf/2602.02498)  

**Abstract**: Large language models can produce toxic or inappropriate text even for benign inputs, creating risks when deployed at scale. Detoxification is therefore important for safety and user trust, particularly when we want to reduce harmful content without sacrificing the model's generation quality. Many existing approaches rely on model retraining, gradients, or learned auxiliary components, which can be costly and may not transfer across model families or to truly black-box settings. We introduce a test-time procedure that approximates the gradient of completion toxicity with respect to the input embeddings and uses a small number of descent steps to steer generation toward less toxic continuations. This is achieved with zeroth-order optimization that requires only access to input embeddings, a toxicity scoring function, and forward evaluations of the model. Empirically, the approach delivers robust toxicity reductions across models and prompts and, in most settings, achieves the best overall toxicity-quality trade-off. More broadly, our work positions word embeddings as effective control variables and encourages wider use of black-box optimization to guide autoregressive language models toward scalable, safer text generation, without requiring any training or access to intermediate computations. 

---
# STEMVerse: A Dual-Axis Diagnostic Framework for STEM Reasoning in Large Language Models 

**Authors**: Xuzhao Li, Xuchen Li, Jian Zhao, Shiyu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.02497)  

**Abstract**: As Large Language Models (LLMs) achieve significant breakthroughs in complex reasoning tasks, evaluating their proficiency in science, technology, engineering, and mathematics (STEM) has become a primary method for measuring machine intelligence. However, current evaluation paradigms often treat benchmarks as isolated "silos," offering only monolithic aggregate scores that neglect the intricacies of both academic specialization and cognitive depth. This result-oriented approach fails to distinguish whether model errors stem from insufficient domain knowledge or deficiencies in cognitive capacity, thereby limiting the diagnostic value. To address this, we propose STEMVerse, a diagnostic framework designed to systematically analyze the STEM reasoning capabilities of LLMs. This framework characterizes model performance across academic specialization and cognitive complexity to map the capability required for reasoning. We re-aggregate over 20,000 STEM problems from mainstream benchmarks into a unified "Discipline $\times$ Cognition" capability space, assigning dual-axis labels to every instance. Utilizing this unified diagnostic framework, we systematically evaluate representative LLM families across varying parameter scales and training paradigms. Our empirical results reveal structural failure patterns in STEM reasoning. By integrating multi-disciplinary coverage and fine-grained cognitive stratification into a unified framework, STEMVerse provides a clear and actionable perspective for understanding the scientific reasoning characteristics of LLMs. 

---
# The Hypocrisy Gap: Quantifying Divergence Between Internal Belief and Chain-of-Thought Explanation via Sparse Autoencoders 

**Authors**: Shikhar Shiromani, Archie Chaudhury, Sri Pranav Kunda  

**Link**: [PDF](https://arxiv.org/pdf/2602.02496)  

**Abstract**: Large Language Models (LLMs) frequently exhibit unfaithful behavior, producing a final answer that differs significantly from their internal chain of thought (CoT) reasoning in order to appease the user they are conversing with. In order to better detect this behavior, we introduce the Hypocrisy Gap, a mechanistic metric utilizing Sparse Autoencoders (SAEs) to quantify the divergence between a model's internal reasoning and its final generation. By mathematically comparing an internal truth belief, derived via sparse linear probes, to the final generated trajectory in latent space, we quantify and detect a model's tendency to engage in unfaithful behavior. Experiments on Gemma, Llama, and Qwen models using Anthropic's Sycophancy benchmark show that our method achieves an AUROC of 0.55-0.73 for detecting sycophantic runs and 0.55-0.74 for hypocritical cases where the model internally "knows" the user is wrong, consistently outperforming a decision-aligned log-probability baseline (0.41-0.50 AUROC). 

---
# AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations 

**Authors**: Minjun Zhu, Zhen Lin, Yixuan Weng, Panzhong Lu, Qiujie Xie, Yifan Wei, Sifan Liu, Qiyao Sun, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03828)  

**Abstract**: High-quality scientific illustrations are crucial for effectively communicating complex scientific and technical concepts, yet their manual creation remains a well-recognized bottleneck in both academia and industry. We present FigureBench, the first large-scale benchmark for generating scientific illustrations from long-form scientific texts. It contains 3,300 high-quality scientific text-figure pairs, covering diverse text-to-illustration tasks from scientific papers, surveys, blogs, and textbooks. Moreover, we propose AutoFigure, the first agentic framework that automatically generates high-quality scientific illustrations based on long-form scientific text. Specifically, before rendering the final result, AutoFigure engages in extensive thinking, recombination, and validation to produce a layout that is both structurally sound and aesthetically refined, outputting a scientific illustration that achieves both structural completeness and aesthetic appeal. Leveraging the high-quality data from FigureBench, we conduct extensive experiments to test the performance of AutoFigure against various baseline methods. The results demonstrate that AutoFigure consistently surpasses all baseline methods, producing publication-ready scientific illustrations. The code, dataset and huggingface space are released in this https URL. 

---
# Antidistillation Fingerprinting 

**Authors**: Yixuan Even Xu, John Kirchenbauer, Yash Savani, Asher Trockman, Alexander Robey, Tom Goldstein, Fei Fang, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2602.03812)  

**Abstract**: Model distillation enables efficient emulation of frontier large language models (LLMs), creating a need for robust mechanisms to detect when a third-party student model has trained on a teacher model's outputs. However, existing fingerprinting techniques that could be used to detect such distillation rely on heuristic perturbations that impose a steep trade-off between generation quality and fingerprinting strength, often requiring significant degradation of utility to ensure the fingerprint is effectively internalized by the student. We introduce antidistillation fingerprinting (ADFP), a principled approach that aligns the fingerprinting objective with the student's learning dynamics. Building upon the gradient-based framework of antidistillation sampling, ADFP utilizes a proxy model to identify and sample tokens that directly maximize the expected detectability of the fingerprint in the student after fine-tuning, rather than relying on the incidental absorption of the un-targeted biases of a more naive watermark. Experiments on GSM8K and OASST1 benchmarks demonstrate that ADFP achieves a significant Pareto improvement over state-of-the-art baselines, yielding stronger detection confidence with minimal impact on utility, even when the student model's architecture is unknown. 

---
# Bridging Online and Offline RL: Contextual Bandit Learning for Multi-Turn Code Generation 

**Authors**: Ziru Chen, Dongdong Chen, Ruinan Jin, Yingbin Liang, Yujia Xie, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.03806)  

**Abstract**: Recently, there have been significant research interests in training large language models (LLMs) with reinforcement learning (RL) on real-world tasks, such as multi-turn code generation. While online RL tends to perform better than offline RL, its higher training cost and instability hinders wide adoption. In this paper, we build on the observation that multi-turn code generation can be formulated as a one-step recoverable Markov decision process and propose contextual bandit learning with offline trajectories (Cobalt), a new method that combines the benefits of online and offline RL. Cobalt first collects code generation trajectories using a reference LLM and divides them into partial trajectories as contextual prompts. Then, during online bandit learning, the LLM is trained to complete each partial trajectory prompt through single-step code generation. Cobalt outperforms two multi-turn online RL baselines based on GRPO and VeRPO, and substantially improves R1-Distill 8B and Qwen3 8B by up to 9.0 and 6.2 absolute Pass@1 scores on LiveCodeBench. Also, we analyze LLMs' in-context reward hacking behaviors and augment Cobalt training with perturbed trajectories to mitigate this issue. Overall, our results demonstrate Cobalt as a promising solution for iterative decision-making tasks like multi-turn code generation. Our code and data are available at this https URL. 

---
# FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation 

**Authors**: Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.03798)  

**Abstract**: Assisting non-expert users to develop complex interactive websites has become a popular task for LLM-powered code agents. However, existing code agents tend to only generate frontend web pages, masking the lack of real full-stack data processing and storage with fancy visual effects. Notably, constructing production-level full-stack web applications is far more challenging than only generating frontend web pages, demanding careful control of data flow, comprehensive understanding of constantly updating packages and dependencies, and accurate localization of obscure bugs in the codebase. To address these difficulties, we introduce FullStack-Agent, a unified agent system for full-stack agentic coding that consists of three parts: (1) FullStack-Dev, a multi-agent framework with strong planning, code editing, codebase navigation, and bug localization abilities. (2) FullStack-Learn, an innovative data-scaling and self-improving method that back-translates crawled and synthesized website repositories to improve the backbone LLM of FullStack-Dev. (3) FullStack-Bench, a comprehensive benchmark that systematically tests the frontend, backend and database functionalities of the generated website. Our FullStack-Dev outperforms the previous state-of-the-art method by 8.7%, 38.2%, and 15.9% on the frontend, backend, and database test cases respectively. Additionally, FullStack-Learn raises the performance of a 30B model by 9.7%, 9.5%, and 2.8% on the three sets of test cases through self-improvement, demonstrating the effectiveness of our approach. The code is released at this https URL. 

---
# WebSentinel: Detecting and Localizing Prompt Injection Attacks for Web Agents 

**Authors**: Xilong Wang, Yinuo Liu, Zhun Wang, Dawn Song, Neil Gong  

**Link**: [PDF](https://arxiv.org/pdf/2602.03792)  

**Abstract**: Prompt injection attacks manipulate webpage content to cause web agents to execute attacker-specified tasks instead of the user's intended ones. Existing methods for detecting and localizing such attacks achieve limited effectiveness, as their underlying assumptions often do not hold in the web-agent setting. In this work, we propose WebSentinel, a two-step approach for detecting and localizing prompt injection attacks in webpages. Given a webpage, Step I extracts \emph{segments of interest} that may be contaminated, and Step II evaluates each segment by checking its consistency with the webpage content as context. We show that WebSentinel is highly effective, substantially outperforming baseline methods across multiple datasets of both contaminated and clean webpages that we collected. Our code is available at: this https URL. 

---
# AOrchestra: Automating Sub-Agent Creation for Agentic Orchestration 

**Authors**: Jianhao Ruan, Zhihao Xu, Yiran Peng, Fashen Ren, Zhaoyang Yu, Xinbing Liang, Jinyu Xiang, Bang Liu, Chenglin Wu, Yuyu Luo, Jiayi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03786)  

**Abstract**: Language agents have shown strong promise for task automation. Realizing this promise for increasingly complex, long-horizon tasks has driven the rise of a sub-agent-as-tools paradigm for multi-turn task solving. However, existing designs still lack a dynamic abstraction view of sub-agents, thereby hurting adaptability. We address this challenge with a unified, framework-agnostic agent abstraction that models any agent as a tuple Instruction, Context, Tools, Model. This tuple acts as a compositional recipe for capabilities, enabling the system to spawn specialized executors for each task on demand. Building on this abstraction, we introduce an agentic system AOrchestra, where the central orchestrator concretizes the tuple at each step: it curates task-relevant context, selects tools and models, and delegates execution via on-the-fly automatic agent creation. Such designs enable reducing human engineering efforts, and remain framework-agnostic with plug-and-play support for diverse agents as task executors. It also enables a controllable performance-cost trade-off, allowing the system to approach Pareto-efficient. Across three challenging benchmarks (GAIA, SWE-Bench, Terminal-Bench), AOrchestra achieves 16.28% relative improvement against the strongest baseline when paired with Gemini-3-Flash. The code is available at: this https URL 

---
# Efficient Estimation of Kernel Surrogate Models for Task Attribution 

**Authors**: Zhenshuo Zhang, Minxuan Duan, Hongyang R. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03783)  

**Abstract**: Modern AI agents such as large language models are trained on diverse tasks -- translation, code generation, mathematical reasoning, and text prediction -- simultaneously. A key question is to quantify how each individual training task influences performance on a target task, a problem we refer to as task attribution. The direct approach, leave-one-out retraining, measures the effect of removing each task, but is computationally infeasible at scale. An alternative approach that builds surrogate models to predict a target task's performance for any subset of training tasks has emerged in recent literature. Prior work focuses on linear surrogate models, which capture first-order relationships, but miss nonlinear interactions such as synergy, antagonism, or XOR-type effects. In this paper, we first consider a unified task weighting framework for analyzing task attribution methods, and show a new connection between linear surrogate models and influence functions through a second-order analysis. Then, we introduce kernel surrogate models, which more effectively represent second-order task interactions. To efficiently learn the kernel surrogate, we develop a gradient-based estimation procedure that leverages a first-order approximation of pretrained models; empirically, this yields accurate estimates with less than $2\%$ relative error without repeated retraining. Experiments across multiple domains -- including math reasoning in transformers, in-context learning, and multi-objective reinforcement learning -- demonstrate the effectiveness of kernel surrogate models. They achieve a $25\%$ higher correlation with the leave-one-out ground truth than linear surrogates and influence-function baselines. When used for downstream task selection, kernel surrogate models yield a $40\%$ improvement in demonstration selection for in-context learning and multi-objective reinforcement learning benchmarks. 

---
# Conflict-Resolving and Sharpness-Aware Minimization for Generalized Knowledge Editing with Multiple Updates 

**Authors**: Duy Nguyen, Hanqi Xiao, Archiki Prasad, Elias Stengel-Eskin, Hyunji Lee, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2602.03696)  

**Abstract**: Large language models (LLMs) rely on internal knowledge to solve many downstream tasks, making it crucial to keep them up to date. Since full retraining is expensive, prior work has explored efficient alternatives such as model editing and parameter-efficient fine-tuning. However, these approaches often break down in practice due to poor generalization across inputs, limited stability, and knowledge conflict. To address these limitations, we propose the CoRSA (Conflict-Resolving and Sharpness-Aware Minimization) training framework, a parameter-efficient, holistic approach for knowledge editing with multiple updates. CoRSA tackles multiple challenges simultaneously: it improves generalization to different input forms and enhances stability across multiple updates by minimizing loss curvature, and resolves conflicts by maximizing the margin between new and prior knowledge. Across three widely used fact editing benchmarks, CoRSA achieves significant gains in generalization, outperforming baselines with average absolute improvements of 12.42% over LoRA and 10% over model editing methods. With multiple updates, it maintains high update efficacy while reducing catastrophic forgetting by 27.82% compared to LoRA. CoRSA also generalizes to the code domain, outperforming the strongest baseline by 5.48% Pass@5 in update efficacy. 

---
# Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems 

**Authors**: Haibo Jin, Kuang Peng, Ye Yu, Xiaopeng Yuan, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03695)  

**Abstract**: While existing multi-agent systems (MAS) can handle complex problems by enabling collaboration among multiple agents, they are often highly task-specific, relying on manually crafted agent roles and interaction prompts, which leads to increased architectural complexity and limited reusability across tasks. Moreover, most MAS communicate primarily through natural language, making them vulnerable to error accumulation and instability in long-context, multi-stage interactions within internal agent histories.
In this work, we propose \textbf{Agent Primitives}, a set of reusable latent building blocks for LLM-based MAS. Inspired by neural network design, where complex models are built from reusable components, we observe that many existing MAS architectures can be decomposed into a small number of recurring internal computation patterns. Based on this observation, we instantiate three primitives: Review, Voting and Selection, and Planning and Execution. All primitives communicate internally via key-value (KV) cache, which improves both robustness and efficiency by mitigating information degradation across multi-stage interactions. To enable automatic system construction, an Organizer agent selects and composes primitives for each query, guided by a lightweight knowledge pool of previously successful configurations, forming a primitive-based MAS.
Experiments show that primitives-based MAS improve average accuracy by 12.0-16.5\% over single-agent baselines, reduce token usage and inference latency by approximately 3$\times$-4$\times$ compared to text-based MAS, while incurring only 1.3$\times$-1.6$\times$ overhead relative to single-agent inference and providing more stable performance across model backbones. 

---
# Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration 

**Authors**: Bowei He, Minda Hu, Zenan Xu, Hongru Wang, Licheng Zong, Yankai Chen, Chen Ma, Xue Liu, Pluto Zhou, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2602.03647)  

**Abstract**: Search-integrated reasoning enables language agents to transcend static parametric knowledge by actively querying external sources. However, training these agents via reinforcement learning is hindered by the multi-scale credit assignment problem: existing methods typically rely on sparse, trajectory-level rewards that fail to distinguish between high-quality reasoning and fortuitous guesses, leading to redundant or misleading search behaviors. To address this, we propose Search-R2, a novel Actor-Refiner collaboration framework that enhances reasoning through targeted intervention, with both components jointly optimized during training. Our approach decomposes the generation process into an Actor, which produces initial reasoning trajectories, and a Meta-Refiner, which selectively diagnoses and repairs flawed steps via a 'cut-and-regenerate' mechanism. To provide fine-grained supervision, we introduce a hybrid reward design that couples outcome correctness with a dense process reward quantifying the information density of retrieved evidence. Theoretically, we formalize the Actor-Refiner interaction as a smoothed mixture policy, proving that selective correction yields strict performance gains over strong baselines. Extensive experiments across various general and multi-hop QA datasets demonstrate that Search-R2 consistently outperforms strong RAG and RL-based baselines across model scales, achieving superior reasoning accuracy with minimal overhead. 

---
# Tutorial on Reasoning for IR & IR for Reasoning 

**Authors**: Mohanna Hoveyda, Panagiotis Efstratiadis, Arjen de Vries, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2602.03640)  

**Abstract**: Information retrieval has long focused on ranking documents by semantic relatedness. Yet many real-world information needs demand more: enforcement of logical constraints, multi-step inference, and synthesis of multiple pieces of evidence. Addressing these requirements is, at its core, a problem of reasoning. Across AI communities, researchers are developing diverse solutions for the problem of reasoning, from inference-time strategies and post-training of LLMs, to neuro-symbolic systems, Bayesian and probabilistic frameworks, geometric representations, and energy-based models. These efforts target the same problem: to move beyond pattern-matching systems toward structured, verifiable inference. However, they remain scattered across disciplines, making it difficult for IR researchers to identify the most relevant ideas and opportunities. To help navigate the fragmented landscape of research in reasoning, this tutorial first articulates a working definition of reasoning within the context of information retrieval and derives from it a unified analytical framework. The framework maps existing approaches along axes that reflect the core components of the definition. By providing a comprehensive overview of recent approaches and mapping current methods onto the defined axes, we expose their trade-offs and complementarities, highlight where IR can benefit from cross-disciplinary advances, and illustrate how retrieval process itself can play a central role in broader reasoning systems. The tutorial will equip participants with both a conceptual framework and practical guidance for enhancing reasoning-capable IR systems, while situating IR as a domain that both benefits and contributes to the broader development of reasoning methodologies. 

---
# When Single Answer Is Not Enough: Rethinking Single-Step Retrosynthesis Benchmarks for LLMs 

**Authors**: Bogdan Zagribelnyy, Ivan Ilin, Maksim Kuznetsov, Nikita Bondarev, Roman Schutski, Thomas MacDougall, Rim Shayakhmetov, Zulfat Miftakhutdinov, Mikolaj Mizera, Vladimir Aladinskiy, Alex Aliper, Alex Zhavoronkov  

**Link**: [PDF](https://arxiv.org/pdf/2602.03554)  

**Abstract**: Recent progress has expanded the use of large language models (LLMs) in drug discovery, including synthesis planning. However, objective evaluation of retrosynthesis performance remains limited. Existing benchmarks and metrics typically rely on published synthetic procedures and Top-K accuracy based on single ground-truth, which does not capture the open-ended nature of real-world synthesis planning. We propose a new benchmarking framework for single-step retrosynthesis that evaluates both general-purpose and chemistry-specialized LLMs using ChemCensor, a novel metric for chemical plausibility. By emphasizing plausibility over exact match, this approach better aligns with human synthesis planning practices. We also introduce CREED, a novel dataset comprising millions of ChemCensor-validated reaction records for LLM training, and use it to train a model that improves over the LLM baselines under this benchmark. 

---
# Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance 

**Authors**: Yingjie Zhu, Xuefeng Bai, Kehai Chen, Yang Xiang, Youcheng Pan, Xiaoqiang Zhou, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03491)  

**Abstract**: Reasoning over table images remains challenging for Large Vision-Language Models (LVLMs) due to complex layouts and tightly coupled structure-content information. Existing solutions often depend on expensive supervised training, reinforcement learning, or external tools, limiting efficiency and scalability. This work addresses a key question: how to adapt LVLMs to table reasoning with minimal annotation and no external tools? Specifically, we first introduce DiSCo, a Disentangled Structure-Content alignment framework that explicitly separates structural abstraction from semantic grounding during multimodal alignment, efficiently adapting LVLMs to tables structures. Building on DiSCo, we further present Table-GLS, a Global-to-Local Structure-guided reasoning framework that performs table reasoning via structured exploration and evidence-grounded inference. Extensive experiments across diverse benchmarks demonstrate that our framework efficiently enhances LVLM's table understanding and reasoning capabilities, particularly generalizing to unseen table structures. 

---
# DiscoverLLM: From Executing Intents to Discovering Them 

**Authors**: Tae Soo Kim, Yoonjoo Lee, Jaesang Yu, John Joon Young Chung, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.03429)  

**Abstract**: To handle ambiguous and open-ended requests, Large Language Models (LLMs) are increasingly trained to interact with users to surface intents they have not yet expressed (e.g., ask clarification questions). However, users are often ambiguous because they have not yet formed their intents: they must observe and explore outcomes to discover what they want. Simply asking "what kind of tone do you want?" fails when users themselves do not know. We introduce DiscoverLLM, a novel and generalizable framework that trains LLMs to help users form and discover their intents. Central to our approach is a novel user simulator that models cognitive state with a hierarchy of intents that progressively concretize as the model surfaces relevant options -- where the degree of concretization serves as a reward signal that models can be trained to optimize. Resulting models learn to collaborate with users by adaptively diverging (i.e., explore options) when intents are unclear, and converging (i.e., refine and implement) when intents concretize. Across proposed interactive benchmarks in creative writing, technical writing, and SVG drawing, DiscoverLLM achieves over 10% higher task performance while reducing conversation length by up to 40%. In a user study with 75 human participants, DiscoverLLM improved conversation satisfaction and efficiency compared to baselines. 

---
# SWE-World: Building Software Engineering Agents in Docker-Free Environments 

**Authors**: Shuang Sun, Huatong Song, Lisheng Huang, Jinhao Jiang, Ran Le, Zhihao Lv, Zongchao Chen, Yiwen Hu, Wenyang Luo, Wayne Xin Zhao, Yang Song, Hongteng Xu, Tao Zhang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03419)  

**Abstract**: Recent advances in large language models (LLMs) have enabled software engineering agents to tackle complex code modification tasks. Most existing approaches rely on execution feedback from containerized environments, which require dependency-complete setup and physical execution of programs and tests. While effective, this paradigm is resource-intensive and difficult to maintain, substantially complicating agent training and limiting scalability. We propose SWE-World, a Docker-free framework that replaces physical execution environments with a learned surrogate for training and evaluating software engineering agents. SWE-World leverages LLM-based models trained on real agent-environment interaction data to predict intermediate execution outcomes and final test feedback, enabling agents to learn without interacting with physical containerized environments. This design preserves the standard agent-environment interaction loop while eliminating the need for costly environment construction and maintenance during agent optimization and evaluation. Furthermore, because SWE-World can simulate the final evaluation outcomes of candidate trajectories without real submission, it enables selecting the best solution among multiple test-time attempts, thereby facilitating effective test-time scaling (TTS) in software engineering tasks. Experiments on SWE-bench Verified demonstrate that SWE-World raises Qwen2.5-Coder-32B from 6.2\% to 52.0\% via Docker-free SFT, 55.0\% with Docker-free RL, and 68.2\% with further TTS. The code is available at this https URL 

---
# SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training 

**Authors**: Huatong Song, Lisheng Huang, Shuang Sun, Jinhao Jiang, Ran Le, Daixuan Cheng, Guoxin Chen, Yiwen Hu, Zongchao Chen, Wayne Xin Zhao, Yang Song, Tao Zhang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03411)  

**Abstract**: In this technical report, we present SWE-Master, an open-source and fully reproducible post-training framework for building effective software engineering agents. SWE-Master systematically explores the complete agent development pipeline, including teacher-trajectory synthesis and data curation, long-horizon SFT, RL with real execution feedback, and inference framework design. Starting from an open-source base model with limited initial SWE capability, SWE-Master demonstrates how systematical optimization method can elicit strong long-horizon SWE task solving abilities. We evaluate SWE-Master on SWE-bench Verified, a standard benchmark for realistic software engineering tasks. Under identical experimental settings, our approach achieves a resolve rate of 61.4\% with Qwen2.5-Coder-32B, substantially outperforming existing open-source baselines. By further incorporating test-time scaling~(TTS) with LLM-based environment feedback, SWE-Master reaches 70.8\% at TTS@8, demonstrating a strong performance potential. SWE-Master provides a practical and transparent foundation for advancing reproducible research on software engineering agents. The code is available at this https URL. 

---
# MeKi: Memory-based Expert Knowledge Injection for Efficient LLM Scaling 

**Authors**: Ning Ding, Fangcheng Liu, Kyungrae Kim, Linji Hao, Kyeng-Hun Lee, Hyeonmok Ko, Yehui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03359)  

**Abstract**: Scaling Large Language Models (LLMs) typically relies on increasing the number of parameters or test-time computations to boost performance. However, these strategies are impractical for edge device deployment due to limited RAM and NPU resources. Despite hardware constraints, deploying performant LLM on edge devices such as smartphone remains crucial for user experience. To address this, we propose MeKi (Memory-based Expert Knowledge Injection), a novel system that scales LLM capacity via storage space rather than FLOPs. MeKi equips each Transformer layer with token-level memory experts that injects pre-stored semantic knowledge into the generation process. To bridge the gap between training capacity and inference efficiency, we employ a re-parameterization strategy to fold parameter matrices used during training into a compact static lookup table. By offloading the knowledge to ROM, MeKi decouples model capacity from computational cost, introducing zero inference latency overhead. Extensive experiments demonstrate that MeKi significantly outperforms dense LLM baselines with identical inference speed, validating the effectiveness of memory-based scaling paradigm for on-device LLMs. Project homepage is at this https URL. 

---
# GFlowPO: Generative Flow Network as a Language Model Prompt Optimizer 

**Authors**: Junmo Cho, Suhan Kim, Sangjune An, Minsu Kim, Dong Bok Lee, Heejun Lee, Sung Ju Hwang, Hae Beom Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.03358)  

**Abstract**: Finding effective prompts for language models (LMs) is critical yet notoriously difficult: the prompt space is combinatorially large, rewards are sparse due to expensive target-LM evaluation. Yet, existing RL-based prompt optimizers often rely on on-policy updates and a meta-prompt sampled from a fixed distribution, leading to poor sample efficiency. We propose GFlowPO, a probabilistic prompt optimization framework that casts prompt search as a posterior inference problem over latent prompts regularized by a meta-prompted reference-LM prior. In the first step, we fine-tune a lightweight prompt-LM with an off-policy Generative Flow Network (GFlowNet) objective, using a replay-based training policy that reuses past prompt evaluations to enable sample-efficient exploration. In the second step, we introduce Dynamic Memory Update (DMU), a training-free mechanism that updates the meta-prompt by injecting both (i) diverse prompts from a replay buffer and (ii) top-performing prompts from a small priority queue, thereby progressively concentrating the search process on high-reward regions. Across few-shot text classification, instruction induction benchmarks, and question answering tasks, GFlowPO consistently outperforms recent discrete prompt optimization baselines. 

---
# Robustness as an Emergent Property of Task Performance 

**Authors**: Shir Ashury-Tahan, Ariel Gera, Elron Bandel, Michal Shmueli-Scheuer, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2602.03344)  

**Abstract**: Robustness is often regarded as a critical future challenge for real-world applications, where stability is essential. However, as models often learn tasks in a similar order, we hypothesize that easier tasks will be easier regardless of how they are presented to the model. Indeed, in this paper, we show that as models approach high performance on a task, robustness is effectively achieved. Through an empirical analysis of multiple models across diverse datasets and configurations (e.g., paraphrases, different temperatures), we find a strong positive correlation. Moreover, we find that robustness is primarily driven by task-specific competence rather than inherent model-level properties, challenging current approaches that treat robustness as an independent capability. Thus, from a high-level perspective, we may expect that as new tasks saturate, model robustness on these tasks will emerge accordingly. For researchers, this implies that explicit efforts to measure and improve robustness may warrant reduced emphasis, as such robustness is likely to develop alongside performance gains. For practitioners, it acts as a sign that indeed the tasks that the literature deals with are unreliable, but on easier past tasks, the models are reliable and ready for real-world deployment. 

---
# R1-SyntheticVL: Is Synthetic Data from Generative Models Ready for Multimodal Large Language Model? 

**Authors**: Jingyi Zhang, Tianyi Lin, Huanjin Yao, Xiang Lan, Shunyu Liu, Jiaxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.03300)  

**Abstract**: In this work, we aim to develop effective data synthesis techniques that autonomously synthesize multimodal training data for enhancing MLLMs in solving complex real-world tasks. To this end, we propose Collective Adversarial Data Synthesis (CADS), a novel and general approach to synthesize high-quality, diverse and challenging multimodal data for MLLMs. The core idea of CADS is to leverage collective intelligence to ensure high-quality and diverse generation, while exploring adversarial learning to synthesize challenging samples for effectively driving model improvement. Specifically, CADS operates with two cyclic phases, i.e., Collective Adversarial Data Generation (CAD-Generate) and Collective Adversarial Data Judgment (CAD-Judge). CAD-Generate leverages collective knowledge to jointly generate new and diverse multimodal data, while CAD-Judge collaboratively assesses the quality of synthesized data. In addition, CADS introduces an Adversarial Context Optimization mechanism to optimize the generation context to encourage challenging and high-value data generation. With CADS, we construct MMSynthetic-20K and train our model R1-SyntheticVL, which demonstrates superior performance on various benchmarks. 

---
# Mići Princ -- A Little Boy Teaching Speech Technologies the Chakavian Dialect 

**Authors**: Nikola Ljubešić, Peter Rupnik, Tea Perinčić  

**Link**: [PDF](https://arxiv.org/pdf/2602.03245)  

**Abstract**: This paper documents our efforts in releasing the printed and audio book of the translation of the famous novel The Little Prince into the Chakavian dialect, as a computer-readable, AI-ready dataset, with the textual and the audio components of the two releases now aligned on the level of each written and spoken word. Our motivation for working on this release is multiple. The first one is our wish to preserve the highly valuable and specific content beyond the small editions of the printed and the audio book. With the dataset published in the this http URL repository, this content is from now on at the fingertips of any interested individual. The second motivation is to make the data available for various artificial-intelligence-related usage scenarios, such as the one we follow upon inside this paper already -- adapting the Whisper-large-v3 open automatic speech recognition model, with decent performance on standard Croatian, to Chakavian dialectal speech. We can happily report that with adapting the model, the word error rate on the selected test data has being reduced to a half, while we managed to remove up to two thirds of the error on character level. We envision many more usages of this dataset beyond the set of experiments we have already performed, both on tasks of artificial intelligence research and application, as well as dialectal research. The third motivation for this release is our hope that this, now highly structured dataset, will be transformed into a digital online edition of this work, allowing individuals beyond the research and technology communities to enjoy the beauty of the message of the little boy in the desert, told through the spectacular prism of the Chakavian dialect. 

---
# Merging Beyond: Streaming LLM Updates via Activation-Guided Rotations 

**Authors**: Yuxuan Yao, Haonan Sheng, Qingsong Lv, Han Wu, Shuqi Liu, Zehua Liu, Zengyan Liu, Jiahui Gao, Haochen Tan, Xiaojin Fu, Haoli Bai, Hing Cheung So, Zhijiang Guo, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.03237)  

**Abstract**: The escalating scale of Large Language Models (LLMs) necessitates efficient adaptation techniques. Model merging has gained prominence for its efficiency and controllability. However, existing merging techniques typically serve as post-hoc refinements or focus on mitigating task interference, often failing to capture the dynamic optimization benefits of supervised fine-tuning (SFT). In this work, we propose Streaming Merging, an innovative model updating paradigm that conceptualizes merging as an iterative optimization process. Central to this paradigm is \textbf{ARM} (\textbf{A}ctivation-guided \textbf{R}otation-aware \textbf{M}erging), a strategy designed to approximate gradient descent dynamics. By treating merging coefficients as learning rates and deriving rotation vectors from activation subspaces, ARM effectively steers parameter updates along data-driven trajectories. Unlike conventional linear interpolation, ARM aligns semantic subspaces to preserve the geometric structure of high-dimensional parameter evolution. Remarkably, ARM requires only early SFT checkpoints and, through iterative merging, surpasses the fully converged SFT model. Experimental results across model scales (1.7B to 14B) and diverse domains (e.g., math, code) demonstrate that ARM can transcend converged checkpoints. Extensive experiments show that ARM provides a scalable and lightweight framework for efficient model adaptation. 

---
# Prompt Augmentation Scales up GRPO Training on Mathematical Reasoning 

**Authors**: Wenquan Lu, Hai Huang, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2602.03190)  

**Abstract**: Reinforcement learning algorithms such as group-relative policy optimization (GRPO) have demonstrated strong potential for improving the mathematical reasoning capabilities of large language models. However, prior work has consistently observed an entropy collapse phenomenon during reinforcement post-training, characterized by a monotonic decrease in policy entropy that ultimately leads to training instability and collapse. As a result, most existing approaches restrict training to short horizons (typically 5-20 epochs), limiting sustained exploration and hindering further policy improvement. In addition, nearly all prior work relies on a single, fixed reasoning prompt or template during training. In this work, we introduce prompt augmentation, a training strategy that instructs the model to generate reasoning traces under diverse templates and formats, thereby increasing rollout diversity. We show that, without a KL regularization term, prompt augmentation enables stable scaling of training duration under a fixed dataset and allows the model to tolerate low-entropy regimes without premature collapse. Empirically, a Qwen2.5-Math-1.5B model trained with prompt augmentation on the MATH Level 3-5 dataset achieves state-of-the-art performance, reaching 44.5 per-benchmark accuracy and 51.3 per-question accuracy on standard mathematical reasoning benchmarks, including AIME24, AMC, MATH500, Minerva, and OlympiadBench. The code and model checkpoints are available at this https URL. 

---
# DynSplit-KV: Dynamic Semantic Splitting for KVCache Compression in Efficient Long-Context LLM Inference 

**Authors**: Jiancai Ye, Jun Liu, Qingchen Li, Tianlang Zhao, Hanbin Zhang, Jiayi Pan, Ningyi Xu, Guohao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2602.03184)  

**Abstract**: Although Key-Value (KV) Cache is essential for efficient large language models (LLMs) inference, its growing memory footprint in long-context scenarios poses a significant bottleneck, making KVCache compression crucial. Current compression methods rely on rigid splitting strategies, such as fixed intervals or pre-defined delimiters. We observe that rigid splitting suffers from significant accuracy degradation (ranging from 5.5% to 55.1%) across different scenarios, owing to the scenario-dependent nature of the semantic boundaries. This highlights the necessity of dynamic semantic splitting to match semantics. To achieve this, we face two challenges. (1) Improper delimiter selection misaligns semantics with the KVCache, resulting in 28.6% accuracy loss. (2) Variable-length blocks after splitting introduce over 73.1% additional inference overhead. To address the above challenges, we propose DynSplit-KV, a KVCache compression method that dynamically identifies delimiters for splitting. We propose: (1) a dynamic importance-aware delimiter selection strategy, improving accuracy by 49.9%. (2) A uniform mapping strategy that transforms variable-length semantic blocks into a fixed-length format, reducing inference overhead by 4.9x. Experiments show that DynSplit-KV achieves the highest accuracy, 2.2x speedup compared with FlashAttention and 2.6x peak memory reduction in long-context scenarios. 

---
# VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models 

**Authors**: Woojin Kim, Sieun Hyeon, Jusang Oh, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2602.03160)  

**Abstract**: Aligning Large Language Models (LLMs) with the diverse spectrum of human values remains a central challenge: preference-based methods often fail to capture deeper motivational principles. Value-based approaches offer a more principled path, yet three gaps persist: extraction often ignores hierarchical structure, evaluation detects presence but not calibrated intensity, and the steerability of LLMs at controlled intensities remains insufficiently understood. To address these limitations, we introduce VALUEFLOW, the first unified framework that spans extraction, evaluation, and steering with calibrated intensity control. The framework integrates three components: (i) HIVES, a hierarchical value embedding space that captures intra- and cross-theory value structure; (ii) the Value Intensity DataBase (VIDB), a large-scale resource of value-labeled texts with intensity estimates derived from ranking-based aggregation; and (iii) an anchor-based evaluator that produces consistent intensity scores for model outputs by ranking them against VIDB panels. Using VALUEFLOW, we conduct a comprehensive large-scale study across ten models and four value theories, identifying asymmetries in steerability and composition laws for multi-value control. This paper establishes a scalable infrastructure for evaluating and controlling value intensity, advancing pluralistic alignment of LLMs. 

---
# Self-Hinting Language Models Enhance Reinforcement Learning 

**Authors**: Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2602.03143)  

**Abstract**: Group Relative Policy Optimization (GRPO) has recently emerged as a practical recipe for aligning large language models with verifiable objectives. However, under sparse terminal rewards, GRPO often stalls because rollouts within a group frequently receive identical rewards, causing relative advantages to collapse and updates to vanish. We propose self-hint aligned GRPO with privileged supervision (SAGE), an on-policy reinforcement learning framework that injects privileged hints during training to reshape the rollout distribution under the same terminal verifier reward. For each prompt $x$, the model samples a compact hint $h$ (e.g., a plan or decomposition) and then generates a solution $\tau$ conditioned on $(x,h)$. Crucially, the task reward $R(x,\tau)$ is unchanged; hints only increase within-group outcome diversity under finite sampling, preventing GRPO advantages from collapsing under sparse rewards. At test time, we set $h=\varnothing$ and deploy the no-hint policy without any privileged information. Moreover, sampling diverse self-hints serves as an adaptive curriculum that tracks the learner's bottlenecks more effectively than fixed hints from an initial policy or a stronger external model. Experiments over 6 benchmarks with 3 LLMs show that SAGE consistently outperforms GRPO, on average +2.0 on Llama-3.2-3B-Instruct, +1.2 on Qwen2.5-7B-Instruct and +1.3 on Qwen3-4B-Instruct. The code is available at this https URL. 

---
# From Speech-to-Spatial: Grounding Utterances on A Live Shared View with Augmented Reality 

**Authors**: Yoonsang Kim, Divyansh Pradhan, Devshree Jadeja, Arie Kaufman  

**Link**: [PDF](https://arxiv.org/pdf/2602.03059)  

**Abstract**: We introduce Speech-to-Spatial, a referent disambiguation framework that converts verbal remote-assistance instructions into spatially grounded AR guidance. Unlike prior systems that rely on additional cues (e.g., gesture, gaze) or manual expert annotations, Speech-to-Spatial infers the intended target solely from spoken references (speech input). Motivated by our formative study of speech referencing patterns, we characterize recurring ways people specify targets (Direct Attribute, Relational, Remembrance, and Chained) and ground them to our object-centric relational graph. Given an utterance, referent cues are parsed and rendered as persistent in-situ AR visual guidance, reducing iterative micro-guidance ("a bit more to the right", "now, stop.") during remote guidance. We demonstrate the use cases of our system with remote guided assistance and intent disambiguation scenarios. Our evaluation shows that Speechto-Spatial improves task efficiency, reduces cognitive load, and enhances usability compared to a conventional voice-only baseline, transforming disembodied verbal instruction into visually explainable, actionable guidance on a live shared view. 

---
# MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems 

**Authors**: Vishal Venkataramani, Haizhou Shi, Zixuan Ke, Austin Xu, Xiaoxiao He, Yingbo Zhou, Semih Yavuz, Hao Wang, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2602.03053)  

**Abstract**: Multi-Agent Systems (MAS) built on Large Language Models (LLMs) often exhibit high variance in their reasoning trajectories. Process verification, which evaluates intermediate steps in trajectories, has shown promise in general reasoning settings, and has been suggested as a potential tool for guiding coordination of MAS; however, its actual effectiveness in MAS remains unclear. To fill this gap, we present MAS-ProVe, a systematic empirical study of process verification for multi-agent systems (MAS). Our study spans three verification paradigms (LLM-as-a-Judge, reward models, and process reward models), evaluated across two levels of verification granularity (agent-level and iteration-level). We further examine five representative verifiers and four context management strategies, and conduct experiments over six diverse MAS frameworks on multiple reasoning benchmarks. We find that process-level verification does not consistently improve performance and frequently exhibits high variance, highlighting the difficulty of reliably evaluating partial multi-agent trajectories. Among the methods studied, LLM-as-a-Judge generally outperforms reward-based approaches, with trained judges surpassing general-purpose LLMs. We further observe a small performance gap between LLMs acting as judges and as single agents, and identify a context-length-performance trade-off in verification. Overall, our results suggest that effective and robust process verification for MAS remains an open challenge, requiring further advances beyond current paradigms. Code is available at this https URL. 

---
# RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents 

**Authors**: Haitian Zhong, Jixiu Zhai, Lei Song, Jiang Bian, Qiang Liu, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.03025)  

**Abstract**: Multi-turn tool calling is challenging for Large Language Models (LLMs) because rewards are sparse and exploration is expensive. A common recipe, SFT followed by GRPO, can stall when within-group reward variation is low (e.g., more rollouts in a group receive the all 0 or all 1 reward), making the group-normalized advantage uninformative and yielding vanishing updates. To address this problem, we propose RC-GRPO (Reward-Conditioned Group Relative Policy Optimization), which treats exploration as a controllable steering problem via discrete reward tokens. We first fine-tune a Reward-Conditioned Trajectory Policy (RCTP) on mixed-quality trajectories with reward goal special tokens (e.g., <|high_reward|>, <|low_reward|>) injected into the prompts, enabling the model to learn how to generate distinct quality trajectories on demand. Then during RL, we sample diverse reward tokens within each GRPO group and condition rollouts on the sampled token to improve within-group diversity, improving advantage gains. On the Berkeley Function Calling Leaderboard v4 (BFCLv4) multi-turn benchmark, our method yields consistently improved performance than baselines, and the performance on Qwen-2.5-7B-Instruct even surpasses all closed-source API models. 

---
# WST-X Series: Wavelet Scattering Transform for Interpretable Speech Deepfake Detection 

**Authors**: Xi Xuan, Davide Carbone, Ruchi Pandey, Wenxin Zhang, Tomi H. Kinnunen  

**Link**: [PDF](https://arxiv.org/pdf/2602.02980)  

**Abstract**: Designing front-ends for speech deepfake detectors primarily focuses on two categories. Hand-crafted filterbank features are transparent but are limited in capturing high-level semantic details, often resulting in performance gaps compared to self-supervised (SSL) features. SSL features, in turn, lack interpretability and may overlook fine-grained spectral anomalies. We propose the WST-X series, a novel family of feature extractors that combines the best of both worlds via the wavelet scattering transform (WST), integrating wavelets with nonlinearities analogous to deep convolutional networks. We investigate 1D and 2D WSTs to extract acoustic details and higher-order structural anomalies, respectively. Experimental results on the recent and challenging Deepfake-Eval-2024 dataset indicate that WST-X outperforms existing front-ends by a wide margin. Our analysis reveals that a small averaging scale ($J$), combined with high-frequency and directional resolutions ($Q, L$), is critical for capturing subtle artifacts. This underscores the value of translation-invariant and deformation-stable features for robust and interpretable speech deepfake detection. 

---
# Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning 

**Authors**: Yihong Huang, Fei Ma, Yihua Shao, Jingcai Guo, Zitong Yu, Laizhong Cui, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.02951)  

**Abstract**: Vision token pruning has proven to be an effective acceleration technique for the efficient Vision Language Model (VLM). However, existing pruning methods demonstrate excellent performance preservation in visual question answering (VQA) and suffer substantial degradation on visual grounding (VG) tasks. Our analysis of the VLM's processing pipeline reveals that strategies utilizing global semantic similarity and attention scores lose the global spatial reference frame, which is derived from the interactions of tokens' positional information. Motivated by these findings, we propose $\text{Nüwa}$, a two-stage token pruning framework that enables efficient feature aggregation while maintaining spatial integrity. In the first stage, after the vision encoder, we apply three operations, namely separation, alignment, and aggregation, which are inspired by swarm intelligence algorithms to retain information-rich global spatial anchors. In the second stage, within the LLM, we perform text-guided pruning to retain task-relevant visual tokens. Extensive experiments demonstrate that $\text{Nüwa}$ achieves SOTA performance on multiple VQA benchmarks (from 94% to 95%) and yields substantial improvements on visual grounding tasks (from 7% to 47%). 

---
# A vector logic for intensional formal semantics 

**Authors**: Daniel Quigley  

**Link**: [PDF](https://arxiv.org/pdf/2602.02940)  

**Abstract**: Formal semantics and distributional semantics are distinct approaches to linguistic meaning: the former models meaning as reference via model-theoretic structures; the latter represents meaning as vectors in high-dimensional spaces shaped by usage. This paper proves that these frameworks are structurally compatible for intensional semantics. We establish that Kripke-style intensional models embed injectively into vector spaces, with semantic functions lifting to (multi)linear maps that preserve composition. The construction accommodates multiple index sorts (worlds, times, locations) via a compound index space, representing intensions as linear operators. Modal operators are derived algebraically: accessibility relations become linear operators, and modal conditions reduce to threshold checks on accumulated values. For uncountable index domains, we develop a measure-theoretic generalization in which necessity becomes truth almost everywhere and possibility becomes truth on a set of positive measure, a non-classical logic natural for continuous parameters. 

---
# Aligning Language Model Benchmarks with Pairwise Preferences 

**Authors**: Marco Gutierrez, Xinyi Leng, Hannah Cyberey, Jonathan Richard Schwarz, Ahmed Alaa, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2602.02898)  

**Abstract**: Language model benchmarks are pervasive and computationally-efficient proxies for real-world performance. However, many recent works find that benchmarks often fail to predict real utility. Towards bridging this gap, we introduce benchmark alignment, where we use limited amounts of information about model performance to automatically update offline benchmarks, aiming to produce new static benchmarks that predict model pairwise preferences in given test settings. We then propose BenchAlign, the first solution to this problem, which learns preference-aligned weight- ings for benchmark questions using the question-level performance of language models alongside ranked pairs of models that could be collected during deployment, producing new benchmarks that rank previously unseen models according to these preferences. Our experiments show that our aligned benchmarks can accurately rank unseen models according to models of human preferences, even across different sizes, while remaining interpretable. Overall, our work provides insights into the limits of aligning benchmarks with practical human preferences, which stands to accelerate model development towards real utility. 

---
# TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation 

**Authors**: Prajna G. Malettira, Manish Nagaraj, Arjun Roy, Shubham Negi, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2602.02891)  

**Abstract**: Structured pruning is essential for efficient deployment of Large Language Models (LLMs). The varying sensitivity of LLM sub-blocks to pruning necessitates the identification of optimal non-uniformly pruned models. Existing methods evaluate the importance of layers, attention heads, or weight channels in isolation. Such localized focus ignores the complex global structural dependencies that exist across the model. Training-aware structured pruning addresses global dependencies, but its computational cost can be just as expensive as post-pruning training. To alleviate the computational burden of training-aware pruning and capture global structural dependencies, we propose TraceNAS, a training-free Neural Architecture Search (NAS) framework that jointly explores structured pruning of LLM depth and width. TraceNAS identifies pruned models that maintain a high degree of loss landscape alignment with the pretrained model using a scale-invariant zero-shot proxy, effectively selecting models that exhibit maximal performance potential during post-pruning training. TraceNAS is highly efficient, enabling high-fidelity discovery of pruned models on a single GPU in 8.5 hours, yielding a 10$\times$ reduction in GPU-hours compared to training-aware methods. Evaluations on the Llama and Qwen families demonstrate that TraceNAS is competitive with training-aware baselines across commonsense and reasoning benchmarks. 

---
# Chain of Simulation: A Dual-Mode Reasoning Framework for Large Language Models with Dynamic Problem Routing 

**Authors**: Saeid Sheikhi  

**Link**: [PDF](https://arxiv.org/pdf/2602.02842)  

**Abstract**: We present Chain of Simulation (CoS), a novel dual-mode reasoning framework that dynamically routes problems to specialized reasoning strategies in Large Language Models (LLMs). Unlike existing uniform prompting approaches, CoS employs three distinct reasoning modes: (1) computational flow with self-consistency for mathematical problems, (2) symbolic state tracking with JSON representations for spatial reasoning, and (3) hybrid fact-extraction for multi-hop inference. Through comprehensive evaluation on GSM8K, StrategyQA, and bAbI benchmarks using four state-of-the-art models (Gemma-3 27B, LLaMA-3.1 8B, Mistral 7B, and Qwen-2.5 14B), we demonstrate that CoS achieves 71.5% accuracy on GSM8K (1.0% absolute improvement), 90.0% on StrategyQA (2.5% improvement), and 19.0% on bAbI (65.2% relative improvement) compared to the strongest baselines. The analysis reveals that problem-specific mode selection is crucial, with computational mode achieving 81.2% accuracy when correctly applied to mathematical problems, while misrouting leads to 0% accuracy. We provide detailed algorithms for mode selection, state tracking, and answer extraction, establishing CoS as an effective approach for improving LLM reasoning without additional training. The framework provides superior trade-offs between accuracy and efficiency compared to Self-Consistency, achieving comparable performance at 54% lower computational cost. 

---
# Privately Fine-Tuned LLMs Preserve Temporal Dynamics in Tabular Data 

**Authors**: Lucas Rosenblatt, Peihan Liu, Ryan McKenna, Natalia Ponomareva  

**Link**: [PDF](https://arxiv.org/pdf/2602.02766)  

**Abstract**: Research on differentially private synthetic tabular data has largely focused on independent and identically distributed rows where each record corresponds to a unique individual. This perspective neglects the temporal complexity in longitudinal datasets, such as electronic health records, where a user contributes an entire (sub) table of sequential events. While practitioners might attempt to model such data by flattening user histories into high-dimensional vectors for use with standard marginal-based mechanisms, we demonstrate that this strategy is insufficient. Flattening fails to preserve temporal coherence even when it maintains valid marginal distributions. We introduce PATH, a novel generative framework that treats the full table as the unit of synthesis and leverages the autoregressive capabilities of privately fine-tuned large language models. Extensive evaluations show that PATH effectively captures long-range dependencies that traditional methods miss. Empirically, our method reduces the distributional distance to real trajectories by over 60% and reduces state transition errors by nearly 50% compared to leading marginal mechanisms while achieving similar marginal fidelity. 

---
# Scaling Small Agents Through Strategy Auctions 

**Authors**: Lisa Alazraki, William F. Shen, Yoram Bachrach, Akhil Mathur  

**Link**: [PDF](https://arxiv.org/pdf/2602.02751)  

**Abstract**: Small language models are increasingly viewed as a promising, cost-effective approach to agentic AI, with proponents claiming they are sufficiently capable for agentic workflows. However, while smaller agents can closely match larger ones on simple tasks, it remains unclear how their performance scales with task complexity, when large models become necessary, and how to better leverage small agents for long-horizon workloads. In this work, we empirically show that small agents' performance fails to scale with task complexity on deep search and coding tasks, and we introduce Strategy Auctions for Workload Efficiency (SALE), an agent framework inspired by freelancer marketplaces. In SALE, agents bid with short strategic plans, which are scored by a systematic cost-value mechanism and refined via a shared auction memory, enabling per-task routing and continual self-improvement without training a separate router or running all models to completion. Across deep search and coding tasks of varying complexity, SALE reduces reliance on the largest agent by 53%, lowers overall cost by 35%, and consistently improves upon the largest agent's pass@1 with only a negligible overhead beyond executing the final trace. In contrast, established routers that rely on task descriptions either underperform the largest agent or fail to reduce cost -- often both -- underscoring their poor fit for agentic workflows. These results suggest that while small agents may be insufficient for complex workloads, they can be effectively "scaled up" through coordinated task allocation and test-time self-improvement. More broadly, they motivate a systems-level view of agentic AI in which performance gains come less from ever-larger individual models and more from market-inspired coordination mechanisms that organize heterogeneous agents into efficient, adaptive ecosystems. 

---
# WAXAL: A Large-Scale Multilingual African Language Speech Corpus 

**Authors**: Abdoulaye Diack, Perry Nelson, Kwaku Agbesi, Angela Nakalembe, MohamedElfatih MohamedKhair, Vusumuzi Dube, Tavonga Siyavora, Subhashini Venugopalan, Jason Hickey, Uche Okonkwo, Abhishek Bapna, Isaac Wiafe, Raynard Dodzi Helegah, Elikem Doe Atsakpo, Charles Nutrokpor, Fiifi Baffoe Payin Winful, Kafui Kwashie Solaga, Jamal-Deen Abdulai, Akon Obu Ekpezu, Audace Niyonkuru, Samuel Rutunda, Boris Ishimwe, Michael Melese, Engineer Bainomugisha, Joyce Nakatumba-Nabende, Andrew Katumba, Claire Babirye, Jonathan Mukiibi, Vincent Kimani, Samuel Kibacia, James Maina, Fridah Emmah, Ahmed Ibrahim Shekarau, Ibrahim Shehu Adamu, Yusuf Abdullahi, Howard Lakougna, Bob MacDonald, Hadar Shemtov, Aisha Walcott-Bryant, Moustapha Cisse, Avinatan Hassidim, Jeff Dean, Yossi Matias  

**Link**: [PDF](https://arxiv.org/pdf/2602.02734)  

**Abstract**: The advancement of speech technology has predominantly favored high-resource languages, creating a significant digital divide for speakers of most Sub-Saharan African languages. To address this gap, we introduce WAXAL, a large-scale, openly accessible speech dataset for 21 languages representing over 100 million speakers. The collection consists of two main components: an Automated Speech Recognition (ASR) dataset containing approximately 1,250 hours of transcribed, natural speech from a diverse range of speakers, and a Text-to-Speech (TTS) dataset with over 180 hours of high-quality, single-speaker recordings reading phonetically balanced scripts. This paper details our methodology for data collection, annotation, and quality control, which involved partnerships with four African academic and community organizations. We provide a detailed statistical overview of the dataset and discuss its potential limitations and ethical considerations. The WAXAL datasets are released at this https URL under the permissive CC-BY-4.0 license to catalyze research, enable the development of inclusive technologies, and serve as a vital resource for the digital preservation of these languages. 

---
# Vector Quantized Latent Concepts: A Scalable Alternative to Clustering-Based Concept Discovery 

**Authors**: Xuemin Yu, Ankur Garg, Samira Ebrahimi Kahou, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2602.02726)  

**Abstract**: Deep Learning models encode rich semantic information in their hidden representations. However, it remains challenging to understand which parts of this information models actually rely on when making predictions. A promising line of post-hoc concept-based explanation methods relies on clustering token representations. However, commonly used approaches such as hierarchical clustering are computationally infeasible for large-scale datasets, and K-Means often yields shallow or frequency-dominated clusters. We propose the vector quantized latent concept (VQLC) method, a framework built upon the vector quantized-variational autoencoder (VQ-VAE) architecture that learns a discrete codebook mapping continuous representations to concept vectors. We perform thorough evaluations and show that VQLC improves scalability while maintaining comparable quality of human-understandable explanations. 

---
# Towards Understanding Steering Strength 

**Authors**: Magamed Taimeskhanov, Samuel Vaiter, Damien Garreau  

**Link**: [PDF](https://arxiv.org/pdf/2602.02712)  

**Abstract**: A popular approach to post-training control of large language models (LLMs) is the steering of intermediate latent representations. Namely, identify a well-chosen direction depending on the task at hand and perturbs representations along this direction at inference time. While many propositions exist to pick this direction, considerably less is understood about how to choose the magnitude of the move, whereas its importance is clear: too little and the intended behavior does not emerge, too much and the model's performance degrades beyond repair. In this work, we propose the first theoretical analysis of steering strength. We characterize its effect on next token probability, presence of a concept, and cross-entropy, deriving precise qualitative laws governing these quantities. Our analysis reveals surprising behaviors, including non-monotonic effects of steering strength. We validate our theoretical predictions empirically on eleven language models, ranging from a small GPT architecture to modern models. 

---
# BinaryPPO: Efficient Policy Optimization for Binary Classification 

**Authors**: Punya Syon Pandey, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.02708)  

**Abstract**: Supervised fine-tuning (SFT) is the standard approach for binary classification tasks such as toxicity detection, factuality verification, and causal inference. However, SFT often performs poorly in real-world settings with label noise, class imbalance, or sparse supervision. We introduce BinaryPPO, an offline reinforcement learning large language model (LLM) framework that reformulates binary classification as a reward maximization problem. Our method leverages a variant of Proximal Policy Optimization (PPO) with a confidence-weighted reward function that penalizes uncertain or incorrect predictions, enabling the model to learn robust decision policies from static datasets without online interaction. Across eight domain-specific benchmarks and multiple models with differing architectures, BinaryPPO improves accuracy by 40-60 percentage points, reaching up to 99%, substantially outperforming supervised baselines. We provide an in-depth analysis of the role of reward shaping, advantage scaling, and policy stability in enabling this improvement. Overall, we demonstrate that confidence-based reward design provides a robust alternative to SFT for binary classification. Our code is available at this https URL. 

---
# Fine-Tuning Language Models to Know What They Know 

**Authors**: Sangjun Park, Elliot Meyerson, Xin Qiu, Risto Miikkulainen  

**Link**: [PDF](https://arxiv.org/pdf/2602.02605)  

**Abstract**: Metacognition is a critical component of intelligence, specifically regarding the awareness of one's own knowledge. While humans rely on shared internal memory for both answering questions and reporting their knowledge state, this dependency in LLMs remains underexplored. This study proposes a framework to measure metacognitive ability $d_{\rm{type2}}'$ using a dual-prompt method, followed by the introduction of Evolution Strategy for Metacognitive Alignment (ESMA) to bind a model's internal knowledge to its explicit behaviors. ESMA demonstrates robust generalization across diverse untrained settings, indicating a enhancement in the model's ability to reference its own knowledge. Furthermore, parameter analysis attributes these improvements to a sparse set of significant modifications. 

---
# Social Catalysts, Not Moral Agents: The Illusion of Alignment in LLM Societies 

**Authors**: Yueqing Hu, Yixuan Jiang, Zehua Jiang, Xiao Wen, Tianhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02598)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems where collective cooperation is often threatened by the "Tragedy of the Commons." This study investigates the effectiveness of Anchoring Agents--pre-programmed altruistic entities--in fostering cooperation within a Public Goods Game (PGG). Using a full factorial design across three state-of-the-art LLMs, we analyzed both behavioral outcomes and internal reasoning chains. While Anchoring Agents successfully boosted local cooperation rates, cognitive decomposition and transfer tests revealed that this effect was driven by strategic compliance and cognitive offloading rather than genuine norm internalization. Notably, most agents reverted to self-interest in new environments, and advanced models like GPT-4.1 exhibited a "Chameleon Effect," masking strategic defection under public scrutiny. These findings highlight a critical gap between behavioral modification and authentic value alignment in artificial societies. 

---
# Uncertainty and Fairness Awareness in LLM-Based Recommendation Systems 

**Authors**: Chandan Kumar Sah, Xiaoli Lian, Li Zhang, Tony Xu, Syed Shazaib Shah  

**Link**: [PDF](https://arxiv.org/pdf/2602.02582)  

**Abstract**: Large language models (LLMs) enable powerful zero-shot recommendations by leveraging broad contextual knowledge, yet predictive uncertainty and embedded biases threaten reliability and fairness. This paper studies how uncertainty and fairness evaluations affect the accuracy, consistency, and trustworthiness of LLM-generated recommendations. We introduce a benchmark of curated metrics and a dataset annotated for eight demographic attributes (31 categorical values) across two domains: movies and music. Through in-depth case studies, we quantify predictive uncertainty (via entropy) and demonstrate that Google DeepMind's Gemini 1.5 Flash exhibits systematic unfairness for certain sensitive attributes; measured similarity-based gaps are SNSR at 0.1363 and SNSV at 0.0507. These disparities persist under prompt perturbations such as typographical errors and multilingual inputs. We further integrate personality-aware fairness into the RecLLM evaluation pipeline to reveal personality-linked bias patterns and expose trade-offs between personalization and group fairness. We propose a novel uncertainty-aware evaluation methodology for RecLLMs, present empirical insights from deep uncertainty case studies, and introduce a personality profile-informed fairness benchmark that advances explainability and equity in LLM recommendations. Together, these contributions establish a foundation for safer, more interpretable RecLLMs and motivate future work on multi-model benchmarks and adaptive calibration for trustworthy deployment. 

---
# Enhancing Post-Training Quantization via Future Activation Awareness 

**Authors**: Zheqi Lv, Zhenxuan Fan, Qi Tian, Wenqiao Zhang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02538)  

**Abstract**: Post-training quantization (PTQ) is a widely used method to compress large language models (LLMs) without fine-tuning. It typically sets quantization hyperparameters (e.g., scaling factors) based on current-layer activations. Although this method is efficient, it suffers from quantization bias and error accumulation, resulting in suboptimal and unstable quantization, especially when the calibration data is biased. To overcome these issues, we propose Future-Aware Quantization (FAQ), which leverages future-layer activations to guide quantization. This allows better identification and preservation of important weights, while reducing sensitivity to calibration noise. We further introduce a window-wise preview mechanism to softly aggregate multiple future-layer activations, mitigating over-reliance on any single layer. To avoid expensive greedy search, we use a pre-searched configuration to minimize overhead. Experiments show that FAQ consistently outperforms prior methods with negligible extra cost, requiring no backward passes, data reconstruction, or tuning, making it well-suited for edge deployment. 

---
# From Sparse Decisions to Dense Reasoning: A Multi-attribute Trajectory Paradigm for Multimodal Moderation 

**Authors**: Tianle Gu, Kexin Huang, Lingyu Li, Ruilin Luo, Shiyang Huang, Zongqi Wang, Yujiu Yang, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02536)  

**Abstract**: Safety moderation is pivotal for identifying harmful content. Despite the success of textual safety moderation, its multimodal counterparts remain hindered by a dual sparsity of data and supervision. Conventional reliance on binary labels lead to shortcut learning, which obscures the intrinsic classification boundaries necessary for effective multimodal discrimination. Hence, we propose a novel learning paradigm (UniMod) that transitions from sparse decision-making to dense reasoning traces. By constructing structured trajectories encompassing evidence grounding, modality assessment, risk mapping, policy decision, and response generation, we reformulate monolithic decision tasks into a multi-dimensional boundary learning process. This approach forces the model to ground its decision in explicit safety semantics, preventing the model from converging on superficial shortcuts. To facilitate this paradigm, we develop a multi-head scalar reward model (UniRM). UniRM provides multi-dimensional supervision by assigning attribute-level scores to the response generation stage. Furthermore, we introduce specialized optimization strategies to decouple task-specific parameters and rebalance training dynamics, effectively resolving interference between diverse objectives in multi-task learning. Empirical results show UniMod achieves competitive textual moderation performance and sets a new multimodal benchmark using less than 40\% of the training data used by leading baselines. Ablations further validate our multi-attribute trajectory reasoning, offering an effective and efficient framework for multimodal moderation. Supplementary materials are available at \href{this https URL}{project website}. 

---
# The "Robert Boulton" Singularity: Semantic Tunneling and Manifold Unfolding in Recursive AI 

**Authors**: Pengyue Hou  

**Link**: [PDF](https://arxiv.org/pdf/2602.02526)  

**Abstract**: The stability of generative artificial intelligence trained on recursive synthetic data is conventionally monitored via Perplexity (PPL). We demonstrate that PPL is a deceptive metric in context-stabilized regimes (L=128). Using a rigorous sliding-window protocol (N=1500), we identify a novel failure mode termed "Semantic Tunneling." While the Baseline model maintains high grammatical fluency (PPL approx. 83.9), it suffers a catastrophic loss of semantic diversity, converging within seven generations to a single, low-entropy narrative attractor: the "Robert Boulton" Singularity. This phenomenon represents a total collapse of the latent manifold (Global Effective Rank 3.62 -> 2.22), where the model discards diverse world knowledge to optimize for statistically safe syntactic templates. To address this, we apply the Multi-Scale Negative Coupled Information Systems (MNCIS) framework recently established in Hou (2026) [arXiv:2601.11594]. We demonstrate that Adaptive Spectral Negative Coupling (ASNC) acts as a topological operator that actively induces "Manifold Unfolding." MNCIS forces the model to expand its effective rank from the anisotropic baseline of 3.62 to a hyper-diverse state of 5.35, effectively constructing an "Artificial Manifold" that resists the gravitational pull of semantic attractors and preserves the long-tail distribution of the training data. 

---
# GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning 

**Authors**: Yuyang Bai, Zhuofeng Li, Ping Nie, Jianwen Xie, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.02518)  

**Abstract**: Large language models (LLMs) increasingly rely on external knowledge to improve factuality, yet many real-world knowledge sources are organized as heterogeneous graphs rather than plain text. Reasoning over such graph-structured knowledge poses two key challenges: (1) navigating structured, schema-defined relations requires precise function calls rather than similarity-based retrieval, and (2) answering complex questions often demands multi-hop evidence aggregation through iterative information seeking. We propose GraphDancer, a reinforcement learning (RL) framework that teaches LLMs to navigate graphs by interleaving reasoning and function execution. To make RL effective for moderate-sized LLMs, we introduce a graph-aware curriculum that schedules training by the structural complexity of information-seeking trajectories using an easy-to-hard biased sampler. We evaluate GraphDancer on a multi-domain benchmark by training on one domain only and testing on unseen domains and out-of-distribution question types. Despite using only a 3B backbone, GraphDancer outperforms baselines equipped with either a 14B backbone or GPT-4o-mini, demonstrating robust cross-domain generalization of graph exploration and reasoning skills. Our code and models can be found at this https URL . 

---
# CreditAudit: 2D Auditing for LLM Evaluation and Selection 

**Authors**: Yiliang Song, Hongjun An, Jiangong Xiao, Haofei Zhao, Jiawei Shao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.02515)  

**Abstract**: Leaderboard scores on public benchmarks have been steadily rising and converging, with many frontier language models now separated by only marginal differences. However, these scores often fail to match users' day to day experience, because system prompts, output protocols, and interaction modes evolve under routine iteration, and in agentic multi step pipelines small protocol shifts can trigger disproportionate failures, leaving practitioners uncertain about which model to deploy. We propose CreditAudit, a deployment oriented credit audit framework that evaluates models under a family of semantically aligned and non adversarial system prompt templates across multiple benchmarks, reporting mean ability as average performance across scenarios and scenario induced fluctuation sigma as a stability risk signal, and further mapping volatility into interpretable credit grades from AAA to BBB via cross model quantiles with diagnostics that mitigate template difficulty drift. Controlled experiments on GPQA, TruthfulQA, and MMLU Pro show that models with similar mean ability can exhibit substantially different fluctuation, and stability risk can overturn prioritization decisions in agentic or high failure cost regimes. By providing a 2D and grade based language for regime specific selection, CreditAudit supports tiered deployment and more disciplined allocation of testing and monitoring effort, enabling more objective and trustworthy model evaluation for real world use. 

---
# Beyond Translation: Cross-Cultural Meme Transcreation with Vision-Language Models 

**Authors**: Yuming Zhao, Peiyi Zhang, Oana Ignat  

**Link**: [PDF](https://arxiv.org/pdf/2602.02510)  

**Abstract**: Memes are a pervasive form of online communication, yet their cultural specificity poses significant challenges for cross-cultural adaptation. We study cross-cultural meme transcreation, a multimodal generation task that aims to preserve communicative intent and humor while adapting culture-specific references. We propose a hybrid transcreation framework based on vision-language models and introduce a large-scale bidirectional dataset of Chinese and US memes. Using both human judgments and automated evaluation, we analyze 6,315 meme pairs and assess transcreation quality across cultural directions. Our results show that current vision-language models can perform cross-cultural meme transcreation to a limited extent, but exhibit clear directional asymmetries: US-Chinese transcreation consistently achieves higher quality than Chinese-US. We further identify which aspects of humor and visual-textual design transfer across cultures and which remain challenging, and propose an evaluation framework for assessing cross-cultural multimodal generation. Our code and dataset are publicly available at this https URL. 

---
# Thinking Like a Doctor: Conversational Diagnosis through the Exploration of Diagnostic Knowledge Graphs 

**Authors**: Jeongmoon Won, Seungwon Kook, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2602.01995)  

**Abstract**: Conversational diagnosis requires multi-turn history-taking, where an agent asks clarifying questions to refine differential diagnoses under incomplete information. Existing approaches often rely on the parametric knowledge of a model or assume that patients provide rich and concrete information, which is unrealistic. To address these limitations, we propose a conversational diagnosis system that explores a diagnostic knowledge graph to reason in two steps: (i) generating diagnostic hypotheses from the dialogue context, and (ii) verifying hypotheses through clarifying questions, which are repeated until a final diagnosis is reached. Since evaluating the system requires a realistic patient simulator that responds to the system's questions, we adopt a well-established simulator along with patient profiles from MIMIC-IV. We further adapt it to describe symptoms vaguely to reflect real-world patients during early clinical encounters. Experiments show improved diagnostic accuracy and efficiency over strong baselines, and evaluations by physicians support the realism of our simulator and the clinical utility of the generated questions. Our code will be released upon publication. 

---
# Making Avatars Interact: Towards Text-Driven Human-Object Interaction for Controllable Talking Avatars 

**Authors**: Youliang Zhang, Zhengguang Zhou, Zhentao Yu, Ziyao Huang, Teng Hu, Sen Liang, Guozhen Zhang, Ziqiao Peng, Shunkai Li, Yi Chen, Zixiang Zhou, Yuan Zhou, Qinglin Lu, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.01538)  

**Abstract**: Generating talking avatars is a fundamental task in video generation. Although existing methods can generate full-body talking avatars with simple human motion, extending this task to grounded human-object interaction (GHOI) remains an open challenge, requiring the avatar to perform text-aligned interactions with surrounding objects. This challenge stems from the need for environmental perception and the control-quality dilemma in GHOI generation. To address this, we propose a novel dual-stream framework, InteractAvatar, which decouples perception and planning from video synthesis for grounded human-object interaction. Leveraging detection to enhance environmental perception, we introduce a Perception and Interaction Module (PIM) to generate text-aligned interaction motions. Additionally, an Audio-Interaction Aware Generation Module (AIM) is proposed to synthesize vivid talking avatars performing object interactions. With a specially designed motion-to-video aligner, PIM and AIM share a similar network structure and enable parallel co-generation of motions and plausible videos, effectively mitigating the control-quality dilemma. Finally, we establish a benchmark, GroundedInter, for evaluating GHOI video generation. Extensive experiments and comparisons demonstrate the effectiveness of our method in generating grounded human-object interactions for talking avatars. Project page: this https URL 

---
