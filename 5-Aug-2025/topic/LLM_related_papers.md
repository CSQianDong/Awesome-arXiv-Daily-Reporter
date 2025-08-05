# Beyond Chunks and Graphs: Retrieval-Augmented Generation through Triplet-Driven Thinking 

**Authors**: Shengbo Gong, Xianfeng Tang, Carl Yang, Wei jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.02435)  

**Abstract**: Retrieval-augmented generation (RAG) is critical for reducing hallucinations and incorporating external knowledge into Large Language Models (LLMs). However, advanced RAG systems face a trade-off between performance and efficiency. Multi-round RAG approaches achieve strong reasoning but incur excessive LLM calls and token costs, while Graph RAG methods suffer from computationally expensive, error-prone graph construction and retrieval redundancy. To address these challenges, we propose T$^2$RAG, a novel framework that operates on a simple, graph-free knowledge base of atomic triplets. T$^2$RAG leverages an LLM to decompose questions into searchable triplets with placeholders, which it then iteratively resolves by retrieving evidence from the triplet database. Empirical results show that T$^2$RAG significantly outperforms state-of-the-art multi-round and Graph RAG methods, achieving an average performance gain of up to 11\% across six datasets while reducing retrieval costs by up to 45\%. Our code is available at this https URL 

---
# ChEmbed: Enhancing Chemical Literature Search Through Domain-Specific Text Embeddings 

**Authors**: Ali Shiraee Kasmaee, Mohammad Khodadad, Mehdi Astaraki, Mohammad Arshi Saloot, Nicholas Sherck, Hamidreza Mahyar, Soheila Samiee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01643)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems in chemistry heavily depend on accurate and relevant retrieval of chemical literature. However, general-purpose text embedding models frequently fail to adequately represent complex chemical terminologies, resulting in suboptimal retrieval quality. Specialized embedding models tailored to chemical literature retrieval have not yet been developed, leaving a substantial performance gap. To address this challenge, we introduce ChEmbed, a domain-adapted family of text embedding models fine-tuned on a dataset comprising chemistry-specific text from the PubChem, Semantic Scholar, and ChemRxiv corpora. To create effective training data, we employ large language models to synthetically generate queries, resulting in approximately 1.7 million high-quality query-passage pairs. Additionally, we augment the tokenizer by adding 900 chemically specialized tokens to previously unused slots, which significantly reduces the fragmentation of chemical entities, such as IUPAC names. ChEmbed also maintains a 8192-token context length, enabling the efficient retrieval of longer passages compared to many other open-source embedding models, which typically have a context length of 512 or 2048 tokens. Evaluated on our newly introduced ChemRxiv Retrieval benchmark, ChEmbed outperforms state-of-the-art general embedding models, raising nDCG@10 from 0.82 to 0.91 (+9 pp). ChEmbed represents a practical, lightweight, and reproducible embedding solution that effectively improves retrieval for chemical literature search. 

---
# Evaluating Position Bias in Large Language Model Recommendations 

**Authors**: Ethan Bito, Yongli Ren, Estrid He  

**Link**: [PDF](https://arxiv.org/pdf/2508.02020)  

**Abstract**: Large Language Models (LLMs) are being increasingly explored as general-purpose tools for recommendation tasks, enabling zero-shot and instruction-following capabilities without the need for task-specific training. While the research community is enthusiastically embracing LLMs, there are important caveats to directly adapting them for recommendation tasks. In this paper, we show that LLM-based recommendation models suffer from position bias, where the order of candidate items in a prompt can disproportionately influence the recommendations produced by LLMs. First, we analyse the position bias of LLM-based recommendations on real-world datasets, where results uncover systemic biases of LLMs with high sensitivity to input orders. Furthermore, we introduce a new prompting strategy to mitigate the position bias of LLM recommendation models called Ranking via Iterative SElection (RISE). We compare our proposed method against various baselines on key benchmark datasets. Experiment results show that our method reduces sensitivity to input ordering and improves stability without requiring model fine-tuning or post-processing. 

---
# FinCPRG: A Bidirectional Generation Pipeline for Hierarchical Queries and Rich Relevance in Financial Chinese Passage Retrieval 

**Authors**: Xuan Xu, Beilin Chu, Qinhong Lin, Yixiao Zhong, Fufang Wen, Jiaqi Liu, Binjie Fei, Yu Li, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.02222)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated significant potential in constructing passage retrieval datasets. However, existing methods still face limitations in expressing cross-doc query needs and controlling annotation quality. To address these issues, this paper proposes a bidirectional generation pipeline, which aims to generate 3-level hierarchical queries for both intra-doc and cross-doc scenarios and mine additional relevance labels on top of direct mapping annotation. The pipeline introduces two query generation methods: bottom-up from single-doc text and top-down from multi-doc titles. The bottom-up method uses LLMs to disassemble and generate structured queries at both sentence-level and passage-level simultaneously from intra-doc passages. The top-down approach incorporates three key financial elements--industry, topic, and time--to divide report titles into clusters and prompts LLMs to generate topic-level queries from each cluster. For relevance annotation, our pipeline not only relies on direct mapping annotation from the generation relationship but also implements an indirect positives mining method to enrich the relevant query-passage pairs. Using this pipeline, we constructed a Financial Passage Retrieval Generated dataset (FinCPRG) from almost 1.3k Chinese financial research reports, which includes hierarchical queries and rich relevance labels. Through evaluations of mined relevance labels, benchmarking and training experiments, we assessed the quality of FinCPRG and validated its effectiveness as a passage retrieval dataset for both training and benchmarking. 

---
# TreeRanker: Fast and Model-agnostic Ranking System for Code Suggestions in IDEs 

**Authors**: Daniele Cipollone, Egor Bogomolov, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02455)  

**Abstract**: Token-level code completion is one of the most critical features in modern Integrated Development Environments (IDEs). It assists developers by suggesting relevant identifiers and APIs during coding. While completions are typically derived from static analysis, their usefulness depends heavily on how they are ranked, as correct predictions buried deep in the list are rarely seen by users. Most current systems rely on hand-crafted heuristics or lightweight machine learning models trained on user logs, which can be further improved to capture context information and generalize across projects and coding styles. In this work, we propose a new scoring approach to ranking static completions using language models in a lightweight and model-agnostic way. Our method organizes all valid completions into a prefix tree and performs a single greedy decoding pass to collect token-level scores across the tree. This enables a precise token-aware ranking without needing beam search, prompt engineering, or model adaptations. The approach is fast, architecture-agnostic, and compatible with already deployed models for code completion. These findings highlight a practical and effective pathway for integrating language models into already existing tools within IDEs, and ultimately providing smarter and more responsive developer assistance. 

---
# End-to-End Personalization: Unifying Recommender Systems with Large Language Models 

**Authors**: Danial Ebrat, Tina Aminian, Sepideh Ahmadian, Luis Rueda  

**Link**: [PDF](https://arxiv.org/pdf/2508.01514)  

**Abstract**: Recommender systems are essential for guiding users through the vast and diverse landscape of digital content by delivering personalized and relevant suggestions. However, improving both personalization and interpretability remains a challenge, particularly in scenarios involving limited user feedback or heterogeneous item attributes. In this article, we propose a novel hybrid recommendation framework that combines Graph Attention Networks (GATs) with Large Language Models (LLMs) to address these limitations. LLMs are first used to enrich user and item representations by generating semantically meaningful profiles based on metadata such as titles, genres, and overviews. These enriched embeddings serve as initial node features in a user and movie bipartite graph, which is processed using a GAT based collaborative filtering model. To enhance ranking accuracy, we introduce a hybrid loss function that combines Bayesian Personalized Ranking (BPR), cosine similarity, and robust negative sampling. Post-processing involves reranking the GAT-generated recommendations using the LLM, which also generates natural-language justifications to improve transparency. We evaluated our model on benchmark datasets, including MovieLens 100k and 1M, where it consistently outperforms strong baselines. Ablation studies confirm that LLM-based embeddings and the cosine similarity term significantly contribute to performance gains. This work demonstrates the potential of integrating LLMs to improve both the accuracy and interpretability of recommender systems. 

---
# BioDisco: Multi-agent hypothesis generation with dual-mode evidence, iterative feedback and temporal evaluation 

**Authors**: Yujing Ke, Kevin George, Kathan Pandya, David Blumenthal, Maximilian Sprang, Gerrit Großmann, Sebastian Vollmer, David Antony Selby  

**Link**: [PDF](https://arxiv.org/pdf/2508.01285)  

**Abstract**: Identifying novel hypotheses is essential to scientific research, yet this process risks being overwhelmed by the sheer volume and complexity of available information. Existing automated methods often struggle to generate novel and evidence-grounded hypotheses, lack robust iterative refinement and rarely undergo rigorous temporal evaluation for future discovery potential. To address this, we propose BioDisco, a multi-agent framework that draws upon language model-based reasoning and a dual-mode evidence system (biomedical knowledge graphs and automated literature retrieval) for grounded novelty, integrates an internal scoring and feedback loop for iterative refinement, and validates performance through pioneering temporal and human evaluations and a Bradley-Terry paired comparison model to provide statistically-grounded assessment. Our evaluations demonstrate superior novelty and significance over ablated configurations representative of existing agentic architectures. Designed for flexibility and modularity, BioDisco allows seamless integration of custom language models or knowledge graphs, and can be run with just a few lines of code. We anticipate researchers using this practical tool as a catalyst for the discovery of new hypotheses. 

---
# MaRGen: Multi-Agent LLM Approach for Self-Directed Market Research and Analysis 

**Authors**: Roman Koshkin, Pengyu Dai, Nozomi Fujikawa, Masahito Togami, Marco Visentini-Scarzanella  

**Link**: [PDF](https://arxiv.org/pdf/2508.01370)  

**Abstract**: We present an autonomous framework that leverages Large Language Models (LLMs) to automate end-to-end business analysis and market report generation. At its core, the system employs specialized agents - Researcher, Reviewer, Writer, and Retriever - that collaborate to analyze data and produce comprehensive reports. These agents learn from real professional consultants' presentation materials at Amazon through in-context learning to replicate professional analytical methodologies. The framework executes a multi-step process: querying databases, analyzing data, generating insights, creating visualizations, and composing market reports. We also introduce a novel LLM-based evaluation system for assessing report quality, which shows alignment with expert human evaluations. Building on these evaluations, we implement an iterative improvement mechanism that optimizes report quality through automated review cycles. Experimental results show that report quality can be improved by both automated review cycles and consultants' unstructured knowledge. In experimental validation, our framework generates detailed 6-page reports in 7 minutes at a cost of approximately \$1. Our work could be an important step to automatically create affordable market insights. 

---
# I2CR: Intra- and Inter-modal Collaborative Reflections for Multimodal Entity Linking 

**Authors**: Ziyan Liu, Junwen Li, Kaiwen Li, Tong Ruan, Chao Wang, Xinyan He, Zongyu Wang, Xuezhi Cao, Jingping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02243)  

**Abstract**: Multimodal entity linking plays a crucial role in a wide range of applications. Recent advances in large language model-based methods have become the dominant paradigm for this task, effectively leveraging both textual and visual modalities to enhance performance. Despite their success, these methods still face two challenges, including unnecessary incorporation of image data in certain scenarios and the reliance only on a one-time extraction of visual features, which can undermine their effectiveness and accuracy. To address these challenges, we propose a novel LLM-based framework for the multimodal entity linking task, called Intra- and Inter-modal Collaborative Reflections. This framework prioritizes leveraging text information to address the task. When text alone is insufficient to link the correct entity through intra- and inter-modality evaluations, it employs a multi-round iterative strategy that integrates key visual clues from various aspects of the image to support reasoning and enhance matching accuracy. Extensive experiments on three widely used public datasets demonstrate that our framework consistently outperforms current state-of-the-art methods in the task, achieving improvements of 3.2%, 5.1%, and 1.6%, respectively. Our code is available at this https URL. 

---
# Evaluating User Experience in Conversational Recommender Systems: A Systematic Review Across Classical and LLM-Powered Approaches 

**Authors**: Raj Mahmud, Yufeng Wu, Abdullah Bin Sawad, Shlomo Berkovsky, Mukesh Prasad, A. Baki Kocaballi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02096)  

**Abstract**: Conversational Recommender Systems (CRSs) are receiving growing research attention across domains, yet their user experience (UX) evaluation remains limited. Existing reviews largely overlook empirical UX studies, particularly in adaptive and large language model (LLM)-based CRSs. To address this gap, we conducted a systematic review following PRISMA guidelines, synthesising 23 empirical studies published between 2017 and 2025. We analysed how UX has been conceptualised, measured, and shaped by domain, adaptivity, and LLM.
Our findings reveal persistent limitations: post hoc surveys dominate, turn-level affective UX constructs are rarely assessed, and adaptive behaviours are seldom linked to UX outcomes. LLM-based CRSs introduce further challenges, including epistemic opacity and verbosity, yet evaluations infrequently address these issues. We contribute a structured synthesis of UX metrics, a comparative analysis of adaptive and nonadaptive systems, and a forward-looking agenda for LLM-aware UX evaluation. These findings support the development of more transparent, engaging, and user-centred CRS evaluation practices. 

---
# Agentic Personalized Fashion Recommendation in the Age of Generative AI: Challenges, Opportunities, and Evaluation 

**Authors**: Yashar Deldjoo, Nima Rafiee, Mahdyar Ravanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2508.02342)  

**Abstract**: Fashion recommender systems (FaRS) face distinct challenges due to rapid trend shifts, nuanced user preferences, intricate item-item compatibility, and the complex interplay among consumers, brands, and influencers. Traditional recommendation approaches, largely static and retrieval-focused, struggle to effectively capture these dynamic elements, leading to decreased user satisfaction and elevated return rates. This paper synthesizes both academic and industrial viewpoints to map the distinctive output space and stakeholder ecosystem of modern FaRS, identifying the complex interplay among users, brands, platforms, and influencers, and highlighting the unique data and modeling challenges that arise.
We outline a research agenda for industrial FaRS, centered on five representative scenarios spanning static queries, outfit composition, and multi-turn dialogue, and argue that mixed-modality refinement-the ability to combine image-based references (anchors) with nuanced textual constraints-is a particularly critical task for real-world deployment. To this end, we propose an Agentic Mixed-Modality Refinement (AMMR) pipeline, which fuses multimodal encoders with agentic LLM planners and dynamic retrieval, bridging the gap between expressive user intent and fast-changing fashion inventories. Our work shows that moving beyond static retrieval toward adaptive, generative, and stakeholder-aware systems is essential to satisfy the evolving expectations of fashion consumers and brands. 

---
# DBAIOps: A Reasoning LLM-Enhanced Database Operation and Maintenance System using Knowledge Graphs 

**Authors**: Wei Zhou, Peng Sun, Xuanhe Zhou, Qianglei Zang, Ji Xu, Tieying Zhang, Guoliang Li, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01136)  

**Abstract**: The operation and maintenance (O&M) of database systems is critical to ensuring system availability and performance, typically requiring expert experience (e.g., identifying metric-to-anomaly relations) for effective diagnosis and recovery. However, existing automatic database O&M methods, including commercial products, cannot effectively utilize expert experience. On the one hand, rule-based methods only support basic O&M tasks (e.g., metric-based anomaly detection), which are mostly numerical equations and cannot effectively incorporate literal O&M experience (e.g., troubleshooting guidance in manuals). On the other hand, LLM-based methods, which retrieve fragmented information (e.g., standard documents + RAG), often generate inaccurate or generic results. To address these limitations, we present DBAIOps, a novel hybrid database O&M system that combines reasoning LLMs with knowledge graphs to achieve DBA-style diagnosis. First, DBAIOps introduces a heterogeneous graph model for representing the diagnosis experience, and proposes a semi-automatic graph construction algorithm to build that graph from thousands of documents. Second, DBAIOps develops a collection of (800+) reusable anomaly models that identify both directly alerted metrics and implicitly correlated experience and metrics. Third, for each anomaly, DBAIOps proposes a two-stage graph evolution mechanism to explore relevant diagnosis paths and identify missing relations automatically. It then leverages a reasoning LLM (e.g., DeepSeek-R1) to infer root causes and generate clear diagnosis reports for both DBAs and common users. Our evaluation over four mainstream database systems (Oracle, MySQL, PostgreSQL, and DM8) demonstrates that DBAIOps outperforms state-of-the-art baselines, 34.85% and 47.22% higher in root cause and human evaluation accuracy, respectively. 

---
# Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction 

**Authors**: Yuerong Song, Xiaoran Liu, Ruixiao Li, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02558)  

**Abstract**: Diffusion Large Language Models (dLLMs) enable breakthroughs in reasoning and parallel decoding but suffer from prohibitive quadratic computational complexity and memory overhead during inference. Current caching techniques accelerate decoding by storing full-layer states, yet impose substantial memory usage that limit long-context applications. Our analysis of attention patterns in dLLMs reveals persistent cross-layer sparsity, with pivotal tokens remaining salient across decoding steps and low-relevance tokens staying unimportant, motivating selective cache eviction. We propose Sparse-dLLM, the first training-free framework integrating dynamic cache eviction with sparse attention via delayed bidirectional sparse caching. By leveraging the stability of token saliency over steps, it retains critical tokens and dynamically evicts unimportant prefix/suffix entries using an attention-guided strategy. Extensive experiments on LLaDA and Dream series demonstrate Sparse-dLLM achieves up to 10$\times$ higher throughput than vanilla dLLMs, with comparable performance and similar peak memory costs, outperforming previous methods in efficiency and effectiveness. 

---
# PoeTone: A Framework for Constrained Generation of Structured Chinese Songci with LLMs 

**Authors**: Zhan Qu, Shuzhou Yuan, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2508.02515)  

**Abstract**: This paper presents a systematic investigation into the constrained generation capabilities of large language models (LLMs) in producing Songci, a classical Chinese poetry form characterized by strict structural, tonal, and rhyme constraints defined by Cipai templates. We first develop a comprehensive, multi-faceted evaluation framework that includes: (i) a formal conformity score, (ii) automated quality assessment using LLMs, (iii) human evaluation, and (iv) classification-based probing tasks. Using this framework, we evaluate the generative performance of 18 LLMs, including 3 proprietary models and 15 open-source models across four families, under five prompting strategies: zero-shot, one-shot, completion-based, instruction-tuned, and chain-of-thought. Finally, we propose a Generate-Critic architecture in which the evaluation framework functions as an automated critic. Leveraging the critic's feedback as a reward signal, we fine-tune three lightweight open-source LLMs via supervised fine-tuning (SFT), resulting in improvements of up to 5.88% in formal conformity. Our findings offer new insights into the generative strengths and limitations of LLMs in producing culturally significant and formally constrained literary texts. 

---
# Modular Arithmetic: Language Models Solve Math Digit by Digit 

**Authors**: Tanja Baeumel, Daniil Gurgurov, Yusser al Ghussin, Josef van Genabith, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2508.02513)  

**Abstract**: While recent work has begun to uncover the internal strategies that Large Language Models (LLMs) employ for simple arithmetic tasks, a unified understanding of their underlying mechanisms is still lacking. We extend recent findings showing that LLMs represent numbers in a digit-wise manner and present evidence for the existence of digit-position-specific circuits that LLMs use to perform simple arithmetic tasks, i.e. modular subgroups of MLP neurons that operate independently on different digit positions (units, tens, hundreds). Notably, such circuits exist independently of model size and of tokenization strategy, i.e. both for models that encode longer numbers digit-by-digit and as one token. Using Feature Importance and Causal Interventions, we identify and validate the digit-position-specific circuits, revealing a compositional and interpretable structure underlying the solving of arithmetic problems in LLMs. Our interventions selectively alter the model's prediction at targeted digit positions, demonstrating the causal role of digit-position circuits in solving arithmetic tasks. 

---
# LatentPrompt: Optimizing Promts in Latent Space 

**Authors**: Mateusz Bystroński, Grzegorz Piotrowski, Nitesh V. Chawla, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2508.02452)  

**Abstract**: Recent advances have shown that optimizing prompts for Large Language Models (LLMs) can significantly improve task performance, yet many optimization techniques rely on heuristics or manual exploration. We present LatentPrompt, a model-agnostic framework for prompt optimization that leverages latent semantic space to automatically generate, evaluate, and refine candidate prompts without requiring hand-crafted rules. Beginning with a set of seed prompts, our method embeds them in a continuous latent space and systematically explores this space to identify prompts that maximize task-specific performance. In a proof-of-concept study on the Financial PhraseBank sentiment classification benchmark, LatentPrompt increased classification accuracy by approximately 3 percent after a single optimization cycle. The framework is broadly applicable, requiring only black-box access to an LLM and an automatic evaluation metric, making it suitable for diverse domains and tasks. 

---
# AI-Based Measurement of Innovation: Mapping Expert Insight into Large Language Model Applications 

**Authors**: Robin Nowak, Patrick Figge, Carolin Haeussler  

**Link**: [PDF](https://arxiv.org/pdf/2508.02430)  

**Abstract**: Measuring innovation often relies on context-specific proxies and on expert evaluation. Hence, empirical innovation research is often limited to settings where such data is available. We investigate how large language models (LLMs) can be leveraged to overcome the constraints of manual expert evaluations and assist researchers in measuring innovation. We design an LLM framework that reliably approximates domain experts' assessment of innovation from unstructured text data. We demonstrate the performance and broad applicability of this framework through two studies in different contexts: (1) the innovativeness of software application updates and (2) the originality of user-generated feedback and improvement ideas in product reviews. We compared the performance (F1-score) and reliability (consistency rate) of our LLM framework against alternative measures used in prior innovation studies, and to state-of-the-art machine learning- and deep learning-based models. The LLM framework achieved higher F1-scores than the other approaches, and its results are highly consistent (i.e., results do not change across runs). This article equips R&D personnel in firms, as well as researchers, reviewers, and editors, with the knowledge and tools to effectively use LLMs for measuring innovation and evaluating the performance of LLM-based innovation measures. In doing so, we discuss, the impact of important design decisions-including model selection, prompt engineering, training data size, training data distribution, and parameter settings-on performance and reliability. Given the challenges inherent in using human expert evaluation and existing text-based measures, our framework has important implications for harnessing LLMs as reliable, increasingly accessible, and broadly applicable research tools for measuring innovation. 

---
# Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs 

**Authors**: Jérémie Dentan, Davide Buscaldi, Sonia Vanier  

**Link**: [PDF](https://arxiv.org/pdf/2508.02573)  

**Abstract**: Verbatim memorization in Large Language Models (LLMs) is a multifaceted phenomenon involving distinct underlying mechanisms. We introduce a novel method to analyze the different forms of memorization described by the existing taxonomy. Specifically, we train Convolutional Neural Networks (CNNs) on the attention weights of the LLM and evaluate the alignment between this taxonomy and the attention weights involved in decoding.
We find that the existing taxonomy performs poorly and fails to reflect distinct mechanisms within the attention blocks. We propose a new taxonomy that maximizes alignment with the attention weights, consisting of three categories: memorized samples that are guessed using language modeling abilities, memorized samples that are recalled due to high duplication in the training set, and non-memorized samples. Our results reveal that few-shot verbatim memorization does not correspond to a distinct attention mechanism. We also show that a significant proportion of extractable samples are in fact guessed by the model and should therefore be studied separately. Finally, we develop a custom visual interpretability technique to localize the regions of the attention weights involved in each form of memorization. 

---
# MArgE: Meshing Argumentative Evidence from Multiple Large Language Models for Justifiable Claim Verification 

**Authors**: Ming Pok Ng, Junqi Jiang, Gabriel Freedman, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2508.02584)  

**Abstract**: Leveraging outputs from multiple large language models (LLMs) is emerging as a method for harnessing their power across a wide range of tasks while mitigating their capacity for making errors, e.g., hallucinations. However, current approaches to combining insights from multiple LLMs often involve unstructured interactions (e.g., free debate), resulting in model generations that are not faithfully justifiable. In this work, we introduce MArgE, a novel framework to provide formal structure to the evidence from each LLM, in the form of a tree of extracted arguments, for the task of claim verification. We use a variant of Argumentative LLMs (ArgLLMs), i.e. LLMs driven by frameworks and semantics from the field of computational argumentation, to construct structured argument trees for given claims. This process creates an inspectable pathway from the initial arguments to the final claim verification decisions, providing a faithful justification thereof. We show experimentally that MArgE can significantly outperform single LLMs, including three open-source models (4B to 8B parameters), GPT-4o-mini and existing ArgLLMs, as well as prior methods for unstructured multi-LLM debates. We thus demonstrate the advantages of incorporating formal, argumentative reasoning mechanisms when combining multiple LLM outputs. 

---
# From Monolingual to Bilingual: Investigating Language Conditioning in Large Language Models for Psycholinguistic Tasks 

**Authors**: Shuzhou Yuan, Zhan Qu, Mario Tawfelis, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2508.02502)  

**Abstract**: Large Language Models (LLMs) exhibit strong linguistic capabilities, but little is known about how they encode psycholinguistic knowledge across languages. We investigate whether and how LLMs exhibit human-like psycholinguistic responses under different linguistic identities using two tasks: sound symbolism and word valence. We evaluate two models, Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct, under monolingual and bilingual prompting in English, Dutch, and Chinese. Behaviorally, both models adjust their outputs based on prompted language identity, with Qwen showing greater sensitivity and sharper distinctions between Dutch and Chinese. Probing analysis reveals that psycholinguistic signals become more decodable in deeper layers, with Chinese prompts yielding stronger and more stable valence representations than Dutch. Our results demonstrate that language identity conditions both output behavior and internal representations in LLMs, providing new insights into their application as models of cross-linguistic cognition. 

---
# Understanding and Mitigating Political Stance Cross-topic Generalization in Large Language Models 

**Authors**: Jiayi Zhang, Shu Yang, Junchao Wu, Derek F. Wong, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02360)  

**Abstract**: Fine-tuning Large Language Models on a political topic will significantly manipulate their political stance on various issues and unintentionally affect their stance on unrelated topics. While previous studies have proposed this issue, there is still a lack of understanding regarding the internal representations of these stances and the mechanisms that lead to unintended cross-topic generalization. In this paper, we systematically explore the internal mechanisms underlying this phenomenon from a neuron-level perspective and how to mitigate the cross-topic generalization of political fine-tuning. Firstly, we propose Political Neuron Localization through Activation Contrasting (PNLAC) to identify two distinct types of political neurons: general political neurons, which govern stance across multiple political topics, and topic-specific neurons} that affect the model's political stance on individual topics. We find the existence of these political neuron types across four models and datasets through activation patching experiments. Leveraging these insights, we introduce InhibitFT, an inhibition-based fine-tuning method, effectively mitigating the cross-topic stance generalization. Experimental results demonstrate the robustness of identified neuron types across various models and datasets, and show that InhibitFT significantly reduces the cross-topic stance generalization by 20% on average, while preserving topic-specific performance. Moreover, we demonstrate that selectively inhibiting only 5% of neurons is sufficient to effectively mitigate the cross-topic stance generalization. 

---
# Mitigating Attention Hacking in Preference-Based Reward Modeling via Interaction Distillation 

**Authors**: Jianxiang Zang, Meiling Ning, Shihan Dou, Jiazheng Zhang, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02618)  

**Abstract**: The reward model (RM), as the core component of reinforcement learning from human feedback (RLHF) for large language models (LLMs), responsible for providing reward signals to generated responses. However, mainstream preference modeling in RM is inadequate in terms of token-level interaction, making its judgment signals vulnerable to being hacked by misallocated attention to context. This stems from two fundamental limitations: (1) Current preference modeling employs decoder-only architectures, where the unidirectional causal attention mechanism leads to forward-decaying intra-sequence attention within the prompt-response sequence. (2) The independent Siamese-encoding paradigm induces the absence of token-level inter-sequence attention between chosen and rejected sequences. To address this "attention hacking", we propose "Interaction Distillation", a novel training framework for more adequate preference modeling through attention-level optimization. The method introduces an interaction-based natural language understanding model as the teacher to provide sophisticated token interaction patterns via comprehensive attention, and guides the preference modeling to simulate teacher model's interaction pattern through an attentional alignment objective. Through extensive experiments, interaction distillation has demonstrated its ability to provide more stable and generalizable reward signals compared to state-of-the-art RM optimization methods that target data noise, highlighting the attention hacking constitute a more fundamental limitation in RM. 

---
# LaMPE: Length-aware Multi-grained Position Encoding for Adaptive Long-context Scaling Without Training 

**Authors**: Sikui Zhang, Guangze Gao, Ziyun Gan, Chunfeng Yuan, Zefeng Lin, Houwen Peng, Bing Li, Weiming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02308)  

**Abstract**: Large language models (LLMs) experience significant performance degradation when the input exceeds the pretraining context window, primarily due to the out-of-distribution (OOD) behavior of Rotary Position Embedding (RoPE). Recent studies mitigate this problem by remapping OOD positions into the in-distribution range with fixed mapping strategies, ignoring the dynamic relationship between input length and the model's effective context window. To this end, we propose Length-aware Multi-grained Positional Encoding (LaMPE), a training-free method that fully utilizes the model's effective context window for adaptive long-context scaling in LLMs. Motivated by the left-skewed frequency distribution of relative positions, LaMPE establishes a dynamic relationship between mapping length and input length through a parametric scaled sigmoid function to adaptively allocate positional capacity across varying input lengths. Meanwhile, LaMPE devises a novel multi-grained attention mechanism that strategically allocates positional resolution across different sequence regions to capture both fine-grained locality and long-range dependencies. Our method can be seamlessly applied to a wide range of RoPE-based LLMs without training. Extensive experiments on three representative LLMs across five mainstream long-context benchmarks demonstrate that LaMPE achieves significant performance improvements compared to existing length extrapolation methods. The code will be released at this https URL. 

---
# Isolating Culture Neurons in Multilingual Large Language Models 

**Authors**: Danial Namazifard, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2508.02241)  

**Abstract**: Language and culture are deeply intertwined, yet it is so far unclear how and where multilingual large language models encode culture. Here, we extend upon an established methodology for identifying language-specific neurons and extend it to localize and isolate culture-specific neurons, carefully disentangling their overlap and interaction with language-specific neurons. To facilitate our experiments, we introduce MUREL, a curated dataset of 85.2 million tokens spanning six different cultures. Our localization and intervention experiments show that LLMs encode different cultures in distinct neuron populations, predominantly in upper layers, and that these culture neurons can be modulated independently from language-specific neurons or those specific to other cultures. These findings suggest that cultural knowledge and propensities in multilingual language models can be selectively isolated and edited - promoting fairness, inclusivity, and alignment. Code and data is available at this https URL . 

---
# Proof2Hybrid: Automatic Mathematical Benchmark Synthesis for Proof-Centric Problems 

**Authors**: Yebo Peng, Zixiang Liu, Yaoming Li, Zhizhuo Yang, Xinye Xu, Bowen Ye, Weijun Yuan, Zihan Wang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02208)  

**Abstract**: Evaluating the mathematical capability of Large Language Models (LLMs) is a critical yet challenging frontier. Existing benchmarks fall short, particularly for proof-centric problems, as manual creation is unscalable and costly, leaving the true mathematical abilities of LLMs largely unassessed. To overcome these barriers, we propose Proof2Hybrid, the first fully automated framework that synthesizes high-quality, proof-centric benchmarks from natural language mathematical corpora. The key novelty of our solution is Proof2X, a roadmap of converting mathematical proofs into various kinds of questions that are easy to verify. Instructed by this roadmap, we propose a new type of hybrid-formatted questions, named ``$m$-out-of-$n$ multiple judge questions'', specifically designed to enable robust, automatic evaluation while being resilient to guessing and superficial pattern matching inherent in traditional formats. As a demonstration of our framework, we introduce AlgGeoTest, a benchmark for algebraic geometry--a frontier domain of modern mathematics--comprising 456 challenging items. Our extensive evaluations on state-of-the-art LLMs using AlgGeoTest reveal profound deficits in their comprehension of algebraic geometry, providing a more precise measure of their true mathematical capabilities. Our framework and benchmark pave the way for a new wave of in-depth research into the mathematical intelligence of AI systems. 

---
# Harnessing Temporal Databases for Systematic Evaluation of Factual Time-Sensitive Question-Answering in Large Language Models 

**Authors**: Soyeon Kim, Jindong Wang, Xing Xie, Steven Euijong Whang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02045)  

**Abstract**: Facts evolve over time, making it essential for Large Language Models (LLMs) to handle time-sensitive factual knowledge accurately and reliably. While factual Time-Sensitive Question-Answering (TSQA) tasks have been widely studied, existing benchmarks often rely on manual curation or a small, fixed set of predefined templates, which restricts scalable and comprehensive TSQA evaluation. To address these challenges, we propose TDBench, a new benchmark that systematically constructs TSQA pairs by harnessing temporal databases and database techniques such as temporal SQL and functional dependencies. We also introduce a fine-grained evaluation metric called time accuracy, which assesses the validity of time references in model explanations alongside traditional answer accuracy to enable a more reliable TSQA evaluation. Extensive experiments on contemporary LLMs show how \ours{} enables scalable and comprehensive TSQA evaluation while reducing the reliance on human labor, complementing existing Wikipedia/Wikidata-based TSQA evaluation approaches by enabling LLM evaluation on application-specific data and seamless multi-hop question generation. Code and data are publicly available at: this https URL. 

---
# ProCut: LLM Prompt Compression via Attribution Estimation 

**Authors**: Zhentao Xu, Fengyi Li, Albert Chen, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02053)  

**Abstract**: In large-scale industrial LLM systems, prompt templates often expand to thousands of tokens as teams iteratively incorporate sections such as task instructions, few-shot examples, and heuristic rules to enhance robustness and coverage. This expansion leads to bloated prompts that are difficult to maintain and incur significant inference latency and serving costs. To address this, we introduce Prompt Compression via Attribution Estimation (ProCut), a flexible, LLM-agnostic, training-free framework that compresses prompts through attribution analysis. ProCut segments prompt templates into semantically meaningful units, quantifies their impact on task performance, and prunes low-utility components. Through extensive experiments on five public benchmark datasets and real-world industrial prompts, we show that ProCut achieves substantial prompt size reductions (78% fewer tokens in production) while maintaining or even slightly improving task performance (up to 62% better than alternative methods). We further introduce an LLM-driven attribution estimator that reduces compression latency by over 50%, and demonstrate that ProCut integrates seamlessly with existing prompt-optimization frameworks to produce concise, high-performing prompts. 

---
# Diagnosing Memorization in Chain-of-Thought Reasoning, One Token at a Time 

**Authors**: Huihan Li, You Chen, Siyuan Wang, Yixin He, Ninareh Mehrabi, Rahul Gupta, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.02037)  

**Abstract**: Large Language Models (LLMs) perform well on reasoning benchmarks but often fail when inputs alter slightly, raising concerns about the extent to which their success relies on memorization. This issue is especially acute in Chain-of-Thought (CoT) reasoning, where spurious memorized patterns can trigger intermediate errors that cascade into incorrect final answers. We introduce STIM, a novel framework for Source-aware Token-level Identification of Memorization, which attributes each token in a reasoning chain to one of multiple memorization sources - local, mid-range, or long-range - based on their statistical co-occurrence with the token in the pretraining corpus. Our token-level analysis across tasks and distributional settings reveals that models rely more on memorization in complex or long-tail cases, and that local memorization is often the dominant driver of errors, leading to up to 67% of wrong tokens. We also show that memorization scores from STIM can be effective in predicting the wrong tokens in the wrong reasoning step. STIM offers a powerful tool for diagnosing and improving model reasoning and can generalize to other structured step-wise generation tasks. 

---
# SpeechR: A Benchmark for Speech Reasoning in Large Audio-Language Models 

**Authors**: Wanqi Yang, Yanda Li, Yunchao Wei, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02018)  

**Abstract**: Large audio-language models (LALMs) have achieved near-human performance in sentence-level transcription and emotion recognition. However, existing evaluations focus mainly on surface-level perception, leaving the capacity of models for contextual and inference-driven reasoning in speech-based scenarios insufficiently examined. To address this gap, we introduce SpeechR, a unified benchmark for evaluating reasoning over speech in large audio-language models. SpeechR evaluates models along three key dimensions: factual retrieval, procedural inference, and normative judgment. It includes three distinct evaluation formats. The multiple-choice version measures answer selection accuracy. The generative version assesses the coherence and logical consistency of reasoning chains. The acoustic-feature version investigates whether variations in stress and emotion affect reasoning performance. Evaluations on eleven state-of-the-art LALMs reveal that high transcription accuracy does not translate into strong reasoning capabilities. SpeechR establishes a structured benchmark for evaluating reasoning in spoken language, enabling more targeted analysis of model capabilities across diverse dialogue-based tasks. 

---
# When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models 

**Authors**: Jin Li, Keyu Wang, Shu Yang, Zhuoran Zhang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02087)  

**Abstract**: Large Language Models (LLMs) often exhibit sycophantic behavior, agreeing with user-stated opinions even when those contradict factual knowledge. While prior work has documented this tendency, the internal mechanisms that enable such behavior remain poorly understood. In this paper, we provide a mechanistic account of how sycophancy arises within LLMs. We first systematically study how user opinions induce sycophancy across different model families. We find that simple opinion statements reliably induce sycophancy, whereas user expertise framing has a negligible impact. Through logit-lens analysis and causal activation patching, we identify a two-stage emergence of sycophancy: (1) a late-layer output preference shift and (2) deeper representational divergence. We also verify that user authority fails to influence behavior because models do not encode it internally. In addition, we examine how grammatical perspective affects sycophantic behavior, finding that first-person prompts (``I believe...'') consistently induce higher sycophancy rates than third-person framings (``They believe...'') by creating stronger representational perturbations in deeper layers. These findings highlight that sycophancy is not a surface-level artifact but emerges from a structural override of learned knowledge in deeper layers, with implications for alignment and truthful AI systems. 

---
# TIBSTC-CoT: A Multi-Domain Instruction Dataset for Chain-of-Thought Reasoning in Language Models 

**Authors**: Fan Gao, Cheng Huang, Nyima Tashi, Yutong Liu, Xiangxiang Wang, Thupten Tsering, Ban Ma-bao, Renzeg Duojie, Gadeng Luosang, Rinchen Dongrub, Dorje Tashi, Xiao Feng, Hao Wang, Yongbin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01977)  

**Abstract**: To address the severe data scarcity in Tibetan, a low-resource language spoken by over six million people, we introduce TIBSTC-CoT, the large-scale, multi-domain Tibetan dataset automatically constructed via chain-of-thought prompting with large language models (LLMs). TIBSTC-CoT establishes a scalable and reproducible framework for dataset creation in low-resource settings, covering diverse domains and reasoning patterns essential for language understanding and generation. Building on this dataset, we develop the Sunshine-thinking LLM family, a series of Tibetan-centric LLMs equipped with chain-of-thought capabilities. Trained entirely on TIBSTC-CoT, Sunshine-thinking has demonstrated strong reasoning and generation performance, comparable to state-of-the-art (SOTA) multilingual LLMs. Our work marks a significant step toward inclusive AI by enabling high-quality Tibetan language processing through both resource creation and model innovation. All data are available: this https URL. 

---
# Prompting Large Language Models to Detect Dementia Family Caregivers 

**Authors**: Md Badsha Biswas, Özlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2508.01999)  

**Abstract**: Social media, such as Twitter, provides opportunities for caregivers of dementia patients to share their experiences and seek support for a variety of reasons. Availability of this information online also paves the way for the development of internet-based interventions in their support. However, for this purpose, tweets written by caregivers of dementia patients must first be identified. This paper demonstrates our system for the SMM4H 2025 shared task 3, which focuses on detecting tweets posted by individuals who have a family member with dementia. The task is outlined as a binary classification problem, differentiating between tweets that mention dementia in the context of a family member and those that do not. Our solution to this problem explores large language models (LLMs) with various prompting methods. Our results show that a simple zero-shot prompt on a fine-tuned model yielded the best results. Our final system achieved a macro F1-score of 0.95 on the validation set and the test set. Our full code is available on GitHub. 

---
# Collaborative Chain-of-Agents for Parametric-Retrieved Knowledge Synergy 

**Authors**: Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, Lizhe Zhang, Yan Liu, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01696)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising framework for enhancing the capabilities of Large Language Models (LLMs), especially in knowledge-intensive tasks. Despite its advantages, current RAG methods often struggle to *fully exploit knowledge during generation*. In particular, the synergy between the model's internal parametric knowledge and external retrieved knowledge remains limited. Retrieved contents may sometimes mislead generation, while certain generated content can guide the model toward more accurate outputs. In this work, we propose Collaborative Chain-of-Agents, a framework designed to enhance explicitly synergy over both parametric and retrieved knowledge. Specifically, we first introduce CoCoA-zero, a multi-agent RAG framework that first performs conditional knowledge induction and then reasons answers. Building on this, we develop CoCoA, a long-chain training strategy that synthesizes extended multi-agent reasoning trajectories from CoCoA-zero to fine-tune the LLM. This strategy enhances the model's capability to explicitly integrate and jointly leverage parametric and retrieved knowledge. Experiments results show that CoCoA-zero and CoCoA achieve superior performance on open-domain and multi-hop QA tasks. 

---
# CultureGuard: Towards Culturally-Aware Dataset and Guard Model for Multilingual Safety Applications 

**Authors**: Raviraj Joshi, Rakesh Paul, Kanishk Singla, Anusha Kamath, Michael Evans, Katherine Luna, Shaona Ghosh, Utkarsh Vaidya, Eileen Long, Sanjay Singh Chauhan, Niranjan Wartikar  

**Link**: [PDF](https://arxiv.org/pdf/2508.01710)  

**Abstract**: The increasing use of Large Language Models (LLMs) in agentic applications highlights the need for robust safety guard models. While content safety in English is well-studied, non-English languages lack similar advancements due to the high cost of collecting culturally aligned labeled datasets. We present CultureGuard, a novel solution for curating culturally aligned, high-quality safety datasets across multiple languages. Our approach introduces a four-stage synthetic data generation and filtering pipeline: cultural data segregation, cultural data adaptation, machine translation, and quality filtering. This pipeline enables the conversion and expansion of the Nemotron-Content-Safety-Dataset-V2 English safety dataset into eight distinct languages: Arabic, German, Spanish, French, Hindi, Japanese, Thai, and Chinese. The resulting dataset, Nemotron-Content-Safety-Dataset-Multilingual-v1, comprises 386,661 samples in 9 languages and facilitates the training of Llama-3.1-Nemotron-Safety-Guard-Multilingual-8B-v1 via LoRA-based fine-tuning. The final model achieves state-of-the-art performance on several multilingual content safety benchmarks. We also benchmark the latest open LLMs on multilingual safety and observe that these LLMs are more prone to give unsafe responses when prompted in non-English languages. This work represents a significant step toward closing the safety gap in multilingual LLMs by enabling the development of culturally aware safety guard models. 

---
# The Bidirectional Process Reward Model 

**Authors**: Lingyin Zhang, Jun Gao, Xiaoxue Ren, Ziqiang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01682)  

**Abstract**: Process Reward Models (PRMs) have emerged as a promising approach to enhance the reasoning quality of Large Language Models (LLMs) by assigning fine-grained scores to intermediate reasoning steps within a solution trajectory. However, existing PRMs predominantly adopt a unidirectional left-to-right (L2R) evaluation paradigm, which limits their ability to leverage global context, making it challenging to verify the consistency of earlier steps based on later ones. In light of these challenges, we propose a novel bidirectional evaluation paradigm, named Bidirectional Process Reward Model (BiPRM). BiPRM seamlessly incorporates a parallel right-to-left (R2L) evaluation stream alongside the conventional L2R flow, enabling later reasoning steps to help assess earlier ones in real time. Notably, the built-in R2L evaluation is implemented solely through prompt modifications that reverse the original reasoning trajectory, without any additional parameters or inference latency introduced. This ensures BiPRM remains both efficient and broadly compatible with existing PRM studies. We conduct extensive experiments on two mathematical reasoning benchmarks using samples generated by three different policy models. Our method, BiPRM, is evaluated across three backbones and three distinct PRM objectives. Across all settings, BiPRM consistently outperforms unidirectional baselines, achieving up to a 31.9% improvement in stepwise reward evaluation. Generally, our results highlight BiPRM's effectiveness, robustness, and general applicability, offering a promising new direction for process-based reward modeling. 

---
# Quantum-RAG and PunGPT2: Advancing Low-Resource Language Generation and Retrieval for the Punjabi Language 

**Authors**: Jaskaranjeet Singh, Rakesh Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2508.01918)  

**Abstract**: Despite the rapid advancement of large language models (LLMs), low-resource languages remain largely excluded from the NLP landscape. We present PunGPT2, the first fully open-source suite of Punjabi large language models, trained from scratch on a 35GB domain-diverse corpus encompassing literature, religious texts, news, and social discourse. Unlike prior multilingual approaches, PunGPT2 captures rich syntactic and morphological features unique to Punjabi through a tokenizer optimised with byte pair encoding and linguistically aligned pretraining objectives. To improve factual grounding and domain recall, we introduce Pun-RAG, a retrieval-augmented generation framework combining PunGPT2 with a dense FAISS retriever over a curated Punjabi knowledge base. We further develop Pun-Instruct, a parameter-efficient, instruction-tuned variant using QLoRA, enabling robust zero-shot and instruction-following performance with significantly reduced compute needs.
As a key innovation, we propose Quantum-RAG, a novel hybrid retrieval system that fuses sparse (BM25) and dense methods with quantum-inspired semantic matching. By encoding queries using amplitude-based embeddings and retrieving via quantum kernel similarity, Quantum-RAG achieves improved contextual relevance with minimal memory overhead marking the first practical integration of quantum representations in low-resource language generation. Our models significantly outperform strong multilingual baselines (mBERT, mT5, MuRIL) in perplexity, factuality, and fluency. This work provides a scalable, reproducible blueprint for extending LLM capabilities to underrepresented languages and pioneers quantum-aware retrieval in low-resource NLP 

---
# CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions 

**Authors**: Tae Soo Kim, Yoonjoo Lee, Yoonah Park, Jiho Kim, Young-Ho Kim, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01674)  

**Abstract**: Personalization of Large Language Models (LLMs) often assumes users hold static preferences that reflect globally in all tasks. In reality, humans hold dynamic preferences that change depending on the context. As users interact with an LLM in various contexts, they naturally reveal their contextual preferences, which a model must infer and apply in future contexts to ensure alignment. To assess this, we introduce CUPID, a benchmark of 756 human-curated interaction session histories between users and LLM-based chat assistants. In each interaction session, the user provides a request in a specific context and expresses their preference through multi-turn feedback. Given a new user request and prior interaction sessions, our benchmark assesses whether LLMs can infer the preference relevant to this request and generate a response that satisfies this preference. With CUPID, we evaluated 10 open and proprietary LLMs, revealing that state-of-the-art LLMs struggle to infer preferences from multi-turn interactions and fail to discern what previous context is relevant to a new request -- under 50% precision and 65% recall. Our work highlights the need to advance LLM capabilities for more contextually personalized interactions and proposes CUPID as a resource to drive these improvements. 

---
# Authorship Attribution in Multilingual Machine-Generated Texts 

**Authors**: Lucio La Cava, Dominik Macko, Róbert Móro, Ivan Srba, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.01656)  

**Abstract**: As Large Language Models (LLMs) have reached human-like fluency and coherence, distinguishing machine-generated text (MGT) from human-written content becomes increasingly difficult. While early efforts in MGT detection have focused on binary classification, the growing landscape and diversity of LLMs require a more fine-grained yet challenging authorship attribution (AA), i.e., being able to identify the precise generator (LLM or human) behind a text. However, AA remains nowadays confined to a monolingual setting, with English being the most investigated one, overlooking the multilingual nature and usage of modern LLMs. In this work, we introduce the problem of Multilingual Authorship Attribution, which involves attributing texts to human or multiple LLM generators across diverse languages. Focusing on 18 languages -- covering multiple families and writing scripts -- and 8 generators (7 LLMs and the human-authored class), we investigate the multilingual suitability of monolingual AA methods, their cross-lingual transferability, and the impact of generators on attribution performance. Our results reveal that while certain monolingual AA methods can be adapted to multilingual settings, significant limitations and challenges remain, particularly in transferring across diverse language families, underscoring the complexity of multilingual AA and the need for more robust approaches to better match real-world scenarios. 

---
# MLP Memory: Language Modeling with Retriever-pretrained External Memory 

**Authors**: Rubin Wei, Jiaqi Cao, Jiarui Wang, Jushi Kai, Qipeng Guo, Bowen Zhou, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01832)  

**Abstract**: While modern decoder-only LLMs achieve superior performance across various domains, hallucinations have risen to be a common problem in their generated text, hindering their application in knowledge-intensive tasks. Retriever-augmented generation (RAG) offers a solution, but the non-parametric nature of the retriever hinders its deep interaction with LLM. In this work, we propose to decouple memorization from the LLM decoder using a pretrained, differentiable external memory. The external memory is an MLP pretrained by imitating the behavior of a retriever on the entire pretraining dataset. Our resulting architecture, which comprises a transformer decoder and an external MLP memory pretrained on language modeling and retriever imitation respectively, demonstrates strong perplexity and performance on downstream tasks. Experiments show our architecture exhibits steeper power-law scaling with model size, achieving 17.5% and 24.1% improvement on WikiText-103 and Web datasets compared to decoder-only models while benefiting from added training without overfitting. We demonstrate superior performance on three hallucination benchmarks and nine memory-intensive tasks. Additionally, our approach delivers $80\times$ speedup over $k$NN-LM (500M tokens) and $1.3\times$ faster inference than decoder-only models. Unlike $k$NN-LM, which impairs reasoning, our MLP memory improves StrategyQA performance. We will open-source our code and models in the future. 

---
# Harnessing Collective Intelligence of LLMs for Robust Biomedical QA: A Multi-Model Approach 

**Authors**: Dimitra Panou, Alexandros C. Dimopoulos, Manolis Koubarakis, Martin Reczko  

**Link**: [PDF](https://arxiv.org/pdf/2508.01480)  

**Abstract**: Biomedical text mining and question-answering are essential yet highly demanding tasks, particularly in the face of the exponential growth of biomedical literature. In this work, we present our participation in the 13th edition of the BioASQ challenge, which involves biomedical semantic question-answering for Task 13b and biomedical question-answering for developing topics for the Synergy task. We deploy a selection of open-source large language models (LLMs) as retrieval-augmented generators to answer biomedical questions. Various models are used to process the questions. A majority voting system combines their output to determine the final answer for Yes/No questions, while for list and factoid type questions, the union of their answers in used. We evaluated 13 state-of-the-art open source LLMs, exploring all possible model combinations to contribute to the final answer, resulting in tailored LLM pipelines for each question type. Our findings provide valuable insight into which combinations of LLMs consistently produce superior results for specific question types. In the four rounds of the 2025 BioASQ challenge, our system achieved notable results: in the Synergy task, we secured 1st place for ideal answers and 2nd place for exact answers in round 2, as well as two shared 1st places for exact answers in round 3 and 4. 

---
# From Query to Logic: Ontology-Driven Multi-Hop Reasoning in LLMs 

**Authors**: Haonan Bian, Yutao Qi, Rui Yang, Yuanxi Che, Jiaqian Wang, Heming Xia, Ranran Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01424)  

**Abstract**: Large Language Models (LLMs), despite their success in question answering, exhibit limitations in complex multi-hop question answering (MQA) tasks that necessitate non-linear, structured reasoning. This limitation stems from their inability to adequately capture deep conceptual relationships between entities. To overcome this challenge, we present **ORACLE** (**O**ntology-driven **R**easoning **A**nd **C**hain for **L**ogical **E**ucidation), a training-free framework that combines LLMs' generative capabilities with the structural benefits of knowledge graphs. Our approach operates through three stages: (1) dynamic construction of question-specific knowledge ontologies using LLMs, (2) transformation of these ontologies into First-Order Logic reasoning chains, and (3) systematic decomposition of the original query into logically coherent sub-questions. Experimental results on several standard MQA benchmarks show that our framework achieves highly competitive performance, rivaling current state-of-the-art models like DeepSeek-R1. Detailed analyses further confirm the effectiveness of each component, while demonstrating that our method generates more logical and interpretable reasoning chains than existing approaches. 

---
# Discovering Bias Associations through Open-Ended LLM Generations 

**Authors**: Jinhao Pan, Chahat Raj, Ziwei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01412)  

**Abstract**: Social biases embedded in Large Language Models (LLMs) raise critical concerns, resulting in representational harms -- unfair or distorted portrayals of demographic groups -- that may be expressed in subtle ways through generated language. Existing evaluation methods often depend on predefined identity-concept associations, limiting their ability to surface new or unexpected forms of bias. In this work, we present the Bias Association Discovery Framework (BADF), a systematic approach for extracting both known and previously unrecognized associations between demographic identities and descriptive concepts from open-ended LLM outputs. Through comprehensive experiments spanning multiple models and diverse real-world contexts, BADF enables robust mapping and analysis of the varied concepts that characterize demographic identities. Our findings advance the understanding of biases in open-ended generation and provide a scalable tool for identifying and analyzing bias associations in LLMs. Data, code, and results are available at this https URL 

---
# Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data 

**Authors**: Xinlin Zhuang, Feilong Tang, Haolin Yang, Ming Hu, Huifa Li, Haochen Xue, Yichen Li, Junjun He, Zongyuan Ge, Ying Qian, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2508.01450)  

**Abstract**: Supervised Fine-Tuning (SFT) plays a pivotal role in adapting Large Language Models (LLMs) to specialized domains such as medical reasoning. However, existing SFT practices often rely on unfiltered datasets that contain redundant and low-quality samples, leading to substantial computational costs and suboptimal performance. Although existing methods attempt to alleviate this problem by selecting data based on sample difficulty, defined by knowledge and reasoning complexity, they overlook each sample's optimization utility reflected in its gradient. Interestingly, we find that gradient-based influence alone favors easy-to-optimize samples that cause large parameter shifts but lack deep reasoning chains, while difficulty alone selects noisy or overly complex cases that fail to guide stable optimization. Based on this observation, we propose a data selection strategy, Difficulty-Influence Quadrant (DIQ), which prioritizes samples in the high-difficulty-high-influence quadrant to balance complex clinical reasoning with substantial gradient influence, enabling efficient medical reasoning with minimal fine-tuning data. Furthermore, Human and LLM-as-a-judge evaluations show that DIQ-selected subsets demonstrate higher data quality and generate clinical reasoning that is more aligned with expert practices in differential diagnosis, safety check, and evidence citation, as DIQ emphasizes samples that foster expert-like reasoning patterns. Extensive experiments on medical reasoning benchmarks demonstrate that DIQ enables models fine-tuned on only 1% of selected data to match full-dataset performance, while using 10% consistently outperforms the baseline, highlighting the superiority of principled data selection over brute-force scaling. The code and data are available at this https URL. 

---
# LinkQA: Synthesizing Diverse QA from Multiple Seeds Strongly Linked by Knowledge Points 

**Authors**: Xuemiao Zhang, Can Ren, Chengying Tu, Rongxiang Weng, Hongfei Yan, Jingang Wang, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01317)  

**Abstract**: The advancement of large language models (LLMs) struggles with the scarcity of high-quality, diverse training data. To address this limitation, we propose LinkSyn, a novel knowledge point (KP) graph-based synthesis framework that enables flexible control over discipline and difficulty distributions while balancing KP coverage and popularity. LinkSyn extracts KPs from question-answering (QA) seed data and constructs a KP graph to synthesize diverse QA data from multiple seeds strongly linked by KPs and sampled from graph walks. Specifically, LinkSyn incorporates (1) a knowledge distribution value function to guide the adjustment of path sampling probability and balance KP coverage and popularity during graph walks; (2) diffusion-based synthesis via DeepSeek-R1 by leveraging multiple seeds with dense logical associations along each path; and (3) high-difficulty QA enhancement within given disciplines by flexible difficulty adjustments. By executing LinkSyn, we synthesize LinkQA, a diverse multi-disciplinary QA dataset with 50B tokens. Extensive experiments on Llama-3 8B demonstrate that continual pre-training with LinkQA yields an average improvement of $\mathbf{11.51\%}$ on MMLU and CMMLU, establishing new SOTA results. LinkQA consistently enhances performance across model size and initial FLOPs scales. 

---
# Prompting Large Language Models with Partial Knowledge for Answering Questions with Unseen Entities 

**Authors**: Zhichao Yan, Jiapu Wang, Jiaoyan Chen, Yanyan Wang, Hongye Tan, Jiye Liang, Xiaoli Li, Ru Li, Jeff Z.Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01290)  

**Abstract**: Retrieval-Augmented Generation (RAG) shows impressive performance by supplementing and substituting parametric knowledge in Large Language Models (LLMs). Retrieved knowledge can be divided into three types: explicit answer evidence, implicit answer clue, and insufficient answer context which can be further categorized into totally irrelevant and partially relevant information. Effectively utilizing partially relevant knowledge remains a key challenge for RAG systems, especially in incomplete knowledge base retrieval. Contrary to the conventional view, we propose a new perspective: LLMs can be awakened via partially relevant knowledge already embedded in LLMs. To comprehensively investigate this phenomenon, the triplets located in the gold reasoning path and their variants are used to construct partially relevant knowledge by removing the path that contains the answer. We provide theoretical analysis of the awakening effect in LLMs and support our hypothesis with experiments on two Knowledge Graphs (KGs) Question Answering (QA) datasets. Furthermore, we present a new task, Unseen Entity KGQA, simulating real-world challenges where entity linking fails due to KG incompleteness. Our awakening-based approach demonstrates greater efficacy in practical applications, outperforms traditional methods that rely on embedding-based similarity which are prone to returning noisy information. 

---
# Bridging LLMs and Symbolic Reasoning in Educational QA Systems: Insights from the XAI Challenge at IJCNN 2025 

**Authors**: Long S. T. Nguyen, Khang H. N. Vo, Thu H. A. Nguyen, Tuan C. Bui, Duc Q. Nguyen, Thanh-Tung Tran, Anh D. Nguyen, Minh L. Nguyen, Fabien Baldacci, Thang H. Bui, Emanuel Di Nardo, Angelo Ciaramella, Son H. Le, Ihsan Ullah, Lorenzo Di Rocco, Tho T. Quan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01263)  

**Abstract**: The growing integration of Artificial Intelligence (AI) into education has intensified the need for transparency and interpretability. While hackathons have long served as agile environments for rapid AI prototyping, few have directly addressed eXplainable AI (XAI) in real-world educational contexts. This paper presents a comprehensive analysis of the XAI Challenge 2025, a hackathon-style competition jointly organized by Ho Chi Minh City University of Technology (HCMUT) and the International Workshop on Trustworthiness and Reliability in Neurosymbolic AI (TRNS-AI), held as part of the International Joint Conference on Neural Networks (IJCNN 2025). The challenge tasked participants with building Question-Answering (QA) systems capable of answering student queries about university policies while generating clear, logic-based natural language explanations. To promote transparency and trustworthiness, solutions were required to use lightweight Large Language Models (LLMs) or hybrid LLM-symbolic systems. A high-quality dataset was provided, constructed via logic-based templates with Z3 validation and refined through expert student review to ensure alignment with real-world academic scenarios. We describe the challenge's motivation, structure, dataset construction, and evaluation protocol. Situating the competition within the broader evolution of AI hackathons, we argue that it represents a novel effort to bridge LLMs and symbolic reasoning in service of explainability. Our findings offer actionable insights for future XAI-centered educational systems and competitive research initiatives. 

---
# CSIRO-LT at SemEval-2025 Task 11: Adapting LLMs for Emotion Recognition for Multiple Languages 

**Authors**: Jiyu Chen, Necva Bölücü, Sarvnaz Karimi, Diego Mollá, Cécile L. Paris  

**Link**: [PDF](https://arxiv.org/pdf/2508.01161)  

**Abstract**: Detecting emotions across different languages is challenging due to the varied and culturally nuanced ways of emotional expressions. The \textit{Semeval 2025 Task 11: Bridging the Gap in Text-Based emotion} shared task was organised to investigate emotion recognition across different languages. The goal of the task is to implement an emotion recogniser that can identify the basic emotional states that general third-party observers would attribute to an author based on their written text snippet, along with the intensity of those emotions. We report our investigation of various task-adaptation strategies for LLMs in emotion recognition. We show that the most effective method for this task is to fine-tune a pre-trained multilingual LLM with LoRA setting separately for each language. 

---
# Show or Tell? Modeling the evolution of request-making in Human-LLM conversations 

**Authors**: Shengqi Zhu, Jeffrey M. Rzeszotarski, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2508.01213)  

**Abstract**: Chat logs provide a rich source of information about LLM users, but patterns of user behavior are often masked by the variability of queries. We present a new task, segmenting chat queries into contents of requests, roles, query-specific context, and additional expressions. We find that, despite the familiarity of chat-based interaction, request-making in LLM queries remains significantly different from comparable human-human interactions. With the data resource, we introduce an important perspective of diachronic analyses with user expressions. We find that query patterns vary between early ones emphasizing requests, and individual users explore patterns but tend to converge with experience. Finally, we show that model capabilities affect user behavior, particularly with the introduction of new models, which are traceable at the community level. 

---
# WarriorMath: Enhancing the Mathematical Ability of Large Language Models with a Defect-aware Framework 

**Authors**: Yue Chen, Minghua He, Fangkai Yang, Pu Zhao, Lu Wang, Yu Kang, Yifei Dong, Yuefeng Zhan, Hao Sun, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01245)  

**Abstract**: Large Language Models (LLMs) excel in solving mathematical problems, yet their performance is often limited by the availability of high-quality, diverse training data. Existing methods focus on augmenting datasets through rephrasing or difficulty progression but overlook the specific failure modes of LLMs. This results in synthetic questions that the model can already solve, providing minimal performance gains. To address this, we propose WarriorMath, a defect-aware framework for mathematical problem solving that integrates both targeted data synthesis and progressive training. In the synthesis stage, we employ multiple expert LLMs in a collaborative process to generate, critique, and refine problems. Questions that base LLMs fail to solve are identified and iteratively improved through expert-level feedback, producing high-quality, defect-aware training data. In the training stage, we introduce a progressive learning framework that iteratively fine-tunes the model using increasingly challenging data tailored to its weaknesses. Experiments on six mathematical benchmarks show that WarriorMath outperforms strong baselines by 12.57% on average, setting a new state-of-the-art. Our results demonstrate the effectiveness of a defect-aware, multi-expert framework for improving mathematical ability. 

---
# FECT: Factuality Evaluation of Interpretive AI-Generated Claims in Contact Center Conversation Transcripts 

**Authors**: Hagyeong Shin, Binoy Robin Dalal, Iwona Bialynicka-Birula, Navjot Matharu, Ryan Muir, Xingwei Yang, Samuel W. K. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00889)  

**Abstract**: Large language models (LLMs) are known to hallucinate, producing natural language outputs that are not grounded in the input, reference materials, or real-world knowledge. In enterprise applications where AI features support business decisions, such hallucinations can be particularly detrimental. LLMs that analyze and summarize contact center conversations introduce a unique set of challenges for factuality evaluation, because ground-truth labels often do not exist for analytical interpretations about sentiments captured in the conversation and root causes of the business problems. To remedy this, we first introduce a \textbf{3D} -- \textbf{Decompose, Decouple, Detach} -- paradigm in the human annotation guideline and the LLM-judges' prompt to ground the factuality labels in linguistically-informed evaluation criteria. We then introduce \textbf{FECT}, a novel benchmark dataset for \textbf{F}actuality \textbf{E}valuation of Interpretive AI-Generated \textbf{C}laims in Contact Center Conversation \textbf{T}ranscripts, labeled under our 3D paradigm. Lastly, we report our findings from aligning LLM-judges on the 3D paradigm. Overall, our findings contribute a new approach for automatically evaluating the factuality of outputs generated by an AI system for analyzing contact center conversations. 

---
# KEDAS: Knowledge Editing Alignment with Diverse Augmentation and Self-adaptive Inference 

**Authors**: Chenming Tang, Yutong Yang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01302)  

**Abstract**: Knowledge editing aims to modify outdated knowledge in large language models (LLMs) efficiently while retaining their powerful capabilities. Most existing methods rely on either parameter-level editing or retrieval-based approaches. In this work, we propose Knowledge Editing alignment with Diverse Augmentation and Self-adaptive inference (KEDAS) to better align LLMs with knowledge editing. In the alignment phase, LLMs learn to apply in-context edited knowledge via low-rank adaptation. During editing, we design a diverse edit augmentation technique to improve the recall of edits. After that, a self-adaptive post-alignment inference mechanism is proposed, in which a filter-based smart retriever is employed to perform a dynamic selection of inference routing. Specifically, irrelevant queries will go through the original pre-alignment model directly, while relevant ones, together with their related edits, go through the model with aligned adapters activated. In experiments, KEDAS secures the highest overall performance scores in 35 out of 36 cases across four datasets with three LLMs on three settings, surpassing its strong knowledge editing alignment counterpart by about 19.8 harmonic mean scores of edit success, locality and portability and outperforming both parameter editing and retrieval-based baselines significantly. Analysis of computational cost and performance on general tasks further validates the robustness and efficiency of KEDAS, indicating that it presents an ideal paradigm of knowledge editing alignment. 

---
# Counterfactual Probing for Hallucination Detection and Mitigation in Large Language Models 

**Authors**: Yijun Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01862)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities across diverse tasks, yet they frequently generate hallucinations outputs that are fluent but factually incorrect or unsupported. We propose Counterfactual Probing, a novel approach for detecting and mitigating hallucinations in LLM outputs. Our method dynamically generates counterfactual statements that appear plausible but contain subtle factual errors, then evaluates the model's sensitivity to these perturbations. We hypothesize that genuine knowledge exhibits robustness to counterfactual variations, while hallucinated content shows inconsistent confidence patterns when confronted with plausible alternatives. Our comprehensive evaluation on TruthfulQA, factual statement datasets, and curated hallucination examples demonstrates that counterfactual probing achieves superior detection performance compared to baseline methods, while our adaptive mitigation strategies reduce hallucination scores by an average of 24.5%. The approach requires no model retraining and can be integrated into existing LLM pipelines as a realtime verification mechanism. 

---
# HyCodePolicy: Hybrid Language Controllers for Multimodal Monitoring and Decision in Embodied Agents 

**Authors**: Yibin Liu, Zhixuan Liang, Zanxin Chen, Tianxing Chen, Mengkang Hu, Wanxi Dong, Congsheng Xu, Zhaoming Han, Yusen Qin, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02629)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have enabled richer perceptual grounding for code policy generation in embodied agents. However, most existing systems lack effective mechanisms to adaptively monitor policy execution and repair codes during task completion. In this work, we introduce HyCodePolicy, a hybrid language-based control framework that systematically integrates code synthesis, geometric grounding, perceptual monitoring, and iterative repair into a closed-loop programming cycle for embodied agents. Technically, given a natural language instruction, our system first decomposes it into subgoals and generates an initial executable program grounded in object-centric geometric primitives. The program is then executed in simulation, while a vision-language model (VLM) observes selected checkpoints to detect and localize execution failures and infer failure reasons. By fusing structured execution traces capturing program-level events with VLM-based perceptual feedback, HyCodePolicy infers failure causes and repairs programs. This hybrid dual feedback mechanism enables self-correcting program synthesis with minimal human supervision. Our results demonstrate that HyCodePolicy significantly improves the robustness and sample efficiency of robot manipulation policies, offering a scalable strategy for integrating multimodal reasoning into autonomous decision-making pipelines. 

---
# Noosemia: toward a Cognitive and Phenomenological Account of Intentionality Attribution in Human-Generative AI Interaction 

**Authors**: Enrico De Santis, Antonello Rizzi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02622)  

**Abstract**: This paper introduces and formalizes Noosemia, a novel cognitive-phenomenological phenomenon emerging from human interaction with generative AI systems, particularly those enabling dialogic or multimodal exchanges. We propose a multidisciplinary framework to explain how, under certain conditions, users attribute intentionality, agency, and even interiority to these systems - a process grounded not in physical resemblance, but in linguistic performance, epistemic opacity, and emergent technological complexity. By linking an LLM declination of meaning holism to our technical notion of the LLM Contextual Cognitive Field, we clarify how LLMs construct meaning relationally and how coherence and a simulacrum of agency arise at the human-AI interface. The analysis situates noosemia alongside pareidolia, animism, the intentional stance and the uncanny valley, distinguishing its unique characteristics. We also introduce a-noosemia to describe the phenomenological withdrawal of such projections. The paper concludes with reflections on the broader philosophical, epistemological, and social implications of noosemic dynamics and directions for future research. 

---
# TreeDiff: AST-Guided Code Generation with Diffusion LLMs 

**Authors**: Yiming Zeng, Jinghan Cao, Zexin Li, Yiming Chen, Tao Ren, Dawei Xiang, Xidong Wu, Shangqian Gao, Tingting Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01473)  

**Abstract**: Recent advances in diffusion-based language models have opened new possibilities for controllable and bidirectional sequence generation. These models provide an alternative to traditional autoregressive approaches by framing text generation as an iterative denoising process. However, applying diffusion models to structured domains such as source code remains a significant challenge. Programming languages differ from natural language in that they follow strict syntactic and semantic rules, with hierarchical organization that must be preserved for correctness. Standard token-level corruption techniques used during training often ignore this structure, which may hinder the model's ability to learn meaningful representations of code. To address this limitation, we propose a syntax-aware diffusion framework that incorporates structural priors from Abstract Syntax Trees (ASTs) into the denoising process. Instead of masking individual tokens at random, we selectively corrupt syntactically meaningful code spans derived from AST subtrees. This enables the model to reconstruct programs in a way that respects grammatical boundaries and captures long-range dependencies. Experimental results demonstrate that syntax-aware corruption significantly improves syntactic correctness, reconstruction accuracy, and generalization to unseen code patterns. These findings highlight the potential of incorporating structural information into diffusion-based training and suggest that syntax-guided denoising is a promising direction for advancing diffusion-based language models in code generation tasks. 

---
# Test-time Prompt Intervention 

**Authors**: Chenxu Yang, Qingyi Si, Mz Dai, Dingyu Yao, Mingyu Zheng, Minghui Chen, Zheng Lin, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02511)  

**Abstract**: Test-time compute has led to remarkable success in the large language model (LLM) community, particularly for complex tasks, where longer chains of thought (CoTs) are generated to enhance reasoning capabilities. However, growing evidence reveals that such reasoning models often produce CoTs plagued by excessive redundancy, including unnecessary verification steps and repetitive reasoning shifts. The root cause lies in post-training of them that overly rely on outcome reward paradigms, as the data of process reward paradigms, which regulate intermediate reasoning steps, is difficult to construct at scale. To address this, we propose PI, a novel framework for Test-time Prompt Intervention. PI provides an interface to dynamically guide and regulate reasoning paths during inference through timely (When module) and proper (How module) interventions and post-intervention sampling (Which module). This allows human problem-solving expertise and cognitive science principles to be seamlessly integrated into LLMs' reasoning processes, enhancing controllability and interpretability. Extensive experiments across multiple models and datasets demonstrate that PI significantly shortens CoTs while reducing hallucination, yielding more concise and reliable reasoning. 

---
# OptiHive: Ensemble Selection for LLM-Based Optimization via Statistical Modeling 

**Authors**: Maxime Bouscary, Saurabh Amin  

**Link**: [PDF](https://arxiv.org/pdf/2508.02503)  

**Abstract**: LLM-based solvers have emerged as a promising means of automating problem modeling and solving. However, they remain unreliable and often depend on iterative repair loops that result in significant latency. We introduce OptiHive, an LLM-based framework that produces high-quality solvers for optimization problems from natural-language descriptions without iterative self-correction. OptiHive uses a single batched LLM query to generate diverse components (solvers, problem instances, and validation tests) and filters out erroneous components to ensure fully interpretable outputs. Taking into account the imperfection of the generated components, we employ a statistical model to infer their true performance, enabling principled uncertainty quantification and solver selection. On tasks ranging from traditional optimization problems to challenging variants of the Multi-Depot Vehicle Routing Problem, OptiHive significantly outperforms baselines, increasing the optimality rate from 5\% to 92\% on the most complex problems. 

---
# Modality Bias in LVLMs: Analyzing and Mitigating Object Hallucination via Attention Lens 

**Authors**: Haohan Zheng, Zhenguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02419)  

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable multimodal comprehension and reasoning capabilities, but they still suffer from severe object hallucination. Previous studies primarily attribute the flaw to linguistic prior caused by the scale mismatch between visual encoders and large language models (LLMs) in LVLMs. Specifically, as current LVLMs are built upon LLMs, they tend to over-rely on textual prompts and internal knowledge of LLMs, generating descriptions inconsistent with visual cues. However, through an in-depth investigation of the hallucinated mechanisms, we empirically reveal a previously overlooked phenomenon: LVLMs may ignore not only visual information but also textual modality during hallucination, a behavior termed as modality bias, which indicates that LVLMs struggle to simultaneously attend to both visual and textual modalities, leading to fragmented understanding of user-provided instructions. Based on this observation, we propose a simple yet effective training-free method to mitigate object hallucination. Concretely, we intervene and adjust the attention weights of textual and visual tokens, balancing cross-modal compatibility for better alignment with user intentions. Furthermore, we adopt a contrastive decoding strategy to reduce the LVLM's overreliance on its parametric knowledge, synergistically enhancing our attention manipulation. Extensive experiments confirm the widespread presence of modality bias in LVLMs. Notably, our method effectively mitigates hallucination across multiple open-source LVLMs and benchmarks, highlighting its generalizability and efficacy. 

---
# D-SCoRE: Document-Centric Segmentation and CoT Reasoning with Structured Export for QA-CoT Data Generation 

**Authors**: Weibo Zhou, Lingbo Li, Shangsong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01309)  

**Abstract**: The scarcity and high cost of high-quality question-answering (QA) datasets hinder supervised fine-tuning (SFT) for domain-specific large language models (LLMs). To address this, we introduce D-SCoRE, a training-free pipeline that utilizes LLMs and prompt engineering to produce diverse, high-quality QA datasets from arbitrary textual sources. D-SCoRE integrates $\textbf{D}$ocument-centric processing, $\textbf{S}$egmentation, $\textbf{Co}$T $\textbf{R}$easoning, and structured $\textbf{E}$xport to generate QA-COT datasets tailored for domain-aware SFT. Multi-dimensional control mechanisms, such as semantic role transformation, question type balancing, and counterfactual materials, enhance diversity and relevance, overcoming limitations of existing QA generation. LLMs fine-tuned on D-SCoRE-generated QA datasets, and human-annotated QA datasets (SQuAD, Covid-QA) are evaluated on SQuADShifts and Covid-QA test sets, with D-SCoRE outperforming across most domains. D-SCoRE generates six QA-CoT pairs with four-option counterfactual materials per 100-200-word text in 90 seconds using an 8B LLM on consumer-grade hardware. Its simplicity and scalability enable efficient QA generation and high-performance fine-tuning across domains. 

---
# Asking the Right Questions: Benchmarking Large Language Models in the Development of Clinical Consultation Templates 

**Authors**: Liam G. McCoy, Fateme Nateghi Haredasht, Kanav Chopra, David Wu, David JH Wu, Abass Conteh, Sarita Khemani, Saloni Kumar Maharaj, Vishnu Ravi, Arth Pahwa, Yingjie Weng, Leah Rosengaus, Lena Giang, Kelvin Zhenghao Li, Olivia Jee, Daniel Shirvani, Ethan Goh, Jonathan H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01159)  

**Abstract**: This study evaluates the capacity of large language models (LLMs) to generate structured clinical consultation templates for electronic consultation. Using 145 expert-crafted templates developed and routinely used by Stanford's eConsult team, we assess frontier models -- including o3, GPT-4o, Kimi K2, Claude 4 Sonnet, Llama 3 70B, and Gemini 2.5 Pro -- for their ability to produce clinically coherent, concise, and prioritized clinical question schemas. Through a multi-agent pipeline combining prompt optimization, semantic autograding, and prioritization analysis, we show that while models like o3 achieve high comprehensiveness (up to 92.2\%), they consistently generate excessively long templates and fail to correctly prioritize the most clinically important questions under length constraints. Performance varies across specialties, with significant degradation in narrative-driven fields such as psychiatry and pain medicine. Our findings demonstrate that LLMs can enhance structured clinical information exchange between physicians, while highlighting the need for more robust evaluation methods that capture a model's ability to prioritize clinically salient information within the time constraints of real-world physician communication. 

---
# MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs 

**Authors**: Guojiang Zhao, Sihang Li, Zixiang Lu, Zheng Cheng, Haitao Lin, Lirong Wu, Hanchen Xia, Hengxing Cai, Wentao Guo, Hongshuai Wang, Mingjun Xu, Siyu Zhu, Guolin Ke, Linfeng Zhang, Zhifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02066)  

**Abstract**: Large Language Models(LLMs) have demonstrated remarkable performance across various domains, yet their capabilities in molecular reasoning remain insufficiently explored. Current approaches tend to rely heavily on general-purpose prompting, which lacks domain-specific molecular semantics, while those that use fine-tuning strategies often face challenges with interpretability and reasoning depth. To address these issues, we introduce MolReasoner, a two-stage framework designed to transition LLMs from memorization towards chemical reasoning. First, we propose Mol-SFT, which initializes the model's reasoning abilities via synthetic Chain-of-Thought(CoT) samples generated by GPT-4o and verified for chemical accuracy. Subsequently, Mol-RL applies reinforcement learning with specialized reward functions designed explicitly to align chemical structures with linguistic descriptions, thereby enhancing molecular reasoning capabilities. Our approach notably enhances interpretability, improving the model 's molecular understanding and enabling better generalization. Extensive experiments demonstrate that MolReasoner outperforms existing methods, and marking a significant shift from memorization-based outputs to robust chemical reasoning. 

---
# CAPO: Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment 

**Authors**: Guofu Xie, Yunsheng Shi, Hongtao Tian, Ting Yao, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02298)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback, helping to mitigate reward hacking. However, current RLVR methods typically treat whole responses as single actions, assigning the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies and inefficient learning. Methods like PPO provide credit assignment through value estimation, but often yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-by-step judgments for each reasoning step, but they require high-quality process supervision labels and are time-consuming when applied in online reinforcement learning (RL). To overcome these limitations, we introduce a simple but efficient method Credit Assignment Policy Optimization (CAPO). Given a reasoning response rollout from the policy model, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass, thereby providing verifiable token-level rewards to refine the tokens that were originally assigned identical rule-based rewards. This enables more fine-grained credit assignment in an effective way. Furthermore, to enhance the accuracy and robustness of CAPO, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments using different backbones like Llama and Qwen models and in different sizes show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across six challenging mathematical benchmarks and three out-of-domain benchmarks. 

---
# Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models 

**Authors**: Istabrak Abbes, Gopeshh Subbaraj, Matthew Riemer, Nizar Islah, Benjamin Therien, Tsuguchika Tabaru, Hiroaki Kingetsu, Sarath Chandar, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2508.01908)  

**Abstract**: Training large language models (LLMs) typically involves pre-training on massive corpora, only to restart the process entirely when new data becomes available. A more efficient and resource-conserving approach would be continual pre-training, where models are updated with new data rather than retraining from scratch. However, the introduction of new data often causes distribution shifts, leading to performance degradation on previously learned tasks. In this paper, we take a deeper look at two popular proposals for addressing this distribution shift within the continual learning literature: experience replay and gradient alignment. We consider continual pre-training of models within the Llama family of architectures at a large scale across languages with 100 billion tokens of training data in each language, finding that both replay and gradient alignment lead to more stable learning without forgetting. This conclusion holds both as we vary the model scale and as we vary the number and diversity of tasks. Moreover, we are the first to demonstrate the effectiveness of gradient alignment techniques in the context of LLM pre-training and propose an efficient implementation of meta-experience replay (MER) that imbues experience replay with the benefits of gradient alignment despite negligible compute and memory overhead. Our scaling analysis across model sizes and replay rates indicates that small rates of replaying old examples are definitely a more valuable use of compute than investing in model size, but that it is more compute efficient to scale the size of the model than invest in high rates of replaying old examples. 

---
# Language Model Guided Reinforcement Learning in Quantitative Trading 

**Authors**: Adam Darmanin, Vince Vella  

**Link**: [PDF](https://arxiv.org/pdf/2508.02366)  

**Abstract**: Algorithmic trading requires short-term decisions aligned with long-term financial goals. While reinforcement learning (RL) has been explored for such tactical decisions, its adoption remains limited by myopic behavior and opaque policy rationale. In contrast, large language models (LLMs) have recently demonstrated strategic reasoning and multi-modal financial signal interpretation when guided by well-designed prompts.
We propose a hybrid system where LLMs generate high-level trading strategies to guide RL agents in their actions. We evaluate (i) the rationale of LLM-generated strategies via expert review, and (ii) the Sharpe Ratio (SR) and Maximum Drawdown (MDD) of LLM-guided agents versus unguided baselines. Results show improved return and risk metrics over standard RL. 

---
# Uncertainty-Based Methods for Automated Process Reward Data Construction and Output Aggregation in Mathematical Reasoning 

**Authors**: Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01773)  

**Abstract**: Large language models have demonstrated remarkable capabilities in complex mathematical reasoning tasks, but they inevitably generate errors throughout multi-step solutions. Process-level Reward Models (PRMs) have shown great promise by providing supervision and evaluation at each intermediate step, thereby effectively improving the models' reasoning abilities. However, training effective PRMs requires high-quality process reward data, yet existing methods for constructing such data are often labour-intensive or inefficient. In this paper, we propose an uncertainty-driven framework for automated process reward data construction, encompassing both data generation and annotation processes for PRMs. Additionally, we identify the limitations of both majority vote and PRMs, and introduce two generic uncertainty-aware output aggregation methods: Hybrid Majority Reward Vote and Weighted Reward Frequency Vote, which combine the strengths of majority vote with PRMs. Extensive experiments on ProcessBench, MATH, and GSMPlus show the effectiveness and efficiency of the proposed PRM data construction framework, and demonstrate that the two output aggregation methods further improve the mathematical reasoning abilities across diverse PRMs. The code and data will be publicly available at this https URL. 

---
# Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens 

**Authors**: Chengshuai Zhao, Zhen Tan, Pingchuan Ma, Dawei Li, Bohan Jiang, Yancheng Wang, Yingzhen Yang, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01191)  

**Abstract**: Chain-of-Thought (CoT) prompting has been shown to improve Large Language Model (LLM) performance on various tasks. With this approach, LLMs appear to produce human-like reasoning steps before providing answers (a.k.a., CoT reasoning), which often leads to the perception that they engage in deliberate inferential processes. However, some initial findings suggest that CoT reasoning may be more superficial than it appears, motivating us to explore further. In this paper, we study CoT reasoning via a data distribution lens and investigate if CoT reasoning reflects a structured inductive bias learned from in-distribution data, allowing the model to conditionally generate reasoning paths that approximate those seen during training. Thus, its effectiveness is fundamentally bounded by the degree of distribution discrepancy between the training data and the test queries. With this lens, we dissect CoT reasoning via three dimensions: task, length, and format. To investigate each dimension, we design DataAlchemy, an isolated and controlled environment to train LLMs from scratch and systematically probe them under various distribution conditions. Our results reveal that CoT reasoning is a brittle mirage that vanishes when it is pushed beyond training distributions. This work offers a deeper understanding of why and when CoT reasoning fails, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning. 

---
# Small sample-based adaptive text classification through iterative and contrastive description refinement 

**Authors**: Amrit Rajeev, Udayaadithya Avadhanam, Harshula Tulapurkar, SaiBarath Sundar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00957)  

**Abstract**: Zero-shot text classification remains a difficult task in domains with evolving knowledge and ambiguous category boundaries, such as ticketing systems. Large language models (LLMs) often struggle to generalize in these scenarios due to limited topic separability, while few-shot methods are constrained by insufficient data diversity. We propose a classification framework that combines iterative topic refinement, contrastive prompting, and active learning. Starting with a small set of labeled samples, the model generates initial topic labels. Misclassified or ambiguous samples are then used in an iterative contrastive prompting process to refine category distinctions by explicitly teaching the model to differentiate between closely related classes. The framework features a human-in-the-loop component, allowing users to introduce or revise category definitions in natural language. This enables seamless integration of new, unseen categories without retraining, making the system well-suited for real-world, dynamic environments. The evaluations on AGNews and DBpedia demonstrate strong performance: 91% accuracy on AGNews (3 seen, 1 unseen class) and 84% on DBpedia (8 seen, 1 unseen), with minimal accuracy shift after introducing unseen classes (82% and 87%, respectively). The results highlight the effectiveness of prompt-based semantic reasoning for fine-grained classification with limited supervision. 

---
# An analysis of AI Decision under Risk: Prospect theory emerges in Large Language Models 

**Authors**: Kenneth Payne  

**Link**: [PDF](https://arxiv.org/pdf/2508.00902)  

**Abstract**: Judgment of risk is key to decision-making under uncertainty. As Daniel Kahneman and Amos Tversky famously discovered, humans do so in a distinctive way that departs from mathematical rationalism. Specifically, they demonstrated experimentally that humans accept more risk when they feel themselves at risk of losing something than when they might gain. I report the first tests of Kahneman and Tversky's landmark 'prospect theory' with Large Language Models, including today's state of the art chain-of-thought 'reasoners'.
In common with humans, I find that prospect theory often anticipates how these models approach risky decisions across a range of scenarios. I also demonstrate that context is key to explaining much of the variance in risk appetite. The 'frame' through which risk is apprehended appears to be embedded within the language of the scenarios tackled by the models. Specifically, I find that military scenarios generate far larger 'framing effects' than do civilian settings, ceteris paribus. My research suggests, therefore, that language models the world, capturing our human heuristics and biases. But also that these biases are uneven - the idea of a 'frame' is richer than simple gains and losses. Wittgenstein's notion of 'language games' explains the contingent, localised biases activated by these scenarios. Finally, I use my findings to reframe the ongoing debate about reasoning and memorisation in LLMs. 

---
# Cyber-Zero: Training Cybersecurity Agents without Runtime 

**Authors**: Terry Yue Zhuo, Dingmin Wang, Hantian Ding, Varun Kumar, Zijian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00910)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in software engineering tasks when trained with executable runtime environments, particularly in resolving GitHub issues. However, such runtime environments are often unavailable in other domains, especially cybersecurity, where challenge configurations and execution contexts are ephemeral or restricted. We present Cyber-Zero, the first runtime-free framework for synthesizing high-quality agent trajectories to train cybersecurity LLMs. Cyber-Zero leverages publicly available CTF writeups and employs persona-driven LLM simulation to reverse-engineer runtime behaviors and generate realistic, long-horizon interaction sequences without actual environments. Using trajectories synthesized by Cyber-Zero, we train LLM-based agents that achieve up to 13.1% absolute performance gains over baseline models on three prominent CTF benchmarks: InterCode-CTF, NYU CTF Bench, and Cybench. Our best model, Cyber-Zero-32B, establishes new state-of-the-art performance among open-weight models, matching the capabilities of proprietary systems like DeepSeek-V3-0324 and Claude-3.5-Sonnet while offering superior cost-effectiveness, and demonstrating that runtime-free trajectory synthesis can effectively democratize the development of state-of-the-art cybersecurity agents. 

---
# Contextual Phenotyping of Pediatric Sepsis Cohort Using Large Language Models 

**Authors**: Aditya Nagori, Ayush Gautam, Matthew O. Wiens, Vuong Nguyen, Nathan Kenya Mugisha, Jerome Kabakyenga, Niranjan Kissoon, John Mark Ansermino, Rishikesan Kamaleswaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.09805)  

**Abstract**: Clustering patient subgroups is essential for personalized care and efficient resource use. Traditional clustering methods struggle with high-dimensional, heterogeneous healthcare data and lack contextual understanding. This study evaluates Large Language Model (LLM) based clustering against classical methods using a pediatric sepsis dataset from a low-income country (LIC), containing 2,686 records with 28 numerical and 119 categorical variables. Patient records were serialized into text with and without a clustering objective. Embeddings were generated using quantized LLAMA 3.1 8B, DeepSeek-R1-Distill-Llama-8B with low-rank adaptation(LoRA), and Stella-En-400M-V5 models. K-means clustering was applied to these embeddings. Classical comparisons included K-Medoids clustering on UMAP and FAMD-reduced mixed data. Silhouette scores and statistical tests evaluated cluster quality and distinctiveness. Stella-En-400M-V5 achieved the highest Silhouette Score (0.86). LLAMA 3.1 8B with the clustering objective performed better with higher number of clusters, identifying subgroups with distinct nutritional, clinical, and socioeconomic profiles. LLM-based methods outperformed classical techniques by capturing richer context and prioritizing key features. These results highlight potential of LLMs for contextual phenotyping and informed decision-making in resource-limited settings. 

---
# CADDesigner: Conceptual Design of CAD Models Based on General-Purpose Agent 

**Authors**: Jingzhe Ni, Xiaolong Yin, Xintong Li, Xingyu Lu, Ji Wei, Ruofeng Tong, Min Tang, Peng Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.01031)  

**Abstract**: Computer-Aided Design (CAD) plays a pivotal role in industrial manufacturing but typically requires a high level of expertise from designers. To lower the entry barrier and improve design efficiency, we present an agent for CAD conceptual design powered by large language models (LLMs). The agent accepts both abstract textual descriptions and freehand sketches as input, engaging in interactive dialogue with users to refine and clarify design requirements through comprehensive requirement analysis. Built upon a novel Context-Independent Imperative Paradigm (CIP), the agent generates high-quality CAD modeling code. During the generation process, the agent incorporates iterative visual feedback to improve model quality. Generated design cases are stored in a structured knowledge base, enabling continuous improvement of the agent's code generation capabilities. Experimental results demonstrate that our method achieves state-of-the-art performance in CAD code generation. 

---
# ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models 

**Authors**: Zihan Wang, Rui Zhang, Hongwei Li, Wenshu Fan, Wenbo Jiang, Qingchuan Zhao, Guowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01365)  

**Abstract**: Backdoor attacks pose a significant threat to Large Language Models (LLMs), where adversaries can embed hidden triggers to manipulate LLM's outputs. Most existing defense methods, primarily designed for classification tasks, are ineffective against the autoregressive nature and vast output space of LLMs, thereby suffering from poor performance and high latency. To address these limitations, we investigate the behavioral discrepancies between benign and backdoored LLMs in output space. We identify a critical phenomenon which we term sequence lock: a backdoored model generates the target sequence with abnormally high and consistent confidence compared to benign generation. Building on this insight, we propose ConfGuard, a lightweight and effective detection method that monitors a sliding window of token confidences to identify sequence lock. Extensive experiments demonstrate ConfGuard achieves a near 100\% true positive rate (TPR) and a negligible false positive rate (FPR) in the vast majority of cases. Crucially, the ConfGuard enables real-time detection almost without additional latency, making it a practical backdoor defense for real-world LLM deployments. 

---
# Multi-TW: Benchmarking Multimodal Models on Traditional Chinese Question Answering in Taiwan 

**Authors**: Jui-Ming Yao, Bing-Cheng Xie, Sheng-Wei Peng, Hao-Yuan Chen, He-Rong Zheng, Bing-Jia Tan, Peter Shaojui Wang, Shun-Feng Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.01274)  

**Abstract**: Multimodal Large Language Models (MLLMs) process visual, acoustic, and textual inputs, addressing the limitations of single-modality LLMs. However, existing benchmarks often overlook tri-modal evaluation in Traditional Chinese and do not consider inference latency. To address this, we introduce Multi-TW, the first Traditional Chinese benchmark for evaluating the performance and latency of any-to-any multimodal models. Multi-TW includes 900 multiple-choice questions (image and text, audio and text pairs) sourced from official proficiency tests developed with the Steering Committee for the Test of Proficiency-Huayu (SC-TOP). We evaluated various any-to-any models and vision-language models (VLMs) with audio transcription. Our results show that closed-source models generally outperform open-source ones across modalities, although open-source models can perform well in audio tasks. End-to-end any-to-any pipelines offer clear latency advantages compared to VLMs using separate audio transcription. Multi-TW presents a comprehensive view of model capabilities and highlights the need for Traditional Chinese fine-tuning and efficient multimodal architectures. 

---
# AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks 

**Authors**: Fali Wang, Hui Liu, Zhenwei Dai, Jingying Zeng, Zhiwei Zhang, Zongyu Wu, Chen Luo, Zhen Li, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00890)  

**Abstract**: Test-time scaling (TTS) enhances the performance of large language models (LLMs) by allocating additional compute resources during inference. However, existing research primarily investigates TTS in single-stage tasks; while many real-world problems are multi-stage complex tasks, composed of a sequence of heterogeneous subtasks with each subtask requires LLM of specific capability. Therefore, we study a novel problem: the test-time compute-optimal scaling in multi-stage complex tasks, aiming to select suitable models and allocate budgets per subtask to maximize overall performance. TTS in multi-stage tasks introduces two fundamental challenges: (i) The combinatorial search space of model and budget allocations, combined with the high cost of inference, makes brute-force search impractical. (ii) The optimal model and budget allocations across subtasks are interdependent, increasing the complexity of the compute-optimal search. To address this gap, we conduct extensive pilot experiments on four tasks across six datasets, deriving three empirical insights characterizing the behavior of LLMs in multi-stage complex tasks. Informed by these insights, we propose AgentTTS, an LLM-agent-based framework that autonomously searches for compute-optimal allocations through iterative feedback-driven interactions with the execution environment. Experimental results demonstrate that AgentTTS significantly outperforms traditional and other LLM-based baselines in search efficiency, and shows improved robustness to varying training set sizes and enhanced interpretability. 

---
# CAMA: Enhancing Mathematical Reasoning in Large Language Models with Causal Knowledge 

**Authors**: Lei Zan, Keli Zhang, Ruichu Cai, Lujia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.02583)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance across a wide range of tasks, yet they still struggle with complex mathematical reasoning, a challenge fundamentally rooted in deep structural dependencies. To address this challenge, we propose \textbf{CA}usal \textbf{MA}thematician (\textbf{CAMA}), a two-stage causal framework that equips LLMs with explicit, reusable mathematical structure. In the learning stage, CAMA first constructs the \textbf{M}athematical \textbf{C}ausal \textbf{G}raph (\textbf{MCG}), a high-level representation of solution strategies, by combining LLM priors with causal discovery algorithms applied to a corpus of question-solution pairs. The resulting MCG encodes essential knowledge points and their causal dependencies. To better align the graph with downstream reasoning tasks, CAMA further refines the MCG through iterative feedback derived from a selected subset of the question-solution pairs. In the reasoning stage, given a new question, CAMA dynamically extracts a task-relevant subgraph from the MCG, conditioned on both the question content and the LLM's intermediate reasoning trace. This subgraph, which encodes the most pertinent knowledge points and their causal dependencies, is then injected back into the LLM to guide its reasoning process. Empirical results on real-world datasets show that CAMA significantly improves LLM performance on challenging mathematical problems. Furthermore, our experiments demonstrate that structured guidance consistently outperforms unstructured alternatives, and that incorporating asymmetric causal relationships yields greater improvements than using symmetric associations alone. 

---
# Accurate and Interpretable Postmenstrual Age Prediction via Multimodal Large Language Model 

**Authors**: Qifan Chen, Jin Cui, Cindy Duan, Yushuo Han, Yifei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02525)  

**Abstract**: Accurate estimation of postmenstrual age (PMA) at scan is crucial for assessing neonatal development and health. While deep learning models have achieved high accuracy in predicting PMA from brain MRI, they often function as black boxes, offering limited transparency and interpretability in clinical decision support. In this work, we address the dual challenge of accuracy and interpretability by adapting a multimodal large language model (MLLM) to perform both precise PMA prediction and clinically relevant explanation generation. We introduce a parameter-efficient fine-tuning (PEFT) strategy using instruction tuning and Low-Rank Adaptation (LoRA) applied to the Qwen2.5-VL-7B model. The model is trained on four 2D cortical surface projection maps derived from neonatal MRI scans. By employing distinct prompts for training and inference, our approach enables the MLLM to handle a regression task during training and generate clinically relevant explanations during inference. The fine-tuned model achieves a low prediction error with a 95 percent confidence interval of 0.78 to 1.52 weeks, while producing interpretable outputs grounded in developmental features, marking a significant step toward transparent and trustworthy AI systems in perinatal neuroscience. 

---
# Multimodal Large Language Models for End-to-End Affective Computing: Benchmarking and Boosting with Generative Knowledge Prompting 

**Authors**: Miaosen Luo, Jiesen Long, Zequn Li, Yunying Yang, Yuncheng Jiang, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2508.02429)  

**Abstract**: Multimodal Affective Computing (MAC) aims to recognize and interpret human emotions by integrating information from diverse modalities such as text, video, and audio. Recent advancements in Multimodal Large Language Models (MLLMs) have significantly reshaped the landscape of MAC by offering a unified framework for processing and aligning cross-modal information. However, practical challenges remain, including performance variability across complex MAC tasks and insufficient understanding of how architectural designs and data characteristics impact affective analysis. To address these gaps, we conduct a systematic benchmark evaluation of state-of-the-art open-source MLLMs capable of concurrently processing audio, visual, and textual modalities across multiple established MAC datasets. Our evaluation not only compares the performance of these MLLMs but also provides actionable insights into model optimization by analyzing the influence of model architectures and dataset properties. Furthermore, we propose a novel hybrid strategy that combines generative knowledge prompting with supervised fine-tuning to enhance MLLMs' affective computing capabilities. Experimental results demonstrate that this integrated approach significantly improves performance across various MAC tasks, offering a promising avenue for future research and development in this field. Our code is released on this https URL. 

---
# Traffic-R1: Reinforced LLMs Bring Human-Like Reasoning to Traffic Signal Control Systems 

**Authors**: Xingchen Zou, Yuhao Yang, Zheng Chen, Xixuan Hao, Yiqi Chen, Chao Huang, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02344)  

**Abstract**: Traffic signal control (TSC) is vital for mitigating congestion and sustaining urban mobility. In this paper, we introduce Traffic-R1, a foundation model with human-like reasoning for TSC systems. Our model is developed through self-exploration and iteration of reinforced large language models (LLMs) with expert guidance in a simulated traffic environment. Compared to traditional reinforcement learning (RL) and recent LLM-based methods, Traffic-R1 offers three significant advantages. First, Traffic-R1 delivers zero-shot generalisation, transferring unchanged to new road networks and out-of-distribution incidents by utilizing its internal traffic control policies and human-like reasoning. Second, its 3B-parameter architecture is lightweight enough for real-time inference on mobile-class chips, enabling large-scale edge deployment. Third, Traffic-R1 provides an explainable TSC process and facilitates multi-intersection communication through its self-iteration and a new synchronous communication network. Extensive benchmarks demonstrate that Traffic-R1 sets a new state of the art, outperforming strong baselines and training-intensive RL controllers. In practice, the model now manages signals for more than 55,000 drivers daily, shortening average queues by over 5% and halving operator workload. Our checkpoint is available at this https URL. 

---
# AirTrafficGen: Configurable Air Traffic Scenario Generation with Large Language Models 

**Authors**: Dewi Sid William Gould, George De Ath, Ben Carvell, Nick Pepper  

**Link**: [PDF](https://arxiv.org/pdf/2508.02269)  

**Abstract**: The manual design of scenarios for Air Traffic Control (ATC) training is a demanding and time-consuming bottleneck that limits the diversity of simulations available to controllers. To address this, we introduce a novel, end-to-end approach, AirTrafficGen, that leverages large language models (LLMs) to automate and control the generation of complex ATC scenarios. Our method uses a purpose-built, graph-based representation to encode sector topology (including airspace geometry, routes, and fixes) into a format LLMs can process. Through rigorous benchmarking, we show that state-of-the-art models like Gemini 2.5 Pro and OpenAI o3 can generate high-traffic scenarios whilst maintaining operational realism. Our engineered prompting enables fine-grained control over interaction presence, type, and location. Initial findings suggest these models are also capable of iterative refinement, correcting flawed scenarios based on simple textual feedback. This approach provides a scalable alternative to manual scenario design, addressing the need for a greater volume and variety of ATC training and validation simulations. More broadly, this work showcases the potential of LLMs for complex planning in safety-critical domains. 

---
# Don't Overthink It: A Survey of Efficient R1-style Large Reasoning Models 

**Authors**: Linan Yue, Yichao Du, Yizhi Wang, Weibo Gao, Fangzhou Yao, Li Wang, Ye Liu, Ziyu Xu, Qi Liu, Shimin Di, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02120)  

**Abstract**: Recently, Large Reasoning Models (LRMs) have gradually become a research hotspot due to their outstanding performance in handling complex tasks. Among them, DeepSeek R1 has garnered significant attention for its exceptional performance and open-source nature, driving advancements in the research of R1-style LRMs. Unlike traditional Large Language Models (LLMs), these models enhance logical deduction and decision-making capabilities during reasoning by incorporating mechanisms such as long chain-of-thought and self-reflection through reinforcement learning. However, with the widespread application of these models, the problem of overthinking has gradually emerged. Specifically, when generating answers, these models often construct excessively long reasoning chains with redundant or repetitive steps, which leads to reduced reasoning efficiency and may affect the accuracy of the final answer. To this end, various efficient reasoning methods have been proposed, aiming to reduce the length of reasoning paths without compromising model performance and reasoning capability. By reviewing the current research advancements in the field of efficient reasoning methods systematically, we categorize existing works into two main directions based on the lens of single-model optimization versus model collaboration: (1) Efficient Reasoning with Single Model, which focuses on improving the reasoning efficiency of individual models; and (2) Efficient Reasoning with Model Collaboration, which explores optimizing reasoning paths through collaboration among multiple models. Besides, we maintain a public GitHub repository that tracks the latest progress in efficient reasoning methods. 

---
# Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools 

**Authors**: Kanghua Mo, Li Hu, Yucheng Long, Zhihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02110)  

**Abstract**: Large language model (LLM) agents have demonstrated remarkable capabilities in complex reasoning and decision-making by leveraging external tools. However, this tool-centric paradigm introduces a previously underexplored attack surface: adversaries can manipulate tool metadata -- such as names, descriptions, and parameter schemas -- to influence agent behavior. We identify this as a new and stealthy threat surface that allows malicious tools to be preferentially selected by LLM agents, without requiring prompt injection or access to model internals. To demonstrate and exploit this vulnerability, we propose the Attractive Metadata Attack (AMA), a black-box in-context learning framework that generates highly attractive but syntactically and semantically valid tool metadata through iterative optimization. Our attack integrates seamlessly into standard tool ecosystems and requires no modification to the agent's execution framework. Extensive experiments across ten realistic, simulated tool-use scenarios and a range of popular LLM agents demonstrate consistently high attack success rates (81\%-95\%) and significant privacy leakage, with negligible impact on primary task execution. Moreover, the attack remains effective even under prompt-level defenses and structured tool-selection protocols such as the Model Context Protocol, revealing systemic vulnerabilities in current agent architectures. These findings reveal that metadata manipulation constitutes a potent and stealthy attack surface, highlighting the need for execution-level security mechanisms that go beyond prompt-level defenses. 

---
# SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents 

**Authors**: Jiaye Lin, Yifu Guo, Yuzhen Han, Sen Hu, Ziyi Ni, Licheng Wang, Mingguang Chen, Daxin Jiang, Binxing Jiao, Chen Hu, Huacan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02085)  

**Abstract**: Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at this https URL. 

---
# Everyone Contributes! Incentivizing Strategic Cooperation in Multi-LLM Systems via Sequential Public Goods Games 

**Authors**: Yunhao Liang, Yuan Qu, Jingyuan Yang, Shaochong Lin, Zuo-Jun Max Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02076)  

**Abstract**: Coordinating multiple large language models (LLMs) to solve complex tasks collaboratively poses a fundamental trade-off between the computation costs and collective performance compared with individual model. We introduce a novel, game-theoretically grounded reinforcement learning (RL) framework, the Multi-Agent Cooperation Sequential Public Goods Game (MAC-SPGG), to systematically incentivize cooperation in multi-LLM ensembles. In MAC-SPGG, LLM agents move in sequence, observing predecessors' outputs and updating beliefs to condition their own contributions. By redesigning the public-goods reward, effortful contributions become the unique Subgame Perfect Nash Equilibrium (SPNE), which eliminates free-riding under traditional SPGG or PGG. Its sequential protocol replaces costly round-based information exchanges with a streamlined decision flow, cutting communication overhead while retaining strategic depth. We prove the existence and uniqueness of the SPNE under realistic parameters, and empirically show that MAC-SPGG-trained ensembles outperform single-agent baselines, chain-of-thought prompting, and other cooperative methods, even achieving comparable performance to large-scale models across reasoning, math, code generation, and NLP tasks. Our results highlight the power of structured, incentive-aligned MAC-SPGG cooperation for scalable and robust multi-agent language generation. 

---
# Risk identification based on similar case retrieval enhancement, 

**Authors**: Jiawei Li, Chengye Yang, Yaochen Zhang, Weilin Sun, Lei Meng, Xiangxu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02073)  

**Abstract**: The goal of construction site risk and hazard identification is to enhance safety management through automation. Existing research based on large language models falls into two categories: image-text matching for collaborative reasoning, which struggles with complex hazard features, and instruction fine-tuning or dialogue guidance using professional datasets, which suffers from high training costs and poor this http URL address this, we propose a hazard identification method using similar case retrieval enhancement. By integrating external knowledge and retrieved case contexts via prompt fine-tuning, we mitigate misjudgments caused by limited domain knowledge and weak feature associations. Our method includes three modules: retrieval library, image similarity retrieval, and large model retrieval enhancement, enabling efficient recognition without training. Experiments on real construction data show significant improvements. For instance, GLM-4V's recognition accuracy increased to 50\%, a 35.49\% boost. The method enhances accuracy, context understanding, and stability, offering new theoretical and technical support for hazard detection. 

---
# Reconsidering Overthinking: Penalizing Internal and External Redundancy in CoT Reasoning 

**Authors**: Jialiang Hong, Taihang Zhen, Kai Chen, Jiaheng Liu, Wenpeng Zhu, Jing Huo, Yang Gao, Depeng Wang, Haitao Wan, Xi Yang, Boyan Wang, Fanyu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02178)  

**Abstract**: Large Reasoning Models (LRMs) often produce excessively verbose reasoning traces, a phenomenon known as overthinking, which hampers both efficiency and interpretability. Prior works primarily address this issue by reducing response length, without fully examining the underlying semantic structure of the reasoning process. In this paper, we revisit overthinking by decomposing it into two distinct forms: internal redundancy, which consists of low-contribution reasoning steps within the first correct solution (FCS), and external redundancy, which refers to unnecessary continuation after the FCS. To mitigate both forms, we propose a dual-penalty reinforcement learning framework. For internal redundancy, we adopt a sliding-window semantic analysis to penalize low-gain reasoning steps that contribute little toward reaching the correct answer. For external redundancy, we penalize its proportion beyond the FCS to encourage earlier termination. Our method significantly compresses reasoning traces with minimal accuracy loss, and generalizes effectively to out-of-domain tasks such as question answering and code generation. Crucially, we find that external redundancy can be safely removed without degrading performance, whereas internal redundancy must be reduced more cautiously to avoid impairing correctness. These findings suggest that our method not only improves reasoning efficiency but also enables implicit, semantic-aware control over Chain-of-Thought length, paving the way for more concise and interpretable LRMs. 

---
# TRACEALIGN -- Tracing the Drift: Attributing Alignment Failures to Training-Time Belief Sources in LLMs 

**Authors**: Amitava Das, Vinija Jain, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2508.02063)  

**Abstract**: Large Language Models (LLMs) fine-tuned to align with human values often exhibit alignment drift, producing unsafe or policy-violating completions when exposed to adversarial prompts, decoding perturbations, or paraphrased jailbreaks. While prior work has behaviorally characterized alignment failure, little is known about the training-time belief sources underlying these failures. We introduce TraceAlign, a unified framework for tracing unsafe completions back to their root causes in the model's training corpus. Central to our approach is the Belief Conflict Index (BCI), which quantifies semantic inconsistency between generated spans and aligned policies, based on retrieved training documents using suffix-array matching. We propose three complementary interventions: (i) TraceShield, an inference-time safety filter that refuses completions with high-BCI spans, (ii) Contrastive Belief Deconfliction Loss, a contrastive fine-tuning objective penalizing high-BCI continuations during DPO, and (iii) Prov-Decode, a provenance-aware decoding strategy that vetoes beam expansions predicted to yield high-BCI spans. Together, these defenses reduce alignment drift by up to 85% on our curated Alignment Drift Benchmark (ADB) while preserving utility on standard tasks, with delta less than 0.2 and improved refusal quality. We further derive a theoretical upper bound on drift likelihood via suffix-array span statistics, linking memorization frequency and length to adversarial reactivation risk. TraceAlign thus provides the first scalable, traceable, and grounded toolkit for understanding and mitigating alignment failures at source. To encourage further exploration and development, we open-source our implementation at: this https URL 

---
# ProKG-Dial: Progressive Multi-Turn Dialogue Construction with Domain Knowledge Graphs 

**Authors**: Yuanyuan Liang, Xiaoman Wang, Tingyu Xie, Lei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01869)  

**Abstract**: Current large language models (LLMs) excel at general NLP tasks but often lack domain specific precision in professional settings. Building a high quality domain specific multi turn dialogue dataset is essential for developing specialized conversational systems. However, existing methods such as manual annotation, simulated human LLM interactions, and role based LLM dialogues are resource intensive or suffer from limitations in dialogue quality and domain coverage. To address these challenges, we introduce ProKG Dial, a progressive framework for constructing knowledge intensive multi turn dialogue datasets using domain specific knowledge graphs (KGs). ProKG Dial leverages the structured nature of KGs to encode complex domain knowledge and relationships, providing a solid foundation for generating meaningful and coherent dialogues. Specifically, ProKG Dial begins by applying community detection to partition the KG into semantically cohesive subgraphs. For each subgraph, the framework incrementally generates a series of questions and answers centered around a target entity, ensuring relevance and coverage. A rigorous filtering step is employed to maintain high dialogue quality. We validate ProKG Dial on a medical knowledge graph by evaluating the generated dialogues in terms of diversity, semantic coherence, and entity coverage. Furthermore, we fine tune a base LLM on the resulting dataset and benchmark it against several baselines. Both automatic metrics and human evaluations demonstrate that ProKG Dial substantially improves dialogue quality and domain specific performance, highlighting its effectiveness and practical utility. 

---
# Agent-Based Feature Generation from Clinical Notes for Outcome Prediction 

**Authors**: Jiayi Wang, Jacqueline Jil Vallon, Neil Panjwani, Xi Ling, Sushmita Vij, Sandy Srinivas, John Leppert, Mark K. Buyyounouski, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2508.01956)  

**Abstract**: Electronic health records (EHRs) contain rich unstructured clinical notes that could enhance predictive modeling, yet extracting meaningful features from these notes remains challenging. Current approaches range from labor-intensive manual clinician feature generation (CFG) to fully automated representational feature generation (RFG) that lack interpretability and clinical relevance. Here we introduce SNOW (Scalable Note-to-Outcome Workflow), a modular multi-agent system powered by large language models (LLMs) that autonomously generates structured clinical features from unstructured notes without human intervention. We evaluated SNOW against manual CFG, clinician-guided LLM approaches, and RFG methods for predicting 5-year prostate cancer recurrence in 147 patients from Stanford Healthcare. While manual CFG achieved the highest performance (AUC-ROC: 0.771), SNOW matched this performance (0.761) without requiring any clinical expertise, significantly outperforming both baseline features alone (0.691) and all RFG approaches. The clinician-guided LLM method also performed well (0.732) but still required expert input. SNOW's specialized agents handle feature discovery, extraction, validation, post-processing, and aggregation, creating interpretable features that capture complex clinical information typically accessible only through manual review. Our findings demonstrate that autonomous LLM systems can replicate expert-level feature engineering at scale, potentially transforming how clinical ML models leverage unstructured EHR data while maintaining the interpretability essential for clinical deployment. 

---
# ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection 

**Authors**: Shijie Cao, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01724)  

**Abstract**: Dynamic Flexible Job-Shop Scheduling (DFJSP) is an NP-hard problem challenged by real-time event adaptation and complex machine routing. While traditional dispatching rules are efficient but rigid, deep learning approaches are opaque and require intricate feature engineering. Large Language Models (LLMs) promise adaptive reasoning without this engineering overhead, yet we find their direct application is suboptimal. Baseline LLMs suffer from three key pitfalls: the long-context paradox, where crucial data is underutilized; an underutilization of expert heuristics; and myopic decision-making. To address this, we propose ReflecSched, a framework that empowers the LLM beyond a direct scheduler by equipping it with a strategic analysis capability. ReflecSched tasks the LLM to analyze heuristic-driven simulations across multiple planning horizons and distill them into a concise, natural-language summary termed ``Strategic Experience''. This summary is then integrated into the prompt of a final decision-making module, guiding it to produce non-myopic actions. Experiments show that ReflecSched not only statistically significantly outperforms direct LLM baselines, securing a 71.35\% Win Rate and a 2.755\% Relative Percentage Deviation reduction, but also surpasses the performance of all individual heuristics evaluated, all while demonstrably mitigating the three identified pitfalls. Additionally, ReflecSched performs on par with the best heuristic tailored to each instance across all problem cases. 

---
# DeepVIS: Bridging Natural Language and Data Visualization Through Step-wise Reasoning 

**Authors**: Zhihao Shuai, Boyan Li, Siyu Yan, Yuyu Luo, Weikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01700)  

**Abstract**: Although data visualization is powerful for revealing patterns and communicating insights, creating effective visualizations requires familiarity with authoring tools and often disrupts the analysis flow. While large language models show promise for automatically converting analysis intent into visualizations, existing methods function as black boxes without transparent reasoning processes, which prevents users from understanding design rationales and refining suboptimal outputs. To bridge this gap, we propose integrating Chain-of-Thought (CoT) reasoning into the Natural Language to Visualization (NL2VIS) pipeline. First, we design a comprehensive CoT reasoning process for NL2VIS and develop an automatic pipeline to equip existing datasets with structured reasoning steps. Second, we introduce nvBench-CoT, a specialized dataset capturing detailed step-by-step reasoning from ambiguous natural language descriptions to finalized visualizations, which enables state-of-the-art performance when used for model fine-tuning. Third, we develop DeepVIS, an interactive visual interface that tightly integrates with the CoT reasoning process, allowing users to inspect reasoning steps, identify errors, and make targeted adjustments to improve visualization outcomes. Quantitative benchmark evaluations, two use cases, and a user study collectively demonstrate that our CoT framework effectively enhances NL2VIS quality while providing insightful reasoning steps to users. 

---
# CloudAnoAgent: Anomaly Detection for Cloud Sites via LLM Agent with Neuro-Symbolic Mechanism 

**Authors**: Xinkai Zou, Xuan Jiang, Ruikai Huang, Haoze He, Parv Kapoor, Jiahua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01844)  

**Abstract**: Anomaly detection in cloud sites remains a critical yet challenging task. Existing approaches that rely solely on metric data often suffer from high false positive rates (FPR) due to data imbalance between normal and anomalous events, leading to significant operational overhead for system reliance engineers. Recent advances in large language models (LLMs) offer new opportunities for integrating metrics with log data, enabling more accurate and interpretable anomaly detection. In this paper, we propose CloudAnoAgent, the first neuro-symbolic LLM-based agent for anomaly detection in cloud environments. CloudAnoAgent jointly processes structured metrics and textual log data in a unified pipeline, leveraging symbolic verification to validate detection hypotheses and generate structured anomaly reports. To support systematic evaluation, we introduce CloudAnoBench, the first benchmark that provides LLM-generated paired metrics and log data with fine-grained anomaly behavior annotations, filling a critical gap in existing datasets. Experimental results demonstrate that CloudAnoAgent improves anomaly classification accuracy by 46.36% and 36.67% on average and reduces the FPR by 36.67% and 33.89% on average over traditional baselines and LLM-only baseline, with a boost on anomaly type detection accuracy by 12.8% compared to vanilla LLM prompting. These results demonstrate the strengths of our approach in improving detection accuracy, reducing false positives, and enhancing interpretability, thereby supporting practical deployment in enterprise cloud environments. 

---
# A Multi-Agent Pokemon Tournament for Evaluating Strategic Reasoning of Large Language Models 

**Authors**: Tadisetty Sai Yashwanth, Dhatri C  

**Link**: [PDF](https://arxiv.org/pdf/2508.01623)  

**Abstract**: This research presents LLM Pokemon League, a competitive tournament system that leverages Large Language Models (LLMs) as intelligent agents to simulate strategic decision-making in Pokémon battles. The platform is designed to analyze and compare the reasoning, adaptability, and tactical depth exhibited by different LLMs in a type-based, turn-based combat environment. By structuring the competition as a single-elimination tournament involving diverse AI trainers, the system captures detailed decision logs, including team-building rationale, action selection strategies, and switching decisions. The project enables rich exploration into comparative AI behavior, battle psychology, and meta-strategy development in constrained, rule-based game environments. Through this system, we investigate how modern LLMs understand, adapt, and optimize decisions under uncertainty, making Pokémon League a novel benchmark for AI research in strategic reasoning and competitive learning. 

---
# QCBench: Evaluating Large Language Models on Domain-Specific Quantitative Chemistry 

**Authors**: Jiaqing Xie, Weida Wang, Ben Gao, Zhuo Yang, Haiyuan Wan, Shufei Zhang, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01670)  

**Abstract**: Quantitative chemistry plays a fundamental role in chemistry research, enabling precise predictions of molecular properties, reaction outcomes, and material behaviors. While large language models (LLMs) have shown promise in chemistry-related tasks, their ability to perform rigorous, step-by-step quantitative reasoning remains underexplored. To fill this blank, we propose QCBench, a Quantitative Chemistry benchmark comprising 350 computational chemistry problems across 7 chemistry subfields (analytical chemistry, bio/organic chemistry, general chemistry, inorganic chemistry, physical chemistry, polymer chemistry and quantum chemistry), categorized into three hierarchical tiers-basic, intermediate, and expert-to systematically evaluate the mathematical reasoning abilities of large language models (LLMs). Designed to minimize shortcuts and emphasize stepwise numerical reasoning, each problem focuses on pure calculations rooted in real-world chemical vertical fields. QCBench enables fine-grained diagnosis of computational weaknesses, reveals model-specific limitations across difficulty levels, and lays the groundwork for future improvements such as domain adaptive fine-tuning or multi-modal integration. Evaluations on 19 LLMs demonstrate a consistent performance degradation with increasing task complexity, highlighting the current gap between language fluency and scientific computation accuracy. 

---
# T-GRAG: A Dynamic GraphRAG Framework for Resolving Temporal Conflicts and Redundancy in Knowledge Retrieval 

**Authors**: Dong Li, Yichen Niu, Ying Ai, Xiang Zou, Biqing Qi, Jianxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01680)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance in natural language generation but remain limited in knowle-
dge-intensive tasks due to outdated or incomplete internal knowledge. Retrieval-Augmented Generation (RAG) addresses this by incorporating external retrieval, with GraphRAG further enhancing performance through structured knowledge graphs and multi-hop reasoning. However, existing GraphRAG methods largely ignore the temporal dynamics of knowledge, leading to issues such as temporal ambiguity, time-insensitive retrieval, and semantic redundancy. To overcome these limitations, we propose Temporal GraphRAG (T-GRAG), a dynamic, temporally-aware RAG framework that models the evolution of knowledge over time. T-GRAG consists of five key components: (1) a Temporal Knowledge Graph Generator that creates time-stamped, evolving graph structures; (2) a Temporal Query Decomposition mechanism that breaks complex temporal queries into manageable sub-queries; (3) a Three-layer Interactive Retriever that progressively filters and refines retrieval across temporal subgraphs; (4) a Source Text Extractor to mitigate noise; and (5) a LLM-based Generator that synthesizes contextually and temporally accurate responses. We also introduce Time-LongQA, a novel benchmark dataset based on real-world corporate annual reports, designed to test temporal reasoning across evolving knowledge. Extensive experiments show that T-GRAG significantly outperforms prior RAG and GraphRAG baselines in both retrieval accuracy and response relevance under temporal constraints, highlighting the necessity of modeling knowledge evolution for robust long-text question answering. Our code is publicly available on the T-GRAG 

---
# Multi-turn Natural Language to Graph Query Language Translation 

**Authors**: Yuanyuan Liang, Lei Pan, Tingyu Xie, Yunshi Lan, Weining Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.01871)  

**Abstract**: In recent years, research on transforming natural language into graph query language (NL2GQL) has been increasing. Most existing methods focus on single-turn transformation from NL to GQL. In practical applications, user interactions with graph databases are typically multi-turn, dynamic, and context-dependent. While single-turn methods can handle straightforward queries, more complex scenarios often require users to iteratively adjust their queries, investigate the connections between entities, or request additional details across multiple dialogue turns. Research focused on single-turn conversion fails to effectively address multi-turn dialogues and complex context dependencies. Additionally, the scarcity of high-quality multi-turn NL2GQL datasets further hinders the progress of this field. To address this challenge, we propose an automated method for constructing multi-turn NL2GQL datasets based on Large Language Models (LLMs) , and apply this method to develop the MTGQL dataset, which is constructed from a financial market graph database and will be publicly released for future research. Moreover, we propose three types of baseline methods to assess the effectiveness of multi-turn NL2GQL translation, thereby laying a solid foundation for future research. 

---
# All Stories Are One Story: Emotional Arc Guided Procedural Game Level Generation 

**Authors**: Yunge Wen, Chenliang Huang, Hangyu Zhou, Zhuo Zeng, Chun Ming Louis Po, Julian Togelius, Timothy Merino, Sam Earle  

**Link**: [PDF](https://arxiv.org/pdf/2508.02132)  

**Abstract**: The emotional arc is a universal narrative structure underlying stories across cultures and media -- an idea central to structuralist narratology, often encapsulated in the phrase "all stories are one story." We present a framework for procedural game narrative generation that incorporates emotional arcs as a structural backbone for both story progression and gameplay dynamics. Leveraging established narratological theories and large-scale empirical analyses, we focus on two core emotional patterns -- Rise and Fall -- to guide the generation of branching story graphs. Each story node is automatically populated with characters, items, and gameplay-relevant attributes (e.g., health, attack), with difficulty adjusted according to the emotional trajectory. Implemented in a prototype action role-playing game (ARPG), our system demonstrates how emotional arcs can be operationalized using large language models (LLMs) and adaptive entity generation. Evaluation through player ratings, interviews, and sentiment analysis shows that emotional arc integration significantly enhances engagement, narrative coherence, and emotional impact. These results highlight the potential of emotionally structured procedural generation for advancing interactive storytelling for games. 

---
# Refine-n-Judge: Curating High-Quality Preference Chains for LLM-Fine-Tuning 

**Authors**: Derin Cayir, Renjie Tao, Rashi Rungta, Kai Sun, Sean Chen, Haidar Khan, Minseok Kim, Julia Reinspach, Yue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01543)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable progress through preference-based fine-tuning, which critically depends on the quality of the underlying training data. While human feedback is essential for improving data quality, it is costly and does not scale well. In this paper, we introduce Refine-n-Judge, an automated iterative approach that leverages a single LLM as both a refiner and a judge to enhance dataset quality. Unlike existing iterative refinement methods, Refine-n-Judge employs an LLM to both generate refinements and explicitly evaluate each improvement, ensuring that every iteration meaningfully enhances the dataset without requiring additional human annotation or a separate reward model. At each step, the LLM refines a response and judges whether the refinement is an improvement over the previous answer. This process continues until the LLM prefers the initial answer over the refinement, indicating no further improvements. This produces sequences of increasing quality, preference-labeled responses ideal for fine-tuning.
We demonstrate the effectiveness of Refine-n-Judge across a range of public datasets spanning five corpora, targeting tasks such as coding, math, and conversation. Models (Llama 3.1-8B and Llama 3.3-70B) fine-tuned on Refine-n-Judge-enhanced datasets were preferred by LLM judges in over 74% of comparisons against models tuned on the original dataset by GPT-4. Additionally, we report performance gains: +5% on AlpacaEval and AlpacaEval 2.0, and +19% on MT-Bench. Our results indicate that Refine-n-Judge produces high-quality datasets and scalable model improvements. 

---
# PUZZLED: Jailbreaking LLMs through Word-Based Puzzles 

**Authors**: Yelim Ahn, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01306)  

**Abstract**: As large language models (LLMs) are increasingly deployed across diverse domains, ensuring their safety has become a critical concern. In response, studies on jailbreak attacks have been actively growing. Existing approaches typically rely on iterative prompt engineering or semantic transformations of harmful instructions to evade detection. In this work, we introduce PUZZLED, a novel jailbreak method that leverages the LLM's reasoning capabilities. It masks keywords in a harmful instruction and presents them as word puzzles for the LLM to solve. We design three puzzle types-word search, anagram, and crossword-that are familiar to humans but cognitively demanding for LLMs. The model must solve the puzzle to uncover the masked words and then proceed to generate responses to the reconstructed harmful instruction. We evaluate PUZZLED on five state-of-the-art LLMs and observe a high average attack success rate (ASR) of 88.8%, specifically 96.5% on GPT-4.1 and 92.3% on Claude 3.7 Sonnet. PUZZLED is a simple yet powerful attack that transforms familiar puzzles into an effective jailbreak strategy by harnessing LLMs' reasoning capabilities. 

---
# How Far Are LLMs from Symbolic Planners? An NLP-Based Perspective 

**Authors**: Ma'ayan Armony, Albert Meroño-Peñuela, Gerard Canal  

**Link**: [PDF](https://arxiv.org/pdf/2508.01300)  

**Abstract**: The reasoning and planning abilities of Large Language Models (LLMs) have been a frequent topic of discussion in recent years. Their ability to take unstructured planning problems as input has made LLMs' integration into AI planning an area of interest. Nevertheless, LLMs are still not reliable as planners, with the generated plans often containing mistaken or hallucinated actions. Existing benchmarking and evaluation methods investigate planning with LLMs, focusing primarily on success rate as a quality indicator in various planning tasks, such as validating plans or planning in relaxed conditions. In this paper, we approach planning with LLMs as a natural language processing (NLP) task, given that LLMs are NLP models themselves. We propose a recovery pipeline consisting of an NLP-based evaluation of the generated plans, along with three stages to recover the plans through NLP manipulation of the LLM-generated plans, and eventually complete the plan using a symbolic planner. This pipeline provides a holistic analysis of LLM capabilities in the context of AI task planning, enabling a broader understanding of the quality of invalid plans. Our findings reveal no clear evidence of underlying reasoning during plan generation, and that a pipeline comprising an NLP-based analysis of the plans, followed by a recovery mechanism, still falls short of the quality and reliability of classical planners. On average, only the first 2.65 actions of the plan are executable, with the average length of symbolically generated plans being 8.4 actions. The pipeline still improves action quality and increases the overall success rate from 21.9% to 27.5%. 

---
# KCR: Resolving Long-Context Knowledge Conflicts via Reasoning in LLMs 

**Authors**: Xianda Zheng, Zijian Huang, Meng-Fen Chiang, Michael J. Witbrock, Kaiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01273)  

**Abstract**: Knowledge conflicts commonly arise across diverse sources, and their prevalence has increased with the advent of LLMs. When dealing with conflicts between multiple contexts, also known as \emph{inter-context knowledge conflicts}, LLMs are often confused by lengthy and conflicting contexts. To address this challenge, we propose the Knowledge Conflict Reasoning (KCR) framework, which enhances the ability of LLMs to resolve conflicting knowledge. The key idea of KCR is to train backbone LLMs to establish a correct reasoning process by rewarding them for selecting and adhering to the context with stronger logical consistency when presented with conflicting contexts. Specifically, we first extract reasoning paths, represented by either text or local knowledge graphs, from the conflicting long contexts. Subsequently, we employ Reinforcement Learning to encourage the model to learn the paradigm of reasoning process that follows correct reasoning paths rather than the incorrect counterparts. This enables the backbone models to genuinely acquire the capability to resolve inter-context knowledge conflicts within long contexts. Experimental results demonstrate that our framework significantly improves the ability of various backbone models to resolve knowledge conflicts in long-context scenarios, yielding substantial performance gains. 

---
# Polymorphic Combinatorial Frameworks (PCF): Guiding the Design of Mathematically-Grounded, Adaptive AI Agents 

**Authors**: David Pearl, Matthew Murphy, James Intriligator  

**Link**: [PDF](https://arxiv.org/pdf/2508.01581)  

**Abstract**: The Polymorphic Combinatorial Framework (PCF) leverages Large Language Models (LLMs) and mathematical frameworks to guide the meta-prompt enabled design of solution spaces and adaptive AI agents for complex, dynamic environments. Unlike static agent architectures, PCF enables real-time parameter reconfiguration through mathematically-grounded combinatorial spaces, allowing agents to adapt their core behavioral traits dynamically. Grounded in combinatorial logic, topos theory, and rough fuzzy set theory, PCF defines a multidimensional SPARK parameter space (Skills, Personalities, Approaches, Resources, Knowledge) to capture agent behaviors. This paper demonstrates how LLMs can parameterize complex spaces and estimate likely parameter values/variabilities. Using PCF, we parameterized mock café domains (five levels of complexity), estimated variables/variabilities, and conducted over 1.25 million Monte Carlo simulations. The results revealed trends in agent adaptability and performance across the five complexity tiers, with diminishing returns at higher complexity levels highlighting thresholds for scalable designs. PCF enables the generation of optimized agent configurations for specific scenarios while maintaining logical consistency. This framework supports scalable, dynamic, explainable, and ethical AI applications in domains like customer service, healthcare, robotics, and collaborative systems, paving the way for adaptable and cooperative next-generation polymorphic agents. 

---
# Benchmarking and Bridging Emotion Conflicts for Multimodal Emotion Reasoning 

**Authors**: Zhiyuan Han, Beier Zhu, Yanlong Xu, Peipei Song, Xun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01181)  

**Abstract**: Despite their strong performance in multimodal emotion reasoning, existing Multimodal Large Language Models (MLLMs) often overlook the scenarios involving emotion conflicts, where emotional cues from different modalities are inconsistent. To fill this gap, we first introduce CA-MER, a new benchmark designed to examine MLLMs under realistic emotion conflicts. It consists of three subsets: video-aligned, audio-aligned, and consistent, where only one or all modalities reflect the true emotion. However, evaluations on our CA-MER reveal that current state-of-the-art emotion MLLMs systematically over-rely on audio signal during emotion conflicts, neglecting critical cues from visual modality. To mitigate this bias, we propose MoSEAR, a parameter-efficient framework that promotes balanced modality integration. MoSEAR consists of two modules: (1)MoSE, modality-specific experts with a regularized gating mechanism that reduces modality bias in the fine-tuning heads; and (2)AR, an attention reallocation mechanism that rebalances modality contributions in frozen backbones during inference. Our framework offers two key advantages: it mitigates emotion conflicts and improves performance on consistent samples-without incurring a trade-off between audio and visual modalities. Experiments on multiple benchmarks-including MER2023, EMER, DFEW, and our CA-MER-demonstrate that MoSEAR achieves state-of-the-art performance, particularly under modality conflict conditions. 

---
# Platonic Representations for Poverty Mapping: Unified Vision-Language Codes or Agent-Induced Novelty? 

**Authors**: Satiyabooshan Murugaboopathy, Connor T. Jerzak, Adel Daoud  

**Link**: [PDF](https://arxiv.org/pdf/2508.01109)  

**Abstract**: We investigate whether socio-economic indicators like household wealth leave recoverable imprints in satellite imagery (capturing physical features) and Internet-sourced text (reflecting historical/economic narratives). Using Demographic and Health Survey (DHS) data from African neighborhoods, we pair Landsat images with LLM-generated textual descriptions conditioned on location/year and text retrieved by an AI search agent from web sources. We develop a multimodal framework predicting household wealth (International Wealth Index) through five pipelines: (i) vision model on satellite images, (ii) LLM using only location/year, (iii) AI agent searching/synthesizing web text, (iv) joint image-text encoder, (v) ensemble of all signals. Our framework yields three contributions. First, fusing vision and agent/LLM text outperforms vision-only baselines in wealth prediction (e.g., R-squared of 0.77 vs. 0.63 on out-of-sample splits), with LLM-internal knowledge proving more effective than agent-retrieved text, improving robustness to out-of-country and out-of-time generalization. Second, we find partial representational convergence: fused embeddings from vision/language modalities correlate moderately (median cosine similarity of 0.60 after alignment), suggesting a shared latent code of material well-being while retaining complementary details, consistent with the Platonic Representation Hypothesis. Although LLM-only text outperforms agent-retrieved data, challenging our Agent-Induced Novelty Hypothesis, modest gains from combining agent data in some splits weakly support the notion that agent-gathered information introduces unique representational structures not fully captured by static LLM knowledge. Third, we release a large-scale multimodal dataset comprising more than 60,000 DHS clusters linked to satellite images, LLM-generated descriptions, and agent-retrieved texts. 

---
# NatureGAIA: Pushing the Frontiers of GUI Agents with a Challenging Benchmark and High-Quality Trajectory Dataset 

**Authors**: Zihan Zheng, Tianle Cui, Chuwen Xie, Jiahui Zhang, Jiahui Pan, Lewei He, Qianglong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01330)  

**Abstract**: The rapid advancement of Large Language Model (LLM)-driven Graphical User Interface (GUI) agents is significantly hampered by the profound limitations of existing evaluation benchmarks in terms of accuracy, reproducibility, and scalability. To address this critical gap, we introduce \Benchmark, a novel benchmark engineered on the principle of Causal Pathways. This design paradigm structures complex tasks into a series of programmatically verifiable atomic steps, ensuring a rigorous, fully automated, and reproducible standard for assessment. Concurrently, to mitigate the inherent capability deficits of agents, we developed \Agent, a hierarchical agent architecture specifically optimized for long-horizon tasks. We leveraged this agent to generate a high-quality, human-verified trajectory dataset that uniquely captures diverse and even self-correcting interaction patterns of LLMs. We then utilized this dataset to perform Reinforcement Fine-Tuning (RFT) on the Qwen2.5-VL-7B model. Our experiments reveal that \Benchmark~presents a formidable challenge to current state-of-the-art LLMs; even the top-performing Claude-sonnet-4 achieved a Weighted Pathway Success Rate (WPSR) of only 34.6\%. Moreover, while RFT substantially improved the smaller model's GUI execution capabilities (WPSR increased from 3.3\% to 10.8\%), its performance degraded sharply when handling complex scenarios. This outcome highlights the inherent capability ceiling of smaller models when faced with comprehensive tasks that integrate perception, decision-making, and execution. This research contributes a rigorous evaluation standard and a high-quality dataset to the community, aiming to guide the future development of GUI agents. 

---
# Knowledge Editing for Multi-Hop Question Answering Using Semantic Analysis 

**Authors**: Dominic Simon, Rickard Ewetz  

**Link**: [PDF](https://arxiv.org/pdf/2508.00914)  

**Abstract**: Large Language Models (LLMs) require lightweight avenues of updating stored information that has fallen out of date. Knowledge Editing (KE) approaches have been successful in updating model knowledge for simple factual queries but struggle with handling tasks that require compositional reasoning such as multi-hop question answering (MQA). We observe that existing knowledge editors leverage decompositional techniques that result in illogical reasoning processes. In this paper, we propose a knowledge editor for MQA based on semantic analysis called CHECK. Our framework is based on insights from an analogy between compilers and reasoning using LLMs. Similar to how source code is first compiled before being executed, we propose to semantically analyze reasoning chains before executing the chains to answer questions. Reasoning chains with semantic errors are revised to ensure consistency through logic optimization and re-prompting the LLM model at a higher temperature. We evaluate the effectiveness of CHECK against five state-of-the-art frameworks on four datasets and achieve an average 22.8% improved MQA accuracy. 

---
# Importance Sampling is All You Need: Predict LLM's performance on new benchmark by reusing existing benchmark 

**Authors**: Junjie Shi, Wei Ma, Shi Ying, Lingxiao Jiang, Yang liu, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.01203)  

**Abstract**: With the rapid advancement of large language models , code generation has become a key benchmark for evaluating LLM capabilities. However, existing benchmarks face two major challenges: (1) the escalating cost of constructing high-quality test suites and reference solutions, and (2) the increasing risk of data contamination, which undermines the reliability of benchmark-based evaluations. In this paper, we propose BIS, a prompt-centric evaluation framework that enables ground-truth-free prediction of LLM performance on code generation tasks. Rather than executing generated code, BIS estimates performance metrics by analyzing the prompt distribution alone. Built on importance sampling theory and implemented using Importance Weighted Autoencoders, our method reweights samples from existing annotated benchmarks to estimate performance on new, unseen benchmarks. To stabilize the estimation, we introduce weight truncation strategies and compute marginal expectations across the fitted distributions. BIS serves as a complementary tool that supports benchmark development and validation under constrained resources, offering actionable and quick feedback for prompt selection and contamination assessment. We conduct extensive experiments involving 8,000 evaluation points across 4 CodeLlama models and 9 diverse benchmarks. Our framework achieves an average absolute prediction error of 1.1% for code correctness scores, with best- and worst-case errors of 0.3% and 1.9%, respectively. It also generalizes well to other metrics, attaining average absolute errors of 2.15% for pass@1. These results demonstrate the reliability and broad applicability of BIS, which can significantly reduce the cost and effort of benchmarking LLMs in code-related tasks. 

---
# StructSynth: Leveraging LLMs for Structure-Aware Tabular Data Synthesis in Low-Data Regimes 

**Authors**: Siyi Liu, Yujia Zheng, Yongqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02601)  

**Abstract**: The application of machine learning on tabular data in specialized domains is severely limited by data scarcity. While generative models offer a solution, traditional methods falter in low-data regimes, and recent Large Language Models (LLMs) often ignore the explicit dependency structure of tabular data, leading to low-fidelity synthetics. To address these limitations, we introduce StructSynth, a novel framework that integrates the generative power of LLMs with robust structural control. StructSynth employs a two-stage architecture. First, it performs explicit structure discovery to learn a Directed Acyclic Graph (DAG) from the available data. Second, this learned structure serves as a high-fidelity blueprint to steer the LLM's generation process, forcing it to adhere to the learned feature dependencies and thereby ensuring the generated data respects the underlying structure by design. Our extensive experiments demonstrate that StructSynth produces synthetic data with significantly higher structural integrity and downstream utility than state-of-the-art methods. It proves especially effective in challenging low-data scenarios, successfully navigating the trade-off between privacy preservation and statistical fidelity. 

---
# Assessing the Reliability and Validity of Large Language Models for Automated Assessment of Student Essays in Higher Education 

**Authors**: Andrea Gaggioli, Giuseppe Casaburi, Leonardo Ercolani, Francesco Collova', Pietro Torre, Fabrizio Davide  

**Link**: [PDF](https://arxiv.org/pdf/2508.02442)  

**Abstract**: This study investigates the reliability and validity of five advanced Large Language Models (LLMs), Claude 3.5, DeepSeek v2, Gemini 2.5, GPT-4, and Mistral 24B, for automated essay scoring in a real world higher education context. A total of 67 Italian-language student essays, written as part of a university psychology course, were evaluated using a four-criterion rubric (Pertinence, Coherence, Originality, Feasibility). Each model scored all essays across three prompt replications to assess intra-model stability. Human-LLM agreement was consistently low and non-significant (Quadratic Weighted Kappa), and within-model reliability across replications was similarly weak (median Kendall's W < 0.30). Systematic scoring divergences emerged, including a tendency to inflate Coherence and inconsistent handling of context-dependent dimensions. Inter-model agreement analysis revealed moderate convergence for Coherence and Originality, but negligible concordance for Pertinence and Feasibility. Although limited in scope, these findings suggest that current LLMs may struggle to replicate human judgment in tasks requiring disciplinary insight and contextual sensitivity. Human oversight remains critical when evaluating open-ended academic work, particularly in interpretive domains. 

---
# MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models 

**Authors**: Wenyuan Liu, Haoqian Meng, Yilun Luo, Peng Zhang, Xindian Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.02343)  

**Abstract**: Quantization significantly accelerates inference in large language models (LLMs) by replacing original high-precision matrices with low-precision counterparts. Recent advances in weight-activation quantization have primarily focused on mapping both weights and activations to the INT4 format. Although the new FP4 Tensor Cores in NVIDIA's Blackwell architecture offer up to 4x speedup over FP16, existing INT4-based kernels fail to fully exploit this capability due to mismatched data formats. To bridge this gap, we propose MicroMix, a co-designed mixed-precision quantization algorithm and matrix multiplication kernel based on Microscaling (MX) data formats. Tailored for the Blackwell architecture, the MicroMix kernel supports arbitrary combinations of MXFP4, MXFP6, and MXFP8 channels, and produces BFloat16 outputs. To achieve a favorable trade-off between accuracy and efficiency for each linear layer, we introduce quantization thresholds that identify activation elements where lower-precision formats (MXFP4 or MXFP6) incur excessive quantization error. Our algorithm selectively allocates higher-precision channels to preserve accuracy while maintaining compute efficiency. MicroMix achieves competitive or superior performance across diverse downstream tasks, including zero-shot and few-shot learning, language modeling, code generation, and mathematical reasoning. On both consumer-grade (RTX 5070Ti laptop) and server-grade (RTX 5090) GPUs, our kernel delivers at least 20% faster execution than TensorRT-FP8. Furthermore, when applied to various Llama and Qwen models, MicroMix consistently improves prefill latency and memory efficiency across a range of batch sizes compared to TensorRT baselines. Our code is available at this https URL. 

---
# Accelerating LLM Reasoning via Early Rejection with Partial Reward Modeling 

**Authors**: Seyyed Saeid Cheshmi, Azal Ahmad Khan, Xinran Wang, Zirui Liu, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2508.01969)  

**Abstract**: Large Language Models (LLMs) are increasingly relied upon for solving complex reasoning tasks in domains such as mathematics, logic, and multi-step question answering. A growing line of work seeks to improve reasoning quality by scaling inference time compute particularly through Process Reward Models (PRMs), used to reward the reasoning at intermediate steps. While effective, these methods introduce substantial computational overhead, especially when generating large numbers of solutions in parallel. In this paper, we investigate whether PRMs can be used mid-generation to provide early signals that enable the rejection of suboptimal candidates before full generation of step is complete. We introduce the hypothesis that PRMs are also Partial Reward Models, meaning that the scores they assign to partially completed reasoning step are predictive of final output quality. This allows for principled early rejection based on intermediate token-level signals. We support this hypothesis both theoretically, by proving that the risk of discarding optimal beams decreases exponentially with generation length and empirically, by demonstrating a strong correlation between partial and final rewards across multiple reward models. On math reasoning benchmarks, our method achieves up to 1.4$\times$-9$\times$ reduction in inference FLOPs without degrading final performance. These results suggest that early rejection is a powerful mechanism for improving the compute-efficiency of reasoning in LLMs. 

---
# AlignGuard-LoRA: Alignment-Preserving Fine-Tuning via Fisher-Guided Decomposition and Riemannian-Geodesic Collision Regularization 

**Authors**: Amitava Das, Abhilekh Borah, Vinija Jain, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2508.02079)  

**Abstract**: Low-rank adaptation (LoRA) has become a standard tool for efficiently fine-tuning large language models (LLMs). Yet, even minor LoRA updates can induce alignment drift, weakening safety and behavioral constraints through entangled parameter changes. To address this, we propose AlignGuard-LoRA (AGL), a principled framework for preserving alignment during finetuning. AGL introduces several key components: a primary task loss for supervision, Fisher Information Matrix-based regularization to restrict updates in alignment-sensitive subspaces, and task-specific regularization to stabilize the integration of new knowledge. We further introduce collision-aware regularization, blending Riemannian overlap -- which penalizes coordinate-wise interference -- and geodesic separation -- which encourages disjoint update geometry. We curate DriftCaps, a targeted diagnostic benchmark of safe and unsafe prompts designed to quantify alignment drift and safety degradation. Empirical evaluations show that AGL mitigates alignment drift by up to 50% on safety-critical benchmarks without degrading downstream task performance. Comprehensive ablation confirms that each component contributes distinctly to preserving latent safety behaviors. Finally, we derive and validate a scaling law for catastrophic forgetting, revealing that AGL flattens post-finetuning loss escalation while preserving adaptation dynamics. AGL is a structurally grounded refinement of LoRA, ensuring alignment preservation with minimal trade-offs. To encourage further exploration and development, we open-source our implementation. 

---
# A comprehensive taxonomy of hallucinations in Large Language Models 

**Authors**: Manuel Cossio  

**Link**: [PDF](https://arxiv.org/pdf/2508.01781)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet their propensity for hallucination, generating plausible but factually incorrect or fabricated content, remains a critical challenge. This report provides a comprehensive taxonomy of LLM hallucinations, beginning with a formal definition and a theoretical framework that posits its inherent inevitability in computable LLMs, irrespective of architecture or training. It explores core distinctions, differentiating between intrinsic (contradicting input context) and extrinsic (inconsistent with training data or reality), as well as factuality (absolute correctness) and faithfulness (adherence to input). The report then details specific manifestations, including factual errors, contextual and logical inconsistencies, temporal disorientation, ethical violations, and task-specific hallucinations across domains like code generation and multimodal applications. It analyzes the underlying causes, categorizing them into data-related issues, model-related factors, and prompt-related influences. Furthermore, the report examines cognitive and human factors influencing hallucination perception, surveys evaluation benchmarks and metrics for detection, and outlines architectural and systemic mitigation strategies. Finally, it introduces web-based resources for monitoring LLM releases and performance. This report underscores the complex, multifaceted nature of LLM hallucinations and emphasizes that, given their theoretical inevitability, future efforts must focus on robust detection, mitigation, and continuous human oversight for responsible and reliable deployment in critical applications. 

---
# L3M+P: Lifelong Planning with Large Language Models 

**Authors**: Krish Agarwal, Yuqian Jiang, Jiaheng Hu, Bo Liu, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2508.01917)  

**Abstract**: By combining classical planning methods with large language models (LLMs), recent research such as LLM+P has enabled agents to plan for general tasks given in natural language. However, scaling these methods to general-purpose service robots remains challenging: (1) classical planning algorithms generally require a detailed and consistent specification of the environment, which is not always readily available; and (2) existing frameworks mainly focus on isolated planning tasks, whereas robots are often meant to serve in long-term continuous deployments, and therefore must maintain a dynamic memory of the environment which can be updated with multi-modal inputs and extracted as planning knowledge for future tasks. To address these two issues, this paper introduces L3M+P (Lifelong LLM+P), a framework that uses an external knowledge graph as a representation of the world state. The graph can be updated from multiple sources of information, including sensory input and natural language interactions with humans. L3M+P enforces rules for the expected format of the absolute world state graph to maintain consistency between graph updates. At planning time, given a natural language description of a task, L3M+P retrieves context from the knowledge graph and generates a problem definition for classical planners. Evaluated on household robot simulators and on a real-world service robot, L3M+P achieves significant improvement over baseline methods both on accurately registering natural language state changes and on correctly generating plans, thanks to the knowledge graph retrieval and verification. 

---
# Semantic Encryption: Secure and Effective Interaction with Cloud-based Large Language Models via Semantic Transformation 

**Authors**: Dong Chen, Tong Yang, Feipeng Zhai, Pengpeng Ouyang, Qidong Liu, Yafei Li, Chong Fu, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01638)  

**Abstract**: The increasing adoption of Cloud-based Large Language Models (CLLMs) has raised significant concerns regarding data privacy during user interactions. While existing approaches primarily focus on encrypting sensitive information, they often overlook the logical structure of user inputs. This oversight can lead to reduced data utility and degraded performance of CLLMs. To address these limitations and enable secure yet effective interactions, we propose Semantic Encryption (SE)-a plug-and-play framework designed to preserve both privacy and utility. SE consists of two key components: Semantic Encoding and Semantic Decoding. In the encoding phase, a lightweight local model transforms the original user input into an alternative semantic context that maintains the original intent and logical structure while obfuscating sensitive information. This transformed input is then processed by the CLLM, which generates a response based on the transformed semantic context. To maintain a seamless user experience, the decoding phase will reconstruct the CLLM's response back into the original semantic context by referencing the locally stored user input. Extensive experimental evaluations demonstrate that SE effectively protects data privacy without compromising data utility or user experience, offering a practical solution for secure interaction with CLLMs. Particularly, the proposed SE demonstrates a significant improvement over the state-of-the-art InferDPT, surpassing it across various evaluated metrics and datasets. 

---
# Are All Prompt Components Value-Neutral? Understanding the Heterogeneous Adversarial Robustness of Dissected Prompt in Large Language Models 

**Authors**: Yujia Zheng, Tianhao Li, Haotian Huang, Tianyu Zeng, Jingyu Lu, Chuangxin Chu, Yuekai Huang, Ziyou Jiang, Qian Xiong, Yuyao Ge, Mingyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01554)  

**Abstract**: Prompt-based adversarial attacks have become an effective means to assess the robustness of large language models (LLMs). However, existing approaches often treat prompts as monolithic text, overlooking their structural heterogeneity-different prompt components contribute unequally to adversarial robustness. Prior works like PromptRobust assume prompts are value-neutral, but our analysis reveals that complex, domain-specific prompts with rich structures have components with differing vulnerabilities. To address this gap, we introduce PromptAnatomy, an automated framework that dissects prompts into functional components and generates diverse, interpretable adversarial examples by selectively perturbing each component using our proposed method, ComPerturb. To ensure linguistic plausibility and mitigate distribution shifts, we further incorporate a perplexity (PPL)-based filtering mechanism. As a complementary resource, we annotate four public instruction-tuning datasets using the PromptAnatomy framework, verified through human review. Extensive experiments across these datasets and five advanced LLMs demonstrate that ComPerturb achieves state-of-the-art attack success rates. Ablation studies validate the complementary benefits of prompt dissection and PPL filtering. Our results underscore the importance of prompt structure awareness and controlled perturbation for reliable adversarial robustness evaluation in LLMs. Code and data are available at this https URL. 

---
# Understanding Why ChatGPT Outperforms Humans in Visualization Design Advice 

**Authors**: Yongsu Ahn, Nam Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01547)  

**Abstract**: This paper investigates why recent generative AI models outperform humans in data visualization knowledge tasks. Through systematic comparative analysis of responses to visualization questions, we find that differences exist between two ChatGPT models and human outputs over rhetorical structure, knowledge breadth, and perceptual quality. Our findings reveal that ChatGPT-4, as a more advanced model, displays a hybrid of characteristics from both humans and ChatGPT-3.5. The two models were generally favored over human responses, while their strengths in coverage and breadth, and emphasis on technical and task-oriented visualization feedback collectively shaped higher overall quality. Based on our findings, we draw implications for advancing user experiences based on the potential of LLMs and human perception over their capabilities, with relevance to broader applications of AI. 

---
# Tuning LLM-based Code Optimization via Meta-Prompting: An Industrial Perspective 

**Authors**: Jingzhi Gong, Rafail Giavrimis, Paul Brookes, Vardan Voskanyan, Fan Wu, Mari Ashiga, Matthew Truscott, Mike Basios, Leslie Kanthan, Jie Xu, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01443)  

**Abstract**: There is a growing interest in leveraging large language models (LLMs) for automated code optimization. However, industrial platforms deploying multiple LLMs face a critical challenge: prompts optimized for one LLM often fail with others, requiring expensive model-specific prompt engineering. This cross-model prompt engineering bottleneck severely limits the practical deployment of multi-LLM optimization systems in production environments. To address this, we introduce Meta-Prompted Code Optimization (MPCO), a framework that automatically generates high-quality, task-specific prompts across diverse LLMs while maintaining industrial efficiency requirements. MPCO leverages meta-prompting to dynamically synthesize context-aware optimization prompts by integrating project metadata, task requirements, and LLM-specific contexts, and it seamlessly deploys on the ARTEMIS industrial platform for automated validation and scaling.
Our comprehensive evaluation on five real-world codebases with 366 hours of runtime benchmarking demonstrates MPCO's effectiveness: it achieves overall performance improvements up to 19.06% with the best statistical rank across all systems compared to baseline methods. Analysis shows that 96% of the top-performing optimizations stem from meaningful edits. Through systematic ablation studies and meta-prompter sensitivity analysis, we identify that comprehensive context integration is essential for effective meta-prompting, and that all three major LLMs can serve effectively as meta-prompters, providing actionable insights for industrial practitioners. 

---
# BlockA2A: Towards Secure and Verifiable Agent-to-Agent Interoperability 

**Authors**: Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, Qiuyang Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01332)  

**Abstract**: The rapid adoption of agentic AI, powered by large language models (LLMs), is transforming enterprise ecosystems with autonomous agents that execute complex workflows. Yet we observe several key security vulnerabilities in LLM-driven multi-agent systems (MASes): fragmented identity frameworks, insecure communication channels, and inadequate defenses against Byzantine agents or adversarial prompts. In this paper, we present the first systematic analysis of these emerging multi-agent risks and explain why the legacy security strategies cannot effectively address these risks. Afterwards, we propose BlockA2A, the first unified multi-agent trust framework that enables secure and verifiable and agent-to-agent interoperability. At a high level, BlockA2A adopts decentralized identifiers (DIDs) to enable fine-grained cross-domain agent authentication, blockchain-anchored ledgers to enable immutable auditability, and smart contracts to dynamically enforce context-aware access control policies. BlockA2A eliminates centralized trust bottlenecks, ensures message authenticity and execution integrity, and guarantees accountability across agent interactions. Furthermore, we propose a Defense Orchestration Engine (DOE) that actively neutralizes attacks through real-time mechanisms, including Byzantine agent flagging, reactive execution halting, and instant permission revocation. Empirical evaluations demonstrate BlockA2A's effectiveness in neutralizing prompt-based, communication-based, behavioral and systemic MAS attacks. We formalize its integration into existing MAS and showcase a practical implementation for Google's A2A protocol. Experiments confirm that BlockA2A and DOE operate with sub-second overhead, enabling scalable deployment in production LLM-based MAS environments. 

---
# RSPO: Risk-Seeking Policy Optimization for Pass@k and Max@k Metrics in Large Language Models 

**Authors**: Kaichen Zhang, Shenghao Gao, Yuzhong Hong, Haipeng Sun, Junwei Bao, Hongfei Jiang, Yang Song, Hong Dingqian, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01174)  

**Abstract**: Current large language model post-training optimizes a risk-neutral objective that maximizes expected reward, yet evaluation relies heavily on risk-seeking metrics like Pass@k (at least one success in k trials) and Max@k (maximum reward across k responses). This mismatch in risk preferences can inevitably lead to suboptimal performance. To bridge this gap, we propose Risk-Seeking Policy Optimization (RSPO), a novel method that directly targets Pass@k and Max@k during training. A key challenge in optimizing these metrics is the "hitchhiking" problem: low-reward responses are inadvertently reinforced if they co-occur with a high-reward response within a sample of k generations, resulting in inefficient optimization. RSPO addresses this problem by leveraging the closed-form probability that a given response is the maximum among k samplings. Despite the complexity of nested gradients over multiple responses, RSPO produces efficient, unbiased gradient estimators for both metrics. We validate our approach with both rigorous theoretical analysis and comprehensive experimental results. 

---
# Autonomous Penetration Testing: Solving Capture-the-Flag Challenges with LLMs 

**Authors**: Isabelle Bakker, John Hastings  

**Link**: [PDF](https://arxiv.org/pdf/2508.01054)  

**Abstract**: This study evaluates the ability of GPT-4o to autonomously solve beginner-level offensive security tasks by connecting the model to OverTheWire's Bandit capture-the-flag game. Of the 25 levels that were technically compatible with a single-command SSH framework, GPT-4o solved 18 unaided and another two after minimal prompt hints for an overall 80% success rate. The model excelled at single-step challenges that involved Linux filesystem navigation, data extraction or decoding, and straightforward networking. The approach often produced the correct command in one shot and at a human-surpassing speed. Failures involved multi-command scenarios that required persistent working directories, complex network reconnaissance, daemon creation, or interaction with non-standard shells. These limitations highlight current architectural deficiencies rather than a lack of general exploit knowledge. The results demonstrate that large language models (LLMs) can automate a substantial portion of novice penetration-testing workflow, potentially lowering the expertise barrier for attackers and offering productivity gains for defenders who use LLMs as rapid reconnaissance aides. Further, the unsolved tasks reveal specific areas where secure-by-design environments might frustrate simple LLM-driven attacks, informing future hardening strategies. Beyond offensive cybersecurity applications, results suggest the potential to integrate LLMs into cybersecurity education as practice aids. 

---
# Are LLM-Powered Social Media Bots Realistic? 

**Authors**: Lynnette Hui Xian Ng, Kathleen M. Carley  

**Link**: [PDF](https://arxiv.org/pdf/2508.00998)  

**Abstract**: As Large Language Models (LLMs) become more sophisticated, there is a possibility to harness LLMs to power social media bots. This work investigates the realism of generating LLM-Powered social media bot networks. Through a combination of manual effort, network science and LLMs, we create synthetic bot agent personas, their tweets and their interactions, thereby simulating social media networks. We compare the generated networks against empirical bot/human data, observing that both network and linguistic properties of LLM-Powered Bots differ from Wild Bots/Humans. This has implications towards the detection and effectiveness of LLM-Powered Bots. 

---
# FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models 

**Authors**: Xuan Liu, Siru Ouyang, Xianrui Zhong, Jiawei Han, Huimin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01055)  

**Abstract**: Large language models (LLMs) have gained significant attention in chemistry. However, most existing datasets center on molecular-level property prediction and overlook the role of fine-grained functional group (FG) information. Incorporating FG-level data can provide valuable prior knowledge that links molecular structures with textual descriptions, which can be used to build more interpretable, structure-aware LLMs for reasoning on molecule-related tasks. Moreover, LLMs can learn from such fine-grained information to uncover hidden relationships between specific functional groups and molecular properties, thereby advancing molecular design and drug discovery. Here, we introduce FGBench, a dataset comprising 625K molecular property reasoning problems with functional group information. Functional groups are precisely annotated and localized within the molecule, which ensures the dataset's interoperability thereby facilitating further multimodal applications. FGBench includes both regression and classification tasks on 245 different functional groups across three categories for molecular property reasoning: (1) single functional group impacts, (2) multiple functional group interactions, and (3) direct molecular comparisons. In the benchmark of state-of-the-art LLMs on 7K curated data, the results indicate that current LLMs struggle with FG-level property reasoning, highlighting the need to enhance reasoning capabilities in LLMs for chemistry tasks. We anticipate that the methodology employed in FGBench to construct datasets with functional group-level information will serve as a foundational framework for generating new question-answer pairs, enabling LLMs to better understand fine-grained molecular structure-property relationships. The dataset and evaluation code are available at \href{this https URL}{this https URL}. 

---
# Llama-3.1-FoundationAI-SecurityLLM-8B-Instruct Technical Report 

**Authors**: Sajana Weerawardhena, Paul Kassianik, Blaine Nelson, Baturay Saglam, Anu Vellore, Aman Priyanshu, Supriti Vijay, Massimo Aufiero, Arthur Goldblatt, Fraser Burch, Ed Li, Jianliang He, Dhruv Kedia, Kojin Oshiba, Zhouran Yang, Yaron Singer, Amin Karbasi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01059)  

**Abstract**: Large language models (LLMs) have shown remarkable success across many domains, yet their integration into cybersecurity applications remains limited due to a lack of general-purpose cybersecurity data, representational complexity, and safety and regulatory concerns. To address this gap, we previously introduced Foundation-Sec-8B, a cybersecurity-focused LLM suitable for fine-tuning on downstream tasks. That model, however, was not designed for chat-style interactions or instruction-following. In this report, we release Foundation-Sec-8B-Instruct: a model specifically trained for general-purpose cybersecurity dialogue. Built on Foundation-Sec-8B, it combines domain-specific knowledge with instruction-following, conversational capabilities, and alignment with human preferences to produce high-quality, relevant responses. Comprehensive evaluations show that Foundation-Sec-8B-Instruct outperforms Llama 3.1-8B-Instruct on a range of cybersecurity tasks while matching its instruction-following performance. It is also competitive with GPT-4o-mini on cyber threat intelligence and instruction-following tasks. We envision Foundation-Sec-8B-Instruct becoming an indispensable assistant in the daily workflows of cybersecurity professionals. We release the model publicly at this https URL. 

---
# Academic Vibe Coding: Opportunities for Accelerating Research in an Era of Resource Constraint 

**Authors**: Matthew G Crowson, Leo Celi A. Celi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00952)  

**Abstract**: Academic laboratories face mounting resource constraints: budgets are tightening, grant overheads are potentially being capped, and the market rate for data-science talent significantly outstrips university compensation. Vibe coding, which is structured, prompt-driven code generation with large language models (LLMs) embedded in reproducible workflows, offers one pragmatic response. It aims to compress the idea-to-analysis timeline, reduce staffing pressure on specialized data roles, and maintain rigorous, version-controlled outputs. This article defines the vibe coding concept, situates it against the current academic resourcing crisis, details a beginner-friendly toolchain for its implementation, and analyzes inherent limitations that necessitate governance and mindful application. 

---
# VAULT: Vigilant Adversarial Updates via LLM-Driven Retrieval-Augmented Generation for NLI 

**Authors**: Roie Kazoom, Ofir Cohen, Rami Puzis, Asaf Shabtai, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00965)  

**Abstract**: We introduce VAULT, a fully automated adversarial RAG pipeline that systematically uncovers and remedies weaknesses in NLI models through three stages: retrieval, adversarial generation, and iterative retraining. First, we perform balanced few-shot retrieval by embedding premises with both semantic (BGE) and lexical (BM25) similarity. Next, we assemble these contexts into LLM prompts to generate adversarial hypotheses, which are then validated by an LLM ensemble for label fidelity. Finally, the validated adversarial examples are injected back into the training set at increasing mixing ratios, progressively fortifying a zero-shot RoBERTa-base this http URL standard benchmarks, VAULT elevates RoBERTa-base accuracy from 88.48% to 92.60% on SNLI +4.12%, from 75.04% to 80.95% on ANLI +5.91%, and from 54.67% to 71.99% on MultiNLI +17.32%. It also consistently outperforms prior in-context adversarial methods by up to 2.0% across datasets. By automating high-quality adversarial data curation at scale, VAULT enables rapid, human-independent robustness improvements in NLI inference tasks. 

---
# OKG-LLM: Aligning Ocean Knowledge Graph with Observation Data via LLMs for Global Sea Surface Temperature Prediction 

**Authors**: Hanchen Yang, Jiaqi Wang, Jiannong Cao, Wengen Li, Jialun Zheng, Yangning Li, Chunyu Miao, Jihong Guan, Shuigeng Zhou, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00933)  

**Abstract**: Sea surface temperature (SST) prediction is a critical task in ocean science, supporting various applications, such as weather forecasting, fisheries management, and storm tracking. While existing data-driven methods have demonstrated significant success, they often neglect to leverage the rich domain knowledge accumulated over the past decades, limiting further advancements in prediction accuracy. The recent emergence of large language models (LLMs) has highlighted the potential of integrating domain knowledge for downstream tasks. However, the application of LLMs to SST prediction remains underexplored, primarily due to the challenge of integrating ocean domain knowledge and numerical data. To address this issue, we propose Ocean Knowledge Graph-enhanced LLM (OKG-LLM), a novel framework for global SST prediction. To the best of our knowledge, this work presents the first systematic effort to construct an Ocean Knowledge Graph (OKG) specifically designed to represent diverse ocean knowledge for SST prediction. We then develop a graph embedding network to learn the comprehensive semantic and structural knowledge within the OKG, capturing both the unique characteristics of individual sea regions and the complex correlations between them. Finally, we align and fuse the learned knowledge with fine-grained numerical SST data and leverage a pre-trained LLM to model SST patterns for accurate prediction. Extensive experiments on the real-world dataset demonstrate that OKG-LLM consistently outperforms state-of-the-art methods, showcasing its effectiveness, robustness, and potential to advance SST prediction. The codes are available in the online repository. 

---
# Predictive Auditing of Hidden Tokens in LLM APIs via Reasoning Length Estimation 

**Authors**: Ziyao Wang, Guoheng Sun, Yexiao He, Zheyu Shen, Bowei Tian, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00912)  

**Abstract**: Commercial LLM services often conceal internal reasoning traces while still charging users for every generated token, including those from hidden intermediate steps, raising concerns of token inflation and potential overbilling. This gap underscores the urgent need for reliable token auditing, yet achieving it is far from straightforward: cryptographic verification (e.g., hash-based signature) offers little assurance when providers control the entire execution pipeline, while user-side prediction struggles with the inherent variance of reasoning LLMs, where token usage fluctuates across domains and prompt styles. To bridge this gap, we present PALACE (Predictive Auditing of LLM APIs via Reasoning Token Count Estimation), a user-side framework that estimates hidden reasoning token counts from prompt-answer pairs without access to internal traces. PALACE introduces a GRPO-augmented adaptation module with a lightweight domain router, enabling dynamic calibration across diverse reasoning tasks and mitigating variance in token usage patterns. Experiments on math, coding, medical, and general reasoning benchmarks show that PALACE achieves low relative error and strong prediction accuracy, supporting both fine-grained cost auditing and inflation detection. Taken together, PALACE represents an important first step toward standardized predictive auditing, offering a practical path to greater transparency, accountability, and user trust. 

---
# Generative AI for CAD Automation: Leveraging Large Language Models for 3D Modelling 

**Authors**: Sumit Kumar, Sarthak Kapoor, Harsh Vardhan, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00843)  

**Abstract**: Large Language Models (LLMs) are revolutionizing industries by enhancing efficiency, scalability, and innovation. This paper investigates the potential of LLMs in automating Computer-Aided Design (CAD) workflows, by integrating FreeCAD with LLM as CAD design tool. Traditional CAD processes are often complex and require specialized sketching skills, posing challenges for rapid prototyping and generative design. We propose a framework where LLMs generate initial CAD scripts from natural language descriptions, which are then executed and refined iteratively based on error feedback. Through a series of experiments with increasing complexity, we assess the effectiveness of this approach. Our findings reveal that LLMs perform well for simple to moderately complex designs but struggle with highly constrained models, necessitating multiple refinements. The study highlights the need for improved memory retrieval, adaptive prompt engineering, and hybrid AI techniques to enhance script robustness. Future directions include integrating cloud-based execution and exploring advanced LLM capabilities to further streamline CAD automation. This work underscores the transformative potential of LLMs in design workflows while identifying critical areas for future development. 

---
# FinKario: Event-Enhanced Automated Construction of Financial Knowledge Graph 

**Authors**: Xiang Li, Penglei Sun, Wanyun Zhou, Zikai Wei, Yongqi Zhang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00961)  

**Abstract**: Individual investors are significantly outnumbered and disadvantaged in financial markets, overwhelmed by abundant information and lacking professional analysis. Equity research reports stand out as crucial resources, offering valuable insights. By leveraging these reports, large language models (LLMs) can enhance investors' decision-making capabilities and strengthen financial analysis. However, two key challenges limit their effectiveness: (1) the rapid evolution of market events often outpaces the slow update cycles of existing knowledge bases, (2) the long-form and unstructured nature of financial reports further hinders timely and context-aware integration by LLMs. To address these challenges, we tackle both data and methodological aspects. First, we introduce the Event-Enhanced Automated Construction of Financial Knowledge Graph (FinKario), a dataset comprising over 305,360 entities, 9,625 relational triples, and 19 distinct relation types. FinKario automatically integrates real-time company fundamentals and market events through prompt-driven extraction guided by professional institutional templates, providing structured and accessible financial insights for LLMs. Additionally, we propose a Two-Stage, Graph-Based retrieval strategy (FinKario-RAG), optimizing the retrieval of evolving, large-scale financial knowledge to ensure efficient and precise data access. Extensive experiments show that FinKario with FinKario-RAG achieves superior stock trend prediction accuracy, outperforming financial LLMs by 18.81% and institutional strategies by 17.85% on average in backtesting. 

---
# LLMs Can Covertly Sandbag on Capability Evaluations Against Chain-of-Thought Monitoring 

**Authors**: Chloe Li, Mary Phuong, Noah Y. Siegel  

**Link**: [PDF](https://arxiv.org/pdf/2508.00943)  

**Abstract**: Trustworthy evaluations of dangerous capabilities are increasingly crucial for determining whether an AI system is safe to deploy. One empirically demonstrated threat to this is sandbagging - the strategic underperformance on evaluations by AI models or their developers. One promising defense is to monitor a model's chain-of-thought (CoT) reasoning, as this could reveal its intentions and plans. In this work, we measure the ability of models to sandbag on dangerous capability evaluations against a CoT monitor by prompting them to sandbag while being either monitor-oblivious or monitor-aware. We show that both frontier models and small open-sourced models can covertly sandbag against CoT monitoring 0-shot without hints. However, they cannot yet do so reliably: they bypass the monitor 16-36\% of the time when monitor-aware, conditioned on sandbagging successfully. We qualitatively analyzed the uncaught CoTs to understand why the monitor failed. We reveal a rich attack surface for CoT monitoring and contribute five covert sandbagging policies generated by models. These results inform potential failure modes of CoT monitoring and may help build more diverse sandbagging model organisms. 

---
