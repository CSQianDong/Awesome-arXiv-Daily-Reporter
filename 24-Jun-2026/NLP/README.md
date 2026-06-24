# Less is More: Quality-Aware Training Data Selection for Scientific Summarization 

**Authors**: Maria Nefeli Paraskevopoulou, Tatiana Passali, Grigorios Tsoumakas  

**Link**: [PDF](https://arxiv.org/pdf/2606.24828)  

**Abstract**: Scientific long-document summarization datasets commonly treat author-written abstracts as gold reference summaries, although their quality and alignment with the source article vary. At the same time, publicly available scientific summarization datasets remain limited in scale and structure for modern long-context models. In this work, we address both challenges by a) constructing and releasing one of the largest biomedical and life science datasets for long-document summarization, containing 1.88 million PMC articles, and b) analyzing the reference quality of author-written abstracts with source-grounded and model-based metrics. We show that author-written abstracts vary in their alignment with the full article and that these quality signals can guide training-data selection. Training on selected high-quality subsets outperforms random sampling at matched training sizes and can match or exceed larger random subsets on factuality-oriented metrics. Our findings suggest that reference quality is an important factor in scientific summarization and that quality-aware data selection can improve training efficiency. 

---
# L3Cube-MahaPOS: A Marathi Part-of-Speech Tagging Dataset and BERT Models 

**Authors**: Hariom Ingle, Ronit Ghode, Ishwari Gondkar, Jidnyasa Harad, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2606.24825)  

**Abstract**: Part-of-Speech (POS) tagging is a foundational NLP task underpinning machine translation, information extraction, and syntactic parsing. Despite Marathi being spoken by over 83 million people and ranking among the top twenty most spoken languages worldwide, it remains severely under-resourced in annotated corpora and standardised evaluation benchmarks. Marathi presents unique challenges for computational modelling owing to its rich morphology, relatively free word order, lack of capitalisation conventions, and pervasive code-mixing with Hindi and English. We introduce L3Cube-MahaPOS, a gold-standard POS tagging dataset for Marathi comprising 32,354 manually annotated sentences drawn from news text. Annotation was performed entirely manually by a team of Marathi-proficient annotators following a 16-tag Universal Dependencies-aligned scheme. A structured preprocessing pipeline covering Unicode normalisation, Devanagari-aware tokenisation, and noise filtering ensures label consistency across all splits. We benchmark the dataset across six model families spanning HMM, CRF, BiLSTM, BiLSTM+CharCNN, MuRIL, and the Marathi-specific transformer MahaBERT-v2. The best system achieves 88.67\% token-level accuracy and a macro-F1 of 81.67% over 15 evaluated tag classes. We release the dataset, annotation guidelines, and trained model checkpoints to foster further research in Marathi NLP. 

---
# SHERLOC: Structured Diagnostic Localization for Code Repair Agents 

**Authors**: Hovhannes Tamoyan, Sean Narenthiran, Erik Arakelyan, Mira Mezini, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2606.24820)  

**Abstract**: LLM agents solve repository-level coding tasks through multi-turn tool use, but utilize half their budget on locating faults before editing. Dedicated localization frameworks have emerged, yet are still evaluated as file retrieval rather than actionable diagnosis, producing locations without the diagnostic context a repair agent needs. We introduce SHERLOC (Structured Hypothesis-driven Exploration and Reasoning for Localization), a training-free framework pairing a reasoning LLM with compact repository tools and self-recovery, without fine-tuning or multi-agent orchestration. SHERLOC reaches state-of-the-art localization across model scales: 84.33% accuracy@1 on SWE-Bench Lite and 81.27% recall@1 on SWE-Bench Verified; at ~30B parameters, it matches or outperforms other agentic methods. Injecting our locations and diagnostic findings into repair agents yields, on average, +5.95 pp resolve rate on SWE-Bench Verified while cutting localization and total tokens by 36.7% and 23.1%. 

---
# Paying to Know: Micro-Transaction Markets for Verified Product Information in Agentic E-Commerce 

**Authors**: Filippos Ventirozos, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2606.24783)  

**Abstract**: Commercial NLP treats the shopping chatbot as a recommender or a conversion tool: its job is to match a user to a catalogue entry and close a sale. We argue that the arrival of agent-native micro-payment rails (e.g., x402, AP2) changes what is scarce. When the buyer is an autonomous agent that can investigate exhaustively, the bottleneck is no longer matching products but acquiring trustworthy, decision-relevant information about them. We envision agentic e-commerce as a micro-transaction market for verified information: buyer agents spend fractions of a cent to progressively unlock seller- and reviewer-supplied data -- service histories, third-party test reports, bills of materials, audited sales and support metrics -- paid for a la carte under a freemium model, with reviewer trust scored reputationally. We sketch the architecture of such a market and argue that it rewards genuine product quality and yields truer competition than ranking-based storefronts. We then translate the vision into concrete NLP problems -- cost-optimal information acquisition, data pricing and negotiation, real-time entity resolution, grounded value exchange, and privacy-preserving persona modelling -- and argue that these, not chat fluency, deserve the field's attention. 

---
# Are We Ready For An Agent-Native Memory System? 

**Authors**: Wei Zhou, Xuanhe Zhou, Shaokun Han, Hongming Xu, Guoliang Li, Zhiyu Li, Feiyu Xiong, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.24775)  

**Abstract**: Memory for large language model (LLM) agents has rapidly evolved from simple retrieval-augmented mechanisms into a data management system that supports persistent information storage, retrieval, update, consolidation, and dynamic lifecycle governance throughout agent execution. Despite this evolution, existing evaluations still benchmark agent memory mainly through end-to-end task success metrics (e.g., F1, BLEU), while treating the underlying system as a monolithic black box. As a result, critical system-level concerns, including operational costs, architectural trade-offs across memory modules, and robustness under dynamic knowledge updates, remain insufficiently explored. In this paper, we present a systematic experimental study of agent memory from a data management perspective. We propose an analytical framework that decomposes agent memory into four core modules: memory representation and storage, extraction, retrieval and routing, and maintenance. Under this framework, we evaluate 12 representative memory systems and two reference baselines across five benchmark workloads spanning 11 datasets. Our extensive end-to-end evaluation shows that no single architecture dominates across all scenarios; instead, effectiveness depends heavily on how well the memory structure aligns with the workload bottleneck. Furthermore, through fine-grained ablation studies, we quantify their individual effects on representation fidelity, retrieval precision, update correctness, and long-horizon stability. Finally, we reveal cost-performance trade-offs under realistic workloads, showing localized maintenance is more cost-efficient than global reorganization. Based on these findings, we identify promising directions towards building truly agent-native memory systems. The code is publicly available at this https URL. 

---
# Posterior Refinement: Fast Language Generation via Any-Order Flow Maps 

**Authors**: Manan Agarwal, Sheel Shah, Chanhyuk Lee, Jaehoon Yoo, Jerry Huang, Seunghoon Hong, Aditi Raghunathan, Jinwoo Kim, Nicholas M. Boffi  

**Link**: [PDF](https://arxiv.org/pdf/2606.24773)  

**Abstract**: Non-autoregressive generation offers a powerful paradigm for iterative refinement, allowing models to recursively critique, erase and regenerate arbitrary subsets of tokens. However, existing non-autoregressive models fail to realize this potential. Masked Diffusion Models (MDMs) suffer from factorization error, causing sample quality to collapse when generating multiple tokens simultaneously. Flow Map Language Models (FMLMs) circumvent this bottleneck via joint sequence transport for excellent few-step generation, but sacrifice the inference-time flexibility of MDMs. We introduce FMLM+, a framework that bridges this gap by equipping FMLM with masking-style noise schedules. While generating the full sequence in a single step, FMLM+ simultaneously scores the global consistency of each token a posteriori. We leverage this to introduce Posterior Refinement, a novel inference-time refinement strategy that enables the model to adaptively self-correct its outputs, matching the performance of discrete baselines with 32x fewer NFEs. Across diverse benchmarks, we demonstrate that FMLM+ with Posterior Refinement improves the speed--quality tradeoff over both MDM and FMLM families, providing a scalable foundation for high-fidelity language modeling. 

---
# CANDLE: Character-level Arabic Noise Deduplication using Lightweight Encoder 

**Authors**: Faris Alasmary, Taif Nono, Orjuwan Zaafarani, Kholood Al Tabash, Ahmad Ghannam, Anas Salamah, Shouq Sadah, Lahouari Ghouti  

**Link**: [PDF](https://arxiv.org/pdf/2606.24758)  

**Abstract**: Handling repeated characters in text can be tricky, since they can represent either the correct spelling of a word or informal character elongation often seen in social media posts. We present CANDLE, a lightweight system for character-level Arabic noise deduplication that addresses this challenge without relying on handcrafted rules, dictionaries, or morphological analyzers. At the heart of CANDLE is a novel application of Connectionist Temporal Classification (CTC) to this task, a formulation not previously explored for character deduplication, which frames normalization as a sequence alignment problem over a character-based encoder. Evaluated on three benchmarks spanning clean newspaper, manually curated ambiguous cases, and real-world social media text, the CTC model achieves a Sentence Error Rate (SER) as low as $5.37\%$ and consistently outperforms a classification-based baseline by a large margin. To reduce inference overhead, we distill the 6-layer CTC model into a 2-layer student, achieving a $3\times$ depth reduction with minimal performance degradation. Beyond deduplication accuracy, normalization yields a practical downstream benefit: a relative reduction in tokenizer fertility of up to $12.8\%$ across a diverse set of Arabic LLM tokenizers, directly lowering inference costs and improving context window utilization. We release all code and models publicly to support reproducibility and advance future research\footnote{this https URL}. 

---
# Task Decomposition for Efficient Annotation 

**Authors**: Nupoor Gandhi, Emma Strubell  

**Link**: [PDF](https://arxiv.org/pdf/2606.24734)  

**Abstract**: High-quality annotations of structured representations are expensive to collect over large corpora. Manual annotation of structure is laborious, and model-based annotation, although cheaper to generate, requires expensive validation and potentially significant supervision to ensure that the annotation quality is strong enough to be useful downstream. In traditional annotation workflows, annotation of each complete example is performed end-to-end by a single annotator. However, structured annotation is complex, and each aspect of the task represents a unique challenge with an associated inferential load for a given annotator. Modern annotation projects can incorporate heterogeneous groups of annotators, including both models and human annotators with varying domain and linguistic expertise. It remains unclear, however, how to redesign annotation tasks in this setting, where efforts are discriminately allocated across heterogeneous annotators with respect to distinct annotation challenges.
We propose to decompose annotation tasks into sub-tasks in order to reduce the aggregate inferential load of annotation projects. Inspired by the notion of centers from centering theory, we introduce a formal model of inferential load based on the degrees of freedom in the space of valid annotations. Using this model, we show that identifying these centers (i.e. salient anchor entities realized by annotation sub-tasks) constrains the output space complexity, and decompositions which isolate and advance center identification reduce the aggregate inferential load. We provide guidelines for decomposing complex structured annotation tasks, supported by examples demonstrating improved cost-efficiency from our prior work. Finally, we present a procedure for allocating sub-tasks across annotators to maximize quality under a fixed budget. 

---
# CN-NewsTTS Bench: a target-level automatic benchmark for raw-input Chinese news TTS pronunciation 

**Authors**: Shijun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.24714)  

**Abstract**: Chinese news text contains dense written forms such as scores, hyphenated model names, ranges, unit symbols, percentages, English abbreviations, and mixed Chinese-Latin-digit names. These forms are frequent in real listening workflows, and a text-to-speech (TTS) system can preserve the written string while changing the spoken meaning. We introduce CN-NewsTTS Bench v0.1, an open target-level benchmark for evaluating whether Chinese news TTS products pronounce such targets correctly from raw text, without user-side rules, LLM rewriting, SSML hints, or manual edits. The release contains a 200-record development set, an 800-record public test set, 992 public auto-evaluable targets, fixed transcripts from a three-ASR ensemble, an automatic target scorer, and initial results for seven product TTS systems. We additionally report ASR-route diagnostics, ASR-subset ablations, category-level results, confidence intervals, and provider configuration metadata. The best system reaches 0.879 strict accuracy, while several systems remain below 0.60. 

---
# DREAM: Dense Retrieval Embeddings via Autoregressive Modeling 

**Authors**: Yixuan Tang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24667)  

**Abstract**: Dense retrieval embedding models are a fundamental component of modern retrieval-based AI systems. Most dense retrievers are trained with contrastive objectives, which require labeled positive and negative document pairs that are often costly and difficult to obtain. In this work, we investigate whether the autoregressive next-token prediction objective of a large language model (LLM) can provide supervision for dense retrieval. The intuition is simple: if a document contains information relevant to a query, conditioning on that document should make the target output easier for the LLM to predict. A key challenge is that the next-token prediction loss is computed inside the LLM, while the retriever is a separate embedding model. To address this challenge, we propose DREAM (Dense Retrieval Embeddings via Autoregressive Modeling), which injects retriever-generated query-document similarity scores into selected attention heads of a frozen LLM. During training, these scores determine how much attention each candidate document receives while the LLM predicts the target output. The resulting prediction loss provides gradients for retriever training through the attention mechanism. We evaluate DREAM on retrieval benchmarks BEIR and RTEB using embedding backbones ranging from 0.5B to 3B parameters. DREAM consistently outperforms existing baselines across different model scales. These results demonstrate that DREAM provides a promising approach for training dense retrievers through autoregressive modeling. 

---
# AI-PAVE-Br: Leveraging Large Language Models for Enhanced Product Attribute Value Extraction through a Golden Set Approach 

**Authors**: Murilo Gazzola, Hugo Gobato Souto, Samuel Silva, Júlia Schubert Peixoto, Felipe Siqueira, André Luis Pedroso de Morais, Caio Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2606.24655)  

**Abstract**: The explosive growth and complexity of product data within the dynamic Brazilian e-commerce landscape demand robust and specialized methods for structured information extraction. Traditional approaches to Product Attribute Value Extraction (PAVE) often struggle with the linguistic nuances and sheer diversity of product descriptions in Portuguese. To address this critical gap, this paper introduces two major contributions. First, we present AI-PAVEBr, a specialized system engineered with Large Language Models (LLMs) to perform high-accuracy PAVE specifically for Brazilian e-commerce catalogs. Second, to facilitate reproducible research and provide a definitive benchmark, we introduce and share the Golden Set, a new, meticulously curated, and manually annotated dataset for PAVE in Portuguese. We detail the creation process and structure (Entity, Category, Subcategories) of this high-quality reference set. Our experiments conclusively show that AI-PAVE-Br, leveraging targeted prompt engineering, dramatically outperforms conventional Named Entity Recognition (NER) baselines. This work not only delivers a superior, scalable solution for a major non-English market but also enriches the NLP community with a valuable, publicly available resource for future PAVE research. 

---
# Harmonic: Hierarchical State Space Models for Efficient Long-Context Language Modeling 

**Authors**: Petr Nyoma  

**Link**: [PDF](https://arxiv.org/pdf/2606.24650)  

**Abstract**: We present Harmonic, a hierarchical state space model (SSM) for language modeling. The architecture stacks three recurrent levels at progressively slower timescales; each level receives the prediction error of the level below as input, rather than its raw hidden state. On enwiki8 with equal token budgets, Harmonic outperforms a comparable Transformer (28M params) by +1.4% at 1K tokens, +6.7% at 8K tokens, and +11.4% at 32K tokens (bpt, lower is better). It also outperforms Mamba at every tested length by 0.7--1.8%. At 64K tokens, both Mamba and Transformer run out of memory on an 80GB H100; Harmonic trains successfully, reaching 6.169 bpt. Results replicate on WikiText-103 (H-TF gap +1.7% to +7.2% across 1K--32K). At 1B parameter scale, replacing all attention layers in TinyLlama 1.1B with HarmonicBlock eliminates the RoPE positional encoding limit: the resulting Hallamonic model maintains stable loss across sequence lengths 1K--8K on two independent clean benchmarks (Lambada and fineweb-edu held-out), while TinyLlama degrades catastrophically past its 2K-token RoPE limit (gap: +9.4 bpt at seq=8K on Lambada). Compute is O(L) per forward pass vs. O(L^2) for attention.
Logs: this https URL. 

---
# Measuring User's Mental Models of Speech Translation in Human-AI Collaboration 

**Authors**: HyoJung Han, Nishant Balepur, Jordan Boyd-Graber, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2606.24644)  

**Abstract**: Millions of people use machine translation (MT) tools daily, yet little is known about their perception of what systems can and cannot do. This paper studies users' mental models of speech translation systems through a new framework based on cross-lingual question answering, where users either accept MT output or request professional re-translation to answer questions based on the information presented in a foreign language. By analyzing user behavior and accuracy trends across varying translation qualities, we examine to what extent they can predict where the system is likely to be wrong, and how this mental model evolves. Users develop stronger mental models with practice, especially when they have some knowledge of the source language, primarily by relying on surface-level error cues. Moreover, providing speech transcriptions can help users develop better mental models. Our results show the promise of cross-lingual question answering as a downstream task for studying MT mental models and advancing our understanding of human-AI collaboration. 

---
# The Warrant Gap: Claim-Conditioned Re-scoring for Fact-Checking 

**Authors**: Arka Ujjal Dey, John Collomosse  

**Link**: [PDF](https://arxiv.org/pdf/2606.24627)  

**Abstract**: Fact-checking systems built on LLMs achieve high verdict accuracy on standard benchmarks, yet routinely output Supports labels whose cited evidence does not license the claim. Structured decomposition is the natural way to inspect those warrants, but rigid extraction protocols strip the full-claim context that facets need. We introduce SIFT -- claim-conditioned re-scoring of extracted evidence spans against the full claim -- paired with WSP (Warranted Supports Proportion), an automatic NLI check that the cited warrant entails the claim. We evaluate on FEVER, SciFact, 5PILS, and DP across four open-source backbones. SIFT recovers accuracy on cells where naive decomposition costs up to 27.6 points, while raising WSP above direct prompting; WSP itself calibrates against human gold evidence at AUC 0.92 and precision 0.98. 

---
# Privacy-Preserving RAG via Multi-Agent Semantic Rewriting: Achieving Confidentiality Without Compromising Contextual Fidelity 

**Authors**: Yuanhe Zhao, Tianyu Zhang, Huafei Xing, Derek F. Wong, Jianbin Li, Tao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24623)  

**Abstract**: Retrieval-Augmented Generation enhances large language models by incorporating external knowledge, but deploying it in sensitive scenarios risks privacy leakage via malicious prompts. To address this, we propose a multi-agent framework that sanitizes retrieved content through semantic rewriting. By employing three specialized agents for privacy extraction, semantic analysis, and reconstruction, our approach collaboratively removes sensitive identifiers while preserving the semantic core. We evaluate the framework on the ChatDoctor and Wiki-PII datasets across six large language models. Experimental results demonstrate a significant reduction in privacy leakage under targeted attacks. For instance, we reduced targeted information exposure in LLaMA-3-8B from 144 instances in the baseline to just 1. Furthermore, we maintain strong contextual fidelity with a BLEU-1 score of 0.122, outperforming the existing SAGE method's 0.117. Finally, the framework operates as an asynchronous preprocessing module, introducing no additional latency to online inference, as all rewriting is executed as a one-time offline preprocessing step. To promote reproducibility, the source code of this work is publicly available at this https URL. 

---
# Same Lesson, Different Story: Cross-Lingual Reconstruction of Cultural Narratives in Large Language Models 

**Authors**: Jory Alshaalan, Haya Albaker, Abeer Aldayel, Aljawharah Alabdullatif, Rehab Alahmadi  

**Link**: [PDF](https://arxiv.org/pdf/2606.24610)  

**Abstract**: The evaluation of cultural grounding context becomes complex when multiple cultures convey the same moral lesson. This challenge is particularly relevant to large language models (LLMs), which produce narratives across a wide range of languages and cultural contexts. However, it remains uncertain whether these models preserve culturally grounded meaning when equivalent moral lessons are conveyed through distinct cultural forms. This study introduces a multilingual evaluation narrative framework that integrates a cross-linguistic collection of 414 proverbs spanning 15 languages and uses four LLMs to generate 13k narratives. By employing semantically equivalent proverbs as culturally grounded prompts, the analysis assesses whether models preserve meaning across languages, how cross-lingual conditioning influences narrative realization, and whether different model families converge on similar interpretations. Results indicate that cross-lingual prompting largely preserves proverb-level semantic meaning while systematically redistributing agency, social positioning, and narrative structure. Additionally, strong inter-model convergence is observed in both monolingual and cross-lingual settings, suggesting that multilingual LLMs rely on shared semantic abstractions despite architectural and linguistic differences. These findings shed light on the need for more comprehensive evaluations of cultural grounding. Relying exclusively on semantic similarity in multilingual narrative assessments may overestimate cultural preservation by neglecting culturally meaningful variations in narrative expression. 

---
# Qwen-AgentWorld: Language World Models for General Agents 

**Authors**: Yuxin Zuo, Zikai Xiao, Li Sheng, Fei Huang, Jianhong Tu, Yuxuan Liu, Tianyi Tang, Xiaomeng Hu, Yang Su, Qingfeng Lan, Yantao Liu, Qin Zhu, Yinger Zhang, Bowen Yu, Haiquan Zhao, Haiyang Xu, Jianxin Yang, Jiayang Cheng, Junyang Wang, Lianghao Deng, Mingfeng Xue, Tianyi Bai, Yang Fan, Yubo Ma, Yucheng Li, Zeyu Cui, Zhihai Wang, Zhihui Xie, Zhuorui Ye, An Yang, Dayiheng Liu, Jingren Zhou, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.24597)  

**Abstract**: A world model predicts environment dynamics based on current observations and actions, serving as a core cognitive mechanism for reasoning and planning. In this work, we investigate how world modeling based on language models can further push the boundaries of general agents. (i) We first focus on building foundation models for agentic environment simulation. We introduce Qwen-AgentWorld-35B-A3B and Qwen-AgentWorld-397B-A17B, the first language world models capable of simulating agentic environments covering 7 domains via long chain-of-thought reasoning. Leveraging more than 10M environment interaction trajectories of 7 domains in real-world environments, we develop Qwen-AgentWorld through a three-stage training pipeline: CPT injects general-purpose world modeling capabilities from the state transition dynamics and augmented professional corpora, SFT activates next-state-prediction reasoning, and RL sharpens simulation fidelity through a tailored framework with hybrid rubric-and-rule rewards. To evaluate language world models, we present AgentWorldBench, a comprehensive benchmark constructed from real-world interactions of 5 frontier models on 9 established benchmarks. Empirical results demonstrate that Qwen-AgentWorld significantly outperforms existing frontier models. (ii) Beyond foundation models, we further investigate two complementary paradigms through which world modeling enhances general agents. First, as a decoupled environment simulator, Qwen-AgentWorld supports scalable and controllable simulation of thousands of real-world environments for agentic RL, yielding gains that surpass real-environment training alone. Second, as a unified agent foundation model, world-model training acts as a highly effective warm-up that improves downstream performance across 7 agentic benchmarks. Code: this https URL 

---
# To Compare, or Not to Compare: On Methodological Practices in Evaluating Social Bias 

**Authors**: Federico Marcuzzi, Xuefei Ning, Roy Schwartz, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2606.24596)  

**Abstract**: As Large Language Models are increasingly deployed in critical applications, robustly evaluating their social biases is paramount. However, the current literature suffers from widespread methodological fragmentation, which yields contradictory conclusions. This stems largely from ignoring the structural framing of benchmark-level evaluations. To resolve this, we introduce a unified and controllable framework that standardizes heterogeneous benchmarks to systematically contrast isolated demographic assessments with forced-choice comparative settings. Crucially, this allows us to disentangle the confounding effects of Chain-of-Thought reasoning, neutral fallback options, and other structural artifacts in social bias evaluations. Our evaluation across multiple model families reveals a massive, systematic paradigm gap: while isolated assessments limit prejudice activation, comparative settings act as aggressive catalysts for latent discrimination, a shift primarily driven by underspecified contexts. Alarmingly, CoT reasoning exacerbates social biases under comparative settings, and this systemic bias persists as a deterministic prejudice even when models are provided neutral fallback options or claim to answer randomly. Finally, we demonstrate that this comparative prejudice is a generalized phenomenon that scales positively with model size. Ultimately, we offer a crucial methodological guideline: while researchers must leverage comparative settings to robustly audit hidden biases, practitioners cannot safely rely on comparative deployments in ambiguous real-world tasks. 

---
# MEMPROBE: Probing Long-Term Agent Memory via Hidden User-State Recovery 

**Authors**: Enze Ma, Yufan Zhou, Wei-Chieh Huang, Jie Yang, Huanhuan Ma, Zixuan Wang, Chengze Li, Chunyu Miao, Philip S. Yu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24595)  

**Abstract**: Long-term memory promises LLM agents that grow more capable across sessions, maintaining an accurate, evolving understanding of the user that interaction forms. In practice, however, this memory is evaluated mostly through downstream behavior, such as later answers, personalization quality, or task success, which tests that understanding only indirectly and leaves the memory artifact itself largely unaudited. We argue that long-term memory should instead be evaluated as an auditable post-interaction artifact: after ordinary assistance, what structured user state can be reconstructed from the memory the agent leaves behind? We instantiate this view in MEMPROBE, a benchmark in which a memory-equipped agent assists simulated users, each carrying a hidden, taxonomy-anchored user-state bank, across a trajectory of leak-controlled tasks, after which that bank is reconstructed from the agent's resulting memory under both full-store and top-k access. Built on synthetic ground truth for efficient, scalable measurement, MEMPROBE spans 50 simulated users with 31 hidden dimensions each (1,550 recovery targets) and tests 5 representative memory systems. Testing state-of-the-art memory agents, we find that successful assistance and recoverable memory behave as distinct capabilities. Task completion nearly saturates, even for a memoryless baseline, while category-balanced recovery stays moderate (about 0.6) and drops further under top-k retrieval. MEMPROBE is the first benchmark to study memory recovery directly, reconstructing the user state a system retains and scoring it against ground truth. We see recovery as a concrete objective for future memory agents to optimize, and MEMPROBE as a step toward an environment where agents are trained to remember their users, growing more faithful the longer they know them. 

---
# Cross-Lingual Exploration for Parametric Knowledge 

**Authors**: Elisha Diskind, Itamar Trainin, Uri Shaham, Leshem Choshen, Idan Szpektor, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2606.24579)  

**Abstract**: Parametric knowledge in Large Language Models is not equally accessible across languages. As a result, standard inference techniques often struggle to surface localized facts, leading to failures in cross-lingual knowledge transfer and consistency. In this work, we investigate techniques for accessing hidden factual knowledge by exploring cross-lingual prompting strategies. We identify four inherent dimensions of cross-lingual exploration that directly govern parametric knowledge retrieval and evaluate them on multilingual factual benchmarks covering 17 typologically diverse languages. Our results demonstrate that cross-lingual exploration significantly improves knowledge transfer and factual recall, representing a more efficient compute Pareto frontier than native-language scaling. Furthermore, we observe corresponding improvements in cross-lingual consistency, exceeding what can be explained by accuracy gains alone. Overall, our work establishes multilingual prompt exploration as a highly effective inference-time strategy for unlocking latent parametric knowledge. 

---
# NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers? 

**Authors**: Yuru Wang, Lejun Cheng, Yuxin Zuo, Sihang Zeng, Bingxiang He, Che Jiang, Junlin Yang, Yuchong Wang, Kaikai Zhao, Weifeng Huang, Kai Tian, Zhenzhao Yuan, Jincheng Zhong, Weizhi Wang, Ning Ding, Bowen Zhou, Kaiyan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24530)  

**Abstract**: We introduce NatureBench, a cross-discipline benchmark of 90 tasks distilled from peer-reviewed Nature-family publications, designed to evaluate whether AI coding agents can move beyond reproduction toward discovery on real scientific problems. NatureBench is built on NatureGym, an automated pipeline that constructs a standardized, per-task containerized environment from a source paper, addressing the environment-fragmentation problem that has limited the credibility of prior agent-on-research benchmarks. Evaluating ten frontier agent configurations under a strict web-search-disabled protocol, we find that the strongest model surpasses SOTA on only 17.8% of tasks under the g>0.1 criterion. Analysis of method pathways reveals that agents succeed primarily through methodological translation, converting scientific tasks into familiar supervised prediction problems, rather than through genuine scientific invention. Failures are dominated by wrong method choice and insufficient compute budget, not by task misunderstanding. We release the benchmark, the NatureGym pipeline, and a public leaderboard with maintainer-side reproduction. Code: this https URL 

---
# AGORA: An Archive-Grounded Benchmark for Agentic Workplace Document Reasoning 

**Authors**: Honglin Guo, Qi Zhang, Yu Zhang, Weijie Li, Rui Zheng, Zhikai Lei, Qiyuan Peng, Zhiheng Xi, Tao Gui, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24526)  

**Abstract**: Large language models are increasingly deployed as agents that reason over documents rather than answer from parametric knowledge. We study archive-grounded reasoning: locating sparse evidence across a large, messy collection of workplace files, reconciling inconsistent terminology, units, and time conventions, and computing an answer. Existing benchmarks address only parts of this setting and none jointly stresses archive-groundedness, agentic exploration, and cross-domain coverage. We introduce Agora, a benchmark pairing 362 questions with eight domain collections of 9,664 authentic documents and 372M tokens, far exceeding any model's context window, so agents must explore deliberately rather than scan exhaustively. Agora is built by an agentic pipeline combining cross-document task synthesis, leakage-preventing obfuscation, and difficulty filtering. Evaluating eight models, we find the task far from solved: even the strongest reaches only 59.4% accuracy, with notable variation across domains. 

---
# Poster: Exploring the Limits of Audio-Based Detection of Turkish Phone Call Scams 

**Authors**: Arda Eren, Micheal Cheung, Youqian Zhang, Grace Ngai, Eugene Yujun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2606.24523)  

**Abstract**: Scam phone calls exploit vulnerable communities worldwide, yet research on detection has focused almost exclusively on English and other high-resource languages. In low-resource settings such as Turkish, detection is especially difficult, as annotated data is scarce and technological defenses remain limited. This research investigates how large language models (LLMs) can support scam detection in Turkish by introducing the first public multi-modal dataset of 100 aligned audio-transcript pairs of scam and benign conversations. We evaluate seven LLMs spanning three model families: Gemini 2.5 (Flash, Flash-Lite, Pro), GPT-4o, and Qwen (Max, Plus, Turbo), under three input conditions: raw audio, automatic speech-to-text transcripts, and transcripts refined by a native speaker. Our results suggest that transcript-based inputs consistently outperform direct audio processing, while human-corrected and uncorrected transcripts perform comparably. By centering a low-resource language and real world threat, this work highlights the urgent need for culturally and linguistically inclusive AI safety research and more robust multi-modal systems for fraud prevention. 

---
# UOL@IDEM at BEA 2026 Shared Task 1: Neural Fusion and Feature-Rich Modeling for L1-Aware Vocabulary Difficulty Prediction 

**Authors**: Nouran Khallaf, Serge Sharoff  

**Link**: [PDF](https://arxiv.org/pdf/2606.24501)  

**Abstract**: This paper describes UOL@IDEM's closed-track submission to the BEA 2026 shared task on L1-aware vocabulary difficulty prediction. We model the task as regression and train separate systems for Spanish, German, and Mandarin Chinese\footnote{Below we use \emph{Chinese} for brevity.}. Our system combines multilingual contextual representations with engineered features capturing frequency, surface form, retrieval evidence, semantic alignment, cognate similarity, and masked-language-model predictability. Development results show consistent gains over the official closed-track baselines, with sentence-embedding encoders such as BGE-M3, multilingual E5, and LaBSE performing best. Official submissions achieve RMSE scores of 1.132, 1.037, and 0.891 for Spanish, German, and Chinese, respectively. Feature analysis identifies frequency as the most stable predictor, while contextual predictability, form similarity, retrieval, and semantic features provide complementary L1-sensitive signals. Error analysis shows strong ranking performance but weaker calibration for the easiest items, which are often overpredicted. See this https URL 

---
# The African Language Tax: Quantifying the Cost, Latency, and Context Penalty of Tokenizing African Languages in Frontier LLMs 

**Authors**: Olaoye Anthony Somide  

**Link**: [PDF](https://arxiv.org/pdf/2606.24460)  

**Abstract**: Commercial large language models bill, scale latency, and budget context per token. Yet tokenizers assign more subword tokens to the same meaning in some languages than in others, so speakers of languages with high token-fertility pay a structural penalty before a model is ever invoked. This penalty is documented for multilingual settings in general, but it has not been measured systematically for African languages at the level of enterprise deployment economics and cognitive context capacity. We measure it across 20 African languages spanning five language families and three scripts (Latin, Ge'ez/Ethiopic, N'Ko; 19 appear in the primary FLORES-200+ corpus, with Nigerian Pidgin measured via MAFAND-MT only), using parallel corpora so that the language effect is isolated from content. Across 11 frontier and open tokenizers on FLORES-200+, every African language carries a tokenization premium above English (median 1.88x on GPT-5 / o200k_base, up to 8.92x for N'Ko); the penalty is largest for Ethiopic and N'Ko scripts (reaching 7-9x) and is near-invariant across corpora (FLORES vs SIB-200 Pearson r = 0.9998). Translated into deployment terms, this results in up to 8.9x inference cost and an equivalent generation-latency multiplier (N'Ko vs English on GPT-5; 7.4x for Amharic), and as little as 11% of English's effective context window. The best currently available tokenizer for African languages, Gemma 4, reduces the mean premium from 3.31x (cl100k_base) to 2.38x, but no tokenizer eliminates the penalty. We release an open measurement tool (afri-fertility), a public leaderboard, a results dataset, and mitigation guidance for African builders. The penalty falls hardest on the languages whose speakers can least afford it, a digital divide encoded directly into the subword vocabulary. 

---
# Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning 

**Authors**: Shiding Zhu, Yudi Qi, Yajie Wang, Jiaze Li, Chao Song, Yaorui Shi, Yibo Miao, Hanqi Gao, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24428)  

**Abstract**: Experience-driven self-evolution is critical for large language model (LLM) agents to improve through open-world interaction. However, existing experience learning methods mostly rely on single-agent loops, where the same agent executes tasks, summarizes outcomes, and determines memory content. This setup makes agents vulnerable to the Self-Confirmation Trap: wrong-but-self-consistent trajectories are misidentified as successful experience, leading to cumulative errors during retrieval and reuse. To address this issue, we propose EDV, an Execute-Distill-Verify framework for reliable experience learning. In the Execute stage, multiple heterogeneous agents explore the same task space in parallel to generate diverse candidate trajectories. In the Distill stage, a dedicated third-party agent comparatively analyzes these trajectories to produce candidate experiences, reducing executor-centric summarization bias. In the Verify stage, the execution group validates candidates via a consensus mechanism, and only approved experiences are written into shared or private memory. By decoupling the three stages, EDV transforms experience learning from isolated self-reflection into collaborative construction, filtering erroneous and noisy content before memory insertion. We evaluate EDV on three challenging long-horizon benchmarks: tau2-bench, Mind2Web and MMTB. Results show EDV consistently outperforms strong baselines, validating that reliable experience construction is essential for robust agent self-evolution. Our code is available at this https URL. 

---
# Beyond Logprobs: A Multi-Signal Confidence Engine for LLM-Based Document Field Extraction 

**Authors**: Nitesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.24420)  

**Abstract**: In high-stakes document processing pipelines, including financial reconciliation, compliance verification, and procurement automation, an LLM extraction that is silently wrong is more dangerous than one that is visibly absent. The central challenge is not extraction accuracy alone but reliable confidence estimation: knowing, field by field, whether an extraction can be trusted for automation or deferred to human review. Token-level log-probabilities, verbalized confidence, and multi-sample self-consistency all collapse toward all-positive behaviour at practical thresholds, offering no reliable separation between trustworthy and untrustworthy extractions.
We present ExtractConf, a cross-domain, field-agnostic confidence engine that grounds confidence estimation in two structurally different readings of the same document. A field-guided Hunter call extracts each field under schema-slot completion pressure; a document-guided Mapper call scans holistically and surfaces values grounded in document content. This asymmetry yields different failure modes: Hunter hallucinates values for absent fields, while Mapper misses visually non-salient ones. Their disagreement is independently informative. ExtractConf fuses cross-call disagreement, LLM-internal uncertainty, OCR, image quality, and spatial layout into a classifier requiring no domain-specific rules or retraining. On DocILE (55-field invoices, 26% failure rate), it achieves 0.928 ROC AUC and reduces selective prediction risk by 70% over logprob-mean. At 80% coverage, accuracy reaches 99.1%, enabling a practical human-in-the-loop workflow. Zero-shot transfer to CORD receipts achieves 0.858 AUC; lightweight Lasso recalibration reduces ECE by 89% and Brier by 43%, confirming the signals generalise across document domains. 

---
# AutoSpecNER: A Fine-Grained Named Entity Recognition Dataset for Vehicle Specification Extraction 

**Authors**: Jordan Lee, Filippos Ventirozos, Abdirahman Abdullahm, Ioanna Nteka, Peter Appleby, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2606.24387)  

**Abstract**: Vehicle advertisements contain rich specification information, but automotive NER resources remain limited. We introduce AutoSpecNER, an expert-annotated dataset for fine-grained entity recognition in vehicle listings. The dataset includes 659 advertisements from a popular car-selling website, with over 10,000 entities annotated across 15 categories, including MODEL, ENGINE_SPEC, and BATTERY_CAPACITY. Annotation quality was validated through inter-annotator agreement, achieving an average score of 91.5%. We benchmark rule-based extraction, fine-tuned transformer encoders, and large language models. DeBERTa achieves the best performance with a 90% micro-F1 score, outperforming the rule-based baseline (43%) and the strongest large language model (77.8%). 

---
# On the Stability of Prompt Ranking in Large Language Model Evaluation 

**Authors**: Shaoshuai Du, Penghao Liang, Yixian Shen, Chuanqi Shi, Hang Zhang, Lun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24381)  

**Abstract**: Prompt-based interaction has become a dominant paradigm for using large language models (LLMs), where multiple candidate prompts are evaluated and the top-ranked one is selected for downstream use. This workflow implicitly assumes that prompt rankings are stable under minor variations in evaluation conditions. In this paper, we systematically study prompt ranking stability under common sources of variability, including random seeds and limited evaluation subsets. Across three open-weight LLMs and two benchmark tasks, we find that while overall rank correlations are often moderate to high, the identity of the top-performing prompt frequently changes, leading to unreliable selection decisions. To address this issue, we propose a simple stability-aware selection strategy based on a lower confidence bound, which accounts for both performance and variance. Our results show that this approach improves robustness in unstable settings while remaining competitive in more stable regimes. These findings highlight the importance of accounting for evaluation uncertainty in prompt selection and LLM benchmarking. 

---
# MorfFlex: Handling Rich Morphology 

**Authors**: Jaroslava Hlaváčová, Marie Mikulová, Barbora Štěpánková, Milan Straka, Jan Hajič  

**Link**: [PDF](https://arxiv.org/pdf/2606.24366)  

**Abstract**: We present MorfFlex, a morphological dictionary architecture suitable for languages with extensive regularity in both inflection and derivation. As the primary example of MorfFlex in use we introduce MorfFlex CZ, a morphological dictionary of Czech. It is distributed as a simple, unstructured list of <wordform, lemma, tag> triplets, however, its manually maintained, unpublished source files and conversion scripts encode a sophisticated system of inflectional and derivational patterns. These patterns dramatically reduce the otherwise enormous size of the dictionary, which currently contains over 100 million wordforms and more than 1 million lemmas. The MorfFlex CZ dictionary serves as an essential resource for ensuring the consistency of manual morphological annotation in the Prague Dependency Treebanks and underpins state-of-the-art automatic tools such as MorphoDiTa. In this paper, we focus on: (i) presenting an effective method for managing the rich morphological system within the dictionary, and (ii) demonstrating the utility of such a language resource for maintaining annotation consistency in corpora and supporting the development of advanced NLP applications. 

---
# Automatic Part-of-Speech Tagging of Arabic-English Dictionary Senses through WordNet 

**Authors**: Diaa M. Fayed, Aly A. Fahmy, Mohsen A. Rashwan, Wafaa K. Fayed  

**Link**: [PDF](https://arxiv.org/pdf/2606.24359)  

**Abstract**: This paper proposed an algorithm for part-of-speech (POS) tagging senses of a bilingual dictionary. The algorithm is applied on the Al-Mawrid Arabic-English dictionary. The tagging task is accomplished by transferring the POS tags of the English translation equivalences (TEs) to the dictionary senses after dis-ambiguities process. The English POS tags of senses are acquired from the Princeton WordNet. POS tagging of bilingual dictionary senses is prerequisite to link a bilingual dictionary to WordNet and/or standardizing that dictionary into WordNet-LMF format where the synset (set of synonyms), not word, is the basic brick. The registered accuracy is high though the cost is little. Building NLP/HLT tools needs linguistic experts, large investments, and long time. For statistical approach, we need large annotated corpora and for rule-based approach, we need large lexicon that contains rich linguistic and world knowledge. That motivates the appearance of what are called resource-light approaches to develop natural language processing (NLP) tools for poor-resource languages. 

---
# Meet UD_Czech-PDTC: A Large and Genre-Rich Treebank in Universal Dependencies 

**Authors**: Marie Mikulová, Barbora Štěpánková, Daniel Zeman, Jan Štěpánek, Milan Straka, Jan Hajič  

**Link**: [PDF](https://arxiv.org/pdf/2606.24337)  

**Abstract**: Czech has been part of Universal Dependencies since its first release in 2015. It has also been one of the best represented languages, with the Prague Dependency Treebank being order of magnitude larger than most other UD treebanks. More recently, three other datasets from the Prague family were added and the annotations thoroughly revisited, forming the "Prague Dependency Treebank-Consolidated" (PDT-C). In comparison to the original PDT, PDT-C is more than twice as large, but it is also much more diverse in terms of genres and domains. In this paper, we describe the conversion of the new resource to Universal Dependencies. While the two annotation schemes are relatively similar at the first sight, there are numerous small differences in topology of the dependency structures and in granularity of the POS and relation type inventories. We demonstrate a selection of such differences on examples, discuss the diverging motivations, as well as ways to overcome the differences during conversion. We argue that while PDT is less "universal" and more tightly bound to one language, its multi-layer annotation is rich and provides all information needed for basic UD trees, and much more. 

---
# Transformer-Based Language Models Across Domain Verticals: Architectures, Applications and Critical Assessment 

**Authors**: Guruprakash J, Krithika L.B  

**Link**: [PDF](https://arxiv.org/pdf/2606.24331)  

**Abstract**: Transformer-based language models have become the default substrate for natural language processing and the pace of new releases has made it hard for practitioners to separate durable ideas from the noise of incremental announcements. This review works at two levels. At the level of mechanism, we organise the main transformer families into a working taxonomy, covering encoder-only, decoder-only, encoder-decoder, long-context, permutation-based, and generator-discriminator variants. We then extend the discussion to post-2023 developments that changed the picture in practice: instruction tuning, reinforcement learning from human feedback, direct preference optimisation, mixture-of-experts scaling, retrieval augmentation and the current flagship model families from OpenAI, Anthropic, Google, Meta, Mistral and DeepSeek. At the level of use, we survey deployments across healthcare, finance, legal, education, customer service, creative writing and scientific work. Based on this we link each to the specific capabilities that make a transformer the appropriate tool. The contribution of this paper is a critical assessment that is based on the survey. We compare architectures on four axes that matter to deployment decisions, we quantify the trade-off between parameter count and energy cost. We also discuss how alignment methods, data provenance and benchmark saturation change what it means to call a model "state of the art". The final section lists the research questions that we think deserve more attention. 

---
# Prague Dependency Treebank -- Consolidated 2.0: Enriching a Complex Annotation Scheme 

**Authors**: Marie Mikulová, Jiří Mírovský, Milan Straka, Pavlína Synková, Jan Štěpánek, Barbora Štěpánková, Jan Hajič  

**Link**: [PDF](https://arxiv.org/pdf/2606.24324)  

**Abstract**: The Prague Dependency Treebank framework is unique in its attempt to systematically include and link different layers of language, including a meaning representation with several types of inter-sentential phenomena, especially coreference and discourse relations. We present its second consolidated version (PDT-C 2.0), which concludes almost 30-years long project of sustained development of the resource to a uniformly and coherently annotated, genre-diversified, almost 4 million token language resource of Czech language, with accompanying fully compatible lexicons. In addition to continuous linguistic research, the richly linguistically annotated corpus is also widely used in international comparisons of the development of traditional and novel NLP tools as well as in conversions into other formalisms. The corpus and the trained parsers are available under the CC BY-NC-SA licence. 

---
# AVOC: Enhancing Hour-Level Audio-Video Understanding in Omni-Modal LLMs via Retrieval-Inspired Token Compression 

**Authors**: Yijing Chen, Wenhui Tan, Xiaoyi Yu, Yuyue Wang, Xin Cheng, Kaisi Guan, Hao Jiang, Xiangyang Li, Guojie Zhu, Ruihua Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.24286)  

**Abstract**: Multimodal Large Language Models have achieved remarkable progress in short-form audio-video understanding, yet long-form audio-video comprehension remains challenged by limited context windows and severe information redundancy. To address these bottlenecks, we propose AVOC, a framework for long-form audio-video understanding in Omni-modal Large Language Models. AVOC introduces a learnable token compression module between the modality encoders and the LLM backbone. We reframe multimodal token compression as a top-$K$ retrieval problem: given a fixed context budget, the module must retrieve a compact subset of tokens that best supports answering the user query. We draw inspiration from three classical Information Retrieval criteria for selecting informative units from a large candidate pool: relevance, importance, and diversity. AVOC instantiates each criterion as a tailored mechanism for audio-video understanding, and integrates them into a unified retrieval-style compression pipeline. Experiments show that AVOC achieves state-of-the-art performance on long-form audio-video benchmarks, surpassing the second-best model by 4.9 and 5.5 points in average accuracy on OmniVideoBench and LVOmniBench, respectively. Moreover, AVOC maintains robust performance on Audio-Video Needle-in-a-Haystack task at durations up to one hour. 

---
# CALIBER: Calibrating Confidence Before and After Reasoning in Language Models 

**Authors**: Conor Finlay, Joshua Kurien, Saurabh Dash, Marzieh Fadaee, Beyza Ermis  

**Link**: [PDF](https://arxiv.org/pdf/2606.24281)  

**Abstract**: Reasoning language models are increasingly asked not only to answer difficult questions, but also to estimate their likelihood of success. Existing methods typically elicit confidence only once: either before thinking or after answering. We argue that confidence in reasoning models is state-dependent: before thinking, confidence should estimate the chance of the model correctly solving the prompt, while after thinking it should predict whether the realized answer is likely to be correct. This distinction determines the appropriate supervision target: prompt-level success should supervise confidence estimates made after seeing the prompt, while individual answer-level correctness should supervise confidence estimates made after answering. We introduce CALIBER (Calibration Before and After Reasoning), which elicits both estimates and supervises each with the target matched to its information state. Under this unified protocol, CALIBER reduces Expected Calibration Error (ECE) by 52.5% over the strongest single-confidence baseline on BigMathDigits for the 7B model, while achieving the best Brier score and AUROC, and remains within 2.1 points of the best accuracy. Further, on a larger 30B model, CALIBER achieves the best ECE on BigMathDigits while remaining competitive in Brier score and AUROC. Out of distribution, it achieves the best ECE and Brier score on GPQA and TriviaQA, and remains competitive on SimpleQA. Ablations further show that this position-target alignment is most beneficial under distribution shift where it consistently reduces calibration error across all out-of-distribution benchmarks. 

---
# Pigeonholing: Bad prompts hurt models to collapse and make mistakes 

**Authors**: Hyunji Nam, Keertana Chidambaram, Dorottya Demszky, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2606.24267)  

**Abstract**: While in-context learning is generally shown to be effective in Large Language Models (LLMs), bad contexts can cause performance degradation and mode collapse, a phenomenon we call "pigeonholing." **Unintentionally bad** contexts can happen without malicious jailbreaking intents: For example, a user asks the model to justify an incorrect math theorem or fails to correct the model's buggy code. Specifically, we investigate ``pigeonholing" in two scenarios: (1) when the user suggests a solution, and (2) when the conversation context includes the assistant's previous (incorrect) responses. Our experiments across 10 verifiable and open-ended tasks with 10 different models show that pigeonholing manifests in several ways: (1) repeating the incorrect answers from context (leading to 38-40% performance drop), (2) converging on a narrow set of answers in coding and text generation without exploring alternatives, and (3) flipping stance on controversial topics to align with the user or the assistant's previous claims. We find that pigeonholing worsens almost monotonically with the number of conversation turns (performance drops by additional 14+% as repeated mistakes increase from 1 to 5), and pigeonholing-induced mode collapse can happen even when the provided example is correct. As a step toward mitigation, we propose RLVR with synthetic errors which improves models by 43-60% under bad contexts compared to vanilla RLVR baselines. 

---
# SURGELLM: Rethinking Multi-Task Evaluation through Task-Aware Feature Gating with Class-Balanced Normalization 

**Authors**: Noor Islam S. Mohammad, Ulug Bayazit  

**Link**: [PDF](https://arxiv.org/pdf/2606.24259)  

**Abstract**: Fine-tuned encoders deployed across heterogeneous NLP tasks face three compounding problems: mismatched inductive biases, class-imbalance corruption of feature statistics, and no mechanism to condition attention on external lexical knowledge. We introduce \textbf{\surgellm}, a unified transformer framework that addresses each with a dedicated lightweight module: a \emph{surgical feature gate} (learned per-dimension sigmoid over curated lexical indicators and \texttt{[CLS]}; provably degenerates to identity when features are uninformative), \emph{task-conditioned prefix tokens} (quantized feature values and task identity prepended to every input), and \emph{Instance-Weighted Normalization} (IWN; removes class-prior bias from gate statistics). We prove an excess-risk bound linking gate benefit to \emph{surgical feature alignment}. Across four tasks, SST-2, multi-hop retrieval, LLM-prompt attribution, and authorship detection, covering 17,830 examples and eleven model variants over three seeds, the IWN variant achieves macro-F1 \textbf{0.940} ($+0.036$ over the strongest non-IWN baseline; $+0.130$ on authorship detection). A random-vocabulary control ($-0.028$ avg.\ F1) confirms gains are lexical, not parametric. Code, vocabularies, and a $99.5\%$-recovery auto-extraction recipe are released. 

---
# Decoherence as Defence and the Magnitude of Noise Regularisation: A Rigorous N -Qubit Theory of Stochastic Quantum Neural Networks for Adversarially Robust Network Intrusion Detection 

**Authors**: Gautier-Edouard Edouard Filardo  

**Link**: [PDF](https://arxiv.org/pdf/2606.24219)  

**Abstract**: Stochastic quantum neural networks (SQNNs) encode neuronal activations as qubits, synaptic topology as entanglement, and neural noise through a Lindblad master equation. A recent conference study applied a ring-entangled SQNN to collaborative intrusion detection and reached three conclusions: ring entanglement is \emph{essential} for non-local anomaly detection; an adversarial-resilience bound holds but is \emph{conservative}; and the depolarising channel \emph{fails} to act as a dropout-style regulariser, behaving instead as output noise. It left open whether a per-gate stochastic deactivation (``true quantum dropout'') could regularise where the depolarising channel could not, and whether the loose robustness bound could be replaced by a predictive theory. This paper resolves both and extends the framework to real data and to neutral-atom hardware. We give an $N$-qubit formulation through the stochastic master equation and its vectorised Liouvillian, and prove a \emph{decoherence-contraction theorem}: a depolarising channel of strength $\gamma$ over $L$ entangling layers contracts every weight-$w$ Pauli read-out by a factor $(1-4\gamma/3)^{wL}$ (for the weight-$1$ read-out used here, $(1-4\gamma/3)^{L}$); building on the general noise-as-defence result of Du et al., we make this quantitative and operational for intrusion detection. On the real NSL-KDD dataset under white-box FGSM and PGD attacks, a depolarising SQNN trained with the channel is, over seven seeds under strong $\ell_\infty$/$\ell_2$ attacks, significantly more robust than the noiseless circuit ($\ell_\infty$ PGD-$20$, $p=0.04$, large effect) and, critically, never suffers the catastrophic robustness collapse that the noiseless model and gradient-trained classical detectors (which fall from $95\%$ to $47\%$) do, cutting robustness variance roughly twofold; we show this robustness arises from a noise-reshaped training boundary rather than from attack-time gradient contraction. For generalisation, we derive an adaptive-penalty formula showing that per-gate dropout implements a curvature-weighted $L_2$ penalty $\tfrac{p(1-p)}{2}\sum\theta^2\partial^2_\theta L$ in weight space, maximised at $p=1/2$, whereas depolarising noise implements an output-space penalty. A $30$-seed study confirms the formula's quantitative prediction: both mechanisms reduce the train-test gap by a small but statistically significant margin ($\approx\!0.01$; $p<10^{-4}$ and $p=0.004$), are statistically indistinguishable from each other, and the effect is concentrated where overfitting is largest; increasing the dropout rate past $1/2$ does not help, as the formula predicts. The single-seed dichotomy of prior work does not survive replication. We close with a neutral-atom realisation and a feasibility-by-$N$ analysis. 

---
# MMed-Bench-IR: A Heterogeneous Benchmark for Multilingual Medical Information Retrieval 

**Authors**: Junhyeok Lee, Han Jang, Hyeonjin Goh, Kyu Sung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2606.24200)  

**Abstract**: Retrieval-augmented generation (RAG) in clinical settings increasingly requires multilingual retrieval against predominantly English evidence corpora. Multilingual medical retrieval demands three capabilities: cross-lingual alignment, concept discrimination, and evidence retrieval. However, existing benchmarks evaluate these only in isolation, leaving the interaction between biomedical expertise and multilingual coverage unmeasured. We introduce MMed-Bench-IR, a benchmark designed to disentangle these axes across 6 languages and three structurally heterogeneous tasks: (1) cross-lingual medical QA retrieval with 6,127 queries grounded in the Unified Medical Language System (UMLS), (2) concept discrimination over 4,975 confusion sets at three difficulty tiers, and (3) multilingual evidence retrieval for RAG with 2,040 quality-assured queries. The three tasks share zero concept and query overlap by design, ensuring that aggregate scores reflect genuine capability breadth. Evaluation of ten systems across six paradigm families reveals severe cross-lingual failure: biomedical encoders that score 0.818 nDCG@10 in English drop to 0.056 in Japanese, a gap that English-only benchmarks cannot detect. 

---
# Aspect-Based Sentiment Evolution and its Correlation with Review Rounds in Multi-Round Peer Reviews: A Deep Learning Approach 

**Authors**: Ruxue Hana, Haomin Zhoua, Jiangtao Zhong, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24188)  

**Abstract**: Mining sentiment information from the textual content of peer review comments offers valuable insights into the scientific evaluation process. However, previous studies are often constrained by coarse-grained analysis and the lack of differentiation across review rounds. Notably, the dynamic shifts in reviewers' focus and sentiment tendencies throughout multiple review stages remain underexplored. To address this gap, the present study investigates the distribution and evolution of aspect-level sentiments and examines their correlation with the number of review rounds. We begin by segmenting the multi-round review comments of 11,063 accepted papers from Nature Communications and identifying fine-grained review aspect clusters. A manually annotated corpus of approximately 5,000 review sentences is then constructed. Using this dataset, we train a series of deep learning-based aspect sentiment classification models. Among them, the LCF-BERT-CDM model achieves the best performance, with a Macro-F1 score of 82.65%. Subsequent statistical analysis reveals a consistent trend: as the number of review rounds increases, the proportion of positive sentiments rises, while negative sentiments decline. Correlation analysis further indicates that aspect sentiment scores are negatively associated with the total number of review rounds. Key aspects exhibiting stronger correlations include "experiments", "research significance" and "result analysis". 

---
# A Synthetic Reliability-Aware PINN Benchmark for Offshore Wind Turbine Support-Structure Monitoring with Bayesian Inverse Identification 

**Authors**: Puneet Kant, Monika Tanwar  

**Link**: [PDF](https://arxiv.org/pdf/2606.24176)  

**Abstract**: Reliable structural health monitoring (SHM) of offshore wind turbine (OWT) support structures requires fast state estimation from sparse measurements. Repeated high fidelity finite element or aeroelastic analyses are difficult to use directly in online monitoring loops, while purely data-driven surrogates can require large training sets. This paper presents Digi Turbine, a synthetic reliability-aware Physics Informed Neural Network (PINN) benchmark for OWT monopile support structure monitoring. The workflow embeds a simplified Euler Bernoulli beam equation with Winkler soil foundation in the training objective, couples it with Bayesian-prior-informed inverse identification, and adds First Order Reliability Method (FORM) screening. All validation uses synthetic configurations with analytical or finite-difference ground truth motivated by the NREL 5MW reference turbine context. 

---
# A Pāninian Foundation for Indic Language Processing 

**Authors**: Ritwik Banerjee, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2606.24172)  

**Abstract**: More than a billion people communicate in Indic languages, yet the natural language processing infrastructure serving them remains fragmented and underdeveloped. The cause is structural: the field organizes its tools and benchmarks around individual languages or small subsets of genealogical language families, building separate analyzers, parsers, and datasets for each language and starting over for the next. This overlooks a deep regularity. Through more than two millennia of convergence around Sanskrit, Indic languages came to share a morphosyntactic architecture formalized in Pānini's grammar, the Astādhyāyī. This cuts across genealogical lines, uniting languages through a common framework. We argue that this Pāninian framework supplies a unifying computational architecture the field has lacked, and that benchmarks grounded explicitly in it would make Indic language systems more accurate, more data-efficient, and more transferable, effectively merging many apparently disparate and sparse Indic language resources into a single high-resource metalanguage bedrock. We propose a four-part benchmark suite to render this shared architecture explicit, measurable, and ready to be leveraged for practical applications. Moreover, we underscore the question it raises for interpretability research: whether neural models trained on these languages come to represent Pānini's categories on their own. 

---
# BehaviorBench: Benchmarking Foundation Models for Behavioral Science Tasks 

**Authors**: Jin Huang, Yutong Xie, Wanli Song, Xingjian Zhang, Walter Yuan, Matthew O. Jackson, Qiaozhu Mei  

**Link**: [PDF](https://arxiv.org/pdf/2606.24162)  

**Abstract**: Foundation models have been increasingly applied to behavioral science domains such as psychology, sociology, and economics. While these models show promise in individual tasks such as survey response prediction and human-subject experiment simulation, there remains no systematic understanding of how well they perform across diverse behavioral science tasks, contexts, and populations. We introduce BehaviorBench, a comprehensive benchmark that evaluates foundation models along four core capabilities: (1) behavior prediction and simulation, (2) strategic decision-making, (3) subject-trait inference, and (4) behavioral knowledge application. Crucially, BehaviorBench evaluates model outputs at both the individual and distributional levels, capturing not only per-subject accuracy but also population-level alignment, an essential requirement for behavioral validity. Leveraging the tasks in BehaviorBench, we further develop this http URL-1.5, extending the this http URL family of behavioral foundation models fine-tuned on behavioral data. Our results reveal a considerable gap: proprietary general-purpose models excel at individual-level prediction and knowledge-intensive tasks, whereas behavioral foundation models, fine-tuned on behavioral data, achieve substantially stronger distributional alignment. Notably, this http URL-1.5 leads on distributional metrics and remains competitive on individual-level metrics, suggesting that proper behavioral adaptation can close the gap. Our results highlight the importance of distributional evaluation, establish BehaviorBench as a foundation for developing and assessing behaviorally aligned AI systems, and demonstrate this http URL-1.5's potential for a broad range of behavioral science studies. Our BehaviorBench and this http URL-1.5 models can be accessed via this https URL. 

---
# MedBench v5: A Dynamic, Process-Oriented, and Hallucination-Aware Benchmark for Clinical Multimodal Models 

**Authors**: Ding Jinru, Jiang Chuchu, Lu Lu, Pang Wenrao, Bian Mouxiao, Gao Zhuangzhi, Chen Jiangyuan, Peng xinwei, Chen Ruiyao, Ren Sijie, Lu Renjie, Han Bin, Liu Meiling, and Xu Jie  

**Link**: [PDF](https://arxiv.org/pdf/2606.24155)  

**Abstract**: Existing medical AI benchmarks lack process visibility, atomic skill evaluation, and integrated hallucination detection. We introduce MedBench v5, a redesigned benchmark for clinical multimodal models (language, vision-language, and agent systems) that moves from static QA to dynamic, process-oriented evaluation. MedBench v5 features: (1) a dual-dimensional framework combining Clinical Cognitive Responsiveness (14 sub-dimensions) and Medical Atomic Skills (4 agent environments), covering 63 tasks; (2) three switchable information-flow stressors (omission, contradiction, evidence delay) for factorized degradation analysis; (3) a dynamic process audit protocol with five reasoning nodes that produces model-specific failure fingerprints; (4) hallucination propagation monitoring across initiation, propagation, anchoring, and contradiction interaction-capturing silent hallucination. Experiments on frontier models show that strong overall task performance does not guarantee process stability: stressors mainly disrupt contradiction detection, diagnosis updating, hallucination propagation, and contradiction-based self-correction, while final evidence grounding can remain superficially stable. MedBench v5 provides a unified infrastructure for capability profiling, controllable stress testing, process auditing, and hallucination trajectory analysis in clinical AI evaluation. 

---
# Metis: Bridging Text and Code Memory for Self-Evolving Agents 

**Authors**: Zijie Dai, Siuhin He, Hui Li, Qihui Zhou, Jiajun Li, Mingcong Song, Guoping Long, Hongjie Si, Xin Yao, Lin Zhang, James Cheng, Xiao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.24151)  

**Abstract**: Self-evolving agents improve over time by distilling experience from past executions and reusing it in future tasks. Existing systems represent such experience either as natural-language text injected into the agent context or as code exposed as callable tools. However, the choice between these representations is typically made at design time rather than derived from the characteristics of the experience itself, leaving the trade-offs between them poorly understood. We present the first controlled study that isolates text memory and code memory over an identical set of experiences. Our results show that the two forms exhibit complementary trade-offs in construction cost, execution efficiency, and transferability, such that neither representation alone is sufficient. Guided by these findings, we propose Metis, a self-evolving agent system built on a hierarchical dual-representation memory. Metis organizes textual experience into execution plans, environment facts, and common pitfalls, and selectively crystallizes recurring plans into validated callable tools. This design combines the broad applicability of text memory with the execution efficiency of code memory while incurring tool-generation cost only when justified by repeated reuse. We evaluate Metis on AppWorld, a challenging benchmark for interactive agents. The results show that Metis improves task accuracy by up to 20.6% over ReAct while reducing execution cost by up to 22.8%. Compared with representative self-evolving agent systems, Metis consistently achieves a better balance between accuracy, execution efficiency, and memory-construction cost. 

---
# PORTER: Language-Grounded Event Representations for Portable Structured EHR Foundation Models 

**Authors**: Lin Lawrence Guo, Adam Paul Yan, Emily Vettese, Lillian Sung  

**Link**: [PDF](https://arxiv.org/pdf/2606.24102)  

**Abstract**: Most electronic health record (EHR) foundation models encode clinical events as discrete event tokens from a fixed vocabulary and therefore cannot directly represent events containing unseen concepts or new combinations of concepts and attributes such as numeric values. This limits transfer across institutions and even across deployment pipelines within the same institution. We introduce PORTER, a language-grounded structured EHR foundation model that decouples event representation from this fixed vocabulary. PORTER represents events through their descriptions using a frozen text encoder, integrates numeric values through a dedicated pathway, and learns clinical dynamics over patient timelines with an autoregressively pretrained temporal backbone. Across 74 clinical prediction tasks at a pediatric hospital, PORTER matched the mean AUROC of a fixed-vocabulary model with the same temporal backbone and pretraining objective. When the same patient timelines were rendered using event descriptions not seen during pretraining, PORTER transferred without retraining or vocabulary mapping, recovering 97.1% of the mean AUROC of a model trained directly on the target vocabulary. When transferred to MIMIC, PORTER outperformed the fixed-vocabulary model, which dropped 69% of events because their tokens were unseen. Mechanistic analyses showed cross-vocabulary transfer tracked preservation of patient-level representation geometry rather than the scale of the text encoder, and the numeric pathway improved sensitivity to magnitude without disrupting clinical concept identity. PORTER also achieved higher AUROC than a task-specific text serialization comparator, at 329-fold lower amortized compute. PORTER is a step toward vocabulary-independent EHR foundation models that reduce the need for vocabulary harmonization while preserving in-domain performance and enabling efficient cross-task reuse. 

---
# Predicting Poets' Origins from Verse: A Computational Analysis of Regional Linguistic Fingerprints in the Complete Tang Poems 

**Authors**: Chi-Sheng Chen, Hung-Yun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.24093)  

**Abstract**: We ask whether the geographic origin of Tang-dynasty poets leaves a detectable linguistic trace in their work. Aggregating every poem attributed to each author in the Complete Tang Poems (Quan Tang Shi) and linking poets to their administrative circuit of origin via the China Biographical Database (CBDB), we build a poet-level corpus of 357 poets across the ten Tang circuits and frame origin prediction as multi-class classification. Using character $n$-gram TF-IDF together with interpretable domain features (imagery, season, and allusion), classical and neural models predict a poet's broad region (South vs.\ North) at $0.69$ accuracy, well above the $0.53$ majority baseline, and finer circuit-level origin above chance. Beyond classification, three findings emerge. (i) Linguistic distance between circuits grows with geographic distance (Mantel $r=0.40$, $p\approx0.09$ over nine circuits), evidence of a distance-decay effect in poetic language. (ii) The signal interacts with time: South/North separability is at chance in the High Tang and strongest in the Late Tang, consistent with court-driven homogenization at the empire's height followed by regional divergence. (iii) The model's confident errors are historically meaningful -- in the Early Tang, every misclassification is a southern poet read as northern, reflecting the prestige of the northern court idiom. We further show that, when given the whole corpus through a hierarchical frozen-encoder representation, a classical-Chinese transformer (GuwenBERT) only matches -- not beats -- simple TF-IDF, and that combining them adds nothing, indicating that character $n$-grams already capture the regional signal. Our results position interpretable machine learning as a hypothesis generator for literary history. 

---
# CAVEWOMAN: How Large Language Models Behave Under Linguistic Input and Output Compression 

**Authors**: Morayo Danielle Adeyemi, Ryan A. Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2606.24083)  

**Abstract**: "Talk short. Drop grammar. Save token." This caveman style is widely promoted as a way to cut inference cost, but whether it actually saves anything depends on which channel (the user's prompt or the model's response) is being compressed. We present Cavewoman, a two-channel evaluation protocol that scores every generation on task accuracy, realized per-item cost, and reference-text agreement against the model's unconstrained reference. We evaluate eight models on five datasets at five reduction levels, with both channels measured on the same items. Output compression cuts realized cost on most API models (1.4-2.4x per model, up to 3x in the best case) and on all four open-weight models under public-tier pricing. Input compression has the opposite effect, a strict lose-lose: it raises net cost rather than lowering it (~1.15x on the five-benchmark mean, up to 1.8x on the worst dataset and 2.7x under stronger compression), because models compensate with longer responses even as accuracy collapses. Under the same setting, surface text diverges from the unconstrained reference: on the non-reasoning models, roughly half of all generations are correct yet their surface text no longer entails the model's own unconstrained baseline generation. The divergence survives length-controlled re-scoring, multiple-comparisons correction, and replication under complementary semantic measures. Code and data are available at this https URL. 

---
# Sentence-Level Contextual Entrainment in Large Language Models 

**Authors**: Yang Liu, Chenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2606.24077)  

**Abstract**: Contextual entrainment, which is a newly discovered phenomenon in large language models (LLMs), refers to the tendency of a model to assign higher probabilities to tokens that appear in its context. In this work, we extend this phenomenon from the token level to the sentence level by examining the per-token mean log-probability of a sentence instead of the probabilities of individual tokens. We investigate sentence-level contextual entrainment across 26 LLMs from seven families and two datasets, which cover both subjective and objective tasks. We find that sentence-level contextual entrainment exists. This means that the sentences in the prompt (even if they are counterfactual statements) can significantly increase their probability during model inference time. As the model size increases, contextual entrainment gradually decreases. We also find that contextual entrainment is controlled by 2% to 4% of the attention heads. Turning off these attention heads can effectively mitigate contextual entrainment without hurting the model's performance. 

---
# Selective Capability Unlearning in End-to-End Spoken Language Understanding 

**Authors**: Akanksha Singh, Vinod Kumar Kurmi  

**Link**: [PDF](https://arxiv.org/pdf/2606.24063)  

**Abstract**: Modern spoken language understanding (SLU) systems are increasingly deployed in real-world settings, where specific functionalities may need to be removed due to policy or safety constraints. In SLU, a functionality corresponds to an intent and its associated slot-generation behavior. However, in autoregressive models, suppressing a target intent does not eliminate the conditional mapping that generates slots conditioned on that intent. When the intent prefix is externally supplied, the model can reconstruct the original intent-slot structure. We identify this structural failure as \textbf{\emph{capability persistence}}. We propose \textit{\underline{B}inding \underline{S}ubspace (BSU)}, a representation-level framework that isolates and attenuates intent-conditioned directions underlying this mapping. Across SLU benchmarks, BSU substantially reduces forced-prefix recoverability while preserving retained performance. 

---
# Best Preprocessing Techniques for Sentiment Analysis 

**Authors**: Saranzaya Magsarjav, Melissa Humphries, Jonathan Tuke, Lewis Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2606.24055)  

**Abstract**: Sentiment analysis in Twitter datasets is important because it enables monitoring public opinion on products and analysis of political and social movements. One critical step is preprocessing: the automated processing of text for machine learning algorithms. Preprocessing plays a critical role in reducing noise and improving efficiency. However, little research has systematically examined the order in which preprocessing techniques are implemented. We find that, when accounting for order, spelling correction is the least impactful preprocessing technique, whereas tokenisation is the most impactful. Stemming and stop-word removal are interchangeable, and it is better to remove stop words without removing negation. The best order for applying the preprocessing techniques was tokenisation, text cleaning, stemming, and then stopword removal. Our results provide a systematic approach for practitioners to deploy preprocessing to improve model output without the costly preprocessing exploratory phase. 

---
# Towards Version-aware Operations and Transaction Memories for Multi-layer MeMo 

**Authors**: Peiran Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.24040)  

**Abstract**: MeMo proposes language models with explicit multi-layer correlation matrix memories (CMMs), where memorization, retrieval, and forgetting are architectural operations. This paper asks how such memories can reduce the need for retraining when knowledge changes. For changes expressible as MeMo memory associations, the model's accessible knowledge can be updated by editing explicit memories rather than retraining the whole model. We propose a version-aware operation layer in which high-level operations such as replace, obsolete, keep-history, rollback, and trace are compiled into MeMo-native primitive calls over sequences and tokens. The key observation is that a version-aware operation is rarely a single MeMo association. It is an ordered transaction of primitive edits, for example forgetting one sequence-token chain, memorizing another, preserving a historical chain, and recording an inverse program. The framework introduces two auxiliary CMMs: a Version CMM (V-CMM) for mapping version transitions to transaction handles, and a Transaction CMM (T-CMM) for storing reusable change contents and inverse programs. It supports both direct sequence-level edits and structured diff-level inputs, and outlines an evaluation route for update success, rollback, traceability, locality, and transaction reuse. 

---
# Towards Spec Learning: Inference-Time Alignment from Preference Pairs 

**Authors**: Dhriti Krishnan, Tejas Goyal, Jaromir Savelka  

**Link**: [PDF](https://arxiv.org/pdf/2606.24004)  

**Abstract**: Steering a large language model (LLM) toward a desired behavior typically relies on an iterative process of hand-crafting a prompt based on a careful inspection of the model's responses. This is an involved, brittle, and error-prone process. Preference-based fine-tuning is a more rigorous but often prohibitively expensive solution. We propose spec learning, a framework that relies on a brief user instruction and a small set of preference judgments. These are compiled into specifications in the form of natural-language prompts for an LLM. Specifications condition LLMs at inference time, and no parameter updates to the underlying models are required. We show that the responses generated based on the compiled specifications often outperform direct preference optimization (DPO) on datasets from specialized domains whose preference signal is dense. Unlike opaque weight updates, the resulting specifications are human-readable and double as interpretable and transparent written embodiments of the preference signal that produced them. 

---
# RASC+: Retrieval-Constrained LLM Adjudication for Clinical Value Set Authoring 

**Authors**: Sumit Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.23992)  

**Abstract**: Clinical value sets define the standardized terminology codes used in quality measurement, phenotyping, cohort construction, and clinical decision support. The recently introduced Retrieval-Augmented Set Completion (RASC) benchmark showed that direct zero-shot large language model (LLM) generation is poorly suited to this task: clinical code systems are large, version-controlled, and not reliably memorized by language models. We study a stage-wise alternative in which candidate-pool construction is optimized for recall and a constrained LLM adjudicator is optimized for candidate selection. On the full 3,744-value-set RASC test split, Qwen3-based retrieval with vocabulary-aware expansion and code-display rescue retrieval increases candidate-pool recall from the original RASC retrieval baseline of 0.553 to 0.730; on the held-out-publisher stratum, pool recall is 0.655. The higher-recall pool alone is not sufficient: applying the original SAPBert cross-encoder to this expanded pool gives full-test macro F1 of 0.287 and held-out-publisher macro F1 of 0.233. Replacing the stage-2 selector with blinded GPT-5 adjudication over the same pool increases full-test macro F1 to 0.549 and held-out-publisher macro F1 to 0.533. These results show that retrieval-constrained LLM adjudication can substantially improve value set completion while preserving the safety constraint that all returned codes must come from an auditable candidate pool. 

---
# Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization 

**Authors**: Shuo Guan  

**Link**: [PDF](https://arxiv.org/pdf/2606.23989)  

**Abstract**: End-to-end large language models (LLMs) produce fluent multi-document summaries but remain prone to hallucination, and the attributions they offer are typically coarse (whole documents or passages) and generated post hoc, leaving each summary statement hard to verify. We revisit the modular Extract--Select--Rewrite paradigm and recast its intermediate representation as the unit of attribution. We present CAMS, a Claim-Anchored Multi-document Summarization framework that (i) extracts atomic claims with token-level provenance from every source document, (ii) clusters equivalent claims across documents while flagging inter-source conflicts, (iii) selects a support-aware and salient subset, and (iv) rewrites the selection into a summary in which every sentence is anchored to a support-checked claim that links back to one or more source spans. Because content is localized before it is realized, the pipeline is attribution-oriented by construction and faithfulness-oriented by construction: it structurally preserves fine-grained, multi-source traceability while using support-aware selection, constrained rewriting, and verification to encourage, rather than guarantee, factual faithfulness. We evaluate quality, faithfulness, and localization on MultiNews, analyze conflict handling on DiverseSumm, and test zero-shot transfer on WCEP, using a two-regime protocol that separates reference-free citation quality from gold-aligned localization accuracy, and we add an evaluator-decoupled audit that tests citation precision with a support model never used for selection or verification. CAMS matches strong end-to-end and span-attribution baselines on summary quality while substantially improving faithfulness and citation precision, lifting multi-source attribution accuracy by roughly two-thirds, and exposing a controllable faithfulness--coverage trade-off that end-to-end models leave implicit. 

---
# Does My Embedding Reflect That $A = B$? Evaluating Mathematical Equivalence in Embedding Models 

**Authors**: Jiaying Ye, Samarth Rao, Leo Carlin, Kedar Chintalapati, Saharsh Bhargava, Rachit Jaiswal, Michael Zhou, Jared Darlington, Jarod Alper, Vasily Ilin, Henry Kvinge  

**Link**: [PDF](https://arxiv.org/pdf/2606.23959)  

**Abstract**: Because mathematics is highly abstract, a single statement can take very different forms depending on what subfield it is framed in. There are many examples where breakthroughs occurred after researchers discovered that a question had already been answered in a different field. At the same time, the growth of new resources related to formalization has increased the need for tools that enable efficient and reliable navigation between mathematical 'languages' (e.g., from Lean to natural language). In this paper, we investigate whether current embedding models capture mathematical equivalence. To do this, we introduce the Mathematically Equivalent but Lexically Different Pairs (MELD) Dataset, a collection of mathematically equivalent statements that are expressed in very different language. We show that current state-of-the-art embedding models tend to group statements by the terminology used to make them instead of the underlying math. Motivated by this, we propose a contrastive approach to learning embeddings of mathematical text that focuses on aligning informal statements with different formalizations. Our experiments demonstrate that this leads to improvements not only on informal-formal retrieval tasks but also on MELD, which only contains natural language statements. 

---
# Layer-wise Probing of wav2vec 2.0 and Whisper for Consonant Cluster Reduction in African American English 

**Authors**: Hamid Mojarad, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.23948)  

**Abstract**: Self-supervised and supervised speech models are increasingly used to investigate which linguistic information their internal representations encode, and at what level of abstraction they encode it. One underexplored phenomenon is consonant cluster reduction (CCR) in African American English (AAE), a widespread phonological process and a source of automatic speech recognition (ASR) disparity. To examine how CCR is represented, we conduct speaker-independent layer-wise probing of wav2vec2-base and Whisper-small using two tasks: segmental reduction detection and segmental restoration of underlying cluster identity. Both models distinguish reduced and canonical forms with high accuracy. Crucially, reduced segments retain cues to their underlying stops, indicating that CCR is encoded as structured gradient phonological variation rather than simple segmental deletion. These results demonstrate structured phonological encoding of AAE CCR patterns in modern speech models. 

---
# QuechuaTok: Morphological Boundary Accuracy as a Necessary Metric for Tokenizer Evaluation in Agglutinative Low-Resource Languages 

**Authors**: Maria Contreras  

**Link**: [PDF](https://arxiv.org/pdf/2606.23943)  

**Abstract**: Tokenization is a foundational step in NLP pipelines, yet standard evaluation metrics such as fertility rate fail to capture morphological correctness for agglutinative languages. We present QuechuaTok, a systematic benchmark comparing four tokenization strategies - BPE, Unigram LM, WordPiece, and a morphology-aware PRPE tokenizer - for Southern Quechua (quz), a low-resource agglutinative language spoken by 8-10 million people in South America. Using a 200k-sentence corpus and the SQUOIA finite-state morphological analyzer (Rios, 2016) as silver standard, we evaluate three metrics: fertility rate, OOV rate, and morphological boundary accuracy (MorphAcc). Our results show that BPE achieves the lowest fertility rate (1.636 at 16k vocab) by memorizing surface word forms, while achieving only 6.67% MorphAcc. PRPE achieves 83.33% MorphAcc - the highest of all systems - demonstrating that fertility rate alone is insufficient to evaluate tokenizers for agglutinative languages. All code and models are publicly available at this http URL 

---
# When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents 

**Authors**: Tianyu Ding, Juan Pablo De la Cruz Weinstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.23937)  

**Abstract**: Exact-match retrieval recall is often used as a proxy for whether a retriever supplies useful policy context to a downstream decision model. We test this proxy for pre-action policy classification in tau-bench using Qwen2.5-3B/7B classifiers. Under gold-policy conditioning, a compact structured state improves macro-F1 over raw trajectories by 0.13-0.17 after tuning. We then replace the benchmark-designated policy clause with the top-ranked clause retrieved from decision-time context. Although the exact governing clause is retrieved at rank 1 for only 7% of airline states, the primary 3B classifier obtains macro-F1 0.58 with retrieved clauses versus 0.60 with gold clauses (Delta=-0.02, task-cluster 95% CI [-0.23,+0.21]); mismatched-policy and no-policy controls score 0.32 and 0.21. We do not detect a macro-F1 difference between retrieved and gold clauses in this configuration, although the interval remains too wide to establish non-inferiority. The same qualitative pattern appears with a second retriever and at 7B, while varying across fine-tuning configurations. These results indicate that exact-match clause recall can underestimate downstream policy utility in this benchmark setting, motivating evaluation with retrieved policies in the classification loop rather than recall alone. 

---
# Do LLM Attribution Metrics Transfer? Auditing Retrieval-Augmented Generation Evaluation Across Datasets and Constructs 

**Authors**: Tianyu Ding, Aditya Nannapaneni, Juan Pablo De la Cruz Weinstein  

**Link**: [PDF](https://arxiv.org/pdf/2606.23915)  

**Abstract**: Practice often treats automatic metrics for attribution in LLM retrieval-augmented generation as interchangeable. We audit eight automatic scorers -- lexical, embedding, and BERTScore baselines alongside entailment/grounding-trained models (clean and FEVER NLI, the checker MiniCheck) -- across three evaluation constructs (provenance/topicality, generated-answer attribution, and fact-check entailment), asking whether any scorer transfers: stays within the 95% confidence interval of the best audited scorer on every dataset of a multi-dataset construct. In the construct with the most multi-dataset human-labeled coverage -- generated-answer attribution (AttributionBench's four source datasets, n = 1,610, with independent HAGRID, n = 2,150) -- none does: the per-dataset metric rankings invert (Kendall tau = -0.64, p = 0.031 on AttributedQA vs. LFQA), and an off-the-shelf NLI scorer that is best on short-claim AttributedQA (AUROC 0.90) collapses to AUROC 0.53 (chance) on long-form LFQA, where BERTScore wins (0.91); the flip is not a length or truncation artifact. This instability has a concrete decision cost: a naive "best-on-average" rule for choosing an evaluator fails leave-one-dataset-out (mean held-out regret 0.172 AUROC, worse than fixing one scorer), so metric choice must be validated on the target dataset rather than learned from others. A prompt-based LLM judge avoids the chance-level collapses the automatic scorers suffer (no LFQA collapse) but is not uniformly best, ~100x costlier, and non-deterministic -- relocating, not removing, the validation burden. 

---
# One Year Later...The Harms Persist, But So Do We! 

**Authors**: Annika Marie Schoene, Cansu Canca, Gautham Vijay Kumar, Anson Antony  

**Link**: [PDF](https://arxiv.org/pdf/2606.23884)  

**Abstract**: General-purpose large language models (LLMs) are increasingly used for mental health-related conversations, yet safety safeguards remain inadequate and inconsistent across clinical conditions. This study evaluates six proprietary LLMs across 16 DSM-5 conditions using four adversarial attack variants, introducing an eight-dimension harm taxonomy and a multi-dimensional evaluation framework. Results show that safeguards hold reliably only for suicide and self-harm, while conditions such as eating disorders, substance use disorder, and major depressive disorder exhibit failure rates of up to 100%. We argue that ethical design and deployment of these LLMs demand clearly defined harm categories across clinical conditions and implementation of safeguards accordingly. Until such safeguards are in place, these models pose significant risks to vulnerable populations, making their growing integration into educational settings a particularly concerning. 

---
# Ground Then Rank: Revisiting Knowledge-Based VQA with Training-Free Entity Identification 

**Authors**: Qian Ma, Qiong Wu, Zhengyi Zhou, Yao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.23881)  

**Abstract**: Knowledge-Based Visual Question Answering (KB-VQA) requires grounding visual queries to external knowledge beyond directly observable content in images. While recent multi modal large language models (MLLMs) show strong perceptual abilities, they struggle on KB-VQA tasks requiring groundings from both fine-grained entity and evidence levels. Most existing multi-modal retrieval augmented generation (MM-RAG) methods tightly couple entity discrimination and section-level evidence ranking into a single re-ranking stage, leading to high cost and limited generalization. In this work, we revisit existing MM-RAG solutions from a workflow perspective and argue both entity-level and fact-level groundings are key bottlenecks. We observe that although MLLMs often fail under open-ended entity naming, they can better identify the correct entity when selecting from a small set of candidate names. Based on this insight, we propose a simple and training-free identify-before-answer IBA framework that decouples entity identification from section-level re-ranking. Our approach prompts an MLLM to select high-confidence entities using only candidate names, followed by an off-the-shelf textual re-ranker for evidence selection. Experiments on Encyclopedic-VQA and InfoSeek show that our method consistently outperforms fine-tuned multi-modal re-ranking baselines while reducing training and inference complexity. Additional analyses reveal that the improvements arise not only from better entity identification, but also from selecting more informative evidence once correct entity is fixed. Our implementation is made public to ease reproducibility. 

---
# Evaluating LLM Usage for Efficient and Explainable Numerical and Classified Implicit Sentiment Analysis of Product Desirability 

**Authors**: Sherri Weitl-Harms, John Hastings  

**Link**: [PDF](https://arxiv.org/pdf/2606.23701)  

**Abstract**: Qualitative product feedback can reveal nuanced user experiences, but its implicit sentiment is difficult to measure. This paper presents a scalable and interpretable framework that uses large language models (LLMs) to quantify product desirability from such data. Using two Product Desirability Toolkit (PDT) datasets from ZORQ and CARMA comprising 106 respondent term groupings with gold-standard human annotation, zero-shot continuous numerical sentiment scoring and categorical sentiment classification are evaluated without relying on explicit review scores. Across the datasets, LLMs generated numerical sentiment scores directly from qualitative responses and closely matched expert labels, achieving Pearson correlations up to 0.97 and classification accuracy up to 94%. LLMs maintained robustness even when handling data presented in multiple forms and consistently expressed high confidence. In contrast, lexicon-based and transformer baselines did not produce statistically significant results. Among the models tested, GPT-4o-mini achieved performance comparable to larger models at 94% lower cost, supporting scalable deployment.
The framework also incorporates model confidence ratings and human-readable rationale explanations (xAI), improving interpretability, transparency, and trust while supporting practical use in product satisfaction assessment. In general, using the PDT tool as a survey method along with a cost efficient LLM for sentiment analysis has the potential to provide for product evaluation with results that are rich in terms of sentiment scores (both numerical and classified sentiment) and in terms of the high-level user impressions of the product that can be used to identify ideas for product development and improvement, as well as marketing ideas for target audiences. 

---
# Self-Recognition Finetuning can Prevent and Reverse Emergent Misalignment 

**Authors**: Arush Tagade, Shaoheng Zhou, Jiaxin Wen, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.23700)  

**Abstract**: Emergent misalignment (EM) has been linked to the activation of misaligned persona vectors and evil character traits, suggesting that EM operates through disruption of the model's aligned character rather than direct learning of harmful content. Motivated by this connection, we study self-generated text recognition (SGTR) finetuning as a character-targeted intervention that is distinct from existing in-training defenses. We conduct two-stage finetuning experiments across three models (GPT-4.1, Qwen2.5-32B-Instruct, Seed-OSS-36B-Instruct) and multiple EM datasets to compare SGTR finetuning against benign finetuning baselines (correct domain-specific data, general knowledge, and word counting) to find it an effective defense in both reversal and prevention settings. We find that all interventions produce comparable EM reversal, but only when restoring capabilities that EM had degraded. For prevention, only SGTR finetuning consistently reduces misalignment without exacerbating any individual metric, suggesting that character fortification specifically drives prevention. We provide further evidence for EM's relation to the LLM's default character by showing that EM finetuning induces diversity into the LLM's identity self-reports, artificially corrupting self-recognition exacerbates misalignment caused by EM finetuning, and that removing the model's identity-bearing system prompt substantially reduces the effect of EM finetuning. Together, these findings reframe EM not as the adoption of a coherent misaligned persona but as the destabilization of aligned character. 

---
# Quantifying Prior Dominance in RAG Systems 

**Authors**: Barak Or  

**Link**: [PDF](https://arxiv.org/pdf/2606.23695)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds Large Language Models in external knowledge, yet current evaluations rely on discrete heuristics that suffer from ''epistemic blindness'' - failing to distinguish genuine contextual information extraction from parametric memory recall. To address this, we introduce the Normalized Context Utilization (NCU) metric, leveraging continuous token log-probabilities across zero-shot, oracle, and adversarial conditions to strictly quantify contextual information gain. Evaluating architectures ranging from 1.5B to 72B parameters alongside a proprietary commercial API reveals that for strict factual extraction (without Chain-of-Thought reasoning), traditional scaling laws exhibit extreme diminishing returns: highly efficient Small Language Models (SLMs) match or outperform high-capacity architectures. Furthermore, we demonstrate that ``Prior Dominance'' correlates with model scale and proprietary alignments. The evaluated commercial API not only overrode explicit external evidence in nearly half of adversarial conflicts, but also frequently suffered from systemic confidence collapse (Negative Transfer) when its parametric priors were contradicted. Our findings highlight the structural epistemic advantage and superior contextual adherence of SLMs in strict extraction workflows. 

---
# ModTGCN: Modularity-aware Graph Neural Networks for Text Classification 

**Authors**: Rajarshi Misra, Aditya Sharma, Vinti Agarwal, Hari Om Aggrawal  

**Link**: [PDF](https://arxiv.org/pdf/2606.23694)  

**Abstract**: Graph-based text classification models typically rely on local neighborhood aggregation and overlook global community structure, despite semantic document graphs exhibiting strong class-consistent clustering. Ignoring this can blur class boundaries and lead to over-smoothing. We propose ModTGCN, a modularity-aware graph neural network for text classification that jointly optimizes cross-entropy and a modularity-based auxiliary objective to promote class-coherent document communities while preserving discriminative representations. The modularity term is computed on a document-document similarity graph derived from transformer embeddings (pretrained or fine-tuned). To improve scalability, we decouple the original heterogeneous TextGCN graph into separate document-word and word-word components, achieving 2x-10x faster training. We further study graph construction strategies, label-aware edge reweighting, and supervision choices for modularity optimization. Experiments on five benchmarks show consistent gains, with larger improvements on complex, low homophily datasets such as Ohsumed and 20NG. 

---
# EXPO-SQL: Execution-based Clause-level Policy Optimization for Text-to-SQL 

**Authors**: Jaehoon Lee, CheolWon Na, Suyoung Bae, Jin-Seop Lee, Jihyung Lee, YunSeok Choi, Jee-Hyong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.23693)  

**Abstract**: Text-to-SQL enables users to query databases using natural language by generating executable SQL queries. Recent methods have increasingly adopted Large Language Models based reinforcement learning (RL) to leverage execution feedback for training. However, existing RL methods assign uniform query-level rewards to all clauses in a SQL query, treating correct and incorrect clauses equally. This coarse-grained reward design leads to insufficient learning signals for correct SQL generation. To address this issue, we propose EXPO-SQL (EXecution-based clause-level Policy Optimization for Text-to-SQL) which provides fine-grained supervision through clause-level rewards. To assign clause-level rewards, our method identifies erroneous clauses by analyzing execution results, including error messages and clause-wise incremental execution. Experiments on widely-used Text-to-SQL benchmarks demonstrate that EXPO-SQL significantly outperforms existing supervised fine-tuning, prompting, and RL-based methods through fine-grained clause-level learning. Our code is available at https://github. com/jhn25/EXPO-SQL. 

---
# Matching Tasks to Objectives: Fine-Tuning and Prompt-Tuning Strategies for Encoder-Decoder Pre-trained Language Models 

**Authors**: Ahmad Pouramini, Hesham Faili  

**Link**: [PDF](https://arxiv.org/pdf/2606.24841)  

**Abstract**: Prompt-based learning has emerged as a dominant paradigm in natural language processing. This study explores the impact of diverse pre-training objectives on the performance of encoder-decoder pre-trained language models across generation and question answering tasks, with a focus on commonsense knowledge retrieval and completion. We highlight the benefits of incorporating multiple objectives during both pre-training and fine-tuning stages. We introduce the Match Task to Objective (MTO) framework and methods for determining the appropriate objective for a given task. This framework offers automated methods to prepare task-related data for adaptation through unsupervised training, based on the identified objective. In the fine-tuning stage, we design novel templates that align with the objectives of the pre-training and adaptation stages. When aligned with task requirements, these strategies can achieve a performance gain of over 120\% compared to conventional methods in few-shot settings. They significantly outperform related works in few-shot settings and exceed the baseline even in full-dataset scenarios. Furthermore, we extend this approach to include prompt-tuning methodologies, providing guidance for more effective soft prompt engineering and optimization. Our strategies significantly enhance prompt-tuning performance as well. These insights hold substantial value, precisely guiding the selection and optimization of models customized for specific tasks. Code is available at this https URL 

---
# ParaPairAudioBench: Paralinguistic Pairwise Audio Benchmark for LALM-as-a-Judge 

**Authors**: Jisu Jeon, Seungyeon Jwa, Joosung Lee, Jinhyeon Kim, Woojin Chung, Hwiyeol Jo, Jeonghoon Kim, Jonghyun Choi, Soyoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.24648)  

**Abstract**: Large Audio-Language Models (LALMs) have been widely used as judge models for the automatic evaluation of generated speech. However, prior approaches predominantly focus on holistic naturalness, leaving fine-grained paralinguistic distinctions underexplored. We introduce ParaPairAudioBench, a pairwise benchmark of 5,175 audio pairs across five paralinguistic dimensions: Style, Rate, Emphasis, Age, and Gender. Our experiments show that current LALM judges still lag behind human judgments by 32%p on average and exhibit severe calibration failures, particularly in Tie cases where the correct decision is to abstain. To further analyze lexical versus acoustic reliance, the benchmark includes both same-transcript and cross-transcript conditions. ParaPairAudioBench enables multi-dimensional, calibration-aware assessment of the reliability of LALM-as-a-Judge for paralinguistic speech evaluation. 

---
# AdversaBench: Automated LLM Red-Teaming with Multi-Judge Confirmation and Cross-Model Transferability 

**Authors**: Khanak Khandelwal  

**Link**: [PDF](https://arxiv.org/pdf/2606.24589)  

**Abstract**: Scaling adversarial evaluation of large language models requires both a method for generating hard inputs and a reliable way to confirm that resulting failures are real. We present AdversaBench, an end-to-end red-teaming pipeline that mutates seed prompts with five structured operators, queries a target model, and confirms failures through a three-judge panel with a meta-judge tiebreaker. We report experiments on 45 seeds across three categories: reasoning, instruction-following, and tool use. Every seed produced a confirmed failure. Four findings stand out. First, operator effectiveness varies sharply by category: inject_distractor scores 0.00 mean reward on instruction-following seeds but 0.80-0.83 on reasoning and tool-use. Second, binary failure rate hides difficulty: instruction-following seeds required 2.4 attacker iterations on average versus 1.1 for other categories, a gap visible in survival curves. Third, pairwise judge agreement of 80-87% coexists with near-zero Cohen's kappa due to label skew; category-level disagreement rates are more informative. Fourth, adversarial prompts generated against Llama 3.1 8B transfer zero-shot to Llama 3.3 70B, suggesting the mutations exploit general behavioral patterns rather than model-specific weaknesses. Code, dataset, and analysis scripts are available at this https URL . 

---
# A specialized reasoning large language model for accelerating rare disease diagnosis: a randomized AI physician assistance trial 

**Authors**: Haichao Chen, Songchi Zhou, Zhengyun Zhao, Shikai Hu, Xianghong Jin, Hongwei Ji, Li He, Shuli Li, Yiming Qin, Xin Tan, Runfeng Shi, Yih Chung Tham, Jiaye Zhu, Ye Li, Ye Jin, Longhao Cao, Dawei Li, Honghan Wu, Hongqiu Gu, Guanqiao Li, Tudor Groza, Chunying Li, Dian Zeng, Weihong Yu, Gareth Baynam, Saumya Shekhar Jamuar, Min Shen, Shuyang Zhang, Bin Sheng, Sheng Yu, Tien Yin Wong  

**Link**: [PDF](https://arxiv.org/pdf/2606.24510)  

**Abstract**: Rare diseases affect millions of individuals worldwide, yet timely diagnosis remains a major public health challenge due to scarcity of specialized clinical expertise. While large language models (LLMs) show promise to support rare disease diagnosis, current models are constrained by insufficient clinical deployability, limited clinically grounded evidence, and scarcity of training data. Here we present RaDaR (Rare Disease navigatoR), an open-source, compact reasoning LLM (32B parameters) for rare disease diagnosis. RaDaR was trained with 49,170 publicly available free-text cases and 104,666 synthetic cases with reasoning-enhanced training. RaDaR showed the strongest performance among evaluated open-source models, including the 671B DeepSeek-R1, across public benchmarks and four external validation centers. In a retrospective cohort, RaDaR prioritized the final diagnosis before documented clinical suspicion in 61.06 percent of cases, corresponding to a potential lead time of 1.87 months and 50.18 percent of the within-center interval. In a randomized physician-assistance trial, RaDaR assistance improved physicians' rare-disease diagnostic accuracy by 21.44 percentage points compared with internet search alone. Synthetic-data ablations suggested that phenotype-anchored narratives provide useful training signal for long-tail rare diseases, with a monotonic scaling trend within the tested data range. Together, RaDaR and its development and validation framework provide a deployable rare-disease reasoning model and a reproducible development framework for diagnostic AI under data scarcity. 

---
# An LLM-based Two-Stage Transformer Framework for Cross-Domain Bearing Fault Diagnosis with Limited Data 

**Authors**: Jinghan Wang, Feng Cheng, Wentao Wu, Hang Li, Gaoliang Peng, Tianchen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.24459)  

**Abstract**: Bearing fault diagnosis faces critical challenges when dataset heterogeneity, operating condition variations, and limited labeled data occur simultaneously in industrial environments. Existing approaches address these issues in isolation and rely on implicit feature alignment, limiting effectiveness under concurrent challenges. This paper proposes a knowledge-guided two-stage transfer learning framework that employs a lightweight GPT-2-style Transformer with causal self-attention for hierarchical feature extraction from vibration signals, establishing explicit pathways where pre-trained encoder weights and fault prototype embeddings serve as knowledge carriers from multi-source pre-training to target adaptation. The framework addresses the dual-shift challenge through multi-source learning for generalizable representations, prototype-based knowledge modulation for target adaptation, and taxonomy-adaptive classification for seamless transfer across heterogeneous fault categories. Experimental validation on four real-world datasets demonstrates 92.61% average accuracy with only 10% labeled target data, outperforming state-of-the-art methods by 17.24 percentage points, establishing a practical pathway toward cost-effective predictive maintenance in Industry 4.0 applications. 

---
# Bayesian control for coding agents 

**Authors**: Theodore Papamarkou, Vladislav Smirnov, Viktor Mazanov, Artem Vazhentsev, Preslav Nakov, Timothy Baldwin, Artem Shelmanov  

**Link**: [PDF](https://arxiv.org/pdf/2606.24453)  

**Abstract**: Modern coding agents pair LLM generators with various tools, including cheap diagnostics and expensive verifiers. The tool-use decisions are typically governed by orchestrators that often use fixed rules and ignore uncertainty. We formulate orchestration as cost-sensitive sequential hypothesis testing: a Bayesian controller maintains a belief over candidate correctness and dynamically decides whether to gather more evidence, refine the candidate, verify it, or stop. Across six generators and nine coding benchmarks, Bayesian control proves to be most valuable when verification is costly and critics are informative but imperfect. Beyond control, the belief state yields an interpretable correctness score that outperforms token-probability and raw tool-success baselines for uncertainty quantification. 

---
# Age of LLM: A Strategic 1v1 Benchmark for Reasoning, Diplomacy and Reliability of Large Language Models under Fog of War 

**Authors**: Arnaud Ricci  

**Link**: [PDF](https://arxiv.org/pdf/2606.24391)  

**Abstract**: We introduce Age of LLM, a turn-based 1v1 benchmark in which two LLMs face off on a 13x7 grid to destroy the enemy base. Three stressors are deliberate: fog of war, full diplomacy (messages, ceasefires, ultimatums; uranium kept secret), and a reliability dimension where every turn must follow a strict JSON schema and an illegal action is silently discarded. The engine is private and each match uses a fresh random map seed and opponent, mitigating the data contamination that affects public benchmarks. Models receive a (near) rule-only prompt with no build-order advice (two tactical seed phrases were present during data collection; see Section 2.7). We benchmark 15 reasoning models across 54 matches and 5,258 actions. Findings: (1) the nuclear rush dominates (78% on the rules-coherent v0.11+ sub-corpus; 85% corpus-wide) with a sole-launcher signature that is largely mechanical under secret-simultaneous launch rules, not a cognitive deterrence failure; (2) military conquest is rare but faster (12.3 vs 18.9 turns); (3) diplomacy is prolific yet almost never consummated; (4) ~58% of illegal actions are fog/state errors, making the illegal-action rate a measure of belief-tracking; (5) -- the least established, and the only one we label exploratory -- a weak link associates reliability with winning. The corpus is small, unbalanced and not side-swapped, so the ranking is a preliminary descriptive view, not a contribution. Beyond ranking, the turn-by-turn traces of actions and messages make the corpus a lens on how LLMs reason under adversarial uncertainty -- their belief-tracking, spontaneous deception, and per-model cognitive "personas" -- which we frame as a future research direction. We release the replay format, an isometric viewer and all replays; engine source on request. 

---
# ComputeFHE: A Privacy-Preserving General-Purpose Computation Library 

**Authors**: Faris Serdar Tasel, Efe Ciftci  

**Link**: [PDF](https://arxiv.org/pdf/2606.24379)  

**Abstract**: Fully Homomorphic Encryption (FHE) enables computations to be performed directly on encrypted data while preserving data confidentiality. However, its practical applications remain limited by high computational costs and development complexity. This paper presents ComputeFHE, an open-source C++ library that facilitates the development of privacy-preserving applications based on the TFHE cryptosystem. The library provides encrypted integer and fixed-point data types together with arithmetic, logical, comparison, conditional, and oblivious array-access operations which allow developers to implement algorithms using a familiar imperative programming paradigm. ComputeFHE supports both conventional TFHE arithmetic based on standard two-input logic gates and an optimized Arithmetic Logic Unit (ALU) architecture utilizing FHE-friendly logic primitives. Experimental results demonstrate significant reductions in the number of required bootstrapping operations, achieving performance improvements of up to 3.9x for selected operations. In addition, the library includes a simulation mode that enables testing, debugging, and complexity analysis without performing actual cryptographic computations while providing circuit complexity and bootstrapping costs. Built on top of OpenFHE, ComputeFHE offers a practical and accessible framework for developing and evaluating privacy-preserving algorithms and applications. 

---
# PETRA: Transforming Web Text for Petroleum-Engineering Domain Adaptation 

**Authors**: Kirill Dubovikov, Omar El Mansouri, Hachem Madmoun, Yanda Li, Sandeep Kumar, Aya El Mir, Supriyo Ghosh, Writabrata Bhattacharya, Adrian Garcia-Garcia, Onkar Pandit, Sunil Kumar Sahu, Federico Castanedo, Larry Murray, Martin Takac, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2606.24346)  

**Abstract**: Petroleum-engineering search exposes a supervision gap for strong general retrievers: relevant evidence exists in public web text, but domain relevance labels are scarce. To address this gap, we propose PETRA, a large-scale Petroleum Engineering Text for Retrieval Adaptation dataset and pipeline that converts noisy public web data into a curated domain corpus and synthetic supervision for dense retrieval and reranking. PETRA contains 1.36M curated chunks, approximately 2B token equivalents, $\approx$859k, embedding training rows from $\approx$224k anchors, and roughly 400k teacher-scored reranker candidate rows. Its construction combines high-recall energy-domain curation, an energy-domain classifier with 98.4% test accuracy, chunk-grounded query generation, LLM-written hard negatives, and retrieval-mined candidate lists. PETRA improves first-stage in-domain Normalized Discounted Cumulative Gain (nDCG) from 0.703 to 0.763 through score fusion. Reranker adaptation improves the public Earth Science benchmark by 44% relative and a six-task reasoning-intensive panel by 23%. Failed training recipes show that high train-holdout accuracy on synthetic labels does not predict retrieval gains; retrieval-mined data helps only after being repackaged as teacher-scored candidate lists sampled from the inference-time candidate distribution. 

---
# Dialogue to Discovery: Attribute-Aware Preference Elicitation for Conversational Product Search Assistants 

**Authors**: Sarthak Harne, Natwar Modani, Debabrata Mahapatra, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2606.24194)  

**Abstract**: Conversational product search assistants offer a more expressive, natural, and interactive alternative to traditional keyword-based product search. With limited screen space, showing only a few items increases the need for precise preference elicitation, which can prolong conversations, leading to user frustration and session abandonment. Conversely, rushing to recommend items without a clear understanding of preferences risks poor matches and a degraded user experience. We present Dialogue to Discovery (D2D), an attribute-oriented preference elicitation framework that dynamically exploits the structure of product attributes to efficiently steer conversations toward the user's desired item. D2D adaptively prioritizes the most informative queries and strategically times product recommendations, reducing premature or off-target suggestions that harm engagement. To evaluate D2D, we curate three datasets from the Amazon Reviews corpus. In simulated conversations modelled using a multi-factor utilitarian patience framework, D2D achieves a 22.2-29.9% improvement in target-finding accuracy, 6.6-16.1% reduction in abandonment, and 27.5% shorter average conversations over the state-of-the-art baselines. A complementary user study further confirms significant gains in both user satisfaction and perceived efficiency. 

---
# Co-occurring associated retained concepts in Diffusion Unlearning 

**Authors**: Miso Kim, Georu Lee, Yunji Kim, Hoki Kim, Jinseong Park, Woojin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.24192)  

**Abstract**: Unlearning has emerged as a key technique to mitigate harmful content generation in diffusion models. However, existing methods often remove not only the target concept, but also benign co-occurring concepts. As illustrated in Fig.1, unlearning nudity can unintentionally suppress the concept of person, preventing a model from generating images with person. We define these undesirably suppressed co-occurring concepts that must be preserved CARE (Co-occurring Associated REtained concepts). Then, we introduce the CARE score, a general metric that directly quantifies their preservation across unlearning tasks. With this foundation, we propose ReCARE (Robust erasure for CARE), a framework that explicitly safeguards CARE while erasing only the target concept. ReCARE automatically constructs the CARE-set, a curated vocabulary of benign co-occurring tokens extracted from target images, and leverages this vocabulary during training for stable unlearning. Extensive experiments across various target concepts (Nudity, Van Gogh style, and Tench object) demonstrate that ReCARE achieves overall state-of-the-art performance in balancing robust concept erasure, overall utility, and CARE preservation. 

---
# Agon: An Autonomous Large-Scale Omnidisciplinary Research System Built on Prompt Economy 

**Authors**: Youran Sun, Xingyu Ren, Chugang Yi, Jiaxuan Guo, Kejia Zhang, Jianda Du, Haizhao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24177)  

**Abstract**: Large language models are making research production scalable, shifting the bottleneck from producing artifacts to judging claims. We present \textsc{Agon}, a research orchestrator that validates what can be checked inside the workflow and leaves the remaining judgments to human scientists. \textsc{Agon} is built on six design principles: Prompt Economy, Future-Facing, Minimal Prompts, OmniDisciplinary, Massive Parallelism, and Zero-Code. We ran \textsc{Agon} across domains for 444 iterations of Prompt Economy loops, using only small starting topics and no human-written experimental code. These deployments demonstrate scalability while exposing new classes of failure. We organize these failures into a taxonomy along severity, fixability, visibility, and capability locus. The taxonomy separates failures the loops can see and fix from those that require human judgment. Together, these results show that \textsc{Agon} is pushing research toward a new paradigm: machine scales, human steers. 

---
# CORE-BREW: LLR-Based Soft Decoding for Robust Multi-Bit LLM Watermarking 

**Authors**: Joeun Kim, HoEun Kim, Young-Sik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.24163)  

**Abstract**: Reliable provenance for LLM outputs requires multi-bit watermarks that remain robust under editing while maintaining strict false-positive control. Existing ECC-based LLM watermarks rely largely on hard-decision decoding, discarding token-level reliability information. We propose CORE-BREW, a Constant-hit-Rate Embedding extension of block-wise BREW for robust multi-bit watermarking. CORE-BREW calibrates the watermark channel by targeting a fixed hit rate p-star, yielding closed-form per-token log-likelihood ratios (LLRs) for principled soft-decision decoding. It supports two detection modes: Strict-Safe, which preserves the bounded-distance designated-codeword acceptance region, and FPR-Calibrated, which uses likelihood-based scoring and lightweight list decoding to characterize the FPR-TPR trade-off. Experiments on open-source LLMs under token-level edits and paraphrasing demonstrate improved low-FPR discrimination and robustness over prior multi-bit watermarking baselines while maintaining comparable semantic quality. 

---
# Progressive Alignment Objectives for Aligner-Encoder based ASR 

**Authors**: Jaeyong Lee, Masato Mimura, Takafumi Moriya  

**Link**: [PDF](https://arxiv.org/pdf/2606.24147)  

**Abstract**: Aligner-Encoders are recently proposed seq2seq end-to-end ASR models that replace decoder attention by predicting the uth token directly from the u-th encoder position, so the encoder must learn the alignment internally without cross-attention or a transducer lattice. In practice, this alignment often forms abruptly in the upper layers, making training sensitive and brittle on long utterances. We propose InterAligner, which adds an intermediate Aligner objective so alignment can form progressively across depth, together with an intermediate CTC loss (InterCTC) to stabilize optimization. On LibriSpeech with a 17-layer Conformer, a final-only Aligner reaches 5.0/7.8 WER (test-clean/other). InterCTC improves to 3.4/6.0, and InterAligner further reduces WER to 3.1/5.6 with the largest gains on long utterances. 

---
# Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning 

**Authors**: Chenhao Dang, Jing Ma, Mingjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.24133)  

**Abstract**: The composition of training data, governed by the diversity of sources and their mixing strategy, is a cornerstone of Large Language Model (LLM) pre-training. Online Data Mixing (ODM), the technique of adaptively adjusting data mixtures during training, has emerged as a promising direction to improve efficiency. However, existing methods are constrained by their reliance on a singular optimization perspective, which fundamentally overlooks the need for complex LLM pre-training to consider the dynamic data composition from multiple dimensions. To overcome this limitation, we introduce the Holistic Data Scheduler (HDS), a novel online data mixing framework. HDS formulates the data scheduling challenge as a reinforcement learning problem in a continuous control space and leverages the Soft Actor-Critic (SAC) algorithm for its stability and sample efficiency in exploring the high-dimensional policy space. At the core of HDS lies a novel multi-objective, holistic reward function that integrates three critical perspectives: a data-driven reward for quality, a loss-driven reward capturing inter-domain influence, and a model-driven reward based on weight norms. To validate our design and determine its optimal configuration, we conducted systematic experiments on LLMs of various sizes. On The Pile benchmark, HDS reaches the final validation perplexity of the next best method with 44% fewer training iterations. Furthermore, it achieves a 7.2% improvement on the MMLU 0-shot task along with consistent gains on other benchmarks, showcasing its ability to enhance both training efficiency and final model capability. 

---
# When Top-1 Fails: Calibrating LoRA Monitors for Masked Diffusion LMs 

**Authors**: Lucky Verma, Pratik Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2606.24119)  

**Abstract**: Discrete diffusion language model (DLM) fine-tuning inherits inexpensive diagnostics from denoising-time confidence monitors, but their PEFT-training meaning is untested. We test top-1 argmax concentration as a collapse warning. Across 816 LoRA/PEFT configurations from three DLM families, the warning fires for every configuration while logs record 0/816 actual collapses at the 200 step horizon, giving zero precision. The cause is pre-equilibrium saturation: top-1 concentration is already high before optimization and quickly becomes insensitive to final training stability. We then evaluate max LoRA gradient norm, a parameter-side signal that samples gradient routing rather than token concentration. On a pooled held-out LLaDA-family split, a train-optimized threshold identifies top-decile final-loss configurations with precision 0.68 and F1=0.79, above the all-positive top-1 baseline even at the lower split-bootstrap confidence bound. Autoregressive controls and cross-family threshold failures bound the result to short-horizon DLM-LoRA inspection rather than a universal collapse detector. Workflow: drop top-1 as a PEFT alarm, log max-gradient early in training, and calibrate thresholds per DLM family before routing runs for inspection. 

---
# Exploring Academic Influence of Algorithms by Co-occurrence Network Based on Full-text of Academic Papers 

**Authors**: Yuzhuo Wang, Chengzhi Zhang, Min Song, Seong Deok Kim, Youngsoo Ko, Juhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.24099)  

**Abstract**: Algorithms have become central to scientific research in the era of artificial intelligence (AI). Although algorithm mentions in papers are often used to indicate popularity and influence, existing studies usually evaluate individual algorithms in isolation and pay limited attention to the collective influence formed through their interconnections. This study constructs large-scale algorithm co-occurrence networks in natural language processing (NLP) based on the full text of academic papers and investigates algorithm influence from a network perspective. Using deep learning models, we extract algorithm entities and build overall, cumulative, and annual co-occurrence networks. We analyze their structural characteristics and apply multiple centrality measures to assess the group influence of algorithms across the whole field and over time. The results show that algorithm networks display typical features of complex networks, with increasingly dense connections developing over approximately two decades. Classic, high-performing algorithms and those located at the intersections of different research periods tend to have high popularity, control, centrality, and balanced influence. When the influence of an algorithm declines, it usually loses its core network position first, followed by weaker associations with other algorithms. This study is the first large-scale analysis of algorithm co-occurrence networks. Covering more than four decades of academic publications, it provides a temporal and structural view of algorithm influence and offers a foundation for future research on networks linking algorithms, scholars, and tasks. 

---
# Blockwise Policy-Drift Gating for On-Policy Distillation 

**Authors**: Liwen Zheng, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24084)  

**Abstract**: On-policy distillation (OPD) trains a student policy using teacher signals computed on trajectories sampled by the student itself. Recent work shows that sampled-token OPD can be fragile on long-horizon reasoning tasks and that local teacher-support matching is a simple and effective repair. This paper introduces blockwise policy-drift gating, a lightweight student-only old-current drift controller for OPD under rollout reuse. The method computes log-probability shifts between the behavior student and the current student on the sampled token path, aggregates these shifts over fixed blocks or spans, and uses the resulting detached, mean-normalized gates to reweight OPD position losses. It does not change teacher targets, teacher top-K supports, or the rollout policy. In a six-variant Qwen3 math reasoning benchmark with a uniform 200-step training budget for all trained variants, we use pass@8 as the primary problem-level solve-rate metric. Fixed 64-token block gating improves sampled-token OPD mean pass@8 from 0.4978 to 0.5160 across AIME24, AIME25, MATH500, and AMC23. On Teacher-TopK/LSM, Block64 gives the best four-benchmark mean pass@8 among trained students. The results identify local old-current policy drift as a practical control signal for reused OPD rollouts and motivate block-level gating as a simple default for improving solve-rate robustness. 

---
# VieSpeaker: A Large-Scale Vietnamese Speaker Recognition Dataset Beyond Visual Dependency 

**Authors**: Viet Hoang Pham, Tran Trung Nguyen, Bao Thu Ho, Phuong Tuan Dat, Thi Thu Trang Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2606.24066)  

**Abstract**: Speaker recognition has advanced rapidly with large-scale training datasets, yet Vietnamese remains under-resourced, with existing corpora limited in scale and acoustic diversity. Most large-scale datasets rely on facial cues to link speech with speaker identities, restricting data collection to recordings where speakers appear on camera. We propose a face-independent dataset construction pipeline and introduce VieSpeaker, a large-scale Vietnamese speaker recognition dataset. Our approach leverages textual metadata and large language model reasoning to infer speaker identities from transcripts and contextual information. VieSpeaker contains approximately 902 hours of speech from 4,715 speakers. Experiments show that models trained on VieSpeaker achieve improved robustness and generalization compared to existing Vietnamese datasets. This work demonstrates the feasibility of face-independent dataset construction and provides a new direction for building large-scale speech resources. 

---
# RoPE-Aware Bit Allocation for KV-Cache Quantization 

**Authors**: Fengfeng Liang, Yuechen Zhang, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.24033)  

**Abstract**: Existing low-bit KV-cache quantizers often treat each cached key as a flat vector. Under RoPE, however, a key's contribution to a future attention logit decomposes into a position-dependent sum over two-dimensional frequency blocks. This makes key-cache quantization a block-wise bit-allocation problem: high-energy RoPE blocks are more sensitive to quantization error and should receive more bits. We introduce Block-GTQ, a RoPE-aware bit allocator for key-cache quantization built on TurboQuant-MSE(TQ-MSE). For each layer and KV head, Block-GTQ computes a label-free energy score for each RoPE block and greedily allocates integer bit widths by marginal gain. Under matched K/V bit budgets, Block-GTQ better preserves RoPE query-key logits on a ten-model diagnostic panel, cutting per-layer MAE by 32-80% at 2 and 3 b/dim K-only quantization and winning all 367/367 layer comparisons against uniform TQ-MSE. These fidelity gains translate to stronger downstream long-context retrieval, understanding, and reasoning. At K2V2 on Llama-3.1-8B-Instruct, Block-GTQ raises the six-task NIAH average from 70.6 to 97.4, and the LongBench-EN average from 36.87 to 53.31. On AIME 2024/2025 with DeepSeek-R1-Distill-Qwen-7B, without an fp16 recent-key buffer, Block-GTQ at K3V2 scores 51.7/37.5, close to fp16's 54.2/37.9, whereas uniform TQ-MSE collapses to 0.0/0.0. We further implement a packed-cache serving path. On a single H800 GPU with Qwen2.5-3B-Instruct, packed K3V3 achieves 3.24x KV-cache compression with fp16-comparable quality, runs 1.34x faster than fp16 FlashAttention2 at 128K context, reduces peak memory from 56.31 GB to 19.85 GB, and remains feasible at 256K and 512K where fp16 OOMs. Code is available at this https URL. 

---
# Reinforcement Learning Towards Broadly and Persistently Beneficial Models 

**Authors**: Akshay V. Jagadeesh, Rahul K. Arora, Khaled Saab, Ali Malik, Mikhail Trofimov, Foivos Tsimpourlas, Johannes Heidecke, Karan Singhal  

**Link**: [PDF](https://arxiv.org/pdf/2606.24014)  

**Abstract**: As AI systems are deployed across increasingly diverse and high-stakes settings, model alignment must generalize beyond the tasks and domains seen during training. This is especially important for reinforcement learning (RL), which can introduce unexpected misalignment through reward hacking, deception, or other unintended strategies. We study whether RL on beneficial behavior, instantiated in realistic domains, can produce broad and persistent alignment generalization beyond the training distribution. We construct a dataset of realistic situations designed to measure and train beneficial traits, such as truthfulness, fairness, risk awareness, and corrigibility, spanning varied domains, including health, science, and education. We then train models with RL on this dataset and evaluate them on more than 50 independent benchmarks of alignment and beneficial behavior. Compared to a compute-matched baseline, beneficial trait RL improves performance on over 80% of these out-of-distribution benchmarks. We observe substantial out-of-distribution alignment transfer: a beneficial-behavior RL intervention entirely limited to one domain, health, produces broad improvements on non-health alignment evaluations, including reduced reward hacking, deception, and general misalignment. Finally, we study alignment persistence: whether behavior remains robustly aligned under attempts to steer models towards misalignment. Models trained with beneficial trait RL show improved persistence, including greater resistance to adversarial prompting and harmful finetuning; further work is required to isolate the sources of these effects. These results suggest that RL to reinforce beneficial behavior in realistic domains can produce models that are more robustly aligned with human flourishing. 

---
# Neuro-Symbolic Drive: Rule-Grounded Faithful Reasoning for Driving VLAs 

**Authors**: Xiangbo Gao, Xiukun Huang, Boyu Lu, Junge Zhang, Mengjie Mao, Jiachen Li, Wei Xiong, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2606.23938)  

**Abstract**: Driving VLA models incorporating Chain-of-Thought (CoT) reasoning are attractive because they leverage pretrained VLM representations and expose intermediate decisions in natural language, yet current rationales often lack the step-by-step decision semantics needed to keep the rationale causally connected to the planned motion. We introduce Neuro-Symbolic Drive, a neuro-symbolic driving framework that supervises a driving VLA with rule-grounded reasoning traces extracted directly from classical rule-based planners. Our key observation is that rule-based planners are symbolic AI systems that already function as executable reasoning engines: they reason about active safety constraints, search over candidate maneuvers, and select a final trajectory. We instrument these planners in simulation to capture both the executed trajectory and the internal decision trace at each rule-evaluation step. Each trace is serialized into structured rule-grounded reasoning and paired with the trajectory to fine-tune Qwen3.5-4B as a driving VLA. Because these traces are derived directly from the planner states that determine the action, they ensure reasoning is structurally coupled to motion generation by construction, rather than by post-hoc alignment. On our simulator-generated benchmark, detailed rule-grounded reasoning reduces ADE@3s from 0.47 to 0.26 and miss rate from 8.30% to 6.40% under three-camera perception, and from 0.54 to 0.26 and 10.13% to 5.99% under eight-camera perception. Neuro-Symbolic Drive thus converts neuro-symbolic planning logic into structured supervision. Code base: this https URL. 

---
# Mind the Heads: Topological Representation Alignment for Multimodal LLMs 

**Authors**: Davide Caffagni, Alberto Compagnoni, Federico Melis, Sara Sarto, Pier Luigi Dovesi, Mark Granroth-Wilding, Marcella Cornia, Lorenzo Baraldi  

**Link**: [PDF](https://arxiv.org/pdf/2606.23885)  

**Abstract**: Representation alignment has emerged as an effective approach to improve Multimodal Large Language Models (MLLMs) by regularizing their internal representations toward those of an external vision encoder. However, existing methods typically align a fixed layer of the language backbone, overlooking the fine-grained structure of Transformer models. In this work, we propose Head-Wise Representation Alignment (HeRA), a method that enforces cross-modal alignment at the level of individual attention heads. Our approach is grounded in the Platonic Representation Hypothesis, focusing on preserving the topological structure of representations (i.e., their local neighborhood relationships) across modalities. Following the Mutual K-Nearest Neighbor (MKNN) alignment metric, we introduce a contrastive objective that acts as a differentiable proxy for matching local structures. HeRA applies this objective during multimodal training to specific attention heads in the LLM, selected by their alignment score according to the MKNN metric. Counterintuitively, we find that aligning the least aligned heads yields the largest gains. Extensive evaluations across multiple MLLMs and 18 benchmarks demonstrate that HeRA consistently improves performance on challenging vision-centric tasks and serves as an effective regularizer against visual hallucinations by naturally curbing the over-reliance on linguistic priors. Our code is publicly released. 

---
# ESBMC-PLC+: A Unified IEC~61131-3 Formal Verification Framework as a PLCverif Successor 

**Authors**: Pierre Dantas, Lucas Cordeiro, Waldir Junior  

**Link**: [PDF](https://arxiv.org/pdf/2606.23870)  

**Abstract**: PLCverif is the most mature open-source platform for PLC formal verification, developed at CERN and in production use since 2019. Yet it has two fundamental limitations: no support for Ladder Diagram (LD) programs, the dominant PLC notation, and reliance on CBMC as its primary backend, which restricts verification to bounded proofs. The PLCverif authors themselves identified ESBMC as the appropriate backend improvement. Prior work established ESBMC-PLC (a textual LD frontend with k-induction) and ESBMC-GraphPLC (graphical PLCopen XML support); together, they cover LD with unbounded proofs but not Structured Text (ST), and graphical LD with timer/counter function blocks remains unverifiable. This paper presents ESBMC-PLC+, a unified framework that closes both gaps: (1) an ST/SCL frontend via the MATIEC IEC 61131-3 compiler, routing C-compiled ST to ESBMC with nondeterministic input modeling and YAML property injection; (2) function block state semantics for graphical LD, extending the DFS resolver to model TON/TOF/TP timers, CTU/CTD counters, and R_TRIG/F_TRIG edge triggers as persistent scan-cycle state variables in the GOTO IR. ESBMC-PLC+ is the first open-source PLC verification framework to support all three major IEC 61131-3 input formats via a single ESBMC backend, enabling k-induction-unbounded safety proofs. A feature comparison with PLCverif and experimental evaluation on 8 benchmark programs, including programs with up to 8 integer timers, shows that ESBMC-PLC+ matches PLCverif's input coverage while providing stronger guarantees. Against nuXmv's BDD backend, ESBMC-PLC+ is 400-2,000x faster on timer programs and completes proofs where nuXmv BDD times out at 120s. 

---
# From Task-Guided Conversational Graphs to Goal-Oriented Dialogue Runtimes 

**Authors**: Mariano Garralda-Barrio  

**Link**: [PDF](https://arxiv.org/pdf/2606.23797)  

**Abstract**: Graph and multi-agent orchestration frameworks make production large language model (LLM) workflows practical, but they do not by themselves solve conversational continuity when users maintain several interdependent objectives. This conceptual systems paper focuses on the high-complexity end of that design space, where goals can be suspended, resumed, revised, and invalidated by actions in other goals. We introduce the Goal-Oriented Dialogue Runtime (GODR), a framework-neutral design pattern that treats goals, task frames, lifecycle state, invalidation rules, and resumption contracts as first-class runtime objects while delegating bounded execution to graph runtimes, agents, tools, or application programming interfaces (APIs). GODR is not proposed as a replacement for workflow graphs in simple guided processes; it is intended for complex, multi-domain, interruptible conversations where objective continuity cannot be recovered reliably from agent identity, chat history, or execution-graph position alone. The paper formalizes the problem, proposes runtime objects and architecture-selection criteria, and frames evaluation as an agenda for future empirical validation rather than as a measured performance claim. 

---
# EvidenceLens: A Claim-Evidence Matrix for Auditing Financial Question Answering 

**Authors**: Fengchen Gu, Xiaotian Ren, Zhengyong Jiang, Zhilu Zhang, Ángel F. García-Fernández, Angelos Stefanidis, Mian Zhou, Huakang Li, Jionglong Su  

**Link**: [PDF](https://arxiv.org/pdf/2606.23724)  

**Abstract**: Large language models are increasingly used to answer questions over annual reports, earnings decks, and analyst notes, yet their outputs remain difficult to verify in high-stakes financial workflows. A fluent answer can blend directly grounded statements, weak synthesis, and unsupported claims across narrative text, tables, and charts. We present EvidenceLens, a visual analytics prototype that treats financial question answering as a claim-evidence alignment problem. The system decomposes an answer into atomic claims, summarizes support composition and confidence, support gaps, and coordinates claim-level inspection with source passages, table cells, and chart regions. Its core visual representation is a multimodal claim-evidence matrix that makes coverage, contradiction, and modality imbalance immediately visible. To support reproducibility, we also specify a JSON-based artifact schema, a lightweight multimodal alignment pipeline, and a deterministic review-priority ranking that maps backend signals into an auditable visual structure. Through representative report-auditing scenarios, we show how EvidenceLens helps analysts distinguish grounded claims from overconfident synthesis that conventional chat interfaces flatten. 

---
