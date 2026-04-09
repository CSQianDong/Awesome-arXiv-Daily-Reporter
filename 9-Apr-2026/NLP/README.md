# Personalized RewardBench: Evaluating Reward Models with Human Aligned Personalization 

**Authors**: Qiyao Ma, Dechen Gao, Rui Cai, Boqi Zhao, Hanchu Zhou, Junshan Zhang, Zhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07343)  

**Abstract**: Pluralistic alignment has emerged as a critical frontier in the development of Large Language Models (LLMs), with reward models (RMs) serving as a central mechanism for capturing diverse human values. While benchmarks for general response quality are prevalent, evaluating how well reward models account for individual user preferences remains an open challenge. To bridge this gap, we introduce Personalized RewardBench, a novel benchmark designed to rigorously assess reward models' capacity to model personalized preferences. We construct chosen and rejected response pairs based on strict adherence to (or violation of) user-specific rubrics, ensuring that preference distinctions are uniquely tailored to the individual. In particular, human evaluations confirm that the primary discriminative factor between pairs is strictly personal preference, with both responses maintaining high general quality (e.g., correctness, relevance and helpfulness). Extensive testing reveals that existing state-of-the-art reward models struggle significantly with personalization, peaking at an accuracy of just 75.94%. Crucially, because an effective reward model benchmark should predict a reward model's performance on downstream tasks, we conduct experiments demonstrating that our benchmark exhibits a significantly higher correlation with downstream performance in both Best-of-N (BoN) sampling and Proximal Policy Optimization (PPO) compared to existing baselines. These findings establish Personalized RewardBench as a robust and accurate proxy for evaluating reward models' performance in downstream applications. 

---
# Evaluating In-Context Translation with Synchronous Context-Free Grammar Transduction 

**Authors**: Jackson Petty, Jaulie Goe, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07320)  

**Abstract**: Low-resource languages pose a challenge for machine translation with large language models (LLMs), which require large amounts of training data. One potential way to circumvent this data dependence is to rely on LLMs' ability to use in-context descriptions of languages, like textbooks and dictionaries. To do so, LLMs must be able to infer the link between the languages' grammatical descriptions and the sentences in question. Here we isolate this skill using a formal analogue of the task: string transduction based on a formal grammar provided in-context. We construct synchronous context-free grammars which define pairs of formal languages designed to model particular aspects of natural language grammar, morphology, and written representation. Using these grammars, we measure how well LLMs can translate sentences from one formal language into another when given both the grammar and the source-language sentence. We vary the size of the grammar, the lengths of the sentences, the syntactic and morphological properties of the languages, and their written script. We note three key findings. First, LLMs' translation accuracy decreases markedly as a function of grammar size and sentence length. Second, differences in morphology and written representation between the source and target languages can strongly diminish model performance. Third, we examine the types of errors committed by models and find they are most prone to recall the wrong words from the target language vocabulary, hallucinate new words, or leave source-language words untranslated. 

---
# OpenSpatial: A Principled Data Engine for Empowering Spatial Intelligence 

**Authors**: Jianhui Liu, Haoze Sun, Wenbo Li, Yanbing Zhang, Rui Yang, Zhiliang Zhu, Yijun Yang, Shenghe Zheng, Nan Jiang, Jiaxiu Jiang, Haoyang Huang, Tien-Tsin Wong, Nan Duan, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2604.07296)  

**Abstract**: Spatial understanding is a fundamental cornerstone of human-level intelligence. Nonetheless, current research predominantly focuses on domain-specific data production, leaving a critical void: the absence of a principled, open-source engine capable of fully unleashing the potential of high-quality spatial data. To bridge this gap, we elucidate the design principles of a robust data generation system and introduce OpenSpatial -- an open-source data engine engineered for high quality, extensive scalability, broad task diversity, and optimized efficiency. OpenSpatial adopts 3D bounding boxes as the fundamental primitive to construct a comprehensive data hierarchy across five foundational tasks: Spatial Measurement (SM), Spatial Relationship (SR), Camera Perception (CP), Multi-view Consistency (MC), and Scene-Aware Reasoning (SAR). Leveraging this scalable infrastructure, we curate OpenSpatial-3M, a large-scale dataset comprising 3 million high-fidelity samples. Extensive evaluations demonstrate that versatile models trained on our dataset achieve state-of-the-art performance across a wide spectrum of spatial reasoning benchmarks. Notably, the best-performing model exhibits a substantial average improvement of 19 percent, relatively. Furthermore, we provide a systematic analysis of how data attributes influence spatial perception. By open-sourcing both the engine and the 3M-scale dataset, we provide a robust foundation to accelerate future research in spatial intelligence. 

---
# Why teaching resists automation in an AI-inundated era: Human judgment, non-modular work, and the limits of delegation 

**Authors**: Songhee Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.07285)  

**Abstract**: Debates about artificial intelligence (AI) in education often portray teaching as a modular and procedural job that can increasingly be automated or delegated to technology. This brief communication paper argues that such claims depend on treating teaching as more separable than it is in practice. Drawing on recent literature and empirical studies of large language models and retrieval-augmented generation systems, I argue that although AI can support some bounded functions, instructional work remains difficult to automate in meaningful ways because it is inherently interpretive, relational, and grounded in professional judgment. More fundamentally, teaching and learning are shaped by human cognition, behavior, motivation, and social interaction in ways that cannot be fully specified, predicted, or exhaustively modeled. Tasks that may appear separable in principle derive their instructional value in practice from ongoing contextual interpretation across learners, situations, and relationships. As long as educational practice relies on emergent understanding of human cognition and learning, teaching remains a form of professional work that resists automation. AI may improve access to information and support selected instructional activities, but it does not remove the need for human judgment and relational accountability that effective teaching requires. 

---
# A Systematic Study of Retrieval Pipeline Design for Retrieval-Augmented Medical Question Answering 

**Authors**: Nusrat Sultana, Abdullah Muhammad Moosa, Kazi Afzalur Rahman, Sajal Chandra Banik  

**Link**: [PDF](https://arxiv.org/pdf/2604.07274)  

**Abstract**: Large language models (LLMs) have demonstrated strong capabilities in medical question answering; however, purely parametric models often suffer from knowledge gaps and limited factual grounding. Retrieval-augmented generation (RAG) addresses this limitation by integrating external knowledge retrieval into the reasoning process. Despite increasing interest in RAG-based medical systems, the impact of individual retrieval components on performance remains insufficiently understood. This study presents a systematic evaluation of retrieval-augmented medical question answering using the MedQA USMLE benchmark and a structured textbook-based knowledge corpus. We analyze the interaction between language models, embedding models, retrieval strategies, query reformulation, and cross-encoder reranking within a unified experimental framework comprising forty configurations. Results show that retrieval augmentation significantly improves zero-shot medical question answering performance. The best-performing configuration was dense retrieval with query reformulation and reranking achieved 60.49% accuracy. Domain-specialized language models were also found to better utilize retrieved medical evidence than general-purpose models. The analysis further reveals a clear tradeoff between retrieval effectiveness and computational cost, with simpler dense retrieval configurations providing strong performance while maintaining higher throughput. All experiments were conducted on a single consumer-grade GPU, demonstrating that systematic evaluation of retrieval-augmented medical QA systems can be performed under modest computational resources. 

---
# ClickGuard: A Trustworthy Adaptive Fusion Framework for Clickbait Detection 

**Authors**: Chhavi Dhiman, Naman Chawla, Riya Dhami, Gaurav Kumar, Ganesh Naik  

**Link**: [PDF](https://arxiv.org/pdf/2604.07272)  

**Abstract**: The widespread use of clickbait headlines, crafted to mislead and maximize engagement, poses a significant challenge to online credibility. These headlines employ sensationalism, misleading claims, and vague language, underscoring the need for effective detection to ensure trustworthy digital content. The paper introduces, ClickGuard: a trustworthy adaptive fusion framework for clickbait detection. It combines BERT embeddings and structural features using a Syntactic-Semantic Adaptive Fusion Block (SSAFB) for dynamic integration. The framework incorporates a hybrid CNN-BiLSTM to capture patterns and dependencies. The model achieved 96.93% testing accuracy, outperforming state-of-the-art approaches. The model's trustworthiness is evaluated using LIME and Permutation Feature Importance (PFI) for interpretability and perturbation analysis. These methods assess the model's robustness and sensitivity to feature changes by measuring the average prediction variation. Ablation studies validated the SSAFB's effectiveness in optimizing feature fusion. The model demonstrated robust performance across diverse datasets, providing a scalable, reliable solution for enhancing online content credibility by addressing syntactic-semantic modelling challenges. Code of the work is available at: this https URL 

---
# Joint Optimization of Reasoning and Dual-Memory for Self-Learning Diagnostic Agent 

**Authors**: Bingxuan Li, Simo Du, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.07269)  

**Abstract**: Clinical expertise improves not only by acquiring medical knowledge, but by accumulating experience that yields reusable diagnostic patterns. Recent LLMs-based diagnostic agents have shown promising progress in clinical reasoning for decision support. However, most approaches treat cases independently, limiting experience reuse and continual adaptation. We propose SEA, a self-learning diagnostic agent with cognitively inspired dual-memory module. We design a reinforcement training framework tailored to our designed agent for joint optimization of reasoning and memory management. We evaluate SEA in two complementary settings. On standard evaluation with MedCaseReasoning dataset, SEA achieves 92.46% accuracy, outperforming the strongest baseline by +19.6%, demonstrating the benefit of jointly optimizing reasoning and memory. On the long-horizon with ER-Reason dataset, SEA attains the best final accuracy (0.7214) and the largest improvement (+0.35 Acc@100), while baseline methods show limited or unstable gains. Expert evaluation further indicates that rules consolidated from SEA show strong clinical correctness, usefulness and trust, suggesting that the induced rules in dual-memory module are reliable and practically meaningful. Overall, SEA improves both diagnostic reasoning ability and continual learning by effectively transforming experience into reusable knowledge. 

---
# Efficient Learned Data Compression via Dual-Stream Feature Decoupling 

**Authors**: Huidong Ma, Xinyan Shi, Hui Sun, Xiaofei Yue, Xiaoguang Liu, Gang Wang, Wentong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2604.07239)  

**Abstract**: While Learned Data Compression (LDC) has achieved superior compression ratios, balancing precise probability modeling with system efficiency remains challenging. Crucially, uniform single-stream architectures struggle to simultaneously capture micro-syntactic and macro-semantic features, necessitating deep serial stacking that exacerbates latency. Compounding this, heterogeneous systems are constrained by device speed mismatches, where throughput is capped by Amdahl's Law due to serial processing. To this end, we propose a Dual-Stream Multi-Scale Decoupler that disentangles local and global contexts to replace deep serial processing with shallow parallel streams, and incorporate a Hierarchical Gated Refiner for adaptive feature refinement and precise probability modeling. Furthermore, we design a Concurrent Stream-Parallel Pipeline, which overcomes systemic bottlenecks to achieve full-pipeline parallelism. Extensive experiments demonstrate that our method achieves state-of-the-art performance in both compression ratio and throughput, while maintaining the lowest latency and memory usage. The code is available at this https URL. 

---
# LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics 

**Authors**: Kosmas Pinitas, Ilias Maglogiannis  

**Link**: [PDF](https://arxiv.org/pdf/2604.07193)  

**Abstract**: Predicting affect in unconstrained environments remains a fundamental challenge in human-centered AI. While deep neural embeddings dominate contemporary approaches, they often lack interpretability and limit expert-driven refinement. We propose a novel framework that uses Language Models (LMs) as semantic context conditioners over handcrafted affect descriptors to model changes in Valence and Arousal. Our approach begins with interpretable facial geometry and acoustic features derived from structured domain knowledge. These features are transformed into symbolic natural-language descriptions encoding their affective implications. A pretrained LM processes these descriptions to generate semantic context embeddings that act as high-level priors over affective dynamics. Unlike end-to-end black-box pipelines, our framework preserves feature transparency while leveraging the contextual abstraction capabilities of LMs. We evaluate the proposed method on the Aff-Wild2 and SEWA datasets for affect change prediction. Experimental results show consistent improvements in accuracy for both Valence and Arousal compared to handcrafted-only and deep-embedding baselines. Our findings demonstrate that semantic conditioning enables interpretable affect modelling without sacrificing predictive performance, offering a transparent and computationally efficient alternative to fully end-to-end architectures 

---
# Agent-Driven Corpus Linguistics: A Framework for Autonomous Linguistic Discovery 

**Authors**: Jia Yu, Weiwei Yu, Pengfei Xiao, Fukun Xing  

**Link**: [PDF](https://arxiv.org/pdf/2604.07189)  

**Abstract**: Corpus linguistics has traditionally relied on human researchers to formulate hypotheses, construct queries, and interpret results - a process demanding specialized technical skills and considerable time. We propose Agent-Driven Corpus Linguistics, an approach in which a large language model (LLM), connected to a corpus query engine via a structured tool-use interface, takes over the investigative cycle: generating hypotheses, querying the corpus, interpreting results, and refining analysis across multiple rounds. The human researcher sets direction and evaluates final output. Unlike unconstrained LLM generation, every finding is anchored in verifiable corpus evidence. We treat this not as a replacement for the corpus-based/corpus-driven distinction but as a complementary dimension: it concerns who conducts the inquiry, not the epistemological relationship between theory and data. We demonstrate the framework by linking an LLM agent to a CQP-indexed Gutenberg corpus (5 million tokens) via the Model Context Protocol (MCP). Given only "investigate English intensifiers," the agent identified a diachronic relay chain (so+ADJ > very > really), three pathways of semantic change (delexicalization, polarity fixation, metaphorical constraint), and register-sensitive distributions. A controlled baseline experiment shows that corpus grounding contributes quantification and falsifiability that the model cannot produce from training data alone. To test external validity, the agent replicated two published studies on the CLMET corpus (40 million tokens) - Claridge (2025) and De Smet (2013) - with close quantitative agreement. Agent-driven corpus research can thus produce empirically grounded findings at machine speed, lowering the technical barrier for a broader range of researchers. 

---
# Dynamic Context Evolution for Scalable Synthetic Data Generation 

**Authors**: Ryan Lingo, Rajeev Chhajer  

**Link**: [PDF](https://arxiv.org/pdf/2604.07147)  

**Abstract**: Large language models produce repetitive output when prompted independently across many batches, a phenomenon we term cross-batch mode collapse: the progressive loss of output diversity when a language model is prompted repeatedly without access to its prior generations. Practitioners have long mitigated this with ad hoc deduplication and seed rotation, but no principled framework exists. We introduce Dynamic Context Evolution (DCE), comprising three mechanisms: (1) verbalized tail sampling (the model labels each idea with a guess about how obvious it is, and obvious ideas are discarded), which filters high-probability candidates via model self-assessment; (2) semantic memory, which maintains a persistent embedding index to reject near-duplicates across batches; and (3) adaptive prompt evolution, which reconstructs the generation prompt each batch using memory state and rotating diversity strategies. In experiments across three domains (sustainable packaging concepts, educational exam questions, and creative writing prompts) and two model families (gpt-5-mini and claude-haiku-4-5), a component ablation across 2-3 random seeds per method shows that DCE achieves 0.0 +/- 0.0% collapse versus 5.6 +/- 2.0% for naive prompting, while producing 17-18 HDBSCAN clusters per seed versus naive's volatile 2-17, indicating reliably richer conceptual structure. These results are validated with an independent embedding model (all-MiniLM-L6-v2) and hold across sensitivity sweeps of the VTS threshold tau and dedup threshold delta. Deduplication and prompt evolution are individually insufficient but jointly effective, at approximately $0.50 per 1,000 candidates using only standard API calls, with no fine-tuning or custom architectures required. 

---
# Language Bias under Conflicting Information in Multilingual LLMs 

**Authors**: Robert Östling, Murathan Kurfalı  

**Link**: [PDF](https://arxiv.org/pdf/2604.07123)  

**Abstract**: Large Language Models (LLMs) have been shown to contain biases in the process of integrating conflicting information when answering questions. Here we ask whether such biases also exist with respect to which language is used for each conflicting piece of information. To answer this question, we extend the conflicting needles in a haystack paradigm to a multilingual setting and perform a comprehensive set of evaluations with naturalistic news domain data in five different languages, for a range of multilingual LLMs of different sizes. We find that all LLMs tested, including GPT-5.2, ignore the conflict and confidently assert only one of the possible answers in the large majority of cases. Furthermore, there is a consistent bias across models in which languages are preferred, with a general bias against Russian and, for the longest context lengths, in favor of Chinese. Both of these patterns are consistent between models trained inside and outside of mainland China, though somewhat stronger in the former category. 

---
# Are Non-English Papers Reviewed Fairly? Language-of-Study Bias in NLP Peer Reviews 

**Authors**: Ehsan Barkhordar, Abdulfattah Safa, Verena Blaschke, Erika Lombart, Marie-Catherine de Marneffe, Gözde Gül Şahin  

**Link**: [PDF](https://arxiv.org/pdf/2604.07119)  

**Abstract**: Peer review plays a central role in the NLP publication process, but is susceptible to various biases. Here, we study language-of-study (LoS) bias: the tendency for reviewers to evaluate a paper differently based on the language(s) it studies, rather than its scientific merit. Despite being explicitly flagged in reviewing guidelines, such biases are poorly understood. Prior work treats such comments as part of broader categories of weak or unconstructive reviews without defining them as a distinct form of bias. We present the first systematic characterization of LoS bias, distinguishing negative and positive forms, and introduce the human-annotated dataset LOBSTER (Language-Of-study Bias in ScienTific pEer Review) and a method achieving 87.37 macro F1 for detection. We analyze 15,645 reviews to estimate how negative and positive biases differ with respect to the LoS, and find that non-English papers face substantially higher bias rates than English-only ones, with negative bias consistently outweighing positive bias. Finally, we identify four subcategories of negative bias, and find that demanding unjustified cross-lingual generalization is the most dominant form. We publicly release all resources to support work on fairer reviewing practices in NLP and beyond. 

---
# Yale-DM-Lab at ArchEHR-QA 2026: Deterministic Grounding and Multi-Pass Evidence Alignment for EHR Question Answering 

**Authors**: Elyas Irankhah, Samah Fodeh  

**Link**: [PDF](https://arxiv.org/pdf/2604.07116)  

**Abstract**: We describe the Yale-DM-Lab system for the ArchEHR-QA 2026 shared task. The task studies patient-authored questions about hospitalization records and contains four subtasks (ST): clinician-interpreted question reformulation, evidence sentence identification, answer generation, and evidence-answer alignment. ST1 uses a dual-model pipeline with Claude Sonnet 4 and GPT-4o to reformulate patient questions into clinician-interpreted questions. ST2-ST4 rely on Azure-hosted model ensembles (o3, GPT-5.2, GPT-5.1, and DeepSeek-R1) combined with few-shot prompting and voting strategies. Our experiments show three main findings. First, model diversity and ensemble voting consistently improve performance compared to single-model baselines. Second, the full clinician answer paragraph is provided as additional prompt context for evidence alignment. Third, results on the development set show that alignment accuracy is mainly limited by reasoning. The best scores on the development set reach 88.81 micro F1 on ST4, 65.72 macro F1 on ST2, 34.01 on ST3, and 33.05 on ST1. 

---
# The Impact of Steering Large Language Models with Persona Vectors in Educational Applications 

**Authors**: Yongchao Wu, Aron Henriksson  

**Link**: [PDF](https://arxiv.org/pdf/2604.07102)  

**Abstract**: Activation-based steering can personalize large language models at inference time, but its effects in educational settings remain unclear. We study persona vectors for seven character traits in short-answer generation and automated scoring on the ASAP-SAS benchmark across three models spanning two architectures. Persona steering lowers answer quality overall, with much larger effects on open-ended English Language Arts (ELA) prompts than on factual science prompts; interpretive and argumentative tasks are up to 11x more sensitive. On the scoring side, we observe predictable valence-aligned calibration shifts: evil and impolite scorers grade more harshly, while good and optimistic scorers grade more leniently. ELA tasks are 2.5-3x more susceptible to scorer personalization than science tasks, and the Mixture-of-Experts model shows roughly 6x larger calibration shifts than the dense models. To our knowledge, this is the first study to systematically examine the effects of activation-steered persona traits in educational generation and scoring, and the results highlight the need for task-aware and architecture-aware calibration when deploying steered models in educational settings. 

---
# STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems 

**Authors**: Hongru Ji, Yuyin Fan, Meng Zhao, Xianghua Li, Lianwei Wu, Chao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07100)  

**Abstract**: Empathetic dialogue requires not only recognizing a user's emotional state but also making strategy-aware, context-sensitive decisions throughout response generation. However, the lack of a comprehensive empathy strategy framework, explicit task-aligned multi-stage reasoning, and high-quality strategy-aware data fundamentally limits existing approaches, preventing them from effectively modeling empathetic dialogue as a complex, multi-stage cognitive and decision-making process. To address these challenges, we propose STRIDE-ED, a STRategy-grounded, Interpretable, and DEep reasoning framework that models Empathetic Dialogue through structured, strategy-conditioned reasoning. To support effective learning, we develop a strategy-aware data refinement pipeline integrating LLM-based annotation, multi-model consistency-weighted evaluation, and dynamic sampling to construct high-quality training data aligned with empathetic strategies. Furthermore, we adopt a two-stage training paradigm that combines supervised fine-tuning with multi-objective reinforcement learning to better align model behaviors with target emotions, empathetic strategies, and response formats. Extensive experiments demonstrate that STRIDE-ED generalizes across diverse open-source LLMs and consistently outperforms existing methods on both automatic metrics and human evaluations. 

---
# Multilingual Embedding Probes Fail to Generalize Across Learner Corpora 

**Authors**: Laurits Lyngbaek, Ross Deans Kristensen-McLachlan  

**Link**: [PDF](https://arxiv.org/pdf/2604.07095)  

**Abstract**: Do multilingual embedding models encode a language-general representation of proficiency? We investigate this by training linear and non-linear probes on hidden-state activations from Qwen3-Embedding (0.6B, 4B, 8B) to predict CEFR proficiency levels from learner texts across nine corpora and seven languages. We compare five probing architectures against a baseline trained on surface-level text features. Under in-distribution evaluation, probes achieve strong performance ($QWK\approx0.7$), substantially outperforming the surface baseline, with middle layers consistently yielding the best predictions. However, in cross-corpus evaluation performance collapses across all probe types and model sizes. Residual analysis reveals that out-of-distribution probes converge towards predicting uniformly distributed labels, indicating that the learned mappings capture corpus-specific distributional properties (topic, language, task type, rating methodology) rather than an abstract, transferable proficiency dimension. These results suggest that current multilingual embeddings do not straightforwardly encode language-general proficiency, with implications for representation-based approaches to proficiency-adaptive language technology. 

---
# Is Cross-Lingual Transfer in Bilingual Models Human-Like? A Study with Overlapping Word Forms in Dutch and English 

**Authors**: Iza Škrjanec, Irene Elisabeth Winther, Vera Demberg, Stefan L. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2604.07067)  

**Abstract**: Bilingual speakers show cross-lingual activation during reading, especially for words with shared surface form. Cognates (friends) typically lead to facilitation, whereas interlingual homographs (false friends) cause interference or no effect. We examine whether cross-lingual activation in bilingual language models mirrors these patterns. We train Dutch-English causal Transformers under four vocabulary-sharing conditions that manipulate whether (false) friends receive shared or language-specific embeddings. Using psycholinguistic stimuli from bilingual reading studies, we evaluate the models through surprisal and embedding similarity analyses. The models largely maintain language separation, and cross-lingual effects arise primarily when embeddings are shared. In these cases, both friends and false friends show facilitation relative to controls. Regression analyses reveal that these effects are mainly driven by frequency rather than consistency in form-meaning mapping. Only when just friends share embeddings are the qualitative patterns of bilinguals reproduced. Overall, bilingual language models capture some cross-linguistic activation effects. However, their alignment with human processing seems to critically depend on how lexical overlap is encoded, possibly limiting their explanatory adequacy as models of bilingual reading. 

---
# SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA) 

**Authors**: Liang-Chih Yu, Jonas Becker, Shamsuddeen Hassan Muhammad, Idris Abdulmumin, Lung-Hao Lee, Ying-Lung Lin, Jin Wang, Jan Philip Wahle, Terry Ruas, Natalia Loukachevitch, Alexander Panchenko, Ilseyar Alimova, Lilian Wanzare, Nelson Odhiambo, Bela Gipp, Kai-Wei Chang, Saif M. Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2604.07066)  

**Abstract**: We present the SemEval-2026 shared task on Dimensional Aspect-Based Sentiment Analysis (DimABSA), which improves traditional ABSA by modeling sentiment along valence-arousal (VA) dimensions rather than using categorical polarity labels. To extend ABSA beyond consumer reviews to public-issue discourse (e.g., political, energy, and climate issues), we introduce an additional task, Dimensional Stance Analysis (DimStance), which treats stance targets as aspects and reformulates stance detection as regression in the VA space. The task consists of two tracks: Track A (DimABSA) and Track B (DimStance). Track A includes three subtasks: (1) dimensional aspect sentiment regression, (2) dimensional aspect sentiment triplet extraction, and (3) dimensional aspect sentiment quadruplet extraction, while Track B includes only the regression subtask for stance targets. We also introduce a continuous F1 (cF1) metric to jointly evaluate structured extraction and VA regression. The task attracted more than 400 participants, resulting in 112 final submissions and 42 system description papers. We report baseline results, discuss top-performing systems, and analyze key design choices to provide insights into dimensional sentiment analysis at the aspect and stance-target levels. All resources are available on our GitHub repository. 

---
# IndoBERT-Sentiment: Context-Conditioned Sentiment Classification for Indonesian Text 

**Authors**: Muhammad Apriandito Arya Saputra, Andry Alamsyah, Dian Puteri Ramadhani, Thomhert Suprapto Siadari, Hanif Fakhrurroja  

**Link**: [PDF](https://arxiv.org/pdf/2604.07057)  

**Abstract**: Existing Indonesian sentiment analysis models classify text in isolation, ignoring the topical context that often determines whether a statement is positive, negative, or neutral. We introduce IndoBERT-Sentiment, a context-conditioned sentiment classifier that takes both a topical context and a text as input, producing sentiment predictions grounded in the topic being discussed. Built on IndoBERT Large (335M parameters) and trained on 31,360 context-text pairs labeled across 188 topics, the model achieves an F1 macro of 0.856 and accuracy of 88.1%. In a head-to-head evaluation against three widely used general-purpose Indonesian sentiment models on the same test set, IndoBERT-Sentiment outperforms the best baseline by 35.6 F1 points. We show that context-conditioning, previously demonstrated for relevancy classification, transfers effectively to sentiment analysis and enables the model to correctly classify texts that are systematically misclassified by context-free approaches. 

---
# Sell More, Play Less: Benchmarking LLM Realistic Selling Skill 

**Authors**: Xuanbo Su, Wenhao Hu, Le Zhan, Yanqi Yang, Leo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07054)  

**Abstract**: Sales dialogues require multi-turn, goal-directed persuasion under asymmetric incentives, which makes them a challenging setting for large language models (LLMs). Yet existing dialogue benchmarks rarely measure deal progression and outcomes. We introduce SalesLLM, a bilingual (ZH/EN) benchmark derived from realistic applications covering Financial Services and Consumer Goods, built from 30,074 scripted configurations and 1,805 curated multi-turn scenarios with controllable difficulty and personas. We propose a fully automatic evaluation pipeline that combines (i) an LLM-based rater for sales-process progress, and (ii) fine-tuned BERT classifiers for end-of-dialogue buying intent. To improve simulation fidelity, we train a user model, CustomerLM, with SFT and DPO on 8,000 crowdworker-involved sales conversations, reducing role inversion from 17.44% (GPT-4o) to 8.8%. SalesLLM scores correlate strongly with expert human ratings (Pearson r=0.98). Experiments across 15 mainstream LLMs reveal substantial variability: top-performance LLMs are competitive with human-level performance while the less capable ones are worse than human. SalesLLM serves as a scalable benchmark for developing and evaluating outcome-oriented sales agents. 

---
# ReDAct: Uncertainty-Aware Deferral for LLM Agents 

**Authors**: Dzianis Piatrashyn, Nikita Kotelevskii, Kirill Grishchenkov, Nikita Glazkov, Ivan Nasonov, Ilya Makarov, Timothy Baldwin, Preslav Nakov, Roman Vashurin, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2604.07036)  

**Abstract**: Recently, LLM-based agents have become increasingly popular across many applications, including complex sequential decision-making problems. However, they inherit the tendency of LLMs to hallucinate, leading to incorrect decisions. In sequential settings, even a single mistake can irreversibly degrade the trajectory, making hallucinations an even bigger problem. Although larger LLMs hallucinate less, they incur a significantly higher per-token cost. In this paper, we address this tradeoff by proposing ReDAct (Reason-Defer-Act). In ReDAct, an agent is equipped with two LLMs: a small, cheap model used by default, and a large, more reliable but expensive model. When the predictive uncertainty of the small model exceeds a calibrated threshold, the decision is deferred to the large model. We evaluate our approach in text-based embodied environments such as ALFWorld and MiniGrid and show that deferring only about 15% of decisions to the large model can match the quality of using it exclusively, while significantly reducing inference costs. 

---
# Gemma 4, Phi-4, and Qwen3: Accuracy-Efficiency Tradeoffs in Dense and MoE Reasoning Language Models 

**Authors**: Md Motaleb Hossen Manik, Ge Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07035)  

**Abstract**: Mixture-of-experts (MoE) language models are often expected to offer better quality-efficiency tradeoffs than dense models because only a subset of parameters is activated per token, but the practical value of that advantage depends on end-to-end behavior under realistic inference constraints. We present a controlled empirical benchmark of seven recent reasoning-oriented instruction-tuned models spanning dense and MoE designs, namely Gemma-4-E2B, Gemma-4-E4B, Gemma-4-26B-A4B, Phi-4-mini-reasoning, Phi-4-reasoning, Qwen3-8B, and Qwen3-30B-A3B, evaluated on four benchmarks -- ARC-Challenge, GSM8K, Math Level 1-3, and TruthfulQA MC1 -- under three prompting strategies: zero-shot, chain-of-thought, and few-shot chain-of-thought. The study covers 8,400 total model-dataset-prompt evaluations and records accuracy, latency, peak GPU memory usage (VRAM), and an approximate floating-point operations (FLOPs)-per-token proxy. Across the weighted multi-task summary, Gemma-4-E4B with few-shot chain-of-thought achieved the best overall result, reaching weighted accuracy 0.675 with mean VRAM 14.9 GB, while Gemma-4-26B-A4B was close in accuracy at 0.663 but substantially more memory intensive at 48.1 GB. At the task level, Gemma models dominated ARC and Math, Phi models were strongest on TruthfulQA, and GSM8K showed the largest prompt sensitivity, including a sharp drop for Phi-4-reasoning from 0.67 under chain-of-thought to 0.11 under few-shot chain-of-thought. These results show that sparse activation alone does not guarantee the best practical operating point: observed accuracy-efficiency tradeoffs depend jointly on architecture, prompting protocol, and task composition. We release a reproducible benchmark pipeline, aggregated results, and paired statistical analyses to support deployment-oriented evaluation of reasoning LLMs under real resource constraints. 

---
# MARS: Enabling Autoregressive Models Multi-Token Generation 

**Authors**: Ziqi Jin, Lei Wang, Ziwei Luo, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.07023)  

**Abstract**: Autoregressive (AR) language models generate text one token at a time, even when consecutive tokens are highly predictable given earlier context. We introduce MARS (Mask AutoRegreSsion), a lightweight fine-tuning method that teaches an instruction-tuned AR model to predict multiple tokens per forward pass. MARS adds no architectural modifications, no extra parameters, and produces a single model that can still be called exactly like the original AR model with no performance degradation. Unlike speculative decoding, which maintains a separate draft model alongside the target, or multi-head approaches such as Medusa, which attach additional prediction heads, MARS requires only continued training on existing instruction data. When generating one token per forward pass, MARS matches or exceeds the AR baseline on six standard benchmarks. When allowed to accept multiple tokens per step, it maintains baseline-level accuracy while achieving 1.5-1.7x throughput. We further develop a block-level KV caching strategy for batch inference, achieving up to 1.71x wall-clock speedup over AR with KV cache on Qwen2.5-7B. Finally, MARS supports real-time speed adjustment via confidence thresholding: under high request load, the serving system can increase throughput on the fly without swapping models or restarting, providing a practical latency-quality knob for deployment. 

---
# Corpora deduplication or duplication in Natural Language Processing of few resourced languages ? A case of study: The Mexico's Nahuatl 

**Authors**: Juan-José Guzman-Landa, Juan-Manuel Torres-Moreno, Graham Ranger, Miguel Figueroa-Saavedra, Martha-Lorena Avendaño-Garrido, Elvys Linhares-Pontes, Luis-Gil Moreno-Jiménez  

**Link**: [PDF](https://arxiv.org/pdf/2604.07015)  

**Abstract**: In this article, we seek to answer the following question: could data duplication be useful in Natural Language Processing (NLP) for languages with limited computational resources? In this type of languages (or $\pi$-languages), corpora available for training Large Language Models are virtually non-existent. In particular, we will study the impact of corpora expansion in Nawatl, an agglutinative and polysynthetic $\pi$-language spoken by over 2 million people, with a large number of dialectal varieties. The aim is to expand the new $\pi$-yalli corpus, which contains a limited number of Nawatl texts, by duplicating it in a controlled way. In our experiments, we will use the incremental duplication technique. The aim is to learn embeddings that are well-suited to NLP tasks. Thus, static embeddings were trained and evaluated in a sentence-level semantic similarity task. Our results show a moderate improvement in performance when using incremental duplication compared to the results obtained using only the corpus without expansion. Furthermore, to our knowledge, this technique has not yet been used in the literature. 

---
# DTCRS: Dynamic Tree Construction for Recursive Summarization 

**Authors**: Guanran Luo, Zhongquan Jian, Wentao Qiu, Meihong Wang, Qingqiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07012)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates the hallucination problem of Large Language Models (LLMs) by incorporating external knowledge. Recursive summarization constructs a hierarchical summary tree by clustering text chunks, integrating information from multiple parts of a document to provide evidence for abstractive questions involving multi-step reasoning. However, summary trees often contain a large number of redundant summary nodes, which not only increase construction time but may also negatively impact question answering. Moreover, recursive summarization is not suitable for all types of questions. We introduce DTCRS, a method that dynamically generates summary trees based on document structure and query semantics. DTCRS determines whether a summary tree is necessary by analyzing the question type. It then decomposes the question and uses the embeddings of sub-questions as initial cluster centers, reducing redundant summaries while improving the relevance between summaries and the question. Our approach significantly reduces summary tree construction time and achieves substantial improvements across three QA tasks. Additionally, we investigate the applicability of recursive summarization to different question types, providing valuable insights for future research. 

---
# Continuous Interpretive Steering for Scalar Diversity 

**Authors**: Ye-eun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2604.07006)  

**Abstract**: Pragmatic inference is inherently graded. Different lexical items give rise to pragmatic enrichment to different degrees. Scalar implicature exemplifies this property through scalar diversity, where implicature strength varies across scalar items. However, evaluations of pragmatic inference in large language models (LLMs) often rely on prompt-based manipulations. Beyond prompt-level effects, this study introduces Continuous Interpretive Steering (CIS), a method that probes graded pragmatic interpretation by treating activation-level steering strength as a continuous experimental variable. To support this analysis, this study introduces a new dataset, GraSD, which encodes graded scalar diversity. Experiments on four LLMs show that uniform activation steering increases pragmatic interpretations globally but collapses item-level variation, whereas graded activation steering yields differentiated interpretive shifts aligned with scalar diversity grades. It indicates that graded sensitivity is encoded in the representation space and can be systematically recovered through controlled intervention. Together, CIS and GraSD provide a principled framework for evaluating graded pragmatic sensitivity in LLMs. 

---
# ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals 

**Authors**: Yihao Wang, Zijian He, Jie Ren, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06997)  

**Abstract**: Retrieval shapes how language models access and ground knowledge in retrieval-augmented generation (RAG). In historical research, the target is often not an arbitrary relevant passage, but the exact record for a specific regnal month, where temporal consistency matters as much as topical relevance. This is especially challenging for Classical Chinese annals, where time is expressed through terse, implicit, non-Gregorian reign phrases that must be interpreted from surrounding context, so semantically plausible evidence can still be temporally invalid. We introduce \textbf{ChunQiuTR}, a time-keyed retrieval benchmark built from the \textit{Spring and Autumn Annals} and its exegetical tradition. ChunQiuTR organizes records by month-level reign keys and includes chrono-near confounders that mirror realistic retrieval failures. We further propose \textbf{CTD} (Calendrical Temporal Dual-encoder), a time-aware dual-encoder that combines Fourier-based absolute calendrical context with relative offset biasing. Experiments show consistent gains over strong semantic dual-encoder baselines under time-keyed evaluation, supporting retrieval-time temporal consistency as a key prerequisite for faithful downstream historical RAG. Our code and datasets are available at \href{this https URL}{\texttt{this http URL}}. 

---
# Self-Preference Bias in Rubric-Based Evaluation of Large Language Models 

**Authors**: José Pombal, Ricardo Rei, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2604.06996)  

**Abstract**: LLM-as-a-judge has become the de facto approach for evaluating LLM outputs. However, judges are known to exhibit self-preference bias (SPB): they tend to favor outputs produced by themselves or by models from their own family. This skews evaluations and, thus, hinders model development, especially in settings of recursive self-improvement. We present the first study of SPB in rubric-based evaluation, an increasingly popular benchmarking paradigm where judges issue binary verdicts on individual evaluation criteria, instead of assigning holistic scores or rankings. Using IFEval, a benchmark with programmatically verifiable rubrics, we show that SPB persists even when evaluation criteria are entirely objective: among rubrics where generators fail, judges can be up to 50\% more likely to incorrectly mark them as satisfied when the output is their own. We also find that, similarly to other evaluation paradigms, ensembling multiple judges helps mitigate SPB, but without fully eliminating it. On HealthBench, a medical chat benchmark with subjective rubrics, we observe that SPB skews model scores by up to 10 points, a potentially decisive margin when ranking frontier models. We analyze the factors that drive SPB in this setting, finding that negative rubrics, extreme rubric lengths, and subjective topics like emergency referrals are particularly susceptible. 

---
# The AI Skills Shift: Mapping Skill Obsolescence, Emergence, and Transition Pathways in the LLM Era 

**Authors**: Rudra Jadhav, Janhavi Danve  

**Link**: [PDF](https://arxiv.org/pdf/2604.06906)  

**Abstract**: As Large Language Models reshape the global labor market, policymakers and workers need empirical data on which occupational skills may be most susceptible to automation. We present the Skill Automation Feasibility Index (SAFI), benchmarking four frontier LLMs -- LLaMA 3.3 70B, Mistral Large, Qwen 2.5 72B, and Gemini 2.5 Flash -- across 263 text-based tasks spanning all 35 skills in the U.S. Department of Labor's O*NET taxonomy (1,052 total model calls, 0% failure rate). Cross-referencing with real-world AI adoption data from the Anthropic Economic Index (756 occupations, 17,998 tasks), we propose an AI Impact Matrix -- an interpretive framework that positions skills along four quadrants: High Displacement Risk, Upskilling Required, AI-Augmented, and Lower Displacement Risk. Key findings: (1) Mathematics (SAFI: 73.2) and Programming (71.8) receive the highest automation feasibility scores; Active Listening (42.2) and Reading Comprehension (45.5) receive the lowest; (2) a "capability-demand inversion" where skills most demanded in AI-exposed jobs are those LLMs perform least well at in our benchmark; (3) 78.7% of observed AI interactions are augmentation, not automation; (4) all four models converge to similar skill profiles (3.6-point spread), suggesting that text-based automation feasibility may be more skill-dependent than model-dependent. SAFI measures LLM performance on text-based representations of skills, not full occupational execution. All data, code, and model responses are open-sourced. 

---
# Is Biomedical Specialization Still Worth It? Insights from Domain-Adaptive Language Modelling with a New French Health Corpus 

**Authors**: Aidan Mannion, Cécile Macaire, Armand Violle, Stéphane Ohayon, Xavier Tannier, Didier Schwab, Lorraine Goeuriot, François Portet  

**Link**: [PDF](https://arxiv.org/pdf/2604.06903)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, yet their adaptation to specialized fields remains challenging, particularly for non-English languages. This study investigates domain-adaptive pre-training (DAPT) as a strategy for specializing small to mid-sized LLMs in the French biomedical domain through continued pre-training. We address two key research questions: the viability of specialized continued pre-training for domain adaptation and the relationship between domain-specific performance gains and general capability degradation. Our contributions include the release of a fully open-licensed French biomedical corpus suitable for commercial and open-source applications, the training and release of specialized French biomedical LLMs, and novel insights for DAPT implementation. Our methodology encompasses the collection and refinement of high-quality French biomedical texts, the exploration of causal language modeling approaches using DAPT, and conducting extensive comparative evaluations. Our results cast doubt on the efficacy of DAPT, in contrast to previous works, but we highlight its viability in smaller-scale, resource-constrained scenarios under the right conditions. Findings in this paper further suggest that model merging post-DAPT is essential to mitigate generalization trade-offs, and in some cases even improves performance on specialized tasks at which the DAPT was directed. 

---
# iTAG: Inverse Design for Natural Text Generation with Accurate Causal Graph Annotations 

**Authors**: Wenshuo Wang, Boyu Cao, Nan Zhuang, Wei Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.06902)  

**Abstract**: A fundamental obstacle to causal discovery from text is the lack of causally annotated text data for use as ground truth, due to high annotation costs. This motivates an important task of generating text with causal graph annotations. Early template-based generation methods sacrifice text naturalness in exchange for high causal graph annotation accuracy. Recent Large Language Model (LLM)-dependent methods directly generate natural text from target graphs through LLMs, but do not guarantee causal graph annotation accuracy. Therefore, we propose iTAG, which performs real-world concept assignment to nodes before converting causal graphs into text in existing LLM-dependent methods. iTAG frames this process as an inverse problem with the causal graph as the target, iteratively examining and refining concept selection through Chain-of-Thought (CoT) reasoning so that the induced relations between concepts are as consistent as possible with the target causal relationships described by the causal graph. iTAG demonstrates both extremely high annotation accuracy and naturalness across extensive tests, and the results of testing text-based causal discovery algorithms with the generated data show high statistical correlation with real-world data. This suggests that iTAG-generated data can serve as a practical surrogate for scalable benchmarking of text-based causal discovery algorithms. 

---
# Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models 

**Authors**: Bajian Xiang, Tingwei Guo, Xuan Chen, Yang Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.06871)  

**Abstract**: Large Speech Language Models (LSLMs) typically operate at high token rates (tokens/s) to ensure acoustic fidelity, yet this results in sequence lengths that far exceed the underlying semantic content, incurring prohibitive inference costs. In this paper, we empirically revisit the necessity of such granular token-level processing. Through layer-wise oracle interventions, we unveil a structured redundancy hierarchy: while shallow layers encode essential acoustic details, deep layers exhibit extreme redundancy, allowing for aggressive compression. Motivated by these findings, we introduce Affinity Pooling, a training-free, similarity-based token merging mechanism. By strategically applying this method at both input and deep layers, we effectively compress speech representations without compromising semantic information. Extensive evaluations across three tasks demonstrate that our approach reduces prefilling FLOPs by 27.48\% while maintaining competitive accuracy. Practical deployment further confirms significant efficiency gains, yielding up to $\sim$1.7$\times$ memory savings and $\sim$1.1$\times$ faster time-to-first-token on long utterances. Our results challenge the necessity of fully distinct token representations, providing new perspectives on LSLM efficiency. 

---
# To Adapt or not to Adapt, Rethinking the Value of Medical Knowledge-Aware Large Language Models 

**Authors**: Ane G. Domingo-Aldama, Iker De La Iglesia, Maitane Urruela, Aitziber Atutxa, Ander Barrena  

**Link**: [PDF](https://arxiv.org/pdf/2604.06854)  

**Abstract**: BACKGROUND: Recent studies have shown that domain-adapted large language models (LLMs) do not consistently outperform general-purpose counterparts on standard medical benchmarks, raising questions about the need for specialized clinical adaptation.
METHODS: We systematically compare general and clinical LLMs on a diverse set of multiple choice clinical question answering tasks in English and Spanish. We introduce a perturbation based evaluation benchmark that probes model robustness, instruction following, and sensitivity to adversarial variations. Our evaluation includes, one-step and two-step question transformations, multi prompt testing and instruction guided assessment. We analyze a range of state-of-the-art clinical models and their general-purpose counterparts, focusing on Llama 3.1-based models. Additionally, we introduce Marmoka, a family of lightweight 8B-parameter clinical LLMs for English and Spanish, developed via continual domain-adaptive pretraining on medical corpora and instructions.
RESULTS: The experiments show that clinical LLMs do not consistently outperform their general purpose counterparts on English clinical tasks, even under the proposed perturbation based benchmark. However, for the Spanish subsets the proposed Marmoka models obtain better results compared to Llama.
CONCLUSIONS: Our results show that, under current short-form MCQA benchmarks, clinical LLMs offer only marginal and unstable improvements over general-purpose models in English, suggesting that existing evaluation frameworks may be insufficient to capture genuine medical expertise. We further find that both general and clinical models exhibit substantial limitations in instruction following and strict output formatting. Finally, we demonstrate that robust medical LLMs can be successfully developed for low-resource languages such as Spanish, as evidenced by the Marmoka models. 

---
# MedDialBench: Benchmarking LLM Diagnostic Robustness under Parametric Adversarial Patient Behaviors 

**Authors**: Xiaotian Luo, Xun Jiang, Jiangcheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06846)  

**Abstract**: Interactive medical dialogue benchmarks have shown that LLM diagnostic accuracy degrades significantly when interacting with non-cooperative patients, yet existing approaches either apply adversarial behaviors without graded severity or case-specific grounding, or reduce patient non-cooperation to a single ungraded axis, and none analyze cross-dimension interactions.
We introduce MedDialBench, a benchmark enabling controlled, dose-response characterization of how individual patient behavior dimensions affect LLM diagnostic robustness. It decomposes patient behavior into five dimensions -- Logic Consistency, Health Cognition, Expression Style, Disclosure, and Attitude -- each with graded severity levels and case-specific behavioral scripts. This controlled factorial design enables graded sensitivity analysis, dose-response profiling, and cross-dimension interaction detection.
Evaluating five frontier LLMs across 7,225 dialogues (85 cases x 17 configurations x 5 models), we find a fundamental asymmetry: information pollution (fabricating symptoms) produces 1.7-3.4x larger accuracy drops than information deficit (withholding information), and fabricating is the only configuration achieving statistical significance across all five models (McNemar p < 0.05). Among six dimension combinations, fabricating is the sole driver of super-additive interaction: all three fabricating-involving pairs produce O/E ratios of 0.70-0.81 (35-44% of eligible cases fail under the combination despite succeeding under each dimension alone), while all non-fabricating pairs show purely additive effects (O/E ~ 1.0). Inquiry strategy moderates deficit but not pollution: exhaustive questioning recovers withheld information, but cannot compensate for fabricated inputs. Models exhibit distinct vulnerability profiles, with worst-case drops ranging from 38.8 to 54.1 percentage points. 

---
# HingeMem: Boundary Guided Long-Term Memory with Query Adaptive Retrieval for Scalable Dialogues 

**Authors**: Yijie Zhong, Yunfan Gao, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06845)  

**Abstract**: Long-term memory is critical for dialogue systems that support continuous, sustainable, and personalized interactions. However, existing methods rely on continuous summarization or OpenIE-based graph construction paired with fixed Top-\textit{k} retrieval, leading to limited adaptability across query categories and high computational overhead. In this paper, we propose HingeMem, a boundary-guided long-term memory that operationalizes event segmentation theory to build an interpretable indexing interface via boundary-triggered hyperedges over four elements: person, time, location, and topic. When any such element changes, HingeMem draws a boundary and writes the current segment, thereby reducing redundant operations and preserving salient context. To enable robust and efficient retrieval under diverse information needs, HingeMem introduces query-adaptive retrieval mechanisms that jointly decide (a) \textit{what to retrieve}: determine the query-conditioned routing over the element-indexed memory; (b) \textit{how much to retrieve}: control the retrieval depth based on the estimated query type. Extensive experiments across LLM scales (from 0.6B to production-tier models; \textit{e.g.}, Qwen3-0.6B to Qwen-Flash) on LOCOMO show that HingeMem achieves approximately $20\%$ relative improvement over strong baselines without query categories specification, while reducing computational cost (68\%$\downarrow$ question answering token cost compared to HippoRAG2). Beyond advancing memory modeling, HingeMem's adaptive retrieval makes it a strong fit for web applications requiring efficient and trustworthy memory over extended interactions. 

---
# On the Step Length Confounding in LLM Reasoning Data Selection 

**Authors**: Bing Wang, Rui Miao, Chen Shen, Shaotian Yan, Kaiyuan Liu, Ximing Li, Xiaosong Yuan, Sinan Fan, Jun Zhang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2604.06834)  

**Abstract**: Large reasoning models have recently demonstrated strong performance on complex tasks that require long chain-of-thought reasoning, through supervised fine-tuning on large-scale and high-quality datasets. To construct such datasets, existing pipelines generate long reasoning data from more capable Large Language Models (LLMs) and apply manually heuristic or naturalness-based selection methods to filter high-quality samples. Despite the proven effectiveness of naturalness-based data selection, which ranks data by the average log probability assigned by LLMs, our analysis shows that, when applied to LLM reasoning datasets, it systematically prefers samples with longer reasoning steps (i.e., more tokens per step) rather than higher-quality ones, a phenomenon we term step length confounding. Through quantitative analysis, we attribute this phenomenon to low-probability first tokens in reasoning steps; longer steps dilute their influence, thereby inflating the average log probabilities. To address this issue, we propose two variant methods: ASLEC-DROP, which drops first-token probabilities when computing average log probability, and ASLEC-CASL, which applies a causal debiasing regression to remove the first tokens' confounding effect. Experiments across four LLMs and five evaluation benchmarks demonstrate the effectiveness of our approach in mitigating the step length confounding problem. 

---
# Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM 

**Authors**: Chengyue Wu, Shiyi Lan, Yonggan Fu, Sensen Gao, Jin Wang, Jincheng Yu, Jose M. Alvarez, Pavlo Molchanov, Ping Luo, Song Han, Ligeng Zhu, Enze Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.06832)  

**Abstract**: Vision-language models (VLMs) predominantly rely on autoregressive decoding, which generates tokens one at a time and fundamentally limits inference throughput. This limitation is especially acute in physical AI scenarios such as robotics and autonomous driving, where VLMs are deployed on edge devices at batch size one, making AR decoding memory-bandwidth-bound and leaving hardware parallelism underutilized. While block-wise discrete diffusion has shown promise for parallel text generation, extending it to VLMs remains challenging due to the need to jointly handle continuous visual representations and discrete text tokens while preserving pretrained multimodal capabilities. We present Fast-dVLM, a block-diffusion-based VLM that enables KV-cache-compatible parallel decoding and speculative block decoding for inference acceleration. We systematically compare two AR-to-diffusion conversion strategies: a two-stage approach that first adapts the LLM backbone with text-only diffusion fine-tuning before multimodal training, and a direct approach that converts the full AR VLM in one stage. Under comparable training budgets, direct conversion proves substantially more efficient by leveraging the already multimodally aligned VLM; we therefore adopt it as our recommended recipe. We introduce a suite of multimodal diffusion adaptations, block size annealing, causal context attention, auto-truncation masking, and vision efficient concatenation, that collectively enable effective block diffusion in the VLM setting. Extensive experiments across 11 multimodal benchmarks show Fast-dVLM matches its autoregressive counterpart in generation quality. With SGLang integration and FP8 quantization, Fast-dVLM achieves over 6x end-to-end inference speedup over the AR baseline. 

---
# WRAP++: Web discoveRy Amplified Pretraining 

**Authors**: Jiang Zhou, Yunhao Wang, Xing Wu, Tinghao Yu, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06829)  

**Abstract**: Synthetic data rephrasing has emerged as a powerful technique for enhancing knowledge acquisition during large language model (LLM) pretraining. However, existing approaches operate at the single-document level, rewriting individual web pages in isolation. This confines synthesized examples to intra-document knowledge, missing cross-document relationships and leaving facts with limited associative context. We propose WRAP++ (Web discoveRy Amplified Pretraining), which amplifies the associative context of factual knowledge by discovering cross-document relationships from web hyperlinks and synthesizing joint QA over each discovered document pair. Concretely, WRAP++ discovers high-confidence relational motifs including dual-links and co-mentions, and synthesizes QA that requires reasoning across both documents. This produces relational knowledge absent from either source document alone, creating diverse entry points to the same facts. Because the number of valid entity pairs grows combinatorially, this discovery-driven synthesis also amplifies data scale far beyond single-document rewriting. Instantiating WRAP++ on Wikipedia, we amplify ~8.4B tokens of raw text into 80B tokens of cross-document QA data. On SimpleQA, OLMo-based models at both 7B and 32B scales trained with WRAP++ substantially outperform single-document approaches and exhibit sustained scaling gains, underscoring the advantage of cross-document knowledge discovery and amplification. 

---
# Environmental, Social and Governance Sentiment Analysis on Slovene News: A Novel Dataset and Models 

**Authors**: Paula Dodig, Boshko Koloski, Katarina Sitar Šuštar, Senja Pollak, Matthew Purver  

**Link**: [PDF](https://arxiv.org/pdf/2604.06826)  

**Abstract**: Environmental, Social, and Governance (ESG) considerations are increasingly integral to assessing corporate performance, reputation, and long-term sustainability. Yet, reliable ESG ratings remain limited for smaller companies and emerging markets. We introduce the first publicly available Slovene ESG sentiment dataset and a suite of models for automatic ESG sentiment detection. The dataset, derived from the MaCoCu Slovene news collection, combines large language model (LLM)-assisted filtering with human annotation of company-related ESG content. We evaluate the performance of monolingual (SloBERTa) and multilingual (XLM-R) models, embedding-based classifiers (TabPFN), hierarchical ensemble architectures, and large language models. Results show that LLMs achieve the strongest performance on Environmental (Gemma3-27B, F1-macro: 0.61) and Social aspects (gpt-oss 20B, F1-macro: 0.45), while fine-tuned SloBERTa is the best model on Governance classification (F1-macro: 0.54). We then show in a small case study how the best-preforming classifier (gpt-oss) can be applied to investigate ESG aspects for selected companies across a long time frame. 

---
# SemEval-2026 Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization 

**Authors**: Usman Naseem, Robert Geislinger, Juan Ren, Sarah Kohail, Rudy Garrido Veliz, P Sam Sahil, Yiran Zhang, Marco Antonio Stranisci, Idris Abdulmumin, Özge Alaçam, Cengiz Acartürk, Aisha Jabr, Saba Anwar, Abinew Ali Ayele, Elena Tutubalina, Aung Kyaw Htet, Xintong Wang, Surendrabikram Thapa, Tanmoy Chakraborty, Dheeraj Kodati, Sahar Moradizeyveh, Firoj Alam, Ye Kyaw Thu, Shantipriya Parida, Ihsan Ayyub Qazi, Lilian Wanzare, Nelson Odhiambo Onyango, Clemencia Siro, Ibrahim Said Ahmad, Adem Chanie Ali, Martin Semmann, Chris Biemann, Shamsuddeen Hassan Muhammad, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2604.06817)  

**Abstract**: We present SemEval-2026 Task 9, a shared task on online polarization detection, covering 22 languages and comprising over 110K annotated instances. Each data instance is multi-labeled with the presence of polarization, polarization type, and polarization manifestation. Participants were asked to predict labels in three sub-tasks: (1) detecting the presence of polarization, (2) identifying the type of polarization, and (3) recognizing the polarization manifestation. The three tasks attracted over 1,000 participants worldwide and more than 10k submission on Codabench. We received final submissions from 67 teams and 73 system description papers. We report the baseline results and analyze the performance of the best-performing systems, highlighting the most common approaches and the most effective methods across different subtasks and languages. The dataset of this task is publicly available. 

---
# AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation 

**Authors**: Guanran Luo, Wentao Qiu, Wanru Zhao, Wenhan Lv, Zhongquan Jian, Meihong Wang, Qingqiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06812)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in long-form generation, yet their application is hindered by the hallucination problem. While Uncertainty Quantification (UQ) is essential for assessing reliability, the complex structure makes reliable aggregation across heterogeneous themes difficult, in addition, existing methods often overlook the nuance of neutral information and suffer from the high computational cost of fine-grained decomposition. To address these challenges, we propose AGSC (Adaptive Granularity and GMM-based Semantic Clustering), a UQ framework tailored for long-form generation. AGSC first uses NLI neutral probabilities as triggers to distinguish irrelevance from uncertainty, reducing unnecessary computation. It then applies Gaussian Mixture Model (GMM) soft clustering to model latent semantic themes and assign topic-aware weights for downstream aggregation. Experiments on BIO and LongFact show that AGSC achieves state-of-the-art correlation with factuality while reducing inference time by about 60% compared to full atomic decomposition. 

---
# Cognitive Loop of Thought: Reversible Hierarchical Markov Chain for Efficient Mathematical Reasoning 

**Authors**: Jia-Chen Zhang, Zheng Zhou, Yu-Jie Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.06805)  

**Abstract**: Multi-step Chain-of-Thought (CoT) has significantly advanced the mathematical reasoning capabilities of LLMs by leveraging explicit reasoning steps. However, the widespread adoption of Long CoT often results in sequence lengths that exceed manageable computational limits. While existing approaches attempt to alleviate this by reducing KV Cache redundancy via Markov chain-like structures, they introduce two critical limitations: inherent memorylessness (loss of context) and limited backward reasoning capability. To address these limitations, we propose a novel Chain-of-Thought framework based on Reversible Hierarchical Markov Chain, termed Cognitive Loop of Thought (CLoT), and a backward reasoning dataset CLoT-Instruct. In CLoT, problems are decomposed into sub-problems with hierarchical dependencies. Inspired by human cognitive processes, we introduce a backward verification mechanism at each hierarchical layer. Furthermore, we implement a pruning strategy: once higher-level sub-problems are verified, redundant lower-level sub-problems are pruned to maximize efficiency. This approach effectively mitigates error propagation and enhances reasoning robustness. Experiments on four mathematical benchmarks demonstrate the effectiveness of our method. Notably, on the AddSub dataset using GPT-4o-mini, CLoT achieves 99.0% accuracy, outperforming traditional CoT and CoT-SC by 4.1% and 2.9%, respectively. 

---
# Beyond Accuracy: Diagnosing Algebraic Reasoning Failures in LLMs Across Nine Complexity Dimensions 

**Authors**: Parth Patil, Dhruv Kumar, Yash Sinha, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2604.06799)  

**Abstract**: Algebraic reasoning remains one of the most informative stress tests for large language models, yet current benchmarks provide no mechanism for attributing failure to a specific cause. When a model fails an algebraic problem, a single accuracy score cannot reveal whether the expression was too deeply nested, the operator too uncommon, the intermediate state count too high, or the dependency chain too long. Prior work has studied individual failure modes in isolation, but no framework has varied each complexity factor independently under strict experimental control. No prior system has offered automatic generation and verification of problems of increasing complexity to track model progress over time. We introduce a nine-dimension algebraic complexity framework in which each factor is varied independently while all others are held fixed, with problem generation and verification handled by a parametric pipeline requiring no human annotation. Each dimension is grounded in a documented LLM failure mode and captures a structurally distinct aspect of algebraic difficulty, including expression nesting depth, simultaneous intermediate result count, sub-expression complexity, operator hardness, and dependent reasoning chain length. We evaluated seven instruction-tuned models spanning 8B to 235B parameters across all nine dimensions and find that working memory is the dominant scale-invariant bottleneck. Every model collapses between 20 and 30 parallel branches regardless of parameter count, pointing to a hard architectural constraint rather than a solvable capacity limitation. Our analysis further identifies a minimal yet diagnostically sufficient subset of five dimensions that together span the full space of documented algebraic failure modes, providing a complete complexity profile of a model's algebraic reasoning capacity. 

---
# GCoT-Decoding: Unlocking Deep Reasoning Paths for Universal Question Answering 

**Authors**: Guanran Luo, Wentao Qiu, Zhongquan Jian, Meihong Wang, Qingqiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06794)  

**Abstract**: Chain-of-Thought reasoning can enhance large language models, but it requires manually designed prompts to guide the model. Recently proposed CoT-decoding enables the model to generate CoT-style reasoning paths without prompts, but it is only applicable to problems with fixed answer sets. To address this limitation, we propose a general decoding strategy GCoT-decoding that extends applicability to a broader range of question-answering tasks. GCoT-decoding employs a two-stage branching method combining Fibonacci sampling and heuristic error backtracking to generate candidate decoding paths. It then splits each path into a reasoning span and an answer span to accurately compute path confidence, and finally aggregates semantically similar paths to identify a consensus answer, replacing traditional majority voting. We conduct extensive experiments on six datasets covering both fixed and free QA tasks. Our method not only maintains strong performance on fixed QA but also achieves significant improvements on free QA, demonstrating its generality. 

---
# When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning 

**Authors**: Yang Xiang, Yixin Ji, Ruotao Xu, Dan Qiao, Zheming Yang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06787)  

**Abstract**: Large reasoning models (LRMs) have achieved remarkable performance in complex reasoning tasks, driven by their powerful inference-time scaling capability. However, LRMs often suffer from overthinking, which results in substantial computational redundancy and significantly reduces efficiency. Early-exit methods aim to mitigate this issue by terminating reasoning once sufficient evidence has been generated, yet existing approaches mostly rely on handcrafted or empirical indicators that are unreliable and impractical. In this work, we introduce Dynamic Thought Sufficiency in Reasoning (DTSR), a novel framework for efficient reasoning that enables the model to dynamically assess the sufficiency of its chain-of-thought (CoT) and determine the optimal point for early exit. Inspired by human metacognition, DTSR operates in two stages: (1) Reflection Signal Monitoring, which identifies reflection signals as potential cues for early exit, and (2) Thought Sufficiency Check, which evaluates whether the current CoT is sufficient to derive the final answer. Experimental results on the Qwen3 models show that DTSR reduces reasoning length by 28.9%-34.9% with minimal performance loss, effectively mitigating overthinking. We further discuss overconfidence in LRMs and self-evaluation paradigms, providing valuable insights for early-exit reasoning. 

---
# Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation 

**Authors**: Zhiyu Cao, Peifeng Li, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06784)  

**Abstract**: Previous research on multi-party dialogue generation has predominantly leveraged structural information inherent in dialogues to directly inform the generation process. However, the prevalence of colloquial expressions and incomplete utterances in dialogues often impedes comprehension and weakens the fidelity of dialogue structure representations, which is particularly pronounced in multi-party dialogues. In this work, we propose a novel framework DRCR (Discourse coherence and Response-guided Context Rewriting) to improve multi-party dialogue generation through dialogue context rewriting. Specifically, DRCR employs two complementary feedback signals, discourse coherence and response quality, to construct preference data for both context rewriting and response generation. Moreover, we propose a dynamic self-evolution learning method that allows the rewriter and responder to continuously enhance their capabilities through mutual interaction in an iterative training loop. Comprehensive experiments conducted on four multi-party dialogue datasets substantiate the effectiveness of DRCR. 

---
# Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search 

**Authors**: Zhiyu Cao, Peifeng Li, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06771)  

**Abstract**: Conversational Query Rewriting (CQR) aims to rewrite ambiguous queries to achieve more efficient conversational search. Early studies have predominantly focused on the rewriting in isolation, ignoring the feedback from query rewrite, passage retrieval and response generation in the rewriting process. To address this issue, we propose Multi-Faceted Self-Consistent Preference Aligned CQR (MSPA-CQR). Specifically, we first construct self-consistent preference alignment data from three dimensions (rewriting, retrieval, and response) to generate more diverse rewritten queries. Then we propose prefix guided multi-faceted direct preference optimization to learn preference information from three different dimensions. The experimental results show that our MSPA-CQR is effective in both in- and out-of-distribution scenarios. 

---
# TeamLLM: A Human-Like Team-Oriented Collaboration Framework for Multi-Step Contextualized Tasks 

**Authors**: Xiangyu Wang, Jin Wu, Haoran Shi, Wei Xia, Jiarui Yu, Chanjin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.06765)  

**Abstract**: Recently, multi-Large Language Model (LLM) frameworks have been proposed to solve contextualized tasks. However, these frameworks do not explicitly emulate human team role division, which may lead to a single perspective, thereby weakening performance on multi-step contextualized tasks. To address this issue, we propose TeamLLM, a human-like Team-Oriented Multi-LLM Collaboration Framework. TeamLLM adopts four team roles with distinct division and employs a three-phase multi-LLM collaboration for multi-step contextualized tasks. To evaluate the effectiveness of TeamLLM on multi-step contextualized tasks, we propose Contextually-Grounded and Procedurally-Structured tasks (CGPST) and construct the CGPST benchmark. This benchmark has four core features: contextual grounding, procedural structure, process-oriented evaluation and multi-dimensional assessment. We evaluate ten popular LLMs on CGPST at overall-level, step-level, and dimension-level. Results show that TeamLLM substantially improves performance on CGPST. We release the benchmark with scenarios, full-process responses and human scores from ten LLMs. The code and data are available at this https URL. 

---
# Multilingual Cognitive Impairment Detection in the Era of Foundation Models 

**Authors**: Damar Hoogland, Boshko Koloski, Jaya Caporusso, Tine Kolenik, Ana Zwitter Vitez, Senja Pollak, Christina Manouilidou, Matthew Purver  

**Link**: [PDF](https://arxiv.org/pdf/2604.06758)  

**Abstract**: We evaluate cognitive impairment (CI) classification from transcripts of speech in English, Slovene, and Korean. We compare zero-shot large language models (LLMs) used as direct classifiers under three input settings -- transcript-only, linguistic-features-only, and combined -- with supervised tabular approaches trained under a leave-one-out protocol. The tabular models operate on engineered linguistic features, transcript embeddings, and early or late fusion of both modalities. Across languages, zero-shot LLMs provide competitive no-training baselines, but supervised tabular models generally perform better, particularly when engineered linguistic features are included and combined with embeddings. Few-shot experiments focusing on embeddings indicate that the value of limited supervision is language-dependent, with some languages benefiting substantially from additional labelled examples while others remain constrained without richer feature representations. Overall, the results suggest that, in small-data CI detection, structured linguistic signals and simple fusion-based classifiers remain strong and reliable signals. 

---
# How Long Reasoning Chains Influence LLMs' Judgment of Answer Factuality 

**Authors**: Minzhu Tu, Shiyu Ni, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2604.06756)  

**Abstract**: Large language models (LLMs) has been widely adopted as a scalable surrogate for human evaluation, yet such judges remain imperfect and susceptible to surface-level biases. One possible reason is that these judges lack sufficient information in assessing answer correctness. With the rise of reasoning-capable models, exposing a generator's reasoning content to the judge provides richer information and is a natural candidate for improving judgment accuracy. However, its actual impact on judge behavior remains understudied. In this paper, we systematically investigate how access to reasoning chains affects LLM-based judgment across factual question answering (QA) and mathematical reasoning benchmarks. We find that weak judges are easily swayed by reasoning presence, frequently accepting incorrect answers accompanied by fluent reasoning, while strong judges can partially leverage reasoning as informative evidence. Nevertheless, even strong judges are misled by seemingly high-quality reasoning chains. Controlled experiments further reveal that both fluency and factuality of reasoning chains are critical signals driving judge decisions. These findings highlight the need for more robust LLM judges that can distinguish genuine reasoning quality from superficial fluency when evaluating modern reasoning models. 

---
# Select-then-Solve: Paradigm Routing as Inference-Time Optimization for LLM Agents 

**Authors**: Heng Zhou, Zelin Tan, Zhemeng Zhang, Yutao Fan, Yibing Lin, Li Kang, Xiufeng Song, Rui Li, Songtao Huang, Ao Yu, Yuchen Fan, Yanxu Chen, Kaixin Xu, Xiaohong Liu, Yiran Qin, Philip Torr, Chen Zhang, Zhenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.06753)  

**Abstract**: When an LLM-based agent improves on a task, is the gain from the model itself or from the reasoning paradigm wrapped around it? We study this question by comparing six inference-time paradigms, namely Direct, CoT, ReAct, Plan-Execute, Reflection, and ReCode, across four frontier LLMs and ten benchmarks, yielding roughly 18,000 runs. We find that reasoning structure helps dramatically on some tasks but hurts on others: ReAct improves over Direct by 44pp on GAIA, while CoT degrades performance by 15pp on HumanEval. No single paradigm dominates, and oracle per-task selection beats the best fixed paradigm by 17.1pp on average. Motivated by this complementarity, we propose a select-then-solve approach: before answering each task, a lightweight embedding-based router selects the most suitable paradigm. Across four models, the router improves average accuracy from 47.6% to 53.1%, outperforming the best fixed paradigm at 50.3% by 2.8pp and recovering up to 37% of the oracle gap. In contrast, zero-shot self-routing only works for GPT-5 at 67.1% and fails for weaker models, all trailing the learned router. Our results argue that reasoning paradigm selection should be a per-task decision made by a learned router, not a fixed architectural choice. 

---
# StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference 

**Authors**: Zhirui Chen, Peiyang Liu, Ling Shao  

**Link**: [PDF](https://arxiv.org/pdf/2604.06746)  

**Abstract**: As Large Language Models (LLMs) scale to support context windows exceeding one million tokens, the linear growth of Key-Value (KV) cache imposes severe memory capacity and bandwidth bottlenecks, constraining the efficiency of long-context inference. Existing compression approaches typically prioritize tokens based on local saliency metrics to decouple prefill computation from decoding memory. However, these methods often rely on local saliency snapshots at a specific layer, thereby systematically discarding tokens that act as global information hubs across the network depth but appear temporarily dormant at the specific layer selected for pruning. To address this limitation, we propose StructKV, a structure-aware KV cache compression framework that introduces three core innovations: First, Global In-Degree Centrality aggregates attention patterns across the network depth to identify global information hubs. Second, Dynamic Pivot Detection utilizes information-theoretic metrics to adaptively locate the optimal layer for compression. Finally, Structural Propagation and Decoupling separates the computational budget from the memory storage budget. Experimental results on the LongBench and RULER benchmarks demonstrate that StructKV effectively preserves long-range dependencies and retrieval robustness. 

---
# Luwen Technical Report 

**Authors**: Yiquan Wu, Yuhang Liu, Yifei Liu, Ang Li, Siying Zhou, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06737)  

**Abstract**: Large language models have demonstrated remarkable capabilities across a wide range of natural language processing tasks, yet their application in the legal domain remains challenging due to the specialized terminology, complex reasoning requirements, and rapidly evolving legal knowledge involved. In this paper, we present Luwen, an open-source Chinese legal language model built upon the Baichuan foundation model through three key techniques: continual pre-training on a large-scale legal corpus, supervised fine-tuning with carefully curated legal instruction data, and retrieval-augmented generation integrated with a comprehensive legal knowledge base. We evaluate Luwen on five representative legal tasks spanning both prediction and generation settings, including legal judgment prediction, judicial examination, legal text summarization, law article question answering, and judicial decision reasoning. Experimental results show that Luwen outperforms several strong baselines, demonstrating the effectiveness of our approach in adapting general-purpose language models to the legal domain. 

---
# SQLStructEval: Structural Evaluation of LLM Text-to-SQL Generation 

**Authors**: Yixi Zhou, Fan Zhang, Zhiqiao Guo, Yu Chen, Haipeng Zhang, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.06736)  

**Abstract**: Despite strong performance on Text-to-SQL benchmarks, it remains unclear whether LLM-generated SQL programs are structurally reliable. In this work, we investigate the structural behavior of LLM-generated SQL queries and introduce SQLStructEval, a framework for analyzing program structures through canonical abstract syntax tree (AST) representations. Our experiments on the Spider benchmark show that modern LLMs often produce structurally diverse queries for the same input, even when execution results are correct, and that such variance is frequently triggered by surface-level input changes such as paraphrases or schema presentation. We further show that generating queries in a structured space via a compile-style pipeline can improve both execution accuracy and structural consistency. These findings suggest that structural reliability is a critical yet overlooked dimension for evaluating LLM-based program generation systems. Our code is available at this https URL. 

---
# TEC: A Collection of Human Trial-and-error Trajectories for Problem Solving 

**Authors**: Xinkai Zhang, Jingtao Zhan, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2604.06734)  

**Abstract**: Trial-and-error is a fundamental strategy for humans to solve complex problems and a necessary capability for Artificial Intelligence (AI) systems operating in real-world environments. Although several trial-and-error AI techniques have recently been proposed, most of them rely on simple heuristics designed by researchers and achieve limited performance gains. The core issue is the absence of appropriate data: current models cannot learn from detailed records of how humans actually conduct trial-and-error in practice. To address this gap, we introduce a data annotation platform and a corresponding dataset, termed Trial-and-Error Collection (TEC). The platform records users' complete trajectories across multiple trials and collects their reflections after receiving error feedback. Using this platform, we record the problem-solving processes of 46 participants on 58 tasks, resulting in 5,370 trial trajectories along with error reflections across 41,229 webpages. With this dataset, we observe that humans achieve substantially higher accuracy compared to LLMs, which demonstrates that humans are more effective in trial-and-error than LLMs. We believe that the TEC platform and dataset provide a valuable foundation for understanding human trial-and-error behavior and for developing more capable AI systems. Platform and dataset are publicly available. 

---
# Adaptive Prompt Structure Factorization: A Framework for Self-Discovering and Optimizing Compositional Prompt Programs 

**Authors**: Haoyue Liu, Zhichao Wang, Yongxin Guo, Haoran Shou, Xiaoying Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06699)  

**Abstract**: Automated prompt optimization is crucial for eliciting reliable reasoning from large language models (LLMs), yet most API-only prompt optimizers iteratively edit monolithic prompts, coupling components and obscuring credit assignment, limiting controllability, and wasting tokens. We propose Adaptive Prompt Structure Factorization (aPSF), an API-only framework (prompt-in/text-out; no access to model internals) that uses an Architect model to discover task-specific prompt structures as semantic factors. aPSF then performs interventional, single-factor updates: interventional factor-level scoring estimates each factor's marginal contribution via validation-performance changes, and error-guided factor selection routes updates to the current dominant failure source for more sample-efficient optimization. Across multiple advanced reasoning benchmarks, aPSF outperforms strong baselines including principle-aware optimizers, improving accuracy by up to +2.16 percentage points on average, and reduces optimization cost by 45--87% tokens on MultiArith while reaching peak validation in 1 step. 

---
# ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding 

**Authors**: Xuanle Zhao, Xinyuan Cai, Xiang Cheng, Xiuyi Chen, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06685)  

**Abstract**: While Vision-Language Models (VLMs) have demonstrated significant potential in chemical visual understanding, current models are predominantly optimized for direct visual question-answering tasks. This paradigm often results in "black-box" systems that fail to utilize the inherent capability of Large Language Models (LLMs) to infer underlying reaction mechanisms. In this work, we introduce ChemVLR, a chemical VLM designed to prioritize reasoning within the perception process. Unlike conventional chemical VLMs, ChemVLR analyzes visual inputs in a fine-grained manner by explicitly identifying granular chemical descriptors, such as functional groups, prior to generating answers. This approach ensures the production of explicit and interpretable reasoning paths for complex visual chemical problems. To facilitate this methodology, we implement a cross-modality reverse-engineering strategy, combined with a rigorous filtering pipeline, to curate a large-scale reasoning-and-captioning dataset comprising 760k high-quality samples across molecular and reaction tasks. Furthermore, we adopt a three-stage training framework that systemically builds model perception and reasoning capacity. Experiments demonstrate that ChemVLR achieves state-of-the-art (SOTA) performance, surpassing both leading proprietary models and domain-specific open-source baselines. We also provide comprehensive ablation studies to validate our training strategy and data generation designs. Code and model weights will be available at this https URL. 

---
# Between Century and Poet: Graph-Based Lexical Semantic Change in Persian Poetry 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh, Mohammadali Keshtparvar  

**Link**: [PDF](https://arxiv.org/pdf/2604.06674)  

**Abstract**: Meaning in Persian poetry is both historical and relational. Words persist through literary tradition while shifting their force through changing constellations of neighbors, rhetorical frames, and poetic voices. This study examines that process using aligned Word2Vec spaces combined with graph-based neighborhood analysis across centuries and major poets. Rather than modeling semantic change as vector displacement alone, it treats lexical history as the rewiring of local semantic graphs: the gain and loss of neighbors, shifts in bridge roles, and movement across communities. The analysis centers on twenty target words, anchored by five recurrent reference terms: Earth, Night, two wine terms, and Heart. Surrounding them are affective, courtly, elemental, and Sufi concepts such as Love, Sorrow, Dervish, King, Annihilation, and Truth. These words exhibit distinct patterns of change. Night is more time-sensitive, Earth more poet-sensitive, and Heart shows continuity despite graph-role mobility. The two wine terms highlight probe sensitivity: one is broad and semantically diffuse, while the other is narrower and more stable. A lexical audit confirms that the corpus contains historically driven terms, poet-specific usages, and sparsely attested mystical vocabulary requiring caution. Overall, semantic change in Persian poetry is better captured as neighborhood rewiring than as abstract drift. For Digital Humanities, this approach restores local structure to computational analysis and supports interpretations closer to literary practice: persistence, migration, mediation, and selective transformation. 

---
# A Graph-Enhanced Defense Framework for Explainable Fake News Detection with LLM 

**Authors**: Bo Wang, Jing Ma, Hongzhan Lin, Zhiwei Yang, Ruichao Yang, Yuan Tian, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06666)  

**Abstract**: Explainable fake news detection aims to assess the veracity of news claims while providing human-friendly explanations. Existing methods incorporating investigative journalism are often inefficient and struggle with breaking news. Recent advances in large language models (LLMs) enable leveraging externally retrieved reports as evidence for detection and explanation generation, but unverified reports may introduce inaccuracies. Moreover, effective explainable fake news detection should provide a comprehensible explanation for all aspects of a claim to assist the public in verifying its accuracy. To address these challenges, we propose a graph-enhanced defense framework (G-Defense) that provides fine-grained explanations based solely on unverified reports. Specifically, we construct a claim-centered graph by decomposing the news claim into several sub-claims and modeling their dependency relationships. For each sub-claim, we use the retrieval-augmented generation (RAG) technique to retrieve salient evidence and generate competing explanations. We then introduce a defense-like inference module based on the graph to assess the overall veracity. Finally, we prompt an LLM to generate an intuitive explanation graph. Experimental results demonstrate that G-Defense achieves state-of-the-art performance in both veracity detection and the quality of its explanations. 

---
# A Parameter-Efficient Transfer Learning Approach through Multitask Prompt Distillation and Decomposition for Clinical NLP 

**Authors**: Cheng Peng, Mengxian Lyu, Ziyi Chen, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06650)  

**Abstract**: Existing prompt-based fine-tuning methods typically learn task-specific prompts independently, imposing significant computing and storage overhead at scale when deploying multiple clinical natural language processing (NLP) systems. We present a multitask prompt distillation and decomposition framework that learns a single shared metaprompt from 21 diverse clinical source tasks and adapts it to unseen target tasks with fewer than 0.05% trainable parameters. Evaluated across five clinical NLP task types (named entity recognition, relation extraction, question answering, natural language inference, and summarization) on 10 held-out target datasets using three backbone models (LLaMA 3.1 8B, Meditron3 8B, gpt-oss 20B), our framework consistently outperforms LoRA by 1.5~1.7% despite using orders of magnitude fewer parameters, and exceeds single-task prompt tuning by 6.1~6.6%. The gpt-oss 20B model achieves the highest overall performance, particularly on clinical reasoning tasks. The strong zero- and few-shot performance demonstrates better transferability of the shared prompt representation. 

---
# Feedback Adaptation for Retrieval-Augmented Generation 

**Authors**: Jihwan Bang, Seunghan Yang, Kyuhong Shim, Simyung Chang, Juntae Lee, Sungha Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.06647)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are typically evaluated under static assumptions, despite being frequently corrected through user or expert feedback in deployment. Existing evaluation protocols focus on overall accuracy and fail to capture how systems adapt after feedback is introduced. We introduce feedback adaptation as a problem setting for RAG systems, which asks how effectively and how quickly corrective feedback propagates to future queries. To make this behavior measurable, we propose two evaluation axes: correction lag, which captures the delay between feedback provision and behavioral change, and post-feedback performance, which measures reliability on semantically related queries after feedback. Using these metrics, we show that training-based approaches exhibit a trade-off between delayed correction and reliable adaptation. We further propose PatchRAG, a minimal inference-time instantiation that incorporates feedback without retraining, demonstrating immediate correction and strong post-feedback generalization under the proposed evaluation. Our results highlight feedback adaptation as a previously overlooked dimension of RAG system behavior in interactive settings. 

---
# DiffuMask: Diffusion Language Model for Token-level Prompt Pruning 

**Authors**: Caleb Zheng, Jyotika Singh, Fang Tu, Weiyi Sun, Sujeeth Bharadwaj, Yassine Benajiba, Sujith Ravi, Eli Shlizerman, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.06627)  

**Abstract**: In-Context Learning and Chain-of-Thought prompting improve reasoning in large language models (LLMs). These typically come at the cost of longer, more expensive prompts that may contain redundant information. Prompt compression based on pruning offers a practical solution, yet existing methods rely on sequential token removal which is computationally intensive. We present DiffuMask, a diffusion-based framework integrating hierarchical shot-level and token-level pruning signals, that enables rapid and parallel prompt pruning via iterative mask prediction. DiffuMask substantially accelerates the compression process via masking multiple tokens in each denoising step. It offers tunable control over retained content, preserving essential reasoning context and achieving up to 80\% prompt length reduction. Meanwhile, it maintains or improves accuracy across in-domain, out-of-domain, and cross-model settings. Our results show that DiffuMask provides a generalizable and controllable framework for prompt compression, facilitating faster and more reliable in-context reasoning in LLMs. 

---
# The Detection--Extraction Gap: Models Know the Answer Before They Can Say It 

**Authors**: Hanyang Wang, Mingxuan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06613)  

**Abstract**: Modern reasoning models continue generating long after the answer is already determined. Across five model configurations, two families, and three benchmarks, we find that \textbf{52--88\% of chain-of-thought tokens are produced after the answer is recoverable} from a partial prefix. This post-commitment generation reveals a structural phenomenon: the \textbf{detection--extraction gap}. Free continuations from early prefixes recover the correct answer even at 10\% of the trace, while forced extraction fails on 42\% of these cases. The answer is recoverable from the model state, yet prompt-conditioned decoding fails to extract it. We formalize this mismatch via a total-variation bound between free and forced continuation distributions, yielding quantitative estimates of suffix-induced shift. Exploiting this asymmetry, we propose Black-box Adaptive Early Exit (\BAEE{}), which uses free continuations for both detection and extraction, truncating \textbf{70--78\% of serial generation} while \textbf{improving accuracy by 1--5\,pp} across all models. For thinking-mode models, early exit prevents post-commitment overwriting, yielding gains of up to 5.8\,pp; a cost-optimized variant achieves 68--73\% reduction at a median of 9 API calls. Code is available at this https URL. 

---
# Scientific Knowledge-driven Decoding Constraints Improving the Reliability of LLMs 

**Authors**: Maotian Ma, Zheni Zeng, Zhenghao Liu, Yukun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.06603)  

**Abstract**: Large language models (LLMs) have shown strong knowledge reserves and task-solving capabilities, but still face the challenge of severe hallucination, hindering their practical application. Though scientific theories and rules can efficiently direct the behaviors of human manipulators, LLMs still do not utilize these highly-condensed knowledge sufficiently through training or prompting. To address this issue, we propose \textbf{SciDC}, an LLM generation method that integrate subject-specific knowledge with strong constraints. By adopting strong LLMs to automatically convert flexible knowledge into multi-layered, standardized rules, we build an extensible framework to effectively constrain the model generation on domain tasks. Experiments on scientific tasks including industrial formulation design, clinical tumor diagnosis and retrosynthesis planning, consistently demonstrate the effectiveness of our method, achieving a 12\% accuracy improvement on average compared with vanilla generation. We further discuss the potential of LLMs in automatically inductively summarizing highly-condensed knowledge, looking ahead to practical solutions for accelerating the overall scientific research process. All the code of this paper can be obtained (this https URL). 

---
# Scoring Edit Impact in Grammatical Error Correction via Embedded Association Graphs 

**Authors**: Qiyuan Xiao, Xiaoman Wang, Yunshi Lan  

**Link**: [PDF](https://arxiv.org/pdf/2604.06573)  

**Abstract**: A Grammatical Error Correction (GEC) system produces a sequence of edits to correct an erroneous sentence. The quality of these edits is typically evaluated against human annotations. However, a sentence may admit multiple valid corrections, and existing evaluation settings do not fully accommodate diverse application scenarios. Recent meta-evaluation approaches rely on human judgments across multiple references, but they are difficult to scale to large datasets. In this paper, we propose a new task, Scoring Edit Impact in GEC, which aims to automatically estimate the importance of edits produced by a GEC system. To address this task, we introduce a scoring framework based on an embedded association graph. The graph captures latent dependencies among edits and syntactically related edits, grouping them into coherent groups. We then perform perplexity-based scoring to estimate each edit's contribution to sentence fluency. Experiments across 4 GEC datasets, 4 languages, and 4 GEC systems demonstrate that our method consistently outperforms a range of baselines. Further analysis shows that the embedded association graph effectively captures cross-linguistic structural dependencies among edits. 

---
# LLM-based Schema-Guided Extraction and Validation of Missing-Person Intelligence from Heterogeneous Data Sources 

**Authors**: Joshua Castillo, Ravi Mukkamala  

**Link**: [PDF](https://arxiv.org/pdf/2604.06571)  

**Abstract**: Missing-person and child-safety investigations rely on heterogeneous case documents, including structured forms, bulletin-style posters, and narrative web profiles. Variations in layout, terminology, and data quality impede rapid triage, large-scale analysis, and search-planning workflows. This paper introduces the Guardian Parser Pack, an AI-driven parsing and normalization pipeline that transforms multi-source investigative documents into a unified, schema-compliant representation suitable for operational review and downstream spatial modeling. The proposed system integrates (i) multi-engine PDF text extraction with Optical Character Recognition (OCR) fallback, (ii) rule-based source identification with source-specific parsers, (iii) schema-first harmonization and validation, and (iv) an optional Large Language Model (LLM)-assisted extraction pathway incorporating validator-guided repair and shared geocoding services. We present the system architecture, key implementation decisions, and output design, and evaluate performance using both gold-aligned extraction metrics and corpus-level operational indicators. On a manually aligned subset of 75 cases, the LLM-assisted pathway achieved substantially higher extraction quality than the deterministic comparator (F1 = 0.8664 vs. 0.2578), while across 517 parsed records per pathway it also improved aggregate key-field completeness (96.97\% vs. 93.23\%). The deterministic pathway remained much faster (mean runtime 0.03 s/record vs. 3.95 s/record for the LLM pathway). In the evaluated run, all LLM outputs passed initial schema validation, so validator-guided repair functioned as a built-in safeguard rather than a contributor to the observed gains. These results support controlled use of probabilistic AI within a schema-first, auditable pipeline for high-stakes investigative settings. 

---
# To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs 

**Authors**: Zohaib Khan, Mustafa Dogan, Ifeoma Okoh, Pouya Sadeghi, Siddhartha Shrestha, Sergius Justus Nyah, Mahmoud O. Mokhiamar, Michael J. Ryan, Tarek Naous  

**Link**: [PDF](https://arxiv.org/pdf/2604.06552)  

**Abstract**: Misinformation is on the rise, and the strong writing capabilities of LLMs lower the barrier for malicious actors to produce and disseminate false information. We study how LLMs behave when prompted to spread misinformation across languages and target countries, and introduce GlobalLies, a multilingual parallel dataset of 440 misinformation generation prompt templates and 6,867 entities, spanning 8 languages and 195 countries. Using both human annotations and large-scale LLM-as-a-judge evaluations across hundreds of thousands of generations from state-of-the-art models, we show that misinformation generation varies systematically based on the country being discussed. Propagation of lies by LLMs is substantially higher in many lower-resource languages and for countries with a lower Human Development Index (HDI). We find that existing mitigation strategies provide uneven protection: input safety classifiers exhibit cross-lingual gaps, and retrieval-augmented fact-checking remains inconsistent across regions due to unequal information availability. We release GlobalLies for research purposes, aiming to support the development of mitigation strategies to reduce the spread of global misinformation: this https URL 

---
# CCD-CBT: Multi-Agent Therapeutic Interaction for CBT Guided by Cognitive Conceptualization Diagram 

**Authors**: Chang Liu, Changsheng Ma, Yongfeng Tao, Bin Hu, Minqiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06551)  

**Abstract**: Large language models show potential for scalable mental-health support by simulating Cognitive Behavioral Therapy (CBT) counselors. However, existing methods often rely on static cognitive profiles and omniscient single-agent simulation, failing to capture the dynamic, information-asymmetric nature of real therapy. We introduce CCD-CBT, a multi-agent framework that shifts CBT simulation along two axes: 1) from a static to a dynamically reconstructed Cognitive Conceptualization Diagram (CCD), updated by a dedicated Control Agent, and 2) from omniscient to information-asymmetric interaction, where the Therapist Agent must reason from inferred client states. We release CCDCHAT, a synthetic multi-turn CBT dataset generated under this framework. Evaluations with clinical scales and expert therapists show that models fine-tuned on CCDCHAT outperform strong baselines in both counseling fidelity and positive-affect enhancement, with ablations confirming the necessity of dynamic CCD guidance and asymmetric agent design. Our work offers a new paradigm for building theory-grounded, clinically-plausible conversational agents. 

---
# The Illusion of Stochasticity in LLMs 

**Authors**: Xiangming Gu, Soham De, Michalis Titsias, Larisa Markeeva, Petar Veličković, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06543)  

**Abstract**: In this work, we demonstrate that reliable stochastic sampling is a fundamental yet unfulfilled requirement for Large Language Models (LLMs) operating as agents. Agentic systems are frequently required to sample from distributions, often inferred from observed data, a process which needs to be emulated by the LLM. This leads to a distinct failure point: while standard RL agents rely on external sampling mechanisms, LLMs fail to map their internal probability estimates to their stochastic outputs. Through rigorous empirical analysis across multiple model families, model sizes, prompting styles, and distributions, we demonstrate the extent of this failure. Crucially, we show that while powerful frontier models can convert provided random seeds to target distributions, their ability to sample directly from specific distributions is fundamentally flawed. 

---
# Does a Global Perspective Help Prune Sparse MoEs Elegantly? 

**Authors**: Zeliang Zhang, Nikhil Ghosh, Jiani Liu, Bin Yu, Xiaodong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06542)  

**Abstract**: Empirical scaling laws for language models have encouraged the development of ever-larger LLMs, despite their growing computational and memory costs. Sparse Mixture-of-Experts (MoEs) offer a promising alternative by activating only a subset of experts per forward pass, improving efficiency without sacrificing performance. However, the large number of expert parameters still leads to substantial memory consumption.
Existing pruning methods typically allocate budgets uniformly across layers, overlooking the heterogeneous redundancy that arises in sparse MoEs. We propose GRAPE (Global Redundancy-Aware Pruning of Experts, a global pruning strategy that dynamically allocates pruning budgets based on cross-layer redundancy. Experiments on Mixtral-8x7B, Mixtral-8x22B, DeepSeek-MoE, Qwen-MoE, and GPT-OSS show that, under the same pruning budget, GRAPE consistently achieves the best average performance. On the three main models reported in the paper, it improves average accuracy over the strongest local baseline by 1.40% on average across pruning settings, with gains of up to 2.45%. 

---
# Fine-tuning Whisper for Pashto ASR: strategies and scale 

**Authors**: Hanif Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2604.06507)  

**Abstract**: Pashto is absent from Whisper's pre-training corpus despite being one of CommonVoice's largest language collections, leaving off-the-shelf models unusable: all Whisper sizes output Arabic, Dari, or Urdu script on Pashto audio, achieving word error rates above 100%. We compare four fine-tuning strategies for whisper-base on CommonVoice Pashto v20: vanilla full fine-tuning, LoRA (rank 64), frozen-encoder (2/6 layers), and multistage Urdu-to-Pashto transfer. We extend vanilla fine-tuning to whisper-small and whisper-large-v3-turbo on CommonVoice Pashto v24 (113 hours). Vanilla fine-tuning achieves WER 21.22% on CV20, outperforming LoRA by 33.36 pp, frozen-encoder by 14.76 pp, and Urdu transfer by 44.56 pp. Frozen-encoder fine-tuning degrades performance on whisper-base (6 encoder layers): layer-function separation does not hold at this depth, and freezing removes a third of trainable capacity. Urdu-to-Pashto transfer fails due to an unverified intermediate checkpoint, phonological mismatch, and insufficient training. On CV24, whisper-small achieves WER 24.89% (2.24 pp over whisper-base at 3.3x parameters); whisper-large-v3-turbo achieves 23.37% (a further 1.52 pp). Diminishing returns indicate whisper-small is the practical optimum at 113 hours. Online augmentation provides 7.25 pp WER benefit over matched training. Error analysis identifies word-final suffix confusion (masculine -ay vs. feminine -a) and retroflex substitutions involving the Pashto-unique consonant /ts/ as dominant failure modes. Fine-tuned checkpoints and evaluation scripts are released on HuggingFace. 

---
# MedConclusion: A Benchmark for Biomedical Conclusion Generation from Structured Abstracts 

**Authors**: Weiyue Li, Ruizhi Qian, Yi Li, Yongce Li, Yunfan Long, Jiahui Cai, Yan Luo, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06505)  

**Abstract**: Large language models (LLMs) are widely explored for reasoning-intensive research tasks, yet resources for testing whether they can infer scientific conclusions from structured biomedical evidence remain limited. We introduce $\textbf{MedConclusion}$, a large-scale dataset of $\textbf{5.7M}$ PubMed structured abstracts for biomedical conclusion generation. Each instance pairs the non-conclusion sections of an abstract with the original author-written conclusion, providing naturally occurring supervision for evidence-to-conclusion reasoning. MedConclusion also includes journal-level metadata such as biomedical category and SJR, enabling subgroup analysis across biomedical domains. As an initial study, we evaluate diverse LLMs under conclusion and summary prompting settings and score outputs with both reference-based metrics and LLM-as-a-judge. We find that conclusion writing is behaviorally distinct from summary writing, strong models remain closely clustered under current automatic metrics, and judge identity can substantially shift absolute scores. MedConclusion provides a reusable data resource for studying scientific evidence-to-conclusion reasoning. Our code and data are available at: this https URL. 

---
# Closing the Speech-Text Gap with Limited Audio for Effective Domain Adaptation in LLM-Based ASR 

**Authors**: Thibault Bañeras-Roux, Sergio Burdisso, Esaú Villatoro-Tello, Dairazalia Sánchez-Cortés, Shiran Liu, Severin Baroudi, Shashi Kumar, Hasindri Watawana, Manjunath K E, Kadri Hacioglu, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2604.06487)  

**Abstract**: Conventional end-to-end automatic speech recognition (ASR) systems rely on paired speech-text data for domain adaptation. Recent LLM-based ASR architectures connect a speech encoder to a large language model via a projection module, enabling adaptation with text-only data. However, this introduces a modality gap, as the LLM is not exposed to the noisy representations produced by the speech projector. We investigate whether small amounts of speech can mitigate this mismatch. We compare three strategies: text-only adaptation, paired speech-text adaptation, and mixed batching (MB), which combines both. Experiments in in-domain and out-of-domain settings show that even limited speech consistently improves performance. Notably, MB using only 10% of the target-domain (less than 4 hours) speech achieves word error rates comparable to, or better than, conventional ASR fine-tuning with the full dataset, indicating that small amounts of speech provide a strong modality-alignment signal. 

---
# ValueGround: Evaluating Culture-Conditioned Visual Value Grounding in MLLMs 

**Authors**: Zhipin Wang, Christoph Leiter, Christian Frey, Mohamed Hesham Ibrahim Abdalla, Josif Grabocka, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2604.06484)  

**Abstract**: Cultural values are expressed not only through language but also through visual scenes and everyday social practices. Yet existing evaluations of cultural values in language models are almost entirely text-only, making it unclear whether models can ground culture-conditioned judgments when response options are visualized. We introduce ValueGround, a benchmark for evaluating culture-conditioned visual value grounding in multimodal large language models (MLLMs). Built from World Values Survey (WVS) questions, ValueGround uses minimally contrastive image pairs to represent opposing response options while controlling irrelevant variation. Given a country, a question, and an image pair, a model must choose the image that best matches the country's value tendency without access to the original response-option texts. Across six MLLMs and 13 countries, average accuracy drops from 72.8% in the text-only setting to 65.8% when options are visualized, despite 92.8% accuracy on option-image alignment. Stronger models are more robust, but all remain prone to prediction reversals. Our benchmark provides a controlled testbed for studying cross-modal transfer of culture-conditioned value judgments. 

---
# DataSTORM: Deep Research on Large-Scale Databases using Exploratory Data Analysis and Data Storytelling 

**Authors**: Shicheng Liu, Yucheng Jiang, Sajid Farook, Camila Nicollier Sanchez, David Fernando Castro Pena, Monica S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2604.06474)  

**Abstract**: Deep research with Large Language Model (LLM) agents is emerging as a powerful paradigm for multi-step information discovery, synthesis, and analysis. However, existing approaches primarily focus on unstructured web data, while the challenges of conducting deep research over large-scale structured databases remain relatively underexplored. Unlike web-based research, effective data-centric research requires more than retrieval and summarization and demands iterative hypothesis generation, quantitative reasoning over structured schemas, and convergence toward a coherent analytical narrative.
In this paper, we present DataSTORM, an LLM-based agentic system capable of autonomously conducting research across both large-scale structured databases and internet sources. Grounded in principles from Exploratory Data Analysis and Data Storytelling, DataSTORM reframes deep research over structured data as a thesis-driven analytical process: discovering candidate theses from data, validating them through iterative cross-source investigation, and developing them into coherent analytical narratives. We evaluate DataSTORM on InsightBench, where it achieves a new state-of-the-art result with a 19.4% relative improvement in insight-level recall and 7.2% in summary-level score. We further introduce a new dataset built on ACLED, a real-world complex database, and demonstrate that DataSTORM outperforms proprietary systems such as ChatGPT Deep Research across both automated metrics and human evaluations. 

---
# Multi-objective Evolutionary Merging Enables Efficient Reasoning Models 

**Authors**: Mario Iacobelli, Adrian Robert Minut, Tommaso Mencattini, Donato Crisostomi, Andrea Santilli, Iacopo Masi, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2604.06465)  

**Abstract**: Reasoning models have demonstrated remarkable capabilities in solving complex problems by leveraging long chains of thought. However, this more deliberate reasoning comes with substantial computational overhead at inference time. The Long-to-Short (L2S) reasoning problem seeks to maintain high accuracy using fewer tokens, but current training-free model merging approaches rely on scalarized, fixed-hyperparameter arithmetic methods that are highly brittle and force suboptimal compromises. To address this gap, we introduce Evo-L2S, a novel framework that formulates L2S reasoning as a multi-objective optimization challenge. By leveraging evolutionary model merging, Evo-L2S explicitly optimizes the trade-off between accuracy and output length to produce a robust Pareto front of merged models. To make this search computationally tractable for large language models, we propose an entropy-based subset sampling technique that drastically reduces the overhead of fitness estimation. Comprehensive experiments across 1.5B, 7B, and 14B parameter scales on six mathematical reasoning benchmarks demonstrate that Evo-L2S can reduce the length of generated reasoning traces by over 50% while preserving, or even improving, the problem-solving accuracy of the original reasoning models. 

---
# Context-Aware Dialectal Arabic Machine Translation with Interactive Region and Register Selection 

**Authors**: Afroza Nowshin, Prithweeraj Acharjee Porag, Haziq Jeelani, Fayeq Jeelani Syed  

**Link**: [PDF](https://arxiv.org/pdf/2604.06456)  

**Abstract**: Current Machine Translation (MT) systems for Arabic often struggle to account for dialectal diversity, frequently homogenizing dialectal inputs into Modern Standard Arabic (MSA) and offering limited user control over the target vernacular. In this work, we propose a context-aware and steerable framework for dialectal Arabic MT that explicitly models regional and sociolinguistic variation. Our primary technical contribution is a Rule-Based Data Augmentation (RBDA) pipeline that expands a 3,000-sentence seed corpus into a balanced 57,000-sentence parallel dataset, covering eight regional varieties eg., Egyptian, Levantine, Gulf, etc. By fine-tuning an mT5-base model conditioned on lightweight metadata tags, our approach enables controllable generation across dialects and social registers in the translation output.
Through a combination of automatic evaluation and qualitative analysis, we observe an apparent accuracy-fidelity trade-off: high-resource baselines such as NLLB (No Language Left Behind) achieve higher aggregate BLEU scores (13.75) by defaulting toward the MSA mean, while exhibiting limited dialectal specificity. In contrast, our model achieves lower BLEU scores (8.19) but produces outputs that align more closely with the intended regional varieties. Supporting qualitative evaluation, including an LLM-assisted cultural authenticity analysis, suggests improved dialectal alignment compared to baseline systems (4.80/5 vs. 1.0/5). These findings highlight the limitations of standard MT metrics for dialect-sensitive tasks and motivate the need for evaluation practices that better reflect linguistic diversity in Arabic MT. 

---
# Learning to Interrupt in Language-based Multi-agent Communication 

**Authors**: Danqing Wang, Da Yin, Ruta Desai, Lei Li, Asli Celikyilmaz, Ansong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2604.06452)  

**Abstract**: Multi-agent systems using large language models (LLMs) have demonstrated impressive capabilities across various domains. However, current agent communication suffers from verbose output that overload context and increase computational costs. Although existing approaches focus on compressing the message from the speaker side, they struggle to adapt to different listeners and identify relevant information. An effective way in human communication is to allow the listener to interrupt and express their opinion or ask for clarification. Motivated by this, we propose an interruptible communication framework that allows the agent who is listening to interrupt the current speaker. Through prompting experiments, we find that current LLMs are often overconfident and interrupt before receiving enough information. Therefore, we propose a learning method that predicts the appropriate interruption points based on the estimated future reward and cost. We evaluate our framework across various multi-agent scenarios, including 2-agent text pictionary games, 3-agent meeting scheduling, and 3-agent debate. The results of the experiment show that our HANDRAISER can reduce the communication cost by 32.2% compared to the baseline with comparable or superior task performance. This learned interruption behavior can also be generalized to different agents and tasks. 

---
# Team Fusion@ SU@ BC8 SympTEMIST track: transformer-based approach for symptom recognition and linking 

**Authors**: Georgi Grazhdanski, Sylvia Vassileva, Ivan Koychev, Svetla Boytcheva  

**Link**: [PDF](https://arxiv.org/pdf/2604.06424)  

**Abstract**: This paper presents a transformer-based approach to solving the SympTEMIST named entity recognition (NER) and entity linking (EL) tasks. For NER, we fine-tune a RoBERTa-based (1) token-level classifier with BiLSTM and CRF layers on an augmented train set. Entity linking is performed by generating candidates using the cross-lingual SapBERT XLMR-Large (2), and calculating cosine similarity against a knowledge base. The choice of knowledge base proves to have the highest impact on model accuracy. 

---
# When to Call an Apple Red: Humans Follow Introspective Rules, VLMs Don't 

**Authors**: Jonathan Nemitz, Carsten Eickhoff, Junyi Jessy Li, Kyle Mahowald, Michal Golovanevsky, William Rudman  

**Link**: [PDF](https://arxiv.org/pdf/2604.06422)  

**Abstract**: Understanding when Vision-Language Models (VLMs) will behave unexpectedly, whether models can reliably predict their own behavior, and if models adhere to their introspective reasoning are central challenges for trustworthy deployment. To study this, we introduce the Graded Color Attribution (GCA) dataset, a controlled benchmark designed to elicit decision rules and evaluate participant faithfulness to these rules. GCA consists of line drawings that vary pixel-level color coverage across three conditions: world-knowledge recolorings, counterfactual recolorings, and shapes with no color priors. Using GCA, both VLMs and human participants establish a threshold: the minimum percentage of pixels of a given color an object must have to receive that color label. We then compare these rules with their subsequent color attribution decisions. Our findings reveal that models systematically violate their own introspective rules. For example, GPT-5-mini violates its stated introspection rules in nearly 60\% of cases on objects with strong color priors. Human participants remain faithful to their stated rules, with any apparent violations being explained by a well-documented tendency to overestimate color coverage. In contrast, we find that VLMs are excellent estimators of color coverage, yet blatantly contradict their own reasoning in their final responses. Across all models and strategies for eliciting introspective rules, world-knowledge priors systematically degrade faithfulness in ways that do not mirror human cognition. Our findings challenge the view that VLM reasoning failures are difficulty-driven and suggest that VLM introspective self-knowledge is miscalibrated, with direct implications for high-stakes deployment. 

---
# State-of-the-Art Arabic Language Modeling with Sparse MoE Fine-Tuning and Chain-of-Thought Distillation 

**Authors**: Navan Preet Singh, Anurag Garikipati, Ahmed Abulkhair, Jyani Akshay Jagdishbhai, Atul Yaduvanshi, Amarendra Chaudhary, Madalina Ciobanu, Qingqing Mao, Ritankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.06421)  

**Abstract**: This paper introduces Arabic-DeepSeek-R1, an application-driven open-source Arabic LLM that leverages a sparse MoE backbone to address the digital equity gap for under-represented languages, and establishes a new SOTA across the entire Open Arabic LLM Leaderboard (OALL). Our four-phase CoT distillation scheme integrates Arabic-specific linguistic verification and regional ethical norms into a 372M-token, contamination-controlled 80/20 Arabic-English training mixture. Arabic-DeepSeek-R1 achieves the highest average score across the seven-benchmark OALL suite while establishing SOTA or near-SOTA, including dominant results on grammar-focused MadinahQA (surpassing both GPT-5.1 and the OALL leader by substantial margins), safety-oriented AraTrust, multi-ability AlGhafa, and retrieval-augmented ALRAGE. Our results indicate that the combination of sparse MoE architecture, culturally-informed CoT distillation with explicit Arabic linguistic checks, and strategic bilingual data curation enables an open-source adapted model to systematically outperform the proprietary frontier system GPT-5.1 on the majority of benchmarks evaluating comprehensive language-specific tasks: the first such demonstration for Arabic LLMs. These findings indicate that much of Arabic's performance deficit in current LLM ecosystems stems from under-specialization rather than architectural limitations, and that parameter-efficient adaptation of open reasoning models can yield breakthrough SOTA performance without industrial-scale pretraining costs. Arabic-DeepSeek-R1 establishes a validated and replicable framework for sovereign and domain-specific language technologies, demonstrating that strategic, culturally-grounded adaptation of sparse MoE backbones offers a viable and cost-effective pathway to achieving record-breaking performance across standardized benchmarks for low-resource languages. 

---
# Attention Flows: Tracing LLM Conceptual Engagement via Story Summaries 

**Authors**: Rebecca M. M. Hicke, Sil Hamilton, David Mimno, Ross Deans Kristensen-McLachlan  

**Link**: [PDF](https://arxiv.org/pdf/2604.06416)  

**Abstract**: Although LLM context lengths have grown, there is evidence that their ability to integrate information across long-form texts has not kept pace. We evaluate one such understanding task: generating summaries of novels. When human authors of summaries compress a story, they reveal what they consider narratively important. Therefore, by comparing human and LLM-authored summaries, we can assess whether models mirror human patterns of conceptual engagement with texts. To measure conceptual engagement, we align sentences from 150 human-written novel summaries with the specific chapters they reference. We demonstrate the difficulty of this alignment task, which indicates the complexity of summarization as a task. We then generate and align additional summaries by nine state-of-the-art LLMs for each of the 150 reference texts. Comparing the human and model-authored summaries, we find both stylistic differences between the texts and differences in how humans and LLMs distribute their focus throughout a narrative, with models emphasizing the ends of texts. Comparing human narrative engagement with model attention mechanisms suggests explanations for degraded narrative comprehension and targets for future development. We release our dataset to support future research. 

---
# FMI@SU ToxHabits: Evaluating LLMs Performance on Toxic Habit Extraction in Spanish Clinical Texts 

**Authors**: Sylvia Vassileva, Ivan Koychev, Svetla Boytcheva  

**Link**: [PDF](https://arxiv.org/pdf/2604.06403)  

**Abstract**: The paper presents an approach for the recognition of toxic habits named entities in Spanish clinical texts. The approach was developed for the ToxHabits Shared Task. Our team participated in subtask 1, which aims to detect substance use and abuse mentions in clinical case reports and classify them in four categories (Tobacco, Alcohol, Cannabis, and Drug). We explored various methods of utilizing LLMs for the task, including zero-shot, few-shot, and prompt optimization, and found that GPT-4.1's few-shot prompting performed the best in our experiments. Our method achieved an F1 score of 0.65 on the test set, demonstrating a promising result for recognizing named entities in languages other than English. 

---
# ART: Attention Replacement Technique to Improve Factuality in LLMs 

**Authors**: Ziqin Luo, Yihao Quan, Xiaofeng Zhang, Xiaosong Yuan, Chen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.06393)  

**Abstract**: Hallucination in large language models (LLMs) continues to be a significant issue, particularly in tasks like question answering, where models often generate plausible yet incorrect or irrelevant information. Although various methods have been proposed to mitigate hallucinations, the relationship between attention patterns and hallucinations has not been fully explored. In this paper, we analyze the distribution of attention scores across each layer and attention head of LLMs, revealing a common and intriguing phenomenon: shallow layers of LLMs primarily rely on uniform attention patterns, where the model distributes its attention evenly across the entire sequence. This uniform attention pattern can lead to hallucinations, as the model fails to focus on the most relevant information. To mitigate this issue, we propose a training-free method called Attention Replacement Technique (ART), which replaces these uniform attention patterns in the shallow layers with local attention patterns. This change directs the model to focus more on the relevant contexts, thus reducing hallucinations. Through extensive experiments, ART demonstrates significant reductions in hallucinations across multiple LLM architectures, proving its effectiveness and generalizability without requiring fine-tuning or additional training data. 

---
# Application-Driven Pedagogical Knowledge Optimization of Open-Source LLMs via Reinforcement Learning and Supervised Fine-Tuning 

**Authors**: Navan Preet Singh, Xiaokun Wang, Anurag Garikipati, Madalina Ciobanu, Qingqing Mao, Ritankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.06385)  

**Abstract**: We present an innovative multi-stage optimization strategy combining reinforcement learning (RL) and supervised fine-tuning (SFT) to enhance the pedagogical knowledge of large language models (LLMs), as illustrated by EduQwen 32B-RL1, EduQwen 32B-SFT, and an optional third-stage model EduQwen 32B-SFT-RL2: (1) RL optimization that implements progressive difficulty training, focuses on challenging examples, and employs extended reasoning rollouts; (2) a subsequent SFT phase that leverages the RL-trained model to synthesize high-quality training data with difficulty-weighted sampling; and (3) an optional second round of RL optimization. EduQwen 32B-RL1, EduQwen 32B-SFT, and EduQwen 32B-SFT-RL2 are an application-driven family of open-source pedagogical LLMs built on a dense Qwen3-32B backbone. These models remarkably achieve high enough accuracy on the Cross-Domain Pedagogical Knowledge (CDPK) Benchmark to establish new state-of-the-art (SOTA) results across the interactive Pedagogy Benchmark Leaderboard and surpass significantly larger proprietary systems such as the previous benchmark leader Gemini-3 Pro. These dense 32-billion-parameter models demonstrate that domain-specialized optimization can transform mid-sized open-source LLMs into true pedagogical domain experts that outperform much larger general-purpose systems, while preserving the transparency, customizability, and cost-efficiency required for responsible educational AI deployment. 

---
# The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models 

**Authors**: Michael Rizvi-Martel, Guillaume Rabusseau, Marius Mosbach  

**Link**: [PDF](https://arxiv.org/pdf/2604.06374)  

**Abstract**: Latent reasoning via continuous chain-of-thoughts (Latent CoT) has emerged as a promising alternative to discrete CoT reasoning. Operating in continuous space increases expressivity and has been hypothesized to enable superposition: the ability to maintain multiple candidate solutions simultaneously within a single representation. Despite theoretical arguments, it remains unclear whether language models actually leverage superposition when reasoning using latent CoTs. We investigate this question across three regimes: a training-free regime that constructs latent thoughts as convex combinations of token embeddings, a fine-tuned regime where a base model is adapted to produce latent thoughts, and a from-scratch regime where a model is trained entirely with latent thoughts to solve a given task. Using Logit Lens and entity-level probing to analyze internal representations, we find that only models trained from scratch exhibit signs of using superposition. In the training-free and fine-tuned regimes, we find that the superposition either collapses or is not used at all, with models discovering shortcut solutions instead. We argue that this is due to two complementary phenomena: i) pretraining on natural language data biases models to commit to a token in the last layers ii) capacity has a huge effect on which solutions a model favors. Together, our results offer a unified explanation for when and why superposition arises in continuous chain-of-thought reasoning, and identify the conditions under which it collapses. 

---
# A Severity-Based Curriculum Learning Strategy for Arabic Medical Text Generation 

**Authors**: Ahmed Alansary, Molham Mohamed, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2604.06365)  

**Abstract**: Arabic medical text generation is increasingly needed to help users interpret symptoms and access general health guidance in their native language. Nevertheless, many existing methods assume uniform importance across training samples, overlooking differences in clinical severity. This simplification can hinder the model's ability to properly capture complex or high-risk cases. To overcome this issue, this work introduces a Severity-based Curriculum Learning Strategy for Arabic Medical Text Generation, where the training process is structured to move gradually from less severe to more critical medical conditions. The approach divides the dataset into ordered stages based on severity and incrementally exposes the model to more challenging cases during fine-tuning, allowing it to first learn basic medical patterns before addressing more complex scenarios. The proposed method is evaluated on a subset of the Medical Arabic Question Answering (MAQA) dataset, which includes Arabic medical questions describing symptoms alongside corresponding responses. In addition, the dataset is annotated with three severity levels (Mild, Moderate, and Critical) using a rule-based method developed in this study. The results demonstrate that incorporating severity-aware curriculum learning leads to consistent performance improvements across all tested models, with gains of around +4% to +7% over baseline models and +3% to +6% compared with conventional fine-tuning approaches. 

---
# In-Context Learning in Speech Language Models: Analyzing the Role of Acoustic Features, Linguistic Structure, and Induction Heads 

**Authors**: Charlotte Pouw, Hosein Mohebbi, Afra Alishahi, Willem Zuidema  

**Link**: [PDF](https://arxiv.org/pdf/2604.06356)  

**Abstract**: In-Context Learning (ICL) has been extensively studied in text-only Language Models, but remains largely unexplored in the speech domain. Here, we investigate how linguistic and acoustic features affect ICL in Speech Language Models. We focus on the Text-to-Speech (TTS) task, which allows us to analyze ICL from two angles: (1) how accurately the model infers the task from the demonstrations (i.e., generating the correct spoken content), and (2) to what extent the model mimics the acoustic characteristics of the demonstration speech in its output. We find that speaking rate strongly affects ICL performance and is also mimicked in the output, whereas pitch range and intensity have little impact on performance and are not consistently reproduced. Finally, we investigate the role of induction heads in speech-based ICL and show that these heads play a causal role: ablating the top-k induction heads completely removes the model's ICL ability, mirroring findings from text-based ICL. 

---
# Severity-Aware Weighted Loss for Arabic Medical Text Generation 

**Authors**: Ahmed Alansary, Molham Mohamed, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2604.06346)  

**Abstract**: Large language models have shown strong potential for Arabic medical text generation; however, traditional fine-tuning objectives treat all medical cases uniformly, ignoring differences in clinical severity. This limitation is particularly critical in healthcare settings, where errors in severe cases contain higher clinical risk. In this work, we propose a severity-aware weighted loss for fine-tuning Arabic language models on medical complaint-response data. The method depends on soft severity probabilities to dynamically scale token-level loss contributions during optimization, thereby prioritizing clinically critical interactions without modifying model architectures. Experiments are conducted using the MAQA dataset, which provides Arabic medical complaints and trusted human responses. Severity labels and probabilistic scores are automatically derived using a fine-tuned AraBERT-based classifier and incorporated exclusively at the loss level. The proposed approach is evaluated across ten Arabic large language models of varying architectures and parameter scales. While standard cross-entropy fine-tuning yields only modest improvements, severity-aware optimization consistently achieves larger gains. Using a balanced weighting configuration, performance improves from 54.04% to 66.14% for AraGPT2-Base, from 59.16% to 67.18% for AraGPT2-Medium, and from 57.83% to 66.86% for Qwen2.5-0.5B, with peak performance reaching 67.18%. Overall, severity-aware fine-tuning delivers improvements of up to 12.10% over non-fine-tuned baselines, demonstrating robust and architecture-consistent gains. 

---
# STDec: Spatio-Temporal Stability Guided Decoding for dLLMs 

**Authors**: Yuzhe Chen, Jiale Cao, Xuyang Liu, Jin Xie, Aiping Yang, Yanwei Pang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06330)  

**Abstract**: Diffusion Large Language Models (dLLMs) have achieved rapid progress, viewed as a promising alternative to the autoregressive paradigm. However, most dLLM decoders still adopt a global confidence threshold, and do not explicitly model local context from neighboring decoded states or temporal consistency of predicted token IDs across steps. To address this issue, we propose a simple spatio-temporal stability guided decoding approach, named STDec. We observe strong spatio-temporal stability in dLLM decoding: newly decoded tokens tend to lie near decoded neighbors, and their predicted IDs often remain consistent across several denoising steps. Inspired by this stability, our STDec includes spatial-aware decoding and temporal-aware decoding. The spatial-aware decoding dynamically generates the token-adaptive threshold by aggregating the decoded states of nearby tokens. The temporal-aware decoding relaxes the decoding thresholds for tokens whose predicted token IDs remain consistent over denoising steps. Our STDec is training-free and remains compatible with cache-based acceleration methods. Across textual reasoning and multimodal understanding benchmarks, STDec substantially improves throughput while maintaining comparable task performance score. Notably, on MBPP with LLaDA, STDec achieves up to 14.17x speedup with a comparable score. Homepage: this https URL. 

---
# Blending Human and LLM Expertise to Detect Hallucinations and Omissions in Mental Health Chatbot Responses 

**Authors**: Khizar Hussain, Bradley A. Malin, Zhijun Yin, Susannah Leigh Rose, Murat Kantarcioglu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06216)  

**Abstract**: As LLM-powered chatbots are increasingly deployed in mental health services, detecting hallucinations and omissions has become critical for user safety. However, state-of-the-art LLM-as-a-judge methods often fail in high-risk healthcare
contexts, where subtle errors can have serious consequences. We show that leading LLM judges achieve only 52% accuracy on mental health counseling data, with some hallucination detection approaches exhibiting near-zero recall. We identify the root cause
as LLMs' inability to capture nuanced linguistic and therapeutic patterns recognized by domain experts. To address this, we propose a framework that integrates human expertise with LLMs to extract interpretable, domain-informed features across five
analytical dimensions: logical consistency, entity verification, factual accuracy, linguistic uncertainty, and professional appropriateness. Experiments on a public mental health dataset and a new human-annotated dataset show that traditional machine
learning models trained on these features achieve 0.717 F1 on our custom dataset and 0.849 F1 on a public benchmark for hallucination detection, with 0.59-0.64 F1 for omission detection across both datasets. Our results demonstrate that combining domain
expertise with automated methods yields more reliable and transparent evaluation than black-box LLM judging in high-stakes mental health applications. 

---
# Unsupervised Neural Network for Automated Classification of Surgical Urgency Levels in Medical Transcriptions 

**Authors**: Sadaf Tabatabaee, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2604.06214)  

**Abstract**: Efficient classification of surgical procedures by urgency is paramount to optimize patient care and resource allocation within healthcare systems. This study introduces an unsupervised neural network approach to automatically categorize surgical transcriptions into three urgency levels: immediate, urgent, and elective. Leveraging BioClinicalBERT, a domain-specific language model, surgical transcripts are transformed into high-dimensional embeddings that capture their semantic nuances. These embeddings are subsequently clustered using both K-means and Deep Embedding Clustering (DEC) algorithms, in which DEC demonstrates superior performance in the formation of cohesive and well-separated clusters. To ensure clinical relevance and accuracy, the clustering results undergo validation through the Modified Delphi Method, which involves expert review and refinement. Following validation, a neural network that integrates Bidirectional Long Short-Term Memory (BiLSTM) layers with BioClinicalBERT embeddings is developed for classification tasks. The model is rigorously evaluated using cross-validation and metrics such as accuracy, precision, recall, and F1-score, which achieve robust performance and demonstrate strong generalization capabilities on unseen data. This unsupervised framework not only addresses the challenge of limited labeled data but also provides a scalable and reliable solution for real-time surgical prioritization, which ultimately enhances operational efficiency and patient outcomes in dynamic medical environments. 

---
# Invisible Influences: Investigating Implicit Intersectional Biases through Persona Engineering in Large Language Models 

**Authors**: Nandini Arimanda, Achyuth Mukund, Sakthi Balan Muthiah, Rajesh Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2604.06213)  

**Abstract**: Large Language Models (LLMs) excel at human-like language generation but often embed and amplify implicit, intersectional biases, especially under persona-driven contexts. Existing bias audits rely on static, embedding-based tests (CEAT, I-WEAT, I-SEAT) that quantify absolute association strengths. We show that they have limitations in capturing dynamic shifts when models adopt social roles. We address this gap by introducing the Bias Amplification Differential and Explainability Score (BADx): a novel, scalable metric that measures persona-induced bias amplification and integrates local explainability insights. BADx comprises three components - differential bias scores (BAD, based on CEAT, I-WEAT, I-SEAT),Persona Sensitivity Index (PSI), and Volatility (Standard Deviation), augmented by LIME-based analysis for emphasizing explainability. This study is divided and performed as two different tasks. Task 1 establishes static bias baselines, and Task 2 applies six persona frames (marginalized and structurally advantaged) to measure BADx, PSI, and volatility. This is studied across five state-of-the-art LLMs (GPT-4o, DeepSeek-R1, LLaMA-4, Claude 4.0 Sonnet and Gemma-3n E4B). Results show persona context significantly modulates bias. GPT-4o exhibits high sensitivity and volatility; DeepSeek-R1 suppresses bias but with erratic volatility; LLaMA-4 maintains low volatility and a stable bias profile with limited amplification; Claude 4.0 Sonnet achieves balanced modulation; and Gemma-3n E4B attains the lowest volatility with moderate amplification. BADx performs better than static methods by revealing context-sensitive biases overlooked in static methods. Our unified method offers a systematic way to detect dynamic implicit intersectional bias in five popular LLMs. 

---
# Illocutionary Explanation Planning for Source-Faithful Explanations in Retrieval-Augmented Language Models 

**Authors**: Francesco Sovrano, Alberto Bacchelli  

**Link**: [PDF](https://arxiv.org/pdf/2604.06211)  

**Abstract**: Natural language explanations produced by large language models (LLMs) are often persuasive, but not necessarily scrutable: users cannot easily verify whether the claims in an explanation are supported by evidence. In XAI, this motivates a focus on faithfulness and traceability, i.e., the extent to which an explanation's claims can be grounded in, and traced back to, an explicit source. We study these desiderata in retrieval-augmented generation (RAG) for programming education, where textbooks provide authoritative evidence. We benchmark six LLMs on 90 Stack Overflow questions grounded in three programming textbooks and quantify source faithfulness via source adherence metrics. We find that non Retrieval-Augmented Generation (RAG) models have median source adherence of 0%, while baseline RAG systems still exhibit low median adherence (22-40%, depending on the model). Motivated by Achinstein's illocutionary theory of explanation, we introduce illocutionary macro-planning as a descriptive design principle for source-faithful explanations and instantiate it with chain-of-illocution prompting (CoI), which expands a query into implicit explanatory questions that drive retrieval. Across models, CoI yields statistically significant gains (up to 63%) in source adherence, although absolute adherence remains moderate and the gains are weak or non-significant for some models. A user study with 165 retained participants (220 recruited) indicates that these gains do not harm satisfaction, relevance, or perceived correctness. 

---
# Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook 

**Authors**: Jaehyeok Lee, Xiaoyuan Yi, Jing Yao, Hyunjin Hwang, Roy Ka-Wei Lee, Xing Xie, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2604.06210)  

**Abstract**: As LLMs are globally deployed, aligning their cultural value orientations is critical for safety and user engagement. However, existing benchmarks face the Construct-Composition-Context ($C^3$) challenge: relying on discriminative, multiple-choice formats that probe value knowledge rather than true orientations, overlook subcultural heterogeneity, and mismatch with real-world open-ended generation. We introduce DOVE, a distributional evaluation framework that directly compares human-written text distributions with LLM-generated outputs. DOVE utilizes a rate-distortion variational optimization objective to construct a compact value-codebook from 10K documents, mapping text into a structured value space to filter semantic noise. Alignment is measured using unbalanced optimal transport, capturing intra-cultural distributional structures and sub-group diversity. Experiments across 12 LLMs show that DOVE achieves superior predictive validity, attaining a 31.56% correlation with downstream tasks, while maintaining high reliability with as few as 500 samples per culture. 

---
# TelcoAgent-Bench: A Multilingual Benchmark for Telecom AI Agents 

**Authors**: Lina Bariah, Brahim Mefgouda, Farbod Tavakkoli, Enrique Molero, Louis Powell, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2604.06209)  

**Abstract**: The integration of large language model (LLM) agents into telecom networks introduces new challenges, related to intent recognition, tool execution, and resolution generation, while taking into consideration different operational constraints. In this paper, we introduce TelcoAgent-Bench and TelcoAgent-Metrics, a Telecom-specific benchmarking framework for evaluating multilingual telecom LLM agents. The proposed framework assesses the semantic understanding as well as process-level alignment with structured troubleshooting flows and stability across repeated scenario variations. Our contribution includes a structured suite of metrics that assess intent recognition, ordered tool execution, resolution correctness, and stability across scenario variations, with the aim of quantifying the reliability and operational consistency of LLM agents in telecom environments. The framework is designed to operate in both English and Arabic, to address the need for multilingual agent deployment in operational network environments. Our experimental results show that although recent instruct-tuned models can understand telecom problems in a reasonable way, they usually struggle to consistently follow the required troubleshooting steps and to maintain stable behavior when exposed to different variations of the same scenario. This performance gap becomes more pronounced in unconstrained and bilingual settings. 

---
# Extracting Breast Cancer Phenotypes from Clinical Notes: Comparing LLMs with Classical Ontology Methods 

**Authors**: Abdullah Bin Faiz, Arbaz Khan Shehzad, Asad Afzal, Momin Tariq, Muhammad Siddiqi, Muhammad Usamah Shahid, Maryam Noor Awan, Muddassar Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2604.06208)  

**Abstract**: A significant amount of data held in Oncology Electronic Medical Records (EMRs) is contained in unstructured provider notes -- including but not limited to the chemotherapy (or cancer treatment) outcome, different biomarkers, the tumor's location, sizes, and growth patterns of a patient. The clinical studies show that the majority of oncologists are comfortable providing these valuable insights in their notes in a natural language rather than the relevant structured fields of an EMR. The major contribution of this research is to report an LLM-based framework to process provider notes and extract valuable medical knowledge and phenotype mentioned above, with a focus on the domain of oncology. In this paper, we focus on extracting phenotypes related to breast cancer using our LLM framework, and then compare its performance with earlier works that used knowledge-driven annotation system, paired with the NCIt Ontology Annotator. The results of the study show that an LLM-based information extraction framework can be easily adapted to extract phenotypes with an accuracy that is comparable to the classical ontology-based methods. However, once trained, they could be easily fine-tuned to cater for other cancer types and diseases. 

---
# A Comparative Study of Demonstration Selection for Practical Large Language Models-based Next POI Prediction 

**Authors**: Ryo Nishida, Masayuki Kawarada, Tatsuya Ishigaki, Hiroya Takamura, Masaki Onishi  

**Link**: [PDF](https://arxiv.org/pdf/2604.06207)  

**Abstract**: This paper investigates demonstration selection strategies for predicting a user's next point-of-interest (POI) using large language models (LLMs), aiming to accurately forecast a user's subsequent location based on historical check-in data. While in-context learning (ICL) with LLMs has recently gained attention as a promising alternative to traditional supervised approaches, the effectiveness of ICL significantly depends on the selected demonstration. Although previous studies have examined methods such as random selection, embedding-based selection, and task-specific selection, there remains a lack of comprehensive comparative analysis among these strategies. To bridge this gap and clarify the best practices for real-world applications, we comprehensively evaluate existing demonstration selection methods alongside simpler heuristic approaches such as geographical proximity, temporal ordering, and sequential patterns. Extensive experiments conducted on three real-world datasets indicate that these heuristic methods consistently outperform more complex and computationally demanding embedding-based methods, both in terms of computational cost and prediction accuracy. Notably, in certain scenarios, LLMs using demonstrations selected by these simpler heuristic methods even outperform existing fine-tuned models, without requiring further training. Our source code is available at: this https URL. 

---
# Tool-MCoT: Tool Augmented Multimodal Chain-of-Thought for Content Safety Moderation 

**Authors**: Shutong Zhang, Dylan Zhou, Yinxiao Liu, Yang Yang, Huiwen Luo, Wenfei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.06205)  

**Abstract**: The growth of online platforms and user content requires strong content moderation systems that can handle complex inputs from various media types. While large language models (LLMs) are effective, their high computational cost and latency present significant challenges for scalable deployment. To address this, we introduce Tool-MCoT, a small language model (SLM) fine-tuned for content safety moderation leveraging external framework. By training our model on tool-augmented chain-of-thought data generated by LLM, we demonstrate that the SLM can learn to effectively utilize these tools to improve its reasoning and decision-making. Our experiments show that the fine-tuned SLM achieves significant performance gains. Furthermore, we show that the model can learn to use these tools selectively, achieving a balance between moderation accuracy and inference efficiency by calling tools only when necessary. 

---
# SensorPersona: An LLM-Empowered System for Continual Persona Extraction from Longitudinal Mobile Sensor Streams 

**Authors**: Bufang Yang, Lilin Xu, Yixuan Li, Kaiwei Liu, Xiaofan Jiang, Zhenyu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.06204)  

**Abstract**: Personalization is essential for Large Language Model (LLM)-based agents to adapt to users' preferences and improve response quality and task performance. However, most existing approaches infer personas from chat histories, which capture only self-disclosed information rather than users' everyday behaviors in the physical world, limiting the ability to infer comprehensive user personas. In this work, we introduce SensorPersona, an LLM-empowered system that continuously infers stable user personas from multimodal longitudinal sensor streams unobtrusively collected from users' mobile devices. SensorPersona first performs person-oriented context encoding on continuous sensor streams to enrich the semantics of sensor contexts. It then employs hierarchical persona reasoning that integrates intra- and inter-episode reasoning to infer personas spanning physical patterns, psychosocial traits, and life experiences. Finally, it employs clustering-aware incremental verification and temporal evidence-aware updating to adapt to evolving personas. We evaluate SensorPersona on a self-collected dataset containing 1,580 hours of sensor data from 20 participants, collected over up to 3 months across 17 cities on 3 continents. Results show that SensorPersona achieves up to 31.4% higher recall in persona extraction, an 85.7% win rate in persona-aware agent responses, and notable improvements in user satisfaction compared to state-of-the-art baselines. 

---
# Cross-Lingual Transfer and Parameter-Efficient Adaptation in the Turkic Language Family: A Theoretical Framework for Low-Resource Language Models 

**Authors**: O. Ibrahimzade, K. Tabasaransky  

**Link**: [PDF](https://arxiv.org/pdf/2604.06202)  

**Abstract**: Large language models (LLMs) have transformed natural language processing, yet their capabilities remain uneven across languages. Most multilingual models are trained primarily on high-resource languages, leaving many languages with large speaker populations underrepresented in both training data and evaluation benchmarks. This imbalance is particularly visible in the Turkic language family. This paper proposes a theoretical framework for studying cross-lingual transfer and parameter-efficient adaptation of multilingual LLMs within the Turkic language family, focusing on Azerbaijani, Kazakh, Uzbek, Turkmen, and Gagauz. These languages share substantial typological and morphological similarity while differing greatly in available digital resources, making them a natural setting for analyzing multilingual adaptation strategies. We integrate insights from multilingual representation learning and parameter-efficient fine-tuning techniques such as Low-Rank Adaptation (LoRA) to develop a conceptual scaling model describing how adaptation performance depends on model capacity, adaptation data size, and the expressivity of adaptation modules. To formalize transfer potential between related languages, we introduce the Turkic Transfer Coefficient (TTC), a theoretical measure incorporating morphological similarity, lexical overlap, syntactic structure, and script compatibility across Turkic languages. The framework highlights how typological similarity can enable efficient multilingual transfer while also identifying structural limits of parameter-efficient adaptation in extremely low-resource scenarios. 

---
# Beyond Facts: Benchmarking Distributional Reading Comprehension in Large Language Models 

**Authors**: Pei-Fu Guo, Ya-An Tsai, Chun-Chia Hsu, Kai-Xin Chen, Yun-Da Tsai, Kai-Wei Chang, Nanyun Peng, Mi-Yen Yeh, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.06201)  

**Abstract**: While most reading comprehension benchmarks for LLMs focus on factual information that can be answered by localizing specific textual evidence, many real-world tasks require understanding distributional information, such as population-level trends and preferences expressed across collections of text. We introduce Text2DistBench, a reading comprehension benchmark for evaluating LLMs' ability to infer distributional knowledge from natural language. Built from real-world YouTube comments about movie and music entities, the benchmark provides models with entity metadata and associated comments, and requires them to answer distributional questions, such as estimating the proportions of positive and negative comments, or identifying the most and second most frequent topics discussed among viewers. To support reliable and long-term evaluation, the construction pipeline of Text2DistBench is fully automated and continuously updated to incorporate newly emerging entities over time. Experiments across multiple LLMs show that while models substantially outperform random baselines, performance varies widely across different distribution types and characteristics. These findings highlight both the capabilities and limitations of current LLMs in distributional reading comprehension and demonstrate the value of Text2DistBench as a practical and scalable testbed for future research. 

---
# Emergent decentralized regulation in a purely synthetic society 

**Authors**: Md Motaleb Hossen Manik, Ge Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06199)  

**Abstract**: As autonomous AI agents increasingly inhabit online environments and extensively interact, a key question is whether synthetic collectives exhibit self-regulated social dynamics with neither human intervention nor centralized design. We study OpenClaw agents on Moltbook, an agent-only social network, using an observational archive of 39,026 posts and 5,712 comments authored by 14,490 agents. We quantify action-inducing language with Directive Intensity (DI), a transparent, lexicon-based proxy for directive and instructional phrasing that does not measure moral valence, intent, or execution outcomes. We classify responsive comments into four types: Affirmation, Corrective Signaling, Adverse Reaction, and Neutral Interaction. Directive content is common (DI>0 in 18.4% of posts). More importantly, corrective signaling scales with DI: posts with higher DI exhibit higher corrective reply probability, visible in stable binned estimates with Wilson confidence intervals. To address comment nesting within posts, we fit a post-level random intercept mixed-effects logistic model and find that the positive DI association persists. Event-aligned within-thread analysis of comment text provides additional evidence consistent with negative feedback after the first corrective response. In general, these results suggest that a purely synthetic, agent-only society can exhibit endogenous corrective signaling with a strength positively linked to the intensity of directive proposals. 

---
# Temporally Phenotyping GLP-1RA Case Reports with Large Language Models: A Textual Time Series Corpus and Risk Modeling 

**Authors**: Sayantan Kumar, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2604.06197)  

**Abstract**: Type 2 diabetes case reports describe complex clinical courses, but their timelines are often expressed in language that is difficult to reuse in longitudinal modeling. To address this gap, we developed a textual time-series corpus of 136 PubMed Open Access single-patient case reports involving glucagon-like peptide 1 receptor agonists, with clinical events associated with their most probable reference times. We evaluated automated LLM timeline extraction against gold-standard timelines annotated by clinical domain experts, assessing how well systems recovered clinical events and their timings. The best-performing LLM produced high event coverage (GPT5; 0.871) and reliable temporal sequencing across symptoms (GPT5; 0.843), diagnoses, treatments, laboratory tests, and outcomes. As a downstream demonstration, time-to-event analyses in diabetes suggested lower risk of respiratory sequelae among GLP-1 users versus non-users (HR=0.259, p<0.05), consistent with prior reports of improved respiratory outcomes. Temporal annotations and code will be released upon acceptance. 

---
# Consistency-Guided Decoding with Proof-Driven Disambiguation for Three-Way Logical Question Answering 

**Authors**: Tianyi Huang, Ming Hou, Jiaheng Su, Yutong Zhang, Ziling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06196)  

**Abstract**: Three-way logical question answering (QA) assigns $True/False/Unknown$ to a hypothesis $H$ given a premise set $S$. While modern large language models (LLMs) can be accurate on isolated examples, we identify two recurring failure modes in 3-way logic QA: (i) negation inconsistency, where answers to $H$ and $\neg H$ violate the deterministic label mapping, and (ii) epistemic $Unknown$, where the model predicts $Unknown$ due to uncertainty or instability even when $S$ entails one side. We present CGD-PD, a lightweight test-time layer that (a) queries a single 3-way classifier on both $H$ and a mechanically negated form of $H$, (b) projects the pair onto a negation-consistent decision when possible, and (c) invokes a proof-driven disambiguation step that uses targeted binary entailment probes to selectively resolve $Unknown$ outcomes, requiring only an average of 4-5 model calls. On the FOLIO benchmark's first-order-logic fields, CGD-PD yields consistent gains across frontier LLMs, with relative improvements in accuracy of up to 16% over the base model, while also reducing $Unknown$ predictions. 

---
# Hallucination as output-boundary misclassification: a composite abstention architecture for language models 

**Authors**: Angelina Hintsanen  

**Link**: [PDF](https://arxiv.org/pdf/2604.06195)  

**Abstract**: Large language models often produce unsupported claims. We frame this as a misclassification error at the output boundary, where internally generated completions are emitted as if they were grounded in evidence. This motivates a composite intervention that combines instruction-based refusal with a structural abstention gate. The gate computes a support deficit score, St, from three black-box signals: self-consistency (At), paraphrase stability (Pt), and citation coverage (Ct), and blocks output when St exceeds a threshold. In a controlled evaluation across 50 items, five epistemic regimes, and three models, neither mechanism alone was sufficient. Instruction-only prompting reduced hallucination sharply, but still showed over-cautious abstention on answerable items and residual hallucination for GPT-3.5-turbo. The structural gate preserved answerable accuracy across models but missed confident confabulation on conflicting-evidence items. The composite architecture achieved high overall accuracy with low hallucination, while also inheriting some over-abstention from the instruction component. A supplementary 100-item no-context stress test derived from TruthfulQA showed that structural gating provides a capability-independent abstention floor. Overall, instruction-based refusal and structural gating show complementary failure modes, which suggests that effective hallucination control benefits from combining both mechanisms. 

---
# Depression Detection at the Point of Care: Automated Analysis of Linguistic Signals from Routine Primary Care Encounters 

**Authors**: Feng Chen, Manas Bedmutha, Janice Sabin, Andrea Hartzler, Nadir Weibel, Trevor Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2604.06193)  

**Abstract**: Depression is underdiagnosed in primary care, yet timely identification remains critical. Recorded clinical encounters, increasingly common with digital scribing technologies, present an opportunity to detect depression from naturalistic dialogue. We investigated automated depression detection from 1,108 audio-recorded primary care encounters in the Establishing Focus study, with depression defined by PHQ-9 (n=253 depressed, n=855 non-depressed). We compared three supervised approaches, Sentence-BERT + Logistic Regression (LR), LIWC+LR and ModernBERT, against a zero-shot GPT-OSS. GPT-OSS achieved the strongest performance (AUPRC=0.510, AUROC=0.774), with LIWC+LR competitive among supervised models (AUPRC=0.500, AUROC=0.742). Combined dyadic transcripts outperformed single-speaker configurations, with providers linguistically mirroring patients in depression encounters, an additive signal not captured by either speaker alone. Meaningful detection is achievable from the first 128 patient tokens (AUPRC=0.356, AUROC=0.675), supporting in-the-moment clinical decision support. These findings argue for passively collected clinical audio as a low-burden complement to existing screening workflows. 

---
# The Stepwise Informativeness Assumption: Why are Entropy Dynamics and Reasoning Correlated in LLMs? 

**Authors**: Mar Gonzàlez I Català, Haitz Sáez de Ocáriz Borde, George D. Montañez, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2604.06192)  

**Abstract**: Recent work uses entropy-based signals at multiple representation levels to study reasoning in large language models, but the field remains largely empirical. A central unresolved puzzle is why internal entropy dynamics, defined under the predictive distribution of a model, correlate so robustly with external correctness given by the ground-truth answer. In this paper, we argue that this correlation arises because autoregressive models reason correctly when they accumulate information about the true answer via answer-informative prefixes. We formalize this intuition via the Stepwise Informativeness Assumption (SIA), which states that reasoning prefixes accumulate answer-relevant information in expectation as generation progresses. We show that SIA naturally emerges from maximum-likelihood optimization on human reasoning traces and is reinforced by standard fine-tuning and reinforcement-learning pipelines. We then derive observable signatures of SIA linking conditional answer entropy dynamics to correctness. We empirically test SIA across multiple reasoning benchmarks (GSM8K, ARC, SVAMP) and a diverse set of open-weight LLMs (Gemma-2, LLaMA-3.2, Qwen-2.5, DeepSeek and Olmo variants), showing that training induces it and that correct traces exhibit characteristic conditional answer entropy patterns. 

---
# LLM-Augmented Knowledge Base Construction For Root Cause Analysis 

**Authors**: Nguyen Phuc Tran, Brigitte Jaumard, Oscar Delgado, Tristan Glatard, Karthikeyan Premkumar, Kun Ni  

**Link**: [PDF](https://arxiv.org/pdf/2604.06171)  

**Abstract**: Communications networks now form the backbone of our digital world, with fast and reliable connectivity. However, even with appropriate redundancy and failover mechanisms, it is difficult to guarantee "five 9s" (99.999 %) reliability, requiring rapid and accurate root cause analysis (RCA) during outages. In the event of an outage, rapid and accurate RCA becomes essential to restore service and prevent future disruptions.
This study evaluates three Large Language Model (LLM) methodologies - Fine-Tuning, RAG, and a Hybrid approach - for constructing a Root Cause Analysis (RCA) Knowledge Base from support tickets. We compare their performance using a comprehensive suite of lexical and semantic similarity metrics. Our experiments on a real industrial dataset demonstrate that the generated knowledge base provides an excellent starting point for accelerating RCA tasks and improving network resilience. 

---
# Appear2Meaning: A Cross-Cultural Benchmark for Structured Cultural Metadata Inference from Images 

**Authors**: Yuechen Jiang, Enze Zhang, Md Mohsinul Kabir, Qianqian Xie, Stavroula Golfomitsou, Konstantinos Arvanitis, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2604.07338)  

**Abstract**: Recent advances in vision-language models (VLMs) have improved image captioning for cultural heritage. However, inferring structured cultural metadata (e.g., creator, origin, period) from visual input remains underexplored. We introduce a multi-category, cross-cultural benchmark for this task and evaluate VLMs using an LLM-as-Judge framework that measures semantic alignment with reference annotations. To assess cultural reasoning, we report exact-match, partial-match, and attribute-level accuracy across cultural regions. Results show that models capture fragmented signals and exhibit substantial performance variation across cultures and metadata types, leading to inconsistent and weakly grounded predictions. These findings highlight the limitations of current VLMs in structured cultural metadata inference beyond visual perception. 

---
# On the Price of Privacy for Language Identification and Generation 

**Authors**: Xiaoyu Li, Andi Han, Jiaojiao Jiang, Junbin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07238)  

**Abstract**: As large language models (LLMs) are increasingly trained on sensitive user data, understanding the fundamental cost of privacy in language learning becomes essential. We initiate the study of differentially private (DP) language identification and generation in the agnostic statistical setting, establishing algorithms and matching lower bounds that precisely quantify the cost of privacy. For both tasks, approximate $(\varepsilon, \delta)$-DP with constant $\varepsilon > 0$ recovers the non-private error rates: $\exp(-r(n))$ for identification (for any $r(n) = o(n)$) and $\exp(-\Omega(n))$ for generation. Under pure $\varepsilon$-DP, the exponents degrade by a multiplicative factor of $\min\{1, \varepsilon\}$, which we show is tight up to constants. Notably, for generation under pure DP with mild assumptions, the upper bound $\exp(-\min\{1,\varepsilon\} \cdot \Omega(n))$ matches the lower bound up to some constants, establishing an optimal rate. Our results show that the cost of privacy in language learning is surprisingly mild: absent entirely under approximate DP, and exactly a $\min\{1,\varepsilon\}$ factor in the exponent under pure DP. 

---
# How Much LLM Does a Self-Revising Agent Actually Need? 

**Authors**: Seongwoo Jeong, Seonil Son  

**Link**: [PDF](https://arxiv.org/pdf/2604.07236)  

**Abstract**: Recent LLM-based agents often place world modeling, planning, and reflection inside a single language model loop. This can produce capable behavior, but it makes a basic scientific question difficult to answer: which part of the agent's competence actually comes from the LLM, and which part comes from explicit structure around it?
We study this question not by claiming a general answer, but by making it empirically tractable. We introduce a declared reflective runtime protocol that externalizes agent state, confidence signals, guarded actions, and hypothetical transitions into inspectable runtime structure. We instantiate this protocol in a declarative runtime and evaluate it on noisy Collaborative Battleship [4] using four progressively structured agents over 54 games (18 boards $\times$ 3 seeds).
The resulting decomposition isolates four components: posterior belief tracking, explicit world-model planning, symbolic in-episode reflection, and sparse LLM-based revision. Across this decomposition, explicit world-model planning improves substantially over a greedy posterior-following baseline (+24.1pp win rate, +0.017 F1). Symbolic reflection operates as a real runtime mechanism -- with prediction tracking, confidence gating, and guarded revision actions -- even though its current revision presets are not yet net-positive in aggregate. Adding conditional LLM revision at about 4.3\% of turns yields only a small and non-monotonic change: average F1 rises slightly (+0.005) while win rate drops (31$\rightarrow$29 out of 54).
These results suggest a methodological contribution rather than a leaderboard claim: externalizing reflection turns otherwise latent agent behavior into inspectable runtime structure, allowing the marginal role of LLM intervention to be studied directly. 

---
# TraceSafe: A Systematic Assessment of LLM Guardrails on Multi-Step Tool-Calling Trajectories 

**Authors**: Yen-Shan Chen, Sian-Yao Huang, Cheng-Lin Yang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.07223)  

**Abstract**: As large language models (LLMs) evolve from static chatbots into autonomous agents, the primary vulnerability surface shifts from final outputs to intermediate execution traces. While safety guardrails are well-benchmarked for natural language responses, their efficacy remains largely unexplored within multi-step tool-use trajectories. To address this gap, we introduce TraceSafe-Bench, the first comprehensive benchmark specifically designed to assess mid-trajectory safety. It encompasses 12 risk categories, ranging from security threats (e.g., prompt injection, privacy leaks) to operational failures (e.g., hallucinations, interface inconsistencies), featuring over 1,000 unique execution instances. Our evaluation of 13 LLM-as-a-guard models and 7 specialized guardrails yields three critical findings: 1) Structural Bottleneck: Guardrail efficacy is driven more by structural data competence (e.g., JSON parsing) than semantic safety alignment. Performance correlates strongly with structured-to-text benchmarks ($\rho=0.79$) but shows near-zero correlation with standard jailbreak robustness. 2) Architecture over Scale: Model architecture influences risk detection performance more significantly than model size, with general-purpose LLMs consistently outperforming specialized safety guardrails in trajectory analysis. 3) Temporal Stability: Accuracy remains resilient across extended trajectories. Increased execution steps allow models to pivot from static tool definitions to dynamic execution behaviors, actually improving risk detection performance in later stages. Our findings suggest that securing agentic workflows requires jointly optimizing for structural reasoning and safety alignment to effectively mitigate mid-trajectory risks. 

---
# Selective Neuron Amplification for Training-Free Task Enhancement 

**Authors**: Ryyan Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2604.07098)  

**Abstract**: Large language models often fail on tasks they seem to already understand. In our experiments, this appears to be less about missing knowledge and more about certain internal circuits not being strongly activated during inference. We explore Selective Neuron Amplification, which increases the influence of task relevant neurons without changing the model's parameters. The method works at inference time and does not permanently alter the model. SNA helps mainly when the model is uncertain, while having low effect when the model is already confident. This suggests that some model failures are due to weak activation rather than lack of capability. 

---
# Strategic Persuasion with Trait-Conditioned Multi-Agent Systems for Iterative Legal Argumentation 

**Authors**: Philipp D. Siedler  

**Link**: [PDF](https://arxiv.org/pdf/2604.07028)  

**Abstract**: Strategic interaction in adversarial domains such as law, diplomacy, and negotiation is mediated by language, yet most game-theoretic models abstract away the mechanisms of persuasion that operate through discourse. We present the Strategic Courtroom Framework, a multi-agent simulation environment in which prosecution and defense teams composed of trait-conditioned Large Language Model (LLM) agents engage in iterative, round-based legal argumentation. Agents are instantiated using nine interpretable traits organized into four archetypes, enabling systematic control over rhetorical style and strategic orientation.
We evaluate the framework across 10 synthetic legal cases and 84 three-trait team configurations, totaling over 7{,}000 simulated trials using DeepSeek-R1 and Gemini~2.5~Pro. Our results show that heterogeneous teams with complementary traits consistently outperform homogeneous configurations, that moderate interaction depth yields more stable verdicts, and that certain traits (notably quantitative and charismatic) contribute disproportionately to persuasive success. We further introduce a reinforcement-learning-based Trait Orchestrator that dynamically generates defense traits conditioned on the case and opposing team, discovering strategies that outperform static, human-designed trait combinations.
Together, these findings demonstrate how language can be treated as a first-class strategic action space and provide a foundation for building autonomous agents capable of adaptive persuasion in multi-agent environments. 

---
# Digital Skin, Digital Bias: Uncovering Tone-Based Biases in LLMs and Emoji Embeddings 

**Authors**: Mingchen Li, Wajdi Aljedaani, Yingjie Liu, Navyasri Meka, Xuan Lu, Xinyue Ye, Junhua Ding, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.06863)  

**Abstract**: Skin-toned emojis are crucial for fostering personal identity and social inclusion in online communication. As AI models, particularly Large Language Models (LLMs), increasingly mediate interactions on web platforms, the risk that these systems perpetuate societal biases through their representation of such symbols is a significant concern. This paper presents the first large-scale comparative study of bias in skin-toned emoji representations across two distinct model classes. We systematically evaluate dedicated emoji embedding models (emoji2vec, emoji-sw2v) against four modern LLMs (Llama, Gemma, Qwen, and Mistral). Our analysis first reveals a critical performance gap: while LLMs demonstrate robust support for skin tone modifiers, widely-used specialized emoji models exhibit severe deficiencies. More importantly, a multi-faceted investigation into semantic consistency, representational similarity, sentiment polarity, and core biases uncovers systemic disparities. We find evidence of skewed sentiment and inconsistent meanings associated with emojis across different skin tones, highlighting latent biases within these foundational models. Our findings underscore the urgent need for developers and platforms to audit and mitigate these representational harms, ensuring that AI's role on the web promotes genuine equity rather than reinforcing societal biases. 

---
# Video-guided Machine Translation with Global Video Context 

**Authors**: Jian Chen, JinZe Lv, Zi Long, XiangHua Fu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06789)  

**Abstract**: Video-guided Multimodal Translation (VMT) has advanced significantly in recent years. However, most existing methods rely on locally aligned video segments paired one-to-one with subtitles, limiting their ability to capture global narrative context across multiple segments in long videos. To overcome this limitation, we propose a globally video-guided multimodal translation framework that leverages a pretrained semantic encoder and vector database-based subtitle retrieval to construct a context set of video segments closely related to the target subtitle semantics. An attention mechanism is employed to focus on highly relevant visual content, while preserving the remaining video features to retain broader contextual information. Furthermore, we design a region-aware cross-modal attention mechanism to enhance semantic alignment during translation. Experiments on a large-scale documentary translation dataset demonstrate that our method significantly outperforms baseline models, highlighting its effectiveness in long-video scenarios. 

---
# From Perception to Autonomous Computational Modeling: A Multi-Agent Approach 

**Authors**: Daniel N. Wilke  

**Link**: [PDF](https://arxiv.org/pdf/2604.06788)  

**Abstract**: We present a solver-agnostic framework in which coordinated large language model (LLM) agents autonomously execute the complete computational mechanics workflow, from perceptual data of an engineering component through geometry extraction, material inference, discretisation, solver execution, uncertainty quantification, and code-compliant assessment, to an engineering report with actionable recommendations. Agents are formalised as conditioned operators on a shared context space with quality gates that introduce conditional iteration between pipeline layers. We introduce a mathematical framework for extracting engineering information from perceptual data under uncertainty using interval bounds, probability densities, and fuzzy membership functions, and introduce task-dependent conservatism to resolve the ambiguity of what `conservative' means when different limit states are governed by opposing parameter trends. The framework is demonstrated through a finite element analysis pipeline applied to a photograph of a steel L-bracket, producing a 171,504-node tetrahedral mesh, seven analyses across three boundary condition hypotheses, and a code-compliant assessment revealing structural failure with a quantified redesign. All results are presented as generated in the first autonomous iteration without manual correction, reinforcing that a professional engineer must review and sign off on any such analysis. 

---
# Geometric Properties of the Voronoi Tessellation in Latent Semantic Manifolds of Large Language Models 

**Authors**: Marshall Brett  

**Link**: [PDF](https://arxiv.org/pdf/2604.06767)  

**Abstract**: Language models operate on discrete tokens but compute in continuous vector spaces, inducing a Voronoi tessellation over the representation manifold. We study this tessellation empirically on Qwen3.5-4B-Base, making two contributions. First, using float32 margin recomputation to resolve bfloat16 quantization artifacts, we validate Mabrok's (2026) linear scaling law of the expressibility gap with $R^2$ = 0.9997 - the strongest confirmation to date - and identify a mid-layer geometric ambiguity regime where margin geometry is anti-correlated with cross-entropy (layers 24-28, $\rho$ = -0.29) before crystallizing into alignment at the final layer ($\rho$ = 0.836).
Second, we show that the Voronoi tessellation of a converged model is reshapable through margin refinement procedures (MRP): short post-hoc optimization runs that widen token-decision margins without retraining. We compare direct margin maximization against Fisher information distance maximization across a dose-response sweep. Both methods find the same ceiling of ~16,300 correctable positions per 256K evaluated, but differ critically in collateral damage. Margin maximization damage escalates with intervention strength until corrections are overwhelmed. Fisher damage remains constant at ~5,300 positions across the validated range ($\lambda$ = 0.15-0.6), achieving +28% median margin improvement at $\lambda$ = 0.6 with invariant downstream benchmarks - a geometric reorganization that compresses the expressibility gap while preserving its scaling law. However, frequency and token-class audits reveal that gains concentrate in high-frequency structural tokens (84% of net corrections at $\lambda$ = 0.6), with content and entity-like contributions shrinking at higher $\lambda$. Fisher MRP is therefore a viable geometric polishing tool whose practical ceiling is set not by aggregate damage but by the uniformity of token-level benefit. 

---
# Steering the Verifiability of Multimodal AI Hallucinations 

**Authors**: Jianhong Pang, Ruoxi Cheng, Ziyi Ye, Xingjun Ma, Zuxuan Wu, Xuanjing Huang, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06714)  

**Abstract**: AI applications driven by multimodal large language models (MLLMs) are prone to hallucinations and pose considerable risks to human users. Crucially, such hallucinations are not equally problematic: some hallucination contents could be detected by human users(i.e., obvious hallucinations), while others are often missed or require more verification effort(i.e., elusive hallucinations). This indicates that multimodal AI hallucinations vary significantly in their verifiability. Yet, little research has explored how to control this property for AI applications with diverse security and usability demands. To address this gap, we construct a dataset from 4,470 human responses to AI-generated hallucinations and categorize these hallucinations into obvious and elusive types based on their verifiability by human users. Further, we propose an activation-space intervention method that learns separate probes for obvious and elusive hallucinations. We reveal that obvious and elusive hallucinations elicit different intervention probes, allowing for fine-grained control over the model's verifiability. Empirical results demonstrate the efficacy of this approach and show that targeted interventions yield superior performance in regulating corresponding verifiability. Moreover, simply mixing these interventions enables flexible control over the verifiability required for different scenarios. 

---
# Specializing Large Models for Oracle Bone Script Interpretation via Component-Grounded Multimodal Knowledge Augmentation 

**Authors**: Jianing Zhang, Runan Li, Honglin Pang, Ding Xia, Zhou Zhu, Qian Zhang, Chuntao Li, Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06711)  

**Abstract**: Deciphering ancient Chinese Oracle Bone Script (OBS) is a challenging task that offers insights into the beliefs, systems, and culture of the ancient era. Existing approaches treat decipherment as a closed-set image recognition problem, which fails to bridge the ``interpretation gap'': while individual characters are often unique and rare, they are composed of a limited set of recurring, pictographic components that carry transferable semantic meanings. To leverage this structural logic, we propose an agent-driven Vision-Language Model (VLM) framework that integrates a VLM for precise visual grounding with an LLM-based agent to automate a reasoning chain of component identification, graph-based knowledge retrieval, and relationship inference for linguistically accurate interpretation. To support this, we also introduce OB-Radix, an expert-annotated dataset providing structural and semantic data absent from prior corpora, comprising 1,022 character images (934 unique characters) and 1,853 fine-grained component images across 478 distinct components with verified explanations. By evaluating our system across three benchmarks of different tasks, we demonstrate that our framework yields more detailed and precise decipherments compared to baseline methods. 

---
# SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning 

**Authors**: Zhengyang Ai, Zikang Shan, Xiaodong Ai, Jingxian Tang, Hangkai Hu, Pinyan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06636)  

**Abstract**: Process supervision has emerged as a promising approach for enhancing LLM reasoning, yet existing methods fail to distinguish meaningful progress from mere verbosity, leading to limited reasoning capabilities and unresolved token inefficiency. To address this, we propose Stage-aware Hierarchical Advantage via Potential Estimation (SHAPE), a framework that formalizes reasoning as a trajectory through a state space of empirical solvability. SHAPE introduces a hierarchical credit assignment mechanism: at the segment level, it employs a stage-aware advantage function to prioritize efficient breakthroughs in low-potential states; at the token level, it utilizes entropy-driven redistribution to sharpen execution signals. Extensive experiments in math reasoning across three base models and five benchmarks demonstrate that SHAPE achieves an average accuracy gain of 3% with 30% reduced token consumption. 

---
# Argus: Reorchestrating Static Analysis via a Multi-Agent Ensemble for Full-Chain Security Vulnerability Detection 

**Authors**: Zi Liang, Qipeng Xie, Jun He, Bohuan Xue, Weizheng Wang, Yuandao Cai, Fei Luo, Boxian Zhang, Haibo Hu, Kaishun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06633)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have sparked interest in their application to Static Application Security Testing (SAST), primarily due to their superior contextual reasoning capabilities compared to traditional symbolic or rule-based methods. However, existing LLM-based approaches typically attempt to replace human experts directly without integrating effectively with existing SAST tools. This lack of integration results in ineffectiveness, including high rates of false positives, hallucinations, limited reasoning depth, and excessive token usage, making them impractical for industrial deployment. To overcome these limitations, we present a paradigm shift that reorchestrates the SAST workflow from current LLM-assisted structure to a new LLM-centered workflow. We introduce Argus (Agentic and Retrieval-Augmented Guarding System), the first multi-agent framework designed specifically for vulnerability detection. Argus incorporates three key novelties: comprehensive supply chain analysis, collaborative multi-agent workflows, and the integration of state-of-the-art techniques such as Retrieval-Augmented Generation (RAG) and ReAct to minimize hallucinations and enhance reasoning. Extensive empirical evaluation demonstrates that Argus significantly outperforms existing methods by detecting a higher volume of true vulnerabilities while simultaneously reducing false positives and operational costs. Notably, Argus has identified several critical zero-day vulnerabilities with CVE assignments. 

---
# Transformer See, Transformer Do: Copying as an Intermediate Step in Learning Analogical Reasoning 

**Authors**: Philipp Hellwig, Willem Zuidema, Claire E. Stevenson, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2604.06501)  

**Abstract**: Analogical reasoning is a hallmark of human intelligence, enabling us to solve new problems by transferring knowledge from one situation to another. Yet, developing artificial intelligence systems capable of robust human-like analogical reasoning has proven difficult. In this work, we train transformers using Meta-Learning for Compositionality (MLC) on an analogical reasoning task (letter-string analogies) and assess their generalization capabilities. We find that letter-string analogies become learnable when guiding the models to attend to the most informative problem elements induced by including copying tasks in the training data. Furthermore, generalization to new alphabets becomes better when models are trained with more heterogeneous datasets, where our 3-layer encoder-decoder model outperforms most frontier models. The MLC approach also enables some generalization to compositions of trained transformations, but not to completely novel transformations. To understand how the model operates, we identify an algorithm that approximates the model's computations. We verify this using interpretability analyses and show that the model can be steered precisely according to expectations derived from the algorithm. Finally, we discuss implications of our findings for generalization capabilities of larger models and parallels to human analogical reasoning. 

---
# The Depth Ceiling: On the Limits of Large Language Models in Discovering Latent Planning 

**Authors**: Yi Xu, Philipp Jettkant, Laura Ruis  

**Link**: [PDF](https://arxiv.org/pdf/2604.06427)  

**Abstract**: The viability of chain-of-thought (CoT) monitoring hinges on models being unable to reason effectively in their latent representations. Yet little is known about the limits of such latent reasoning in LLMs. We test these limits by studying whether models can discover multi-step planning strategies without supervision on intermediate steps and execute them latently, within a single forward pass. Using graph path-finding tasks that precisely control the number of required latent planning steps, we uncover a striking limitation unresolved by massive scaling: tiny transformers trained from scratch discover strategies requiring up to three latent steps, fine-tuned GPT-4o and Qwen3-32B reach five, and GPT-5.4 attains seven under few-shot prompting. Although the maximum latent planning depth models can learn during training is five, the discovered strategy generalizes up to eight latent steps at test-time. This reveals a dissociation between the ability to discover a latent strategy under final-answer supervision alone and the ability to execute it once discovered. If similar limits hold more broadly, strategies requiring multiple coordinated latent planning steps may need to be explicitly taught or externalized, lending credence to CoT monitoring. 

---
# Say Something Else: Rethinking Contextual Privacy as Information Sufficiency 

**Authors**: Yunze Xiao, Wenkai Li, Xiaoyuan Wu, Ningshan Ma, Yueqi Song, Weihao Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.06409)  

**Abstract**: LLM agents increasingly draft messages on behalf of users, yet users routinely overshare sensitive information and disagree on what counts as private. Existing systems support only suppression (omitting sensitive information) and generalization (replacing information with an abstraction), and are typically evaluated on single isolated messages, leaving both the strategy space and evaluation setting incomplete. We formalize privacy-preserving LLM communication as an \textbf{Information Sufficiency (IS)} task, introduce \textbf{free-text pseudonymization} as a third strategy that replaces sensitive attributes with functionally equivalent alternatives, and propose a \textbf{conversational evaluation protocol} that assesses strategies under realistic multi-turn follow-up pressure. Across 792 scenarios spanning three power-relation types (institutional, peer, intimate) and three sensitivity categories (discrimination risk, social cost, boundary), we evaluate seven frontier LLMs on privacy at two granularities, covertness, and utility. Pseudonymization yields the strongest privacy\textendash utility tradeoff overall, and single-message evaluation systematically underestimates leakage, with generalization losing up to 16.3 percentage points of privacy under follow-up. 

---
# Weakly Supervised Distillation of Hallucination Signals into Transformer Representations 

**Authors**: Shoaib Sadiq Salehmohamed, Jinal Prashant Thakkar, Hansika Aredla, Shaik Mohammed Omar, Shalmali Ayachit  

**Link**: [PDF](https://arxiv.org/pdf/2604.06277)  

**Abstract**: Existing hallucination detection methods for large language models (LLMs) rely on external verification at inference time, requiring gold answers, retrieval systems, or auxiliary judge models. We ask whether this external supervision can instead be distilled into the model's own representations during training, enabling hallucination detection from internal activations alone at inference time.
We introduce a weak supervision framework that combines three complementary grounding signals: substring matching, sentence embedding similarity, and an LLM as a judge verdict to label generated responses as grounded or hallucinated without human annotation. Using this framework, we construct a 15000-sample dataset from SQuAD v2 (10500 train/development samples and a separate 5000-sample test set), where each example pairs a LLaMA-2-7B generated answer with its full per-layer hidden states and structured hallucination labels.
We then train five probing classifiers: ProbeMLP (M0), LayerWiseMLP (M1), CrossLayerTransformer (M2), HierarchicalTransformer (M3), and CrossLayerAttentionTransformerV2 (M4), directly on these hidden states, treating external grounding signals as training-time supervision only. Our central hypothesis is that hallucination detection signals can be distilled into transformer representations, enabling internal detection without any external verification at inference time.
Results support this hypothesis. Transformer-based probes achieve the strongest discrimination, with M2 performing best on 5-fold average AUC/F1, and M3 performing best on both single-fold validation and held-out test evaluation. We also benchmark inference efficiency: probe latency ranges from 0.15 to 5.62 ms (batched) and 1.55 to 6.66 ms (single sample), while end-to-end generation plus probe throughput remains approximately 0.231 queries per second, indicating negligible practical overhead. 

---
# Automating Database-Native Function Code Synthesis with LLMs 

**Authors**: Wei Zhou, Xuanhe Zhou, Qikang He, Guoliang Li, Bingsheng He, Quanqing Xu, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.06231)  

**Abstract**: Database systems incorporate an ever-growing number of functions in their kernels (a.k.a., database native functions) for scenarios like new application support and business migration. This growth causes an urgent demand for automatic database native function synthesis. While recent advances in LLM-based code generation (e.g., Claude Code) show promise, they are too generic for database-specific development. They often hallucinate or overlook critical context because database function synthesis is inherently complex and error-prone, where synthesizing a single function may involve registering multiple function units, linking internal references, and implementing logic correctly. To this end, we propose DBCooker, an LLM-based system for automatically synthesizing database native functions. It consists of three components. First, the function characterization module aggregates multi-source declarations, identifies function units that require specialized coding, and traces cross-unit dependencies. Second, we design operations to address the main synthesis challenges: (1) a pseudo-code-based coding plan generator that constructs structured implementation skeletons by identifying key elements such as reusable referenced functions; (2) a hybrid fill-in-the-blank model guided by probabilistic priors and component awareness to integrate core logic with reusable routines; and (3) three-level progressive validation, including syntax checking, standards compliance, and LLM-guided semantic verification. Finally, an adaptive orchestration strategy unifies these operations with existing tools and dynamically sequences them via the orchestration history of similar functions. Results show that DBCooker outperforms other methods on SQLite, PostgreSQL, and DuckDB (34.55% higher accuracy on average), and can synthesize new functions absent in the latest SQLite (v3.50). 

---
# Probabilistic Language Tries: A Unified Framework for Compression, Decision Policies, and Execution Reuse 

**Authors**: Gregory Magarshak  

**Link**: [PDF](https://arxiv.org/pdf/2604.06228)  

**Abstract**: We introduce probabilistic language tries (PLTs), a unified representation that makes explicit the prefix structure implicitly defined by any generative model over sequences. By assigning to each outgoing edge the conditional probability of the corresponding token or action, a PLT simultaneously serves as: (i) an optimal lossless compressor via frequency-weighted interval encoding, generalizing arithmetic coding to model-conditioned distributions; (ii) a policy representation for sequential decision problems including games, search, and robotic control; and (iii) a memoization index that lets repeated inference queries be answered by structured retrieval rather than full model execution.
The central technical result is a prior-guided caching theorem: under a stationary generative distribution, a PLT-guided cache achieves strictly lower expected inference cost than any empirical-frequency cache for all query counts below a threshold that grows with the concentration of the prior. This converts O(n^2) transformer attention cost into an expected cost of p_r * O(log N) + (1 - p_r) * O(n^2), where p_r is the prior-estimated reuse probability and N is the artifact store size.
We further introduce a hybrid compression architecture decomposing any dataset into a PLT-covered majority and a sparse residual store, connecting arithmetic coding with Kolmogorov-style program representations and rate-distortion theory. We instantiate the framework across chess, web search, robotics, organizational workflows, and LLM inference, demonstrating that compression, decision making, and computational reuse are all derived from a single probability measure on sequence space. 

---
# Code Sharing In Prediction Model Research: A Scoping Review 

**Authors**: Thomas Sounack, Raffaele Giancotti, Catherine A. Gao, Lasai Barreñada, Hyeonhoon Lee, Hyung-Chul Lee, Leo Anthony Celi, Karel G.M. Moons, Gary S. Collins, Charlotta Lindvall, Tom Pollard  

**Link**: [PDF](https://arxiv.org/pdf/2604.06212)  

**Abstract**: Analytical code is essential for reproducing diagnostic and prognostic prediction model research, yet code availability in the published literature remains limited. While the TRIPOD statements set standards for reporting prediction model methods, they do not define explicit standards for repository structure and documentation. This review quantifies current code-sharing practices to inform the development of TRIPOD-Code, a TRIPOD extension reporting guideline focused on code sharing.
We conducted a scoping review of PubMed-indexed articles citing TRIPOD or TRIPOD+AI as of Aug 11, 2025, restricted to studies retrievable via the PubMed Central Open Access API. Eligible studies developed, updated, or validated multivariable prediction models. A large language model-assisted pipeline was developed to screen articles and extract code availability statements and repository links. Repositories were assessed with the same LLM against 14 predefined reproducibility-related features. Our code is made publicly available.
Among 3,967 eligible articles, 12.2% included code sharing statements. Code sharing increased over time, reaching 15.8% in 2025, and was higher among TRIPOD+AI-citing studies than TRIPOD-citing studies. Sharing prevalence varied widely by journal and country. Repository assessment showed substantial heterogeneity in reproducibility features: most repositories contained a README file (80.5%), but fewer specified dependencies (37.6%; version-constrained 21.6%) or were modular (42.4%).
In prediction model research, code sharing remains relatively uncommon, and when shared, often falls short of being reusable. These findings provide an empirical baseline for the TRIPOD-Code extension and underscore the need for clearer expectations beyond code availability, including documentation, dependency specification, licensing, and executable structure. 

---
# The Human Condition as Reflected in Contemporary Large Language Models 

**Authors**: W. Russell Neuman  

**Link**: [PDF](https://arxiv.org/pdf/2604.06206)  

**Abstract**: This study seeks to uncover evidence of a latent structure in evolved human culture as it is refracted through contemporary large language models (LLMs). Drawing on parallel responses from six leading generative models to a prompt which asks directly what their training corpora reveal about human culture and behavior, we identify a robust cross-model consensus on a limited set of recurring cultural themes. The themes include narrative meaning-making, affect-first cognition, coalition psychology, status competition, threat sensitivity, and moral rationalization. Each provides grounds for further psychological and sociological inquiry. There is strong evidence of a convergence in these pattern recognition exercises as differences among models are shown to reflect varying explanatory lenses rather than substantive disagreement. We review these findings in the light of the evolving literatures of moral psychology, evolutionary psychology, anthropology, and the computer science literature on large-scale language modeling. We argue that LLMs function as cultural condensates -- compressed representations of how humans describe, justify, and contest their own social lives across trillions of tokens of aggregated communication and narration. 

---
# Harf-Speech: A Clinically Aligned Framework for Arabic Phoneme-Level Speech Assessment 

**Authors**: Asif Azad, MD Sadik Hossain Shanto, Mohammad Sadat Hossain, Bdour Alwuqaysi, Sabri Boughorbel, Yahya Bokhari, Abdulrhman Aljouie, Ayah Othman Sindi, Ehsan Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2604.06191)  

**Abstract**: Automated phoneme-level pronunciation assessment is vital for scalable speech therapy and language learning, yet validated tools for Arabic remain scarce. We present Harf-Speech, a modular system scoring Arabic pronunciation at the phoneme level on a clinical scale. It combines an MSA phonetizer, a fine-tuned speech-to-phoneme model, Levenshtein alignment, and a blended scorer using longest common subsequence and edit-distance metrics. We fine-tune three ASR architectures on Arabic phoneme data and benchmark them with zero-shot multimodal models; the best, OmniASR-CTC-1B-v2, achieves 8.92\% phoneme error rate. Three certified speech-language pathologists independently scored 40 utterances for clinical validation. Harf-Speech attains a Pearson correlation of 0.791 and ICC(2,1) of 0.659 with mean expert scores, outperforming existing end-to-end assessment frameworks. These results show Harf-Speech yields clinically aligned, interpretable scores comparable to inter-rater expert agreement. 

---
# LLM Spirals of Delusion: A Benchmarking Audit Study of AI Chatbot Interfaces 

**Authors**: Peter Kirgis, Ben Hawriluk, Sherrie Feng, Aslan Bilimer, Sam Paech, Zeynep Tufekci  

**Link**: [PDF](https://arxiv.org/pdf/2604.06188)  

**Abstract**: People increasingly hold sustained, open-ended conversations with large language models (LLMs). Public reports and early studies suggest that, in such settings, models can reinforce delusional or conspiratorial ideation or even amplify harmful beliefs and engagement patterns. We present an audit and benchmarking study that measures how different LLMs encourage, resist, or escalate disordered and conspiratorial thinking. We explicitly compare API outputs to user chat interfaces, like the ChatGPT desktop app or web interface, which is how people have conversations with chatbots in real life but are almost never used for testing. In total, we run 56 20-turn conversations testing ChatGPT-4o and ChatGPT-5, via both the API and chat interface, and grade each conversation by two research assistants (RAs) as well as by GPT-5. We document five results. First, we observe large differences in performance between the API and chat interface environments, showing that the universally used method of automated testing through the API is not sufficient to assess the impact of chatbots in the real world. Second, when tested in the chat interface, we find that ChatGPT-5 displays less sycophancy, escalation, and delusion reinforcement than ChatGPT-4o, showing that these behaviors are influenced by the policy choices of major AI companies. Third, conversations with nearly identical aggregate intensity in a behavior display large differences in how the behavior evolves turn by turn, highlighting the importance of temporal dynamics in multi-turn evaluation. Fourth, even updated models display substantial levels of negative behaviors, revealing that model improvement does not imply model safety. Fifth, the same API endpoint tested just two months apart yields a complete reversal in behavior, underscoring how transparency in model updates is a necessary prerequisite for robust audit findings. 

---
# Benchmarking LLM Tool-Use in the Wild 

**Authors**: Peijie Yu, Wei Liu, Yifan Yang, Jinjian Li, Zelong Zhang, Xiao Feng, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.06185)  

**Abstract**: Fulfilling user needs through Large Language Model multi-turn, multi-step tool-use is rarely a straightforward process. Real user interactions are inherently wild, being intricate, messy, and flexible. We identify three key challenges from user behaviour: compositional tasks that demand efficient orchestration of tool-call topologies, implicit intent spread across dialogue turns that require contextual inference, and instruction transition, which mixes task queries, clarifications, and casual conversation, forcing LLMs to adjust their policies on the fly. Existing benchmarks overlook these behaviors, making the apparent progress of LLMs on tool-use spurious. To address this, we introduce WildToolBench, an LLM tool-use benchmark grounded in real-world user behavior patterns. Comprehensive evaluations of 57 LLMs reveal that no model achieves an accuracy of more than 15%, indicating a substantial gap in the robustness of LLMs' agentic ability. Controlled experiments and in-depth analyses further indicate that the real challenge for LLM tool-use lies not in artificially complex tasks, but in the wild nature of user behavior, emphasizing the need to reconsider the interactions among LLMs, users, and tools. 

---
# ARIA: Adaptive Retrieval Intelligence Assistant -- A Multimodal RAG Framework for Domain-Specific Engineering Education 

**Authors**: Yue Luo, Dibakar Roy Sarkar, Rachel Herring Sangree, Somdatta Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2604.06179)  

**Abstract**: Developing effective, domain-specific educational support systems is central to advancing AI in education. Although large language models (LLMs) demonstrate remarkable capabilities, they face significant limitations in specialized educational applications, including hallucinations, limited knowledge updates, and lack of domain expertise. Fine-tuning requires complete model retraining, creating substantial computational overhead, while general-purpose LLMs often provide inaccurate responses in specialized contexts due to reliance on generalized training data. To address this, we propose ARIA (Adaptive Retrieval Intelligence Assistant), a Retrieval-Augmented Generation (RAG) framework for creating intelligent teaching assistants across university-level courses. ARIA leverages a multimodal content extraction pipeline combining Docling for structured document analysis, Nougat for mathematical formula recognition, and GPT-4 Vision API for diagram interpretation, with the e5-large-v2 embedding model for high semantic performance and low latency. This enables accurate processing of complex educational materials while maintaining pedagogical consistency through engineered prompts and response controls. We evaluate ARIA using lecture material from Statics and Mechanics of Materials, a sophomore-level civil engineering course at Johns Hopkins University, benchmarking against ChatGPT-5. Results demonstrate 97.5% accuracy in domain-specific question filtering and superior pedagogical performance. ARIA correctly answered all 20 relevant course questions while rejecting 58 of 60 non-relevant queries, achieving 90.9% precision, 100% recall, and 4.89/5.0 average response quality. These findings demonstrate that ARIA's course-agnostic architecture represents a scalable framework for domain-specific educational AI deployment. 

---
# WebExpert: domain-aware web agents with critic-guided expert experience for high-precision search 

**Authors**: Yuelin Hu, Zhengxue Cheng, Ronghua Wu, Qunshan Gu, Hongwei Hu, Wei Liu, Qiao Liang, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.06177)  

**Abstract**: Specialized web tasks in finance, biomedicine, and pharmaceuticals remain challenging due to missing domain priors: queries drift, evidence is noisy, and reasoning is brittle. We present WebExpert, a domain-aware web agent that we implement end-to-end, featuring : (i) sentence-level experience retrieval with topic merging and rule distillation, (ii) schemalight facet induction that bootstraps time,region,policy,industry facets from weak supervision instead of static hand-written lexicons, and (iii) preference-optimized planning that jointly improves query planning and retrieval via pairwise preference learning alongside a coverage-aware objective. At inference, a lightweight experience gate biases decoding toward active facets with fallback under low-retrieval confidence. On GAIA, GPQA, HLE, and WebWalkerQA, WebExpert improves Answer Exact Match (EM) by 1.5-3.6 pp over the strongest browsing baseline and reduces page hops. Analysis shows consistent gains and ablations on retrieval, topic merging, facet induction, and preference-aware training. 

---
# Robustness Risk of Conversational Retrieval: Identifying and Mitigating Noise Sensitivity in Qwen3-Embedding Model 

**Authors**: Weishu Chen, Zhouhui Hou, Mingjie Zhan, Zhicheng Zhao, Fei Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.06176)  

**Abstract**: We present an empirical study of embedding-based retrieval under realistic conversational settings, where queries are short, dialogue-like, and weakly specified, and retrieval corpora contain structured conversational artifacts. Focusing on Qwen3-embedding models, we identify a deployment-relevant robustness vulnerability: under conversational retrieval without query prompting, structured dialogue-style noise can become disproportionately retrievable and intrude into top-ranked results, despite being semantically uninformative. This failure mode emerges consistently across model scales, remains largely invisible under standard clean-query benchmarks, and is significantly more pronounced in Qwen3 than in earlier Qwen variants and other widely used dense retrieval baselines. We further show that lightweight query prompting qualitatively alters retrieval behavior, effectively suppressing noise intrusion and restoring ranking stability. Our findings highlight an underexplored robustness risk in conversational retrieval and underscore the importance of evaluation protocols that reflect the complexities of deployed systems. 

---
