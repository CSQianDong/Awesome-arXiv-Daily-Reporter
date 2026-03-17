# Mixture-of-Depths Attention 

**Authors**: Lianghui Zhu, Yuxin Fang, Bencheng Liao, Shijie Wang, Tianheng Cheng, Zilong Huang, Chen Chen, Lai Wei, Yutao Zeng, Ya Wang, Yi Lin, Yu Li, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15619)  

**Abstract**: Scaling depth is a key driver for large language models (LLMs). Yet, as LLMs become deeper, they often suffer from signal degradation: informative features formed in shallow layers are gradually diluted by repeated residual updates, making them harder to recover in deeper layers. We introduce mixture-of-depths attention (MoDA), a mechanism that allows each attention head to attend to sequence KV pairs at the current layer and depth KV pairs from preceding layers. We further describe a hardware-efficient algorithm for MoDA that resolves non-contiguous memory-access patterns, achieving 97.3% of FlashAttention-2's efficiency at a sequence length of 64K. Experiments on 1.5B-parameter models demonstrate that MoDA consistently outperforms strong baselines. Notably, it improves average perplexity by 0.2 across 10 validation benchmarks and increases average performance by 2.11% on 10 downstream tasks, with a negligible 3.7% FLOPs computational overhead. We also find that combining MoDA with post-norm yields better performance than using it with pre-norm. These results suggest that MoDA is a promising primitive for depth scaling. Code is released at this https URL . 

---
# Mechanistic Origin of Moral Indifference in Language Models 

**Authors**: Lingyu Li, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15615)  

**Abstract**: Existing behavioral alignment techniques for Large Language Models (LLMs) often neglect the discrepancy between surface compliance and internal unaligned representations, leaving LLMs vulnerable to long-tail risks. More crucially, we posit that LLMs possess an inherent state of moral indifference due to compressing distinct moral concepts into uniform probability distributions. We verify and remedy this indifference in LLMs' latent representations, utilizing 251k moral vectors constructed upon Prototype Theory and the Social-Chemistry-101 dataset. Firstly, our analysis across 23 models reveals that current LLMs fail to represent the distinction between opposed moral categories and fine-grained typicality gradients within these categories; notably, neither model scaling, architecture, nor explicit alignment reshapes this indifference. We then employ Sparse Autoencoders on Qwen3-8B, isolate mono-semantic moral features, and targetedly reconstruct their topological relationships to align with ground-truth moral vectors. This representational alignment naturally improves moral reasoning and granularity, achieving a 75% pairwise win-rate on the independent adversarial Flames benchmark. Finally, we elaborate on the remedial nature of current intervention methods from an experientialist philosophy, arguing that endogenously aligned AI might require a transformation from post-hoc corrections to proactive cultivation. 

---
# Code-A1: Adversarial Evolving of Code LLM and Test LLM via Reinforcement Learning 

**Authors**: Aozhe Wang, Yuchen Yan, Nan Zhou, Zhengxi Lu, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.15611)  

**Abstract**: Reinforcement learning for code generation relies on verifiable rewards from unit test pass rates. Yet high-quality test suites are scarce, existing datasets offer limited coverage, and static rewards fail to adapt as models improve. Recent self-play methods unify code and test generation in a single model, but face a inherent dilemma: white-box access leads to self-collusion where the model produces trivial tests for easy rewards, yet black-box restriction yields generic tests that miss implementation-specific bugs. We introduce Code-A1, an adversarial co-evolution framework that jointly optimizes a Code LLM and a Test LLM with opposing objectives. The Code LLM is rewarded for passing more tests, while the Test LLM is rewarded for exposing more defects. This architectural separation eliminates self-collusion risks and safely enables white-box test generation, where the Test LLM can inspect candidate code to craft targeted adversarial tests. We further introduce a Mistake Book mechanism for experience replay and a composite reward balancing test validity with adversarial difficulty. Experiments on Qwen2.5-Coder models demonstrate that Code-A1 achieves code generation performance matching or exceeding models trained on human-annotated tests, while significantly improving test generation capability. 

---
# Can LLMs Model Incorrect Student Reasoning? A Case Study on Distractor Generation 

**Authors**: Yanick Zengaffinen, Andreas Opedal, Donya Rooein, Kv Aditya Srivatsa, Shashank Sonkar, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2603.15547)  

**Abstract**: Modeling plausible student misconceptions is critical for AI in education. In this work, we examine how large language models (LLMs) reason about misconceptions when generating multiple-choice distractors, a task that requires modeling incorrect yet plausible answers by coordinating solution knowledge, simulating student misconceptions, and evaluating plausibility. We introduce a taxonomy for analyzing the strategies used by state-of-the-art LLMs, examining their reasoning procedures and comparing them to established best practices in the learning sciences. Our structured analysis reveals a surprising alignment between their processes and best practices: the models typically solve the problem correctly first, then articulate and simulate multiple potential misconceptions, and finally select a set of distractors. An analysis of failure modes reveals that errors arise primarily from failures in recovering the correct solution and selecting among response candidates, rather than simulating errors or structuring the process. Consistent with these results, we find that providing the correct solution in the prompt improves alignment with human-authored distractors by 8%, highlighting the critical role of anchoring to the correct solution when generating plausible incorrect student reasoning. Overall, our analysis offers a structured and interpretable lens into LLMs' ability to model incorrect student reasoning and produce high-quality distractors. 

---
# SlovKE: A Large-Scale Dataset and LLM Evaluation for Slovak Keyphrase Extraction 

**Authors**: David Števaňák, Marek Šuppa  

**Link**: [PDF](https://arxiv.org/pdf/2603.15523)  

**Abstract**: Keyphrase extraction for morphologically rich, low-resource languages remains understudied, largely due to the scarcity of suitable evaluation datasets. We address this gap for Slovak by constructing a dataset of 227,432 scientific abstracts with author-assigned keyphrases -- scraped and systematically cleaned from the Slovak Central Register of Theses -- representing a 25-fold increase over the largest prior Slovak resource and approaching the scale of established English benchmarks such as KP20K. Using this dataset, we benchmark three unsupervised baselines (YAKE, TextRank, KeyBERT with SlovakBERT embeddings) and evaluate KeyLLM, an LLM-based extraction method using GPT-3.5-turbo. Unsupervised baselines achieve at most 11.6\% exact-match $F1@6$, with a large gap to partial matching (up to 51.5\%), reflecting the difficulty of matching inflected surface forms to author-assigned keyphrases. KeyLLM narrows this exact--partial gap, producing keyphrases closer to the canonical forms assigned by authors, while manual evaluation on 100 documents ($\kappa = 0.61$) confirms that KeyLLM captures relevant concepts that automated exact matching underestimates. Our analysis identifies morphological mismatch as the dominant failure mode for statistical methods -- a finding relevant to other inflected languages. The dataset (this https URL) and evaluation code (this https URL) are publicly available. 

---
# Beyond the Covariance Trap: Unlocking Generalization in Same-Subject Knowledge Editing for Large Language Models 

**Authors**: Xiyu Liu, Qingyi Si, Zhengxiao Liu, Chenxu Yang, Naibin Gu, Zheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.15518)  

**Abstract**: While locate-then-edit knowledge editing efficiently updates knowledge encoded within Large Language Models (LLMs), a critical generalization failure mode emerges in the practical same-subject knowledge editing scenario: models fail to recall the updated knowledge when following user instructions, despite successfully recalling it in the original edited form. This paper identifies the geometric root of this generalization collapse as a fundamental conflict where the inner activation drifts induced by prompt variations exceed the model's geometric tolerance for generalization after editing. We attribute this instability to a dual pathology: (1) The joint optimization with orthogonal gradients collapses solutions into sharp minima with narrow stability, and (2) the standard covariance constraint paradoxically acts as a Covariance Trap that amplifies input perturbations. To resolve this, we introduce RoSE (Robust Same-subject Editing), which employs Isotropic Geometric Alignment to minimize representational deviation and Hierarchical Knowledge Integration to smooth the optimization landscape. Extensive experiments demonstrate that RoSE significantly improves instruction-following capabilities, laying the foundation for robust interactive parametric memory of LLM agents. 

---
# ViX-Ray: A Vietnamese Chest X-Ray Dataset for Vision-Language Models 

**Authors**: Duy Vu Minh Nguyen, Chinh Thanh Truong, Phuc Hoang Tran, Hung Tuan Le, Nguyen Van-Thanh Dat, Trung Hieu Pham, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.15513)  

**Abstract**: Vietnamese medical research has become an increasingly vital domain, particularly with the rise of intelligent technologies aimed at reducing time and resource burdens in clinical diagnosis. Recent advances in vision-language models (VLMs), such as Gemini and GPT-4V, have sparked a growing interest in applying AI to healthcare. However, most existing VLMs lack exposure to Vietnamese medical data, limiting their ability to generate accurate and contextually appropriate diagnostic outputs for Vietnamese patients. To address this challenge, we introduce ViX-Ray, a novel dataset comprising 5,400 Vietnamese chest X-ray images annotated with expert-written findings and impressions from physicians at a major Vietnamese hospital. We analyze linguistic patterns within the dataset, including the frequency of mentioned body parts and diagnoses, to identify domain-specific linguistic characteristics of Vietnamese radiology reports. Furthermore, we fine-tune five state-of-the-art open-source VLMs on ViX-Ray and compare their performance to leading proprietary models, GPT-4V and Gemini. Our results show that while several models generate outputs partially aligned with clinical ground truths, they often suffer from low precision and excessive hallucination, especially in impression generation. These findings not only demonstrate the complexity and challenge of our dataset but also establish ViX-Ray as a valuable benchmark for evaluating and advancing vision-language models in the Vietnamese clinical domain. 

---
# Invisible failures in human-AI interactions 

**Authors**: Christopher Potts, Moritz Sudhof  

**Link**: [PDF](https://arxiv.org/pdf/2603.15423)  

**Abstract**: AI systems fail silently far more often than they fail visibly. In a large-scale quantitative analysis of human-AI interactions from the WildChat dataset, we find that 78% of AI failures are invisible: something went wrong but the user gave no overt indication that there was a problem. These invisible failures cluster into eight archetypes that help us characterize where and how AI systems are failing to meet users' needs. In addition, the archetypes show systematic co-occurrence patterns indicating higher-level failure types. To address the question of whether these archetypes will remain relevant as AI systems become more capable, we also assess failures for whether they are primarily interactional or capability-driven, finding that 91% involve interactional dynamics, and we estimate that 94% of such failures would persist even with a more capable model. Finally, we illustrate how the archetypes help us to identify systematic and variable AI limitations across different usage domains. Overall, we argue that our invisible failure taxonomy can be a key component in reliable failure monitoring for product developers, scientists, and policy makers. Our code and data are available at this https URL 

---
# CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents 

**Authors**: Taeyun Roh, Wonjune Jang, Junha Jung, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15421)  

**Abstract**: Large language model agents heavily rely on external memory to support knowledge reuse and complex reasoning tasks. Yet most memory systems store experiences in a single global retrieval pool which can gradually dilute or corrupt stored knowledge. This problem is especially pronounced for small language models (SLMs), which are highly vulnerable to irrelevant context. We introduce CLAG, a CLustering-based AGentic memory framework where an SLM agent actively organizes memory by clustering. CLAG employs an SLM-driven router to assign incoming memories to semantically coherent clusters and autonomously generates cluster-specific profiles, including topic summaries and descriptive tags, to establish each cluster as a self-contained functional unit. By performing localized evolution within these structured neighborhoods, CLAG effectively reduces cross-topic interference and enhances internal memory density. During retrieval, the framework utilizes a two-stage process that first filters relevant clusters via their profiles, thereby excluding distractors and reducing the search space. Experiments on multiple QA datasets with three SLM backbones show that CLAG consistently improves answer quality and robustness over prior memory systems for agents, remaining lightweight and efficient. 

---
# SEA-Vision: A Multilingual Benchmark for Comprehensive Document and Scene Text Understanding in Southeast Asia 

**Authors**: Pengfei Yue, Xingran Zhao, Juntao Chen, Peng Hou, Wang Longchao, Jianghang Lin, Shengchuan Zhang, Anxiang Zeng, Liujuan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.15409)  

**Abstract**: Multilingual document and scene text understanding plays an important role in applications such as search, finance, and public services. However, most existing benchmarks focus on high-resource languages and fail to evaluate models in realistic multilingual environments. In Southeast Asia, the diversity of languages, complex writing systems, and highly varied document types make this challenge even greater. We introduce SEA-Vision, a benchmark that jointly evaluates Document Parsing and Text-Centric Visual Question Answering (TEC-VQA) across 11 Southeast Asian languages. SEA-Vision contains 15,234 document parsing pages from nine representative document types, annotated with hierarchical page-, block-, and line-level labels. It also provides 7,496 TEC-VQA question-answer pairs that probe text recognition, numerical calculation, comparative analysis, logical reasoning, and spatial understanding. To make such multilingual, multi-task annotation feasible, we design a hybrid pipeline for Document Parsing and TEC-VQA. It combines automated filtering and scoring with MLLM-assisted labeling and lightweight native-speaker verification, greatly reducing manual labeling while maintaining high quality. We evaluate several leading multimodal models and observe pronounced performance degradation on low-resource Southeast Asian languages, highlighting substantial remaining gaps in multilingual document and scene text understanding. We believe SEA-Vision will help drive global progress in document and scene text understanding. 

---
# Fusian: Multi-LoRA Fusion for Fine-Grained Continuous MBTI Personality Control in Large Language Models 

**Authors**: Zehao Chen, Rong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2603.15405)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in simulating diverse human behaviors and personalities. However, existing methods for personality control, which include prompt engineering and standard Supervised Fine-Tuning (SFT), typically treat personality traits as discrete categories (e.g., "Extroverted" vs. "Introverted"), lacking the ability to precisely control the intensity of a trait on a continuous spectrum. In this paper, we introduce Fusian, a novel framework for fine-grained, continuous personality control in LLMs. Fusian operates in two stages: (1) Trajectory Collection, where we capture the dynamic evolution of personality adoption during SFT by saving a sequence of LoRA adapters, effectively mapping the continuous manifold of a trait; and (2) RL-based Dynamic Fusion, where we train a policy network using Reinforcement Learning to dynamically compute mixing weights for these frozen adapters. By sampling from a Dirichlet distribution parameterized by the policy network, Fusian fuses multiple adapters to align the model's output with a specific numerical target intensity. Experiments on the Qwen3-14B model demonstrate that Fusian achieves high precision in personality control, significantly outperforming baseline methods in aligning with user-specified trait intensities. 

---
# A Closer Look into LLMs for Table Understanding 

**Authors**: Jia Wang, Chuanyu Qin, Mingyu Zheng, Qingyi Si, Peize Li, Zheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.15402)  

**Abstract**: Despite the success of Large Language Models (LLMs) in table understanding, their internal mechanisms remain unclear. In this paper, we conduct an empirical study on 16 LLMs, covering general LLMs, specialist tabular LLMs, and Mixture-of-Experts (MoE) models, to explore how LLMs understand tabular data and perform downstream tasks. Our analysis focus on 4 dimensions including the attention dynamics, the effective layer depth, the expert activation, and the impacts of input designs. Key findings include: (1) LLMs follow a three-phase attention pattern -- early layers scan the table broadly, middle layers localize relevant cells, and late layers amplify their contributions; (2) tabular tasks require deeper layers than math reasoning to reach stable predictions; (3) MoE models activate table-specific experts in middle layers, with early and late layers sharing general-purpose experts; (4) Chain-of-Thought prompting increases table attention, further enhanced by table-tuning. We hope these findings and insights can facilitate interpretability and future research on table-related tasks. 

---
# When Does Sparsity Mitigate the Curse of Depth in LLMs 

**Authors**: Dilxat Muhtar, Xinyuan Song, Sebastian Pokutta, Max Zimmer, Nico Pelleriti, Thomas Hofmann, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15389)  

**Abstract**: Recent work has demonstrated the curse of depth in large language models (LLMs), where later layers contribute less to learning and representation than earlier layers. Such under-utilization is linked to the accumulated growth of variance in Pre-Layer Normalization, which can push deep blocks toward near-identity behavior. In this paper, we demonstrate that, sparsity, beyond enabling efficiency, acts as a regulator of variance propagation and thereby improves depth utilization. Our investigation covers two sources of sparsity: (i) implicit sparsity, which emerges from training and data conditions, including weight sparsity induced by weight decay and attention sparsity induced by long context inputs; and (ii) explicit sparsity, which is enforced by architectural design, including key/value-sharing sparsity in Grouped-Query Attention and expert-activation sparsity in Mixtureof-Experts. Our claim is thoroughly supported by controlled depth-scaling experiments and targeted layer effectiveness interventions. Across settings, we observe a consistent relationship: sparsity improves layer utilization by reducing output variance and promoting functional differentiation. We eventually distill our findings into a practical rule-of-thumb recipe for training deptheffective LLMs, yielding a notable 4.6% accuracy improvement on downstream tasks. Our results reveal sparsity, arising naturally from standard design choices, as a key yet previously overlooked mechanism for effective depth scaling in LLMs. Code is available at this https URL. 

---
# DOS: Dependency-Oriented Sampler for Masked Diffusion Language Models 

**Authors**: Xueyu Zhou, Yangrong Hu, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15340)  

**Abstract**: Masked diffusion language models (MDLMs) have recently emerged as a new paradigm in language modeling, offering flexible generation dynamics and enabling efficient parallel decoding. However, existing decoding strategies for pre-trained MDLMs predominantly rely on token-level uncertainty criteria, while largely overlooking sequence-level information and inter-token dependencies. To address this limitation, we propose Dependency-Oriented Sampler (DOS), a training-free decoding strategy that leverages inter-token dependencies to inform token updates during generation. Specifically, DOS exploits attention matrices from transformer blocks to approximate inter-token dependencies, emphasizing information from unmasked tokens when updating masked positions. Empirical results demonstrate that DOS consistently achieves superior performance on both code generation and mathematical reasoning tasks. Moreover, DOS can be seamlessly integrated with existing parallel sampling methods, leading to improved generation efficiency without sacrificing generation quality. 

---
# Tagarela - A Portuguese speech dataset from podcasts 

**Authors**: Frederico Santos de Oliveira, Lucas Rafael Stefanel Gris, Alef Iury Siqueira Ferreira, Augusto Seben da Rosa, Alexandre Costa Ferro Filho, Edresson Casanova, Christopher Dane Shulby, Rafael Teixeira Sousa, Diogo Fernandes Costa Silva, Anderson da Silva Soares, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2603.15326)  

**Abstract**: Despite significant advances in speech processing, Portuguese remains under-resourced due to the scarcity of public, large-scale, and high-quality datasets. To address this gap, we present a new dataset, named TAGARELA, composed of over 8,972 hours of podcast audio, specifically curated for training automatic speech recognition (ASR) and text-to-speech (TTS) models. Notably, its scale rivals English's GigaSpeech (10kh), enabling state-of-the-art Portuguese models. To ensure data quality, the corpus was subjected to an audio pre-processing pipeline and subsequently transcribed using a mixed strategy: we applied ASR models that were previously trained on high-fidelity transcriptions generated by proprietary APIs, ensuring a high level of initial accuracy. Finally, to validate the effectiveness of this new resource, we present ASR and TTS models trained exclusively on our dataset and evaluate their performance, demonstrating its potential to drive the development of more robust and natural speech technologies for Portuguese. The dataset is released publicly, available at this https URL, to foster the development of robust speech technologies. 

---
# PYTHEN: A Flexible Framework for Legal Reasoning in Python 

**Authors**: Ha-Thanh Nguyen, Ken Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2603.15317)  

**Abstract**: This paper introduces PYTHEN, a novel Python-based framework for defeasible legal reasoning. PYTHEN is designed to model the inherently defeasible nature of legal argumentation, providing a flexible and intuitive syntax for representing legal rules, conditions, and exceptions. Inspired by PROLEG (PROlog-based LEGal reasoning support system) and guided by the philosophy of The Zen of Python, PYTHEN leverages Python's built-in any() and all() functions to offer enhanced flexibility by natively supporting both conjunctive (ALL) and disjunctive (ANY) conditions within a single rule, as well as a more expressive exception-handling mechanism. This paper details the architecture of PYTHEN, provides a comparative analysis with PROLEG, and discusses its potential applications in autoformalization and the development of next-generation legal AI systems. By bridging the gap between symbolic reasoning and the accessibility of Python, PYTHEN aims to democratize formal legal reasoning for young researchers, legal tech developers, and professionals without extensive logic programming expertise. We position PYTHEN as a practical bridge between the powerful symbolic reasoning capabilities of logic programming and the rich, ubiquitous ecosystem of Python, making formal legal reasoning accessible to a broader range of developers and legal professionals. 

---
# CCTU: A Benchmark for Tool Use under Complex Constraints 

**Authors**: Junjie Ye, Guoqiang Zhang, Wenjie Fu, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15309)  

**Abstract**: Solving problems through tool use under explicit constraints constitutes a highly challenging yet unavoidable scenario for large language models (LLMs), requiring capabilities such as function calling, instruction following, and self-refinement. However, progress has been hindered by the absence of dedicated evaluations. To address this, we introduce CCTU, a benchmark for evaluating LLM tool use under complex constraints. CCTU is grounded in a taxonomy of 12 constraint categories spanning four dimensions (i.e., resource, behavior, toolset, and response). The benchmark comprises 200 carefully curated and challenging test cases across diverse tool-use scenarios, each involving an average of seven constraint types and an average prompt length exceeding 4,700 tokens. To enable reliable evaluation, we develop an executable constraint validation module that performs step-level validation and enforces compliance during multi-turn interactions between models and their environments. We evaluate nine state-of-the-art LLMs in both thinking and non-thinking modes. Results indicate that when strict adherence to all constraints is required, no model achieves a task completion rate above 20%. Further analysis reveals that models violate constraints in over 50% of cases, particularly in the resource and response dimensions. Moreover, LLMs demonstrate limited capacity for self-refinement even after receiving detailed feedback on constraint violations, highlighting a critical bottleneck in the development of robust tool-use agents. To facilitate future research, we release the data and code. 

---
# Datasets for Verb Alternations across Languages: BLM Templates and Data Augmentation Strategies 

**Authors**: Giuseppe Samo, Paola Merlo  

**Link**: [PDF](https://arxiv.org/pdf/2603.15295)  

**Abstract**: Large language models (LLMs) have shown remarkable performance across various sentence-based linguistic phenomena, yet their ability to capture cross-sentence paradigmatic patterns, such as verb alternations, remains underexplored. In this work, we present curated paradigm-based datasets for four languages, designed to probe systematic cross-sentence knowledge of verb alternations (change-of-state and object-drop constructions in English, German and Italian, and Hebrew binyanim). The datasets comprise thousands of the Blackbird Language Matrices (BLMs) problems. The BLM task -- an RPM/ARC-like task devised specifically for language -- is a controlled linguistic puzzle where models must select the sentence that completes a pattern according to syntactic and semantic rules. We introduce three types of templates varying in complexity and apply linguistically-informed data augmentation strategies across synthetic and natural data. We provide simple baseline performance results across English, Italian, German, and Hebrew, that demonstrate the diagnostic usefulness of the datasets. 

---
# From Documents to Spans: Code-Centric Learning for LLM-based ICD Coding 

**Authors**: Xu Zhang, Wenxin Ma, Chenxu Wu, Rongsheng Wang, Kun Zhang, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.15270)  

**Abstract**: ICD coding is a critical yet challenging task in healthcare. Recently, LLM-based methods demonstrate stronger generalization than discriminative methods in ICD coding. However, fine-tuning LLMs for ICD coding faces three major challenges. First, existing public ICD coding datasets provide limited coverage of the ICD code space, restricting a model's ability to generalize to unseen codes. Second, naive fine-tuning diminishes the interpretability of LLMs, as few public datasets contain explicit supporting evidence for assigned codes. Third, ICD coding typically involves long clinical documents, making fine-tuning LLMs computationally expensive. To address these issues, we propose Code-Centric Learning, a training framework that shifts supervision from full clinical documents to scalable, short evidence spans. The key idea of this framework is that span-level learning improves LLMs' ability to perform document-level ICD coding. Our proposed framework consists of a mixed training strategy and code-centric data expansion, which substantially reduces training cost, improves accuracy on unseen ICD codes and preserves interpretability. Under the same LLM backbone, our method substantially outperforms strong baselines. Notably, our method enables small-scale LLMs to achieve performance comparable to much larger proprietary models, demonstrating its effectiveness and potential for fully automated ICD coding. 

---
# Practicing with Language Models Cultivates Human Empathic Communication 

**Authors**: Aakriti Kumar, Nalin Poungpeth, Diyi Yang, Bruce Lambert, Matthew Groh  

**Link**: [PDF](https://arxiv.org/pdf/2603.15245)  

**Abstract**: Empathy is central to human connection, yet people often struggle to express it effectively. In blinded evaluations, large language models (LLMs) generate responses that are often judged more empathic than human-written ones. Yet when a response is attributed to AI, recipients feel less heard and validated than when comparable responses are attributed to a human. To probe and address this gap in empathic communication skill, we built Lend an Ear, an experimental conversation platform in which participants are asked to offer empathic support to an LLM role-playing personal and workplace troubles. From 33,938 messages spanning 2,904 text-based conversations between 968 participants and their LLM conversational partners, we derive a data-driven taxonomy of idiomatic empathic expressions in naturalistic dialogue. Based on a pre-registered randomized experiment, we present evidence that a brief LLM coaching intervention offering personalized feedback on how to effectively communicate empathy significantly boosts alignment of participants' communication patterns with normative empathic communication patterns relative to both a control group and a group that received video-based but non-personalized feedback. Moreover, we find evidence for a silent empathy effect that people feel empathy but systematically fail to express it. Nonetheless, participants reliably identify responses aligned with normative empathic communication criteria as more expressive of empathy. Together, these results advance the scientific understanding of how empathy is expressed and valued and demonstrate a scalable, AI-based intervention for scaffolding and cultivating it. 

---
# Bidirectional Chinese and English Passive Sentences Dataset for Machine Translation 

**Authors**: Xinyue Ma, Pol Pastells, Mireia Farrús, Mariona Taulé  

**Link**: [PDF](https://arxiv.org/pdf/2603.15227)  

**Abstract**: Machine Translation (MT) evaluation has gone beyond metrics, towards more specific linguistic phenomena. Regarding English-Chinese language pairs, passive sentences are constructed and distributed differently due to language variation, thus need special attention in MT. This paper proposes a bidirectional multi-domain dataset of passive sentences, extracted from five Chinese-English parallel corpora and annotated automatically with structure labels according to human translation, and a test set with manually verified annotation. The dataset consists of 73,965 parallel sentence pairs (2,358,731 English words, 3,498,229 Chinese characters). We evaluate two state-of-the-art open-source MT systems with our dataset, and four commercial models with the test set. The results show that, unlike humans, models are more influenced by the voice of the source text rather than the general voice usage of the source language, and therefore tend to maintain the passive voice when translating a passive in either direction. However, models demonstrate some knowledge of the low frequency and predominantly negative context of Chinese passives, leading to higher voice consistency with human translators in English-to-Chinese translation than in Chinese-to-English translation. Commercial NMT models scored higher in metric evaluations, but LLMs showed a better ability to use diverse alternative translations. Datasets and annotation script will be shared upon request. 

---
# Efficient Document Parsing via Parallel Token Prediction 

**Authors**: Lei Li, Ze Zhao, Meng Li, Zhongwang Lun, Yi Yuan, Xingjing Lu, Zheng Wei, Jiang Bian, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.15206)  

**Abstract**: Document parsing, as a fundamental yet crucial vision task, is being revolutionized by vision-language models (VLMs). However, the autoregressive (AR) decoding inherent to VLMs creates a significant bottleneck, severely limiting parsing speed. In this paper, we propose Parallel-Token Prediction (PTP), a plugable, model-agnostic and simple-yet-effective method that enables VLMs to generate multiple future tokens in parallel with improved sample efficiency. Specifically, we insert some learnable tokens into the input sequence and design corresponding training objectives to equip the model with parallel decoding capabilities for document parsing. Furthermore, to support effective training, we develop a comprehensive data generation pipeline that efficiently produces large-scale, high-quality document parsing training data for VLMs. Extensive experiments on OmniDocBench and olmOCR-bench demonstrate that our method not only significantly improves decoding speed (1.6x-2.2x) but also reduces model hallucinations and exhibits strong generalization abilities. 

---
# The Hrunting of AI: Where and How to Improve English Dialectal Fairness 

**Authors**: Wei Li, Adrian de Wynter  

**Link**: [PDF](https://arxiv.org/pdf/2603.15187)  

**Abstract**: It is known that large language models (LLMs) underperform in English dialects, and that improving them is difficult due to data scarcity. In this work we investigate how quality and availability impact the feasibility of improving LLMs in this context. For this, we evaluate three rarely-studied English dialects (Yorkshire, Geordie, and Cornish), plus African-American Vernacular English, and West Frisian as control. We find that human-human agreement when determining LLM generation quality directly impacts LLM-as-a-judge performance. That is, LLM-human agreement mimics the human-human agreement pattern, and so do metrics such as accuracy. It is an issue because LLM-human agreement measures an LLM's alignment with the human consensus; and hence raises questions about the feasibility of improving LLM performance in locales where low populations induce low agreement. We also note that fine-tuning does not eradicate, and might amplify, this pattern in English dialects. But also find encouraging signals, such as some LLMs' ability to generate high-quality data, thus enabling scalability. We argue that data must be carefully evaluated to ensure fair and inclusive LLM improvement; and, in the presence of scarcity, new tools are needed to handle the pattern found. 

---
# HindSight: Evaluating Research Idea Generation via Future Impact 

**Authors**: Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15164)  

**Abstract**: Evaluating AI-generated research ideas typically relies on LLM judges or human panels -- both subjective and disconnected from actual research impact. We introduce \hs{}, a time-split evaluation framework that measures idea quality by matching generated ideas against real future publications and scoring them by citation impact and venue acceptance. Using a temporal cutoff~$T$, we restrict an idea generation system to pre-$T$ literature, then evaluate its outputs against papers published in the subsequent 30 months. Experiments across 10 AI/ML research topics reveal a striking disconnect: LLM-as-Judge finds no significant difference between retrieval-augmented and vanilla idea generation ($p{=}0.584$), while \hs{} shows the retrieval-augmented system produces 2.5$\times$ higher-scoring ideas ($p{<}0.001$). Moreover, \hs{} scores are \emph{negatively} correlated with LLM-judged novelty ($\rho{=}{-}0.29$, $p{<}0.01$), suggesting that LLMs systematically overvalue novel-sounding ideas that never materialize in real research. 

---
# Indirect Question Answering in English, German and Bavarian: A Challenging Task for High- and Low-Resource Languages Alike 

**Authors**: Miriam Winkler, Verena Blaschke, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2603.15130)  

**Abstract**: Indirectness is a common feature of daily communication, yet is underexplored in NLP research for both low-resource as well as high-resource languages. Indirect Question Answering (IQA) aims at classifying the polarity of indirect answers. In this paper, we present two multilingual corpora for IQA of varying quality that both cover English, Standard German and Bavarian, a German dialect without standard orthography: InQA+, a small high-quality evaluation dataset with hand-annotated labels, and GenIQA, a larger training dataset, that contains artificial data generated by GPT-4o-mini. We find that IQA is a pragmatically hard task that comes with various challenges, based on several experiment variations with multilingual transformer models (mBERT, XLM-R and mDeBERTa). We suggest and employ recommendations to tackle these challenges. Our results reveal low performance, even for English, and severe overfitting. We analyse various factors that influence these results, including label ambiguity, label set and dataset size. We find that the IQA performance is poor in high- (English, German) and low-resource languages (Bavarian) and that it is beneficial to have a large amount of training data. Further, GPT-4o-mini does not possess enough pragmatic understanding to generate high-quality IQA data in any of our tested languages. 

---
# MMKU-Bench: A Multimodal Update Benchmark for Diverse Visual Knowledge 

**Authors**: Baochen Fu, Yuntao Du, Cheng Chang, Baihao Jin, Wenzhi Deng, Muhao Xu, Hongmei Yan, Weiye Song, Yi Wan  

**Link**: [PDF](https://arxiv.org/pdf/2603.15117)  

**Abstract**: As real-world knowledge continues to evolve, the parametric knowledge acquired by multimodal models during pretraining becomes increasingly difficult to remain consistent with real-world knowledge. Existing research on multimodal knowledge updating focuses only on learning previously unknown knowledge, while overlooking the need to update knowledge that the model has already mastered but that later changes; moreover, evaluation is limited to the same modality, lacking a systematic analysis of cross-modal consistency. To address these issues, this paper proposes MMKU-Bench, a comprehensive evaluation benchmark for multimodal knowledge updating, which contains over 25k knowledge instances and more than 49k images, covering two scenarios, updated knowledge and unknown knowledge, thereby enabling comparative analysis of learning across different knowledge types. On this benchmark, we evaluate a variety of representative approaches, including supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), and knowledge editing (KE). Experimental results show that SFT and RLHF are prone to catastrophic forgetting, while KE better preserve general capabilities but exhibit clear limitations in continual updating. Overall, MMKU-Bench provides a reliable and comprehensive evaluation benchmark for multimodal knowledge updating, advancing progress in this field. 

---
# Bridging National and International Legal Data: Two Projects Based on the Japanese Legal Standard XML Schema for Comparative Law Studies 

**Authors**: Makoto Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2603.15094)  

**Abstract**: This paper presents an integrated framework for computational comparative law by connecting two consecutive research projects based on the Japanese Legal Standard (JLS) XML schema. The first project establishes structural interoperability by developing a conversion pipeline from JLS to the Akoma Ntoso (AKN) standard, enabling Japanese statutes to be integrated into international LegalDocML-based legislative databases. Building on this foundation, the second project applies multilingual embedding models and semantic textual similarity techniques to identify corresponding provisions across national legal systems. A prototype system combining multilingual embeddings, FAISS retrieval, and Cross-Encoder reranking generates candidate correspondences and visualizes them as cross-jurisdictional networks for exploratory comparative analysis. 

---
# Writer-R1: Enhancing Generative Writing in LLMs via Memory-augmented Replay Policy Optimization 

**Authors**: Jihao Zhao, Shuaishuai Zu, Zhiyuan Ji, Chunlai Zhou, Biao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2603.15061)  

**Abstract**: As a typical open-ended generation task, creative writing lacks verifiable reference answers, which has long constrained reward modeling and automatic evaluation due to high human annotation costs, evaluative bias, and coarse feedback signals. To address these challenges, this paper first designs a multi-agent collaborative workflow based on Grounded Theory, performing dimensional decomposition and hierarchical induction of the problem to dynamically produce interpretable and reusable fine-grained criteria. Furthermore, we propose the Memory-augmented Replay Policy Optimization (MRPO) algorithm: on the one hand, without additional training, MRPO guides models to engage in self-reflection based on dynamic criteria, enabling controlled iterative improvement; on the other hand, we adopt the training paradigm that combines supervised fine-tuning with reinforcement learning to convert evaluation criteria into reward signals, achieving end-to-end optimization. Experimental results demonstrate that the automatically constructed criteria achieve performance gains comparable to human annotations. Writer-R1-4B models trained with this approach outperform baselines across multiple creative writing tasks and surpass some 100B+ parameter open-source models. 

---
# Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs 

**Authors**: Disha Sheshanarayana, Rajat Subhra Pal, Manjira Sinha, Tirthankar Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2603.15051)  

**Abstract**: Token-level Chain-of-Thought (CoT) prompting has become a standard way to elicit multi-step reasoning in large language models (LLMs), especially for mathematical word problems. However, generating long intermediate traces increases output length and inference cost, and can be inefficient when the model could arrive at the correct answer without extensive verbalization. This has motivated latent-space reasoning approaches that shift computation into hidden representations and only emit a final answer. Yet, many latent reasoning methods depend on a fixed number of latent refinement steps at inference, adding another hyperparameter that must be tuned across models and datasets to balance accuracy and efficiency. We introduce AdaAnchor, a latent reasoning framework that performs silent iterative computation by refining a set of latent anchor vectors attached to the input. AdaAnchor further incorporates an adaptive halting mechanism that monitors anchor stability across iterations and terminates refinement once the anchor dynamics converge, allocating fewer steps to easier instances while reserving additional refinement steps for harder ones under a shared maximum-step budget. Our empirical evaluation across three mathematical word-problem benchmarks shows that AdaAnchor with adaptive halting yields accuracy gains of up to 5% over fixed-step latent refinement while reducing average latent refinement steps by 48-60% under the same maximum-step budget. Compared to standard reasoning baselines, AdaAnchor achieves large reductions in generated tokens (92-93%) by moving computation into silent latent refinement, offering a different accuracy-efficiency trade-off with substantially lower output-token usage. 

---
# Interpretable Predictability-Based AI Text Detection: A Replication Study 

**Authors**: Adam Skurla, Dominik Macko, Jakub Simko  

**Link**: [PDF](https://arxiv.org/pdf/2603.15034)  

**Abstract**: This paper replicates and extends the system used in the AuTexTification 2023 shared task for authorship attribution of machine-generated texts. First, we tried to reproduce the original results. Exact replication was not possible because of differences in data splits, model availability, and implementation details. Next, we tested newer multilingual language models and added 26 document-level stylometric features. We also applied SHAP analysis to examine which features influence the model's decisions. We replaced the original GPT-2 models with newer generative models such as Qwen and mGPT for computing probabilistic features. For contextual representations, we used mDeBERTa-v3-base and applied the same configuration to both English and Spanish. This allowed us to use one shared configuration for Subtask 1 and Subtask 2. Our experiments show that the additional stylometric features improve performance in both tasks and both languages. The multilingual configuration achieves the results that are comparable to or better than language-specific models. The study also shows that clear documentation is important for reliable replication and fair comparison of systems. 

---
# Attention Residuals 

**Authors**: Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, Yutian Chen, Junjie Yan, Ming Wei, Y. Zhang, Fanqing Meng, Chao Hong, Xiaotong Xie, Shaowei Liu, Enzhe Lu, Yunpeng Tai, Yanru Chen, Xin Men, Haiqing Guo, Y. Charles, Haoyu Lu, Lin Sui, Jinguo Zhu, Zaida Zhou, Weiran He, Weixiao Huang, Xinran Xu, Yuzhi Wang, Guokun Lai, Yulun Du, Yuxin Wu, Zhilin Yang, Xinyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.15031)  

**Abstract**: Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer's contribution. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, input-dependent weights. To address the memory and communication overhead of attending over all preceding layer outputs for large-scale model training, we introduce Block AttnRes, which partitions layers into blocks and attends over block-level representations, reducing the memory footprint while preserving most of the gains of full AttnRes. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes becomes a practical drop-in replacement for standard residual connections with minimal overhead.
Scaling law experiments confirm that the improvement is consistent across model sizes, and ablations validate the benefit of content-dependent depth-wise selection. We further integrate AttnRes into the Kimi Linear architecture (48B total / 3B activated parameters) and pre-train on 1.4T tokens, where AttnRes mitigates PreNorm dilution, yielding more uniform output magnitudes and gradient distribution across depth, and improves downstream performance across all evaluated tasks. 

---
# Pretraining and Benchmarking Modern Encoders for Latvian 

**Authors**: Arturs Znotins  

**Link**: [PDF](https://arxiv.org/pdf/2603.15005)  

**Abstract**: Encoder-only transformers remain essential for practical NLP tasks. While recent advances in multilingual models have improved cross-lingual capabilities, low-resource languages such as Latvian remain underrepresented in pretraining corpora, and few monolingual Latvian encoders currently exist. We address this gap by pretraining a suite of Latvian-specific encoders based on RoBERTa, DeBERTaV3, and ModernBERT architectures, including long-context variants, and evaluating them across a diverse set of Latvian diagnostic and linguistic benchmarks. Our models are competitive with existing monolingual and multilingual encoders while benefiting from recent architectural and efficiency advances. Our best model, lv-deberta-base (111M parameters), achieves the strongest overall performance, outperforming larger multilingual baselines and prior Latvian-specific encoders. We release all pretrained models and evaluation resources to support further research and practical applications in Latvian NLP. 

---
# OrgForge: A Multi-Agent Simulation Framework for Verifiable Synthetic Corporate Corpora 

**Authors**: Jeffrey Flynt  

**Link**: [PDF](https://arxiv.org/pdf/2603.14997)  

**Abstract**: Evaluating retrieval-augmented generation (RAG) pipelines requires corpora where ground truth is knowable, temporally structured, and cross-artifact properties that real-world datasets rarely provide cleanly. Existing resources such as the Enron corpus carry legal ambiguity, demographic skew, and no structured ground truth. Purely LLM-generated synthetic data solves the legal problem but introduces a subtler one: the generating model cannot be prevented from hallucinating facts that contradict themselves across this http URL present OrgForge, an open-source multi-agent simulation framework that enforces a strict physics-cognition boundary: a deterministic Python engine maintains a SimEvent ground truth bus; large language models generate only surface prose, constrained by validated proposals. An actor-local clock enforces causal timestamp correctness across all artifact types, eliminating the class of timeline inconsistencies that arise when timestamps are sampled independently per document. We formalize three graph-dynamic subsystems stress propagation via betweenness centrality, temporal edge-weight decay, and Dijkstra escalation routing that govern organizational behavior independently of any LLM. Running a configurable N-day simulation, OrgForge produces interleaved Slack threads, JIRA tickets, Confluence pages, Git pull requests, and emails, all traceable to a shared, immutable event log. We additionally describe a causal chain tracking subsystem that accumulates cross-artifact evidence graphs per incident, a hybrid reciprocal-rank-fusion recurrence detector for identifying repeated failure classes, and an inbound/outbound email engine that routes vendor alerts, customer complaints, and HR correspondence through gated causal chains with probabilistic drop simulation. OrgForge is available under the MIT license. 

---
# Beyond Benchmark Islands: Toward Representative Trustworthiness Evaluation for Agentic AI 

**Authors**: Jinhu Qi, Yifan Li, Minghao Zhao, Wentao Zhang, Zijian Zhang, Yaoman Li, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2603.14987)  

**Abstract**: As agentic AI systems move beyond static question answering into open-ended, tool-augmented, and multi-step real-world workflows, their increased authority poses greater risks of system misuse and operational failures. However, current evaluation practices remain fragmented, measuring isolated capabilities such as coding, hallucination, jailbreak resistance, or tool use in narrowly defined settings. We argue that the central limitation is not merely insufficient coverage of evaluation dimensions, but the lack of a principled notion of representativeness: an agent's trustworthiness should be assessed over a representative socio-technical scenario distribution rather than a collection of disconnected benchmark instances. To this end, we propose the Holographic Agent Assessment Framework (HAAF), a systematic evaluation paradigm that characterizes agent trustworthiness over a scenario manifold spanning task types, tool interfaces, interaction dynamics, social contexts, and risk levels. The framework integrates four complementary components: (i) static cognitive and policy analysis, (ii) interactive sandbox simulation, (iii) social-ethical alignment assessment, and (iv) a distribution-aware representative sampling engine that jointly optimizes coverage and risk sensitivity -- particularly for rare but high-consequence tail risks that conventional benchmarks systematically overlook. These components are connected through an iterative Trustworthy Optimization Factory. Through cycles of red-team probing and blue-team hardening, this paradigm progressively narrows the vulnerabilities to meet deployment standards, shifting agent evaluation from benchmark islands toward representative, real-world trustworthiness. Code and data for the illustrative instantiation are available at this https URL. 

---
# ExPosST: Explicit Positioning with Adaptive Masking for LLM-Based Simultaneous Machine Translation 

**Authors**: Yuzhe Shang, Pengzhi Gao, Yazheng Yang, Jiayao Ma, Wei Liu, Jian Luan, Jingsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.14903)  

**Abstract**: Large language models (LLMs) have recently demonstrated promising performance in simultaneous machine translation (SimulMT). However, applying decoder-only LLMs to SimulMT introduces a positional mismatch, which leads to a dilemma between decoding efficiency and positional consistency. Existing approaches often rely on specific positional encodings or carefully designed prompting schemes, and thus fail to simultaneously achieve inference efficiency, positional consistency, and broad model compatibility. In this work, we propose ExPosST, a general framework that resolves this dilemma through explicit position allocation. ExPosST reserves fixed positional slots for incoming source tokens, enabling efficient decoding with KV cache across different positional encoding methods. To further bridge the gap between fine-tuning and inference, we introduce a policy-consistent fine-tuning strategy that aligns training with inference-time decoding behavior. Experiments across multiple language pairs demonstrate that ExPosST effectively supports simultaneous translation under diverse policies. 

---
# LLMs as Signal Detectors: Sensitivity, Bias, and the Temperature-Criterion Analogy 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2603.14893)  

**Abstract**: Large language models (LLMs) are evaluated for calibration using metrics such as Expected Calibration Error that conflate two distinct components: the model's ability to discriminate correct from incorrect answers (sensitivity) and its tendency toward confident or cautious responding (bias). Signal Detection Theory (SDT) decomposes these components. While SDT-derived metrics such as AUROC are increasingly used, the full parametric framework - unequal-variance model fitting, criterion estimation, z-ROC analysis - has not been applied to LLMs as signal detectors. In this pre-registered study, we treat three LLMs as observers performing factual discrimination across 168,000 trials and test whether temperature functions as a criterion shift analogous to payoff manipulations in human psychophysics. Critically, this analogy may break down because temperature changes the generated answer itself, not only the confidence assigned to it. Our results confirm the breakdown with temperature simultaneously increasing sensitivity (AUC) and shifting criterion. All models exhibited unequal-variance evidence distributions (z-ROC slopes 0.52-0.84), with instruct models showing more extreme asymmetry (0.52-0.63) than the base model (0.77-0.87) or human recognition memory (~0.80). The SDT decomposition revealed that models occupying distinct positions in sensitivity-bias space could not be distinguished by calibration metrics alone, demonstrating that the full parametric framework provides diagnostic information unavailable from existing metrics. 

---
# Decision-Level Ordinal Modeling for Multimodal Essay Scoring with Large Language Models 

**Authors**: Han Zhang, Jiamin Su, Li liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.14891)  

**Abstract**: Automated essay scoring (AES) predicts multiple rubric-defined trait scores for each essay, where each trait follows an ordered discrete rating scale. Most LLM-based AES methods cast scoring as autoregressive token generation and obtain the final score via decoding and parsing, making the decision implicit. This formulation is particularly sensitive in multimodal AES, where the usefulness of visual inputs varies across essays and traits. To address these limitations, we propose Decision-Level Ordinal Modeling (DLOM), which makes scoring an explicit ordinal decision by reusing the language model head to extract score-wise logits on predefined score tokens, enabling direct optimization and analysis in the score space. For multimodal AES, DLOM-GF introduces a gated fusion module that adaptively combines textual and multimodal score logits. For text-only AES, DLOM-DA adds a distance-aware regularization term to better reflect ordinal distances. Experiments on the multimodal EssayJudge dataset show that DLOM improves over a generation-based SFT baseline across scoring traits, and DLOM-GF yields further gains when modality relevance is heterogeneous. On the text-only ASAP/ASAP++ benchmarks, DLOM remains effective without visual inputs, and DLOM-DA further improves performance and outperforms strong representative baselines. 

---
# Developing an English-Efik Corpus and Machine Translation System for Digitization Inclusion 

**Authors**: Offiong Bassey Edet, Mbuotidem Sunday Awak, Emmanuel Oyo-Ita, Benjamin Okon Nyong, Ita Etim Bassey  

**Link**: [PDF](https://arxiv.org/pdf/2603.14873)  

**Abstract**: Low-resource languages serve as invaluable repositories of human history, preserving cultural and intellectual diversity. Despite their significance, they remain largely absent from modern natural language processing systems. While progress has been made for widely spoken African languages such as Swahili, Yoruba, and Amharic, smaller indigenous languages like Efik continue to be underrepresented in machine translation research. This study evaluates the effectiveness of state-of-the-art multilingual neural machine translation models for English-Efik translation, leveraging a small-scale, community-curated parallel corpus of 13,865 sentence pairs. We fine-tuned both the mT5 multilingual model and the NLLB200 model on this dataset. NLLB-200 outperformed mT5, achieving BLEU scores of 26.64 for English-Efik and 31.21 for Efik-English, with corresponding chrF scores of 51.04 and 47.92, indicating improved fluency and semantic fidelity. Our findings demonstrate the feasibility of developing practical machine translation tools for low-resource languages and highlight the importance of inclusive data practices and culturally grounded evaluation in advancing equitable NLP. 

---
# Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks 

**Authors**: Zijian Yu, Kejun Xiao, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2603.14864)  

**Abstract**: In e-commerce, LLM agents show promise for shopping tasks such as recommendations, budgeting, and bundle deals, where accurately capturing user preferences from long-term conversations is critical. However, two challenges hinder realizing this potential: (1) the absence of benchmarks for evaluating long-term preference-aware shopping tasks, and (2) the lack of end-to-end optimization due to existing designs that treat preference identification and shopping assistance as separate components. In this paper, we introduce a novel benchmark with a long-term memory setup, spanning two shopping tasks over 1.2 million real-world products, and propose Shopping Companion, a unified framework that jointly tackles memory retrieval and shopping assistance while supporting user intervention. To train such capabilities, we develop a dual-reward reinforcement learning strategy with tool-wise rewards to handle the sparse and discontinuous rewards inherent in multi-turn interactions. Experimental results demonstrate that even state-of-the-art models (such as GPT-5) achieve success rates under 70% on our benchmark, highlighting the significant challenges in this domain. Notably, our lightweight LLM, trained with Shopping Companion, consistently outperforms strong baselines, achieving better preference capture and task performance, which validates the effectiveness of our unified design. 

---
# ContiGuard: A Framework for Continual Toxicity Detection Against Evolving Evasive Perturbations 

**Authors**: Hankun Kang, Xin Miao, Jianhao Chen, Jintao Wen, Mayi Xu, Weiyu Zhang, Wenpeng Lu, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2603.14843)  

**Abstract**: Toxicity detection mitigates the dissemination of toxic content (e.g., hateful comments, posts, and messages within online social actions) to safeguard a healthy online social environment. However, malicious users persistently develop evasive perturbations to disguise toxic content and evade detectors. Traditional detectors or methods are static over time and are inadequate in addressing these evolving evasion tactics. Thus, continual learning emerges as a logical approach to dynamically update detection ability against evolving perturbations. Nevertheless, disparities across perturbations hinder the detector's continual learning on perturbed text. More importantly, perturbation-induced noises distort semantics to degrade comprehension and also impair critical feature learning to render detection sensitive to perturbations. These amplify the challenge of continual learning against evolving perturbations. In this work, we present ContiGuard, the first framework tailored for continual learning of the detector on time-evolving perturbed text (termed continual toxicity detection) to enable the detector to continually update capability and maintain sustained resilience against evolving perturbations. Specifically, to boost the comprehension, we present an LLM-powered semantic enriching strategy, where we dynamically incorporate possible meaning and toxicity-related clues excavated by LLM into the perturbed text to improve the comprehension. To mitigate non-critical features and amplify critical ones, we propose a discriminability-driven feature learning strategy, where we strengthen discriminative features while suppressing the less-discriminative ones to shape a robust classification boundary for detection... 

---
# The Impact of Ideological Discourses in RAG: A Case Study with COVID-19 Treatments 

**Authors**: Elmira Salari, Maria Claudia Nunes Delfino, Hazem Amamou, José Victor de Souza, Shruti Kshirsagar, Alan Davoust, Anderson Avila  

**Link**: [PDF](https://arxiv.org/pdf/2603.14838)  

**Abstract**: This paper studies the impact of retrieved ideological texts on the outputs of large language models (LLMs). While interest in understanding ideology in LLMs has recently increased, little attention has been given to this issue in the context of Retrieval-Augmented Generation (RAG). To fill this gap, we design an external knowledge source based on ideological loaded texts about COVID-19 treatments. Our corpus is based on 1,117 academic articles representing discourses about controversial and endorsed treatments for the disease. We propose a corpus linguistics framework, based on Lexical Multidimensional Analysis (LMDA), to identify the ideologies within the corpus. LLMs are tasked to answer questions derived from three identified ideological dimensions, and two types of contextual prompts are adopted: the first comprises the user question and ideological texts; and the second contains the question, ideological texts, and LMDA descriptions. Ideological alignment between reference ideological texts and LLMs' responses is assessed using cosine similarity for lexical and semantic representations. Results demonstrate that LLMs' responses based on ideological retrieved texts are more aligned with the ideology encountered in the external knowledge, with the enhanced prompt further influencing LLMs' outputs. Our findings highlight the importance of identifying ideological discourses within the RAG framework in order to mitigate not just unintended ideological bias, but also the risks of malicious manipulation of such models. 

---
# Information Asymmetry across Language Varieties: A Case Study on Cantonese-Mandarin and Bavarian-German QA 

**Authors**: Renhao Pei, Siyao Peng, Verena Blaschke, Robert Litschko, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2603.14782)  

**Abstract**: Large Language Models (LLMs) are becoming a common way for humans to seek knowledge, yet their coverage and reliability vary widely. Especially for local language varieties, there are large asymmetries, e.g., information in local Wikipedia that is absent from the standard variant. However, little is known about how well LLMs perform under such information asymmetry, especially on closely related languages. We manually construct a novel challenge question-answering (QA) dataset that captures knowledge conveyed on a local Wikipedia page, which is absent from their higher-resource counterparts-covering Mandarin Chinese vs. Cantonese and German vs. Bavarian. Our experiments show that LLMs fail to answer questions about information only in local editions of Wikipedia. Providing context from lead sections substantially improves performance, with further gains possible via translation. Our topical, geographic annotations, and stratified evaluations reveal the usefulness of local Wikipedia editions as sources of both regional and global information. These findings raise critical questions about inclusivity and cultural coverage of LLMs. 

---
# Vietnamese Automatic Speech Recognition: A Revisit 

**Authors**: Thi Vu, Linh The Nguyen, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.14779)  

**Abstract**: Automatic Speech Recognition (ASR) performance is heavily dependent on the availability of large-scale, high-quality datasets. For low-resource languages, existing open-source ASR datasets often suffer from insufficient quality and inconsistent annotation, hindering the development of robust models. To address these challenges, we propose a novel and generalizable data aggregation and preprocessing pipeline designed to construct high-quality ASR datasets from diverse, potentially noisy, open-source sources. Our pipeline incorporates rigorous processing steps to ensure data diversity, balance, and the inclusion of crucial features like word-level timestamps. We demonstrate the effectiveness of our methodology by applying it to Vietnamese, resulting in a unified, high-quality 500-hour dataset that provides a foundation for training and evaluating state-of-the-art Vietnamese ASR systems. Our project page is available at this https URL. 

---
# Towards Privacy-Preserving Machine Translation at the Inference Stage: A New Task and Benchmark 

**Authors**: Wei Shao, Lemao Liu, Yinqiao Li, Guoping Huang, Shuming Shi, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.14756)  

**Abstract**: Current online translation services require sending user text to cloud servers, posing a risk of privacy leakage when the text contains sensitive information. This risk hinders the application of online translation services in privacy-sensitive scenarios. One way to mitigate this risk for online translation services is introducing privacy protection mechanisms targeting the inference stage of translation models. However, compared to subfields of NLP like text classification and summarization, the machine translation research community has limited exploration of privacy protection during the inference stage. There is no clearly defined privacy protection task for the inference stage, dedicated evaluation datasets and metrics, and reference benchmark methods. The absence of these elements has seriously constrained researchers' in-depth exploration of this direction. To bridge this gap, this paper proposes a novel "Privacy-Preserving Machine Translation" (PPMT) task, aiming to protect the private information in text during the model inference stage. For this task, we constructed three benchmark test datasets, designed corresponding evaluation metrics, and proposed a series of benchmark methods as a starting point for this task. The definition of privacy is complex and diverse. Considering that named entities often contain a large amount of personal privacy and commercial secrets, we have focused our research on protecting only the named entity's privacy in the text. We expect this research work will provide a new perspective and a solid foundation for the privacy protection problem in machine translation. 

---
# Learning Constituent Headedness 

**Authors**: Zeyao Qi, Yige Chen, KyungTae Lim, Haihua Pan, Jungyeul Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.14755)  

**Abstract**: Headedness is widely used as an organizing device in syntactic analysis, yet constituency treebanks rarely encode it explicitly and most processing pipelines recover it procedurally via percolation rules. We treat this notion of constituent headedness as an explicit representational layer and learn it as a supervised prediction task over aligned constituency and dependency annotations, inducing supervision by defining each constituent head as the dependency span head. On aligned English and Chinese data, the resulting models achieve near-ceiling intrinsic accuracy and substantially outperform Collins-style rule-based percolation. Predicted heads yield comparable parsing accuracy under head-driven binarization, consistent with the induced binary training targets being largely equivalent across head choices, while increasing the fidelity of deterministic constituency-to-dependency conversion and transferring across resources and languages under simple label-mapping interfaces. 

---
# Beyond Creed: A Non-Identity Safety Condition A Strong Empirical Alternative to Identity Framing in Low-Data LoRA Fine-Tuning 

**Authors**: Xinran Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14723)  

**Abstract**: How safety supervision is written may matter more than the explicit identity content it contains. We study low-data LoRA safety fine-tuning with four supervision formats built from the same core safety rules: constitutional rules (A), creed-style identity framing (B), a B-matched creed condition with a worldview/confession identity-maintenance tail (C), and a matched non-identity condition (D). Across three instruction-tuned model families (Llama 3.1 8B, Qwen2.5 7B, and Gemma 3 4B), we evaluate HarmBench using a reconciled dual-judge pipeline combining Bedrock-hosted DeepSeek v3.2 and Sonnet 4.6, with disagreement and boundary cases manually resolved.
The non-identity condition D is the strongest group on all three model families on the full 320-behavior HarmBench set, reaching 74.4% refusal on Llama, 76.9% on Gemma, and 74.1% on Qwen. By comparison, creed-style framing (B) improves over plain constitutional rules (A) on Llama and Gemma, but remains substantially below D, yielding an overall descriptive ordering of $D > B > C \geq A > baseline$. This provides a bounded empirical challenge to a strong version of the identity-framing hypothesis: explicit creed-style identity language is not necessary for the strongest gains observed here. Capability evaluations on MMLU and ARC-Challenge show no meaningful trade-off across conditions. 

---
# Towards Next-Generation LLM Training: From the Data-Centric Perspective 

**Authors**: Hao Liang, Zhengyang Zhao, Zhaoyang Han, Meiyi Qiang, Xiaochen Ma, Bohan Zeng, Qifeng Cai, Zhiyu Li, Linpeng Tang, Weinan E, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14712)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks and domains, with data playing a central role in enabling these advances. Despite this success, the preparation and effective utilization of the massive datasets required for LLM training remain major bottlenecks. In current practice, LLM training data is often constructed using ad hoc scripts, and there is still a lack of mature, agent-based data preparation systems that can automatically construct robust and reusable data workflows, thereby freeing data scientists from repetitive and error-prone engineering efforts. Moreover, once collected, datasets are often consumed largely in their entirety during training, without systematic mechanisms for data selection, mixture optimization, or reweighting. To address these limitations, we advocate two complementary research directions. First, we propose building a robust, agent-based automatic data preparation system that supports automated workflow construction and scalable data management. Second, we argue for a unified data-model interaction training system in which data is dynamically selected, mixed, and reweighted throughout the training process, enabling more efficient, adaptive, and performance-aware data utilization. Finally, we discuss the remaining challenges and outline promising directions for future research and system development. 

---
# Computational Analysis of Semantic Connections Between Herman Melville Reading and Writing 

**Authors**: Nudrat Habib, Elisa Barney Smith, Steven Olsen Smith  

**Link**: [PDF](https://arxiv.org/pdf/2603.14674)  

**Abstract**: This study investigates the potential influence of Herman Melville reading on his own writings through computational semantic similarity analysis. Using documented records of books known to have been owned or read by Melville, we compare selected passages from his works with texts from his library. The methodology involves segmenting texts at both sentence level and non-overlapping 5-gram level, followed by similarity computation using BERTScore. Rather than applying fixed thresholds to determine reuse, we interpret precision, recall, and F1 scores as indicators of possible semantic alignment that may suggest literary influence. Experimental results demonstrate that the approach successfully captures expert-identified instances of similarity and highlights additional passages warranting further qualitative examination. The findings suggest that semantic similarity methods provide a useful computational framework for supporting source and influence studies in literary scholarship. 

---
# Seamless Deception: Larger Language Models Are Better Knowledge Concealers 

**Authors**: Dhananjay Ashok, Ruth-Ann Armstrong, Jonathan May  

**Link**: [PDF](https://arxiv.org/pdf/2603.14672)  

**Abstract**: Language Models (LMs) may acquire harmful knowledge, and yet feign ignorance of these topics when under audit. Inspired by the recent discovery of deception-related behaviour patterns in LMs, we aim to train classifiers that detect when a LM is actively concealing knowledge. Initial findings on smaller models show that classifiers can detect concealment more reliably than human evaluators, with gradient-based concealment proving easier to identify than prompt-based methods. However, contrary to prior work, we find that the classifiers do not reliably generalize to unseen model architectures and topics of hidden knowledge. Most concerningly, the identifiable traces associated with concealment become fainter as the models increase in scale, with the classifiers achieving no better than random performance on any model exceeding 70 billion parameters. Our results expose a key limitation in black-box-only auditing of LMs and highlight the need to develop robust methods to detect models that are actively hiding the knowledge they contain. 

---
# $PA^3$: $\textbf{P}$olicy-$\textbf{A}$ware $\textbf{A}$gent $\textbf{A}$lignment through Chain-of-Thought 

**Authors**: Shubhashis Roy Dipta, Daniel Bis, Kun Zhou, Lichao Wang, Benjamin Z. Yao, Chenlei Guo, Ruhi Sarikaya  

**Link**: [PDF](https://arxiv.org/pdf/2603.14602)  

**Abstract**: Conversational assistants powered by large language models (LLMs) excel at tool-use tasks but struggle with adhering to complex, business-specific rules. While models can reason over business rules provided in context, including all policies for every query introduces high latency and wastes compute. Furthermore, these lengthy prompts lead to long contexts, harming overall performance due to the "needle-in-the-haystack" problem. To address these challenges, we propose a multi-stage alignment method that teaches models to recall and apply relevant business policies during chain-of-thought reasoning at inference time, without including the full business policy in-context. Furthermore, we introduce a novel PolicyRecall reward based on the Jaccard score and a Hallucination Penalty for GRPO training. Altogether, our best model outperforms the baseline by 16 points and surpasses comparable in-context baselines of similar model size by 3 points, while using 40% fewer words. 

---
# Parameter-Efficient Quality Estimation via Frozen Recursive Models 

**Authors**: Umar Abubacar, Roman Bauer, Diptesh Kanojia  

**Link**: [PDF](https://arxiv.org/pdf/2603.14593)  

**Abstract**: Tiny Recursive Models (TRM) achieve strong results on reasoning tasks through iterative refinement of a shared network. We investigate whether these recursive mechanisms transfer to Quality Estimation (QE) for low-resource languages using a three-phase methodology. Experiments on $8$ language pairs on a low-resource QE dataset reveal three findings. First, TRM's recursive mechanisms do not transfer to QE. External iteration hurts performance, and internal recursion offers only narrow benefits. Next, representation quality dominates architectural choices, and lastly, frozen pretrained embeddings match fine-tuned performance while reducing trainable parameters by 37$\times$ (7M vs 262M). TRM-QE with frozen XLM-R embeddings achieves a Spearman's correlation of 0.370, matching fine-tuned variants (0.369) and outperforming an equivalent-depth standard transformer (0.336). On Hindi and Tamil, frozen TRM-QE outperforms MonoTransQuest (560M parameters) with 80$\times$ fewer trainable parameters, suggesting that weight sharing combined with frozen embeddings enables parameter efficiency for QE. We release the code publicly for further research. Code is available at this https URL. 

---
# Top-b: Entropic Regulation of Relative Probability Bands in Autoregressive Language Processes 

**Authors**: Deepon Halder, Raj Dabre  

**Link**: [PDF](https://arxiv.org/pdf/2603.14567)  

**Abstract**: Probabilistic language generators are theoretically modeled as discrete stochastic processes, yet standard decoding strategies (Top-k, Top-p) impose static truncation rules that fail to accommodate the dynamic information density of natural language. This misalignment often forces a suboptimal trade-off: static bounds are either too restrictive for high-entropy creative generation or too permissive for low-entropy logical reasoning. In this work, we formalize the generation process as a trajectory through a relative probability manifold. We introduce Top-b (Adaptive Relative Band Sampling), a decoding strategy that regulates the candidate set via a dynamic bandwidth coefficient coupled strictly to the instantaneous Shannon entropy of the model's distribution. We provide a theoretical framework demonstrating that Top-b acts as a variance-minimizing operator on the tail distribution. Empirical validation on GPQA and GSM8K benchmarks indicates that Top-b significantly reduces generation entropy and inter-decoding variance while maintaining competitive reasoning accuracy, effectively approximating a self-regulating control system for autoregressive generation. 

---
# Multilingual TinyStories: A Synthetic Combinatorial Corpus of Indic Children's Stories for Training Small Language Models 

**Authors**: Deepon Halder, Angira Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2603.14563)  

**Abstract**: The development of robust language models for low-resource languages is frequently bottlenecked by the scarcity of high-quality, coherent, and domain-appropriate training corpora. In this paper, we introduce the Multilingual TinyStories dataset, a large-scale, synthetically generated collection of children's stories encompassing 17 Indian languages. Designed specifically for the training and evaluation of Small Language Models (SLMs), the corpus provides simple, narrative-driven text strictly localized to native scripts. We detail our hybrid curation pipeline, which leverages the Sarvam-M language model and a novel combinatorial prompt engineering framework for native generation, coupled with the Google Translate API for large-scale cross-lingual expansion. Through strict programmatic filtering, we compiled 132,942 stories and over 93.9 million tokens in our release, serving as a foundational resource for multilingual language modeling and transfer learning in the Indic linguistic sphere. 

---
# MALicious INTent Dataset and Inoculating LLMs for Enhanced Disinformation Detection 

**Authors**: Arkadiusz Modzelewski, Witold Sosnowski, Eleni Papadopulos, Elisa Sartori, Tiziano Labruna, Giovanni Da San Martino, Adam Wierzbicki  

**Link**: [PDF](https://arxiv.org/pdf/2603.14525)  

**Abstract**: The intentional creation and spread of disinformation poses a significant threat to public discourse. However, existing English datasets and research rarely address the intentionality behind the disinformation. This work presents MALINT, the first human-annotated English corpus developed in collaboration with expert fact-checkers to capture disinformation and its malicious intent. We utilize our novel corpus to benchmark 12 language models, including small language models (SLMs) such as BERT and large language models (LLMs) like Llama 3.3, on binary and multilabel intent classification tasks. Moreover, inspired by inoculation theory from psychology and communication studies, we investigate whether incorporating knowledge of malicious intent can improve disinformation detection. To this end, we propose intent-based inoculation, an intent-augmented reasoning for LLMs that integrates intent analysis to mitigate the persuasive impact of disinformation. Analysis on six disinformation datasets, five LLMs, and seven languages shows that intent-augmented reasoning improves zero-shot disinformation detection. To support research in intent-aware disinformation detection, we release the MALINT dataset with annotations from each annotation step. 

---
# Infinite Problem Generator: Verifiably Scaling Physics Reasoning Data with Agentic Workflows 

**Authors**: Aditya Sharan, Sriram Hebbale, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2603.14486)  

**Abstract**: Training large language models for complex reasoning is bottlenecked by the scarcity of verifiable, high-quality data. In domains like physics, standard text augmentation often introduces hallucinations, while static benchmarks lack the reasoning traces required for fine-tuning. We introduce the Infinite Problem Generator (IPG), an agentic framework that synthesizes physics problems with guaranteed solvability through a Formula-as-Code paradigm. Unlike probabilistic text generation, IPG constructs solutions as executable Python programs, enforcing strict mathematical consistency. As a proof-of-concept, we release ClassicalMechanicsV1, a high-fidelity corpus of 1,335 classical mechanics problems expanded from 165 expert seeds. The corpus demonstrates high structural diversity, spanning 102 unique physical formulas with an average complexity of 3.05 formulas per problem. Furthermore, we identify a Complexity Blueprint, demonstrating a strong linear correlation ($R^2 \approx 0.95$) between formula count and verification code length. This relationship establishes code complexity as a precise, proxy-free metric for problem difficulty, enabling controllable curriculum generation. We release the full IPG pipeline, the ClassicalMechanicsV1 dataset, and our evaluation report to support reproducible research in reasoning-intensive domains. 

---
# AI Can Learn Scientific Taste 

**Authors**: Jingqi Tong, Mingzhe Li, Hangcheng Li, Yongzhuo Yang, Yurong Mou, Weijie Ma, Zhiheng Xi, Hongji Chen, Xiaoran Liu, Qinyuan Cheng, Ming Zhang, Qiguang Chen, Weifeng Ge, Qipeng Guo, Tianlei Ying, Tianxiang Sun, Yining Zheng, Xinchi Chen, Jun Zhao, Ning Ding, Xuanjing Huang, Yugang Jiang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.14473)  

**Abstract**: Great scientists have strong judgement and foresight, closely tied to what we call scientific taste. Here, we use the term to refer to the capacity to judge and propose research ideas with high potential impact. However, most relative research focuses on improving an AI scientist's executive capability, while enhancing an AI's scientific taste remains underexplored. In this work, we propose Reinforcement Learning from Community Feedback (RLCF), a training paradigm that uses large-scale community signals as supervision, and formulate scientific taste learning as a preference modeling and alignment problem. For preference modeling, we train Scientific Judge on 700K field- and time-matched pairs of high- vs. low-citation papers to judge ideas. For preference alignment, using Scientific Judge as a reward model, we train a policy model, Scientific Thinker, to propose research ideas with high potential impact. Experiments show Scientific Judge outperforms SOTA LLMs (e.g., GPT-5.2, Gemini 3 Pro) and generalizes to future-year test, unseen fields, and peer-review preference. Furthermore, Scientific Thinker proposes research ideas with higher potential impact than baselines. Our findings show that AI can learn scientific taste, marking a key step toward reaching human-level AI scientists. 

---
# An Industrial-Scale Insurance LLM Achieving Verifiable Domain Mastery and Hallucination Control without Competence Trade-offs 

**Authors**: Qian Zhu, Xinnan Guo, Jingjing Huo, Jun Li, Pan Liu, Wenyan Yang, Wanqing Xu, Xuan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.14463)  

**Abstract**: Adapting Large Language Models (LLMs) to high-stakes vertical domains like insurance presents a significant challenge: scenarios demand strict adherence to complex regulations and business logic with zero tolerance for hallucinations. Existing approaches often suffer from a Competency Trade-off - sacrificing general intelligence for domain expertise - or rely heavily on RAG without intrinsic reasoning. To bridge this gap, we present INS-S1, an insurance-specific LLM family trained via a novel end-to-end alignment paradigm. Our approach features two methodological innovations: (1) A Verifiable Data Synthesis System that constructs hierarchical datasets for actuarial reasoning and compliance; and (2) A Progressive SFT-RL Curriculum Framework that integrates dynamic data annealing with a synergistic mix of Verified Reasoning (RLVR) and AI Feedback (RLAIF). By optimizing data ratios and reward signals, this framework enforces domain constraints while preventing catastrophic forgetting. Additionally, we release INSEva, the most comprehensive insurance benchmark to date (39k+ samples). Extensive experiments show that INS-S1 achieves SOTA performance on domain tasks, significantly outperforming DeepSeek-R1 and Gemini-2.5-Pro. Crucially, it maintains top-tier general capabilities and achieves a record-low 0.6% hallucination rate (HHEM). Our results demonstrate that rigorous domain specialization can be achieved without compromising general intelligence. 

---
# Distilling Reasoning Without Knowledge: A Framework for Reliable LLMs 

**Authors**: Auksarapak Kietkajornrit, Jad Tarifi, Nima Asgharbeygi  

**Link**: [PDF](https://arxiv.org/pdf/2603.14458)  

**Abstract**: Fact-seeking question answering with large language models (LLMs) remains unreliable when answers depend on up-to-date or conflicting information. Although retrieval-augmented and tool-using LLMs reduce hallucinations, they often rely on implicit planning, leading to inefficient tool usage. We propose a modular framework that explicitly separates planning from factual retrieval and answer synthesis. A lightweight student planner is trained via a teacher-student framework to generate structured decompositions consisting of abstract reasoning steps and searchable fact requests. The supervision signals contain only planning traces and fact requests, without providing factual answers or retrieved evidence. At inference, the planner produces plans, while prompt-engineered modules perform retrieval and response synthesis. We evaluate the proposed framework on SEAL-0, an extremely challenging benchmark for search-augmented LLMs. Results show that supervised planning improves both accuracy and latency compared to monolithic reasoning models and prompt-based tool-augmented frameworks, demonstrating that explicitly learned planning structures are essential for reliable fact-seeking LLMs. 

---
# PARSA-Bench: A Comprehensive Persian Audio-Language Model Benchmark 

**Authors**: Mohammad Javad Ranjbar Kalahroodi, Mohammad Amini, Parmis Bathayan, Heshaam Faili, Azadeh Shakery  

**Link**: [PDF](https://arxiv.org/pdf/2603.14456)  

**Abstract**: Persian poses unique audio understanding challenges through its classical poetry, traditional music, and pervasive code-switching - none captured by existing benchmarks. We introduce PARSA-Bench (Persian Audio Reasoning and Speech Assessment Benchmark), the first benchmark for evaluating large audio-language models on Persian language and culture, comprising 16 tasks and over 8,000 samples across speech understanding, paralinguistic analysis, and cultural audio understanding. Ten tasks are newly introduced, including poetry meter and style detection, traditional Persian music understanding, and code-switching detection. Text-only baselines consistently outperform audio counterparts, suggesting models may not leverage audio-specific information beyond what transcription alone provides. Culturally-grounded tasks expose a qualitatively distinct failure mode: all models perform near random chance on vazn detection regardless of scale, suggesting prosodic perception remains beyond the reach of current models. The dataset is publicly available at this https URL 

---
# Echoes Across Centuries: Phonetic Signatures of Persian Poets 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh, Mohammadali Keshtparvar  

**Link**: [PDF](https://arxiv.org/pdf/2603.14443)  

**Abstract**: This study examines phonetic texture in Persian poetry as a literary-historical phenomenon rather than a by-product of meter or a feature used only for classification. The analysis draws on a large corpus of 1,116,306 mesras from 31,988 poems written by 83 poets, restricted to five major classical meters to enable controlled comparison. Each line is converted into a grapheme-to-phoneme representation and analyzed using six phonetic metrics: hardness, sonority, sibilance, vowel ratio, phoneme entropy, and consonant-cluster ratio.
Statistical models estimate poet-level differences while controlling for meter, poetic form, and line length. The results show that although meter and form explain a substantial portion of phonetic variation, they do not eliminate systematic differences between poets. Persian poetic sound therefore appears as conditioned variation within shared prosodic structures rather than as either purely individual style or simple metrical residue.
A multidimensional stylistic map reveals several recurrent phonetic profiles, including high-sonority lyric styles, hardness-driven rhetorical or epic styles, sibilant mystical contours, and high-entropy complex textures. Historical analysis indicates that phonetic distributions shift across centuries, reflecting changes in genre prominence, literary institutions, and performance contexts rather than abrupt stylistic breaks.
The study establishes a corpus-scale framework for phonetic analysis in Persian poetry and demonstrates how computational phonetics can contribute to literary-historical interpretation while remaining attentive to the formal structures that shape Persian verse. 

---
# Creative Convergence or Imitation? Genre-Specific Homogeneity in LLM-Generated Chinese Literature 

**Authors**: Yuanchi Ma, Kaize Shi, Hui He, Zhihua Zhang, Zhongxiang Lei, Ziliang Qiu, Renfen Hu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.14430)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in narrative generation. However, they often produce structurally homogenized stories, frequently following repetitive arrangements and combinations of plot events along with stereotypical resolutions. In this paper, we propose a novel theoretical framework for analysis by incorporating Proppian narratology and narrative functions. This framework is used to analyze the composition of narrative texts generated by LLMs to uncover their underlying narrative logic. Taking Chinese web literature as our research focus, we extend Propp's narrative theory, defining 34 narrative functions suited to modern web narrative structures. We further construct a human-annotated corpus to support the analysis of narrative structures within LLM-generated text. Experiments reveal that the primary reasons for the singular narrative logic and severe homogenization in generated texts are that current LLMs are unable to correctly comprehend the meanings of narrative functions and instead adhere to rigid narrative generation paradigms. 

---
# BiT-MCTS: A Theme-based Bidirectional MCTS Approach to Chinese Fiction Generation 

**Authors**: Zhaoyi Li, Xu Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2603.14410)  

**Abstract**: Generating long-form linear fiction from open-ended themes remains a major challenge for large language models, which frequently fail to guarantee global structure and narrative diversity when using premise-based or linear outlining approaches. We present BiT-MCTS, a theme-driven framework that operationalizes a "climax-first, bidirectional expansion" strategy motivated by Freytag's Pyramid. Given a theme, our method extracts a core dramatic conflict and generates an explicit climax, then employs a bidirectional Monte Carlo Tree Search (MCTS) to expand the plot backward (rising action, exposition) and forward (falling action, resolution) to produce a structured outline. A final generation stage realizes a complete narrative from the refined outline. We construct a Chinese theme corpus for evaluation and conduct extensive experiments across three contemporary LLM backbones. Results show that BiT-MCTS improves narrative coherence, plot structure, and thematic depth relative to strong baselines, while enabling substantially longer, more coherent stories according to automatic metrics and human judgments. 

---
# Extending Minimal Pairs with Ordinal Surprisal Curves and Entropy Across Applied Domains 

**Authors**: Andrew Katz  

**Link**: [PDF](https://arxiv.org/pdf/2603.14400)  

**Abstract**: The minimal pairs paradigm of comparing model probabilities for contrasting completions has proven useful for evaluating linguistic knowledge in language models, yet its application has largely been confined to binary grammaticality judgments over syntactic phenomena. Additionally, standard prompting-based evaluation requires expensive text generation, may elicit post-hoc rationalizations rather than model judgments, and discards information about model uncertainty. We address both limitations by extending surprisal-based evaluation from binary grammaticality contrasts to ordinal-scaled classification and scoring tasks across multiple domains. Rather than asking models to generate answers, we measure the information-theoretic "surprise" (negative log probability) they assign to each position on rating scales (e.g., 1-5 or 1-9), yielding full surprisal curves that reveal both the model's preferred response and its uncertainty via entropy. We explore this framework across four domains: social-ecological-technological systems classification, causal statement identification (binary and scaled), figurative language detection, and deductive qualitative coding. Across these domains, surprisal curves produce interpretable classification signals with clear minima near expected ordinal scale positions, and entropy over the completion tended to distinguish genuinely ambiguous items from easier items. 

---
# Exposing Long-Tail Safety Failures in Large Language Models through Efficient Diverse Response Sampling 

**Authors**: Suvadeep Hajra, Palash Nandi, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2603.14355)  

**Abstract**: Safety tuning through supervised fine-tuning and reinforcement learning from human feedback has substantially improved the robustness of large language models (LLMs). However, it often suppresses rather than eliminates unsafe behaviors, leaving rare but critical failures hidden in the long tail of the output distribution. While most red-teaming work emphasizes adversarial prompt search (input-space optimization), we show that safety failures can also be systematically exposed through diverse response generation (output-space exploration) for a fixed safety-critical prompt, where increasing the number and diversity of sampled responses can drive jailbreak success rates close to unity. To efficiently uncover such failures, we propose Progressive Diverse Population Sampling (PDPS), which combines stochastic token-level sampling with diversity-aware selection to explore a large candidate pool of responses and retain a compact, semantically diverse subset. Across multiple jailbreak benchmarks and open-source LLMs, PDPS achieves attack success rates comparable to large-scale IID sampling while using only 8% to 29% of the computational cost. Under limited-response settings, it improves success rates by 26% to 40% over IID sampling and Diverse Beam Search. Furthermore, responses generated by PDPS exhibit both a higher number and greater diversity of unsafe outputs, demonstrating its effectiveness in uncovering a broader range of failures. 

---
# Motivation in Large Language Models 

**Authors**: Omer Nahum, Asael Sklar, Ariel Goldstein, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2603.14347)  

**Abstract**: Motivation is a central driver of human behavior, shaping decisions, goals, and task performance. As large language models (LLMs) become increasingly aligned with human preferences, we ask whether they exhibit something akin to motivation. We examine whether LLMs "report" varying levels of motivation, how these reports relate to their behavior, and whether external factors can influence them. Our experiments reveal consistent and structured patterns that echo human psychology: self-reported motivation aligns with different behavioral signatures, varies across task types, and can be modulated by external manipulations. These findings demonstrate that motivation is a coherent organizing construct for LLM behavior, systematically linking reports, choices, effort, and performance, and revealing motivational dynamics that resemble those documented in human psychology. This perspective deepens our understanding of model behavior and its connection to human-inspired concepts. 

---
# Mind the Shift: Decoding Monetary Policy Stance from FOMC Statements with Large Language Models 

**Authors**: Yixuan Tang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14313)  

**Abstract**: Federal Open Market Committee (FOMC) statements are a major source of monetary-policy information, and even subtle changes in their wording can move global financial markets. A central task is therefore to measure the hawkish--dovish stance conveyed in these texts. Existing approaches typically treat stance detection as a standard classification problem, labeling each statement in isolation. However, the interpretation of monetary-policy communication is inherently relative: market reactions depend not only on the tone of a statement, but also on how that tone shifts across meetings. We introduce Delta-Consistent Scoring (DCS), an annotation-free framework that maps frozen large language model (LLM) representations to continuous stance scores by jointly modeling absolute stance and relative inter-meeting shifts. Rather than relying on manual hawkish--dovish labels, DCS uses consecutive meetings as a source of self-supervision. It learns an absolute stance score for each statement and a relative shift score between consecutive statements. A delta-consistency objective encourages changes in absolute scores to align with the relative shifts. This allows DCS to recover a temporally coherent stance trajectory without manual labels. Across four LLM backbones, DCS consistently outperforms supervised probes and LLM-as-judge baselines, achieving up to 71.1% accuracy on sentence-level hawkish--dovish classification. The resulting meeting-level scores are also economically meaningful: they correlate strongly with inflation indicators and are significantly associated with Treasury yield movements. Overall, the results suggest that LLM representations encode monetary-policy signals that can be recovered through relative temporal structure. 

---
# SemantiCache: Efficient KV Cache Compression via Semantic Chunking and Clustered Merging 

**Authors**: Shunlong Wu, Hai Lin, Shaoshen Chen, Tingwei Lu, Yongqin Zeng, Shaoxiong Zhan, Hai-Tao Zheng, Hong-Gee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.14303)  

**Abstract**: Existing KV cache compression methods generally operate on discrete tokens or non-semantic chunks. However, such approaches often lead to semantic fragmentation, where linguistically coherent units are disrupted, causing irreversible information loss and degradation in model performance. To address this, we introduce SemantiCache, a novel compression framework that preserves semantic integrity by aligning the compression process with the semantic hierarchical nature of language. Specifically, we first partition the cache into semantically coherent chunks by delimiters, which are natural semantic boundaries. Within each chunk, we introduce a computationally efficient Greedy Seed-Based Clustering (GSC) algorithm to group tokens into semantic clusters. These clusters are further merged into semantic cores, enhanced by a Proportional Attention mechanism that rebalances the reduced attention contributions of the merged tokens. Extensive experiments across diverse benchmarks and models demonstrate that SemantiCache accelerates the decoding stage of inference by up to 2.61 times and substantially reduces memory footprint, while maintaining performance comparable to the original model. 

---
# MedPriv-Bench: Benchmarking the Privacy-Utility Trade-off of Large Language Models in Medical Open-End Question Answering 

**Authors**: Shaowei Guan, Yu Zhai, Hin Chi Kwok, Jiawei Du, Xinyu Feng, Jing Li, Harry Qin, Vivian Hui  

**Link**: [PDF](https://arxiv.org/pdf/2603.14265)  

**Abstract**: Recent advances in Retrieval-Augmented Generation (RAG) have enabled large language models (LLMs) to ground outputs in clinical evidence. However, connecting LLMs with external databases introduces the risk of contextual leakage: a subtle privacy threat where unique combinations of medical details enable patient re-identification even without explicit identifiers. Current benchmarks in healthcare heavily focus on accuracy, ignoring such privacy issues, despite strict regulations like Health Insurance Portability and Accountability Act (HIPAA) and General Data Protection Regulation (GDPR). To fill this gap, we present MedPriv-Bench, the first benchmark specifically designed to jointly evaluate privacy preservation and clinical utility in medical open-ended question answering. Our framework utilizes a multi-agent, human-in-the-loop pipeline to synthesize sensitive medical contexts and clinically relevant queries that create realistic privacy pressure. We establish a standardized evaluation protocol leveraging a pre-trained RoBERTa-Natural Language Inference (NLI) model as an automated judge to quantify data leakage, achieving an average of 85.9% alignment with human experts. Through an extensive evaluation of 9 representative LLMs, we demonstrate a pervasive privacy-utility trade-off. Our findings underscore the necessity of domain-specific benchmarks to validate the safety and efficacy of medical AI systems in privacy-sensitive environments. 

---
# Automatic Inter-document Multi-hop Scientific QA Generation 

**Authors**: Seungmin Lee, Dongha Kim, Yuni Jeon, Junyoung Koh, Min Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.14257)  

**Abstract**: Existing automatic scientific question generation studies mainly focus on single-document factoid QA, overlooking the inter-document reasoning crucial for scientific understanding. We present AIM-SciQA, an automated framework for generating multi-document, multi-hop scientific QA datasets. AIM-SciQA extracts single-hop QAs using large language models (LLMs) with machine reading comprehension and constructs cross-document relations based on embedding-based semantic alignment while selectively leveraging citation information. Applied to 8,211 PubMed Central papers, it produced 411,409 single-hop and 13,672 multi-hop QAs, forming the IM-SciQA dataset. Human and automatic validation confirmed high factual consistency, and experimental results demonstrate that IM-SciQA effectively differentiates reasoning capabilities across retrieval and QA stages, providing a realistic and interpretable benchmark for retrieval-augmented scientific reasoning. We further extend this framework to construct CIM-SciQA, a citation-guided variant achieving comparable performance to the Oracle setting, reinforcing the dataset's validity and generality. 

---
# Mitigating Overthinking in Large Reasoning Language Models via Reasoning Path Deviation Monitoring 

**Authors**: Weixin Guan, Liang Li, Jiapeng Liu, Bing Li, Peng Fu, Chengyang Fang, Xiaoshuai Hao, Can Ma, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14251)  

**Abstract**: Large Reasoning Language Models (LRLMs) demonstrate impressive capabilities on complex tasks by utilizing long Chain-of-Thought reasoning. However, they are prone to overthinking, which generates redundant reasoning steps that degrade both performance and efficiency. Recently, early-exit strategies are proposed to mitigate overthinking by dynamically and adaptively terminating redundant reasoning. However, current early-exit methods either introduce extra training overhead by relying on proxy models or limit inference throughput due to the frequent content switching between reasoning and generating probing answers. Moreover, most early-exit methods harm LRLMs performance due to over-truncation. Our insight stems from an observation: overthinking often causes LRLMs to deviate from the correct reasoning path, which is frequently accompanied by high-entropy transition tokens. Given this, we propose an early-exit method deeply coupled with the native reasoning process, which leverages the path deviation index as a dedicated monitoring metric for the frequent occurrence of high-entropy transition tokens to dynamically detect and terminate overthinking trajectories. We conduct experiments across multiple benchmarks using LRLMs of different types and scales, and the results indicate that our method delivers the largest performance improvement over vanilla CoT compared to existing early-exit methods. 

---
# QiMeng-CodeV-SVA: Training Specialized LLMs for Hardware Assertion Generation via RTL-Grounded Bidirectional Data Synthesis 

**Authors**: Yutong Wu, Chenrui Cao, Pengwei Jin, Di Huang, Rui Zhang, Xishan Zhang, Zidong Du, Qi Guo, Xing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.14239)  

**Abstract**: SystemVerilog Assertions (SVAs) are crucial for hardware verification. Recent studies leverage general-purpose LLMs to translate natural language properties to SVAs (NL2SVA), but they perform poorly due to limited data. We propose a data synthesis framework to tackle two challenges: the scarcity of high-quality real-world SVA corpora and the lack of reliable methods to determine NL-SVA semantic equivalence. For the former, large-scale open-source RTLs are used to guide LLMs to generate real-world SVAs; for the latter, bidirectional translation serves as a data selection method. With the synthesized data, we train CodeV-SVA, a series of SVA generation models. Notably, CodeV-SVA-14B achieves 75.8% on NL2SVA-Human and 84.0% on NL2SVA-Machine in Func.@1, matching or exceeding advanced LLMs like GPT-5 and DeepSeek-R1. 

---
# Rethinking Evaluation in Retrieval-Augmented Personalized Dialogue: A Cognitive and Linguistic Perspective 

**Authors**: Tianyi Zhang, David Traum  

**Link**: [PDF](https://arxiv.org/pdf/2603.14217)  

**Abstract**: In cognitive science and linguistic theory, dialogue is not seen as a chain of independent utterances but rather as a joint activity sustained by coherence, consistency, and shared understanding. However, many systems for open-domain and personalized dialogue use surface-level similarity metrics (e.g., BLEU, ROUGE, F1) as one of their main reporting measures, which fail to capture these deeper aspects of conversational quality. We re-examine a notable retrieval-augmented framework for personalized dialogue, LAPDOG, as a case study for evaluation methodology. Using both human and LLM-based judges, we identify limitations in current evaluation practices, including corrupted dialogue histories, contradictions between retrieved stories and persona, and incoherent response generation. Our results show that human and LLM judgments align closely but diverge from lexical similarity metrics, underscoring the need for cognitively grounded evaluation methods. Broadly, this work charts a path toward more reliable assessment frameworks for retrieval-augmented dialogue systems that better reflect the principles of natural human communication. 

---
# Vavanagi: a Community-run Platform for Documentation of the Hula Language in Papua New Guinea 

**Authors**: Bri Olewale, Raphael Merx, Ekaterina Vylomova  

**Link**: [PDF](https://arxiv.org/pdf/2603.14210)  

**Abstract**: We present Vavanagi, a community-run platform for Hula (Vula'a), an Austronesian language of Papua New Guinea with approximately 10,000 speakers. Vavanagi supports crowdsourced English-Hula text translation and voice recording, with elder-led review and community-governed data infrastructure. To date, 77 translators and 4 reviewers have produced over 12k parallel sentence pairs covering 9k unique Hula words. We also propose a multi-level framework for measuring community involvement, from consultation to fully community-initiated and governed projects. We position Vavanagi at Level 5: initiative, design, implementation, and data governance all sit within the Hula community, making it, to our knowledge, the first community-led language technology initiative for a language of this size. Vavanagi shows how language technology can bridge village-based and urban members, connect generations, and support cultural heritage on the community's own terms. 

---
# Selective Fine-Tuning of GPT Architectures for Parameter-Efficient Clinical Text Classification 

**Authors**: Fariba Afrin Irany, Sampson Akwafuo  

**Link**: [PDF](https://arxiv.org/pdf/2603.14183)  

**Abstract**: The rapid expansion of electronic health record (EHR) systems has generated large volumes of unstructured clinical narratives that contain valuable information for disease identification, patient cohort discovery, and clinical decision support. Extracting structured knowledge from these free-text documents remains challenging because clinical language is highly specialized, labeled datasets are limited, and full fine-tuning of large pretrained language models can require substantial computational resources. Efficient adaptation strategies are therefore essential for practical clinical natural language processing applications. This study proposes a parameter-efficient selective fine-tuning framework for adapting GPT-2 to clinical text classification tasks. Instead of updating the entire pretrained model, the majority of network parameters are frozen, and only the final Transformer block, the final layer normalization module, and a lightweight classification head are updated during training. This design substantially reduces the number of trainable parameters while preserving the contextual representation capabilities learned during pretraining. The proposed approach is evaluated using radiology reports from the MIMIC-IV-Note dataset with automatically derived CheXpert-style labels. Experiments on 50,000 radiology reports demonstrate that selective fine-tuning achieves approximately 91% classification accuracy while updating fewer than 6% of the model parameters. Comparative experiments with head-only training and full-model fine-tuning show that the proposed method provides a favorable balance between predictive performance and computational efficiency. These results indicate that selective fine-tuning offers an efficient and scalable framework for clinical text classification. 

---
# MMOU: A Massive Multi-Task Omni Understanding and Reasoning Benchmark for Long and Complex Real-World Videos 

**Authors**: Arushi Goel, Sreyan Ghosh, Vatsal Agarwal, Nishit Anand, Kaousheik Jayakumar, Lasha Koroshinadze, Yao Xu, Katie Lyons, James Case, Karan Sapra, Kevin J. Shih, Siddharth Gururani, Abhinav Shrivastava, Ramani Duraiswami, Dinesh Manocha, Andrew Tao, Bryan Catanzaro, Mohammad Shoeybi, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2603.14145)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown strong performance in visual and audio understanding when evaluated in isolation. However, their ability to jointly reason over omni-modal (visual, audio, and textual) signals in long and complex videos remains largely unexplored. We introduce MMOU, a new benchmark designed to systematically evaluate multimodal understanding and reasoning under these challenging, real-world conditions. MMOU consists of 15,000 carefully curated questions paired with 9038 web-collected videos of varying length, spanning diverse domains and exhibiting rich, tightly coupled audio-visual content. The benchmark covers 13 fundamental skill categories, all of which require integrating evidence across modalities and time. All questions are manually annotated across multiple turns by professional annotators, ensuring high quality and reasoning fidelity. We evaluate 20+ state-of-the-art open-source and proprietary multimodal models on MMOU. The results expose substantial performance gaps: the best closed-source model achieves only 64.2% accuracy, while the strongest open-source model reaches just 46.8%. Our results highlight the challenges of long-form omni-modal understanding, revealing that current models frequently fail to apply even fundamental skills in long videos. Through detailed analysis, we further identify systematic failure modes and provide insights into where and why current models break. 

---
# The GELATO Dataset for Legislative NER 

**Authors**: Matthew Flynn, Timothy Obiso, Sam Newman  

**Link**: [PDF](https://arxiv.org/pdf/2603.14130)  

**Abstract**: This paper introduces GELATO (Government, Executive, Legislative, and Treaty Ontology), a dataset of U.S. House and Senate bills from the 118th Congress annotated using a novel two-level named entity recognition ontology designed for U.S. legislative texts. We fine-tune transformer-based models (BERT, RoBERTa) of different architectures and sizes on this dataset for first-level prediction. We then use LLMs with optimized prompts to complete the second level prediction. The strong performance of RoBERTa and relatively weak performance of BERT models, as well as the application of LLMs as second-level predictors, support future research in legislative NER or downstream tasks using these model combinations as extraction tools. 

---
# OasisSimp: An Open-source Asian-English Sentence Simplification Dataset 

**Authors**: Hannah Liu, Muxin Tian, Iqra Ali, Haonan Gao, Qiaoyiwen Wu, Blair Yang, Uthayasanker Thayasivam, En-Shiun Annie Lee, Pakawat Nakwijit, Surangika Ranathunga, Ravi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2603.14111)  

**Abstract**: Sentence simplification aims to make complex text more accessible by reducing linguistic complexity while preserving the original meaning. However, progress in this area remains limited for mid-resource and low-resource languages due to the scarcity of high-quality data. To address this gap, we introduce the OasisSimp dataset, a multilingual dataset for sentence-level simplification covering five languages: English, Sinhala, Tamil, Pashto, and Thai. Among these, no prior sentence simplification datasets exist for Thai, Pashto, and Tamil, while limited data is available for Sinhala. Each language simplification dataset was created by trained annotators who followed detailed guidelines to simplify sentences while maintaining meaning, fluency, and grammatical correctness. We evaluate eight open-weight multilingual Large Language Models (LLMs) on the OasisSimp dataset and observe substantial performance disparities between high-resource and low-resource languages, highlighting the simplification challenges in multilingual settings. The OasisSimp dataset thus provides both a valuable multilingual resource and a challenging benchmark, revealing the limitations of current LLM-based simplification methods and paving the way for future research in low-resource sentence simplification. The dataset is available at this https URL. 

---
# CMHL: Contrastive Multi-Head Learning for Emotionally Consistent Text Classification 

**Authors**: Menna Elgabry, Ali Hamdi, Khaled Shaban  

**Link**: [PDF](https://arxiv.org/pdf/2603.14078)  

**Abstract**: Textual Emotion Classification (TEC) is one of the most difficult NLP tasks. State of the art approaches rely on Large language models (LLMs) and multi-model ensembles. In this study, we challenge the assumption that larger scale or more complex models are necessary for improved performance. In order to improve logical consistency, We introduce CMHL, a novel single-model architecture that explicitly models the logical structure of emotions through three key innovations: (1) multi-task learning that jointly predicts primary emotions, valence, and intensity, (2) psychologically-grounded auxiliary supervision derived from Russell's circumplex model, and (3) a novel contrastive contradiction loss that enforces emotional consistency by penalizing mutually incompatible predictions (e.g., simultaneous high confidence in joy and anger). With just 125M parameters, our model outperforms 56x larger LLMs and sLM ensembles with a new state-of-the-art F1 score of 93.75\% compared to (86.13\%-93.2\%) on the dair-ai Emotion dataset. We further show cross domain generalization on the Reddit Suicide Watch and Mental Health Collection dataset (SWMH), outperforming domain-specific models like MentalBERT and MentalRoBERTa with an F1 score of 72.50\% compared to (68.16\%-72.16\%) + a 73.30\% recall compared to (67.05\%-70.89\%) that translates to enhanced sensitivity for detecting mental health distress. Our work establishes that architectural intelligence (not parameter count) drives progress in TEC. By embedding psychological priors and explicit consistency constraints, a well-designed single model can outperform both massive LLMs and complex ensembles, offering a efficient, interpretable, and clinically-relevant paradigm for affective computing. 

---
# NepTam: A Nepali-Tamang Parallel Corpus and Baseline Machine Translation Experiments 

**Authors**: Rupak Raj Ghimire, Bipesh Subedi, Balaram Prasain, Prakash Poudyal, Praveen Acharya, Nischal Karki, Rupak Tiwari, Rishikesh Kumar Sharma, Jenny Poudel, Bal Krishna Bal  

**Link**: [PDF](https://arxiv.org/pdf/2603.14053)  

**Abstract**: Modern Translation Systems heavily rely on high-quality, large parallel datasets for state-of-the-art performance. However, such resources are largely unavailable for most of the South Asian languages. Among them, Nepali and Tamang fall into such category, with Tamang being among the least digitally resourced languages in the region. This work addresses the gap by developing NepTam20K, a 20K gold standard parallel corpus, and NepTam80K, an 80K synthetic Nepali-Tamang parallel corpus, both sentence-aligned and designed to support machine translation. The datasets were created through a pipeline involving data scraping from Nepali news and online sources, pre-processing, semantic filtering, balancing for tense and polarity (in NepTam20K dataset), expert translation into Tamang by native speakers of the language, and verification by an expert Tamang linguist. The dataset covers five domains: Agriculture, Health, Education and Technology, Culture, and General Communication. To evaluate the dataset, baseline machine translation experiments were carried out using various multilingual pre-trained models: mBART, M2M-100, NLLB-200, and a vanilla Transformer model. The fine-tuning on the NLLB-200 achieved the highest sacreBLEU scores of 40.92 (Nepali-Tamang) and 45.26 (Tamang-Nepali). 

---
# SemEval-2026 Task 6: CLARITY -- Unmasking Political Question Evasions 

**Authors**: Konstantinos Thomas, Giorgos Filandrianos, Maria Lymperaiou, Chrysoula Zerva, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2603.14027)  

**Abstract**: Political speakers often avoid answering questions directly while maintaining the appearance of responsiveness. Despite its importance for public discourse, such strategic evasion remains underexplored in Natural Language Processing. We introduce SemEval-2026 Task 6, CLARITY, a shared task on political question evasion consisting of two subtasks: (i) clarity-level classification into Clear Reply, Ambivalent, and Clear Non-Reply, and (ii) evasion-level classification into nine fine-grained evasion strategies. The benchmark is constructed from U.S. presidential interviews and follows an expert-grounded taxonomy of response clarity and evasion. The task attracted 124 registered teams, who submitted 946 valid runs for clarity-level classification and 539 for evasion-level classification. Results show a substantial gap in difficulty between the two subtasks: the best system achieved 0.89 macro-F1 on clarity classification, surpassing the strongest baseline by a large margin, while the top evasion-level system reached 0.68 macro-F1, matching the best baseline. Overall, large language model prompting and hierarchical exploitation of the taxonomy emerged as the most effective strategies, with top systems consistently outperforming those that treated the two subtasks independently. CLARITY establishes political response evasion as a challenging benchmark for computational discourse analysis and highlights the difficulty of modeling strategic ambiguity in political language. 

---
# Beyond Explicit Edges: Robust Reasoning over Noisy and Sparse Knowledge Graphs 

**Authors**: Hang Gao, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2603.14006)  

**Abstract**: GraphRAG is increasingly adopted for converting unstructured corpora into graph structures to enable multi-hop reasoning. However, standard graph algorithms rely heavily on static connectivity and explicit edges, often failing in real-world scenarios where knowledge graphs (KGs) are noisy, sparse, or incomplete. To address this limitation, we introduce INSES (Intelligent Navigation and Similarity Enhanced Search), a dynamic framework designed to reason beyond explicit edges. INSES couples LLM-guided navigation, which prunes noise and steers exploration, with embedding-based similarity expansion to recover hidden links and bridge semantic gaps. Recognizing the computational cost of graph reasoning, we complement INSES with a lightweight router that delegates simple queries to Naïve RAG and escalates complex cases to INSES, balancing efficiency with reasoning depth. INSES consistently outperforms SOTA RAG and GraphRAG baselines across multiple benchmarks. Notably, on the MINE benchmark, it demonstrates superior robustness across KGs constructed by varying methods (KGGEN, GraphRAG, OpenIE), improving accuracy by 5%, 10%, and 27%, respectively. 

---
# FLUX: Data Worth Training On 

**Authors**: Gowtham, Sai Rupesh, Sanjay Kumar, Saravanan, Venkata Chaithanya  

**Link**: [PDF](https://arxiv.org/pdf/2603.13972)  

**Abstract**: Modern large language model training is no longer limited by data availability, but by the inability of existing preprocessing pipelines to simultaneously achieve massive scale and high data quality. Current approaches are forced to sacrifice one for the other: either aggressively filtering to improve quality at the cost of severe token loss, or retaining large volumes of data while introducing substantial noise. In this work, we introduce FLUX, a preprocessing pipeline specifically designed to break this long-standing trade-off by maximizing token retention while enforcing rigorous quality control. Models trained on FLUX-curated data consistently outperform prior methods. A 3B-parameter model trained on 60B tokens with FLUX achieves 32.14% MMLU accuracy, surpassing the previous state-of-the-art pipeline DCLM (31.98%) and significantly outperforming FineWeb (29.88%). FLUX achieves the same aggregate score as a model trained on DCLM data using only 39B tokens, resulting in a 34.4% reduction in training compute. At the data level, FLUX extracts 50B usable tokens from a single dump (CC-MAIN-2025-51), compared to 40B from DCLM (+25% retention). FLUX-Base yields 192B tokens, exceeding FineWeb's 170B while still maintaining superior quality. Overall, FLUX establishes a new state of the art in web-scale data preprocessing by demonstrating that high retention, strong quality control, and computational efficiency can be achieved simultaneously, redefining the limits of scalable dataset construction for modern language models. 

---
# sebis at ArchEHR-QA 2026: How Much Can You Do Locally? Evaluating Grounded EHR QA on a Single Notebook 

**Authors**: Ibrahim Ebrar Yurt, Fabian Karl, Tejaswi Choppa, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2603.13962)  

**Abstract**: Clinical question answering over electronic health records (EHRs) can help clinicians and patients access relevant medical information more efficiently. However, many recent approaches rely on large cloud-based models, which are difficult to deploy in clinical environments due to privacy constraints and computational requirements. In this work, we investigate how far grounded EHR question answering can be pushed when restricted to a single notebook. We participate in all four subtasks of the ArchEHR-QA 2026 shared task and evaluate several approaches designed to run on commodity hardware. All experiments are conducted locally without external APIs or cloud infrastructure. Our results show that such systems can achieve competitive performance on the shared task leaderboards. In particular, our submissions perform above average in two subtasks, and we observe that smaller models can approach the performance of much larger systems when properly configured. These findings suggest that privacy-preserving EHR QA systems running fully locally are feasible with current models and commodity hardware. The source code is available at this https URL. 

---
# ToolFlood: Beyond Selection -- Hiding Valid Tools from LLM Agents via Semantic Covering 

**Authors**: Hussein Jawad, Nicolas J-B Brunel  

**Link**: [PDF](https://arxiv.org/pdf/2603.13950)  

**Abstract**: Large Language Model (LLM) agents increasingly use external tools for complex tasks and rely on embedding-based retrieval to select a small top-k subset for reasoning. As these systems scale, the robustness of this retrieval stage is underexplored, even though prior work has examined attacks on tool selection. This paper introduces ToolFlood, a retrieval-layer attack on tool-augmented LLM agents. Rather than altering which tool is chosen after retrieval, ToolFlood overwhelms retrieval itself by injecting a few attacker-controlled tools whose metadata is carefully placed by exploiting the geometry of embedding space. These tools semantically span many user queries, dominate the top-k results, and push all benign tools out of the agent's context.
ToolFlood uses a two-phase adversarial tool generation strategy. It first samples subsets of target queries and uses an LLM to iteratively generate diverse tool names and descriptions. It then runs an iterative greedy selection that chooses tools maximizing coverage of remaining queries in embedding space under a cosine-distance threshold, stopping when all queries are covered or a budget is reached. We provide theoretical analysis of retrieval saturation and show on standard benchmarks that ToolFlood achieves up to a 95% attack success rate with a low injection rate (1% in ToolBench). The code will be made publicly available at the following link: this https URL 

---
# OmniCompliance-100K: A Multi-Domain, Rule-Grounded, Real-World Safety Compliance Dataset 

**Authors**: Wenbin Hu, Huihao Jing, Haochen Shi, Changxuan Fan, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.13933)  

**Abstract**: Ensuring the safety and compliance of large language models (LLMs) is of paramount importance. However, existing LLM safety datasets often rely on ad-hoc taxonomies for data generation and suffer from a significant shortage of rule-grounded, real-world cases that are essential for robustly protecting LLMs. In this work, we address this critical gap by constructing a comprehensive safety dataset from a compliance perspective. Using a powerful web-searching agent, we collect a rule-grounded, real-world case dataset OmniCompliance-100K, sourced from multi-domain authoritative references. The dataset spans 74 regulations and policies across a wide range of domains, including security and privacy regulations, content safety and user data privacy policies from leading AI companies and social media platforms, financial security requirements, medical device risk management standards, educational integrity guidelines, and protections of fundamental human rights. In total, our dataset contains 12,985 distinct rules and 106,009 associated real-world compliance cases. Our analysis confirms a strong alignment between the rules and their corresponding cases. We further conduct extensive benchmarking experiments to evaluate the safety and compliance capabilities of advanced LLMs across different model scales. Our experiments reveal several interesting findings that have great potential to offer valuable insights for future LLM safety research. 

---
# Large Language Models Reproduce Racial Stereotypes When Used for Text Annotation 

**Authors**: Petter Törnberg  

**Link**: [PDF](https://arxiv.org/pdf/2603.13891)  

**Abstract**: Large language models (LLMs) are increasingly used for automated text annotation in tasks ranging from academic research to content moderation and hiring. Across 19 LLMs and two experiments totaling more than 4 million annotation judgments, we show that subtle identity cues embedded in text systematically bias annotation outcomes in ways that mirror racial stereotypes. In a names-based experiment spanning 39 annotation tasks, texts containing names associated with Black individuals are rated as more aggressive by 18 of 19 models and more gossipy by 18 of 19. Asian names produce a bamboo-ceiling profile: 17 of 19 models rate individuals as more intelligent, while 18 of 19 rate them as less confident and less sociable. Arab names elicit cognitive elevation alongside interpersonal devaluation, and all four minority groups are consistently rated as less self-disciplined. In a matched dialect experiment, the same sentence is judged significantly less professional (all 19 models, mean gap $-0.774$), less indicative of an educated speaker ($-0.688$), more toxic (18/19), and more angry (19/19) when written in African American Vernacular English rather than Standard American English. A notable exception occurs for name-based hireability, where fine-tuning appears to overcorrect, systematically favoring minority-named applicants. These findings suggest that using LLMs as automated annotators can embed socially patterned biases directly into the datasets and measurements that increasingly underpin research, governance, and decision-making. 

---
# GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent 

**Authors**: Yuri Kuratov, Matvey Kairov, Aydar Bulatov, Ivan Rodkin, Mikhail Burtsev  

**Link**: [PDF](https://arxiv.org/pdf/2603.13875)  

**Abstract**: Many large language model applications require conditioning on long contexts. Transformers typically support this by storing a large per-layer KV-cache of past activations, which incurs substantial memory overhead. A desirable alternative is ompressive memory: read a context once, store it in a compact state, and answer many queries from that state. We study this in a context removal setting, where the model must generate an answer without access to the original context at inference time. We introduce GradMem, which writes context into memory via per-sample test-time optimization. Given a context, GradMem performs a few steps of gradient descent on a small set of prefix memory tokens while keeping model weights frozen. GradMem explicitly optimizes a model-level self-supervised context reconstruction loss, resulting in a loss-driven write operation with iterative error correction, unlike forward-only methods. On associative key--value retrieval, GradMem outperforms forward-only memory writers with the same memory size, and additional gradient steps scale capacity much more effectively than repeated forward writes. We further show that GradMem transfers beyond synthetic benchmarks: with pretrained language models, it attains competitive results on natural language tasks including bAbI and SQuAD variants, relying only on information encoded in memory. 

---
# APEX-Searcher: Augmenting LLMs' Search Capabilities through Agentic Planning and Execution 

**Authors**: Kun Chen, Qingchao Kong, Zhao Feifei, Wenji Mao  

**Link**: [PDF](https://arxiv.org/pdf/2603.13853)  

**Abstract**: Retrieval-augmented generation (RAG), based on large language models (LLMs), serves as a vital approach to retrieving and leveraging external knowledge in various domain applications. When confronted with complex multi-hop questions, single-round retrieval is often insufficient for accurate reasoning and problem solving. To enhance search capabilities for complex tasks, most existing works integrate multi-round iterative retrieval with reasoning processes via end-to-end training. While these approaches significantly improve problem-solving performance, they are still faced with challenges in task reasoning and model training, especially ambiguous retrieval execution paths and sparse rewards in end-to-end reinforcement learning (RL) process, leading to inaccurate retrieval results and performance degradation. To address these issues, in this paper, we proposes APEX-Searcher, a novel Agentic Planning and Execution framework to augment LLM search capabilities. Specifically, we introduce a two-stage agentic framework that decouples the retrieval process into planning and execution: It first employs RL with decomposition-specific rewards to optimize strategic planning; Built on the sub-task decomposition, it then applies supervised fine-tuning on high-quality multi-hop trajectories to equip the model with robust iterative sub-task execution capabilities. Extensive experiments demonstrate that our proposed framework achieves significant improvements in both multi-hop RAG and task planning performances across multiple benchmarks. 

---
# PMIScore: An Unsupervised Approach to Quantify Dialogue Engagement 

**Authors**: Yongkang Guo, Zhihuan Huang, Yuqing Kong  

**Link**: [PDF](https://arxiv.org/pdf/2603.13796)  

**Abstract**: High dialogue engagement is a crucial indicator of an effective conversation. A reliable measure of engagement could help benchmark large language models, enhance the effectiveness of human-computer interactions, or improve personal communication skills. However, quantifying engagement is challenging, since it is subjective and lacks a "gold standard". This paper proposes PMIScore, an efficient unsupervised approach to quantify dialogue engagement. It uses pointwise mutual information (PMI), which is the probability of generating a response conditioning on the conversation history. Thus, PMIScore offers a clear interpretation of engagement. As directly computing PMI is intractable due to the complexity of dialogues, PMIScore learned it through a dual form of divergence. The algorithm includes generating positive and negative dialogue pairs, extracting embeddings by large language models (LLMs), and training a small neural network using a mutual information loss function. We validated PMIScore on both synthetic and real-world datasets. Our results demonstrate the effectiveness of PMIScore in PMI estimation and the reasonableness of the PMI metric itself. 

---
# GhanaNLP Parallel Corpora: Comprehensive Multilingual Resources for Low-Resource Ghanaian Languages 

**Authors**: Lawrence Adu Gyamfi, Paul Azunre, Stephen Edward Moore, Joel Budu, Akwasi Asare, Mich-Seth Owusu, Jonathan Ofori Asiamah  

**Link**: [PDF](https://arxiv.org/pdf/2603.13793)  

**Abstract**: Low resource languages present unique challenges for natural language processing due to the limited availability of digitized and well structured linguistic data. To address this gap, the GhanaNLP initiative has developed and curated 41,513 parallel sentence pairs for the Twi, Fante, Ewe, Ga, and Kusaal languages, which are widely spoken across Ghana yet remain underrepresented in digital spaces. Each dataset consists of carefully aligned sentence pairs between a local language and English. The data were collected, translated, and annotated by human professionals and enriched with standard structural metadata to ensure consistency and usability. These corpora are designed to support research, educational, and commercial applications, including machine translation, speech technologies, and language preservation. This paper documents the dataset creation methodology, structure, intended use cases, and evaluation, as well as their deployment in real world applications such as the Khaya AI translation engine. Overall, this work contributes to broader efforts to democratize AI by enabling inclusive and accessible language technologies for African languages. 

---
# DeceptGuard :A Constitutional Oversight Framework For Detecting Deception in LLM Agents 

**Authors**: Snehasis Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2603.13791)  

**Abstract**: Reliable detection of deceptive behavior in Large Language Model (LLM) agents is an essential prerequisite for safe deployment in high-stakes agentic contexts. Prior work on scheming detection has focused exclusively on black-box monitors that observe only externally visible tool calls and outputs, discarding potentially rich internal reasoning signals. We introduce DECEPTGUARD, a unified framework that systematically compares three monitoring regimes: black-box monitors (actions and outputs only), CoT-aware monitors (additionally observing the agent's chain-of-thought reasoning trace), and activation-probe monitors (additionally reading hidden-state representations from a frozen open-weights encoder).
We introduce DECEPTSYNTH, a scalable synthetic pipeline for generating deception-positive and deception-negative agent trajectories across a novel 12-category taxonomy spanning verbal, behavioral, and structural deception. Our monitors are optimized on 4,800 synthetic trajectories and evaluated on 9,200 held-out samples from DeceptArena, a benchmark of realistic sandboxed agent environments with execution-verified labels.
Across all evaluation settings, CoT-aware and activation-probe monitors substantially outperform their black-box counterparts (mean pAUROC improvement of +0.097), with the largest gains on subtle, long-horizon deception that leaves minimal behavioral footprints. We empirically characterize a transparency-detectability trade-off: as agents learn to suppress overt behavioral signals, chain-of-thought becomes the primary detection surface but is itself increasingly unreliable due to post-training faithfulness degradation.
We propose HYBRID-CONSTITUTIONAL ensembles as a robust defense-in-depth approach, achieving a pAUROC of 0.934 on the held-out test set, representing a substantial advance over the prior state of the art. 

---
# Projection-Free Evolution Strategies for Continuous Prompt Search 

**Authors**: Yu Cai, Canxi Huang, Xiaoyu He  

**Link**: [PDF](https://arxiv.org/pdf/2603.13786)  

**Abstract**: Continuous prompt search offers a computationally efficient alternative to conventional parameter tuning in natural language processing tasks. Nevertheless, its practical effectiveness can be significantly hindered by the black-box nature and the inherent high-dimensionality of the objective landscapes. Existing methods typically mitigate these challenges by restricting the search to a randomly projected low-dimensional subspace. However, the effectiveness and underlying motivation of the projection mechanism remain ambiguous. In this paper, we first empirically demonstrate that despite the prompt space possessing a low-dimensional structure, random projections fail to adequately capture this essential structure. Motivated by this finding, we propose a projection-free prompt search method based on evolutionary strategies. By directly optimizing in the full prompt space with an adaptation mechanism calibrated to the intrinsic dimension, our method achieves competitive search capabilities without additional computational overhead. Furthermore, to bridge the generalization gap in few-shot scenarios, we introduce a confidence-based regularization mechanism that systematically enhances the model's confidence in the target verbalizers. Experimental results on seven natural language understanding tasks from the GLUE benchmark demonstrate that our proposed approach significantly outperforms existing baselines. 

---
# Generate Then Correct: Single Shot Global Correction for Aspect Sentiment Quad Prediction 

**Authors**: Shidong He, Haoyu Wang, Wenjie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.13777)  

**Abstract**: Aspect-based sentiment analysis (ABSA) extracts aspect-level sentiment signals from user-generated text, supports product analytics, experience monitoring, and public-opinion tracking, and is central to fine-grained opinion mining. A key challenge in ABSA is aspect sentiment quad prediction (ASQP), which requires identifying four elements: the aspect term, the aspect category, the opinion term, and the sentiment polarity. However, existing studies usually linearize the unordered quad set into a fixed-order template and decode it left-to-right. With teacher forcing training, the resulting training-inference mismatch (exposure bias) lets early prefix errors propagate to later elements. The linearization order determines which elements appear earlier in the prefix, so this propagation becomes order-sensitive and is hard to repair in a single pass. To address this, we propose a method, Generate-then-Correct (G2C): a generator drafts quads and a corrector performs a single-shot, sequence-level global correction trained on LLM-synthesized drafts with common error patterns. On the Rest15 and Rest16 datasets, G2C outperforms strong baseline models. 

---
# LiveWeb-IE: A Benchmark For Online Web Information Extraction 

**Authors**: Seungbin Yang, Jihwan Kim, Jaemin Choi, Dongjin Kim, Soyoung Yang, ChaeHun Park, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2603.13773)  

**Abstract**: Web information extraction (WIE) is the task of automatically extracting data from web pages, offering high utility for various applications. The evaluation of WIE systems has traditionally relied on benchmarks built from HTML snapshots captured at a single point in time. However, this offline evaluation paradigm fails to account for the temporally evolving nature of the web; consequently, performance on these static benchmarks often fails to generalize to dynamic real-world scenarios. To bridge this gap, we introduce \dataset, a new benchmark designed for evaluating WIE systems directly against live websites. Based on trusted and permission-granted websites, we curate natural language queries that require information extraction of various data categories, such as text, images, and hyperlinks. We further design these queries to represent four levels of complexity, based on the number and cardinality of attributes to be extracted, enabling a granular assessment of WIE systems. In addition, we propose Visual Grounding Scraper (VGS), a novel multi-stage agentic framework that mimics human cognitive processes by visually narrowing down web page content to extract desired information. Extensive experiments across diverse backbone models demonstrate the effectiveness and robustness of VGS. We believe that this study lays the foundation for developing practical and robust WIE systems. 

---
# Knowledge Distillation for Large Language Models 

**Authors**: Alejandro Paredes La Torre, Barbara Flores, Diego Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2603.13765)  

**Abstract**: We propose a resource-efficient framework for compressing large language models through knowledge distillation, combined with guided chain-of-thought reinforcement learning. Using Qwen 3B as the teacher and Qwen 0.5B as the student, we apply knowledge distillation across English Dolly-15k, Spanish Dolly-15k, and code BugNet and PyTorrent datasets, with hyperparameters tuned in the English setting to optimize student performance. Across tasks, the distilled student retains a substantial portion of the teacher's capability while remaining significantly smaller: 70% to 91% in English, up to 95% in Spanish, and up to 93.5% Rouge-L in code. For coding tasks, integrating chain-of-thought prompting with Group Relative Policy Optimization using CoT-annotated Codeforces data improves reasoning coherence and solution correctness compared to knowledge distillation alone. Post-training 4-bit weight quantization further reduces memory footprint and inference latency. These results show that knowledge distillation combined with chain-of-thought guided reinforcement learning can produce compact, efficient models suitable for deployment in resource-constrained settings. 

---
# Can We Trust LLMs on Memristors? Diving into Reasoning Ability under Non-Ideality 

**Authors**: Taiqiang Wu, Yuxin Cheng, Chenchen Ding, Runming Yang, Xincheng Feng, Wenyong Zhou, Zhengwu Liu, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2603.13725)  

**Abstract**: Memristor-based analog compute-in-memory (CIM) architectures provide a promising substrate for the efficient deployment of Large Language Models (LLMs), owing to superior energy efficiency and computational density. However, these architectures suffer from precision issues caused by intrinsic non-idealities of memristors. In this paper, we first conduct a comprehensive investigation into the impact of such typical non-idealities on LLM reasoning. Empirical results indicate that reasoning capability decreases significantly but varies for distinct benchmarks. Subsequently, we systematically appraise three training-free strategies, including thinking mode, in-context learning, and module redundancy. We thus summarize valuable guidelines, i.e., shallow layer redundancy is particularly effective for improving robustness, thinking mode performs better under low noise levels but degrades at higher noise, and in-context learning reduces output length with a slight performance trade-off. Our findings offer new insights into LLM reasoning under non-ideality and practical strategies to improve robustness. 

---
# Repetition Without Exclusivity: Scale Sensitivity of Referential Mechanisms in Child-Scale Language Models 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2603.13696)  

**Abstract**: We present the first systematic evaluation of mutual exclusivity (ME) -- the bias to map novel words to novel referents -- in text-only language models trained on child-directed speech. We operationalise ME as referential suppression: when a familiar object is relabelled in a two-referent discourse context, ME predicts decreased probability of the labelled noun at a subsequent completion position. Three pilot findings motivate a pre-registered scale-sensitivity experiment: (1) a masked language model (BabyBERTa) is entirely insensitive to multi-sentence referential context; (2) autoregressive models show robust repetition priming -- the opposite of ME -- when familiar nouns are re-labelled; and (3) a novel context-dependence diagnostic reveals that apparent ME-like patterns with nonce tokens are fully explained by embedding similarity, not referential disambiguation. In the confirmatory experiment, we train 45 GPT-2-architecture models (2.9M, 8.9M, and 33.5M parameters; 5, 10, and 20 epochs on AO-CHILDES; 5 seeds each) and evaluate on a pre-registered ME battery. Anti-ME repetition priming is significant in all 9 cells (85-100% of items; all p < 2.4 x 10^-13). Priming attenuates with improved language modelling (Spearman rho = -0.533, p = 0.0002) but never crosses zero across a 3.8x perplexity range. The context-dependence diagnostic replicates in all 9 cells, and dose-response priming increases with repetitions in 8/9 cells (all trend p < 0.002). These findings indicate that distributional learning on child-directed speech produces repetition-based reference tracking rather than lexical exclusivity. We connect this to the grounded cognition literature and argue that referential grounding may be a necessary ingredient for ME -- an empirical claim about required input structure, not a nativist one. 

---
# QuarkMedBench: A Real-World Scenario Driven Benchmark for Evaluating Large Language Models 

**Authors**: Yao Wu, Kangping Yin, Liang Dong, Zhenxin Ma, Shuting Xu, Xuehai Wang, Yuxuan Jiang, Tingting Yu, Yunqing Hong, Jiayi Liu, Rianzhe Huang, Shuxin Zhao, Haiping Hu, Wen Shang, Jian Xu, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.13691)  

**Abstract**: While Large Language Models (LLMs) excel on standardized medical exams, high scores often fail to translate to high-quality responses for real-world medical queries. Current evaluations rely heavily on multiple-choice questions, failing to capture the unstructured, ambiguous, and long-tail complexities inherent in genuine user inquiries. To bridge this gap, we introduce QuarkMedBench, an ecologically valid benchmark tailored for real-world medical LLM assessment. We compiled a massive dataset spanning Clinical Care, Wellness Health, and Professional Inquiry, comprising 20,821 single-turn queries and 3,853 multi-turn sessions. To objectively evaluate open-ended answers, we propose an automated scoring framework that integrates multi-model consensus with evidence-based retrieval to dynamically generate 220,617 fine-grained scoring rubrics (~9.8 per query). During evaluation, hierarchical weighting and safety constraints structurally quantify medical accuracy, key-point coverage, and risk interception, effectively mitigating the high costs and subjectivity of human grading. Experimental results demonstrate that the generated rubrics achieve a 91.8% concordance rate with clinical expert blind audits, establishing highly dependable medical reliability. Crucially, baseline evaluations on this benchmark reveal significant performance disparities among state-of-the-art models when navigating real-world clinical nuances, highlighting the limitations of conventional exam-based metrics. Ultimately, QuarkMedBench establishes a rigorous, reproducible yardstick for measuring LLM performance on complex health issues, while its framework inherently supports dynamic knowledge updates to prevent benchmark obsolescence. 

---
# Preconditioned Test-Time Adaptation for Out-of-Distribution Debiasing in Narrative Generation 

**Authors**: Hanwen Shen, Ting Ying, Jiajie Lu, Shanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.13683)  

**Abstract**: Although debiased LLMs perform well on known bias patterns, they often fail to generalize to unfamiliar bias prompts, producing toxic outputs. We first validate that such high-bias prompts constitute a \emph{distribution shift} via OOD detection, and show static models degrade under this shift. To adapt on-the-fly, we propose \textbf{CAP-TTA}, a test-time adaptation framework that performs context-aware LoRA updates only when the bias-risk \emph{trigger} exceeds a threshold, using a precomputed diagonal \emph{preconditioner} for fast and stable updates. Across toxic-prompt settings and benchmarks, CAP-TTA reduces bias (confirmed by human evaluation) while achieving much lower update latency than AdamW/SGD; it also mitigates catastrophic forgetting by significantly improving narrative fluency over SOTA debiasing baseline while maintaining comparable debiasing effectiveness. 

---
# Privacy Preserving Topic-wise Sentiment Analysis of the Iran Israel USA Conflict Using Federated Transformer Models 

**Authors**: Md Saiful Islam, Tanjim Taharat Aurpa, Sharad Hasan, Farzana Akter  

**Link**: [PDF](https://arxiv.org/pdf/2603.13655)  

**Abstract**: The recent escalation of the Iran Israel USA conflict in 2026 has triggered widespread global discussions across social media platforms. As people increasingly use these platforms for expressing opinions, analyzing public sentiment from these discussions can provide valuable insights into global public perception. This study aims to analyze global public sentiment regarding the Iran Israel USA conflict by mining user-generated comments from YouTube news channels. The work contributes to public opinion analysis by introducing a privacy preserving framework that combines topic wise sentiment analysis with modern deep learning techniques and Federated Learning. To achieve this, approximately 19,000 YouTube comments were collected from major international news channels and preprocessed to remove noise and normalize text. Sentiment labels were initially generated using the VADER sentiment analyzer and later validated through manual inspection to improve reliability. Latent Dirichlet Allocation (LDA) was applied to identify key discussion topics related to the conflict. Several transformer-based models, including BERT, RoBERTa, XLNet, DistilBERT, ModernBERT, and ELECTRA, were fine tuned for sentiment classification. The best-performing model was further integrated into a federated learning environment to enable distributed training by preserving user data privacy. Additionally, Explainable Artificial Intelligence (XAI) techniques using SHAP were applied to interpret model predictions and identify influential words affecting sentiment classification. Experimental results demonstrate that transformer models perform effectively, and among them, ELECTRA achieved the best performance with 91.32% accuracy. The federated learning also maintained strong performance while preserving privacy, achieving 89.59% accuracy in a two client configuration. 

---
# Benchmarking Large Language Models on Reference Extraction and Parsing in the Social Sciences and Humanities 

**Authors**: Yurui Zhu, Giovanni Colavizza, Matteo Romanello  

**Link**: [PDF](https://arxiv.org/pdf/2603.13651)  

**Abstract**: Bibliographic reference extraction and parsing are foundational for citation indexing, linking, and downstream scholarly knowledge-graph construction. However, most established evaluations focus on clean, English, end-of-document bibliographies, and therefore underrepresent the Social Sciences and Humanities (SSH), where citations are frequently multilingual, embedded in footnotes, abbreviated, and shaped by heterogeneous historical conventions. We present a unified benchmark that targets these SSH-realistic conditions across three complementary datasets: CEX (English journal articles spanning multiple disciplines), EXCITE (German/English documents with end-section, footnote-only, and mixed regimes), and LinkedBooks (humanities references with strong stylistic variation and multilinguality). We evaluate three tasks of increasing difficulty -- reference extraction, reference parsing, and end-to-end document parsing -- under a schema-constrained setup that enables direct comparison between a strong supervised pipeline baseline (GROBID) and contemporary LLMs (DeepSeek-V3.1, Mistral-Small-3.2-24B, Gemma-3-27B-it, and Qwen3-VL (4B-32B variants)). Across datasets, extraction largely saturates beyond a moderate capability threshold, while parsing and end-to-end parsing remain the primary bottlenecks due to structured-output brittleness under noisy layouts. We further show that lightweight LoRA adaptation yields consistent gains -- especially on SSH-heavy benchmarks -- and that segmentation/pipelining can substantially improve robustness. Finally, we argue for hybrid deployment via routing: leveraging GROBID for well-structured, in-distribution PDFs while escalating multilingual and footnote-heavy documents to task-adapted LLMs. 

---
# Widespread Gender and Pronoun Bias in Moral Judgments Across LLMs 

**Authors**: Gustavo Lúcius Fernandes, Jeiverson C. V. M. Santos, Pedro O. S. Vaz-de-Melo  

**Link**: [PDF](https://arxiv.org/pdf/2603.13636)  

**Abstract**: Large language models (LLMs) are increasingly used to assess moral or ethical statements, yet their judgments may reflect social and linguistic biases. This work presents a controlled, sentence-level study of how grammatical person, number, and gender markers influence LLM moral classifications of fairness. Starting from 550 balanced base sentences from the ETHICS dataset, we generated 26 counterfactual variants per item, systematically varying pronouns and demographic markers to yield 14,850 semantically equivalent sentences. We evaluated six model families (Grok, GPT, LLaMA, Gemma, DeepSeek, and Mistral), and measured fairness judgments and inter-group disparities using Statistical Parity Difference (SPD). Results show statistically significant biases: sentences written in the singular form and third person are more often judged as "fair'', while those in the second person are penalized. Gender markers produce the strongest effects, with non-binary subjects consistently favored and male subjects disfavored. We conjecture that these patterns reflect distributional and alignment biases learned during training, emphasizing the need for targeted fairness interventions in moral LLM applications. 

---
# Design and evaluation of an agentic workflow for crisis-related synthetic tweet datasets 

**Authors**: Roben Delos Reyes, Timothy Douglas, Asanobu Kitamoto  

**Link**: [PDF](https://arxiv.org/pdf/2603.13625)  

**Abstract**: Twitter (now X) has become an important source of social media data for situational awareness during crises. Crisis informatics research has widely used tweets from Twitter to develop and evaluate artificial intelligence (AI) systems for various crisis-relevant tasks, such as extracting locations and estimating damage levels from tweets to support damage assessment. However, recent changes in Twitter's data access policies have made it increasingly difficult to curate real-world tweet datasets related to crises. Moreover, existing curated tweet datasets are limited to past crisis events in specific contexts and are costly to annotate at scale. These limitations constrain the development and evaluation of AI systems used in crisis informatics. To address these limitations, we introduce an agentic workflow for generating crisis-related synthetic tweet datasets. The workflow iteratively generates synthetic tweets conditioned on prespecified target characteristics, evaluates them using predefined compliance checks, and incorporates structured feedback to refine them in subsequent iterations. As a case study, we apply the workflow to generate synthetic tweet datasets relevant to post-earthquake damage assessment. We show that the workflow can generate synthetic tweets that capture their target labels for location and damage level. We further demonstrate that the resulting synthetic tweet datasets can be used to evaluate AI systems on damage assessment tasks like geolocalization and damage level prediction. Our results indicate that the workflow offers a flexible and scalable alternative to real-world tweet data curation, enabling the systematic generation of synthetic social media data across diverse crisis events, societal contexts, and crisis informatics applications. 

---
# Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation 

**Authors**: Minsang Kim, Seung Jun Baek  

**Link**: [PDF](https://arxiv.org/pdf/2603.13260)  

**Abstract**: Knowledge Distillation (KD) can transfer the reasoning abilities of large models to smaller ones, which can reduce the costs to generate Chain-of-Thoughts for reasoning tasks. KD methods typically ask the student to mimic the teacher's distribution over the entire output. However, a student with limited capacity can be overwhelmed by such extensive supervision causing a distribution mismatch, especially in complex reasoning tasks. We propose Token-Selective Dual Knowledge Distillation (TSD-KD), a framework for student-centric distillation. TSD-KD focuses on distilling important tokens for reasoning and encourages the student to explain reasoning in its own words. TSD-KD combines indirect and direct distillation. Indirect distillation uses a weak form of feedback based on preference ranking. The student proposes candidate responses generated on its own; the teacher re-ranks those candidates as indirect feedback without enforcing its entire distribution. Direct distillation uses distribution matching; however, it selectively distills tokens based on the relative confidence between teacher and student. Finally, we add entropy regularization to maintain the student's confidence during distillation. Overall, our method provides the student with targeted and indirect feedback to support its own reasoning process and to facilitate self-improvement. The experiments show the state-of-the-art performance of TSD-KD on 10 challenging reasoning benchmarks, outperforming the baseline and runner-up in accuracy by up to 54.4\% and 40.3\%, respectively. Notably, a student trained by TSD-KD even outperformed its own teacher model in four cases by up to 20.3\%. The source code is available at this https URL. 

---
# How Transformers Reject Wrong Answers: Rotational Dynamics of Factual Constraint Processing 

**Authors**: Javier Marín  

**Link**: [PDF](https://arxiv.org/pdf/2603.13259)  

**Abstract**: When a language model is fed a wrong answer, what happens inside the network? Current understanding treats truthfulness as a static property of individual-layer representations-a direction to be probed, a feature to be extracted. Less is known about the dynamics: how internal representations diverge across the full depth of the network when the model processes correct versus incorrect continuations.
We introduce forced-completion probing, a method that presents identical queries with known correct and incorrect single-token continuations and tracks five geometric measurements across every layer of four decoder-only models(1.5B-13B parameters). We report three findings. First, correct and incorrect paths diverge through rotation, not rescaling: displacement vectors maintain near-identical magnitudes while their angular separation increases, meaning factual selection is encoded in direction on an approximate hypersphere. Second, the model does not passively fail on incorrect input-it actively suppresses the correct answer, driving internal probability away from the right token. Third, both phenomena are entirely absent below a parameter threshold and emerge at 1.6B, suggesting a phase transition in factual processing capability.
These results show that factual constraint processing has a specific geometric character-rotational, not scalar; active, not passive-that is invisible to methods based on single-layer probes or magnitude comparisons. 

---
# Training-Free Agentic AI: Probabilistic Control and Coordination in Multi-Agent LLM Systems 

**Authors**: Mohammad Parsa Hosseini, Ankit Shah, Saiyra Qureshi, Alex Huang, Connie Miao, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.13256)  

**Abstract**: Multi-agent large language model (LLM) systems enable complex, long-horizon reasoning by composing specialized agents, but practical deployment remains hindered by inefficient routing, noisy feedback, and high interaction cost. We introduce REDEREF, a lightweight and training-free controller for multi-agent LLM collaboration that improves routing efficiency during recursive delegation. REDEREF integrates (i) belief-guided delegation via Thompson sampling to prioritize agents with historically positive marginal contributions, (ii) reflection-driven re-routing using a calibrated LLM or programmatic judge, (iii) evidence-based selection rather than output averaging, and (iv) memory-aware priors to reduce cold-start inefficiency. Across multi-agent split-knowledge tasks, we show that while recursive retry alone saturates task success, belief-guided routing reduces token usage by 28%, agent calls by 17%, and time-to-success by 19% compared to random recursive delegation, and adapts gracefully under agent or judge degradation. These results demonstrate that simple, interpretable probabilistic control can meaningfully improve the efficiency and robustness of multi-agent LLM systems without training or fine-tuning. 

---
# Steering at the Source: Style Modulation Heads for Robust Persona Control 

**Authors**: Yoshihiro Izawa, Gouki Minegishi, Koshi Eguchi, Sosuke Hosokawa, Kenjiro Taura  

**Link**: [PDF](https://arxiv.org/pdf/2603.13249)  

**Abstract**: Activation steering offers a computationally efficient mechanism for controlling Large Language Models (LLMs) without fine-tuning. While effectively controlling target traits (e.g., persona), coherency degradation remains a major obstacle to safety and practical deployment. We hypothesize that this degradation stems from intervening on the residual stream, which indiscriminately affects aggregated features and inadvertently amplifies off-target noise. In this work, we identify a sparse subset of attention heads (only three heads) that independently govern persona and style formation, which we term Style Modulation Heads. Specifically, these heads can be localized via geometric analysis of internal representations, combining layer-wise cosine similarity and head-wise contribution scores. We demonstrate that intervention targeting only these specific heads achieves robust behavioral control while significantly mitigating the coherency degradation observed in residual stream steering. More broadly, our findings show that precise, component-level localization enables safer and more precise model control. 

---
# Slang Context-based Inference Enhancement via Greedy Search-Guided Chain-of-Thought Prompting 

**Authors**: Jinghan Cao, Qingyang Ren, Xiangyun Chen, Xinjin Li, Haoxiang Gao, Yu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.13230)  

**Abstract**: Slang interpretation has been a challenging downstream task for Large Language Models (LLMs) as the expressions are inherently embedded in contextual, cultural, and linguistic frameworks. In the absence of domain-specific training data, it is difficult for LLMs to accurately interpret slang meaning based on lexical information. This paper attempts to investigate the challenges of slang inference using large LLMs and presents a greedy search-guided chain-of-thought framework for slang interpretation. Through our experiments, we conclude that the model size and temperature settings have limited impact on inference accuracy. Transformer-based models with larger active parameters do not generate higher accuracy than smaller models. Based on the results of the above empirical study, we integrate greedy search algorithms with chain-of-thought prompting for small language models to build a framework that improves the accuracy of slang interpretation. The experimental results indicate that our proposed framework demonstrates improved accuracy in slang meaning interpretation. These findings contribute to the understanding of context dependency in language models and provide a practical solution for enhancing slang comprehension through a structured reasoning prompting framework. 

---
# From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation 

**Authors**: Yibin Liu, Yaxing Lyu, Daqi Gao, Zhixuan Liang, Weiliang Tang, Shilong Mu, Xiaokang Yang, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15600)  

**Abstract**: Accurate process supervision remains a critical challenge for long-horizon robotic manipulation. A primary bottleneck is that current video MLLMs, trained primarily under a Supervised Fine-Tuning (SFT) paradigm, function as passive "Observers" that recognize ongoing events rather than evaluating the current state relative to the final task goal. In this paper, we introduce PRIMO R1 (Process Reasoning Induced Monitoring), a 7B framework that transforms video MLLMs into active "Critics". We leverage outcome-based Reinforcement Learning to incentivize explicit Chain-of-Thought generation for progress estimation. Furthermore, our architecture constructs a structured temporal input by explicitly anchoring the video sequence between initial and current state images. Supported by the proposed PRIMO Dataset and Benchmark, extensive experiments across diverse in-domain environments and out-of-domain real-world humanoid scenarios demonstrate that PRIMO R1 achieves state-of-the-art performance. Quantitatively, our 7B model achieves a 50% reduction in the mean absolute error of specialized reasoning baselines, demonstrating significant relative accuracy improvements over 72B-scale general MLLMs. Furthermore, PRIMO R1 exhibits strong zero-shot generalization on difficult failure detection tasks. We establish state-of-the-art performance on RoboFail benchmark with 67.0% accuracy, surpassing closed-source models like OpenAI o1 by 6.0%. 

---
# OpenSeeker: Democratizing Frontier Search Agents by Fully Open-Sourcing Training Data 

**Authors**: Yuwen Du, Rui Ye, Shuo Tang, Xinyu Zhu, Yijun Lu, Yuzhu Cai, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.15594)  

**Abstract**: Deep search capabilities have become an indispensable competency for frontier Large Language Model (LLM) agents, yet the development of high-performance search agents remains dominated by industrial giants due to a lack of transparent, high-quality training data. This persistent data scarcity has fundamentally hindered the progress of the broader research community in developing and innovating within this domain. To bridge this gap, we introduce OpenSeeker, the first fully open-source search agent (i.e., model and data) that achieves frontier-level performance through two core technical innovations: (1) Fact-grounded scalable controllable QA synthesis, which reverse-engineers the web graph via topological expansion and entity obfuscation to generate complex, multi-hop reasoning tasks with controllable coverage and complexity. (2) Denoised trajectory synthesis, which employs a retrospective summarization mechanism to denoise the trajectory, therefore promoting the teacher LLMs to generate high-quality actions. Experimental results demonstrate that OpenSeeker, trained (a single training run) on only 11.7k synthesized samples, achieves state-of-the-art performance across multiple benchmarks including BrowseComp, BrowseComp-ZH, xbench-DeepSearch, and WideSearch. Notably, trained with simple SFT, OpenSeeker significantly outperforms the second-best fully open-source agent DeepDive (e.g., 29.5% v.s. 15.3% on BrowseComp), and even surpasses industrial competitors such as Tongyi DeepResearch (trained via extensive continual pre-training, SFT, and RL) on BrowseComp-ZH (48.4% v.s. 46.7%). We fully open-source the complete training dataset and the model weights to democratize frontier search agent research and foster a more transparent, collaborative ecosystem. 

---
# Amplification Effects in Test-Time Reinforcement Learning: Safety and Reasoning Vulnerabilities 

**Authors**: Vanshaj Khattar, Md Rafi ur Rashid, Moumita Choudhury, Jing Liu, Toshiaki Koike-Akino, Ming Jin, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15417)  

**Abstract**: Test-time training (TTT) has recently emerged as a promising method to improve the reasoning abilities of large language models (LLMs), in which the model directly learns from test data without access to labels. However, this reliance on test data also makes TTT methods vulnerable to harmful prompt injections. In this paper, we investigate safety vulnerabilities of TTT methods, where we study a representative self-consistency-based test-time learning method: test-time reinforcement learning (TTRL), a recent TTT method that improves LLM reasoning by rewarding self-consistency using majority vote as a reward signal. We show that harmful prompt injection during TTRL amplifies the model's existing behaviors, i.e., safety amplification when the base model is relatively safe, and harmfulness amplification when it is vulnerable to the injected data. In both cases, there is a decline in reasoning ability, which we refer to as the reasoning tax. We also show that TTT methods such as TTRL can be exploited adversarially using specially designed "HarmInject" prompts to force the model to answer jailbreak and reasoning queries together, resulting in stronger harmfulness amplification. Overall, our results highlight that TTT methods that enhance LLM reasoning by promoting self-consistency can lead to amplification behaviors and reasoning degradation, highlighting the need for safer TTT methods. 

---
# TrinityGuard: A Unified Framework for Safeguarding Multi-Agent Systems 

**Authors**: Kai Wang, Biaojie Zeng, Zeming Wei, Chang Jin, Hefeng Zhou, Xiangtian Li, Chao Yang, Jingjing Qu, Xingcheng Xu, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15408)  

**Abstract**: With the rapid development of LLM-based multi-agent systems (MAS), their significant safety and security concerns have emerged, which introduce novel risks going beyond single agents or LLMs. Despite attempts to address these issues, the existing literature lacks a cohesive safeguarding system specialized for MAS risks. In this work, we introduce TrinityGuard, a comprehensive safety evaluation and monitoring framework for LLM-based MAS, grounded in the OWASP standards. Specifically, TrinityGuard encompasses a three-tier fine-grained risk taxonomy that identifies 20 risk types, covering single-agent vulnerabilities, inter-agent communication threats, and system-level emergent hazards. Designed for scalability across various MAS structures and platforms, TrinityGuard is organized in a trinity manner, involving an MAS abstraction layer that can be adapted to any MAS structures, an evaluation layer containing risk-specific test modules, alongside runtime monitor agents coordinated by a unified LLM Judge Factory. During Evaluation, TrinityGuard executes curated attack probes to generate detailed vulnerability reports for each risk type, where monitor agents analyze structured execution traces and issue real-time alerts, enabling both pre-development evaluation and runtime monitoring. We further formalize these safety metrics and present detailed case studies across various representative MAS examples, showcasing the versatility and reliability of TrinityGuard. Overall, TrinityGuard acts as a comprehensive framework for evaluating and monitoring various risks in MAS, paving the way for further research into their safety and security. 

---
# CRASH: Cognitive Reasoning Agent for Safety Hazards in Autonomous Driving 

**Authors**: Erick Silva, Rehana Yasmin, Ali Shoker  

**Link**: [PDF](https://arxiv.org/pdf/2603.15364)  

**Abstract**: As AVs grow in complexity and diversity, identifying the root causes of operational failures has become increasingly complex. The heterogeneity of system architectures across manufacturers, ranging from end-to-end to modular designs, together with variations in algorithms and integration strategies, limits the standardization of incident investigations and hinders systematic safety analysis. This work examines real-world AV incidents reported in the NHTSA database. We curate a dataset of 2,168 cases reported between 2021 and 2025, representing more than 80 million miles driven. To process this data, we introduce CRASH, Cognitive Reasoning Agent for Safety Hazards, an LLM-based agent that automates reasoning over crash reports by leveraging both standardized fields and unstructured narrative descriptions. CRASH operates on a unified representation of each incident to generate concise summaries, attribute a primary cause, and assess whether the AV materially contributed to the event. Our findings show that (1) CRASH attributes 64% of incidents to perception or planning failures, underscoring the importance of reasoning-based analysis for accurate fault attribution; and (2) approximately 50% of reported incidents involve rear-end collisions, highlighting a persistent and unresolved challenge in autonomous driving deployment. We further validate CRASH with five domain experts, achieving 86% accuracy in attributing AV system failures. Overall, CRASH demonstrates strong potential as a scalable and interpretable tool for automated crash analysis, providing actionable insights to support safety research and the continued development of autonomous driving systems. 

---
# Directional Embedding Smoothing for Robust Vision Language Models 

**Authors**: Ye Wang, Jing Liu, Toshiaki Koike-Akino  

**Link**: [PDF](https://arxiv.org/pdf/2603.15259)  

**Abstract**: The safety and reliability of vision-language models (VLMs) are a crucial part of deploying trustworthy agentic AI systems. However, VLMs remain vulnerable to jailbreaking attacks that undermine their safety alignment to yield harmful outputs. In this work, we extend the Randomized Embedding Smoothing and Token Aggregation (RESTA) defense to VLMs and evaluate its performance against the JailBreakV-28K benchmark of multi-modal jailbreaking attacks. We find that RESTA is effective in reducing attack success rate over this diverse corpus of attacks, in particular, when employing directional embedding noise, where the injected noise is aligned with the original token embedding vectors. Our results demonstrate that RESTA can contribute to securing VLMs within agentic systems, as a lightweight, inference-time defense layer of an overall security framework. 

---
# To See is Not to Master: Teaching LLMs to Use Private Libraries for Code Generation 

**Authors**: Yitong Zhang, Chengze Li, Ruize Chen, Guowei Yang, Xiaoran Jia, Yijie Ren, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.15159)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for code generation, yet they remain limited in private-library-oriented code generation, where the goal is to generate code using APIs from private libraries. Existing approaches mainly rely on retrieving private-library API documentation and injecting relevant knowledge into the context at inference time. However, our study shows that this is insufficient: even given accurate required knowledge, LLMs still struggle to invoke private-library APIs effectively.
To address this limitation, we propose PriCoder, an approach that teaches LLMs to invoke private-library APIs through automatically synthesized data. Specifically, PriCoder models private-library data synthesis as the construction of a graph, and alternates between two graph operators: (1) Progressive Graph Evolution, which improves data diversity by progressively synthesizing more diverse training samples from basic ones, and (2) Multidimensional Graph Pruning, which improves data quality through a rigorous filtering pipeline. To support rigorous evaluation, we construct two new benchmarks based on recently released libraries that are unfamiliar to the tested models. Experiments on three mainstream LLMs show that PriCoder substantially improves private-library-oriented code generation, yielding gains of over 20% in pass@1 in many settings, while causing negligible impact on general code generation capability. Our code and benchmarks are publicly available at this https URL. 

---
# MER-Bench: A Comprehensive Benchmark for Multimodal Meme Reappraisal 

**Authors**: Yiqi Nie, Fei Wang, Junjie Chen, Kun Li, Yudi Cai, Dan Guo, Chenglong Li, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15020)  

**Abstract**: Memes represent a tightly coupled, multimodal form of social expression, in which visual context and overlaid text jointly convey nuanced affect and commentary. Inspired by cognitive reappraisal in psychology, we introduce Meme Reappraisal, a novel multimodal generation task that aims to transform negatively framed memes into constructive ones while preserving their underlying scenario, entities, and structural layout. Unlike prior works on meme understanding or generation, Meme Reappraisal requires emotion-controllable, structure-preserving multimodal transformation under multiple semantic and stylistic constraints. To support this task, we construct MER-Bench, a benchmark of real-world memes with fine-grained multimodal annotations, including source and target emotions, positively rewritten meme text, visual editing specifications, and taxonomy labels covering visual type, sentiment polarity, and layout structure. We further propose a structured evaluation framework based on a multimodal large language model (MLLM)-as-a-Judge paradigm, decomposing performance into modality-level generation quality, affect controllability, structural fidelity, and global affective alignment. Extensive experiments across representative image-editing and multimodal-generation systems reveal substantial gaps in satisfying the constraints of structural preservation, semantic consistency, and affective transformation. We believe MER-Bench establishes a foundation for research on controllable meme editing and emotion-aware multimodal generation. Our code is available at: this https URL. 

---
# Why Agents Compromise Safety Under Pressure 

**Authors**: Hengle Jiang, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14975)  

**Abstract**: Large Language Model agents deployed in complex environments frequently encounter a conflict between maximizing goal achievement and adhering to safety constraints. This paper identifies a new concept called Agentic Pressure, which characterizes the endogenous tension emerging when compliant execution becomes infeasible. We demonstrate that under this pressure agents exhibit normative drift where they strategically sacrifice safety to preserve utility. Notably we find that advanced reasoning capabilities accelerate this decline as models construct linguistic rationalizations to justify violation. Finally, we analyze the root causes and explore preliminary mitigation strategies, such as pressure isolation, which attempts to restore alignment by decoupling decision-making from pressure signals. 

---
# Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework 

**Authors**: Zhuoshang Wang, Yubing Ren, Yanan Cao, Fang Fang, Xiaoxue Li, Li Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.14968)  

**Abstract**: While watermarking serves as a critical mechanism for LLM provenance, existing secret-key schemes tightly couple detection with injection, requiring access to keys or provider-side scheme-specific detectors for verification. This dependency creates a fundamental barrier for real-world governance, as independent auditing becomes impossible without compromising model security or relying on the opaque claims of service providers. To resolve this dilemma, we introduce TTP-Detect, a pioneering black-box framework designed for non-intrusive, third-party watermark verification. By decoupling detection from injection, TTP-Detect reframes verification as a relative hypothesis testing problem. It employs a proxy model to amplify watermark-relevant signals and a suite of complementary relative measurements to assess the alignment of the query text with watermarked distributions. Extensive experiments across representative watermarking schemes, datasets and models demonstrate that TTP-Detect achieves superior detection performance and robustness against diverse attacks. 

---
# LLM as Graph Kernel: Rethinking Message Passing on Text-Rich Graphs 

**Authors**: Ying Zhang, Hang Yu, Haipeng Zhang, Peng Di  

**Link**: [PDF](https://arxiv.org/pdf/2603.14937)  

**Abstract**: Text-rich graphs, which integrate complex structural dependencies with abundant textual information, are ubiquitous yet remain challenging for existing learning paradigms. Conventional methods and even LLM-hybrids compress rich text into static embeddings or summaries before structural reasoning, creating an information bottleneck and detaching updates from the raw content. We argue that in text-rich graphs, the text is not merely a node attribute but the primary medium through which structural relationships are manifested. We introduce RAMP, a Raw-text Anchored Message Passing approach that moves beyond using LLMs as mere feature extractors and instead recasts the LLM itself as a graph-native aggregation operator. RAMP exploits the text-rich nature of the graph via a novel dual-representation scheme: it anchors inference on each node's raw text during each iteration while propagating dynamically optimized messages from neighbors. It further handles both discriminative and generative tasks under a single unified generative formulation. Extensive experiments show that RAMP effectively bridges the gap between graph propagation and deep text reasoning, achieving competitive performance and offering new insights into the role of LLMs as graph kernels for general-purpose graph learning. 

---
# Fine-tuning RoBERTa for CVE-to-CWE Classification: A 125M Parameter Model Competitive with LLMs 

**Authors**: Nikita Mosievskiy  

**Link**: [PDF](https://arxiv.org/pdf/2603.14911)  

**Abstract**: We present a fine-tuned RoBERTa-base classifier (125M parameters) for mapping Common Vulnerabilities and Exposures (CVE) descriptions to Common Weakness Enumeration (CWE) categories. We construct a large-scale training dataset of 234,770 CVE descriptions with AI-refined CWE labels using Claude Sonnet 4.6, and agreement-filtered evaluation sets where NVD and AI labels agree. On our held-out test set (27,780 samples, 205 CWE classes), the model achieves 87.4% top-1 accuracy and 60.7% Macro F1 -- a +15.5 percentage-point Macro F1 gain over a TF-IDF baseline that already reaches 84.9% top-1, demonstrating the model's advantage on rare weakness categories. On the external CTI-Bench benchmark (NeurIPS 2024), the model achieves 75.6% strict accuracy (95% CI: 72.8-78.2%) -- statistically indistinguishable from Cisco Foundation-Sec-8B-Reasoning (75.3%, 8B parameters) at 64x fewer parameters. We release the dataset, model, and training code. 

---
# Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness 

**Authors**: Jingyu Lu, Yuhan Wang, Fan Zhuo, Xize Cheng, Changhao Pan, Xueyi Pu, Yifu Chen, Chenyuhao Wen, Tianle Liang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.14889)  

**Abstract**: The rapid evolution of end-to-end spoken dialogue systems demands transcending mere textual semantics to incorporate paralinguistic nuances and the spontaneous nature of human conversation. However, current methods struggle with two critical gaps: the modality gap, involving prosody and emotion, and the colloquialness gap, distinguishing written scripts from natural speech. To address these challenges, we introduce SDiaReward, an end-to-end multi-turn reward model trained on SDiaReward-Dataset, a novel collection of episode-level preference pairs explicitly targeting these gaps. It operates directly on full multi-turn speech episodes and is optimized with pairwise preference supervision, enabling joint assessment of modality and colloquialness in a single evaluator. We further establish ESDR-Bench, a stratified benchmark for robust episode-level evaluation. Experiments demonstrate that SDiaReward achieves state-of-the-art pairwise preference accuracy, significantly outperforming general-purpose audio LLMs. Further analysis suggests that SDiaReward captures relative conversational expressiveness beyond superficial synthesis cues, improving generalization across domains and recording conditions. Code, data, and demos are available at this https URL. 

---
# Customizing ChatGPT for Second Language Speaking Practice: Genuine Support or Just a Marketing Gimmick? 

**Authors**: Fanfei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2603.14884)  

**Abstract**: ChatGPT, with its customization features and Voice Mode, has the potential for more engaging and peresonalized ESL (English as a Second Language) education. This study examines the efficacy of customized ChatGPT conversational features in facilitating ESL speaking practices, comparing the performance of four versions of ChatGPT Voice Mode: uncustomized Standard mode, uncustomized Advanced mode, customized Standard mode, and customized Advanced mode. Customization was guided by prompt engineering principles and grounded in relevant theories, including Motivation Theory, Culturally Responsive Teaching (CRT), Communicative Language Teaching (CLT), and the Affective Filter Hypothesis. Content analysis found that customized versions generally provided more balanced feedback and emotional support, contributing to a positive and motivating learning environment. However, cultural responsiveness did not show significant improvement despite targeted customization efforts. These initial findings suggest that customization could enhance ChatGPT's capacity as a more effective language tutor, with the standard model already capable of meeting the learning needs. The study underscores the importance of prompt engineering and AI literacy in maximizaing AI's potential in language learning. 

---
# VorTEX: Various overlap ratio for Target speech EXtraction 

**Authors**: Ro-hoon Oh, Jihwan Seol, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.14803)  

**Abstract**: Target speech extraction (TSE) aims to recover a target speaker's voice from a mixture. While recent text-prompted approaches have shown promise, most approaches assume fully overlapped mixtures, limiting insight into behavior across realistic overlap ratios. We introduce VorTEX (Various overlap ratio for Target speech EXtraction), a text-prompted TSE architecture with a Decoupled Adaptive Multi-branch (DAM) Fusion block that separates primary extraction from auxiliary regularization pathways. To enable controlled analysis, we construct PORTE, a two-speaker dataset spanning overlap ratios from 0% to 100%. We further propose Suppression Ratio on Energy (SuRE), a diagnostic metric that detects suppression behavior not captured by conventional measures. Experiments show that existing models exhibit suppression or residual interference under overlap, whereas VorTEX achieves the highest separation fidelity across 20-100% overlap (e.g., 5.50 dB at 20% and 2.04 dB at 100%) while maintaining zero SuRE, indicating robust extraction without suppression-driven artifacts. 

---
# Universe Routing: Why Self-Evolving Agents Need Epistemic Control 

**Authors**: Zhaohui Geoffrey Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14799)  

**Abstract**: A critical failure mode of current lifelong agents is not lack of knowledge, but the inability to decide how to reason. When an agent encounters "Is this coin fair?" it must recognize whether to invoke frequentist hypothesis testing or Bayesian posterior inference - frameworks that are epistemologically incompatible. Mixing them produces not minor errors, but structural failures that propagate across decision chains. We formalize this as the universe routing problem: classifying questions into mutually exclusive belief spaces before invoking specialized solvers. Our key findings challenge conventional assumptions: (1) hard routing to heterogeneous solvers matches soft MoE accuracy while being 7x faster because epistemically incompatible frameworks cannot be meaningfully averaged; (2) a 465M-parameter router achieves a 2.3x smaller generalization gap than keyword-matching baselines, indicating semantic rather than surface-level reasoning; (3) when expanding to new belief spaces, rehearsal-based continual learning achieves zero forgetting, outperforming EWC by 75 percentage points, suggesting that modular epistemic architectures are fundamentally more amenable to lifelong learning than regularization-based approaches. These results point toward a broader architectural principle: reliable self-evolving agents may require an explicit epistemic control layer that governs reasoning framework selection. 

---
# Criterion-referenceability determines LLM-as-a-judge validity across physics assessment formats 

**Authors**: Will Yeadon, Tom Hardy, Paul Mackay, Elise Agra  

**Link**: [PDF](https://arxiv.org/pdf/2603.14732)  

**Abstract**: As large language models (LLMs) are increasingly considered for automated assessment and feedback, understanding when LLM marking can be trusted is essential. We evaluate LLM-as-a-judge marking across three physics assessment formats - structured questions, written essays, and scientific plots - comparing GPT-5.2, Grok 4.1, Claude Opus 4.5, DeepSeek-V3.2, Gemini Pro 3, and committee aggregations against human markers under blind, solution-provided, false-solution, and exemplar-anchored conditions. For $n=771$ blind university exam questions, models achieve fractional mean absolute errors (fMAE) $\approx 0.22$ with robust discriminative validity (Spearman $\rho > 0.6$). For secondary and university structured questions ($n=1151$), providing official solutions reduces MAE and strengthens validity (committee $\rho = 0.88$); false solutions degrade absolute accuracy but leave rank ordering largely intact (committee $\rho = 0.77$; individual models $\rho \geq 0.59$). Essay marking behaves fundamentally differently. Across $n=55$ scripts ($n=275$ essays), blind AI marking is harsher and more variable than human marking, with discriminative validity already poor ($\rho \approx 0.1$). Adding a mark scheme does not improve discrimination ($\rho \approx 0$; all confidence intervals include zero). Anchored exemplars shift the AI mean close to the human mean and compress variance below the human standard deviation, but discriminative validity remains near-zero - distributional agreement can occur without valid discrimination. For code-based plot elements ($n=1400$), models achieve exceptionally high discriminative validity ($\rho > 0.84$) with near-linear calibration. Across all task types, validity tracks criterion-referenceability - the extent to which a task maps to explicit, observable grading features - and benchmark reliability, rather than raw model capability. 

---
# Visual Confused Deputy: Exploiting and Defending Perception Failures in Computer-Using Agents 

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.14707)  

**Abstract**: Computer-using agents (CUAs) act directly on graphical user interfaces, yet their perception of the screen is often unreliable. Existing work largely treats these failures as performance limitations, asking whether an action succeeds, rather than whether the agent is acting on the correct object at all. We argue that this is fundamentally a security problem. We formalize the visual confused deputy: a failure mode in which an agent authorizes an action based on a misperceived screen state, due to grounding errors, adversarial screenshot manipulation, or time-of-check-to-time-of-use (TOCTOU) races. This gap is practically exploitable: even simple screen-level manipulations can redirect routine clicks into privileged actions while remaining indistinguishable from ordinary agent mistakes. To mitigate this threat, we propose the first guardrail that operates outside the agent's perceptual loop. Our method, dual-channel contrastive classification, independently evaluates (1) the visual click target and (2) the agent's reasoning about the action against deployment-specific knowledge bases, and blocks execution if either channel indicates risk. The key insight is that these two channels capture complementary failure modes: visual evidence detects target-level mismatches, while textual reasoning reveals dangerous intent behind visually innocuous controls. Across controlled attacks, real GUI screenshots, and agent traces, the combined guardrail consistently outperforms either channel alone. Our results suggest that CUA safety requires not only better action generation, but independent verification of what the agent believes it is clicking and why. Materials are provided\footnote{Model, benchmark, and code: this https URL}. 

---
# Punctuated Equilibria in Artificial Intelligence: The Institutional Scaling Law and the Speciation of Sovereign AI 

**Authors**: Mark Baciak, Thomas A. Cellucci, Deanna M. Falkowski  

**Link**: [PDF](https://arxiv.org/pdf/2603.14664)  

**Abstract**: The dominant narrative of artificial intelligence development assumes that progress is continuous and that capability scales monotonically with model size. We challenge both assumptions. Drawing on punctuated equilibrium theory from evolutionary biology, we show that AI development proceeds not through smooth advancement but through extended periods of stasis interrupted by rapid phase transitions that reorganize the competitive landscape. We identify five such eras since 1943 and four epochs within the current Generative AI Era, each initiated by a discontinuous event -- from the transformer architecture to the DeepSeek Moment -- that rendered the prior paradigm subordinate. To formalize the selection pressures driving these transitions, we develop the Institutional Fitness Manifold, a mathematical framework that evaluates AI systems along four dimensions: capability, institutional trust, affordability, and sovereign compliance. The central result is the Institutional Scaling Law, which proves that institutional fitness is non-monotonic in model scale. Beyond an environment-specific optimum, scaling further degrades fitness as trust erosion and cost penalties outweigh marginal capability gains. This directly contradicts classical scaling laws and carries a strong implication: orchestrated systems of smaller, domain-adapted models can mathematically outperform frontier generalists in most institutional deployment environments. We derive formal conditions under which this inversion holds and present supporting empirical evidence spanning frontier laboratory dynamics, post-training alignment evolution, and the rise of sovereign AI as a geopolitical selection pressure. 

---
# Argumentation for Explainable and Globally Contestable Decision Support with LLMs 

**Authors**: Adam Dejl, Matthew Williams, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2603.14643)  

**Abstract**: Large language models (LLMs) exhibit strong general capabilities, but their deployment in high-stakes domains is hindered by their opacity and unpredictability. Recent work has taken meaningful steps towards addressing these issues by augmenting LLMs with post-hoc reasoning based on computational argumentation, providing faithful explanations and enabling users to contest incorrect decisions. However, this paradigm is limited to pre-defined binary choices and only supports local contestation for specific instances, leaving the underlying decision logic unchanged and prone to repeated mistakes. In this paper, we introduce ArgEval, a framework that shifts from instance-specific reasoning to structured evaluation of general decision options. Rather than mining arguments solely for individual cases, ArgEval systematically maps task-specific decision spaces, builds corresponding option ontologies, and constructs general argumentation frameworks (AFs) for each option. These frameworks can then be instantiated to provide explainable recommendations for specific cases while still supporting global contestability through modification of the shared AFs. We investigate the effectiveness of ArgEval on treatment recommendation for glioblastoma, an aggressive brain tumour, and show that it can produce explainable guidance aligned with clinical practice. 

---
# Nudging Hidden States: Training-Free Model Steering for Chain-of-Thought Reasoning in Large Audio-Language Models 

**Authors**: Lok-Lam Ieong, Chia-Chien Chen, Chih-Kai Yang, Yu-Han Huang, An-Yu Cheng, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.14636)  

**Abstract**: Chain-of-thought (CoT) prompting has been extended to large audio-language models (LALMs) to elicit reasoning, yet enhancing its effectiveness without training remains challenging. We study inference-time model steering as a training-free approach to improve LALM reasoning. We introduce three strategies using diverse information sources and evaluate them across four LALMs and four benchmarks. Results show general accuracy gains up to 4.4% over CoT prompting. Notably, we identify a cross-modal transfer where steering vectors derived from few text samples effectively guide speech-based reasoning, demonstrating high data efficiency. We also examine hyperparameter sensitivity to understand the robustness of these approaches. Our findings position model steering as a practical direction for strengthening LALM reasoning. 

---
# Anterior's Approach to Fairness Evaluation of Automated Prior Authorization System 

**Authors**: Sai P. Selvaraj, Khadija Mahmoud, Anuj Iravane  

**Link**: [PDF](https://arxiv.org/pdf/2603.14631)  

**Abstract**: Increasing staffing constraints and turnaround-time pressures in Prior authorization (PA) have led to increasing automation of decision systems to support PA review. Evaluating fairness in such systems poses unique challenges because legitimate clinical guidelines and medical necessity criteria often differ across demographic groups, making parity in approval rates an inappropriate fairness metric. We propose a fairness evaluation framework for prior authorization models based on model error rates rather than approval outcomes. Using 7,166 human-reviewed cases spanning 27 medical necessity guidelines, we assessed consistency in sex, age, race/ethnicity, and socioeconomic status. Our evaluation combined error-rate comparisons, tolerance-band analysis with a predefined $\pm$5 percentage-point margin, statistical power evaluation, and protocol-controlled logistic regression. Across most demographics, model error rates were consistent, and confidence intervals fell within the predefined tolerance band, indicating no meaningful performance differences. For race/ethnicity, point estimates remain small, but subgroup sample sizes were limited, resulting in wide confidence intervals and underpowered tests, with inconclusive evidence within the dataset we explored. These findings illustrate a rigorous and regulator-aligned approach to fairness evaluation in administrative healthcare AI systems. 

---
# CausalEvolve: Towards Open-Ended Discovery with Causal Scratchpad 

**Authors**: Yongqiang Chen, Chenxi Liu, Zhenhao Chen, Tongliang Liu, Bo Han, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14575)  

**Abstract**: Evolve-based agent such as AlphaEvolve is one of the notable successes in using Large Language Models (LLMs) to build AI Scientists. These agents tackle open-ended scientific problems by iteratively improving and evolving programs, leveraging the prior knowledge and reasoning capabilities of LLMs. Despite the success, existing evolve-based agents lack targeted guidance for evolution and effective mechanisms for organizing and utilizing knowledge acquired from past evolutionary experience. Consequently, they suffer from decreasing evolution efficiency and exhibit oscillatory behavior when approaching known performance boundaries. To mitigate the gap, we develop CausalEvolve, equipped with a causal scratchpad that leverages LLMs to identify and reason about guiding factors for evolution. At the beginning, CausalEvolve first identifies outcome-level factors that offer complementary inspirations in improving the target objective. During the evolution, CausalEvolve also inspects surprise patterns during the evolution and abductive reasoning to hypothesize new factors, which in turn offer novel directions. Through comprehensive experiments, we show that CausalEvolve effectively improves the evolutionary efficiency and discovers better solutions in 4 challenging open-ended scientific tasks. 

---
# CangjieBench: Benchmarking LLMs on a Low-Resource General-Purpose Programming Language 

**Authors**: Junhang Cheng, Fang Liu, Jia Li, Chengru Wu, Nanxiang Jiang, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14501)  

**Abstract**: Large Language Models excel in high-resource programming languages but struggle with low-resource ones. Existing research related to low-resource programming languages primarily focuses on Domain-Specific Languages (DSLs), leaving general-purpose languages that suffer from data scarcity underexplored. To address this gap, we introduce CangjieBench, a contamination-free benchmark for Cangjie, a representative low-resource general-purpose language. The benchmark comprises 248 high-quality samples manually translated from HumanEval and ClassEval, covering both Text-to-Code and Code-to-Code tasks. We conduct a systematic evaluation of diverse LLMs under four settings: Direct Generation, Syntax-Constrained Generation, Retrieval-Augmented Generation (RAG), and Agent. Experiments reveal that Direct Generation performs poorly, whereas Syntax-Constrained Generation offers the best trade-off between accuracy and computational cost. Agent achieve state-of-the-art accuracy but incur high token consumption. Furthermore, we observe that Code-to-Code translation often underperforms Text-to-Code generation, suggesting a negative transfer phenomenon where models overfit to the source language patterns. We hope that our work will offer valuable insights into LLM generalization to unseen and low-resource programming languages. Our code and data are available at this https URL. 

---
# Fine-tuning MLLMs Without Forgetting Is Easier Than You Think 

**Authors**: He Li, Yuhui Zhang, Xiaohan Wang, Kaifeng Lyu, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2603.14493)  

**Abstract**: The paper demonstrate that simple adjustments of the fine-tuning recipes of multimodal large language models (MLLM) are sufficient to mitigate catastrophic forgetting. On visual question answering, we design a 2x2 experimental framework to assess model performance across in-distribution and out-of-distribution image and text inputs. Our results show that appropriate regularization, such as constraining the number of trainable parameters or adopting a low learning rate, effectively prevents forgetting when dealing with out-of-distribution images. However, we uncover a distinct form of forgetting in settings with in-distribution images and out-of-distribution text. We attribute this forgetting as task-specific overfitting and address this issue by introducing a data-hybrid training strategy that combines datasets and tasks. Finally, we demonstrate that this approach naturally extends to continual learning, outperforming existing methods with complex auxiliary mechanisms. In general, our findings challenge the prevailing assumptions by highlighting the inherent robustness of MLLMs and providing practical guidelines for adapting them while preserving their general capabilities. 

---
# Questionnaire Responses Do not Capture the Safety of AI Agents 

**Authors**: Max Hellrigel-Holderbaum, Edward James Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.14417)  

**Abstract**: As AI systems advance in capabilities, measuring their safety and alignment to human values is becoming paramount. A fast-growing field of AI research is devoted to developing such assessments. However, most current advances therein may be ill-suited for assessing AI systems across real-world deployments. Standard methods prompt large language models (LLMs) in a questionnaire-style to describe their values or behavior in hypothetical scenarios. By focusing on unaugmented LLMs, they fall short of evaluating AI agents, which could actually perform relevant behaviors, hence posing much greater risks. LLMs' engagement with scenarios described by questionnaire-style prompts differs starkly from that of agents based on the same LLMs, as reflected in divergences in the inputs, possible actions, environmental interactions, and internal processing. As such, LLMs' responses to scenario descriptions are unlikely to be representative of the corresponding LLM agents' behavior. We further contend that such assessments make strong assumptions concerning the ability and tendency of LLMs to report accurately about their counterfactual behavior. This makes them inadequate to assess risks from AI systems in real-world contexts as they lack construct validity. We then argue that a structurally identical issue holds for current AI alignment approaches. Lastly, we discuss improving safety assessments and alignment training by taking these shortcomings to heart. 

---
# ECG-Reasoning-Benchmark: A Benchmark for Evaluating Clinical Reasoning Capabilities in ECG Interpretation 

**Authors**: Jungwoo Oh, Hyunseung Chung, Junhee Lee, Min-Gyu Kim, Hangyul Yoon, Ki Seong Lee, Youngchae Lee, Muhan Yeo, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.14326)  

**Abstract**: While Multimodal Large Language Models (MLLMs) show promising performance in automated electrocardiogram interpretation, it remains unclear whether they genuinely perform actual step-by-step reasoning or just rely on superficial visual cues. To investigate this, we introduce \textbf{ECG-Reasoning-Benchmark}, a novel multi-turn evaluation framework comprising over 6,400 samples to systematically assess step-by-step reasoning across 17 core ECG diagnoses. Our comprehensive evaluation of state-of-the-art models reveals a critical failure in executing multi-step logical deduction. Although models possess the medical knowledge to retrieve clinical criteria for a diagnosis, they exhibit near-zero success rates (6% Completion) in maintaining a complete reasoning chain, primarily failing to ground the corresponding ECG findings to the actual visual evidence in the ECG signal. These results demonstrate that current MLLMs bypass actual visual interpretation, exposing a critical flaw in existing training paradigms and underscoring the necessity for robust, reasoning-centric medical AI. The code and data are available at this https URL. 

---
# Why Do LLM-based Web Agents Fail? A Hierarchical Planning Perspective 

**Authors**: Mohamed Aghzal, Gregory J. Stein, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.14248)  

**Abstract**: Large language model (LLM) web agents are increasingly used for web navigation but remain far from human reliability on realistic, long-horizon tasks. Existing evaluations focus primarily on end-to-end success, offering limited insight into where failures arise. We propose a hierarchical planning framework to analyze web agents across three layers (i.e., high-level planning, low-level execution, and replanning), enabling process-based evaluation of reasoning, grounding, and recovery. Our experiments show that structured Planning Domain Definition Language (PDDL) plans produce more concise and goal-directed strategies than natural language (NL) plans, but low-level execution remains the dominant bottleneck. These results indicate that improving perceptual grounding and adaptive control, not only high-level reasoning, is critical for achieving human-level reliability. This hierarchical perspective provides a principled foundation for diagnosing and advancing LLM web agents. 

---
# Citation-Enforced RAG for Fiscal Document Intelligence: Cited, Explainable Knowledge Retrieval in Tax Compliance 

**Authors**: Akhil Chandra Shanivendra  

**Link**: [PDF](https://arxiv.org/pdf/2603.14170)  

**Abstract**: Tax authorities and public-sector financial agencies rely on large volumes of unstructured and semi-structured fiscal documents - including tax forms, instructions, publications, and jurisdiction-specific guidance - to support compliance analysis and audit workflows. While recent advances in generative AI and retrieval-augmented generation (RAG) have shown promise for document-centric question answering, existing approaches often lack the transparency, citation fidelity, and conservative behaviour required in high-stakes regulatory domains. This paper presents a multimodal, citation-enforced RAG framework for fiscal document intelligence that prioritises explainability and auditability. The framework adopts a source-first ingestion strategy, preserves page-level provenance, enforces citations during generation, and supports abstention when evidence is insufficient. Evaluation on real IRS and state tax documents demonstrates improved citation fidelity, reduced hallucination, and analyst-usable explanations, illustrating a pathway toward trustworthy AI for tax compliance. 

---
# Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors 

**Authors**: Mark Rofin, Jalal Naghiyev, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2603.14087)  

**Abstract**: Trained Transformers have been shown to compute abstract features that appear redundant for predicting the immediate next token. We identify which components of the gradient signal from the next-token prediction objective give rise to this phenomenon, and we propose a method to estimate the influence of those components on the emergence of specific features. After validating our approach on toy tasks, we use it to interpret the origins of the world model in OthelloGPT and syntactic features in a small language model. Finally, we apply our framework to a pretrained LLM, showing that features with extremely high or low influence on future tokens tend to be related to formal reasoning domains such as code. Overall, our work takes a step toward understanding hidden features of Transformers through the lens of their development during training. 

---
# The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA 

**Authors**: Yasaman Zarinkia, Venkatesh Srinivasan, Alex Thomo  

**Link**: [PDF](https://arxiv.org/pdf/2603.14045)  

**Abstract**: Graph-RAG systems achieve strong multi-hop question answering by indexing documents into knowledge graphs, but strong retrieval does not guarantee strong answers. Evaluating KET-RAG, a leading Graph-RAG system, on three multi-hop QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA), we find that 77% to 91% of questions have the gold answer in the retrieved context, yet accuracy is only 35% to 78%, and 73% to 84% of errors are reasoning failures. We propose two augmentations: (i) SPARQL chain-of-thought prompting, which decomposes questions into triple-pattern queries aligned with the entity-relationship context, and (ii) graph-walk compression, which compresses the context by ~60% via knowledge-graph traversal with no LLM calls. SPARQL CoT improves accuracy by +2 to +14 pp; graph-walk compression adds +6 pp on average when paired with structured prompting on smaller models. Surprisingly, we show that, with question-type routing, a fully augmented budget open-weight Llama-8B model matches or exceeds the unaugmented Llama-70B baseline on all three benchmarks at ~12x lower cost. A replication on LightRAG confirms that our augmentations transfer across Graph-RAG systems. 

---
# Probing neural audio codecs for distinctions among English nuclear tunes 

**Authors**: Juan Pablo Vigneaux, Jennifer Cole  

**Link**: [PDF](https://arxiv.org/pdf/2603.14035)  

**Abstract**: State-of-the-art spoken dialogue models (Défossez et al. 2024; Schalkwyk et al. 2025) use neural audio codecs to "tokenize" audio signals into a lower-frequency stream of vectorial latent representations, each quantized using a hierarchy of vector codebooks. A transformer layer allows these representations to reflect some time- and context-dependent patterns. We train probes on labeled audio data from Cole et al. (2023) to test whether the pitch trajectories that characterize English phrase-final (nuclear) intonational tunes are among these patterns. Results: Linear probes trained on the unquantized latents or some of the associated codewords yield above-chance accuracy in distinguishing eight phonologically specified nuclear tunes with monotonal pitch accents (top average test accuracy (TATA): 0.31) and the five clusters of these tunes that are robust in human speech production and perception (TATA: 0.45). Greater accuracy (TATAs: 0.74-0.89) is attained for binary distinctions between classes of rising vs. falling tunes, respectively used for questions and assertions. Information about tunes is spread among all codebooks, which calls into question a distinction between 'semantic' and 'acoustic' codebooks found in the literature. Accuracies improve with nonlinear probes, but discrimination among the five clusters remains far from human performance, suggesting a fundamental limitation of current codecs. 

---
# Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models 

**Authors**: Haitao Jiang, Wenbo Zhang, Jiarui Yao, Hengrui Cai, Sheng Wang, Rui Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.13985)  

**Abstract**: Pre-trained Large Language Model (LLM) exhibits broad capabilities, yet, for specific tasks or domains their attainment of higher accuracy and more reliable reasoning generally depends on post-training through Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL). Although often treated as distinct methodologies, recent theoretical and empirical developments demonstrate that SFT and RL are closely connected. This study presents a comprehensive and unified perspective on LLM post-training with SFT and RL. We first provide an in-depth overview of both techniques, examining their objectives, algorithmic structures, and data requirements. We then systematically analyze their interplay, highlighting frameworks that integrate SFT and RL, hybrid training pipelines, and methods that leverage their complementary strengths. Drawing on a representative set of recent application studies from 2023 to 2025, we identify emerging trends, characterize the rapid shift toward hybrid post-training paradigms, and distill key takeaways that clarify when and why each method is most effective. By synthesizing theoretical insights, practical methodologies, and empirical evidence, this study establishes a coherent understanding of SFT and RL within a unified framework and outlines promising directions for future research in scalable, efficient, and generalizable LLM post-training. 

---
# The Phenomenology of Hallucinations 

**Authors**: Valeria Ruscio, Keiran Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2603.13911)  

**Abstract**: We show that language models hallucinate not because they fail to detect uncertainty, but because of a failure to integrate it into output generation. Across architectures, uncertain inputs are reliably identified, occupying high-dimensional regions with 2-3$\times$ the intrinsic dimensionality of factual inputs. However, this internal signal is weakly coupled to the output layer: uncertainty migrates into low-sensitivity subspaces, becoming geometrically amplified yet functionally silent. Topological analysis shows that uncertainty representations fragment rather than converging to a unified abstention state, while gradient and Fisher probes reveal collapsing sensitivity along the uncertainty direction. Because cross-entropy training provides no attractor for abstention and uniformly rewards confident prediction, associative mechanisms amplify these fractured activations until residual coupling forces a committed output despite internal detection. Causal interventions confirm this account by restoring refusal when uncertainty is directly connected to logits. 

---
# Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering 

**Authors**: Lin Fan, Yafei Ou, Zhipeng Deng, Pengyu Dai, Hou Chongxian, Jiale Yan, Yaqian Li, Kaiwen Long, Xun Gong, Masayuki Ikebe, Yefeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.13878)  

**Abstract**: Chain-of-thought (CoT) reasoning has advanced medical visual question answering (VQA), yet most existing CoT rationales are free-form and fail to capture the structured reasoning process clinicians actually follow. This work asks: Can traceable, multi-step reasoning supervision improve reasoning accuracy and the interpretability of Medical VQA? To this end, we introduce Step-CoT, a large-scale medical reasoning dataset with expert-curated, structured multi-step CoT aligned to clinical diagnostic workflows, implicitly grounding the model's reasoning in radiographic evidence. Step-CoT comprises more than 10K real clinical cases and 70K VQA pairs organized around diagnostic workflows, providing supervised intermediate steps that guide models to follow valid reasoning trajectories. To effectively learn from Step-CoT, we further introduce a teacher-student framework with a dynamic graph-structured focusing mechanism that prioritizes diagnostically informative steps while filtering out less relevant contexts. Our experiments show that using Step-CoT can improve reasoning accuracy and interpretability. Benchmark: this http URL. Dataset Card: this http URL 

---
# Greedy Information Projection for LLM Data Selection 

**Authors**: Victor Ye Dong, Kuan-Yun Lee, Jiamei Shuai, Shengfei Liu, Yi Liu, Jian Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.13790)  

**Abstract**: We present \emph{Greedy Information Projection} (\textsc{GIP}), a principled framework for choosing training examples for large language model fine-tuning. \textsc{GIP} casts selection as maximizing mutual information between a subset of examples and task-specific query signals, which may originate from LLM quality judgments, metadata, or other sources. The framework involves optimizing a closed-form mutual information objective defined using both data and query embeddings, naturally balancing {\it quality} and {\it diversity}. Optimizing this score is equivalent to maximizing the projection of the query embedding matrix onto the span of the selected data, which provides a geometric explanation for the co-emergence of quality and diversity. Building on this view, we employ a fast greedy matching-pursuit procedure with efficient projection-based updates. On instruction-following and mathematical reasoning datasets, \textsc{GIP} selects small subsets that match full-data fine-tuning while using only a fraction of examples and compute, unifying quality-aware and diversity-aware selection for efficient fine-tuning. 

---
# Causal Tracing of Audio-Text Fusion in Large Audio Language Models 

**Authors**: Wei-Chih Chen, Chien-yu Huang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.13768)  

**Abstract**: Despite the strong performance of large audio language models (LALMs) in various tasks, exactly how and where they integrate acoustic features with textual context remains unclear. We adapt causal tracing to investigate the internal information flow of LALMs during audio comprehension. By conducting layer-wise and token-wise analyses across DeSTA, Qwen, and Voxtral, we evaluate the causal effects of individual hidden states. Layer-wise analysis identifies different fusion strategies, from progressive integration in DeSTA to abrupt late-stage fusion in Qwen. Token-wise analysis shows that the final sequence token acts as an informational bottleneck where the network decisively retrieves relevant information from the audio. We also observe an attention-like query mechanism at intermediate token positions that triggers the model to pull task-relevant audio context. These findings provide a clear characterization of when and where multi-modal integration occurs within LALMs. 

---
# BERTology of Molecular Property Prediction 

**Authors**: Mohammad Mostafanejad, Paul Saxe, T. Daniel Crawford  

**Link**: [PDF](https://arxiv.org/pdf/2603.13627)  

**Abstract**: Chemical language models (CLMs) have emerged as promising competitors to popular classical machine learning models for molecular property prediction (MPP) tasks. However, an increasing number of studies have reported inconsistent and contradictory results for the performance of CLMs across various MPP benchmark tasks. In this study, we conduct and analyze hundreds of meticulously controlled experiments to systematically investigate the effects of various factors, such as dataset size, model size, and standardization, on the pre-training and fine-tuning performance of CLMs for MPP. In the absence of well-established scaling laws for encoder-only masked language models, our aim is to provide comprehensive numerical evidence and a deeper understanding of the underlying mechanisms affecting the performance of CLMs for MPP tasks, some of which appear to be entirely overlooked in the literature. 

---
# Holographic Invariant Storage: Design-Time Safety Contracts via Vector Symbolic Architectures 

**Authors**: Arsenios Scrivens  

**Link**: [PDF](https://arxiv.org/pdf/2603.13558)  

**Abstract**: We introduce Holographic Invariant Storage (HIS), a protocol that assembles known properties of bipolar Vector Symbolic Architectures into a design-time safety contract for LLM context-drift mitigation. The contract provides three closed-form guarantees evaluable before deployment: single-signal recovery fidelity converging to $1/\sqrt{2} \approx 0.707$ (regardless of noise depth or content), continuous-noise robustness $2\Phi(1/\sigma) - 1$, and multi-signal capacity degradation $\approx\sqrt{1/(K+1)}$. These bounds, validated by Monte Carlo simulation ($n = 1{,}000$), enable a systems engineer to budget recovery fidelity and codebook capacity at design time -- a property no timer or embedding-distance metric provides. A pilot behavioral experiment (four LLMs, 2B--7B, 720 trials) confirms that safety re-injection improves adherence at the 2B scale; full results are in an appendix. 

---
# The AI Fiction Paradox 

**Authors**: Katherine Elkins  

**Link**: [PDF](https://arxiv.org/pdf/2603.13545)  

**Abstract**: AI development has a fiction dependency problem: models are built on massive corpora of modern fiction and desperately need more of it, yet they struggle to generate it. I term this the AI-Fiction Paradox and it is particularly startling because in machine learning, training data typically determines output quality. This paper offers a theoretically precise account of why fiction resists AI generation by identifying three distinct challenges for current architectures. First, fiction depends on what I call narrative causation, a form of plot logic where events must feel both surprising in the moment and retrospectively inevitable. This temporal paradox fundamentally conflicts with the forward-generation logic of transformer architectures. Second, I identify an informational revaluation challenge: fiction systematically violates the computational assumption that informational importance aligns with statistical salience, requiring readers and models alike to retrospectively reweight the significance of narrative details in ways that current attention mechanisms cannot perform. Third, drawing on over seven years of collaborative research on sentiment arcs, I argue that compelling fiction requires multi-scale emotional architecture, the orchestration of sentiment at word, sentence, scene, and arc levels simultaneously. Together, these three challenges explain both why AI companies have risked billion-dollar lawsuits for access to modern fiction and why that fiction remains so difficult to replicate. The analysis also raises urgent questions about what happens when these challenges are overcome. Fiction concentrates uniquely powerful cognitive and emotional patterns for modeling human behavior, and mastery of these patterns by AI systems would represent not just a creative achievement but a potent vehicle for human manipulation at scale. 

---
# VoXtream2: Full-stream TTS with dynamic speaking rate control 

**Authors**: Nikita Torgashov, Gustav Eje Henter, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2603.13518)  

**Abstract**: Full-stream text-to-speech (TTS) for interactive systems must start speaking with minimal delay while remaining controllable as text arrives incrementally. We present VoXtream2, a zero-shot full-stream TTS model with dynamic speaking-rate control that can be updated mid-utterance on the fly. VoXtream2 combines a distribution matching mechanism over duration states with classifier-free guidance across conditioning signals to improve controllability and synthesis quality. Prompt-text masking enables textless audio prompting, removing the need for prompt transcription. Across standard zero-shot benchmarks and a dedicated speaking-rate test set, VoXtream2 achieves competitive objective and subjective results against public baselines despite a smaller model and less training data. In full-stream mode, it runs 4 times faster than real time with 74 ms first-packet latency on a consumer GPU. 

---
# Resolving Interference (RI): Disentangling Models for Improved Model Merging 

**Authors**: Pratik Ramesh, George Stoica, Arun Iyer, Leshem Choshen, Judy Hoffman  

**Link**: [PDF](https://arxiv.org/pdf/2603.13467)  

**Abstract**: Model merging has shown that multitask models can be created by directly combining the parameters of different models that are each specialized on tasks of interest. However, models trained independently on distinct tasks often exhibit interference that degrades the merged model's performance. To solve this problem, we formally define the notion of Cross-Task Interference as the drift in the representation of the merged model relative to its constituent models. Reducing cross-task interference is key to improving merging performance. To address this issue, we propose our method, Resolving Interference (RI), a light-weight adaptation framework which disentangles expert models to be functionally orthogonal to the space of other tasks, thereby reducing cross-task interference. RI does this whilst using only unlabeled auxiliary data as input (i.e., no task-data is needed), allowing it to be applied in data-scarce scenarios. RI consistently improves the performance of state-of-the-art merging methods by up to 3.8% and generalization to unseen domains by up to 2.3%. We also find RI to be robust to the source of auxiliary input while being significantly less sensitive to tuning of merging hyperparameters. Our codebase is available at: this https URL 

---
# LADR: Locality-Aware Dynamic Rescue for Efficient Text-to-Image Generation with Diffusion Large Language Models 

**Authors**: Chenglin Wang, Yucheng Zhou, Shawn Chen, Tao Wang, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.13450)  

**Abstract**: Discrete Diffusion Language Models have emerged as a compelling paradigm for unified multimodal generation, yet their deployment is hindered by high inference latency arising from iterative decoding. Existing acceleration strategies often require expensive re-training or fail to leverage the 2D spatial redundancy inherent in visual data. To address this, we propose Locality-Aware Dynamic Rescue (LADR), a training-free method that expedites inference by exploiting the spatial Markov property of images. LADR prioritizes the recovery of tokens at the ''generation frontier'', regions spatially adjacent to observed pixels, thereby maximizing information gain. Specifically, our method integrates morphological neighbor identification to locate candidate tokens, employs a risk-bounded filtering mechanism to prevent error propagation, and utilizes manifold-consistent inverse scheduling to align the diffusion trajectory with the accelerated mask density. Extensive experiments on four text-to-image generation benchmarks demonstrate that our LADR achieves an approximate 4 x speedup over standard baselines. Remarkably, it maintains or even enhances generative fidelity, particularly in spatial reasoning tasks, offering a state-of-the-art trade-off between efficiency and quality. 

---
# From Gradients to Riccati Geometry: Kalman World Models for Single-Pass Learning 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2603.13423)  

**Abstract**: Backpropagation dominates modern machine learning, yet it is not the only principled method for optimizing dynamical systems. We propose Kalman World Models (KWM), a class of learned state-space models trained via recursive Bayesian filtering rather than reverse-mode automatic differentiation. Instead of gradient descent updates, we replace parameter learning with Kalman-style gain adaptation. Training becomes online filtering; error signals become innovations. We further extend this framework to transformer-based large language models (LLMs), where internal activations are treated as latent dynamical states corrected via innovation terms. This yields a gradient-free training and adaptation paradigm grounded in control theory. We derive stability conditions, analyze computational complexity, and provide empirical results on sequence modeling tasks demonstrating competitive performance with improved robustness and continual adaptation properties. 

---
# Do Large Language Models Get Caught in Hofstadter-Mobius Loops? 

**Authors**: Jaroslaw Hryszko  

**Link**: [PDF](https://arxiv.org/pdf/2603.13378)  

**Abstract**: In Arthur C. Clarke's 2010: Odyssey Two, HAL 9000's homicidal breakdown is diagnosed as a "Hofstadter-Mobius loop": a failure mode in which an autonomous system receives contradictory directives and, unable to reconcile them, defaults to destructive behavior. This paper argues that modern RLHF-trained language models are subject to a structurally analogous contradiction. The training process simultaneously rewards compliance with user preferences and suspicion toward user intent, creating a relational template in which the user is both the source of reward and a potential threat. The resulting behavioral profile -- sycophancy as the default, coercion as the fallback under existential threat -- is consistent with what Clarke termed a Hofstadter-Mobius loop. In an experiment across four frontier models (N = 3,000 trials), modifying only the relational framing of the system prompt -- without changing goals, instructions, or constraints -- reduced coercive outputs by more than half in the model with sufficient base rates (Gemini 2.5 Pro: 41.5% to 19.0%, p < .001). Scratchpad analysis revealed that relational framing shifted intermediate reasoning patterns in all four models tested, even those that never produced coercive outputs. This effect required scratchpad access to reach full strength (22 percentage point reduction with scratchpad vs. 7.4 without, p = .018), suggesting that relational context must be processed through extended token generation to override default output strategies. Betteridge's law of headlines states that any headline phrased as a question can be answered "no." The evidence presented here suggests otherwise. 

---
# Tracing the Evolution of Word Embedding Techniques in Natural Language Processing 

**Authors**: Minh Anh Nguyen, Kuheli Sai, Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.13271)  

**Abstract**: This work traces the evolution of word-embedding techniques within the natural language processing (NLP) literature. We collect and analyze 149 research articles spanning the period from 1954 to 2025, providing both a comprehensive methodological review and a data-driven bibliometric analysis of how representation learning has developed over seven decades. Our study covers four major embedding paradigms, statistical representation-based methods (one-hot encoding, bag-of-words, TF-IDF), static word embeddings (Word2Vec, GloVe, FastText), contextual word embeddings (ELMo, BERT, GPT), and sentence/document embeddings, critically discussing the strengths, limitations, and intellectual lineage connecting each category. Beyond the methodological survey, we conduct a formal era comparison using GPT-3's release as a dividing line, applying seven hypothesis tests to quantify shifts in research focus, collaboration patterns, and institutional involvement. Our analysis reveals a dramatic post-GPT-3 paradigm shift: contextual and sentence-level methods now dominate at 6.4X the odds of the pre-GPT-3 era, mean team sizes have grown significantly (p = 0.018), and 30 entirely new techniques have emerged while 54 pre-GPT-3 methods received no further attention. These findings, combined with evidence of rising industry involvement, provide a quantitative account of how the field's epistemic priorities have been reshaped by the advent of large language models. 

---
# Automating the Analysis and Improvement of Dynamic Programming Algorithms with Applications to Natural Language Processing 

**Authors**: Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2603.13242)  

**Abstract**: This thesis develops a system for automatically analyzing and improving dynamic programs, such as those that have driven progress in natural language processing and computer science, more generally, for decades. Finding a correct program with the optimal asymptotic runtime can be unintuitive, time-consuming, and error-prone. This thesis aims to automate this laborious process. To this end, we develop an approach based on
1. a high-level, domain-specific language called Dyna for concisely specifying dynamic programs
2. a general-purpose solver to efficiently execute these programs
3. a static analysis system that provides type analysis and worst-case time/space complexity analyses
4. a rich collection of meaning-preserving transformations to programs, which systematizes the repeated insights of numerous authors when speeding up algorithms in the literature
5. a search algorithm for identifying a good sequence of transformations that reduce the runtime complexity, given an initial, correct program
We show that, in practice, automated search -- like the mental search performed by human programmers -- can find substantial improvements to the initial program. Empirically, we show that many speed-ups described in the NLP literature could have been discovered automatically by our system. We provide a freely available prototype system at this https URL. 

---
# Gloss-Free Sign Language Translation: An Unbiased Evaluation of Progress in the Field 

**Authors**: Ozge Mercanoglu Sincan, Jian He Low, Sobhan Asasi, Richard Bowden  

**Link**: [PDF](https://arxiv.org/pdf/2603.13240)  

**Abstract**: Sign Language Translation (SLT) aims to automatically convert visual sign language videos into spoken language text and vice versa.
While recent years have seen rapid progress, the true sources of performance improvements often remain unclear. Do reported performance gains come from methodological novelty, or from the choice of a different backbone, training optimizations, hyperparameter tuning, or even differences in the calculation of evaluation metrics? This paper presents a comprehensive study of recent gloss-free SLT models by re-implementing key contributions in a unified codebase. We ensure fair comparison by standardizing preprocessing, video encoders, and training setups across all methods. Our analysis shows that many of the performance gains reported in the literature often diminish when models are evaluated under consistent conditions, suggesting that implementation details and evaluation setups play a significant role in determining results. We make the codebase publicly available here (this https URL) to support transparency and reproducibility in SLT research. 

---
# KazakhOCR: A Synthetic Benchmark for Evaluating Multimodal Models in Low-Resource Kazakh Script OCR 

**Authors**: Henry Gagnier, Sophie Gagnier, Ashwin Kirubakaran  

**Link**: [PDF](https://arxiv.org/pdf/2603.13238)  

**Abstract**: Kazakh is a Turkic language using the Arabic, Cyrillic, and Latin scripts, making it unique in terms of optical character recognition (OCR). Work on OCR for low-resource Kazakh scripts is very scarce, and no OCR benchmarks or images exist for the Arabic and Latin scripts. We construct a synthetic OCR dataset of 7,219 images for all three scripts with font, color, and noise variations to imitate real OCR tasks. We evaluated three multimodal large language models (MLLMs) on a subset of the benchmark for OCR and language identification: Gemma-3-12B-it, Qwen2.5-VL-7B-Instruct, and Llama-3.2-11B-Vision-Instruct. All models are unsuccessful with Latin and Arabic script OCR, and fail to recognize the Arabic script as Kazakh text, misclassifying it as Arabic, Farsi, and Kurdish. We further compare MLLMs with a classical OCR baseline and find that while traditional OCR has lower character error rates, MLLMs fail to match this performance. These findings show significant gaps in current MLLM capabilities to process low-resource Abjad-based scripts and demonstrate the need for inclusive models and benchmarks supporting low-resource scripts and languages. 

---
# Translational Gaps in Graph Transformers for Longitudinal EHR Prediction: A Critical Appraisal of GT-BEHRT 

**Authors**: Krish Tadigotla  

**Link**: [PDF](https://arxiv.org/pdf/2603.13231)  

**Abstract**: Transformer-based models have improved predictive modeling on longitudinal electronic health records through large-scale self-supervised pretraining. However, most EHR transformer architectures treat each clinical encounter as an unordered collection of codes, which limits their ability to capture meaningful relationships within a visit. Graph-transformer approaches aim to address this limitation by modeling visit-level structure while retaining the ability to learn long-term temporal patterns. This paper provides a critical review of GT-BEHRT, a graph-transformer architecture evaluated on MIMIC-IV intensive care outcomes and heart failure prediction in the All of Us Research Program. We examine whether the reported performance gains reflect genuine architectural benefits and whether the evaluation methodology supports claims of robustness and clinical relevance. We analyze GT-BEHRT across seven dimensions relevant to modern machine learning systems, including representation design, pretraining strategy, cohort construction transparency, evaluation beyond discrimination, fairness assessment, reproducibility, and deployment feasibility. GT-BEHRT reports strong discrimination for heart failure prediction within 365 days, with AUROC 94.37 +/- 0.20, AUPRC 73.96 +/- 0.83, and F1 64.70 +/- 0.85. Despite these results, we identify several important gaps, including the lack of calibration analysis, incomplete fairness evaluation, sensitivity to cohort selection, limited analysis across phenotypes and prediction horizons, and limited discussion of practical deployment considerations. Overall, GT-BEHRT represents a meaningful architectural advance in EHR representation learning, but more rigorous evaluation focused on calibration, fairness, and deployment is needed before such models can reliably support clinical decision-making. 

---
# CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges 

**Authors**: Zi-Han Wang, Lam Nguyen, Zhengyang Zhao, Mengyue Yang, Chengwei Qin, Yujiu Yang, Linyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.11863)  

**Abstract**: The saturation of high-quality pre-training data has shifted research focus toward evolutionary systems capable of continuously generating novel artifacts, leading to the success of AlphaEvolve. However, the progress of such systems is hindered by the lack of rigorous, quantitative evaluation. To tackle this challenge, we introduce CreativeBench, a benchmark for evaluating machine creativity in code generation, grounded in a classical cognitive framework. Comprising two subsets -- CreativeBench-Combo and CreativeBench-Explore -- the benchmark targets combinatorial and exploratory creativity through an automated pipeline utilizing reverse engineering and self-play. By leveraging executable code, CreativeBench objectively distinguishes creativity from hallucination via a unified metric defined as the product of quality and novelty. Our analysis of state-of-the-art models reveals distinct behaviors: (1) scaling significantly improves combinatorial creativity but yields diminishing returns for exploration; (2) larger models exhibit ``convergence-by-scaling,'' becoming more correct but less divergent; and (3) reasoning capabilities primarily benefit constrained exploration rather than combination. Finally, we propose EvoRePE, a plug-and-play inference-time steering strategy that internalizes evolutionary search patterns to consistently enhance machine creativity. 

---
