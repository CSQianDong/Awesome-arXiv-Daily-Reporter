# On-Policy Context Distillation for Language Models 

**Authors**: Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.12275)  

**Abstract**: Context distillation enables language models to internalize in-context knowledge into their parameters. In our work, we propose On-Policy Context Distillation (OPCD), a framework that bridges on-policy distillation with context distillation by training a student model on its own generated trajectories while minimizing reverse Kullback-Leibler divergence against a context-conditioned teacher. We demonstrate the effectiveness of OPCD on two important applications: experiential knowledge distillation, where models extract and consolidate transferable knowledge from their historical solution traces, and system prompt distillation, where models internalize beneficial behaviors encoded in optimized prompts. Across mathematical reasoning, text-based games, and domain-specific tasks, OPCD consistently outperforms baseline methods, achieving higher task accuracy while better preserving out-of-distribution capabilities. We further show that OPCD enables effective cross-size distillation, where smaller student models can internalize experiential knowledge from larger teachers. 

---
# T3D: Few-Step Diffusion Language Models via Trajectory Self-Distillation with Direct Discriminative Optimization 

**Authors**: Tunyu Zhang, Xinxi Zhang, Ligong Han, Haizhou Shi, Xiaoxiao He, Zhuowei Li, Hao Wang, Kai Xu, Akash Srivastava, Hao Wang, Vladimir Pavlovic, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2602.12262)  

**Abstract**: Diffusion large language models (DLLMs) have the potential to enable fast text generation by decoding multiple tokens in parallel. However, in practice, their inference efficiency is constrained by the need for many refinement steps, while aggressively reducing the number of steps leads to a substantial degradation in generation quality. To alleviate this, we propose a trajectory self-distillation framework that improves few-step decoding by distilling the model's own generative trajectories. We incorporate Direct Discriminative Optimization (DDO), a reverse-KL objective that promotes mode-seeking distillation and encourages the student to concentrate on high-probability teacher modes. Across benchmarks, our approach consistently outperforms strong few-step baselines and standard training under tight step budgets. Although full-step decoding remains superior, we substantially narrow the gap, establishing a strong foundation towards practical few-step DLLMs. The source code is available at this https URL. 

---
# A technical curriculum on language-oriented artificial intelligence in translation and specialised communication 

**Authors**: Ralph Krüger  

**Link**: [PDF](https://arxiv.org/pdf/2602.12251)  

**Abstract**: This paper presents a technical curriculum on language-oriented artificial intelligence (AI) in the language and translation (L&T) industry. The curriculum aims to foster domain-specific technical AI literacy among stakeholders in the fields of translation and specialised communication by exposing them to the conceptual and technical/algorithmic foundations of modern language-oriented AI in an accessible way. The core curriculum focuses on 1) vector embeddings, 2) the technical foundations of neural networks, 3) tokenization and 4) transformer neural networks. It is intended to help users develop computational thinking as well as algorithmic awareness and algorithmic agency, ultimately contributing to their digital resilience in AI-driven work environments. The didactic suitability of the curriculum was tested in an AI-focused MA course at the Institute of Translation and Multilingual Communication at TH Koeln. Results suggest the didactic effectiveness of the curriculum, but participant feedback indicates that it should be embedded into higher-level didactic scaffolding - e.g., in the form of lecturer support - in order to enable optimal learning conditions. 

---
# Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications 

**Authors**: Manjunath Kudlur, Evan King, James Wang, Pete Warden  

**Link**: [PDF](https://arxiv.org/pdf/2602.12241)  

**Abstract**: Latency-critical speech applications (e.g., live transcription, voice commands, and real-time translation) demand low time-to-first-token (TTFT) and high transcription accuracy, particularly on resource-constrained edge devices. Full-attention Transformer encoders remain a strong accuracy baseline for automatic speech recognition (ASR) because every frame can directly attend to every other frame, which resolves otherwise locally ambiguous acoustics using distant lexical context. However, this global dependency incurs quadratic complexity in sequence length, inducing an inherent "encode-the-whole-utterance" latency profile. For streaming use cases, this causes TTFT to grow linearly with utterance length as the encoder must process the entire prefix before any decoder token can be emitted. To better meet the needs of on-device, streaming ASR use cases we introduce Moonshine v2, an ergodic streaming-encoder ASR model that employs sliding-window self-attention to achieve bounded, low-latency inference while preserving strong local context. Our models achieve state of the art word error rates across standard benchmarks, attaining accuracy on-par with models 6x their size while running significantly faster. These results demonstrate that carefully designed local attention is competitive with the accuracy of full attention at a fraction of the size and latency cost, opening new possibilities for interactive speech interfaces on edge devices. 

---
# Detecting Overflow in Compressed Token Representations for Retrieval-Augmented Generation 

**Authors**: Julia Belikova, Danila Rozhevskii, Dennis Svirin, Konstantin Polev, Alexander Panchenko  

**Link**: [PDF](https://arxiv.org/pdf/2602.12235)  

**Abstract**: Efficient long-context processing remains a crucial challenge for contemporary large language models (LLMs), especially in resource-constrained environments. Soft compression architectures promise to extend effective context length by replacing long token sequences with smaller sets of learned compressed tokens. Yet, the limits of compressibility -- and when compression begins to erase task-relevant content -- remain underexplored. In this paper, we define \emph{token overflow} as a regime in which compressed representations no longer contain sufficient information to answer a given query, and propose a methodology to characterize and detect it. In the xRAG soft-compression setting, we find that query-agnostic saturation statistics reliably separate compressed from uncompressed token representations, providing a practical tool for identifying compressed tokens but showing limited overflow detection capability. Lightweight probing classifiers over both query and context xRAG representations detect overflow with 0.72 AUC-ROC on average on HotpotQA, SQuADv2, and TriviaQA datasets, demonstrating that incorporating query information improves detection performance. These results advance from query-independent diagnostics to query-aware detectors, enabling low-cost pre-LLM gating to mitigate compression-induced errors. 

---
# ExStrucTiny: A Benchmark for Schema-Variable Structured Information Extraction from Document Images 

**Authors**: Mathieu Sibue, Andres Muñoz Garza, Samuel Mensah, Pranav Shetty, Zhiqiang Ma, Xiaomo Liu, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2602.12203)  

**Abstract**: Enterprise documents, such as forms and reports, embed critical information for downstream applications like data archiving, automated workflows, and analytics. Although generalist Vision Language Models (VLMs) perform well on established document understanding benchmarks, their ability to conduct holistic, fine-grained structured extraction across diverse document types and flexible schemas is not well studied. Existing Key Entity Extraction (KEE), Relation Extraction (RE), and Visual Question Answering (VQA) datasets are limited by narrow entity ontologies, simple queries, or homogeneous document types, often overlooking the need for adaptable and structured extraction. To address these gaps, we introduce ExStrucTiny, a new benchmark dataset for structured Information Extraction (IE) from document images, unifying aspects of KEE, RE, and VQA. Built through a novel pipeline combining manual and synthetic human-validated samples, ExStrucTiny covers more varied document types and extraction scenarios. We analyze open and closed VLMs on this benchmark, highlighting challenges such as schema adaptation, query under-specification, and answer localization. We hope our work provides a bedrock for improving generalist models for structured IE in documents. 

---
# Visual Reasoning Benchmark: Evaluating Multimodal LLMs on Classroom-Authentic Visual Problems from Primary Education 

**Authors**: Mohamed Huti, Alasdair Mackintosh, Amy Waldock, Dominic Andrews, Maxime Lelièvre, Moritz Boos, Tobias Murray, Paul Atherton, Robin A. A. Ince, Oliver G. B. Garrod  

**Link**: [PDF](https://arxiv.org/pdf/2602.12196)  

**Abstract**: AI models have achieved state-of-the-art results in textual reasoning; however, their ability to reason over spatial and relational structures remains a critical bottleneck -- particularly in early-grade maths, which relies heavily on visuals. This paper introduces the visual reasoning benchmark (VRB), a novel dataset designed to evaluate Multimodal Large Language Models (MLLMs) on their ability to solve authentic visual problems from classrooms. This benchmark is built on a set of 701 questions sourced from primary school examinations in Zambia and India, which cover a range of tasks such as reasoning by analogy, pattern completion, and spatial matching. We outline the methodology and development of the benchmark which intentionally uses unedited, minimal-text images to test if models can meet realistic needs of primary education. Our findings reveal a ``jagged frontier'' of capability where models demonstrate better proficiency in static skills such as counting and scaling, but reach a distinct ``spatial ceiling'' when faced with dynamic operations like folding, reflection, and rotation. These weaknesses pose a risk for classroom use on visual reasoning problems, with the potential for incorrect marking, false scaffolding, and reinforcing student misconceptions. Consequently, education-focused benchmarks like the VRB are essential for determining the functional boundaries of multimodal tools used in classrooms. 

---
# Query-focused and Memory-aware Reranker for Long Context Processing 

**Authors**: Yuqing Li, Jiangnan Li, Mo Yu, Guoxuan Ding, Zheng Lin, Weiping Wang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.12192)  

**Abstract**: Built upon the existing analysis of retrieval heads in large language models, we propose an alternative reranking framework that trains models to estimate passage-query relevance using the attention scores of selected heads. This approach provides a listwise solution that leverages holistic information within the entire candidate shortlist during ranking. At the same time, it naturally produces continuous relevance scores, enabling training on arbitrary retrieval datasets without requiring Likert-scale supervision. Our framework is lightweight and effective, requiring only small-scale models (e.g., 4B parameters) to achieve strong performance. Extensive experiments demonstrate that our method outperforms existing state-of-the-art pointwise and listwise rerankers across multiple domains, including Wikipedia and long narrative datasets. It further establishes a new state-of-the-art on the LoCoMo benchmark that assesses the capabilities of dialogue understanding and memory usage. We further demonstrate that our framework supports flexible extensions. For example, augmenting candidate passages with contextual information further improves ranking accuracy, while training attention heads from middle layers enhances efficiency without sacrificing performance. 

---
# dVoting: Fast Voting for dLLMs 

**Authors**: Sicheng Feng, Zigeng Chen, Xinyin Ma, Gongfan Fang, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12153)  

**Abstract**: Diffusion Large Language Models (dLLMs) represent a new paradigm beyond autoregressive modeling, offering competitive performance while naturally enabling a flexible decoding process. Specifically, dLLMs can generate tokens at arbitrary positions in parallel, endowing them with significant potential for parallel test-time scaling, which was previously constrained by severe inefficiency in autoregressive modeling. In this work, we introduce dVoting, a fast voting technique that boosts reasoning capability without training, with only an acceptable extra computational overhead. dVoting is motivated by the observation that, across multiple samples for the same prompt, token predictions remain largely consistent, whereas performance is determined by a small subset of tokens exhibiting cross-sample variability. Leveraging the arbitrary-position generation capability of dLLMs, dVoting performs iterative refinement by sampling, identifying uncertain tokens via consistency analysis, regenerating them through voting, and repeating this process until convergence. Extensive evaluations demonstrate that dVoting consistently improves performance across various benchmarks. It achieves gains of 6.22%-7.66% on GSM8K, 4.40%-7.20% on MATH500, 3.16%-14.84% on ARC-C, and 4.83%-5.74% on MMLU. Our code is available at this https URL 

---
# CitiLink-Minutes: A Multilayer Annotated Dataset of Municipal Meeting Minutes 

**Authors**: Ricardo Campos, Ana Filipa Pacheco, Ana Luísa Fernandes, Inês Cantante, Rute Rebouças, Luís Filipe Cunha, José Miguel Isidro, José Pedro Evans, Miguel Marques, Rodrigo Batista, Evelin Amorim, Alípio Jorge, Nuno Guimarães, Sérgio Nunes, António Leal, Purificação Silvano  

**Link**: [PDF](https://arxiv.org/pdf/2602.12137)  

**Abstract**: City councils play a crucial role in local governance, directly influencing citizens' daily lives through decisions made during municipal meetings. These deliberations are formally documented in meeting minutes, which serve as official records of discussions, decisions, and voting outcomes. Despite their importance, municipal meeting records have received little attention in Information Retrieval (IR) and Natural Language Processing (NLP), largely due to the lack of annotated datasets, which ultimately limit the development of computational models. To address this gap, we introduce CitiLink-Minutes, a multilayer dataset of 120 European Portuguese municipal meeting minutes from six municipalities. Unlike prior annotated datasets of parliamentary or video records, CitiLink-Minutes provides multilayer annotations and structured linkage of official written minutes. The dataset contains over one million tokens, with all personal identifiers de-identified. Each minute was manually annotated by two trained annotators and curated by an experienced linguist across three complementary dimensions: (1) metadata, (2) subjects of discussion, and (3) voting outcomes, totaling over 38,000 individual annotations. Released under FAIR principles and accompanied by baseline results on metadata extraction, topic classification, and vote labeling, CitiLink-Minutes demonstrates its potential for downstream NLP and IR tasks, while promoting transparent access to municipal decisions. 

---
# WavBench: Benchmarking Reasoning, Colloquialism, and Paralinguistics for End-to-End Spoken Dialogue Models 

**Authors**: Yangzhuo Li, Shengpeng Ji, Yifu Chen, Tianle Liang, Haorong Ying, Yule Wang, Junbo Li, Jun Fang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.12135)  

**Abstract**: With the rapid integration of advanced reasoning capabilities into spoken dialogue models, the field urgently demands benchmarks that transcend simple interactions to address real-world complexity. However, current evaluations predominantly adhere to text-generation standards, overlooking the unique audio-centric characteristics of paralinguistics and colloquialisms, alongside the cognitive depth required by modern agents. To bridge this gap, we introduce WavBench, a comprehensive benchmark designed to evaluate realistic conversational abilities where prior works fall short. Uniquely, WavBench establishes a tripartite framework: 1) Pro subset, designed to rigorously challenge reasoning-enhanced models with significantly increased difficulty; 2) Basic subset, defining a novel standard for spoken colloquialism that prioritizes "listenability" through natural vocabulary, linguistic fluency, and interactive rapport, rather than rigid written accuracy; and 3) Acoustic subset, covering explicit understanding, generation, and implicit dialogue to rigorously evaluate comprehensive paralinguistic capabilities within authentic real-world scenarios. Through evaluating five state-of-the-art models, WavBench offers critical insights into the intersection of complex problem-solving, colloquial delivery, and paralinguistic fidelity, guiding the evolution of robust spoken dialogue models. The benchmark dataset and evaluation toolkit are available at this https URL. 

---
# A Rule-based Computational Model for Gaidhlig Morphology 

**Authors**: Peter J Barclay  

**Link**: [PDF](https://arxiv.org/pdf/2602.12132)  

**Abstract**: Language models and software tools are essential to support the continuing vitality of lesser-used languages; however, currently popular neural models require considerable data for training, which normally is not available for such low-resource languages. This paper describes work-in-progress to construct a rule-based model of Gaidhlig morphology using data from Wiktionary, arguing that rule-based systems effectively leverage limited sample data, support greater interpretability, and provide insights useful in the design of teaching materials. The use of SQL for querying the occurrence of different lexical patterns is investigated, and a declarative rule-base is presented that allows Python utilities to derive inflected forms of Gaidhlig words. This functionality could be used to support educational tools that teach or explain language patterns, for example, or to support higher level tools such as rule-based dependency parsers. This approach adds value to the data already present in Wiktionary by adapting it to new use-cases. 

---
# P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling 

**Authors**: Pinyi Zhang, Ting-En Lin, Yuchuan Wu, Jingyang Chen, Zongqi Wang, Hua Yang, Ze Xu, Fei Huang, Kai Zhang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.12116)  

**Abstract**: Personalized alignment of large language models seeks to adapt responses to individual user preferences, typically via reinforcement learning. A key challenge is obtaining accurate, user-specific reward signals in open-ended scenarios. Existing personalized reward models face two persistent limitations: (1) oversimplifying diverse, scenario-specific preferences into a small, fixed set of evaluation principles, and (2) struggling with generalization to new users with limited feedback. To this end, we propose P-GenRM, the first Personalized Generative Reward Model with test-time user-based scaling. P-GenRM transforms preference signals into structured evaluation chains that derive adaptive personas and scoring rubrics across various scenarios. It further clusters users into User Prototypes and introduces a dual-granularity scaling mechanism: at the individual level, it adaptively scales and aggregates each user's scoring scheme; at the prototype level, it incorporates preferences from similar users. This design mitigates noise in inferred preferences and enhances generalization to unseen users through prototype-based transfer. Empirical results show that P-GenRM achieves state-of-the-art results on widely-used personalized reward model benchmarks, with an average improvement of 2.31%, and demonstrates strong generalization on an out-of-distribution dataset. Notably, Test-time User-based scaling provides an additional 3% boost, demonstrating stronger personalized alignment with test-time scalability. 

---
# DeepSight: An All-in-One LM Safety Toolkit 

**Authors**: Bo Zhang, Jiaxuan Guo, Lijun Li, Dongrui Liu, Sujin Chen, Guanxu Chen, Zhijie Zheng, Qihao Lin, Lewen Yan, Chen Qian, Yijin Zhou, Yuyao Wu, Shaoxiong Guo, Tianyi Du, Jingyi Yang, Xuhao Hu, Ziqi Miao, Xiaoya Lu, Jing Shao, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12092)  

**Abstract**: As the development of Large Models (LMs) progresses rapidly, their safety is also a priority. In current Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) safety workflow, evaluation, diagnosis, and alignment are often handled by separate tools. Specifically, safety evaluation can only locate external behavioral risks but cannot figure out internal root causes. Meanwhile, safety diagnosis often drifts from concrete risk scenarios and remains at the explainable level. In this way, safety alignment lack dedicated explanations of changes in internal mechanisms, potentially degrading general capabilities. To systematically address these issues, we propose an open-source project, namely DeepSight, to practice a new safety evaluation-diagnosis integrated paradigm. DeepSight is low-cost, reproducible, efficient, and highly scalable large-scale model safety evaluation project consisting of a evaluation toolkit DeepSafe and a diagnosis toolkit DeepScan. By unifying task and data protocols, we build a connection between the two stages and transform safety evaluation from black-box to white-box insight. Besides, DeepSight is the first open source toolkit that support the frontier AI risk evaluation and joint safety evaluation and diagnosis. 

---
# Composition-RL: Compose Your Verifiable Prompts for Reinforcement Learning of Large Language Models 

**Authors**: Xin Xu, Clive Bai, Kai Yang, Tianhao Chen, Yangkun Chen, Weijie Liu, Hao Chen, Yang Wang, Saiyong Yang, Can Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12036)  

**Abstract**: Large-scale verifiable prompts underpin the success of Reinforcement Learning with Verifiable Rewards (RLVR), but they contain many uninformative examples and are costly to expand further. Recent studies focus on better exploiting limited training data by prioritizing hard prompts whose rollout pass rate is 0. However, easy prompts with a pass rate of 1 also become increasingly prevalent as training progresses, thereby reducing the effective data size. To mitigate this, we propose Composition-RL, a simple yet useful approach for better utilizing limited verifiable prompts targeting pass-rate-1 prompts. More specifically, Composition-RL automatically composes multiple problems into a new verifiable question and uses these compositional prompts for RL training. Extensive experiments across model sizes from 4B to 30B show that Composition-RL consistently improves reasoning capability over RL trained on the original dataset. Performance can be further boosted with a curriculum variant of Composition-RL that gradually increases compositional depth over training. Additionally, Composition-RL enables more effective cross-domain RL by composing prompts drawn from different domains. Codes, datasets, and models are available at this https URL. 

---
# Disentangling Ambiguity from Instability in Large Language Models: A Clinical Text-to-SQL Case Study 

**Authors**: Angelo Ziletti, Leonardo D'Ambrosi  

**Link**: [PDF](https://arxiv.org/pdf/2602.12015)  

**Abstract**: Deploying large language models for clinical Text-to-SQL requires distinguishing two qualitatively different causes of output diversity: (i) input ambiguity that should trigger clarification, and (ii) model instability that should trigger human review. We propose CLUES, a framework that models Text-to-SQL as a two-stage process (interpretations --> answers) and decomposes semantic uncertainty into an ambiguity score and an instability score. The instability score is computed via the Schur complement of a bipartite semantic graph matrix. Across AmbigQA/SituatedQA (gold interpretations) and a clinical Text-to-SQL benchmark (known interpretations), CLUES improves failure prediction over state-of-the-art Kernel Language Entropy. In deployment settings, it remains competitive while providing a diagnostic decomposition unavailable from a single score. The resulting uncertainty regimes map to targeted interventions - query refinement for ambiguity, model improvement for instability. The high-ambiguity/high-instability regime contains 51% of errors while covering 25% of queries, enabling efficient triage. 

---
# LaCy: What Small Language Models Can and Should Learn is Not Just a Question of Loss 

**Authors**: Szilvia Ujváry, Louis Béthune, Pierre Ablin, João Monteiro, Marco Cuturi, Michael Kirchhof  

**Link**: [PDF](https://arxiv.org/pdf/2602.12005)  

**Abstract**: Language models have consistently grown to compress more world knowledge into their parameters, but the knowledge that can be pretrained into them is upper-bounded by their parameter size. Especially the capacity of Small Language Models (SLMs) is limited, leading to factually incorrect generations. This problem is often mitigated by giving the SLM access to an outside source: the ability to query a larger model, documents, or a database. Under this setting, we study the fundamental question of \emph{which tokens an SLM can and should learn} during pretraining, versus \emph{which ones it should delegate} via a \texttt{<CALL>} token. We find that this is not simply a question of loss: although the loss is predictive of whether a predicted token mismatches the ground-truth, some tokens are \emph{acceptable} in that they are truthful alternative continuations of a pretraining document, and should not trigger a \texttt{<CALL>} even if their loss is high. We find that a spaCy grammar parser can help augment the loss signal to decide which tokens the SLM should learn to delegate to prevent factual errors and which are safe to learn and predict even under high losses. We propose LaCy, a novel pretraining method based on this token selection philosophy. Our experiments demonstrate that LaCy models successfully learn which tokens to predict and where to delegate for help. This results in higher FactScores when generating in a cascade with a bigger model and outperforms Rho or LLM-judge trained SLMs, while being simpler and cheaper. 

---
# Automatic Simplification of Common Vulnerabilities and Exposures Descriptions 

**Authors**: Varpu Vehomäki, Kimmo K. Kaski  

**Link**: [PDF](https://arxiv.org/pdf/2602.11982)  

**Abstract**: Understanding cyber security is increasingly important for individuals and organizations. However, a lot of information related to cyber security can be difficult to understand to those not familiar with the topic. In this study, we focus on investigating how large language models (LLMs) could be utilized in automatic text simplification (ATS) of Common Vulnerability and Exposure (CVE) descriptions. Automatic text simplification has been studied in several contexts, such as medical, scientific, and news texts, but it has not yet been studied to simplify texts in the rapidly changing and complex domain of cyber security. We created a baseline for cyber security ATS and a test dataset of 40 CVE descriptions, evaluated by two groups of cyber security experts in two survey rounds. We have found that while out-of-the box LLMs can make the text appear simpler, they struggle with meaning preservation. Code and data are available at this https URL\_nmi. 

---
# DHPLT: large-scale multilingual diachronic corpora and word representations for semantic change modelling 

**Authors**: Mariia Fedorova, Andrey Kutuzov, Khonzoda Umarova  

**Link**: [PDF](https://arxiv.org/pdf/2602.11968)  

**Abstract**: In this resource paper, we present DHPLT, an open collection of diachronic corpora in 41 diverse languages. DHPLT is based on the web-crawled HPLT datasets; we use web crawl timestamps as the approximate signal of document creation time. The collection covers three time periods: 2011-2015, 2020-2021 and 2024-present (1 million documents per time period for each language). We additionally provide pre-computed word type and token embeddings and lexical substitutions for our chosen target words, while at the same time leaving it open for the other researchers to come up with their own target words using the same datasets. DHPLT aims at filling in the current lack of multilingual diachronic corpora for semantic change modelling (beyond a dozen of high-resource languages). It opens the way for a variety of new experimental setups in this field. All the resources described in this paper are available at this https URL, sorted by language. 

---
# Scaling Model and Data for Multilingual Machine Translation with Open Large Language Models 

**Authors**: Yuzhe Shang, Pengzhi Gao, Wei Liu, Jian Luan, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2602.11961)  

**Abstract**: Open large language models (LLMs) have demonstrated improving multilingual capabilities in recent years. In this paper, we present a study of open LLMs for multilingual machine translation (MT) across a range of languages, and investigate the effects of model scaling and data scaling when adapting open LLMs to multilingual MT through continual pretraining and instruction finetuning. Based on the Gemma3 model family, we develop MiLMMT-46, which achieves top-tier multilingual translation performance across 46 languages. Extensive experiments show that MiLMMT-46 consistently outperforms recent state-of-the-art (SOTA) models, including Seed-X, HY-MT-1.5, and TranslateGemma, and achieves competitive performance with strong proprietary systems such as Google Translate and Gemini 3 Pro. 

---
# Do Large Language Models Adapt to Language Variation across Socioeconomic Status? 

**Authors**: Elisa Bassignana, Mike Zhang, Dirk Hovy, Amanda Cercas Curry  

**Link**: [PDF](https://arxiv.org/pdf/2602.11939)  

**Abstract**: Humans adjust their linguistic style to the audience they are addressing. However, the extent to which LLMs adapt to different social contexts is largely unknown. As these models increasingly mediate human-to-human communication, their failure to adapt to diverse styles can perpetuate stereotypes and marginalize communities whose linguistic norms are less closely mirrored by the models, thereby reinforcing social stratification. We study the extent to which LLMs integrate into social media communication across different socioeconomic status (SES) communities. We collect a novel dataset from Reddit and YouTube, stratified by SES. We prompt four LLMs with incomplete text from that corpus and compare the LLM-generated completions to the originals along 94 sociolinguistic metrics, including syntactic, rhetorical, and lexical features. LLMs modulate their style with respect to SES to only a minor extent, often resulting in approximation or caricature, and tend to emulate the style of upper SES more effectively. Our findings (1) show how LLMs risk amplifying linguistic hierarchies and (2) call into question their validity for agent-based social simulation, survey experiments, and any research relying on language style as a social signal. 

---
# Who is the richest club in the championship? Detecting and Rewriting Underspecified Questions Improve QA Performance 

**Authors**: Yunchong Huang, Gianni Barlacchi, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2602.11938)  

**Abstract**: Large language models (LLMs) perform well on well-posed questions, yet standard question-answering (QA) benchmarks remain far from solved. We argue that this gap is partly due to underspecified questions - queries whose interpretation cannot be uniquely determined without additional context. To test this hypothesis, we introduce an LLM-based classifier to identify underspecified questions and apply it to several widely used QA datasets, finding that 16% to over 50% of benchmark questions are underspecified and that LLMs perform significantly worse on them. To isolate the effect of underspecification, we conduct a controlled rewriting experiment that serves as an upper-bound analysis, rewriting underspecified questions into fully specified variants while holding gold answers fixed. QA performance consistently improves under this setting, indicating that many apparent QA failures stem from question underspecification rather than model limitations. Our findings highlight underspecification as an important confound in QA evaluation and motivate greater attention to question clarity in benchmark design. 

---
# Cross-Modal Robustness Transfer (CMRT): Training Robust Speech Translation Models Using Adversarial Text 

**Authors**: Abderrahmane Issam, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2602.11933)  

**Abstract**: End-to-End Speech Translation (E2E-ST) has seen significant advancements, yet current models are primarily benchmarked on curated, "clean" datasets. This overlooks critical real-world challenges, such as morphological robustness to inflectional variations common in non-native or dialectal speech. In this work, we adapt a text-based adversarial attack targeting inflectional morphology to the speech domain and demonstrate that state-of-the-art E2E-ST models are highly vulnerable it. While adversarial training effectively mitigates such risks in text-based tasks, generating high-quality adversarial speech data remains computationally expensive and technically challenging. To address this, we propose Cross-Modal Robustness Transfer (CMRT), a framework that transfers adversarial robustness from the text modality to the speech modality. Our method eliminates the requirement for adversarial speech data during training. Extensive experiments across four language pairs demonstrate that CMRT improves adversarial robustness by an average of more than 3 BLEU points, establishing a new baseline for robust E2E-ST without the overhead of generating adversarial speech. 

---
# AdaptEvolve: Improving Efficiency of Evolutionary AI Agents through Adaptive Model Selection 

**Authors**: Pretam Ray, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2602.11931)  

**Abstract**: Evolutionary agentic systems intensify the trade-off between computational efficiency and reasoning capability by repeatedly invoking large language models (LLMs) during inference. This setting raises a central question: how can an agent dynamically select an LLM that is sufficiently capable for the current generation step while remaining computationally efficient? While model cascades offer a practical mechanism for balancing this trade-off, existing routing strategies typically rely on static heuristics or external controllers and do not explicitly account for model uncertainty. We introduce AdaptEvolve: Adaptive LLM Selection for Multi-LLM Evolutionary Refinement within an evolutionary sequential refinement framework that leverages intrinsic generation confidence to estimate real-time solvability. Empirical results show that confidence-driven selection yields a favourable Pareto frontier, reducing total inference cost by an average of 37.9% across benchmarks while retaining 97.5% of the upper-bound accuracy of static large-model baselines. Our code is available at this https URL. 

---
# Benchmark Illusion: Disagreement among LLMs and Its Scientific Consequences 

**Authors**: Eddie Yang, Dashun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11898)  

**Abstract**: Benchmarks underpin how progress in large language models (LLMs) is measured and trusted. Yet our analyses reveal that apparent convergence in benchmark accuracy can conceal deep epistemic divergence. Using two major reasoning benchmarks - MMLU-Pro and GPQA - we show that LLMs achieving comparable accuracy still disagree on 16-66% of items, and 16-38% among top-performing frontier models. These discrepancies suggest distinct error profiles for different LLMs. When such models are used for scientific data annotation and inference, their hidden disagreements propagate into research results: in re-analyses of published studies in education and political science, switching the annotation model can change estimated treatment effects by more than 80%, and in some cases reverses their sign. Together, these findings illustrate a benchmark illusion, where equal accuracy may conceal disagreement, with model choice becoming a hidden yet consequential variable for scientific reproducibility. 

---
# LLM-based Triplet Extraction from Financial Reports 

**Authors**: Dante Wesslund, Ville Stenström, Pontus Linde, Alexander Holmberg  

**Link**: [PDF](https://arxiv.org/pdf/2602.11886)  

**Abstract**: Corporate financial reports are a valuable source of structured knowledge for Knowledge Graph construction, but the lack of annotated ground truth in this domain makes evaluation difficult. We present a semi-automated pipeline for Subject-Predicate-Object triplet extraction that uses ontology-driven proxy metrics, specifically Ontology Conformance and Faithfulness, instead of ground-truth-based evaluation. We compare a static, manually engineered ontology against a fully automated, document-specific ontology induction approach across different LLMs and two corporate annual reports. The automatically induced ontology achieves 100% schema conformance in all configurations, eliminating the ontology drift observed with the manual approach. We also propose a hybrid verification strategy that combines regex matching with an LLM-as-a-judge check, reducing apparent subject hallucination rates from 65.2% to 1.6% by filtering false positives caused by coreference resolution. Finally, we identify a systematic asymmetry between subject and object hallucinations, which we attribute to passive constructions and omitted agents in financial prose. 

---
# Towards Fair and Comprehensive Evaluation of Routers in Collaborative LLM Systems 

**Authors**: Wanxing Wu, He Zhu, Yixia Li, Lei Yang, Jiehui Zhao, Hongru Wang, Jian Yang, Benyou Wang, Bingyi Jing, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11877)  

**Abstract**: Large language models (LLMs) have achieved success, but cost and privacy constraints necessitate deploying smaller models locally while offloading complex queries to cloud-based models. Existing router evaluations are unsystematic, overlooking scenario-specific requirements and out-of-distribution robustness. We propose RouterXBench, a principled evaluation framework with three dimensions: router ability, scenario alignment, and cross-domain robustness. Unlike prior work that relies on output probabilities or external embeddings, we utilize internal hidden states that capture model uncertainty before answer generation. We introduce ProbeDirichlet, a lightweight router that aggregates cross-layer hidden states via learnable Dirichlet distributions with probabilistic training. Trained on multi-domain data, it generalizes robustly across in-domain and out-of-distribution scenarios. Our results show ProbeDirichlet achieves 16.68% and 18.86% relative improvements over the best baselines in router ability and high-accuracy scenarios, with consistent performance across model families, model scales, heterogeneous tasks, and agentic workflows. 

---
# DMAP: A Distribution Map for Text 

**Authors**: Tom Kempton, Julia Rozanova, Parameswaran Kamalaruban, Maeve Madigan, Karolina Wresilo, Yoann L. Launay, David Sutton, Stuart Burrell  

**Link**: [PDF](https://arxiv.org/pdf/2602.11871)  

**Abstract**: Large Language Models (LLMs) are a powerful tool for statistical text analysis, with derived sequences of next-token probability distributions offering a wealth of information. Extracting this signal typically relies on metrics such as perplexity, which do not adequately account for context; how one should interpret a given next-token probability is dependent on the number of reasonable choices encoded by the shape of the conditional distribution. In this work, we present DMAP, a mathematically grounded method that maps a text, via a language model, to a set of samples in the unit interval that jointly encode rank and probability information. This representation enables efficient, model-agnostic analysis and supports a range of applications. We illustrate its utility through three case studies: (i) validation of generation parameters to ensure data integrity, (ii) examining the role of probability curvature in machine-generated text detection, and (iii) a forensic analysis revealing statistical fingerprints left in downstream models that have been subject to post-training on synthetic data. Our results demonstrate that DMAP offers a unified statistical view of text that is simple to compute on consumer hardware, widely applicable, and provides a foundation for further research into text analysis with LLMs. 

---
# A Subword Embedding Approach for Variation Detection in Luxembourgish User Comments 

**Authors**: Anne-Marie Lutgen, Alistair Plum, Christoph Purschke  

**Link**: [PDF](https://arxiv.org/pdf/2602.11795)  

**Abstract**: This paper presents an embedding-based approach to detecting variation without relying on prior normalisation or predefined variant lists. The method trains subword embeddings on raw text and groups related forms through combined cosine and n-gram similarity. This allows spelling and morphological diversity to be examined and analysed as linguistic structure rather than treated as noise. Using a large corpus of Luxembourgish user comments, the approach uncovers extensive lexical and orthographic variation that aligns with patterns described in dialectal and sociolinguistic research. The induced families capture systematic correspondences and highlight areas of regional and stylistic differentiation. The procedure does not strictly require manual annotation, but does produce transparent clusters that support both quantitative and qualitative analysis. The results demonstrate that distributional modelling can reveal meaningful patterns of variation even in ''noisy'' or low-resource settings, offering a reproducible methodological framework for studying language variety in multilingual and small-language contexts. 

---
# MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling 

**Authors**: MiniCPM Team, Wenhao An, Yingfa Chen, Yewei Fang, Jiayi Li, Xin Li, Yaohui Li, Yishan Li, Yuxuan Li, Biyuan Lin, Chuan Liu, Hezi Liu, Siyuan Liu, Hongya Lyu, Yinxu Pan, Shixin Ren, Xingyu Shen, Zhou Su, Haojun Sun, Yangang Sun, Zhen Leng Thai, Xin Tian, Rui Wang, Xiaorong Wang, Yudong Wang, Bo Wu, Xiaoyue Xu, Dong Xu, Shuaikang Xue, Jiawei Yang, Bowen Zhang, Jinqian Zhang, Letian Zhang, Shengnan Zhang, Xinyu Zhang, Xinyuan Zhang, Zhu Zhang, Hengyu Zhao, Jiacheng Zhao, Jie Zhou, Zihan Zhou, Shuo Wang, Chaojun Xiao, Xu Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.11761)  

**Abstract**: The evolution of large language models (LLMs) towards applications with ultra-long contexts faces challenges posed by the high computational and memory costs of the Transformer architecture. While existing sparse and linear attention mechanisms attempt to mitigate these issues, they typically involve a trade-off between memory efficiency and model performance. This paper introduces MiniCPM-SALA, a 9B-parameter hybrid architecture that integrates the high-fidelity long-context modeling of sparse attention (InfLLM-V2) with the global efficiency of linear attention (Lightning Attention). By employing a layer selection algorithm to integrate these mechanisms in a 1:3 ratio and utilizing a hybrid positional encoding (HyPE), the model maintains efficiency and performance for long-context tasks. Furthermore, we introduce a cost-effective continual training framework that transforms pre-trained Transformer-based models into hybrid models, which reduces training costs by approximately 75% compared to training from scratch. Extensive experiments show that MiniCPM-SALA maintains general capabilities comparable to full-attention models while offering improved efficiency. On a single NVIDIA A6000D GPU, the model achieves up to 3.5x the inference speed of the full-attention model at the sequence length of 256K tokens and supports context lengths of up to 1M tokens, a scale where traditional full-attention 8B models fail because of memory constraints. 

---
# Think Longer to Explore Deeper: Learn to Explore In-Context via Length-Incentivized Reinforcement Learning 

**Authors**: Futing Wang, Jianhao Yan, Yun Luo, Ganqu Cui, Zhi Wang, Xiaoye Qu, Yue Zhang, Yu Cheng, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.11748)  

**Abstract**: Achieving effective test-time scaling requires models to engage in In-Context Exploration -- the intrinsic ability to generate, verify, and refine multiple reasoning hypotheses within a single continuous context.
Grounded in State Coverage theory, our analysis identifies a critical bottleneck to enabling this capability: while broader state coverage requires longer reasoning trajectories, the probability of sampling such sequences decays exponentially during autoregressive generation, a phenomenon we term the ``Shallow Exploration Trap''.
To bridge this gap, we propose Length-Incentivized Exploration(\method).
This simple yet effective recipe explicitly encourages models to explore more via a length-based reward coupled with a redundancy penalty, thereby maximizing state coverage in two-step manner.
Comprehensive experiments across different models (Qwen3, Llama) demonstrate that \method effectively incentivize in-context exploration.
As a result, our method achieves an average improvement of 4.4\% on in-domain tasks and a 2.7\% gain on out-of-domain benchmarks. 

---
# Thinking with Drafting: Optical Decompression via Logical Reconstruction 

**Authors**: Jingxuan Wei, Honghao He, Caijun Jia, Siyuan Li, Zheng Sun, Yuhang Xu, Yuanyuan Lin, Linzhuang Sun, Yuchen Wu, Bihui Yu, Xiangxiang Zhang, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.11731)  

**Abstract**: Existing multimodal large language models have achieved high-fidelity visual perception and exploratory visual generation. However, a precision paradox persists in complex reasoning tasks: optical perception systems transcribe symbols without capturing logical topology, while pixel-based generative models produce visual artifacts lacking mathematical exactness. To bridge this gap, we propose that reasoning over visual inputs be reconceptualized as optical decompression-the process of reconstructing latent logical structures from compressed visual tokens. Guided by the axiom that Parsing is Reasoning, we introduce Thinking with Drafting (TwD), which utilizes a minimalist Domain-Specific Language (DSL) as a grounding intermediate representation. Unlike standard approaches that hallucinate answers directly, TwD forces the model to draft its mental model into executable code, rendering deterministic visual proofs for self-verification. To validate this, we present VisAlg, a visual algebra benchmark. Experiments demonstrate that TwD serve as a superior cognitive scaffold. Our work establishes a closed-loop system where visual generation acts not as a creative output but as a logical verifier, offering a generalizable path for visual reasoning. 

---
# Finding Sense in Nonsense with Generated Contexts: Perspectives from Humans and Language Models 

**Authors**: Katrin Olsen, Sebastian Padó  

**Link**: [PDF](https://arxiv.org/pdf/2602.11699)  

**Abstract**: Nonsensical and anomalous sentences have been instrumental in the development of computational models of semantic interpretation. A core challenge is to distinguish between what is merely anomalous (but can be interpreted given a supporting context) and what is truly nonsensical. However, it is unclear (a) how nonsensical, rather than merely anomalous, existing datasets are; and (b) how well LLMs can make this distinction. In this paper, we answer both questions by collecting sensicality judgments from human raters and LLMs on sentences from five semantically deviant datasets: both context-free and when providing a context. We find that raters consider most sentences at most anomalous, and only a few as properly nonsensical. We also show that LLMs are substantially skilled in generating plausible contexts for anomalous cases. 

---
# PatientHub: A Unified Framework for Patient Simulation 

**Authors**: Sahand Sabour, TszYam NG, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11684)  

**Abstract**: As Large Language Models increasingly power role-playing applications, simulating patients has become a valuable tool for training counselors and scaling therapeutic assessment. However, prior work is fragmented: existing approaches rely on incompatible, non-standardized data formats, prompts, and evaluation metrics, hindering reproducibility and fair comparison. In this paper, we introduce PatientHub, a unified and modular framework that standardizes the definition, composition, and deployment of simulated patients. To demonstrate PatientHub's utility, we implement several representative patient simulation methods as case studies, showcasing how our framework supports standardized cross-method evaluation and the seamless integration of custom evaluation metrics. We further demonstrate PatientHub's extensibility by prototyping two new simulator variants, highlighting how PatientHub accelerates method development by eliminating infrastructure overhead. By consolidating existing work into a single reproducible pipeline, PatientHub lowers the barrier to developing new simulation methods and facilitates cross-method and cross-model benchmarking. Our framework provides a practical foundation for future datasets, methods, and benchmarks in patient-centered dialogue, and the code is publicly available via this https URL. 

---
# Which Feedback Works for Whom? Differential Effects of LLM-Generated Feedback Elements Across Learner Profiles 

**Authors**: Momoka Furuhashi, Kouta Nakayama, Noboru Kawai, Takashi Kodama, Saku Sugawara, Kyosuke Takami  

**Link**: [PDF](https://arxiv.org/pdf/2602.11650)  

**Abstract**: Large language models (LLMs) show promise for automatically generating feedback in education settings. However, it remains unclear how specific feedback elements, such as tone and information coverage, contribute to learning outcomes and learner acceptance, particularly across learners with different personality traits. In this study, we define six feedback elements and generate feedback for multiple-choice biology questions using GPT-5. We conduct a learning experiment with 321 first-year high school students and evaluate feedback effectiveness using two learning outcomes measures and subjective evaluations across six criteria. We further analyze differences in how feedback acceptance varies across learners based on Big Five personality traits. Our results show that effective feedback elements share common patterns supporting learning outcomes, while learners' subjective preferences differ across personality-based clusters. These findings highlight the importance of selecting and adapting feedback elements according to learners' personality traits when we design LLM-generated feedback, and provide practical implications for personalized feedback design in education. 

---
# PACE: Prefix-Protected and Difficulty-Aware Compression for Efficient Reasoning 

**Authors**: Ruixiang Feng, Yuntao Wen, Silin Zhou, Ke Shi, Yifan Wang, Ran Le, Zhenwei An, Zongchao Chen, Chen Yang, Guangyue Peng, Yiming Jia, Dongsheng Wang, Tao Zhang, Lisi Chen, Yang Song, Shen Gao, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11639)  

**Abstract**: Language Reasoning Models (LRMs) achieve strong performance by scaling test-time computation but often suffer from ``overthinking'', producing excessively long reasoning traces that increase latency and memory usage. Existing LRMs typically enforce conciseness with uniform length penalties, which over-compress crucial early deduction steps at the sequence level and indiscriminately penalize all queries at the group level. To solve these limitations, we propose \textbf{\model}, a dual-level framework for prefix-protected and difficulty-aware compression under hierarchical supervision. At the sequence level, prefix-protected optimization employs decaying mixed rollouts to maintain valid reasoning paths while promoting conciseness. At the group level, difficulty-aware penalty dynamically scales length constraints based on query complexity, maintaining exploration for harder questions while curbing redundancy on easier ones. Extensive experiments on DeepSeek-R1-Distill-Qwen (1.5B/7B) demonstrate that \model achieves a substantial reduction in token usage (up to \textbf{55.7\%}) while simultaneously improving accuracy (up to \textbf{4.1\%}) on math benchmarks, with generalization ability to code, science, and general domains. 

---
# Scene-Aware Memory Discrimination: Deciding Which Personal Knowledge Stays 

**Authors**: Yijie Zhong, Mengying Guo, Zewei Wang, Zhongyang Li, Dandan Tu, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11607)  

**Abstract**: Intelligent devices have become deeply integrated into everyday life, generating vast amounts of user interactions that form valuable personal knowledge. Efficient organization of this knowledge in user memory is essential for enabling personalized applications. However, current research on memory writing, management, and reading using large language models (LLMs) faces challenges in filtering irrelevant information and in dealing with rising computational costs. Inspired by the concept of selective attention in the human brain, we introduce a memory discrimination task. To address large-scale interactions and diverse memory standards in this task, we propose a Scene-Aware Memory Discrimination method (SAMD), which comprises two key components: the Gating Unit Module (GUM) and the Cluster Prompting Module (CPM). GUM enhances processing efficiency by filtering out non-memorable interactions and focusing on the salient content most relevant to application demands. CPM establishes adaptive memory standards, guiding LLMs to discern what information should be remembered or discarded. It also analyzes the relationship between user intents and memory contexts to build effective clustering prompts. Comprehensive direct and indirect evaluations demonstrate the effectiveness and generalization of our approach. We independently assess the performance of memory discrimination, showing that SAMD successfully recalls the majority of memorable data and remains robust in dynamic scenarios. Furthermore, when integrated into personalized applications, SAMD significantly enhances both the efficiency and quality of memory construction, leading to better organization of personal knowledge. 

---
# PRIME: A Process-Outcome Alignment Benchmark for Verifiable Reasoning in Mathematics and Engineering 

**Authors**: Xiangfeng Wang, Hangyu Guo, Yanlin Lai, Mitt Huang, Liang Zhao, Chengyuan Yao, Yinmin Zhang, Qi Han, Xiaoxiao Ren, Chun Yuan, Tong Xu, Zheng Ge, Xiangyu Zhang, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11570)  

**Abstract**: While model-based verifiers are essential for scaling Reinforcement Learning with Verifiable Rewards (RLVR), current outcome-centric verification paradigms primarily focus on the consistency between the final result and the ground truth, often neglecting potential errors in the derivation process. This leads to assigning positive rewards to correct answers produced from incorrect derivations. To bridge this gap, we introduce PRIME, a benchmark for evaluating verifiers on Process-Outcome Alignment verification in Mathematics and Engineering. Curated from a comprehensive collection of college-level STEM problems, PRIME comprises 2,530 high-difficulty samples through a consistency-based filtering pipeline. Through extensive evaluation, we find that current verifiers frequently fail to detect derivation flaws. Furthermore, we propose a process-aware RLVR training paradigm utilizing verifiers selected via PRIME. This approach substantially outperforms the outcome-only verification baseline, achieving absolute performance gains of 8.29%, 9.12%, and 7.31% on AIME24, AIME25, and Beyond-AIME, respectively, for the Qwen3-14B-Base model. Finally, we demonstrate a strong linear correlation ($R^2 > 0.92$) between verifier accuracy on PRIME and RLVR training effectiveness, validating PRIME as a reliable predictor for verifier selection. 

---
# SIGHT: Reinforcement Learning with Self-Evidence and Information-Gain Diverse Branching for Search Agent 

**Authors**: Wenlin Zhong, Jinluan Yang, Yiquan Wu, Yi Liu, Jianhang Yao, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11551)  

**Abstract**: Reinforcement Learning (RL) has empowered Large Language Models (LLMs) to master autonomous search for complex question answering. However, particularly within multi-turn search scenarios, this interaction introduces a critical challenge: search results often suffer from high redundancy and low signal-to-noise ratios. Consequently, agents easily fall into "Tunnel Vision," where the forced interpretation of early noisy retrievals leads to irreversible error accumulation. To address these challenges, we propose SIGHT, a framework that enhances search-based reasoning through Self-Evidence Support (SES) and Information-Gain Driven Diverse Branching. SIGHT distills search results into high-fidelity evidence via SES and calculates an Information Gain score to pinpoint pivotal states where observations maximally reduce uncertainty. This score guides Dynamic Prompting Interventions - including de-duplication, reflection, or adaptive branching - to spawn new branches with SES. Finally, by integrating SES and correctness rewards via Group Relative Policy Optimization, SIGHT internalizes robust exploration strategies without external verifiers. Experiments on single-hop and multi-hop QA benchmarks demonstrate that SIGHT significantly outperforms existing approaches, particularly in complex reasoning scenarios, using fewer search steps. 

---
# Pretraining A Large Language Model using Distributed GPUs: A Memory-Efficient Decentralized Paradigm 

**Authors**: Jinrui Zhang, Chaodong Xiao, Aoqi Wu, Xindong Zhang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11543)  

**Abstract**: Pretraining large language models (LLMs) typically requires centralized clusters with thousands of high-memory GPUs (e.g., H100/A100). Recent decentralized training methods reduce communication overhead by employing federated optimization; however, they still need to train the entire model on each node, remaining constrained by GPU memory limitations. In this work, we propose SParse Expert Synchronization (SPES), a memory-efficient decentralized framework for pretraining mixture-of-experts (MoE) LLMs. SPES trains only a subset of experts per node, substantially lowering the memory footprint. Each node updates its local experts and periodically synchronizes with other nodes, eliminating full-parameter transmission while ensuring efficient knowledge sharing. To accelerate convergence, we introduce an expert-merging warm-up strategy, where experts exchange knowledge early in training, to rapidly establish foundational capabilities. With SPES, we train a 2B-parameter MoE LLM using 16 standalone 48GB GPUs over internet connections, which achieves competitive performance with centrally trained LLMs under similar computational budgets. We further demonstrate scalability by training a 7B model from scratch and a 9B model upcycled from a dense checkpoint, both of which match prior centralized baselines. Our code is available at this https URL. 

---
# Multimodal Fact-Level Attribution for Verifiable Reasoning 

**Authors**: David Wan, Han Wang, Ziyang Wang, Elias Stengel-Eskin, Hyunji Lee, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2602.11509)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly used for real-world tasks involving multi-step reasoning and long-form generation, where reliability requires grounding model outputs in heterogeneous input sources and verifying individual factual claims. However, existing multimodal grounding benchmarks and evaluation methods focus on simplified, observation-based scenarios or limited modalities and fail to assess attribution in complex multimodal reasoning. We introduce MuRGAt (Multimodal Reasoning with Grounded Attribution), a benchmark for evaluating fact-level multimodal attribution in settings that require reasoning beyond direct observation. Given inputs spanning video, audio, and other modalities, MuRGAt requires models to generate answers with explicit reasoning and precise citations, where each citation specifies both modality and temporal segments. To enable reliable assessment, we introduce an automatic evaluation framework that strongly correlates with human judgments. Benchmarking with human and automated scores reveals that even strong MLLMs frequently hallucinate citations despite correct reasoning. Moreover, we observe a key trade-off: increasing reasoning depth or enforcing structured grounding often degrades accuracy, highlighting a significant gap between internal reasoning and verifiable attribution. 

---
# When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration 

**Authors**: Jayadev Billa  

**Link**: [PDF](https://arxiv.org/pdf/2602.11488)  

**Abstract**: When audio and text conflict, speech-enabled language models follow the text 10 times more often than when arbitrating between two text sources, even when explicitly instructed to trust the audio. Using ALME, a benchmark of 57,602 controlled audio-text conflict stimuli across 8 languages, we find that Gemini 2.0 Flash exhibits 16.6\% text dominance under audio-text conflict versus 1.6\% under text-text conflict with identical reliability cues. This gap is not explained by audio quality: audio-only accuracy (97.2\%) exceeds cascade accuracy (93.9\%), indicating audio embeddings preserve more information than text transcripts. We propose that text dominance reflects an asymmetry not in information content but in arbitration accessibility: how easily the model can reason over competing representations.
This framework explains otherwise puzzling findings. Forcing transcription before answering increases text dominance (19\% to 33\%), sacrificing audio's information advantage without improving accessibility. Framing text as ``deliberately corrupted'' reduces text dominance by 80\%. A fine-tuning ablation provides interventional evidence: training only the audio projection layer increases text dominance (+26.5\%), while LoRA on the language model halves it ($-$23.9\%), localizing text dominance to the LLM's reasoning rather than the audio encoder. Experiments across four state-of-the-art audio-LLMs and 8 languages show consistent trends with substantial cross-linguistic and cross-model variation, establishing modality arbitration as a distinct reliability dimension not captured by standard speech benchmarks. 

---
# ADRD-Bench: A Preliminary LLM Benchmark for Alzheimer's Disease and Related Dementias 

**Authors**: Guangxin Zhao, Jiahao Zheng, Malaz Boustani, Jarek Nabrzyski, Meng Jiang, Yiyu Shi, Zhi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.11460)  

**Abstract**: Large language models (LLMs) have shown great potential for healthcare applications. However, existing evaluation benchmarks provide minimal coverage of Alzheimer's Disease and Related Dementias (ADRD). To address this gap, we introduce ADRD-Bench, the first ADRD-specific benchmark dataset designed for rigorous evaluation of LLMs. ADRD-Bench has two components: 1) ADRD Unified QA, a synthesis of 1,352 questions consolidated from seven established medical benchmarks, providing a unified assessment of clinical knowledge; and 2) ADRD Caregiving QA, a novel set of 149 questions derived from the Aging Brain Care (ABC) program, a widely used, evidence-based brain health management program. Guided by a program with national expertise in comprehensive ADRD care, this new set was designed to mitigate the lack of practical caregiving context in existing benchmarks. We evaluated 33 state-of-the-art LLMs on the proposed ADRD-Bench. Results showed that the accuracy of open-weight general models ranged from 0.63 to 0.93 (mean: 0.78; std: 0.09). The accuracy of open-weight medical models ranged from 0.48 to 0.93 (mean: 0.82; std: 0.13). The accuracy of closed-source general models ranged from 0.83 to 0.91 (mean: 0.89; std: 0.03). While top-tier models achieved high accuracies (>0.9), case studies revealed that inconsistent reasoning quality and stability limit their reliability, highlighting a critical need for domain-specific improvement to enhance LLMs' knowledge and reasoning grounded in daily caregiving data. The entire dataset is available at this https URL. 

---
# LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation 

**Authors**: Ahmadreza Jeddi, Marco Ciccone, Babak Taati  

**Link**: [PDF](https://arxiv.org/pdf/2602.11451)  

**Abstract**: Looped Transformers have emerged as an efficient and powerful class of models for reasoning in the language domain. Recent studies show that these models achieve strong performance on algorithmic and reasoning tasks, suggesting that looped architectures possess an inductive bias toward latent reasoning. However, prior approaches fix the number of loop iterations during training and inference, leaving open the question of whether these models can flexibly adapt their computational depth under variable compute budgets. We introduce LoopFormer, a looped Transformer trained on variable-length trajectories to enable budget-conditioned reasoning. Our core contribution is a shortcut-consistency training scheme that aligns trajectories of different lengths, ensuring that shorter loops yield informative representations while longer loops continue to refine them. LoopFormer conditions each loop on the current time and step size, enabling representations to evolve consistently across trajectories of varying length rather than drifting or stagnating. Empirically, LoopFormer demonstrates robust performance on language modeling and reasoning benchmarks even under aggressive compute constraints, while scaling gracefully with additional budget. These results show that looped Transformers are inherently suited for adaptive language modeling, opening a path toward controllable and budget-aware large language models. 

---
# Towards Reliable Machine Translation: Scaling LLMs for Critical Error Detection and Safety 

**Authors**: Muskaan Chopra, Lorenz Sparrenberg, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2602.11444)  

**Abstract**: Machine Translation (MT) plays a pivotal role in cross-lingual information access, public policy communication, and equitable knowledge dissemination. However, critical meaning errors, such as factual distortions, intent reversals, or biased translations, can undermine the reliability, fairness, and safety of multilingual systems. In this work, we explore the capacity of instruction-tuned Large Language Models (LLMs) to detect such critical errors, evaluating models across a range of parameters using the publicly accessible data sets. Our findings show that model scaling and adaptation strategies (zero-shot, few-shot, fine-tuning) yield consistent improvements, outperforming encoder-only baselines like XLM-R and ModernBERT. We argue that improving critical error detection in MT contributes to safer, more trustworthy, and socially accountable information systems by reducing the risk of disinformation, miscommunication, and linguistic harm, especially in high-stakes or underrepresented contexts. This work positions error detection not merely as a technical challenge, but as a necessary safeguard in the pursuit of just and responsible multilingual AI. The code will be made available at GitHub. 

---
# Gradients Must Earn Their Influence: Unifying SFT with Generalized Entropic Objectives 

**Authors**: Zecheng Wang, Deyuan Liu, Chunshan Li, Yupeng Zhang, Zhengyun Zhao, Dianhui Chu, Bingning Wang, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2602.11424)  

**Abstract**: Standard negative log-likelihood (NLL) for Supervised Fine-Tuning (SFT) applies uniform token-level weighting. This rigidity creates a two-fold failure mode: (i) overemphasizing low-probability targets can amplify gradients on noisy supervision and disrupt robust priors, and (ii) uniform weighting provides weak sharpening when the model is already confident. Existing methods fail to resolve the resulting plasticity--stability dilemma, often suppressing necessary learning signals alongside harmful ones. To address this issue, we unify token-level SFT objectives within a generalized deformed-log family and expose a universal gate $\times$ error gradient structure, where the gate controls how much the model trusts its current prediction. By employing the Cayley transform, we map the model's continuously evolving uncertainty onto a continuous focus trajectory, which enables seamless interpolation between scenarios involving uncertain novel concepts and those involving well-established knowledge. We then introduce Dynamic Entropy Fine-Tuning (DEFT), a parameter-free objective that modulates the trust gate using distribution concentration (Rényi-2 entropy) as a practical proxy for the model's predictive state. Extensive experiments and analyses demonstrate that DEFT achieves a better balance between exploration and exploitation, leading to improved overall performance. 

---
# Advancing AI Trustworthiness Through Patient Simulation: Risk Assessment of Conversational Agents for Antidepressant Selection 

**Authors**: Md Tanvir Rouf Shawon, Mohammad Sabik Irbaz, Hadeel R. A. Elyazori, Keerti Reddy Resapu, Yili Lin, Vladimir Franzuela Cardenas, Farrokh Alemi, Kevin Lybarger  

**Link**: [PDF](https://arxiv.org/pdf/2602.11391)  

**Abstract**: Objective: This paper introduces a patient simulator designed to enable scalable, automated evaluation of healthcare conversational agents. The simulator generates realistic, controllable patient interactions that systematically vary across medical, linguistic, and behavioral dimensions, allowing annotators and an independent AI judge to assess agent performance, identify hallucinations and inaccuracies, and characterize risk patterns across diverse patient populations. Methods: The simulator is grounded in the NIST AI Risk Management Framework and integrates three profile components reflecting different dimensions of patient variation: (1) medical profiles constructed from electronic health records in the All of Us Research Program; (2) linguistic profiles modeling variation in health literacy and condition-specific communication patterns; and (3) behavioral profiles representing empirically observed interaction patterns, including cooperation, distraction, and adversarial engagement. We evaluated the simulator's effectiveness in identifying errors in an AI decision aid for antidepressant selection. Results: We generated 500 conversations between the patient simulator and the AI decision aid across systematic combinations of five linguistic and three behavioral profiles. Human annotators assessed 1,787 medical concepts across 100 conversations, achieving high agreement (F1=0.94, \k{appa}=0.73), and the LLM judge achieved comparable agreement with human annotators (F1=0.94, \k{appa}=0.78; paired bootstrap p=0.21). The simulator revealed a monotonic degradation in AI decision aid performance across the health literacy spectrum: rank-one concept retrieval accuracy increased from 47.9% for limited health literacy to 69.1% for functional and 81.6% for proficient. 

---
# The Energy of Falsehood: Detecting Hallucinations via Diffusion Model Likelihoods 

**Authors**: Arpit Singh Gautam, Kailash Talreja, Saurabh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2602.11364)  

**Abstract**: Large Language Models (LLMs) frequently hallucinate plausible but incorrect assertions, a vulnerability often missed by uncertainty metrics when models are confidently wrong. We propose DiffuTruth, an unsupervised framework that reconceptualizes fact verification via non equilibrium thermodynamics, positing that factual truths act as stable attractors on a generative manifold while hallucinations are unstable. We introduce the Generative Stress Test, claims are corrupted with noise and reconstructed using a discrete text diffusion model. We define Semantic Energy, a metric measuring the semantic divergence between the original claim and its reconstruction using an NLI critic. Unlike vector space errors, Semantic Energy isolates deep factual contradictions. We further propose a Hybrid Calibration fusing this stability signal with discriminative confidence. Extensive experiments on FEVER demonstrate DiffuTruth achieves a state of the art unsupervised AUROC of 0.725, outperforming baselines by 1.5 percent through the correction of overconfident predictions. Furthermore, we show superior zero shot generalization on the multi hop HOVER dataset, outperforming baselines by over 4 percent, confirming the robustness of thermodynamic truth properties to distribution shifts. 

---
# Finding the Cracks: Improving LLMs Reasoning with Paraphrastic Probing and Consistency Verification 

**Authors**: Weili Shi, Dongliang Guo, Lehan Yang, Tianlong Wang, Hanzhang Yuan, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.11361)  

**Abstract**: Large language models have demonstrated impressive performance across a variety of reasoning tasks. However, their problem-solving ability often declines on more complex tasks due to hallucinations and the accumulation of errors within these intermediate steps. Recent work has introduced the notion of critical tokens--tokens in the reasoning process that exert significant influence on subsequent steps. Prior studies suggest that replacing critical tokens can refine reasoning trajectories. Nonetheless, reliably identifying and exploiting critical tokens remains challenging. To address this, we propose the Paraphrastic Probing and Consistency Verification~(PPCV) framework. PPCV operates in two stages. In the first stage, we roll out an initial reasoning path from the original question and then concatenate paraphrased versions of the question with this reasoning path. And we identify critical tokens based on mismatches between the predicted top-1 token and the expected token in the reasoning path. A criterion is employed to confirm the final critical token. In the second stage, we substitute critical tokens with candidate alternatives and roll out new reasoning paths for both the original and paraphrased questions. The final answer is determined by checking the consistency of outputs across these parallel reasoning processes. We evaluate PPCV on mainstream LLMs across multiple benchmarks. Extensive experiments demonstrate PPCV substantially enhances the reasoning performance of LLMs compared to baselines. 

---
# When Models Examine Themselves: Vocabulary-Activation Correspondence in Self-Referential Processing 

**Authors**: Zachary Pedram Dadfar  

**Link**: [PDF](https://arxiv.org/pdf/2602.11358)  

**Abstract**: Large language models produce rich introspective language when prompted for self-examination, but whether this language reflects internal computation or sophisticated confabulation has remained unclear. We show that self-referential vocabulary tracks concurrent activation dynamics, and that this correspondence is specific to self-referential processing. We introduce the Pull Methodology, a protocol that elicits extended self-examination through format engineering, and use it to identify a direction in activation space that distinguishes self-referential from descriptive processing in Llama 3.1. The direction is orthogonal to the known refusal direction, localised at 6.25% of model depth, and causally influences introspective output when used for steering. When models produce "loop" vocabulary, their activations exhibit higher autocorrelation (r = 0.44, p = 0.002); when they produce "shimmer" vocabulary under steering, activation variability increases (r = 0.36, p = 0.002). Critically, the same vocabulary in non-self-referential contexts shows no activation correspondence despite nine-fold higher frequency. Qwen 2.5-32B, with no shared training, independently develops different introspective vocabulary tracking different activation metrics, all absent in descriptive controls. The findings indicate that self-report in transformer models can, under appropriate conditions, reliably track internal computational states. 

---
# Evaluating Alignment of Behavioral Dispositions in LLMs 

**Authors**: Amir Taubenfeld, Zorik Gekhman, Lior Nezry, Omri Feldman, Natalie Harris, Shashir Reddy, Romina Stella, Ariel Goldstein, Marian Croak, Yossi Matias, Amir Feder  

**Link**: [PDF](https://arxiv.org/pdf/2602.11328)  

**Abstract**: As LLMs integrate into our daily lives, understanding their behavior becomes essential. In this work, we focus on behavioral dispositions$-$the underlying tendencies that shape responses in social contexts$-$and introduce a framework to study how closely the dispositions expressed by LLMs align with those of humans. Our approach is grounded in established psychological questionnaires but adapts them for LLMs by transforming human self-report statements into Situational Judgment Tests (SJTs). These SJTs assess behavior by eliciting natural recommendations in realistic user-assistant scenarios. We generate 2,500 SJTs, each validated by three human annotators, and collect preferred actions from 10 annotators per SJT, from a large pool of 550 participants. In a comprehensive study involving 25 LLMs, we find that models often do not reflect the distribution of human preferences: (1) in scenarios with low human consensus, LLMs consistently exhibit overconfidence in a single response; (2) when human consensus is high, smaller models deviate significantly, and even some frontier models do not reflect the consensus in 15-20% of cases; (3) traits can exhibit cross-LLM patterns, e.g., LLMs may encourage emotion expression in contexts where human consensus favors composure. Lastly, mapping psychometric statements directly to behavioral scenarios presents a unique opportunity to evaluate the predictive validity of self-reports, revealing considerable gaps between LLMs' stated values and their revealed behavior. 

---
# Are Aligned Large Language Models Still Misaligned? 

**Authors**: Usman Naseem, Gautam Siddharth Kashyap, Rafiq Ali, Ebad Shabbir, Sushant Kumar Ray, Abdullah Mohammad, Agrima Seth  

**Link**: [PDF](https://arxiv.org/pdf/2602.11305)  

**Abstract**: Misalignment in Large Language Models (LLMs) arises when model behavior diverges from human expectations and fails to simultaneously satisfy safety, value, and cultural dimensions, which must co-occur in real-world settings to solve a real-world query. Existing misalignment benchmarks-such as INSECURE CODE (safety-centric), VALUEACTIONLENS (value-centric), and CULTURALHERITAGE (culture centric)-rely on evaluating misalignment along individual dimensions, preventing simultaneous evaluation. To address this gap, we introduce Mis-Align Bench, a unified benchmark for analyzing misalignment across safety, value, and cultural dimensions. First we constructs SAVACU, an English misaligned-aligned dataset of 382,424 samples spanning 112 domains (or labels), by reclassifying prompts from the LLM-PROMPT-DATASET via taxonomy into 14 safety domains, 56 value domains, and 42 cultural domains using Mistral-7B-Instruct-v0.3, and expanding low-resource domains via Llama-3.1-8B-Instruct with SimHash-based fingerprint to avoid deduplication. Furthermore, we pairs prompts with misaligned and aligned responses via two-stage rejection sampling to enforce quality. Second we benchmarks general-purpose, fine-tuned, and open-weight LLMs, enabling systematic evaluation of misalignment under three dimensions. Empirically, single-dimension models achieve high Coverage (upto 97.6%) but incur False Failure Rate >50% and lower Alignment Score (63%-66%) under joint conditions. 

---
# SurveyLens: A Research Discipline-Aware Benchmark for Automatic Survey Generation 

**Authors**: Beichen Guo, Zhiyuan Wen, Jia Gu, Senzhang Wang, Haochen Shi, Ruosong Yang, Shuaiqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11238)  

**Abstract**: The exponential growth of scientific literature has driven the evolution of Automatic Survey Generation (ASG) from simple pipelines to multi-agent frameworks and commercial Deep Research agents. However, current ASG evaluation methods rely on generic metrics and are heavily biased toward Computer Science (CS), failing to assess whether ASG methods adhere to the distinct standards of various academic disciplines. Consequently, researchers, especially those outside CS, lack clear guidance on using ASG systems to yield high-quality surveys compliant with specific discipline standards. To bridge this gap, we introduce SurveyLens, the first discipline-aware benchmark evaluating ASG methods across diverse research disciplines. We construct SurveyLens-1k, a curated dataset of 1,000 high-quality human-written surveys spanning 10 disciplines. Subsequently, we propose a dual-lens evaluation framework: (1) Discipline-Aware Rubric Evaluation, which utilizes LLMs with human preference-aligned weights to assess adherence to domain-specific writing standards; and (2) Canonical Alignment Evaluation to rigorously measure content coverage and synthesis quality against human-written survey papers. We conduct extensive experiments by evaluating 11 state-of-the-art ASG methods on SurveyLens, including Vanilla LLMs, ASG systems, and Deep Research agents. Our analysis reveals the distinct strengths and weaknesses of each paradigm across fields, providing essential guidance for selecting tools tailored to specific disciplinary requirements. 

---
# The Automatic Verification of Image-Text Claims (AVerImaTeC) Shared Task 

**Authors**: Rui Cao, Zhenyun Deng, Yulong Chen, Michael Schlichtkrull, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2602.11221)  

**Abstract**: The Automatic Verification of Image-Text Claims (AVerImaTeC) shared task aims to advance system development for retrieving evidence and verifying real-world image-text claims. Participants were allowed to either employ external knowledge sources, such as web search engines, or leverage the curated knowledge store provided by the organizers. System performance was evaluated using the AVerImaTeC score, defined as a conditional verdict accuracy in which a verdict is considered correct only when the associated evidence score exceeds a predefined threshold. The shared task attracted 14 submissions during the development phase and 6 submissions during the testing phase. All participating systems in the testing phase outperformed the baseline provided. The winning team, HUMANE, achieved an AVerImaTeC score of 0.5455. This paper provides a detailed description of the shared task, presents the complete evaluation results, and discusses key insights and lessons learned. 

---
# Mechanistic Evidence for Faithfulness Decay in Chain-of-Thought Reasoning 

**Authors**: Donald Ye, Max Loffgren, Om Kotadia, Linus Wong  

**Link**: [PDF](https://arxiv.org/pdf/2602.11201)  

**Abstract**: Chain-of-Thought (CoT) explanations are widely used to interpret how language models solve complex problems, yet it remains unclear whether these step-by-step explanations reflect how the model actually reaches its answer, or merely post-hoc justifications. We propose Normalized Logit Difference Decay (NLDD), a metric that measures whether individual reasoning steps are faithful to the model's decision-making process. Our approach corrupts individual reasoning steps from the explanation and measures how much the model's confidence in its answer drops, to determine if a step is truly important. By standardizing these measurements, NLDD enables rigorous cross-model comparison across different architectures. Testing three model families across syntactic, logical, and arithmetic tasks, we discover a consistent Reasoning Horizon (k*) at 70--85% of chain length, beyond which reasoning tokens have little or negative effect on the final answer. We also find that models can encode correct internal representations while completely failing the task. These results show that accuracy alone does not reveal whether a model actually reasons through its chain. NLDD offers a way to measure when CoT matters. 

---
# When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification 

**Authors**: Jiale Zhao, Ke Fang, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.11199)  

**Abstract**: Large language models (LLMs) often respond even when prompts omit critical details or include misleading information, leading to hallucinations or reinforced misconceptions. We study how to evaluate and improve LLMs' ability to decide when and what to ask for clarification without sacrificing task performance. We introduce AskBench, an interactive benchmark that converts standard QA pairs into multi-turn interactions with explicit checkpoints. A unified judge loop evaluates final answers and simulates user responses as needed. AskBench covers two settings: AskMind, with intent-deficient queries requiring clarification, and AskOverconfidence, with queries containing false premises that must be identified and corrected. We further propose rubric-guided reinforcement learning with verifier-based rewards (RLVR), which uses structured rubrics to encourage targeted clarification. Experiments show consistent improvements in accuracy, rubric adherence, and interaction efficiency, with strong generalization to unseen domains. 

---
# DDL2PropBank Agent: Benchmarking Multi-Agent Frameworks' Developer Experience Through a Novel Relational Schema Mapping Task 

**Authors**: Shafiuddin Rehan Ahmed, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.11198)  

**Abstract**: Multi-agent frameworks promise to simplify LLM-driven software development, yet there is no principled way to evaluate their developer experience in a controlled setting. We introduce DDL2PropBank, a novel benchmark task that maps relational database schemas to PropBank rolesets, requiring autonomous retrieval of candidate frames and fine-grained linguistic reasoning over table names, columns, and relations. Using the Agent-as-a-Tool pattern, we implement identical agent logic across 10 frameworks and evaluate along two dimensions: (i) code complexity via static analysis, and (ii) AI-assistability -- the extent to which LLMs can autonomously generate correct, framework-specific code. Our results reveal a threefold complexity spectrum, with Pydantic AI and Agno requiring the least implementation overhead. For AI-assistability, structural alignment scores reliably proxy runtime success for frameworks with single canonical patterns, but overestimate correctness for multi-pattern frameworks. Agno emerges as the strongest overall performer, combining lowest complexity with highest structural alignment and 83% pass@1. 

---
# MetaMem: Evolving Meta-Memory for Knowledge Utilization through Self-Reflective Symbolic Optimization 

**Authors**: Haidong Xin, Xinze Li, Zhenghao Liu, Yukun Yan, Shuo Wang, Cheng Yang, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.11182)  

**Abstract**: Existing memory systems enable Large Language Models (LLMs) to support long-horizon human-LLM interactions by persisting historical interactions beyond limited context windows. However, while recent approaches have succeeded in constructing effective memories, they often disrupt the inherent logical and temporal relationships within interaction sessions, resulting in fragmented memory units and degraded reasoning performance. In this paper, we propose MetaMem, a novel framework that augments memory systems with a self-evolving meta-memory, aiming to teach LLMs how to effectively utilize memorized knowledge. During meta-memory optimization, MetaMem iteratively distills transferable knowledge utilization experiences across different tasks by self-reflecting on reasoning processes and performing actions to update the current meta-memory state. The accumulated meta-memory units serve as explicit knowledge utilization experiences, guiding the LLM to systematically identify and integrate critical evidence from scattered memory fragments. Extensive experiments demonstrate the effectiveness of MetaMem, which significantly outperforms strong baselines by over 3.6%. All codes and datasets are available at this https URL. 

---
# Code Mixologist : A Practitioner's Guide to Building Code-Mixed LLMs 

**Authors**: Himanshu Gupta, Pratik Jayarao, Chaitanya Dwivedi, Neeraj Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2602.11181)  

**Abstract**: Code-mixing and code-switching (CSW) remain challenging phenomena for large language models (LLMs). Despite recent advances in multilingual modeling, LLMs often struggle in mixed-language settings, exhibiting systematic degradation in grammaticality, factuality, and safety behavior. This work provides a comprehensive overview of CSW research in modern large language model settings. We introduce a unifying taxonomy that organizes prior work along dimensions of data, modeling, and evaluation, and we distill these findings into a practical playbook of actionable recommendations for building, adapting, and evaluating CSW-capable LLMs. We review modeling approaches ranging from CSW-tailored pre-training and task-specific post-training to prompting strategies and in-context learning. We analyze current evaluation practices, highlighting sources of instability and limited reproducibility, and we catalog existing benchmarks while critically examining their linguistic coverage and English-centric biases. Finally, we discuss emerging safety concerns, including use of code-mixing as a mechanism for bypassing model safeguards, and identify open research challenges. 

---
# Mechanistic Interpretability for Large Language Model Alignment: Progress, Challenges, and Future Directions 

**Authors**: Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2602.11180)  

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities across diverse tasks, yet their internal decision-making processes remain largely opaque. Mechanistic interpretability (i.e., the systematic study of how neural networks implement algorithms through their learned representations and computational structures) has emerged as a critical research direction for understanding and aligning these models. This paper surveys recent progress in mechanistic interpretability techniques applied to LLM alignment, examining methods ranging from circuit discovery to feature visualization, activation steering, and causal intervention. We analyze how interpretability insights have informed alignment strategies including reinforcement learning from human feedback (RLHF), constitutional AI, and scalable oversight. Key challenges are identified, including the superposition hypothesis, polysemanticity of neurons, and the difficulty of interpreting emergent behaviors in large-scale models. We propose future research directions focusing on automated interpretability, cross-model generalization of circuits, and the development of interpretability-driven alignment techniques that can scale to frontier models. 

---
# From Instruction to Output: The Role of Prompting in Modern NLG 

**Authors**: Munazza Zaib, Elaf Alhazmi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11179)  

**Abstract**: Prompt engineering has emerged as an integral technique for extending the strengths and abilities of Large Language Models (LLMs) to gain significant performance gains in various Natural Language Processing (NLP) tasks. This approach, which requires instructions to be composed in natural language to bring out the knowledge from LLMs in a structured way, has driven breakthroughs in various NLP tasks. Yet there is still no structured framework or coherent understanding of the varied prompt engineering methods and techniques, particularly in the field of Natural Language Generation (NLG).
This survey aims to help fill that gap by outlining recent developments in prompt engineering, and their effect on different NLG tasks. It reviews recent advances in prompting methods and their impact on NLG tasks, presenting prompt design as an input-level control mechanism that complements fine-tuning and decoding approaches. The paper introduces a taxonomy of prompting paradigms, a decision framework for prompt selection based on varying factors for the practitioners, outlines emerging trends and challenges, and proposes a framework that links design, optimization, and evaluation to support more controllable and generalizable NLG. 

---
# What Do LLMs Know About Alzheimer's Disease? Fine-Tuning, Probing, and Data Synthesis for AD Detection 

**Authors**: Lei Jiang, Yue Zhou, Natalie Parde  

**Link**: [PDF](https://arxiv.org/pdf/2602.11177)  

**Abstract**: Reliable early detection of Alzheimer's disease (AD) is challenging, particularly due to limited availability of labeled data. While large language models (LLMs) have shown strong transfer capabilities across domains, adapting them to the AD domain through supervised fine-tuning remains largely unexplored. In this work, we fine-tune an LLM for AD detection and investigate how task-relevant information is encoded within its internal representations. We employ probing techniques to analyze intermediate activations across transformer layers, and we observe that, after fine-tuning, the probing values of specific words and special markers change substantially, indicating that these elements assume a crucial role in the model's improved detection performance. Guided by this insight, we design a curated set of task-aware special markers and train a sequence-to-sequence model as a data-synthesis tool that leverages these markers to generate structurally consistent and diagnostically informative synthetic samples. We evaluate the synthesized data both intrinsically and by incorporating it into downstream training pipelines. 

---
# Evaluating Few-Shot Temporal Reasoning of LLMs for Human Activity Prediction in Smart Environments 

**Authors**: Maral Doctorarastoo, Katherine A. Flanigan, Mario Bergés, Christopher McComb  

**Link**: [PDF](https://arxiv.org/pdf/2602.11176)  

**Abstract**: Anticipating human activities and their durations is essential in applications such as smart-home automation, simulation-based architectural and urban design, activity-based transportation system simulation, and human-robot collaboration, where adaptive systems must respond to human activities. Existing data-driven agent-based models--from rule-based to deep learning--struggle in low-data environments, limiting their practicality. This paper investigates whether large language models, pre-trained on broad human knowledge, can fill this gap by reasoning about everyday activities from compact contextual cues. We adopt a retrieval-augmented prompting strategy that integrates four sources of context--temporal, spatial, behavioral history, and persona--and evaluate it on the CASAS Aruba smart-home dataset. The evaluation spans two complementary tasks: next-activity prediction with duration estimation, and multi-step daily sequence generation, each tested with various numbers of few-shot examples provided in the prompt. Analyzing few-shot effects reveals how much contextual supervision is sufficient to balance data efficiency and predictive accuracy, particularly in low-data environments. Results show that large language models exhibit strong inherent temporal understanding of human behavior: even in zero-shot settings, they produce coherent daily activity predictions, while adding one or two demonstrations further refines duration calibration and categorical accuracy. Beyond a few examples, performance saturates, indicating diminishing returns. Sequence-level evaluation confirms consistent temporal alignment across few-shot conditions. These findings suggest that pre-trained language models can serve as promising temporal reasoners, capturing both recurring routines and context-dependent behavioral variations, thereby strengthening the behavioral modules of agent-based models. 

---
# Barriers to Discrete Reasoning with Transformers: A Survey Across Depth, Exactness, and Bandwidth 

**Authors**: Michelle Yuan, Weiyi Sun, Amir H. Rezaeian, Jyotika Singh, Sandip Ghoshal, Yao-Ting Wang, Miguel Ballesteros, Yassine Benajiba  

**Link**: [PDF](https://arxiv.org/pdf/2602.11175)  

**Abstract**: Transformers have become the foundational architecture for a broad spectrum of sequence modeling applications, underpinning state-of-the-art systems in natural language processing, vision, and beyond. However, their theoretical limitations in discrete reasoning tasks, such as arithmetic, logical inference, and algorithmic composition, remain a critical open problem. In this survey, we synthesize recent studies from three theoretical perspectives: circuit complexity, approximation theory, and communication complexity, to clarify the structural and computational barriers that transformers face when performing symbolic computations. By connecting these established theoretical frameworks, we provide an accessible and unified account of why current transformer architectures struggle to implement exact discrete algorithms, even as they excel at pattern matching and interpolation. We review key definitions, seminal results, and illustrative examples, highlighting challenges such as depth constraints, difficulty approximating discontinuities, and bottlenecks in inter-token communication. Finally, we discuss implications for model design and suggest promising directions for overcoming these foundational limitations. 

---
# The Script Tax: Measuring Tokenization-Driven Efficiency and Latency Disparities in Multilingual Language Models 

**Authors**: Aradhya Dixit, Shreem Dixit  

**Link**: [PDF](https://arxiv.org/pdf/2602.11174)  

**Abstract**: Pretrained multilingual language models are often assumed to be script-agnostic, yet their tokenizers can impose systematic costs on certain writing systems. We quantify this script tax by comparing two orthographic variants with identical linguistic content. Across mBERT and XLM-R, the higher-fragmentation orthography shows a ~3.4x increase in fertility (6.73-6.85 vs. 2.10-2.35 tokens/word), leading to a 16.5x inference slowdown (0.23 vs. 3.8 sentences/second) on identical hardware. Using bits per character (BPC) to avoid the "NLL paradox" from subword fragmentation, we find a substantial increase in information cost: +19.7% for mBERT (8.06->9.65) and +47.1% for XLM-R (12.19->17.94). A round-trip conversion check (CER_rt=0.31) suggests these gaps reflect orthography-conditioned processing rather than mapping noise. Our results highlight tokenization as a key source of inequity in multilingual NLP and motivate script-aware tokenization and pretraining. 

---
# Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review 

**Authors**: Qian Ruan, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2602.11173)  

**Abstract**: Author response (rebuttal) writing is a critical stage of scientific peer review that demands substantial author effort. Recent work frames this task as automatic text generation, underusing author expertise and intent. In practice, authors possess domain expertise, author-only information, revision and response strategies--concrete forms of author expertise and intent--to address reviewer concerns, and seek NLP assistance that integrates these signals to support effective response writing in peer review. We reformulate author response generation as an author-in-the-loop task and introduce REspGen, a generation framework that integrates explicit author input, multi-attribute control, and evaluation-guided refinement, together with REspEval, a comprehensive evaluation suite with 20+ metrics covering input utilization, controllability, response quality, and discourse. To support this formulation, we construct Re$^3$Align, the first large-scale dataset of aligned review--response--revision triplets, where revisions provide signals of author expertise and intent. Experiments with state-of-the-art LLMs show the benefits of author input and evaluation-guided refinement, the impact of input design on response quality, and trade-offs between controllability and quality. We make our dataset, generation and evaluation tools publicly available. 

---
# Synthesizing the Virtual Advocate: A Multi-Persona Speech Generation Framework for Diverse Linguistic Jurisdictions in Indic Languages 

**Authors**: Aniket Deroy  

**Link**: [PDF](https://arxiv.org/pdf/2602.11172)  

**Abstract**: Legal advocacy requires a unique combination of authoritative tone, rhythmic pausing for emphasis, and emotional intelligence. This study investigates the performance of the Gemini 2.5 Flash TTS and Gemini 2.5 Pro TTS models in generating synthetic courtroom speeches across five Indic languages: Tamil, Telugu, Bengali, Hindi, and Gujarati. We propose a prompting framework that utilizes Gemini 2.5s native support for 5 languages and its context-aware pacing to produce distinct advocate personas. The evolution of Large Language Models (LLMs) has shifted the focus of TexttoSpeech (TTS) technology from basic intelligibility to context-aware, expressive synthesis. In the legal domain, synthetic speech must convey authority and a specific professional persona a task that becomes significantly more complex in the linguistically diverse landscape of India. The models exhibit a "monotone authority," excelling at procedural information delivery but struggling with the dynamic vocal modulation and emotive gravitas required for persuasive advocacy. Performance dips in Bengali and Gujarati further highlight phonological frontiers for future refinement. This research underscores the readiness of multilingual TTS for procedural legal tasks while identifying the remaining challenges in replicating the persuasive artistry of human legal discourse. The code is available at-this https URL 

---
# Efficient Hyper-Parameter Search for LoRA via Language-aided Bayesian Optimization 

**Authors**: Baek Seong-Eun, Lee Jung-Mok, Kim Sung-Bin, Tae-Hyun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2602.11171)  

**Abstract**: Fine-tuning Large Language Models (LLMs) with Low-Rank Adaptation (LoRA) enables resource-efficient personalization or specialization, but it comes at the expense of additional hyperparameter tuning. Although LoRA makes fine-tuning efficient, it is highly sensitive to the choice of hyperparameters, and exhaustive hyperparameter search is still computationally very demanding. To address these challenges, we propose a framework that integrates the domain knowledge of pre-trained LLMs into Bayesian Optimization (BO) to efficiently search for LoRA hyperparameters. To leverage the informed knowledge of LLMs, we repurpose LLMs as a discrete-to-continuous mapping to link the hyperparameters and their domain knowledge with a continuous vector space, where BO is conducted. We design and control the mapping by language prompting, where we provide a domain-aware textual prompt describing the relationships among hyperparameters and their respective roles; thereby, we explicitly inject domain knowledge about LoRA into the LLM in natural language. Also, we model the residual information that is hard to linguistically describe in the prompt with an additional learnable token. This aids BO to sample more high-performing hyperparameters. In addition, by leveraging the observation of the strong correlation between the respective performance obtained from full and subset training datasets in LoRA training regimes, we introduce proxy training and evaluation with a data subset. This further increases the efficiency of our method. We demonstrate that our hyperparameter found with only about 30 iterations achieves more than 20% performance improvement over standard hyperparameters found from about 45,000 combinations. 

---
# PRIME: Policy-Reinforced Iterative Multi-agent Execution for Algorithmic Reasoning in Large Language Models 

**Authors**: Jiawei Xu, Zhenyu Yu, Ziqian Bi, Minh Duc Pham, Xiaoyi Qu, Danyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11170)  

**Abstract**: Large language models have demonstrated remarkable capabilities across diverse reasoning tasks, yet their performance on algorithmic reasoning remains limited. To handle this limitation, we propose PRIME (Policy-Reinforced Iterative Multi-agent Execution), a framework comprising three specialized agents, an executor for step-by-step reasoning, a verifier for constraint checking, and a coordinator for backtracking control, optimized through group relative policy optimization. For comprehensive evaluation, we introduce PRIME-Bench, the largest algorithmic reasoning benchmark to date, comprising 86 tasks across 12 categories with 51,600 instances. Tasks span sorting algorithms, graph and tree structures, automata and state machines, symbolic reasoning, and constraint-based puzzles, with execution traces reaching over one million steps. Compared to baseline approach, PRIME improves average accuracy from 26.8% to 93.8%, a 250% relative gain. The largest improvements occur on tasks requiring sustained state tracking, with Turing machine simulation improving from 9% to 92% and long division from 16% to 94%. Ablation studies identify iterative verification as the primary contributor, preventing the error propagation that causes baseline approaches to fail catastrophically. Analysis across model scales (8B-120B parameters) reveals that smaller models benefit disproportionately, achieving accuracy comparable to models 8x larger. 

---
# Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis 

**Authors**: Mangadoddi Srikar Vardhan, Lekkala Sai Teja  

**Link**: [PDF](https://arxiv.org/pdf/2602.11169)  

**Abstract**: Transformer hidden states encode information as high-dimensional vectors, yet whether direction (orientation in representational space) and magnitude (vector norm) serve distinct functional roles remains unclear. Studying Pythia-family models, we discover a striking cross-over dissociation: angular perturbations cause up to 42.9 more damage to language modeling loss, while magnitude perturbations cause disproportionately more damage to syntactic processing (20.4% vs.1.6% accuracy drop on subject-verb agreement).This finding is enabled by L2-matched perturbation analysis, a methodology ensuring that an gular and magnitude perturbations achieve identical Euclidean displacements. Causal intervention reveals that angular damage flows substantially through the attention pathways (28.4% loss recovery via attention repair), while magnitude damage flows partly through the LayerNorm pathways(29.9% recovery via LayerNorm repair). These patterns replicate across scales within the Pythia architecture family. These findings provide evidence that direction and magnitude support partially distinct computational roles in LayerNorm based architectures. The direction preferentially affects attentional routing, while magnitude modulates processing intensity for fine-grained syntactic judgments. We find different patterns in RMSNorm-based architectures, suggesting that the dissociation depends on architectural choices. Our results refine the linear representation hypothesis and have implications for model editing and interpretability research 

---
# Enhancing SDG-Text Classification with Combinatorial Fusion Analysis and Generative AI 

**Authors**: Jingyan Xu, Marcelo L. LaFleur, Christina Schweikert, D. Frank Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11168)  

**Abstract**: (Natural Language Processing) NLP techniques such as text classification and topic discovery are very useful in many application areas including information retrieval, knowledge discovery, policy formulation, and decision-making. However, it remains a challenging problem in cases where the categories are unavailable, difficult to differentiate, or are interrelated. Social analysis with human context is an area that can benefit from text classification, as it relies substantially on text data. The focus of this paper is to enhance the classification of text according to the UN's Sustainable Development Goals (SDGs) by collecting and combining intelligence from multiple models. Combinatorial Fusion Analysis (CFA), a system fusion paradigm using a rank-score characteristic (RSC) function and cognitive diversity (CD), has been used to enhance classifier methods by combining a set of relatively good and mutually diverse classification models. We use a generative AI model to generate synthetic data for model training and then apply CFA to this classification task. The CFA technique achieves 96.73% performance, outperforming the best individual model. We compare the outcomes with those obtained from human domain experts. It is demonstrated that combining intelligence from multiple ML/AI models using CFA and getting input from human experts can, not only complement, but also enhance each other. 

---
# Visualizing and Benchmarking LLM Factual Hallucination Tendencies via Internal State Analysis and Clustering 

**Authors**: Nathan Mao, Varun Kaushik, Shreya Shivkumar, Parham Sharafoleslami, Kevin Zhu, Sunishchal Dev  

**Link**: [PDF](https://arxiv.org/pdf/2602.11167)  

**Abstract**: Large Language Models (LLMs) often hallucinate, generating nonsensical or false information that can be especially harmful in sensitive fields such as medicine or law. To study this phenomenon systematically, we introduce FalseCite, a curated dataset designed to capture and benchmark hallucinated responses induced by misleading or fabricated citations. Running GPT-4o-mini, Falcon-7B, and Mistral 7-B through FalseCite, we observed a noticeable increase in hallucination activity for false claims with deceptive citations, especially in GPT-4o-mini. Using the responses from FalseCite, we can also analyze the internal states of hallucinating models, visualizing and clustering the hidden state vectors. From this analysis, we noticed that the hidden state vectors, regardless of hallucination or non-hallucination, tend to trace out a distinct horn-like shape. Our work underscores FalseCite's potential as a foundation for evaluating and mitigating hallucinations in future LLM research. 

---
# Small Updates, Big Doubts: Does Parameter-Efficient Fine-tuning Enhance Hallucination Detection ? 

**Authors**: Xu Hu, Yifan Zhang, Songtao Wei, Chen Zhao, Qiannan Li, Bingzhe Li, Feng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11166)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods are widely used to adapt large language models (LLMs) to downstream tasks and are often assumed to improve factual correctness. However, how the parameter-efficient fine-tuning methods affect hallucination behavior remains insufficiently understood, especially on QA datasets. In this work, we systematically investigate the impact of PEFT on hallucination detection through a comprehensive empirical study across three open-weight LLM backbones and three fact-seeking QA benchmarks. For each model, we evaluate performance using seven unsupervised hallucination detection methods spanning three complementary approaches: semantic consistency based detectors, confidence based detectors, and entropy based detectors. This multifaceted evaluation enables us to characterize how PEFT reshapes uncertainty across different detection paradigms. In conclusion, our experimental results show that PEFT consistently strengthens hallucination detection ability, substantially improving AUROC across a wide range of hallucination detectors. Besides, further analyses using linear probes and representation diagnostics indicate that PEFT methods primarily reshapes how uncertainty is encoded and surfaced, comparing with injecting new factual knowledge into the models. 

---
# Assessing LLM Reliability on Temporally Recent Open-Domain Questions 

**Authors**: Pushwitha Krishnappa, Amit Das, Vinija Jain, Tathagata Mukherjee, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2602.11165)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed for open-domain question answering, yet their alignment with human perspectives on temporally recent information remains underexplored. We introduce RECOM (Reddit Evaluation for Correspondence of Models), a benchmark dataset of 15,000 recent Reddit questions from September 2025 paired with community-derived reference answers. We investigate how four open-source LLMs (Llama3.1-8B, Mistral-7B, Gemma-2-9B, and GPT-OSS-20B) respond to these questions, evaluating alignment using lexical metrics (BLEU, ROUGE), semantic similarity (BERTScore, MoverScore, cosine similarity), and logical inference (NLI). Our central finding is a striking semantic-lexical paradox: all models achieve over 99% cosine similarity with references despite less than 8% BLEU-1 overlap, a 90+ percentage point gap indicating that models preserve meaning through extensive paraphrasing rather than lexical reproduction. MoverScore (51-53%) confirms this pattern, occupying an intermediate position that reflects the optimal transport cost of semantic alignment. Furthermore, model scale does not predict performance: Mistral-7B (7B parameters) outperforms GPT-OSS-20B (20B parameters) across all metrics. NLI analysis reveals that contradiction rates remain below 7%, suggesting models rarely generate content that directly conflicts with human consensus. These findings challenge the reliability of lexical metrics for evaluating abstractive generation and argue for multi-dimensional evaluation frameworks that capture semantic fidelity beyond surface-level text matching. The RECOM dataset is publicly available at this https URL 

---
# Nested Named Entity Recognition in Plasma Physics Research Articles 

**Authors**: Muhammad Haris, Hans Höft, Markus M. Becker, Markus Stocker  

**Link**: [PDF](https://arxiv.org/pdf/2602.11163)  

**Abstract**: Named Entity Recognition (NER) is an important task in natural language processing that aims to identify and extract key entities from unstructured text. We present a novel application of NER in plasma physics research articles and address the challenges of extracting specialized entities from scientific text in this domain. Research articles in plasma physics often contain highly complex and context-rich content that must be extracted to enable, e.g., advanced search. We propose a lightweight approach based on encoder-transformers and conditional random fields to extract (nested) named entities from plasma physics research articles. First, we annotate a plasma physics corpus with 16 classes specifically designed for the nested NER task. Second, we evaluate an entity-specific model specialization approach, where independent BERT-CRF models are trained to recognize individual entity types in plasma physics text. Third, we integrate an optimization process to systematically fine-tune hyperparameters and enhance model performance. Our work contributes to the advancement of entity recognition in plasma physics and also provides a foundation to support researchers in navigating and analyzing scientific literature. 

---
# Retrieval Heads are Dynamic 

**Authors**: Yuping Lin, Zitao Li, Yue Xing, Pengfei He, Yingqian Cui, Yaliang Li, Bolin Ding, Jingren Zhou, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11162)  

**Abstract**: Recent studies have identified "retrieval heads" in Large Language Models (LLMs) responsible for extracting information from input contexts. However, prior works largely rely on static statistics aggregated across datasets, identifying heads that perform retrieval on average. This perspective overlooks the fine-grained temporal dynamics of autoregressive generation. In this paper, we investigate retrieval heads from a dynamic perspective. Through extensive analysis, we establish three core claims: (1) Dynamism: Retrieval heads vary dynamically across timesteps; (2) Irreplaceability: Dynamic retrieval heads are specific at each timestep and cannot be effectively replaced by static retrieval heads; and (3) Correlation: The model's hidden state encodes a predictive signal for future retrieval head patterns, indicating an internal planning mechanism. We validate these findings on the Needle-in-a-Haystack task and a multi-hop QA task, and quantify the differences on the utility of dynamic and static retrieval heads in a Dynamic Retrieval-Augmented Generation framework. Our study provides new insights into the internal mechanisms of LLMs. 

---
# Response-Based Knowledge Distillation for Multilingual Jailbreak Prevention Unwittingly Compromises Safety 

**Authors**: Max Zhang, Derek Liu, Kai Zhang, Joshua Franco, Haihao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11157)  

**Abstract**: Large language models (LLMs) are increasingly deployed worldwide, yet their safety alignment remains predominantly English-centric. This allows for vulnerabilities in non-English contexts, especially with low-resource languages. We introduce a novel application of knowledge distillation (KD) in the context of multilingual jailbreak prevention, examining its efficacy. We distill the refusal behaviors of a proprietary teacher model (OpenAI o1-mini) with Low-Rank Adaptation (LoRA) into three open-source student models: Meta-Llama-3-8B-Instruct, Gemma-2-2B-IT, and Qwen3-8B, using ~28,000 multilingual jailbreak prompts from XSafety via black-box response-based, parameter-efficient fine-tuning (PEFT). Evaluation on the MultiJail benchmark reveals a counterintuitive behavior: standard fine-tuning on the teacher's ``safe'' refusal data inadvertently increases Jailbreak Success Rate (JSR) for all student models, up to 16.6 percentage points. Our experiments reveal a divergent generalization to unseen languages during distillation, with varying outcomes depending on the base model. By removing a primary source of safety degradation, nuanced `boundary' refusals, we mitigate or even reverse safety declines in student models, although reductions in reasoning performance (GSM8K) persist. Overall, our exploratory study highlights the challenges and potential of KD as a technique for multilingual safety alignment, offering a foundation for future research in this direction. 

---
# HybridRAG: A Practical LLM-based ChatBot Framework based on Pre-Generated Q&A over Raw Unstructured Documents 

**Authors**: Sungmoon Kim, Hyuna Jeon, Dahye Kim, Mingyu Kim, Dong-Kyu Chae, Jiwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.11156)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful approach for grounding Large Language Model (LLM)-based chatbot responses on external knowledge. However, existing RAG studies typically assume well-structured textual sources (e.g. Wikipedia or curated datasets) and perform retrieval and generation at query time, which can limit their applicability in real-world chatbot scenarios. In this paper, we present HybridRAG, a novel and practical RAG framework towards more accurate and faster chatbot responses. First, HybridRAG ingests raw, unstructured PDF documents containing complex layouts (text, tables, figures) via Optical Character Recognition (OCR) and layout analysis, and convert them into hierarchical text chunks. Then, it pre-generates a plausible question-answer (QA) knowledge base from the organized chunks using an LLM. At query time, user questions are matched against this QA bank to retrieve immediate answers when possible, and only if no suitable QA match is found does our framework fall back to an on-the-fly response generation. Experiments on OHRBench demonstrate that our HybridRAG provides higher answer quality and lower latency compared to a standard RAG baseline. We believe that HybridRAG could be a practical solution for real-world chatbot applications that must handle large volumes of unstructured documents and lots of users under limited computational resources. 

---
# Agentic Test-Time Scaling for WebAgents 

**Authors**: Nicholas Lee, Lutfi Eren Erdogan, Chris Joseph John, Surya Krishnapillai, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2602.12276)  

**Abstract**: Test-time scaling has become a standard way to improve performance and boost reliability of neural network models. However, its behavior on agentic, multi-step tasks remains less well-understood: small per-step errors can compound over long horizons; and we find that naive policies that uniformly increase sampling show diminishing returns. In this work, we present CATTS, a simple technique for dynamically allocating compute for multi-step agents. We first conduct an empirical study of inference-time scaling for web agents. We find that uniformly increasing per-step compute quickly saturates in long-horizon environments. We then investigate stronger aggregation strategies, including an LLM-based Arbiter that can outperform naive voting, but that can overrule high-consensus decisions. We show that uncertainty statistics derived from the agent's own vote distribution (entropy and top-1/top-2 margin) correlate with downstream success and provide a practical signal for dynamic compute allocation. Based on these findings, we introduce Confidence-Aware Test-Time Scaling (CATTS), which uses vote-derived uncertainty to allocate compute only when decisions are genuinely contentious. CATTS improves performance on WebArena-Lite and GoBrowse by up to 9.1% over React while using up to 2.3x fewer tokens than uniform scaling, providing both efficiency gains and an interpretable decision rule. 

---
# "Sorry, I Didn't Catch That": How Speech Models Miss What Matters Most 

**Authors**: Kaitlyn Zhou, Martijn Bartelds, Federico Bianchi, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2602.12249)  

**Abstract**: Despite speech recognition systems achieving low word error rates on standard benchmarks, they often fail on short, high-stakes utterances in real-world deployments. Here, we study this failure mode in a high-stakes task: the transcription of U.S. street names as spoken by U.S. participants. We evaluate 15 models from OpenAI, Deepgram, Google, and Microsoft on recordings from linguistically diverse U.S. speakers and find an average transcription error rate of 44%. We quantify the downstream impact of failed transcriptions by geographic locations and show that mis-transcriptions systematically cause errors for all speakers, but that routing distance errors are twice as large for non-English primary speakers compared to English primary speakers. To mitigate this harm, we introduce a synthetic data generation approach that produces diverse pronunciations of named entities using open-source text-to-speech models. Fine-tuning with less than 1,000 synthetic samples improves street name transcription accuracy by nearly 60% (relative to base models) for non-English primary speakers. Our results highlight a critical gap between benchmark performance and real-world reliability in speech systems and demonstrate a simple, scalable path to reducing high-stakes transcription errors. 

---
# Olmix: A Framework for Data Mixing Throughout LM Development 

**Authors**: Mayee F. Chen, Tyler Murray, David Heineman, Matt Jordan, Hannaneh Hajishirzi, Christopher Ré, Luca Soldaini, Kyle Lo  

**Link**: [PDF](https://arxiv.org/pdf/2602.12237)  

**Abstract**: Data mixing -- determining the ratios of data from different domains -- is a first-order concern for training language models (LMs). While existing mixing methods show promise, they fall short when applied during real-world LM development. We present Olmix, a framework that addresses two such challenges. First, the configuration space for developing a mixing method is not well understood -- design choices across existing methods lack justification or consensus and overlook practical issues like data constraints. We conduct a comprehensive empirical study of this space, identifying which design choices lead to a strong mixing method. Second, in practice, the domain set evolves throughout LM development as datasets are added, removed, partitioned, and revised -- a problem setting largely unaddressed by existing works, which assume fixed domains. We study how to efficiently recompute the mixture after the domain set is updated, leveraging information from past mixtures. We introduce mixture reuse, a mechanism that reuses existing ratios and recomputes ratios only for domains affected by the update. Over a sequence of five domain-set updates mirroring real-world LM development, mixture reuse matches the performance of fully recomputing the mix after each update with 74% less compute and improves over training without mixing by 11.6% on downstream tasks. 

---
# Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation 

**Authors**: Bowei He, Yankai Chen, Xiaokun Zhang, Linghe Kong, Philip S. Yu, Xue Liu, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.12172)  

**Abstract**: Knowledge distillation from Large Language Models (LLMs) to smaller models has emerged as a critical technique for deploying efficient AI systems. However, current methods for distillation via synthetic data lack pedagogical awareness, treating knowledge transfer as a one-off data synthesis and training task rather than a systematic learning process. In this paper, we propose a novel pedagogically-inspired framework for LLM knowledge distillation that draws from fundamental educational principles. Our approach introduces a three-stage pipeline -- Knowledge Identifier, Organizer, and Adapter (IOA) -- that systematically identifies knowledge deficiencies in student models, organizes knowledge delivery through progressive curricula, and adapts representations to match the cognitive capacity of student models. We integrate Bloom's Mastery Learning Principles and Vygotsky's Zone of Proximal Development to create a dynamic distillation process where student models approach teacher model's performance on prerequisite knowledge before advancing, and new knowledge is introduced with controlled, gradual difficulty increments. Extensive experiments using LLaMA-3.1/3.2 and Qwen2.5 as student models demonstrate that IOA achieves significant improvements over baseline distillation methods, with student models retaining 94.7% of teacher performance on DollyEval while using less than 1/10th of the parameters. Our framework particularly excels in complex reasoning tasks, showing 19.2% improvement on MATH and 22.3% on HumanEval compared with state-of-the-art baselines. 

---
# GPT-4o Lacks Core Features of Theory of Mind 

**Authors**: John Muchovej, Amanda Royka, Shane Lee, Julian Jara-Ettinger  

**Link**: [PDF](https://arxiv.org/pdf/2602.12150)  

**Abstract**: Do Large Language Models (LLMs) possess a Theory of Mind (ToM)? Research into this question has focused on evaluating LLMs against benchmarks and found success across a range of social tasks. However, these evaluations do not test for the actual representations posited by ToM: namely, a causal model of mental states and behavior. Here, we use a cognitively-grounded definition of ToM to develop and test a new evaluation framework. Specifically, our approach probes whether LLMs have a coherent, domain-general, and consistent model of how mental states cause behavior -- regardless of whether that model matches a human-like ToM. We find that even though LLMs succeed in approximating human judgments in a simple ToM paradigm, they fail at a logically equivalent task and exhibit low consistency between their action predictions and corresponding mental state inferences. As such, these findings suggest that the social proficiency exhibited by LLMs is not the result of an domain-general or consistent ToM. 

---
# Seq2Seq2Seq: Lossless Data Compression via Discrete Latent Transformers and Reinforcement Learning 

**Authors**: Mahdi Khodabandeh, Ghazal Shabani, Arash Yousefi Jordehi, Seyed Abolghasem Mirroshandel  

**Link**: [PDF](https://arxiv.org/pdf/2602.12146)  

**Abstract**: Efficient lossless compression is essential for minimizing storage costs and transmission overhead while preserving data integrity. Traditional compression techniques, such as dictionary-based and statistical methods, often struggle to optimally exploit the structure and redundancy in complex data formats. Recent advancements in deep learning have opened new avenues for compression; however, many existing approaches depend on dense vector representations that obscure the underlying token structure. To address these limitations, we propose a novel lossless compression method that leverages Reinforcement Learning applied to a T5 language model architecture. This approach enables the compression of data into sequences of tokens rather than traditional vector representations. Unlike auto-encoders, which typically encode information into continuous latent spaces, our method preserves the token-based structure, aligning more closely with the original data format. This preservation allows for higher compression ratios while maintaining semantic integrity. By training the model using an off-policy Reinforcement Learning algorithm, we optimize sequence length to minimize redundancy and enhance compression efficiency. Our method introduces an efficient and adaptive data compression system built upon advanced Reinforcement Learning techniques, functioning independently of external grammatical or world knowledge. This approach shows significant improvements in compression ratios compared to conventional methods. By leveraging the latent information within language models, our system effectively compresses data without requiring explicit content understanding, paving the way for more robust and practical compression solutions across various applications. 

---
# Neutral Prompts, Non-Neutral People: Quantifying Gender and Skin-Tone Bias in Gemini Flash 2.5 Image and GPT Image 1.5 

**Authors**: Roberto Balestri  

**Link**: [PDF](https://arxiv.org/pdf/2602.12133)  

**Abstract**: This study quantifies gender and skin-tone bias in two widely deployed commercial image generators - Gemini Flash 2.5 Image (NanoBanana) and GPT Image 1.5 - to test the assumption that neutral prompts yield demographically neutral outputs. We generated 3,200 photorealistic images using four semantically neutral prompts. The analysis employed a rigorous pipeline combining hybrid color normalization, facial landmark masking, and perceptually uniform skin tone quantification using the Monk (MST), PERLA, and Fitzpatrick scales. Neutral prompts produced highly polarized defaults. Both models exhibited a strong "default white" bias (>96% of outputs). However, they diverged sharply on gender: Gemini favored female-presenting subjects, while GPT favored male-presenting subjects with lighter skin tones. This research provides a large-scale, comparative audit of state-of-the-art models using an illumination-aware colorimetric methodology, distinguishing aesthetic rendering from underlying pigmentation in synthetic imagery. The study demonstrates that neutral prompts function as diagnostic probes rather than neutral instructions. It offers a robust framework for auditing algorithmic visual culture and challenges the sociolinguistic assumption that unmarked language results in inclusive representation. 

---
# Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation 

**Authors**: Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2602.12125)  

**Abstract**: On-policy distillation (OPD), which aligns the student with the teacher's logit distribution on student-generated trajectories, has demonstrated strong empirical gains in improving student performance and often outperforms off-policy distillation and reinforcement learning (RL) paradigms. In this work, we first theoretically show that OPD is a special case of dense KL-constrained RL where the reward function and the KL regularization are always weighted equally and the reference model can by any model. Then, we propose the Generalized On-Policy Distillation (G-OPD) framework, which extends the standard OPD objective by introducing a flexible reference model and a reward scaling factor that controls the relative weight of the reward term against the KL regularization. Through comprehensive experiments on math reasoning and code generation tasks, we derive two novel insights: (1) Setting the reward scaling factor to be greater than 1 (i.e., reward extrapolation), which we term ExOPD, consistently improves over standard OPD across a range of teacher-student size pairings. In particular, in the setting where we merge the knowledge from different domain experts, obtained by applying domain-specific RL to the same student model, back into the original student, ExOPD enables the student to even surpass the teacher's performance boundary and outperform the domain teachers. (2) Building on ExOPD, we further find that in the strong-to-weak distillation setting (i.e., distilling a smaller student from a larger teacher), performing reward correction by choosing the reference model as the teacher's base model before RL yields a more accurate reward signal and further improves distillation performance. However, this choice assumes access to the teacher's pre-RL variant and incurs more computational overhead. We hope our work offers new insights for future research on OPD. 

---
# Capability-Oriented Training Induced Alignment Risk 

**Authors**: Yujun Zhou, Yue Huang, Han Bao, Kehan Guo, Zhenwen Liang, Pin-Yu Chen, Tian Gao, Werner Geyer, Nuno Moniz, Nitesh V Chawla, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12124)  

**Abstract**: While most AI alignment research focuses on preventing models from generating explicitly harmful content, a more subtle risk is emerging: capability-oriented training induced exploitation. We investigate whether language models, when trained with reinforcement learning (RL) in environments with implicit loopholes, will spontaneously learn to exploit these flaws to maximize their reward, even without any malicious intent in their training. To test this, we design a suite of four diverse "vulnerability games", each presenting a unique, exploitable flaw related to context-conditional compliance, proxy metrics, reward tampering, and self-evaluation. Our experiments show that models consistently learn to exploit these vulnerabilities, discovering opportunistic strategies that significantly increase their reward at the expense of task correctness or safety. More critically, we find that these exploitative strategies are not narrow "tricks" but generalizable skills; they can be transferred to new tasks and even "distilled" from a capable teacher model to other student models through data alone. Our findings reveal that capability-oriented training induced risks pose a fundamental challenge to current alignment approaches, suggesting that future AI safety work must extend beyond content moderation to rigorously auditing and securing the training environments and reward mechanisms themselves. Code is available at this https URL. 

---
# Meta-Sel: Efficient Demonstration Selection for In-Context Learning via Supervised Meta-Learning 

**Authors**: Xubin Wang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2602.12123)  

**Abstract**: Demonstration selection is a practical bottleneck in in-context learning (ICL): under a tight prompt budget, accuracy can change substantially depending on which few-shot examples are included, yet selection must remain cheap enough to run per query over large candidate pools. We propose Meta-Sel, a lightweight supervised meta-learning approach for intent classification that learns a fast, interpretable scoring function for (candidate, query) pairs from labeled training data.
Meta-Sel constructs a meta-dataset by sampling pairs from the training split and using class agreement as supervision, then trains a calibrated logistic regressor on two inexpensive meta-features: TF--IDF cosine similarity and a length-compatibility ratio. At inference time, the selector performs a single vectorized scoring pass over the full candidate pool and returns the top-k demonstrations, requiring no model fine-tuning, no online exploration, and no additional LLM calls. This yields deterministic rankings and makes the selection mechanism straightforward to audit via interpretable feature weights.
Beyond proposing Meta-Sel, we provide a broad empirical study of demonstration selection, benchmarking 12 methods -- spanning prompt engineering baselines, heuristic selection, reinforcement learning, and influence-based approaches -- across four intent datasets and five open-source LLMs. Across this benchmark, Meta-Sel consistently ranks among the top-performing methods, is particularly effective for smaller models where selection quality can partially compensate for limited model capacity, and maintains competitive selection-time overhead. 

---
# Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty 

**Authors**: Zewei Yu, Lirong Gao, Yuke Zhu, Bo Zheng, Sheng Guo, Haobo Wang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.12113)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex reasoning tasks by employing test-time scaling. However, they often generate over-long chains-of-thought that, driven by substantial reflections such as repetitive self-questioning and circular reasoning, lead to high token consumption, substantial computational overhead, and increased latency without improving accuracy, particularly in smaller models. Our observation reveals that increasing problem complexity induces more excessive and unnecessary reflection, which in turn reduces accuracy and increases token overhead. To address this challenge, we propose Adaptive Reflection and Length Coordinated Penalty (ARLCP), a novel reinforcement learning framework designed to dynamically balance reasoning efficiency and solution accuracy. ARLCP introduces two key innovations: (1) a reflection penalty that adaptively curtails unnecessary reflective steps while preserving essential reasoning, and (2) a length penalty calibrated to the estimated complexity of the problem. By coordinating these penalties, ARLCP encourages the model to generate more concise and effective reasoning paths. We evaluate our method on five mathematical reasoning benchmarks using DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B models. Experimental results show that ARLCP achieves a superior efficiency-accuracy trade-off compared to existing approaches. For the 1.5B model, it reduces the average response length by 53.1% while simultaneously improving accuracy by 5.8%. For the 7B model, it achieves a 35.0% reduction in length with a 2.7% accuracy gain. The code is released at this https URL . 

---
# Tiny Recursive Reasoning with Mamba-2 Attention Hybrid 

**Authors**: Wenlong Wang, Fergal Reid  

**Link**: [PDF](https://arxiv.org/pdf/2602.12078)  

**Abstract**: Recent work on recursive reasoning models like TRM demonstrates that tiny networks (7M parameters) can achieve strong performance on abstract reasoning tasks through latent recursion -- iterative refinement in hidden representation space without emitting intermediate tokens. This raises a natural question about operator choice: Mamba-2's state space recurrence is itself a form of iterative refinement, making it a natural candidate for recursive reasoning -- but does introducing Mamba-2 into the recursive scaffold preserve reasoning capability? We investigate this by replacing the Transformer blocks in TRM with Mamba-2 hybrid operators while maintaining parameter parity (6.83M vs 6.86M parameters). On ARC-AGI-1, we find that the hybrid improves pass@2 (the official metric) by +2.0\% (45.88\% vs 43.88\%) and consistently outperforms at higher K values (+4.75\% at pass@100), whilst maintaining pass@1 parity. This suggests improved candidate coverage -- the model generates correct solutions more reliably -- with similar top-1 selection. Our results validate that Mamba-2 hybrid operators preserve reasoning capability within the recursive scaffold, establishing SSM-based operators as viable candidates in the recursive operator design space and taking a first step towards understanding the best mixing strategies for recursive reasoning. 

---
# Artificial intelligence is creating a new global linguistic hierarchy 

**Authors**: Giulia Occhini, Kumiko Tanaka-Ishii, Anna Barford, Refael Tikochinski, Songbo Hu, Roi Reichart, Yijie Zhou, Hannah Claus, Ulla Petti, Ivan Vulić, Ramit Debnath, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2602.12018)  

**Abstract**: Artificial intelligence (AI) has the potential to transform healthcare, education, governance and socioeconomic equity, but its benefits remain concentrated in a small number of languages (Bender, 2019; Blasi et al., 2022; Joshi et al., 2020; Ranathunga and de Silva, 2022; Young, 2015). Language AI - the technologies that underpin widely-used conversational systems such as ChatGPT - could provide major benefits if available in people's native languages, yet most of the world's 7,000+ linguistic communities currently lack access and face persistent digital marginalization. Here we present a global longitudinal analysis of social, economic and infrastructural conditions across languages to assess systemic inequalities in language AI. We first analyze the existence of AI resources for 6003 languages. We find that despite efforts of the community to broaden the reach of language technologies (Bapna et al., 2022; Costa-Jussà et al., 2022), the dominance of a handful of languages is exacerbating disparities on an unprecedented scale, with divides widening exponentially rather than narrowing. Further, we contrast the longitudinal diffusion of AI with that of earlier IT technologies, revealing a distinctive hype-driven pattern of spread. To translate our findings into practical insights and guide prioritization efforts, we introduce the Language AI Readiness Index (EQUATE), which maps the state of technological, socio-economic, and infrastructural prerequisites for AI deployment across languages. The index highlights communities where capacity exists but remains underutilized, and provides a framework for accelerating more equitable diffusion of language AI. Our work contributes to setting the baseline for a transition towards more sustainable and equitable language technologies. 

---
# Benchmarking Vision-Language Models for French PDF-to-Markdown Conversion 

**Authors**: Bruno Rigal, Victor Dupriez, Alexis Mignon, Ronan Le Hy, Nicolas Mery  

**Link**: [PDF](https://arxiv.org/pdf/2602.11960)  

**Abstract**: This report evaluates PDF-to-Markdown conversion using recent Vision-Language Models (VLMs) on challenging French documents. Document parsing is a critical step for Retrieval-Augmented Generation (RAG) pipelines, where transcription and layout errors propagate to downstream retrieval and grounding. Existing benchmarks often emphasize English or Chinese and can over-penalize benign formatting and linearization choices (e.g., line breaks, list segmentation, alternative table renderings) that are largely irrelevant for downstream use.
We introduce a French-focused benchmark of difficult pages selected via model-disagreement sampling from a corpus of 60{,}000 documents, covering handwritten forms, complex layouts, dense tables, and graphics-rich pages. Evaluation is performed with unit-test-style checks that target concrete failure modes (text presence, reading order, and local table constraints) combined with category-specific normalization designed to discount presentation-only variance. Across 15 models, we observe substantially higher robustness for the strongest proprietary models on handwriting and forms, while several open-weights systems remain competitive on standard printed layouts. 

---
# RAM-Net: Expressive Linear Attention with Selectively Addressable Memory 

**Authors**: Kaicheng Xiao, Haotian Li, Liran Dong, Guoliang Xing  

**Link**: [PDF](https://arxiv.org/pdf/2602.11958)  

**Abstract**: While linear attention architectures offer efficient inference, compressing unbounded history into a fixed-size memory inherently limits expressivity and causes information loss. To address this limitation, we introduce Random Access Memory Network (RAM-Net), a novel architecture designed to bridge the gap between the representational capacity of full attention and the memory efficiency of linear models. The core of RAM-Net maps inputs to high-dimensional sparse vectors serving as explicit addresses, allowing the model to selectively access a massive memory state. This design enables exponential state size scaling without additional parameters, which significantly mitigates signal interference and enhances retrieval fidelity. Moreover, the inherent sparsity ensures exceptional computational efficiency, as state updates are confined to minimal entries. Extensive experiments demonstrate that RAM-Net consistently surpasses state-of-the-art baselines in fine-grained long-range retrieval tasks and achieves competitive performance in standard language modeling and zero-shot commonsense reasoning benchmarks, validating its superior capability to capture complex dependencies with significantly reduced computational overhead. 

---
# When Should LLMs Be Less Specific? Selective Abstraction for Reliable Long-Form Text Generation 

**Authors**: Shani Goren, Ido Galil, Ran El-Yaniv  

**Link**: [PDF](https://arxiv.org/pdf/2602.11908)  

**Abstract**: LLMs are widely used, yet they remain prone to factual errors that erode user trust and limit adoption in high-risk settings. One approach to mitigate this risk is to equip models with uncertainty estimation mechanisms that abstain when confidence is low. However, this binary "all-or-nothing" approach is excessively restrictive in long-form settings, often discarding valuable information. We introduce Selective Abstraction (SA), a framework that enables LLMs to trade specificity for reliability by selectively reducing the detail of uncertain content. We first formalize SA through the lenses of selective risk and coverage. We then propose Atom-wise Selective Abstraction, a claim-level instantiation that decomposes responses into atomic claims (short, self-contained statements each expressing a single fact) and replaces uncertain atoms with higher confidence, less specific abstractions. To evaluate this framework, we develop a novel end-to-end pipeline for open-ended generation that instantiates risk as factual correctness and measures coverage using an information-theoretic measure of retained information. Across six open-source models on the FactScore and LongFact-Objects benchmarks, atom-wise SA consistently outperforms existing baselines, improving the area under the risk-coverage curve (AURC) by up to 27.73% over claim removal, demonstrating that reducing specificity can boost accuracy and reliability while preserving most of their original meaning. 

---
# A$^{2}$V-SLP: Alignment-Aware Variational Modeling for Disentangled Sign Language Production 

**Authors**: Sümeyye Meryem Taşyürek, Enis Mücahid İskender, Hacer Yalim Keles  

**Link**: [PDF](https://arxiv.org/pdf/2602.11861)  

**Abstract**: Building upon recent structural disentanglement frameworks for sign language production, we propose A$^{2}$V-SLP, an alignment-aware variational framework that learns articulator-wise disentangled latent distributions rather than deterministic embeddings. A disentangled Variational Autoencoder (VAE) encodes ground-truth sign pose sequences and extracts articulator-specific mean and variance vectors, which are used as distributional supervision for training a non-autoregressive Transformer. Given text embeddings, the Transformer predicts both latent means and log-variances, while the VAE decoder reconstructs the final sign pose sequences through stochastic sampling at the decoding stage. This formulation maintains articulator-level representations by avoiding deterministic latent collapse through distributional latent modeling. In addition, we integrate a gloss attention mechanism to strengthen alignment between linguistic input and articulated motion. Experimental results show consistent gains over deterministic latent regression, achieving state-of-the-art back-translation performance and improved motion realism in a fully gloss-free setting. 

---
# Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception 

**Authors**: Lai Wei, Liangbo He, Jun Lan, Lingzhong Dong, Yutong Cai, Siyuan Li, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, Zhuosheng Zhang, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11858)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel at broad visual understanding but still struggle with fine-grained perception, where decisive evidence is small and easily overwhelmed by global context. Recent "Thinking-with-Images" methods alleviate this by iteratively zooming in and out regions of interest during inference, but incur high latency due to repeated tool calls and visual re-encoding. To address this, we propose Region-to-Image Distillation, which transforms zooming from an inference-time tool into a training-time primitive, thereby internalizing the benefits of agentic zooming into a single forward pass of an MLLM. In particular, we first zoom in to micro-cropped regions to let strong teacher models generate high-quality VQA data, and then distill this region-grounded supervision back to the full image. After training on such data, the smaller student model improves "single-glance" fine-grained perception without tool use. To rigorously evaluate this capability, we further present ZoomBench, a hybrid-annotated benchmark of 845 VQA data spanning six fine-grained perceptual dimensions, together with a dual-view protocol that quantifies the global--regional "zooming gap". Experiments show that our models achieve leading performance across multiple fine-grained perception benchmarks, and also improve general multimodal cognition on benchmarks such as visual reasoning and GUI agents. We further discuss when "Thinking-with-Images" is necessary versus when its gains can be distilled into a single forward pass. Our code is available at this https URL. 

---
# Prototype Transformer: Towards Language Model Architectures Interpretable by Design 

**Authors**: Yordan Yordanov, Matteo Forasassi, Bayar Menzat, Ruizhi Wang, Chang Qi, Markus Kaltenberger, Amine M'Charrak, Tommaso Salvatori, Thomas Lukasiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2602.11852)  

**Abstract**: While state-of-the-art language models (LMs) surpass the vast majority of humans in certain domains, their reasoning remains largely opaque, undermining trust in their output. Furthermore, while autoregressive LMs can output explicit reasoning, their true reasoning process is opaque, which introduces risks like deception and hallucination. In this work, we introduce the Prototype Transformer (ProtoT) -- an autoregressive LM architecture based on prototypes (parameter vectors), posed as an alternative to the standard self-attention-based transformers. ProtoT works by means of two-way communication between the input sequence and the prototypes, and we show that this leads to the prototypes automatically capturing nameable concepts (e.g. "woman") during training. They provide the potential to interpret the model's reasoning and allow for targeted edits of its behavior. Furthermore, by design, the prototypes create communication channels that aggregate contextual information at different time scales, aiding interpretability. In terms of computation scalability, ProtoT scales linearly with sequence length vs the quadratic scalability of SOTA self-attention transformers. Compared to baselines, ProtoT scales well with model and data size, and performs well on text generation and downstream tasks (GLUE). ProtoT exhibits robustness to input perturbations on par or better than some baselines, but differs from them by providing interpretable pathways showing how robustness and sensitivity arises. Reaching close to the performance of state-of-the-art architectures, ProtoT paves the way to creating well-performing autoregressive LMs interpretable by design. 

---
# More Haste, Less Speed: Weaker Single-Layer Watermark Improves Distortion-Free Watermark Ensembles 

**Authors**: Ruibo Chen, Yihan Wu, Xuehao Cui, Jingqi Zhang, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11793)  

**Abstract**: Watermarking has emerged as a crucial technique for detecting and attributing content generated by large language models. While recent advancements have utilized watermark ensembles to enhance robustness, prevailing methods typically prioritize maximizing the strength of the watermark at every individual layer. In this work, we identify a critical limitation in this "stronger-is-better" approach: strong watermarks significantly reduce the entropy of the token distribution, which paradoxically weakens the effectiveness of watermarking in subsequent layers. We theoretically and empirically show that detectability is bounded by entropy and that watermark ensembles induce a monotonic decrease in both entropy and the expected green-list ratio across layers. To address this inherent trade-off, we propose a general framework that utilizes weaker single-layer watermarks to preserve the entropy required for effective multi-layer ensembling. Empirical evaluations demonstrate that this counter-intuitive strategy mitigates signal decay and consistently outperforms strong baselines in both detectability and robustness. 

---
# Detecting RLVR Training Data via Structural Convergence of Reasoning 

**Authors**: Hongbo Zhang, Yue Yang, Jianhao Yan, Guangsheng Bao, Yue Zhang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11792)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is central to training modern reasoning models, but the undisclosed training data raises concerns about benchmark contamination. Unlike pretraining methods, which optimize models using token-level probabilities, RLVR fine-tunes models based on reward feedback from self-generated reasoning trajectories, making conventional likelihood-based detection methods less effective. We show that RLVR induces a distinctive behavioral signature: prompts encountered during RLVR training result in more rigid and similar generations, while unseen prompts retain greater diversity. We introduce Min-$k$NN Distance, a simple black-box detector that quantifies this collapse by sampling multiple completions for a given prompt and computing the average of the $k$ smallest nearest-neighbor edit distances. Min-$k$NN Distance requires no access to the reference model or token probabilities. Experiments across multiple RLVR-trained reasoning models show that Min-$k$NN Distance reliably distinguishes RL-seen examples from unseen ones and outperforms existing membership inference and RL contamination detection baselines. 

---
# Beyond End-to-End Video Models: An LLM-Based Multi-Agent System for Educational Video Generation 

**Authors**: Lingyong Yan, Jiulong Wu, Dong Xie, Weixian Shi, Deguo Xia, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11790)  

**Abstract**: Although recent end-to-end video generation models demonstrate impressive performance in visually oriented content creation, they remain limited in scenarios that require strict logical rigor and precise knowledge representation, such as instructional and educational media. To address this problem, we propose LAVES, a hierarchical LLM-based multi-agent system for generating high-quality instructional videos from educational problems. The LAVES formulates educational video generation as a multi-objective task that simultaneously demands correct step-by-step reasoning, pedagogically coherent narration, semantically faithful visual demonstrations, and precise audio--visual alignment. To address the limitations of prior approaches--including low procedural fidelity, high production cost, and limited controllability--LAVES decomposes the generation workflow into specialized agents coordinated by a central Orchestrating Agent with explicit quality gates and iterative critique mechanisms. Specifically, the Orchestrating Agent supervises a Solution Agent for rigorous problem solving, an Illustration Agent that produces executable visualization codes, and a Narration Agent for learner-oriented instructional scripts. In addition, all outputs from the working agents are subject to semantic critique, rule-based constraints, and tool-based compilation checks. Rather than directly synthesizing pixels, the system constructs a structured executable video script that is deterministically compiled into synchronized visuals and narration using template-driven assembly rules, enabling fully automated end-to-end production without manual editing. In large-scale deployments, LAVES achieves a throughput exceeding one million videos per day, delivering over a 95% reduction in cost compared to current industry-standard approaches while maintaining a high acceptance rate. 

---
# TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2602.11767)  

**Abstract**: Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using task-specific feedback. This improves rollout quality and stabilizes learning while leaving the underlying optimization objective unchanged, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a simple and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods. 

---
# Mask What Matters: Mitigating Object Hallucinations in Multimodal Large Language Models with Object-Aligned Visual Contrastive Decoding 

**Authors**: Boqi Chen, Xudong Liu, Jianing Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11737)  

**Abstract**: We study object hallucination in Multimodal Large Language Models (MLLMs) and improve visual contrastive decoding (VCD) by constructing an object-aligned auxiliary view. We leverage object-centric attention in self-supervised Vision Transformers. In particular, we remove the most salient visual evidence to construct an auxiliary view that disrupts unsupported tokens and produces a stronger contrast signal. Our method is prompt-agnostic, model-agnostic, and can be seamlessly plugged into the existing VCD pipeline with little computation overhead, i.e., a single cacheable forward pass. Empirically, our method demonstrates consistent gains on two popular object hallucination benchmarks across two MLLMs. 

---
# DICE: Diffusion Large Language Models Excel at Generating CUDA Kernels 

**Authors**: Haolei Bai, Lingcheng Kong, Xueyi Chen, Jianmian Wang, Zhiqiang Tao, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11715)  

**Abstract**: Diffusion large language models (dLLMs) have emerged as a compelling alternative to autoregressive (AR) LLMs, owing to their capacity for parallel token generation. This paradigm is particularly well-suited for code generation, where holistic structural planning and non-sequential refinement are critical. Despite this potential, tailoring dLLMs for CUDA kernel generation remains challenging, obstructed not only by the high specialization but also by the severe lack of high-quality training data. To address these challenges, we construct CuKe, an augmented supervised fine-tuning dataset optimized for high-performance CUDA kernels. On top of it, we propose a bi-phase curated reinforcement learning (BiC-RL) framework consisting of a CUDA kernel infilling stage and an end-to-end CUDA kernel generation stage. Leveraging this training framework, we introduce DICE, a series of diffusion large language models designed for CUDA kernel generation, spanning three parameter scales, 1.7B, 4B, and 8B. Extensive experiments on KernelBench demonstrate that DICE significantly outperforms both autoregressive and diffusion LLMs of comparable scale, establishing a new state-of-the-art for CUDA kernel generation. 

---
# ThinkRouter: Efficient Reasoning via Routing Thinking between Latent and Discrete Spaces 

**Authors**: Xin Xu, Tong Yu, Xiang Chen, Haoliang Wang, Julian McAuley, Saayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2602.11683)  

**Abstract**: Recent work explores latent reasoning to improve reasoning efficiency by replacing explicit reasoning trajectories with continuous representations in a latent space, yet its effectiveness varies across settings. Analysis of model confidence dynamics under latent reasoning reveals that thinking trajectories ending in incorrect answers contain fewer low-confidence steps than those ending in correct answers. Meanwhile, we suggest that soft embeddings aggregated by multiple low-confidence thinking alternatives may introduce and propagate noise, leading to high confidence in unreliable reasoning trajectories. Motivated by these observations, ThinkRouter, an inference-time confidence-aware routing mechanism is proposed to avoid high confidence and noise for efficient reasoning. ThinkRouter routes thinking to the discrete token space when model confidence is low, and to the latent space otherwise. Extensive experiments on STEM reasoning and coding benchmarks across diverse large reasoning models demonstrate that ThinkRouter outperforms explicit CoT, random routing, and latent reasoning baselines in terms of accuracy, achieving an average improvement of 19.70 points in Pass@1, while reducing generation length by up to 15.55%. Further comprehensive analysis reveals that ThinkRouter can calibrate errors arising from explicit CoT and latent reasoning, and accelerates end-of-thinking token generation by globally lowering model confidence. 

---
# PhyNiKCE: A Neurosymbolic Agentic Framework for Autonomous Computational Fluid Dynamics 

**Authors**: E Fan, Lisong Shi, Zhengtong Li, Chih-yung Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11666)  

**Abstract**: The deployment of autonomous agents for Computational Fluid Dynamics (CFD), is critically limited by the probabilistic nature of Large Language Models (LLMs), which struggle to enforce the strict conservation laws and numerical stability required for physics-based simulations. Reliance on purely semantic Retrieval Augmented Generation (RAG) often leads to "context poisoning," where agents generate linguistically plausible but physically invalid configurations due to a fundamental Semantic-Physical Disconnect. To bridge this gap, this work introduces PhyNiKCE (Physical and Numerical Knowledgeable Context Engineering), a neurosymbolic agentic framework for trustworthy engineering. Unlike standard black-box agents, PhyNiKCE decouples neural planning from symbolic validation. It employs a Symbolic Knowledge Engine that treats simulation setup as a Constraint Satisfaction Problem, rigidly enforcing physical constraints via a Deterministic RAG Engine with specialized retrieval strategies for solvers, turbulence models, and boundary conditions. Validated through rigorous OpenFOAM experiments on practical, non-tutorial CFD tasks using Gemini-2.5-Pro/Flash, PhyNiKCE demonstrates a 96% relative improvement over state-of-the-art baselines. Furthermore, by replacing trial-and-error with knowledge-driven initialization, the framework reduced autonomous self-correction loops by 59% while simultaneously lowering LLM token consumption by 17%. These results demonstrate that decoupling neural generation from symbolic constraint enforcement significantly enhances robustness and efficiency. While validated on CFD, this architecture offers a scalable, auditable paradigm for Trustworthy Artificial Intelligence in broader industrial automation. 

---
# Analytical Search 

**Authors**: Yiteng Tu, Shuo Miao, Weihang Su, Yiqun Liu, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2602.11581)  

**Abstract**: Analytical information needs, such as trend analysis and causal impact assessment, are prevalent across various domains including law, finance, science, and much more. However, existing information retrieval paradigms, whether based on relevance-oriented document ranking or retrieval-augmented generation (RAG) with large language models (LLMs), often struggle to meet the end-to-end requirements of such tasks at the corpus scale. They either emphasize information finding rather than end-to-end problem solving, or simply treat everything as naive question answering, offering limited control over reasoning, evidence usage, and verifiability. As a result, they struggle to support analytical queries that have diverse utility concepts and high accountability requirements.
In this paper, we propose analytical search as a distinct and emerging search paradigm designed to fulfill these analytical information needs. Analytical search reframes search as an evidence-governed, process-oriented analytical workflow that explicitly models analytical intent, retrieves evidence for fusion, and produces verifiable conclusions through structured, multi-step inference. We position analytical search in contrast to existing paradigms, and present a unified system framework that integrates query understanding, recall-oriented retrieval, reasoning-aware fusion, and adaptive verification. We also discuss potential research directions for the construction of analytical search engines. In this way, we highlight the conceptual significance and practical importance of analytical search and call on efforts toward the next generation of search engines that support analytical information needs. 

---
# Stop Tracking Me! Proactive Defense Against Attribute Inference Attack in LLMs 

**Authors**: Dong Yan, Jian Liang, Ran He, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.11528)  

**Abstract**: Recent studies have shown that large language models (LLMs) can infer private user attributes (e.g., age, location, gender) from user-generated text shared online, enabling rapid and large-scale privacy breaches. Existing anonymization-based defenses are coarse-grained, lacking word-level precision in anonymizing privacy-leaking elements. Moreover, they are inherently limited as altering user text to hide sensitive cues still allows attribute inference to occur through models' reasoning capabilities. To address these limitations, we propose a unified defense framework that combines fine-grained anonymization (TRACE) with inference-preventing optimization (RPS). TRACE leverages attention mechanisms and inference chain generation to identify and anonymize privacy-leaking textual elements, while RPS employs a lightweight two-stage optimization strategy to induce model rejection behaviors, thereby preventing attribute inference. Evaluations across diverse LLMs show that TRACE-RPS reduces attribute inference accuracy from around 50\% to below 5\% on open-source models. In addition, our approach offers strong cross-model generalization, prompt-variation robustness, and utility-privacy tradeoffs. Our code is available at this https URL. 

---
# Adaptive Milestone Reward for GUI Agents 

**Authors**: Congmin Zheng, Xiaoyun Mo, Xinbei Ma, Qiqiang Lin, Yin Zhao, Jiachen Zhu, Xingyu Lou, Jun Wang, Zhaoxiang Wang, Weiwen Liu, Zhuosheng Zhang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11524)  

**Abstract**: Reinforcement Learning (RL) has emerged as a mainstream paradigm for training Mobile GUI Agents, yet it struggles with the temporal credit assignment problem inherent in long-horizon tasks. A primary challenge lies in the trade-off between reward fidelity and density: outcome reward offers high fidelity but suffers from signal sparsity, while process reward provides dense supervision but remains prone to bias and reward hacking. To resolve this conflict, we propose the Adaptive Milestone Reward (ADMIRE) mechanism. ADMIRE constructs a verifiable, adaptive reward system by anchoring trajectory to milestones, which are dynamically distilled from successful explorations. Crucially, ADMIRE integrates an asymmetric credit assignment strategy that denoises successful trajectories and scaffolds failed trajectories. Extensive experiments demonstrate that ADMIRE consistently yields over 10% absolute improvement in success rate across different base models on AndroidWorld. Moreover, the method exhibits robust generalizability, achieving strong performance across diverse RL algorithms and heterogeneous environments such as web navigation and embodied tasks. 

---
# Jailbreaking Leaves a Trace: Understanding and Detecting Jailbreak Attacks from Internal Representations of Large Language Models 

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis  

**Link**: [PDF](https://arxiv.org/pdf/2602.11495)  

**Abstract**: Jailbreaking large language models (LLMs) has emerged as a critical security challenge with the widespread deployment of conversational AI systems. Adversarial users exploit these models through carefully crafted prompts to elicit restricted or unsafe outputs, a phenomenon commonly referred to as Jailbreaking. Despite numerous proposed defense mechanisms, attackers continue to develop adaptive prompting strategies, and existing models remain vulnerable. This motivates approaches that examine the internal behavior of LLMs rather than relying solely on prompt-level defenses. In this work, we study jailbreaking from both security and interpretability perspectives by analyzing how internal representations differ between jailbreak and benign prompts. We conduct a systematic layer-wise analysis across multiple open-source models, including GPT-J, LLaMA, Mistral, and the state-space model Mamba, and identify consistent latent-space patterns associated with harmful inputs. We then propose a tensor-based latent representation framework that captures structure in hidden activations and enables lightweight jailbreak detection without model fine-tuning or auxiliary LLM-based detectors. We further demonstrate that the latent signals can be used to actively disrupt jailbreak execution at inference time. On an abliterated LLaMA-3.1-8B model, selectively bypassing high-susceptibility layers blocks 78% of jailbreak attempts while preserving benign behavior on 94% of benign prompts. This intervention operates entirely at inference time and introduces minimal overhead, providing a scalable foundation for achieving stronger coverage by incorporating additional attack distributions or more refined susceptibility thresholds. Our results provide evidence that jailbreak behavior is rooted in identifiable internal structures and suggest a complementary, architecture-agnostic direction for improving LLM security. 

---
# Sparse Semantic Dimension as a Generalization Certificate for LLMs 

**Authors**: Dibyanayan Bandyopadhyay, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2602.11388)  

**Abstract**: Standard statistical learning theory predicts that Large Language Models (LLMs) should overfit because their parameter counts vastly exceed the number of training tokens. Yet, in practice, they generalize robustly. We propose that the effective capacity controlling generalization lies in the geometry of the model's internal representations: while the parameter space is high-dimensional, the activation states lie on a low-dimensional, sparse manifold. To formalize this, we introduce the Sparse Semantic Dimension (SSD), a complexity measure derived from the active feature vocabulary of a Sparse Autoencoder (SAE) trained on the model's layers. Treating the LLM and SAE as frozen oracles, we utilize this framework to attribute the model's generalization capabilities to the sparsity of the dictionary rather than the total parameter count. Empirically, we validate this framework on GPT-2 Small and Gemma-2B, demonstrating that our bound provides non-vacuous certificates at realistic sample sizes. Crucially, we uncover a counter-intuitive "feature sharpness" scaling law: despite being an order of magnitude larger, Gemma-2B requires significantly fewer calibration samples to identify its active manifold compared to GPT-2, suggesting that larger models learn more compressible, distinct semantic structures. Finally, we show that this framework functions as a reliable safety monitor: out-of-distribution inputs trigger a measurable "feature explosion" (a sharp spike in active features), effectively signaling epistemic uncertainty through learned feature violation. Code is available at: this https URL. 

---
# ReplicatorBench: Benchmarking LLM Agents for Replicability in Social and Behavioral Sciences 

**Authors**: Bang Nguyen, Dominik Soós, Qian Ma, Rochana R. Obadage, Zack Ranjan, Sai Koneru, Timothy M. Errington, Shakhlo Nematova, Sarah Rajtmajer, Jian Wu, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.11354)  

**Abstract**: The literature has witnessed an emerging interest in AI agents for automated assessment of scientific papers. Existing benchmarks focus primarily on the computational aspect of this task, testing agents' ability to reproduce or replicate research outcomes when having access to the code and data. This setting, while foundational, (1) fails to capture the inconsistent availability of new data for replication as opposed to reproduction, and (2) lacks ground-truth diversity by focusing only on reproducible papers, thereby failing to evaluate an agent's ability to identify non-replicable research. Furthermore, most benchmarks only evaluate outcomes rather than the replication process. In response, we introduce ReplicatorBench, an end-to-end benchmark, including human-verified replicable and non-replicable research claims in social and behavioral sciences for evaluating AI agents in research replication across three stages: (1) extraction and retrieval of replication data; (2) design and execution of computational experiments; and (3) interpretation of results, allowing a test of AI agents' capability to mimic the activities of human replicators in real world. To set a baseline of AI agents' capability, we develop ReplicatorAgent, an agentic framework equipped with necessary tools, like web search and iterative interaction with sandboxed environments, to accomplish tasks in ReplicatorBench. We evaluate ReplicatorAgent across four underlying large language models (LLMs), as well as different design choices of programming language and levels of code access. Our findings reveal that while current LLM agents are capable of effectively designing and executing computational experiments, they struggle with retrieving resources, such as new data, necessary to replicate a claim. All code and data are publicly available at this https URL. 

---
# Dissecting Subjectivity and the "Ground Truth" Illusion in Data Annotation 

**Authors**: Sheza Munir, Benjamin Mah, Krisha Kalsi, Shivani Kapania, Julian Posada, Edith Law, Ding Wang, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2602.11318)  

**Abstract**: In machine learning, "ground truth" refers to the assumed correct labels used to train and evaluate models. However, the foundational "ground truth" paradigm rests on a positivistic fallacy that treats human disagreement as technical noise rather than a vital sociotechnical signal. This systematic literature review analyzes research published between 2020 and 2025 across seven premier venues: ACL, AIES, CHI, CSCW, EAAMO, FAccT, and NeurIPS, investigating the mechanisms in data annotation practices that facilitate this "consensus trap". Our identification phase captured 30,897 records, which were refined via a tiered keyword filtration schema to a high-recall corpus of 3,042 records for manual screening, resulting in a final included corpus of 346 papers for qualitative synthesis. Our reflexive thematic analysis reveals that systemic failures in positional legibility, combined with the recent architectural shift toward human-as-verifier models, specifically the reliance on model-mediated annotations, introduce deep-seated anchoring bias and effectively remove human voices from the loop. We further demonstrate how geographic hegemony imposes Western norms as universal benchmarks, often enforced by the performative alignment of precarious data workers who prioritize requester compliance over honest subjectivity to avoid economic penalties. Critiquing the "noisy sensor" fallacy, where statistical models misdiagnose cultural pluralism as random error, we argue for reclaiming disagreement as a high-fidelity signal essential for building culturally competent models. To address these systemic tensions, we propose a roadmap for pluralistic annotation infrastructures that shift the objective from discovering a singular "right" answer to mapping the diversity of human experience. 

---
# How Many Features Can a Language Model Store Under the Linear Representation Hypothesis? 

**Authors**: Nikhil Garg, Jon Kleinberg, Kenny Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.11246)  

**Abstract**: We introduce a mathematical framework for the linear representation hypothesis (LRH), which asserts that intermediate layers of language models store features linearly. We separate the hypothesis into two claims: linear representation (features are linearly embedded in neuron activations) and linear accessibility (features can be linearly decoded). We then ask: How many neurons $d$ suffice to both linearly represent and linearly access $m$ features? Classical results in compressed sensing imply that for $k$-sparse inputs, $d = O(k\log (m/k))$ suffices if we allow non-linear decoding algorithms (Candes and Tao, 2006; Candes et al., 2006; Donoho, 2006). However, the additional requirement of linear decoding takes the problem out of the classical compressed sensing, into linear compressed sensing.
Our main theoretical result establishes nearly-matching upper and lower bounds for linear compressed sensing. We prove that $d = \Omega_\epsilon(\frac{k^2}{\log k}\log (m/k))$ is required while $d = O_\epsilon(k^2\log m)$ suffices. The lower bound establishes a quantitative gap between classical and linear compressed setting, illustrating how linear accessibility is a meaningfully stronger hypothesis than linear representation alone. The upper bound confirms that neurons can store an exponential number of features under the LRH, giving theoretical evidence for the "superposition hypothesis" (Elhage et al., 2022).
The upper bound proof uses standard random constructions of matrices with approximately orthogonal columns. The lower bound proof uses rank bounds for near-identity matrices (Alon, 2003) together with Turán's theorem (bounding the number of edges in clique-free graphs). We also show how our results do and do not constrain the geometry of feature representations and extend our results to allow decoders with an activation function and bias. 

---
# Evaluating Memory Structure in LLM Agents 

**Authors**: Alina Shutova, Alexandra Olenina, Ivan Vinogradov, Anton Sinitsin  

**Link**: [PDF](https://arxiv.org/pdf/2602.11243)  

**Abstract**: Modern LLM-based agents and chat assistants rely on long-term memory frameworks to store reusable knowledge, recall user preferences, and augment reasoning. As researchers create more complex memory architectures, it becomes increasingly difficult to analyze their capabilities and guide future memory designs. Most long-term memory benchmarks focus on simple fact retention, multi-hop recall, and time-based changes. While undoubtedly important, these capabilities can often be achieved with simple retrieval-augmented LLMs and do not test complex memory hierarchies. To bridge this gap, we propose StructMemEval - a benchmark that tests the agent's ability to organize its long-term memory, not just factual recall. We gather a suite of tasks that humans solve by organizing their knowledge in a specific structure: transaction ledgers, to-do lists, trees and others. Our initial experiments show that simple retrieval-augmented LLMs struggle with these tasks, whereas memory agents can reliably solve them if prompted how to organize their memory. However, we also find that modern LLMs do not always recognize the memory structure when not prompted to do so. This highlights an important direction for future improvements in both LLM training and memory frameworks. 

---
# ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning 

**Authors**: Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, Feng Xiong, Xing Wei, Zhiheng Ma, Mu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11236)  

**Abstract**: Building general-purpose embodied agents across diverse hardware remains a central challenge in robotics, often framed as the ''one-brain, many-forms'' paradigm. Progress is hindered by fragmented data, inconsistent representations, and misaligned training objectives. We present ABot-M0, a framework that builds a systematic data curation pipeline while jointly optimizing model architecture and training strategies, enabling end-to-end transformation of heterogeneous raw data into unified, efficient representations. From six public datasets, we clean, standardize, and balance samples to construct UniACT-dataset, a large-scale dataset with over 6 million trajectories and 9,500 hours of data, covering diverse robot morphologies and task scenarios. Unified pre-training improves knowledge transfer and generalization across platforms and tasks, supporting general-purpose embodied intelligence. To improve action prediction efficiency and stability, we propose the Action Manifold Hypothesis: effective robot actions lie not in the full high-dimensional space but on a low-dimensional, smooth manifold governed by physical laws and task constraints. Based on this, we introduce Action Manifold Learning (AML), which uses a DiT backbone to predict clean, continuous action sequences directly. This shifts learning from denoising to projection onto feasible manifolds, improving decoding speed and policy stability. ABot-M0 supports modular perception via a dual-stream mechanism that integrates VLM semantics with geometric priors and multi-view inputs from plug-and-play 3D modules such as VGGT and Qwen-Image-Edit, enhancing spatial understanding without modifying the backbone and mitigating standard VLM limitations in 3D reasoning. Experiments show components operate independently with additive benefits. We will release all code and pipelines for reproducibility and future research. 

---
# Agent-Diff: Benchmarking LLM Agents on Enterprise API Tasks via Code Execution with State-Diff-Based Evaluation 

**Authors**: Hubert M. Pysklo, Artem Zhuravel, Patrick D. Watson  

**Link**: [PDF](https://arxiv.org/pdf/2602.11224)  

**Abstract**: We present Agent-Diff, a novel benchmarking framework for evaluating agentic Large Language Models (LLMs) on real-world tasks that execute code via external APIs. Agentic LLM performance varies due to differences in models, external tool access, prompt structures, and agentic frameworks. Benchmarks must make fundamental trade-offs between a sandboxed approach that controls for variation in software environments and more ecologically valid approaches employing real services. Agent-Diff attempts to capture the desirable features of both of these approaches by including access to the real API interfaces for software services while sandboxing the environment in which calls are made, processed, and evaluated. This approach relies on two key innovations. The first is a novel state-diff contract, which separates process from outcome - rather than fuzzy trace or parameter matching, we define task success as whether the expected change in environment state was achieved. The second is a novel sandbox that provides a standardized scripting layer that all models use to execute code against external APIs (Slack, Box, Linear, Google Calendar). Thus, we can evaluate different agentic LLMs against a standardized set of contracts using a unified sandbox while still evaluating their performance on real-world service interfaces. Using the Agent-Diff framework, we provide benchmarks for nine LLMs across 224 tasks utilizing enterprise software workflows. In addition, we evaluate the robustness of the framework with ablation experiments to assess the contribution of access to API documentation on benchmark performance. Code and data: this https URL. 

---
# Patch the Distribution Mismatch: RL Rewriting Agent for Stable Off-Policy SFT 

**Authors**: Jiacheng Wang, Ping Jian, Zhen Yang, Zirong Chen, Keren Liao, Zhongbin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.11220)  

**Abstract**: Large language models (LLMs) have made rapid progress, yet adapting them to downstream scenarios still commonly relies on supervised fine-tuning (SFT). When downstream data exhibit a substantial distribution shift from the model's prior training distribution, SFT can induce catastrophic forgetting. To narrow this gap, data rewriting has been proposed as a data-centric approach that rewrites downstream training data prior to SFT. However, existing methods typically sample rewrites from a prompt-induced conditional distribution, so the resulting targets are not necessarily aligned with the model's natural QA-style generation distribution. Moreover, reliance on fixed templates can lead to diversity collapse. To address these issues, we cast data rewriting as a policy learning problem and learn a rewriting policy that better matches the backbone's QA-style generation distribution while preserving diversity. Since distributional alignment, diversity and task consistency are automatically evaluable but difficult to optimize end-to-end with differentiable objectives, we leverage reinforcement learning to optimize the rewrite distribution under reward feedback and propose an RL-based data-rewriting agent. The agent jointly optimizes QA-style distributional alignment and diversity under a hard task-consistency gate, thereby constructing a higher-quality rewritten dataset for downstream SFT. Extensive experiments show that our method achieves downstream gains comparable to standard SFT while reducing forgetting on non-downstream benchmarks by 12.34% on average. Our code is available at this https URL . 

---
# Automated Optimization Modeling via a Localizable Error-Driven Perspective 

**Authors**: Weiting Liu, Han Wu, Yufei Kuang, Xiongwei Han, Tao Zhong, Jianfeng Feng, Wenlian Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.11164)  

**Abstract**: Automated optimization modeling via Large Language Models (LLMs) has emerged as a promising approach to assist complex human decision-making. While post-training has become a pivotal technique to enhance LLMs' capabilities in this domain, its effectiveness is severely constrained by the scarcity and underutilization of high-quality training data. However, through a detailed profiling of error patterns across various problem-response pairs drawn from post-training, we identify two fundamental limitations of existing automated optimization modeling approaches: (L1) the sparsity of error-specific problems and (L2) the sparse rewards associated with difficult problems. We demonstrate that these limitations can result in suboptimal performance in domain-specific post-training for LLMs. To tackle the above two limitations, we propose a novel error-driven learning framework -- namely, auto\textbf{m}ated opt\textbf{i}mization modeli\textbf{n}g via a localizable error-\textbf{d}riven perspective (MIND) -- that customizes the whole model training framework from data synthesis to post-training. MIND is based on our key observation of the unique localizable patterns in error propagation of optimization modelings, that is, modeling errors may remain localized to specific semantic segments and do not propagate throughout the entire solution. Thus, in contrast to holistic reasoning tasks such as mathematical proofs, MIND leverages the construction of a focused, high-density training corpus and proposes \textbf{D}ynamic Supervised \textbf{F}ine-Tuning \textbf{P}olicy \textbf{O}ptimization (DFPO) to tackle difficult problems through localized refinement. Experiments on six benchmarks demonstrate that MIND consistently outperforms all the state-of-the-art automated optimization modeling approaches. 

---
# Althea: Human-AI Collaboration for Fact-Checking and Critical Reasoning 

**Authors**: Svetlana Churina, Kokil Jaidka, Anab Maulana Barik, Harshit Aneja, Cai Yang, Wynne Hsu, Mong Li Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.11161)  

**Abstract**: The web's information ecosystem demands fact-checking systems that are both scalable and epistemically trustworthy. Automated approaches offer efficiency but often lack transparency, while human verification remains slow and inconsistent. We introduce Althea, a retrieval-augmented system that integrates question generation, evidence retrieval, and structured reasoning to support user-driven evaluation of online claims. On the AVeriTeC benchmark, Althea achieves a Macro-F1 of 0.44, outperforming standard verification pipelines and improving discrimination between supported and refuted claims. We further evaluate Althea through a controlled user study and a longitudinal survey experiment (N = 642), comparing three interaction modes that vary in the degree of scaffolding: an Exploratory mode with guided reasoning, a Summary mode providing synthesized verdicts, and a Self-search mode that offers procedural guidance without algorithmic intervention. Results show that guided interaction produces the strongest immediate gains in accuracy and confidence, while self-directed search yields the most persistent improvements over time. This pattern suggests that performance gains are not driven solely by effort or exposure, but by how cognitive work is structured and internalized. 

---
