# Semantic Soft Bootstrapping: Long Context Reasoning in LLMs without Reinforcement Learning 

**Authors**: Purbesh Mitra, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2512.05105)  

**Abstract**: Long context reasoning in large language models (LLMs) has demonstrated enhancement of their cognitive capabilities via chain-of-thought (CoT) inference. Training such models is usually done via reinforcement learning with verifiable rewards (RLVR) in reasoning based problems, like math and programming. However, RLVR is limited by several bottlenecks, such as, lack of dense reward, and inadequate sample efficiency. As a result, it requires significant compute resources in post-training phase. To overcome these limitations, in this work, we propose \textbf{Semantic Soft Bootstrapping (SSB)}, a self-distillation technique, in which the same base language model plays the role of both teacher and student, but receives different semantic contexts about the correctness of its outcome at training time. The model is first prompted with a math problem and several rollouts are generated. From them, the correct and most common incorrect response are filtered, and then provided to the model in context to produce a more robust, step-by-step explanation with a verified final answer. This pipeline automatically curates a paired teacher-student training set from raw problem-answer data, without any human intervention. This generation process also produces a sequence of logits, which is what the student model tries to match in the training phase just from the bare question alone. In our experiment, Qwen2.5-3B-Instruct on GSM8K dataset via parameter-efficient fine-tuning. We then tested its accuracy on MATH500, and AIME2024 benchmarks. Our experiments show a jump of 10.6%, and 10% improvements in accuracy, respectively, over group relative policy optimization (GRPO), which is a commonly used RLVR algorithm. Our code is available at this https URL, and the model, curated dataset is available at this https URL. 

---
# Structured Document Translation via Format Reinforcement Learning 

**Authors**: Haiyue Song, Johannes Eschbach-Dymanus, Hour Kaing, Sumire Honda, Hideki Tanaka, Bianka Buschbeck, Masao Utiyama  

**Link**: [PDF](https://arxiv.org/pdf/2512.05100)  

**Abstract**: Recent works on structured text translation remain limited to the sentence level, as they struggle to effectively handle the complex document-level XML or HTML structures. To address this, we propose \textbf{Format Reinforcement Learning (FormatRL)}, which employs Group Relative Policy Optimization on top of a supervised fine-tuning model to directly optimize novel structure-aware rewards: 1) TreeSim, which measures structural similarity between predicted and reference XML trees and 2) Node-chrF, which measures translation quality at the level of XML nodes. Additionally, we apply StrucAUC, a fine-grained metric distinguishing between minor errors and major structural failures. Experiments on the SAP software-documentation benchmark demonstrate improvements across six metrics and an analysis further shows how different reward functions contribute to improvements in both structural and translation quality. 

---
# Arbitrage: Efficient Reasoning via Advantage-Aware Speculation 

**Authors**: Monishwaran Maheswaran, Rishabh Tiwari, Yuezhou Hu, Kerem Dilmen, Coleman Hooper, Haocheng Xi, Nicholas Lee, Mehrdad Farajtabar, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2512.05033)  

**Abstract**: Modern Large Language Models achieve impressive reasoning capabilities with long Chain of Thoughts, but they incur substantial computational cost during inference, and this motivates techniques to improve the performance-cost ratio. Among these techniques, Speculative Decoding accelerates inference by employing a fast but inaccurate draft model to autoregressively propose tokens, which are then verified in parallel by a more capable target model. However, due to unnecessary rejections caused by token mismatches in semantically equivalent steps, traditional token-level Speculative Decoding struggles in reasoning tasks. Although recent works have shifted to step-level semantic verification, which improve efficiency by accepting or rejecting entire reasoning steps, existing step-level methods still regenerate many rejected steps with little improvement, wasting valuable target compute. To address this challenge, we propose Arbitrage, a novel step-level speculative generation framework that routes generation dynamically based on the relative advantage between draft and target models. Instead of applying a fixed acceptance threshold, Arbitrage uses a lightweight router trained to predict when the target model is likely to produce a meaningfully better step. This routing approximates an ideal Arbitrage Oracle that always chooses the higher-quality step, achieving near-optimal efficiency-accuracy trade-offs. Across multiple mathematical reasoning benchmarks, Arbitrage consistently surpasses prior step-level Speculative Decoding baselines, reducing inference latency by up to $\sim2\times$ at matched accuracy. 

---
# Factuality and Transparency Are All RAG Needs! Self-Explaining Contrastive Evidence Re-ranking 

**Authors**: Francielle Vargas, Daniel Pedronette  

**Link**: [PDF](https://arxiv.org/pdf/2512.05012)  

**Abstract**: This extended abstract introduces Self-Explaining Contrastive Evidence Re-Ranking (CER), a novel method that restructures retrieval around factual evidence by fine-tuning embeddings with contrastive learning and generating token-level attribution rationales for each retrieved passage. Hard negatives are automatically selected using a subjectivity-based criterion, forcing the model to pull factual rationales closer while pushing subjective or misleading explanations apart. As a result, the method creates an embedding space explicitly aligned with evidential reasoning. We evaluated our method on clinical trial reports, and initial experimental results show that CER improves retrieval accuracy, mitigates the potential for hallucinations in RAG systems, and provides transparent, evidence-based retrieval that enhances reliability, especially in safety-critical domains. 

---
# Nex-N1: Agentic Models Trained via a Unified Ecosystem for Large-Scale Environment Construction 

**Authors**: Nex-AGI Team, Yuxuan Cai, Lu Chen, Qiaoling Chen, Yuyang Ding, Liwen Fan, Wenjie Fu, Yufei Gao, Honglin Guo, Pinxue Guo, Zhenhua Han, Zhengfu He, Hanglei Hu, Kai Hu, Shengjia Hua, Tianyu Huai, Baodai Huang, Li Ji, Zhen Jiang, Zhikai Lei, Bufan Li, Jiahang Lin, Lizhi Lin, Jinxiu Liu, Shichun Liu, Ziming Liu, Yuchen Ni, Pengfang Qian, Yujiong Shen, Qingyun Shi, Wentao Shu, Peng Sun, Yiran Suo, Tian Tang, Boyu Tian, Guoteng Wang, Junzhe Wang, Peixin Wang, Zhiheng Xi, Hang Yan, Jie Yang, Zhixiong Yang, Tianchu Yao, Guangze Ye, Qianxi Yu, Shuo Zhang, Xinyue Zhang, Yiqi Zhang, Jiarong Zhao, Miao Zheng, Rui Zheng, Enyu Zhou, Jiazheng Zhou, Maosen Zhou, Yuhao Zhou, Tao Gui, Yining Zheng, Xinchi Chen, Jie Zhou, Siyuan Feng, Qin Chen, Liang He, Qi Zhang, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2512.04987)  

**Abstract**: The evolution of Large Language Models (LLMs) from passive responders to autonomous agents necessitates a fundamental shift in learning paradigms -- from static imitation to incentive-driven decision making. However, this transition is significantly impeded by the lack of scalable infrastructure capable of constructing high-quality interaction signals for effective policy learning. To address this, we introduce a comprehensive method designed to systematically scale the diversity and complexity of interactive environments. Our method realizes this scaling by addressing three orthogonal dimensions: (1) Complexity: NexAU, a flexible agent framework that supports building complex agent hierarchies via simple configurations; (2) Diversity: NexA4A automatically generates diverse agent hierarchies from natural language to cover infinite domains; and (3) Fidelity: NexGAP bridges the simulation-reality gap by integrating dynamic real-world environment for grounded trajectories synthesis. We train Nex-N1 upon the diverse and complex interactive environments established by our infrastructure. Empirical results on benchmarks such as SWE-bench and tau2 demonstrate that Nex-N1 consistently outperforms SOTA open-source models and achieves competitive performance against frontier proprietary models on complex agentic tasks. We open-source the Nex ecosystem and model weights to facilitate further research. 

---
# LLMs Know More Than Words: A Genre Study with Syntax, Metaphor & Phonetics 

**Authors**: Weiye Shi, Zhaowei Zhang, Shaoheng Yan, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.04957)  

**Abstract**: Large language models (LLMs) demonstrate remarkable potential across diverse language related tasks, yet whether they capture deeper linguistic properties, such as syntactic structure, phonetic cues, and metrical patterns from raw text remains unclear. To analysis whether LLMs can learn these features effectively and apply them to important nature language related tasks, we introduce a novel multilingual genre classification dataset derived from Project Gutenberg, a large-scale digital library offering free access to thousands of public domain literary works, comprising thousands of sentences per binary task (poetry vs. novel;drama vs. poetry;drama vs. novel) in six languages (English, French, German, Italian, Spanish, and Portuguese). We augment each with three explicit linguistic feature sets (syntactic tree structures, metaphor counts, and phonetic metrics) to evaluate their impact on classification performance. Experiments demonstrate that although LLM classifiers can learn latent linguistic structures either from raw text or from explicitly provided features, different features contribute unevenly across tasks, which underscores the importance of incorporating more complex linguistic signals during model training. 

---
# SEAL: Self-Evolving Agentic Learning for Conversational Question Answering over Knowledge Graphs 

**Authors**: Hao Wang, Jialun Zhong, Changcheng Wang, Zhujun Nie, Zheng Li, Shunyu Yao, Yanzeng Li, Xinchi Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.04868)  

**Abstract**: Knowledge-based conversational question answering (KBCQA) confronts persistent challenges in resolving coreference, modeling contextual dependencies, and executing complex logical reasoning. Existing approaches, whether end-to-end semantic parsing or stepwise agent-based reasoning, often suffer from structural inaccuracies and prohibitive computational costs, particularly when processing intricate queries over large knowledge graphs. To address these limitations, we introduce SEAL, a novel two-stage semantic parsing framework grounded in self-evolving agentic learning. In the first stage, a large language model (LLM) extracts a minimal S-expression core that captures the essential semantics of the input query. This core is then refined by an agentic calibration module, which corrects syntactic inconsistencies and aligns entities and relations precisely with the underlying knowledge graph. The second stage employs template-based completion, guided by question-type prediction and placeholder instantiation, to construct a fully executable S-expression. This decomposition not only simplifies logical form generation but also significantly enhances structural fidelity and linking efficiency. Crucially, SEAL incorporates a self-evolving mechanism that integrates local and global memory with a reflection module, enabling continuous adaptation from dialog history and execution feedback without explicit retraining. Extensive experiments on the SPICE benchmark demonstrate that SEAL achieves state-of-the-art performance, especially in multi-hop reasoning, comparison, and aggregation tasks. The results validate notable gains in both structural accuracy and computational efficiency, underscoring the framework's capacity for robust and scalable conversational reasoning. 

---
# Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates 

**Authors**: Atsuki Yamaguchi, Terufumi Morishita, Aline Villavicencio, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2512.04844)  

**Abstract**: Expanding the linguistic diversity of instruct large language models (LLMs) is crucial for global accessibility but is often hindered by the reliance on costly specialized target language labeled data and catastrophic forgetting during adaptation. We tackle this challenge under a realistic, low-resource constraint: adapting instruct LLMs using only unlabeled target language data. We introduce Source-Shielded Updates (SSU), a selective parameter update strategy that proactively preserves source knowledge. Using a small set of source data and a parameter importance scoring method, SSU identifies parameters critical to maintaining source abilities. It then applies a column-wise freezing strategy to protect these parameters before adaptation. Experiments across five typologically diverse languages and 7B and 13B models demonstrate that SSU successfully mitigates catastrophic forgetting. It reduces performance degradation on monolingual source tasks to just 3.4% (7B) and 2.8% (13B) on average, a stark contrast to the 20.3% and 22.3% from full fine-tuning. SSU also achieves target-language performance highly competitive with full fine-tuning, outperforming it on all benchmarks for 7B models and the majority for 13B models. 

---
# DAMASHA: Detecting AI in Mixed Adversarial Texts via Segmentation with Human-interpretable Attribution 

**Authors**: L. D. M. S. Sai Teja, N. Siva Gopala Krishna, Ufaq Khan, Muhammad Haris Khan, Partha Pakray, Atul Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2512.04838)  

**Abstract**: In the age of advanced large language models (LLMs), the boundaries between human and AI-generated text are becoming increasingly blurred. We address the challenge of segmenting mixed-authorship text, that is identifying transition points in text where authorship shifts from human to AI or vice-versa, a problem with critical implications for authenticity, trust, and human oversight. We introduce a novel framework, called Info-Mask for mixed authorship detection that integrates stylometric cues, perplexity-driven signals, and structured boundary modeling to accurately segment collaborative human-AI content. To evaluate the robustness of our system against adversarial perturbations, we construct and release an adversarial benchmark dataset Mixed-text Adversarial setting for Segmentation (MAS), designed to probe the limits of existing detectors. Beyond segmentation accuracy, we introduce Human-Interpretable Attribution (HIA overlays that highlight how stylometric features inform boundary predictions, and we conduct a small-scale human study assessing their usefulness. Across multiple architectures, Info-Mask significantly improves span-level robustness under adversarial conditions, establishing new baselines while revealing remaining challenges. Our findings highlight both the promise and limitations of adversarially robust, interpretable mixed-authorship detection, with implications for trust and oversight in human-AI co-authorship. 

---
# DaLA: Danish Linguistic Acceptability Evaluation Guided by Real World Errors 

**Authors**: Gianluca Barmina, Nathalie Carmen Hau Norman, Peter Schneider-Kamp, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2512.04799)  

**Abstract**: We present an enhanced benchmark for evaluating linguistic acceptability in Danish. We first analyze the most common errors found in written Danish. Based on this analysis, we introduce a set of fourteen corruption functions that generate incorrect sentences by systematically introducing errors into existing correct Danish sentences. To ensure the accuracy of these corruptions, we assess their validity using both manual and automatic methods. The results are then used as a benchmark for evaluating Large Language Models on a linguistic acceptability judgement task. Our findings demonstrate that this extension is both broader and more comprehensive than the current state of the art. By incorporating a greater variety of corruption types, our benchmark provides a more rigorous assessment of linguistic acceptability, increasing task difficulty, as evidenced by the lower performance of LLMs on our benchmark compared to existing ones. Our results also suggest that our benchmark has a higher discriminatory power which allows to better distinguish well-performing models from low-performing ones. 

---
# AdiBhashaa: A Community-Curated Benchmark for Machine Translation into Indian Tribal Languages 

**Authors**: Pooja Singh, Sandeep Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.04765)  

**Abstract**: Large language models and multilingual machine translation (MT) systems increasingly drive access to information, yet many languages of the tribal communities remain effectively invisible in these technologies. This invisibility exacerbates existing structural inequities in education, governance, and digital participation. We present AdiBhashaa, a community-driven initiative that constructs the first open parallel corpora and baseline MT systems for four major Indian tribal languages-Bhili, Mundari, Gondi, and Santali. This work combines participatory data creation with native speakers, human-in-the-loop validation, and systematic evaluation of both encoder-decoder MT models and large language models. In addition to reporting technical findings, we articulate how AdiBhashaa illustrates a possible model for more equitable AI research: it centers local expertise, builds capacity among early-career researchers from marginalized communities, and foregrounds human validation in the development of language technologies. 

---
# Challenging the Abilities of Large Language Models in Italian: a Community Initiative 

**Authors**: Malvina Nissim, Danilo Croce, Viviana Patti, Pierpaolo Basile, Giuseppe Attanasio, Elio Musacchio, Matteo Rinaldi, Federico Borazio, Maria Francis, Jacopo Gili, Daniel Scalena, Begoña Altuna, Ekhi Azurmendi, Valerio Basile, Luisa Bentivogli, Arianna Bisazza, Marianna Bolognesi, Dominique Brunato, Tommaso Caselli, Silvia Casola, Maria Cassese, Mauro Cettolo, Claudia Collacciani, Leonardo De Cosmo, Maria Pia Di Buono, Andrea Esuli, Julen Etxaniz, Chiara Ferrando, Alessia Fidelangeli, Simona Frenda, Achille Fusco, Marco Gaido, Andrea Galassi, Federico Galli, Luca Giordano, Mattia Goffetti, Itziar Gonzalez-Dios, Lorenzo Gregori, Giulia Grundler, Sandro Iannaccone, Chunyang Jiang, Moreno La Quatra, Francesca Lagioia, Soda Marem Lo, Marco Madeddu, Bernardo Magnini, Raffaele Manna, Fabio Mercorio, Paola Merlo, Arianna Muti, Vivi Nastase, Matteo Negri, Dario Onorati, Elena Palmieri, Sara Papi, Lucia Passaro, Giulia Pensa, Andrea Piergentili, Daniele Potertì, Giovanni Puccetti, Federico Ranaldi, Leonardo Ranaldi, Andrea Amelio Ravelli, Martina Rosola, Elena Sofia Ruzzetti, Giuseppe Samo, Andrea Santilli, Piera Santin, Gabriele Sarti, Giovanni Sartor, Beatrice Savoldi, Antonio Serino, Andrea Seveso, Lucia Siciliani, Paolo Torroni, Rossella Varvara, Andrea Zaninello, Asya Zanollo, Fabio Massimo Zanzotto, Kamyar Zeinalipour, Andrea Zugarini  

**Link**: [PDF](https://arxiv.org/pdf/2512.04759)  

**Abstract**: The rapid progress of Large Language Models (LLMs) has transformed natural language processing and broadened its impact across research and society. Yet, systematic evaluation of these models, especially for languages beyond English, remains limited. "Challenging the Abilities of LAnguage Models in ITAlian" (CALAMITA) is a large-scale collaborative benchmarking initiative for Italian, coordinated under the Italian Association for Computational Linguistics. Unlike existing efforts that focus on leaderboards, CALAMITA foregrounds methodology: it federates more than 80 contributors from academia, industry, and the public sector to design, document, and evaluate a diverse collection of tasks, covering linguistic competence, commonsense reasoning, factual consistency, fairness, summarization, translation, and code generation. Through this process, we not only assembled a benchmark of over 20 tasks and almost 100 subtasks, but also established a centralized evaluation pipeline that supports heterogeneous datasets and metrics. We report results for four open-weight LLMs, highlighting systematic strengths and weaknesses across abilities, as well as challenges in task-specific evaluation. Beyond quantitative results, CALAMITA exposes methodological lessons: the necessity of fine-grained, task-representative metrics, the importance of harmonized pipelines, and the benefits and limitations of broad community engagement. CALAMITA is conceived as a rolling benchmark, enabling continuous integration of new tasks and models. This makes it both a resource -- the most comprehensive and diverse benchmark for Italian to date -- and a framework for sustainable, community-driven evaluation. We argue that this combination offers a blueprint for other languages and communities seeking inclusive and rigorous LLM evaluation practices. 

---
# EtCon: Edit-then-Consolidate for Reliable Knowledge Editing 

**Authors**: Ruilin Li, Yibin Wang, Wenhong Zhu, Chenglin Li, Jinghao Zhang, Chenliang Li, Junchi Yan, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.04753)  

**Abstract**: Knowledge editing aims to update specific facts in large language models (LLMs) without full retraining. Prior efforts sought to tune the knowledge layers of LLMs, proving effective for making selective edits. However, a significant gap exists between their performance in controlled, teacher-forcing evaluations and their real-world effectiveness in lifelong learning scenarios, which greatly limits their practical applicability. This work's empirical analysis reveals two recurring issues associated with this gap: (1) Most traditional methods lead the edited model to overfit to the new fact, thereby degrading pre-trained capabilities; (2) There is a critical absence of a knowledge consolidation stage, leaving new facts insufficiently integrated into LLMs' inference-time behavior under autoregressive generation, thereby leading to a mismatch between parametric knowledge and actual generation behavior. To this end, we propose Edit-then-Consolidate, a novel knowledge editing paradigm that aims to bridge the gap between theoretical knowledge editing methods and their real-world applicability. Specifically, (1) our framework mitigates overfitting via Targeted Proximal Supervised Fine-Tuning (TPSFT) that localizes the edit via a trust-region objective to limit policy drift; (2) Then, a consolidation stage using Group Relative Policy Optimization (GRPO) aligns the edited knowledge with CoT-based inference policy by optimizing trajectory-level behavior under comprehensive reward signals. Extensive experiments demonstrate our framework consistently improves editing reliability and generalization under real-world evaluations, while better preserving locality and pre-trained capabilities. 

---
# Model Whisper: Steering Vectors Unlock Large Language Models' Potential in Test-time 

**Authors**: Xinyue Kang, Diwei Shi, Li Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.04748)  

**Abstract**: It is a critical challenge to efficiently unlock the powerful reasoning potential of Large Language Models (LLMs) for specific tasks or new distributions. Existing test-time adaptation methods often require tuning model parameters, which is not only computationally expensive but also risks degrading the model's pre-existing this http URL address this, we introduce a lightweight component, Test-Time Steering Vectors (TTSV), which is prepended to the input while keeping the LLM's parameters entirely frozen. By optimizing the TTSV on test data to minimize the model's output entropy, we steer the model towards an internal state of higher confidence, activating its inherent abilities most relevant to the current task. TTSV is both lightweight and highly efficient to optimize, making it a true plug-and-play enhancement. Extensive experiments validate our approach's effectiveness on both base models and reasoning-enhanced models. For instance, on the MATH500 task, TTSV achieves a 45.88% relative performance gain on the Qwen2.5-Math-7B model and a 16.22% relative gain on the Qwen3-4B model. Furthermore, our approach exhibits robust generalization, with its steering vectors proving highly transferable across diverse tasks. 

---
# SignRoundV2: Closing the Performance Gap in Extremely Low-Bit Post-Training Quantization for LLMs 

**Authors**: Wenhua Cheng, Weiwei Zhang, Heng Guo, Haihao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.04746)  

**Abstract**: Extreme low-bit quantization is critical for efficiently deploying Large Language Models (LLMs), yet it often leads to severe performance degradation at 2-bits and even 4-bits (e.g., MXFP4). We present SignRoundV2, a post-training quantization framework that is highly effective even without mixed-precision. SignRoundV2 introduces (1) a fast sensitivity metric that combines gradient information with quantization-induced deviations to guide layer-wise bit allocation, and (2) a lightweight pre-tuning search for quantization scales to improve extremely low-bit quantization. These components allow SignRoundV2 to close the gap with full-precision models. Extensive experiments indicate that our method sustains competitive accuracy for LLMs, achieving production-grade performance with about 1 percent variance at 4-5 bits and strong results even at 2 bits. The implementation is available at this https URL. 

---
# OsmT: Bridging OpenStreetMap Queries and Natural Language with Open-source Tag-aware Language Models 

**Authors**: Zhuoyue Wan, Wentao Hu, Chen Jason Zhang, Yuanfeng Song, Shuaimin Li, Ruiqiang Xiao, Xiao-Yong Wei, Raymond Chi-Wing Wong  

**Link**: [PDF](https://arxiv.org/pdf/2512.04738)  

**Abstract**: Bridging natural language and structured query languages is a long-standing challenge in the database community. While recent advances in language models have shown promise in this direction, existing solutions often rely on large-scale closed-source models that suffer from high inference costs, limited transparency, and lack of adaptability for lightweight deployment. In this paper, we present OsmT, an open-source tag-aware language model specifically designed to bridge natural language and Overpass Query Language (OverpassQL), a structured query language for accessing large-scale OpenStreetMap (OSM) data. To enhance the accuracy and structural validity of generated queries, we introduce a Tag Retrieval Augmentation (TRA) mechanism that incorporates contextually relevant tag knowledge into the generation process. This mechanism is designed to capture the hierarchical and relational dependencies present in the OSM database, addressing the topological complexity inherent in geospatial query formulation. In addition, we define a reverse task, OverpassQL-to-Text, which translates structured queries into natural language explanations to support query interpretation and improve user accessibility. We evaluate OsmT on a public benchmark against strong baselines and observe consistent improvements in both query generation and interpretation. Despite using significantly fewer parameters, our model achieves competitive accuracy, demonstrating the effectiveness of open-source pre-trained language models in bridging natural language and structured query languages within schema-rich geospatial environments. 

---
# Geschlechtsübergreifende Maskulina im Sprachgebrauch Eine korpusbasierte Untersuchung zu lexemspezifischen Unterschieden 

**Authors**: Carolin Mueller-Spitzer, Samira Ochs, Jan Oliver Ruediger, Sascha Wolfer  

**Link**: [PDF](https://arxiv.org/pdf/2512.04683)  

**Abstract**: This study examines the distribution and linguistic characteristics of generic masculines (GM) in contemporary German press texts. The use of masculine personal nouns to refer to mixed-gender groups or unspecified individuals has been widely debated in academia and the public, with con-flicting perspectives on its gender-neutrality. While psycholinguistic studies suggest that GM is more readily associated with male referents, corpus-based analyses of its actual use remain scarce. We investigate GM in a large corpus of press texts, focusing on lexeme-specific differences across dif-ferent types of personal nouns. We conducted manual annotations of the whole inflectional para-digm of 21 personal nouns, resulting in 6,195 annotated tokens. Our findings reveal considerable differences between lexical items, especially between passive role nouns and prestige-related per-sonal nouns. On a grammatical level, we find that GM occurs predominantly in the plural and in indefinite noun phrases. Furthermore, our data shows that GM is not primarily used to denote entire classes of people, as has been previously claimed. By providing an empirical insight into the use of GM in authentic written language, we contribute to a more nuanced understanding of its forms and manifestations. These findings provide a solid basis for aligning linguistic stimuli in psy-cholinguistic studies more closely with real-world language use. 

---
# LexGenius: An Expert-Level Benchmark for Large Language Models in Legal General Intelligence 

**Authors**: Wenjin Liu, Haoran Luo, Xin Feng, Xiang Ji, Lijuan Zhou, Rui Mao, Jiapu Wang, Shirui Pan, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2512.04578)  

**Abstract**: Legal general intelligence (GI) refers to artificial intelligence (AI) that encompasses legal understanding, reasoning, and decision-making, simulating the expertise of legal experts across domains. However, existing benchmarks are result-oriented and fail to systematically evaluate the legal intelligence of large language models (LLMs), hindering the development of legal GI. To address this, we propose LexGenius, an expert-level Chinese legal benchmark for evaluating legal GI in LLMs. It follows a Dimension-Task-Ability framework, covering seven dimensions, eleven tasks, and twenty abilities. We use the recent legal cases and exam questions to create multiple-choice questions with a combination of manual and LLM reviews to reduce data leakage risks, ensuring accuracy and reliability through multiple rounds of checks. We evaluate 12 state-of-the-art LLMs using LexGenius and conduct an in-depth analysis. We find significant disparities across legal intelligence abilities for LLMs, with even the best LLMs lagging behind human legal professionals. We believe LexGenius can assess the legal intelligence abilities of LLMs and enhance legal GI development. Our project is available at this https URL. 

---
# ADAPT: Learning Task Mixtures for Budget-Constrained Instruction Tuning 

**Authors**: Pritam Kadasi, Abhishek Upperwal, Mayank SIngh  

**Link**: [PDF](https://arxiv.org/pdf/2512.04555)  

**Abstract**: We propose ADAPT, a meta-learning algorithm that \emph{learns} task sampling proportions under an explicit token budget for multi-task instruction tuning. Instead of fixing task weights by hand, \adapt{} maintains a continuous distribution over tasks and updates it via meta-gradients of a smooth worst-case validation objective, inducing an adaptive curriculum that allocates more tokens to useful tasks while avoiding collapse. We instantiate ADAPT on three $\sim$1B-parameter open-weight LLMs (Gemma-3-1B, LLaMA-3.2-1B, Qwen-0.6B), training on 20 Natural Instructions task types under budgets of $1\%$, $5\%$, and $10\%$ of the available supervised tokens, and compare against strong supervised fine-tuning baselines with uniform and size-proportional mixing. We conduct evaluations on 11 out-of-domain benchmarks spanning reasoning, reading comprehension, code generation, and instruction following, we find that ADAPT matches or slightly improves average downstream performance relative to the best static mixture, while using fewer effective training tokens and reallocating budget toward harder, benchmark-aligned tasks. 

---
# AdmTree: Compressing Lengthy Context with Adaptive Semantic Trees 

**Authors**: Yangning Li, Shaoshen Chen, Yinghui Li, Yankai Chen, Hai-Tao Zheng, Hui Wang, Wenhao Jiang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.04550)  

**Abstract**: The quadratic complexity of self-attention constrains Large Language Models (LLMs) in processing long contexts, a capability essential for many advanced applications. Context compression aims to alleviate this computational bottleneck while retaining critical semantic information. However, existing approaches often fall short: explicit methods may compromise local detail, whereas implicit methods can suffer from positional biases, information degradation, or an inability to capture long-range semantic dependencies. We propose AdmTree, a novel framework for adaptive, hierarchical context compression with a central focus on preserving high semantic fidelity while maintaining efficiency. AdmTree dynamically segments input based on information density, utilizing gist tokens to summarize variable-length segments as the leaves of a semantic binary tree. This structure, together with a lightweight aggregation mechanism and a frozen backbone LLM (thereby minimizing new trainable parameters), enables efficient hierarchical abstraction of the context. By preserving fine-grained details alongside global semantic coherence, mitigating positional bias, and dynamically adapting to content, AdmTree robustly retains the semantic information of long contexts. 

---
# EvoEdit: Lifelong Free-Text Knowledge Editing through Latent Perturbation Augmentation and Knowledge-driven Parameter Fusion 

**Authors**: Pengfei Cao, Zeao Ji, Daojian Zeng, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.04545)  

**Abstract**: Adjusting the outdated knowledge of large language models (LLMs) after deployment remains a major challenge. This difficulty has spurred the development of knowledge editing, which seeks to accurately and efficiently modify a model's internal (parametric) knowledge without retraining it from scratch. However, existing methods suffer from two limitations. First, they depend on structured triplets that are misaligned with the free-text nature of LLM pretraining and fail to capture the nuanced relationships among facts. Second, they typically support one-time knowledge updates, with relatively limited research on the problem of sequential or lifelong editing. To address these gaps, we propose a new task, Lifelong Free-text Knowledge Editing (LF-Edit), which enables models to incorporate updates expressed in natural language and supports continual editing over time. Despite its promise, LF-Edit faces the dual challenge of integrating new knowledge while mitigating the forgetting of prior information. To foster research on this new task, we construct a large-scale benchmark, Multi-Rank Lifelong Free-text Editing Benchmark (MRLF-Bench), containing 16,835 free-text edit requests. We further design a cognitively inspired multi-rank evaluation framework encompassing four levels: memorization, understanding, constrained comprehension, and reasoning. To tackle the challenges inherent in LF-Edit, we introduce a novel approach named EvoEdit that enhances knowledge injection through Latent Perturbation Augmentation and preserves prior information via Knowledge-driven Parameter Fusion. Experimental results demonstrate that EvoEdit substantially outperforms existing knowledge editing methods on the proposed LF-Edit task. 

---
# UW-BioNLP at ChemoTimelines 2025: Thinking, Fine-Tuning, and Dictionary-Enhanced LLM Systems for Chemotherapy Timeline Extraction 

**Authors**: Tianmai M. Zhang, Zhaoyi Sun, Sihang Zeng, Chenxi Li, Neil F. Abernethy, Barbara D. Lam, Fei Xia, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2512.04518)  

**Abstract**: The ChemoTimelines shared task benchmarks methods for constructing timelines of systemic anticancer treatment from electronic health records of cancer patients. This paper describes our methods, results, and findings for subtask 2 -- generating patient chemotherapy timelines from raw clinical notes. We evaluated strategies involving chain-of-thought thinking, supervised fine-tuning, direct preference optimization, and dictionary-based lookup to improve timeline extraction. All of our approaches followed a two-step workflow, wherein an LLM first extracted chemotherapy events from individual clinical notes, and then an algorithm normalized and aggregated events into patient-level timelines. Each specific method differed in how the associated LLM was utilized and trained. Multiple approaches yielded competitive performances on the test set leaderboard, with fine-tuned Qwen3-14B achieving the best official score of 0.678. Our results and analyses could provide useful insights for future attempts on this task as well as the design of similar tasks. 

---
# MSME: A Multi-Stage Multi-Expert Framework for Zero-Shot Stance Detection 

**Authors**: Yuanshuo Zhang, Aohua Li, Bo Chen, Jingbo Sun, Xiaobing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.04492)  

**Abstract**: LLM-based approaches have recently achieved impressive results in zero-shot stance detection. However, they still struggle in complex real-world scenarios, where stance understanding requires dynamic background knowledge, target definitions involve compound entities or events that must be explicitly linked to stance labels, and rhetorical devices such as irony often obscure the author's actual intent. To address these challenges, we propose MSME, a Multi-Stage, Multi-Expert framework for zero-shot stance detection. MSME consists of three stages: (1) Knowledge Preparation, where relevant background knowledge is retrieved and stance labels are clarified; (2) Expert Reasoning, involving three specialized modules-Knowledge Expert distills salient facts and reasons from a knowledge perspective, Label Expert refines stance labels and reasons accordingly, and Pragmatic Expert detects rhetorical cues such as irony to infer intent from a pragmatic angle; (3) Decision Aggregation, where a Meta-Judge integrates all expert analyses to produce the final stance prediction. Experiments on three public datasets show that MSME achieves state-of-the-art performance across the board. 

---
# RapidUn: Influence-Driven Parameter Reweighting for Efficient Large Language Model Unlearning 

**Authors**: Guoshenghui Zhao, Huawei Lin, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.04457)  

**Abstract**: Removing specific data influence from large language models (LLMs) remains challenging, as retraining is costly and existing approximate unlearning methods are often unstable. The challenge is exacerbated when the forget set is small or imbalanced. We introduce RapidUn, an influence-driven and parameter-efficient unlearning framework. It first estimates per-sample influence through a fast estimation module, then maps these scores into adaptive update weights that guide selective parameter updates -- forgetting harmful behavior while retaining general knowledge. On Mistral-7B and Llama-3-8B across Dolly-15k and Alpaca-57k, RapidUn achieves up to 100 times higher efficiency than full retraining and consistently outperforms Fisher, GA, and LoReUn on both in-distribution and out-of-distribution forgetting. These results establish influence-guided parameter reweighting as a scalable and interpretable paradigm for LLM unlearning. 

---
# Sarcasm Detection on Reddit Using Classical Machine Learning and Feature Engineering 

**Authors**: Subrata Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2512.04396)  

**Abstract**: Sarcasm is common in online discussions, yet difficult for machines to identify because the intended meaning often contradicts the literal wording. In this work, I study sarcasm detection using only classical machine learning methods and explicit feature engineering, without relying on neural networks or context from parent comments. Using a 100,000-comment subsample of the Self-Annotated Reddit Corpus (SARC 2.0), I combine word-level and character-level TF-IDF features with simple stylistic indicators. Four models are evaluated: logistic regression, a linear SVM, multinomial Naive Bayes, and a random forest. Naive Bayes and logistic regression perform the strongest, achieving F1-scores around 0.57 for sarcastic comments. Although the lack of conversational context limits performance, the results offer a clear and reproducible baseline for sarcasm detection using lightweight and interpretable methods. 

---
# MASE: Interpretable NLP Models via Model-Agnostic Saliency Estimation 

**Authors**: Zhou Yang, Shunyan Luo, Jiazhen Zhu, Fang Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.04386)  

**Abstract**: Deep neural networks (DNNs) have made significant strides in Natural Language Processing (NLP), yet their interpretability remains elusive, particularly when evaluating their intricate decision-making processes. Traditional methods often rely on post-hoc interpretations, such as saliency maps or feature visualization, which might not be directly applicable to the discrete nature of word data in NLP. Addressing this, we introduce the Model-agnostic Saliency Estimation (MASE) framework. MASE offers local explanations for text-based predictive models without necessitating in-depth knowledge of a model's internal architecture. By leveraging Normalized Linear Gaussian Perturbations (NLGP) on the embedding layer instead of raw word inputs, MASE efficiently estimates input saliency. Our results indicate MASE's superiority over other model-agnostic interpretation methods, especially in terms of Delta Accuracy, positioning it as a promising tool for elucidating the operations of text-based models in NLP. 

---
# LangSAT: A Novel Framework Combining NLP and Reinforcement Learning for SAT Solving 

**Authors**: Muyu Pan, Matthew Walter, Dheeraj Kodakandla, Mahfuza Farooque  

**Link**: [PDF](https://arxiv.org/pdf/2512.04374)  

**Abstract**: Our work presents a novel reinforcement learning (RL) based framework to optimize heuristic selection within the conflict-driven clause learning (CDCL) process, improving the efficiency of Boolean satisfia- bility (SAT) solving. The proposed system, LangSAT, bridges the gap between natural language inputs and propositional logic by converting English descriptions into Conjunctive Normal Form (CNF) expressions and solving them using an RL-enhanced CDCL SAT solver. Unlike existing SAT-solving platforms that require CNF as input, LangSAT enables users to input standard English descriptions, making SAT-solving more accessible. The framework comprises two key components: Lang2Logic, which translates English sentences into CNF expressions, and SmartSAT, an RL-based SAT solver. SmartSAT encodes clause-variable relationships as structured graph representations and extracts global features specific to the SAT problem. This implementation provides the RL agent with deeper contextual information, enabling SAT problems to be solved more efficiently. Lang2Logic was evaluated on diverse natural language inputs, processing descriptions up to 450 words. The generated CNFs were solved by SmartSAT, which demonstrated comparable performance to traditional CDCL heuristics with respect to solving time. The combined LangSAT framework offers a more accessible and scalable solution for SAT-solving tasks across reasoning, formal verification, and debugging. 

---
# ClusterFusion: Hybrid Clustering with Embedding Guidance and LLM Adaptation 

**Authors**: Yiming Xu, Yuan Yuan, Vijay Viswanathan, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2512.04350)  

**Abstract**: Text clustering is a fundamental task in natural language processing, yet traditional clustering algorithms with pre-trained embeddings often struggle in domain-specific contexts without costly fine-tuning. Large language models (LLMs) provide strong contextual reasoning, yet prior work mainly uses them as auxiliary modules to refine embeddings or adjust cluster boundaries. We propose ClusterFusion, a hybrid framework that instead treats the LLM as the clustering core, guided by lightweight embedding methods. The framework proceeds in three stages: embedding-guided subset partition, LLM-driven topic summarization, and LLM-based topic assignment. This design enables direct incorporation of domain knowledge and user preferences, fully leveraging the contextual adaptability of LLMs. Experiments on three public benchmarks and two new domain-specific datasets demonstrate that ClusterFusion not only achieves state-of-the-art performance on standard tasks but also delivers substantial gains in specialized domains. To support future work, we release our newly constructed dataset and results on all benchmarks. 

---
# DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle 

**Authors**: Fangyu Lei, Jinxiang Meng, Yiming Huang, Junjie Zhao, Yitong Zhang, Jianwen Luo, Xin Zou, Ruiyi Yang, Wenbo Shi, Yan Gao, Shizhu He, Zuo Wang, Qian Liu, Yang Wang, Ke Wang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.04324)  

**Abstract**: Real-world enterprise data intelligence workflows encompass data engineering that turns raw sources into analytical-ready tables and data analysis that convert those tables into decision-oriented insights. We introduce DAComp, a benchmark of 210 tasks that mirrors these complex workflows. Data engineering (DE) tasks require repository-level engineering on industrial schemas, including designing and building multi-stage SQL pipelines from scratch and evolving existing systems under evolving requirements. Data analysis (DA) tasks pose open-ended business problems that demand strategic planning, exploratory analysis through iterative coding, interpretation of intermediate results, and the synthesis of actionable recommendations. Engineering tasks are scored through execution-based, multi-metric evaluation. Open-ended tasks are assessed by a reliable, experimentally validated LLM-judge, which is guided by hierarchical, meticulously crafted rubrics. Our experiments reveal that even state-of-the-art agents falter on DAComp. Performance on DE tasks is particularly low, with success rates under 20%, exposing a critical bottleneck in holistic pipeline orchestration, not merely code generation. Scores on DA tasks also average below 40%, highlighting profound deficiencies in open-ended reasoning and demonstrating that engineering and analysis are distinct capabilities. By clearly diagnosing these limitations, DAComp provides a rigorous and realistic testbed to drive the development of truly capable autonomous data agents for enterprise settings. Our data and code are available at this https URL 

---
# SQuARE: Structured Query & Adaptive Retrieval Engine For Tabular Formats 

**Authors**: Chinmay Gondhalekar, Urjitkumar Patel, Fang-Chun Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2512.04292)  

**Abstract**: Accurate question answering over real spreadsheets remains difficult due to multirow headers, merged cells, and unit annotations that disrupt naive chunking, while rigid SQL views fail on files lacking consistent schemas. We present SQuARE, a hybrid retrieval framework with sheet-level, complexity-aware routing. It computes a continuous score based on header depth and merge density, then routes queries either through structure-preserving chunk retrieval or SQL over an automatically constructed relational representation. A lightweight agent supervises retrieval, refinement, or combination of results across both paths when confidence is low. This design maintains header hierarchies, time labels, and units, ensuring that returned values are faithful to the original cells and straightforward to verify. Evaluated on multi-header corporate balance sheets, a heavily merged World Bank workbook, and diverse public datasets, SQuARE consistently surpasses single-strategy baselines and ChatGPT-4o on both retrieval precision and end-to-end answer accuracy while keeping latency predictable. By decoupling retrieval from model choice, the system is compatible with emerging tabular foundation models and offers a practical bridge toward a more robust table understanding. 

---
# Computational Linguistics Meets Libyan Dialect: A Study on Dialect Identification 

**Authors**: Mansour Essgaer, Khamis Massud, Rabia Al Mamlook, Najah Ghmaid  

**Link**: [PDF](https://arxiv.org/pdf/2512.04257)  

**Abstract**: This study investigates logistic regression, linear support vector machine, multinomial Naive Bayes, and Bernoulli Naive Bayes for classifying Libyan dialect utterances gathered from Twitter. The dataset used is the QADI corpus, which consists of 540,000 sentences across 18 Arabic dialects. Preprocessing challenges include handling inconsistent orthographic variations and non-standard spellings typical of the Libyan dialect. The chi-square analysis revealed that certain features, such as email mentions and emotion indicators, were not significantly associated with dialect classification and were thus excluded from further analysis. Two main experiments were conducted: (1) evaluating the significance of meta-features extracted from the corpus using the chi-square test and (2) assessing classifier performance using different word and character n-gram representations. The classification experiments showed that Multinomial Naive Bayes (MNB) achieved the highest accuracy of 85.89% and an F1-score of 0.85741 when using a (1,2) word n-gram and (1,5) character n-gram representation. In contrast, Logistic Regression and Linear SVM exhibited slightly lower performance, with maximum accuracies of 84.41% and 84.73%, respectively. Additional evaluation metrics, including log loss, Cohen kappa, and Matthew correlation coefficient, further supported the effectiveness of MNB in this task. The results indicate that carefully selected n-gram representations and classification models play a crucial role in improving the accuracy of Libyan dialect identification. This study provides empirical benchmarks and insights for future research in Arabic dialect NLP applications. 

---
# On GRPO Collapse in Search-R1: The Lazy Likelihood-Displacement Death Spiral 

**Authors**: Wenlong Deng, Yushu Li, Boying Gong, Yi Ren, Christos Thrampoulidis, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.04220)  

**Abstract**: Tool-integrated (TI) reinforcement learning (RL) enables large language models (LLMs) to perform multi-step reasoning by interacting with external tools such as search engines and retrievers. Group Relative Policy Optimization (GRPO), exemplified by the recent Search-R1, offers fast convergence and a value-free formulation that makes it appealing for this setting, yet consistently suffers from training collapse. We identify Lazy Likelihood Displacement (LLD), a systematic reduction or stagnation in the likelihood of both correct and incorrect responses, as the core mechanism driving this failure. LLD emerges early and triggers a self-reinforcing LLD Death Spiral, where declining likelihood leads to low-confidence responses, inflating gradients, and ultimately causing collapse. We empirically characterize this process across models on a Search-R1-style, search-integrated question answering task, revealing a consistent three-phase trajectory: early stagnation, steady decay, and accelerated collapse. To address this, we propose a lightweight likelihood-preserving regularization LLDS for GRPO that activates only when a trajectory's likelihood decreases, and regularizes only the tokens responsible. This fine-grained structure mitigates LLD with minimal interference to optimization. Across seven open-domain and multi-hop QA benchmarks, our method stabilizes training, prevents gradient explosion, and yields substantial performance improvements, including +37.8% gains on Qwen2.5-3B and +32.0% gains on Qwen2.5-7B. Our results establish LLD as a fundamental bottleneck in GRPO-based TIRL and provide a practical path toward stable, scalable training of tool-integrated LLM. 

---
# DraCo: Draft as CoT for Text-to-Image Preview and Rare Concept Generation 

**Authors**: Dongzhi Jiang, Renrui Zhang, Haodong Li, Zhuofan Zong, Ziyu Guo, Jun He, Claire Guo, Junyan Ye, Rongyao Fang, Weijia Li, Rui Liu, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.05112)  

**Abstract**: Recent unified multimodal large language models (MLLMs) have shown impressive capabilities, incorporating chain-of-thought (CoT) reasoning for enhanced text-to-image generation. However, existing approaches remain limited, either treating the model merely as a standalone generator or relying on abstract textual planning. To this end, we propose Draft-as-CoT (DraCo), a novel interleaved reasoning paradigm that fully leverages both textual and visual contents in CoT for better planning and verification. Our method first generates a low-resolution draft image as preview, providing more concrete and structural visual planning and guidance. Then, we employ the model's inherent understanding capability to verify potential semantic misalignments between the draft and input prompt, and performs refinement through selective corrections with super-resolution. In this way, our approach addresses two fundamental challenges: the coarse-grained nature of textual planning and the difficulty in generating rare attribute combinations. To support training, we curate DraCo-240K, aiming to enhance three atomic capabilities spanning general correction, instance manipulation, and layout reorganization. Supported by DraCo-CFG, a specialized classifier-free guidance (CFG) strategy for interleaved reasoning, DraCo achieves a tremendous increase on GenEval (+8%), Imagine-Bench (+0.91), and GenEval++ (+3%), significantly outperforming direct generation and other generation methods empowered by CoT. 

---
# Multi-LLM Collaboration for Medication Recommendation 

**Authors**: Huascar Sanchez, Briland Hitaj, Jules Bergmann, Linda Briesemeister  

**Link**: [PDF](https://arxiv.org/pdf/2512.05066)  

**Abstract**: As healthcare increasingly turns to AI for scalable and trustworthy clinical decision support, ensuring reliability in model reasoning remains a critical challenge. Individual large language models (LLMs) are susceptible to hallucinations and inconsistency, whereas naive ensembles of models often fail to deliver stable and credible recommendations. Building on our previous work on LLM Chemistry, which quantifies the collaborative compatibility among LLMs, we apply this framework to improve the reliability in medication recommendation from brief clinical vignettes. Our approach leverages multi-LLM collaboration guided by Chemistry-inspired interaction modeling, enabling ensembles that are effective (exploiting complementary strengths), stable (producing consistent quality), and calibrated (minimizing interference and error amplification). We evaluate our Chemistry-based Multi-LLM collaboration strategy on real-world clinical scenarios to investigate whether such interaction-aware ensembles can generate credible, patient-specific medication recommendations. Preliminary results are encouraging, suggesting that LLM Chemistry-guided collaboration may offer a promising path toward reliable and trustworthy AI assistants in clinical practice. 

---
# CARL: Critical Action Focused Reinforcement Learning for Multi-Step Agent 

**Authors**: Leyang Shen, Yang Zhang, Chun Kai Ling, Xiaoyan Zhao, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2512.04949)  

**Abstract**: Agents capable of accomplishing complex tasks through multiple interactions with the environment have emerged as a popular research direction. However, in such multi-step settings, the conventional group-level policy optimization algorithm becomes suboptimal because of its underlying assumption that each action holds equal contribution, which deviates significantly from reality. Our analysis reveals that only a small fraction of actions are critical in determining the final outcome. Building on this insight, we propose CARL, a critical-action-focused reinforcement learning algorithm tailored for multi-step agents. CARL achieves focused training through providing action-level optimization signals for high-criticality actions while excluding low-criticality actions from model update. Extensive experiments demonstrate that CARL achieves both stronger performance and higher efficiency during training and inference across diverse evaluation settings. 

---
# Algorithmic Thinking Theory 

**Authors**: MohammadHossein Bateni, Vincent Cohen-Addad, Yuzhou Gu, Silvio Lattanzi, Simon Meierhans, Christopher Mohri  

**Link**: [PDF](https://arxiv.org/pdf/2512.04923)  

**Abstract**: Large language models (LLMs) have proven to be highly effective for solving complex reasoning tasks. Surprisingly, their capabilities can often be improved by iterating on previously generated solutions. In this context, a reasoning plan for generating and combining a set of solutions can be thought of as an algorithm for reasoning using a probabilistic oracle.
We introduce a theoretical framework for analyzing such reasoning algorithms. This framework formalizes the principles underlying popular techniques for iterative improvement and answer aggregation, providing a foundation for designing a new generation of more powerful reasoning methods. Unlike approaches for understanding models that rely on architectural specifics, our model is grounded in experimental evidence. As a result, it offers a general perspective that may extend to a wide range of current and future reasoning oracles. 

---
# The AI Consumer Index (ACE) 

**Authors**: Julien Benchek, Rohit Shetty, Benjamin Hunsberger, Ajay Arun, Zach Richards, Brendan Foody, Osvald Nitski, Bertie Vidgen  

**Link**: [PDF](https://arxiv.org/pdf/2512.04921)  

**Abstract**: We introduce the first version of the AI Consumer Index (ACE), a benchmark for assessing whether frontier AI models can perform high-value consumer tasks. ACE contains a hidden heldout set of 400 test cases, split across four consumer activities: shopping, food, gaming, and DIY. We are also open sourcing 80 cases as a devset with a CC-BY license. For the ACE leaderboard we evaluated 10 frontier models (with websearch turned on) using a novel grading methodology that dynamically checks whether relevant parts of the response are grounded in the retrieved web sources. GPT 5 (Thinking = High) is the top-performing model, scoring 56.1%, followed by o3 Pro (Thinking = On) (55.2%) and GPT 5.1 (Thinking = High) (55.1%). Models differ across domains, and in Shopping the top model scores under 50%. For some requests (such as giving the correct price or providing working links), models are highly prone to hallucination. Overall, ACE shows a substantial gap between the performance of even the best models and consumers' AI needs. 

---
# STELLA: Guiding Large Language Models for Time Series Forecasting with Semantic Abstractions 

**Authors**: Junjie Fan, Hongye Zhao, Linduo Wei, Jiayu Rao, Guijia Li, Jiaxin Yuan, Wenqi Xu, Yong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2512.04871)  

**Abstract**: Recent adaptations of Large Language Models (LLMs) for time series forecasting often fail to effectively enhance information for raw series, leaving LLM reasoning capabilities underutilized. Existing prompting strategies rely on static correlations rather than generative interpretations of dynamic behavior, lacking critical global and instance-specific context. To address this, we propose STELLA (Semantic-Temporal Alignment with Language Abstractions), a framework that systematically mines and injects structured supplementary and complementary information. STELLA employs a dynamic semantic abstraction mechanism that decouples input series into trend, seasonality, and residual components. It then translates intrinsic behavioral features of these components into Hierarchical Semantic Anchors: a Corpus-level Semantic Prior (CSP) for global context and a Fine-grained Behavioral Prompt (FBP) for instance-level patterns. Using these anchors as prefix-prompts, STELLA guides the LLM to model intrinsic dynamics. Experiments on eight benchmark datasets demonstrate that STELLA outperforms state-of-the-art methods in long- and short-term forecasting, showing superior generalization in zero-shot and few-shot settings. Ablation studies further validate the effectiveness of our dynamically generated semantic anchors. 

---
# Are LLMs Truly Multilingual? Exploring Zero-Shot Multilingual Capability of LLMs for Information Retrieval: An Italian Healthcare Use Case 

**Authors**: Vignesh Kumar Kembu, Pierandrea Morandini, Marta Bianca Maria Ranzini, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2512.04834)  

**Abstract**: Large Language Models (LLMs) have become a key topic in AI and NLP, transforming sectors like healthcare, finance, education, and marketing by improving customer service, automating tasks, providing insights, improving diagnostics, and personalizing learning experiences. Information extraction from clinical records is a crucial task in digital healthcare. Although traditional NLP techniques have been used for this in the past, they often fall short due to the complexity, variability of clinical language, and high inner semantics in the free clinical text. Recently, Large Language Models (LLMs) have become a powerful tool for better understanding and generating human-like text, making them highly effective in this area. In this paper, we explore the ability of open-source multilingual LLMs to understand EHRs (Electronic Health Records) in Italian and help extract information from them in real-time. Our detailed experimental campaign on comorbidity extraction from EHR reveals that some LLMs struggle in zero-shot, on-premises settings, and others show significant variation in performance, struggling to generalize across various diseases when compared to native pattern matching and manual annotations. 

---
# MemLoRA: Distilling Expert Adapters for On-Device Memory Systems 

**Authors**: Massimo Bini, Ondrej Bohdal, Umberto Michieli, Zeynep Akata, Mete Ozay, Taha Ceritli  

**Link**: [PDF](https://arxiv.org/pdf/2512.04763)  

**Abstract**: Memory-augmented Large Language Models (LLMs) have demonstrated remarkable consistency during prolonged dialogues by storing relevant memories and incorporating them as context. Such memory-based personalization is also key in on-device settings that allow users to keep their conversations and data private. However, memory-augmented systems typically rely on LLMs that are too costly for local on-device deployment. Even though Small Language Models (SLMs) are more suitable for on-device inference than LLMs, they cannot achieve sufficient performance. Additionally, these LLM-based systems lack native visual capabilities, limiting their applicability in multimodal contexts. In this paper, we introduce (i) MemLoRA, a novel memory system that enables local deployment by equipping SLMs with specialized memory adapters, and (ii) its vision extension MemLoRA-V, which integrates small Vision-Language Models (SVLMs) to memory systems, enabling native visual understanding. Following knowledge distillation principles, each adapter is trained separately for specific memory operations$\unicode{x2013}$knowledge extraction, memory update, and memory-augmented generation. Equipped with memory adapters, small models enable accurate on-device memory operations without cloud dependency. On text-only operations, MemLoRA outperforms 10$\times$ larger baseline models (e.g., Gemma2-27B) and achieves performance comparable to 60$\times$ larger models (e.g., GPT-OSS-120B) on the LoCoMo benchmark. To evaluate visual understanding operations instead, we extend LoCoMo with challenging Visual Question Answering tasks that require direct visual reasoning. On this, our VLM-integrated MemLoRA-V shows massive improvements over caption-based approaches (81.3 vs. 23.7 accuracy) while keeping strong performance in text-based tasks, demonstrating the efficacy of our method in multimodal contexts. 

---
# Towards Ethical Multi-Agent Systems of Large Language Models: A Mechanistic Interpretability Perspective 

**Authors**: Jae Hee Lee, Anne Lauscher, Stefano V. Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2512.04691)  

**Abstract**: Large language models (LLMs) have been widely deployed in various applications, often functioning as autonomous agents that interact with each other in multi-agent systems. While these systems have shown promise in enhancing capabilities and enabling complex tasks, they also pose significant ethical challenges. This position paper outlines a research agenda aimed at ensuring the ethical behavior of multi-agent systems of LLMs (MALMs) from the perspective of mechanistic interpretability. We identify three key research challenges: (i) developing comprehensive evaluation frameworks to assess ethical behavior at individual, interactional, and systemic levels; (ii) elucidating the internal mechanisms that give rise to emergent behaviors through mechanistic interpretability; and (iii) implementing targeted parameter-efficient alignment techniques to steer MALMs towards ethical behaviors without compromising their performance. 

---
# Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs 

**Authors**: Jinbo Liu, Defu Cao, Yifei Wei, Tianyao Su, Yuan Liang, Yushun Dong, Yue Zhao, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.04668)  

**Abstract**: Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a framework that measures how network structure shapes leakage. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over up to 10 interaction rounds, we quantify leakage as the fraction of ground-truth PII recovered from attacking agent outputs via exact matching. We systematically evaluate six common network topologies (fully connected, ring, chain, binary tree, star, and star-ring), varying agent counts $n\in\{4,5,6\}$, attacker-target placements, and base models. Our findings reveal consistent patterns: fully connected graphs exhibit maximum leakage while chains provide strongest protection; shorter attacker-target graph distance and higher target centrality significantly increase vulnerability; leakage rises sharply in early rounds before plateauing; model choice shifts absolute leakage rates but preserves topology rankings; temporal/locational PII attributes leak more readily than identity credentials or regulated identifiers. These results provide the first systematic mapping from architectural choices to measurable privacy risk, yielding actionable guidance: prefer sparse or hierarchical connectivity, maximize attacker-target separation, limit node degree and network radius, avoid shortcuts bypassing hubs, and implement topology-aware access controls. 

---
# SEASON: Mitigating Temporal Hallucination in Video Large Language Models via Self-Diagnostic Contrastive Decoding 

**Authors**: Chang-Hsun Wu, Kai-Po Chang, Yu-Yang Sheng, Hung-Kai Chung, Kuei-Chun Wang, Yu-Chiang Frank Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.04643)  

**Abstract**: Video Large Language Models (VideoLLMs) have shown remarkable progress in video understanding. However, these models still struggle to effectively perceive and exploit rich temporal information in videos when responding to user queries. Therefore, they often generate descriptions of events that are temporal inconsistent or causally implausible, causing severe hallucination issues. While most prior studies have focused on spatial hallucinations (e.g. object mismatches), temporal reasoning in video understanding remains relatively underexplored. To address this issue, we propose Self-Diagnostic Contrastive Decoding (SEASON), a training-free method that adaptively enhances temporal and spatial faithfulness for each output token. It achieves this by dynamically diagnosing each token's hallucination tendency and applying adaptive contrastive decoding against its corresponding temporal and spatial negatives. Extensive experiments demonstrate that SEASON outperforms all existing training-free hallucination mitigation approaches on three hallucination examination benchmarks, while further improves VideoLLMs across four general video understanding benchmarks. The code will be released upon acceptance. 

---
# Limit cycles for speech 

**Authors**: Adamantios I. Gafos, Stephan R. Kuberski  

**Link**: [PDF](https://arxiv.org/pdf/2512.04642)  

**Abstract**: Rhythmic fluctuations in acoustic energy and accompanying neuronal excitations in cortical oscillations are characteristic of human speech, yet whether a corresponding rhythmicity inheres in the articulatory movements that generate speech remains unclear. The received understanding of speech movements as discrete, goal-oriented actions struggles to make contact with the rhythmicity findings. In this work, we demonstrate that an unintuitive -- but no less principled than the conventional -- representation for discrete movements reveals a pervasive limit cycle organization and unlocks the recovery of previously inaccessible rhythmic structure underlying the motor activity of speech. These results help resolve a time-honored tension between the ubiquity of biological rhythmicity and discreteness in speech, the quintessential human higher function, by revealing a rhythmic organization at the most fundamental level of individual articulatory actions. 

---
# Natural Language Actor-Critic: Scalable Off-Policy Learning in Language Space 

**Authors**: Joey Hong, Kang Liu, Zhan Ling, Jiecao Chen, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2512.04601)  

**Abstract**: Large language model (LLM) agents -- LLMs that dynamically interact with an environment over long horizons -- have become an increasingly important area of research, enabling automation in complex tasks involving tool-use, web browsing, and dialogue with people. In the absence of expert demonstrations, training LLM agents has relied on policy gradient methods that optimize LLM policies with respect to an (often sparse) reward function. However, in long-horizon tasks with sparse rewards, learning from trajectory-level rewards can be noisy, leading to training that is unstable and has high sample complexity. Furthermore, policy improvement hinges on discovering better actions through exploration, which can be difficult when actions lie in natural language space. In this paper, we propose Natural Language Actor-Critic (NLAC), a novel actor-critic algorithm that trains LLM policies using a generative LLM critic that produces natural language rather than scalar values. This approach leverages the inherent strengths of LLMs to provide a richer and more actionable training signal; particularly, in tasks with large, open-ended action spaces, natural language explanations for why an action is suboptimal can be immensely useful for LLM policies to reason how to improve their actions, without relying on random exploration. Furthermore, our approach can be trained off-policy without policy gradients, offering a more data-efficient and stable alternative to existing on-policy methods. We present results on a mixture of reasoning, web browsing, and tool-use with dialogue tasks, demonstrating that NLAC shows promise in outperforming existing training approaches and offers a more scalable and stable training paradigm for LLM agents. 

---
# Mitigating Object and Action Hallucinations in Multimodal LLMs via Self-Augmented Contrastive Alignment 

**Authors**: Kai-Po Chang, Wei-Yuan Cheng, Chi-Pin Huang, Fu-En Yang, Yu-Chiang Frank Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.04356)  

**Abstract**: Recent advancement in multimodal LLMs (MLLMs) has demonstrated their remarkable capability to generate descriptive captions for input videos. However, these models suffer from factual inaccuracies in the generated descriptions, causing severe hallucination issues. While prior works have explored alleviating hallucinations for static images, jointly mitigating visual object and temporal action hallucinations for dynamic videos remains a challenging and unsolved task. To tackle this challenge, we propose a Self-Augmented Contrastive Alignment (SANTA) framework for enabling object and action faithfulness by exempting the spurious correlations and enforcing the emphasis on visual facts. SANTA employs a hallucinative self-augmentation scheme to identify the potential hallucinations that lie in the MLLM and transform the original captions to the contrasted negatives. Furthermore, we develop a tracklet-phrase contrastive alignment to match the regional objects and relation-guided actions with their corresponding visual and temporal phrases. Extensive experiments demonstrate that SANTA outperforms existing methods in alleviating object and action hallucinations, yielding superior performance on the hallucination examination benchmarks. 

---
# Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction 

**Authors**: Rui Fonseca, Bruno Martins, Gil Rocha  

**Link**: [PDF](https://arxiv.org/pdf/2512.04309)  

**Abstract**: Image captioning has drawn considerable attention from the natural language processing and computer vision fields. Aiming to reduce the reliance on curated data, several studies have explored image captioning without any humanly-annotated image-text pairs for training, although existing methods are still outperformed by fully supervised approaches. This paper proposes TOMCap, i.e., an improved text-only training method that performs captioning without the need for aligned image-caption pairs. The method is based on prompting a pre-trained language model decoder with information derived from a CLIP representation, after undergoing a process to reduce the modality gap. We specifically tested the combined use of retrieved examples of captions, and latent vector representations, to guide the generation process. Through extensive experiments, we show that TOMCap outperforms other training-free and text-only methods. We also analyze the impact of different choices regarding the configuration of the retrieval-augmentation and modality gap reduction components. 

---
# Balancing Safety and Helpfulness in Healthcare AI Assistants through Iterative Preference Alignment 

**Authors**: Huy Nghiem, Swetasudha Panda, Devashish Khatwani, Huy V. Nguyen, Krishnaram Kenthapadi, Hal Daumé III  

**Link**: [PDF](https://arxiv.org/pdf/2512.04210)  

**Abstract**: Large Language Models (LLMs) are increasingly used in healthcare, yet ensuring their safety and trustworthiness remains a barrier to deployment. Conversational medical assistants must avoid unsafe compliance without over-refusing benign queries. We present an iterative post-deployment alignment framework that applies Kahneman-Tversky Optimization (KTO) and Direct Preference Optimization (DPO) to refine models against domain-specific safety signals. Using the CARES-18K benchmark for adversarial robustness, we evaluate four LLMs (Llama-3B/8B, Meditron-8B, Mistral-7B) across multiple cycles. Our results show up to 42% improvement in safety-related metrics for harmful query detection, alongside interesting trade-offs against erroneous refusals, thereby exposing architecture-dependent calibration biases. We also perform ablation studies to identify when self-evaluation is reliable and when external or finetuned judges are necessary to maximize performance gains. Our findings underscore the importance of adopting best practices that balance patient safety, user trust, and clinical utility in the design of conversational medical assistants. 

---
# Network of Theseus (like the ship) 

**Authors**: Vighnesh Subramaniam, Colin Conwell, Boris Katz, Andrei Barbu, Brian Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2512.04198)  

**Abstract**: A standard assumption in deep learning is that the inductive bias introduced by a neural network architecture must persist from training through inference. The architecture you train with is the architecture you deploy. This assumption constrains the community from selecting architectures that may have desirable efficiency or design properties due to difficulties with optimization. We challenge this assumption with Network of Theseus (NoT), a method for progressively converting a trained, or even untrained, guide network architecture part-by-part into an entirely different target network architecture while preserving the performance of the guide network. At each stage, components in the guide network architecture are incrementally replaced with target architecture modules and aligned via representational similarity metrics. This procedure largely preserves the functionality of the guide network even under substantial architectural changes-for example, converting a convolutional network into a multilayer perceptron, or GPT-2 into a recurrent neural network. By decoupling optimization from deployment, NoT expands the space of viable inference-time architectures, opening opportunities for better accuracy-efficiency tradeoffs and enabling more directed exploration of the architectural design space. 

---
# Can machines perform a qualitative data analysis? Reading the debate with Alan Turing 

**Authors**: Stefano De Paoli  

**Link**: [PDF](https://arxiv.org/pdf/2512.04121)  

**Abstract**: This paper reflects on the literature that rejects the use of Large Language Models (LLMs) in qualitative data analysis. It illustrates through empirical evidence as well as critical reflections why the current critical debate is focusing on the wrong problems. The paper proposes that the focus of researching the use of the LLMs for qualitative analysis is not the method per se, but rather the empirical investigation of an artificial system performing an analysis. The paper builds on the seminal work of Alan Turing and reads the current debate using key ideas from Turing "Computing Machinery and Intelligence". This paper therefore reframes the debate on qualitative analysis with LLMs and states that rather than asking whether machines can perform qualitative analysis in principle, we should ask whether with LLMs we can produce analyses that are sufficiently comparable to human analysts. In the final part the contrary views to performing qualitative analysis with LLMs are analysed using the same writing and rhetorical style that Turing used in his seminal work, to discuss the contrary views to the main question. 

---
# Towards Contextual Sensitive Data Detection 

**Authors**: Liang Telkamp, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2512.04120)  

**Abstract**: The emergence of open data portals necessitates more attention to protecting sensitive data before datasets get published and exchanged. While an abundance of methods for suppressing sensitive data exist, the conceptualization of sensitive data and methods to detect it, focus particularly on personal data that, if disclosed, may be harmful or violate privacy. We observe the need for refining and broadening our definitions of sensitive data, and argue that the sensitivity of data depends on its context. Based on this definition, we introduce two mechanisms for contextual sensitive data detection that con- sider the broader context of a dataset at hand. First, we introduce type contextualization, which first detects the semantic type of particular data values, then considers the overall context of the data values within the dataset or document. Second, we introduce domain contextualization which determines sensitivity of a given dataset in the broader context based on the retrieval of relevant rules from documents that specify data sensitivity (e.g., data topic and geographic origin). Experiments with these mechanisms, assisted by large language models (LLMs), confirm that: 1) type-contextualization significantly reduces the number of false positives for type-based sensitive data detection and reaches a recall of 94% compared to 63% with commercial tools, and 2) domain-contextualization leveraging sensitivity rule retrieval is effective for context-grounded sensitive data detection in non-standard data domains such as humanitarian datasets. Evaluation with humanitarian data experts also reveals that context-grounded LLM explanations provide useful guidance in manual data auditing processes, improving consistency. We open-source mechanisms and annotated datasets for contextual sensitive data detection at this https URL. 

---
# Retrieval-Augmented Few-Shot Prompting Versus Fine-Tuning for Code Vulnerability Detection 

**Authors**: Fouad Trad, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2512.04106)  

**Abstract**: Few-shot prompting has emerged as a practical alternative to fine-tuning for leveraging the capabilities of large language models (LLMs) in specialized tasks. However, its effectiveness depends heavily on the selection and quality of in-context examples, particularly in complex domains. In this work, we examine retrieval-augmented prompting as a strategy to improve few-shot performance in code vulnerability detection, where the goal is to identify one or more security-relevant weaknesses present in a given code snippet from a predefined set of vulnerability categories. We perform a systematic evaluation using the Gemini-1.5-Flash model across three approaches: (1) standard few-shot prompting with randomly selected examples, (2) retrieval-augmented prompting using semantically similar examples, and (3) retrieval-based labeling, which assigns labels based on retrieved examples without model inference. Our results show that retrieval-augmented prompting consistently outperforms the other prompting strategies. At 20 shots, it achieves an F1 score of 74.05% and a partial match accuracy of 83.90%. We further compare this approach against zero-shot prompting and several fine-tuned models, including Gemini-1.5-Flash and smaller open-source models such as DistilBERT, DistilGPT2, and CodeBERT. Retrieval-augmented prompting outperforms both zero-shot (F1 score: 36.35%, partial match accuracy: 20.30%) and fine-tuned Gemini (F1 score: 59.31%, partial match accuracy: 53.10%), while avoiding the training time and cost associated with model fine-tuning. On the other hand, fine-tuning CodeBERT yields higher performance (F1 score: 91.22%, partial match accuracy: 91.30%) but requires additional training, maintenance effort, and resources. 

---
# Human-Centred Evaluation of Text-to-Image Generation Models for Self-expression of Mental Distress: A Dataset Based on GPT-4o 

**Authors**: Sui He, Shenbin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2512.04087)  

**Abstract**: Effective communication is central to achieving positive healthcare outcomes in mental health contexts, yet international students often face linguistic and cultural barriers that hinder their communication of mental distress. In this study, we evaluate the effectiveness of AI-generated images in supporting self-expression of mental distress. To achieve this, twenty Chinese international students studying at UK universities were invited to describe their personal experiences of mental distress. These descriptions were elaborated using GPT-4o with four persona-based prompt templates rooted in contemporary counselling practice to generate corresponding images. Participants then evaluated the helpfulness of generated images in facilitating the expression of their feelings based on their original descriptions. The resulting dataset comprises 100 textual descriptions of mental distress, 400 generated images, and corresponding human evaluation scores. Findings indicate that prompt design substantially affects perceived helpfulness, with the illustrator persona achieving the highest ratings. This work introduces the first publicly available text-to-image evaluation dataset with human judgment scores in the mental health domain, offering valuable resources for image evaluation, reinforcement learning with human feedback, and multi-modal research on mental health communication. 

---
