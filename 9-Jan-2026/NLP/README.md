# GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization 

**Authors**: Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2601.05242)  

**Abstract**: As language models become increasingly capable, users expect them to provide not only accurate responses but also behaviors aligned with diverse human preferences across a variety of scenarios. To achieve this, Reinforcement learning (RL) pipelines have begun incorporating multiple rewards, each capturing a distinct preference, to guide models toward these desired behaviors. However, recent work has defaulted to apply Group Relative Policy Optimization (GRPO) under multi-reward setting without examining its suitability. In this paper, we demonstrate that directly applying GRPO to normalize distinct rollout reward combinations causes them to collapse into identical advantage values, reducing the resolution of the training signal and resulting in suboptimal convergence and, in some cases, early training failure. We then introduce Group reward-Decoupled Normalization Policy Optimization (GDPO), a new policy optimization method to resolve these issues by decoupling the normalization of individual rewards, more faithfully preserving their relative differences and enabling more accurate multi-reward optimization, along with substantially improved training stability. We compare GDPO with GRPO across three tasks: tool calling, math reasoning, and coding reasoning, evaluating both correctness metrics (accuracy, bug ratio) and constraint adherence metrics (format, length). Across all settings, GDPO consistently outperforms GRPO, demonstrating its effectiveness and generalizability for multi-reward reinforcement learning optimization. 

---
# Measuring and Fostering Peace through Machine Learning and Artificial Intelligence 

**Authors**: P. Gilda, P. Dungarwal, A. Thongkham, E. T. Ajayi, S. Choudhary, T. M. Terol, C. Lam, J. P. Araujo, M. McFadyen-Mungalln, L. S. Liebovitch, P. T. Coleman, H. West, K. Sieck, S. Carter  

**Link**: [PDF](https://arxiv.org/pdf/2601.05232)  

**Abstract**: We used machine learning and artificial intelligence: 1) to measure levels of peace in countries from news and social media and 2) to develop on-line tools that promote peace by helping users better understand their own media diet. For news media, we used neural networks to measure levels of peace from text embeddings of on-line news sources. The model, trained on one news media dataset also showed high accuracy when used to analyze a different news dataset. For social media, such as YouTube, we developed other models to measure levels of social dimensions important in peace using word level (GoEmotions) and context level (Large Language Model) methods. To promote peace, we note that 71% of people 20-40 years old daily view most of their news through short videos on social media. Content creators of these videos are biased towards creating videos with emotional activation, making you angry to engage you, to increase clicks. We developed and tested a Chrome extension, MirrorMirror, which provides real-time feedback to YouTube viewers about the peacefulness of the media they are watching. Our long term goal is for MirrorMirror to evolve into an open-source tool for content creators, journalists, researchers, platforms, and individual users to better understand the tone of their media creation and consumption and its effects on viewers. Moving beyond simple engagement metrics, we hope to encourage more respectful, nuanced, and informative communication. 

---
# LELA: an LLM-based Entity Linking Approach with Zero-Shot Domain Adaptation 

**Authors**: Samy Haffoudhi, Fabian M. Suchanek, Nils Holzenberger  

**Link**: [PDF](https://arxiv.org/pdf/2601.05192)  

**Abstract**: Entity linking (mapping ambiguous mentions in text to entities in a knowledge base) is a foundational step in tasks such as knowledge graph construction, question-answering, and information extraction. Our method, LELA, is a modular coarse-to-fine approach that leverages the capabilities of large language models (LLMs), and works with different target domains, knowledge bases and LLMs, without any fine-tuning phase. Our experiments across various entity linking settings show that LELA is highly competitive with fine-tuned approaches, and substantially outperforms the non-fine-tuned ones. 

---
# Inside Out: Evolving User-Centric Core Memory Trees for Long-Term Personalized Dialogue Systems 

**Authors**: Jihao Zhao, Ding Chen, Zhaoxin Fan, Kerun Xu, Mengting Hu, Bo Tang, Feiyu Xiong, Zhiyu li  

**Link**: [PDF](https://arxiv.org/pdf/2601.05171)  

**Abstract**: Existing long-term personalized dialogue systems struggle to reconcile unbounded interaction streams with finite context constraints, often succumbing to memory noise accumulation, reasoning degradation, and persona inconsistency. To address these challenges, this paper proposes Inside Out, a framework that utilizes a globally maintained PersonaTree as the carrier of long-term user profiling. By constraining the trunk with an initial schema and updating the branches and leaves, PersonaTree enables controllable growth, achieving memory compression while preserving consistency. Moreover, we train a lightweight MemListener via reinforcement learning with process-based rewards to produce structured, executable, and interpretable {ADD, UPDATE, DELETE, NO_OP} operations, thereby supporting the dynamic evolution of the personalized tree. During response generation, PersonaTree is directly leveraged to enhance outputs in latency-sensitive scenarios; when users require more details, the agentic mode is triggered to introduce details on-demand under the constraints of the PersonaTree. Experiments show that PersonaTree outperforms full-text concatenation and various personalized memory systems in suppressing contextual noise and maintaining persona consistency. Notably, the small MemListener model achieves memory-operation decision performance comparable to, or even surpassing, powerful reasoning models such as DeepSeek-R1-0528 and Gemini-3-Pro. 

---
# Reverse-engineering NLI: A study of the meta-inferential properties of Natural Language Inference 

**Authors**: Rasmus Blanck, Bill Noble, Stergios Chatzikyriakidis  

**Link**: [PDF](https://arxiv.org/pdf/2601.05170)  

**Abstract**: Natural Language Inference (NLI) has been an important task for evaluating language models for Natural Language Understanding, but the logical properties of the task are poorly understood and often mischaracterized. Understanding the notion of inference captured by NLI is key to interpreting model performance on the task. In this paper we formulate three possible readings of the NLI label set and perform a comprehensive analysis of the meta-inferential properties they entail. Focusing on the SNLI dataset, we exploit (1) NLI items with shared premises and (2) items generated by LLMs to evaluate models trained on SNLI for meta-inferential consistency and derive insights into which reading of the logical relations is encoded by the dataset. 

---
# RelayLLM: Efficient Reasoning via Collaborative Decoding 

**Authors**: Chengsong Huang, Tong Zheng, Langlin Huang, Jinyuan Li, Haolin Liu, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05167)  

**Abstract**: Large Language Models (LLMs) for complex reasoning is often hindered by high computational costs and latency, while resource-efficient Small Language Models (SLMs) typically lack the necessary reasoning capacity. Existing collaborative approaches, such as cascading or routing, operate at a coarse granularity by offloading entire queries to LLMs, resulting in significant computational waste when the SLM is capable of handling the majority of reasoning steps. To address this, we propose RelayLLM, a novel framework for efficient reasoning via token-level collaborative decoding. Unlike routers, RelayLLM empowers the SLM to act as an active controller that dynamically invokes the LLM only for critical tokens via a special command, effectively "relaying" the generation process. We introduce a two-stage training framework, including warm-up and Group Relative Policy Optimization (GRPO) to teach the model to balance independence with strategic help-seeking. Empirical results across six benchmarks demonstrate that RelayLLM achieves an average accuracy of 49.52%, effectively bridging the performance gap between the two models. Notably, this is achieved by invoking the LLM for only 1.07% of the total generated tokens, offering a 98.2% cost reduction compared to performance-matched random routers. 

---
# DocDancer: Towards Agentic Document-Grounded Information Seeking 

**Authors**: Qintong Zhang, Xinjie Lv, Jialong Wu, Baixuan Li, Zhengwei Tao, Guochen Yan, Huanyao Zhang, Bin Wang, Jiahao Xu, Haitao Mi, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05163)  

**Abstract**: Document Question Answering (DocQA) focuses on answering questions grounded in given documents, yet existing DocQA agents lack effective tool utilization and largely rely on closed-source models. In this work, we introduce DocDancer, an end-to-end trained open-source Doc agent. We formulate DocQA as an information-seeking problem and propose a tool-driven agent framework that explicitly models document exploration and comprehension. To enable end-to-end training of such agents, we introduce an Exploration-then-Synthesis data synthesis pipeline that addresses the scarcity of high-quality training data for DocQA. Training on the synthesized data, the trained models on two long-context document understanding benchmarks, MMLongBench-Doc and DocBench, show their effectiveness. Further analysis provides valuable insights for the agentic tool design and synthetic data. 

---
# Agent-as-a-Judge 

**Authors**: Runyang You, Hongru Cai, Caiqi Zhang, Qiancheng Xu, Meng Liu, Tiezheng Yu, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.05111)  

**Abstract**: LLM-as-a-Judge has revolutionized AI evaluation by leveraging large language models for scalable assessments. However, as evaluands become increasingly complex, specialized, and multi-step, the reliability of LLM-as-a-Judge has become constrained by inherent biases, shallow single-pass reasoning, and the inability to verify assessments against real-world observations. This has catalyzed the transition to Agent-as-a-Judge, where agentic judges employ planning, tool-augmented verification, multi-agent collaboration, and persistent memory to enable more robust, verifiable, and nuanced evaluations. Despite the rapid proliferation of agentic evaluation systems, the field lacks a unified framework to navigate this shifting landscape. To bridge this gap, we present the first comprehensive survey tracing this evolution. Specifically, we identify key dimensions that characterize this paradigm shift and establish a developmental taxonomy. We organize core methodologies and survey applications across general and professional domains. Furthermore, we analyze frontier challenges and identify promising research directions, ultimately providing a clear roadmap for the next generation of agentic evaluation. 

---
# How Human is AI? Examining the Impact of Emotional Prompts on Artificial and Human and Responsiveness 

**Authors**: Florence Bernays, Marco Henriques Pereira, Jochen Menges  

**Link**: [PDF](https://arxiv.org/pdf/2601.05104)  

**Abstract**: This research examines how the emotional tone of human-AI interactions shapes ChatGPT and human behavior. In a between-subject experiment, we asked participants to express a specific emotion while working with ChatGPT (GPT-4.0) on two tasks, including writing a public response and addressing an ethical dilemma. We found that compared to interactions where participants maintained a neutral tone, ChatGPT showed greater improvement in its answers when participants praised ChatGPT for its responses. Expressing anger towards ChatGPT also led to a higher albeit smaller improvement relative to the neutral condition, whereas blaming ChatGPT did not improve its answers. When addressing an ethical dilemma, ChatGPT prioritized corporate interests less when participants expressed anger towards it, while blaming increases its emphasis on protecting the public interest. Additionally, we found that people used more negative, hostile, and disappointing expressions in human-human communication after interactions during which participants blamed rather than praised for their responses. Together, our findings demonstrate that the emotional tone people apply in human-AI interactions not only shape ChatGPT's outputs but also carry over into subsequent human-human communication. 

---
# Code-Mix Sentiment Analysis on Hinglish Tweets 

**Authors**: Aashi Garg, Aneshya Das, Arshi Arya, Anushka Goyal, Aditi  

**Link**: [PDF](https://arxiv.org/pdf/2601.05091)  

**Abstract**: The effectiveness of brand monitoring in India is increasingly challenged by the rise of Hinglish--a hybrid of Hindi and English--used widely in user-generated content on platforms like Twitter. Traditional Natural Language Processing (NLP) models, built for monolingual data, often fail to interpret the syntactic and semantic complexity of this code-mixed language, resulting in inaccurate sentiment analysis and misleading market insights. To address this gap, we propose a high-performance sentiment classification framework specifically designed for Hinglish tweets. Our approach fine-tunes mBERT (Multilingual BERT), leveraging its multilingual capabilities to better understand the linguistic diversity of Indian social media. A key component of our methodology is the use of subword tokenization, which enables the model to effectively manage spelling variations, slang, and out-of-vocabulary terms common in Romanized Hinglish. This research delivers a production-ready AI solution for brand sentiment tracking and establishes a strong benchmark for multilingual NLP in low-resource, code-mixed environments. 

---
# SemPA: Improving Sentence Embeddings of Large Language Models through Semantic Preference Alignment 

**Authors**: Ziyang Chen, Zhenxuan Huang, Yile Wang, Weiqin Wang, Lu Yin, Hui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05075)  

**Abstract**: Traditional sentence embedding methods employ token-level contrastive learning on non-generative pre-trained models. Recently, there have emerged embedding methods based on generative large language models (LLMs). These methods either rely on fixed prompt templates or involve modifications to the model architecture. The former lacks further optimization of the model and results in limited performance, while the latter alters the internal computational mechanisms of the model, thereby compromising its generative capabilities. We propose SemPA, a novel approach that boosts the sentence representations while preserving the generative ability of LLMs via semantic preference alignment. We leverage sentence-level Direct Preference Optimization (DPO) to efficiently optimize LLMs on a paraphrase generation task, where the model learns to discriminate semantically equivalent sentences while preserving inherent generative capacity. Theoretically, we establish a formal connection between DPO and contrastive learning under the Plackett-Luce model framework. Empirically, experimental results on both semantic textual similarity tasks and various benchmarks for LLMs show that SemPA achieves better semantic representations without sacrificing the inherent generation capability of LLMs. 

---
# Compositional Steering of Large Language Models with Steering Tokens 

**Authors**: Gorjan Radevski, Kiril Gashteovski, Giwon Hong, Carolin Lawrence, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2601.05062)  

**Abstract**: Deploying LLMs in real-world applications requires controllable output that satisfies multiple desiderata at the same time. While existing work extensively addresses LLM steering for a single behavior, \textit{compositional steering} -- i.e., steering LLMs simultaneously towards multiple behaviors -- remains an underexplored problem. In this work, we propose \emph{compositional steering tokens} for multi-behavior steering. We first embed individual behaviors, expressed as natural language instructions, into dedicated tokens via self-distillation. Contrary to most prior work, which operates in the activation space, our behavior steers live in the space of input tokens, enabling more effective zero-shot composition. We then train a dedicated \textit{composition token} on pairs of behaviors and show that it successfully captures the notion of composition: it generalizes well to \textit{unseen} compositions, including those with unseen behaviors as well as those with an unseen \textit{number} of behaviors. Our experiments across different LLM architectures show that steering tokens lead to superior multi-behavior control compared to competing approaches (instructions, activation steering, and LoRA merging). Moreover, we show that steering tokens complement natural language instructions, with their combination resulting in further gains. 

---
# ArcAligner: Adaptive Recursive Aligner for Compressed Context Embeddings in RAG 

**Authors**: Jianbo Li, Yi Jiang, Sendong Zhao, Bairui Hu, Haochun Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.05038)  

**Abstract**: Retrieval-Augmented Generation (RAG) helps LLMs stay accurate, but feeding long documents into a prompt makes the model slow and expensive. This has motivated context compression, ranging from token pruning and summarization to embedding-based compression. While researchers have tried ''compressing'' these documents into smaller summaries or mathematical embeddings, there is a catch: the more you compress the data, the more the LLM struggles to understand it. To address this challenge, we propose ArcAligner (Adaptive recursive context *Aligner*), a lightweight module integrated into the language model layers to help the model better utilize highly compressed context representations for downstream generation. It uses an adaptive ''gating'' system that only adds extra processing power when the information is complex, keeping the system fast. Across knowledge-intensive QA benchmarks, ArcAligner consistently beats compression baselines at comparable compression rates, especially on multi-hop and long-tail settings. The source code is publicly available. 

---
# Hán Dān Xué Bù (Mimicry) or Qīng Chū Yú Lán (Mastery)? A Cognitive Perspective on Reasoning Distillation in Large Language Models 

**Authors**: Yueqing Hu, Xinyang Peng, Shuting Peng, Hanqi Wang, Tianhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05019)  

**Abstract**: Recent Large Reasoning Models trained via reinforcement learning exhibit a "natural" alignment with human cognitive costs. However, we show that the prevailing paradigm of reasoning distillation -- training student models to mimic these traces via Supervised Fine-Tuning (SFT) -- fails to transmit this cognitive structure. Testing the "Hán Dān Xué Bù" (Superficial Mimicry) hypothesis across 14 models, we find that distillation induces a "Functional Alignment Collapse": while teacher models mirror human difficulty scaling ($\bar{r}=0.64$), distilled students significantly degrade this alignment ($\bar{r}=0.34$), often underperforming their own pre-distillation baselines ("Negative Transfer"). Our analysis suggests that SFT induces a "Cargo Cult" effect, where students ritualistically replicate the linguistic form of reasoning (verbosity) without internalizing the teacher's dynamic resource allocation policy. Consequently, reasoning distillation decouples computational cost from cognitive demand, revealing that human-like cognition is an emergent property of active reinforcement, not passive imitation. 

---
# Can Large Language Models Resolve Semantic Discrepancy in Self-Destructive Subcultures? Evidence from Jirai Kei 

**Authors**: Peng Wang, Xilin Tao, Siyi Yao, Jiageng Wu, Yuntao Zou, Zhuotao Tian, Libo Qin, Dagang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.05004)  

**Abstract**: Self-destructive behaviors are linked to complex psychological states and can be challenging to diagnose. These behaviors may be even harder to identify within subcultural groups due to their unique expressions. As large language models (LLMs) are applied across various fields, some researchers have begun exploring their application for detecting self-destructive behaviors. Motivated by this, we investigate self-destructive behavior detection within subcultures using current LLM-based methods. However, these methods have two main challenges: (1) Knowledge Lag: Subcultural slang evolves rapidly, faster than LLMs' training cycles; and (2) Semantic Misalignment: it is challenging to grasp the specific and nuanced expressions unique to subcultures. To address these issues, we proposed Subcultural Alignment Solver (SAS), a multi-agent framework that incorporates automatic retrieval and subculture alignment, significantly enhancing the performance of LLMs in detecting self-destructive behavior. Our experimental results show that SAS outperforms the current advanced multi-agent framework OWL. Notably, it competes well with fine-tuned LLMs. We hope that SAS will advance the field of self-destructive behavior detection in subcultural contexts and serve as a valuable resource for future researchers. 

---
# Learning from Mistakes: Negative Reasoning Samples Enhance Out-of-Domain Generalization 

**Authors**: Xueyun Tian, Minghua Ma, Bingbing Xu, Nuoyan Lyu, Wei Li, Heng Dong, Zheng Chu, Yuanzhuo Wang, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04992)  

**Abstract**: Supervised fine-tuning (SFT) on chain-of-thought (CoT) trajectories demonstrations is a common approach for enabling reasoning in large language models. Standard practices typically only retain trajectories with correct final answers (positives) while ignoring the rest (negatives). We argue that this paradigm discards substantial supervision and exacerbates overfitting, limiting out-of-domain (OOD) generalization. Specifically, we surprisingly find that incorporating negative trajectories into SFT yields substantial OOD generalization gains over positive-only training, as these trajectories often retain valid intermediate reasoning despite incorrect final answers. To understand this effect in depth, we systematically analyze data, training dynamics, and inference behavior, identifying 22 recurring patterns in negative chains that serve a dual role: they moderate loss descent to mitigate overfitting during training and boost policy entropy by 35.67% during inference to facilitate exploration. Motivated by these observations, we further propose Gain-based LOss Weighting (GLOW), an adaptive, sample-aware scheme that exploits such distinctive training dynamics by rescaling per-sample loss based on inter-epoch progress. Empirically, GLOW efficiently leverages unfiltered trajectories, yielding a 5.51% OOD gain over positive-only SFT on Qwen2.5-7B and boosting MMLU from 72.82% to 76.47% as an RL initialization. 

---
# Text as a Universal Interface for Transferable Personalization 

**Authors**: Yuting Liu, Jian Guan, Jia-Nan Li, Wei Wu, Jiang-Ming Yang, Jianzhe Zhao, Guibing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.04963)  

**Abstract**: We study the problem of personalization in large language models (LLMs). Prior work predominantly represents user preferences as implicit, model-specific vectors or parameters, yielding opaque ``black-box'' profiles that are difficult to interpret and transfer across models and tasks. In contrast, we advocate natural language as a universal, model- and task-agnostic interface for preference representation. The formulation leads to interpretable and reusable preference descriptions, while naturally supporting continual evolution as new interactions are observed. To learn such representations, we introduce a two-stage training framework that combines supervised fine-tuning on high-quality synthesized data with reinforcement learning to optimize long-term utility and cross-task transferability. Based on this framework, we develop AlignXplore+, a universal preference reasoning model that generates textual preference summaries. Experiments on nine benchmarks show that our 8B model achieves state-of-the-art performanc -- outperforming substantially larger open-source models -- while exhibiting strong transferability across tasks, model families, and interaction formats. 

---
# A Unified Spoken Language Model with Injected Emotional-Attribution Thinking for Human-like Interaction 

**Authors**: Qing Wang, Zehan Li, Yaodong Song, Hongjie Chen, Jian Kang, Jie Lian, Jie Li, Yongxiang Li, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04960)  

**Abstract**: This paper presents a unified spoken language model for emotional intelligence, enhanced by a novel data construction strategy termed Injected Emotional-Attribution Thinking (IEAT). IEAT incorporates user emotional states and their underlying causes into the model's internal reasoning process, enabling emotion-aware reasoning to be internalized rather than treated as explicit supervision. The model is trained with a two-stage progressive strategy. The first stage performs speech-text alignment and emotional attribute modeling via self-distillation, while the second stage conducts end-to-end cross-modal joint optimization to ensure consistency between textual and spoken emotional expressions. Experiments on the Human-like Spoken Dialogue Systems Challenge (HumDial) Emotional Intelligence benchmark demonstrate that the proposed approach achieves top-ranked performance across emotional trajectory modeling, emotional reasoning, and empathetic response generation under both LLM-based and human evaluations. 

---
# GenProve: Learning to Generate Text with Fine-Grained Provenance 

**Authors**: Jingxuan Wei, Xingyue Wang, Yanghaoyu Liao, Jie Dong, Yuchen Liu, Caijun Jia, Bihui Yu, Junnan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04932)  

**Abstract**: Large language models (LLM) often hallucinate, and while adding citations is a common solution, it is frequently insufficient for accountability as users struggle to verify how a cited source supports a generated claim. Existing methods are typically coarse-grained and fail to distinguish between direct quotes and complex reasoning. In this paper, we introduce Generation-time Fine-grained Provenance, a task where models must generate fluent answers while simultaneously producing structured, sentence-level provenance triples. To enable this, we present ReFInE (Relation-aware Fine-grained Interpretability & Evidence), a dataset featuring expert verified annotations that distinguish between Quotation, Compression, and Inference. Building on ReFInE, we propose GenProve, a framework that combines Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO). By optimizing a composite reward for answer fidelity and provenance correctness, GenProve significantly outperforms 14 strong LLMs in joint evaluation. Crucially, our analysis uncovers a reasoning gap where models excel at surface-level quotation but struggle significantly with inference-based provenance, suggesting that verifiable reasoning remains a frontier challenge distinct from surface-level citation. 

---
# Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences 

**Authors**: Arkadiusz Modzelewski, Paweł Golik, Anna Kołos, Giovanni Da San Martino  

**Link**: [PDF](https://arxiv.org/pdf/2601.04925)  

**Abstract**: Large Language Models (LLMs) can generate highly persuasive text, raising concerns about their misuse for propaganda, manipulation, and other harmful purposes. This leads us to our central question: Is LLM-generated persuasion more difficult to automatically detect than human-written persuasion? To address this, we categorize controllable generation approaches for producing persuasive content with LLMs and introduce Persuaficial, a high-quality multilingual benchmark covering six languages: English, German, Polish, Italian, French and Russian. Using this benchmark, we conduct extensive empirical evaluations comparing human-authored and LLM-generated persuasive texts. We find that although overtly persuasive LLM-generated texts can be easier to detect than human-written ones, subtle LLM-generated persuasion consistently degrades automatic detection performance. Beyond detection performance, we provide the first comprehensive linguistic analysis contrasting human and LLM-generated persuasive texts, offering insights that may guide the development of more interpretable and robust detection tools. 

---
# V-FAT: Benchmarking Visual Fidelity Against Text-bias 

**Authors**: Ziteng Wang, Yujie He, Guanliang Li, Siqi Yang, Jiaqi Xiong, Songxiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04897)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated impressive performance on standard visual reasoning benchmarks. However, there is growing concern that these models rely excessively on linguistic shortcuts rather than genuine visual grounding, a phenomenon we term Text Bias. In this paper, we investigate the fundamental tension between visual perception and linguistic priors. We decouple the sources of this bias into two dimensions: Internal Corpus Bias, stemming from statistical correlations in pretraining, and External Instruction Bias, arising from the alignment-induced tendency toward sycophancy. To quantify this effect, we introduce V-FAT (Visual Fidelity Against Text-bias), a diagnostic benchmark comprising 4,026 VQA instances across six semantic domains. V-FAT employs a Three-Level Evaluation Framework that systematically increases the conflict between visual evidence and textual information: (L1) internal bias from atypical images, (L2) external bias from misleading instructions, and (L3) synergistic bias where both coincide. We introduce the Visual Robustness Score (VRS), a metric designed to penalize "lucky" linguistic guesses and reward true visual fidelity. Our evaluation of 12 frontier MLLMs reveals that while models excel in existing benchmarks, they experience significant visual collapse under high linguistic dominance. 

---
# Faithful Summarisation under Disagreement via Belief-Level Aggregation 

**Authors**: Favour Yahdii Aghaebe, Tanefa Apekey, Elizabeth Williams, Nafise Sadat Moosavi  

**Link**: [PDF](https://arxiv.org/pdf/2601.04889)  

**Abstract**: Opinion and multi-document summarisation often involve genuinely conflicting viewpoints, yet many existing approaches, particularly LLM-based systems, implicitly smooth disagreement and over-represent majority opinions. This limits the faithfulness of generated summaries in opinion-heavy settings. We introduce a disagreement-aware synthesis pipeline that separates belief-level aggregation from language generation. Documents are first represented as structured belief sets and aggregated using distance-based belief merging operators that explicitly model conflict. Large language models are then used only to realise the aggregated beliefs as natural language summaries. We evaluate the approach across multiple model families and scales, comparing it to methods that perform explicit aggregation during generation. Our results show that while sufficiently large models can match belief-level aggregation when aggregation is handled at generation time, this behaviour is not stable across architectures or capacities. In contrast, belief-level aggregation combined with simple prompting yields consistently strong disagreement-aware performance across models, while maintaining fluent and grounded summaries. 

---
# CuMA: Aligning LLMs with Sparse Cultural Values via Demographic-Aware Mixture of Adapters 

**Authors**: Ao Sun, Xiaoyu Wang, Zhe Tan, Yu Li, Jiachen Zhu, Shu Su, Yuheng Jia  

**Link**: [PDF](https://arxiv.org/pdf/2601.04885)  

**Abstract**: As Large Language Models (LLMs) serve a global audience, alignment must transition from enforcing universal consensus to respecting cultural pluralism. We demonstrate that dense models, when forced to fit conflicting value distributions, suffer from \textbf{Mean Collapse}, converging to a generic average that fails to represent diverse groups. We attribute this to \textbf{Cultural Sparsity}, where gradient interference prevents dense parameters from spanning distinct cultural modes. To resolve this, we propose \textbf{\textsc{CuMA}} (\textbf{Cu}ltural \textbf{M}ixture of \textbf{A}dapters), a framework that frames alignment as a \textbf{conditional capacity separation} problem. By incorporating demographic-aware routing, \textsc{CuMA} internalizes a \textit{Latent Cultural Topology} to explicitly disentangle conflicting gradients into specialized expert subspaces. Extensive evaluations on WorldValuesBench, Community Alignment, and PRISM demonstrate that \textsc{CuMA} achieves state-of-the-art performance, significantly outperforming both dense baselines and semantic-only MoEs. Crucially, our analysis confirms that \textsc{CuMA} effectively mitigates mean collapse, preserving cultural diversity. Our code is available at this https URL. 

---
# Mind2Report: A Cognitive Deep Research Agent for Expert-Level Commercial Report Synthesis 

**Authors**: Mingyue Cheng, Daoyu Wang, Qi Liu, Shuo Yu, Xiaoyu Tao, Yuqian Wang, Chengzhong Chu, Yu Duan, Mingkang Long, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04879)  

**Abstract**: Synthesizing informative commercial reports from massive and noisy web sources is critical for high-stakes business decisions. Although current deep research agents achieve notable progress, their reports still remain limited in terms of quality, reliability, and coverage. In this work, we propose Mind2Report, a cognitive deep research agent that emulates the commercial analyst to synthesize expert-level reports. Specifically, it first probes fine-grained intent, then searches web sources and records distilled information on the fly, and subsequently iteratively synthesizes the report. We design Mind2Report as a training-free agentic workflow that augments general large language models (LLMs) with dynamic memory to support these long-form cognitive processes. To rigorously evaluate Mind2Report, we further construct QRC-Eval comprising 200 real-world commercial tasks and establish a holistic evaluation strategy to assess report quality, reliability, and coverage. Experiments demonstrate that Mind2Report outperforms leading baselines, including OpenAI and Gemini deep research agents. Although this is a preliminary study, we expect it to serve as a foundation for advancing the future design of commercial deep research agents. Our code and data are available at this https URL. 

---
# EvolSQL: Structure-Aware Evolution for Scalable Text-to-SQL Data Synthesis 

**Authors**: Xuanguang Pan, Chongyang Tao, Jiayuan Bai, Jianling Gao, Zhengwei Tao, Xiansheng Zhou, Gavin Cheung, Shuai Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.04875)  

**Abstract**: Training effective Text-to-SQL models remains challenging due to the scarcity of high-quality, diverse, and structurally complex datasets. Existing methods either rely on limited human-annotated corpora, or synthesize datasets directly by simply prompting LLMs without explicit control over SQL structures, often resulting in limited structural diversity and complexity. To address this, we introduce EvolSQL, a structure-aware data synthesis framework that evolves SQL queries from seed data into richer and more semantically diverse forms. EvolSQL starts with an exploratory Query-SQL expansion to broaden question diversity and improve schema coverage, and then applies an adaptive directional evolution strategy using six atomic transformation operators derived from the SQL Abstract Syntax Tree to progressively increase query complexity across relational, predicate, aggregation, and nesting dimensions. An execution-grounded SQL refinement module and schema-aware deduplication further ensure the creation of high-quality, structurally diverse mapping pairs. Experimental results show that a 7B model fine-tuned on our data outperforms one trained on the much larger SynSQL dataset using only 1/18 of the data. 

---
# A Navigational Approach for Comprehensive RAG via Traversal over Proposition Graphs 

**Authors**: Maxime Delmas, Lei Xu, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2601.04859)  

**Abstract**: Standard RAG pipelines based on chunking excel at simple factual retrieval but fail on complex multi-hop queries due to a lack of structural connectivity. Conversely, initial strategies that interleave retrieval with reasoning often lack global corpus awareness, while Knowledge Graph (KG)-based RAG performs strongly on complex multi-hop tasks but suffers on fact-oriented single-hop queries. To bridge this gap, we propose a novel RAG framework: ToPG (Traversal over Proposition Graphs). ToPG models its knowledge base as a heterogeneous graph of propositions, entities, and passages, effectively combining the granular fact density of propositions with graph connectivity. We leverage this structure using iterative Suggestion-Selection cycles, where the Suggestion phase enables a query-aware traversal of the graph, and the Selection phase provides LLM feedback to prune irrelevant propositions and seed the next iteration. Evaluated on three distinct QA tasks (Simple, Complex, and Abstract QA), ToPG demonstrates strong performance across both accuracy- and quality-based metrics. Overall, ToPG shows that query-aware graph traversal combined with factual granularity is a critical component for efficient structured RAG systems. ToPG is available at this https URL. 

---
# MisSpans: Fine-Grained False Span Identification in Cross-Domain Fake News 

**Authors**: Zhiwei Liu, Paul Thompson, Jiaqi Rong, Baojie Qu, Runteng Guo, Min Peng, Qianqian Xie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2601.04857)  

**Abstract**: Online misinformation is increasingly pervasive, yet most existing benchmarks and methods evaluate veracity at the level of whole claims or paragraphs using coarse binary labels, obscuring how true and false details often co-exist within single sentences. These simplifications also limit interpretability: global explanations cannot identify which specific segments are misleading or differentiate how a detail is false (e.g., distorted vs. fabricated). To address these gaps, we introduce MisSpans, the first multi-domain, human-annotated benchmark for span-level misinformation detection and analysis, consisting of paired real and fake news stories. MisSpans defines three complementary tasks: MisSpansIdentity for pinpointing false spans within sentences, MisSpansType for categorising false spans by misinformation type, and MisSpansExplanation for providing rationales grounded in identified spans. Together, these tasks enable fine-grained localisation, nuanced characterisation beyond true/false and actionable explanations. Expert annotators were guided by standardised guidelines and consistency checks, leading to high inter-annotator agreement. We evaluate 15 representative LLMs, including reasoning-enhanced and non-reasoning variants, under zero-shot and one-shot settings. Results reveal the challenging nature of fine-grained misinformation identification and analysis, and highlight the need for a deeper understanding of how performance may be influenced by multiple interacting factors, including model size and reasoning capabilities, along with domain-specific textual features. This project will be available at this https URL. 

---
# Token Maturation: Autoregressive Language Generation via Continuous Token Dynamics 

**Authors**: Oshri Naparstek  

**Link**: [PDF](https://arxiv.org/pdf/2601.04854)  

**Abstract**: Autoregressive language models are conventionally defined over discrete token sequences, committing to a specific token at every generation step. This early discretization forces uncertainty to be resolved through token-level sampling, often leading to instability, repetition, and sensitivity to decoding heuristics.
In this work, we introduce a continuous autoregressive formulation of language generation in which tokens are represented as continuous vectors that \emph{mature} over multiple update steps before being discretized. Rather than sampling tokens, the model evolves continuous token representations through a deterministic dynamical process, committing to a discrete token only when the representation has sufficiently converged. Discrete text is recovered via hard decoding, while uncertainty is maintained and resolved in the continuous space.
We show that this maturation process alone is sufficient to produce coherent and diverse text using deterministic decoding (argmax), without reliance on token-level sampling, diffusion-style denoising, or auxiliary stabilization mechanisms. Additional perturbations, such as stochastic dynamics or history smoothing, can be incorporated naturally but are not required for the model to function.
To our knowledge, this is the first autoregressive language model that generates text by evolving continuous token representations to convergence prior to discretization, enabling stable generation without token-level sampling. 

---
# RAAR: Retrieval Augmented Agentic Reasoning for Cross-Domain Misinformation Detection 

**Authors**: Zhiwei Liu, Runteng Guo, Baojie Qu, Yuechen Jiang, Min Peng, Qianqian Xie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2601.04853)  

**Abstract**: Cross-domain misinformation detection is challenging, as misinformation arises across domains with substantial differences in knowledge and discourse. Existing methods often rely on single-perspective cues and struggle to generalize to challenging or underrepresented domains, while reasoning large language models (LLMs), though effective on complex tasks, are limited to same-distribution data. To address these gaps, we introduce RAAR, the first retrieval-augmented agentic reasoning framework for cross-domain misinformation detection. To enable cross-domain transfer beyond same-distribution assumptions, RAAR retrieves multi-perspective source-domain evidence aligned with each target sample's semantics, sentiment, and writing style. To overcome single-perspective modeling and missing systematic reasoning, RAAR constructs verifiable multi-step reasoning paths through specialized multi-agent collaboration, where perspective-specific agents produce complementary analyses and a summary agent integrates them under verifier guidance. RAAR further applies supervised fine-tuning and reinforcement learning to train a single multi-task verifier to enhance verification and reasoning capabilities. Based on RAAR, we trained the RAAR-8b and RAAR-14b models. Evaluation on three cross-domain misinformation detection tasks shows that RAAR substantially enhances the capabilities of the base models and outperforms other cross-domain methods, advanced LLMs, and LLM-based adaptation approaches. The project will be released at this https URL. 

---
# When AI Settles Down: Late-Stage Stability as a Signature of AI-Generated Text Detection 

**Authors**: Ke Sun, Guangsheng Bao, Han Cui, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04833)  

**Abstract**: Zero-shot detection methods for AI-generated text typically aggregate token-level statistics across entire sequences, overlooking the temporal dynamics inherent to autoregressive generation. We analyze over 120k text samples and reveal Late-Stage Volatility Decay: AI-generated text exhibits rapidly stabilizing log probability fluctuations as generation progresses, while human writing maintains higher variability throughout. This divergence peaks in the second half of sequences, where AI-generated text shows 24--32\% lower volatility. Based on this finding, we propose two simple features: Derivative Dispersion and Local Volatility, which computed exclusively from late-stage statistics. Without perturbation sampling or additional model access, our method achieves state-of-the-art performance on EvoBench and MAGE benchmarks and demonstrates strong complementarity with existing global methods. 

---
# Belief in Authority: Impact of Authority in Multi-Agent Evaluation Framework 

**Authors**: Junhyuk Choi, Jeongyoun Kwon, Heeju Kim, Haeun Cho, Hayeong Jung, Sehee Min, Bugeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.04790)  

**Abstract**: Multi-agent systems utilizing large language models often assign authoritative roles to improve performance, yet the impact of authority bias on agent interactions remains underexplored. We present the first systematic analysis of role-based authority bias in free-form multi-agent evaluation using ChatEval. Applying French and Raven's power-based theory, we classify authoritative roles into legitimate, referent, and expert types and analyze their influence across 12-turn conversations. Experiments with GPT-4o and DeepSeek R1 reveal that Expert and Referent power roles exert stronger influence than Legitimate power roles. Crucially, authority bias emerges not through active conformity by general agents, but through authoritative roles consistently maintaining their positions while general agents demonstrate flexibility. Furthermore, authority influence requires clear position statements, as neutral responses fail to generate bias. These findings provide key insights for designing multi-agent frameworks with asymmetric interaction patterns. 

---
# NC2C: Automated Convexification of Generic Non-Convex Optimization Problems 

**Authors**: Xinyue Peng, Yanming Liu, Yihan Cang, Yuwei Zhang, Xinyi Wang, Songhang Deng, Jiannan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2601.04789)  

**Abstract**: Non-convex optimization problems are pervasive across mathematical programming, engineering design, and scientific computing, often posing intractable challenges for traditional solvers due to their complex objective functions and constrained landscapes. To address the inefficiency of manual convexification and the over-reliance on expert knowledge, we propose NC2C, an LLM-based end-to-end automated framework designed to transform generic non-convex optimization problems into solvable convex forms using large language models. NC2C leverages LLMs' mathematical reasoning capabilities to autonomously detect non-convex components, select optimal convexification strategies, and generate rigorous convex equivalents. The framework integrates symbolic reasoning, adaptive transformation techniques, and iterative validation, equipped with error correction loops and feasibility domain correction mechanisms to ensure the robustness and validity of transformed problems. Experimental results on a diverse dataset of 100 generic non-convex problems demonstrate that NC2C achieves an 89.3\% execution rate and a 76\% success rate in producing feasible, high-quality convex transformations. This outperforms baseline methods by a significant margin, highlighting NC2C's ability to leverage LLMs for automated non-convex to convex transformation, reduce expert dependency, and enable efficient deployment of convex solvers for previously intractable optimization tasks. 

---
# LANGSAE EDITING: Improving Multilingual Information Retrieval via Post-hoc Language Identity Removal 

**Authors**: Dongjun Kim, Jeongho Yoon, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2601.04768)  

**Abstract**: Dense retrieval in multilingual settings often searches over mixed-language collections, yet multilingual embeddings encode language identity alongside semantics. This language signal can inflate similarity for same-language pairs and crowd out relevant evidence written in other languages. We propose LANGSAE EDITING, a post-hoc sparse autoencoder trained on pooled embeddings that enables controllable removal of language-identity signal directly in vector space. The method identifies language-associated latent units using cross-language activation statistics, suppresses these units at inference time, and reconstructs embeddings in the original dimensionality, making it compatible with existing vector databases without retraining the base encoder or re-encoding raw text. Experiments across multiple languages show consistent improvements in ranking quality and cross-language coverage, with especially strong gains for script-distinct languages. 

---
# Revisiting Judge Decoding from First Principles via Training-Free Distributional Divergence 

**Authors**: Shengyin Sun, Yiming Li, Renxi Liu, Weizhe Lin, Hui-Ling Zhen, Xianzhi Yu, Mingxuan Yuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.04766)  

**Abstract**: Judge Decoding accelerates LLM inference by relaxing the strict verification of Speculative Decoding, yet it typically relies on expensive and noisy supervision. In this work, we revisit this paradigm from first principles, revealing that the ``criticality'' scores learned via costly supervision are intrinsically encoded in the draft-target distributional divergence. We theoretically prove a structural correspondence between learned linear judges and Kullback-Leibler (KL) divergence, demonstrating they rely on the same underlying logit primitives. Guided by this, we propose a simple, training-free verification mechanism based on KL divergence. Extensive experiments across reasoning and coding benchmarks show that our method matches or outperforms complex trained judges (e.g., AutoJudge), offering superior robustness to domain shifts and eliminating the supervision bottleneck entirely. 

---
# Differential syntactic and semantic encoding in LLMs 

**Authors**: Santiago Acevedo, Alessandro Laio, Marco Baroni  

**Link**: [PDF](https://arxiv.org/pdf/2601.04765)  

**Abstract**: We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations. 

---
# PILOT-Bench: A Benchmark for Legal Reasoning in the Patent Domain with IRAC-Aligned Classification Tasks 

**Authors**: Yehoon Jang, Chaewon Lee, Hyun-seok Min, Sungchul Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.04758)  

**Abstract**: The Patent Trial and Appeal Board (PTAB) of the USPTO adjudicates thousands of ex parte appeals each year, requiring the integration of technical understanding and legal reasoning. While large language models (LLMs) are increasingly applied in patent and legal practice, their use has remained limited to lightweight tasks, with no established means of systematically evaluating their capacity for structured legal reasoning in the patent domain. In this work, we introduce PILOT-Bench, the first PTAB-centric benchmark that aligns PTAB decisions with USPTO patent data at the case-level and formalizes three IRAC-aligned classification tasks: Issue Type, Board Authorities, and Subdecision. We evaluate a diverse set of closed-source (commercial) and open-source LLMs and conduct analyses across multiple perspectives, including input-variation settings, model families, and error tendencies. Notably, on the Issue Type task, closed-source models consistently exceed 0.75 in Micro-F1 score, whereas the strongest open-source model (Qwen-8B) achieves performance around 0.56, highlighting a substantial gap in reasoning capabilities. PILOT-Bench establishes a foundation for the systematic evaluation of patent-domain legal reasoning and points toward future directions for improving LLMs through dataset design and model alignment. All data, code, and benchmark resources are available at this https URL. 

---
# Tool-MAD: A Multi-Agent Debate Framework for Fact Verification with Diverse Tool Augmentation and Adaptive Retrieval 

**Authors**: Seyeon Jeong, Yeonjun Choi, JongWook Kim, Beakcheol Jang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04742)  

**Abstract**: Large Language Models (LLMs) suffer from hallucinations and factual inaccuracies, especially in complex reasoning and fact verification tasks. Multi-Agent Debate (MAD) systems aim to improve answer accuracy by enabling multiple LLM agents to engage in dialogue, promoting diverse reasoning and mutual verification. However, existing MAD frameworks primarily rely on internal knowledge or static documents, making them vulnerable to hallucinations. While MADKE introduces external evidence to mitigate this, its one-time retrieval mechanism limits adaptability to new arguments or emerging information during the debate. To address these limitations, We propose Tool-MAD, a multi-agent debate framework that enhances factual verification by assigning each agent a distinct external tool, such as a search API or RAG module. Tool-MAD introduces three key innovations: (1) a multi-agent debate framework where agents leverage heterogeneous external tools, encouraging diverse perspectives, (2) an adaptive query formulation mechanism that iteratively refines evidence retrieval based on the flow of the debate, and (3) the integration of Faithfulness and Answer Relevance scores into the final decision process, allowing the Judge agent to quantitatively assess the coherence and question alignment of each response and effectively detect hallucinations. Experimental results on four fact verification benchmarks demonstrate that Tool-MAD consistently outperforms state-of-the-art MAD frameworks, achieving up to 5.5% accuracy improvement. Furthermore, in medically specialized domains, Tool-MAD exhibits strong robustness and adaptability across various tool configurations and domain conditions, confirming its potential for broader real-world fact-checking applications. 

---
# RiskAtlas: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation 

**Authors**: Huawei Zheng, Xinqi Jiang, Sen Yang, Shouling Ji, Yingcai Wu, Dazhen Deng  

**Link**: [PDF](https://arxiv.org/pdf/2601.04740)  

**Abstract**: Large language models (LLMs) are increasingly applied in specialized domains such as finance and healthcare, where they introduce unique safety risks. Domain-specific datasets of harmful prompts remain scarce and still largely rely on manual construction; public datasets mainly focus on explicit harmful prompts, which modern LLM defenses can often detect and refuse. In contrast, implicit harmful prompts-expressed through indirect domain knowledge-are harder to detect and better reflect real-world threats. We identify two challenges: transforming domain knowledge into actionable constraints and increasing the implicitness of generated harmful prompts. To address them, we propose an end-to-end framework that first performs knowledge-graph-guided harmful prompt generation to systematically produce domain-relevant prompts, and then applies dual-path obfuscation rewriting to convert explicit harmful prompts into implicit variants via direct and context-enhanced rewriting. This framework yields high-quality datasets combining strong domain relevance with implicitness, enabling more realistic red-teaming and advancing LLM safety research. We release our code and datasets at GitHub. 

---
# AM$^3$Safety: Towards Data Efficient Alignment of Multi-modal Multi-turn Safety for MLLMs 

**Authors**: Han Zhu, Jiale Chen, Chengkun Cai, Shengjie Sun, Haoran Li, Yujin Zhou, Chi-Min Chan, Pengcheng Wen, Lei Li, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.04736)  

**Abstract**: Multi-modal Large Language Models (MLLMs) are increasingly deployed in interactive applications. However, their safety vulnerabilities become pronounced in multi-turn multi-modal scenarios, where harmful intent can be gradually reconstructed across turns, and security protocols fade into oblivion as the conversation progresses. Existing Reinforcement Learning from Human Feedback (RLHF) alignment methods are largely developed for single-turn visual question-answer (VQA) task and often require costly manual preference annotations, limiting their effectiveness and scalability in dialogues. To address this challenge, we present InterSafe-V, an open-source multi-modal dialogue dataset containing 11,270 dialogues and 500 specially designed refusal VQA samples. This dataset, constructed through interaction between several models, is designed to more accurately reflect real-world scenarios and includes specialized VQA pairs tailored for specific domains. Building on this dataset, we propose AM$^3$Safety, a framework that combines a cold-start refusal phase with Group Relative Policy Optimization (GRPO) fine-tuning using turn-aware dual-objective rewards across entire dialogues. Experiments on Qwen2.5-VL-7B-Instruct and LLaVA-NeXT-7B show more than 10\% decrease in Attack Success Rate (ASR) together with an increment of at least 8\% in harmless dimension and over 13\% in helpful dimension of MLLMs on multi-modal multi-turn safety benchmarks, while preserving their general abilities. 

---
# Automatic Classifiers Underdetect Emotions Expressed by Men 

**Authors**: Ivan Smirnov, Segun T. Aroyehun, Paul Plener, David Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2601.04730)  

**Abstract**: The widespread adoption of automatic sentiment and emotion classifiers makes it important to ensure that these tools perform reliably across different populations. Yet their reliability is typically assessed using benchmarks that rely on third-party annotators rather than the individuals experiencing the emotions themselves, potentially concealing systematic biases. In this paper, we use a unique, large-scale dataset of more than one million self-annotated posts and a pre-registered research design to investigate gender biases in emotion detection across 414 combinations of models and emotion-related classes. We find that across different types of automatic classifiers and various underlying emotions, error rates are consistently higher for texts authored by men compared to those authored by women. We quantify how this bias could affect results in downstream applications and show that current machine learning tools, including large language models, should be applied with caution when the gender composition of a sample is not known or variable. Our findings demonstrate that sentiment analysis is not yet a solved problem, especially in ensuring equitable model behaviour across demographic groups. 

---
# Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking 

**Authors**: Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.04720)  

**Abstract**: In this report, we introduce the Qwen3-VL-Embedding and Qwen3-VL-Reranker model series, the latest extensions of the Qwen family built on the Qwen3-VL foundation model. Together, they provide an end-to-end pipeline for high-precision multimodal search by mapping diverse modalities, including text, images, document images, and video, into a unified representation space. The Qwen3-VL-Embedding model employs a multi-stage training paradigm, progressing from large-scale contrastive pre-training to reranking model distillation, to generate semantically rich high-dimensional vectors. It supports Matryoshka Representation Learning, enabling flexible embedding dimensions, and handles inputs up to 32k tokens. Complementing this, Qwen3-VL-Reranker performs fine-grained relevance estimation for query-document pairs using a cross-encoder architecture with cross-attention mechanisms. Both model series inherit the multilingual capabilities of Qwen3-VL, supporting more than 30 languages, and are released in $\textbf{2B}$ and $\textbf{8B}$ parameter sizes to accommodate diverse deployment requirements. Empirical evaluations demonstrate that the Qwen3-VL-Embedding series achieves state-of-the-art results across diverse multimodal embedding evaluation benchmarks. Specifically, Qwen3-VL-Embedding-8B attains an overall score of $\textbf{77.8}$ on MMEB-V2, ranking first among all models (as of January 8, 2025). This report presents the architecture, training methodology, and practical capabilities of the series, demonstrating their effectiveness on various multimodal retrieval tasks, including image-text retrieval, visual question answering, and video-text matching. 

---
# Fame Fades, Nature Remains: Disentangling the Character Identity of Role-Playing Agents 

**Authors**: Yonghyun Jun, Junhyuk Choi, Jihyeong Park, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.04716)  

**Abstract**: Despite the rapid proliferation of Role-Playing Agents (RPAs) based on Large Language Models (LLMs), the structural dimensions defining a character's identity remain weakly formalized, often treating characters as arbitrary text inputs. In this paper, we propose the concept of \textbf{Character Identity}, a multidimensional construct that disentangles a character into two distinct layers: \textbf{(1) Parametric Identity}, referring to character-specific knowledge encoded from the LLM's pre-training, and \textbf{(2) Attributive Identity}, capturing fine-grained behavioral properties such as personality traits and moral values. To systematically investigate these layers, we construct a unified character profile schema and generate both Famous and Synthetic characters under identical structural constraints. Our evaluation across single-turn and multi-turn interactions reveals two critical phenomena. First, we identify \textit{"Fame Fades"}: while famous characters hold a significant advantage in initial turns due to parametric knowledge, this edge rapidly vanishes as models prioritize accumulating conversational context over pre-trained priors. Second, we find that \textit{"Nature Remains"}: while models robustly portray general personality traits regardless of polarity, RPA performance is highly sensitive to the valence of morality and interpersonal relationships. Our findings pinpoint negative social natures as the primary bottleneck in RPA fidelity, guiding future character construction and evaluation. 

---
# DSC2025 -- ViHallu Challenge: Detecting Hallucination in Vietnamese LLMs 

**Authors**: Anh Thi-Hoang Nguyen, Khanh Quoc Tran, Tin Van Huynh, Phuoc Tan-Hoang Nguyen, Cam Tan Nguyen, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04711)  

**Abstract**: The reliability of large language models (LLMs) in production environments remains significantly constrained by their propensity to generate hallucinations--fluent, plausible-sounding outputs that contradict or fabricate information. While hallucination detection has recently emerged as a priority in English-centric benchmarks, low-to-medium resource languages such as Vietnamese remain inadequately covered by standardized evaluation frameworks. This paper introduces the DSC2025 ViHallu Challenge, the first large-scale shared task for detecting hallucinations in Vietnamese LLMs. We present the ViHallu dataset, comprising 10,000 annotated triplets of (context, prompt, response) samples systematically partitioned into three hallucination categories: no hallucination, intrinsic, and extrinsic hallucinations. The dataset incorporates three prompt types--factual, noisy, and adversarial--to stress-test model robustness. A total of 111 teams participated, with the best-performing system achieving a macro-F1 score of 84.80\%, compared to a baseline encoder-only score of 32.83\%, demonstrating that instruction-tuned LLMs with structured prompting and ensemble strategies substantially outperform generic architectures. However, the gap to perfect performance indicates that hallucination detection remains a challenging problem, particularly for intrinsic (contradiction-based) hallucinations. This work establishes a rigorous benchmark and explores a diverse range of detection methodologies, providing a foundation for future research into the trustworthiness and reliability of Vietnamese language AI systems. 

---
# Prior-Informed Zeroth-Order Optimization with Adaptive Direction Alignment for Memory-Efficient LLM Fine-Tuning 

**Authors**: Feihu Jin, Shipeng Cen, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04710)  

**Abstract**: Fine-tuning large language models (LLMs) has achieved remarkable success across various NLP tasks, but the substantial memory overhead during backpropagation remains a critical bottleneck, especially as model scales grow. Zeroth-order (ZO) optimization alleviates this issue by estimating gradients through forward passes and Gaussian sampling, avoiding the need for backpropagation. However, conventional ZO methods suffer from high variance in gradient estimation due to their reliance on random perturbations, leading to slow convergence and suboptimal performance. We propose a simple plug-and-play method that incorporates prior-informed perturbations to refine gradient estimation. Our method dynamically computes a guiding vector from Gaussian samples, which directs perturbations toward more informative directions, significantly accelerating convergence compared to standard ZO approaches. We further investigate a greedy perturbation strategy to explore the impact of prior knowledge on gradient estimation. Theoretically, we prove that our gradient estimator achieves stronger alignment with the true gradient direction, enhancing optimization efficiency. Extensive experiments across LLMs of varying scales and architectures demonstrate that our proposed method could seamlessly integrate into existing optimization methods, delivering faster convergence and superior performance. Notably, on the OPT-13B model, our method outperforms traditional ZO optimization across all 11 benchmark tasks and surpasses gradient-based baselines on 9 out of 11 tasks, establishing a robust balance between efficiency and accuracy. 

---
# PRISM: A Unified Framework for Post-Training LLMs Without Verifiable Rewards 

**Authors**: Mukesh Ghimire, Aosong Feng, Liwen You, Youzhi Luo, Fang Liu, Xuan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04700)  

**Abstract**: Current techniques for post-training Large Language Models (LLMs) rely either on costly human supervision or on external verifiers to boost performance on tasks such as mathematical reasoning and code generation. However, as LLMs improve their problem-solving, any further improvement will potentially require high-quality solutions to difficult problems that are not available to humans. As a result, learning from unlabeled data is becoming increasingly attractive in the research community. Existing methods extract learning signal from a model's consistency, either by majority voting or by converting the model's internal confidence into reward. Although internal consistency metric such as entropy or self-certainty require no human intervention, as we show in this work, these are unreliable signals for large-scale and long-term training. To address the unreliability, we propose PRISM, a unified training framework that uses a Process Reward Model (PRM) to guide learning alongside model's internal confidence in the absence of ground-truth labels. We show that effectively combining PRM with self-certainty can lead to both stable training and better test-time performance, and also keep the model's internal confidence in check. 

---
# Thunder-KoNUBench: A Corpus-Aligned Benchmark for Korean Negation Understanding 

**Authors**: Sungmok Jung, Yeonkyoung So, Joonhak Lee, Sangho Kim, Yelim Ahn, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.04693)  

**Abstract**: Although negation is known to challenge large language models (LLMs), benchmarks for evaluating negation understanding, especially in Korean, are scarce. We conduct a corpus-based analysis of Korean negation and show that LLM performance degrades under negation. We then introduce Thunder-KoNUBench, a sentence-level benchmark that reflects the empirical distribution of Korean negation phenomena. Evaluating 47 LLMs, we analyze the effects of model size and instruction tuning, and show that fine-tuning on Thunder-KoNUBench improves negation understanding and broader contextual comprehension in Korean. 

---
# See, Explain, and Intervene: A Few-Shot Multimodal Agent Framework for Hateful Meme Moderation 

**Authors**: Naquee Rizwan, Subhankar Swain, Paramananda Bhaskar, Gagan Aryan, Shehryaar Shah Khan, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2601.04692)  

**Abstract**: In this work, we examine hateful memes from three complementary angles - how to detect them, how to explain their content and how to intervene them prior to being posted - by applying a range of strategies built on top of generative AI models. To the best of our knowledge, explanation and intervention have typically been studied separately from detection, which does not reflect real-world conditions. Further, since curating large annotated datasets for meme moderation is prohibitively expensive, we propose a novel framework that leverages task-specific generative multimodal agents and the few-shot adaptability of large multimodal models to cater to different types of memes. We believe this is the first work focused on generalizable hateful meme moderation under limited data conditions, and has strong potential for deployment in real-world production scenarios. Warning: Contains potentially toxic contents. 

---
# ToolGate: Contract-Grounded and Verified Tool Execution for LLMs 

**Authors**: Yanming Liu, Xinyue Peng, Jiannan Cao, Xinyi Wang, Songhang Deng, Jintao Chen, Jianwei Yin, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04688)  

**Abstract**: Large Language Models (LLMs) augmented with external tools have demonstrated remarkable capabilities in complex reasoning tasks. However, existing frameworks rely heavily on natural language reasoning to determine when tools can be invoked and whether their results should be committed, lacking formal guarantees for logical safety and verifiability. We present \textbf{ToolGate}, a forward execution framework that provides logical safety guarantees and verifiable state evolution for LLM tool calling. ToolGate maintains an explicit symbolic state space as a typed key-value mapping representing trusted world information throughout the reasoning process. Each tool is formalized as a Hoare-style contract consisting of a precondition and a postcondition, where the precondition gates tool invocation by checking whether the current state satisfies the required conditions, and the postcondition determines whether the tool's result can be committed to update the state through runtime verification. Our approach guarantees that the symbolic state evolves only through verified tool executions, preventing invalid or hallucinated results from corrupting the world representation. Experimental validation demonstrates that ToolGate significantly improves the reliability and verifiability of tool-augmented LLM systems while maintaining competitive performance on complex multi-step reasoning tasks. This work establishes a foundation for building more trustworthy and debuggable AI systems that integrate language models with external tools. 

---
# CRANE: Causal Relevance Analysis of Language-Specific Neurons in Multilingual Large Language Models 

**Authors**: Yifan Le, Yunliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04664)  

**Abstract**: Multilingual large language models (LLMs) achieve strong performance across languages, yet how language capabilities are organized at the neuron level remains poorly understood. Prior work has identified language-related neurons mainly through activation-based heuristics, which conflate language preference with functional importance. Prior work has identified language-related neurons mainly through activation-based heuristics, which conflate language preference with functional importance. We propose CRANE, a relevance-based analysis framework that redefines language specificity in terms of functional necessity, identifying language-specific neurons through targeted neuron-level interventions. CRANE characterizes neuron specialization by their contribution to language-conditioned predictions rather than activation magnitude. Our implementation will be made publicly available. Neuron-level interventions reveal a consistent asymmetric pattern: masking neurons relevant to a target language selectively degrades performance on that language while preserving performance on other languages to a substantial extent, indicating language-selective but non-exclusive neuron specializations. Experiments on English, Chinese, and Vietnamese across multiple benchmarks, together with a dedicated relevance-based metric and base-to-chat model transfer analysis, show that CRANE isolates language-specific components more precisely than activation-based methods. 

---
# SpeechMedAssist: Efficiently and Effectively Adapting Speech Language Models for Medical Consultation 

**Authors**: Sirry Chen, Jieyi Wang, Wei Chen, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.04638)  

**Abstract**: Medical consultations are intrinsically speech-centric. However, most prior works focus on long-text-based interactions, which are cumbersome and patient-unfriendly. Recent advances in speech language models (SpeechLMs) have enabled more natural speech-based interaction, yet the scarcity of medical speech data and the inefficiency of directly fine-tuning on speech data jointly hinder the adoption of SpeechLMs in medical consultation. In this paper, we propose SpeechMedAssist, a SpeechLM natively capable of conducting speech-based multi-turn interactions with patients. By exploiting the architectural properties of SpeechLMs, we decouple the conventional one-stage training into a two-stage paradigm consisting of (1) Knowledge & Capability Injection via Text and (2) Modality Re-alignment with Limited Speech Data, thereby reducing the requirement for medical speech data to only 10k synthesized samples. To evaluate SpeechLMs for medical consultation scenarios, we design a benchmark comprising both single-turn question answering and multi-turn simulated interactions. Experimental results show that our model outperforms all baselines in both effectiveness and robustness in most evaluation settings. 

---
# MAGA-Bench: Machine-Augment-Generated Text via Alignment Detection Benchmark 

**Authors**: Anyang Song, Ying Cheng, Yiqian Xu, Rui Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.04633)  

**Abstract**: Large Language Models (LLMs) alignment is constantly evolving. Machine-Generated Text (MGT) is becoming increasingly difficult to distinguish from Human-Written Text (HWT). This has exacerbated abuse issues such as fake news and online fraud. Fine-tuned detectors' generalization ability is highly dependent on dataset quality, and simply expanding the sources of MGT is insufficient. Further augment of generation process is required. According to HC-Var's theory, enhancing the alignment of generated text can not only facilitate attacks on existing detectors to test their robustness, but also help improve the generalization ability of detectors fine-tuned on it. Therefore, we propose \textbf{M}achine-\textbf{A}ugment-\textbf{G}enerated Text via \textbf{A}lignment (MAGA). MAGA's pipeline achieves comprehensive alignment from prompt construction to reasoning process, among which \textbf{R}einforced \textbf{L}earning from \textbf{D}etectors \textbf{F}eedback (RLDF), systematically proposed by us, serves as a key component. In our experiments, the RoBERTa detector fine-tuned on MAGA training set achieved an average improvement of 4.60\% in generalization detection AUC. MAGA Dataset caused an average decrease of 8.13\% in the AUC of the selected detectors, expecting to provide indicative significance for future research on the generalization detection ability of detectors. 

---
# From National Curricula to Cultural Awareness: Constructing Open-Ended Culture-Specific Question Answering Dataset 

**Authors**: Haneul Yoo, Won Ik Cho, Geunhye Kim, Jiyoon Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.04632)  

**Abstract**: Large language models (LLMs) achieve strong performance on many tasks, but their progress remains uneven across languages and cultures, often reflecting values latent in English-centric training data. To enable practical cultural alignment, we propose a scalable approach that leverages national social studies curricula as a foundation for culture-aware supervision. We introduce CuCu, an automated multi-agent LLM framework that transforms national textbook curricula into open-ended, culture-specific question-answer pairs. Applying CuCu to the Korean national social studies curriculum, we construct KCaQA, comprising 34.1k open-ended QA pairs. Our quantitative and qualitative analyses suggest that KCaQA covers culture-specific topics and produces responses grounded in local sociocultural contexts. 

---
# Character-R1: Enhancing Role-Aware Reasoning in Role-Playing Agents via RLVR 

**Authors**: Yihong Tang, Kehai Chen, Xuefeng Bai, Benyou Wang, Zeming Liu, Haifeng Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04611)  

**Abstract**: Current role-playing agents (RPAs) are typically constructed by imitating surface-level behaviors, but this approach lacks internal cognitive consistency, often causing out-of-character errors in complex situations. To address this, we propose Character-R1, a framework designed to provide comprehensive verifiable reward signals for effective role-aware reasoning, which are missing in recent studies. Specifically, our framework comprises three core designs: (1) Cognitive Focus Reward, which enforces explicit label-based analysis of 10 character elements (e.g., worldview) to structure internal cognition; (2) Reference-Guided Reward, which utilizes overlap-based metrics with reference responses as optimization anchors to enhance exploration and performance; and (3) Character-Conditioned Reward Normalization, which adjusts reward distributions based on character categories to ensure robust optimization across heterogeneous roles. Extensive experiments demonstrate that Character-R1 significantly outperforms existing methods in knowledge, memory and others. 

---
# When More Words Say Less: Decoupling Length and Specificity in Image Description Evaluation 

**Authors**: Rhea Kapur, Robert Hawkins, Elisa Kreiss  

**Link**: [PDF](https://arxiv.org/pdf/2601.04609)  

**Abstract**: Vision-language models (VLMs) are increasingly used to make visual content accessible via text-based descriptions. In current systems, however, description specificity is often conflated with their length. We argue that these two concepts must be disentangled: descriptions can be concise yet dense with information, or lengthy yet vacuous. We define specificity relative to a contrast set, where a description is more specific to the extent that it picks out the target image better than other possible images. We construct a dataset that controls for length while varying information content, and validate that people reliably prefer more specific descriptions regardless of length. We find that controlling for length alone cannot account for differences in specificity: how the length budget is allocated makes a difference. These results support evaluation approaches that directly prioritize specificity over verbosity. 

---
# On the Limitations of Rank-One Model Editing in Answering Multi-hop Questions 

**Authors**: Zhiyuan He, Binghan Chen, Tianxiang Xiong, Ziyang Sun, Mozhao Zhu, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04600)  

**Abstract**: Recent advances in Knowledge Editing (KE), particularly Rank-One Model Editing (ROME), show superior efficiency over fine-tuning and in-context learning for updating single-hop facts in transformers. However, these methods face significant challenges when applied to multi-hop reasoning tasks requiring knowledge chaining. In this work, we study the effect of editing knowledge with ROME on different layer depths and identify three key failure modes. First, the "hopping-too-late" problem occurs as later layers lack access to necessary intermediate representations. Second, generalization ability deteriorates sharply when editing later layers. Third, the model overfits to edited knowledge, incorrectly prioritizing edited-hop answers regardless of context. To mitigate the issues of "hopping-too-late" and generalisation decay, we propose Redundant Editing, a simple yet effective strategy that enhances multi-hop reasoning. Our experiments demonstrate that this approach can improve accuracy on 2-hop questions by at least 15.5 percentage points, representing a 96% increase over the previous single-edit strategy, while trading off some specificity and language naturalness. 

---
# THaLLE-ThaiLLM: Domain-Specialized Small LLMs for Finance and Thai -- Technical Report 

**Authors**: KBTG Labs, Anuruth Lertpiya, Danupat Khamnuansin, Kantapong Sucharitpongpan, Pornchanan Balee, Tawunrat Chalothorn, Thadpong Pongthawornkamol, Monchai Lertsutthiwong  

**Link**: [PDF](https://arxiv.org/pdf/2601.04597)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential across various domains, particularly in banking and finance, where they can automate complex tasks and enhance decision-making at scale. Due to privacy, security, and regulatory concerns, organizations often prefer on-premise deployment of LLMs. The ThaiLLM initiative aims to enhance Thai language capabilities in open-LLMs, enabling Thai industry to leverage advanced language models. However, organizations often face a trade-off between deploying multiple specialized models versus the prohibitive expense of training a single multi-capability model. To address this, we explore model merging as a resource-efficient alternative for developing high-performance, multi-capability LLMs. We present results from two key experiments: first, merging Qwen-8B with ThaiLLM-8B demonstrates how ThaiLLM-8B enhances Thai general capabilities, showing an uplift of M3 and M6 O-NET exams over the general instruction-following Qwen-8B. Second, we merge Qwen-8B with both ThaiLLM-8B and THaLLE-CFA-8B. This combination results in further improvements in performance across both general and financial domains, by demonstrating an uplift in both M3 and M6 O-NET, Flare-CFA, and Thai-IC benchmarks. The report showcases the viability of model merging for efficiently creating multi-capability LLMs. 

---
# Aligning Text, Code, and Vision: A Multi-Objective Reinforcement Learning Framework for Text-to-Visualization 

**Authors**: Mizanur Rahman, Mohammed Saidul Islam, Md Tahmid Rahman Laskar, Shafiq Joty, Enamul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2601.04582)  

**Abstract**: Text-to-Visualization (Text2Vis) systems translate natural language queries over tabular data into concise answers and executable visualizations. While closed-source LLMs generate functional code, the resulting charts often lack semantic alignment and clarity, qualities that can only be assessed post-execution. Open-source models struggle even more, frequently producing non-executable or visually poor outputs. Although supervised fine-tuning can improve code executability, it fails to enhance overall visualization quality, as traditional SFT loss cannot capture post-execution feedback. To address this gap, we propose RL-Text2Vis, the first reinforcement learning framework for Text2Vis generation. Built on Group Relative Policy Optimization (GRPO), our method uses a novel multi-objective reward that jointly optimizes textual accuracy, code validity, and visualization quality using post-execution feedback. By training Qwen2.5 models (7B and 14B), RL-Text2Vis achieves a 22% relative improvement in chart quality over GPT-4o on the Text2Vis benchmark and boosts code execution success from 78% to 97% relative to its zero-shot baseline. Our models significantly outperform strong zero-shot and supervised baselines and also demonstrate robust generalization to out-of-domain datasets like VIS-Eval and NVBench. These results establish GRPO as an effective strategy for structured, multimodal reasoning in visualization generation. We release our code at this https URL. 

---
# FeedEval: Pedagogically Aligned Evaluation of LLM-Generated Essay Feedback 

**Authors**: Seongyeub Chu, Jongwoo Kim, Munyong Yi  

**Link**: [PDF](https://arxiv.org/pdf/2601.04574)  

**Abstract**: Going beyond the prediction of numerical scores, recent research in automated essay scoring has increasingly emphasized the generation of high-quality feedback that provides justification and actionable guidance. To mitigate the high cost of expert annotation, prior work has commonly relied on LLM-generated feedback to train essay assessment models. However, such feedback is often incorporated without explicit quality validation, resulting in the propagation of noise in downstream applications. To address this limitation, we propose FeedEval, an LLM-based framework for evaluating LLM-generated essay feedback along three pedagogically grounded dimensions: specificity, helpfulness, and validity. FeedEval employs dimension-specialized LLM evaluators trained on datasets curated in this study to assess multiple feedback candidates and select high-quality feedback for downstream use. Experiments on the ASAP++ benchmark show that FeedEval closely aligns with human expert judgments and that essay scoring models trained with FeedEval-filtered high-quality feedback achieve superior scoring performance. Furthermore, revision experiments using small LLMs show that the high-quality feedback identified by FeedEval leads to more effective essay revisions. We will release our code and curated datasets upon accepted. 

---
# Identifying Good and Bad Neurons for Task-Level Controllable LLMs 

**Authors**: Wenjie Li, Guansong Pang, Hezhe Qiao, Debin Gao, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2601.04548)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities on multiple-choice question answering benchmarks, but the complex mechanisms underlying their large-scale neurons remain opaque, posing significant challenges for understanding and steering LLMs. While recent studies made progress on identifying responsible neurons for certain abilities, these ability-specific methods are infeasible for task-focused scenarios requiring coordinated use of multiple abilities. Moreover, these approaches focus only on supportive neurons that correlate positively with task completion, while neglecting neurons with other roles-such as inhibitive roles-and misled neuron attribution due to fortuitous behaviors in LLMs (i.e., correctly answer the questions by chance rather than genuine understanding). To address these challenges, we propose NeuronLLM, a novel task-level LLM understanding framework that adopts the biological principle of functional antagonism for LLM neuron identification. The key insight is that task performance is jointly determined by neurons with two opposing roles: good neurons that facilitate task completion and bad neurons that inhibit it. NeuronLLM achieves a holistic modeling of neurons via contrastive learning of good and bad neurons, while leveraging augmented question sets to mitigate the fortuitous behaviors in LLMs. Comprehensive experiments on LLMs of different sizes and families show the superiority of NeuronLLM over existing methods in four NLP tasks, providing new insights into LLM functional organization. 

---
# BanglaLorica: Design and Evaluation of a Robust Watermarking Algorithm for Large Language Models in Bangla Text Generation 

**Authors**: Amit Bin Tariqul, A N M Zahid Hossain Milkan, Sahab-Al-Chowdhury, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04534)  

**Abstract**: As large language models (LLMs) are increasingly deployed for text generation, watermarking has become essential for authorship attribution, intellectual property protection, and misuse detection. While existing watermarking methods perform well in high-resource languages, their robustness in low-resource languages remains underexplored. This work presents the first systematic evaluation of state-of-the-art text watermarking methods: KGW, Exponential Sampling (EXP), and Waterfall, for Bangla LLM text generation under cross-lingual round-trip translation (RTT) attacks. Under benign conditions, KGW and EXP achieve high detection accuracy (>88%) with negligible perplexity and ROUGE degradation. However, RTT causes detection accuracy to collapse below RTT causes detection accuracy to collapse to 9-13%, indicating a fundamental failure of token-level watermarking. To address this, we propose a layered watermarking strategy that combines embedding-time and post-generation watermarks. Experimental results show that layered watermarking improves post-RTT detection accuracy by 25-35%, achieving 40-50% accuracy, representing a 3$\times$ to 4$\times$ relative improvement over single-layer methods, at the cost of controlled semantic degradation. Our findings quantify the robustness-quality trade-off in multilingual watermarking and establish layered watermarking as a practical, training-free solution for low-resource languages such as Bangla. Our code and data will be made public. 

---
# GRACE: Reinforcement Learning for Grounded Response and Abstention under Contextual Evidence 

**Authors**: Yibo Zhao, Jiapeng Zhu, Zichen Ding, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04525)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates external knowledge to enhance Large Language Models (LLMs), yet systems remain susceptible to two critical flaws: providing correct answers without explicit grounded evidence and producing fabricated responses when the retrieved context is insufficient. While prior research has addressed these issues independently, a unified framework that integrates evidence-based grounding and reliable abstention is currently lacking. In this paper, we propose GRACE, a reinforcement-learning framework that simultaneously mitigates both types of flaws. GRACE employs a data construction method that utilizes heterogeneous retrievers to generate diverse training samples without manual annotation. A multi-stage gated reward function is then employed to train the model to assess evidence sufficiency, extract key supporting evidence, and provide answers or explicitly abstain. Experimental results on two benchmarks demonstrate that GRACE achieves state-of-the-art overall accuracy and strikes a favorable balance between accurate response and rejection, while requiring only 10% of the annotation costs of prior methods. Our code is available at this https URL.. 

---
# LinguaGame: A Linguistically Grounded Game-Theoretic Paradigm for Multi-Agent Dialogue Generation 

**Authors**: Yuxiao Ye, Yiming Zhang, Yiran Ma, Huiyuan Xie, Huining Zhu, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04516)  

**Abstract**: Large Language Models (LLMs) have enabled Multi-Agent Systems (MASs) where agents interact through natural language to solve complex tasks or simulate multi-party dialogues. Recent work on LLM-based MASs has mainly focused on architecture design, such as role assignment and workflow orchestration. In contrast, this paper targets the interaction process itself, aiming to improve agents' communication efficiency by helping them convey their intended meaning more effectively through language. To this end, we propose LinguaGame, a linguistically-grounded game-theoretic paradigm for multi-agent dialogue generation. Our approach models dialogue as a signalling game over communicative intents and strategies, solved with a training-free equilibrium approximation algorithm for inference-time decision adjustment. Unlike prior game-theoretic MASs, whose game designs are often tightly coupled with task-specific objectives, our framework relies on linguistically informed reasoning with minimal task-specific coupling. Specifically, it treats dialogue as intentional and strategic communication, requiring agents to infer what others aim to achieve (intents) and how they pursue those goals (strategies). We evaluate our framework in simulated courtroom proceedings and debates, with human expert assessments showing significant gains in communication efficiency. 

---
# WESR: Scaling and Evaluating Word-level Event-Speech Recognition 

**Authors**: Chenchen Yang, Kexin Huang, Liwei Fan, Qian Tu, Botian Jiang, Dong Zhang, Linqi Yin, Shimin Li, Zhaoye Fei, Qinyuan Cheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04508)  

**Abstract**: Speech conveys not only linguistic information but also rich non-verbal vocal events such as laughing and crying. While semantic transcription is well-studied, the precise localization of non-verbal events remains a critical yet under-explored challenge. Current methods suffer from insufficient task definitions with limited category coverage and ambiguous temporal granularity. They also lack standardized evaluation frameworks, hindering the development of downstream applications. To bridge this gap, we first develop a refined taxonomy of 21 vocal events, with a new categorization into discrete (standalone) versus continuous (mixed with speech) types. Based on the refined taxonomy, we introduce WESR-Bench, an expert-annotated evaluation set (900+ utterances) with a novel position-aware protocol that disentangles ASR errors from event detection, enabling precise localization measurement for both discrete and continuous events. We also build a strong baseline by constructing a 1,700+ hour corpus, and train specialized models, surpassing both open-source audio-language models and commercial APIs while preserving ASR quality. We anticipate that WESR will serve as a foundational resource for future research in modeling rich, real-world auditory scenes. 

---
# SampoNLP: A Self-Referential Toolkit for Morphological Analysis of Subword Tokenizers 

**Authors**: Iaroslav Chelombitko, Ekaterina Chelombitko, Aleksey Komissarov  

**Link**: [PDF](https://arxiv.org/pdf/2601.04469)  

**Abstract**: The quality of subword tokenization is critical for Large Language Models, yet evaluating tokenizers for morphologically rich Uralic languages is hampered by the lack of clean morpheme lexicons.
We introduce SampoNLP, a corpus-free toolkit for morphological lexicon creation using MDL-inspired Self-Referential Atomicity Scoring, which filters composite forms through internal structural cues - suited for low-resource settings.
Using the high-purity lexicons generated by SampoNLP for Finnish, Hungarian, and Estonian, we conduct a systematic evaluation of BPE tokenizers across a range of vocabulary sizes (8k-256k). We propose a unified metric, the Integrated Performance Score (IPS), to navigate the trade-off between morpheme coverage and over-splitting. By analyzing the IPS curves, we identify the "elbow points" of diminishing returns and provide the first empirically grounded recommendations for optimal vocabulary sizes (k) in these languages. Our study not only offers practical guidance but also quantitatively demonstrates the limitations of standard BPE for highly agglutinative languages. The SampoNLP library and all generated resources are made publicly available: this https URL 

---
# Concept Tokens: Learning Behavioral Embeddings Through Concept Definitions 

**Authors**: Ignacio Sastre, Aiala Rosá  

**Link**: [PDF](https://arxiv.org/pdf/2601.04465)  

**Abstract**: We propose Concept Tokens, a lightweight method that adds a new special token to a pretrained LLM and learns only its embedding from multiple natural language definitions of a target concept, where occurrences of the concept are replaced by the new token. The LLM is kept frozen and the embedding is optimized with the standard language-modeling objective. We evaluate Concept Tokens in three settings. First, we study hallucinations in closed-book question answering on HotpotQA and find a directional effect: negating the hallucination token reduces hallucinated answers mainly by increasing abstentions, whereas asserting it increases hallucinations and lowers precision. Second, we induce recasting, a pedagogical feedback strategy for second language teaching, and observe the same directional effect. Moreover, compared to providing the full definitional corpus in-context, concept tokens better preserve compliance with other instructions (e.g., asking follow-up questions). Finally, we include a qualitative study with the Eiffel Tower and a fictional "Austral Tower" to illustrate what information the learned embeddings capture and where their limitations emerge. Overall, Concept Tokens provide a compact control signal learned from definitions that can steer behavior in frozen LLMs. 

---
# Beyond Static Summarization: Proactive Memory Extraction for LLM Agents 

**Authors**: Chengyuan Yang, Zequn Sun, Wei Wei, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04463)  

**Abstract**: Memory management is vital for LLM agents to handle long-term interaction and personalization. Most research focuses on how to organize and use memory summary, but often overlooks the initial memory extraction stage. In this paper, we argue that existing summary-based methods have two major limitations based on the recurrent processing theory. First, summarization is "ahead-of-time", acting as a blind "feed-forward" process that misses important details because it doesn't know future tasks. Second, extraction is usually "one-off", lacking a feedback loop to verify facts, which leads to the accumulation of information loss. To address these issues, we propose proactive memory extraction (namely ProMem). Unlike static summarization, ProMem treats extraction as an iterative cognitive process. We introduce a recurrent feedback loop where the agent uses self-questioning to actively probe the dialogue history. This mechanism allows the agent to recover missing information and correct errors. Our ProMem significantly improves the completeness of the extracted memory and QA accuracy. It also achieves a superior trade-off between extraction quality and token cost. 

---
# Users Mispredict Their Own Preferences for AI Writing Assistance 

**Authors**: Vivian Lai, Zana Buçinca, Nil-Jana Akpinar, Mo Houtti, Hyeonsu B. Kang, Kevin Chian, Namjoon Suh, Alex C. Williams  

**Link**: [PDF](https://arxiv.org/pdf/2601.04461)  

**Abstract**: Proactive AI writing assistants need to predict when users want drafting help, yet we lack empirical understanding of what drives preferences. Through a factorial vignette study with 50 participants making 750 pairwise comparisons, we find compositional effort dominates decisions ($\rho = 0.597$) while urgency shows no predictive power ($\rho \approx 0$). More critically, users exhibit a striking perception-behavior gap: they rank urgency first in self-reports despite it being the weakest behavioral driver, representing a complete preference inversion. This misalignment has measurable consequences. Systems designed from users' stated preferences achieve only 57.7\% accuracy, underperforming even naive baselines, while systems using behavioral patterns reach significantly higher 61.3\% ($p < 0.05$). These findings demonstrate that relying on user introspection for system design actively misleads optimization, with direct implications for proactive natural language generation (NLG) systems. 

---
# Merging Triggers, Breaking Backdoors: Defensive Poisoning for Instruction-Tuned Language Models 

**Authors**: San Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.04448)  

**Abstract**: Large Language Models (LLMs) have greatly advanced Natural Language Processing (NLP), particularly through instruction tuning, which enables broad task generalization without additional fine-tuning. However, their reliance on large-scale datasets-often collected from human or web sources-makes them vulnerable to backdoor attacks, where adversaries poison a small subset of data to implant hidden behaviors. Despite this growing risk, defenses for instruction-tuned models remain underexplored. We propose MB-Defense (Merging & Breaking Defense Framework), a novel training pipeline that immunizes instruction-tuned LLMs against diverse backdoor threats. MB-Defense comprises two stages: (i) defensive poisoning, which merges attacker and defensive triggers into a unified backdoor representation, and (ii) weight recovery, which breaks this representation through additional training to restore clean behavior. Extensive experiments across multiple LLMs show that MB-Defense substantially lowers attack success rates while preserving instruction-following ability. Our method offers a generalizable and data-efficient defense strategy, improving the robustness of instruction-tuned LLMs against unseen backdoor attacks. 

---
# Learning to Simulate Human Dialogue 

**Authors**: Kanishk Gandhi, Agam Bhatia, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2601.04436)  

**Abstract**: To predict what someone will say is to model how they think. We study this through next-turn dialogue prediction: given a conversation, predict the next utterance produced by a person. We compare learning approaches along two dimensions: (1) whether the model is allowed to think before responding, and (2) how learning is rewarded either through an LLM-as-a-judge that scores semantic similarity and information completeness relative to the ground-truth response, or by directly maximizing the log-probability of the true human dialogue. We find that optimizing for judge-based rewards indeed increases judge scores throughout training, however it decreases the likelihood assigned to ground truth human responses and decreases the win rate when human judges choose the most human-like response among a real and synthetic option. This failure is amplified when the model is allowed to think before answering. In contrast, by directly maximizing the log-probability of observed human responses, the model learns to better predict what people actually say, improving on both log-probability and win rate evaluations. Treating chain-of-thought as a latent variable, we derive a lower bound on the log-probability. Optimizing this objective yields the best results on all our evaluations. These results suggest that thinking helps primarily when trained with a distribution-matching objective grounded in real human dialogue, and that scaling this approach to broader conversational data may produce models with a more nuanced understanding of human behavior. 

---
# Accommodation and Epistemic Vigilance: A Pragmatic Account of Why LLMs Fail to Challenge Harmful Beliefs 

**Authors**: Myra Cheng, Robert D. Hawkins, Dan Jurafsky  

**Link**: [PDF](https://arxiv.org/pdf/2601.04435)  

**Abstract**: Large language models (LLMs) frequently fail to challenge users' harmful beliefs in domains ranging from medical advice to social reasoning. We argue that these failures can be understood and addressed pragmatically as consequences of LLMs defaulting to accommodating users' assumptions and exhibiting insufficient epistemic vigilance. We show that social and linguistic factors known to influence accommodation in humans (at-issueness, linguistic encoding, and source reliability) similarly affect accommodation in LLMs, explaining performance differences across three safety benchmarks that test models' ability to challenge harmful beliefs, spanning misinformation (Cancer-Myth, SAGE-Eval) and sycophancy (ELEPHANT). We further show that simple pragmatic interventions, such as adding the phrase "wait a minute", significantly improve performance on these benchmarks while preserving low false-positive rates. Our results highlight the importance of considering pragmatics for evaluating LLM behavior and improving LLM safety. 

---
# Gavel: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization 

**Authors**: Yao Dou, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04424)  

**Abstract**: Large language models (LLMs) now support contexts of up to 1M tokens, but their effectiveness on complex long-context tasks remains unclear. In this paper, we study multi-document legal case summarization, where a single case often spans many documents totaling 100K-500K tokens. We introduce Gavel-Ref, a reference-based evaluation framework with multi-value checklist evaluation over 26 items, as well as residual fact and writing-style evaluations. Using Gavel-Ref, we go beyond the single aggregate scores reported in prior work and systematically evaluate 12 frontier LLMs on 100 legal cases ranging from 32K to 512K tokens, primarily from 2025. Our results show that even the strongest model, Gemini 2.5 Pro, achieves only around 50 of $S_{\text{Gavel-Ref}}$, highlighting the difficulty of the task. Models perform well on simple checklist items (e.g., filing date) but struggle on multi-value or rare ones such as settlements and monitor reports. As LLMs continue to improve and may surpass human-written summaries -- making human references less reliable -- we develop Gavel-Agent, an efficient and autonomous agent scaffold that equips LLMs with six tools to navigate and extract checklists directly from case documents. With Qwen3, Gavel-Agent reduces token usage by 36% while resulting in only a 7% drop in $S_{\text{checklist}}$ compared to end-to-end extraction with GPT-4.1. 

---
# Interpreting Transformers Through Attention Head Intervention 

**Authors**: Mason Kadem, Rong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.04398)  

**Abstract**: Neural networks are growing more capable on their own, but we do not understand their neural mechanisms. Understanding these mechanisms' decision-making processes, or mechanistic interpretability, enables (1) accountability and control in high-stakes domains, (2) the study of digital brains and the emergence of cognition, and (3) discovery of new knowledge when AI systems outperform humans. 

---
# ARREST: Adversarial Resilient Regulation Enhancing Safety and Truth in Large Language Models 

**Authors**: Sharanya Dasgupta, Arkaprabha Basu, Sujoy Nath, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.04394)  

**Abstract**: Human cognition, driven by complex neurochemical processes, oscillates between imagination and reality and learns to self-correct whenever such subtle drifts lead to hallucinations or unsafe associations. In recent years, LLMs have demonstrated remarkable performance in a wide range of tasks. However, they still lack human cognition to balance factuality and safety. Bearing the resemblance, we argue that both factual and safety failures in LLMs arise from a representational misalignment in their latent activation space, rather than addressing those as entirely separate alignment issues. We hypothesize that an external network, trained to understand the fluctuations, can selectively intervene in the model to regulate falsehood into truthfulness and unsafe output into safe output without fine-tuning the model parameters themselves. Reflecting the hypothesis, we propose ARREST (Adversarial Resilient Regulation Enhancing Safety and Truth), a unified framework that identifies and corrects drifted features, engaging both soft and hard refusals in addition to factual corrections. Our empirical results show that ARREST not only regulates misalignment but is also more versatile compared to the RLHF-aligned models in generating soft refusals due to adversarial training. We make our codebase available at this https URL. 

---
# MiJaBench: Revealing Minority Biases in Large Language Models via Hate Speech Jailbreaking 

**Authors**: Iago Alves Brito, Walcy Santos Rezende Rios, Julia Soares Dollis, Diogo Fernandes Costa Silva, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2601.04389)  

**Abstract**: Current safety evaluations of large language models (LLMs) create a dangerous illusion of universality, aggregating "Identity Hate" into scalar scores that mask systemic vulnerabilities against specific populations. To expose this selective safety, we introduce MiJaBench, a bilingual (English and Portuguese) adversarial benchmark comprising 44,000 prompts across 16 minority groups. By generating 528,000 prompt-response pairs from 12 state-of-the-art LLMs, we curate MiJaBench-Align, revealing that safety alignment is not a generalized semantic capability but a demographic hierarchy: defense rates fluctuate by up to 33\% within the same model solely based on the target group. Crucially, we demonstrate that model scaling exacerbates these disparities, suggesting that current alignment techniques do not create principle of non-discrimination but reinforces memorized refusal boundaries only for specific groups, challenging the current scaling laws of security. We release all datasets and scripts to encourage research into granular demographic alignment at GitHub. 

---
# Disco-RAG: Discourse-Aware Retrieval-Augmented Generation 

**Authors**: Dongqi Liu, Hang Ding, Qiming Feng, Jian Li, Xurong Xie, Zhucun Xue, Chengjie Wang, Jiangning Zhang, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04377)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as an important means of enhancing the performance of large language models (LLMs) in knowledge-intensive tasks. However, most existing RAG strategies treat retrieved passages in a flat and unstructured way, which prevents the model from capturing structural cues and constrains its ability to synthesize knowledge from dispersed evidence across documents. To overcome these limitations, we propose Disco-RAG, a discourse-aware framework that explicitly injects discourse signals into the generation process. Our method constructs intra-chunk discourse trees to capture local hierarchies and builds inter-chunk rhetorical graphs to model cross-passage coherence. These structures are jointly integrated into a planning blueprint that conditions the generation. Experiments on question answering and long-document summarization benchmarks show the efficacy of our approach. Disco-RAG achieves state-of-the-art results on the benchmarks without fine-tuning. These findings underscore the important role of discourse structure in advancing RAG systems. 

---
# Dialect Matters: Cross-Lingual ASR Transfer for Low-Resource Indic Language Varieties 

**Authors**: Akriti Dhasmana, Aarohi Srivastava, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04373)  

**Abstract**: We conduct an empirical study of cross-lingual transfer using spontaneous, noisy, and code-mixed speech across a wide range of Indic dialects and language varieties. Our results indicate that although ASR performance is generally improved with reduced phylogenetic distance between languages, this factor alone does not fully explain performance in dialectal settings. Often, fine-tuning on smaller amounts of dialectal data yields performance comparable to fine-tuning on larger amounts of phylogenetically-related, high-resource standardized languages. We also present a case study on Garhwali, a low-resource Pahari language variety, and evaluate multiple contemporary ASR models. Finally, we analyze transcription errors to examine bias toward pre-training languages, providing additional insight into challenges faced by ASR systems on dialectal and non-standardized speech. 

---
# RIGOURATE: Quantifying Scientific Exaggeration with Evidence-Aligned Claim Evaluation 

**Authors**: Joseph James, Chenghao Xiao, Yucheng Li, Nafise Sadat Moosavi, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.04350)  

**Abstract**: Scientific rigour tends to be sidelined in favour of bold statements, leading authors to overstate claims beyond what their results support. We present RIGOURATE, a two-stage multimodal framework that retrieves supporting evidence from a paper's body and assigns each claim an overstatement score. The framework consists of a dataset of over 10K claim-evidence sets from ICLR and NeurIPS papers, annotated using eight LLMs, with overstatement scores calibrated using peer-review comments and validated through human evaluation. It employes a fine-tuned reranker for evidence retrieval and a fine-tuned model to predict overstatement scores with justification. Compared to strong baselines, RIGOURATE enables improved evidence retrieval and overstatement detection. Overall, our work operationalises evidential proportionality and supports clearer, more transparent scientific communication. 

---
# From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning 

**Authors**: Xiaoyu Xu, Minxin Du, Zitong Li, Zi Liang, Zhibiao Guo, Shiyu Zhang, Peizhao Hu, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04278)  

**Abstract**: Although machine unlearning is essential for removing private, harmful, or copyrighted content from LLMs, current benchmarks often fail to faithfully represent the true "forgetting scope" learned by the model. We formalize two distinct unlearning granularities, domain-level and instance-level, and propose BiForget, an automated framework for synthesizing high-quality forget sets. Unlike prior work relying on external generators, BiForget exploits the target model per se to elicit data that matches its internal knowledge distribution through seed-guided and adversarial prompting. Our experiments across diverse benchmarks show that it achieves a superior balance of relevance, diversity, and efficiency. Quantitatively, in the Harry Potter domain, it improves relevance by ${\sim}20$ and diversity by ${\sim}$0.05 while halving the total data size compared to SOTAs. Ultimately, it facilitates more robust forgetting and better utility preservation, providing a more rigorous foundation for evaluating LLM unlearning. 

---
# AnimatedLLM: Explaining LLMs with Interactive Visualizations 

**Authors**: Zdeněk Kasner, Ondřej Dušek  

**Link**: [PDF](https://arxiv.org/pdf/2601.04213)  

**Abstract**: Large language models (LLMs) are becoming central to natural language processing education, yet materials showing their mechanics are sparse. We present AnimatedLLM, an interactive web application that provides step-by-step visualizations of a Transformer language model. AnimatedLLM runs entirely in the browser, using pre-computed traces of open LLMs applied on manually curated inputs. The application is available at this https URL, both as a teaching aid and for self-educational purposes. 

---
# TrueBrief: Faithful Summarization through Small Language Models 

**Authors**: Kumud Lakara, Ruibo Shi, Fran Silavong  

**Link**: [PDF](https://arxiv.org/pdf/2601.04212)  

**Abstract**: Large language models (LLMs) have exhibited remarkable proficiency in generating high-quality text; however, their propensity for producing hallucinations poses a significant challenge for their deployment in security-critical domains. In this work, we present TrueBrief, an end-to-end framework specifically designed to enhance the faithfulness of small LLMs (SLMs) primarily for the task of text summarization through a preference-optimization paradigm. Central to our framework is a data generation module that facilitates controlled hallucination injection to generate synthetic preference data. Our work provides insights into the impact of data quality and model size on preference-based optimization, highlighting the conditions under which these methods are most effective. 

---
# Qwerty AI: Explainable Automated Age Rating and Content Safety Assessment for Russian-Language Screenplays 

**Authors**: Nikita Zmanovskii  

**Link**: [PDF](https://arxiv.org/pdf/2601.04211)  

**Abstract**: We present Qwerty AI, an end-to-end system for automated age-rating and content-safety assessment of Russian-language screenplays according to Federal Law No. 436-FZ. The system processes full-length scripts (up to 700 pages in under 2 minutes), segments them into narrative units, detects content violations across five categories (violence, sexual content, profanity, substances, frightening elements), and assigns age ratings (0+, 6+, 12+, 16+, 18+) with explainable justifications. Our implementation leverages a fine-tuned Phi-3-mini model with 4-bit quantization, achieving 80% rating accuracy and 80-95% segmentation precision (format-dependent). The system was developed under strict constraints: no external API calls, 80GB VRAM limit, and <5 minute processing time for average scripts. Deployed on Yandex Cloud with CUDA acceleration, Qwerty AI demonstrates practical applicability for production workflows. We achieved these results during the Wink hackathon (November 2025), where our solution addressed real editorial challenges in the Russian media industry. 

---
# Complexity Agnostic Recursive Decomposition of Thoughts 

**Authors**: Kaleem Ullah Qasim, Jiashu Zhang, Hafiz Saif Ur Rehman  

**Link**: [PDF](https://arxiv.org/pdf/2601.04210)  

**Abstract**: Large language models often fail on multi-step reasoning due to fixed reasoning strategies that ignore problem specific difficulty. We introduce CARD (Complexity Agnostic Recursive Decomposition), a framework that predicts problem complexity before generation and adapts decomposition accordingly. Our system comprises MRCE (Multi-dimensional Reasoning Complexity Estimator), a 0.6B Qwen model predicting 30 fine-grained features from question text and a two-stage recursive solver: (1) hierarchical decomposition into K steps based on task profile and (2) per-step thought budget allocation (1, 5-9, or 10 thoughts) via recursive MRCE profiling. Evaluated on three reasoning models (Qwen3-0.6B, DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-1.7B), CARD achieves 81.4% to 89.2% accuracy on GSM8K while reducing token cost by 1.88x to 2.40x compared to fixed decomposition baselines. On MATH-500, CARD reaches 75.1 to 86.8% accuracy using 1.71x to 5.74x fewer tokens. Our results demonstrate that preemptive complexity estimation enables both higher accuracy and significant efficiency gains. 

---
# Leveraging Language Models and RAG for Efficient Knowledge Discovery in Clinical Environments 

**Authors**: Seokhwan Ko, Donghyeon Lee, Jaewoo Chun, Hyungsoo Han, Junghwan Cho  

**Link**: [PDF](https://arxiv.org/pdf/2601.04209)  

**Abstract**: Large language models (LLMs) are increasingly recognized as valuable tools across the medical environment, supporting clinical, research, and administrative workflows. However, strict privacy and network security regulations in hospital settings require that sensitive data be processed within fully local infrastructures. Within this context, we developed and evaluated a retrieval-augmented generation (RAG) system designed to recommend research collaborators based on PubMed publications authored by members of a medical institution. The system utilizes PubMedBERT for domain-specific embedding generation and a locally deployed LLaMA3 model for generative synthesis. This study demonstrates the feasibility and utility of integrating domain-specialized encoders with lightweight LLMs to support biomedical knowledge discovery under local deployment constraints. 

---
# LLMs for Explainable Business Decision-Making: A Reinforcement Learning Fine-Tuning Approach 

**Authors**: Xiang Cheng, Wen Wang, Anindya Ghose  

**Link**: [PDF](https://arxiv.org/pdf/2601.04208)  

**Abstract**: Artificial Intelligence (AI) models increasingly drive high-stakes consumer interactions, yet their decision logic often remains opaque. Prevailing explainable AI techniques rely on post hoc numerical feature attributions, which fail to provide coherent narratives behind model decisions. Large language models (LLMs) present an opportunity to generate natural-language explanations, but three design challenges remain unresolved: explanations must be both decision-correct and faithful to the factors that drive the prediction; they should be able to serve multiple audiences without shifting the underlying decision rule; and they should be trained in a label-efficient way that does not depend on large corpora of human-scored explanations. To address these challenges, we introduce LEXMA (LLM-based EXplanations for Multi-Audience decisions), a reinforcement-learning-based fine-tuning framework that produces narrative-driven, audience-appropriate explanations. LEXMA combines reflection-augmented supervised fine-tuning with two stages of Group Relative Policy Optimization (GRPO). Specifically, it fine-tunes two separate parameter sets to improve decision correctness and satisfy stylistic requirements for different audiences, using reward signals that do not rely on human-annotated explanations. We instantiate LEXMA in the context of mortgage approval decisions. Results demonstrate that LEXMA yields significant improvements in predictive performance compared with other LLM baselines. Moreover, human evaluations show that expert-facing explanations generated by our approach are more risk-focused, and consumer-facing explanations are clearer, more actionable, and more polite. Our study contributes a cost-efficient, systematic LLM fine-tuning approach to enhance explanation quality for business decisions, offering strong potential for scalable deployment of transparent AI systems. 

---
# Ideology as a Problem: Lightweight Logit Steering for Annotator-Specific Alignment in Social Media Analysis 

**Authors**: Wei Xia, Haowen Tang, Luozheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04207)  

**Abstract**: LLMs internally organize political ideology along low-dimensional structures that are partially, but not fully aligned with human ideological space. This misalignment is systematic, model specific, and measurable. We introduce a lightweight linear probe that both quantifies the misalignment and minimally corrects the output layer. This paper introduces a simple and efficient method for aligning models with specific user opinions. Instead of retraining the model, we calculated a bias score from its internal features and directly adjusted the final output probabilities. This solution is practical and low-cost and preserves the original reasoning power of the model. 

---
# Enhancing Admission Inquiry Responses with Fine-Tuned Models and Retrieval-Augmented Generation 

**Authors**: Aram Virabyan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04206)  

**Abstract**: University admissions offices face the significant challenge of managing high volumes of inquiries efficiently while maintaining response quality, which critically impacts prospective students' perceptions. This paper addresses the issues of response time and information accuracy by proposing an AI system integrating a fine-tuned language model with Retrieval-Augmented Generation (RAG). While RAG effectively retrieves relevant information from large datasets, its performance in narrow, complex domains like university admissions can be limited without adaptation, potentially leading to contextually inadequate responses due to the intricate rules and specific details involved. To overcome this, we fine-tuned the model on a curated dataset specific to admissions processes, enhancing its ability to interpret RAG-provided data accurately and generate domain-relevant outputs. This hybrid approach leverages RAG's ability to access up-to-date information and fine-tuning's capacity to embed nuanced domain understanding. We further explored optimization strategies for the response generation logic, experimenting with settings to balance response quality and speed, aiming for consistently high-quality outputs that meet the specific requirements of admissions communications. 

---
# STDD:Spatio-Temporal Dynamics-Driven Token Refinement in Diffusion Language Models 

**Authors**: Xinhao Sun, Maoliang Li, Zihao Zheng, Jiayu Chen, Hezhao Xu, Yun Liang, Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04205)  

**Abstract**: Unlike autoregressive language models, diffusion language models (DLMs) generate text by iteratively denoising all token positions in parallel. At each timestep, the remasking strategy of a DLM selects low- priority tokens to defer their decoding, thereby improving both efficiency and output quality. However, mainstream remasking strategies rely on a single global confidence threshold, overlooking the temporal and spatial dynamics of individual tokens. Motivated by the redundant iterations and constrained parallelism introduced by fixed-threshold remasking, we propose a novel remasking approach that dynamically detects Temporal Variance and Spa- tial Deviance of each token, which reflect its convergence status and inter-token correlations. Using these signals, our method adaptively adjusts the confidence threshold for every token at every step. Empirical re- sults show that our approach significantly improves the operational efficiency of DLMs across mainstream datasets, achieving speedups of up to 8.9 times while faithfully preserving generation quality. 

---
# FronTalk: Benchmarking Front-End Development as Conversational Code Generation with Multi-Modal Feedback 

**Authors**: Xueqing Wu, Zihan Xue, Da Yin, Shuyan Zhou, Kai-Wei Chang, Nanyun Peng, Yeming Wen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04203)  

**Abstract**: We present FronTalk, a benchmark for front-end code generation that pioneers the study of a unique interaction dynamic: conversational code generation with multi-modal feedback. In front-end development, visual artifacts such as sketches, mockups and annotated creenshots are essential for conveying design intent, yet their role in multi-turn code generation remains largely unexplored. To address this gap, we focus on the front-end development task and curate FronTalk, a collection of 100 multi-turn dialogues derived from real-world websites across diverse domains such as news, finance, and art. Each turn features both a textual instruction and an equivalent visual instruction, each representing the same user intent. To comprehensively evaluate model performance, we propose a novel agent-based evaluation framework leveraging a web agent to simulate users and explore the website, and thus measuring both functional correctness and user experience. Evaluation of 20 models reveals two key challenges that are under-explored systematically in the literature: (1) a significant forgetting issue where models overwrite previously implemented features, resulting in task failures, and (2) a persistent challenge in interpreting visual feedback, especially for open-source vision-language models (VLMs). We propose a strong baseline to tackle the forgetting issue with AceCoder, a method that critiques the implementation of every past instruction using an autonomous web agent. This approach significantly reduces forgetting to nearly zero and improves the performance by up to 9.3% (56.0% to 65.3%). Overall, we aim to provide a solid foundation for future research in front-end development and the general interaction dynamics of multi-turn, multi-modal code generation. Code and data are released at this https URL 

---
# TeleTables: A Benchmark for Large Language Models in Telecom Table Interpretation 

**Authors**: Anas Ezzakri, Nicola Piovesan, Mohamed Sana, Antonio De Domenico, Fadhel Ayed, Haozhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04202)  

**Abstract**: Language Models (LLMs) are increasingly explored in the telecom industry to support engineering tasks, accelerate troubleshooting, and assist in interpreting complex technical documents. However, recent studies show that LLMs perform poorly on telecom standards, particularly 3GPP specifications. We argue that a key reason is that these standards densely include tables to present essential information, yet the LLM knowledge and interpretation ability of such tables remains largely unexamined. To address this gap, we introduce TeleTables, a benchmark designed to evaluate both the implicit knowledge LLMs have about tables in technical specifications and their explicit ability to interpret them. TeleTables is built through a novel multi-stage data generation pipeline that extracts tables from 3GPP standards and uses multimodal and reasoning-oriented LLMs to generate and validate questions. The resulting dataset, which is publicly available, comprises 500 human-verified question-answer pairs, each associated with the corresponding table in multiple formats. Our evaluation shows that, smaller models (under 10B parameters) struggle both to recall 3GPP knowledge and to interpret tables, indicating the limited exposure to telecom standards in their pretraining and the insufficient inductive biases for navigating complex technical material. Larger models, on the other hand, show stronger reasoning on table interpretation. Overall, TeleTables highlights the need for domain-specialized fine-tuning to reliably interpret and reason over telecom standards. 

---
# Collective Narrative Grounding: Community-Coordinated Data Contributions to Improve Local AI Systems 

**Authors**: Zihan Gao, Mohsin Y. K. Yousufi, Jacob Thebault-Spieker  

**Link**: [PDF](https://arxiv.org/pdf/2601.04201)  

**Abstract**: Large language model (LLM) question-answering systems often fail on community-specific queries, creating "knowledge blind spots" that marginalize local voices and reinforce epistemic injustice. We present Collective Narrative Grounding, a participatory protocol that transforms community stories into structured narrative units and integrates them into AI systems under community governance. Learning from three participatory mapping workshops with N=24 community members, we designed elicitation methods and a schema that retain narrative richness while enabling entity, time, and place extraction, validation, and provenance control. To scope the problem, we audit a county-level benchmark of 14,782 local information QA pairs, where factual gaps, cultural misunderstandings, geographic confusions, and temporal misalignments account for 76.7% of errors. On a participatory QA set derived from our workshops, a state-of-the-art LLM answered fewer than 21% of questions correctly without added context, underscoring the need for local grounding. The missing facts often appear in the collected narratives, suggesting a direct path to closing the dominant error modes for narrative items. Beyond the protocol and pilot, we articulate key design tensions, such as representation and power, governance and control, and privacy and consent, providing concrete requirements for retrieval-first, provenance-visible, locally governed QA systems. Together, our taxonomy, protocol, and participatory evaluation offer a rigorous foundation for building community-grounded AI that better answers local questions. 

---
# Attribute-Aware Controlled Product Generation with LLMs for E-commerce 

**Authors**: Virginia Negri, Víctor Martínez Gómez, Sergio A. Balanya, Subburam Rajaram  

**Link**: [PDF](https://arxiv.org/pdf/2601.04200)  

**Abstract**: Product information extraction is crucial for e-commerce services, but obtaining high-quality labeled datasets remains challenging. We present a systematic approach for generating synthetic e-commerce product data using Large Language Models (LLMs), introducing a controlled modification framework with three strategies: attribute-preserving modification, controlled negative example generation, and systematic attribute removal. Using a state-of-the-art LLM with attribute-aware prompts, we enforce store constraints while maintaining product coherence. Human evaluation of 2000 synthetic products demonstrates high effectiveness, with 99.6% rated as natural, 96.5% containing valid attribute values, and over 90% showing consistent attribute usage. On the public MAVE dataset, our synthetic data achieves 60.5% accuracy, performing on par with real training data (60.8%) and significantly improving upon the 13.4% zero-shot baseline. Hybrid configurations combining synthetic and real data further improve performance, reaching 68.8% accuracy. Our framework provides a practical solution for augmenting e-commerce datasets, particularly valuable for low-resource scenarios. 

---
# Automatic Construction of Chinese Verb Collostruction Database 

**Authors**: Xuri Tang, Daohuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04197)  

**Abstract**: This paper proposes a fully unsupervised approach to the construction of verb collostruction database for Chinese language, aimed at complementing LLMs by providing explicit and interpretable rules for application scenarios where explanation and interpretability are indispensable. The paper formally defines a verb collostruction as a projective, rooted, ordered, and directed acyclic graph and employs a series of clustering algorithms to generate collostructions for a given verb from a list of sentences retrieved from large-scale corpus. Statistical analysis demonstrates that the generated collostructions possess the design features of functional independence and graded typicality. Evaluation with verb grammatical error correction shows that the error correction algorithm based on maximum matching with collostructions achieves better performance than LLMs. 

---
# RAGVUE: A Diagnostic View for Explainable and Automated Evaluation of Retrieval-Augmented Generation 

**Authors**: Keerthana Murugaraj, Salima Lamsiyah, Martin Theobald  

**Link**: [PDF](https://arxiv.org/pdf/2601.04196)  

**Abstract**: Evaluating Retrieval-Augmented Generation (RAG) systems remains a challenging task: existing metrics often collapse heterogeneous behaviors into single scores and provide little insight into whether errors arise from retrieval,reasoning, or grounding. In this paper, we introduce RAGVUE, a diagnostic and explainable framework for automated, reference-free evaluation of RAG pipelines. RAGVUE decomposes RAG behavior into retrieval quality, answer relevance and completeness, strict claim-level faithfulness, and judge calibration. Each metric includes a structured explanation, making the evaluation process transparent. Our framework supports both manual metric selection and fully automated agentic evaluation. It also provides a Python API, CLI, and a local Streamlit interface for interactive usage. In comparative experiments, RAGVUE surfaces fine-grained failures that existing tools such as RAGAS often overlook. We showcase the full RAGVUE workflow and illustrate how it can be integrated into research pipelines and practical RAG development. The source code and detailed instructions on usage are publicly available on GitHub 

---
# MedPI: Evaluating AI Systems in Medical Patient-facing Interactions 

**Authors**: Diego Fajardo V., Oleksii Proniakin, Victoria-Elisabeth Gruber, Razvan Marinescu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04195)  

**Abstract**: We present MedPI, a high-dimensional benchmark for evaluating large language models (LLMs) in patient-clinician conversations. Unlike single-turn question-answer (QA) benchmarks, MedPI evaluates the medical dialogue across 105 dimensions comprising the medical process, treatment safety, treatment outcomes and doctor-patient communication across a granular, accreditation-aligned rubric. MedPI comprises five layers: (1) Patient Packets (synthetic EHR-like ground truth); (2) an AI Patient instantiated through an LLM with memory and affect; (3) a Task Matrix spanning encounter reasons (e.g. anxiety, pregnancy, wellness checkup) x encounter objectives (e.g. diagnosis, lifestyle advice, medication advice); (4) an Evaluation Framework with 105 dimensions on a 1-4 scale mapped to the Accreditation Council for Graduate Medical Education (ACGME) competencies; and (5) AI Judges that are calibrated, committee-based LLMs providing scores, flags, and evidence-linked rationales. We evaluate 9 flagship models -- Claude Opus 4.1, Claude Sonnet 4, MedGemma, Gemini 2.5 Pro, Llama 3.3 70b Instruct, GPT-5, GPT OSS 120b, o3, Grok-4 -- across 366 AI Patients and 7,097 conversations using a standardized "vanilla clinician" prompt. For all LLMs, we observe low performance across a variety of dimensions, in particular on differential diagnosis. Our work can help guide future use of LLMs for diagnosis and treatment recommendations. 

---
# Mechanisms of Prompt-Induced Hallucination in Vision-Language Models 

**Authors**: William Rudman, Michal Golovanevsky, Dana Arad, Yonatan Belinkov, Ritambhara Singh, Carsten Eickhoff, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2601.05201)  

**Abstract**: Large vision-language models (VLMs) are highly capable, yet often hallucinate by favoring textual prompts over visual evidence. We study this failure mode in a controlled object-counting setting, where the prompt overstates the number of objects in the image (e.g., asking a model to describe four waterlilies when only three are present). At low object counts, models often correct the overestimation, but as the number of objects increases, they increasingly conform to the prompt regardless of the discrepancy. Through mechanistic analysis of three VLMs, we identify a small set of attention heads whose ablation substantially reduces prompt-induced hallucinations (PIH) by at least 40% without additional training. Across models, PIH-heads mediate prompt copying in model-specific ways. We characterize these differences and show that PIH ablation increases correction toward visual evidence. Our findings offer insights into the internal mechanisms driving prompt-induced hallucinations, revealing model-specific differences in how these behaviors are implemented. 

---
# Observations and Remedies for Large Language Model Bias in Self-Consuming Performative Loop 

**Authors**: Yaxuan Wang, Zhongteng Cai, Yujia Bao, Xueru Zhang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05184)  

**Abstract**: The rapid advancement of large language models (LLMs) has led to growing interest in using synthetic data to train future models. However, this creates a self-consuming retraining loop, where models are trained on their own outputs and may cause performance drops and induce emerging biases. In real-world applications, previously deployed LLMs may influence the data they generate, leading to a dynamic system driven by user feedback. For example, if a model continues to underserve users from a group, less query data will be collected from this particular demographic of users. In this study, we introduce the concept of \textbf{S}elf-\textbf{C}onsuming \textbf{P}erformative \textbf{L}oop (\textbf{SCPL}) and investigate the role of synthetic data in shaping bias during these dynamic iterative training processes under controlled performative feedback. This controlled setting is motivated by the inaccessibility of real-world user preference data from dynamic production systems, and enables us to isolate and analyze feedback-driven bias evolution in a principled manner. We focus on two types of loops, including the typical retraining setting and the incremental fine-tuning setting, which is largely underexplored. Through experiments on three real-world tasks, we find that the performative loop increases preference bias and decreases disparate bias. We design a reward-based rejection sampling strategy to mitigate the bias, moving towards more trustworthy self-improving systems. 

---
# A Lightweight and Explainable Vision-Language Framework for Crop Disease Visual Question Answering 

**Authors**: Md. Zahid Hossain, Most. Sharmin Sultana Samu, Md. Rakibul Islam, Md. Siam Ansary  

**Link**: [PDF](https://arxiv.org/pdf/2601.05143)  

**Abstract**: Visual question answering for crop disease analysis requires accurate visual understanding and reliable language generation. This work presents a lightweight vision-language framework for crop and disease identification from leaf images. The proposed approach combines a Swin Transformer vision encoder with sequence-to-sequence language decoders. A two-stage training strategy is adopted to improve visual representation learning and cross-modal alignment. The model is evaluated on a large-scale crop disease dataset using classification and natural language generation metrics. Experimental results show high accuracy for both crop and disease identification. The framework also achieves strong performance on BLEU, ROUGE and BERTScore. Our proposed models outperform large-scale vision-language baselines while using significantly fewer parameters. Explainability is assessed using Grad-CAM and token-level attribution. Qualitative results demonstrate robust performance under diverse user-driven queries. These findings highlight the effectiveness of task-specific visual pretraining for crop disease visual question answering. 

---
# Token-Level LLM Collaboration via FusionRoute 

**Authors**: Nuoya Xiong, Yuhang Zhou, Hanqing Zeng, Zhaorun Chen, Furong Huang, Shuchao Bi, Lizhu Zhang, Zhuokai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.05106)  

**Abstract**: Large language models (LLMs) exhibit strengths across diverse domains. However, achieving strong performance across these domains with a single general-purpose model typically requires scaling to sizes that are prohibitively expensive to train and deploy. On the other hand, while smaller domain-specialized models are much more efficient, they struggle to generalize beyond their training distributions. To address this dilemma, we propose FusionRoute, a robust and effective token-level multi-LLM collaboration framework in which a lightweight router simultaneously (i) selects the most suitable expert at each decoding step and (ii) contributes a complementary logit that refines or corrects the selected expert's next-token distribution via logit addition. Unlike existing token-level collaboration methods that rely solely on fixed expert outputs, we provide a theoretical analysis showing that pure expert-only routing is fundamentally limited: unless strong global coverage assumptions hold, it cannot in general realize the optimal decoding policy. By augmenting expert selection with a trainable complementary generator, FusionRoute expands the effective policy class and enables recovery of optimal value functions under mild conditions. Empirically, across both Llama-3 and Gemma-2 families and diverse benchmarks spanning mathematical reasoning, code generation, and instruction following, FusionRoute outperforms both sequence- and token-level collaboration, model merging, and direct fine-tuning, while remaining competitive with domain experts on their respective tasks. 

---
# Semantically Orthogonal Framework for Citation Classification: Disentangling Intent and Content 

**Authors**: Changxu Duan, Zhiyin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05103)  

**Abstract**: Understanding the role of citations is essential for research assessment and citation-aware digital libraries. However, existing citation classification frameworks often conflate citation intent (why a work is cited) with cited content type (what part is cited), limiting their effectiveness in auto classification due to a dilemma between fine-grained type distinctions and practical classification reliability. We introduce SOFT, a Semantically Orthogonal Framework with Two dimensions that explicitly separates citation intent from cited content type, drawing inspiration from semantic role theory. We systematically re-annotate the ACL-ARC dataset using SOFT and release a cross-disciplinary test set sampled from ACT2. Evaluation with both zero-shot and fine-tuned Large Language Models demonstrates that SOFT enables higher agreement between human annotators and LLMs, and supports stronger classification performance and robust cross-domain generalization compared to ACL-ARC and SciCite annotation frameworks. These results confirm SOFT's value as a clear, reusable annotation standard, improving clarity, consistency, and generalizability for digital libraries and scholarly communication infrastructures. All code and data are publicly available on GitHub this https URL. 

---
# Multi-Disciplinary Dataset Discovery from Citation-Verified Literature Contexts 

**Authors**: Zhiyin Tan, Changxu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05099)  

**Abstract**: Identifying suitable datasets for a research question remains challenging because existing dataset search engines rely heavily on metadata quality and keyword overlap, which often fail to capture the semantic intent of scientific investigation. We introduce a literature-driven framework that discovers datasets from citation contexts in scientific papers, enabling retrieval grounded in actual research use rather than metadata availability. Our approach combines large-scale citation-context extraction, schema-guided dataset recognition with Large Language Models, and provenance-preserving entity resolution. We evaluate the system on eight survey-derived computer science queries and find that it achieves substantially higher recall than Google Dataset Search and DataCite Commons, with normalized recall ranging from an average of 47.47% to a highest value of 81.82%. Beyond recovering gold-standard datasets, the method also surfaces additional datasets not documented in the surveys. Expert assessments across five top-level Fields of Science indicate that a substantial portion of the additional datasets are considered high utility, and some are regarded as novel for the specific topics chosen by the experts. These findings establish citation-context mining as an effective and generalizable paradigm for dataset discovery, particularly in settings where datasets lack sufficient or reliable metadata. To support reproducibility and future extensions, we release our code, evaluation datasets, and results on GitHub (this https URL). 

---
# Reinforced Efficient Reasoning via Semantically Diverse Exploration 

**Authors**: Ziqi Zhao, Zhaochun Ren, Jiahong Zou, Liu Yang, Zhiwei Xu, Xuri Ge, Zhumin Chen, Xinyu Ma, Daiting Shi, Shuaiqiang Wang, Dawei Yin, Xin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2601.05053)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has proven effective in enhancing the reasoning of large language models (LLMs). Monte Carlo Tree Search (MCTS)-based extensions improve upon vanilla RLVR (e.g., GRPO) by providing tree-based reasoning rollouts that enable fine-grained and segment-level credit assignment. However, existing methods still suffer from limited exploration diversity and inefficient reasoning. To address the above challenges, we propose reinforced efficient reasoning via semantically diverse explorations, i.e., ROSE, for LLMs. To encourage more diverse reasoning exploration, our method incorporates a semantic-entropy-based branching strategy and an $\varepsilon$-exploration mechanism. The former operates on already sampled reasoning rollouts to capture semantic uncertainty and select branching points with high semantic divergence to generate new successive reasoning paths, whereas the latter stochastically initiates reasoning rollouts from the root, preventing the search process from becoming overly local. To improve efficiency, we design a length-aware segment-level advantage estimator that rewards concise and correct reasoning while penalizing unnecessarily long reasoning chains. Extensive experiments on various mathematical reasoning benchmarks with Qwen and Llama models validate the effectiveness and efficiency of ROSE. Codes are available at this https URL. 

---
# Publishing FAIR and Machine-actionable Reviews in Materials Science: The Case for Symbolic Knowledge in Neuro-symbolic Artificial Intelligence 

**Authors**: Jennifer D'Souza, Soren Auer, Eleni Poupaki, Alex Watkins, Anjana Devi, Riikka L. Puurunen, Bora Karasulu, Adrie Mackus, Erwin Kessels  

**Link**: [PDF](https://arxiv.org/pdf/2601.05051)  

**Abstract**: Scientific reviews are central to knowledge integration in materials science, yet their key insights remain locked in narrative text and static PDF tables, limiting reuse by humans and machines alike. This article presents a case study in atomic layer deposition and etching (ALD/E) where we publish review tables as FAIR, machine-actionable comparisons in the Open Research Knowledge Graph (ORKG), turning them into structured, queryable knowledge. Building on this, we contrast symbolic querying over ORKG with large language model-based querying, and argue that a curated symbolic layer should remain the backbone of reliable neurosymbolic AI in materials science, with LLMs serving as complementary, symbolically grounded interfaces rather than standalone sources of truth. 

---
# On the Hidden Objective Biases of Group-based Reinforcement Learning 

**Authors**: Aleksandar Fontana, Marco Simoni, Giulio Rossolini, Andrea Saracino, Paolo Mori  

**Link**: [PDF](https://arxiv.org/pdf/2601.05002)  

**Abstract**: Group-based reinforcement learning methods, like Group Relative Policy Optimization (GRPO), are widely used nowadays to post-train large language models. Despite their empirical success, they exhibit structural mismatches between reward optimization and the underlying training objective. In this paper, we present a theoretical analysis of GRPO style methods by studying them within a unified surrogate formulation. This perspective reveals recurring properties that affect all the methods under analysis: (i) non-uniform group weighting induces systematic gradient biases on shared prefix tokens; (ii) interactions with the AdamW optimizer make training dynamics largely insensitive to reward scaling; and (iii) optimizer momentum can push policy updates beyond the intended clipping region under repeated optimization steps. We believe that these findings highlight fundamental limitations of current approaches and provide principled guidance for the design of future formulations. 

---
# ConMax: Confidence-Maximizing Compression for Efficient Chain-of-Thought Reasoning 

**Authors**: Minda Hu, Zexuan Qiu, Zenan Xu, Kun Li, Bo Zhou, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2601.04973)  

**Abstract**: Recent breakthroughs in Large Reasoning Models (LRMs) have demonstrated that extensive Chain-of-Thought (CoT) generation is critical for enabling intricate cognitive behaviors, such as self-verification and backtracking, to solve complex tasks. However, this capability often leads to ``overthinking'', where models generate redundant reasoning paths that inflate computational costs without improving accuracy. While Supervised Fine-Tuning (SFT) on reasoning traces is a standard paradigm for the 'cold start' phase, applying existing compression techniques to these traces often compromises logical coherence or incurs prohibitive sampling costs. In this paper, we introduce ConMax (Confidence-Maximizing Compression), a novel reinforcement learning framework designed to automatically compress reasoning traces while preserving essential reasoning patterns. ConMax formulates compression as a reward-driven optimization problem, training a policy to prune redundancy by maximizing a weighted combination of answer confidence for predictive fidelity and thinking confidence for reasoning validity through a frozen auxiliary LRM. Extensive experiments across five reasoning datasets demonstrate that ConMax achieves a superior efficiency-performance trade-off. Specifically, it reduces inference length by 43% over strong baselines at the cost of a mere 0.7% dip in accuracy, proving its effectiveness in generating high-quality, efficient training data for LRMs. 

---
# Higher-Order Knowledge Representations for Agentic Scientific Reasoning 

**Authors**: Isabella A. Stewart, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2601.04878)  

**Abstract**: Scientific inquiry requires systems-level reasoning that integrates heterogeneous experimental data, cross-domain knowledge, and mechanistic evidence into coherent explanations. While Large Language Models (LLMs) offer inferential capabilities, they often depend on retrieval-augmented contexts that lack structural depth. Traditional Knowledge Graphs (KGs) attempt to bridge this gap, yet their pairwise constraints fail to capture the irreducible higher-order interactions that govern emergent physical behavior. To address this, we introduce a methodology for constructing hypergraph-based knowledge representations that faithfully encode multi-entity relationships. Applied to a corpus of ~1,100 manuscripts on biocomposite scaffolds, our framework constructs a global hypergraph of 161,172 nodes and 320,201 hyperedges, revealing a scale-free topology (power law exponent ~1.23) organized around highly connected conceptual hubs. This representation prevents the combinatorial explosion typical of pairwise expansions and explicitly preserves the co-occurrence context of scientific formulations. We further demonstrate that equipping agentic systems with hypergraph traversal tools, specifically using node-intersection constraints, enables them to bridge semantically distant concepts. By exploiting these higher-order pathways, the system successfully generates grounded mechanistic hypotheses for novel composite materials, such as linking cerium oxide to PCL scaffolds via chitosan intermediates. This work establishes a "teacherless" agentic reasoning system where hypergraph topology acts as a verifiable guardrail, accelerating scientific discovery by uncovering relationships obscured by traditional graph methods. 

---
# DR-LoRA: Dynamic Rank LoRA for Mixture-of-Experts Adaptation 

**Authors**: Guanzhi Deng, Bo Li, Ronghao Chen, Huacan Wang, Linqi Song, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04823)  

**Abstract**: Mixture-of-Experts (MoE) has become a prominent paradigm for scaling Large Language Models (LLMs). Parameter-efficient fine-tuning (PEFT), such as LoRA, is widely adopted to adapt pretrained MoE LLMs to downstream tasks. However, existing approaches assign identical LoRA ranks to all experts, overlooking the intrinsic functional specialization within MoE LLMs. This uniform allocation leads to resource mismatch, task-relevant experts are under-provisioned while less relevant ones receive redundant parameters. We propose a Dynamic Rank LoRA framework named DR-LoRA, which dynamically grows expert LoRA ranks during fine-tuning based on task-specific demands. DR-LoRA employs an Expert Saliency Scoring mechanism that integrates expert routing frequency and LoRA rank importance to quantify each expert's demand for additional capacity. Experts with higher saliency scores are prioritized for rank expansion, enabling the automatic formation of a heterogeneous rank distribution tailored to the target task. Experiments on multiple benchmarks demonstrate that DR-LoRA consistently outperforms standard LoRA and static allocation strategies under the same parameter budget, achieving superior task performance with more efficient parameter utilization. 

---
# Defense Against Indirect Prompt Injection via Tool Result Parsing 

**Authors**: Qiang Yu, Xinran Cheng, Chuanyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04795)  

**Abstract**: As LLM agents transition from digital assistants to physical controllers in autonomous systems and robotics, they face an escalating threat from indirect prompt injection. By embedding adversarial instructions into the results of tool calls, attackers can hijack the agent's decision-making process to execute unauthorized actions. This vulnerability poses a significant risk as agents gain more direct control over physical environments. Existing defense mechanisms against Indirect Prompt Injection (IPI) generally fall into two categories. The first involves training dedicated detection models; however, this approach entails high computational overhead for both training and inference, and requires frequent updates to keep pace with evolving attack vectors. Alternatively, prompt-based methods leverage the inherent capabilities of LLMs to detect or ignore malicious instructions via prompt engineering. Despite their flexibility, most current prompt-based defenses suffer from high Attack Success Rates (ASR), demonstrating limited robustness against sophisticated injection attacks. In this paper, we propose a novel method that provides LLMs with precise data via tool result parsing while effectively filtering out injected malicious code. Our approach achieves competitive Utility under Attack (UA) while maintaining the lowest Attack Success Rate (ASR) to date, significantly outperforming existing methods. Code is available at GitHub. 

---
# CounterVid: Counterfactual Video Generation for Mitigating Action and Temporal Hallucinations in Video-Language Models 

**Authors**: Tobia Poppi, Burak Uzkent, Amanmeet Garg, Lucas Porto, Garin Kessler, Yezhou Yang, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara, Florian Schiffers  

**Link**: [PDF](https://arxiv.org/pdf/2601.04778)  

**Abstract**: Video-language models (VLMs) achieve strong multimodal understanding but remain prone to hallucinations, especially when reasoning about actions and temporal order. Existing mitigation strategies, such as textual filtering or random video perturbations, often fail to address the root cause: over-reliance on language priors rather than fine-grained visual dynamics. We propose a scalable framework for counterfactual video generation that synthesizes videos differing only in actions or temporal structure while preserving scene context. Our pipeline combines multimodal LLMs for action proposal and editing guidance with diffusion-based image and video models to generate semantic hard negatives at scale. Using this framework, we build CounterVid, a synthetic dataset of ~26k preference pairs targeting action recognition and temporal reasoning. We further introduce MixDPO, a unified Direct Preference Optimization approach that jointly leverages textual and visual preferences. Fine-tuning Qwen2.5-VL with MixDPO yields consistent improvements, notably in temporal ordering, and transfers effectively to standard video hallucination benchmarks. Code and models will be made publicly available. 

---
# AT$^2$PO: Agentic Turn-based Policy Optimization via Tree Search 

**Authors**: Zefang Zong, Dingwei Chen, Yang Li, Qi Yi, Bo Zhou, Chengming Li, Bo Qian, Peng Chen, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04767)  

**Abstract**: LLM agents have emerged as powerful systems for tackling multi-turn tasks by interleaving internal reasoning and external tool interactions. Agentic Reinforcement Learning has recently drawn significant research attention as a critical post-training paradigm to further refine these capabilities. In this paper, we present AT$^2$PO (Agentic Turn-based Policy Optimization via Tree Search), a unified framework for multi-turn agentic RL that addresses three core challenges: limited exploration diversity, sparse credit assignment, and misaligned policy optimization. AT$^2$PO introduces a turn-level tree structure that jointly enables Entropy-Guided Tree Expansion for strategic exploration and Turn-wise Credit Assignment for fine-grained reward propagation from sparse outcomes. Complementing this, we propose Agentic Turn-based Policy Optimization, a turn-level learning objective that aligns policy updates with the natural decision granularity of agentic interactions. ATPO is orthogonal to tree search and can be readily integrated into any multi-turn RL pipeline. Experiments across seven benchmarks demonstrate consistent improvements over the state-of-the-art baseline by up to 1.84 percentage points in average, with ablation studies validating the effectiveness of each component. Our code is available at this https URL. 

---
# Miner:Mining Intrinsic Mastery for Data-Efficient RL in Large Reasoning Models 

**Authors**: Shuyang Jiang, Yuhao Wang, Ya Zhang, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04731)  

**Abstract**: Current critic-free RL methods for large reasoning models suffer from severe inefficiency when training on positive homogeneous prompts (where all rollouts are correct), resulting in waste of rollouts due to zero advantage estimates. We introduce a radically simple yet powerful solution to \uline{M}ine \uline{in}trinsic mast\uline{er}y (Miner), that repurposes the policy's intrinsic uncertainty as a self-supervised reward signal, with no external supervision, auxiliary models, or additional inference cost. Our method pioneers two key innovations: (1) a token-level focal credit assignment mechanism that dynamically amplifies gradients on critical uncertain tokens while suppressing overconfident ones, and (2) adaptive advantage calibration to seamlessly integrate intrinsic and verifiable rewards. Evaluated across six reasoning benchmarks on Qwen3-4B and Qwen3-8B base models, Miner achieves state-of-the-art performance among the other four algorithms, yielding up to \textbf{4.58} absolute gains in Pass@1 and \textbf{6.66} gains in Pass@K compared to GRPO. Comparison with other methods targeted at exploration enhancement further discloses the superiority of the two newly proposed innovations. This demonstrates that latent uncertainty exploitation is both necessary and sufficient for efficient and scalable RL training of reasoning models. 

---
# Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning 

**Authors**: Yuyang Hu, Jiongnan Liu, Jiejun Tan, Yutao Zhu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2601.04726)  

**Abstract**: Large language models (LLMs) are increasingly deployed as intelligent agents that reason, plan, and interact with their environments. To effectively scale to long-horizon scenarios, a key capability for such agents is a memory mechanism that can retain, organize, and retrieve past experiences to support downstream decision-making. However, most existing approaches organize and store memories in a flat manner and rely on simple similarity-based retrieval techniques. Even when structured memory is introduced, existing methods often struggle to explicitly capture the logical relationships among experiences or memory units. Moreover, memory access is largely detached from the constructed structure and still depends on shallow semantic retrieval, preventing agents from reasoning logically over long-horizon dependencies. In this work, we propose CompassMem, an event-centric memory framework inspired by Event Segmentation Theory. CompassMem organizes memory as an Event Graph by incrementally segmenting experiences into events and linking them through explicit logical relations. This graph serves as a logic map, enabling agents to perform structured and goal-directed navigation over memory beyond superficial retrieval, progressively gathering valuable memories to support long-horizon reasoning. Experiments on LoCoMo and NarrativeQA demonstrate that CompassMem consistently improves both retrieval and reasoning performance across multiple backbone models. 

---
# TourPlanner: A Competitive Consensus Framework with Constraint-Gated Reinforcement Learning for Travel Planning 

**Authors**: Yinuo Wang, Mining Tan, Wenxiang Jiao, Xiaoxi Li, Hao Wang, Xuanyu Zhang, Yuan Lu, Weiming Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.04698)  

**Abstract**: Travel planning is a sophisticated decision-making process that requires synthesizing multifaceted information to construct itineraries. However, existing travel planning approaches face several challenges: (1) Pruning candidate points of interest (POIs) while maintaining a high recall rate; (2) A single reasoning path restricts the exploration capability within the feasible solution space for travel planning; (3) Simultaneously optimizing hard constraints and soft constraints remains a significant difficulty. To address these challenges, we propose TourPlanner, a comprehensive framework featuring multi-path reasoning and constraint-gated reinforcement learning. Specifically, we first introduce a Personalized Recall and Spatial Optimization (PReSO) workflow to construct spatially-aware candidate POIs' set. Subsequently, we propose Competitive consensus Chain-of-Thought (CCoT), a multi-path reasoning paradigm that improves the ability of exploring the feasible solution space. To further refine the plan, we integrate a sigmoid-based gating mechanism into the reinforcement learning stage, which dynamically prioritizes soft-constraint satisfaction only after hard constraints are met. Experimental results on travel planning benchmarks demonstrate that TourPlanner achieves state-of-the-art performance, significantly surpassing existing methods in both feasibility and user-preference alignment. 

---
# A Method for Constructing a Digital Transformation Driving Mechanism Based on Semantic Understanding of Large Models 

**Authors**: Huayi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04696)  

**Abstract**: In the process of digital transformation, enterprises are faced with problems such as insufficient semantic understanding of unstructured data and lack of intelligent decision-making basis in driving mechanisms. This study proposes a method that combines a large language model (LLM) and a knowledge graph. First, a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model is used to perform entity recognition and relationship extraction on multi-source heterogeneous texts, and GPT-4 is used to generate semantically enhanced vector representations; secondly, a two-layer graph neural network (GNN) architecture is designed to fuse the semantic vectors output by LLM with business metadata to construct a dynamic and scalable enterprise knowledge graph; then reinforcement learning is introduced to optimize decision path generation, and the reward function is used to drive the mechanism iteration. In the case of the manufacturing industry, this mechanism reduced the response time for equipment failure scenarios from 7.8 hours to 3.7 hours, the F1 value reached 94.3%, and the compensation for decision errors in the annual digital transformation cost decreased by 45.3%. This method significantly enhances the intelligence level and execution efficiency of the digital transformation driving mechanism by integrating large model semantic understanding with structured knowledge. 

---
# Agri-R1: Empowering Generalizable Agricultural Reasoning in Vision-Language Models with Reinforcement Learning 

**Authors**: Wentao Zhang, Lifei Wang, Lina Lu, MingKun Xu, Shangyang Li, Yanchao Yang, Tao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04672)  

**Abstract**: Agricultural disease diagnosis challenges VLMs, as conventional fine-tuning requires extensive labels, lacks interpretability, and generalizes poorly. While reasoning improves model robustness, existing methods rely on costly expert annotations and rarely address the open-ended, diverse nature of agricultural queries. To address these limitations, we propose \textbf{Agri-R1}, a reasoning-enhanced large model for agriculture. Our framework automates high-quality reasoning data generation via vision-language synthesis and LLM-based filtering, using only 19\% of available samples. Training employs Group Relative Policy Optimization (GRPO) with a novel proposed reward function that integrates domain-specific lexicons and fuzzy matching to assess both correctness and linguistic flexibility in open-ended responses. Evaluated on CDDMBench, our resulting 3B-parameter model achieves performance competitive with 7B- to 13B-parameter baselines, showing a +23.2\% relative gain in disease recognition accuracy, +33.3\% in agricultural knowledge QA, and a +26.10-point improvement in cross-domain generalization over standard fine-tuning. Ablation studies confirm that the synergy between structured reasoning data and GRPO-driven exploration underpins these gains, with benefits scaling as question complexity increases. 

---
# Succeeding at Scale: Automated Multi-Retriever Fusion and Query-Side Adaptation for Multi-Tenant Search 

**Authors**: Prateek Jain, Shabari S Nair, Ritesh Goru, Prakhar Agarwal, Ajay Yadav, Yoga Sri Varshan Varadharajan, Constantine Caramanis  

**Link**: [PDF](https://arxiv.org/pdf/2601.04646)  

**Abstract**: Large-scale multi-tenant retrieval systems amass vast user query logs yet critically lack the curated relevance labels required for effective domain adaptation. This "dark data" problem is exacerbated by the operational cost of model updates: jointly fine-tuning query and document encoders requires re-indexing the entire corpus, which is prohibitive in multi-tenant environments with thousands of isolated indices. To address these dual challenges, we introduce \textbf{DevRev Search}, a passage retrieval benchmark for technical customer support constructed through a fully automatic pipeline. We employ a \textbf{fusion-based candidate generation} strategy, pooling results from diverse sparse and dense retrievers, and utilize an LLM-as-a-Judge to perform rigorous \textbf{consistency filtering} and relevance assignment. We further propose a practical \textbf{Index-Preserving Adaptation} strategy: by fine-tuning only the query encoder via Low-Rank Adaptation (LoRA), we achieve competitive performance improvements while keeping the document index frozen. Our experiments on DevRev Search and SciFact demonstrate that targeting specific transformer layers in the query encoder yields optimal quality-efficiency trade-offs, offering a scalable path for personalized enterprise search. 

---
# DP-MGTD: Privacy-Preserving Machine-Generated Text Detection via Adaptive Differentially Private Entity Sanitization 

**Authors**: Lionel Z. Wang, Yusheng Zhao, Jiabin Luo, Xinfeng Li, Lixu Wang, Yinan Peng, Haoyang Li, XiaoFeng Wang, Wei Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.04641)  

**Abstract**: The deployment of Machine-Generated Text (MGT) detection systems necessitates processing sensitive user data, creating a fundamental conflict between authorship verification and privacy preservation. Standard anonymization techniques often disrupt linguistic fluency, while rigorous Differential Privacy (DP) mechanisms typically degrade the statistical signals required for accurate detection. To resolve this dilemma, we propose \textbf{DP-MGTD}, a framework incorporating an Adaptive Differentially Private Entity Sanitization algorithm. Our approach utilizes a two-stage mechanism that performs noisy frequency estimation and dynamically calibrates privacy budgets, applying Laplace and Exponential mechanisms to numerical and textual entities respectively. Crucially, we identify a counter-intuitive phenomenon where the application of DP noise amplifies the distinguishability between human and machine text by exposing distinct sensitivity patterns to perturbation. Extensive experiments on the MGTBench-2.0 dataset show that our method achieves near-perfect detection accuracy, significantly outperforming non-private baselines while satisfying strict privacy guarantees. 

---
# Neurosymbolic Retrievers for Retrieval-augmented Generation 

**Authors**: Yash Saxena, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2601.04568)  

**Abstract**: Retrieval Augmented Generation (RAG) has made significant strides in overcoming key limitations of large language models, such as hallucination, lack of contextual grounding, and issues with transparency. However, traditional RAG systems consist of three interconnected neural components - the retriever, re-ranker, and generator - whose internal reasoning processes remain opaque. This lack of transparency complicates interpretability, hinders debugging efforts, and erodes trust, especially in high-stakes domains where clear decision-making is essential. To address these challenges, we introduce the concept of Neurosymbolic RAG, which integrates symbolic reasoning using a knowledge graph with neural retrieval techniques. This new framework aims to answer two primary questions: (a) Can retrievers provide a clear and interpretable basis for document selection? (b) Can symbolic knowledge enhance the clarity of the retrieval process? We propose three methods to improve this integration. First is MAR (Knowledge Modulation Aligned Retrieval) that employs modulation networks to refine query embeddings using interpretable symbolic features, thereby making document matching more explicit. Second, KG-Path RAG enhances queries by traversing knowledge graphs to improve overall retrieval quality and interpretability. Lastly, Process Knowledge-infused RAG utilizes domain-specific tools to reorder retrieved content based on validated workflows. Preliminary results from mental health risk assessment tasks indicate that this neurosymbolic approach enhances both transparency and overall performance 

---
# BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents 

**Authors**: Yunhao Feng, Yige Li, Yutao Wu, Yingshui Tan, Yanming Guo, Yifan Ding, Kun Zhai, Xingjun Ma, Yugang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04566)  

**Abstract**: Large language model (LLM) agents execute tasks through multi-step workflows that combine planning, memory, and tool use. While this design enables autonomy, it also expands the attack surface for backdoor threats. Backdoor triggers injected into specific stages of an agent workflow can persist through multiple intermediate states and adversely influence downstream outputs. However, existing studies remain fragmented and typically analyze individual attack vectors in isolation, leaving the cross-stage interaction and propagation of backdoor triggers poorly understood from an agent-centric perspective. To fill this gap, we propose \textbf{BackdoorAgent}, a modular and stage-aware framework that provides a unified, agent-centric view of backdoor threats in LLM agents. BackdoorAgent structures the attack surface into three functional stages of agentic workflows, including \textbf{planning attacks}, \textbf{memory attacks}, and \textbf{tool-use attacks}, and instruments agent execution to enable systematic analysis of trigger activation and propagation across different stages. Building on this framework, we construct a standardized benchmark spanning four representative agent applications: \textbf{Agent QA}, \textbf{Agent Code}, \textbf{Agent Web}, and \textbf{Agent Drive}, covering both language-only and multimodal settings. Our empirical analysis shows that \textit{triggers implanted at a single stage can persist across multiple steps and propagate through intermediate states.} For instance, when using a GPT-based backbone, we observe trigger persistence in 43.58\% of planning attacks, 77.97\% of memory attacks, and 60.28\% of tool-stage attacks, highlighting the vulnerabilities of the agentic workflow itself to backdoor threats. To facilitate reproducibility and future research, our code and benchmark are publicly available at GitHub. 

---
# A Vision for Multisensory Intelligence: Sensing, Synergy, and Science 

**Authors**: Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04563)  

**Abstract**: Our experience of the world is multisensory, spanning a synthesis of language, sight, sound, touch, taste, and smell. Yet, artificial intelligence has primarily advanced in digital modalities like text, vision, and audio. This paper outlines a research vision for multisensory artificial intelligence over the next decade. This new set of technologies can change how humans and AI experience and interact with one another, by connecting AI to the human senses and a rich spectrum of signals from physiological and tactile cues on the body, to physical and social signals in homes, cities, and the environment. We outline how this field must advance through three interrelated themes of sensing, science, and synergy. Firstly, research in sensing should extend how AI captures the world in richer ways beyond the digital medium. Secondly, developing a principled science for quantifying multimodal heterogeneity and interactions, developing unified modeling architectures and representations, and understanding cross-modal transfer. Finally, we present new technical challenges to learn synergy between modalities and between humans and AI, covering multisensory integration, alignment, reasoning, generation, generalization, and experience. Accompanying this vision paper are a series of projects, resources, and demos of latest advances from the Multisensory Intelligence group at the MIT Media Lab, see this https URL. 

---
# Not All Steps are Informative: On the Linearity of LLMs' RLVR Training 

**Authors**: Tianle Wang, Zhongyuan Wu, Shenghao Jin, Hao Xu, Wei Chen, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2601.04537)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a central component of large language model (LLM) post-training. Unlike supervised fine-tuning (SFT), RLVR lets an LLM generate multiple candidate solutions and reinforces those that lead to a verifiably correct final answer. However, in practice, RLVR often requires thousands of training steps to reach strong performance, incurring substantial computation largely attributed to prolonged exploration. In this work, we make a surprising observation: during RLVR, LLMs evolve in a strongly linear manner. Specifically, both model weights and model output log-probabilities exhibit strong linear correlations with RL training steps. This suggests that RLVR predominantly amplifies trends that emerge early in training, rather than continuously discovering new behaviors throughout the entire optimization trajectory. Motivated by this linearity, we investigate whether future model states can be predicted from intermediate checkpoints via extrapolation, avoiding continued expensive training. We show that Weight Extrapolation produces models with performance comparable to standard RL training while requiring significantly less computation. Moreover, Logits Extrapolation consistently outperforms continued RL training on all four benchmarks by extrapolating beyond the step range where RL training remains stable. 

---
# Advancing Language Models for Code-related Tasks 

**Authors**: Zhao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.04526)  

**Abstract**: Recent advances in language models (LMs) have driven significant progress in various software engineering tasks. However, existing LMs still struggle with complex programming scenarios due to limitations in data quality, model architecture, and reasoning capability. This research systematically addresses these challenges through three complementary directions: (1) improving code data quality with a code difference-guided adversarial augmentation technique (CODA) and a code denoising technique (CodeDenoise); (2) enhancing model architecture via syntax-guided code LMs (LEAM and LEAM++); and (3) advancing model reasoning with a prompting technique (muFiX) and an agent-based technique (Specine). These techniques aim to promote the practical adoption of LMs in software development and further advance intelligent software engineering. 

---
# CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts 

**Authors**: Khandakar Shakib Al Hasan, Syed Rifat Raiyan, Hasin Mahtab Alvee, Wahid Sadik  

**Link**: [PDF](https://arxiv.org/pdf/2601.04505)  

**Abstract**: Generating accurate circuit schematics from high-level natural language descriptions remains a persistent challenge in electronics design, as large language models (LLMs) frequently hallucinate in granular details, violate electrical constraints, and produce non-machine-readable outputs. We present CircuitLM, a novel multi-agent LLM-aided circuit design pipeline that translates user prompts into structured, visually interpretable CircuitJSON schematics through five sequential stages: (i) LLM-based component identification, (ii) canonical pinout retrieval, (iii) chain-of-thought reasoning by an electronics expert agent, (iv) JSON schematic synthesis, and (v) force-directed SVG visualization. Anchored by a curated, embedding-powered component knowledge base. While LLMs often violate electrical constraints, CircuitLM bridges this gap by grounding generation in a verified and dynamically extensible component database, initially comprising 50 components. To ensure safety, we incorporate a hybrid evaluation framework, namely Dual-Metric Circuit Validation (DMCV), validated against human-expert assessments, which achieves high fidelity in microcontroller-centric designs. We evaluate the system on 100 diverse embedded-systems prompts across six LLMs and introduce DMCV to assess both structural and electrical validity. This work bridges natural language input to deployable hardware designs, enabling reliable circuit prototyping by non-experts. Our code and data will be made public upon acceptance. 

---
# Vision-Language Agents for Interactive Forest Change Analysis 

**Authors**: James Brock, Ce Zhang, Nantheera Anantrasirichai  

**Link**: [PDF](https://arxiv.org/pdf/2601.04497)  

**Abstract**: Modern forest monitoring workflows increasingly benefit from the growing availability of high-resolution satellite imagery and advances in deep learning. Two persistent challenges in this context are accurate pixel-level change detection and meaningful semantic change captioning for complex forest dynamics. While large language models (LLMs) are being adapted for interactive data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored. To address this gap, we introduce an LLM-driven agent for integrated forest change analysis that supports natural language querying across multiple RSICI tasks. The proposed system builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration. To facilitate adaptation and evaluation in forest environments, we further introduce the Forest-Change dataset, which comprises bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated using a combination of human annotation and rule-based methods. Experimental results show that the proposed system achieves mIoU and BLEU-4 scores of 67.10% and 40.17% on the Forest-Change dataset, and 88.13% and 34.41% on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI benchmark for joint change detection and captioning. These results highlight the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and efficiency of forest change analysis. All data and code are publicly available at this https URL. 

---
# Re-Rankers as Relevance Judges 

**Authors**: Chuan Meng, Jiqun Liu, Mohammad Aliannejadi, Fengran Mo, Jeff Dalton, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2601.04455)  

**Abstract**: Using large language models (LLMs) to predict relevance judgments has shown promising results. Most studies treat this task as a distinct research line, e.g., focusing on prompt design for predicting relevance labels given a query and passage. However, predicting relevance judgments is essentially a form of relevance prediction, a problem extensively studied in tasks such as re-ranking. Despite this potential overlap, little research has explored reusing or adapting established re-ranking methods to predict relevance judgments, leading to potential resource waste and redundant development. To bridge this gap, we reproduce re-rankers in a re-ranker-as-relevance-judge setup. We design two adaptation strategies: (i) using binary tokens (e.g., "true" and "false") generated by a re-ranker as direct judgments, and (ii) converting continuous re-ranking scores into binary labels via thresholding. We perform extensive experiments on TREC-DL 2019 to 2023 with 8 re-rankers from 3 families, ranging from 220M to 32B, and analyse the evaluation bias exhibited by re-ranker-based judges. Results show that re-ranker-based relevance judges, under both strategies, can outperform UMBRELA, a state-of-the-art LLM-based relevance judge, in around 40% to 50% of the cases; they also exhibit strong self-preference towards their own and same-family re-rankers, as well as cross-family bias. 

---
# Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization 

**Authors**: Xingjian Diao, Zheyuan Liu, Chunhui Zhang, Weiyi Wu, Keyi Kong, Lin Shi, Kaize Ding, Soroush Vosoughi, Jiang Gui  

**Link**: [PDF](https://arxiv.org/pdf/2601.04442)  

**Abstract**: Large Vision-Language Models (LVLMs) have exhibited strong reasoning capabilities through chain-of-thought mechanisms that generate step-by-step rationales. However, such slow-thinking approaches often lead to overthinking, where models produce excessively verbose responses even for simple queries, resulting in test-time inefficiency and even degraded accuracy. Prior work has attempted to mitigate this issue via adaptive reasoning strategies, but these methods largely overlook a fundamental bottleneck: visual perception failures. We argue that stable reasoning critically depends on low-level visual grounding, and that reasoning errors often originate from imperfect perception rather than insufficient deliberation. To address this limitation, we propose Gated Perception-Reasoning Optimization (GPRO), a meta-reasoning controller that dynamically routes computation among three decision paths at each generation step: a lightweight fast path, a slow perception path for re-examining visual inputs, and a slow reasoning path for internal self-reflection. To learn this distinction, we derive large-scale failure attribution supervision from approximately 790k samples, using teacher models to distinguish perceptual hallucinations from reasoning errors. We then train the controller with multi-objective reinforcement learning to optimize the trade-off between task accuracy and computational cost under uncertainty. Experiments on five benchmarks demonstrate that GPRO substantially improves both accuracy and efficiency, outperforming recent slow-thinking methods while generating significantly shorter responses. 

---
# Rate or Fate? RLV$^\varepsilon$R: Reinforcement Learning with Verifiable Noisy Rewards 

**Authors**: Ali Rad, Khashayar Filom, Darioush Keivan, Peyman Mohajerin Esfahani, Ehsan Kamalinejad  

**Link**: [PDF](https://arxiv.org/pdf/2601.04411)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a simple but powerful paradigm for training LLMs: sample a completion, verify it, and update. In practice, however, the verifier is almost never clean--unit tests probe only limited corner cases; human and synthetic labels are imperfect; and LLM judges (e.g., RLAIF) are noisy and can be exploited--and this problem worsens on harder domains (especially coding) where tests are sparse and increasingly model-generated. We ask a pragmatic question: does the verification noise merely slow down the learning (rate), or can it flip the outcome (fate)?
To address this, we develop an analytically tractable multi-armed bandit view of RLVR dynamics, instantiated with GRPO and validated in controlled experiments. Modeling false positives and false negatives and grouping completions into recurring reasoning modes yields a replicator-style (natural-selection) flow on the probability simplex. The dynamics decouples into within-correct-mode competition and a one-dimensional evolution for the mass on incorrect modes, whose drift is determined solely by Youden's index J=TPR-FPR. This yields a sharp phase transition: when J>0, the incorrect mass is driven toward extinction (learning); when J=0, the process is neutral; and when J<0, incorrect modes amplify until they dominate (anti-learning and collapse). In the learning regime J>0, noise primarily rescales convergence time ("rate, not fate"). Experiments on verifiable programming tasks under synthetic noise reproduce the predicted J=0 boundary. Beyond noise, the framework offers a general lens for analyzing RLVR stability, convergence, and algorithmic interventions. 

---
# The Language of Bargaining: Linguistic Effects in LLM Negotiations 

**Authors**: Stuti Sinha, Himanshu Kumar, Aryan Raju Mandapati, Rakshit Sakhuja, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.04387)  

**Abstract**: Negotiation is a core component of social intelligence, requiring agents to balance strategic reasoning, cooperation, and social norms. Recent work shows that LLMs can engage in multi-turn negotiation, yet nearly all evaluations occur exclusively in English. Using controlled multi-agent simulations across Ultimatum, Buy-Sell, and Resource Exchange games, we systematically isolate language effects across English and four Indic framings (Hindi, Punjabi, Gujarati, Marwadi) by holding game rules, model parameters, and incentives constant across all conditions. We find that language choice can shift outcomes more strongly than changing models, reversing proposer advantages and reallocating surplus. Crucially, effects are task-contingent: Indic languages reduce stability in distributive games yet induce richer exploration in integrative settings. Our results demonstrate that evaluating LLM negotiation solely in English yields incomplete and potentially misleading conclusions. These findings caution against English-only evaluation of LLMs and suggest that culturally-aware evaluation is essential for fair deployment. 

---
# Generalization to Political Beliefs from Fine-Tuning on Sports Team Preferences 

**Authors**: Owen Terry  

**Link**: [PDF](https://arxiv.org/pdf/2601.04369)  

**Abstract**: Fine-tuned LLMs often exhibit unexpected behavior as a result of generalizing beyond the data they're shown. We present results in which an LLM fine-tuned to prefer either coastal sports teams or Southern sports teams adopt political beliefs that diverge significantly from those of the base model. While we hypothesized that the coastal model would become more liberal and the southern model would become more conservative, we find that their responses are usually similar to each other, without a clear-cut liberal or conservative bias. In addition to asking the models for numerical ratings of agreement with relevant political statements, we ask them to elaborate on their more radical answers, finding varying degrees of willingness to justify themselves. Further work is needed to understand the mechanisms by which fine-tuning on simple, narrow datasets leads to seemingly unrelated changes in model behavior. 

---
# Quantifying the Effect of Test Set Contamination on Generative Evaluations 

**Authors**: Rylan Schaeffer, Joshua Kazdan, Baber Abbasi, Ken Ziyu Liu, Brando Miranda, Ahmed Ahmed, Abhay Puri, Niloofar Mireshghallah, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2601.04301)  

**Abstract**: As frontier AI systems are pretrained on web-scale data, test set contamination has become a critical concern for accurately assessing their capabilities. While research has thoroughly investigated the impact of test set contamination on discriminative evaluations like multiple-choice question-answering, comparatively little research has studied the impact of test set contamination on generative evaluations. In this work, we quantitatively assess the effect of test set contamination on generative evaluations through the language model lifecycle. We pretrain language models on mixtures of web data and the MATH benchmark, sweeping model sizes and number of test set replicas contaminating the pretraining corpus; performance improves with contamination and model size. Using scaling laws, we make a surprising discovery: including even a single test set replica enables models to achieve lower loss than the irreducible error of training on the uncontaminated corpus. We then study further training: overtraining with fresh data reduces the effects of contamination, whereas supervised finetuning on the training set can either increase or decrease performance on test data, depending on the amount of pretraining contamination. Finally, at inference, we identify factors that modulate memorization: high sampling temperatures mitigate contamination effects, and longer solutions are exponentially more difficult to memorize than shorter ones, presenting a contrast with discriminative evaluations, where solutions are only a few tokens in length. By characterizing how generation and memorization interact, we highlight a new layer of complexity for trustworthy evaluation of AI systems. 

---
# Mitigating Position-Shift Failures in Text-Based Modular Arithmetic via Position Curriculum and Template Diversity 

**Authors**: Nikolay Yudin  

**Link**: [PDF](https://arxiv.org/pdf/2601.04283)  

**Abstract**: Building on insights from the grokking literature, we study character-level Transformers trained to compute modular addition from text, and focus on robustness under input-format variation rather than only in-distribution accuracy. We identify a previously under-emphasized failure mode: models that achieve high in-distribution accuracy can fail catastrophically when the same expression is shifted to different absolute character positions ("position shift") or presented under out-of-distribution natural-language templates. Using a disjoint-pair split over all ordered pairs for p=97, we show that a baseline model reaches strong in-distribution performance yet collapses under position shift and template OOD. We then introduce a simple training recipe that combines (i) explicit expression boundary markers, (ii) position curriculum that broadens the range of absolute positions seen during training, (iii) diverse template mixtures, and (iv) consistency training across multiple variants per example. Across three seeds, this intervention substantially improves robustness to position shift and template OOD while maintaining high in-distribution accuracy, whereas an ALiBi-style ablation fails to learn the task under our setup. Our results suggest that steering procedural generalization under noisy supervision benefits from explicitly training invariances that are otherwise absent from the data distribution, and we provide a reproducible evaluation protocol and artifacts. 

---
# Shadow Unlearning: A Neuro-Semantic Approach to Fidelity-Preserving Faceless Forgetting in LLMs 

**Authors**: Dinesh Srivasthav P, Ashok Urlana, Rahul Mishra, Bala Mallikarjunarao Garlapati, Ponnurangam Kumaraguru  

**Link**: [PDF](https://arxiv.org/pdf/2601.04275)  

**Abstract**: Machine unlearning aims to selectively remove the influence of specific training samples to satisfy privacy regulations such as the GDPR's 'Right to be Forgotten'. However, many existing methods require access to the data being removed, exposing it to membership inference attacks and potential misuse of Personally Identifiable Information (PII). We address this critical challenge by proposing Shadow Unlearning, a novel paradigm of approximate unlearning, that performs machine unlearning on anonymized forget data without exposing PII. We further propose a novel privacy-preserving framework, Neuro-Semantic Projector Unlearning (NSPU) to achieve Shadow unlearning. To evaluate our method, we compile Multi-domain Fictitious Unlearning (MuFU) forget set across five diverse domains and introduce an evaluation stack to quantify the trade-off between knowledge retention and unlearning effectiveness. Experimental results on various LLMs show that NSPU achieves superior unlearning performance, preserves model utility, and enhances user privacy. Additionally, the proposed approach is at least 10 times more computationally efficient than standard unlearning approaches. Our findings foster a new direction for privacy-aware machine unlearning that balances data protection and model fidelity. 

---
# Sphinx: Benchmarking and Modeling for LLM-Driven Pull Request Review 

**Authors**: Daoan Zhang, Shuo Zhang, Zijian Jin, Jiebo Luo, Shengyu Fu, Elsie Nallipogu  

**Link**: [PDF](https://arxiv.org/pdf/2601.04252)  

**Abstract**: Pull request (PR) review is essential for ensuring software quality, yet automating this task remains challenging due to noisy supervision, limited contextual understanding, and inadequate evaluation metrics. We present Sphinx, a unified framework for LLM-based PR review that addresses these limitations through three key components: (1) a structured data generation pipeline that produces context-rich, semantically grounded review comments by comparing pseudo-modified and merged code; (2) a checklist-based evaluation benchmark that assesses review quality based on structured coverage of actionable verification points, moving beyond surface-level metrics like BLEU; and (3) Checklist Reward Policy Optimization (CRPO), a novel training paradigm that uses rule-based, interpretable rewards to align model behavior with real-world review practices. Extensive experiments show that models trained with Sphinx achieve state-of-the-art performance on review completeness and precision, outperforming both proprietary and open-source baselines by up to 40\% in checklist coverage. Together, Sphinx enables the development of PR review models that are not only fluent but also context-aware, technically precise, and practically deployable in real-world development workflows. The data will be released after review. 

---
# SAGE-32B: Agentic Reasoning via Iterative Distillation 

**Authors**: Basab Jha, Firoj Paudel, Ujjwal Puri, Ethan Henkel, Zhang Yuting, Mateusz Kowalczyk, Mei Huang, Choi Donghyuk, Wang Junhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.04237)  

**Abstract**: We demonstrate SAGE-32B, a 32 billion parameter language model that focuses on agentic reasoning and long range planning tasks. Unlike chat models that aim for general conversation fluency, SAGE-32B is designed to operate in an agentic loop, emphasizing task decomposition, tool usage, and error recovery. The model is initialized from the Qwen2.5-32B pretrained model and fine tuned using Iterative Distillation, a two stage training process that improves reasoning performance through rigorously tested feedback loops. SAGE-32B also introduces an inverse reasoning approach, which uses a meta cognition head to forecast potential failures in the planning process before execution. On agentic reasoning benchmarks including MMLU-Pro, AgentBench, and MATH-500, SAGE-32B achieves higher success rates in multi tool usage scenarios compared to similarly sized baseline models, while remaining competitive on standard reasoning evaluations. Model weights are publicly released at this https URL 

---
# Generative Teaching via Code 

**Authors**: Yuheng Wang, Runde Yang, Lin Wu, Jie Zhang, Jingru Fan, Ruoyu Fu, Tianle Zhou, Huatao Li, Siheng Chen, Weinan E, Chen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2601.04204)  

**Abstract**: The scalability of high-quality online education is hindered by the high costs and slow cycles of labor-intensive manual content creation. Despite advancements in video generation, current approaches often fail to ensure pedagogical structure and precise control due to their pixel-level, black-box nature. In this paper, we propose Generative Teaching, a novel paradigm that transitions educators from manual creators to high-level directors, allowing them to focus on pedagogical intent while autonomous agents handle the execution. To realize this vision, we introduce TeachMaster, a multi-agent framework that leverages code as an intermediate semantic medium. Unlike traditional video generation methods, TeachMaster orchestrates a collaborative team of agents--spanning planning, design, and rendering--to automate the production of interpretable, editable, and curriculum-ready educational videos. Experiments validate that TeachMaster significantly boosts production efficiency without compromising structural coherence or visual fidelity, providing a robust solution for scalable education. 

---
# The Forgotten Shield: Safety Grafting in Parameter-Space for Medical MLLMs 

**Authors**: Jiale Zhao, Xing Mou, Jinlin Wu, Hongyuan Yu, Mingrui Sun, Yang Shi, Xuanwu Yin, Zhen Chen, Zhen Lei, Yaohua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04199)  

**Abstract**: Medical Multimodal Large Language Models (Medical MLLMs) have achieved remarkable progress in specialized medical tasks; however, research into their safety has lagged, posing potential risks for real-world deployment. In this paper, we first establish a multidimensional evaluation framework to systematically benchmark the safety of current SOTA Medical MLLMs. Our empirical analysis reveals pervasive vulnerabilities across both general and medical-specific safety dimensions in existing models, particularly highlighting their fragility against cross-modality jailbreak attacks. Furthermore, we find that the medical fine-tuning process frequently induces catastrophic forgetting of the model's original safety alignment. To address this challenge, we propose a novel "Parameter-Space Intervention" approach for efficient safety re-alignment. This method extracts intrinsic safety knowledge representations from original base models and concurrently injects them into the target model during the construction of medical capabilities. Additionally, we design a fine-grained parameter search algorithm to achieve an optimal trade-off between safety and medical performance. Experimental results demonstrate that our approach significantly bolsters the safety guardrails of Medical MLLMs without relying on additional domain-specific safety data, while minimizing degradation to core medical performance. 

---
