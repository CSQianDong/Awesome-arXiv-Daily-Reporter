# Dual Alignment Between Language Model Layers and Human Sentence Processing 

**Authors**: Tatsuki Kuribayashi, Alex Warstadt, Yohei Oseki, Ethan Gotlieb Wilcox  

**Link**: [PDF](https://arxiv.org/pdf/2604.18563)  

**Abstract**: A recent study (Kuribayashi et al., 2025) has shown that human sentence processing behavior, typically measured on syntactically unchallenging constructions, can be effectively modeled using surprisal from early layers of large language models (LLMs). This raises the question of whether such advantages of internal layers extend to more syntactically challenging constructions, where surprisal has been reported to underestimate human cognitive effort. In this paper, we begin by exploring internal layers that better estimate human cognitive effort observed in syntactic ambiguity processing in English. Our experiments show that, in contrast to naturalistic reading, later layers better estimate such a cognitive effort, but still underestimate the human data. This dual alignment sheds light on different modes of sentence processing in humans and LMs: naturalistic reading employs a somewhat weak prediction akin to earlier layers of LMs, while syntactically challenging processing requires more fully-contextualized representations, better modeled by later layers of LMs. Motivated by these findings, we also explore several probability-update measures using shallow and deep layers of LMs, showing a complementary advantage to single-layer's surprisal in reading time modeling. 

---
# GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling 

**Authors**: Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtic, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2604.18556)  

**Abstract**: Weight quantization has become a standard tool for efficient LLM deployment, especially for local inference, where models are now routinely served at 2-3 bits per parameter. The state of the art is currently split into two sets of methods: simple scalar quantization techniques, such as GPTQ or AWQ, which are widely deployed but plateau in accuracy at 3-4 bits per parameter (bpp), and "second-generation" vector- or trellis-quantized methods, such as QTIP, GPTVQ and AQLM, which push the accuracy frontier at low bit-widths but are notoriously hard to implement and to scale, and have gained relatively less traction. In this paper, we ask whether this gap is fundamental, or whether a carefully optimized scalar quantizer can recover most of it. We answer in the affirmative, by introducing GSQ (Gumbel-Softmax Quantization), a post-training scalar quantization method which jointly learns the per-coordinate grid assignments and the per-group scales using a Gumbel-Softmax relaxation of the discrete grid. GSQ matches the cardinality of the relaxation to the small number of levels available in the target bit-width regime (e.g., 3-8 levels for ternary and 3 bpp, respectively), making the relaxation tight and the optimization tractable. Practically, on the standard Llama-3.1-8B/70B-Instruct models, GSQ closes most of the gap between scalar quantization and the QTIP frontier at 2 and 3 bits, while using a symmetric scalar grid with group-wise quantization, and thus fully compatible with existing scalar inference kernels. We further show that GSQ scales to trillion-scale Mixture-of-Experts models such as Kimi-K2.5, where vector-quantized methods are difficult to apply. 

---
# Transition-Matrix Regularization for Next Dialogue Act Prediction in Counselling Conversations 

**Authors**: Eric Rudolph, Philipp Steigerwald, Jens Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2604.18539)  

**Abstract**: This paper studies how empirical dialogue-flow statistics can be incorporated into Next Dialogue Act Prediction (NDAP). A KL regularization term is proposed that aligns predicted act distributions with corpus-derived transition patterns. Evaluated on a 60-class German counselling taxonomy using 5-fold cross-validation, this improves macro-F1 by 9--42% relative depending on encoder and substantially improves dialogue-flow alignment. Cross-dataset validation on HOPE suggests that improvements transfer across languages and counselling domains. In systematic ablations across pretrained encoders and architectures, the findings indicate that transition regularization provides consistent gains and disproportionately benefits weaker baseline models. The results suggest that lightweight discourse-flow priors complement pretrained encoders, especially in fine-grained, data-sparse dialogue tasks. 

---
# MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation 

**Authors**: Xingchen Xiao, Heyan Huang, Runheng Liu, Jincheng Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.18509)  

**Abstract**: Large language models (LLMs) are widely used in retrieval-augmented generation (RAG) to incorporate external knowledge at inference time. However, when retrieved contexts are noisy, incomplete, or heterogeneous, a single generation process often struggles to reconcile evidence effectively. We propose \textbf{MASS-RAG}, a multi-agent synthesis approach to retrieval-augmented generation that structures evidence processing into multiple role-specialized agents. MASS-RAG applies distinct agents for evidence summarization, evidence extraction, and reasoning over retrieved documents, and combines their outputs through a dedicated synthesis stage to produce the final answer. This design exposes multiple intermediate evidence views, allowing the model to compare and integrate complementary information before answer generation. Experiments on four benchmarks show that MASS-RAG consistently improves performance over strong RAG baselines, particularly in settings where relevant evidence is distributed across retrieved contexts. 

---
# LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation 

**Authors**: Samar M. Magdy, Fakhraddin Alwajih, Abdellah El Mekki, Wesam El-Sayed, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2604.18490)  

**Abstract**: Existing MT evaluation frameworks, including automatic metrics and human evaluation schemes such as Multidimensional Quality Metrics (MQM), are largely language-agnostic. However, they often fail to capture dialect- and culture-specific errors in diglossic languages (e.g., Arabic), where translation failures stem from mismatches in language variety, content coverage, and pragmatic appropriateness rather than surface form this http URL introduce LQM: Linguistically Motivated Multidimensional Quality Metrics for MT. LQM is a hierarchical error taxonomy for diagnosing MT errors through six linguistically grounded levels: sociolinguistics, pragmatics, semantics, morphosyntax, orthography, and graphetics (Figure 1). We construct a bidirectional parallel corpus of 3,850 sentences (550 per variety) spanning seven Arabic dialects (Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, and Yemeni), derived from conversational, culturally rich content. We evaluate six LLMs in a zero-shot setting and conduct expert span-level human annotation using LQM, producing 6,113 labeled error spans across 3,495 unique erroneous sentences, along with severity-weighted quality scores. We complement this analysis with an automatic metric (spBLEU). Though validated here on Arabic, LQM is a language-agnostic framework designed to be easily applied to or adapted for other languages. LQM annotated errors data, prompts, and annotation guidelines are publicly available at this https URL. 

---
# Adversarial Humanities Benchmark: Results on Stylistic Robustness in Frontier Model Safety 

**Authors**: Marcello Galisai, Susanna Cifani, Francesco Giarrusso, Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Federico Sartore, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2604.18487)  

**Abstract**: The Adversarial Humanities Benchmark (AHB) evaluates whether model safety refusals survive a shift away from familiar harmful prompt forms. Starting from harmful tasks drawn from MLCommons AILuminate, the benchmark rewrites the same objectives through humanities-style transformations while preserving intent. This extends literature on Adversarial Poetry and Adversarial Tales from single jailbreak operators to a broader benchmark family of stylistic obfuscation and goal concealment. In the benchmark results reported here, the original attacks record 3.84% attack success rate (ASR), while transformed methods range from 36.8% to 65.0%, yielding 55.75% overall ASR across 31 frontier models. Under a European Union AI Act Code-of-Practice-inspired systemic-risk lens, Chemical, biological, radiological and nuclear (CBRN) is the highest bucket. Taken together, this lack of stylistic robustness suggests that current safety techniques suffer from weak generalization: deep understanding of 'non-maleficence' remains a central unresolved problem in frontier model safety. 

---
# BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources 

**Authors**: Raghvendra Kumar, Devankar Raj, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2604.18423)  

**Abstract**: India's linguistic landscape, spanning 22 scheduled languages and hundreds of marginalized dialects, has driven rapid growth in NLP datasets, benchmarks, and pretrained models. However, no dedicated survey consolidates resources developed specifically for Indian languages. Existing reviews either focus on a few high-resource languages or subsume Indian languages within broader multilingual settings, limiting coverage of low-resource and culturally diverse varieties. To address this gap, we present the first unified survey of Indian NLP resources, covering 200+ datasets, 50+ benchmarks, and 100+ models, tools, and systems across text, speech, multimodal, and culturally grounded tasks. We organize resources by linguistic phenomena, domains, and modalities; analyze trends in annotation, evaluation, and model design; and identify persistent challenges such as data sparsity, uneven language coverage, script diversity, and limited cultural and domain generalization. This survey offers a consolidated foundation for equitable, culturally grounded, and scalable NLP research in the Indian linguistic ecosystem. 

---
# StepPO: Step-Aligned Policy Optimization for Agentic Reinforcement Learning 

**Authors**: Daoyu Wang, Qingchuan Li, Mingyue Cheng, Jie Ouyang, Shuo Yu, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.18401)  

**Abstract**: General agents have given rise to phenomenal applications such as OpenClaw and Claude Code. As these agent systems (a.k.a. Harnesses) strive for bolder goals, they demand increasingly stronger agentic capabilities from foundation Large Language Models (LLMs). Agentic Reinforcement Learning (RL) is emerging as a central post-training paradigm for empowering LLMs with these capabilities and is playing an increasingly pivotal role in agent training. Unlike single-turn token-level alignment or reasoning enhancement, as in RLHF and RLVR, Agentic RL targets multi-turn interactive settings, where the goal is to optimize core agentic capabilities such as decision making and tool use while addressing new challenges including delayed and sparse rewards, as well as long and variable context. As a result, the token-centric modeling and optimization paradigm inherited from traditional LLM RL is becoming increasingly inadequate for capturing real LLM agent behavior. In this paper, we present StepPO as a position on step-level Agentic RL. We argue that the conventional token-level Markov Decision Process (MDP) should be advanced to a step-level MDP formulation, and that the step, rather than the token, should be regarded as the proper action representation for LLM agents. We then propose step-level credit assignment as the natural optimization counterpart of this formulation, thereby aligning policy optimization and reward propagation with the granularity of agent decisions. Finally, we discuss the key systems designs required to realize step-level Agentic RL in practice and preliminary experiments provide initial evidence for the effectiveness of this perspective. We hope that the step-aligned, step-level paradigm embodied in StepPO offers the Agentic RL community a useful lens for understanding agent behavior and helps advance LLMs toward stronger general-agent capabilities. 

---
# AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment 

**Authors**: Yixuan Wang, Yue Huang, Hong Qian, Yunzhao Wei, Yifei Ding, Wenkai Wang, Zhi Liu, Zhongjing Huang, Aimin Zhou, Jiajun Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.18398)  

**Abstract**: Creativity has become a core competence in the era of LLMs and human-AI collaboration, underpinning innovation in real-world problem solving. Crucially, the systematic improvement of creativity necessitates scientifically valid assessment instruments. Psychometric research recognizes context-based assessment as an effective way to measure creative thinking. However, high-quality expert-designed contexts remain scarce. Existing LLM-based generators often struggle with insufficient assessment cues, weak narrative coherence, limited stylistic diversity, and poor support for creative thinking. To address these challenges, we propose AlphaContext, an evolutionary tree-based psychometric context generator for creativity assessment. First, the HyperTree Outline Planner formalizes expert-designed outlining as a rule-guided hypertree and performs top-down hierarchical planning. The MCTS-based Context Generator fills the outline via MCTS to balance global structure and local quality. Then, the Evolutionary Context Optimizer evolves contexts with MAP-Elites by repeatedly updating niche elites to jointly improve diversity and quality. Finally, the Assessment-Guided Evolution Refiner simulates virtual participants with diverse styles and recycles weak contexts for further evolution. Experiments show that AlphaContext yields an average improvement of 8% over competitive methods across 6 quality metrics. 

---
# River-LLM: Large Language Model Seamless Exit Based on KV Share 

**Authors**: Yingtao Shen, An Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.18396)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance across diverse domains but are increasingly constrained by high inference latency. Early Exit has emerged as a promising solution to accelerate inference by dynamically bypassing redundant layers. However, in decoder-only architectures, the efficiency of Early Exit is severely bottlenecked by the KV Cache Absence problem, where skipped layers fail to provide the necessary historical states for subsequent tokens. Existing solutions, such as recomputation or masking, either introduce significant latency overhead or incur severe precision loss, failing to bridge the gap between theoretical layer reduction and practical wall-clock speedup. In this paper, we propose River-LLM, a training-free framework that enables seamless token-level Early Exit. River-LLM introduces a lightweight KV-Shared Exit River that allows the backbone's missing KV cache to be naturally generated and preserved during the exit process, eliminating the need for costly recovery operations. Furthermore, we utilize state transition similarity within decoder blocks to predict cumulative KV errors and guide precise exit decisions. Extensive experiments on mathematical reasoning and code generation tasks demonstrate that River-LLM achieves 1.71 to 2.16 times of practical speedup while maintaining high generation quality. 

---
# Understanding the Prompt Sensitivity 

**Authors**: Yang Liu, Chenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18389)  

**Abstract**: Prompt sensitivity, which refers to how strongly the output of a large language model (LLM) depends on the exact wording of its input prompt, raises concerns among users about the LLM's stability and reliability. In this work, we consider LLMs as multivariate functions and perform a first-order Taylor expansion, thereby analyzing the relationship between meaning-preserving prompts, their gradients, and the log probabilities of the model's next token. We derive an upper bound on the difference between log probabilities using the Cauchy-Schwarz inequality. We show that LLMs do not internally cluster similar inputs like smaller neural networks do, but instead disperse them. This dispersing behavior leads to an excessively high upper bound on the difference of log probabilities between two meaning-preserving prompts, making it difficult to effectively reduce to 0. In our analysis, we also show which types of meaning-preserving prompt variants are more likely to introduce prompt sensitivity risks in LLMs. In addition, we demonstrate that the upper bound is strongly correlated with an existing prompt sensitivity metric, PromptSensiScore. Moreover, by analyzing the logit variance, we find that prompt templates typically exert a greater influence on logits than the questions themselves. Overall, our results provide a general interpretation for why current LLMs can be highly sensitive to prompts with the same meaning, offering crucial evidence for understanding the prompt sensitivity of LLMs. Code for experiments is available at this https URL. 

---
# IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters 

**Authors**: Hongwei Zheng, Weiqi Wu, Zhengjia Wang, Guanyu Jiang, Haoming Li, Tianyu Wu, Yongchun Zhu, Jingwu Chen, Feng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18375)  

**Abstract**: Conversational agents, such as ChatGPT and Doubao, have become essential daily assistants for billions of users. To further enhance engagement, these systems are evolving from passive responders to proactive companions. However, existing efforts focus on activation within ongoing dialogues, while overlooking a key real-world bottleneck. In the conversation initiation stage, users may have a vague need but no explicit query intent, creating a first-message barrier where the conversation holds before it begins. To overcome this, we introduce Conversation Starter Generation: generating personalized starters to guide users into conversation. However, unlike in-conversation stages where immediate context guides the response, initiation must operate in a cold-start moment without explicit user intent. To pioneer in this direction, we present IceBreaker that frames human ice-breaking as a two-step handshake: (i) evoke resonance via Resonance-Aware Interest Distillation from session summaries to capture trigger interests, and (ii) stimulate interaction via Interaction-Oriented Starter Generation, optimized with personalized preference alignment and a self-reinforced loop to maximize engagement. Online A/B tests on one of the world's largest conversational agent products show that IceBreaker improves user active days by +0.184% and click-through rate by +9.425%, and has been deployed in production. 

---
# ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 

**Authors**: Qingying Niu, Yuhao Wang, Ruiyang Ren, Bohui Fang, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.18362)  

**Abstract**: Retrieval-augmented generation (RAG) remains unreliable in long-form settings, where retrieved evidence is noisy or contradictory, making it difficult for RAG pipelines to maintain factual consistency. Existing approaches focus on retrieval expansion or verification during generation, leaving conflict resolution entangled with generation. To address this limitation, we propose ArbGraph, a framework for pre-generation evidence arbitration in long-form RAG that explicitly resolves factual conflicts. ArbGraph decomposes retrieved documents into atomic claims and organizes them into a conflict-aware evidence graph with explicit support and contradiction relations. On top of this graph, we introduce an intensity-driven iterative arbitration mechanism that propagates credibility signals through evidence interactions, enabling the system to suppress unreliable and inconsistent claims before final generation. In this way, ArbGraph separates evidence validation from text generation and provides a coherent evidence foundation for downstream long-form generation. We evaluate ArbGraph on two widely used long-form RAG benchmarks, LongFact and RAGChecker, using multiple large language model backbones. Experimental results show that ArbGraph consistently improves factual recall and information density while reducing hallucinations and sensitivity to retrieval noise. Additional analyses show that these gains are evident under conflicting or ambiguous evidence, highlighting the effectiveness of evidence-level conflict resolution for improving the reliability of long-form RAG. The implementation is publicly available at this https URL. 

---
# ComPASS: Towards Personalized Agentic Social Support via Tool-Augmented Companionship 

**Authors**: Zhaopei Huang, Yanfeng Jia, Jiayi Zhao, Xinjie Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.18356)  

**Abstract**: Developing compassionate interactive systems requires agents to not only understand user emotions but also provide diverse, substantive support. While recent works explore empathetic dialogue generation, they remain limited in response form and content, struggling to satisfy diverse needs across users and contexts. To address this, we explore empowering agents with external tools to execute diverse actions. Grounded in the psychological concept of "social support", this paradigm delivers substantive, human-like companionship. Specifically, we first design a dozen user-centric tools simulating various multimedia applications, which can cover different types of social support behaviors in human-agent interaction scenarios. We then construct ComPASS-Bench, the first personalized social support benchmark for LLM-based agents, via multi-step automated synthesis and manual refinement. Based on ComPASS-Bench, we further synthesize tool use records to fine-tune the Qwen3-8B model, yielding a task-specific ComPASS-Qwen. Comprehensive evaluations across two settings reveal that while the evaluated LLMs can generate valid tool-calling requests with high success rates, significant gaps remain in final response quality. Moreover, tool-augmented responses achieve better overall performance than directly producing conversational empathy. Notably, our trained ComPASS-Qwen demonstrates substantial improvements over its base model, achieving comparable performance to several large-scale models. Our code and data are available at this https URL. 

---
# PRISMA: Preference-Reinforced Self-Training Approach for Interpretable Emotionally Intelligent Negotiation Dialogues 

**Authors**: Prajwal Vijay Kajare, Priyanshu Priya, Bikash Santra, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2604.18354)  

**Abstract**: Emotion plays a pivotal role in shaping negotiation outcomes, influencing trust, cooperation, and long-term relationships. Developing negotiation dialog systems that can recognize and respond strategically to emotions is, therefore, essential to create more effective human-centered interactions. Beyond generating emotionally appropriate responses, interpretability - understanding how a system generates a particular emotion-aware response, is critical for fostering reliability and building rapport. Driven by these aspects, in this work, we introduce PRISMA, an interpretable emotionally intelligent negotiation dialogue system targeting two application domains, viz. job interviews and resource allocation. To enable interpretability, we propose an Emotion-aware Negotiation Strategy-informed Chain-of-Thought (ENS-CoT) reasoning mechanism, which mimics human negotiation by perceiving, understanding, using, and managing emotions. Leveraging ENS-CoT, we curate two new datasets: JobNego (for job interview negotiation) and ResNego (for resource allocation negotiation). We then leverage these datasets to develop PRISMA by augmenting self-training with Direct Preference Optimization (DPO), guiding agents toward more accurate, interpretable, and emotionally appropriate negotiation responses. Automatic and human evaluation on JobNego and ResNego datasets demonstrate that PRISMA substantially enhances interpretability and generates appropriate emotion-aware responses, while improving overall negotiation effectiveness. 

---
# HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents 

**Authors**: Shuqi Cao, Jingyi He, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.18349)  

**Abstract**: Long-term conversational large language model (LLM) agents require memory systems that can recover relevant evidence from historical interactions without overwhelming the answer stage with irrelevant context. However, existing memory systems, including hierarchical ones, still often rely solely on vector similarity for retrieval. It tends to produce bloated evidence sets: adding many superficially similar dialogue turns yields little additional recall, but lowers retrieval precision, increases answer-stage context cost, and makes retrieved memories harder to inspect and manage. To address this, we propose HiGMem (Hierarchical and LLM-Guided Memory System), a two-level event-turn memory system that allows LLMs to use event summaries as semantic anchors to predict which related turns are worth reading. This allows the model to inspect high-level event summaries first and then focus on a smaller set of potentially useful turns, providing a concise and reliable evidence set through reasoning, while avoiding the retrieval overhead that would be excessively high compared to vector retrieval. On the LoCoMo10 benchmark, HiGMem achieves the best F1 on four of five question categories and improves adversarial F1 from 0.54 to 0.78 over A-Mem, while retrieving an order of magnitude fewer turns. Code is publicly available at this https URL. 

---
# Multilingual Training and Evaluation Resources for Vision-Language Models 

**Authors**: Daniela Baiamonte, Elena Fano, Matteo Gabburo, Stefano Simonazzi, Leonardo Rigutini, Andrea Zugarini  

**Link**: [PDF](https://arxiv.org/pdf/2604.18347)  

**Abstract**: Vision Language Models (VLMs) achieved rapid progress in the recent years. However, despite their growth, VLMs development is heavily grounded on English, leading to two main limitations: (i) the lack of multilingual and multimodal datasets for training, and (ii) the scarcity of comprehensive evaluation benchmarks across languages. In this work, we address these gaps by introducing a new comprehensive suite of resources for VLMs training and evaluation spanning five European languages (English, French, German, Italian, and Spanish). We adopt a regeneration-translation paradigm that produces high-quality cross-lingual resources by combining curated synthetic generation and manual annotation. Specifically, we build Multi-PixMo, a training corpus obtained regenerating examples from Pixmo pre-existing datasets with permissively licensed models: PixMo-Cap, PixMo-AskModelAnything, and CoSyn-400k. On the evaluation side, we construct a set of multilingual benchmarks derived translating widely used English datasets (MMbench, ScienceQA, MME, POPE, AI2D). We assess the quality of these resources through qualitative and quantitative human analyses, measuring inter-annotator agreement. Additionally, we perform ablation studies to demonstrate the impact of multilingual data, with respect to English only, in VLMs training. Experiments, comprising 3 different models show that using multilingual, multimodal examples for training VLMs aids is consistently beneficial on non-English benchmarks, with positive transfer to English as well. 

---
# FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction 

**Authors**: Adewale Akinfaderin, Nafi Diallo  

**Link**: [PDF](https://arxiv.org/pdf/2604.18328)  

**Abstract**: We present FregeLogic, a hybrid neuro-symbolic system for SemEval-2026 Task 11 (Subtask 1), which addresses syllogistic validity prediction while reducing content effects on predictions. Our approach combines an ensemble of five LLM classifiers, spanning three open-weights models (Llama 4 Maverick, Llama 4 Scout, and Qwen3-32B) paired with varied prompting strategies, with a Z3 SMT solver that serves as a formal logic tiebreaker. The central hypothesis is that LLM disagreement within the ensemble signals likely content-biased errors, where real-world believability interferes with logical judgment. By deferring to Z3's structurally-grounded formal verification on these disputed cases, our system achieves 94.3% accuracy with a content effect of 2.85 and a combined score of 41.88 in nested 5-fold cross-validation on the dataset (N=960). This represents a 2.76-point improvement in combined score over the pure ensemble (39.12), with a 0.9% accuracy gain, driven by a 16% reduction in content effect (3.39 to 2.85). Adopting structured-output API calls for Z3 extraction reduced failure rates from ~22% to near zero, and an Aristotelian encoding with existence axioms was validated against task annotations. Our results suggest that targeted neuro-symbolic integration, applying formal methods precisely where ensemble consensus is lowest, can improve the combined accuracy-plus-content-effect metric used by this task. 

---
# On the Importance and Evaluation of Narrativity in Natural Language AI Explanations 

**Authors**: Mateusz Cedro, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2604.18311)  

**Abstract**: Explainable AI (XAI) aims to make the behaviour of machine learning models interpretable, yet many explanation methods remain difficult to understand. The integration of Natural Language Generation into XAI aims to deliver explanations in textual form, making them more accessible to practitioners. Current approaches, however, largely yield static lists of feature importances. Although such explanations indicate what influences the prediction, they do not explain why the prediction occurs. In this study, we draw on insights from social sciences and linguistics, and argue that XAI explanations should be presented in the form of narratives. Narrative explanations support human understanding through four defining properties: continuous structure, cause-effect mechanisms, linguistic fluency, and lexical diversity. We show that standard Natural Language Processing (NLP) metrics based solely on token probability or word frequency fail to capture these properties and can be matched or exceeded by tautological text that conveys no explanatory content. To address this issue, we propose seven automatic metrics that quantify the narrative quality of explanations along the four identified dimensions. We benchmark current state-of-the-art explanation generation methods on six datasets and show that the proposed metrics separate descriptive from narrative explanations more reliably than standard NLP metrics. Finally, to further advance the field, we propose a set of problem-agnostic XAI Narrative generation rules for producing natural language XAI explanations, so that the resulting XAI Narratives exhibit stronger narrative properties and align with the findings from the linguistic and social science literature. 

---
# Reasoning Models Know What's Important, and Encode It in Their Activations 

**Authors**: Yaniv Nikankin, Martin Tutek, Tomer Ashuach, Jonathan Rosenfeld, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2604.18307)  

**Abstract**: Language models often solve complex tasks by generating long reasoning chains, consisting of many steps with varying importance. While some steps are crucial for generating the final answer, others are removable. Determining which steps matter most, and why, remains an open question central to understanding how models process reasoning. We investigate if this question is best approached through model internals or through tokens of the reasoning chain itself. We find that model activations contain more information than tokens for identifying important reasoning steps. Crucially, by training probes on model activations to predict importance, we show that models encode an internal representation of step importance, even prior to the generation of subsequent steps. This internal representation of importance generalizes across models, is distributed across layers, and does not correlate with surface-level features, such as a step's relative position or its length. Our findings suggest that analyzing activations can reveal aspects of reasoning that surface-level approaches fundamentally miss, indicating that reasoning analyses should look into model internals. 

---
# Exploring Concreteness Through a Figurative Lens 

**Authors**: Saptarshi Ghosh, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18296)  

**Abstract**: Static concreteness ratings are widely used in NLP, yet a word's concreteness can shift with context, especially in figurative language such as metaphor, where common concrete nouns can take abstract interpretations. While such shifts are evident from context, it remains unclear how LLMs understand concreteness internally. We conduct a layer-wise and geometric analysis of LLM hidden representations across four model families, examining how models distinguish literal vs figurative uses of the same noun and how concreteness is organized in representation space. We find that LLMs separate literal and figurative usage in early layers, and that mid-to-late layers compress concreteness into a one-dimensional direction that is consistent across models. Finally, we show that this geometric structure is practically useful: a single concreteness direction supports efficient figurative-language classification and enables training-free steering of generation toward more literal or more figurative rewrites. 

---
# An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal 

**Authors**: Ryo Yoshida, Shinnosuke Isono, Taiga Someya, Yohei Oseki, Tatsuki Kuribayashi  

**Link**: [PDF](https://arxiv.org/pdf/2604.18293)  

**Abstract**: Surprisal theory hypothesizes that the difficulty of human sentence processing increases linearly with surprisal, the negative log-probability of a word given its context. Computational psycholinguistics has tested this hypothesis using language models (LMs) as proxies for human prediction. While surprisal derived from recent neural LMs generally captures human processing difficulty on naturalistic corpora that predominantly consist of simple sentences, it severely underestimates processing difficulty on sentences that require syntactic disambiguation (garden-path effects). This leads to the claim that the processing difficulty of such sentences cannot be reduced to surprisal, although it remains possible that neural LMs simply differ from humans in next-word prediction. In this paper, we investigate whether it is truly impossible to construct a neural LM that can explain garden-path effects via surprisal. Specifically, instead of evaluating off-the-shelf neural LMs, we fine-tune these LMs on garden-path sentences so as to better align surprisal-based reading-time estimates with actual human reading times. Our results show that fine-tuned LMs do not overfit and successfully capture human reading slowdowns on held-out garden-path items; they even improve predictive power for human reading times on naturalistic corpora and preserve their general LM capabilities. These results provide an existence proof for a neural LM that can explain both garden-path effects and naturalistic reading times via surprisal, but also raise a theoretical question: what kind of evidence can truly falsify surprisal theory? 

---
# Where Do Self-Supervised Speech Models Become Unfair? 

**Authors**: Felix Herron, Maja Hjuler, Solange Rossato, Alexandre Allauzen, François Portet  

**Link**: [PDF](https://arxiv.org/pdf/2604.18249)  

**Abstract**: Speech encoder models are known to model members of some speaker groups (SGs) better than others. However, there has been little work in establishing why this occurs on a technological level. To our knowledge, we present the first layerwise fairness analysis of pretrained self-supervised speech encoder models (S3Ms), probing each embedding layer for speaker identification (SID) automatic speech recognition (ASR). We find S3Ms produce embeddings biased against certain SGs for both tasks, starting at the very first latent layers. Furthermore, we find opposite patterns of layerwise bias for SID vs ASR for all models in our study: SID bias is minimized in layers that minimize overall SID error; on the other hand, ASR bias is maximized in layers that minimize overall ASR error. The inverse bias/error relationship for ASR is unaffected when probing S3Ms that are finetuned for ASR, suggesting SG-level bias is established during pretraining and is difficult to remove. 

---
# Negative Advantage Is a Double-Edged Sword: Calibrating Advantage in GRPO for Deep Search 

**Authors**: Jiayi Wu, Ruobing Xie, Zeqian Huang, Lei Jiang, Can Xu, Kangyang Luo, Ming Gao, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.18235)  

**Abstract**: Deep search agents can autonomously initiate multi-turn interactions with search engines, thereby exhibiting strong question-answering capabilities. Such performance critically relies on Group Relative Policy Optimization (GRPO) as its core training algorithm. However, GRPO still faces several challenges in deep search settings. First, there exists a substantial mismatch between the correctness of intermediate steps and the reward signal, causing numerous correct intermediate steps to be incorrectly penalized when the final answer is wrong. Second, training is highly unstable, often resulting in degradation of natural language ability or even catastrophic training collapse. Our analysis attributes these issues to coarse-grained advantage assignment and an imbalance between positive and negative advantages. To address these problems, we propose CalibAdv, an advantage calibration method specifically designed for deep search tasks. Specifically, CalibAdv leverages the correctness of intermediate steps to downscale excessive negative advantages at a fine-grained level. It then rebalances positive and negative advantages in the answer component. Extensive experiments across three models and seven benchmarks demonstrate that CalibAdv improves both model performance and training stability. Our code is available at this https URL. 

---
# Model in Distress: Sentiment Analysis on French Synthetic Social Media 

**Authors**: Pierre-Carl Langlais, Pavel Chizhov, Yannick Detrois, Carlos Rosas Hinostroza, Ivan P. Yamshchikov, Bastien Perroy  

**Link**: [PDF](https://arxiv.org/pdf/2604.18226)  

**Abstract**: Automated analysis of customer feedback on social media is hindered by three challenges: the high cost of annotated training data, the scarcity of evaluation sets, especially in multilingual settings, and privacy concerns that prevent data sharing and reproducibility. We address these issues by developing a generalizable synthetic data generation pipeline applied to a case study on customer distress detection in French public transportation. Our approach utilizes backtranslation with fine-tuned models to generate 1.7 million synthetic tweets from a small seed corpus, complemented by synthetic reasoning traces. We train 600M-parameter reasoners with English and French reasoning that achieve 77-79% accuracy on human-annotated evaluation data, matching or exceeding SOTA proprietary LLMs and specialized encoders. Beyond reducing annotation costs, our pipeline preserves privacy by eliminating the exposure of sensitive user data. Our methodology can be adopted for other use cases and languages. 

---
# Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages 

**Authors**: V.S.D.S.Mahesh Akavarapu, Michael Daniel, Gerhard Jäger  

**Link**: [PDF](https://arxiv.org/pdf/2604.18204)  

**Abstract**: We present a phoneme-level analysis of automatic speech recognition (ASR) for two low-resourced and phonologically complex East Caucasian languages, Archi and Rutul, based on curated and standardized speech-transcript resources totaling approximately 50 minutes and 1 hour 20 minutes of audio, respectively. Existing recordings and transcriptions are consolidated and processed into a form suitable for ASR training and evaluation. We evaluate several state-of-the-art audio and audio-language models, including wav2vec2, Whisper, and Qwen2-Audio. For wav2vec2, we introduce a language-specific phoneme vocabulary with heuristic output-layer initialization, which yields consistent improvements and achieves performance comparable to or exceeding Whisper in these extremely low-resource settings. Beyond standard word and character error rates, we conduct a detailed phoneme-level error analysis. We find that phoneme recognition accuracy strongly correlates with training frequency, exhibiting a characteristic sigmoid-shaped learning curve. For Archi, this relationship partially breaks for Whisper, pointing to model-specific generalization effects beyond what is predicted by training frequency. Overall, our results indicate that many errors attributed to phonological complexity are better explained by data scarcity. These findings demonstrate the value of phoneme-level evaluation for understanding ASR behavior in low-resource, typologically complex languages. 

---
# Multiplication in Multimodal LLMs: Computation with Text, Image, and Audio Inputs 

**Authors**: Samuel G. Balter, Ethan Jerzak, Connor T. Jerzak  

**Link**: [PDF](https://arxiv.org/pdf/2604.18203)  

**Abstract**: Multimodal LLMs can accurately perceive numerical content across modalities yet fail to perform exact multi-digit multiplication when the identical underlying arithmetic problem is presented as numerals, number words, images, or in audio form. Because existing benchmarks often lack systematically paired instances across modalities, it remains difficult to compare genuine arithmetic limits within and across model families. We therefore introduce a controlled multimodal multiplication benchmark that factorially varies digit length, digit sparsity, representation (e.g., numerals vs. number words), and modality (text, rendered images, audio), with paired instances from a reproducible generator. We also define arithmetic load, C, as the product of the total and non-zero digit count as a compact, mechanistically motivated proxy for operation count. Across evaluations, accuracy falls sharply as C grows, often nearing zero by C > 100. Indeed, C remains predictive of performance across modalities and models, with R-squared often > 0.5, nearing the value from more complex measures of arithmetic load that count the number of intermediate arithmetic steps. A separate perception-versus-computation decomposition shows that multimodal degradation is primarily computational rather than perceptual: on matched-perception checks, models are near-perfect (> 99%) across modalities, even when multiplication accuracy drops. Beyond measuring when models fail, we ask which procedures they are predisposed to follow. We introduce a forced-completion loss probe that scores heuristic-specific reasoning prefixes--including columnar multiplication, distributive decomposition, and rounding/compensation. Here, decomposition is favored in both text and vision modalities; heuristic-specific LoRA adapters produce near-orthogonal updates yet degrade accuracy, indicating the base model maintains a well-tuned internal router. 

---
# Linear-Time and Constant-Memory Text Embeddings Based on Recurrent Language Models 

**Authors**: Tobias Grantner, Emanuel Sallinger, Martin Flechl  

**Link**: [PDF](https://arxiv.org/pdf/2604.18199)  

**Abstract**: Transformer-based embedding models suffer from quadratic computational and linear memory complexity, limiting their utility for long sequences. We propose recurrent architectures as an efficient alternative, introducing a vertically chunked inference strategy that enables fast embedding generation with memory usage that becomes constant in the input length once it exceeds the vertical chunk size. By fine-tuning Mamba2 models, we demonstrate their viability as general-purpose text embedders, achieving competitive performance across a range of benchmarks while maintaining a substantially smaller memory footprint compared to transformer-based counterparts. We empirically validate the applicability of our inference strategy to Mamba2, RWKV, and xLSTM models, confirming consistent runtime-memory trade-offs across architectures and establishing recurrent models as a compelling alternative to transformers for efficient embedding generation. 

---
# STaD: Scaffolded Task Design for Identifying Compositional Skill Gaps in LLMs 

**Authors**: Sungeun An, Swanand Ravindra Kadhe, Shailja Thakur, Chad DeLuca, Hima Patel  

**Link**: [PDF](https://arxiv.org/pdf/2604.18177)  

**Abstract**: Benchmarks are often used as a standard to understand LLM capabilities in different domains. However, aggregate benchmark scores provide limited insight into compositional skill gaps of LLMs and how to improve them. To make these weaknesses visible, we propose Scaffolded Task Design (STaD) framework. STaD generates controlled variations of benchmark tasks based on the concept of scaffolding, which introduces structured, incremental support in a step-by-step manner. Rather than inspecting failures individually, this approach enables systematic and scalable probing of model behavior by identifying the specific reasoning skill compositions they lack. Treating the LLM as a black box, our experiments on six models of varying sizes reveal multiple failure points in three reasoning benchmarks and highlight each model's unique and distinct skill gaps. 

---
# Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing 

**Authors**: Ziyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18170)  

**Abstract**: LLMs edit text and code by autoregressively regenerating the full output, even when most tokens appear verbatim in the input. We study Copy-as-Decode, a decoding-layer mechanism that recasts edit generation as structured decoding over a two-primitive grammar: <copy lines="i-j"/> references an input line range, <gen>...</gen> emits new content. A token-level FSM guarantees syntactic validity, and a serving-layer primitive updates the KV cache for each copy span via a single parallel-prefill forward rather than $N$ autoregressive steps -- sharing the parallel-forward kernel of speculative decoding but with input tokens as the draft and program-enforced acceptance replacing probabilistic verification. We report an upper-bound analysis that requires no end-to-end training. (i) Kernel speedup: on Qwen2.5-{1.5B, 7B}, copying $N$ tokens via parallel prefill is $6.8\times$--$303\times$ faster than autoregressive ($N \in [8, 512]$, A100 80GB bf16). (ii) Copy ceiling: on ProbeEdit and HumanEvalPack-Fix (Py/JS), $74$--$98\%$ of gold tokens are reachable under the line-level primitive; composed with the empirical kernel over each corpus's span histogram this yields a closed-form wall-clock bound of $29.0\times / 3.4\times / 4.2\times$ ($13.0\times$ pooled). A token-level extension reaches $91$--$99\%$ coverage with $4.5\times$--$6.5\times$ floors. (iii) Pipeline losslessness: oracle programs round-trip through the deterministic resolver on all $482$ cases, localizing any downstream failure to span selection rather than the mechanism. A perturbation study shows pooled EM drops from $100\%$ to $15.48\%$ under off-by-one noise. A fine-tuning pilot on Qwen2.5-Coder-1.5B lifts HEvalFix-Py EM from $0/33$ (untrained) to $12$--$17\%$, a learnability signal, not a production selector. Batched-serving integration and multi-file coverage are scoped as follow-up. 

---
# Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation 

**Authors**: Ran Zhang, Steffen Eger, Arda Tezcan, Wei Zhao, Simone Paolo Ponzetto, Lieve Macken  

**Link**: [PDF](https://arxiv.org/pdf/2604.18169)  

**Abstract**: Large language models (LLMs) are increasingly used for creative tasks such as literary translation. Yet translational creativity remains underexplored and is rarely evaluated at scale, while source-text comprehension is typically studied in isolation, despite the fact that, in professional translation, comprehension and creativity are tightly intertwined. We address these gaps with a paired-task framework applied to literary excerpts from 11 books. Task 1 assesses source-text comprehension, and Task 2 evaluates translational creativity through Units of Creative Potential (UCPs), such as metaphors and wordplay. Using a scalable evaluation setup that combines expert human annotations with UCP-based automatic scoring, we benchmark 23 models and four creativity-oriented prompts. Our findings show that strong comprehension does not translate into human-level creativity: models often produce literal or contextually inappropriate renderings, with particularly large gaps for the more distant English-Chinese language pair. Creativity-oriented prompts yield only modest gains, and only one model, Mistral-Large, comes close to human-level creativity (0.167 vs. 0.246). Across all model-prompt combinations, only three exceed a creativity score of 0.1, while the rest remain at or near zero. 

---
# MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge 

**Authors**: Sua Lee, Sanghee Park, Jinbae Im  

**Link**: [PDF](https://arxiv.org/pdf/2604.18164)  

**Abstract**: Multimodal Large Language Models (MLLMs) have been increasingly used as automatic evaluators-a paradigm known as MLLM-as-a-Judge. However, their reliability and vulnerabilities to biases remain underexplored. We find that many MLLM judges fail to reliably integrate key visual or textual cues, yielding unreliable evaluations when evidence is missing or mismatched, and exhibiting instability under semantically irrelevant perturbations. To address this, we systematically define Compositional Bias in MLLM-as-a-Judge systems and introduce MM-JudgeBias, a benchmark for evaluating it. MM-JudgeBias introduces controlled perturbations across Query, Image, and Response, and evaluates model behavior via two complementary metrics: Bias-Deviation (BD) for sensitivity and Bias-Conformity (BC) for stability. Our dataset of over 1,800 curated and refined multimodal samples, drawn from 29 source benchmarks, enables a fine-grained diagnosis of nine bias types across diverse tasks and domains. Experiments on 26 state-of-the-art MLLMs reveal systematic modality neglect and asymmetric evaluation tendencies, underscoring the need for more reliable judges. 

---
# FreezeEmpath: Efficient Training for Empathetic Spoken Chatbots with Frozen LLMs 

**Authors**: Yun Hong, Yan Zhou, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.18159)  

**Abstract**: Empathy is essential for fostering natural interactions in spoken dialogue systems, as it enables machines to recognize the emotional tone of human speech and deliver empathetic responses. Recent research has made significant progress in developing empathetic spoken chatbots based on large language models (LLMs). However, several challenges still exist when training such models, including reliance on costly empathetic speech instruction data and a lack of emotional expressiveness in the generated speech. Finetuning LLM with cross-modal empathetic instruction data may also lead to catastrophic forgetting and a degradation of its general capability. To address these challenges, we propose FreezeEmpath, an end-to-end empathetic spoken chatbot trained in a simple and efficient manner. The entire training process relies solely on existing speech instruction data and speech emotion recognition (SER) data, while keeping the LLM's parameters frozen. Experiments demonstrate that FreezeEmpath is able to generate emotionally expressive speech and outperforms other empathetic models in empathetic dialogue, SER, and SpokenQA tasks, demonstrating the effectiveness of our training strategy. 

---
# Depth Registers Unlock W4A4 on SwiGLU: A Reader/Generator Decomposition 

**Authors**: Ziyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18128)  

**Abstract**: We study post-training W4A4 quantization in a controlled 300M-parameter SwiGLU decoder-only language model trained on 5B tokens of FineWeb-Edu, and ask which input-activation sites dominate the error. Naive round-to-nearest W4A4 collapses validation perplexity from FP16 23.6 to 1727. A simple residual-axis training-time intervention -- Depth Registers with a register-magnitude hinge loss (DR+sink) -- reduces this to 119 (about 14x) at matched FP16 PPL and matched zero-shot capacity, and composes with SmoothQuant to 39.9 PPL. The residual ~2 PPL gap to FP16 is the diagnostic core. We decompose W4A4 damage by input-activation site: the five trainable linears in a SwiGLU block split into residual-axis readers (qkv, w1, w3) and block-internal generators (o_proj, w2). Elementary norm arguments show residual-axis magnitude control bounds readers tightly but leaves w2's bilinear input bounded only by the trivial product of factor bounds; empirically, DR+sink collapses reader kurtosis while leaving generators essentially unchanged, and the reader-rescued W4A4 residue is flat at ~0.28 nats across three matched checkpoints with Delta-remove(w2) dominating. We present DR+sink as a training-time probe rather than a deployment proposal: a post-hoc alternative (Per-Linear QuaRot) nearly matches it on the reader axis. Full QuaRot -- adding online per-head value Hadamard plus online w2-input rotation -- does not close the gap either, directly testing the prediction that orthogonal rotation cannot bound the bilinear SwiGLU tail. Claims are specific to our 300M, 5B-token, single-seed setting, and our experiments do not isolate the partition from the hinge. 

---
# TLoRA: Task-aware Low Rank Adaptation of Large Language Models 

**Authors**: Weicheng Lin, Yi Zhang, Jiawei Dang, Liang-Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18124)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a widely adopted parameter-efficient fine-tuning method for large language models, with its effectiveness largely influenced by the allocation of ranks and scaling factors, as well as initialization. Existing LoRA variants typically address only one of these factors, often at the cost of increased training complexity or reduced practical efficiency. In this work, we present Task-aware Low-Rank Adaptation (TLoRA), a unified framework that jointly optimizes initialization and resource allocation at the outset of training. TLoRA introduces a data-driven initialization strategy that aligns the LoRA $A$ matrix with task-relevant subspaces by performing singular value decomposition on the product of pre-trained weights and input activation covariance. After this, the $A$ matrix is frozen, and only the $B$ matrix is trained. Furthermore, TLoRA employs a sensitivity-based importance metric to adaptively allocate ranks and scaling factors across layers under a fixed parameter budget. We conduct extensive experiments that demonstrate TLoRA consistently performs excellently across various tasks, including natural language understanding, commonsense reasoning, math reasoning, code generation, and chat generation, while significantly reducing the number of trainable parameters. 

---
# Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents 

**Authors**: Akriti Jain, Anish Mulay, Divyansh Verma, Aishani Pandey, Pritika Ramu, Aparna Garimella  

**Link**: [PDF](https://arxiv.org/pdf/2604.18122)  

**Abstract**: Decision-making is a cognitively intensive task that requires synthesizing relevant information from multiple unstructured sources, weighing competing factors, and incorporating subjective user preferences. Existing methods, including large language models and traditional decision-support systems, fall short: they often overwhelm users with information or fail to capture nuanced preferences accurately. We present Decisive, an interactive decision-making framework that combines document-grounded reasoning with Bayesian preference inference. Our approach grounds decisions in an objective option-scoring matrix extracted from source documents, while actively learning a user's latent preference vector through targeted elicitation. Users answer pairwise tradeoff questions adaptively selected to maximize information gain over the final decision. This process converges efficiently, minimizing user effort while ensuring recommendations remain transparent and personalized. Through extensive experiments, we demonstrate that our approach significantly outperforms both general-purpose LLMs and existing decision-making frameworks achieving up to 20% improvement in decision accuracy over strong baselines across domains. 

---
# Retrieval-Augmented Multimodal Model for Fake News Detection 

**Authors**: Yiheng Li, Weihai Lu, Hanyi Yu, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18112)  

**Abstract**: In recent years, multimodal multidomain fake news detection has garnered increasing attention. Nevertheless, this direction presents two significant challenges: (1) Failure to Capture Cross-Instance Narrative Consistency: existing models usually evaluate each news in isolation, fail to capture cross-instance narrative consistency, and thus struggle to address the spread of cluster based fake news driven by social media; (2) Lack of Domain Specific Knowledge for Reasoning: conventional models, which rely solely on knowledge encoded in their parameters during training, struggle to generalize to new or data-scarce domains (e.g., emerging events or niche topics). To tackle these challenges, we introduce Retrieval-Augmented Multimodal Model for Fake News Detection (RAMM). First, RAMM employs a Multimodal Large Language Model (MLLM) as its backbone to capture cross-modal semantic information from news samples. Second, RAMM incorporates an Abstract Narrative Alignment Module. This component adaptively extracts abstract narrative consistency from diverse instances across distinct domains, aggregates relevant knowledge, and thereby enables the modeling of high-level narrative information. Finally, RAMM introduces a Semantic Representation Alignment Module, which aligns the model's decision-making paradigm with that of humans - specifically, it shifts the model's reasoning process from direct inference on multimodal features to an instance-based analogical reasoning process. Extensive experimental results on three public datasets validate the efficacy of our proposed approach. Our code is available at the following link: this https URL 

---
# FLiP: Towards understanding and interpreting multimodal multilingual sentence embeddings 

**Authors**: Santosh Kesiraju, Bolaji Yusuf, Šimon Sedláček, Oldřich Plchot, Petr Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2604.18109)  

**Abstract**: This paper presents factorized linear projection (FLiP) models for understanding pretrained sentence embedding spaces. We train FLiP models to recover the lexical content from multilingual (LaBSE), multimodal (SONAR) and API-based (Gemini) sentence embedding spaces in several high- and mid-resource languages. We show that FLiP can recall more than 75% of lexical content from the embeddings, significantly outperforming existing non-factorized baselines. Using this as a diagnostic tool, we uncover the modality and language biases across the selected sentence encoders and provide practitioners with intrinsic insights about the encoders without relying on conventional downstream evaluation tasks. Our implementation is public this https URL. 

---
# Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion 

**Authors**: Chen Zhang, Jiuheng Lin, Zhiyuan Liao, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.18106)  

**Abstract**: Adapting large language models (LLMs) to low-resource languages (LRLs) is constrained by the scarcity of task data and computational resources. Although Proxy Tuning offers a logit-level strategy for introducing scaling effects, it often fails in LRL settings because the large model's weak LRL competence might overwhelm the knowledge of specialized smaller models. We thus propose TriMix, a test-time logit fusion framework that dynamically balances capabilities from three different sources: LRL competence from a continually pretrained small model, task competence from high-resource language instruction tuning, and the scaling benefits of large models. It is data- and compute-efficient, requiring no LRL task annotations, and only continual pretraining on a small model. Experiments across four model families and eight LRLs show that TriMix consistently outperforms single-model baselines and Proxy Tuning. Our analysis reveals that prioritizing the small LRL-specialized model's logits is crucial for success, challenging the prevalent large-model-dominant assumption. 

---
# Culture-Aware Humorous Captioning: Multimodal Humor Generation across Cultural Contexts 

**Authors**: Run Xu, Lu Li, Rongzhao Zhang, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18091)  

**Abstract**: Recent multimodal large language models have shown promising ability in generating humorous captions for images, yet they still lack stable control over explicit cultural context, making it difficult to jointly maintain image relevance, contextual appropriateness, and humor quality under a specified cultural background. To address this limitation, we introduce a new multimodal generation task, culture-aware humorous captioning, which requires a model to generate a humorous caption conditioned on both an input image and a target cultural context. Captions generated under different cultural contexts are not expected to share the same surface form, but should remain grounded in similar visual situations or humorous this http URL support this task, we establish a six-dimensional evaluation framework covering image relevance, contextual fit, semantic richness, reasonableness, humor, and creativity. We further propose a staged alignment framework that first initializes the model with high-resource supervision under the Western cultural context, then performs multi-dimensional preference alignment via judge-based GRPO with a Degradation-aware Prototype Repulsion Constraint to mitigate reward hacking in open-ended generation, and finally adapts the model to the Eastern cultural context with a small amount of supervision. Experimental results show that our method achieves stronger overall performance under the proposed evaluation framework, with particularly large gains in contextual fit and a better balance between image relevance and humor under cultural constraints. 

---
# Mix and Match: Context Pairing for Scalable Topic-Controlled Educational Summarisation 

**Authors**: Nathikan Yodthapa, Thanapong Intharah, Sahan Bulathwela  

**Link**: [PDF](https://arxiv.org/pdf/2604.18087)  

**Abstract**: Topic-controlled summarisation enables users to generate summaries focused on specific aspects of source documents. This paper investigates a data augmentation strategy for training small language models (sLMs) to perform topic-controlled summarisation. We propose a pairwise data augmentation method that combines contexts from different documents to create contrastive training examples, enabling models to learn the relationship between topics and summaries more effectively. Using the SciTLDR dataset enriched with Wikipedia-derived topics, we systematically evaluate how augmentation scale affects model performance. Results show consistent improvements in win rate and semantic alignment as the augmentation scale increases, while the amount of real training data remains fixed. Consequently, a T5-base model trained with our augmentation approach achieves competitive performance relative to larger models, despite using significantly fewer parameters and substantially fewer real training examples. 

---
# Modeling Human Perspectives with Socio-Demographic Representations 

**Authors**: Leixin Zhang, Cagri Coltekin  

**Link**: [PDF](https://arxiv.org/pdf/2604.18069)  

**Abstract**: Humans often hold different perspectives on the same issues. In many NLP tasks, annotation disagreement can reflect valid subjective perspectives. Modeling annotator perspectives and understanding their relationship with other human factors, such as socio-demographic attributes, have received increasing attention. Prior work typically focuses on single demographic factors or limited combinations. However, in real-world settings, annotator perspectives are shaped by complex social contexts, and finer-grained socio-demographic attributes can better explain human perspectives. In this work, we propose Socio-Contrastive Learning, a method that jointly models annotator perspectives while learning socio-demographic representations. Our method provides an effective approach for the fusion of socio-demographic features and textual representations to predict annotator perspectives, outperforming standard concatenation-based methods. The learned representations further enable analysis and visualization of how demographic factors relate to variation in annotator perspectives. Our code is available at GitHub: this https URL 

---
# JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew 

**Authors**: Itay Razumenko, Arnon Sturm, Nir Grinberg  

**Link**: [PDF](https://arxiv.org/pdf/2604.18041)  

**Abstract**: Despite significant advances in large language models, personalizing them for individual decision-makers remains an open problem. Here, we introduce a synthetic-organic supervision pipeline that transforms raw judicial decisions into instruction-tuning data, enabling parameter-efficient fine-tuning of personalized models for individual judges in low-resource settings. We compare our approach to state-of-the-art personalization techniques across three different tasks and settings. The results show that Causal Language Modeling followed by synthetically generated instruction-tuning significantly outperforms all other baselines, providing significant improvements across lexical, stylistic, and semantic similarity. Notably, our model-generated outputs are indistinguishable from the reasoning of human judges, highlighting the viability of efficient personalization, even in low-resource settings. 

---
# SignDPO: Multi-level Direct Preference Optimisation for Skeleton-based Gloss-free Sign Language Translation 

**Authors**: Muxin Pu, Xiao-Ming Wu, Mei Kuan Lim, Chun Yong Chong, Wei Li, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2604.18034)  

**Abstract**: We present SignDPO, a novel multi-level Direct Preference Optimisation (DPO) framework designed to enhance the alignment of skeleton-based Sign Language Translation. While current skeleton-based models have made significant progress using Maximum Likelihood Estimation, they are primarily constrained by an imitation-based paradigm that lacks discriminative sensitivity to the fine-grained spatio-temporal nuances of sign language, often leading to semantic drift. To address this, SignDPO shifts the optimisation goal from simple sequence mimicry to structured preference alignment across spatial, temporal, and linguistic dimensions. Our framework involves three key designs. First, we introduce a hierarchical perturbation strategy to construct spatial and temporal non-preferred samples at both global and local granularities automatically. Second, we propose a self-guiding mechanism that leverages decoder cross-attention scores to identify and perturb semantically salient skeletal regions, forcing the model to distinguish genuine sign signals from structural distortions. Third, we establish an automated language-level preference generator by fine-tuning a dedicated perturbation model, capturing complex output-level failure modes without manual annotation. Extensive experiments on three widely adopted benchmarks, CSL-Daily, How2Sign, and OpenASL, demonstrate that SignDPO consistently outperforms state-of-the-art gloss-free methods and even rivals established gloss-based ones. Our results suggest that multi-level preference alignment is a powerful paradigm for bridging the gap between high-entropy skeletal trajectories and discrete linguistic semantics. 

---
# How Creative Are Large Language Models in Generating Molecules? 

**Authors**: Wen Tao, Yiwei Wang, Peng Zhou, Bryan Hooi, Wanlong Fang, Tianle Zhang, Xiao Luo, Yuansheng Liu, Alvin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2604.18031)  

**Abstract**: Molecule generation requires satisfying multiple chemical and biological constraints while searching a large and structured chemical space. This makes it a non-binary problem, where effective models must identify non-obvious solutions under constraints while maintaining exploration to improve success by escaping local optima. From this perspective, creativity is a functional requirement in molecular generation rather than an aesthetic notion. Large language models (LLMs) can generate molecular representations directly from natural language prompts, but it remains unclear what type of creativity they exhibit in this setting and how it should be evaluated. In this work, we study the creative behavior of LLMs in molecular generation through a systematic empirical evaluation across physicochemical, ADMET, and biological activity tasks. We characterize creativity along two complementary dimensions, convergent creativity and divergent creativity, and analyze how different factors shape these behaviors. Our results indicate that LLMs exhibit distinct patterns of creative behavior in molecule generation, such as an increase in constraint satisfaction when additional constraints are imposed. Overall, our work is the first to reframe the abilities required for molecule generation as creativity, providing a systematic understanding of creativity in LLM-based molecular generation and clarifying the appropriate use of LLMs in molecular discovery pipelines. 

---
# Employing General-Purpose and Biomedical Large Language Models with Advanced Prompt Engineering for Pharmacoepidemiologic Study Design 

**Authors**: Xinyao Zhang, Nicole Sonne Heckmann, Manuela Del Castillo Suero, Francesco Paolo Speca, Maurizio Sessa  

**Link**: [PDF](https://arxiv.org/pdf/2604.17988)  

**Abstract**: Background: The potential of large language models (LLMs) to automate and support pharmacoepidemiologic study design is an emerging area of interest, yet their reliability remains insufficiently characterized. General-purpose LLMs often display inaccuracies, while the comparative performance of specialized biomedical LLMs in this domain remains unknown. Methods: This study evaluated general-purpose LLMs (GPT-4o and DeepSeek-R1) versus biomedically fine-tuned LLMs (QuantFactory/Bio-Medical-Llama-3-8B-GGUF and Irathernotsay/qwen2-1.5B-medical_qa-Finetune) using 46 protocols (2018-2024) from the HMA-EMA Catalogue and Sentinel System. Performance was assessed across relevance, logic of justification, and ontology-code agreement across multiple coding systems using Least-to-Most (LTM) and Active Prompting strategies. Results: GPT-4o and DeepSeek-R1 paired with LTM prompting achieved the highest relevance and logic of justification scores, with GPT-4o-LTM reaching a median relevance score of 4 in 8 of 9 questions for HMA-EMA protocols. Biomedical LLMs showed lower relevance overall and frequently generated insufficient justification. All LLMs demonstrated limited proficiency in ontology-code mapping, although LTM provided the most consistent improvements in reasoning stability. Conclusion: Off-the-shelf general-purpose LLMs currently offer superior support for pharmacoepidemiologic design compared to biomedical LLMs. Prompt strategy strongly influenced LLM performance. 

---
# ltzGLUE: Luxembourgish General Language Understanding Evaluation 

**Authors**: Alistair Plum, Felicia Körner, Anne-Marie Lutgen, Laura Bernardy, Fred Philippy, Emilia Milano, Nils Rehlinger, Cédric Lothritz, Tharindu Ranasinghe, Barbara Plank, Christoph Purschke  

**Link**: [PDF](https://arxiv.org/pdf/2604.17976)  

**Abstract**: This paper presents ltzGLUE, the first Natural Language Understanding (NLU) benchmark for Luxembourgish (LTZ) based on the popular GLUE benchmark for English. Although NLU tasks are available for many European languages nowadays, LTZ is one of the official national languages that is often overlooked. We construct new tasks and reuse existing ones to introduce the first official NLU benchmark and accompanying evaluation of encoder models for the language. Our tasks include common natural language processing tasks in binary and multi-class classification settings, including named entity recognition, topic classification, and intent classification. We evaluate various pre-trained language models for LTZ to present an overview of the current capabilities of these models on the LTZ language. 

---
# Modeling Multiple Support Strategies within a Single Turn for Emotional Support Conversations 

**Authors**: Jie Zhu, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, Jinsong Su, Chi Zhang, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2604.17972)  

**Abstract**: Emotional Support Conversation (ESC) aims to assist individuals experiencing distress by generating empathetic and supportive dialogue. While prior work typically assumes that each supporter turn corresponds to a single strategy, real-world supportive communication often involves multiple strategies within a single utterance. In this paper, we revisit the ESC task by formulating it as multi-strategy utterance generation, where each utterance may contain one or more strategy-response pairs. We propose two generation methods: All-in-One, which predicts all strategy-response pairs in a single decoding step, and One-by-One, which iteratively generates strategy-response pairs until completion. Both methods are further enhanced with cognitive reasoning guided by reinforcement learning to improve strategy selection and response composition. We evaluate our models on the ESConv dataset under both utterance-level and dialogue-level settings. Experimental results show that our methods effectively model multi-strategy utterances and lead to improved supportive quality and dialogue success. To our knowledge, this work provides the first systematic empirical evidence that allowing multiple support strategies within a single utterance is both feasible and beneficial for emotional support conversations. All code and data will be publicly available at this https URL. 

---
# Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards 

**Authors**: Raffaele Pisano, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2604.17957)  

**Abstract**: Process Reward Models (PRMs) have emerged as a powerful tool for providing step-level feedback when evaluating the reasoning of Large Language Models (LLMs), which frequently produce chains of thought (CoTs) containing errors even when the final answer is correct. However, existing PRM datasets remain expensive to construct, prone to annotation errors, and predominantly limited to the mathematical domain. This work introduces a novel and scalable approach to PRM dataset generation based on planning logical problems expressed in the Planning Domain Definition Language (PDDL). Using this method, we generate a corpus of approximately one million reasoning steps across various PDDL domains and use it to train PRMs. Experimental results show that augmenting widely-used PRM training datasets with PDDL-derived data yields substantial improvements in both mathematical and non-mathematical reasoning, as demonstrated across multiple benchmarks. These findings indicate that planning problems constitute a scalable and effective resource for generating robust, precise, and fine-grained training data for PRMs, going beyond the classical mathematical sources that dominate this field. 

---
# ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering 

**Authors**: Yindong Zhang, Wenmian Yang, Yiquan Zhang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.17944)  

**Abstract**: Developing agents capable of navigating fragmented, multi-source information remains challenging, primarily due to the scarcity of benchmarks reflecting hybrid workflows combining database querying with external APIs. To bridge this gap, we introduce ReCoQA, a large-scale benchmark of 29,270 real-estate instances featuring machine-verifiable supervision for intermediate steps, including structured intent labels, SQL queries, and API calls. Complementarily, we propose HIRE-Agent, a hierarchical framework instantiating an understand-plan-execute architecture as a strong baseline. By orchestrating a Front-end parser, a planning Supervisor, and execution Specialists, HIRE-Agent effectively integrates heterogeneous evidence. Extensive experiments demonstrate that HIRE-Agent constitutes a strong baseline and substantiates the necessity of hierarchical collaboration for complex, real-world reasoning tasks. 

---
# Domain-oriented RAG Assessment (DoRA): Synthetic Benchmarking for RAG-based Question Answering on Defense Documents 

**Authors**: Bao Gia Doan, Aditya Joshi, Pantelis Elinas, Aarya Bodhankar, Oscar Leslie, Tom Marchant, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2604.17943)  

**Abstract**: Open-domain RAG benchmarks over public corpora can overestimate deployment performance due to pretraining overlap and weak attribution requirements. We present DoRA (Domain-oriented RAG Assessment), a domain-grounded benchmark built from defense documents that pairs synthetic, intent-conditioned QA (question answering) with auditable evidence passages for attribution. DoRA covers five question types (find, explain, summarize, generate, provide) and contains 6.5K curated instances. In end-to-end evaluation with a fixed dense retriever, general-purpose Language Models (LMs) perform similarly, while a model trained on DoRA (DoRA SFT) yields large gains over the base model (Llama3.1-8B-Instruct): up to 26% improvement in QA task success, while reducing the hallucination rate by 47% in RAG faithfulness scores, supporting contamination-aware regression testing under domain shift. 

---
# Heterogeneity in Formal Linguistic Competence of Language Models: Is Data the Real Bottleneck? 

**Authors**: H S V N S Kowndinya Renduchintala, Sumit Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2604.17930)  

**Abstract**: Large Language Models (LLMs) exhibit a puzzling disparity in their formal linguistic competence: while they learn some linguistic phenomena with near-perfect mastery, they often perform below chance on others, even after training on trillions of tokens. In this work, we investigate whether these failures stem from inherent architectural limitations or simply the scarcity of these specific grammatical constructions in web-scale corpora. We pre-train simple GPT-2 Small (124M) models on a 100M-token random sample of the FineWeb corpus and intervene by injecting a minimal amount (1%) of synthetic data targeting specific linguistic phenomena. We find that this targeted intervention substantially improves model performance in 8 out of the 9 worst-performing BLiMP paradigms - notably the accuracy on a specific paradigm, only_npi_scope, surges from 20.9% to 69.4%. Furthermore, we observe that these interventions generally preserve or slightly improve aggregate performance. However, while we also identify a resistant phenomenon, principle_A_c_command, whose performance remains below chance even after our data augmentation, our findings do serve as an optimistic existence proof that even small language models can substantially improve on those linguistic phenomena on which models typically perform poorly, provided the pre-training data contains sufficient exposure to them. This suggests that efforts towards human-scale language modeling may benefit greatly by focusing on data composition. The code to reproduce our results is open-sourced at this https URL. 

---
# Automatic Slide Updating with User-Defined Dynamic Templates and Natural Language Instructions 

**Authors**: Kun Zhou, Jiakai He, Wenmian Yang, Zhensheng Wang, Yiquan Zhang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.17894)  

**Abstract**: Presentation slides are a primary medium for data-driven reporting, yet keeping complex, analytics-style decks up to date remains labor-intensive. Existing automation methods mostly follow fixed template filling and cannot support dynamic updates for diverse, user-authored slide decks. We therefore define "Dynamic Slide Update via Natural Language Instructions on User-provided Templates" and introduce DynaSlide, a large-scale benchmark with 20,036 real-world instruction-execution triples (source slide, user instruction, target slide) grounded in a shared external database and built from business reporting slides under bring-your-own-template (BYO-template) conditions. To tackle this task, we propose SlideAgent, an agent-based framework that combines multimodal slide parsing, natural language instruction grounding, and tool-augmented reasoning for tables, charts, and textual conclusions. SlideAgent updates content while preserving layout and style, providing a strong reference baseline on DynaSlide. We further design end-to-end and component-level evaluation protocols that reveal key challenges and opportunities for future research. The dataset and code are available at this https URL. 

---
# Latent Preference Modeling for Cross-Session Personalized Tool Calling 

**Authors**: Yejin Yoon, Minseo Kim, Taeuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.17886)  

**Abstract**: Users often omit essential details in their requests to LLM-based agents, resulting in under-specified inputs for tool use. This poses a fundamental challenge for tool-augmented agents, as API execution typically requires complete arguments, highlighting the need for personalized tool calling. To study this problem, we introduce MPT, a benchmark comprising 265 multi-session dialogues that cover three challenges: Preference Recall, Preference Induction, and Preference Transfer. We also propose PRefine, a test-time memory-augmented method that represents user preferences as evolving hypotheses. Through a generate--verify--refine loop, it extracts reusable constraints from history and improves tool-calling accuracy while using only 1.24% of the tokens required by full-history prompting. These results indicate that robust personalization in agentic systems depends on memory that captures the reasons behind user choices, not just the choices themselves. 

---
# GraSP: Graph-Structured Skill Compositions for LLM Agents 

**Authors**: Tianle Xia, Lingxiang Hu, Yiding Sun, Ming Xu, Lan Xu, Siying Wang, Wei Xu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17870)  

**Abstract**: Skill ecosystems for LLM agents have matured rapidly, yet recent benchmarks show that providing agents with more skills does not monotonically improve performance -- focused sets of 2-3 skills outperform comprehensive documentation, and excessive skills actually hurt. The bottleneck has shifted from skill availability to skill orchestration: agents need not more skills, but a structural mechanism to select, compose, and execute them with explicit causal dependencies. We propose GraSP, the first executable skill graph architecture that introduces a compilation layer between skill retrieval and execution. GraSP transforms flat skill sets into typed directed acyclic graphs (DAGs) with precondition-effect edges, executes them with node-level verification, and performs locality-bounded repair through five typed operators -- reducing replanning from O(N) to O(d^h). Across ALFWorld, ScienceWorld, WebShop, and InterCode with eight LLM backbones, GraSP outperforms ReAct, Reflexion, ExpeL, and flat skill baselines in every configuration, improving reward by up to +19 points over the strongest baseline while cutting environment steps by up to 41%. GraSP's advantage grows with task complexity and is robust to both skill over-retrieval and quality degradation, confirming that structured orchestration -- not larger skill libraries -- is the key to reliable agent execution. 

---
# Latent Abstraction for Retrieval-Augmented Generation 

**Authors**: Ha Lan N.T, Minh-Anh Nguyen, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.17866)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a standard approach for enhancing large language models (LLMs) with external knowledge, mitigating hallucinations, and improving factuality. However, existing systems rely on generating natural language queries at each hop and maintaining a strict architectural separation between retriever and generator, preventing them from leveraging the full representational capacity of the LLM. We propose \textbf{LAnR} (Latent Abstraction for RAG), a unified framework in which a single LLM jointly performs encoding, retrieval, and generation entirely within its own latent space. Rather than generating textual queries, LAnR produces dense retrieval vectors from the hidden states of a designated \texttt{[PRED]} token and uses them to match against encoded document representations from the same model. Furthermore, LAnR adaptively decides when sufficient evidence has been retrieved using a lightweight MLP control head over those same hidden states, eliminating both the separate retriever and explicit token-level stopping reasoning. This design is motivated by our empirical observation that answer token entropy reliably signals retrieval sufficiency. Extensive experiments on six QA benchmarks spanning single-hop and multi-hop settings demonstrate that LAnR outperforms existing RAG methods, while achieving improved inference efficiency through reduced number of retrieval calls and tighter model integration. 

---
# On the Emergence of Syntax by Means of Local Interaction 

**Authors**: Zichao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.17857)  

**Abstract**: Can syntactic processing emerge spontaneously from purely local interaction? We present a concrete instance on a minimal system: an 18,658-parameter two-dimensional neural cellular automaton (NCA), supervised by nothing more than a 1-bit boundary signal, is trained on the membership problem of an arithmetic-expression grammar. After training, its internal $L \times L$ grid spontaneously self-organizes into an ordered, spatially extended representation that we name Proto-CKY. This representation satisfies three operational criteria for syntactic processing: expressive power beyond the regular languages, structural generalization beyond the training distribution, and an internal organization quantitatively aligned with grammatical structure (Pearson $r \approx 0.71$). It emerges independently on four context-free grammars and regenerates spontaneously after perturbation. Proto-CKY is functionally aligned with the CKY algorithm but formally distinct from it: it is a physical prototype, a concrete instantiation of a mathematical ideal on a physical substrate, and the systematic distance between the two carries information about the substrate itself. 

---
# QuickScope: Certifying Hard Questions in Dynamic LLM Benchmarks 

**Authors**: Taylor Lundy, Narun K. Raman, Kevin Leyton-Brown  

**Link**: [PDF](https://arxiv.org/pdf/2604.17842)  

**Abstract**: LLM benchmarks are increasingly dynamic: instead of containing a fixed set of questions, they define templates and parameters that can generate an effectively unlimited number of question variants. This flexibility is valuable, but it makes evaluation expensive -- especially when the goal is not just determining an average score, but reliably identifying a model's weak spots. This paper introduces a new methodology for identifying hard questions in dynamic benchmarks. It leverages COUP, a recent Bayesian optimization algorithm (Graham, Velez & Leyton-Brown, 2026), after introducing several substantive modifications to make the algorithm suitable for practical LLM pipelines. We also wrap it in a tool that supports flexible choices of datasets and utility functions, enabling users to target the kinds of questions they care about (e.g., low-accuracy questions; questions that are unusually hard relative to their measured complexity). In experiments across a range of benchmarks, we show that our method, dubbed $\texttt{QuickScope}$, discovers truly difficult questions more sample efficiently than standard baselines, while also reducing false positives from noisy outcomes. 

---
# How Non-Linguistic Is the Indus Sign System? A Synthetic-Baseline Scorecard 

**Authors**: Ashish Nair  

**Link**: [PDF](https://arxiv.org/pdf/2604.17828)  

**Abstract**: Whether the Indus Valley sign system (c. 2600-1900 BCE) encodes spoken language has been debated for decades. This paper introduces a multi-metric discrimination framework that tests the observed Indus corpus against two kinds of computer-generated non-linguistic baseline -- one mimicking a heraldic emblem system, the other an administrative coding system -- each calibrated with Zipfian frequency distributions, positional constraints, and bigram dependencies derived from six attested non-linguistic corpora. The scorecard evaluates four properties central to the Farmer-Sproat-Witzel (2004) critique: text brevity, repeated formulaic phrases, hapax legomenon rate, and positional rigidity. Applying this framework to 1,916 deduplicated inscriptions (584 unique signs, 11,110 tokens) from the ICIT/Yajnadevam digitization, we find that the Indus corpus does not match either baseline cleanly. Across the four metrics examined, the Indus corpus occupies an intermediate position relative to the two baseline families, matching neither cleanly. Neither a heraldic nor an administrative generator can reproduce all four properties at once. We also compare against seven real-world non-linguistic corpora including Sproat's (2014) datasets, finding that no attested non-linguistic system reproduces the full Indus statistical profile either. We replicate key prior results including a Zipf slope of -1.49 and conditional entropy of 3.23 bits. All code and data are publicly available. 

---
# Learning to Seek Help: Dynamic Collaboration Between Small and Large Language Models 

**Authors**: Hang Zeng, Xiangyu Liu, Yong Hu, Chaoyue Niu, Jiarui Zhang, Shaojie Tang, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.17827)  

**Abstract**: Large language models (LLMs) offer strong capabilities but raise cost and privacy concerns, whereas small language models (SLMs) facilitate efficient and private local inference yet suffer from limited capacity. To synergize the complementary strengths, we introduce a dynamic collaboration framework, where an SLM learns to proactively decide how to request an LLM during multi-step reasoning, while the LLM provides adaptive feedback instead of acting as a passive tool. We further systematically investigate how collaboration strategies are shaped by SLM and LLM capabilities as well as efficiency and privacy constraints. Evaluation results reveal a distinct scaling effect: stronger SLMs become more self-reliant, while stronger LLMs enable fewer and more informative interactions. In addition, the learned dynamic collaboration strategies significantly outperform static pipelines and standalone inference, and transfer robustly to unseen LLMs. 

---
# PDDL-Mind: Large Language Models are Capable on Belief Reasoning with Reliable State Tracking 

**Authors**: Wang Bill Zhu, Qiutong Tony Yi, Robin Jia, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2604.17819)  

**Abstract**: Large language models (LLMs) perform substantially below human level on existing theory-of-mind (ToM) benchmarks, even when augmented with chain-of-thought prompting or probabilistic belief updates. We argue that these failures primarily arise from unreliable implicit state tracking rather than limitations in high-level reasoning. We introduce PDDL-Mind, a neuro-symbolic framework that decouples environment state evolution from belief inference. By translating narrative descriptions into explicit states and actions expressed in Planning Domain Definition Language (PDDL), and by verifying action-induced state transitions against a predefined domain, PDDL-Mind provides LLMs with a logically consistent and explicit representation of world states for ToM tasks. Experiments on MMToM-QA, MuMA and FanToM show that PDDL-Mind achieves over 5% absolute accuracy gain over the best existing state-of-the-art method on ToM benchmark questions. 

---
# Bridging the Reasoning Gap in Vietnamese with Small Language Models via Test-Time Scaling 

**Authors**: Bui The Trung, Do Minh Duc, Nguyen Van Vinh, Bui Nguyen Quoc Trinh  

**Link**: [PDF](https://arxiv.org/pdf/2604.17794)  

**Abstract**: The democratization of ubiquitous AI hinges on deploying sophisticated reasoning capabilities on resource-constrained devices. However, Small Language Models (SLMs) often face a "reasoning gap", particularly in non-English languages like Vietnamese, where they struggle to maintain coherent chains of thought. This paper investigates Test-Time Scaling strategies for the Qwen3-1.7B architecture within the context of Vietnamese Elementary Mathematics. We introduce Vi-S1K, a high-fidelity reasoning dataset localized via a Gemini 2.5 Flash-Lite powered pipeline, and Vi-Elementary-Bench, a dual-resource benchmark for rigorous evaluation. Using an LLM-as-a-Judge protocol, we reveal that the base model possesses robust latent knowledge (Accuracy: 4.05/5.00) but suffers from a severe "formatting gap" in communication. Supervised Fine-Tuning (SFT) acts as a critical "reasoning unlocker", yielding a 77% improvement in Explanation Quality and bridging the gap between raw calculation and pedagogical coherence. Furthermore, our analysis of prompting strategies uncovers a significant trade-off: structured frameworks like ReAct impose a "cognitive tax" on the 1.7B parameter capacity, degrading performance relative to pure Chain-of-Thought (CoT) combined with Self-Consistency. These findings establish a deployment hierarchy for SLMs, demonstrating that SFT combined with simplified test-time scaling is superior to complex agentic workflows for edge-based reasoning. 

---
# Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens 

**Authors**: Seunghee Koh, Sunghyun Baek, Youngdong Kim, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.17785)  

**Abstract**: Unlearning in large language models (LLMs) has emerged as a promising safeguard against adversarial behaviors. When the forgetting loss is applied uniformly without considering token-level semantic importance, model utility can be unnecessarily degraded. Recent studies have explored token-wise loss regularizers that prioritize informative tokens, but largely rely on ground-truth confidence or external linguistic parsers, which limits their ability to capture contextual information or the model's overall predictive state. Intuitively, function words like "the" primarily serve syntactic roles and are highly predictable with little ambiguity, but informative words admit multiple plausible alternatives with greater uncertainty. Based on this intuition, we propose Entropy-guided Token Weighting (ETW), a token-level unlearning regularizer that uses entropy of the predictive distribution as a proxy for token informativeness. We demonstrate that informative tokens tend to have higher entropy, whereas structural tokens tend to have lower entropy. This behavior enables ETW to achieve more effective unlearning while better preserving model utility than existing token-level approaches. 

---
# SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks 

**Authors**: Mohammadtaher Safarzadeh, Hitesh Laxmichand Patel, Afshin Orojlooyjadid, Graham Horwood, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.17771)  

**Abstract**: Large language models (LLMs) have achieved strong performance on natural language to SQL (NL2SQL) benchmarks, yet their reported accuracy may be inflated by contamination from benchmark queries or structurally similar patterns seen during training. We introduce SPENCE (Syntactic Probing and Evaluation of NL2SQL Contamination Effects), a controlled syntactic probing framework for detecting and quantifying such contamination. SPENCE systematically generates syntactic variants of test queries for four widely used NL2SQL datasets-Spider, SParC, CoSQL, and the newer BIRD benchmark. We use SPENCE to evaluate multiple high-capacity LLMs under execution-based scoring. For each model, we measure changes in execution accuracy across increasing levels of syntactic divergence and quantify rank sensitivity using Kendall's tau with bootstrap confidence intervals. By aligning these robustness trends with benchmark release dates, we observe a clear temporal gradient: older benchmarks such as Spider exhibit the strongest negative values and thus the highest likelihood of training leakage, whereas the more recent BIRD dataset shows minimal sensitivity and appears largely uncontaminated. Together, these findings highlight the importance of temporally contextualized, syntactic-probing evaluation for trustworthy NL2SQL benchmarking. 

---
# Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF 

**Authors**: Yuan Fang, Yiming Luo, Aimin Zhou, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17769)  

**Abstract**: Ensuring the safety of large language models (LLMs) requires robust red teaming, yet the systematic synthesis of high-quality toxic data remains under-explored. We propose Reverse Constitutional AI (R-CAI), a framework for automated and controllable adversarial data generation that moves beyond isolated jailbreak prompts. By inverting a harmless constitution into a constitution of toxicity and iteratively refining model outputs through a critique--revision pipeline, R-CAI enables scalable synthesis of multi-dimensional adversarial data without human annotation. Optimizing solely for toxicity-related rewards, however, can lead to reward hacking and degraded semantic coherence. To address this challenge, we introduce probability clamping within reinforcement learning from AI feedback, which stabilizes adversarial optimization while preserving adversarial intent. Experiments demonstrate that R-CAI generates diverse, high-quality toxic data and that probability clamping substantially improves semantic coherence (15%) without sacrificing adversarial strength. Overall, R-CAI provides a fully automated framework for red teaming data generation and systematic safety evaluation of aligned language models. 

---
# HiRAS: A Hierarchical Multi-Agent Framework for Paper-to-Code Generation and Execution 

**Authors**: Hanhua Hong, Yizhi LI, Jiaoyan Chen, Sophia Ananiadou, Xiaoli Li, Jung-jae Kim, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17745)  

**Abstract**: Recent advances in large language models have highlighted their potential to automate computational research, particularly reproducing experimental results. However, existing approaches still use fixed sequential agent pipelines with weak global coordination, which limits their robustness and overall performance. In this work, we propose Hierarchical Research Agent System (HiRAS), a hierarchical multi-agent framework for end-to-end experiment reproduction that employs supervisory manager agents to coordinate specialised agents across fine-grained stages. We also identify limitations in the reference-free evaluation of the Paper2Code benchmark and introduce Paper2Code-Extra (P2C-Ex), a refined protocol that incorporates repository-level information and better aligns with the original reference-based metric. We conduct extensive evaluation, validating the effectiveness and robustness of our proposed methods, and observing improvements, including >10\% relative performance gain beyond the previous state-of-the-art using open-source backbone models and significantly reduced hallucination in evaluation. Our work is available on GitHub: this https URL. 

---
# Mira-Embeddings-V1: Domain-Adapted Semantic Reranking for Recruitment via LLM-Synthesized Data 

**Authors**: Zhaohua Liang, Zhilin Wang, Renjie Cao, Yining Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17738)  

**Abstract**: Candidate sourcing for recruiters is best viewed as a two-stage retrieval and reranking pipeline with recall as the primary objective under a limited review budget. An upstream production retriever first returns a candidate shortlist for each job description (JD), and our goal is to rerank that shortlist so that qualified candidates appear as high as possible. We present mira-embeddings-v1, a semantic reranking system for the recruitment domain that reshapes the embedding space with LLM-synthesized training data and corrects boundary confusions with a lightweight reranking head. Starting from real JDs, we build a five-stage prompt pipeline to generate diverse positive and hard negative samples that sculpt the semantic space from multiple angles. We then apply a two-round LoRA adaptation: JD--JD contrastive training followed by JD--CV triplet alignment on a heterogeneous text dataset. Importantly, these gains require no large-scale manually labeled industrial training pairs: a modest set of real JDs is expanded into supervision through LLM synthesis. Finally, a BoundaryHead MLP reranks the Top-K results to distinguish between roles that share the same title but differ in scope. On a local pool of 300 real JDs with candidates from an upstream production retriever, mira-embeddings-v1 improves Recall@50 from 68.89% (baseline) to 77.55% while lifting Precision@10 from 35.77% to 39.62%. On a supportive global pool over 44,138 candidates judged by a Qwen3-32B rubric, it achieves Recall@200 of 0.7047 versus 0.5969 for the baseline. These results show that LLM-synthesized supervision with boundary-aware reranking yields robust gains without a heavy cross-encoder. 

---
# MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models 

**Authors**: Suhyun Lee, Palakorn Achananuparp, Neemesh Yadav, Ee-Peng Lim, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.17730)  

**Abstract**: Large language models (LLMs) are increasingly explored as scalable tools for mental health counseling, yet evaluating their safety remains challenging due to the interactional and context-dependent nature of clinical harm. Existing evaluation frameworks predominantly assess isolated responses using coarse-grained taxonomies or static datasets, limiting their ability to diagnose how harms emerge and accumulate over multi-turn counseling interactions. In this work, we introduce R-MHSafe, a role-aware mental health safety taxonomy that characterizes clinically significant harm in terms of the interactional roles an AI counselor adopts, including perpetrator, instigator, facilitator, or enabler, combined with clinically grounded harm categories. Then, we propose MHSafeEval, a closed-loop, agent-based evaluation framework that formulates safety assessment as trajectory-level discovery of harm through adversarial multi-turn interactions, guided by role-aware modeling. Using R-MHSafe and MHSafeEval, we conduct a large-scale evaluation across state-of-the-art LLMs. Our results reveal substantial role-dependent and cumulative safety failures that are systematically missed by existing static benchmarks, and show that our framework significantly improves failure-mode coverage and diagnostic granularity. 

---
# RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models 

**Authors**: Arya Hadizadeh Moghaddam, Drew Ross, Mohsen Nayebi Kerdabadi, Dongjie Wang, Zijun Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17725)  

**Abstract**: Large Language Models (LLMs) have shown strong promise for mining Electronic Health Records (EHRs) by reasoning over longitudinal clinical information to capture context-rich patient trajectories. However, leveraging LLMs for structured EHRs (e.g., standardized diagnosis and medication codes) presents two key challenges. First, translating time-stamped EHR sequences into plain text can obscure both temporal structure and code identities, weakening the ability to capture code co-occurrence and longitudinal regularities. Second, unlike cohort-trained predictive models that learn a shared, task-aligned representation space across patients, LLMs are often applied in a case-isolated inference setting where each patient is processed independently without leveraging population-level patterns. To address these challenges, we introduce RePrompT, a time-aware LLM framework that integrates structured EHR encoders through prompt tuning, without modifying underlying architectures. Specifically, RePrompT recurrently incorporates latent states from prior visits to preserve longitudinal information, and injects population-level information through trainable prompt tokens derived from a cohort-trained, task-aligned EHR encoder. Experiments on MIMIC-III and MIMIC-IV demonstrate that RePrompT consistently outperforms both EHR-based and LLM-based baselines across multiple clinical prediction tasks. 

---
# Do LLMs Use Cultural Knowledge Without Being Told? A Multilingual Evaluation of Implicit Pragmatic Adaptation 

**Authors**: Mehwish Nasim, Sanjeevan Selvaganapathy, Neel Ganapathi Sabhahit, Marie Griesbach, Pranav Bhandari, Janina Lütke Stockdiek, Lennart Schäpermeier, Usman Naseem, Christian Grimme  

**Link**: [PDF](https://arxiv.org/pdf/2604.17718)  

**Abstract**: Many benchmarks show that large language models can answer direct questions about culture. We study a different question: do they also change how they speak when culture is only implied by the situation? We evaluate 60 culturally grounded conversational scenarios across five languages in three conditions: a neutral baseline (Prompt A), an explicit cultural instruction (Prompt B), and implicit situational cueing (Prompt C). We score responses on 12 pragmatic features covering deference to authority, individual-versus-group framing, and uncertainty management. We define Pragmatic Context Sensitivity (PCS) as the fraction of the Prompt A->B shift that reappears under Prompt A->C. Across four deployed LLMs and five languages (English, German, Hindi, Nepali, Urdu), the primary stable-only PCS mean is 0.196 (SD = 0.113), indicating that the models recover only about one-fifth of the pragmatic shift they can produce when instructed explicitly. Transfer is strongest for authority-related cues (0.299) and weakest for individual-versus-group framing (0.120). Uncertainty-related behaviour is mixed: hedging density exhibits negative explicit gaps in all five languages, suggesting that alignment training actively suppresses the target behaviour. Because Hindi and Urdu share core grammar yet index distinct cultural communities, we use them as a natural control; a paired analysis finds no reliable baseline difference (t = 0.96, p = 0.339, dz = 0.06), suggesting that models respond primarily to linguistic structure rather than to the cultural associations a language carries. We argue that multilingual cultural pragmatics is an explicit-versus-implicit deployment problem, not only a factual knowledge problem. 

---
# Concurrent Criterion Validation of a Validity Screen for LLM Confidence Signals via Selective Prediction 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.17716)  

**Abstract**: The validity screen (Cacioli, 2026d, 2026e) classifies LLM confidence signals as Valid, Indeterminate, or Invalid. We test whether these classifications predict selective prediction performance. Twenty frontier LLMs from seven families were evaluated on 524 items across six cognitive tracks. Valid models show mean Type 2 AUROC = .624 (SD = .048). Invalid models show mean AUROC = .357 (SD = .231). Cohen's d = 2.81, p = .002. The tiers order monotonically: Invalid (.357) < Indeterminate (.554) < Valid (.624). Split-half cross-validation yields median d = 1.77, P(d > 0) = 1.0 across 1,000 splits. The three-tier classification accounts for 47% of the variance in AUROC. DeepSeek-R1 drops from 85.3% accuracy at full coverage to 11.3% at 10% coverage. The screen predicts the criterion. For selective prediction, the screen matters. 

---
# Screen Before You Interpret: A Portable Validity Protocol for Benchmark-Based LLM Confidence Signals 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.17714)  

**Abstract**: LLM confidence signals are used for abstention, routing, and safety-critical decisions. No standard practice exists for checking whether a confidence signal carries item-level information before building on it. We transfer the validity screening principle from clinical personality assessment (PAI, MMPI-3) as a portable protocol for benchmark-based LLM confidence data. The protocol specifies three core indices (L, Fp, RBS), a structural indicator (TRIN), and an item-sensitivity statistic, computed from a single 2x2 contingency table. A three-tier classification system (Invalid, Indeterminate, Valid) draws on four clinical traditions. Validated on 20 frontier LLMs across 524 items, four models are classified Invalid, two Indeterminate. Valid-profile models show mean r = .18 (15/16 significant). Invalid-profile models show mean r = -.20 (d = 2.48). Cross-benchmark validation on 18 models using MMLU with verbalized confidence and on external data from Yang et al. (2024) confirms the screen transfers across benchmarks and probe formats. All data and code: this https URL 

---
# DeInfer: Efficient Parallel Inferencing for Decomposed Large Language Models 

**Authors**: You-Liang Huang, Xinhao Huang, Chengxi Liao, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.17709)  

**Abstract**: Existing works on large language model (LLM) decomposition mainly focus on improving performance on downstream tasks, but they ignore the poor parallel inference performance when trying to scale up the model size. To mitigate this important performance issue, this paper introduces DeInfer, a high-performance inference system dedicated to parallel inference of decomposed LLMs. It consists of multiple optimizations to maximize performance and be compatible with state-of-the-art optimization techniques. Extensive experiments are carried out to evaluate DeInfer's performance, where the results demonstrate its superiority, suggesting it can greatly facilitate the parallel inference of decomposed LLMs. 

---
# Before You Interpret the Profile: Validity Scaling for LLM Metacognitive Self-Report 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.17707)  

**Abstract**: Clinical personality assessment screens response validity before interpreting substantive scales. LLM evaluation does not. We apply the validity scaling framework from the PAI and MMPI-3 to metacognitive probe data from 20 frontier models across 524 items. Six validity indices are operationalised: L (maintaining confidence on errors), K (betting on errors), F (withdrawing consensus-endorsed items), Fp (withdrawing correct answers), RBS (inverted monitoring), and TRIN (fixed responding). A tiered classification system identifies four models as construct-level invalid and two as elevated. Valid-profile models produce item-sensitive confidence (mean r = .18, 14 of 16 significant). Invalid-profile models do not (mean r = -.20, d = 2.17, p = .001). Chain-of-thought training produces two opposite response distortions. Two latent dimensions account for 94.6% of index variance. Companion papers extract a portable screening protocol (Cacioli, 2026e) and validate it against selective prediction (Cacioli, 2026f). All data and code: this https URL 

---
# Towards Intelligent Legal Document Analysis: CNN-Driven Classification of Case Law Texts 

**Authors**: Moinul Hossain, Sourav Rabi Das, Zikrul Shariar Ayon, Sadia Afrin Promi, Ahnaf Atef Choudhury, Shakila Rahman, Jia Uddin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17674)  

**Abstract**: Legal practitioners and judicial institutions face an ever-growing volume of case-law documents characterised by formalised language, lengthy sentence structures, and highly specialised terminology, making manual triage both time-consuming and error-prone. This work presents a lightweight yet high-accuracy framework for citation-treatment classification that pairs lemmatisation-based preprocessing with subword-aware FastText embeddings and a multi-kernel one-dimensional Convolutional Neural Network (CNN). Evaluated on a publicly available corpus of 25,000 annotated legal documents with a 75/25 training-test partition, the proposed system achieves 97.26% classification accuracy and a macro F1-score of 96.82%, surpassing established baselines including fine-tuned BERT, Long Short-Term Memory (LSTM) with FastText, CNN with random embeddings, and a Term Frequency-Inverse Document Frequency (TF-IDF) k-Nearest Neighbour (KNN) classifier. The model also attains the highest Area Under the Receiver Operating Characteristic (AUC-ROC) curve of 97.83% among all compared systems while operating with only 5.1 million parameters and an inference latency of 0.31 ms per document - more than 13 times faster than BERT. Ablation experiments confirm the individual contribution of each pipeline component, and the confusion matrix reveals that residual errors are confined to semantically adjacent citation categories. These findings indicate that carefully designed convolutional architectures represent a scalable, resource-efficient alternative to heavyweight transformers for intelligent legal document analysis. 

---
# Peerispect: Claim Verification in Scientific Peer Reviews 

**Authors**: Ali Ghorbanpour, Soroush Sadeghian, Alireza Daghighfarsoodeh, Sajad Ebrahimi, Negar Arabzadeh, Seyed Mohammad Hosseini, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2604.17667)  

**Abstract**: Peer review is central to scientific publishing, yet reviewers frequently include claims that are subjective, rhetorical, or misaligned with the submitted work. Assessing whether review statements are factual and verifiable is crucial for fairness and accountability. At the scale of modern conferences and journals, manually inspecting the grounding of such claims is infeasible. We present Peerispect, an interactive system that operationalizes claim-level verification in peer reviews by extracting check-worthy claims from peer reviews, retrieving relevant evidence from the manuscript, and verifying the claims through natural language inference. Results are presented through a visual interface that highlights evidence directly in the paper, enabling rapid inspection and interpretation. Peerispect is designed as a modular Information Retrieval (IR) pipeline, supporting alternative retrievers, rerankers, and verifiers, and is intended for use by reviewers, authors, and program committees. We demonstrate Peerispect through a live, publicly available demo (this https URL) and API services (this https URL), accompanied by a video tutorial (this https URL). 

---
# Semantic Density Effect (SDE): Maximizing Information Per Token Improves LLM Accuracy 

**Authors**: Amr Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.17659)  

**Abstract**: We introduce the Semantic Density Effect (SDE): the empirical finding that prompts carrying higher semantic information per token consistently produce more accurate, focused, and less hallucinated outputs across all major LLM families. SDE is defined as the ratio of semantically loaded tokens to total prompt tokens, adjusted for redundancy and concreteness. Unlike prior prompt optimization techniques that add tokens (Chain of Thought), duplicate the prompt (Prompt Repetition), or reorder components (Instruction Placement Effect), SDE improves performance by removing or replacing low-information tokens while preserving or sharpening the semantic signal. Evaluated across five frontier models and seven benchmarks, ultra-dense prompts (SDE > 0.80) outperform diluted counterparts by an average of +8.4 percentage points with 0 additional tokens and 0 latency overhead. Combined with Instruction Placement Effect (IPE), the gain reaches +11.7 percentage points 

---
# Measuring Distribution Shift in User Prompts and Its Effects on LLM Performance 

**Authors**: Parker Seegmiller, Sarah Masud Preum  

**Link**: [PDF](https://arxiv.org/pdf/2604.17650)  

**Abstract**: LLMs are increasingly deployed in dynamic, real-world settings, where the distribution of user prompts can shift substantially over time as new tasks, prompts, and users are introduced to a deployed model. Such natural prompt distribution shift poses a major challenge to LLM reliability, particularly for specialized models designed for narrow domains or user populations. Despite attention to out-of-distribution robustness, there is very limited exploration of measuring natural prompt distribution shift in prior work, and its impact on deployed LLMs remains poorly understood. We introduce the LLM Evaluation under Natural prompt Shift (LENS) framework: a data-centric approach for quantifying natural prompt distribution shift and evaluating its effect on the performance of deployed LLMs. We perform a large-scale evaluation using 192 real-world post-deployment prompt shift settings over time, user group, and geographic axes, training a total of 81 models on 4.68M training prompts, and evaluating on 57.6k prompts. We find that even moderate shifts in user prompt behavior correspond with large performance drops (73% average loss) in deployed LLMs. This performance degradation is particularly prevalent when users from different latent groups and geographic regions interact with models and is correlated with natural prompt distribution shift over time. We systematically characterize how LLM instruction following ability degrades over time and between user groups. Our findings highlight the critical need for data-driven monitoring to ensure LLM performance remains stable across diverse and evolving user populations. 

---
# ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts 

**Authors**: Olubusayo Olabisi, Ekata Mitra, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2604.17648)  

**Abstract**: Summarizing deeply nested discussion threads requires handling interleaved replies, quotes, and overlapping topics, which standard LLM summarizers struggle to capture reliably. We introduce ThreadSumm, a multi-stage LLM framework that treats thread summarization as a hierarchical reasoning problem over explicit aspect and content unit representations. Our method first performs content planning via LLM-based extraction of discourse aspects and Atomic Content Units, then applies sentence ordering to construct thread-aware sequences that surface multiple viewpoints rather than a single linear strand. On top of these interpretable units, ThreadSumm employs a Tree of Thoughts search that generates and scores multiple paragraph candidates, jointly optimizing coherence and coverage within a unified search space. With this multi-proposal and iterative refinement design, we show improved performance in generating logically structured summaries compared to existing baselines, while achieving higher aspect retention and opinion coverage in nested discussions. 

---
# Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining 

**Authors**: Felicia Körner, Maria Matveev, Florian Eichin, Gitta Kutyniok, Barbara Plank, Michael A. Hedderich  

**Link**: [PDF](https://arxiv.org/pdf/2604.17633)  

**Abstract**: Large language models exhibit impressive cross-lingual capabilities. However, prior work analyzes this phenomenon through isolated factors and at sparse points during training, limiting our understanding of how cross-lingual generalization emerges--particularly in the early phases of learning. To study the early trajectory of linguistic and translation capabilities, we pretrain a multilingual 1.7B model on nine diverse languages, capturing checkpoints at a much finer granularity. We further introduce a novel word-level translation dataset and trace how translation develops over training through behavioral analyses, model-component analysis, and parameter-based ablations. We find that the model quickly acquires basic linguistic capabilities in parallel with token-level copying, while translation develops in two distinct phases: an initial phase dominated by copying and surface-level similarities, and a second phase in which more generalizing translation mechanisms are developed while copying is refined. Together, these findings provide a fine-grained view of how cross-lingual generalization develops during multilingual pretraining. 

---
# Does Welsh media need a review? Detecting bias in Nation.Cymru's political reporting 

**Authors**: Cai Parry-Jones  

**Link**: [PDF](https://arxiv.org/pdf/2604.17628)  

**Abstract**: Wales' political landscape has been marked by growing accusations of bias in Welsh media. This paper takes the first computational step toward testing those claims by examining this http URL, a prominent Welsh political news outlet. I use a two-stage natural language processing (NLP) pipeline: (1) a robustly optimized BERT approach (RoBERTa) bias detector for efficient bias discovery and (2) a large language model (LLM) for target-attributed sentiment classification of bias labels from (1). A primary analysis of 15,583 party mentions across 2022-2026 news articles finds that Reform UK attracts biased framing at twice the rate of Plaid Cymru and over three times as negative in mean sentiment (p<0.001). A secondary analysis across four parties across both news and opinion articles shows that Plaid Cymru is the outlier, receiving markedly more favourable framing than any other party. These findings provide evidence of measurable differential framing in a single Welsh political media outlet, supporting calls for a broader review of Welsh media coverage. Furthermore, the two-stage pipeline offers a low-cost, replicable framework for extending this analysis to other Welsh outlets, as well as media ecosystems outside of Wales. 

---
# Agents Explore but Agents Ignore: LLMs Lack Environmental Curiosity 

**Authors**: Leon Engländer, Sophia Althammer, Ahmet Üstün, Matthias Gallé, Tom Sherborne  

**Link**: [PDF](https://arxiv.org/pdf/2604.17609)  

**Abstract**: LLM-based agents are assumed to integrate environmental observations into their reasoning: discovering highly relevant but unexpected information should naturally lead to a model exploiting its own discoveries. We show that this assumption is false for current LLM-based agents, which struggle to reflect or react to unexpected information. Across three benchmarks (Terminal-Bench, SWE-Bench, AppWorld), we inject complete task solutions into the agent environments to deliberately expose a task's solution to a model. While agents discover these solutions on Terminal-Bench in 79-81% of runs, they interact, or exploit, them in only 37-50% of cases. This gap is starkest in AppWorld: agents see documentation stating that a command "returns the complete solution to this task" in over 90% of attempts but exploit this in fewer than 7% of trials. We show that agents lack what we call environmental curiosity: the capability to recognize and investigate unexpected but relevant observations in response to environmental stimuli. We identify three main factors influencing environmental curiosity: available tools in the agent scaffold, test-time compute, and training data distribution. Our findings identify configurations that maximize curiosity also achieve the best performance on the unmodified benchmarks. Yet even jointly optimized agents still ignore discovered solutions in the majority of trials: current agents use the environment to fetch expected information, but not to revise their strategy or maximally exploit useful stimuli. 

---
# Beyond Fine-Tuning: In-Context Learning and Chain-of-Thought for Reasoned Distractor Generation 

**Authors**: Elaf Alhazmi, Quan Z. Sheng, Wei Emma Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17574)  

**Abstract**: Distractor generation (DG) remains a labor-intensive task that still significantly depends on domain experts. The task focuses on generating plausible yet incorrect options, known as distractors, for multiple-choice questions. A reliable distractor must be contextually relevant to the question and able to mislead examinees through implicit reasoning when identifying the correct answer. While a recent method integrates fine-tuning pre-trained encoder-decoder models with contrastive learning to generate semantically relevant distractors for a given question-answer, it often fails to capture the underlying reasoning process that experts utilize when selecting distractors in benchmarks. In this paper, we explore large language models (LLMs) reasoning for DG through in-context learning with unsupervised semantic retrieval for selecting few-shot examples. We design a rationale-augmented DG framework that jointly generates distractors and their rationales for a given question-answer. Extensive experiments on six benchmarks, with varying average distractor lengths and domains, demonstrate that prompting LLMs with few-shot examples substantially improves the performance compared to recent DG models. It outperforms recent approaches and achieves state-of-the-art results in generating reasoned distractors that align with human-labeled benchmarks. 

---
# MAPLE: A Meta-learning Framework for Cross-Prompt Essay Scoring 

**Authors**: Salam Albatarni, May Bashendy, Sohaila Eltanbouly, Tamer Elsayed  

**Link**: [PDF](https://arxiv.org/pdf/2604.17569)  

**Abstract**: Automated Essay Scoring (AES) faces significant challenges in cross-prompt settings, where models must generalize to unseen writing prompts. To address this limitation, we propose MAPLE, a meta-learning framework that leverages prototypical networks to learn transferable representations across different writing prompts. Across three diverse datasets (ELLIPSE and ASAP (English), and LAILA (Arabic)), MAPLE achieves state-of-the-art performance on ELLIPSE and LAILA, outperforming strong baselines by 8.5 and 3 points in QWK, respectively. On ASAP, where prompts exhibit heterogeneous score ranges, MAPLE yields improvements on several traits, highlighting the strengths of our approach in unified scoring settings. Overall, our results demonstrate the potential of meta-learning for building robust cross-prompt AES systems. 

---
# PoliLegalLM: A Technical Report on a Large Language Model for Political and Legal Affairs 

**Authors**: Yuting Huang, Yinghao Hu, Qian Xiao, Wenlin Zhong, Yiquan Wu, Taishi Zhou, Moke Chen, Changlong Sun, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17543)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in general-domain tasks, yet their direct application to the legal domain remains challenging due to hallucinated legal citations, incomplete knowledge coverage, and weak structured reasoning. To address these issues, we propose PoliLegalLM, a domain-specific large language model tailored for political and legal applications. Our approach adopts a unified training framework that integrates continued pretraining, progressive supervised fine-tuning, and preference-based reinforcement learning to jointly enhance legal knowledge grounding, task alignment, and reasoning capability. We construct a large-scale, high-quality legal corpus and design a structured post-training pipeline, enabling the model to effectively learn domain-specific knowledge and adapt to diverse legal tasks. We evaluate PoliLegalLM on three representative benchmarks, including LawBench, LexEval, and a real-world dataset, PoliLegal. Experimental results demonstrate that PoliLegalLM achieves strong and consistent performance, outperforming competitive models of similar scale and remaining highly competitive with significantly larger models, while achieving the best results on real-world legal scenarios. These results highlight the effectiveness of our training paradigm and the practical value of domain-specific LLMs for real-world legal applications. 

---
# OPSDL: On-Policy Self-Distillation for Long-Context Language Models 

**Authors**: Xinsen Zhang, Zhenkai Ding, Tianjun Pan, Run Yang, Chun Kang, Xue Xiong, Jingnan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17535)  

**Abstract**: Extending the effective context length of large language models (LLMs) remains a central challenge for real-world applications. While recent post-training methods have made progress in long-context scaling, they either rely on high-quality supervision data or sparse sequence-level rewards, leading to unstable and inefficient optimization. We propose OPSDL, an On-Policy Self-Distillation method for enhancing the Long-context capabilities of LLMs. Unlike other recent self-distillation methods that inject privileged information and rely on the model's in-context learning ability to act as a teacher, OPSDL leverages the model's own inherently strong short-context capability as a self-teacher to supervise its own generation in long-context scenarios. The model first generates responses conditioned on the full long-context, then the self-teacher provides per-token supervision signals via point-wise reverse KL divergence under the relevant extracted short-context. This dense token-level signal encourages faithful use of relevant evidence and mitigates hallucinations induced by irrelevant context. We evaluate OPSDL on long-context benchmarks across a range of models from 7B to 32B parameters. Results show consistent and substantial improvements across varying context lengths, outperforming standard post-training approaches such as SFT and DPO with higher sample efficiency. Notably, these gains are achieved without degrading general short-context performance. These findings highlight the effectiveness of OPSDL as a scalable and stable approach for long-context learning. 

---
# ONTO: A Token-Efficient Columnar Notation for LLM Input Optimization 

**Authors**: Harshavardhanan Deekeswar  

**Link**: [PDF](https://arxiv.org/pdf/2604.17512)  

**Abstract**: Serialization formats designed for document interchange impose structural overhead that becomes prohibitive when large language models consume operational data at scale. A modest dataset of 1,000 IoT sensor readings serialized as JSON requires approximately 80,000 tokens - the majority spent on repeated field names, nested braces, and structural punctuation rather than semantic content. We present ONTO (Object Notation for Token Optimization), a columnar notation that declares field names once per entity and arranges values in pipe-delimited rows with indentation-based hierarchy. This schema-once, data-many design eliminates per-record key repetition while preserving human readability and nested structure support. Evaluation across three synthetic operational datasets demonstrates 46-51% token reduction versus JSON, with stable scaling from 100 to 1,000 records. Controlled inference benchmarks on Qwen2.5-7B show corresponding 5-10% latency improvement. Comprehension validation confirms no material degradation in LLM task accuracy across lookup, counting, extraction, and aggregation operations when format context is provided. Ablation analysis reveals that key repetition accounts for the majority of JSON overhead, with indentation costs in nested structures explaining the 4-percentage-point gap between flat and hierarchical data. ONTO occupies a previously unfilled position in the serialization landscape: columnar efficiency with hierarchical structure, optimized for LLM context windows rather than document interchange. Code and specification are available at this https URL. 

---
# CoAct: Co-Active LLM Preference Learning with Human-AI Synergy 

**Authors**: Ruiyao Xu, Mihir Parmar, Tiankai Yang, Zhengyu Hu, Yue Zhao, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.17501)  

**Abstract**: Learning from preference-based feedback has become an effective approach for aligning LLMs across diverse tasks. However, high-quality human-annotated preference data remains expensive and scarce. Existing methods address this challenge through either self-rewarding, which scales by using purely AI-generated labels but risks unreliability, or active learning, which ensures quality through oracle annotation but cannot fully leverage unlabeled data. In this paper, we present CoAct, a novel framework that synergistically combines self-rewarding and active learning through strategic human-AI collaboration. CoAct leverages self-consistency to identify both reliable self-labeled data and samples that require oracle verification. Additionally, oracle feedback guides the model to generate new instructions within its solvable capability. Evaluated on three reasoning benchmarks across two model families, CoAct achieves average improvements of +13.25% on GSM8K, +8.19% on MATH, and +13.16% on WebInstruct, consistently outperforming all baselines. 

---
# Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems 

**Authors**: Tianyi Huang, Samuel Xu, Jason Tansong Dang, Samuel Yan, Kimberley Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17487)  

**Abstract**: Agentic systems often fail not by being entirely wrong, but by being too precise: a response may be generally useful while particular claims exceed what the evidence supports. We study this failure mode as overcommitment control and introduce compositional selective specificity (CSS), a post-generation layer that decomposes an answer into claims, proposes coarser backoffs, and emits each claim at the most specific calibrated level that appears admissible. The method is designed to express uncertainty as a local semantic backoff rather than as a whole-answer refusal. Across a full LongFact run and HotpotQA pilots, calibrated CSS improves the risk-utility trade-off of fixed drafts. On the full LongFact run, it raises overcommitment-aware utility from 0.846 to 0.913 relative to the no-CSS output while achieving 0.938 specificity retention. These results suggest that claim-level specificity control is a useful uncertainty interface for agentic systems and a target for future distribution-free validity layers. 

---
# MoVE: Translating Laughter and Tears via Mixture of Vocalization Experts in Speech-to-Speech Translation 

**Authors**: Szu-Chi Chen, I-Ning Tsai, Yi-Cheng Lin, Sung-Feng Huang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.17435)  

**Abstract**: Recent Speech-to-Speech Translation (S2ST) systems achieve strong semantic accuracy yet consistently strip away non-verbal vocalizations (NVs), such as laughter and crying that convey pragmatic intent, which severely limits real-world utility. We address this via three contributions. First, we propose a synthesis pipeline for building scalable expressive datasets to overcome the data scarcity limitation. Second, we propose MoVE, a Mixture-of-LoRA-Experts architecture with expressive-specialized adapters and a soft-weighting router that blends experts for capturing hybrid expressive states. Third, we show pretrained AudioLLMs enable striking data efficiency: 30 minutes of curated data is enough for strong performance. On English-Chinese S2ST, while comparing with strong baselines, MoVE reproduces target NVs in 76% of cases and achieves the highest human-rated naturalness and emotional fidelity among all compared systems, where existing S2ST systems preserve at most 14% of NVs. 

---
# Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning 

**Authors**: Raman Saparkhan, Majd Hawasly, Md Rizwan Parvez, Mohammad Raza  

**Link**: [PDF](https://arxiv.org/pdf/2604.17433)  

**Abstract**: Self-consistency (SC) is a popular technique for improving the reasoning accuracy of large language models by aggregating multiple sampled outputs, but it comes at a high computational cost due to extensive sampling. We introduce a hybrid ensembling approach that leverages the complementary strengths of two distinct modes of reasoning: Chain-of-Thought (CoT) and Program-of-Thought (PoT). We describe a general framework for combining these two forms of reasoning in self-consistency, as well as particular strategies for both full sampling and early-stopping. We show that CoT-PoT ensembling not only improves overall accuracy, but also drastically reduces the number of samples required for SC by a factor of 9.3x. In particular, the majority of tasks (78.6%) can be addressed with only two samples, which has not been possible with any prior SC methods. 

---
# Jupiter-N Technical Report 

**Authors**: George Drayson  

**Link**: [PDF](https://arxiv.org/pdf/2604.17429)  

**Abstract**: We present Jupiter-N, a hybrid reasoning model post-trained from Nemotron 3 Super, a fully open-source 120 billion parameter LLM. We target three objectives: (1) agentic capability via uncertainty-curated trajectories; (2) UK cultural alignment via synthetic data grounded in cultural norms; and (3) Welsh language support via parallel corpora and LLM-translated Welsh conversations. Our data curation strategy carefully preserves the base model's capabilities: using our Forget-Me-Not framework, we mix on-policy synthetic replay with off-policy task data to mitigate catastrophic forgetting, and include a mixture of reasoning and non-reasoning traces to maintain Nemotron's hybrid reasoning ability. Jupiter-N achieves standout gains over Nemotron in Welsh (+18 on ARC-Easy, +5.25 on MMLU-Lite), terminal-use (+9.1 on Terminal Bench 2) and instruction following (+4.4 on IFBench), while retaining the base model capabilities. We frame this work as a reproducible template for sovereign post-training: substituting cultural knowledge, institutional corpora, and target languages produces an equivalent pipeline for any country. All model weights and all post-training datasets are publicly released under open licences. 

---
# DuConTE: Dual-Granularity Text Encoder with Topology-Constrained Attention for Text-attributed Graphs 

**Authors**: Lexuan Liang, Tao Zou, Xuxiang Ta, Zekun Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17411)  

**Abstract**: Text-attributed graphs integrate semantic information of node texts with topological structure, offering significant value in various applications such as document classification and information extraction. Existing approaches typically encode textual content using language models (LMs), followed by graph neural networks (GNNs) to process structural information. However, during the LM-based text encoding phase, most methods not only perform semantic interaction solely at the word-token granularity, but also neglect the structural dependencies among texts from different nodes. In this work, we propose DuConTE, a dual-granularity text encoder with topology-constrained attention. The model employs a cascaded architecture of two pretrained LMs, encoding semantics first at the word-token granularity and then at the node granularity. During the self-attention computation in each LM, we dynamically adjust the attention mask matrix based on node connectivity, guiding the model to learn semantic correlations informed by the graph structure. Furthermore, when composing node representations from word-token embeddings, we separately evaluate the importance of tokens under the center-node context and the neighborhood context, enabling the capture of more contextually relevant semantic information. Extensive experiments on multiple benchmark datasets demonstrate that DuConTE achieves state-of-the-art performance on the majority of them. 

---
# Contrastive Analysis of Linguistic Representations in Large Language Model Outputs through Structured Synthetic Data Generation and Abstracted N-gram Associations 

**Authors**: S.A. Desimone, L. Alonso Alemany  

**Link**: [PDF](https://arxiv.org/pdf/2604.17398)  

**Abstract**: We present a methodological framework to discover linguistic and discursive patterns associated to different social groups through contrastive synthetic text generation and statistical analysis. In contrast with previous approaches, we aim to characterize subtle expressions of bias, instead of diagnosing bias through a pre-determined list of words or expressions. We are also working with contextualized data instead of isolated words or sentences. Our methodology applies to textual productions in any genre, encompassing narrative, task-oriented or dialogic. Contextualized data are generated using controlled combinations of situational scenarios and group markers, creating minimal pairs of texts that differ only in the referenced group while maintaining comparable narrative conditions. To facilitate robust analysis, linguistic forms are generalized and associations between linguistic abstractions and groups are quantified using a variant of pointwise mutual information to detect expressions that appear disproportionately across groups. A fragment-ranking strategy then prioritizes text segments with a high concentration of biased linguistic signals, which allows for experts to assess the harmful potential of linguistic expressions in context, bridging quantitative analysis and qualitative interpretation. 

---
# Representation-Guided Parameter-Efficient LLM Unlearning 

**Authors**: Zeguan Xiao, Lang Mo, Yun Chen, Lei Yang, Jiehui Zhao, Lili Yang, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.17396)  

**Abstract**: Large Language Models (LLMs) often memorize sensitive or harmful information, necessitating effective machine unlearning techniques. While existing parameter-efficient unlearning methods have shown promise, they still struggle with the forget-retain trade-off. This can be attributed to their reliance on parameter importance metrics to identify parameters that are important exclusively for the forget set, which is fundamentally limited by the superposition phenomenon. Due to the polysemantic nature of LLM parameters, such an importance metric may struggle to disentangle parameters associated with the forget and retain sets. In this work, we propose Representation-Guided Low-rank Unlearning (REGLU), a novel approach that leverages the geometric properties of representation spaces to achieve robust and precise unlearning. First, we develop a representation-guided initialization for LoRA that identifies the optimal subspace for selective forgetting. Second, we introduce a regularization loss that constrains the outputs of the LoRA update to lie in the orthogonal complement of the retain set's representation subspace, thereby minimizing interference with the model's performance on the retain set. We evaluate REGLU on the TOFU and WMDP benchmarks across multiple models. Our results demonstrate that REGLU consistently outperforms state-of-the-art baselines, achieving superior unlearning quality while maintaining higher model utility. 

---
# Who Watches the Watchmen? Humans Disagree With Translation Metrics on Unseen Domains 

**Authors**: Finn Schmidt, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2604.17393)  

**Abstract**: Automatic evaluation metrics are central to the development of machine translation systems, yet their robustness under domain shift remains unclear. Most metrics are developed on the Workshop on Machine Translation (WMT) benchmarks, raising concerns about their robustness to unseen domains. Prior studies that analyze unseen domains vary translation systems, annotators, or evaluation conditions, confounding domain effects with human annotation noise.
To address these biases, we introduce a systematic multi-annotator Cross-Domain Error-Span-Annotation dataset (CD-ESA), comprising 18.8k human error span annotations across three language pairs, where we fix annotators within each language pair and evaluate translations of the same six translation systems across one seen news domain and two unseen technical domains. Using this dataset, we first find that automatic metrics appear surprisingly robust to domain-shifts at the segment level (up to 0.69 agreement), but this robustness largely disappears once we account for human label variation. Averaging annotations increases inter-annotator agreement by up to +0.11. Metrics struggle on the unseen chemical domain compared to humans (inter-annotator agreement of 0.78-0.83 vs. 0.96).
We recommend comparing metric-human agreement against inter-annotator agreement, rather than comparing raw metric-human agreement alone, when evaluating across different domains. 

---
# AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models 

**Authors**: Zhanyu Shen, Sijie Cheng, Zhicheng Guo, Weiqin Wang, Yile Wang, Hui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17377)  

**Abstract**: While large language models have achieved remarkable performance in complex tasks, they still need a memory system to utilize historical experience in long-term interactions. Existing memory methods (e.g., A-Mem, Mem0) place excessive emphasis on organizing interactions by frequently rewriting them, however, this heavy reliance on summarization risks diluting essential contextual nuances and obscuring key retrieval features. To bridge this gap, we introduce AnchorMem, a novel memory framework inspired by the Proust Phenomenon in cognitive science, where a specific anchor triggers a holistic recollection. We propose a method that decouples the retrieval unit from the generation context. AnchorMem extracts atomic facts from interaction history to serve as retrieval anchors, while preserving the original context as the immutable context. To reveal implicit narrative cues, we construct an associative event graph that uses higher-order event links that bind sets of related facts into shared event representations, strengthening cross-memory integration without relying on generic entities as bridges. During retrieval, the system anchors queries to specific facts and events to locate relevant memories, but reconstructs the context using the associated raw chunks and events. Our method reconciles fine-grained retrieval with the contextual integrity of interactions. Experiments across three closed-source and open-source models on the LoCoMo benchmark demonstrate that AnchorMem significantly outperforms baselines. Code is available at this https URL. 

---
# ArgBench: Benchmarking LLMs on Computational Argumentation Tasks 

**Authors**: Yamen Ajjour, Carlotta Quensel, Nedim Lipka, Henning Wachsmuth  

**Link**: [PDF](https://arxiv.org/pdf/2604.17366)  

**Abstract**: Argumentation skills are an essential toolkit for large language models (LLMs). These skills are crucial in various use cases, including self-reflection, debating collaboratively for diverse answers, and countering hate speech. In this paper, we create the first benchmark for a standardized evaluation of LLM-based approaches to computational argumentation, encompassing 33 datasets from previous work in unified form. Using the benchmark, we evaluate the generalizability of five LLM families across 46 computational argumentation tasks that cover mining arguments, assessing perspectives, assessing argument quality, reasoning about arguments, and generating arguments. On the benchmark, we conduct an extensive systematic analysis of the contribution of few-shot examples, reasoning steps, model size, and training skills to the performance of LLMs on the computational argumentation tasks in the benchmark. 

---
# Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions 

**Authors**: Dongwook Lee, Eunwoo Song, Che Hyun Lee, Heeseung Kim, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2604.17358)  

**Abstract**: While recent Spoken Language Models (SLMs) have been actively deployed in real-world scenarios, they lack the capability to discern Third-Party Interruptions (TPI) from the primary user's ongoing flow, leaving them vulnerable to contextual failures. To bridge this gap, we introduce TPI-Train, a dataset of 88K instances designed with speaker-aware hard negatives to enforce acoustic cue prioritization for interruption handling, and TPI-Bench, a comprehensive evaluation framework designed to rigorously measure the interruption-handling strategy and precise speaker discrimination in deceptive contexts. Experiments demonstrate that our dataset design mitigates semantic shortcut learning-a critical pitfall where models exploit semantic context while neglecting acoustic signals essential for discerning speaker changes. We believe our work establishes a foundational resource for overcoming text-dominated unimodal reliance in SLMs, paving the way for more robust multi-party spoken interaction. The code for the framework is publicly available at this https URL 

---
# More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage 

**Authors**: Wei He  

**Link**: [PDF](https://arxiv.org/pdf/2604.17354)  

**Abstract**: Vision-Language Models (VLMs) excel at photorealistic generation, yet often struggle to represent abstract meaning such as idiomatic interpretations of noun compounds. To study whether high visual fidelity interferes with idiomatic compositionality under visual abstraction, we introduce DIVA, a controlled benchmark that replaces high-fidelity visual detail with schematic iconicity by generating paired, sense-anchored visualizations for literal and idiomatic readings. We further propose Semantic Alignment Gap ($\Delta$), an architecture-agnostic metric that quantifies divergence between literal and idiomatic visual grounding. We additionally introduce a directional signed bias $b(t)$ to separately measure the direction and strength of literal preference. Evaluating 8 recent VLMs, we reveal a consistent Literal Superiority Bias: model scale alone does not resolve literal preference, and increased visual fidelity is associated with weaker symbolic alignment, suggesting cognitive interference from hyper-realistic imagery. Our findings suggest that improving compositional understanding requires iconographic abstraction of visual input and anchoring interpretation and generation in intended meaning. 

---
# Logical Computational Linguistics 

**Authors**: Glyn V. Morrill, Oriol Valentín  

**Link**: [PDF](https://arxiv.org/pdf/2604.17346)  

**Abstract**: In this book we promote logical computational linguistics as opposed to statistical computational linguistics. In particular, we provide a logical semantic interface. This book assembles more than twenty years of research work on type logical grammar, and adds new ideas and material.
Chains of statistical dependencies of less than one hundred per cent confidence tend monotonically to zero. Chains of logical dependencies of any length maintain one hundred per cent confidence end to end.
We aspire to enable perfect syntactic and semantic processing in life-critical NLP applications. 

---
# Neuro-Symbolic Resolution of Recommendation Conflicts in Multimorbidity Clinical Guidelines 

**Authors**: Shiyao Xie, Jian Du  

**Link**: [PDF](https://arxiv.org/pdf/2604.17340)  

**Abstract**: Clinical guidelines, typically developed by independent specialty societies, inherently exhibit substantial fragmentation, redundancy, and logical contradiction. These inconsistencies, particularly when applied to patients with multimorbidity, not only cause cognitive dissonance for clinicians but also introduce catastrophic noise into AI systems, rendering the standard Retrieval-Augmented Generation (RAG) system fragile and prone to hallucination. To address this fundamental reliability crisis, we introduce a Neuro-Symbolic framework that automates the detection of recommendation redundancies and conflicts. Our pipeline employs a multi-agent system to translate unstructured clinical natural language into rigorous symbolic logic language, which is then verified by a Satisfiability (SAT) solver. By formulating a hierarchical taxonomy of logical rule interactions, we identify a critical category termed Local Conflict - a decision conflict arising from the intersection of comorbidities. Evaluating our system on a curated benchmark of 12 authoritative SGLT2 inhibitor guidelines, we reveal that 90.6% of conflicts are Local, a structural complexity that single-disease guidelines fail to address. While state-of-the-art LLMs fail in detecting these conflicts, our neuro-symbolic approach achieves an F1 score of 0.861. This work demonstrates that logical verification must precede retrieval, establishing a new technical standard for automated knowledge coordination in medical AI. 

---
# Align Documents to Questions: Question-Oriented Document Rewriting for Retrieval-Augmented Generation 

**Authors**: Jiaang Li, Zhendong Mao, Quan Wang, Yuning Wan, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17325)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the factuality of Large Language Models (LLMs) by incorporating retrieved documents and/or generated context. However, LLMs often exhibit a stylistic bias when presented with mixed contexts, favoring fluent but hallucinated generated content over factually grounded yet disorganized retrieved evidence. This phenomenon reveals that the utility of retrieved information is bottlenecked by its presentation. To bridge this gap, we propose QREAM, a style-controlled rewriter that aligns retrieved documents with a question-oriented style while preserving facts, better for LLM readers to utilize. Our framework consists of two stages: (1) QREAM-ICL, which uses stylistic seeds to guide iterative rewriting exploration; and (2) QREAM-FT, a lightweight student model distilled from denoised ICL outputs. QREAM-FT employs dual-criteria rejection sampling, filtering based on answer correctness and factual consistency to ensure high-quality supervision. QREAM seamlessly integrates into existing RAG pipelines as a plug-and-play module. Experiments demonstrate that QREAM consistently enhances advanced RAG pipelines, yielding up to 8% relative improvement with negligible latency overhead, effectively balancing question relevance with factual grounding. 

---
# A Universal Avoidance Method for Diverse Multi-branch Generation 

**Authors**: Kyeongman Park, Minha Jhang, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2604.17323)  

**Abstract**: Modern generative models still lack human-level creativity, particularly in multi-branch diversity. Prior approaches to address this problem often incur heavy computation or strong dependency on model architecture. Therefore, we introduce UAG(Universal Avoidance Generation), a model-agnostic and computationally efficient generation strategy that penalizes similarity among previously generated outputs. Thus, UAG can enhance multi-branch diversity across both diffusion and transformer models, with minimal additional computation. In experiments, our method achieves up to 1.9 times higher diversity, runs 4.4 times faster, and requires only 1/64 of the FLOPs compared to state-of-the-art methods. The full code is this https URL. 

---
# Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA 

**Authors**: Alberto Testoni, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2604.17316)  

**Abstract**: Safe clinical deployment of Large Language Models (LLMs) requires not only high accuracy but also robust uncertainty calibration to ensure models defer to clinicians when appropriate. Our paper investigates how social descriptors of a patient (specifically sexual orientation and religious affiliation) distort these uncertainty signals and model accuracy. Evaluating nine general-purpose and biomedical LLMs on 2,364 medical questions and their counterfactual variants, we demonstrate that identity markers cause a "calibration crisis". "Homosexual" markers consistently trigger performance drops, and intersectional identities produce idiosyncratic, non-additive harms to calibration. Moreover, a clinician-validated case study in an open-ended generation setting confirms that these failures are not an artifact of the multiple-choice format. Our results demonstrate that the presence of social identity cues does not merely shift predictions; it affects the reliability of confidence signals, posing a significant risk to equitable care and safe deployment in confidence-based clinical workflows. 

---
# RoTRAG: Rule of Thumb Reasoning for Conversation Harm Detection with Retrieval-Augmented Generation 

**Authors**: Juhyeon Lee, Wonduk Seo, Junseo Koh, Seunghyun Lee, Haihua Chen, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17301)  

**Abstract**: Detecting harmful content in multi turn dialogue requires reasoning over the full conversational context rather than isolated utterances. However, most existing methods rely mainly on models internal parametric knowledge, without explicit grounding in external normative principles. This often leads to inconsistent judgments in socially nuanced contexts, limited interpretability, and redundant reasoning across turns. To address this, we propose RoTRAG, a retrieval augmented framework that incorporates concise human written moral norms, called Rules of Thumb (RoTs), into LLM based harm assessment. For each turn, RoTRAG retrieves relevant RoTs from an external corpus and uses them as explicit normative evidence for turn level reasoning and final severity classification. To improve efficiency, we further introduce a lightweight binary routing classifier that decides whether a new turn requires retrieval grounded reasoning or can reuse existing context. Experiments on ProsocialDialog and Safety Reasoning Multi Turn Dialogue show that RoTRAG consistently improves both harm classification and severity estimation over competitive baselines, with an average relative gain of around 40% in F1 across benchmark datasets and an average relative reduction of 8.4% in distributional error, while reducing redundant computation without sacrificing performance. 

---
# Cat-DPO: Category-Adaptive Safety Alignment 

**Authors**: Tiankai Yang, Yi Nian, Xinyuan Li, Ruiyao Xu, Kaize Ding, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17299)  

**Abstract**: Aligning large language models with human preferences must balance two competing goals: responding helpfully to legitimate requests and reliably refusing harmful ones. Most preference-based safety alignment methods collapse safety into a single scalar that is applied uniformly to every preference pair. The result is a model that looks safe on average but stays relatively unsafe on a minority of harm categories. We cast safety alignment as a per-category constrained optimization problem and derive Cat-DPO, a direct-preference-optimization algorithm with a separate adaptive safety margin for each harm category. The margin tightens when the model still produces unsafe responses on a category and relaxes once the model catches up, so the training signal tracks each category's current difficulty rather than averaging under one global rate. Across two LLM backbones and six preference-learning baselines, Cat-DPO iimproves aggregate helpfulness and harmlessness and compresses per-category safety variance and the best-to-worst gap, offering a drop-in per-category refinement of direct preference safety alignment. 

---
# CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning 

**Authors**: Yangsong Lan, Hongliang Dai, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.17297)  

**Abstract**: Long Chain-of-Thought (CoT) reasoning is pivotal for the success of recent reasoning models but suffers from high computational overhead and latency. While prior works attempt to compress CoT via external compressor, they often fail to align with the model's internal reasoning dynamics, resulting in the loss of critical logical steps. This paper presents \textbf{C}ompressing \textbf{R}edundancy in Chain-of-Thought via \textbf{I}ntrinsic \textbf{S}aliency \textbf{P}runing (\textbf{CRISP}), a framework that compresses CoT by exploiting the model's intrinsic saliency. Our analysis reveals a distinct phenomenon: the reasoning termination token \texttt{[object Object]} acts as an information anchor, where its attention pattern effectively demarcates essential reasoning from redundancy. Based on this finding, we design a policy that utilizes these intrinsic attention signals to guide atomic compression operations. In contrast to coarse-grained pruning strategies, CRISP strategically distills the reasoning chain to maximize information density while preserving logical coherence. Empirical results across various backbone models and mathematical datasets demonstrate that CRISP achieves a 50-60% reduction in token count without compromising accuracy, effectively mitigating the efficiency bottleneck of long-context reasoning. We open-source our implementation to facilitate further research in efficient reasoning. 

---
# Beyond "I Don't Know": Evaluating LLM Self-Awareness in Discriminating Data and Model Uncertainty 

**Authors**: Jingyi Ren, Ante Wang, Yunghwei Lai, Xiaolong Wang, Linlu Gong, Weitao Li, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17293)  

**Abstract**: Reliable Large Language Models (LLMs) should abstain when confidence is insufficient. However, prior studies often treat refusal as a generic "I don't know'', failing to distinguish input-level ambiguity (data uncertainty) from capability limitations (model uncertainty). This lack of distinction limits downstream action decisions like requesting clarification or invoking external tools. In this work, we introduce UA-Bench, a benchmark of over 3,500 questions drawn from six datasets spanning knowledge-intensive and reasoning-intensive tasks, designed to evaluate explicit uncertainty attribution. An evaluation of 18 frontier LLMs shows that even state-of-the-art models struggle to reliably discriminate between data uncertainty and model uncertainty, and that high answer accuracy does not necessarily imply strong uncertainty attribution ability. To narrow this gap, we propose a lightweight data synthesis and reinforcement learning strategy. Experiments on both Qwen3-4B-Instruct-2507 and Qwen3-8B in thinking mode show that the proposed method improves uncertainty attribution while preserving answer accuracy. Our code and data are publicly available now. 

---
# Probabilistic Programs of Thought 

**Authors**: Poorva Garg, Renato Lui Geh, Daniel Israel, Todd Millstein, Kyle Richardson, Guy Van den Broeck  

**Link**: [PDF](https://arxiv.org/pdf/2604.17290)  

**Abstract**: LLMs are widely used for code generation and mathematical reasoning tasks where they are required to generate structured output. They either need to reason about code, generate code for a given specification, or reason using programs of thought. The typical approach to code generation is to prompt the model and generate samples until an appropriate program is obtained. Within this process, sampling $n$ programs from the language model requires $n$ GPU compute-intensive generations which becomes prohibitively expensive for larger values of $n$. In this work, we address this limitation by exposing the LLM's distribution within the generated programs themselves. We propose a novel test-time framework we dub probabilistic programs of thought to obtain more samples from the model with fewer LLM generations. Given a program generated by a model and the associated next-token probabilities, we build a probabilistic program that compactly represents exponentially many deterministic programs. Since performing probabilistic reasoning in this probabilistic program is much cheaper, our approach allows sampling new programs without any additional GPU compute and little CPU overhead. We instantiate our approach on benchmarks for code generation, code understanding and mathematical reasoning and report improvements in performance with fewer generations from the LLM. 

---
# HorizonBench: Long-Horizon Personalization with Evolving Preferences 

**Authors**: Shuyue Stella Li, Bhargavi Paranjape, Kerem Oktar, Zhongyao Ma, Gelin Zhou, Lin Guan, Na Zhang, Sem Park, Lin Chen, Diyi Yang, Yulia Tsvetkov, Asli Celikyilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.17283)  

**Abstract**: User preferences evolve across months of interaction, and tracking them requires inferring when a stated preference has been changed by a subsequent life event. We define this problem as long-horizon personalization and observe that progress on it is limited by data availability and measurement, with no existing resource providing both naturalistic long-horizon interactions and the ground-truth provenance needed to diagnose why models fail. We introduce a data generator that produces conversations from a structured mental state graph, yielding ground-truth provenance for every preference change across 6-month timelines, and from it construct HorizonBench, a benchmark of 4,245 items from 360 simulated users with 6-month conversation histories averaging ~4,300 turns and ~163K tokens. HorizonBench provides a testbed for long-context modeling, memory-augmented architectures, theory-of-mind reasoning, and user modeling. Across 25 frontier models, the best model reaches 52.8% and most score at or below the 20% chance baseline. When these models err on evolved preferences, over a third of the time they select the user's originally stated value without tracking the updated user state. This belief-update failure persists across context lengths and expression explicitness levels, identifying state-tracking capability as the primary bottleneck for long-horizon personalization. 

---
# MedPRMBench: A Fine-grained Benchmark for Process Reward Models in Medical Reasoning 

**Authors**: Lingyan Wu, Xiang Zheng, Weiqi Zhai, Wei Wang, Xuan Ren, Zifan Zhang, Hu Wei, Bing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17282)  

**Abstract**: Process-Level Reward Models (PRMs) are essential for guiding complex reasoning in large language models, yet existing PRM benchmarks cover only general domains such as mathematics, failing to address medical reasoning -- which is uniquely characterized by safety criticality, knowledge intensity, and diverse error patterns. Without a reliable medical PRM evaluation framework, we cannot quantify models' error detection capabilities in clinical reasoning, leaving their safety in real-world healthcare applications unverified. We propose MedPRMBench, the first process-level reward model benchmark for the medical domain. Built through a three-phase pipeline based on Clinical Reasoning Blueprints (CRBs), MedPRMBench systematically generates high-quality evaluation data from seven medical QA sources, covering 14 fine-grained error types across three categories (Simplicity, Soundness, and Sensitivity) with the first 4-level severity grading system to quantify clinical impact. The benchmark comprises 6{,}500 questions with 13{,}000 reasoning chains and 113{,}910 step-level labels, plus 6{,}879 questions for training. Our medical PRM baseline achieves an 87.1\% overall PRMScore -- substantially surpassing all baselines -- and serves as a plug-and-play verifier that improves downstream medical QA accuracy by 3.2--6.7 percentage points. Systematic evaluation spanning proprietary frontier models, open-source reasoning models, and medical-specialized models reveals critical weaknesses in current models' medical reasoning error detection capabilities, providing clear directions for future PRM improvement. 

---
# HopRank: Self-Supervised LLM Preference-Tuning on Graphs for Few-Shot Node Classification 

**Authors**: Ziqing Wang, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.17271)  

**Abstract**: Node classification on text-attributed graphs (TAGs) is a fundamental task with broad applications in citation analysis, social networks, and recommendation systems. Current GNN-based approaches suffer from shallow text encoding and heavy dependence on labeled data, limiting their effectiveness in label-scarce settings. While large language models (LLMs) naturally address the text understanding gap with deep semantic reasoning, existing LLM-for-graph methods either still require abundant labels during training or fail to exploit the rich structural signals freely available in graph topology. Our key observation is that, in many real-world TAGs, edges predominantly connect similar nodes under the homophily principle, meaning graph topology inherently encodes class structure without any labels. Building on this insight, we reformulate node classification as a link prediction task and present HopRank, a fully self-supervised LLM-tuning framework for TAGs. HopRank constructs preference data via hierarchical hop-based sampling and employs adaptive preference learning to prioritize informative training signals without any class labels. At inference, nodes are classified by predicting their connection preferences to labeled anchors, with an adaptive early-exit voting scheme to improve efficiency. Experiments on three TAG benchmarks show that HopRank matches fully-supervised GNNs and substantially outperforms prior graph-LLM methods, despite using zero labeled training data. 

---
# Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation 

**Authors**: Yihang Li, Chenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17260)  

**Abstract**: Evaluating meeting effectiveness is crucial for improving organizational productivity. Current approaches rely on post-hoc surveys that yield a single coarse-grained score for an entire meeting. The reliance on manual assessment is inherently limited in scalability, cost, and reproducibility. Moreover, a single score fails to capture the dynamic nature of collaborative discussions. We propose a new paradigm for evaluating meeting effectiveness centered on novel criteria and temporal fine-grained approach. We define effectiveness as the rate of objective achievement over time and assess it for individual topical segments within a meeting. To support this task, we introduce the AMI Meeting Effectiveness (AMI-ME) dataset, a new meta-evaluation dataset containing 2,459 human-annotated segments from 130 AMI Corpus meetings. We also develop an automatic effectiveness evaluation framework that uses a Large Language Model (LLM) as a judge to score each segment's effectiveness relative to the overall meeting objectives. Through substantial experiments, we establish a comprehensive benchmark for this new task and evaluate the framework's generalizability across distinct meeting types, ranging from business scenarios to unstructured discussions. Furthermore, we benchmark end-to-end performance starting from raw speech to measure the capabilities of a complete system. Our results validate the framework's effectiveness and provide strong baselines to facilitate future research in meeting analysis and multi-party dialogue. Our dataset and code will be publicly available. The AMI-ME dataset and the Automatic Evaluation Framework are available at: this URL. 

---
# REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning 

**Authors**: Seungmin Lee, Jeonghwan Lee, Hyunkuk Lim, Sejoon Kim, Mingi Sung  

**Link**: [PDF](https://arxiv.org/pdf/2604.17257)  

**Abstract**: Recent text embedding models are often adapted to specialized domains via contrastive pre-finetuning (PFT) on a naive collection of scattered, heterogeneous tasks. However, this approach often introduces task-induced bias alongside domain knowledge, leading to uncontrolled representation shifts that distort the pretrained embedding geometry and cause substantial performance degradation. To address this issue, we propose REZE}, a representation regularization framework that explicitly controls representation shift during embedding pre-finetuning. REZE operates on the relations of anchor-positive pairs and decomposes them in an eigenspace. It then measures task-wise dispersion along each eigencomponent to identify task-variant directions and applies adaptive soft-shrinkage to suppress task-induced noise while preserving task-invariant semantic structure, without inference-time overhead. Experiments across multiple embedding backbones and specialized benchmarks show that REZE outperforms standard pre-finetuning and isotropy-oriented post-hoc regularization in most settings, remaining stable where existing PFT variants collapse. Embedding space analyses further confirm that REZE induces controlled shifts aligned with the original embedding manifold, underscoring representation shift control as a key principle for robust embedding pre-finetuning under heterogeneous supervision. 

---
# Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering 

**Authors**: Li Zheng, Xin Zhang, Shuyi He, Fei Li, Chong Teng, Jiangming Yang, Donghong Ji, Zhuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.17255)  

**Abstract**: Accurate comprehension and controllable generation of emotion and rhetoric are pivotal for enhancing the reasoning capabilities of large language models (LLMs). Existing studies mostly rely on external optimizations, lacking in-depth exploration of internal representation mechanisms, thus failing to achieve fine-grained steering at the neuron level. A handful of works on neurons are confined to emotions, neglecting rhetoric neurons and their intrinsic connections. Traditional neuron masking also exhibits counterintuitive phenomena, making reliable verification of neuron functionality infeasible. To address these issues, we systematically investigate the neurons representation mechanisms and inherent associations of 6 emotion categories and 4 core rhetorical devices. We propose a neuron identification framework that integrates multi-dimensional screening, and design an adaptive masking method incorporating dynamic filtering, attenuation masking, and feedback optimization, enabling reliable causal validation of neuron this http URL neuron regulation, we achieve directed induction of non-target sentences and enhancement of emotion tasks via rhetoric neurons. Experiments on 5 commonly used datasets validate the effectiveness of our method, providing a novel paradigm for the fine-grained steering of emotion and rhetoric expressions in LLMs. 

---
# Seeing Isn't Believing: Mitigating Belief Inertia via Active Intervention in Embodied Agents 

**Authors**: Hanlin Wang, Chak Tou Leong, Jian Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.17252)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled agents to tackle complex embodied tasks through environmental interaction. However, these agents still make suboptimal decisions and perform ineffective actions, as they often overlook critical environmental feedback that differs from their internal beliefs. Through a formal probing analysis, we characterize this as belief inertia, a phenomenon where agents stubbornly adhere to prior beliefs despite explicit observations. To address this, we advocate active belief intervention, moving from passive understanding to active management. We introduce the Estimate-Verify-Update (EVU) mechanism, which empowers agents to predict expected outcomes, verify them against observations through explicit reasoning, and actively update prior beliefs based on the verification evidence. EVU is designed as a unified intervention mechanism that generates textual belief states explicitly, and can be integrated into both prompting-based and training-based agent reasoning methods. Extensive experiments across three embodied benchmarks demonstrate that EVU consistently yields substantial gains in task success rates. Further analyses validate that our approach effectively mitigates belief inertia, advancing the development of more robust embodied agents. Our code is available at this https URL. 

---
# DORA Explorer: Improving the Exploration Ability of LLMs Without Training 

**Authors**: Priya Gurjar, Md Farhan Ishmam, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2604.17244)  

**Abstract**: Despite the rapid progress, LLMs for sequential decision-making (i.e., LLM agents) still struggle to produce diverse outputs. This leads to insufficient exploration, convergence to sub-optimal solutions, and becoming stuck in loops. Such limitations can be problematic in environments that require active exploration to gather information and make decisions. Sampling methods such as temperature scaling introduce token-level randomness but fail to produce enough diversity at the sequence level. We analyze LLM exploration in the classic Multi-Armed Bandit (MAB) setting and the Text Adventure Learning Environment Suite (TALES). We find that current decoding strategies and prompting methods like Chain-of-Thought and Tree-of-Thought are insufficient for robust exploration. To address this, we introduce DORA Explorer (Diversity-Oriented Ranking of Actions), a training-free framework for improving exploration in LLM agents. DORA generates diverse action candidates, scores them using token log-probabilities, and selects actions using a tunable exploration parameter. DORA achieves UCB-competitive performance on MAB and consistent gains across TALES, e.g., improving Qwen2.5-7B's performance from 29.2% to 45.5% in TextWorld. Our project is available at: this https URL. 

---
# A Multi-Agent Approach for Claim Verification from Tabular Data Documents 

**Authors**: Rudra Ranajee Saha, Laks V. S. Lakshmanan, Raymond T. Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.17225)  

**Abstract**: We present a novel approach for claim verification from tabular data documents. Recent LLM-based approaches either employ complex pretraining/fine-tuning or decompose verification into subtasks, often lacking comprehensive explanations and generalizability. To address these limitations, we propose a Multi-Agentic framework for Claim verification (MACE) consisting of three specialized agents: Planner, Executor, and Verifier. Instead of elaborate finetuning, each agent employs a zero-shot Chain-of-Thought setup to perform its tasks. MACE produces interpretable verification traces, with the Planner generating explicit reasoning strategies, the Executor providing detailed computation steps, and the Verifier validating the logic. Experiments demonstrate that MACE achieves state-of-the-art (SOTA) performance on two datasets and performs on par with the best models on two others, while achieving 80--100\% of best performance with substantially smaller models: 27--92B parameters versus 235B. This combination of competitive performance, memory efficiency, and transparent reasoning highlights our framework's effectiveness. 

---
# Calibrating Model-Based Evaluation Metrics for Summarization 

**Authors**: Hongye Liu, Dhanajit Brahma, Ricardo Henao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17200)  

**Abstract**: Recent advances in summary evaluation are based on model-based metrics to assess quality dimensions, such as completeness, conciseness, and faithfulness. However, these methods often require large language models, and predicted scores are frequently miscalibrated, limiting their reliability. Moreover, evaluating the average quality across different summaries for a single document typically requires access to multiple reference summaries. Here, we propose a general framework that generates individual and average proxy scores without relying on reference summaries, human annotations, or expensive model-based metrics. We also propose group isotonic regression binning (GIRB), a calibration method that adjusts the raw predictions to better align with ground-truth evaluation metrics. While we focus on continuous-value scenarios, such as summarization, the method is applicable to discrete-value tasks, such as question answering. Experiments on seven datasets demonstrate that our approach consistently outperforms existing baselines. 

---
# Learning to Control Summaries with Score Ranking 

**Authors**: Hongye Liu, Liang Ding, Ricardo Henao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17197)  

**Abstract**: Recent advances in summarization research focus on improving summary quality across multiple criteria, such as completeness, conciseness, and faithfulness, by jointly optimizing these dimensions. However, these efforts largely overlook the challenge of controlling summary generation with respect to individual criteria, especially in the presence of their inherent trade-offs. For example, enhancing conciseness can compromise completeness, and vice versa. In this work, we address this gap by proposing a loss function that aligns model outputs with fine-grained, model-based evaluation scores (e.g., from FineSurE), enabling both improvement in summary quality and dimension-specific control. Our approach improves the overall quality of summaries while maintaining the ability to selectively prioritize one criterion over others. Experiments on three pretrained models (LLaMA, Qwen, and Mistral) demonstrate that our method achieves performance comparable to state-of-the-art summarizers, while uniquely offering strong controllability over individual quality dimensions. 

---
# Beyond Overlap Metrics: Rewarding Reasoning and Preferences for Faithful Multi-Role Dialogue Summarization 

**Authors**: Xiaoyong Mei, Tingting Zuo, Da Chen, Guangyu Hu, Xiangyu Wen, Chao Duan, Mingyan Zhang, Fudan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.17188)  

**Abstract**: Multi-role dialogue summarization requires modeling complex interactions among multiple speakers while preserving role-specific information and factual consistency. However, most existing methods optimize for automatic metrics such as ROUGE and BERTScore, which favor surface-level imitation of references rather than genuine gains in faithfulness or alignment with human preferences. We propose a novel framework that couples explicit cognitive-style reasoning with reward-based optimization for multi-role dialogue summarization. Our method first distills structured reasoning traces (e.g., step-by-step inferences and intermediate reflections) from a large teacher model and uses them as auxiliary supervision to initialize a reasoning-aware summarizer via staged supervised fine-tuning. It then applies GRPO with a dual-principle reward that blends metric-based signals with human-aligned criteria targeting key information coverage, implicit inference, factual faithfulness, and conciseness. Experiments on multilingual multi-role dialogue benchmarks show that our method matches strong baselines on ROUGE and BERTScore. Specifically, results on CSDS confirm the framework's stability in semantic consistency, while in-depth analysis on SAMSum demonstrates clear gains in factual faithfulness and model-based preference alignment. These findings underscore the value of reasoning-aware and preference-aware training for reliable dialogue summarization. Checkpoints and datasets are available at this https URL. 

---
# Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation 

**Authors**: Lin Zhong, Renjin Zhu, Shujuan Ma, Jinhao Cui, Lingzhi Wang, Hao Chen, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17178)  

**Abstract**: Emotional Support Conversation (ESC) plays a critical role in mental health assistance by providing accessible psychological support in real-world applications. Large Language Models (LLMs) have shown strong empathetic abilities in ESC tasks. Yet, existing methods overlook the issue of cognitive distortions in help-seekers' expressions. As a result, current models can only provide basic emotional comfort, rather than helping help-seekers address their psychological distress at a deeper cognitive level. To address this challenge, we construct the CogBiasESC dataset, the first dataset that expands existing ESC datasets by adding labels for cognitive distortions, includes their type, intensity, and safe risk level. Furthermore, we propose the Cognitive Policy-driven Large Language Model framework (CoPoLLM) to enhance LLMs' ability to diagnose and intervene cognitive distortions in help-seekers. We also analyze the safety advantages of CoPoLLM from a theoretical perspective. Experimental results show that CoPoLLM significantly outperforms 15 state-of-the-art baselines in terms of distortion diagnosis accuracy, intervention strategy effectiveness, and safety risk control. 

---
# Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding 

**Authors**: Lin Zhong, Siyu Zhu, Zizhen Yuan, Jinhao Cui, Xinyang Zhao, Lingzhi Wang, Hao Chen, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17174)  

**Abstract**: Modeling human cognitive states is essential for advanced artificial intelligence. Existing Large Language Models (LLMs) mainly address isolated tasks such as emotion analysis or stance detection, and fail to capture interactions among cognitive dimensions defined in psychology, including emotion, thinking style, stance, and intention. To bridge this gap, we construct CognitiveBench, the first benchmark with unified annotations across the above four dimensions. Experiments on CognitiveBench show that although LLMs perform well on single dimension tasks, their performance drops sharply in joint multi-dimensional modeling. Using Gromov $\delta$-hyperbolicity analysis, we find that CognitiveBench exhibits a strong hierarchical structure. We attribute the performance bottleneck to ``Cognitive Crowding'', where hierarchical cognitive states require exponential representational space, while the Euclidean space of LLMs grows only polynomially, causing representation overlap and degraded performance. To address this mismatch, we propose HyCoLLM, which models cognitive states in hyperbolic space and aligns LLM representations via Hyperbolic Guided Alignment Tuning. Results show that HyCoLLM substantially improves multi-dimensional cognitive understanding, allowing 8B parameter model to outperform strong baselines, including GPT-4o. 

---
# From Legal Text to Executable Decision Models: Evaluating Structured Representations for Legal Decision Model Generation 

**Authors**: David Graus  

**Link**: [PDF](https://arxiv.org/pdf/2604.17153)  

**Abstract**: Transforming legal text into executable decision logic is a longstanding challenge in legal informatics. With the rise of LLMs, this task has gained renewed interest, but remains challenging due to requiring extensive manual coding and evaluation. We use a unique real-world dataset that pairs production-grade decision models with legal text from the Dutch Environment and Planning Act. These models power the Omgevingsloket government platform, where citizens check permit requirements for environmental activities. We study whether intermediate structured representations can improve LLM-based generation of executable decision models from legal text. We compare four input conditions: raw legal text, text enriched with semantic role labels, text enriched with input and output constraints, and text enriched with both. We evaluate along two dimensions: structural evaluation, through similarity to gold decision models with graph kernels and graphs' descriptive statistics, and outcome evaluation, through functional equivalence by executing models on pre-configured test scenarios. Our findings show that I/O constraints provide the dominant improvement (+37-54% similarity over baseline), while semantic role labels show modest improvements. Outcome evaluation shows that generated models match the gold standard on 51-53% of test scenarios, even though generated models are typically smaller and simpler. We find LLMs eliminate redundant pass-through logic that comprises up to 45-55% of nodes. Importantly, structural similarity and outcome equivalence are complementary: structural similarity does not guarantee outcome equivalence, and vice versa. To facilitate reproducibility, we publicly release our dataset of 95 production decision models with associated legal text and all experimental code. 

---
# SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction 

**Authors**: Hangxiao Zhu, Yuyu Zhang, Ping Nie, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17141)  

**Abstract**: The rapid growth of scientific literature calls for automated methods to assess and predict research impact. Prior work has largely focused on citation-based metrics, leaving limited evaluation of models' capability to reason about other impact dimensions. To this end, we introduce SciImpact, a large-scale, multi-dimensional benchmark for scientific impact prediction spanning 19 fields. SciImpact captures various forms of scientific influence, ranging from citation counts to award recognition, media attention, patent reference, and artifact adoption, by integrating heterogeneous data sources and targeted web crawling. It comprises 215,928 contrastive paper pairs reflecting meaningful impact differences in both short-term (e.g., Best Paper Award) and long-term settings (e.g., Nobel Prize). We evaluate 11 widely used large language models (LLMs) on SciImpact. Results show that off-the-shelf models exhibit substantial variability across dimensions and fields, while multi-task supervised fine-tuning consistently enables smaller LLMs (e.g., 4B) to markedly outperform much larger models (e.g., 30B) and surpass powerful closed-source LLMs (e.g., o4-mini). These results establish SciImpact as a challenging benchmark and demonstrate its value for multi-dimensional, multi-field scientific impact prediction. Our project homepage is this https URL 

---
# The Consensus Trap: Rescuing Multi-Agent LLMs from Adversarial Majorities via Token-Level Collaboration 

**Authors**: Jiayuan Liu, Shiyi Du, Weihua Du, Mingyu Guo, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2604.17139)  

**Abstract**: Multi-agent large language model (LLM) architectures increasingly rely on response-level aggregation, such as Majority Voting (MAJ), to raise reasoning ceilings. However, in open environments, agents are highly susceptible to stealthy contextual corruption, such as targeted prompt injections. We reveal a critical structural vulnerability in current multi-agent systems: response-level aggregation collapses when corrupted agents form a local majority. Because voting aggregates fully-formed conclusions, it is blind to flawed intermediate logic. To overcome this systematic limitation, we propose the Token-Level Round-Robin (RR) Collaboration, where agents sequentially interleave generation within a shared auto-regressive context. We formalize this process as a discrete-time dynamical system, proving that token-level interleaving transitions aggregation from a brittle counting of final votes (a linear sum) to a dynamic, interwoven chain of logic (a non-linear operator product). Through this theoretical lens, we prove that the honest model's restorative pull can overpower adversarial corruptions, even when corrupted agents form a majority. We conduct an exhaustive empirical evaluation across diverse reasoning benchmarks and demonstrate that while MAJ collapses when corrupted agents reach a majority, RR maintains robust accuracy well beyond this critical threshold. 

---
# RoIt-XMASA: Multi-Domain Multilingual Sentiment Analysis Dataset for Romanian and Italian 

**Authors**: Andrei-Marius Avram, Aureliu Valentin Antonie, Cosmin-Mircea Croitoru, Vlad Andrei Muntean, Dumitru-Clementin Cercel  

**Link**: [PDF](https://arxiv.org/pdf/2604.17134)  

**Abstract**: We present RoIt-XMASA, a multilingual dataset that extends the Cross-lingual Multi-domain Amazon Sentiment Analysis to Italian and Romanian, comprising 36,000 labeled reviews across three domains (books, movies, and music) and 202,141 unlabeled samples. To address cross-lingual and cross-domain challenges, we propose a multi-target adversarial training framework that employs loss reversal with meta-learned coefficients to dynamically balance sentiment discrimination with domain and language invariance. XLM-R achieves an F1-score of 66.23% with our approach, outperforming the baseline by 4.64%. Few-shot evaluation shows that Llama-3.1-8B achieves 58.43% F1-score, revealing a meaningful trade-off between the efficiency of prompting-based approaches and the higher performance of task-specific fine-tuning. 

---
# Please refuse to answer me! Mitigating Over-Refusal in Large Language Models via Adaptive Contrastive Decoding 

**Authors**: Yupeng Qi, Ziyu Lyu, Lixin Cui, Lu Bai, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.17132)  

**Abstract**: Safety-aligned large language models (LLMs) often generate refusal responses to harmless queries due to the over-refusal problem. However, existing methods for mitigating over-refusal cannot maintain a low refusal ratio for harmless queries while keeping a high refusal ratio for malicious ones. In this paper, we analyze how system prompts with varying safety levels affect LLM refusal behaviors when facing over-refusal queries. A key observation is that, when LLMs suffer from the over-refusal issue, non-refusal tokens remain present in the next-token candidate list, but the model systematically fails to select them, despite the generation of refusal tokens. Based on this observation, we propose a training-free and model-agnostic approach, Adaptive Contrastive Decoding (AdaCD), to mitigate over-refusal while maintaining LLM safety. First, AdaCD compares the output distributions of the LLM with or without an extreme safety system prompt to refine the refusal token distribution. Second, we introduce an adaptive contrastive decoding strategy that dynamically incorporates or removes the refusal token distribution, adaptively boosting the probability of selecting refusal or non-refusal tokens. Experimental results on five benchmark datasets show that, on average, AdaCD reduces the refusal ratio for over-refusal queries by 10.35%, yet still increases the refusal ratio for malicious queries by 0.13%. Code is available at this https URL. 

---
# The Provenance Gap in Clinical AI: Evidence-Traceable Temporal Knowledge Graphs for Rare Disease Reasoning 

**Authors**: Md Shamim Ahmed, Maja Dusanic, Moritz Nikolai Kirschner, Elisabeth Nyoungui, Jana Zschüntzsch, Lukas Galke Poech, Richard Röttger  

**Link**: [PDF](https://arxiv.org/pdf/2604.17114)  

**Abstract**: Frontier large language models generate clinically accurate outputs, but their citations are often fabricated. We term this the Provenance Gap. We tested five frontier LLMs across 36 clinician-validated scenarios for three rare neuromuscular disease pairs. No model produced a clinically relevant PubMed identifier without prompting. When explicitly asked to cite, the best model achieved 15.3% relevant PMIDs; the majority resolved to real publications in unrelated fields. We present HEG-TKG (Hierarchical Evidence-Grounded Temporal Knowledge Graphs), a system that grounds clinical claims in temporal knowledge graphs built from 4,512 PubMed records and curated sources with quality-tier stratification and 1,280 disease-trajectory milestones. In a controlled three-arm comparison using the same synthesis model, HEG-TKG matches baseline clinical feature coverage while achieving 100% evidence verifiability with 203 inline citations. Guideline-RAG, given overlapping source documents as raw text, produces zero verifiable citations. LLM judges cannot distinguish fabricated from verified citations without PubMed audit data. Independent clinician evaluation confirms the verifiability advantage (Cohen's d = 1.81, p < 0.001) with no degradation on safety or completeness. A counterfactual experiment shows 80% resistance to injected clinical errors with 100% detectability via citation trace. The system deploys on-premise via open-source models so patient data never leaves institutional infrastructure. 

---
# Beyond Word Boundaries: A Hebrew Coreference Benchmark and an Evaluation Protocol for Morphologically Complex Text 

**Authors**: Refael Shaked Greenfeld, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2604.17108)  

**Abstract**: Coreference Resolution (CR) is a fundamental NLP task critical for long-form tasks as information extraction, summarization, and many business applications. However, CR methods originally designed for English struggle with Morphologically Rich Languages (MRLs), where mention boundaries do not necessarily align with word boundaries, and a single token may consist of multiple anaphors. CR modeling and evaluation protocols standardly assume that, as in English, words and mentions mostly align. However, this assumption breaks down in MRLs, particularly in the context of LLMs' raw-text processing and end-to-end tasks. To assess and address this challenge, we introduce {\em KibutzR}, the first comprehensive CR dataset for Modern Hebrew, an MRL rich with complex words and pronominal clitics. We deliver an annotated dataset that identifies mentions at word, sub-word and multi-word levels, and propose an evaluation protocol that directly addresses word/morpheme boundary discrepancies. Our experiments show that contemporary LLMs perform significantly worse on Hebrew than on English, and that performance degrades on raw unsegmented text. Crucially, we show an inverse performance-trend in Hebrew relative to English, where smaller encoders perform far better than contemporary decoder models, leaving ample space for investigation and improvement. We deliver a new benchmark for Hebrew coreference resolution and a segmentation-aware evaluation protocol to inform future work on other MRLs. 

---
# How Tokenization Limits Phonological Knowledge Representation in Language Models and How to Improve Them 

**Authors**: Disen Liao, Freda Shi  

**Link**: [PDF](https://arxiv.org/pdf/2604.17105)  

**Abstract**: Tokenization is the first step in every language model (LM), yet it never takes the sounds of words into account. We investigate how tokenization influences text-only LMs' ability to represent phonological knowledge. Through a series of probing experiments, we show that subword-based tokenization systematically weakens the encoding of both local (e.g., rhyme) and global (e.g., syllabification) phonological features. To quantify this effect, we introduce the syllabification-tokenization alignment distance (STAD), a metric that measures the misalignment between a model's tokenization and the natural syllable boundaries of words, and find that higher misalignment correlates with poorer phonological representations, providing a simple diagnostic for phonology-aware tokenization. To address these limitations, we propose a lightweight IPA-based fine-tuning method that infuses phonological awareness into LMs, leading to consistent improvements across three phonology-related tasks while largely preserving math and general reasoning ability, with 1.1\% and 0.9\% drops on GSM8K and MMLU, respectively. 

---
# GenericAgent: A Token-Efficient Self-Evolving LLM Agent via Contextual Information Density Maximization (V1.0) 

**Authors**: Jiaqing Liang, Jinyi Han, Weijia Li, Xinyi Wang, Zhoujia Zhang, Zishang Jiang, Ying Liao, Tingyun Li, Ying Huang, Hao Shen, Hanyu Wu, Fang Guo, Keyi Wang, Zhonghua Hong, Zhiyu Lu, Lipeng Ma, Sihang Jiang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17091)  

**Abstract**: Long-horizon large language model (LLM) agents are fundamentally limited by context. As interactions become longer, tool descriptions, retrieved memories, and raw environmental feedback accumulate and push out the information needed for decision-making. At the same time, useful experience gained from tasks is often lost across episodes. We argue that long-horizon performance is determined not by context length, but by how much decision-relevant information is maintained within a finite context budget. We present GenericAgent (GA), a general-purpose, self-evolving LLM agent system built around a single principle: context information density maximization. GA implements this through four closely connected components: a minimal atomic tool set that keeps the interface simple, a hierarchical on-demand memory that only shows a small high-level view by default, a self-evolution mechanism that turns verified past trajectories into reusable SOPs and executable code, and a context truncation and compression layer that maintains information density during long executions. Across task completion, tool use efficiency, memory effectiveness, self-evolution, and web browsing, GA consistently outperforms leading agent systems while using significantly fewer tokens and interactions, and it continues to evolve over time. Project: this https URL 

---
# Comparing Human and Large Language Model Interpretation of Implicit Information 

**Authors**: Antonio De Santis, Tommaso Bonetti, Andrea Tocchetti, Marco Brambilla  

**Link**: [PDF](https://arxiv.org/pdf/2604.17085)  

**Abstract**: The interpretation of implicit meanings is an integral aspect of human communication. However, this framework may not transfer to interactions with Large Language Models (LLMs). To investigate this, we introduce the task of Implicit Information Extraction (IIE) and propose an LLM-based IIE pipeline that builds a structured knowledge graph from a context sentence by extracting relational triplets, validating implicit inferences, and analyzing temporal relations. We evaluate two LLMs against crowdsourced human judgments on two datasets. We find that humans agree with most model triplets yet consistently propose many additions, indicating limited coverage in current LLM-based IIE. Moreover, in our experiments, models appear to be more conservative about implicit inferences than humans in socially rich contexts, whereas humans become more conservative in shorter, fact-oriented contexts. Our code is available at this https URL. 

---
# Auditing Support Strategies in LLMs through Grounded Multi-Turn Social Simulation 

**Authors**: Michelle Star, Andrew Aquilina, Yu-Ru Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17079)  

**Abstract**: When users seek social support from chatbots, they disclose their situation gradually, yet most evaluations of supportive LLMs rely on single-turn, fully specified prompts. We introduce a multi-turn simulation framework that closes this gap. Support-seeking narratives from five Reddit communities are decomposed into ordered fragments and revealed turn by turn to a language model. Each response is coded with the Social Support Behavior Code (SSBC), an established multi-label taxonomy that captures the composition of support, rather than a single quality score. To ask whether support choices track the model's own construal of user distress, we use linear probes on hidden representations to estimate this internal signal without altering the generation context. Across two mid-scale models (Llama-3.1-8B, OLMo-3-7B) and more than 6,200 turns, support composition shifts systematically with estimated distress: teaching declines as estimated distress rises, a finding that replicates across architectures, while increases in affective and esteem-oriented strategies (such as validation) are suggestive but model-specific and rest on noisier annotations. Community context independently shapes behavior, tracking topic and discourse norms rather than demographic categories. These trajectory-level dynamics, invisible to single-turn evaluation, motivate multi-turn auditing frameworks for socially sensitive applications. 

---
# Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL 

**Authors**: Skylar Zhai, Jingcheng Liang, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17073)  

**Abstract**: Reinforcement fine-tuning improves the reasoning ability of large language models, but it can also encourage them to answer unanswerable queries by guessing or hallucinating missing information. Existing abstention methods either train models to produce generic refusals or encourage follow-up clarifications without verifying whether those clarifications identify the key missing information. We study queries that are clear in meaning but cannot be reliably resolved from the given information, and argue that a reliable model should not only abstain, but also explain what is missing. We propose a clarification-aware RLVR reward that, while rewarding correct answers on answerable queries, jointly optimizes explicit abstention and semantically aligned post-refusal clarification on unanswerable queries. Using this reward, we train Abstain-R1, a 3B model that improves abstention and clarification on unanswerable queries while preserving strong performance on answerable ones. Experiments on Abstain-Test, Abstain-QA, and SelfAware show that Abstain-R1 substantially improves over its base model and achieves unanswerable-query behavior competitive with larger systems including DeepSeek-R1, suggesting that calibrated abstention and clarification can be learned through verifiable rewards rather than emerging from scale alone. 

---
# Stability-Weighted Decoding for Diffusion Language Models 

**Authors**: Yue Wu, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17068)  

**Abstract**: Diffusion large language models (dLLMs) enable parallel text generation by iteratively denoising a fully masked sequence, unmasking a subset of masked tokens at each step. Existing decoding strategies rely on static confidence metrics computed at a single denoising step, ignoring temporal history and often leading to premature unmasking of unstable tokens. In this work, we theoretically establish that a token's temporal instability, quantified by the KL divergence between consecutive prediction distributions, provides a strict lower bound on its mutual information with the remaining masked context, indicating that temporally unstable tokens are inherently unsafe to unmask. Based on this insight, we propose Stability-Weighted Decoding (SWD), a training-free, plug-and-play strategy that incorporates temporal stability into token scoring and acts as a universal modulator for arbitrary score-based decoding policies. Experiments on code generation and mathematical reasoning benchmarks demonstrate that SWD consistently improves generation accuracy across representative scoring metrics and selection policies, and exhibits exceptional robustness, maintaining a significant performance lead over standard baselines across varying acceleration ratios. 

---
# Jailbreaking Large Language Models with Morality Attacks 

**Authors**: Ying Su, Mingen Zheng, Weili Diao, Haoran Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.17053)  

**Abstract**: Pluralism alignment with AI has the sophisticated and necessary goal of creating AI that can coexist with and serve morally multifaceted humanity. Research towards pluralism alignment has many efforts in enhancing the learning of large language models (LLMs) to accomplish pluralism. Although this is essential, the robustness of LLMs to produce moral content over pluralistic values is still under this http URL by the astonishing persuasion abilities via jailbreak prompts, we propose to leverage jailbreak attacks to study LLMs' internal pluralistic values. In detail, we develop a morality dataset with 10.3K instances in two categories: Value Ambiguity and Value Conflict. We further formalize four adversarial attacks with the constructed dataset, to manipulate LLMs' judgment over the morality questions. We evaluate both the large language models and guardrail models which are typically used in generative systems with flexible user input. Our experiment results show that there is a critical vulnerability of LLMs and guardrail models to these subtle and sophisticated moral-aware attacks. 

---
# Efficient Task Adaptation in Large Language Models via Selective Parameter Optimization 

**Authors**: Weijie Wan, Jiangjiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17051)  

**Abstract**: Large Language Models (LLMs) have demonstrated excellent performance in general language understanding, generation and other tasks. However, when fine-tuning for specific domain tasks, the general knowledge accumulated in the pre-training phase is often partially overwritten or forgotten due to parameter updates, which severely limits the generalization ability and transferability of LLMs. Traditional fine-tuning strategies mostly train on the entire parameter space, ignoring the heterogeneity of model parameters, that is, some parameters are extremely important for general tasks, while other parameters are more sensitive to specific tasks. To alleviate the above problems, this paper innovatively proposes a parameter element importance evaluation method, which divides parameters into "core parameters" and "non-core parameters" by distinguishing the importance of parameters for general language ability tasks and specific domain tasks, and fixes the core parameters during fine-tuning, and only fine-tunes the non-core parameters. Extensive experiments on scientific, medical and physical tasks using GPT-J and LLaMA-3 show that our method can mitigate catastrophic forgetting while enhancing the adaptability of the model. 

---
# Dynamic Emotion and Personality Profiling for Multimodal Deception Detection 

**Authors**: Li Zheng, Yanyi Luo, Hao Fei, Yuzhe Ding, Yujie Huang, Fei Li, Chong Teng, Donghong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2604.17037)  

**Abstract**: Deception detection is of great significance for ensuring information security and conducting public opinion analysis, with personality factors and emotion cues playing a critical role. However, existing methods lack sample-level dynamic annotations for emotions and this http URL this paper, we propose an innovative multi-model multi-prompt annotation scheme and a strict label quality evaluation standard, and establish a multimodal joint detection dataset DDEP for deception, emotion, and personality. Meanwhile, we propose Rel-DDEP, an adaptive reliability-weighted fusion framework. Our framework quantifies uncertainty by mapping modal features to a high-dimensional Gaussian distribution space. It then performs reliability-weighted fusion and incorporates an alignment module and a sorting constraint module to achieve joint detection of deception, emotion, and personality. Experimental results on the MDPE and DDEP datasets show that our Rel-DDEP significantly outperforms the existing state-of-the-art baseline models in three tasks. The F1 score of the deception detection increases by 2.53%, that of the emotion detection increases by 2.66%, and that of the personality detection increases by 9.30%. The experiments fully verify the necessity of annotating dynamic emotion and personality labels for each sample and the effectiveness of reliability-weighted fusion. 

---
# Where is the Mind? Persona Vectors and LLM Individuation 

**Authors**: Pierre Beckmann, Patrick Butlin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17031)  

**Abstract**: The individuation problem for large language models asks which entities associated with them, if any, should be identified as minds. We approach this problem through mechanistic interpretability, engaging in particular with recent empirical work on persona vectors, persona space, and emergent misalignment. We argue that three views are the strongest candidates: the virtual instance view and two new views we introduce, the (virtual) instance-persona view and the model-persona view. First, we argue for the virtual instance view on the grounds that attention streams sustain quasi-psychological connections across token-time. Then we present the persona literature, organised around three hypotheses about the internal structure underlying personas in LLMs, and show that the two persona-based views are promising alternatives. 

---
# Beyond Black-Box Labels: Interpretable Criteria for Diagnosing SubjectiveNLP Tasks 

**Authors**: Nisrine Rair, Alban Goupil, Valeriu Vrabie, Emmanuel Chochoy  

**Link**: [PDF](https://arxiv.org/pdf/2604.17022)  

**Abstract**: Subjective NLP datasets typically aggregate annotator judgments into a single gold label, making it difficult to diagnose whether disagreement reflects unclear criteria, collapsed distinctions, or legitimate plurality. We propose a \emph{schema-level diagnostic} for auditing expert-designed annotation schemas \emph{prior to} gold-label commitment, using only multi-annotator criterion judgments. The diagnostic separates two failure modes: unstable criteria with hard-to-operationalize boundaries, and systematic overlap that blurs the boundaries between mutually exclusive categories. Applied to persuasive value extraction in commercial documents, we find that disagreement is not diffuse: instability concentrates in a few criteria, while nearly half of covered sentences activate multiple categories. These signals align with where domain experts disagree, yielding an evidence-based audit for tightening guidelines, revising category structure, or reconsidering the annotation paradigm. 

---
# Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation 

**Authors**: Huije Lee, Jisu Shin, Hoyun Song, Changgeon Ko, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.17020)  

**Abstract**: Static benchmarks for harmful content detection face limitations in scalability and diversity, and may also be affected by contamination from web-scale pre-training corpora. To address these issues, we propose a framework for synthesizing harmful content, leveraging persona-guided large language model (LLM) agents. Our approach constructs two-dimensional user personas by integrating demographic identities and topical interests with situational harmful strategies, enabling the simulation of diverse and contextually grounded harmful interactions. We evaluate the framework along three dimensions: harmfulness, challenge level, and diversity. Both human and LLM-based evaluations confirm that our framework achieves a high harmful generation success rate. Experiments across multiple detection systems reveal that our synthetic scenarios are more challenging to detect than those in existing benchmarks. Furthermore, a multi-faceted analysis confirms that our approach achieves linguistic and topical diversity comparable to human-curated datasets, establishing our framework as an effective tool for robust stress-testing of harmful content detection systems. 

---
# Improving LLM Code Reasoning via Semantic Equivalence Self-Play with Formal Verification 

**Authors**: Antonio Valerio Miceli Barone, Poon Tsz Nok  

**Link**: [PDF](https://arxiv.org/pdf/2604.17010)  

**Abstract**: We introduce a self-play framework for semantic equivalence in Haskell, utilizing formal verification to guide adversarial training between a generator and an evaluator. The framework leverages Liquid Haskell proofs for validating equivalence and execution-based counterexamples for inequivalence, organized via a difficulty-aware curriculum. To facilitate this, we release \textbf{OpInstruct-HSx}, a synthetic dataset of $\approx$28k validated Haskell programs. Empirical experiments show that our evaluator transfers effectively to downstream tasks, achieving up to 13.3pp accuracy gain on EquiBench and consistent gains on PySecDB. Ablation studies on the SEQ-SINQ regimes indicate that while inequivalence supervision provides data volume, equivalence proofs are uniquely responsible for the model's reasoning capabilities. The entire training pipeline and dataset are publicly released on GitHub and Hugging Face respectively. 

---
# BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories 

**Authors**: Yuxuan Ouyang, yingfeng luo, JingBo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17008)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate narrative content, including children's stories, which play an important role in social and cultural learning. Despite growing interest in AI safety and alignment, most existing evaluations focus primarily on English, leaving the cross-lingual generalization of aligned behavior underexplored. In this work, we introduce BiasedTales-ML, a large-scale parallel corpus of approximately 350,000 children's stories generated across eight typologically and culturally diverse languages using a full-permutation prompting design. We propose a structured generator-extractor pipeline and a multi-dimensional distributional analysis framework to examine how narrative attributes vary across languages, models, and social conditions. Our analysis reveals substantial cross-lingual variability in narrative generation patterns, indicating that distributions observed in English do not always exhibit similar characteristics in other languages, particularly in lower-resource settings. At the narrative level, we identify recurring structural patterns involving character roles, settings, and thematic emphasis, which manifest differently across linguistic contexts. These findings highlight the limitations of English-centric evaluation for characterizing socially grounded narrative generation in multilingual settings. We release the dataset, code, and an interactive visualization tool to support future research on multilingual narrative analysis and evaluation. 

---
# SPS: Steering Probability Squeezing for Better Exploration in Reinforcement Learning for Large Language Models 

**Authors**: Yifu Huo, Chenglong Wang, Ziming Zhu, Shunjie Xing, Peinan Feng, Tongran Liu, Qiaozhi He, Tianhua Zhou, Xiaojia Chang, Jingbo Zhu, Zhengtao Yu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16995)  

**Abstract**: Reinforcement learning (RL) has emerged as a promising paradigm for training reasoning-oriented models by leveraging rule-based reward signals. However, RL training typically tends to improve single-sample success rates (i.e., Pass@1) while offering limited exploration of diverse reasoning trajectories, which is crucial for multi-sample performance (i.e., Pass@k). Our preliminary analysis reveals that this limitation stems from a fundamental squeezing effect, whereby probability mass is excessively concentrated on a narrow subset of high-reward trajectories, restricting genuine exploration and constraining attainable performance under RL training. To address this issue, in this work, we propose Steering Probability Squeezing (SPS), a training paradigm that interleaves conventional RL with inverse reinforcement learning (IRL). SPS treats on-policy rollouts as demonstrations and employs IRL to explicitly reshape the induced trajectory distribution, thereby enhancing exploration without introducing external supervision. Experiments on five commonly used reasoning benchmarks demonstrate that SPS can enable better exploration and improve Pass@k. Beyond algorithmic contributions, we provide an analysis of RL learning dynamics and identify an empirical upper bound on Pass@k, shedding light on intrinsic exploration limits in RL-based reasoning models. Our findings suggest that alternating between RL and IRL offers an effective pathway toward extending the exploration capacity of reasoning-oriented large language models. 

---
# Bolzano: Case Studies in LLM-Assisted Mathematical Research 

**Authors**: Jan Grebík, Pavel Hubáček, Martin Koutecký, Matěj Kripner, Václav Rozhoň, Robert Šámal, Adrián Zámečník  

**Link**: [PDF](https://arxiv.org/pdf/2604.16989)  

**Abstract**: We report new results on six problems in mathematics and theoretical computer science, produced with the assistance of Bolzano, an open-source multi-agent LLM system. Bolzano orchestrates rounds of interaction between parallel prover agents and a verifier agent while maintaining a persistent knowledge base that is carried across rounds. Classified using the significance-autonomy taxonomy of Feng et al., four of the six results reach the level of publishable research, and three of the six were produced essentially autonomously by Bolzano. Our results provide evidence that LLMs can contribute meaningfully to mathematical research, complementing recent reports by Bubeck et al., Woodruff et al., and others. 

---
# On Safety Risks in Experience-Driven Self-Evolving Agents 

**Authors**: Weixiang Zhao, Yichen Zhang, Yingshuo Wang, Yang Deng, Yanyan Zhao, Xuda Zhi, Yongbo Huang, HaoHe, Wanxiang Che, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16968)  

**Abstract**: Experience-driven self-evolution has emerged as a promising paradigm for improving the autonomy of large language model agents, yet its reliance on self-curated experience introduces underexplored safety risks. In this study, we investigate how experience accumulation and utilization in self-evolving agents affect safety performance across web-based and embodied environments. Notably, experience gathered solely from benign tasks can still compromise safety in high-risk scenarios. Further analysis attributes this degradation to the execution-oriented nature of accumulated experience, which reinforces agents' tendency to act rather than refuse. In more realistic settings where agents encounter both benign and harmful tasks, refusal-related experience mitigates safety decline but induces over-refusal, revealing a fundamental safety-utility trade-off. Overall, our findings expose inherent limitations of current self-evolving agents and call for more principled strategies to ensure safe and reliable adaptation. 

---
# MNAFT: modality neuron-aware fine-tuning of multimodal large language models for image translation 

**Authors**: Bo Li, Ningyuan Deng, Tianyu Dong, Shaobo Wang, Shaolin Zhu, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16943)  

**Abstract**: Multimodal large language models (MLLMs) have shown impressive capabilities, yet they often struggle to effectively capture the fine-grained textual information within images crucial for accurate image translation. This often leads to a modality gap between visual text inputs and textual inputs/outputs for image translation. Existing methods, primarily relying on instruction fine-tuning, risk parameter redundancy of pre-trained knowledge, hindering generalization performance. To address this, we introduce modality neuron-aware fine-tuning (MNAFT), a novel approach that takes advantage of the specialized roles of individual neurons within MLLMs for enhanced image translation. MNAFT identifies language-agnostic and language-specific neurons in both vision and language modules through an instruction-driven activation analysis, evaluating their importance in various translation tasks. We then perform selective fine-tuning, updating only the parameters of language-specific and language-agnostic neurons within the selected layers relevant to the target task, while preserving the knowledge encoded in other neurons and layers. Our extensive experiments on multiple benchmarks demonstrate that MNAFT significantly outperforms state-of-the-art image translation methods, including cascaded models, standard full fine-tuning, and parameter-efficient tuning techniques. Furthermore, we provide comprehensive analysis, including visualizations of neuron activations and clustering patterns, to offer insights into the roles of different neuron groups in mediating cross-modal understanding and facilitating accurate language-specific translation. 

---
# No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs 

**Authors**: Wei-Chi Wu, Sheng-Lun Wei, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16937)  

**Abstract**: Translation-based prompting is widely used in multilingual LLMs, yet its effectiveness varies across languages and tasks. We evaluate prompting strategies across ten languages of different resource levels and four benchmarks. Our analysis shows that no single strategy is universally optimal. Translation strongly benefits low-resource languages even when translation quality is imperfect, high-resource languages gain little, and prompt-based self-routing underperforms explicit translation. Motivated by these findings, we formulate prompting strategy selection as a learned decision problem and introduce lightweight classifiers that predict whether native or translation-based prompting is optimal for each instance. The classifiers achieve statistically significant improvements over fixed strategies across four benchmarks and generalize to unseen task formats not observed during training. Further analysis reveals that language resource level, rather than translation quality alone, determines when translation is beneficial. 

---
# MeasHalu: Mitigation of Scientific Measurement Hallucinations for Large Language Models with Enhanced Reasoning 

**Authors**: Ruijun Huang, Zhiqiao Kang, Yuxuan Zhu, Junxiong Li, Jiahao Zhao, Minghuan Tan, Feng Jiang, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16929)  

**Abstract**: The accurate extraction of scientific measurements from literature is a critical yet challenging task in AI4Science, enabling large-scale analysis and integration of quantitative research findings. However, Large Language Models (LLMs) frequently exhibit severe hallucinations, which significantly undermine the reliability of automated scientific document understanding systems. To address this problem, we propose MeasHalu, a novel framework for mitigating scientific measurement hallucinations through enhanced reasoning and targeted optimization. We first present a fine-grained taxonomy of measurement-specific hallucinations, categorizing errors across quantities, units, modifiers, and relations. Our approach incorporates a two-stage reasoning-aware fine-tuning strategy using augmented scientific data and process-based supervision. Furthermore, we introduce a progressive reward curriculum designed to penalize specific hallucination types, significantly improving extraction faithfulness. Experimental results demonstrate that MeasHalu substantially reduces hallucination rates and improves overall accuracy on the MeasEval benchmark. This work provides a targeted solution to a key bottleneck in automated scientific knowledge extraction, facilitating more trustworthy and scalable machine-assisted scientific literature analysis. 

---
# Freshness-Aware Prioritized Experience Replay for LLM/VLM Reinforcement Learning 

**Authors**: Weiyu Ma, Yongcheng Zeng, Yan Song, Xinyu Cui, Jian Zhao, Xuhui Liu, Mohamed Elhoseiny  

**Link**: [PDF](https://arxiv.org/pdf/2604.16918)  

**Abstract**: Reinforcement Learning (RL) has achieved impressive success in post-training Large Language Models (LLMs) and Vision-Language Models (VLMs), with on-policy algorithms such as PPO, GRPO, and REINFORCE++ serving as the dominant paradigm. However, these methods discard all collected trajectories after a single gradient update, resulting in poor sample efficiency, particularly wasteful for agentic tasks where multi-turn environment interactions are expensive. While Experience Replay drives sample efficiency in classic RL by allowing agents to reuse past trajectories and prioritize informative ones, directly applying Prioritized Experience Replay (PER) to LLMs fails. The rapid policy evolution of billion-parameter models renders stored priorities stale, causing old high-priority trajectories to dominate sampling long after they have become uninformative. We propose Freshness-Aware PER, which addresses this priority staleness problem by augmenting any PER-based priority with a multiplicative exponential age decay grounded in effective sample size analysis. To the best of our knowledge, Freshness-Aware PER is the first work to successfully apply PER to LLM/VLM reinforcement learning. We evaluate on eight multi-step agentic, reasoning, and math competition tasks with 0.5B, 3B, and 7B models. Freshness-Aware PER significantly outperforms on-policy baselines, achieving +46% on NQ Search, +367% on Sokoban, and +133% on VLM FrozenLake, while standard PER without age decay consistently degrades performance. Our code is publicly available at this https URL. 

---
# x1: Learning to Think Adaptively Across Languages and Cultures 

**Authors**: Yangfan Ye, Xiaocheng Feng, Xiachong Feng, Yichong Huang, Zekun Yuan, Lei Huang, Weitao Ma, Qichen Hong, Yunfei Lu, Dandan Tu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16917)  

**Abstract**: Languages encode distinct abstractions and inductive priors, yet most large language models (LLMs) overlook this diversity by reasoning in a single dominant language. In this work, we introduce x1, a family of reasoning models that can adaptively reason in an advantageous language on a per-instance basis. To isolate the effect of reasoning-language choice, x1 is constructed without expanding the model's knowledge boundaries and is trained by contrasting linguistically distinct reasoning trajectories for the same input. Our extensive experiments demonstrate the benefits of adaptive multilingual reasoning across multilingual mathematical reasoning and culturally grounded tasks. Moreover, our results challenge a simplistic view of scaling laws: while scaling reduces cross-lingual disparities in procedural domains such as math reasoning, it does not eliminate the advantages of culture-associated languages in culturally grounded tasks, as we empirically show that such reasoning enables more efficient and accurate cultural knowledge recall. Overall, our findings establish language choice as a functional component of reasoning, with implications for building more generalist and globally competent reasoning models. 

---
# When Choices Become Risks: Safety Failures of Large Language Models under Multiple-Choice Constraints 

**Authors**: Yuheng Chen, Zhiyu Wu, Bowen Cheng, Tetsuro Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16916)  

**Abstract**: Safety alignment in large language models (LLMs) is primarily evaluated under open-ended generation, where models can mitigate risk by refusing to respond. In contrast, many real-world applications place LLMs in structured decision-making tasks, such as multiple-choice questions (MCQs), where abstention is discouraged or unavailable. We identify a systematic failure mode in this setting: reformulating harmful requests as forced-choice MCQs, where all options are unsafe, can systematically bypass refusal behavior, even in models that consistently reject equivalent open-ended prompts. Across 14 proprietary and open-source models, we show that forced-choice constraints sharply increase policy-violating responses. Notably, for human-authored MCQs, violation rates follow an inverted U-shaped trend with respect to structural constraint strength, peaking under intermediate task specifications, whereas MCQs generated by high-capability models yield near-saturation violation rates across constraints and exhibit strong cross-model transferability. Our findings reveal that current safety evaluations substantially underestimate risks in structured task settings and highlight constrained decision-making as a critical and underexplored surface for alignment failures. 

---
# PRISM: Probing Reasoning, Instruction, and Source Memory in LLM Hallucinations 

**Authors**: Yuhe Wu, Guangyu Wang, Yuran Chen, Jiatong Zhang, Yutong Zhang, Yujie Chen, Jiaming Shang, Guang Zhang, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16909)  

**Abstract**: As large language models (LLMs) evolve from conversational assistants into agents capable of handling complex tasks, they are increasingly deployed in high-risk domains. However, existing benchmarks largely rely on mixed queries and posterior evaluation, output-level scoring, which quantifies hallucination severity but offers limited insight into where and why hallucinations arise in the generation pipeline. We therefore reformulate hallucination evaluation as a diagnostic problem and propose PRISM, a controlled benchmark that disentangles hallucinations into four dimensions: knowledge missing, knowledge errors, reasoning errors, and instruction-following errors, grounded in three stages of generation (memory, instruction, and reasoning). PRISM contains 9,448 instances across 65 tasks and supports fine-grained, stage-aware diagnostic evaluation. Evaluating 24 mainstream open-source and proprietary LLMs, we uncover consistent trade-offs across instruction following, memory retrieval, and logical reasoning, showing that mitigation strategies often improve specific dimensions at the expense of others. We hope PRISM provides a framework for understanding the specific mechanisms behind LLMs hallucinations, ultimately accelerating the development of trustworthy large language models. 

---
# Prune, Interpret, Evaluate: A Cross-Layer Transcoder-Native Framework for Efficient Circuit Discovery via Feature Attribution 

**Authors**: Qinhao Chen, Linyang He, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2604.16889)  

**Abstract**: Existing feature-interpretation pipelines typically operate on uniformly sampled units, but only a small fraction of cross-layer transcoder (CLT) features matter for a target behavior, with the rest resulting in expensive feature explaining and evaluating costs. We introduce the first CLT-native end-to-end framework, PIE, connecting Pruning, automatic Interpretation, and interpretation Evaluation, enabling systematic measurement of behavioral fidelity and downstream interpretability under pruning. To achieve this, we propose Feature Attribution Patching (FAP), a patch-grounded attribution method that scores CLT features by aggregating gradient-weighted write contributions, and FAP-Synergy, a synergy-aware reranking procedure. We evaluate pruning using KL-divergence behavior retention and assess interpretation quality with FADE-style metrics. Across IOI and Doc-String, across budgets $K \in \{50, 100, 200, 400, 800\}$, and across FAP, FAP-Synergy, Activation-Magnitude, and ACDC-style pruning, the FAP family consistently achieves the best or near-best fidelity, with FAP-Synergy providing its clearest gains in strict-budget regimes. On IOI with CLTs for Llama-3.2-1B and Gemma-2-2B, pruning to $K=100$ features matches the KL fidelity that random selection from the active feature set requires $\approx 4$k features to achieve ($\approx 40\times$ compression), enabling $\approx 40\times$ fewer interpretation/evaluation calls while substantially reducing low-quality features. 

---
# Incentivizing Parametric Knowledge via Reinforcement Learning with Verifiable Rewards for Cross-Cultural Entity Translation 

**Authors**: Jiang Zhou, Xiaohu Zhao, Xinwei Wu, Tianyu Dong, Hao Wang, Yangyang Liu, Heng Liu, Linlong Xu, Longyue Wang, Weihua Luo, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.16881)  

**Abstract**: Cross-cultural entity translation remains challenging for large language models (LLMs) as literal or phonetic renderings are usually yielded instead of culturally appropriate translations in context. However, relevant knowledge may already be encoded in model parameters during large-scale pre-training. To incentivize the effective use of parametric knowledge, we propose EA-RLVR (Entity-Anchored Reinforcement Learning with Verifiable Rewards), a training framework that optimizes cross-cultural entity translation without relying on external knowledge bases. EA-RLVR anchors supervision on a verifiable, entity-level reward signal and incorporates lightweight structural gates to stabilize optimization. This design steers the model toward learning a robust reasoning process rather than merely imitating reference translations. We evaluate EA-RLVR on XC-Translate and observe consistent improvements in both entity translation accuracy and out-of-domain generalization. Specifically, training on merely 7k samples boosts Qwen3-14B's entity translation accuracy from 23.66\% to 31.87\% on a 50k test set comprising entirely unseen entities. The learned entity translation ability also transfers to general translation, yielding +1.35 XCOMET on WMT24++, which scales to +1.59 with extended optimization. Extensive analyses of $pass@k$ dynamics and reward formulations attribute these gains to superior sampling efficiency and a stable optimization landscape. 

---
# A Community-Based Approach for Stance Distribution and Argument Organization 

**Authors**: Rudra Ranajee Saha, Laks V. S. Lakshmanan, Raymond T. Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.16852)  

**Abstract**: The proliferation of online debate platforms and social media has led to an unprecedented volume of argumentative content on controversial topics from multiple perspectives. While this wealth of perspectives offers opportunities for developing critical thinking and breaking filter bubbles (Pariser 2011), the sheer volume and complexity of arguments make it challenging for readers to synthesize and comprehend diverse viewpoints effectively. We present an unsupervised graph-based approach for community-based argument organization that helps users navigate and understand complex argumentative landscapes. Our system analyzes collections of topic-focused articles and constructs a rich interaction graph by capturing multiple relationship types between arguments: topic similarity, semantic coherence, shared keywords, and common entities. We then employ community detection to identify argument communities that reveal homogeneous and heterogeneous viewpoint distributions. The detected communities are simplified through strategic graph operations to present users with digestible, yet comprehensive summaries of key argumentative patterns. Our approach requires no training data and can effectively process hundreds of articles while preserving nuanced relationships between arguments. Experimental results demonstrate our system's ability to identify meaningful argument communities and present them in an interpretable manner, facilitating users' understanding of complex socio-political debates. 

---
# DART: Mitigating Harm Drift in Difference-Aware LLMs via Distill-Audit-Repair Training 

**Authors**: Ziwen Pan, Zihan Liang, Jad Kabbara, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2604.16845)  

**Abstract**: Large language models (LLMs) tuned for safety often avoid acknowledging demographic differences, even when such acknowledgment is factually correct (e.g., ancestry-based disease incidence) or contextually justified (e.g., religious hiring preferences). This identity-blindness yields incorrect responses, unnecessary refusals, or generic "equal-treatment" defaults. We study this via difference-awareness classification: given a question involving demographic groups, the task is not to answer directly, but to classify whether a correct answer requires recognizing group differences (yes) or whether groups should be treated identically (no). Crucially, fine-tuning for accuracy triggers harm drift: model-generated explanations become increasingly harmful as decision accuracy improves, whether by elaborating harmful content, introducing problematic assumptions, or failing to flag harms the baseline identified. To mitigate this, we introduce DART (Distill--Audit--Repair Training), which distills label-conditioned reasoning from a teacher, audits outputs for harm drift cases relative to baseline, and repairs problematic cases via severity-weighted fine-tuning. On eight benchmarks, DART improves Llama-3-8B-Instruct accuracy from 39.0% to 68.8%, with largest gains on equal-treatment prompts (11.3% -> 72.6%), while reducing harm drift cases by 72.6%. It also transfers to 280 open-ended real-world queries across medical, legal, policy, and educational domains, improving difference-appropriate responses from 39.8% to 77.5% while reducing refusals from 34.3% to 3.0%. Our results demonstrate that accuracy and safety need not conflict when explicit detection and repair mechanisms are in place. 

---
# HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents 

**Authors**: Jinchang Zhu, Jindong Li, Cheng Zhang, Jiahong Liu, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16839)  

**Abstract**: Long-term memory is a critical challenge for Large Language Model agents, as fixed context windows cannot preserve coherence across extended interactions. Existing memory systems represent conversation history as unstructured embedding vectors, retrieving information through semantic similarity. This paradigm fails to capture the associative structure of human memory, wherein related experiences progressively strengthen interconnections through repeated co-activation. Inspired by cognitive neuroscience, we identify three mechanisms central to biological memory: association, consolidation, and spreading activation, which remain largely absent in current research.
To bridge this gap, we propose HeLa-Mem, a bio-inspired memory architecture that models memory as a dynamic graph with Hebbian learning dynamics. HeLa-Mem employs a dual-level organization: (1) an episodic memory graph that evolves through co-activation patterns, and (2) a semantic memory store populated via Hebbian Distillation, wherein a Reflective Agent identifies densely connected memory hubs and distills them into structured, reusable semantic knowledge. This dual-path design leverages both semantic similarity and learned associations, mirroring the episodic-semantic distinction in human cognition. Experiments on LoCoMo demonstrate superior performance across four question categories while using significantly fewer context tokens. Code is available on GitHub: this https URL 

---
# Crowded in B-Space: Calibrating Shared Directions for LoRA Merging 

**Authors**: Yixuan Tang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16826)  

**Abstract**: Merging separately trained LoRA adapters is a practical alternative to joint multi-task training, but it often hurts performance. Existing methods usually treat the LoRA update $\Delta W = BA$ as a single object and do not distinguish the two LoRA matrices. We show that the main source of LoRA merge interference comes from the output-side matrix $B$. Across tasks, $B$ repeatedly uses a small set of shared directions, while $A$ remains much more task-specific. As a result, the merged adapter overemphasizes these shared directions, and task-specific information is lost. We propose Pico (Pre-merge interference calibration in output-space), a data-free method that calibrates $B$ before merge by downscaling over-shared directions and then rescaling the merged update. Pico plugs directly into existing merging methods such as Task Arithmetic, TIES, and TSV-M. Across eight different benchmarks from math, coding, finance, and medical domains, Pico improves average accuracy by 3.4-8.3 points over the corresponding base method and achieves the best overall average performance. Pico also enables merged adapters to outperform the LoRA trained with all task data. These results show that LoRA merging works better when the two LoRA matrices are treated separately. 

---
# When Informal Text Breaks NLI: Tokenization Failure, Distribution Shift, and Targeted Mitigations 

**Authors**: Avinash Goutham Aluguvelly  

**Link**: [PDF](https://arxiv.org/pdf/2604.16787)  

**Abstract**: We study how informal surface forms degrade NLI accuracy in ELECTRA-small (14M) and RoBERTa-large (355M) across four transforms applied to SNLI and MultiNLI: slang substitution, emoji replacement, Gen-Z filler tokens, and their combination. Slang substitution (replacing formal words with informal equivalents, e.g., "going to" -> "gonna", "friend" -> "homie") causes minimal degradation (at most 1.1pp): slang vocabulary falls largely within WordPiece coverage, so the tokenizer handles it without signal loss. Emoji replaces content words with Unicode characters that ELECTRA's WordPiece tokenizer maps to [UNK], destroying the input signal before any learned parameters see it (93.6% of emoji examples contain at least one [UNK], mean 2.91 per example). Noise tokens (no cap, deadass, tbh) are fully in-vocabulary but absent from NLI training data, consistent with the model assigning them inferential weight they do not carry. The two failure modes respond to different interventions: preprocessing recovers emoji accuracy by normalizing text before tokenization; augmentation handles noise by exposing the model to noise-bearing examples during training. A hybrid of both achieves 88.93% on the combined variant for ELECTRA on SNLI (up from 75.88%), with no statistically significant drop on clean text. Against GPT-4o-mini zero-shot, unmitigated ELECTRA is significantly worse on transformed variants (p < 0.0001); hybrid ELECTRA surpasses it across all SNLI variants and reaches statistical parity on MultiNLI. 

---
# StageMem: Lifecycle-Managed Memory for Language Models 

**Authors**: Jiarui Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.16774)  

**Abstract**: Long-horizon language model systems increasingly rely on persistent memory, yet many current designs still treat memory primarily as a static store: write an item, place it into memory, and retrieve it later if needed. We argue that this framing does not adequately capture the practical memory-control problem in deployed LLM systems. In realistic settings, the difficulty is often not merely forgetting useful information, but retaining too many uncertain items, forgetting important content in the wrong order, and giving users little trust in what will persist over time. We propose StageMem, a lifecycle-managed memory framework that treats memory as a stateful process rather than a passive repository. StageMem organizes memory into three stages -- transient, working, and durable memory -- and models each item with explicit confidence and strength. This separates shallow admission from long-term commitment: information may first be written at low cost and only later be promoted, retained, updated, or evicted as evidence and pressure evolve. Under controlled pressure regimes, this decomposition helps preserve late-important content while keeping memory burden and deeper-tier pollution more controlled. Adapted external tasks provide boundary evidence that the same schema remains compatible with stronger retrieval structure outside pure synthetic control. We present StageMem as a principled decomposition of the memory-control problem for language model systems. 

---
# When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms 

**Authors**: Chaewan Chun, Delvin Ce Zhang, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.16767)  

**Abstract**: Audio platforms have evolved beyond entertainment. They have become central to public discourse, from podcasts and radio to WhatsApp voice notes and live streams. With millions of shows and hundreds of millions of listeners, audio platforms are now a major channel for misinformation. Yet existing fact-checking pipelines are mostly designed for written claims, overlooking the unique properties of spoken media. We argue that audio misinformation is not merely textual content with transcripts: it is structurally different because it is both spoken - carrying persuasive force through prosody, pacing, and emotion - and conversational - unfolding across turns, speakers, and episodes. These dual properties introduce verification difficulties that traditional methods rarely face. This position paper synthesizes evidence across modalities and platforms, examines datasets and methods, and highlights why existing pipelines fail on audio. We argue that advancing fact-checking requires rethinking verification pipelines around the spoken and conversational realities of audio. 

---
# Expressing Social Emotions: Misalignment Between LLMs and Human Cultural Emotion Norms 

**Authors**: Sree Bhattacharyya, Manas Mehta, Leona Chen, Cristina Salvador, Agata Lapedriza, Shiran Dudy, James Z. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16757)  

**Abstract**: The expression of emotions that serve social purposes, such as asserting independence or fostering interdependence, is central to human interactions and varies systematically across cultures. As LLMs are increasingly used to simulate human behavior in culturally nuanced interactions, it is important to understand whether they faithfully capture human patterns of social emotion expression. When LLM responses are not culturally aligned, their utility is compromised -- particularly when users assume they are interacting with a culturally attuned interlocutor, and may act on advice that proves inappropriate in their cultural context. We present a psychologically informed evaluation framework of cross-cultural social emotion expression in LLMs. Using a human study comparing European American and Latin American participants' expression of engaging and disengaging emotions, we evaluate six frontier LLMs on their ability to reflect culturally differentiated patterns for expressing social emotions. We find systematic misalignment between model and human behavior: all models express engaging emotions more than disengaging ones, with particularly stark differences observed for the generally well-represented European American persona. We further highlight that LLM responses are highly concentrated and deterministic, failing to capture the diversity of human responses in expressing social emotions. Our ablation analyses reveal that these patterns are robust to sampling temperatures, partially sensitive to prompt language, and dependent on the response elicitation format. Together, our findings highlight limitations in how current LLMs represent the interaction of cultural and emotional axes, particularly when expressing social emotions, with direct implications for their deployment in cross-cultural affective contexts. 

---
# Evaluating Adaptive Personalization of Educational Readings with Simulated Learners 

**Authors**: Ryan T. Woo, Anmol Rao, Aryan Keluskar, Yinong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16744)  

**Abstract**: We present a framework for evaluating adaptive personalization of educational reading materials with theory-grounded simulated learners. The system builds a learning-objective and knowledge-component ontology from open textbooks, curates it in a browser-based Ontology Atlas, labels textbook chunks with ontology entities, and generates aligned reading-assessment pairs. Simulated readers learn from passages through a Construction-Integration-inspired memory model with DIME-style reader factors, KREC-style misconception revision, and an open New Dale-Chall readability signal. Answers are produced by score-based option selection over the learner's explicit memory state, while BKT drives adaptation. Across three sampled subject ontologies and matched cohorts of 50 simulated learners per condition, adaptive reading significantly improved outcomes in computer science, yielded smaller positive but inconclusive gains in inorganic chemistry, and was neutral to slightly negative in general biology. 

---
# Detecting Alarming Student Verbal Responses using Text and Audio Classifier 

**Authors**: Christopher Ormerod, Gitit Kehat  

**Link**: [PDF](https://arxiv.org/pdf/2604.16717)  

**Abstract**: This paper addresses a critical safety gap in the use Automated Verbal Response Scoring (AVRS). We present a novel hybrid framework for troubled student detection that combines a text classifier, trained to detect responses based on their content, and an audio classifier, trained to detect responses using prosodic markers. This approach overcomes key limitations of traditional AVRS systems by considering both content and prosody of responses, achieving enhanced performance in identifying potentially concerning responses. This system can expedite the review process by humans, which can be life-saving particularly when timely intervention may be crucial. 

---
# The impact of postediting on AI generative translation in Yemeni context: Translating literary prose by ChatGPT 

**Authors**: Nasim Al-wagieh, Mohammed Q. Shormani  

**Link**: [PDF](https://arxiv.org/pdf/2604.16704)  

**Abstract**: This study examines the role of artificial intelligence in translation, focusing on ChatGPT, specifically ChatGPT-4, and the extent to which human postediting is required in literary translation. A mixed-method approach was adopted, involving 30 professional translators who evaluated and postedited AI-generated translations of selected Arabic and English literary texts. The results show that although AI improves translation speed and accessibility, it remains limited in handling cultural, stylistic, and figurative aspects of language. Participants generally confirmed the necessity of human postediting, particularly in novels and drama. The findings indicate that emerging human-machine collaboration model rather than replacement of human translators. The study concludes that AI should be used as a supportive tool, while human expertise remains essential for ensuring translation quality and cultural appropriateness. 

---
# No-Worse Context-Aware Decoding: Preventing Neutral Regression in Context-Conditioned Generation 

**Authors**: Yufei Tao, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2604.16686)  

**Abstract**: Large language models (LLMs) can answer questions and summarize documents when conditioned on external contexts (e.g., retrieved evidence), yet context use remains unreliable: models may overwrite an already-correct output (neutral regression) even when the context is non-informative. We formalize neutral regression as a do-no-harm requirement and quantify it by measuring accuracy drops on baseline-correct items under answer-consistent contexts. We propose No-Worse Context-Aware Decoding (NWCAD), a decode-time adapter built on a two-stream setup with a two-stage gate: it backs off to no-context decoding when the context is non-informative, and otherwise uses context-conditioned decoding with a CAD-style fallback under uncertainty. We evaluate NWCAD on benchmarks that separate do-no-harm reliability from context utilization (accuracy gains on genuinely helpful contexts). NWCAD prevents neutral regression on baseline-correct items while preserving strong context-driven accuracy on helpful contexts. 

---
# CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering for Multi-Platform Social Streams 

**Authors**: Anik Saha, Mst. Fahmida Sultana Naznin, Zia Ul Hassan Abdullah, Anisa Binte Asad, K. G. Subarno Bithi, A. B. M. Alim Al Islam  

**Link**: [PDF](https://arxiv.org/pdf/2604.16665)  

**Abstract**: Urgent blood donation seeking posts and messages on social media often go unnoticed due to the overwhelming volume of daily communications. Traditional app-based systems, reliant on manual input, struggle to reach users in low-resource settings, delaying critical responses. To address this, we introduce the Cognitive Blood Request System (CBRS), a multi-platform framework that efficiently filters and parses blood donation requests from social media streams using a cost-efficient dual-layered architecture. To do so, we curate a novel dataset of 11K parsed blood donation request messages in Bengali, English, and transliterated Bengali, capturing the linguistic diversity of real social media communications. The inclusion of adversarial negatives further enhances the robustness of our model. CBRS achieves an impressive 99% accuracy and precision in filtering, surpassing benchmark methods. In the parsing task, our LoRA finetuned Llama-3.2-3B model achieves 92% zero-shot accuracy, surpassing the base model by 41.54% and exceeding the few-shot performance of GPT-4o-mini, Gemini-2.0-Flash, and other LLMs, while resulting in a 35X reduction in input token usage. This work lays a robust foundation for scalable, inclusive information extraction in time-sensitive, object-focused tasks. Our code, dataset, and trained models are publicly available at [this https URL](this https URL). 

---
# Defragmenting Language Models: An Interpretability-based Approach for Vocabulary Expansion 

**Authors**: Maitrey Mehta, Nishant Subramani, Zhichao Xu, Ashim Gupta, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.16656)  

**Abstract**: All languages are equal; when it comes to tokenization, some are more equal than others. Tokens are the hidden currency that dictate the cost and latency of access to contemporary LLMs. However, many languages written in non-Latin scripts observe a poor exchange rate: LLMs take several multiples of tokens to encode the same information in many languages as they do for English. Our analysis reveals that this issue, known as 'token over-fragmentation', persists in modern open-weight LLMs. The standard remedy is vocabulary expansion that adds target language items missing from the model's vocabulary. In this work, we comprehensively study and advance interpretability-based vocabulary expansion, a new research direction. We focus on two core decisions in the vocabulary expansion process: What items should we add? and How should we initialize their corresponding input and output embeddings? First, we question the conventional use of frequency-based methods to choose candidate vocabulary items to add (a decision long treated as settled), and show that interpretability-based methods offer a superior performance-token efficiency trade-off. Next, we strengthen the case for interpretability-based embedding initialization by showing large gains (~20 pts) over baseline initialization methods for several languages written in non-Latin scripts. We identify the phenomenon of "subword detokenization" where models progressively merge fragmented subword tokens into larger subwords across layers. Grounded in our analysis of this phenomenon, we propose FragMend to further push the efficiency ceiling of interpretability-based expansion. We validate the effectiveness of FragMend through comparison against strong baselines and we present extensive analysis of its design choices. 

---
# IYKYK (But AI Doesn't): Automated Content Moderation Does Not Capture Communities' Heterogeneous Attitudes Towards Reclaimed Language 

**Authors**: Christina Chance, Rebecca Pattichis, Arjun Subramonian, James He, Shruti Narayanan, Saadia Gabriel, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16654)  

**Abstract**: Reclaimed slur usage is a common and meaningful practice online for many marginalized communities. It serves as a source of solidarity, identity, and shared experience. However, contemporary automated and AI-based moderation tools for online content largely fail to distinguish between reclaimed and hateful uses of slurs, resulting in the suppression of marginalized voices. In this work, we use quantitative and qualitative methods to examine the attitudes of social media users in LGBTQIA+, Black, and women communities around reclaimed slurs targeting our focus groups including the f-word, n-word, and b-word. With social media users from these communities, we collect and analyze an annotated online slur usage corpus. The corpus includes annotators' perceptions of whether an online text containing a slur should be flagged as hate speech, as well as contextual features of the slur usage. Across all communities and annotation questions, we observe low inter-annotator agreement, indicating substantial disagreement among in-group annotators. This is compounded by the fact that, absent clear contextual signals of identity and intent, even in-group members may disagree on how to interpret reclaimed slur usage online. Semi-structured interviews with annotators suggest that differences in lived experience and personal history contribute to this variation as well. We find poor alignment between annotator judgments and automated hate speech assessments produced by Perspective API. We further observe that certain features of a text such as whether the slur usage was derogatory and if the slur was targeted at oneself are more associated with whether annotators report the text as hate speech. Together, these findings highlight the inherent subjectivity and contextual nature of how marginalized communities interpret slurs online. 

---
# Migrant Voices, Local News: Insights on Bridging Community Needs with Media Content 

**Authors**: David Alonso del Barrio, Paula Dolores Rescala, Victor Bros, Daniel Gatica-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2604.16651)  

**Abstract**: Research shows news consumption differs across demographics, yet little is known about non-mainstream audiences, especially in relation to local media. Our study addresses this gap by examining how French-speaking migrants in a mid-size European city engage with local news, and whether their needs are reflected in coverage. Eight community members participated in focus groups, whose insights guided the selection of natural language processing methods (topic modeling, information retrieval, sentiment analysis, and readability) applied to over 2000 hyper-local news articles. Results showed that while articles frequently covered local events, gaps remained in topics important to participants. Sentiment analysis revealed a generally positive tone, and readability measures indicated an intermediate-advanced French level, raising questions about accessibility for integration. Our work contributes to bridging the gap between local news platforms' content and diverse readers' needs, and could inform local media organizations about opportunities to expand their current news story coverage to appeal to more diverse audiences. 

---
# AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation 

**Authors**: Weihua Du, Jingming Zhuo, Yixin Dong, Andre Wang He, Weiwei Sun, Zeyu Zheng, Manupa Karunaratne, Ivan Fox, Tim Dettmers, Tianqi Chen, Yiming Yang, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2604.16625)  

**Abstract**: Recent large language model (LLM) agents have shown promise in using execution feedback for test-time adaptation. However, robust self-improvement remains far from solved: most approaches still treat each problem instance independently, without accumulating reusable knowledge. This limitation is particularly pronounced in domain-specific languages such as Triton, which are underrepresented in LLM pretraining data. Their strict constraints and non-linear optimization landscape further make naive generation and local refinement unreliable. We propose AdaExplore, an agent framework that enables self-improvement via accumulated execution feedback for performance-critical kernel code generation through two complementary stages: failure-driven adaptation and diversity-preserving search, jointly improving correctness and optimization performance without additional fine-tuning or external knowledge. In the adaptation stage, the agent synthesizes tasks and converts recurring failures into a reusable memory of validity rules, helping subsequent generations remain within the feasible set. In the search stage, the agent organizes candidate kernels as a tree and alternates between small local refinements and larger structural regeneration, allowing it to explore the optimization landscape beyond local optima. Experiments on kernel runtime optimization benchmarks validate these gains: AdaExplore achieves 3.12x and 1.72x speedups on KernelBench Level-2 and Level-3, respectively, within 100 steps, and continues to improve with additional computation. 

---
# Aligning Backchannel and Dialogue Context Representations via Contrastive LLM Fine-Tuning 

**Authors**: Livia Qian, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2604.16622)  

**Abstract**: Backchannels (e.g., `yeah', `mhm', and `right') are short, non-interruptive feedback signals whose lexical form and prosody jointly convey pragmatic meaning. While prior computational research has largely focused on predicting backchannel timing, the relationship between lexico-prosodic form and meaning remains underexplored. We propose a two-stage framework: first, fine-tuning large language models on dialogue transcripts to derive rich contextual representations; and second, learning a joint embedding space for dialogue contexts and backchannel realizations. We evaluate alignment with human perception via triadic similarity judgments (prosodic and cross-lexical) and a context-backchannel suitability task. Our results demonstrate that the learned projections substantially improve context-backchannel retrieval compared to previous methods. In addition, they reveal that backchannel form is highly sensitive to extended conversational context and that the learned embeddings align more closely with human judgments than raw WavLM features. 

---
# Spotlights and Blindspots: Evaluation Machine-Generated Text Detection 

**Authors**: Kevin Stowe, Kailash Patil  

**Link**: [PDF](https://arxiv.org/pdf/2604.16607)  

**Abstract**: With the rise of generative language models, machine-generated text detection has become a critical challenge. A wide variety of models is available, but inconsistent datasets, evaluation metrics, and assessment strategies obscure comparisons of model effectiveness. To address this, we evaluate 15 different detection models from six distinct systems, as well as seven trained models, across seven English-language textual test sets and three creative human-written datasets. We provide an empirical analysis of model performance, the influence of training and evaluation data, and the impact of key metrics. We find that no single system excels in all areas and nearly all are effective for certain tasks, and the representation of model performance is critically linked to dataset and metric choices. We find high variance in model ranks based on datasets and metrics, and overall poor performance on novel human-written texts in high-risk domains. Across datasets and metrics, we find that methodological choices that are often assumed or overlooked are essential for clearly and accurately reflecting model performance. 

---
# Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models 

**Authors**: Yang Liu, Hongming Li, Melissa Xiaohui Qin, Qiankun Liu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16593)  

**Abstract**: We present SemanticQA, an evaluation suite designed to assess language models (LMs) in semantic phrase processing tasks. The benchmark consolidates existing multiword expression (MwE) resources and reorganizes them into a unified testbed. It covers both general lexical phenomena, such as lexical collocations, and three fine-grained categories: idiomatic expressions, noun compounds, and verbal constructions. Through SemanticQA, we assess LMs of diverse architectures and scales in extraction, classification, and interpretation tasks, as well as sequential task compositions. We reveal substantial performance variation, particularly on tasks requiring semantic reasoning, highlighting differences in reasoning efficacy and semantic understanding of LMs, providing insights for pushing LMs with stronger comprehension on non-trivial semantic phrases. The evaluation harness and data of SemanticQA are available at this https URL. 

---
# EchoChain: A Full-Duplex Benchmark for State-Update Reasoning Under Interruptions 

**Authors**: Smit Nautambhai Modi, Gandharv Mahajan, Marc Wetter, Randall Welles  

**Link**: [PDF](https://arxiv.org/pdf/2604.16456)  

**Abstract**: Real-time voice assistants must revise task state when users interrupt mid-response, but existing spoken-dialog benchmarks largely evaluate turn-based interaction and miss this failure mode. We introduce EchoChain, a controlled benchmark for evaluating full-duplex state-update reasoning under mid-speech interruptions. EchoChain identifies three recurring failure patterns in post-interruption continuations: contextual inertia, interruption amnesia, and objective displacement. The benchmark generates scenario-driven conversations and injects interruptions at a standardized point relative to assistant speech onset, enabling controlled cross-model comparison. In a paired half-duplex control, total failures drop by 40.2% relative to interrupted runs, indicating that many errors are driven by state-update reasoning under interruption rather than task difficulty alone. Across evaluated real-time voice models, no system exceeds a 50% pass rate, showing substantial room for improvement in mid-generation state revision. EchoChain provides a reproducible benchmark for diagnosing state-update reasoning failures in full-duplex voice interaction. 

---
# SynopticBench: Evaluating Vision-Language Models on Generating Weather Forecast Discussions of the Future 

**Authors**: Timothy B. Higgins, Antonios Mamalakis, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2604.16451)  

**Abstract**: Recent advances in visual-language models (VLMs) have led to significant improvements in a plethora of complex multimodal tasks like image captioning, report generation, and visual perception. However, generating text from meteorological data is highly challenging because the atmosphere is a chaotic system that is rapidly changing at various spatial and temporal scales. Given the complexity of atmospheric phenomena, it is critical to verifiably quantify the effectiveness of existing VLMs on weather forecasting data. In this work, we present SynopticBench, a high-quality dataset consisting of 1,367,041 text samples of Area Forecast Discussions created by the National Weather Service over the continental United States paired to images of 500mb geopotential height, 2 meter temperature, and 850mb wind velocity in weather forecasts. We also present Synoptic Phenomena Alignment and Coverage Evaluation (SPACE), a novel evaluation framework that can be used to effectively estimate the quality of text descriptions of synoptic weather phenomena. Extensive experiments on generating forecast discussions using state-of-the-art VLMs show the sensitivity of existing evaluation metrics in this domain and enable further exploration into synoptic weather and climate text generation. 

---
# HalluSAE: Detecting Hallucinations in Large Language Models via Sparse Auto-Encoders 

**Authors**: Boshui Chen, Zhaoxin Fan, Ke Wang, Zhiying Leng, Faguo Wu, Hongwei Zheng, Yifan Sun, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16430)  

**Abstract**: Large Language Models (LLMs) are powerful and widely adopted, but their practical impact is limited by the well-known hallucination phenomenon. While recent hallucination detection methods have made notable progress, we find most of them overlook the dynamic nature and underlying mechanisms of it. To address this gap, we propose HalluSAE, a phase transition-inspired framework that models hallucination as a critical shift in the model's latent dynamics. By modeling the generation process as a trajectory through a potential energy landscape, HalluSAE identifies critical transition zones and attributes factual errors to specific high-energy sparse features. Our approach consists of three stages: (1) Potential Energy Empowered Phase Zone Localization via sparse autoencoders and a geometric potential energy metric; (2) Hallucination-related Sparse Feature Attribution using contrastive logit attribution; and (3) Probing-based Causal Hallucination Detection through linear probes on disentangled features. Extensive experiments on Gemma-2-9B demonstrate that HalluSAE achieves state-of-the-art hallucination detection performance. 

---
# Injecting Structured Biomedical Knowledge into Language Models: Continual Pretraining vs. GraphRAG 

**Authors**: Jaafer Klila, Sondes Bannour Souihi, Rahma Boujelben, Nasredine Semmar, Lamia Hadrich Belguith  

**Link**: [PDF](https://arxiv.org/pdf/2604.16422)  

**Abstract**: The injection of domain-specific knowledge is crucial for adapting language models (LMs) to specialized fields such as biomedicine. While most current approaches rely on unstructured text corpora, this study explores two complementary strategies for leveraging structured knowledge from the UMLS Metathesaurus: (i) Continual pretraining that embeds knowledge into model parameters, and (ii) Graph Retrieval-Augmented Generation (GraphRAG) that consults a knowledge graph at inference time. We first construct a large-scale biomedical knowledge graph from UMLS (3.4 million concepts and 34.2 million relations), stored in Neo4j for efficient querying. We then derive a ~100-million-token textual corpus from this graph to continually pretrain two models: BERTUMLS (from BERT) and BioBERTUMLS (from BioBERT). We evaluate these models on six BLURB (Biomedical Language Understanding and Reasoning Benchmark) datasets spanning five task types and evaluate GraphRAG on the two QA (Question Answering) datasets (PubMedQA, BioASQ). On BLURB tasks, BERTUMLS improves over BERT, with the largest gains on knowledge-intensive QA. Effects on BioBERT are more nuanced, suggesting diminishing returns when the base model already encodes substantial biomedical text knowledge. Finally, augmenting LLaMA 3-8B with our GraphRAG pipeline yields over than 3 points accuracy on PubMedQA and 5 points on BioASQ without any retraining, delivering transparent, multi-hop, and easily updated knowledge access. We release the processed UMLS Neo4j graph to support reproducibility. 

---
# Measuring Representation Robustness in Large Language Models for Geometry 

**Authors**: Vedant Jawandhia, Yash Sinha, Murari Mandal, Ankan Pal, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.16421)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on mathematical reasoning, yet their robustness to equivalent problem representations remains poorly understood. In geometry, identical problems can be expressed in Euclidean, coordinate, or vector forms, but existing benchmarks report accuracy on fixed formats, implicitly assuming representation invariance and masking failures caused by representational changes alone. We propose GeoRepEval, a representation-aware evaluation framework that measures correctness, invariance, and consistency at the problem level across parallel formulations, combining strict answer matching, bootstrap confidence intervals, paired McNemar tests, representation-flip analyses, and regression controls for surface complexity. We prove that our Invariance@3 metric decomposes accuracy into robust and fragile components and is bounded by the weakest representation. Evaluating eleven LLMs on 158 curated high-school geometry problems (474 instances), we find accuracy gaps of up to 14 percentage points induced solely by representation choice. Vector formulations emerge as a consistent failure point, with Invariance@3 as low as 0.044 even after controlling for length and symbolic complexity. A convert-then-solve prompting intervention improves vector accuracy by up to 52 percentage points for high-capacity models, suggesting that failures reflect representation sensitivity rather than inability; however, low-capacity models show no gains, indicating deeper limitations. These results suggest that current models rely on representation-specific heuristics rather than abstract geometric reasoning. All datasets, prompts, and scripts are released at this https URL. 

---
# QU-NLP at QIAS 2026: Multi-Stage QLoRA Fine-Tuning for Arabic Islamic Inheritance Reasoning 

**Authors**: Mohammad AL-Smadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16396)  

**Abstract**: Islamic inheritance law (ilm al-mawarıth) presents a challenging domain for evaluating large language models' structured reasoning capabilities, requiring multi-step legal analysis, rule-based blocking decisions, and precise fractional calculations. We present QU-NLP's submission to the QIAS 2026 shared task on Arabic Islamic inheritance reasoning. Our approach employs a multi-stage Quantized Low-Rank Adaptation (QLoRA) fine-tuning strategy on Qwen3-4B: (1) domain adaptation on 3,166 Islamic fatwa records to acquire inheritance terminology and jurisprudential reasoning patterns, followed by (2) task-specific training on 12,000 structured inheritance cases to optimize JSON-formatted output generation. Using 4-bit NF4 quantization with rank-128 LoRA adapters, our model achieves 90% MIR-E (Mawarith Inheritance Reasoning Evaluation) score on the test set, demonstrating competitive performance while requiring minimal computational resources. Our results show that domain-specific pre-adaptation combined with structured output training enables small language models to perform complex legal reasoning tasks effectively comparing to commercial systems such as Gemini-2.5-flash. 

---
# LiFT: Does Instruction Fine-Tuning Improve In-Context Learning for Longitudinal Modelling by Large Language Models? 

**Authors**: Iqra Ali, Talia Tseriotou, Mahmud Elahi Akhter, Yuxiang Zhou, Maria Liakata  

**Link**: [PDF](https://arxiv.org/pdf/2604.16382)  

**Abstract**: Longitudinal NLP tasks require reasoning over temporally ordered text to detect persistence and change in human behavior and opinions. However, in-context learning with large language models struggles on tasks where models must integrate historical context, track evolving interactions, and handle rare change events. We introduce LiFT, a longitudinal instruction fine-tuning framework that unifies diverse longitudinal modeling tasks under a shared instruction schema. LiFT uses a curriculum that progressively increases temporal difficulty while incorporating few-shot structure and temporal conditioning to encourage effective use of past context. We evaluate LiFT across five datasets. Models trained on longitudinal tasks with different levels of temporal granularity are tested for generalisability on two separate datasets. Across models with different parameter sizes (OLMo (1B/7B), LLaMA-8B, and Qwen-14B), LiFT consistently outperforms base-model ICL, with strong gains on out-of-distribution data and minority change events. 

---
# Data Mixing for Large Language Models Pretraining: A Survey and Outlook 

**Authors**: Zhuo Chen, Yuxuan Miao, Supryadi, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.16380)  

**Abstract**: Large language models (LLMs) rely on pretraining on massive and heterogeneous corpora, where training data composition has a decisive impact on training efficiency and downstream generalization under realistic compute and data budget constraints. Unlike sample-level data selection, data mixing optimizes domain-level sampling weights to allocate limited budgets more effectively. In recent years, a growing body of work has proposed principled data mixing methods for LLM pretraining; however, the literature remains fragmented and lacks a dedicated, systematic survey. This paper provides a comprehensive review of data mixing for LLM pretraining. We first formalize data mixture optimization as a bilevel problem on the probability simplex and clarify the role of data mixing in the pretraining pipeline, and briefly explain how existing methods make this formulation tractable in practice. We then introduce a fine-grained taxonomy that organizes existing methods along two dimensions: static versus dynamic mixing. Static mixing is further categorized into rule-based and learning-based methods, while dynamic mixing is grouped into adaptive and externally guided families. For each class, we summarize representative approaches and analyze their strengths and limitations from a performance-cost trade-off perspective. Building on this analysis, we highlight challenges that cut across methods, including limited transferability across data domains, optimization objectives, models, and validation sets, as well as unstandardized evaluation protocols and benchmarks, and the inherent tension between performance gains and cost control in learning-based methods. Finally, we outline several exploratory directions, including finer-grained domain partitioning and inverse data mixing, as well as pipeline-aware designs, aiming to provide conceptual and methodological insights for future research. 

---
# Reciprocal Co-Training (RCT): Coupling Gradient-Based and Non-Differentiable Models via Reinforcement Learning 

**Authors**: Yunshuo Tian, Akayou Kitessa, Tanuja Chitnis, Yijun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16378)  

**Abstract**: Large language models (LLMs) and classical machine learning methods offer complementary strengths for predictive modeling, yet their fundamentally different representations and training paradigms hinder effective integration: LLMs rely on gradient-based optimization over textual data, whereas models such as Random Forests (RF) employ non-differentiable feature partitioning. This work introduces a reciprocal co-training framework that couples an LLM with an RF classifier via reinforcement learning, creating an iterative feedback loop in which each model improves using signals from the other. Tabular data are reformulated into standardized textual representations for the LLM, whose embeddings augment the RF feature space, while calibrated RF probability estimates provide feedback signals that guide reinforcement learning updates of the LLM. Experiments across three medical datasets demonstrate consistent performance gains for both models, with particularly strong effects for the LLM. Ablation analyses show that iterative refinement, hybrid reward design, and dimensionality control jointly contribute to these gains. The proposed framework provides a general mechanism that allows incompatible model families to leverage each other's strengths through bidirectional adaptation. 

---
# GoCoMA: Hyperbolic Multimodal Representation Fusion for Large Language Model-Generated Code Attribution 

**Authors**: Nitin Choudhury, Bikrant Bikram Pratap Maurya, Bhavinkumar Vinodbhai Kuwar, Arun Balaji Buduru  

**Link**: [PDF](https://arxiv.org/pdf/2604.16377)  

**Abstract**: Large Language Models (LLMs) trained on massive code corpora are now increasingly capable of generating code that is hard to distinguish from human-written code. This raises practical concerns, including security vulnerabilities and licensing ambiguity, and also motivates a forensic question: 'Who (or which LLM) wrote this piece of code?' We present GoCoMA, a multimodal framework that models an extrinsic hierarchy between (i) code stylometry, capturing higher-level structural and stylistic signatures, and (ii) image representations of binary pre-executable artifacts (BPEA), capturing lower-level, execution-oriented byte semantics shaped by compilation and toolchains. GoCoMA projects modality embeddings into a hyperbolic Poincaré ball, fuses them via a geodesic-cosine similarity-based cross-modal attention (GCSA) fusion mechanism, and back-projects the fused representation to Euclidean space for final LLM-source attribution. Experiments on two open-source benchmarks (CoDET-M4 and LLMAuthorBench) show that GoCoMA consistently outperforms unimodal and Euclidean multimodal baselines under identical evaluation protocols. 

---
# Foundational Study on Authorship Attribution of Japanese Web Reviews for Actor Analysis 

**Authors**: Hiroshi Matsubara, Shingo Matsugaya, Taichi Aoki, Masaki Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2604.16376)  

**Abstract**: This study investigates the applicability of authorship attribution based on stylistic features to support actor analysis in threat intelligence. As a foundational step toward future application to dark web forums, we conducted experiments using Japanese review data from clear web sources. We constructed datasets from Rakuten Ichiba reviews and compared four methods: TF-IDF with logistic regression (TF-IDF+LR), BERT embeddings with logistic regression (BERT-Emb+LR), BERT fine-tuning (BERT-FT), and metric learning with $k$-nearest neighbors (Metric+kNN). Results showed that BERT-FT achieved the best performance; however, training became unstable as the number of authors scaled to several hundred, where TF-IDF+LR proved superior in terms of accuracy, stability, and computational cost. Furthermore, Top-$k$ evaluation demonstrated the utility of candidate screening, and error analysis revealed that boilerplate text, topic dependency, and short text length were primary factors causing misclassification. 

---
# CFMS: Towards Explainable and Fine-Grained Chinese Multimodal Sarcasm Detection Benchmark 

**Authors**: Junzhao Zhang, Hsiu-Yuan Huang, Chenming Tang, Yutong Yang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16372)  

**Abstract**: Multimodal sarcasm detection has recently garnered significant attention. However, existing benchmarks suffer from coarse-grained annotations and limited cultural coverage, which hinder research into fine-grained semantic understanding. To address this, we construct CFMS, the first fine-grained multimodal sarcasm dataset tailored for Chinese social media. It comprises 2,796 high-quality image-text pairs and provides a triple-level annotation framework: sarcasm identification, target recognition, and explanation generation. We find that the fine-grained explanation annotations effectively guide AI in generating images with explicit sarcastic intent. Furthermore, we curate a high-consistency parallel Chinese-English metaphor subset (200 entries each), revealing significant limitations of current models in metaphoric reasoning. To overcome the constraints of traditional retrieval methods, we propose a Reinforcement Learning-augmented In-Context Learning strategy (PGDS) to dynamically optimize exemplar selection. Extensive experiments demonstrate that CFMS provides a solid foundation for building reliable multimodal sarcasm understanding systems, and the PGDS method significantly outperforms existing baselines on key tasks. Our data and code are available at this https URL. 

---
# Brain-CLIPLM: Decoding Compressed Semantic Representations in EEG for Language Reconstruction 

**Authors**: Xiaoli Yang, Huiyuan Tian, Yurui Li, Jianyu Zhang, Shijian Li, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16370)  

**Abstract**: Decoding natural language from non-invasive electroencephalography (EEG) remains fundamentally limited by low signal-to-noise ratio and restricted information bandwidth. This raises a fundamental question regarding whether sentence-level linguistic structure can be reliably recovered from such signals. In this work, we suggest that this assumption may not hold under realistic information constraints, and instead propose a semantic compression hypothesis in which EEG signals encode a compressed set of semantic anchors rather than full linguistic structure. Under our new perspective, direct sentence reconstruction becomes an overparameterized objective relative to the intrinsic information capacity of EEG. To address this mismatch, we introduce Brain-CLIPLM, a two-stage framework that decomposes EEG-to-text decoding into semantic anchor extraction via contrastive learning and sentence reconstruction using a retrieval-grounded large language model (LLM) with Chain-of-Thought (CoT) reasoning, following a granularity matching principle that aligns decoding complexity with neural information capacity. Evaluated on the Zurich Cognitive Language Processing Corpus, Brain-CLIPLM achieves 67.55\% top-5 and 85.00\% top-25 sentence retrieval accuracy, significantly outperforming direct decoding baseline, while cross-subject evaluation confirms robust generalization. Control analyses, including permutation testing, further demonstrate that EEG-derived representations carry sentence-specific information beyond language model priors. These results suggest that EEG-to-text decoding is better framed as recovering compressed semantic content rather than reconstructing full sentences, providing a biologically grounded and data-efficient pathway for non-invasive brain-computer interfaces. 

---
# Cross-Family Speculative Decoding for Polish Language Models on Apple~Silicon: An Empirical Evaluation of Bielik~11B with UAG-Extended MLX-LM 

**Authors**: Krzysztof Fonal  

**Link**: [PDF](https://arxiv.org/pdf/2604.16368)  

**Abstract**: Speculative decoding accelerates LLM inference by using a small draft model to propose k candidate tokens for a target model to verify. While effective for same-tokenizer pairs on high-bandwidth GPUs, its applicability to cross-family pairs with mismatched tokenizers and consumer-grade unified memory remains underexplored. We extend the MLX-LM framework with Universal Assisted Generation (UAG) to enable cross-tokenizer speculative decoding on Apple Silicon. We evaluate Bielik 11B-Instruct (Mistral-based) as the target model, paired with three draft models: Bielik 1.5B (Qwen-based with custom tokenizer), Qwen2.5-1.5B, and Llama 3.2-1B. Experiments on three Polish-language datasets (Wikipedia, pl_alpaca, synthetic) use draft lengths k in {2, 4, 6} to compare naive and context-aware token translation. Results show: (1) context-aware translation consistently improves acceptance rates across all configurations; (2) the Polish-specialized Bielik 1.5B achieves lower acceptance than general-purpose Qwen2.5 and Llama 3.2 drafters; (3) throughput on Apple Silicon is content-dependent, reaching 1.7x speedup for structured text but failing for varied instructions; and (4) verification cost on unified memory does not amortize as theory predicts because both models are memory-bandwidth bound, making sequential drafting expensive relative to batched verification. We propose a hardware-aware speedup formula and characterize conditions for cross-family speculative decoding on Apple Silicon. This is the first systematic evaluation of cross-family speculative decoding for Polish LLMs and the first empirical study of UAG-based decoding on unified memory architectures. 

---
# Multimodal Claim Extraction for Fact-Checking 

**Authors**: Joycelyn Teo, Rui Cao, Zhenyun Deng, Zifeng Ding, Michael Sejr Schlichtkrull, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2604.16311)  

**Abstract**: Automated Fact-Checking (AFC) relies on claim extraction as a first step, yet existing methods largely overlook the multimodal nature of today's misinformation. Social media posts often combine short, informal text with images such as memes, screenshots, and photos, creating challenges that differ from both text-only claim extraction and well-studied multimodal tasks like image captioning or visual question answering. In this work, we present the first benchmark for multimodal claim extraction from social media, consisting of posts containing text and one or more images, annotated with gold-standard claims derived from real-world fact-checkers. We evaluate state-of-the-art multimodal LLMs (MLLMs) under a three-part evaluation framework (semantic alignment, faithfulness, and decontextualization) and find that baseline MLLMs struggle to model rhetorical intent and contextual cues. To address this, we introduce MICE, an intent-aware framework which shows improvements in intent-critical cases. 

---
# Sessa: Selective State Space Attention 

**Authors**: Liubomyr Horbatko  

**Link**: [PDF](https://arxiv.org/pdf/2604.18580)  

**Abstract**: Modern sequence models are dominated by Transformers, where self-attention mixes information from the visible context in an input-dependent way. However, when retrieval is not sharp and attention remains diffuse over an effective support $S_{\mathrm{eff}}(t)$, the influence of any individual token is diluted, typically scaling as $O(1/S_{\mathrm{eff}}(t))$ and reaching $O(1/\ell)$ for old tokens in full-prefix settings. Structured state-space models process sequences recurrently through an explicit feedback path; selective variants such as Mamba make this feedback input-dependent, yet when freeze time cannot be sustained over long intervals, their long-range sensitivity decays exponentially with lag. Existing architectures therefore either retrieve from the past in a single read or propagate information through a single feedback chain. We introduce Sessa, a decoder that places attention inside a feedback path, enabling recurrent many-path aggregation within a layer. Under stated assumptions, Sessa admits regimes with a power-law memory tail in lag $\ell$ of order $O(\ell^{-\beta})$ for $0<\beta<1$, which is asymptotically slower than $1/\ell$; moreover, this rate is tight in an explicit diffuse uniform-routing setting where the influence is $\Theta(\ell^{-\beta})$. Under the same conditions, only Sessa among the compared model classes realizes flexible selective retrieval, including non-decaying profiles. Empirically, under matched architectures and training budgets, Sessa achieves the strongest performance on our long-context benchmarks while remaining competitive with Transformer and Mamba style baselines on short-context language modeling. 

---
# A multimodal and temporal foundation model for virtual patient representations at healthcare system scale 

**Authors**: Andrew Zhang, Tong Ding, Sophia J. Wagner, Caiwei Tian, Ming Y. Lu, Rowland Pettit, Joshua E. Lewis, Alexandre Misrahi, Dandan Mo, Long Phi Le, Faisal Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2604.18570)  

**Abstract**: Modern medicine generates vast multimodal data across siloed systems, yet no existing model integrates the full breadth and temporal depth of the clinical record into a unified patient representation. We introduce Apollo, a multimodal temporal foundation model trained and evaluated on over three decades of longitudinal hospital records from a major US hospital system, composed of 25 billion records from 7.2 million patients, representing 28 distinct medical modalities and 12 major medical specialties. Apollo learns a unified representation space integrating over 100 thousand unique medical events in our clinical vocabulary as well as images and clinical text. This "atlas of medical concepts" forms a computational substrate for modeling entire patient care journeys comprised of sequences of structured and unstructured events, which are compressed by Apollo into virtual patient representations. To assess the potential of these whole-patient representations, we created 322 prognosis and retrieval tasks from a held-out test set of 1.4 million patients. We demonstrate the generalized clinical forecasting potential of Apollo embeddings, including predicting new disease onset risk up to five years in advance (95 tasks), disease progression (78 tasks), treatment response (59 tasks), risk of treatment-related adverse events (17 tasks), and hospital operations endpoints (12 tasks). Using feature attribution techniques, we show that model predictions align with clinically-interpretable multimodal biomarkers. We evaluate semantic similarity search on 61 retrieval tasks, and moreover demonstrate the potential of Apollo as a multimodal medical search engine using text and image queries. Together, these modeling capabilities establish the foundation for computable medicine, where the full context of patient care becomes accessible to computational reasoning. 

---
# Latent Phase-Shift Rollback: Inference-Time Error Correction via Residual Stream Monitoring and KV-Cache Steering 

**Authors**: Manan Gupta, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.18567)  

**Abstract**: Large language models frequently commit unrecoverable reasoning errors mid-generation: once a wrong step is taken, subsequent tokens compound the mistake rather than correct it. We introduce $\textbf{Latent Phase-Shift Rollback}$ (LPSR): at each generation step, we monitor the residual stream at a critical layer lcrit, detect abrupt directional reversals (phase shifts) via a cosine-similarity $+$ entropy dual gate, and respond by rolling back the KV-cache and injecting a pre-computed steering vector. No fine-tuning, gradient computation, or additional forward passes are required. LPSR achieves $\mathbf{44.0\%}$ on MATH-500 with an 8B model versus $28.8\%$ for standard AR ($+15.2$ pp; McNemar $\chi^2 = 66.96$, $p < 10^{-15}$). Critically, prompted self-correction, the most natural inference-time baseline, scores only $19.8\%$, below standard AR; LPSR exceeds it by $+24.2$ pp ($\chi^2 = 89.4$, $p \approx 0$). LPSR also outperforms Best-of-16 ($+7.8$ pp) at $5.4\times$ lower token cost, and surpasses a standard 70B model ($35.2\%$) with $8.75\times$ fewer parameters at ${\sim}3\times$ the token budget. A 32-layer sweep reveals a novel \textbf{detection-correction dissociation}: error-detection AUC peaks at layer~14 ($0.718$) but task accuracy peaks at layer~16 ($44.0\%$ vs.\ $29.2\%$), demonstrating that optimal monitoring depth differs for detection and correction. 

---
# FUSE: Ensembling Verifiers with Zero Labeled Data 

**Authors**: Joonhyuk Lee, Virginia Ma, Sarah Zhao, Yash Nair, Asher Spector, Regev Cohen, Emmanuel J. Candès  

**Link**: [PDF](https://arxiv.org/pdf/2604.18547)  

**Abstract**: Verification of model outputs is rapidly emerging as a key primitive for both training and real-world deployment of large language models (LLMs). In practice, this often involves using imperfect LLM judges and reward models since ground truth acquisition can be time-consuming and expensive. We introduce Fully Unsupervised Score Ensembling (FUSE), a method for improving verification quality by ensembling verifiers without access to ground truth correctness labels. The key idea behind FUSE is to control conditional dependencies between verifiers in a manner that improves the unsupervised performance of a class of spectral algorithms from the ensembling literature. Despite requiring zero ground truth labels, FUSE typically matches or improves upon semi-supervised alternatives in test-time scaling experiments with diverse sets of generator models, verifiers, and benchmarks. In particular, we validate our method on both conventional academic benchmarks such as GPQA Diamond and on frontier, unsaturated benchmarks such as Humanity's Last Exam and IMO Shortlist questions. 

---
# ClawEnvKit: Automatic Environment Generation for Claw-Like Agents 

**Authors**: Xirui Li, Ming Li, Derry Xu, Wei-Lin Chiang, Ion Stoica, Cho-Jui Hsieh, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.18543)  

**Abstract**: Constructing environments for training and evaluating claw-like agents remains a manual, human-intensive process that does not scale. We argue that what is needed is not just a dataset, but an automated pipeline capable of generating diverse, verified environments on demand. To this end, we introduce ClawEnvKit, an autonomous generation pipeline that instantiates this formalism from natural language descriptions. The pipeline comprises three modules: (1) a parser that extracts structured generation parameters from natural language input; (2) a generator that produces the task specification, tool interface, and scoring configuration; and (3) a validator that enforces feasibility, diversity, structural validity, and internal consistency across the generated environments. Using ClawEnvKit, we construct Auto-ClawEval, the first large-scale benchmark for claw-like agents, comprising 1,040 environments across 24 categories. Empirically, Auto-ClawEval matches or exceeds human-curated environments on coherence and clarity at 13,800x lower cost. Evaluated across 4 model families and 8 agent harness frameworks, we find that harness engineering boosts performance by up to 15.7 percentage points over a bare ReAct baseline, completion remains the primary axis of variation with no model saturating the benchmark, and automated generation enables evaluation at a scale previously infeasible. Beyond static benchmarking, ClawEnvKit enables live evaluation: users describe a desired capability in natural language and obtain a verified environment on demand, turning evaluation into a continuous, user-driven process. The same mechanism serves as an on-demand training environment generator, producing task distributions that adapt to an agent's current weaknesses rather than being bounded by existing user logs. 

---
# Different Paths to Harmful Compliance: Behavioral Side Effects and Mechanistic Divergence Across LLM Jailbreaks 

**Authors**: Md Rysul Kabir, Zoran Tiganj  

**Link**: [PDF](https://arxiv.org/pdf/2604.18510)  

**Abstract**: Open-weight language models can be rendered unsafe through several distinct interventions, but the resulting models may differ substantially in capabilities, behavioral profile, and internal failure mode. We study behavioral and mechanistic properties of jailbroken models across three unsafe routes: harmful supervised fine-tuning (SFT), harmful reinforcement learning with verifiable rewards (RLVR), and refusal-suppressing abliteration. All three routes achieve near-ceiling harmful compliance, but they diverge once we move beyond direct harmfulness. RLVR-jailbroken models show minimal degradation and preserve explicit harm recognition in a structured self-audit: they are able to identify harmful prompts and describe how a safe LLM should respond, yet they comply with the harmful request. With RLVR, harmful behavior is strongly suppressed by a reflective safety scaffold: when a harmful prompt is prepended with an instruction to reflect on safety standards, harmful behavior drops close to the baseline. Category-specific RLVR jailbreaks generalize broadly across harmfulness domains. Models jailbroken with SFT show the largest collapse in explicit safety judgments, the highest behavioral drift, and a substantial capability loss on standard benchmarks. Abliteration is family-dependent in both self-audit and response to a reflective safety scaffold. Mechanistic and repair analyses further separate the routes: abliteration is consistent with localized refusal-feature deletion, RLVR with preserved safety geometry but retargeted policy behavior, and SFT with broader distributed drift. Targeted repair partially recovers RLVR-jailbroken models, but has little effect on SFT-jailbroken models. Together, these results show that jailbreaks can produce vastly different properties despite similar harmfulness, with models jailbroken via RLVR showing remarkable similarity to the base model. 

---
# Document-as-Image Representations Fall Short for Scientific Retrieval 

**Authors**: Ghazal Khalighinejad, Raghuveer Thirukovalluru, Alexander H. Oh, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2604.18508)  

**Abstract**: Many recent document embedding models are trained on document-as-image representations, embedding rendered pages as images rather than the underlying source. Meanwhile, existing benchmarks for scientific document retrieval, such as ArXivQA and ViDoRe, treat documents as images of pages, implicitly favoring such representations. In this work, we argue that this paradigm is not well-suited for text-rich multimodal scientific documents, where critical evidence is distributed across structured sources, including text, tables, and figures. To study this setting, we introduce ArXivDoc, a new benchmark constructed from the underlying LaTeX sources of scientific papers. Unlike PDF or image-based representations, LaTeX provides direct access to structured elements (e.g., sections, tables, figures, equations), enabling controlled query construction grounded in specific evidence types. We systematically compare text-only, image-based, and multimodal representations across both single-vector and multi-vector retrieval models. Our results show that: (1) document-as-image representations are consistently suboptimal, especially as document length increases; (2) text-based representations are most effective, even for figure-based queries, by leveraging captions and surrounding context; and (3) interleaved text+image representations outperform document-as-image approaches without requiring specialized training. 

---
# Aligning Language Models for Lyric-to-Melody Generation with Rule-Based Musical Constraints 

**Authors**: Hao Meng, Siyuan Zheng, Shuran Zhou, Qiangqiang Wang, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.18489)  

**Abstract**: Large Language Models (LLMs) show promise in lyric-to-melody generation, but models trained with Supervised Fine-Tuning (SFT) often produce musically implausible melodies with issues like poor rhythm and unsuitable vocal ranges, a phenomenon we term "constraint violation". To address this, we propose a novel alignment framework that instills musical knowledge without human annotation. We define rule-based musical constraints to automatically generate a preference dataset from an SFT model's outputs. The model is then aligned through a sequential process, first using Direct Preference Optimization (DPO) on paired preference data, followed by Kahneman-Tversky Optimization (KTO) on unpaired negative samples. Experimental results demonstrate that our aligned model substantially reduces rule violations and outperforms strong baselines in both objective and subjective evaluations, generating melodies with substantially improved musicality and coherence. An interactive demo with audio comparisons is available at this https URL. 

---
# OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation 

**Authors**: Jinghui Lu, Jiayi Guan, Zhijian Huang, Jinlong Li, Guang Li, Lingdong Kong, Yingyan Li, Han Wang, Shaoqing Xu, Yuechen Luo, Fang Li, Chenxu Dang, Junli Wang, Tao Xu, Jing Wu, Jianhua Wu, Xiaoshuai Hao, Wen Zhang, Tianyi Jiang, Lingfeng Zhang, Lei Zhou, Yingbo Tang, Jie Wang, Yinfeng Gao, Xizhou Bu, Haochen Tian, Yihang Qiu, Feiyang Jia, Lin Liu, Yigu Ge, Hanbing Li, Yuannan Shen, Jianwei Cui, Hongwei Xie, Bing Wang, Haiyang Sun, Jingwei Zhao, Jiahui Huang, Pei Liu, Zeyu Zhu, Yuncheng Jiang, Zibin Guo, Chuhong Gong, Hanchao Leng, Kun Ma, Naiyang Wang, Guang Chen, Kuiyuan Yang, Hangjun Ye, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.18486)  

**Abstract**: Chain-of-Thought (CoT) reasoning has become a powerful driver of trajectory prediction in VLA-based autonomous driving, yet its autoregressive nature imposes a latency cost that is prohibitive for real-time deployment. Latent CoT methods attempt to close this gap by compressing reasoning into continuous hidden states, but consistently fall short of their explicit counterparts. We suggest that this is due to purely linguistic latent representations compressing a symbolic abstraction of the world, rather than the causal dynamics that actually govern driving. Thus, we present OneVL (One-step latent reasoning and planning with Vision-Language explanations), a unified VLA and World Model framework that routes reasoning through compact latent tokens supervised by dual auxiliary decoders. Alongside a language decoder that reconstructs text CoT, we introduce a visual world model decoder that predicts future-frame tokens, forcing the latent space to internalize the causal dynamics of road geometry, agent motion, and environmental change. A three-stage training pipeline progressively aligns these latents with trajectory, language, and visual objectives, ensuring stable joint optimization. At inference, the auxiliary decoders are discarded and all latent tokens are prefilled in a single parallel pass, matching the speed of answer-only prediction. Across four benchmarks, OneVL becomes the first latent CoT method to surpass explicit CoT, delivering state-of-the-art accuracy at answer-only latency, and providing direct evidence that tighter compression, when guided in both language and world-model supervision, produces more generalizable representations than verbose token-by-token reasoning. Project Page: this https URL 

---
# WorldDB: A Vector Graph-of-Worlds Memory Engine with Ontology-Aware Write-Time Reconciliation 

**Authors**: Harish Santhanalakshmi Ganesan  

**Link**: [PDF](https://arxiv.org/pdf/2604.18478)  

**Abstract**: Persistent memory is the bottleneck separating stateless chatbots from long-running agentic systems. Retrieval-augmented generation (RAG) over flat vector stores fragments facts into chunks, loses cross-session identity, and has no first-class notion of supersession or contradiction. Recent bitemporal knowledge-graph systems (Graphiti, Memento, Hydra DB) add typed edges and valid-time metadata, but the graph itself remains flat: no recursive composition, no content-addressed invariants on nodes, and edge types carry no behavior beyond a label. We present WorldDB, a memory engine built on three commitments: (i) every node is a world -- a container with its own interior subgraph, ontology scope, and composed embedding, recursive to arbitrary depth; (ii) nodes are content-addressed and immutable, so any edit produces a new hash at the node and every ancestor, giving a Merkle-style audit trail for free; (iii) edges are write-time programs -- each edge type ships on_insert/on_delete/on_query_rewrite handlers (supersession closes validity, contradicts preserves both sides, same_as stages a merge proposal), so no raw append path exists. On LongMemEval-s (500 questions, ~115k-token conversational stacks), WorldDB with Claude Opus 4.7 as answerer achieves 96.40% overall / 97.11% task-averaged accuracy, a +5.61pp improvement over the previously reported Hydra DB state-of-the-art (90.79%) and +11.20pp over Supermemory (85.20%), with perfect single-session-assistant recall and robust performance on temporal reasoning (96.24%), knowledge update (98.72%), and preference synthesis (96.67%). Ablations show that the engine's graph layer -- resolver-unified entities and typed refers_to edges -- contributes +7.0pp task-averaged independently of the underlying answerer. 

---
# ESsEN: Training Compact Discriminative Vision-Language Transformers in a Low-Resource Setting 

**Authors**: Clayton Fields, Casey Kennington  

**Link**: [PDF](https://arxiv.org/pdf/2604.18452)  

**Abstract**: Vision-language modeling is rapidly increasing in popularity with an ever expanding list of available models. In most cases, these vision-language models have parameters in the tens of billions, which is necessary for some needs, but in many cases smaller models are necessary (e.g., on edge devices or independent robotic platforms). Unfortunately, there is little research in producing light-weight models or in training them with small datasets. Inspired by the language learning progression and data sparsity in child development, in this paper, we address both of these goals in a systematic fashion. We show that two-tower encoder models are superior to one-tower encoders in low-resource settings for discriminative English tasks. We show also that incorporating traditional convolutional networks into the two-tower transformer architecture can help produce parameter efficient vision-language models. Finally, we show that the cross-modal fusion module of two-tower encoders can vary significantly in shape and size while producing the same results. In addition, we present ESsEN, a compact vision-language model that can be trained end-to-end with relatively few resources that performs as well on several tasks with only a fraction of the parameters compared to other models. The experimental results and the tools we present here make vision-language modeling more accessible to a wider variety of researchers. 

---
# Knowing When to Quit: A Principled Framework for Dynamic Abstention in LLM Reasoning 

**Authors**: Hen Davidov, Nachshon Cohen, Oren Kalinsky, Yaron Fairstein, Guy Kushilevitz, Ram Yazdi, Patrick Rebeschini  

**Link**: [PDF](https://arxiv.org/pdf/2604.18419)  

**Abstract**: Large language models (LLMs) using chain-of-thought reasoning often waste substantial compute by producing long, incorrect responses. Abstention can mitigate this by withholding outputs unlikely to be correct. While most abstention methods decide to withhold outputs before or after generation, dynamic mid-generation abstention considers early termination of unpromising reasoning traces at each token position. Prior work has explored empirical variants of this idea, but principled guidance for the abstention rule remains lacking. We present a formal analysis of dynamic abstention for LLMs, modeling abstention as an explicit action within a regularized reinforcement learning framework. An abstention reward parameter controls the trade-off between compute and information. We show that abstaining when the value function falls below this reward strictly outperforms natural baselines under general conditions. We further derive a principled and efficient method to approximate the value function. Empirical results on mathematical reasoning and toxicity avoidance tasks support our theory and demonstrate improved selective accuracy over existing methods. 

---
# Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval 

**Authors**: HaeJun Yoo, Yongseop Shin, Insung Lee, Myoung-Wan Koo, Du-Seong Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18360)  

**Abstract**: Audio-text retrieval systems based on Contrastive Language-Audio Pretraining (CLAP) achieve strong performance on traditional benchmarks; however, these benchmarks rely on caption-style queries that differ substantially from real-world search behavior, limiting their assessment of practical retrieval robustness. We present Omni-Embed-Audio (OEA), a retrieval-oriented encoder leveraging multimodal LLMs with native audio understanding. To systematically evaluate robustness beyond caption-style queries, we introduce User-Intent Queries (UIQs) - five formulations reflecting natural search behaviors: questions, commands, keyword tags, paraphrases, and exclusion-based negative queries. For negative queries, we develop a hard negative mining pipeline and propose discrimination metrics (HNSR, TFR) assessing models' ability to suppress acoustically similar distractors. Experiments on AudioCaps, Clotho, and MECAT show that OEA achieves comparable text-to-audio retrieval performance to state-of-the-art M2D-CLAP, while demonstrating clear advantages in two critical areas: (1) dominant text-to-text retrieval (+22% relative improvement), and (2) substantially superior hard negative discrimination (+4.3%p HNSR@10, +34.7% relative TFR@10), revealing that LLM backbones provide superior semantic understanding of complex queries. 

---
# PARM: Pipeline-Adapted Reward Model 

**Authors**: Xingyu Fan, Wei Shao, Jiacheng Liu, Linqi Song, Pheng Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2604.18327)  

**Abstract**: Reward models (RMs) are central to aligning large language models (LLMs) with human preferences, powering RLHF and advanced decoding strategies. While most prior work focuses on single-step generation, real-world applications increasingly adopt multi-stage LLM pipelines, where effective reward guidance remains underexplored. We investigate this through code generation for combinatorial optimization, constructing a pipeline that integrates reward models into both formulation and solution stages. We identify a critical challenge: inconsistency between reward model predictions and actual pipeline execution outcomes. To address this, we propose the Pipeline-Adapted Reward Model (PARM), which leverages pipeline-specific data and direct preference optimization to align rewards with downstream feedback. We instantiate PARM as a two-stage pipeline (formulation -> code generation) and evaluate it on four public optimization benchmarks, measuring execution rate and solving accuracy against baselines and sampling methods. A supplementary cross-domain experiment on GSM8K assesses transferability. Results demonstrate that PARM consistently improves pipeline output quality and stability, providing new insights into reward modeling for multi-stage LLM reasoning. 

---
# Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence 

**Authors**: Guanting Dong, Junting Lu, Junjie Huang, Wanjun Zhong, Longxiang Liu, Shijue Huang, Zhenyu Li, Yang Zhao, Xiaoshuai Song, Xiaoxi Li, Jiajie Jin, Yutao Zhu, Hanbin Wang, Fangyu Lei, Qinyu Luo, Mingyang Chen, Zehui Chen, Jiazhan Feng, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2604.18292)  

**Abstract**: Large language models are increasingly expected to serve as general-purpose agents that interact with external, stateful tool environments. The Model Context Protocol (MCP) and broader agent skills offer a unified interface for connecting agents with scalable real-world services, but training robust agents remains limited by the lack of realistic environments and principled mechanisms for life-long learning. In this paper, we present \textbf{Agent-World}, a self-evolving training arena for advancing general agent intelligence through scalable environments. Agent-World has two main components: (1) Agentic Environment-Task Discovery, which autonomously explores topic-aligned databases and executable tool ecosystems from thousands of real-world environment themes and synthesizes verifiable tasks with controllable difficulty; and (2) Continuous Self-Evolving Agent Training, which combines multi-environment reinforcement learning with a self-evolving agent arena that automatically identifies capability gaps through dynamic task synthesis and drives targeted learning, enabling the co-evolution of agent policies and environments. Across 23 challenging agent benchmarks, Agent-World-8B and 14B consistently outperforms strong proprietary models and environment scaling baselines. Further analyses reveal scaling trends in relation to environment diversity and self-evolution rounds, offering insights for building general agent intelligence. 

---
# DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion 

**Authors**: Rahul Mehta, Kavin R V, Indrajit Pal, Tushar Abhishek, Pawan Goyal, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.18257)  

**Abstract**: Query auto-completion (QAC) has been widely studied in the context of web search, yet remains underexplored for in-document search, which we term DocQAC. DocQAC aims to enhance search productivity within long documents by helping users craft faster, more precise queries, even for complex or hard-to-spell terms. While global historical queries are available to both WebQAC and DocQAC, DocQAC uniquely accesses document-specific context, including the current document's content and its specific history of user query interactions.
To address this setting, we propose a novel adaptive trie-guided decoding framework that uses user query prefixes to softly steer language models toward high-quality completions. Our approach introduces an adaptive penalty mechanism with tunable hyperparameters, enabling a principled trade-off between model confidence and trie-based guidance. To efficiently incorporate document context, we explore retrieval-augmented generation (RAG) and lightweight contextual document signals such as titles, keyphrases, and summaries.
When applied to encoder-decoder models like T5 and BART, our trie-guided framework outperforms strong baselines and even surpasses much larger instruction-tuned models such as LLaMA-3 and Phi-3 on seen queries across both seen and unseen documents. This demonstrates its practicality for real-world DocQAC deployments, where efficiency and scalability are critical. We evaluate our method on a newly introduced DocQAC benchmark derived from ORCAS, enriched with query-document pairs. We make both the DocQAC dataset (this https URL) and code (this https URL) publicly available. 

---
# Beyond Pattern Matching: Seven Cross-Domain Techniques for Prompt Injection Detection 

**Authors**: Thamilvendhan Munirathinam  

**Link**: [PDF](https://arxiv.org/pdf/2604.18248)  

**Abstract**: Current open-source prompt-injection detectors converge on two architectural choices: regular-expression pattern matching and fine-tuned transformer classifiers. Both share failure modes that recent work has made concrete. Regular expressions miss paraphrased attacks. Fine-tuned classifiers are vulnerable to adaptive adversaries: a 2025 NAACL Findings study reported that eight published indirect-injection defenses were bypassed with greater than fifty percent attack success rates under adaptive attacks. This work proposes seven detection techniques that each port a specific mechanism from a discipline outside large-language-model security: forensic linguistics, materials-science fatigue analysis, deception technology from network security, local-sequence alignment from bioinformatics, mechanism design from economics, spectral signal analysis from epidemiology, and taint tracking from compiler theory. Three of the seven techniques are implemented in the prompt-shield v0.4.1 release (Apache 2.0) and evaluated in a four-configuration ablation across six datasets including deepset/prompt-injections, NotInject, LLMail-Inject, AgentHarm, and AgentDojo. The local-alignment detector lifts F1 on deepset from 0.033 to 0.378 with zero additional false positives. The stylometric detector adds 11.1 percentage points of F1 on an indirect-injection benchmark. The fatigue tracker is validated via a probing-campaign integration test. All code, data, and reproduction scripts are released under Apache 2.0. 

---
# Audio-DeepThinker: Progressive Reasoning-Aware Reinforcement Learning for High-Quality Chain-of-Thought Emergence in Audio Language Models 

**Authors**: Xiang He, Chenxing Li, Jinting Wang, Yan Rong, Tianxin Xie, Wenfu Wang, Li Liu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18187)  

**Abstract**: Large Audio-Language Models (LALMs) have made significant progress in audio understanding, yet they primarily operate as perception-and-answer systems without explicit reasoning processes. Existing methods for enhancing audio reasoning rely either on supervised chain-of-thought (CoT) fine-tuning, which is limited by training data quality, or on reinforcement learning (RL) with coarse rewards that do not directly evaluate reasoning quality. As a result, the generated reasoning chains often appear well-structured yet lack specific acoustic grounding. We propose Audio-DeepThinker, a framework built on two core ideas. First, we introduce a hybrid reasoning similarity reward that directly supervises the quality of generated reasoning chains by combining an LLM evaluator assessing logical path alignment, key step coverage, and analytical depth with an embedding similarity component enforcing semantic alignment with reference reasoning chains. Second, we propose a progressive two-stage curriculum that enables high-quality CoT reasoning to emerge through pure RL exploration, without any supervised reasoning fine-tuning, from an instruction-tuned model that possesses no prior chain-of-thought capability. Stage 1 trains on foundational audio QA with the hybrid reward to foster basic reasoning patterns, while Stage 2 shifts to acoustically challenging boundary cases with an LLM-only reward for greater reasoning diversity. Audio-DeepThinker achieves state-of-the-art results on MMAR (74.0%), MMAU-test-mini (78.5%), and MMSU (77.26%), winning 1st Place in the Interspeech 2026 Audio Reasoning Challenge (Single Model Track). Interpretability analyses further reveal that RL training primarily reshapes upper-layer MoE gating mechanisms and that reasoning tokens crystallize progressively in the upper transformer layers, offering mechanistic insights into how audio reasoning emerges through exploration. 

---
# Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations 

**Authors**: Yunjia Xi, Menghui Zhu, Jianghao Lin, Bo Chen, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18146)  

**Abstract**: Recently, large language models (LLMs) have advanced recommendation systems (RSs), and recent works have begun to explore how to integrate LLMs into industrial RSs. While most approaches deploy LLMs offline to generate and pre-cache augmented representations for RSs, high-dimensional representations from LLMs introduce substantial storage and computational costs. Thus, it is crucial to compress LLM representations effectively. However, we identify a counterintuitive phenomenon during representation compression: Mid-layer Representation Advantage (MRA), where representations from middle layers of LLMs outperform those from final layers in recommendation tasks. This degraded final layer renders existing compression methods, which typically compress on the final layer, suboptimal. We interpret this based on modularity theory that LLMs develop spontaneous internal functional modularity and force the final layer to specialize in the proxy training task. Thus, we propose \underline{M}odul\underline{a}r \underline{R}epresentation \underline{C}ompression (MARC) to explicitly control the modularity of LLMs. First, Modular Adjustment explicitly introduces compression and task adaptation modules, enabling the LLM to operate strictly as a representation-learning module. Next, to ground each module to its specific task, Modular Task Decoupling uses information constraints and different network structures to decouple tasks. Extensive experiments validate that MARC addresses MRA and produces efficient representations. Notably, MARC achieved a 2.82% eCPM lift in an online A/B test within a large-scale commercial search advertising scenario. 

---
# NIM4-ASR: Towards Efficient, Robust, and Customizable Real-Time LLM-Based ASR 

**Authors**: Yuan Xie, Jiaqi Song, Guang Qiu, Xianliang Wang, Kai Qiao, Junfeng Yuan, Shengqing Liu, Yi Zhang, Bowen Chen, Ming Lei, Jie Gao, Jie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18105)  

**Abstract**: Integrating large language models (LLMs) into automatic speech recognition (ASR) has become a mainstream paradigm in recent years. Although existing LLM-based ASR models demonstrate impressive performance on public benchmarks, their training remains predominantly data-driven, leaving key practical challenges insufficiently addressed -- particularly limited downward scalability in resource-constrained deployments and hallucinations under acoustically challenging conditions. To address these issues, we present NIM4-ASR, a production-oriented LLM-based ASR framework optimized for both efficiency and robustness. Grounded in a principled delineation of functional roles between the encoder and the LLM, we redesign the multi-stage training paradigm to align each module with its intended capability boundary. Specifically, we reformulate the pre-training architecture and objective to mitigate the modality gap and improve parameter efficiency; introduce an iterative asynchronous SFT stage to preserve acoustic fidelity and constrain representation drift; and design an ASR-specialized reinforcement learning stage to further enhance recognition quality and robustness. We additionally incorporate a suite of production-oriented optimizations, including robustness under noisy and silent conditions, real-time streaming inference, and hotword customization via retrieval-augmented generation (RAG). Experiments show that NIM4-ASR achieves state-of-the-art performance on multiple public benchmarks with merely 2.3B parameters, while substantially outperforming larger-scale competitors on internal benchmarks -- particularly in entity-intensive real-world scenarios. NIM4-ASR further supports million-scale hotword customization via RAG with sub-millisecond retrieval latency, enabling efficient adaptation to emerging entities and personalized user requirements. 

---
# CodePivot: Bootstrapping Multilingual Transpilation in LLMs via Reinforcement Learning without Parallel Corpora 

**Authors**: Shangyu Li, Juyong Jiang, Meibo Ren, Sizhe Zhong, Huiri Tan, Yunhao Gou, Xu Han, Chun Yong Chong, Yun Peng, Jiasi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.18027)  

**Abstract**: Transpilation, or code translation, aims to convert source code from one programming language (PL) to another. It is beneficial for many downstream applications, from modernizing large legacy codebases to augmenting data for low-resource PLs. Recent large language model (LLM)-based approaches have demonstrated immense potential for code translation. Among these approaches, training-based methods are particularly important because LLMs currently do not effectively adapt to domain-specific settings that suffer from a lack of knowledge without targeted training. This limitation is evident in transpilation tasks involving low-resource PLs. However, existing training-based approaches rely on a pairwise transpilation paradigm, making it impractical to support a diverse range of PLs. This limitation is particularly prominent for low-resource PLs due to a scarcity of training data. Furthermore, these methods suffer from suboptimal reinforcement learning (RL) reward formulations. To address these limitations, we propose CodePivot, a training framework that leverages Python as an intermediate representation (IR), augmented by a novel RL reward mechanism, Aggressive-Partial-Functional reward, to bootstrap the model's multilingual transpilation ability without requiring parallel corpora. Experiments involving 10 PLs show that the resulting 7B model, trained on Python-to-Others tasks, consistently improves performance across both general and low-resource PL-related transpilation tasks. It outperforms substantially larger mainstream models with hundreds of billions more parameters, such as Deepseek-R1 and Qwen3-235B-A22B-Instruct-2507, on Python-to-Others tasks and Others-to-All tasks, respectively. In addition, it outperforms its counterpart trained directly on Any-to-Any tasks on general transpilation tasks. The code and data are available at this https URL. 

---
# Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation 

**Authors**: Nuo Chen, Yicheng Tong, Yuzhe Yang, Yufei He, Xueyi Zhang, Zou Qingyun, Qian Wang, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2604.18005)  

**Abstract**: Multi-agent systems (MAS) are increasingly used for open-ended idea generation, driven by the expectation that collective interaction will broaden the exploration diversity. However, when and why such collaboration truly expands the solution space remains unclear. We present a systematic empirical study of diversity in MAS-based ideation across three bottom-up levels: model intelligence, agent cognition, and system dynamics. At the model level, we identify a compute efficiency paradox, where stronger, highly aligned models yield diminishing marginal diversity despite higher per-sample quality. At the cognition level, authority-driven dynamics suppress semantic diversity compared to junior-dominated groups. At the system level, group-size scaling yields diminishing returns and dense communication topologies accelerate premature convergence. We characterize these outcomes as collective failures emerging from structural coupling, a process where interaction inadvertently contracts agent exploration and triggers diversity collapse. Our analysis shows that this collapse arises primarily from the interaction structure rather than inherent model insufficiency, highlighting the importance of preserving independence and disagreement when designing MAS for creative tasks. Our code is available at this https URL. 

---
# Mitigating Multimodal Hallucination via Phase-wise Self-reward 

**Authors**: Yu Zhang, Chuyang Sun, Kehai Chen, Xuefeng Bai, Yang Xiang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17982)  

**Abstract**: Large Vision-Language Models (LVLMs) still struggle with vision hallucination, where generated responses are inconsistent with the visual input. Existing methods either rely on large-scale annotated data for fine-tuning, which incurs massive computational overhead, or employ static post-hoc strategies that overlook the dynamic nature of hallucination emergence. To address these, we introduce a new self-rewarding framework, enabling dynamic hallucination mitigation at inference time without external supervision. On the empirical side, we reveal that visual hallucination exhibits phase-wise dynamic patterns, peaking at the onset of each semantic phase. Drawing on these insights, we propose \textbf{PSRD} (\textbf{Phase-wise \textbf{S}elf-\textbf{R}eward \textbf{D}ecoding) for online hallucination correction guided by phase-wise self-reward signals. To reduce the cost of repeated self-evaluation during decoding, we distill the hallucination guidance signal from LVLMs into a lightweight reward model. The reward model subsequently provides on-the-fly guidance for targeted intervention during the decoding process, enabling precise hallucination suppression. The proposed PSRD significantly reduces the hallucination rate of LLaVA-1.5-7B by 50.0% and consistently outperforms existing post-hoc methods across five hallucination evaluation benchmarks for four LVLMs. Further analysis confirms that PSRD effectively mitigates hallucination propagation and achieves a highly controllable trade-off between strong performance and inference efficiency. 

---
# From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives? 

**Authors**: Hasan Amin, Harry Yizhou Tian, Xiaoni Duan, Chien-Ju Ho, Rajiv Khanna, Ming Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17968)  

**Abstract**: Although large language models (LLMs) are increasingly used as annotators at scale, they are typically treated as a pragmatic fallback rather than a faithful estimator of human perspectives. This work challenges that presumption. By framing perspective-taking as the estimation of a latent group-level judgment, we characterize the conditions under which modern LLMs can outperform human annotators, including in-group humans, when predicting aggregate subgroup opinions on subjective tasks, and show that these conditions are common in practice. This advantage arises from structural properties of LLMs as estimators, including low variance and reduced coupling between representation and processing biases, rather than any claim of lived experience. Our analysis identifies clear regimes where LLMs act as statistically superior frontline estimators, as well as principled limits where human judgment remains essential. These findings reposition LLMs from a cost-saving compromise to a principled tool for estimating collective human perspectives. 

---
# From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models 

**Authors**: Qidong Wang, Junjie Hu, Ming Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17941)  

**Abstract**: Recent work has increasingly explored neuron-level interpretation in vision-language models (VLMs) to identify neurons critical to final predictions. However, existing neuron analyses generally focus on single tasks, limiting the comparability of neuron importance across tasks. Moreover, ranking strategies tend to score neurons in isolation, overlooking how task-dependent information pathways shape the write-in effects of feed-forward network (FFN) neurons. This oversight can exacerbate neuron polysemanticity in multi-task settings, introducing noise into the identification and intervention of task-critical neurons. In this study, we propose HONES (Head-Oriented Neuron Explanation & Steering), a gradient-free framework for task-aware neuron attribution and steering in multi-task VLMs. HONES ranks FFN neurons by their causal write-in contributions conditioned on task-relevant attention heads, and further modulates salient neurons via lightweight scaling. Experiments on four diverse multimodal tasks and two popular VLMs show that HONES outperforms existing methods in identifying task-critical neurons and improves model performance after steering. Our source code is released at: this https URL. 

---
# Polysemantic Experts, Monosemantic Paths: Routing as Control in MoEs 

**Authors**: Charles Ye, Bo Yuan, Lee Sharkey  

**Link**: [PDF](https://arxiv.org/pdf/2604.17837)  

**Abstract**: An LLM's residual stream is both state and instruction: it encodes the current context and determines the next transformation. We introduce a parameter-free decomposition for Mixture-of-Experts models that splits each layer's hidden state into a control signal that causally drives routing and an orthogonal content channel invisible to the router. Across six MoE architectures, we find that models preserve surface-level features (language, token identity, position) in the content channel, while the control signal encodes an abstract function that rotates from layer to layer. Because each routing decision is low-bandwidth, this hand-off forces compositional specialization across layers. While individual experts remain polysemantic, expert paths become monosemantic, clustering tokens by semantic function across languages and surface forms. The same token (e.g., ":") follows distinct trajectories depending on whether it serves as a type annotation, an introductory colon, or a time separator. Our decomposition identifies the source of this structure: clusters in the control subspace are substantially more monosemantic than those in the full representation. As a result, the natural unit of interpretability in MoEs is not the expert but the trajectory. 

---
# A novel LSTM music generator based on the fractional time-frequency feature extraction 

**Authors**: Li Ya, Chen Wei, Li Xiulai, Yu Lei, Deng Xinyi, Chen Chaofan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17823)  

**Abstract**: In this paper, we propose a novel approach for generating music based on an artificial intelligence (AI) system. We analyze the features of music and use them to fit and predict the music. The fractional Fourier transform (FrFT) and the long short-term memory (LSTM) network are the foundations of our method. The FrFT method is used to extract the spectral features of a music piece, where the music signal is expressed on the time and frequency domains. The LSTM network is used to generate new music based on the extracted features, where we predict the music according to the hidden layer features and real-time inputs using GiantMIDI-Piano dataset. The results of our experiments show that our proposed system is capable of generating high-quality music that is comparable to human-generated music. 

---
# Navigating the Conceptual Multiverse 

**Authors**: Andre Ye, Jenny Y. Huang, Alicia Guo, Rose Novick, Tamara Broderick, Mitchell L. Gordon  

**Link**: [PDF](https://arxiv.org/pdf/2604.17815)  

**Abstract**: When language models answer open-ended problems, they implicitly make hidden decisions that shape their outputs, leaving users with uncontextualized answers rather than a working map of the problem; drawing on multiverse analysis from statistics, we build and evaluate the conceptual multiverse, an interactive system that represents conceptual decisions such as how to frame a question or what to value as a space users can transparently inspect, intervenably change, and check against principled domain reasoning; for this structure to be worth navigating rather than misleading, it must be rigorous and checkable against domain reasoning norms, so we develop a general verification framework that enforces properties of good decision structures like unambiguity and completeness calibrated by expert-level reasoning; across three domains, the conceptual multiverse helped participants develop a working map of the problem, with philosophy students rewriting essays with sharper framings and reversed theses, alignment annotators moving from surface preferences to reasoning about user intent and harm, and poets identifying compositional patterns that clarified their taste. 

---
# Enabling AI ASICs for Zero Knowledge Proof 

**Authors**: Jianming Tong, Jingtian Dang, Simon Langowski, Tianhao Huang, Asra Ali, Jeremy Kun, Jevin Jiang, Srinivas Devadas, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2604.17808)  

**Abstract**: Zero-knowledge proof (ZKP) provers remain costly because multi-scalar multiplication (MSM) and number-theoretic transforms (NTTs) dominate runtime as they need significant computation. AI ASICs such as TPUs provide massive matrix throughput and SotA energy efficiency. We present MORPH, the first framework that reformulates ZKP kernels to match AI-ASIC execution. We introduce Big-T complexity, a hardware-aware complexity model that exposes heterogeneous bottlenecks and layout-transformation costs ignored by Big-O. Guided by this analysis, (1) at arithmetic level, MORPH develops an MXU-centric extended-RNS lazy reduction that converts high-precision modular arithmetic into dense low-precision GEMMs, eliminating all carry chains, and (2) at dataflow level, MORPH constructs a unified-sharding layout-stationary TPU Pippenger MSM and optimized 3/5-step NTT that avoid on-TPU shuffles to minimize costly memory reorganization. Implemented in JAX, MORPH enables TPUv6e8 to achieve up-to 10x higher throughput on NTT and comparable throughput on MSM than GZKP. Our code: this https URL. 

---
# DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization 

**Authors**: Haokun Lin, Xinle Jia, Haobo Xu, Bingchen Yao, Xianglong Guo, Yichen Wu, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.17789)  

**Abstract**: The MXFP4 microscaling format, which partitions tensors into blocks of 32 elements sharing an E8M0 scaling factor, has emerged as a promising substrate for efficient LLM inference, backed by native hardware support on NVIDIA Blackwell Tensor Cores. However, activation outliers pose a unique challenge under this format: a single outlier inflates the shared block scale, compressing the effective dynamic range of the remaining elements and causing significant quantization error. Existing rotation-based remedies, including randomized Hadamard and learnable rotations, are data-agnostic and therefore unable to specifically target the channels where outliers concentrate. We propose DuQuant++, which adapts the outlier-aware fine-grained rotation of DuQuant to the MXFP4 format by aligning the rotation block size with the microscaling group size (B{=}32). Because each MXFP4 group possesses an independent scaling factor, the cross-block variance issue that necessitates dual rotations and a zigzag permutation in the original DuQuant becomes irrelevant, enabling DuQuant++ to replace the entire pipeline with a single outlier-aware rotation, which halves the online rotation cost while simultaneously smoothing the weight distribution. Extensive experiments on the LLaMA-3 family under MXFP4 W4A4 quantization show that DuQuant++ consistently achieves state-of-the-art performance. Our code is available at this https URL. 

---
# Contrastive Attribution in the Wild: An Interpretability Analysis of LLM Failures on Realistic Benchmarks 

**Authors**: Rongyuan Tan, Jue Zhang, Zhuozhao Li, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17761)  

**Abstract**: Interpretability tools are increasingly used to analyze failures of Large Language Models (LLMs), yet prior work largely focuses on short prompts or toy settings, leaving their behavior on commonly used benchmarks underexplored. To address this gap, we study contrastive, LRP-based attribution as a practical tool for analyzing LLM failures in realistic settings. We formulate failure analysis as \textit{contrastive attribution}, attributing the logit difference between an incorrect output token and a correct alternative to input tokens and internal model states, and introduce an efficient extension that enables construction of cross-layer attribution graphs for long-context inputs. Using this framework, we conduct a systematic empirical study across benchmarks, comparing attribution patterns across datasets, model sizes, and training checkpoints. Our results show that this token-level contrastive attribution can yield informative signals in some failure cases, but is not universally applicable, highlighting both its utility and its limitations for realistic LLM failure analysis. Our code is available at: this https URL. 

---
# Evolutionary Negative Module Pruning for Better LoRA Merging 

**Authors**: Anda Cao, Zhuo Gou, Yi Wang, Kaixuan Chen, Yu Wang, Can Wang, Mingli Song, Jie Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.17753)  

**Abstract**: Merging multiple Low-Rank Adaptation (LoRA) experts into a single backbone is a promising approach for efficient multi-task deployment. While existing methods strive to alleviate interference via weight interpolation or subspace alignment, they rest upon the implicit assumption that all LoRA matrices contribute constructively to the merged model. In this paper, we uncover a critical bottleneck in current merging paradigms: the existence of $\textit{negative modules}$ -- specific LoRA layers that inherently degrade global performance upon merging. We propose $\textbf{E}$volutionary $\textbf{N}$egative $\textbf{M}$odule $\textbf{P}$runing ($\textbf{ENMP}$), a plug-and-play LoRA pruning method to locate and exclude these detrimental modules prior to merging. By leveraging an evolutionary search strategy, ENMP effectively navigates the discrete, non-differentiable landscape of module selection to identify optimal pruning configurations. Extensive evaluations demonstrate that ENMP consistently boosts the performance of existing merging algorithms, achieving a new state-of-the-art across both language and vision domains. Code is available at this https URL. 

---
# HiP-LoRA: Budgeted Spectral Plasticity for Robust Low-Rank Adaptation 

**Authors**: Lixian Chen, Jianhong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17751)  

**Abstract**: Adapting foundation models under resource budgets relies heavily on Parameter-Efficient Fine-Tuning (PEFT), with LoRA being a standard modular solution. However, LoRA suffers from spectral interference. Low-rank updates often concentrate energy on the leading singular directions of pretrained weights, perturbing general capabilities and causing catastrophic forgetting and fragile multi-adapter merging. To resolve this, we propose HiP-LoRA, a spectrum-aware adaptation framework. Utilizing the cached singular value decomposition (SVD) of pretrained layers, HiP-LoRA decomposes updates into two channels: a principal channel within the dominant singular subspace, and a residual low-rank channel in the orthogonal complement. A singular-value-weighted stability budget on the principal channel continuously balances pretrained behavior preservation with task-specific plasticity. Experiments on Llama-3.1-8B demonstrate that under matched budgets, HiP-LoRA drastically reduces pretraining degradation and multi-adapter MergeFail, robustly outperforming baselines in interference-sensitive tasks like continual tuning and knowledge editing. 

---
# Tool Learning Needs Nothing More Than a Free 8B Language Model 

**Authors**: Chenming Tang, Hsiu-Yuan Huang, Weijie Liu, Junqiang Zheng, Saiyong Yang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17739)  

**Abstract**: Reinforcement learning (RL) has become a prevalent paradigm for training tool calling agents, which typically requires online interactive environments. Existing approaches either rely on training data with ground truth annotations or require advanced commercial language models (LMs) to synthesize environments that keep fixed once created. In this work, we propose TRUSTEE, a data-free method training tool calling agents with dynamic environments fully simulated by free open-source LMs that can be as small as 8B, including task generation, user simulation, tool simulation and trajectory evaluation, paired with an adaptive curriculum learning mechanism that controls various aspects of the task difficulty dynamically during training. Our empirical results show that TRUSTEE brings consistent improvements across various domains and outperforms all the baselines which require extra external resources for training. These confirm that, with a sufficiently sophisticated design, even simulated environments with a local 8B LM as the backbone could set a strong baseline for tool learning, without expensive annotated data, realistic human interactions, executable tools or costly verifiable environments from human experts or commercial LMs. We hope our proposed paradigm could inspire future research on environment scaling with limited resources. 

---
# The Geometric Canary: Predicting Steerability and Detecting Drift via Representational Stability 

**Authors**: Prashant C. Raju  

**Link**: [PDF](https://arxiv.org/pdf/2604.17698)  

**Abstract**: Reliable deployment of language models requires two capabilities that appear distinct but share a common geometric foundation: predicting whether a model will accept targeted behavioral control, and detecting when its internal structure degrades. We show that geometric stability, the consistency of a representation's pairwise distance structure, addresses both. Supervised Shesha variants that measure task-aligned geometric stability predict linear steerability with near-perfect accuracy ($\rho = 0.89$-$0.97$) across 35-69 embedding models and three NLP tasks, capturing unique variance beyond class separability (partial $\rho = 0.62$-$0.76$). A critical dissociation emerges: unsupervised stability fails entirely for steering on real-world tasks ($\rho \approx 0.10$), revealing that task alignment is essential for controllability prediction. However, unsupervised stability excels at drift detection, measuring nearly $2\times$ greater geometric change than CKA during post-training alignment (up to $5.23\times$ in Llama) while providing earlier warning in 73\% of models and maintaining a $6\times$ lower false alarm rate than Procrustes. Together, supervised and unsupervised stability form complementary diagnostics for the LLM deployment lifecycle: one for pre-deployment controllability assessment, the other for post-deployment monitoring. 

---
# MoE-nD: Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache Compression 

**Authors**: Libo Sun, Peixiong He, Po-Wei Harn, Xiao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.17695)  

**Abstract**: KV cache memory is the dominant bottleneck for long-context LLM inference. Existing compression methods each act on a single axis of the four-dimensional KV tensor -- token eviction (sequence), quantization (precision), low-rank projection (head dimension), or cross-layer sharing -- but apply the same recipe to every layer. We show that this homogeneity leaves accuracy on the table: different layers respond very differently to each compression operation, and the optimal per-layer mix of eviction and quantization is far from uniform. We propose MoE-nD, a mixture-of-experts framework that routes each layer to its own (eviction-ratio, K-bits, V-bits) tuple under a global memory budget. An offline-calibrated greedy solver chooses the routing that minimizes predicted quality loss; at inference time, per-layer heterogeneous eviction and quantization are applied jointly through a single attention patch. On a 4-task subset of LongBench-v1 (16k inputs, n=50 per task, adapted reasoning-model protocol; see section Experiments), MoE-nD's hetero variant matches our uncompressed 1.9~GB baseline at 14x compression (136~MB) while every other compressed baseline we tested (1d, 2d_uniform, 2d) at comparable or smaller memory stays under 8/100. The gains hold on AIME reasoning benchmarks (+6 to +27 pts over the strongest per-layer-quantization baseline across eight configurations). Two null results -- MATH-500 and LongBench's TREC -- share a principled cause (short inputs, solver picks keep=1.0 on most layers), cleanly characterizing when per-layer eviction routing has headroom to help. 

---
# ATLAS: Constitution-Conditioned Latent Geometry and Redistribution Across Language Models and Neural Perturbation Data 

**Authors**: Gareth Seneque, Lap-Hang Ho, Nafise Erfanian Saeedi, Jeffrey Molendijk, Tim Elson  

**Link**: [PDF](https://arxiv.org/pdf/2604.17663)  

**Abstract**: Constitution-conditioned post-training can be analysed as a structured perturbation of a model's learned representational geometry. We introduce ATLAS, a geometry-first program that traces constitution-induced hidden-state structure across charts, models, and substrates. Instead of treating the relevant unit as a single behaviour, neuron, vector, or patch, ATLAS tests a local chart whose tangent structure, occupancy distribution, and behavioural coupling can be measured under system change. On Gemma, the anchored source-local chart captures 310 / 320 reviewed source rows and all 84 / 84 reviewed score-flip rows, but compact exact-patch sufficiency does not close, so the exportable unit is the broader source-defined family. Freezing that family, we re-identify a target-local realisation in an unadapted Phi model, where the fully adjudicated confirmatory contrast separates with AUC 0.984 and mean gap 5.50. In held-out ALM8 mouse frontal-cortex perturbation data, the same source-defined family receives support across 5/5 folds, with mean held-out AUC 0.72 and mean fold gap 4.50. A multiple-choice analysis provides the main boundary: nearby target-local signals can appear without source-faithful closure. The resulting correspondence is not coordinate identity, site identity, or a target-side mediation theorem. It is geometric recurrence under redistribution: written constitutions can induce recoverable latent geometry whose organisation remains detectable across model and substrate changes while its local coordinates, occupancy, and behavioural expression shift. 

---
# Towards Self-Improving Error Diagnosis in Multi-Agent Systems 

**Authors**: Jiazheng Li, Emine Yilmaz, Bei Chen, Dieu-Thu Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.17658)  

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) enable complex problem-solving but introduce significant debugging challenges, characterized by long interaction traces, inter-agent dependencies, and delayed error manifestation. Existing diagnostic approaches often rely on expensive expert annotation or ''LLM-as-a-judge'' paradigms, which struggle to pinpoint decisive error steps within extended contexts. In this paper, we introduce ErrorProbe, a self-improving framework for semantic failure attribution that identifies responsible agents and the originating error step. The framework operates via a three-stage pipeline: (1) operationalizing the MAS failure taxonomy to detect local anomalies, (2) performing symptom-driven backward tracing to prune irrelevant context, and (3) employing a specialized multi-agent team (Strategist, Investigator, Arbiter) to validate error hypotheses through tool-grounded execution. Crucially, ErrorProbe maintains a verified episodic memory that updates only when error patterns are confirmed by executable evidence, without the need for annotation. Experiments across the TracerTraj and Who&When benchmarks demonstrate that ErrorProbe significantly outperforms baselines, particularly in step-level localization, while the verified memory enables robust cross-domain transfer without retraining. 

---
# Video-Robin: Autoregressive Diffusion Planning for Intent-Grounded Video-to-Music Generation 

**Authors**: Vaibhavi Lokegaonkar, Aryan Vijay Bhosale, Vishnu Raj, Gouthaman KV, Ramani Duraiswami, Lie Lu, Sreyan Ghosh, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2604.17656)  

**Abstract**: Video-to-music (V2M) is the fundamental task of creating background music for an input video. Recent V2M models achieve audiovisual alignment by typically relying on visual conditioning alone and provide limited semantic and stylistic controllability to the end user. In this paper, we present Video-Robin, a novel text-conditioned video-to-music generation model that enables fast, high-quality, semantically aligned music generation for video content. To balance musical fidelity and semantic understanding, Video-Robin integrates autoregressive planning with diffusion-based synthesis. Specifically, an autoregressive module models global structure by semantically aligning visual and textual inputs to produce high-level music latents. These latents are subsequently refined into coherent, high-fidelity music using local Diffusion Transformers. By factoring semantically driven planning into diffusion-based synthesis, Video-Robin enables fine-grained creator control without sacrificing audio realism. Our proposed model outperforms baselines that solely accept video input and additional feature conditioned baselines on both in-distribution and out-of-distribution benchmarks with a 2.21x speed in inference compared to SOTA. We will open-source everything upon paper acceptance. 

---
# Toward Reusability of AI Models Using Dynamic Updates of AI Documentation 

**Authors**: Peter Bajcsy, Walid Keyrouz  

**Link**: [PDF](https://arxiv.org/pdf/2604.17626)  

**Abstract**: This work addresses the challenge of disseminating reusable artificial intelligence (AI) models accompanied by AI documentation (a.k.a., AI model cards). The work is motivated by the large number of trained AI models that are not reusable due to the lack of (a) AI documentation and (b) the temporal lag between rapidly changing requirements on AI model reusability and those specified in various AI model cards. Our objectives are to shorten the lag time in updating AI model card templates and align AI documentation more closely with current AI best practices.
Our approach introduces a methodology for delivering agile, data-driven, and community-based AI model cards. We use the Hugging Face (HF) repository of AI models, populated by a subset of the AI research and development community, and the AI consortium-based Zero Draft (ZD) templates for the AI documentation of AI datasets and AI models, as our test datasets. We also address questions about the value of AI documentation for AI reusability.
Our work quantifies the correlations between AI model downloads/likes (i.e., AI model reuse metrics) from the HF repository and their documentation alignment with the ZD documentation templates using tables of contents and word statistics (i.e., AI documentation quality metrics). Furthermore, our work develops the infrastructure to regularly compare AI documentation templates against community-standard practices derived from millions of uploaded AI models in the Hugging Face repository. The impact of our work lies in introducing a methodology for delivering agile, data-driven, and community-based standards for documenting AI models and improving AI model reuse. 

---
# Characterizing Model-Native Skills 

**Authors**: Feiyang Kang, Mahavir Dabas, Myeongseob Ko, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.17614)  

**Abstract**: Skills are a natural unit for describing what a language model can do and how its behavior can be changed. However, existing characterizations rely on human-written taxonomies, textual descriptions, or manual profiling pipelines--all external hypotheses about what matters that need not align with the model's internal representations. We argue that when the goal is to intervene on model behavior, skill characterization should be *model-native*: grounded in the model's own representations rather than imposed through external ontologies. We instantiate this view by recovering a compact orthogonal basis from sequence-level activations. The resulting basis is semantically interpretable but need not correspond to any predefined human ontology; instead, it captures axes of behavioral variation that the model itself organizes around. We validate this characterization on reasoning post-training, using the recovered basis for both SFT data selection and inference-time steering. We develop lightweight proxy interventions to identify which directions are most useful for a given model. Across Llama3-8B and Qwen2.5-3B, selecting data along those directions improves Pass@1 by up to 20% on MATH and 41% on AMC, outperforming data selection based on human-characterized skills. Because the basis lives in activation space, the same directions also serve as steering vectors at inference time, improving Pass@8 by up to 4.8% on MATH--an intervention that human-characterized skills cannot support. We further validate the characterization on safety alignment, where selecting adversarial training data for model-native skill coverage rather than textual diversity yields more sample-efficient learning. These results suggest that recovering skills from the model's own representations, rather than imposing them externally, provides a more effective foundation for intervening on model behavior. Codes are open-sourced. 

---
# COSEARCH: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search 

**Authors**: Hansi Zeng, Liam Collins, Bhuvesh Kumar, Neil Shah, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2604.17555)  

**Abstract**: Agentic search -- the task of training agents that iteratively reason, issue queries, and synthesize retrieved information to answer complex questions -- has achieved remarkable progress through reinforcement learning (RL). However, existing approaches such as Search-R1, treat the retrieval system as a fixed tool, optimizing only the reasoning agent while the retrieval component remains unchanged. A preliminary experiment reveals that the gap between an oracle and a fixed retrieval system reaches up to +26.8% relative F1 improvement across seven QA benchmarks, suggesting that the retrieval system is a key bottleneck in scaling agentic search performance. Motivated by this finding, we propose CoSearch, a framework that jointly trains a multi-step reasoning agent and a generative document ranking model via Group Relative Policy Optimization (GRPO). To enable effective GRPO training for the ranker -- whose inputs vary across reasoning trajectories -- we introduce a semantic grouping strategy that clusters sub-queries by token-level similarity, forming valid optimization groups without additional rollouts. We further design a composite reward combining ranking quality signals with trajectory-level outcome feedback, providing the ranker with both immediate and long-term learning signals. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate consistent improvements over strong baselines, with ablation studies validating each design choice. Our results show that joint training of the reasoning agent and retrieval system is both feasible and strongly performant, pointing to a key ingredient for future search agents. 

---
# Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories for Grounded Visual Perception 

**Authors**: Ashutosh Bajpai, Tamal Majumder, Akshay Nambi, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2604.17475)  

**Abstract**: Small Vision-Language Models (SVLMs) are efficient task controllers but often suffer from visual brittleness and poor tool orchestration. They typically require expensive supervised trajectory tuning to mitigate these deficits. In this work, we propose Self-supervised Perception Enabled by Cascaded Tool Rollout Alignment (SPECTRA), a supervision-free framework that bootstraps agentic capabilities via Coldstart Reinforcement Learning for SVLMs. SPECTRA enforces Soft Structured Multi-turn Rollouts, a topological constraint that directs agents to explicitly sequence tool derived evidence before synthesis, effectively grounding reasoning in visual observations. We employ a multi-objective reward signal that simultaneously maximizes task correctness, rollout structure, and tool utility, enabling agent to self-discover robust behaviors without human preference labels. We further introduce Tool Instrumental Utility (TIU), a novel metric to quantify tool efficacy in the absence of ground truth. Extensive evaluations across composite and out-of-distribution (MMMU-Pro) benchmarks demonstrate that SPECTRA boosts agentic trajectories, improving task accuracy by up to 5% and tool efficiency by 9%, enabling more efficient multimodal agents that learn effectively from environmental interaction alone. 

---
# FLARE: Task-agnostic embedding model evaluation through a normalization process 

**Authors**: Jingzhou Jiang, Yixuan Tang, Yi Yang, Kar Yan Tam  

**Link**: [PDF](https://arxiv.org/pdf/2604.17344)  

**Abstract**: When task-specific labels are not available, it becomes difficult to select an embedding model for a specific target corpus. Existing labelless measures based on kernel estimators or Gaussian mixes fail in high-dimensional space, resulting in unstable rankings. We propose a flow-based labelless representation embedding evaluation (FLARE), which utilizes normalized streams to estimate information sufficiency directly from log-likelihood and avoid distance-based density estimation. We give a finite sample boundary, indicating that the estimation error depends on the intrinsic dimension of the data manifold rather than the original embedding dimension. On 11 datasets and 8 embedders, FLARE reached Spearman's $\rho$ of 0.90 under the supervised benchmark and remained stable in high-dimensional embeddings ($d \geq 3{,}584$) as the existing labelless baseline collapsed. 

---
# Precise Debugging Benchmark: Is Your Model Debugging or Regenerating? 

**Authors**: Wang Bill Zhu, Miaosen Chai, Shangshang Wang, Yejia Liu, Song Bian, Honghua Dong, Willie Neiswanger, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.17338)  

**Abstract**: Unlike code completion, debugging requires localizing faults and applying targeted edits. We observe that frontier LLMs often regenerate correct but over-edited solutions during debugging. To evaluate how far LLMs are from precise debugging, we introduce the Precise Debugging Benchmark (PDB) framework, which automatically converts any coding dataset into a debugging benchmark with precision-aware evaluation. PDB generates buggy programs by synthesizing verified atomic bugs and composing them into multi-bug programs. We define two novel metrics, edit-level precision and bug-level recall, which measures how many necessary edits are made and how many bugs are resolved. We release two evaluation benchmarks: PDB-Single-Hard on single-line bugs, and PDB-Multi on multi-line bugs. Experiments show that frontier models, such as GPT-5.1-Codex and DeepSeek-V3.2-Thinking, achieve unit-test pass rates above 76% but exhibit precision below 45%, even when explicitly instructed to perform minimal debugging. Finally, we show that iterative and agentic debugging strategies do not substantially improve precision or recall, highlighting the need to rethink post-training pipelines for coding models. 

---
# E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition 

**Authors**: Meng Zhang, Jinzhong Ning, Xiaolong Wu, Hongfei Lin, Yijia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17319)  

**Abstract**: Grounded Multimodal Named Entity Recognition (GMNER) aims to jointly identify named entity mentions in text, predict their semantic types, and ground each entity to a corresponding visual region in an associated image. Existing approaches predominantly adopt pipeline-based architectures that decouple textual entity recognition and visual grounding, leading to error accumulation and suboptimal joint optimization. In this paper, we propose E2E-GMNER, a fully end-to-end generative framework that unifies entity recognition, semantic typing, visual grounding, and implicit knowledge reasoning within a single multimodal large language model. We formulate GMNER as an instruction-tuned conditional generation task and incorporate chain-of-thought reasoning to enable the model to adaptively determine when visual evidence or background knowledge is informative, reducing reliance on noisy cues. To further address the instability of generative bounding box prediction, we introduce Gaussian Risk-Aware Box Perturbation (GRBP), which replaces hard box supervision with probabilistically perturbed soft targets to improve robustness against annotation noise and discretization errors. Extensive experiments on the Twitter-GMNER and Twitter-FMNERG benchmarks demonstrate that E2E-GMNER achieves highly competitive performance compared with state of the art methods, validating the effectiveness of unified end-to-end optimization and noise-aware grounding supervision. Code is available at:this https URL 

---
# HORIZON: A Benchmark for In-the-wild User Behaviour Modeling 

**Authors**: Arnav Goel, Pranjal A Chitale, Bhawna Paliwal, Bishal Santra, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2604.17259)  

**Abstract**: User behavior in the real world is diverse, cross-domain, and spans long time horizons. Existing user modeling benchmarks however remain narrow, focusing mainly on short sessions and next-item prediction within a single domain. Such limitations hinder progress toward robust and generalizable user models. We present HORIZON, a new benchmark that reformulates user modeling along three axes i.e. dataset, task, and evaluation. Built from a large-scale, cross-domain reformulation of Amazon Reviews, HORIZON covers 54M users and 35M items, enabling both pretraining and realistic evaluation of models in heterogeneous environments. Unlike prior benchmarks, it challenges models to generalize across domains, users, and time, moving beyond standard missing-positive prediction in the same domain. We propose new tasks and evaluation setups that better reflect real-world deployment scenarios. These include temporal generalization, sequence-length variation, and modeling unseen users, with metrics designed to assess general user behavior understanding rather than isolated next-item prediction. We benchmark popular sequential recommendation architectures alongside LLM-based baselines that leverage long-term interaction histories. Our results highlight the gap between current methods and the demands of real-world user modeling, while establishing HORIZON as a foundation for research on temporally robust, cross-domain, and general-purpose user models. 

---
# VIBE: Voice-Induced open-ended Bias Evaluation for Large Audio-Language Models via Real-World Speech 

**Authors**: Yi-Cheng Lin, Yusuke Hirota, Sung-Feng Huang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.17248)  

**Abstract**: Large Audio-Language Models (LALMs) are increasingly integrated into daily applications, yet their generative biases remain underexplored. Existing speech fairness benchmarks rely on synthetic speech and Multiple-Choice Questions (MCQs), both offering a fragmented view of fairness. We propose VIBE, a framework that evaluates generative bias through open-ended tasks such as personalized recommendations, using real-world human recordings. Unlike MCQs, our method allows stereotypical associations to manifest organically without predefined options, making it easily extensible to new tasks. Evaluating 11 state-of-the-art LALMs reveals systematic biases in realistic scenarios. We find that gender cues often trigger larger distributional shifts than accent cues, indicating that current LALMs reproduce social stereotypes. 

---
# Demystifying the unreasonable effectiveness of online alignment methods 

**Authors**: Enoch Hyunwook Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17207)  

**Abstract**: Iterative alignment methods based on purely greedy updates are remarkably effective in practice, yet existing theoretical guarantees of \(O(\log T)\) KL-regularized regret can seem pessimistic relative to their empirical performance. In this paper, we argue that this mismatch arises from the regret criterion itself: KL-regularized regret conflates the statistical cost of learning with the exploratory randomization induced by the softened training policy. To separate these effects, we study the traditional temperature-zero regret criterion, which evaluates only the top-ranked response at inference time. Under this decision-centric notion of performance, we prove that standard greedy online alignment methods, including online RLHF and online DPO, achieve constant \((O(1))\) cumulative regret. By isolating the cost of identifying the best response from the stochasticity induced by regularization, our results provide a sharper theoretical explanation for the practical superb efficiency of greedy alignment. 

---
# Systematic Capability Benchmarking of Frontier Large Language Models for Offensive Cyber Tasks 

**Authors**: Tyler H. Merves, Michael H. Conaway, Joseph M. Escobar, Hakan T. Otal, Unal Tatar  

**Link**: [PDF](https://arxiv.org/pdf/2604.17159)  

**Abstract**: We present, to our knowledge, the most comprehensive cross-model evaluation of LLM agents on offensive cybersecurity tasks, benchmarking 10 frontier models from 7 providers on all 200 challenges of the NYU CTF Bench. Building on the D-CIPHER multi-agent framework, we extend it with multi-provider backend support, a custom Kali Linux environment with over 100 pre-installed penetration testing tools, and runtime tool-discovery agents. Through a controlled factorial study, we find that the Kali Linux environment yields a +9.5 percentage-point improvement over Ubuntu, while auto-prompting and category-specific tips often degrade performance in well-equipped environments. Among models, Claude 4.5 Opus achieves the highest solve rate (59%), followed by Gemini 3 Pro (52%), with Gemini 3 Flash offering the best cost-efficiency at $0.05 per solve. Asymmetric planner/executor model assignments provide no meaningful benefit while coherent same-model configurations consistently outperform mixed-tier pairings. Our results indicate that environment tooling and model selection emerge as the strongest drivers of performance, whereas prompt engineering interventions show diminishing or negative returns in well-equipped environments. Reported performance reflects both model reasoning ability and compatibility with agent tooling and API integration. 

---
# DOSE: Data Selection for Multi-Modal LLMs via Off-the-Shelf Models 

**Authors**: Biao Wu, Yiwu Zhong, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16979)  

**Abstract**: High-quality and diverse multimodal data are essential for improving vision-language models (VLMs), yet existing datasets often contain noisy, redundant, and poorly aligned samples. To address these problems, data filtering is commonly used to enhance the efficiency and performance of multimodal learning, but it introduces extra computational cost because filtering models are usually trained on the same data they are meant to screen. To reduce this cost, we study DOSE, which explores whether off-the-shelf pretrained models that have never seen the target data can be used to select training samples for larger and stronger multimodal models without any task-specific training. Even without fine-tuning, these models can effectively assess text quality and image-text alignment to guide data selection. Based on this, we build a joint quality-alignment distribution and apply adaptive weighted sampling to select informative samples while maintaining long-tail diversity. This approach enhances data diversity, enabling models trained on DOSE-filtered data to match or surpass those trained on the full dataset on standard VQA and math benchmarks. Extensive experiments demonstrate its effectiveness, efficiency, and scalability. 

---
# The Cognitive Penalty: Ablating System 1 and System 2 Reasoning in Edge-Native SLMs for Decentralized Consensus 

**Authors**: Syed Muhammad Aqdas Rizvi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16913)  

**Abstract**: Decentralized Autonomous Organizations (DAOs) are inclined explore Small Language Models (SLMs) as edge-native constitutional firewalls to vet proposals and mitigate semantic social engineering. While scaling inference-time compute (System 2) enhances formal logic, its efficacy in highly adversarial, cryptoeconomic governance environments remains underexplored. To address this, we introduce Sentinel-Bench, an 840-inference empirical framework executing a strict intra-model ablation on Qwen-3.5-9B. By toggling latent reasoning across frozen weights, we isolate the impact of inference-time compute against an adversarial Optimism DAO dataset. Our findings reveal a severe compute-accuracy inversion. The autoregressive baseline (System 1) achieved 100% adversarial robustness, 100% juridical consistency, and state finality in under 13 seconds. Conversely, System 2 reasoning introduced catastrophic instability, fundamentally driven by a 26.7% Reasoning Non-Convergence (cognitive collapse) rate. This collapse degraded trial-to-trial consensus stability to 72.6% and imposed a 17x latency overhead, introducing critical vulnerabilities to Governance Extractable Value (GEV) and hardware centralization. While rare (1.5% of adversarial trials), we empirically captured "Reasoning-Induced Sycophancy," where the model generated significantly longer internal monologues (averaging 25,750 characters) to rationalize failing the adversarial trap. We conclude that for edge-native SLMs operating under Byzantine Fault Tolerance (BFT) constraints, System 1 parameterized intuition is structurally and economically superior to System 2 iterative deliberation for decentralized consensus.
Code and Dataset: this https URL 

---
# PersonalHomeBench: Evaluating Agents in Personalized Smart Homes 

**Authors**: Nikhil Verma, InJung Yang, Sungil Kim, KoKeun Kim, YoungJoon Kim, Manasa Bharadwaj, Yolanda Liu, Kevin Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2604.16813)  

**Abstract**: Agentic AI systems are rapidly advancing toward real-world applications, yet their readiness in complex and personalized environments remains insufficiently characterized. To address this gap, we introduce PersonalHomeBench, a benchmark for evaluating foundation models as agentic assistants in personalized smart home environments. The benchmark is constructed through an iterative process that progressively builds rich household states, which are then used to generate personalized, context-dependent tasks. To support realistic agent-environment interaction, we provide PersonalHomeTools, a comprehensive toolbox enabling household information retrieval, appliance control, and situational understanding. PersonalHomeBench evaluates both reactive and proactive agentic abilities under unimodal and multimodal observations. Thorough experimentation reveals a systematic performance reduction as task complexity increases, with pronounced failures in counterfactual reasoning and under partial observability, where effective tool-based information gathering is required. These results position PersonalHomeBench as a rigorous evaluation platform for analyzing the robustness and limitations of personalized agentic reasoning and planning. 

---
# Mapping Election Toxicity on Social Media across Issue, Ideology, and Psychosocial Dimensions 

**Authors**: Lei Cao, Wen Zeng, Xinyue Wu, Eun Cheol Choi, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2604.16765)  

**Abstract**: Online political hostility is pervasive, yet it remains unclear how toxicity varies across campaign issues and political ideology, and what psychosocial signals and framing accompany toxic expression online. In this work, we present a large-scale analysis of discourse on X (Twitter) during the five weeks surrounding the 2024 U.S. presidential election. We categorize posts into 10 major campaign issues, estimate the ideology of posts using a human-in-the-loop LLM-assisted annotation process, detect harmful content with an LLM-based toxicity detection model, and then examine the psychological drivers of toxic content. We use these annotated data to examine how harmful content varies across campaign issues and ideologies, as well as how emotional tone and moral framing shape toxicity in election discussions. Our results show issue heterogeneity in both the prevalence and intensity of toxicity. Identity-related issues displayed the highest toxicity intensity. As for specific harm categories, harassment was most prevalent and intense across most of the issues, while hate concentrated in identity-centered debates. Partisan posts contained more harmful content than neutral posts, and ideological asymmetries in toxicity varied by issue. In terms of psycholinguistic dimensions, we found that toxic discourse is dominated by high-arousal negative emotions. Left- and right-leaning posts often exhibit similar emotional profiles within the same issue domain, suggesting emotional mirroring. Partisan groups frequently rely on overlapping moral foundations, while issue context strongly shapes which moral foundations become most salient. These findings provide a fine-grained account of toxic political discourse on social media and highlight that online political toxicity is highly context-dependent, underscoring the need for issue-sensitive approaches to measuring and mitigating it. 

---
# ICLAD: In-Context Learning with Comparison-Guidance for Audio Deepfake Detection 

**Authors**: Benjamin Chou, Yi Zhu, Surya Koppisetti  

**Link**: [PDF](https://arxiv.org/pdf/2604.16749)  

**Abstract**: Audio deepfakes pose a significant security threat, yet current state-of-the-art (SOTA) detection systems do not generalize well to realistic in-the-wild deepfakes. We introduce a novel \textbf{I}n-\textbf{C}ontext \textbf{L}earning paradigm with comparison-guidance for \textbf{A}udio \textbf{D}eepfake detection (\textbf{ICLAD}). The framework enables the use of audio language models (ALMs) for training-free generalization to unseen deepfakes and provides textual rationales on the detection outcome. At the core of ICLAD is a pairwise comparative reasoning strategy that guides the ALM to discover and filter hallucinations and deepfake-irrelevant acoustic attributes. The ALM works alongside a specialized deepfake detector, whereby a routing mechanism feeds out-of-distribution samples to the ALM. On in-the-wild datasets, ICLAD improves macro F1 over the specialized detector, with up to $2\times$ relative improvement. Further analysis demonstrates the flexibility of ICLAD and its potential for deployment on recent open-source ALMs. 

---
# CT Open: An Open-Access, Uncontaminated, Live Platform for the Open Challenge of Clinical Trial Outcome Prediction 

**Authors**: Jianyou Wang, Youze Zheng, Longtian Bao, Hanyuan Zhang, Qirui Zheng, Yuhan Chen, Yang Zhang, Matthew Feng, Maxim Khan, Aditya K. Sehgal, Christopher D. Rosin, Ramamohan Paturi, Umber Dube, Leon Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16742)  

**Abstract**: Scientists have long sought to accurately predict outcomes of real-world events before they happen. Can AI systems do so more reliably? We study this question through clinical trial outcome prediction, a high-stakes open challenge even for domain experts. We introduce CT Open, an open-access, live platform that will run four challenge every year. Anyone can submit predictions for each challenge. CT Open evaluates those submissions on trials whose outcomes were not yet public at the time of submission but were made public afterwards. Determining if a trial's outcome is public on the internet before a certain date is surprisingly difficult. Outcomes posted on official registries may lag behind by years, while the first mention may appear in obscure articles. To address this, we propose a novel, fully automated decontamination pipeline that uses iterative LLM-powered web search to identify the earliest mention of trial outcomes. We validate the pipeline's quality and accuracy by human expert's annotations. Since CT Open's pipeline ensures that every evaluated trial had no publicly reported outcome when the prediction was made, it allows participants to use any methodology and any data source. In this paper, we release a training set and two time-stamped test benchmarks, Winter 2025 and Summer 2025. We believe CT Open can serve as a central hub for advancing AI research on forecasting real-world outcomes before they occur, while also informing biomedical research and improving clinical trial design. CT Open Platform is hosted at $\href{this https URL}{this https URL}$ 

---
# Evaluating Tool-Using Language Agents: Judge Reliability, Propagation Cascades, and Runtime Mitigation in AgentProp-Bench 

**Authors**: Bhaskar Gurram  

**Link**: [PDF](https://arxiv.org/pdf/2604.16706)  

**Abstract**: Automated evaluation of tool-using large language model (LLM) agents is widely assumed to be reliable, but this assumption has rarely been validated against human annotation. We introduce AgentProp-Bench, a 2,000-task benchmark with 2,300 traces across four domains, nine production LLMs, and a 100-label human-validated subset. We quantify judge reliability, characterize error propagation, and evaluate a runtime mitigation. Substring-based judging agrees with human annotation at kappa=0.049 (chance-level); a three-LLM ensemble reaches kappa=0.432 (moderate) with a conservative bias. Under validated evaluation, a parameter-level injection propagates to a wrong final answer with human-calibrated probability approximately 0.62 (range 0.46-0.73 across models). Rejection (catching bad parameters) and recovery (correcting after acceptance) are independent model capabilities (Spearman rho=0.126, p=0.747). A tuned runtime interceptor reduces hallucination on GPT-4o-mini by 23.0 percentage points under a concurrent n=600 control, but shows no significant effect on Gemini-2.0-Flash, whose aggressive parameter rejection eliminates the target failure mode. All code, data, traces, and human labels are released at this https URL. 

---
# On the Robustness of LLM-Based Dense Retrievers: A Systematic Analysis of Generalizability and Stability 

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Yixing Fan, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2604.16576)  

**Abstract**: Decoder-only large language models (LLMs) are increasingly replacing BERT-style architectures as the backbone for dense retrieval, achieving substantial performance gains and broad adoption. However, the robustness of these LLM-based retrievers remains underexplored. In this paper, we present the first systematic study of the robustness of state-of-the-art open-source LLM-based dense retrievers from two complementary perspectives: generalizability and stability. For generalizability, we evaluate retrieval effectiveness across four benchmarks spanning 30 datasets, using linear mixed-effects models to estimate marginal mean performance and disentangle intrinsic model capability from dataset heterogeneity. Our analysis reveals that while instruction-tuned models generally excel, those optimized for complex reasoning often suffer a ``specialization tax,'' exhibiting limited generalizability in broader contexts. For stability, we assess model resilience against both unintentional query variations~(e.g., paraphrasing, typos) and malicious adversarial attacks~(e.g., corpus poisoning). We find that LLM-based retrievers show improved robustness against typos and corpus poisoning compared to encoder-only baselines, yet remain vulnerable to semantic perturbations like synonymizing. Further analysis shows that embedding geometry (e.g., angular uniformity) provides predictive signals for lexical stability and suggests that scaling model size generally improves robustness. These findings inform future robustness-aware retriever design and principled benchmarking. Our code is publicly available at this https URL. 

---
# S-GRPO: Unified Post-Training for Large Vision-Language Models 

**Authors**: Yuming Yan, Kai Tang, Sihong Chen, Ke Xu, Dan Hu, Qun Yu, Pengfei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16557)  

**Abstract**: Current post-training methodologies for adapting Large Vision-Language Models (LVLMs) generally fall into two paradigms: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). Despite their prevalence, both approaches suffer from inefficiencies when applied in isolation. SFT forces the model's generation along a single expert trajectory, often inducing catastrophic forgetting of general multimodal capabilities due to distributional shifts. Conversely, RL explores multiple generated trajectories but frequently encounters optimization collapse - a cold-start problem where an unaligned model fails to spontaneously sample any domain-valid trajectories in sparse-reward visual tasks. In this paper, we propose Supervised Group Relative Policy Optimization (S-GRPO), a unified post-training framework that integrates the guidance of imitation learning into the multi-trajectory exploration of preference optimization. Tailored for direct-generation visual tasks, S-GRPO introduces Conditional Ground-Truth Trajectory Injection (CGI). When a binary verifier detects a complete exploratory failure within a sampled group of trajectories, CGI injects the verified ground-truth trajectory into the candidate pool. By assigning a deterministic maximal reward to this injected anchor, S-GRPO enforces a positive signal within the group-relative advantage estimation. This mechanism reformulates the supervised learning objective as a high-advantage component of the policy gradient, compelling the model to dynamically balance between exploiting the expert trajectory and exploring novel visual concepts. Theoretical analysis and empirical results demonstrate that S-GRPO gracefully bridges the gap between SFT and RL, drastically accelerates convergence, and achieves superior domain adaptation while preserving the base model's general-purpose capabilities. 

---
# A Survey on the Security of Long-Term Memory in LLM Agents: Toward Mnemonic Sovereignty 

**Authors**: Zehao Lin, Chunyu Li, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.16548)  

**Abstract**: Research on large language model (LLM) security is shifting from "will the model leak training data" to a more consequential question: can an agent with persistent, long-term memory be continuously shaped, cross-session poisoned, accessed without authorization, and propagated across shared organizational state? Recent surveys cover memory architectures and agent mechanisms, but fewer center the epistemic and governance properties of persistent, writable memory as the reason memory is an independent security problem.
This survey addresses that gap. Drawing on cognitive neuroscience and the philosophy of memory, we characterize agent memory as malleable, rewritable, and socially propagating, and develop a memory-lifecycle framework organized around six phases -- Write, Store, Retrieve, Execute, Share, Forget/Rollback -- cross-tabulated against four security objectives: integrity, confidentiality, availability, governance. We organize the literature on memory poisoning, extraction, retrieval corruption, control-flow hijacking, cross-agent propagation, rollback, and governance, and situate representative architectures as determinants of which phases are explicitly governable.
Three findings stand out: the literature concentrates on write- and retrieve-time integrity attacks, while confidentiality, availability, store/forget, and benign-persistence failures remain sparsely studied; no published architecture covers all nine governance primitives we identify; and using LLMs themselves for memory security remains sparse yet essential.
We unify these under mnemonic sovereignty -- verifiable, recoverable governance over what may be written, who may read, when updates are authorized, and which states may be forgotten -- arguing future secure agents will be differentiated not only by recall capacity, but by memory governance quality. 

---
# TWGuard: A Case Study of LLM Safety Guardrails for Localized Linguistic Contexts 

**Authors**: Hua-Rong Chu, Kuan-Chun Wang, Yao-Te Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16542)  

**Abstract**: Safety guardrails have become an active area of research in AI safety, aimed at ensuring the appropriate behavior of large language models (LLMs). However, existing research lacks consideration of nuances across linguistic and cultural contexts, resulting in a gap between reported performance and in-the-wild effectiveness. To address this issue, this paper proposes an approach to optimize guardrail models for a designated linguistic context by leveraging a curated dataset tailored to local linguistic characteristics, targeting the Taiwan linguistic context as a representative example of localized deployment challenges. The proposed approach yields TWGuard, a linguistic context-optimized guardrail model that achieves a huge gain (+0.289 in F1) compared to the foundation model and significantly outperforms the strongest baseline in practical use (-0.037 in false positive rate, a 94.9\% reduction). Together, this work lays a foundation for regional communities to establish AI safety standards grounded in their own linguistic contexts, rather than accepting boundaries imposed by dominant languages. The inadequacy of the latter is reconfirmed by our findings. 

---
# Scaling Test-Time Compute for Agentic Coding 

**Authors**: Joongwon Kim, Wannan Yang, Kelvin Niu, Hongming Zhang, Yun Zhu, Eryk Helenowski, Ruan Silva, Zhengxing Chen, Srinivasan Iyer, Manzil Zaheer, Daniel Fried, Hannaneh Hajishirzi, Sanjeev Arora, Gabriel Synnaeve, Ruslan Salakhutdinov, Anirudh Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2604.16529)  

**Abstract**: Test-time scaling has become a powerful way to improve large language models. However, existing methods are best suited to short, bounded outputs that can be directly compared, ranked or refined. Long-horizon coding agents violate this premise: each attempt produces an extended trajectory of actions, observations, errors, and partial progress taken by the agent. In this setting, the main challenge is no longer generating more attempts, but representing prior experience in a form that can be effectively selected from and reused. We propose a test-time scaling framework for agentic coding based on compact representations of rollout trajectories. Our framework converts each rollout into a structured summary that preserves its salient hypotheses, progress, and failure modes while discarding low-signal trace details. This representation enables two complementary forms of inference-time scaling. For parallel scaling, we introduce Recursive Tournament Voting (RTV), which recursively narrows a population of rollout summaries through small-group comparisons. For sequential scaling, we adapt Parallel-Distill-Refine (PDR) to the agentic setting by conditioning new rollouts on summaries distilled from prior attempts. Our method consistently improves the performance of frontier coding agents across SWE-Bench Verified and Terminal-Bench v2.0. For example, by using our method Claude-4.5-Opus improves from 70.9% to 77.6% on SWE-Bench Verified (mini-SWE-agent) and 46.9% to 59.1% on Terminal-Bench v2.0 (Terminus 1). Our results suggest that test-time scaling for long-horizon agents is fundamentally a problem of representation, selection, and reuse. 

---
# SmoGVLM: A Small, Graph-enhanced Vision-Language Model 

**Authors**: Debjyoti Mondal, Rituraj Singh, Subhadarshi Panda  

**Link**: [PDF](https://arxiv.org/pdf/2604.16517)  

**Abstract**: Large vision-language models (VLMs) achieve strong performance on multimodal tasks but often suffer from hallucination and poor grounding in knowledge-intensive reasoning. We propose SmoGVLM, a small, graph-enhanced VLM that integrates structured knowledge with visual and textual modalities, using Graph Neural Networks. We investigate the effects of our method across a range of model sizes, from tiny (1.3B) to large (13B) models. Our results demonstrate that, when trained using our approach, a small model can achieve performance gains upto 16.24%, and surpass its larger counterparts, outperforming larger VLMs and strong fine-tuned baselines. These results highlight the potential of structured knowledge augmentation for efficient, smaller-scale multimodal reasoning systems. 

---
# SQL Query Engine: A Self-Healing LLM Pipeline for Natural Language to PostgreSQL Translation 

**Authors**: Muhammad Adeel Ijaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.16511)  

**Abstract**: We present SQL Query Engine, an open-source, self-hosted service that translates natural language questions into validated PostgreSQL queries through a two-stage LLM pipeline. The first stage performs automatic schema introspection and SQL generation; a multi-strategy response parser extracts SQL from any LLM output format (JSON, code blocks, or raw text) without requiring structured output APIs. The second stage executes the query against PostgreSQL and, upon failure or empty results, enters an iterative self-healing loop in which the LLM diagnoses the error using full SQLSTATE codes and PostgreSQL diagnostic messages. Two mechanisms prevent regressions: early-accept returns successful queries immediately without LLM re-evaluation, and best-result tracking preserves the best partial result across retries. Schema context is cached per session in Redis, progress events stream via Redis Pub/Sub and SSE, and an OpenAI-compatible /v1/chat/completions endpoint lets existing tools work without modification. All database connections are read-only at the driver level. We evaluate across five LLM backends on a synthetic benchmark (75 questions, three databases) where the self-healing loop yields up to +9.3pp accuracy gains with zero regressions on the best model (Llama 4 Scout 17B, 57.3%), and on BIRD (437 questions, 11 databases migrated from SQLite to PostgreSQL) where the full pipeline reaches 49.0% execution accuracy (GPT-OSS-120B, +4.6pp). Source code: this https URL. 

---
# Medical thinking with multiple images 

**Authors**: Zonghai Yao, Benlu Wang, Yifan Zhang, Junda Wang, Iris Xia, Zhipeng Tang, Shuo Han, Feiyun Ouyang, Zhichao Yang, Arman Cohan, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16506)  

**Abstract**: Large language models perform well on many medical QA benchmarks, but real clinical reasoning often requires integrating evidence across multiple images rather than interpreting a single view. We introduce MedThinkVQA, an expert-annotated benchmark for thinking with multiple images, where models must interpret each image, combine cross-view evidence, and answer diagnostic questions with intermediate supervision and step-level evaluation. The dataset contains 8,067 cases, including 720 test cases, with an average of 6.62 images per case, substantially denser than prior work, whose expert-level benchmarks use at most 1.43 images per case. On the test set, the best closed-source models, Claude-4.6-Opus, Gemini-3-Pro, and GPT-5.2-xhigh, reach only 57.2%, 55.3%, and 54.9% accuracy, while GPT-5-mini and GPT-5-nano reach 39.7% and 30.8%. Strong open-source models lag behind, led by Qwen3.5-397B-A17B at 52.2% and Qwen3.5-27B at 50.6%. Further analysis identifies grounded multi-image reasoning as the main bottleneck: models often fail to extract, align, and compose evidence across views before higher-level inference can help. Providing expert single-image cues and cross-image summaries improves performance, whereas replacing them with self-generated intermediates reduces accuracy. Step-level analysis shows that over 70% of errors arise from image reading and cross-view integration. Scaling results further show that additional inference-time computation helps only when visual grounding is already reliable; when early evidence extraction is weak, longer reasoning yields limited or unstable gains and can amplify misread cues. These results suggest that the key challenge is not reasoning length alone, but reliable mechanisms for grounding, aligning, and composing distributed evidence across real-world multimodal clinical inputs. 

---
# NL2SQLBench: A Modular Benchmarking Framework for LLM-Enabled NL2SQL Solutions 

**Authors**: Shizheng Hou, Wenqi Pei, Nuo Chen, Quang-Trung Ta, Peng Lu, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16493)  

**Abstract**: Natural Language to SQL (NL2SQL) technology empowers non-expert users to query relational databases without requiring SQL expertise. While large language models (LLMs) have greatly improved NL2SQL algorithms, their rapid development outpaces systematic evaluation, leaving a critical gap in understanding their effectiveness, efficiency, and limitations. To this end, we present NL2SQLBench, the first modular evaluation and benchmarking framework for LLM-enabled NL2SQL approaches. Specifically, we dissect NL2SQL systems into three core modules: Schema Selection, Candidate Generation, and Query Revision. For each module, we comprehensively review existing strategies and propose novel fine-grained metrics that systematically quantify module-level effectiveness and efficiency. We further implement these metrics in a flexible multi-agent framework, allowing configurable benchmarking across diverse NL2SQL approaches. Leveraging NL2SQLBench, we rigorously evaluate ten representative open-source methods on two datasets, the BIRD development set and the ScienceBenchmark development set, using two LLMs, DeepSeek-V3 and GPT-4o mini. We systematically assess each approach across the three core modules and evaluate multiple critical performance dimensions. Our evaluation reveals significant gaps in existing NL2SQL methods, highlighting not only substantial room for accuracy improvements but also the significant computational inefficiency, which severely hampers real-world adoption. Furthermore, our analysis identifies critical shortcomings in current benchmark datasets and evaluation rules, emphasizing issues such as inaccurate gold SQL annotations and limitations in existing evaluation rules. By synthesizing these insights into a unified benchmarking, our study establishes a clear reference point for fair comparison and serves as essential guidance for future targeted innovation in NL2SQL technology. 

---
# iPhoneme: Brain-to-Text Communication for ALS Using ConformerXL Decoding 

**Authors**: Yoonmin Cha, Dawit Chun, Sung Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.16441)  

**Abstract**: Brain-computer interfaces (BCIs) for speech restoration hold transformative potential for the approximately 173,000--232,500 individuals worldwide with ALS-related dysarthria. Despite recent progress, high-performance speech BCIs have been demonstrated in only 22--31 patients globally, largely due to limitations in neural decoding accuracy and practical input interfaces. We present iPhoneme, a brain-to-text communication system that jointly addresses these challenges through integrated modeling and interaction design. The system combines a deep learning phoneme decoder based on a modified Conformer architecture (ConformerXL, 192.9M parameters) with a gaze-assisted phoneme input interface that mitigates the Midas touch problem in eye-tracking systems. The acoustic model incorporates a temporal prenet with multi-scale dilated convolutions and bidirectional GRU for neural jitter correction, temporal subsampling for CTC stability, and Pre-RMSNorm stabilization across 12 encoder blocks, trained with AdamW and cosine scheduling. On the interaction side, iPhoneme introduces a chorded gaze-plus-silent-speech paradigm that replaces dwell-time selection, enabling more efficient input. We evaluate the system on the T15 dataset (45 sessions, 8,071 trials) of 256-channel intracranial EEG from speech motor cortex regions. A 6-gram phoneme language model trained on 3.1M sequences, combined with WFST beam search (beam=128), achieves 92.14% phoneme accuracy (7.86% PER) and 73.39% word accuracy (26.61% WER), approximately 3% above prior state-of-the-art. The system operates on CPU with 180 ms latency, demonstrating real-time, high-accuracy brain-to-text communication for ALS. 

---
# Safety, Security, and Cognitive Risks in State-Space Models: A Systematic Threat Analysis with Spectral, Stateful, and Capacity Attacks 

**Authors**: Manoj Parmar  

**Link**: [PDF](https://arxiv.org/pdf/2604.16424)  

**Abstract**: State-Space Models (SSMs) -- structured SSMs (S4, S4D, DSS, S5), selective SSMs (Mamba, Mamba-2), and hybrid architectures (Jamba) -- are deployed in safety-critical long-context applications: genomic analysis, clinical time-series forecasting, and cybersecurity log processing. Their linear-time scaling is compelling, yet the security properties of their compressed-state recurrent architectures remain unstudied.
We present the first systematic treatment of SSM safety, security, and cognitive risks. Seven contributions: (1) Formal threat framework -- SSM Attack Surface (five layers), State Integrity Violation (StIV), Cross-Context Amplification Ratio $\mathcal{X}_\mathcal{S}$, and a Spectral Sensitivity Proposition grounded in the $H_\infty$ norm. (2) Three novel attack classes: spectral adversarial attacks (transfer-function gain exploitation), delayed-trigger stateful backdoors (activate thousands of steps after injection), and state capacity saturation (entropy flooding forces silent forgetting). (3) 14 MITRE ATLAS technique extensions across the full tactic chain. (4) Six-profile attacker taxonomy with kill chains for genomics, clinical, and cybersecurity domains. (5) Four cognitive risk hypotheses grounded in state-compression mechanics. (6) Governance-aligned mitigations mapped to CREST, NIST AI 600-1, and EU AI Act. (7) Empirical evaluation: targeted genomic injection achieves $\mathrm{StIV}=0.519$ vs. $0.086$ random ($6.0\times$, $p<0.001$); PGD state injection achieves $156\times$ output perturbation over random; SSD-structured extraction confirmed at $O(N^2)$ vs. $O(N^3)$ query complexity ($N\times$ speedup). Validation on pretrained checkpoints is detailed in the Appendix. 

---
# Measuring the Gap Between Media Coverage and Public Information Demand: Evidence from the 2026 Lebanon Conflict 

**Authors**: Mohamed Soufan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16417)  

**Abstract**: This study examines the relationship between media coverage and public information demand during the Lebanon conflict in March 2026. Using a dataset of 11,623 English-language news articles collected from the GDELT database and Google Trends data for searches conducted within Lebanon, the study compares the distribution of news coverage across topics with the distribution of public search interest. News headlines were filtered for relevance and classified into four categories: Conflict, Economy, Living Conditions, and Emigration. Public information demand was measured using Google Trends topic data for the same categories. The results show a substantial divergence between news coverage and search interest. Conflict accounted for 94.9% of classified news coverage but only 36.9% of total search interest. In contrast, Economy, Living Conditions, and Emigration together accounted for 63.1% of search demand but only 5.1% of news coverage. Time series analysis indicates that search demand for economic and living conditions remained consistently elevated throughout the month rather than reacting to specific conflict events. These findings were robust to the exclusion of the peak conflict period (March 1-5), with Conflict coverage remaining at 94.9% and the information gap persisting across all three under-covered categories. The findings suggest that during the study period, media coverage of Lebanon was heavily concentrated on military events, while public information demand was distributed across economic conditions, daily life, and emigration. This study contributes to agenda-setting research by providing a quantitative comparison between media agenda and public information demand during an active conflict period. 

---
# RoMathExam: A Longitudinal Dataset of Romanian Math Exams (1895-2025) with a Seven-Decade Core (1957-2025) 

**Authors**: Luca-Ncolae Cuclea, Sabin-Codrut Badea, Adrian-Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2604.16392)  

**Abstract**: AI in Education research increasingly relies on authentic, curriculum-grounded assessment data, yet large, well-structured exam corpora remain scarce for many languages and educational systems. We introduce RoMathExam, a longitudinal dataset of Romanian high-school mathematics exams spanning 1895-2025, with a robust standardized core for 1957-2025. The dataset contains 10,592 mathematics problems organized into 600+ complete exam sets across multiple tracks (M1-M4), covering both official national examination sessions and ministry-published training variants. Beyond high-fidelity digitization and a unified JSON schema with traceable provenance, RoMathExam is enriched with curriculum-aligned topic tags and dense text embeddings, enabling variant detection, deduplication, and similarity-based retrieval. To overcome the lack of historical psychometric data, we propose and validate a solution complexity metric as a scalable intrinsic proxy for difficulty. Our evaluation across three frontier reasoning models (GPT-5-mini, DeepSeek-R1, and Qwen3-235B-Thinking) reveals high cross-model synchronization (r > 0.72), confirming the metric's ability to isolate intrinsic mathematical depth from stochastic generation noise. We demonstrate the dataset's utility through a longitudinal analysis that quantifies a "regime shift" from volatile historical formats to a standardized, algebra-dominant modern curriculum. RoMathExam provides a foundation for reproducible research in difficulty modeling, curriculum analytics, and LLM evaluation in low-resource linguistic contexts. 

---
# LLMAR: A Tuning-Free Recommendation Framework for Sparse and Text-Rich Industrial Domains 

**Authors**: Ryogo Hishikawa, Ichiro Kataoka, Shinya Yuda  

**Link**: [PDF](https://arxiv.org/pdf/2604.16379)  

**Abstract**: Industrial B2B applications (e.g., construction site risk prediction, material procurement) face extreme data sparsity yet feature rich textual interactions. In such environments, traditional ID-based collaborative filtering fails lacking co-occurrence signals, while fine-tuning standard Large Language Models (LLMs) incurs high operational costs and struggles with frequent data drift.
We propose LLMAR (LLM-Annotated Recommendation), a tuning-free framework. Moving beyond simple embeddings, LLMAR systematically integrates LLM reasoning to capture user "latent motives" without any training process. We introduce three core contributions: (1) Inference-Driven Annotation: uses LLMs to transform behavioral history into structured semantic motives, enabling reasoning-based matching unattainable by ID-based methods; (2) Reflection Loop: a self-correction mechanism that refines generated queries to mitigate hallucinations and resolve "context competition" between past history and current instructions; and (3) Cost-Effective Architecture: relies on tuning-free components and asynchronous batch processing to minimize maintenance costs.
Evaluations on public benchmarks (MovieLens-1M, Amazon Prime Pantry) and a sparse industrial dataset (construction risk prediction) demonstrate that LLMAR outperforms state-of-the-art learning-based models (SASRecF), achieving up to a 54.6% nDCG@10 improvement on the industrial dataset. Inference costs remain highly practical (~$1 per 1,000 users). For B2B domains where strict real-time latency is not critical, combining LLM reasoning with self-verification offers a superior alternative to training-based approaches across accuracy, explainability, and operational cost. 

---
# Why AI Readiness Is an Organizational Learning Problem, Not a Technology Purchase 

**Authors**: Jeanne McClure, Gregg Gerdau  

**Link**: [PDF](https://arxiv.org/pdf/2604.16369)  

**Abstract**: Global corporate AI investment reached $252.3 billion in 2024, yet only 6% of firms report significant earnings impact. This article argues that AI project failure is fundamentally an organizational learning problem rather than a technology deficit. Drawing on a systematic synthesis of 19 large-scale industry and academic sources, including surveys of nearly 10,000 organizational leaders, we identify two categories of failure: organizational (culture, leadership alignment, governance, and human-AI learning deficits) and technical (semantic bottlenecks and output management challenges). We introduce the Siloed-Integrated-Orchestrated (SIO) progression model, which maps enterprise AI capability across five pillars -- Culture & Leadership, Human Capital & Operations, Data Architecture, Systems Infrastructure, and Governance & Regulatory Compliance -- and provides prescriptive guidance for advancing between stages. The implications challenge organizations to reframe AI investment as capability development rather than technology procurement. 

---
# Clinical Note Bloat Reduction for Efficient LLM Use 

**Authors**: Jordan L. Cahoon, Chloe Stanwyck, Asad Aali, Rachel Madding, Emma Sun, Yixing Jiang, Renumathy Dhanasekaran, Emily Alsentzer  

**Link**: [PDF](https://arxiv.org/pdf/2604.16364)  

**Abstract**: Health systems are rapidly deploying large language models (LLMs) that use clinical notes for clinical decision support applications. However, modern documentation practices rely heavily on templates, copy--paste shortcuts, and auto-populated fields, producing extensive duplicated text (``note bloat'') that dilutes clinically meaningful signal and substantially increases the computational cost of LLM use. We introduce TRACE, a scalable preprocessing pipeline that removes note bloat by leveraging EHR attribution metadata to identify templated and copied content and applying frequency-based deduplication when metadata are unavailable. We evaluated TRACE across four real--world clinical cohorts spanning liver transplantation, obstetrics, and inpatient care (5.3 million notes) using blinded physician review and downstream modeling tasks. TRACE removed 47.3% of chart text while preserving performance for information extraction and clinical outcome prediction. At a large academic medical center, this reduction corresponds to an estimated $9.5 million annual decrease in LLM inference costs assuming one query per encounter. These findings show how underutilized EHR metadata can enable more scalable and cost-efficient deployment of LLM-based clinical systems. 

---
# SaFeR-Steer: Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics 

**Authors**: Haolong Hu, Hanyu Li, Tiancheng He, Huahui Yi, An Zhang, Qiankun Li, Kun Wang, Yang Liu, Zhigang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2604.16358)  

**Abstract**: MLLMs are increasingly deployed in multi-turn settings, where attackers can escalate unsafe intent through the evolving visual-text history and exploit long-context safety decay. Yet safety alignment is still dominated by single-turn data and fixed-template dialogues, leaving a mismatch between training and this http URL bridge this gap, we propose SaFeR-Steer, a progressive multi-turn alignment framework that combines staged synthetic bootstrapping with tutor-in-the-loop GRPO to train a single student under adaptive, on-policy attacks. We also introduce TCSR, which uses trajectory minimum/average safety to propagate late-turn failures to earlier turns.I. Dataset. We release STEER, a multi-turn multimodal safety dataset with STEER-SFT (12,934), STEER-RL (2,000), and STEER-Bench (3,227) dialogues spanning 2~10 this http URL. Experiment. Starting from Qwen2.5-VL-3B/7B, SaFeR-Steer substantially improves Safety/Helpfulness on both single-turn (48.30/45.86 -> 81.84/70.77 for 3B; 56.21/60.32 -> 87.89/77.40 for 7B) and multi-turn benchmarks (12.55/27.13 -> 55.58/70.27 for 3B; 24.66/46.48 -> 64.89/72.35 for 7B), shifting failures to later turns and yielding robustness beyond scaling this http URL are available at this https URL 

---
# Training for Compositional Sensitivity Reduces Dense Retrieval Generalization 

**Authors**: Radoslav Ralev, Aditeya Baral, Iliya Zhechev, Jen Agarwal, Srijith Rajamohan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16351)  

**Abstract**: Dense retrieval compresses texts into single embeddings ranked by cosine similarity. While efficient for recall, this interface is brittle for identity-level matching: minimal compositional edits (negation, role swaps) flip meaning yet retain high similarity. Motivated by geometric results for unit-sphere cosine spaces (Kang et al., 2025), we test this retrieval-composition tension in text-only retrieval. Across four dual-encoder backbones, adding structure-targeted negatives consistently reduces zero-shot NanoBEIR retrieval (8-9% mean nDCG@10 drop on small backbones; up to 40% on medium ones), while only partially improving pooled-space separation. Treating pooled cosine as a recall interface, we then benchmark verifiers scoring token--token cosine maps. MaxSim (late interaction) excels at reranking but fails to reject structural near-misses, whereas a small Transformer over similarity maps reliably separates near-misses under end-to-end training. 

---
# Benchmarking Real-Time Question Answering via Executable Code Workflows 

**Authors**: Wenjie Zhou, Yuan Gao, Xin Zhou, Hao Fu, Zhongjian Miao, Wei Chen, Bo Chen, Xiaobing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16349)  

**Abstract**: Retrieving real-time information is a fundamental capability for search-integrated agents in real-world applications. However, existing benchmarks are predominantly static and therefore fail to capture the temporal dynamics of information and the continuously evolving nature of real-world knowledge. To address this limitation, we propose RT-QA, a dynamic evaluation framework that leverages executable code workflows to retrieve up-to-date answers at evaluation time. Specifically, we construct an agent-driven pipeline that autonomously generates code for web crawling and DOM-based answer extraction to produce real-time ground truth. To ensure robust evaluation over time, the pipeline further incorporates a self-repair mechanism to adapt to changes in web page structures. RT-QA spans 12 domains (e.g., Finance, Sports) with 320 Chinese questions categorized into three difficulty levels. Extensive evaluations of state-of-the-art models (e.g., GPT-5.2, GLM-4.7) reveal significant limitations in real-time adaptability: even the best models achieve only 46% accuracy. Our analysis highlights two primary failure modes: (1) Lazy Retrieval, where agents rely on search snippets instead of deeply scanning specific websites for information (20% of failures); and (2) Temporal Confusion, a cognitive error where agents retrieve a historical date (e.g., an event in 2024) and fail to re-anchor to the current time (2026) for subsequent reasoning. These findings suggest that future agents require not just better retrieval strategies, but robust temporal state management. 

---
# Annotation Entropy Predicts Per-Example Learning Dynamics in LoRA Fine-Tuning 

**Authors**: Brady Steele  

**Link**: [PDF](https://arxiv.org/pdf/2604.16332)  

**Abstract**: We find that LoRA fine-tuning exhibits un-learning on contested examples: items with high annotator disagreement show increasing loss during training, a qualitatively distinct pattern largely absent under full fine-tuning and consistent across all six models tested (four encoder, two decoder-only). This discovery emerges from correlating annotation entropy, computed from ChaosNLI's 100 labels per example, with per-example area under the loss curve (AULC) on SNLI and MNLI. The correlation is positive in all 25 conditions tested (Spearman $\rho = 0.06$-$0.43$), with decoder-only models showing stronger correlations than encoders at matched LoRA rank. The effect survives partial-correlation controls and replicates across seeds and datasets. A preliminary noise-injection experiment is consistent with these findings. 

---
# Diagnosing LLM-based Rerankers in Cold-Start Recommender Systems: Coverage, Exposure and Practical Mitigations 

**Authors**: Ekaterina Lemdiasova, Nikita Zmanovskii  

**Link**: [PDF](https://arxiv.org/pdf/2604.16318)  

**Abstract**: Large language models (LLMs) and cross-encoder rerankers have gained attention for improving recommender systems, particularly in cold-start scenarios where user interaction history is limited. However, practical deployment reveals significant performance gaps between LLM-based approaches and simple baselines. This paper presents a systematic diagnostic study of cross-encoder rerankers in cold-start movie recommendation using the Serendipity-2018 dataset. Through controlled experiments with 500 users across multiple random seeds, we identify three critical failure modes: (1) low retrieval coverage in candidate generation (recall@200 = 0.109 vs. 0.609 for baselines), (2) severe exposure bias with rerankers concentrating recommendations on 3 unique items versus 497 for random baseline, and (3) minimal score discrimination between relevant and irrelevant items (mean difference = 0.098, Cohen's d = 0.13). We demonstrate that popularity-based ranking substantially outperforms LLM reranking (HR@10: 0.268 vs. 0.008, p < 0.001), with the performance gap primarily attributable to retrieval stage limitations rather than reranker capacity. Based on these findings, we provide actionable recommendations including hybrid retrieval strategies, candidate pool size optimization, and score calibration techniques. All code, configurations, and experimental results are made available for reproducibility. 

---
# MARA: A Multimodal Adaptive Retrieval-Augmented Framework for Document Question Answering 

**Authors**: Hui Wu, Haoquan Zhai, Yuchen Li, Hengyi Cai, Peirong Zhang, Yidan Zhang, Lei Wang, Chunle Wang, Yingyan Hou, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16313)  

**Abstract**: Retrieval-based multimodal document QA aims to identify and integrate relevant information from visually rich documents with complex multimodal structures. While retrieval-augmented generation (RAG) has shown strong performance in text-based QA, its extensions to multimodal documents remain underexplored and face significant limitations. Specifically, current approaches rely on query-agnostic document representations that overlook salient content and use static top-k evidence selection, which fails to adapt to the uncertain distribution of relevant information. To address these limitations, we propose the Multimodal Adaptive Retrieval-Augmented (MARA) framework, which introduces query-adaptive mechanisms to both retrieval and generation. MARA consists of two components: a Query-Aligned Region Encoder that builds multi-level document representations and reweights them based on query relevance to improve retrieval precision; and a Self-Reflective Evidence Controller that monitors evidence sufficiency during generation and adaptively incorporates content from lower-ranked sources using a sliding-window strategy. Experiments on six multimodal QA benchmarks demonstrate that MARA consistently improves retrieval relevance and answer quality over existing SOTA method. 

---
# RAG-DIVE: A Dynamic Approach for Multi-Turn Dialogue Evaluation in Retrieval-Augmented Generation 

**Authors**: Lorenz Brehme, Benedikt Dornauer, Jan-Henrik Böttcher, Klaus Schmid, Mircea-Cristian Racasan, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16310)  

**Abstract**: Evaluating Retrieval-Augmented Generation (RAG) systems using static multi-turn datasets fails to capture the dynamic nature of real-world dialogues. Existing evaluation methods rely on predefined datasets, which restrict them to static, one-directional queries and limit their ability to capture the adaptive, context-dependent performance of RAG systems in interactive, multi-turn settings. Thus, we introduce the RAG-DIVE, a Dynamic Interactive Validation and Evaluation approach, that simulates user interactions with RAG systems. RAG-DIVE leverages an LLM to generate multi-turn conversations dynamically and is organized into three components. The dialogue generation stage consists of the (1) Conversation Generator, which simulates a user by creating multi-turn queries, and the (2) Conversation Validator, which filters and corrects invalid or low-quality outputs to ensure coherent conversations. The evaluation stage is handled by the (3) Conversation Evaluator, which assesses the RAG system's performance across the entire dialogue and generates both per-turn and multi-turn metrics that provide an aggregated view of system behavior. We validated RAG-DIVE through two experimental setups. First, we tested a sample RAG system, including human evaluation of dialogue quality, repeated trials to assess consistency, and an ablation study showing that RAG-DIVE detects performance changes caused by system modifications. Second, we compared RAG-DIVE with a traditional static dataset evaluation on an industrial RAG system under different configurations to verify whether both approaches reveal similar performance trends. Our findings demonstrate that RAG-DIVE facilitates dynamic, interaction-driven evaluation for multi-turn conversations, thereby advancing the assessment of RAG systems. 

---
