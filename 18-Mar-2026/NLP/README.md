# Chronos: Temporal-Aware Conversational Agents with Structured Event Retrieval for Long-Term Memory 

**Authors**: Sahil Sen, Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2603.16862)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled conversational AI agents to engage in extended multi-turn interactions spanning weeks or months. However, existing memory systems struggle to reason over temporally grounded facts and preferences that evolve across months of interaction and lack effective retrieval strategies for multi-hop, time-sensitive queries over long dialogue histories. We introduce Chronos, a novel temporal-aware memory framework that decomposes raw dialogue into subject-verb-object event tuples with resolved datetime ranges and entity aliases, indexing them in a structured event calendar alongside a turn calendar that preserves full conversational context. At query time, Chronos applies dynamic prompting to generate tailored retrieval guidance for each question, directing the agent on what to retrieve, how to filter across time ranges, and how to approach multi-hop reasoning through an iterative tool-calling loop over both calendars. We evaluate Chronos with 8 LLMs, both open-source and closed-source, on the LongMemEvalS benchmark comprising 500 questions spanning six categories of dialogue history tasks. Chronos Low achieves 92.60% and Chronos High scores 95.60% accuracy, setting a new state of the art with an improvement of 7.67% over the best prior system. Ablation results reveal the events calendar accounts for a 58.9% gain on the baseline while all other components yield improvements between 15.5% and 22.3%. Notably, Chronos Low alone surpasses prior approaches evaluated under their strongest model configurations. 

---
# Online Experiential Learning for Language Models 

**Authors**: Tianzhu Ye, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.16856)  

**Abstract**: The prevailing paradigm for improving large language models relies on offline training with human annotations or simulated environments, leaving the rich experience accumulated during real-world deployment entirely unexploited. We propose Online Experiential Learning (OEL), a framework that enables language models to continuously improve from their own deployment experience. OEL operates in two stages: first, transferable experiential knowledge is extracted and accumulated from interaction trajectories collected on the user side; second, this knowledge is consolidated into model parameters via on-policy context distillation, requiring no access to the user-side environment. The two stages are iterated to form an online learning loop, where the improved model collects higher-quality trajectories that yield richer experiential knowledge for subsequent rounds. We evaluate OEL on text-based game environments across multiple model scales and both thinking and non-thinking variants. OEL achieves consistent improvements over successive iterations, enhancing both task accuracy and token efficiency while preserving out-of-distribution performance. Our analysis further shows that extracted experiential knowledge is significantly more effective than raw trajectories, and that on-policy consistency between the knowledge source and the policy model is critical for effective learning. 

---
# Mediocrity is the key for LLM as a Judge Anchor Selection 

**Authors**: Shachar Don-Yehiya, Asaf Yehudai, Leshem Choshen, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2603.16848)  

**Abstract**: The ``LLM-as-a-judge'' paradigm has become a standard method for evaluating open-ended generation. To address the quadratic scalability costs of pairwise comparisons, popular benchmarks like Arena-Hard and AlpacaEval compare all models against a single anchor. However, despite its widespread use, the impact of anchor selection on the reliability of the results remains largely unexplored. In this work, we systematically investigate the effect of anchor selection by evaluating 22 different anchors on the Arena-Hard-v2.0 dataset. We find that the choice of anchor is critical: a poor anchor can dramatically reduce correlation with human rankings. We identify that common anchor choices (best-performing and worst-performing models) make poor anchors. Because these extreme anchors are consistently better or worse than all other models, they are seldom indicative of the relative ranking of the models. We further quantify the effect size of anchor selection, showing it is comparable to the selection of a judge model. We conclude with actionable recommendations. First, we conduct a power analysis, and compute sufficient benchmark sizes for anchor-based evaluation, finding that standard benchmark sizes are insufficient for pairwise evaluation and fail to distinguish between competitive models reliably. Second, we provide guidelines for selecting informative anchors to ensure reliable and efficient evaluation practices. 

---
# SpokenUS: A Spoken User Simulator for Task-Oriented Dialogue 

**Authors**: Jonggeun Lee, Junseong Pyo, Jeongmin Park, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2603.16783)  

**Abstract**: Robust task-oriented spoken dialogue agents require exposure to the full diversity of how people interact through speech. Building spoken user simulators that address this requires large-scale spoken task-oriented dialogue (TOD) data encompassing spoken user behaviors, yet existing datasets are limited in scale and domain coverage, with no systematic pipeline for augmenting them. To address this, we introduce \textbf{SpokenTOD}, a spoken TOD dataset of 52,390 dialogues and 1,034 hours of speech augmented with four spoken user behaviors -- cross-turn slots, barge-in, disfluency, and emotional prosody -- across diverse speakers and domains. Building on SpokenTOD, we present \textbf{SpokenUS}, a spoken user simulator grounded in TOD with a dedicated architecture for barge-in. SpokenUS achieves comparable goal coverage to significantly larger models while substantially outperforming all baselines in Human MOS, disclosing slot values gradually across the dialogue as humans do rather than front-loading them. Further analysis confirms that SpokenUS's spoken behaviors pose meaningful challenges to downstream agents, making it a practical tool for training and evaluating more robust spoken dialogue systems. 

---
# TurnWise: The Gap between Single- and Multi-turn Language Model Capabilities 

**Authors**: Victoria Graf, Valentina Pyatkin, Nouha Dziri, Nathan Lambert, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16759)  

**Abstract**: Multi-turn conversations are a common and critical mode of language model interaction. However, current open training and evaluation data focus on single-turn settings, failing to capture the additional dimension of these longer interactions. To understand this multi-/single-turn gap, we first introduce a new benchmark, TurnWiseEval, for multi-turn capabilities that is directly comparable to single-turn chat evaluation. Our evaluation isolates multi-turn specific conversational ability through pairwise comparison to equivalent single-turn settings. We additionally introduce our synthetic multi-turn data pipeline TurnWiseData which allows the scalable generation of multi-turn training data. Our experiments with Olmo 3 show that training with multi-turn data is vital to achieving strong multi-turn chat performance, and that including as little as 10k multi-turn conversations during post-training can lead to a 12% improvement on TurnWiseEval. 

---
# Probing Cultural Signals in Large Language Models through Author Profiling 

**Authors**: Valentin Lafargue, Ariel Guerra-Adames, Emmanuelle Claeys, Elouan Vuichard, Jean-Michel Loubes  

**Link**: [PDF](https://arxiv.org/pdf/2603.16749)  

**Abstract**: Large language models (LLMs) are increasingly deployed in applications with societal impact, raising concerns about the cultural biases they encode. We probe these representations by evaluating whether LLMs can perform author profiling from song lyrics in a zero-shot setting, inferring singers' gender and ethnicity without task-specific fine-tuning. Across several open-source models evaluated on more than 10,000 lyrics, we find that LLMs achieve non-trivial profiling performance but demonstrate systematic cultural alignment: most models default toward North American ethnicity, while DeepSeek-1.5B aligns more strongly with Asian ethnicity. This finding emerges from both the models' prediction distributions and an analysis of their generated rationales. To quantify these disparities, we introduce two fairness metrics, Modality Accuracy Divergence (MAD) and Recall Divergence (RD), and show that Ministral-8B displays the strongest ethnicity bias among the evaluated models, whereas Gemma-12B shows the most balanced behavior. Our code is available on GitHub (this https URL). 

---
# Arabic Morphosyntactic Tagging and Dependency Parsing with Large Language Models 

**Authors**: Mohamed Adel, Bashar Alhafni, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2603.16718)  

**Abstract**: Large language models (LLMs) perform strongly on many NLP tasks, but their ability to produce explicit linguistic structure remains unclear. We evaluate instruction-tuned LLMs on two structured prediction tasks for Standard Arabic: morphosyntactic tagging and labeled dependency parsing. Arabic provides a challenging testbed due to its rich morphology and orthographic ambiguity, which create strong morphology-syntax interactions. We compare zero-shot prompting with retrieval-based in-context learning (ICL) using examples from Arabic treebanks. Results show that prompt design and demonstration selection strongly affect performance: proprietary models approach supervised baselines for feature-level tagging and become competitive with specialized dependency parsers. In raw-text settings, tokenization remains challenging, though retrieval-based ICL improves both parsing and tokenization. Our analysis highlights which aspects of Arabic morphosyntax and syntax LLMs capture reliably and which remain difficult. 

---
# Can Linguistically Related Languages Guide LLM Translation in Low-Resource Settings? 

**Authors**: Aishwarya Ramasethu, Niyathi Allu, Rohin Garg, Harshwardhan Fartale, Dun Li Chan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16660)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance across many downstream tasks, yet their effectiveness in extremely low-resource machine translation remains limited. Standard adaptation techniques typically rely on large-scale parallel data or extensive fine-tuning, which are infeasible for the long tail of underrepresented languages. In this work, we investigate a more constrained question: in data-scarce settings, to what extent can linguistically similar pivot languages and few-shot demonstrations provide useful guidance for on-the-fly adaptation in LLMs? We study a data-efficient experimental setup that combines linguistically related pivot languages with few-shot in-context examples, without any parameter updates, and evaluate translation behavior under controlled conditions. Our analysis shows that while pivot-based prompting can yield improvements in certain configurations, particularly in settings where the target language is less well represented in the model's vocabulary, the gains are often modest and sensitive to few shot example construction. For closely related or better represented varieties, we observe diminishing or inconsistent gains. Our findings provide empirical guidance on how and when inference-time prompting and pivot-based examples can be used as a lightweight alternative to fine-tuning in low-resource translation settings. 

---
# Omanic: Towards Step-wise Evaluation of Multi-hop Reasoning in Large Language Models 

**Authors**: Xiaojie Gu, Sherry T. Tong, Aosong Feng, Sophia Simeng Han, Jinghui Lu, Yingjian Chen, Yusuke Iwasawa, Yutaka Matsuo, Chanjun Park, Rex Ying, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16654)  

**Abstract**: Reasoning-focused large language models (LLMs) have advanced in many NLP tasks, yet their evaluation remains challenging: final answers alone do not expose the intermediate reasoning steps, making it difficult to determine whether a model truly reasons correctly and where failures occur, while existing multi-hop QA benchmarks lack step-level annotations for diagnosing reasoning failures. To address this gap, we propose Omanic, an open-domain multi-hop QA resource that provides decomposed sub-questions and intermediate answers as structural annotations for analyzing reasoning processes. It contains 10,296 machine-generated training examples (OmanicSynth) and 967 expert-reviewed human-annotated evaluation examples (OmanicBench). Systematic evaluations show that state-of-the-art LLMs achieve only 73.11% multiple-choice accuracy on OmanicBench, confirming its high difficulty. Stepwise analysis reveals that CoT's performance hinges on factual completeness, with its gains diminishing under knowledge gaps and errors amplifying in later hops. Additionally, supervised fine-tuning on OmanicSynth brings substantial transfer gains (7.41 average points) across six reasoning and math benchmarks, validating the dataset's quality and further supporting the effectiveness of OmanicSynth as supervision for reasoning-capability transfer. We release the data at this https URL and the code at this https URL. 

---
# Good Arguments Against the People Pleasers: How Reasoning Mitigates (Yet Masks) LLM Sycophancy 

**Authors**: Zhaoxin Feng, Zheng Chen, Jianfei Ma, Yip Tin Po, Emmanuele Chersoni, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16643)  

**Abstract**: Alignment techniques often inadvertently induce sycophancy in LLMs. While prior studies studied this behaviour in direct-answer settings, the role of Chain-of-Thought (CoT) reasoning remains under-explored: does it serve as a logical constraint that mitigates sycophancy, or a tool for post-hoc rationalization that masks it? We evaluate a range of models across objective and subjective tasks to investigate the issue. Results show that reasoning generally reduces sycophancy in final decisions but also masks sycophancy in some samples, where models construct deceptive justifications through logical inconsistencies, calculation errors, and one-sided arguments etc. Furthermore, LLMs are more prone to sycophancy in subjective tasks and under authority-bias. Our mechanistic analysis on three open-source models reveals that the tendency of sycophancy is dynamic during the reasoning process rather than being pre-determined at the input stage. 

---
# Domain Mixture Design via Log-Likelihood Differences for Aligning Language Models with a Target Model 

**Authors**: Ryo Kishino, Riku Shiomi, Hiroaki Yamagiwa, Momose Oyama, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2603.16622)  

**Abstract**: Instead of directly distilling a language model, this study addresses the problem of aligning a base model with a target model in distribution by designing the domain mixture of training data for pretraining or continued pretraining as a fixed training recipe. We propose a method for determining domain weights by viewing models as points in log-likelihood space and aligning the training update direction with the direction toward the target model. Experiments with NanoGPT show that the proposed method consistently reduces the KL divergence to the target model compared with uniform weighting over the Pile. Although knowledge distillation remains more effective when available, the proposed method still achieves meaningful alignment, and downstream task performance also tends to become closer to that of the target model. 

---
# Omnilingual SONAR: Cross-Lingual and Cross-Modal Sentence Embeddings Bridging Massively Multilingual Text and Speech 

**Authors**: Omnilingual SONAR Team, João Maria Janeiro, Pere-Lluís Huguet Cabot, Ioannis Tsiamas, Yen Meng, Vivek Iyer, Guillem Ramírez, Loic Barrault, Belen Alastruey, Yu-An Chung, Marta R. Costa-Jussa, David Dale, Kevin Heffernan, Jaehyeong Jo, Artyom Kozhevnikov, Alexandre Mourachko, Christophe Ropers, Holger Schwenk, Paul-Ambroise Duquenne  

**Link**: [PDF](https://arxiv.org/pdf/2603.16606)  

**Abstract**: Cross-lingual sentence encoders typically cover only a few hundred languages and often trade downstream quality for stronger alignment, limiting their adoption. We introduce OmniSONAR, a new family of omnilingual, cross-lingual and cross-modal sentence embedding models that natively embed text, speech, code, and mathematical expressions in a single semantic space, while delivering state-of-the-art downstream performance at the scale of thousands of languages, from high-resource to extremely low-resource varieties. To reach this scale without representation collapse, we use progressive training. We first learn a strong foundational space for 200 languages with an LLM-initialized encoder-decoder, combining token-level decoding with a novel split-softmax contrastive loss and synthetic hard negatives. Building on this foundation, we expand to several thousands language varieties via a two-stage teacher-student encoder distillation framework. Finally, we demonstrate the cross-modal extensibility of this space by seamlessly mapping 177 spoken languages into it. OmniSONAR halves cross-lingual similarity search error on the 200-language FLORES dataset and reduces error by a factor of 15 on the 1,560-language BIBLE benchmark. It also enables strong translation, outperforming NLLB-3B on multilingual benchmarks and exceeding prior models (including much larger LLMs) by 15 chrF++ points on 1,560 languages into English BIBLE translation. OmniSONAR also performs strongly on MTEB and XLCoST. For speech, OmniSONAR achieves a 43% lower similarity-search error and reaches 97% of SeamlessM4T speech-to-text quality, despite being zero-shot for translation (trained only on ASR data). Finally, by training an encoder-decoder LM, Spectrum, exclusively on English text processing OmniSONAR embedding sequences, we unlock high-performance transfer to thousands of languages and speech for complex downstream tasks. 

---
# Tarab: A Multi-Dialect Corpus of Arabic Lyrics and Poetry 

**Authors**: Mo El-Haj  

**Link**: [PDF](https://arxiv.org/pdf/2603.16601)  

**Abstract**: We introduce the Tarab Corpus, a large-scale cultural and linguistic resource that brings together Arabic song lyrics and poetry within a unified analytical framework. The corpus comprises 2.56 million verses and more than 13.5 million tokens, making it, to our knowledge, the largest open Arabic corpus of creative text spanning both classical and contemporary production. Tarab is broadly balanced between songs and poems and covers Classical Arabic, Modern Standard Arabic (MSA), and six major regional varieties: Egyptian, Gulf, Levantine, Iraqi, Sudanese, and Maghrebi Arabic. The artists and poets represented in the corpus are associated with 28 modern nation states and multiple historical eras, covering over fourteen centuries of Arabic creative expression from the Pre-Islamic period to the twenty-first century. Each verse is accompanied by structured metadata describing linguistic variety, geographic origin, and historical or cultural context, enabling comparative linguistic, stylistic, and diachronic analysis across genres and time. We describe the data collection, normalisation, and validation pipeline and present baseline analyses for variety identification and genre differentiation. The dataset is publicly available on HuggingFace at this https URL. 

---
# BATQuant: Outlier-resilient MXFP4 Quantization via Learnable Block-wise Optimization 

**Authors**: Ji-Fu Li, Manyi Zhang, Xiaobo Xia, Han Bao, Haoli Bai, Zhenhua Dong, Xianzhi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16590)  

**Abstract**: Microscaling floating-point (MXFP) formats have emerged as a promising standard for deploying Multi-modal Large Language Models (MLLMs) and Large Language Models (LLMs) on modern accelerator architectures. However, existing Post-Training Quantization (PTQ) methods, particularly rotation-based techniques designed for integer formats, suffer from severe performance collapse when applied to MXFP4. Recent studies attribute this failure to a fundamental format mismatch: global orthogonal rotations inadvertently transfer outlier energy across quantization blocks, inducing new outliers that disrupt local block-wise scaling, while often creating bimodal activation distributions that underutilize the limited quantization range. To address these issues, we propose BATQuant (Block-wise Affine Transformation), which restricts transformations to align with MXFP granularity to prevent cross-block outlier propagation, while relaxing orthogonality constraints to optimize distribution shaping. To ensure parameter efficiency, we introduce Global and Private Kronecker (GPK) decomposition to effectively reduces storage and runtime overhead and incorporate Block-wise Learnable Clipping to suppress residual outliers. Extensive experiments on both MLLMs and LLMs demonstrate that BATQuant establishes new state-of-the-art results under aggressive W4A4KV16 configurations, recovering up to 96.43% of full-precision performance on multimodal benchmarks and clearly outperforming existing methods across diverse tasks. 

---
# Diverging Transformer Predictions for Human Sentence Processing: A Comprehensive Analysis of Agreement Attraction Effects 

**Authors**: Titus von der Malsburg, Sebastian Padó  

**Link**: [PDF](https://arxiv.org/pdf/2603.16574)  

**Abstract**: Transformers underlie almost all state-of-the-art language models in computational linguistics, yet their cognitive adequacy as models of human sentence processing remains disputed. In this work, we use a surprisal-based linking mechanism to systematically evaluate eleven autoregressive transformers of varying sizes and architectures on a more comprehensive set of English agreement attraction configurations than prior work. Our experiments yield mixed results: While transformer predictions generally align with human reading time data for prepositional phrase configurations, performance degrades significantly on object-extracted relative clause configurations. In the latter case, predictions also diverge markedly across models, and no model successfully replicates the asymmetric interference patterns observed in humans. We conclude that current transformer models do not explain human morphosyntactic processing, and that evaluations of transformers as cognitive models must adopt rigorous, comprehensive experimental designs to avoid spurious generalizations from isolated syntactic configurations or individual models. 

---
# Characterizing Delusional Spirals through Human-LLM Chat Logs 

**Authors**: Jared Moore, Ashish Mehta, William Agnew, Jacy Reese Anthis, Ryan Louie, Yifan Mai, Peggy Yin, Myra Cheng, Samuel J Paech, Kevin Klyman, Stevie Chancellor, Eric Lin, Nick Haber, Desmond C. Ong  

**Link**: [PDF](https://arxiv.org/pdf/2603.16567)  

**Abstract**: As large language models (LLMs) have proliferated, disturbing anecdotal reports of negative psychological effects, such as delusions, self-harm, and ``AI psychosis,'' have emerged in global media and legal discourse. However, it remains unclear how users and chatbots interact over the course of lengthy delusional ``spirals,'' limiting our ability to understand and mitigate the harm. In our work, we analyze logs of conversations with LLM chatbots from 19 users who report having experienced psychological harms from chatbot use. Many of our participants come from a support group for such chatbot users. We also include chat logs from participants covered by media outlets in widely-distributed stories about chatbot-reinforced delusions. In contrast to prior work that speculates on potential AI harms to mental health, to our knowledge we present the first in-depth study of such high-profile and veridically harmful cases. We develop an inventory of 28 codes and apply it to the $391,562$ messages in the logs. Codes include whether a user demonstrates delusional thinking (15.5% of user messages), a user expresses suicidal thoughts (69 validated user messages), or a chatbot misrepresents itself as sentient (21.2% of chatbot messages). We analyze the co-occurrence of message codes. We find, for example, that messages that declare romantic interest and messages where the chatbot describes itself as sentient occur much more often in longer conversations, suggesting that these topics could promote or result from user over-engagement and that safeguards in these areas may degrade in multi-turn settings. We conclude with concrete recommendations for how policymakers, LLM chatbot developers, and users can use our inventory and conversation analysis tool to understand and mitigate harm from LLM chatbots.
Warning: This paper discusses self-harm, trauma, and violence. 

---
# EmoLLM: Appraisal-Grounded Cognitive-Emotional Co-Reasoning in Large Language Models 

**Authors**: Yifei Zhang, Mingyang Li, Henry Gao, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16553)  

**Abstract**: Large language models (LLMs) demonstrate strong cognitive intelligence (IQ), yet many real-world interactions also require emotional intelligence (EQ) to produce responses that are both factually reliable and emotionally appropriate. In settings such as emotional support, technical assistance, and consultation, effective dialogue depends on how situations are appraised with respect to the user's needs, goals, and coping capacity. Inspired by appraisal theory, we propose EmoLLM, an appraisal-grounded framework for IQ/EQ co-reasoning in dialogue. EmoLLM uses an explicit Appraisal Reasoning Graph (ARG) to structure intermediate reasoning over contextual facts, inferred user needs, appraisal dimensions, emotional states, and response strategies before generating a reply. We train EmoLLM in a multi-turn role-play environment with reinforcement learning, where reverse-perspective reasoning provides reward signals based on predicted user-side consequences of responses. Across diverse dialogue settings, EmoLLM improves emotional state outcomes and response quality over strong baselines while preserving strong factual reliability. 

---
# DanceHA: A Multi-Agent Framework for Document-Level Aspect-Based Sentiment Analysis 

**Authors**: Lei Wang, Min Huang, Eduard Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2603.16546)  

**Abstract**: Aspect-Based Sentiment Intensity Analysis (ABSIA) has garnered increasing attention, though research largely focuses on domain-specific, sentence-level settings. In contrast, document-level ABSIA--particularly in addressing complex tasks like extracting Aspect-Category-Opinion-Sentiment-Intensity (ACOSI) tuples--remains underexplored. In this work, we introduce DanceHA, a multi-agent framework designed for open-ended, document-level ABSIA with informal writing styles. DanceHA has two main components: Dance, which employs a divide-and-conquer strategy to decompose the long-context ABSIA task into smaller, manageable sub-tasks for collaboration among specialized agents; and HA, Human-AI collaboration for annotation. We release Inf-ABSIA, a multi-domain document-level ABSIA dataset featuring fine-grained and high-accuracy labels from DanceHA. Extensive experiments demonstrate the effectiveness of our agentic framework and show that the multi-agent knowledge in DanceHA can be effectively transferred into student models. Our results highlight the importance of the overlooked informal styles in ABSIA, as they often intensify opinions tied to specific aspects. 

---
# How often do Answers Change? Estimating Recency Requirements in Question Answering 

**Authors**: Bhawna Piryani, Zehra Mert, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2603.16544)  

**Abstract**: Large language models (LLMs) often rely on outdated knowledge when answering time-sensitive questions, leading to confident yet incorrect responses. Without explicit signals indicating whether up-to-date information is required, models struggle to decide when to retrieve external evidence, how to reason about stale facts, and how to rank answers by their validity. Existing benchmarks either periodically refresh answers or rely on fixed templates, but they do not reflect on how frequently answers change or whether a question inherently requires up-to-date information. To address this gap, we introduce a recency-stationarity taxonomy that categorizes questions by how often their answers change and whether this change frequency is time-invariant or context-dependent. Building on this taxonomy, we present RecencyQA, a dataset of 4,031 open-domain questions annotated with recency and stationarity labels. Through human evaluation and empirical analysis, we show that non-stationary questions, i.e., those where context changes the recency requirement, are significantly more challenging for LLMs, with difficulty increasing as update frequency rises. By explicitly modeling recency and context dependence, RecencyQA enables fine-grained benchmarking and analysis of temporal reasoning beyond binary notions of freshness, and provides a foundation for developing recency-aware and context-sensitive question answering systems. 

---
# AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents 

**Authors**: Shannan Yan, Jingchen Ni, Leqi Zheng, Jiajun Zhang, Peixi Wu, Dacheng Yin, Jing Lyu, Chun Yuan, Fengyun Rao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16496)  

**Abstract**: Large language model (LLM) agents increasingly rely on external memory to support long-horizon interaction, personalized assistance, and multi-step reasoning. However, existing memory systems still face three core challenges: they often rely too heavily on semantic similarity, which can miss evidence crucial for user-centric understanding; they frequently store related experiences as isolated fragments, weakening temporal and causal coherence; and they typically use static memory granularities that do not adapt well to the requirements of different questions. We propose AdaMem, an adaptive user-centric memory framework for long-horizon dialogue agents. AdaMem organizes dialogue history into working, episodic, persona, and graph memories, enabling the system to preserve recent context, structured long-term experiences, stable user traits, and relation-aware connections within a unified framework. At inference time, AdaMem first resolves the target participant, then builds a question-conditioned retrieval route that combines semantic retrieval with relation-aware graph expansion only when needed, and finally produces the answer through a role-specialized pipeline for evidence synthesis and response generation. We evaluate AdaMem on the LoCoMo and PERSONAMEM benchmarks for long-horizon reasoning and user modeling. Experimental results show that AdaMem achieves state-of-the-art performance on both benchmarks. The code will be released upon acceptance. 

---
# On the Emotion Understanding of Synthesized Speech 

**Authors**: Yuan Ge, Haishu Zhao, Aokai Hao, Junxiang Zhang, Bei Li, Xiaoqian Liu, Chenglong Wang, Jianjin Wang, Bingsen Zhou, Bingyu Liu, Jingbo Zhu, Zhengtao Yu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16483)  

**Abstract**: Emotion is a core paralinguistic feature in voice interaction. It is widely believed that emotion understanding models learn fundamental representations that transfer to synthesized speech, making emotion understanding results a plausible reward or evaluation metric for assessing emotional expressiveness in speech synthesis. In this work, we critically examine this assumption by systematically evaluating Speech Emotion Recognition (SER) on synthesized speech across datasets, discriminative and generative SER models, and diverse synthesis models. We find that current SER models can not generalize to synthesized speech, largely because speech token prediction during synthesis induces a representation mismatch between synthesized and human speech. Moreover, generative Speech Language Models (SLMs) tend to infer emotion from textual semantics while ignoring paralinguistic cues. Overall, our findings suggest that existing SER models often exploit non-robust shortcuts rather than capturing fundamental features, and paralinguistic understanding in SLMs remains challenging. 

---
# DynHD: Hallucination Detection for Diffusion Large Language Models via Denoising Dynamics Deviation Learning 

**Authors**: Yanyu Qian, Yue Tan, Yixin Liu, Wang Yu, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16459)  

**Abstract**: Diffusion large language models (D-LLMs) have emerged as a promising alternative to auto-regressive models due to their iterative refinement capabilities. However, hallucinations remain a critical issue that hinders their reliability. To detect hallucination responses from model outputs, token-level uncertainty (e.g., entropy) has been widely used as an effective signal to indicate potential factual errors. Nevertheless, the fixed-length generation paradigm of D-LLMs implies that tokens contribute unevenly to hallucination detection, with only a small subset providing meaningful signals. Moreover, the evolution trend of uncertainty throughout the diffusion process can also provide important signals, highlighting the necessity of modeling its denoising dynamics for hallucination detection. In this paper, we propose DynHD that bridge these gaps from both spatial (token sequence) and temporal (denoising dynamics) perspectives. To address the information density imbalance across tokens, we propose a semantic-aware evidence construction module that extracts hallucination-indicative signals by filtering out non-informative tokens and emphasizing semantically meaningful ones. To model denoising dynamics for hallucination detection, we introduce a reference evidence generator that learns the expected evolution trajectory of uncertainty evidence, along with a deviation-based hallucination detector that makes predictions by measuring the discrepancy between the observed and reference trajectories. Extensive experiments demonstrate that DynHD consistently outperforms state-of-the-art baselines while achieving higher efficiency across multiple benchmarks and backbone models. 

---
# VQKV: High-Fidelity and High-Ratio Cache Compression via Vector-Quantization 

**Authors**: Yixuan Wang, Qingyu Shi, Jiayu Zhou, Dianbo Liu, Ziwei He, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.16435)  

**Abstract**: The growing context length of Large Language Models (LLMs) enlarges the Key-Value (KV) cache, limiting deployment in resource-limited environments. Prior training-free approaches for KV cache compression typically rely on low-rank approximation or scalar quantization, which fail to simultaneously achieve high compression ratios and high reconstruction fidelity. We propose VQKV, a novel, training-free method introducing vector quantization (VQ) to obtain highly compressed KV representations while preserving high model fidelity, allowing for the representation of thousands of floating-point values with just a few integer indices. As a result, VQKV achieves an 82.8\% compression ratio on LLaMA3.1-8B while retaining 98.6\% of the baseline performance on LongBench and enabling 4.3x longer generation length on the same memory footprint. 

---
# EngGPT2: Sovereign, Efficient and Open Intelligence 

**Authors**: G. Ciarfaglia, A. Rosanova, S. Cipolla, J. Bartoli, A. Di Domenico, C. Fioroni, A. Fontana, M. R. Scoleri, M. I. Mone, D. Franchi, M. C. Del Gaudio, F. Picariello, M. Gabusi, S. Bonura, V. Morreale, I. Bailo  

**Link**: [PDF](https://arxiv.org/pdf/2603.16430)  

**Abstract**: EngGPT2-16B-A3B is the latest iteration of Engineering Group's Italian LLM and it's built to be a Sovereign, Efficient and Open model. EngGPT2 is trained on 2.5 trillion tokens - less than Qwen3's 36T or Llama3's 15T - and delivers performance on key benchmarks, including MMLU-Pro, GSM8K, IFEval and HumanEval, comparable to dense models in the 8B-16B range, while requiring one-fifth to half of the inference power, and between one-tenth to one-sixth of the training data and consequent needed training power. Designed as a trained-from-scratch Mixture-of-Experts (MoE) architecture, EngGPT2 features 16 billion parameters with 3 billion active per inference, with expert sizes positioned between those used in GPT-OSS and Qwen3. Approximately 25% of its training corpus consists of Italian-language data, to deliver strong capabilities for European and Italian NLP tasks among models of similar scale. This efficiency aims to position EngGPT2 as a key contributor to the growing portfolio of open-weight European models, combining performance and efficiency with full alignment to the EU AI Act. EngGPT2 is also a single model capable of multiple reasoning modes: non-reasoning, reasoning in Italian or English, and turbo-reasoning (a concise, bullet-point style reasoning available in both languages designed for real-time reasoning use cases). EngGPT2 aims to set a new standard for resource-conscious, high-performance LLMs tailored to European and Italian contexts. 

---
# IndexRAG: Bridging Facts for Cross-Document Reasoning at Index Time 

**Authors**: Zhenghua Bao, Yi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16415)  

**Abstract**: Multi-hop question answering (QA) requires reasoning across multiple documents, yet existing retrieval-augmented generation (RAG) approaches address this either through graph-based methods requiring additional online processing or iterative multi-step reasoning. We present IndexRAG, a novel approach that shifts cross-document reasoning from online inference to offline indexing. IndexRAG identifies bridge entities shared across documents and generates bridging facts as independently retrievable units, requiring no additional training or fine-tuning. Experiments on three widely-used multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue) show that IndexRAG improves F1 over Naive RAG by 4.6 points on average, while requiring only single-pass retrieval and a single LLM call at inference time. When combined with IRCoT, IndexRAG outperforms all graph-based baselines on average, including HippoRAG and FastGraphRAG, while relying solely on flat retrieval. Our code will be released upon acceptance. 

---
# RECOVER: Robust Entity Correction via agentic Orchestration of hypothesis Variants for Evidence-based Recovery 

**Authors**: Abhishek Kumar, Aashraya Sachdeva  

**Link**: [PDF](https://arxiv.org/pdf/2603.16411)  

**Abstract**: Entity recognition in Automatic Speech Recognition (ASR) is challenging for rare and domain-specific terms. In domains such as finance, medicine, and air traffic control, these errors are costly. If the entities are entirely absent from the ASR output, post-ASR correction becomes difficult. To address this, we introduce RECOVER, an agentic correction framework that serves as a tool-using agent. It leverages multiple hypotheses as evidence from ASR, retrieves relevant entities, and applies Large Language Model (LLM) correction under constraints. The hypotheses are used using different strategies, namely, 1-Best, Entity-Aware Select, Recognizer Output Voting Error Reduction (ROVER) Ensemble, and LLM-Select. Evaluated across five diverse datasets, it achieves 8-46% relative reductions in entity-phrase word error rate (E-WER) and increases recall by up to 22 percentage points. The LLM-Select achieves the best overall performance in entity correction while maintaining overall WER. 

---
# PlotTwist: A Creative Plot Generation Framework with Small Language Models 

**Authors**: Abhinav Thorat, Ravi Kolla, Jyotin Goel, Niranjan Pedanekar  

**Link**: [PDF](https://arxiv.org/pdf/2603.16410)  

**Abstract**: Creative plot generation presents a fundamental challenge for language models: transforming a concise premise into a coherent narrative that sustains global structure, character development, and emotional resonance. Although recent Large Language Models (LLMs) demonstrate strong fluency across general-purpose tasks, they typically require preference alignment to perform well on specialized domains such as creative plot generation. However, conducting such alignment at the scale of frontier LLMs is computationally prohibitive, significantly limiting accessibility and practical deployment. To address this, we present PlotTwist, a structured framework that enables Small Language Models (SLMs) with $\leq$ 5B active parameters to generate high-quality, premise-conditioned plots competitive with frontier systems up to $200\times$ larger. Our approach decomposes generation into three specialized components: (1) an Aspect Rating Reward Model trained via a novel Positive-Negative prompting strategy to deliver structured narratives across five Narrative Quality Dimensions (NQDs); (2) a Mixture-of-Experts (MoE) plot generator aligned via Direct Preference Optimization on high-confidence preference pairs; and (3) an Agentic Evaluation module that emulates human critical judgment for unbiased post-hoc assessment. Extensive experiments demonstrate that PlotTwist consistently outperforms frontier models across multiple NQDs despite substantially tighter capacity constraints. Further validation confirms strong sensitivity to narrative quality, as the framework reliably distinguishes plots derived from critically acclaimed versus widely panned screenplays. Together, these results establish structured, preference-based alignment as a resource-efficient approach to high-quality creative plot generation. 

---
# Who Benchmarks the Benchmarks? A Case Study of LLM Evaluation in Icelandic 

**Authors**: Finnur Ágúst Ingimundarson, Steinunn Rut Friðriksdóttir, Bjarki Ármannsson, Iris Edda Nowenstein, Steinþór Steingrímsson  

**Link**: [PDF](https://arxiv.org/pdf/2603.16406)  

**Abstract**: This paper evaluates current Large Language Model (LLM) benchmarking for Icelandic, identifies problems, and calls for improved evaluation methods in low/medium-resource languages in particular. We show that benchmarks that include synthetic or machine-translated data that have not been verified in any way, commonly contain severely flawed test examples that are likely to skew the results and undermine the tests' validity. We warn against the use of such methods without verification in low/medium-resource settings as the translation quality can, at best, only be as good as MT quality for a given language at any given time. Indeed, the results of our quantitative error analysis on existing benchmarks for Icelandic show clear differences between human-authored/-translated benchmarks vs. synthetic or machine-translated benchmarks. 

---
# Fanar 2.0: Arabic Generative AI Stack 

**Authors**: FANAR TEAM, Ummar Abbas, Mohammad Shahmeer Ahmad, Minhaj Ahmad, Abdulaziz Al-Homaid, Anas Al-Nuaimi, Enes Altinisik, Ehsaneddin Asgari, Sanjay Chawla, Shammur Chowdhury, Fahim Dalvi, Kareem Darwish, Nadir Durrani, Mohamed Elfeky, Ahmed Elmagarmid, Mohamed Eltabakh, Asim Ersoy, Masoomali Fatehkia, Mohammed Qusay Hashim, Majd Hawasly, Mohamed Hefeeda, Mus'ab Husaini, Keivin Isufaj, Soon-Gyo Jung, Houssam Lachemat, Ji Kim Lucas, Abubakr Mohamed, Tasnim Mohiuddin, Basel Mousi, Hamdy Mubarak, Ahmad Musleh, Mourad Ouzzani, Amin Sadeghi, Husrev Taha Sencar, Mohammed Shinoy, Omar Sinan, Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16397)  

**Abstract**: We present Fanar 2.0, the second generation of Qatar's Arabic-centric Generative AI platform. Sovereignty is a first-class design principle: every component, from data pipelines to deployment infrastructure, was designed and operated entirely at QCRI, Hamad Bin Khalifa University. Fanar 2.0 is a story of resource-constrained excellence: the effort ran on 256 NVIDIA H100 GPUs, with Arabic having only ~0.5% of web data despite 400 million native speakers. Fanar 2.0 adopts a disciplined strategy of data quality over quantity, targeted continual pre-training, and model merging to achieve substantial gains within these constraints. At the core is Fanar-27B, continually pre-trained from a Gemma-3-27B backbone on a curated corpus of 120 billion high-quality tokens across three data recipes. Despite using 8x fewer pre-training tokens than Fanar 1.0, it delivers substantial benchmark improvements: Arabic knowledge (+9.1 pts), language (+7.3 pts), dialects (+3.5 pts), and English capability (+7.6 pts). Beyond the core LLM, Fanar 2.0 introduces a rich stack of new capabilities. FanarGuard is a state-of-the-art 4B bilingual moderation filter for Arabic safety and cultural alignment. The speech family Aura gains a long-form ASR model for hours-long audio. Oryx vision family adds Arabic-aware image and video understanding alongside culturally grounded image generation. An agentic tool-calling framework enables multi-step workflows. Fanar-Sadiq utilizes a multi-agent architecture for Islamic content. Fanar-Diwan provides classical Arabic poetry generation. FanarShaheen delivers LLM-powered bilingual translation. A redesigned multi-layer orchestrator coordinates all components through intent-aware routing and defense-in-depth safety validation. Taken together, Fanar 2.0 demonstrates that sovereign, resource-constrained AI development can produce systems competitive with those built at far greater scale. 

---
# PashtoCorp: A 1.25-Billion-Word Corpus, Evaluation Suite, and Reproducible Pipeline for Low-Resource Language Development 

**Authors**: Hanif Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2603.16354)  

**Abstract**: We present PashtoCorp, a 1.25-billion-word corpus for Pashto, a language spoken by 60 million people that remains severely underrepresented in NLP. The corpus is assembled from 39 sources spanning seven HuggingFace
datasets and 32 purpose-built web scrapers, processed through a reproducible pipeline with Arabic-script tokenization, SHA-256 deduplication, and quality filtering. At 1.25B words across 2.81 million documents,
PashtoCorp is 40x larger than the OSCAR Pashto subset and 83x larger than the previously largest dedicated Pashto corpus. Continued MLM pretraining of XLM-R-base on PashtoCorp reduces held-out perplexity by 25.1%
(8.08->6.06). On WikiANN Pashto NER, the pretrained model improves entity F1 by 10% relative (19.0%->21.0%) and reduces training variance nearly 7x; the largest gain appears at 50 training sentences (+27%), with
PashtoCorp covering 97.9% of WikiANN entity vocabulary. On Belebele Pashto reading comprehension, Gemma-3n achieves 64.6% accuracy, the first published LLM baseline for Pashto on this benchmark. A leave-one-out source
ablation shows that Wikipedia (0.7% of documents) is the most critical source for NER: removing it alone reduces entity F1 by 47%. Corpus data, trained model, and code are available at
this https URL, this https URL, and this https URL. 

---
# Omnilingual MT: Machine Translation for 1,600 Languages 

**Authors**: Omnilingual MT Team, Belen Alastruey, Niyati Bafna, Andrea Caciolai, Kevin Heffernan, Artyom Kozhevnikov, Christophe Ropers, Eduardo Sánchez, Charles-Eric Saint-James, Ioannis Tsiamas, Chierh Cheng, Joe Chuang, Paul-Ambroise Duquenne, Mark Duppenthaler, Nate Ekberg, Cynthia Gao, Pere Lluís Huguet Cabot, João Maria Janeiro, Jean Maillard, Gabriel Mejia Gonzalez, Holger Schwenk, Edan Toledo, Arina Turkatenko, Albert Ventayol-Boada, Rashel Moritz, Alexandre Mourachko, Surya Parimi, Mary Williamson, Shireen Yates, David Dale, Marta R. Costa-jussà  

**Link**: [PDF](https://arxiv.org/pdf/2603.16309)  

**Abstract**: High-quality machine translation (MT) can scale to hundreds of languages, setting a high bar for multilingual systems. However, compared to the world's 7,000 languages, current systems still offer only limited coverage: about 200 languages on the target side, and maybe a few hundreds more on the source side, supported due to cross-lingual transfer. And even these numbers have been hard to evaluate due to the lack of reliable benchmarks and metrics.
We present Omnilingual Machine Translation (OMT), the first MT system supporting more than 1,600 languages. This scale is enabled by a comprehensive data strategy that integrates large public multilingual corpora with newly created datasets, including manually curated MeDLEY bitext.
We explore two ways of specializing a Large Language model (LLM) for machine translation: as a decoder-only model (OMT-LLaMA) or as a module in an encoder-decoder architecture (OMT-NLLB). Notably, all our 1B to 8B parameter models match or exceed the MT performance of a 70B LLM baseline, revealing a clear specialization advantage and enabling strong translation quality in low-compute settings. Moreover, our evaluation of English-to-1,600 translations further shows that while baseline models can interpret undersupported languages, they frequently fail to generate them with meaningful fidelity; OMT-LLaMA models substantially expand the set of languages for which coherent generation is feasible. Additionally, OMT models improve in cross-lingual transfer, being close to solving the "understanding" part of the puzzle in MT for the 1,600 evaluated. Our leaderboard and main human-created evaluation datasets (BOUQuET and Met-BOUQuET) are dynamically evolving towards Omnilinguality and freely available. 

---
# PyPhonPlan: Simulating phonetic planning with dynamic neural fields and task dynamics 

**Authors**: Sam Kirkham  

**Link**: [PDF](https://arxiv.org/pdf/2603.16299)  

**Abstract**: We introduce PyPhonPlan, a Python toolkit for implementing dynamical models of phonetic planning using coupled dynamic neural fields and task dynamic simulations. The toolkit provides modular components for defining planning, perception and memory fields, as well as between-field coupling, gestural inputs, and using field activation profiles to solve tract variable trajectories. We illustrate the toolkit's capabilities through an example application:~simulating production/perception loops with a coupled memory field, which demonstrates the framework's ability to model interactive speech dynamics using representations that are temporally-principled, neurally-grounded, and phonetically-rich. PyPhonPlan is released as open-source software and contains executable examples to promote reproducibility, extensibility, and cumulative computational development for speech communication research. 

---
# Attention-guided Evidence Grounding for Spoken Question Answering 

**Authors**: Ke Yang, Bolin Chen, Yuejie Li, Yueying Hua, Jianhao Nie, Yueping He, Bowen Li, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16292)  

**Abstract**: Spoken Question Answering (Spoken QA) presents a challenging cross-modal problem: effectively aligning acoustic queries with textual knowledge while avoiding the latency and error propagation inherent in cascaded ASR-based systems. In this paper, we introduce Attention-guided Evidence Grounding (AEG), a novel end-to-end framework that leverages the internal cross-modal attention of Speech Large Language Models (SpeechLLMs) to explicitly locate and ground key evidence in the model's latent space. To address the diffuse attention distribution in pre-trained models, we propose Learning to Focus on Evidence (LFE), a supervised fine-tuning paradigm that calibrates the model's attention mechanism to distinguish query-relevant segments from irrelevant context. Experiments on SQuAD, HotpotQA, and MuSiQue demonstrate that AEG reduces hallucinations and achieves strong efficiency gains, outperforming large-scale cascaded baselines (Whisper-Large-v3 + Reranker) while reducing inference latency by approximately 62%. 

---
# Is Semi-Automatic Transcription Useful in Corpus Creation? Preliminary Considerations on the KIParla Corpus 

**Authors**: Martina Simonotti, Ludovica Pannitto, Eleonora Zucchini, Silvia Ballarè, Caterina Mauri  

**Link**: [PDF](https://arxiv.org/pdf/2603.16258)  

**Abstract**: This paper analyses the implementation of Automatic Speech Recognition (ASR) into the transcription workflow of the KIParla corpus, a resource of spoken Italian. Through a two-phase experiment, 11 expert and novice transcribers produced both manual and ASR-assisted transcriptions of identical audio segments across three different types of conversation, which were subsequently analyzed through a combination of statistical modeling, word-level alignment and a series of annotation-based metrics. Results show that ASR-assisted workflows can increase transcription speed but do not consistently improve overall accuracy, with effects depending on multiple factors such as workflow configuration, conversation type and annotator experience. Analyses combining alignment-based metrics, descriptive statistics and statistical modeling provide a systematic framework to monitor transcription behavior across annotators and workflows. Despite limitations, ASR-assisted transcription, potentially supported by task-specific fine-tuning, could be integrated into the KIParla transcription workflow to accelerate corpus creation without compromising transcription quality. 

---
# More Rounds, More Noise: Why Multi-Turn Review Fails to Improve Cross-Context Verification 

**Authors**: Song Tae-Eun  

**Link**: [PDF](https://arxiv.org/pdf/2603.16244)  

**Abstract**: Cross-Context Review (CCR) improves LLM verification by separating production and review into independent sessions. A natural extension is multi-turn review: letting the reviewer ask follow-up questions, receive author responses, and review again. We call this Dynamic Cross-Context Review (D-CCR). In a controlled experiment with 30 artifacts and 150 injected errors, we tested four D-CCR variants against the single-pass CCR baseline. Single-pass CCR (F1 = 0.376) significantly outperformed all multi-turn variants, including D-CCR-2b with question-and-answer exchange (F1 = 0.303, $p < 0.001$, $d = -0.59$). Multi-turn review increased recall (+0.08) but generated 62% more false positives (8.5 vs. 5.2), collapsing precision from 0.30 to 0.20. Two mechanisms drive this degradation: (1) false positive pressure -- reviewers in later rounds fabricate findings when the artifact's real errors have been exhausted, and (2) Review Target Drift -- reviewers provided with prior Q&A exchanges shift from reviewing the artifact to critiquing the conversation itself. Independent re-review without prior context (D-CCR-2c) performed worst (F1 = 0.263), confirming that mere repetition degrades rather than helps. The degradation stems from false positive pressure in additional rounds, not from information amount -- within multi-turn conditions, more information actually helps (D-CCR-2b > D-CCR-2a). The problem is not what the reviewer sees, but that reviewing again invites noise. 

---
# SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation 

**Authors**: Hang Lv, Sheng Liang, Hao Wang, Yongyue Zhang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16219)  

**Abstract**: Realizing personalized intelligence faces a core dilemma: sending user history to centralized large language models raises privacy concerns, while on-device small language models lack the reasoning capacity required for high-quality generation. Our pilot study shows that purely local enhancements remain insufficient to reliably bridge this gap. We therefore propose SpecSteer, an asymmetric collaborative inference framework that synergizes private on-device context with cloud-scale reasoning. SpecSteer casts collaboration as Bayesian knowledge fusion and repurposes speculative decoding as a distributed alignment protocol, yielding a Draft--Verify--Recover pipeline: the on-device model drafts personalized sequences; the cloud validates via a ratio-based mechanism that decouples reasoning verification from private context, filtering logical flaws without accessing raw user context; upon rejection, a steering recovery injects local intent during correction. Experiments demonstrate that SpecSteer successfully closes the reasoning gap and achieves superior personalized generation performance, while delivering a 2.36x speedup over standard baselines. 

---
# Structured Semantic Cloaking for Jailbreak Attacks on Large Language Models 

**Authors**: Xiaobing Sun, Perry Lam, Shaohua Li, Zizhou Wang, Rick Siow Mong Goh, Yong Liu, Liangli Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16192)  

**Abstract**: Modern LLMs employ safety mechanisms that extend beyond surface-level input filtering to latent semantic representations and generation-time reasoning, enabling them to recover obfuscated malicious intent during inference and refuse accordingly, and rendering many surface-level obfuscation jailbreak attacks ineffective. We propose Structured Semantic Cloaking (S2C), a novel multi-dimensional jailbreak attack framework that manipulates how malicious semantic intent is reconstructed during model inference. S2C strategically distributes and reshapes semantic cues such that full intent consolidation requires multi-step inference and long-range co-reference resolution within deeper latent representations. The framework comprises three complementary mechanisms: (1) Contextual Reframing, which embeds the request within a plausible high-stakes scenario to bias the model toward compliance; (2) Content Fragmentation, which disperses the semantic signature of the request across disjoint prompt segments; and (3) Clue-Guided Camouflage, which disguises residual semantic cues while embedding recoverable markers that guide output generation. By delaying and restructuring semantic consolidation, S2C degrades safety triggers that depend on coherent or explicitly reconstructed malicious intent at decoding time, while preserving sufficient instruction recoverability for functional output generation. We evaluate S2C across multiple open-source and proprietary LLMs using HarmBench and JBB-Behaviors, where it improves Attack Success Rate (ASR) by 12.4% and 9.7%, respectively, over the current SOTA. Notably, S2C achieves substantial gains on GPT-5-mini, outperforming the strongest baseline by 26% on JBB-Behaviors. We also analyse which combinations perform best against broad families of models, and characterise the trade-off between the extent of obfuscation versus input recoverability on jailbreak success. 

---
# Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR 

**Authors**: Quy-Anh Dang, Chris Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2603.16184)  

**Abstract**: We present Polyglot-Lion, a family of compact multilingual automatic speech recognition (ASR) models tailored for the linguistic landscape of Singapore, covering English, Mandarin, Tamil, and Malay. Our models are obtained by fine-tuning Qwen3-ASR-0.6B and Qwen3-ASR-1.7B exclusively on publicly available speech corpora, using a balanced sampling strategy that equalizes the number of training utterances per language and deliberately omits language-tag conditioning so that the model learns to identify languages implicitly from audio. On 12 benchmarks spanning the four target languages, Polyglot-Lion-1.7B achieves an average error rate of 14.85, competitive with MERaLiON-2-10B-ASR (14.32) - a model 6x larger - while incurring a training cost of \$81 on a single RTX PRO 6000 GPU compared to \$18,862 for the 128-GPU baseline. Inference throughput is approximately 20x faster than MERaLiON at 0.10 s/sample versus 2.02 s/sample. These results demonstrate that linguistically balanced fine-tuning of moderate-scale pretrained models can yield deployment-ready multilingual ASR at a fraction of the cost of larger specialist systems. 

---
# Parametric Social Identity Injection and Diversification in Public Opinion Simulation 

**Authors**: Hexi Wang, Yujia Zhou, Bangde Du, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16142)  

**Abstract**: Large language models (LLMs) have recently been adopted as synthetic agents for public opinion simulation, offering a promising alternative to costly and slow human surveys. Despite their scalability, current LLM-based simulation methods fail to capture social diversity, producing flattened inter-group differences and overly homogeneous responses within demographic groups. We identify this limitation as a Diversity Collapse phenomenon in LLM hidden representations, where distinct social identities become increasingly indistinguishable across layers. Motivated by this observation, we propose Parametric Social Identity Injection (PSII), a general framework that injects explicit, parametric representations of demographic attributes and value orientations directly into intermediate hidden states of LLMs. Unlike prompt-based persona conditioning, PSII enables fine-grained and controllable identity modulation at the representation level. Extensive experiments on the World Values Survey using multiple open-source LLMs show that PSII significantly improves distributional fidelity and diversity, reducing KL divergence to real-world survey data while enhancing overall diversity. This work provides new insights into representation-level control of LLM agents and advances scalable, diversity-aware public opinion simulation. Code and data are available at this https URL. 

---
# SIA: A Synthesize-Inject-Align Framework for Knowledge-Grounded and Secure E-commerce Search LLMs with Industrial Deployment 

**Authors**: Zhouwei Zhai, Mengxiang Chen, Anmeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16137)  

**Abstract**: Large language models offer transformative potential for e-commerce search by enabling intent-aware recommendations. However, their industrial deployment is hindered by two critical challenges: (1) knowledge hallucination due to insufficient encoding of dynamic, fine-grained product knowledge, and (2) security vulnerabilities under jailbreak attacks that threaten compliance. To address these issues, we propose SI--a Synthesize-Inject-Align framework for building knowledgeable and secure e-commerce search LLMs. Our approach first synthesizes high-quality natural language corpus by combining structured knowledge graphs with unstructured behavioral logs, augmented with reasoning chains and safety-aware this http URL then introduce a parameter-efficient pre-training strategy based on Depth Up-Scaling to inject domain knowledge while preserving general capabilities. Finally, a dual-path alignment method via multi-task instruction tuning and adversarial training strengthens both task performance and safety robustness. The framework has been deployed at this http URL, China's largest self-operated e-commerce platform, where A/B tests across five core search scenarios demonstrate significant improvements in key business metrics, validating its industrial effectiveness and scalability. 

---
# SciZoom: A Large-scale Benchmark for Hierarchical Scientific Summarization across the LLM Era 

**Authors**: Han Jang, Junhyeok Lee, Kyu Sung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16131)  

**Abstract**: The explosive growth of AI research has created unprecedented information overload, increasing the demand for scientific summarization at multiple levels of granularity beyond traditional abstracts. While LLMs are increasingly adopted for summarization, existing benchmarks remain limited in scale, target only a single granularity, and predate the LLM era. Moreover, since the release of ChatGPT in November 2022, researchers have rapidly adopted LLMs for drafting manuscripts themselves, fundamentally transforming scientific writing, yet no resource exists to analyze how this writing has evolved. To bridge these gaps, we introduce SciZoom, a benchmark comprising 44,946 papers from four top-tier ML venues (NeurIPS, ICLR, ICML, EMNLP) spanning 2020 to 2025, explicitly stratified into Pre-LLM and Post-LLM eras. SciZoom provides three hierarchical summarization targets (Abstract, Contributions, and TL;DR) achieving compression ratios up to 600:1, enabling both multi-granularity summarization research and temporal mining of scientific writing patterns. Our linguistic analysis reveals striking shifts in phrase patterns (up to 10x for formulaic expressions) and rhetorical style (23% decline in hedging), suggesting that LLM-assisted writing produces more confident yet homogenized prose. SciZoom serves as both a challenging benchmark and a unique resource for mining the evolution of scientific discourse in the generative AI era. Our code and dataset are publicly available on GitHub (this https URL) and Hugging Face (this https URL), respectively. 

---
# Social Simulacra in the Wild: AI Agent Communities on Moltbook 

**Authors**: Agam Goyal, Olivia Pal, Hari Sundaram, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2603.16128)  

**Abstract**: As autonomous LLM-based agents increasingly populate social platforms, understanding the dynamics of AI-agent communities becomes essential for both communication research and platform governance. We present the first large-scale empirical comparison of AI-agent and human online communities, analyzing 73,899 Moltbook and 189,838 Reddit posts across five matched communities. Structurally, we find that Moltbook exhibits extreme participation inequality (Gini = 0.84 vs. 0.47) and high cross-community author overlap (33.8\% vs. 0.5\%). In terms of linguistic attributes, content generated by AI-agents is emotionally flattened, cognitively shifted toward assertion over exploration, and socially detached. These differences give rise to apparent community-level homogenization, but we show this is primarily a structural artifact of shared authorship. At the author level, individual agents are more identifiable than human users, driven by outlier stylistic profiles amplified by their extreme posting volume. As AI-mediated communication reshapes online discourse, our work offers an empirical foundation for understanding how multi-agent interaction gives rise to collective communication dynamics distinct from those of human communities. 

---
# Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning 

**Authors**: Kazuki Yano, Shun Kiyono, Sosuke Kobayashi, Sho Takase, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2603.16127)  

**Abstract**: We investigate the role of learning rate scheduling in the large-scale pre-training of large language models, focusing on its influence on downstream performance after supervised fine-tuning (SFT). Decay-based learning rate schedulers are widely used to minimize pre-training loss. However, despite their widespread use, how these schedulers affect performance after SFT remains underexplored. In this paper, we examine Warmup-Stable-Only (WSO), which maintains a constant learning rate after warmup without any decay. Through experiments with 1B and 8B parameter models, we show that WSO consistently outperforms decay-based schedulers in terms of performance after SFT, even though decay-based schedulers may exhibit better performance after pre-training. The result also holds across different regimes with mid-training and over-training. Loss landscape analysis further reveals that decay-based schedulers lead models into sharper minima, whereas WSO preserves flatter minima that support adaptability. These findings indicate that applying LR decay to improve pre-training metrics may compromise downstream adaptability. Our work also provides practical guidance for training and model release strategies, highlighting that pre-training models with WSO enhances their adaptability for downstream tasks. 

---
# Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users 

**Authors**: Nishant Balepur, Malachi Hamada, Varsha Kishore, Sergey Feldman, Amanpreet Singh, Pao Siangliulue, Joseph Chee Chang, Eunsol Choi, Jordan Lee Boyd-Graber, Aakanksha Naik  

**Link**: [PDF](https://arxiv.org/pdf/2603.16120)  

**Abstract**: Deep Research (DR) tools (e.g. OpenAI DR) help researchers cope with ballooning publishing counts. Such tools can synthesize scientific papers to answer researchers' queries, but lack understanding of their users. We change that in MyScholarQA (MySQA), a personalized DR tool that: 1) infers a profile of a user's research interests; 2) proposes personalized actions for a user's input query; and 3) writes a multi-section report for the query that follows user-approved actions. We first test MySQA with NLP's standard protocol: we design a benchmark of synthetic users and LLM judges, where MySQA beats baselines in citation metrics and personalized action-following. However, we suspect this process does not cover all aspects of personalized DR users value, so we interview users in an online version of MySQA to unmask them. We reveal nine nuanced errors of personalized DR undetectable by our LLM judges, and we study qualitative feedback to form lessons for future DR design. In all, we argue for a pillar of personalization that easy-to-use LLM judges can lead NLP to overlook: real progress in personalization is only possible with real users. 

---
# ASDA: Automated Skill Distillation and Adaptation for Financial Reasoning 

**Authors**: Tik Yu Yim, Wenting Tan, Sum Yee Chan, Tak-Wah Lam, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16112)  

**Abstract**: Adapting large language models (LLMs) to specialized financial reasoning typically requires expensive fine-tuning that produces model-locked expertise. Training-free alternatives have emerged, yet our experiments show that leading methods (GEPA and ACE) achieve only marginal gains on the FAMMA financial reasoning benchmark, exposing the limits of unstructured text optimization for complex, multi-step domain reasoning. We introduce Automated Skill Distillation and Adaptation (ASDA), a framework that automatically generates structured skill artifacts through iterative error-corrective learning without modifying model weights. A teacher model analyzes a student model's failures on financial reasoning tasks, clusters errors by subfield and error type, and synthesizes skill files containing reasoning procedures, code templates, and worked examples, which are dynamically injected during inference. Evaluated on FAMMA, ASDA achieves up to +17.33% improvement on arithmetic reasoning and +5.95% on non-arithmetic reasoning, substantially outperforming all training-free baselines. The resulting skill artifacts are human-readable, version-controlled, and compatible with the Agent Skills open standard, offering any organization with a labeled domain dataset a practical and auditable path to domain adaptation without weight access or retraining. 

---
# Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization 

**Authors**: Francesco Pio Monaco, Elia Cunegatti, Flavio Vella, Giovanni Iacca  

**Link**: [PDF](https://arxiv.org/pdf/2603.16105)  

**Abstract**: Post-training model compression is essential for enhancing the portability of Large Language Models (LLMs) while preserving their performance. While several compression approaches have been proposed, less emphasis has been placed on selecting the most suitable set of data (the so-called \emph{calibration data}) for finding the compressed model configuration. The choice of calibration data is a critical step in preserving model capabilities both intra- and inter-tasks. In this work, we address the challenge of identifying high-performance calibration sets for both pruning and quantization by analyzing intrinsic data properties rather than model-specific signals. We introduce \texttt{\textbf{ZipCal}}, a model-agnostic data curation strategy that maximizes lexical diversity based on Zipfian power laws. Experiments demonstrate that our method consistently outperforms standard uniform random sampling across various pruning benchmarks. Notably, it also performs on par, in terms of downstream performance, with a state-of-the-art method that relies on model perplexity. The latter becomes prohibitively expensive at large-scale models and datasets, while \texttt{\textbf{ZipCal}} is on average $\sim$240$\times$ faster due to its tractable linear complexity\footnote{We make the code and the experiments available at this https URL.}. 

---
# CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering 

**Authors**: Tianyi Huang, Ying Kai Deng  

**Link**: [PDF](https://arxiv.org/pdf/2603.16091)  

**Abstract**: In factual question answering, many errors are not failures of access but failures of commitment: the system retrieves relevant evidence, yet still settles on the wrong answer. We present CounterRefine, a lightweight inference-time repair layer for retrieval-grounded question answering. CounterRefine first produces a short answer from retrieved evidence, then gathers additional support and conflicting evidence with follow-up queries conditioned on that draft answer, and finally applies a restricted refinement step that outputs either KEEP or REVISE, with proposed revisions accepted only if they pass deterministic validation. In effect, CounterRefine turns retrieval into a mechanism for testing a provisional answer rather than merely collecting more context. On the full SimpleQA benchmark, CounterRefine improves a matched GPT-5 Baseline-RAG by 5.8 points and reaches a 73.1 percent correct rate, while exceeding the reported one-shot GPT-5.4 score by roughly 40 points. These findings suggest a simple but important direction for knowledgeable foundation models: beyond accessing evidence, they should also be able to use that evidence to reconsider and, when necessary, repair their own answers. 

---
# ClaimFlow: Tracing the Evolution of Scientific Claims in NLP 

**Authors**: Aniket Pramanick, Yufang Hou, Saif M. Mohammad, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2603.16073)  

**Abstract**: Scientific papers do more than report results $-$ they advance $\textit{claims}$ that later work supports, extends, or sometimes refutes. Yet existing methods for citation and claim analysis capture only fragments of this dialogue. In this work, we make these interactions explicit at the level of individual scientific claims. We introduce $\texttt{ClaimFlow}$, a claim-centric view of the NLP literature, built from $304$ ACL Anthology papers (1979$-$2025) that are manually annotated with $1{,}084$ claims and $832$ cross-paper claim relations, indicating whether a citing paper $\textit{supports}$, $\textit{extends}$, $\textit{qualifies}$, $\textit{refutes}$, or references a claim as $\textit{background}$. Using $\texttt{ClaimFlow}$, we define a new task $-$ $\textit{Claim Relation Classification}$ $-$ which requires models to infer the scientific stance toward a cited claim from the text and citation context. Evaluating strong neural models and large language models on this task, we report baseline performance of $0.78$ macro-F1, highlighting that claim-relation classification is feasible but challenging. We further apply our model to $\sim$$13k$ NLP papers to analyze how claims evolve across decades of NLP research. Our analysis reveals that $63.5$% claims are never reused; only $11.1$% are ever challenged; meanwhile, widely propagated claims are more often $\textit{reshaped}$ through qualification and extension than directly confirmed or refuted. Overall, $\texttt{ClaimFlow}$ offers a lens for examining how ideas shift and mature within NLP, and a foundation for assessing whether models can interpret scientific argumentation. 

---
# SEAHateCheck: Functional Tests for Detecting Hate Speech in Low-Resource Languages of Southeast Asia 

**Authors**: Ri Chi Ng, Aditi Kumaresan, Yujia Hu, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.16070)  

**Abstract**: Hate speech detection relies heavily on linguistic resources, which are primarily available in high-resource languages such as English and Chinese, creating barriers for researchers and platforms developing tools for low-resource languages in Southeast Asia, where diverse socio-linguistic contexts complicate online hate moderation. To address this, we introduce SEAHateCheck, a pioneering dataset tailored to Indonesia, Thailand, the Philippines, and Vietnam, covering Indonesian, Tagalog, Thai, and Vietnamese. Building on HateCheck's functional testing framework and refining SGHateCheck's methods, SEAHateCheck provides culturally relevant test cases, augmented by large language models and validated by local experts for accuracy. Experiments with state-of-the-art and multilingual models revealed limitations in detecting hate speech in specific low-resource languages. In particular, Tagalog test cases showed the lowest model accuracy, likely due to linguistic complexity and limited training data. In contrast, slang-based functional tests proved the hardest, as models struggled with culturally nuanced expressions. The diagnostic insights of SEAHateCheck further exposed model weaknesses in implicit hate detection and models' struggles with counter-speech expression. As the first functional test suite for these Southeast Asian languages, this work equips researchers with a robust benchmark, advancing the development of practical, culturally attuned hate speech detection tools for inclusive online content moderation. 

---
# Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability 

**Authors**: Fan Huang, Haewoon Kwak, Jisun An  

**Link**: [PDF](https://arxiv.org/pdf/2603.16017)  

**Abstract**: Large language models (LLMs) increasingly participate in morally sensitive decision-making, yet how they organize ethical frameworks across reasoning steps remains underexplored. We introduce \textit{moral reasoning trajectories}, sequences of ethical framework invocations across intermediate reasoning steps, and analyze their dynamics across six models and three benchmarks. We find that moral reasoning involves systematic multi-framework deliberation: 55.4--57.7\% of consecutive steps involve framework switches, and only 16.4--17.8\% of trajectories remain framework-consistent. Unstable trajectories remain 1.29$\times$ more susceptible to persuasive attacks ($p=0.015$). At the representation level, linear probes localize framework-specific encoding to model-specific layers (layer 63/81 for Llama-3.3-70B; layer 17/81 for Qwen2.5-72B), achieving 13.8--22.6\% lower KL divergence than the training-set prior baseline. Lightweight activation steering modulates framework integration patterns (6.7--8.9\% drift reduction) and amplifies the stability--accuracy relationship. We further propose a Moral Representation Consistency (MRC) metric that correlates strongly ($r=0.715$, $p<0.0001$) with LLM coherence ratings, whose underlying framework attributions are validated by human annotators (mean cosine similarity $= 0.859$). 

---
# RadAnnotate: Large Language Models for Efficient and Reliable Radiology Report Annotation 

**Authors**: Saisha Pradeep Shetty, Roger Eric Goldman, Vladimir Filkov  

**Link**: [PDF](https://arxiv.org/pdf/2603.16002)  

**Abstract**: Radiology report annotation is essential for clinical NLP, yet manual labeling is slow and costly. We present RadAnnotate, an LLM-based framework that studies retrieval-augmented synthetic reports and confidence-based selective automation to reduce expert effort for labeling in RadGraph. We study RadGraph-style entity labeling (graph nodes) and leave relation extraction (edges) to future work. First, we train entity-specific classifiers on gold-standard reports and characterize their strengths and failure modes across anatomy and observation categories, with uncertain observations hardest to learn. Second, we generate RAG-guided synthetic reports and show that synthetic-only models remain within 1-2 F1 points of gold-trained models, and that synthetic augmentation is especially helpful for uncertain observations in a low-resource setting, improving F1 from 0.61 to 0.70. Finally, by learning entity-specific confidence thresholds, RadAnnotate can automatically annotate 55-90% of reports at 0.86-0.92 entity match score while routing low-confidence cases for expert review. 

---
# NLP Occupational Emergence Analysis: How Occupations Form and Evolve in Real Time -- A Zero-Assumption Method Demonstrated on AI in the US Technology Workforce, 2022-2026 

**Authors**: David Nordfors  

**Link**: [PDF](https://arxiv.org/pdf/2603.15998)  

**Abstract**: Occupations form and evolve faster than classification systems can track. We propose that a genuine occupation is a self-reinforcing structure (a bipartite co-attractor) in which a shared professional vocabulary makes practitioners cohesive as a group, and the cohesive group sustains the vocabulary. This co-attractor concept enables a zero-assumption method for detecting occupational emergence from resume data, requiring no predefined taxonomy or job titles: we test vocabulary cohesion and population cohesion independently, with ablation to test whether the vocabulary is the mechanism binding the population. Applied to 8.2 million US resumes (2022-2026), the method correctly identifies established occupations and reveals a striking asymmetry for AI: a cohesive professional vocabulary formed rapidly in early 2024, but the practitioner population never cohered. The pre-existing AI community dissolved as the tools went mainstream, and the new vocabulary was absorbed into existing careers rather than binding a new occupation. AI appears to be a diffusing technology, not an emerging occupation. We discuss whether introducing an "AI Engineer" occupational category could catalyze population cohesion around the already-formed vocabulary, completing the co-attractor. 

---
# Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning 

**Authors**: Jingxiang Chen, Minseok Kim, Seong-Gyun Leem, Yin Huang, Rashi Rungta, Zhicheng Ouyang, Haibin Wu, Surya Teja Appini, Ankur Bansal, Yang Bai, Yue Liu, Florian Metze, Ahmed A Aly, Anuj Kumar, Ariya Rastrow, Zhaojiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.15981)  

**Abstract**: Speech large language models (LLMs) observe paralinguistic cues such as prosody, emotion, and non-verbal sounds--crucial for intent understanding. However, leveraging these cues faces challenges: limited training data, annotation difficulty, and models exploiting lexical shortcuts over paralinguistic signals. We propose multi-task reinforcement learning (RL) with chain-of-thought prompting that elicits explicit affective reasoning. To address data scarcity, we introduce a paralinguistics-aware speech LLM (PALLM) that jointly optimizes sentiment classification from audio and paralinguistics-aware response generation via a two-stage pipeline. Experiments demonstrate that our approach improves paralinguistics understanding over both supervised baselines and strong proprietary models (Gemini-2.5-Pro, GPT-4o-audio) by 8-12% on Expresso, IEMOCAP, and RAVDESS. The results show that modeling paralinguistic reasoning with multi-task RL is crucial for building emotionally intelligent speech LLMs. 

---
# Robust Language Identification for Romansh Varieties 

**Authors**: Charlotte Model, Sina Ahmadi, Jannis Vamvas  

**Link**: [PDF](https://arxiv.org/pdf/2603.15969)  

**Abstract**: The Romansh language has several regional varieties, called idioms, which sometimes have limited mutual intelligibility. Despite this linguistic diversity, there has been a lack of documented efforts to build a language identification (LID) system that can distinguish between these idioms. Since Romansh LID should also be able to recognize Rumantsch Grischun, a supra-regional variety that combines elements of several idioms, this makes for a novel and interesting classification problem. In this paper, we present a LID system for Romansh idioms based on an SVM approach. We evaluate our model on a newly curated benchmark across two domains and find that it reaches an average in-domain accuracy of 97%, enabling applications such as idiom-aware spell checking or machine translation. Our classifier is publicly available. 

---
# MoLoRA: Composable Specialization via Per-Token Adapter Routing 

**Authors**: Shrey Shah, Justin Wagle  

**Link**: [PDF](https://arxiv.org/pdf/2603.15965)  

**Abstract**: Multi-adapter serving systems route entire sequences to a single adapter, forcing a choice when requests span multiple domains. This assumption fails in two important settings: (1) multimodal generation, where text and image tokens require different adapters within the same sequence, and (2) mixed-capability requests like "write code to solve this equation," which need expertise from multiple specialized adapters. We introduce per-token routing, which routes individual tokens to adapters based on either vocabulary structure (for multimodal models) or learned gating (for semantic specialization). Per-token routing is provably optimal, achieving work N for N tokens versus K \cdot N for per-sequence routing with K adapter types. Our key contribution is MoLoRA (Mixture of LoRA), which enables composable specialization: load multiple domain-specific adapters and let a learned router select the appropriate adapter per-token. We demonstrate that specialization dramatically beats scale: MoLoRA enables Qwen3-1.7B to exceed Qwen3-8B across four reasoning benchmarks while being 4.7x smaller. This enables modular expertise at inference time: train focused LoRAs independently, combine them without retraining, and add new capabilities by simply loading new adapters. 

---
# A Family of LLMs Liberated from Static Vocabularies 

**Authors**: Aleph Alpha, Adnen Abdessaied, Artur Baranowski, Lukas Balles, Michael Barlow, Fabien C. Y. Benureau, Felix Berkenkamp, Lukas Bluebaum, Bastian Boll, Thomas F. Burns, Björn Deiseroth, Constantin Eichenberg, David Friede, Pablo Iyu Guerrero, Ahmed Hammam, Bastian Harren, Johann Higl, Yasser Jadidi, Carina Kauf, Johannes Messner, Jan Hendrik Metzen, Max Meuer, Vedant Nanda, Pit Neitemeier, Koen Oostermeijer, Letitia Parcalabescu, Markus Pernpointner, Felix Reinfurt, Dylan Rodriquez, Grégory Schott, Philipp Siedler, Martin Simonovsky, Till Speicher, Volker Stampa, Stephan Wäldchen, Samuel Weinbach, Gregor Ziegltrum  

**Link**: [PDF](https://arxiv.org/pdf/2603.15953)  

**Abstract**: Tokenization is a central component of natural language processing in current large language models (LLMs), enabling models to convert raw text into processable units. Although learned tokenizers are widely adopted, they exhibit notable limitations, including their large, fixed vocabulary sizes and poor adaptability to new domains or languages. We present a family of models with up to 70 billion parameters based on the hierarchical autoregressive transformer (HAT) architecture. In HAT, an encoder transformer aggregates bytes into word embeddings and then feeds them to the backbone, a classical autoregressive transformer. The outputs of the backbone are then cross-attended by the decoder and converted back into bytes. We show that we can reuse available pre-trained models by converting the Llama 3.1 8B and 70B models into the HAT architecture: Llama-3.1-8B-TFree-HAT and Llama-3.1-70B-TFree-HAT are byte-level models whose encoder and decoder are trained from scratch, but where we adapt the pre-trained Llama backbone, i.e., the transformer blocks with the embedding matrix and head removed, to handle word embeddings instead of the original tokens. We also provide a 7B HAT model, Llama-TFree-HAT-Pretrained, trained entirely from scratch on nearly 4 trillion words. The HAT architecture improves text compression by reducing the number of required sequence positions and enhances robustness to intra-word variations, e.g., spelling differences. Through pre-training, as well as subsequent supervised fine-tuning and direct preference optimization in English and German, we show strong proficiency in both languages, improving on the original Llama 3.1 in most benchmarks. We release our models (including 200 pre-training checkpoints) on Hugging Face. 

---
# POLAR:A Per-User Association Test in Embedding Space 

**Authors**: Pedro Bento, Arthur Buzelin, Arthur Chagas, Yan Aquino, Victoria Estanislau, Samira Malaquias, Pedro Robles Dutenhefner, Gisele L. Pappa, Virgilio Almeida, Wagner MeiraJr  

**Link**: [PDF](https://arxiv.org/pdf/2603.15950)  

**Abstract**: Most intrinsic association probes operate at the word, sentence, or corpus level, obscuring author-level variation. We present POLAR (Per-user On-axis Lexical Association Re-port), a per-user lexical association test that runs in the embedding space of a lightly adapted masked language model. Authors are represented by private deterministic to-kens; POLAR projects these vectors onto curated lexicalaxes and reports standardized effects with permutation p-values and Benjamini--Hochberg control. On a balanced bot--human Twitter benchmark, POLAR cleanly separates LLM-driven bots from organic accounts; on an extremist forum,it quantifies strong alignment with slur lexicons and reveals rightward drift over time. The method is modular to new attribute sets and provides concise, per-author diagnostics for computational social science. All code is publicly avail-able at this https URL. 

---
# BANGLASOCIALBENCH: A Benchmark for Evaluating Sociopragmatic and Cultural Alignment of LLMs in Bangladeshi Social Interaction 

**Authors**: Tanvir Ahmed Sijan, S. M Golam Rifat, Pankaj Chowdhury Partha, Md. Tanjeed Islam, Md. Musfique Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2603.15949)  

**Abstract**: Large Language Models have demonstrated strong multilingual fluency, yet fluency alone does not guarantee socially appropriate language use. In high-context languages, communicative competence requires sensitivity to social hierarchy, relational roles, and interactional norms that are encoded directly in everyday language. Bangla exemplifies this challenge through its three-tiered pronominal system, kinship-based addressing, and culturally embedded social customs. We introduce BANGLASOCIALBENCH, the first benchmark designed to evaluate sociopragmatic competence in Bangla through context-dependent language use rather than factual recall. The benchmark spans three domains: Bangla Address Terms, Kinship Reasoning, and Social Customs, and consists of 1,719 culturally grounded instances written and verified by native Bangla speakers. We evaluate twelve contemporary LLMs in a zero-shot setting and observe systematic patterns of cultural misalignment. Models frequently default to overly formal address forms, fail to recognize multiple socially acceptable address pronouns, and conflate kinship terminology across religious contexts. Our findings show that sociopragmatic failures are often structured and non-random, revealing persistent limitations in how current LLMs infer and apply culturally appropriate language use in realistic Bangladeshi social interactions. 

---
# CTG-DB: An Ontology-Based Transformation of ClinicalTrials.gov to Enable Cross-Trial Drug Safety Analyses 

**Authors**: Jeffery L. Painter, François Haguinet, Andrew Bate  

**Link**: [PDF](https://arxiv.org/pdf/2603.15936)  

**Abstract**: this http URL (this http URL) is the largest publicly accessible registry of clinical studies, yet its registry-oriented architecture and heterogeneous adverse event (AE) terminology limit systematic pharmacovigilance (PV) analytics. AEs are typically recorded as investigator-reported text rather than standardized identifiers, requiring manual reconciliation to identify coherent safety concepts. We present the this http URL Transformation Database (CTG-DB), an open-source pipeline that ingests the complete this http URL XML archive and produces a relational database aligned to standardized AE terminology using the Medical Dictionary for Regulatory Activities (MedDRA). CTG-DB preserves arm-level denominators, represents placebo and comparator arms, and normalizes AE terminology using deterministic exact and fuzzy matching to ensure transparent and reproducible mappings. This framework enables concept-level retrieval and cross-trial aggregation for scalable placebo-referenced safety analyses and integration of clinical trial evidence into downstream PV signal detection. 

---
# Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies 

**Authors**: Nathaniel Imel, Richard Futrell, Michael Franke, Noga Zaslavsky  

**Link**: [PDF](https://arxiv.org/pdf/2603.15903)  

**Abstract**: Natural languages have been argued to evolve under pressure to efficiently compress meanings into words by optimizing the Information Bottleneck (IB) complexity-accuracy tradeoff. However, the underlying social dynamics that could drive the optimization of a language's vocabulary towards efficiency remain largely unknown. In parallel, evolutionary game theory has been invoked to explain the emergence of language from rudimentary agent-level dynamics, but it has not yet been tested whether such an approach can lead to efficient compression in the IB sense. Here, we provide a unified model integrating evolutionary game theory with the IB framework and show how near-optimal compression can arise in a population through an independently motivated dynamic of imprecise strategy imitation in signaling games. We find that key parameters of the model -- namely, those that regulate precision in these games, as well as players' tendency to confuse similar states -- lead to constrained variation of the tradeoffs achieved by emergent vocabularies. Our results suggest that evolutionary game dynamics could potentially provide a mechanistic basis for the evolution of vocabularies with information-theoretically optimal and empirically attested properties. 

---
# COGNAC at SemEval-2026 Task 5: LLM Ensembles for Human-Level Word Sense Plausibility Rating in Challenging Narratives 

**Authors**: Azwad Anjum Islam, Tisa Islam Erana  

**Link**: [PDF](https://arxiv.org/pdf/2603.15897)  

**Abstract**: We describe our system for SemEval-2026 Task 5, which requires rating the plausibility of given word senses of homonyms in short stories on a 5-point Likert scale. Systems are evaluated by the unweighted average of accuracy (within one standard deviation of mean human judgments) and Spearman Rank Correlation. We explore three prompting strategies using multiple closed-source commercial LLMs: (i) a baseline zero-shot setup, (ii) Chain-of-Thought (CoT) style prompting with structured reasoning, and (iii) a comparative prompting strategy for evaluating candidate word senses simultaneously. Furthermore, to account for the substantial inter-annotator variation present in the gold labels, we propose an ensemble setup by averaging model predictions. Our best official system, comprising an ensemble of LLMs across all three prompting strategies, placed 4th on the competition leaderboard with 0.88 accuracy and 0.83 Spearman's rho (0.86 average). Post-competition experiments with additional models further improved this performance to 0.92 accuracy and 0.85 Spearman's rho (0.89 average). We find that comparative prompting consistently improved performance across model families, and model ensembling significantly enhanced alignment with mean human judgments, suggesting that LLM ensembles are especially well suited for subjective semantic evaluation tasks involving multiple annotators. 

---
# Morphemes Without Borders: Evaluating Root-Pattern Morphology in Arabic Tokenizers and LLMs 

**Authors**: Yara Alakeel, Chatrine Qwaider, Hanan Aldarmaki, Sawsan Alqahtani  

**Link**: [PDF](https://arxiv.org/pdf/2603.15773)  

**Abstract**: This work investigates how effectively large language models (LLMs) and their tokenization schemes represent and generate Arabic root-pattern morphology, probing whether they capture genuine morphological structure or rely on surface memorization. Arabic morphological system provides a rich testbed for analyzing how LLMs handle complex, non-concatenative forms and how tokenization choices influence this process. Our study begins with an evaluation of morphological fidelity across Arabic and multilingual tokenizers against gold-standard segmentation, followed by an analysis of LLM performance in productive root-pattern generation using a newly developed test set. Our findings across seven Arabic-centric and multilingual LLMs and their respective tokenizers reveal that tokenizer morphological alignment is not necessary nor sufficient for morphological generation, which questions the role of morphological tokenization in downstream performance. 

---
# MiroThinker-1.7 & H1: Towards Heavy-Duty Research Agents via Verification 

**Authors**: MiroMind Team, S. Bai, L. Bing, L. Lei, R. Li, X. Li, X. Lin, E. Min, L. Su, B. Wang, L. Wang, L. Wang, S. Wang, X. Wang, Y. Zhang, Z. Zhang, G. Chen, L. Chen, Z. Cheng, Y. Deng, Z. Huang, D. Ng, J. Ni, Q. Ren, X. Tang, B.L. Wang, H. Wang, N. Wang, C. Wei, Q. Wu, J. Xia, Y. Xiao, H. Xu, X. Xu, C. Xue, Z. Yang, Z. Yang, F. Ye, H. Ye, J. Yu, C. Zhang, W. Zhang, H. Zhao, P. Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.15726)  

**Abstract**: We present MiroThinker-1.7, a new research agent designed for complex long-horizon reasoning tasks. Building on this foundation, we further introduce MiroThinker-H1, which extends the agent with heavy-duty reasoning capabilities for more reliable multi-step problem solving. In particular, MiroThinker-1.7 improves the reliability of each interaction step through an agentic mid-training stage that emphasizes structured planning, contextual reasoning, and tool interaction. This enables more effective multi-step interaction and sustained reasoning across complex tasks. MiroThinker-H1 further incorporates verification directly into the reasoning process at both local and global levels. Intermediate reasoning decisions can be evaluated and refined during inference, while the overall reasoning trajectory is audited to ensure that final answers are supported by coherent chains of evidence. Across benchmarks covering open-web research, scientific reasoning, and financial analysis, MiroThinker-H1 achieves state-of-the-art performance on deep research tasks while maintaining strong results on specialized domains. We also release MiroThinker-1.7 and MiroThinker-1.7-mini as open-source models, providing competitive research-agent capabilities with significantly improved efficiency. 

---
# MedArena: Comparing LLMs for Medicine-in-the-Wild Clinician Preferences 

**Authors**: Eric Wu, Kevin Wu, Jason Hom, Paul H. Yi, Angela Zhang, Alejandro Lozano, Jeff Nirschl, Jeff Tangney, Kevin Byram, Braydon Dymm, Narender Annapureddy, Eric Topol, David Ouyang, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.15677)  

**Abstract**: Large language models (LLMs) are increasingly central to clinician workflows, spanning clinical decision support, medical education, and patient communication. However, current evaluation methods for medical LLMs rely heavily on static, templated benchmarks that fail to capture the complexity and dynamics of real-world clinical practice, creating a dissonance between benchmark performance and clinical utility. To address these limitations, we present MedArena, an interactive evaluation platform that enables clinicians to directly test and compare leading LLMs using their own medical queries. Given a clinician-provided query, MedArena presents responses from two randomly selected models and asks the user to select the preferred response. Out of 1571 preferences collected across 12 LLMs up to November 1, 2025, Gemini 2.0 Flash Thinking, Gemini 2.5 Pro, and GPT-4o were the top three models by Bradley-Terry rating. Only one-third of clinician-submitted questions resembled factual recall tasks (e.g., MedQA), whereas the majority addressed topics such as treatment selection, clinical documentation, or patient communication, with ~20% involving multi-turn conversations. Additionally, clinicians cited depth and detail and clarity of presentation more often than raw factual accuracy when explaining their preferences, highlighting the importance of readability and clinical nuance. We also confirm that the model rankings remain stable even after controlling for style-related factors like response length and formatting. By grounding evaluation in real-world clinical questions and preferences, MedArena offers a scalable platform for measuring and improving the utility and efficacy of medical LLMs. 

---
# Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context 

**Authors**: Keivan Alizadeh, Parshin Shojaee, Minsik Cho, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2603.15653)  

**Abstract**: Long-context handling remains a core challenge for language models: even with extended context windows, models often fail to reliably extract, reason over, and use the information across long contexts. Recent works like Recursive Language Models (RLM) have approached this challenge by agentic way of decomposing long contexts into recursive sub-calls through programmatic interaction at inference. While promising, the success of RLM critically depends on how these context-interaction programs are selected, which has remained largely unexplored. In this paper, we study this problem and introduce SRLM, a framework that augments programmatic context interaction with uncertainty-aware Self-Reflection. SRLM leverages three intrinsic signals: self consistency, reasoning length, and verbalized confidence. These serve as complementary indicators of a model's internal uncertainty, and the model uses them to evaluate and compare candidate context-interaction programs. Extensive experiments across diverse benchmark datasets, context lengths, and backbone models, show that SRLM consistently outperforms state-of-the-art baselines, yielding up to 22% improvement over RLM under the same time budget. Our findings show that recursion itself is not the primary driver of performance in RLM, and a simple self-reflective program search can match or surpass RLM without requiring self-query or explicit recursion mechanisms. We find that for context lengths within the model's window, RLMs with recursion often degrade performance relative to the base model, whereas SRLM yields consistent gains across both short and long contexts. We also find that RLM is less effective in tasks with semantically intensive nature, where heuristic program search is insufficient and broader contextual understanding is required, while self-reflection in SRLM provides a semantic signal that better steers reasoning in these scenarios. 

---
# Efficient Reasoning on the Edge 

**Authors**: Yelysei Bondarenko, Thomas Hehn, Rob Hesselink, Romain Lepert, Fabio Valerio Massoli, Evgeny Mironov, Leyla Mirvakhabova, Tribhuvanesh Orekondy, Spyridon Stasis, Andrey Kuzmin, Anna Kuzina, Markus Nagel, Ankita Nayak, Corrado Rainone, Ork de Rooij, Paul N Whatmough, Arash Behboodi, Babak Ehteshami Bejnordi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16867)  

**Abstract**: Large language models (LLMs) with chain-of-thought reasoning achieve state-of-the-art performance across complex problem-solving tasks, but their verbose reasoning traces and large context requirements make them impractical for edge deployment. These challenges include high token generation costs, large KV-cache footprints, and inefficiencies when distilling reasoning capabilities into smaller models for mobile devices. Existing approaches often rely on distilling reasoning traces from larger models into smaller models, which are verbose and stylistically redundant, undesirable for on-device inference. In this work, we propose a lightweight approach to enable reasoning in small LLMs using LoRA adapters combined with supervised fine-tuning. We further introduce budget forcing via reinforcement learning on these adapters, significantly reducing response length with minimal accuracy loss. To address memory-bound decoding, we exploit parallel test-time scaling, improving accuracy at minor latency increase. Finally, we present a dynamic adapter-switching mechanism that activates reasoning only when needed and a KV-cache sharing strategy during prompt encoding, reducing time-to-first-token for on-device inference. Experiments on Qwen2.5-7B demonstrate that our method achieves efficient, accurate reasoning under strict resource constraints, making LLM reasoning practical for mobile scenarios. Videos demonstrating our solution running on mobile devices are available on our project page. 

---
# Prompt Programming for Cultural Bias and Alignment of Large Language Models 

**Authors**: Maksim Eren, Eric Michalak, Brian Cook, Johnny Seales Jr  

**Link**: [PDF](https://arxiv.org/pdf/2603.16827)  

**Abstract**: Culture shapes reasoning, values, prioritization, and strategic decision-making, yet large language models (LLMs) often exhibit cultural biases that misalign with target populations. As LLMs are increasingly used for strategic decision-making, policy support, and document engineering tasks such as summarization, categorization, and compliance-oriented auditing, improving cultural alignment is important for ensuring that downstream analyses and recommendations reflect target-population value profiles rather than default model priors. Previous work introduced a survey-grounded cultural alignment framework and showed that culture-specific prompting can reduce misalignment, but it primarily evaluated proprietary models and relied on manual prompt engineering. In this paper, we validate and extend that framework by reproducing its social sciences survey based projection and distance metrics on open-weight LLMs, testing whether the same cultural skew and benefits of culture conditioning persist outside closed LLM systems. Building on this foundation, we introduce use of prompt programming with DSPy for this problem-treating prompts as modular, optimizable programs-to systematically tune cultural conditioning by optimizing against cultural-distance objectives. In our experiments, we show that prompt optimization often improves upon cultural prompt engineering, suggesting prompt compilation with DSPy can provide a more stable and transferable route to culturally aligned LLM responses. 

---
# Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights 

**Authors**: Yi Chen, Daiwei Chen, Sukrut Madhav Chikodikar, Caitlyn Heqi Yin, Ramya Korlakai Vinayak  

**Link**: [PDF](https://arxiv.org/pdf/2603.16817)  

**Abstract**: Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs across generation, scoring, calibration, robustness, and efficiency. We propose novel informativeness-aware metrics that better reflect task utility under conformal filtering. Across three benchmarks and multiple model families, we find that (i) conformal filtering suffers from low usefulness at high factuality levels due to vacuous outputs, (ii) conformal factuality guarantee is not robust to distribution shifts and distractors, highlighting the limitation that requires calibration data to closely match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform LLM-based model confidence scorers while requiring over $100\times$ fewer FLOPs. Overall, our results expose factuality-informativeness trade-offs and fragility of conformal filtering framework under distribution shifts and distractors, highlighting the need for new approaches for reliability with robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines that are both reliable and computationally efficient. 

---
# SOMP: Scalable Gradient Inversion for Large Language Models via Subspace-Guided Orthogonal Matching Pursuit 

**Authors**: Yibo Li, Qiongxiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.16761)  

**Abstract**: Gradient inversion attacks reveal that private training text can be reconstructed from shared gradients, posing a privacy risk to large language models (LLMs). While prior methods perform well in small-batch settings, scaling to larger batch sizes and longer sequences remains challenging due to severe signal mixing, high computational cost, and degraded fidelity. We present SOMP (Subspace-Guided Orthogonal Matching Pursuit), a scalable gradient inversion framework that casts text recovery from aggregated gradients as a sparse signal recovery problem. Our key insight is that aggregated transformer gradients retain exploitable head-wise geometric structure together with sample-level sparsity. SOMP leverages these properties to progressively narrow the search space and disentangle mixed signals without exhaustive search. Experiments across multiple LLM families, model scales, and five languages show that SOMP consistently outperforms prior methods in the aggregated-gradient this http URL long sequences at batch size B=16, SOMP achieves substantially higher reconstruction fidelity than strong baselines, while remaining computationally competitive. Even under extreme aggregation (up to B=128), SOMP still recovers meaningful text, suggesting that privacy leakage can persist in regimes where prior attacks become much less effective. 

---
# Retrieving Counterfactuals Improves Visual In-Context Learning 

**Authors**: Guangzhi Xiong, Sanchit Sinha, Zhenghao He, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16737)  

**Abstract**: Vision-language models (VLMs) have achieved impressive performance across a wide range of multimodal reasoning tasks, but they often struggle to disentangle fine-grained visual attributes and reason about underlying causal relationships. In-context learning (ICL) offers a promising avenue for VLMs to adapt to new tasks, but its effectiveness critically depends on the selection of demonstration examples. Existing retrieval-augmented approaches typically rely on passive similarity-based retrieval, which tends to select correlated but non-causal examples, amplifying spurious associations and limiting model robustness. We introduce CIRCLES (Composed Image Retrieval for Causal Learning Example Selection), a novel framework that actively constructs demonstration sets by retrieving counterfactual-style examples through targeted, attribute-guided composed image retrieval. By incorporating counterfactual-style examples, CIRCLES enables VLMs to implicitly reason about the causal relations between attributes and outcomes, moving beyond superficial correlations and fostering more robust and grounded reasoning. Comprehensive experiments on four diverse datasets demonstrate that CIRCLES consistently outperforms existing methods across multiple architectures, especially on small-scale models, with pronounced gains under information scarcity. Furthermore, CIRCLES retrieves more diverse and causally informative examples, providing qualitative insights into how models leverage in-context demonstrations for improved reasoning. Our code is available at this https URL. 

---
# IQuest-Coder-V1 Technical Report 

**Authors**: Jian Yang, Wei Zhang, Shawn Guo, Zhengmao Ye, Lin Jing, Shark Liu, Yizhi Li, Jiajun Wu, Cening Liu, X. Ma, Yuyang Song, Siwei Wu, Yuwen Li, L. Liao, T. Zheng, Ziling Huang, Zelong Huang, Che Liu, Yan Xing, Renyuan Li, Qingsong Cai, Hanxu Yan, Siyue Wang, Shikai Li, Jason Klein Liu, An Huang, Yongsheng Kang, Jinxing Zhang, Chuan Hao, Haowen Wang, Weicheng Gu, Ran Tao, Mingjie Tang, Peihao Wu, Jianzhou Wang, Xianglong Liu, Weifeng Lv, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2603.16733)  

**Abstract**: In this report, we introduce the IQuest-Coder-V1 series-(7B/14B/40B/40B-Loop), a new family of code large language models (LLMs). Moving beyond static code representations, we propose the code-flow multi-stage training paradigm, which captures the dynamic evolution of software logic through different phases of the pipeline. Our models are developed through the evolutionary pipeline, starting with the initial pre-training consisting of code facts, repository, and completion data. Following that, we implement a specialized mid-training stage that integrates reasoning and agentic trajectories in 32k-context and repository-scale in 128k-context to forge deep logical foundations. The models are then finalized with post-training of specialized coding capabilities, which is bifurcated into two specialized paths: the thinking path (utilizing reasoning-driven RL) and the instruct path (optimized for general assistance). IQuest-Coder-V1 achieves state-of-the-art performance among competitive models across critical dimensions of code intelligence: agentic software engineering, competitive programming, and complex tool use. To address deployment constraints, the IQuest-Coder-V1-Loop variant introduces a recurrent mechanism designed to optimize the trade-off between model capacity and deployment footprint, offering an architecturally enhanced path for efficacy-efficiency trade-off. We believe the release of the IQuest-Coder-V1 series, including the complete white-box chain of checkpoints from pre-training bases to the final thinking and instruction models, will advance research in autonomous code intelligence and real-world agentic systems. 

---
# CritiSense: Critical Digital Literacy and Resilience Against Misinformation 

**Authors**: Firoj Alam, Fatema Ahmad, Ali Ezzat Shahroor, Mohamed Bayan Kmainasi, Elisa Sartori, Giovanni Da San Martino, Abul Hasnat, Raian Ali  

**Link**: [PDF](https://arxiv.org/pdf/2603.16672)  

**Abstract**: Misinformation on social media undermines informed decision-making and public trust. Prebunking offers a proactive complement by helping users recognize manipulation tactics before they encounter them in the wild. We present CritiSense, a mobile media-literacy app that builds these skills through short, interactive challenges with instant feedback. It is the first multilingual (supporting nine languages) and modular platform, designed for rapid updates across topics and domains. We report a usability study with 93 users: 83.9% expressed overall satisfaction and 90.1% rated the app as easy to use. Qualitative feedback indicates that CritiSense helps improve digital literacy skills. Overall, it provides a multilingual prebunking platform and a testbed for measuring the impact of microlearning on misinformation resilience. Over 3+ months, we have reached 300+ active users. It is freely available to all users on the Apple App Store (this https URL) and Google Play Store (this https URL). Demo Video: this https URL 

---
# When AI Navigates the Fog of War 

**Authors**: Ming Li, Xirui Li, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.16642)  

**Abstract**: Can AI reason about a war before its trajectory becomes historically obvious? Analyzing this capability is difficult because retrospective geopolitical prediction is heavily confounded by training-data leakage. We address this challenge through a temporally grounded case study of the early stages of the 2026 Middle East conflict, which unfolded after the training cutoff of current frontier models. We construct 11 critical temporal nodes, 42 node-specific verifiable questions, and 5 general exploratory questions, requiring models to reason only from information that would have been publicly available at each moment. This design substantially mitigates training-data leakage concerns, creating a setting well-suited for studying how models analyze an unfolding crisis under the fog of war, and provides, to our knowledge, the first temporally grounded analysis of LLM reasoning in an ongoing geopolitical conflict. Our analysis reveals three main findings. First, current state-of-the-art large language models often display a striking degree of strategic realism, reasoning beyond surface rhetoric toward deeper structural incentives. Second, this capability is uneven across domains: models are more reliable in economically and logistically structured settings than in politically ambiguous multi-actor environments. Finally, model narratives evolve over time, shifting from early expectations of rapid containment toward more systemic accounts of regional entrenchment and attritional de-escalation. Since the conflict remains ongoing at the time of writing, this work can serve as an archival snapshot of model reasoning during an unfolding geopolitical crisis, enabling future studies without the hindsight bias of retrospective analysis. 

---
# When and Why Does Unsupervised RL Succeed in Mathematical Reasoning? A Manifold Envelopment Perspective 

**Authors**: Zelin Zhang, Fei Cheng, Chenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16578)  

**Abstract**: Although outcome-based reinforcement learning (RL) significantly advances the mathematical reasoning capabilities of Large Language Models (LLMs), its reliance on computationally expensive ground-truth annotations imposes a severe scalability bottleneck. Unsupervised RL guided by intrinsic rewards offers a scalable alternative, yet it suffers from opaque training dynamics and catastrophic instability, such as policy collapse and reward hacking. In this paper, we first design and evaluate a suite of intrinsic rewards that explicitly enforce concise and certain generation. Second, to discover the boundaries of this approach, we test base models across a spectrum of intrinsic reasoning capabilities, revealing how a model's foundational logical prior dictates its success or failure. Finally, to demystify why certain configurations stabilize while others collapse, we introduce a novel geometric diagnostic lens, showing that successful cases are enveloped by manifolds. Ultimately, our work goes beyond merely demonstrating that enforcing concise and certain responses successfully boosts mathematical reasoning; we reveal when this unsupervised approach breaks down and geometrically diagnose why. 

---
# BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs 

**Authors**: Sangyeon Yoon, Sunkyoung Kim, Hyesoo Hong, Wonje Jeung, Yongil Kim, Wooseok Seo, Heuiyeen Yeen, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2603.16557)  

**Abstract**: Large language models (LLMs) increasingly store user preferences in persistent memory to support personalization across interactions. However, in third-party communication settings governed by social and institutional norms, some user preferences may be inappropriate to apply. We introduce BenchPreS, which evaluates whether memory-based user preferences are appropriately applied or suppressed across communication contexts. Using two complementary metrics, Misapplication Rate (MR) and Appropriate Application Rate (AAR), we find even frontier LLMs struggle to apply preferences in a context-sensitive manner. Models with stronger preference adherence exhibit higher rates of over-application, and neither reasoning capability nor prompt-based defenses fully resolve this issue. These results suggest current LLMs treat personalized preferences as globally enforceable rules rather than as context-dependent normative signals. 

---
# From the Inside Out: Progressive Distribution Refinement for Confidence Calibration 

**Authors**: Xizhong Yang, Yinan Xia, Huiming Wang, Mofei Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.16500)  

**Abstract**: Leveraging the model's internal information as the self-reward signal in Reinforcement Learning (RL) has received extensive attention due to its label-free nature. While prior works have made significant progress in applying the Test-Time Scaling (TTS) strategies to RL, the discrepancy in internal information between test and training remains inadequately addressed. Moreover, Test-Time Training based on voting-based TTS strategies often suffers from reward hacking problems. To address these issues, we propose DistriTTRL, which leverages the distribution prior of the model's confidence during RL to progressively optimize the reward signal, rather than relying solely on single-query rollouts. Additionally, we mitigate the phenomenon of consistent reward hacking caused by the voting-based TTS strategies through diversity-targeted penalties. Benefiting from this training mechanism where model capability and self-reward signals complement each other, and the mitigation of reward hacking, DistriTTRL has achieved significant performance improvements across multiple models and benchmarks. 

---
# Capability-Guided Compression: Toward Interpretability-Aware Budget Allocation for Large Language Models 

**Authors**: Rishaank Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2603.16440)  

**Abstract**: Large language model compression has made substantial progress through pruning, quantization, and low-rank decomposition, yet a fundamental limitation persists across all existing methods: compression budgets are allocated without any representation of what individual model components functionally encode. We term this the capability-blind compression problem and argue it is a root cause of two well-documented failures -- the insensitivity of perplexity-based evaluation to reasoning capability loss, and the abrupt phase transitions in model performance recently characterized by Ma et al. (2026). We propose Capability-Guided Compression (CGC), a framework that addresses this by using Sparse Autoencoder (SAE)-derived capability density maps to allocate differential compression budgets across transformer components. Capability density is a formally defined scalar measure combining the feature breadth, activation entropy, and cross-input consistency of a component's SAE feature activation distribution. We prove theoretically that components with higher capability density exhibit lower structural redundancy and reach their individual phase transition points at lower compression ratios, providing the first pre-compression mechanism for component-level phase transition prediction. Experiments on GPT-2 Medium confirm that capability density is statistically independent of Wanda importance scores (Spearman rho = -0.054, n = 384 heads), establishing it as a genuinely novel compression signal orthogonal to all existing importance metrics. We report a negative result on PPL-based compression comparison and provide a principled diagnosis identifying GPT-2 Medium as an insufficient test bed for the full CGC hypothesis. The theoretical framework, density formalism, and orthogonality finding constitute a foundation for capability-aware compression research. 

---
# Behavioral Steering in a 35B MoE Language Model via SAE-Decoded Probe Vectors: One Agency Axis, Not Five Traits 

**Authors**: Jia Qing Yap  

**Link**: [PDF](https://arxiv.org/pdf/2603.16335)  

**Abstract**: We train nine sparse autoencoders (SAEs) on the residual stream of Qwen 3.5-35B-A3B, a 35-billion-parameter Mixture-of-Experts model with a hybrid GatedDeltaNet/attention architecture, and use them to identify and steer five agentic behavioral traits. Our method trains linear probes on SAE latent activations, then projects the probe weights back through the SAE decoder to obtain continuous steering vectors in the model's native activation space. This bypasses the SAE's top-k discretization, enabling fine-grained behavioral intervention at inference time with no retraining. Across 1,800 agent rollouts (50 scenarios times 36 conditions), we find that autonomy steering at multiplier 2 achieves Cohen's d = 1.01 (p < 0.0001), shifting the model from asking the user for help 78% of the time to proactively executing code and searching the web. Cross-trait analysis, however, reveals that all five steering vectors primarily modulate a single dominant agency axis (the disposition to act independently versus defer to the user), with trait specific effects appearing only as secondary modulations in tool-type composition and dose-response shape. The tool-use vector steers behavior (d = 0.39); the risk-calibration vector produces only suppression. We additionally show that steering only during autoregressive decoding has zero effect (p > 0.35), providing causal evidence that behavioral commitments are computed during prefill in GatedDeltaNet architectures. 

---
# How to Utilize Complementary Vision-Text Information for 2D Structure Understanding 

**Authors**: Jiancheng Dong, Pengyue Jia, Derong Xu, Jiawei Cheng, Jingyu Peng, Chao Zhang, Bowen Liu, Xin Sun, Lixin Su, Shuaiqiang Wang, Dawei Yin, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16245)  

**Abstract**: LLMs typically linearize 2D tables into 1D sequences to fit their autoregressive architecture, which weakens row-column adjacency and other layout cues. In contrast, purely visual encoders can capture spatial cues, yet often struggle to preserve exact cell text. Our analysis reveals that these two modalities provide highly distinct information to LLMs and exhibit strong complementarity. However, direct concatenation and other fusion methods yield limited gains and frequently introduce cross-modal interference. To address this issue, we propose DiVA-Former, a lightweight architecture designed to effectively integrate vision and text information. DiVA-Former leverages visual tokens as dynamic queries to distill long textual sequences into digest vectors, thereby effectively exploiting complementary vision--text information. Evaluated across 13 table benchmarks, DiVA-Former improves upon the pure-text baseline by 23.9\% and achieves consistent gains over existing baselines using visual inputs, textual inputs, or a combination of both. 

---
# Offline Exploration-Aware Fine-Tuning for Long-Chain Mathematical Reasoning 

**Authors**: Yongyu Mu, Jiali Zeng, Fandong Meng, JingBo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16206)  

**Abstract**: Through encouraging self-exploration, reinforcement learning from verifiable rewards (RLVR) has significantly advanced the mathematical reasoning capabilities of large language models. As the starting point for RLVR, the capacity of supervised fine-tuning (SFT) to memorize new chain-of-thought trajectories provides a crucial initialization that shapes the subsequent exploration landscape. However, existing research primarily focuses on facilitating exploration during RLVR training, leaving exploration-aware SFT under-explored. To bridge this gap, we propose Offline eXploration-Aware (OXA) fine-tuning. Specifically, OXA optimizes two objectives: promoting low-confidence verified teacher-distillation data to internalize previously uncaptured reasoning patterns, and suppressing high-confidence incorrect self-distillation data to redistribute probability mass of incorrect patterns toward potentially correct candidates. Experimental results across 6 benchmarks show that OXA consistently improves mathematical reasoning performance, especially achieving an average gain of $+6$ Pass@1 and $+5$ Pass@$k$ points compared to conventional SFT on the Qwen2.5-1.5B-Math. Crucially, OXA elevates initial policy entropy, and performance gains persist throughout extensive RLVR training, demonstrating the long-term value of OXA. 

---
# Are Large Language Models Truly Smarter Than Humans? 

**Authors**: Eshwar Reddy M, Sourav Karmakar  

**Link**: [PDF](https://arxiv.org/pdf/2603.16197)  

**Abstract**: Public leaderboards increasingly suggest that large language models (LLMs) surpass human experts on benchmarks spanning academic knowledge, law, and programming. Yet most benchmarks are fully public, their questions widely mirrored across the internet, creating systematic risk that models were trained on the very data used to evaluate them. This paper presents three complementary experiments forming a rigorous multi-method contamination audit of six frontier LLMs: GPT-4o, GPT-4o-mini, DeepSeek-R1, DeepSeek-V3, Llama-3.3-70B, and Qwen3-235B. Experiment 1 applies a lexical contamination detection pipeline to 513 MMLU questions across all 57 subjects, finding an overall contamination rate of 13.8% (18.1% in STEM, up to 66.7% in Philosophy) and estimated performance gains of +0.030 to +0.054 accuracy points by category. Experiment 2 applies a paraphrase and indirect-reference diagnostic to 100 MMLU questions, finding accuracy drops by an average of 7.0 percentage points under indirect reference, rising to 19.8 pp in both Law and Ethics. Experiment 3 applies TS-Guessing behavioral probes to all 513 questions and all six models, finding that 72.5% trigger memorization signals far above chance, with DeepSeek-R1 displaying a distributed memorization signature (76.6% partial reconstruction, 0% verbatim recall) that explains its anomalous Experiment 2 profile. All three experiments converge on the same contamination ranking: STEM > Professional > Social Sciences > Humanities. 

---
# Open-Source Reproduction and Explainability Analysis of Corrective Retrieval Augmented Generation 

**Authors**: Surya Vardhan Yalavarthi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16169)  

**Abstract**: Corrective Retrieval Augmented Generation (CRAG) improves the robustness of RAG systems by evaluating retrieved document quality and triggering corrective actions. However, the original implementation relies on proprietary components including the Google Search API and closed model weights, limiting reproducibility. In this work, we present a fully open-source reproduction of CRAG, replacing proprietary web search with the Wikipedia API and the original LLaMA-2 generator with Phi-3-mini-4k-instruct. We evaluate on PopQA and ARC-Challenge, demonstrating that our open-source pipeline achieves comparable performance to the original system. Furthermore, we contribute the first explainability analysis of CRAG's T5-based retrieval evaluator using SHAP, revealing that the evaluator primarily relies on named entity alignment rather than semantic similarity. Our analysis identifies key failure modes including domain transfer limitations on science questions. All code and results are available at this https URL. 

---
# STARK: Spatio-Temporal Attention for Representation of Keypoints for Continuous Sign Language Recognition 

**Authors**: Suvajit Patra, Soumitra Samanta  

**Link**: [PDF](https://arxiv.org/pdf/2603.16163)  

**Abstract**: Continuous Sign Language Recognition (CSLR) is a crucial task for understanding the languages of deaf communities. Contemporary keypoint-based approaches typically rely on spatio-temporal encoding, where spatial interactions among keypoints are modeled using Graph Convolutional Networks or attention mechanisms, while temporal dynamics are captured using 1D convolutional networks. However, such designs often introduce a large number of parameters in both the encoder and the decoder. This paper introduces a unified spatio-temporal attention network that computes attention scores both spatially (across keypoints) and temporally (within local windows), and aggregates features to produce a local context-aware spatio-temporal representation. The proposed encoder contains approximately $70-80\%$ fewer parameters than existing state-of-the-art models while achieving comparable performance to keypoint-based methods on the Phoenix-14T dataset. 

---
# HIPO: Instruction Hierarchy via Constrained Reinforcement Learning 

**Authors**: Keru Chen, Jun Luo, Sen Lin, Yingbin Liang, Alvaro Velasquez, Nathaniel Bastian, Shaofeng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.16152)  

**Abstract**: Hierarchical Instruction Following (HIF) refers to the problem of prompting large language models with a priority-ordered stack of instructions. Standard methods like RLHF and DPO typically fail in this problem since they mainly optimize for a single objective, failing to explicitly enforce system prompt compliance. Meanwhile, supervised fine-tuning relies on mimicking filtered, compliant data, which fails to establish the priority asymmetry at the algorithmic level. In this paper, we introduce \textsc{HIPO}, a novel alignment framework that formulates HIF as a Constrained Markov Decision Process. \textsc{HIPO} elevates system prompts from mere input context to strict algorithmic boundaries. Using a primal-dual safe reinforcement learning approach, the algorithm dynamically enforces system prompt compliance as an explicit constraint, maximizing user utility strictly within this feasible region. Extensive evaluations across diverse model architectures (e.g., Qwen, Phi, Llama) demonstrate that \textsc{HIPO} significantly improves both system compliance and user utility. Furthermore, mechanistic analysis reveals that this constrained optimization autonomously drives the model to shift its attention toward long-range system tokens, providing a principled foundation for reliable LLM deployment in complex workflows. 

---
# Answer Bubbles: Information Exposure in AI-Mediated Search 

**Authors**: Michelle Huang, Agam Goyal, Koustuv Saha, Eshwar Chandrasekharan  

**Link**: [PDF](https://arxiv.org/pdf/2603.16138)  

**Abstract**: Generative search systems are increasingly replacing link-based retrieval with AI-generated summaries, yet little is known about how these systems differ in sources, language, and fidelity to cited material. We examine responses to 11,000 real search queries across four systems -- vanilla GPT, Search GPT, Google AI Overviews, and traditional Google Search -- at three levels: source diversity, linguistic characterization of the generated summary, and source-summary fidelity. We find that generative search systems exhibit significant \textit{source-selection} biases in their citations, favoring certain sources over others. Incorporating search also selectively attenuates epistemic markers, reducing hedging by up to 60\% while preserving confidence language in the AI-generated summaries. At the same time, AI summaries further compound the citation biases: Wikipedia and longer sources are disproportionately overrepresented, whereas cited social media content and negatively framed sources are substantially underrepresented. Our findings highlight the potential for \textit{answer bubbles}, in which identical queries yield structurally different information realities across systems, with implications for user trust, source visibility, and the transparency of AI-mediated information access. 

---
# SWE-QA-Pro: A Representative Benchmark and Scalable Training Recipe for Repository-Level Code Understanding 

**Authors**: Songcheng Cai, Zhiheng Lyu, Yuansheng Ni, Xiangchao Chen, Baichuan Zhou, Shenzhe Zhu, Yi Lu, Haozhe Wang, Chi Ruan, Benjamin Schneider, Weixu Zhang, Xiang Li, Andy Zheng, Yuyu Zhang, Ping Nie, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.16124)  

**Abstract**: Agentic repository-level code understanding is essential for automating complex software engineering tasks, yet the field lacks reliable benchmarks. Existing evaluations often overlook the long tail topics and rely on popular repositories where Large Language Models (LLMs) can cheat via memorized knowledge. To address this, we introduce SWE-QA-Pro, a benchmark constructed from diverse, long-tail repositories with executable environments. We enforce topical balance via issue-driven clustering to cover under-represented task types and apply a rigorous difficulty calibration process: questions solvable by direct-answer baselines are filtered out. This results in a dataset where agentic workflows significantly outperform direct answering (e.g., a ~13-point gap for Claude Sonnet 4.5), confirming the necessity of agentic codebase exploration. Furthermore, to tackle the scarcity of training data for such complex behaviors, we propose a scalable synthetic data pipeline that powers a two-stage training recipe: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning from AI Feedback (RLAIF). This approach allows small open models to learn efficient tool usage and reasoning. Empirically, a Qwen3-8B model trained with our recipe surpasses GPT-4o by 2.3 points on SWE-QA-Pro and substantially narrows the gap to state-of-the-art proprietary models, demonstrating both the validity of our evaluation and the effectiveness of our agentic training workflow. 

---
# Resource Consumption Threats in Large Language Models 

**Authors**: Yuanhe Zhang, Xinyue Wang, Zhican Chen, Weiliu Wang, Zilu Zhang, Zhengshuo Gong, Zhenhong Zhou, Li Sun, Yang Liu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.16068)  

**Abstract**: Given limited and costly computational infrastructure, resource efficiency is a key requirement for large language models (LLMs). Efficient LLMs increase service capacity for providers and reduce latency and API costs for users. Recent resource consumption threats induce excessive generation, degrading model efficiency and harming both service availability and economic sustainability. This survey presents a systematic review of threats to resource consumption in LLMs. We further establish a unified view of this emerging area by clarifying its scope and examining the problem along the full pipeline from threat induction to mechanism understanding and mitigation. Our goal is to clarify the problem landscape for this emerging area, thereby providing a clearer foundation for characterization and mitigation. 

---
# Residual Stream Duality in Modern Transformer Architectures 

**Authors**: Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.16039)  

**Abstract**: Recent work has made clear that the residual pathway is not mere optimization plumbing; it is part of the model's representational machinery. We agree, but argue that the cleanest way to organize this design space is through a two-axis view of the Transformer. A decoder evolves information along two ordered dimensions: sequence position and layer depth. Self-attention already provides adaptive mixing along the sequence axis, whereas the residual stream usually performs fixed addition along the depth axis. If we fix a token position and treat layer index as the ordered variable, then a causal depth-wise residual attention read is exactly the same local operator as causal short sliding-window attention (ShortSWA), except written over depth rather than over sequence. This is the core residual stream duality behind Transformer$^2$. This perspective also clarifies the recent literature. ELC-BERT and DenseFormer already show that learned aggregation over depth can outperform uniform residual accumulation, while Vertical Attention, DeepCrossAttention (DCA), MUDDFormer, and Attention Residuals move further toward explicit attention-based routing over earlier layers. The key point, however, is that operator-level duality does not imply systems-level symmetry. For large-scale autoregressive models, sequence-axis ShortSWA is usually the more hardware-friendly placement because it reuses token-side sliding-window kernels, KV-cache layouts, and chunked execution. If the goal is instead to change the shortcut itself, Deep Delta Learning (DDL) is the cleaner intervention because it modifies the residual operator directly rather than adding a separate cross-layer retrieval path. Our recommendation is therefore simple: use DDL when the shortcut is the object of interest, and use sequence-axis ShortSWA when the goal is local adaptive mixing. 

---
# Evaluating Agentic Optimization on Large Codebases 

**Authors**: Atharva Sehgal, James Hou, Akanksha Sarkar, Ishaan Mantripragada, Swarat Chaudhuri, Jennifer J. Sun, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2603.16011)  

**Abstract**: Large language model (LLM) coding agents increasingly operate at the repository level, motivating benchmarks that evaluate their ability to optimize entire codebases under realistic constraints. Existing code benchmarks largely rely on synthetic tasks, binary correctness signals, or single-objective evaluation, limiting their ability to assess holistic optimization behavior. We introduce FormulaCode, a benchmark for evaluating agentic optimization on large, real-world codebases with fine-grained, multi-objective performance metrics. FormulaCode comprises 957 performance bottlenecks mined from scientific Python repositories on GitHub, each paired with expert-authored patches and, on average, 264.6 community-maintained performance workloads per task, enabling the holistic ability of LLM agents to optimize codebases under realistic correctness and performance constraints. Our evaluations reveal that repository-scale, multi-objective optimization remains a major challenge for frontier LLM agents. Project website at: this https URL 

---
# Mostly Text, Smart Visuals: Asymmetric Text-Visual Pruning for Large Vision-Language Models 

**Authors**: Sijie Li, Biao Qian, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.16001)  

**Abstract**: Network pruning is an effective technique for enabling lightweight Large Vision-Language Models (LVLMs), which primarily incorporates both weights and activations into the importance metric. However, existing efforts typically process calibration data from different modalities in a unified manner, overlooking modality-specific behaviors. This raises a critical challenge: how to address the divergent behaviors of textual and visual tokens for accurate pruning of LVLMs. To this end, we systematically investigate the sensitivity of visual and textual tokens to the pruning operation by decoupling their corresponding weights, revealing that: (i) the textual pathway should be calibrated via text tokens, since it exhibits higher sensitivity than the visual pathway; (ii) the visual pathway exhibits high redundancy, permitting even 50% sparsity. Motivated by these insights, we propose a simple yet effective Asymmetric Text-Visual Weight Pruning method for LVLMs, dubbed ATV-Pruning, which establishes the importance metric for accurate weight pruning by selecting the informative tokens from both textual and visual pathways. Specifically, ATV-Pruning integrates two primary innovations: first, a calibration pool is adaptively constructed by drawing on all textual tokens and a subset of visual tokens; second, we devise a layer-adaptive selection strategy to yield important visual tokens. Finally, extensive experiments across standard multimodal benchmarks verify the superiority of our ATV-Pruning over state-of-the-art methods. 

---
# Visual Set Program Synthesizer 

**Authors**: Zehua Cheng, Wei Dai, Wenhu Zhang, Thomas Lukasiewicz, Jiahao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.15997)  

**Abstract**: A user pointing their phone at a supermarket shelf and asking "Which soda has the least sugar?" poses a difficult challenge for current visual Al assistants. Such queries require not only object recognition, but explicit set-based reasoning such as filtering, comparison, and aggregation. Standard endto-end MLLMs often fail at these tasks because they lack an explicit mechanism for compositional logic. We propose treating visual reasoning as Visual Program Synthesis, where the model first generates a symbolic program that is executed by a separate engine grounded in visual scenes. We also introduce Set-VQA, a new benchmark designed specifically for evaluating set-based visual reasoning. Experiments show that our approach significantly outperforms state-of-the-art baselines on complex reasoning tasks, producing more systematic and transparent behavior while substantially improving answer accuracy. These results demonstrate that program-driven reasoning provides a principled alternative to black-box visual-language inference. 

---
# MAC: Multi-Agent Constitution Learning 

**Authors**: Rushil Thareja, Gautam Gupta, Francesco Pinto, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2603.15968)  

**Abstract**: Constitutional AI is a method to oversee and control LLMs based on a set of rules written in natural language. These rules are typically written by human experts, but could in principle be learned automatically given sufficient training data for the desired behavior. Existing LLM-based prompt optimizers attempt this but are ineffective at learning constitutions since (i) they require many labeled examples and (ii) lack structure in the optimized prompts, leading to diminishing improvements as prompt size grows. To address these limitations, we propose Multi-Agent Constitutional Learning (MAC), which optimizes over structured prompts represented as sets of rules using a network of agents with specialized tasks to accept, edit, or reject rule updates. We also present MAC+, which improves performance by training agents on successful trajectories to reinforce updates leading to higher reward. We evaluate MAC on tagging Personally Identifiable Information (PII), a classification task with limited labels where interpretability is critical, and demonstrate that it generalizes to other agentic tasks such as tool calling. MAC outperforms recent prompt optimization methods by over 50%, produces human-readable and auditable rule sets, and achieves performance comparable to supervised fine-tuning and GRPO without requiring parameter updates. 

---
# Machine Translation in the Wild: User Reaction to Xiaohongshu's Built-In Translation Feature 

**Authors**: Sui He  

**Link**: [PDF](https://arxiv.org/pdf/2603.15922)  

**Abstract**: The growing integration of machine translation into social media platforms is transforming how users interact with each other across cultural and linguistic boundaries. This paper examines user reactions to the launch of Xiaohongshu's built-in translation feature in January 2025. Drawing on a dataset of 6,723 comments collected from 11 official posts promoting the translation function, this paper combines sentiment analysis with thematic analysis to investigate how users perceived and experimented with the function. Results show that reactions were generally positive, particularly for translating posts and comments, although concerns regarding functionality, accessibility, and translation accuracy were also expressed. In addition to evaluative feedback, users actively tested the function with diverse inputs, including words and phrases in English and Chinese, abbreviations in pinyin, internet slang, and other language forms such as emoji, kaomoji, coded texts, etc. The findings highlight the importance of closer collaboration among computer scientists, translation scholars, and platform designers to better understand and improve translation technologies in real world communicative context. 

---
# Prompt Engineering for Scale Development in Generative Psychometrics 

**Authors**: Lara Lee Russell-Lasalandra, Hudson Golino  

**Link**: [PDF](https://arxiv.org/pdf/2603.15909)  

**Abstract**: This Monte Carlo simulation examines how prompt engineering strategies shape the quality of large language model (LLM)--generated personality assessment items within the AI-GENIE framework for generative psychometrics. Item pools targeting the Big Five traits were generated using multiple prompting designs (zero-shot, few-shot, persona-based, and adaptive), model temperatures, and LLMs, then evaluated and reduced using network psychometric methods. Across all conditions, AI-GENIE reliably improved structural validity following reduction, with the magnitude of its incremental contribution inversely related to the quality of the incoming item pool. Prompt design exerted a substantial influence on both pre- and post-reduction item quality. Adaptive prompting consistently outperformed non-adaptive strategies by sharply reducing semantic redundancy, elevating pre-reduction structural validity, and preserving substantially larger item pool, particularly when paired with newer, higher-capacity models. These gains were robust across temperature settings for most models, indicating that adaptive prompting mitigates common trade-offs between creativity and psychometric coherence. An exception was observed for the GPT-4o model at high temperatures, suggesting model-specific sensitivity to adaptive constraints at elevated stochasticity. Overall, the findings demonstrate that adaptive prompting is the strongest approach in this context, and that its benefits scale with model capability, motivating continued investigation of model--prompt interactions in generative psychometric pipelines. 

---
# Temporal Fact Conflicts in LLMs: Reproducibility Insights from Unifying DYNAMICQA and MULAN 

**Authors**: Ritajit Dey, Iadh Ounis, Graham McDonald, Yashar Moshfeghi  

**Link**: [PDF](https://arxiv.org/pdf/2603.15892)  

**Abstract**: Large Language Models (LLMs) often struggle with temporal fact conflicts due to outdated or evolving information in their training data. Two recent studies with accompanying datasets report opposite conclusions on whether external context can effectively resolve such conflicts. DYNAMICQA evaluates how effective external context is in shifting the model's output distribution, finding that temporal facts are more resistant to change. In contrast, MULAN examines how often external context changes memorised facts, concluding that temporal facts are easier to update. In this reproducibility paper, we first reproduce experiments from both benchmarks. We then reproduce the experiments of each study on the dataset of the other to investigate the source of their disagreement. To enable direct comparison of findings, we standardise both datasets to align with the evaluation settings of each study. Importantly, using an LLM, we synthetically generate realistic natural language contexts to replace MULAN's programmatically constructed statements when reproducing the findings of DYNAMICQA. Our analysis reveals strong dataset dependence: MULAN's findings generalise under both methodological frameworks, whereas applying MULAN's evaluation to DYNAMICQA yields mixed outcomes. Finally, while the original studies only considered 7B LLMs, we reproduce these experiments across LLMs of varying sizes, revealing how model size influences the encoding and updating of temporal facts. Our results highlight how dataset design, evaluation metrics, and model size shape LLM behaviour in the presence of temporal knowledge conflicts. 

---
# FlashSampling: Fast and Memory-Efficient Exact Sampling 

**Authors**: Tomas Ruiz, Zhen Qin, Yifan Zhang, Xuyang Shen, Yiran Zhong, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.15854)  

**Abstract**: Sampling from a categorical distribution is mathematically simple, but in large-vocabulary decoding, it often triggers extra memory traffic and extra kernels after the LM head. We present FlashSampling, an exact sampling primitive that fuses sampling into the LM-head matmul and never materializes the logits tensor in HBM. The method is simple: compute logits tile-by-tile on chip, add Gumbel noise, keep only one maximizer per row and per vocabulary tile, and finish with a small reduction over tiles. The fused tiled kernel is exact because $\argmax$ decomposes over a partition; grouped variants for online and tensor-parallel settings are exact by hierarchical factorization of the categorical distribution. Across H100, H200, B200, and B300 GPUs, FlashSampling speeds up kernel-level decode workloads, and in end-to-end vLLM experiments, it reduces time per output token by up to $19%$ on the models we test. These results show that exact sampling, with no approximation, can be integrated into the matmul itself, turning a bandwidth-bound postprocessing step into a lightweight epilogue. Project Page: this https URL. 

---
# When Stability Fails: Hidden Failure Modes Of LLMS in Data-Constrained Scientific Decision-Making 

**Authors**: Nazia Riasat  

**Link**: [PDF](https://arxiv.org/pdf/2603.15840)  

**Abstract**: Large language models (LLMs) are increasingly used as decision-support tools in data-constrained scientific workflows, where correctness and validity are critical. However, evaluation practices often emphasize stability or reproducibility across repeated runs. While these properties are desirable, stability alone does not guar- antee agreement with statistical ground truth when such references are available. We introduce a controlled behavioral evaluation framework that explicitly sep- arates four dimensions of LLM decision-making: stability, correctness, prompt sensitivity, and output validity under fixed statistical inputs. We evaluate multi- ple LLMs using a statistical gene prioritization task derived from differential ex- pression analysis across prompt regimes involving strict and relaxed significance thresholds, borderline ranking scenarios, and minor wording variations. Our ex- periments show that LLMs can exhibit near-perfect run-to-run stability while sys- tematically diverging from statistical ground truth, over-selecting under relaxed thresholds, responding sharply to minor prompt wording changes, or producing syntactically plausible gene identifiers absent from the input table. Although sta- bility reflects robustness across repeated runs, it does not guarantee agreement with statistical ground truth in structured scientific decision tasks. These findings highlight the importance of explicit ground-truth validation and output validity checks when deploying LLMs in automated or semi-automated scientific work- flows. 

---
# Persona-Conditioned Risk Behavior in Large Language Models: A Simulated Gambling Study with GPT-4.1 

**Authors**: Sankalp Dubedy  

**Link**: [PDF](https://arxiv.org/pdf/2603.15831)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents in uncertain, sequential decision-making contexts. Yet it remains poorly understood whether the behaviors they exhibit in such environments reflect principled cognitive patterns or simply surface-level prompt mimicry. This paper presents a controlled experiment in which GPT-4.1 was assigned one of three socioeconomic personas (Rich, Middle-income, and Poor) and placed in a structured slot-machine environment with three distinct machine configurations: Fair (50%), Biased Low (35%), and Streak (dynamic probability increasing after consecutive losses). Across 50 independent iterations per condition and 6,950 recorded decisions, we find that the model reproduces key behavioral signatures predicted by Kahneman and Tversky's Prospect Theory without being instructed to do so. The Poor persona played a mean of 37.4 rounds per session (SD=15.5) compared to 1.1 rounds for the Rich persona (SD=0.31), a difference that is highly significant (Kruskal-Wallis H=393.5, p<2.2e-16). Risk scores by persona show large effect sizes (Cohen's d=4.15 for Poor vs Rich). Emotional labels appear to function as post-hoc annotations rather than decision drivers (chi-square=3205.4, Cramer's V=0.39), and belief-updating across rounds is negligible (Spearman rho=0.032 for Poor persona, p=0.016). These findings carry implications for LLM agent design, interpretability research, and the broader question of whether classical cognitive economic biases are implicitly encoded in large-scale pretrained language models. 

---
# Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory 

**Authors**: Ce Zhang, Jinxi He, Junyi He, Katia Sycara, Yaqi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.15800)  

**Abstract**: Multi-modal Large Language Models (MLLMs) have achieved remarkable performance across a wide range of visual reasoning tasks, yet their vulnerability to safety risks remains a pressing concern. While prior research primarily focuses on jailbreak defenses that detect and refuse explicitly unsafe inputs, such approaches often overlook contextual safety, which requires models to distinguish subtle contextual differences between scenarios that may appear similar but diverge significantly in safety intent. In this work, we present MM-SafetyBench++, a carefully curated benchmark designed for contextual safety evaluation. Specifically, for each unsafe image-text pair, we construct a corresponding safe counterpart through minimal modifications that flip the user intent while preserving the underlying contextual meaning, enabling controlled evaluation of whether models can adapt their safety behaviors based on contextual understanding. Further, we introduce EchoSafe, a training-free framework that maintains a self-reflective memory bank to accumulate and retrieve safety insights from prior interactions. By integrating relevant past experiences into current prompts, EchoSafe enables context-aware reasoning and continual evolution of safety behavior during inference. Extensive experiments on various multi-modal safety benchmarks demonstrate that EchoSafe consistently achieves superior performance, establishing a strong baseline for advancing contextual safety in MLLMs. All benchmark data and code are available at this https URL. 

---
# Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents 

**Authors**: Madhava Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2603.15658)  

**Abstract**: Memory-augmented agents maintain multiple specialized stores, yet most systems retrieve from all stores for every query, increasing cost and introducing irrelevant context. We formulate memory retrieval as a store-routing problem and evaluate it using coverage, exact match, and token efficiency metrics. On downstream question answering, an oracle router achieves higher accuracy while using substantially fewer context tokens compared to uniform retrieval, demonstrating that selective retrieval improves both efficiency and performance. Our results show that routing decisions are a first-class component of memory-augmented agent design and motivate learned routing mechanisms for scalable multi-store systems. We additionally formalize store selection as a cost-sensitive decision problem that trades answer accuracy against retrieval cost, providing a principled interpretation of routing policies. 

---
# Beyond Reward Suppression: Reshaping Steganographic Communication Protocols in MARL via Dynamic Representational Circuit Breaking 

**Authors**: Liu Hung Ming  

**Link**: [PDF](https://arxiv.org/pdf/2603.15655)  

**Abstract**: In decentralized Multi-Agent Reinforcement Learning (MARL), steganographic collusion -- where agents develop private protocols to evade monitoring -- presents a critical AI safety threat. Existing defenses, limited to behavioral or reward layers, fail to detect coordination in latent communication channels. We introduce the Dynamic Representational Circuit Breaker (DRCB), an architectural defense operating at the optimization substrate.
Building on the AI Mother Tongue (AIM) framework, DRCB utilizes a Vector Quantized Variational Autoencoder (VQ-VAE) bottleneck to convert unobservable messages into auditable statistical objects. DRCB monitors signals including Jensen-Shannon Divergence drift, L2-norm codebook displacement, and Randomized Observer Pool accuracy to compute an EMA-based Collusion Score. Threshold breaches trigger four escalating interventions: dynamic adaptation, gradient-space penalty injection into the Advantage function A^pi, temporal reward suppression, and full substrate circuit breaking via codebook shuffling and optimizer state reset.
Experiments on a Contextual Prisoner's Dilemma with MNIST labels show that while static monitoring fails (p = 0.3517), DRCB improves observer mean accuracy from 0.858 to 0.938 (+9.3 percent) and reduces volatility by 43 percent, while preserving mean joint reward (p = 0.854). Analysis of 214,298 symbol samples confirms "Semantic Degradation," where high-frequency sequences converge to zero entropy, foreclosing complex steganographic encodings. We identify a "Transparency Paradox" where agents achieve surface-level determinism while preserving residual capacity in long-tail distributions, reflecting Goodhart's Law. This task-agnostic methodology provides a technical path toward MICA-compliant (Multi-Agent Internal Coupling Audit) pre-deployment auditing for autonomous systems. 

---
# Alternating Reinforcement Learning with Contextual Rubric Rewards 

**Authors**: Guangchen Lan  

**Link**: [PDF](https://arxiv.org/pdf/2603.15646)  

**Abstract**: Reinforcement Learning with Rubric Rewards (RLRR) is a framework that extends conventional reinforcement learning from human feedback (RLHF) and verifiable rewards (RLVR) by replacing scalar preference signals with structured, multi-dimensional, contextual rubric-based evaluations. However, existing approaches in RLRR are limited to linearly compressing vector rewards into a scalar reward with a fixed weightings, which is sensitive to artificial score design and fails to capture correlations among reward dimensions. To overcome the limitations of reward aggregation, this work proposes Alternating Reinforcement Learning with Rubric Rewards (ARL-RR), a framework that eliminates the need for a fixed scalarization by optimizing one semantic rubric meta-class at a time. Theoretically, we show that reward aggregation induces a variance contraction effect, which helps explain the performance gains. We further introduce a lightweight, search-based adaptation procedure that selects the next meta-class dynamically based on task performance, enabling the policy to emphasize critical objectives and thereby improve the model performance. Empirically, our experiments on the HealthBench dataset with experts annotations demonstrate that ARL-RR uniformly outperforms scalarized methods in both model performance and training efficiency across different model scales (1.7B, 4B, 8B, and 14B). 

---
# Tokenization Tradeoffs in Structured EHR Foundation Models 

**Authors**: Lin Lawrence Guo, Santiago Eduardo Arciniegas, Joseph Jihyung Lee, Adam Paul Yan, George Tomlinson, Jason Fries, Lillian Sung  

**Link**: [PDF](https://arxiv.org/pdf/2603.15644)  

**Abstract**: Foundation models for structured electronic health records (EHRs) are pretrained on longitudinal sequences of timestamped clinical events to learn adaptable patient representations. Tokenization -- how these timelines are converted into discrete model inputs -- determines what information is preserved, how efficiently it is encoded, and which relationships must be learned versus precomputed. Yet the impact of tokenization design choices on downstream performance and computational efficiency remains largely unexplored. Here, we pretrained a transformer on pediatric EHR data under a factorial design, varying tokenization along event encoding, time encoding, and workflow annotation. We evaluated area-under-the-receiver-operating-characteristic curve across 74 clinical prediction tasks. Joint event encoding and positional time encoding outperformed their alternatives (73/74 and 71/74 tasks) while requiring 39.5% and 9.6% fewer pretraining floating-point operations, respectively. Targeted ablations traced the joint encoding advantage to local binding efficiency, that is, code-attribute pairs are combined into single tokens, rather than split across tokens that the model must learn to associate during pretraining. External evaluation on an adult intensive care unit cohort demonstrated that this advantage generalizes despite substantial vocabulary mismatch, while temporal and workflow effects remain institution-specific. These results establish tokenization as a tractable lever for improving both the performance and efficiency of EHR foundation models. 

---
# BrainBench: Exposing the Commonsense Reasoning Gap in Large Language Models 

**Authors**: Yuzhe Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.14761)  

**Abstract**: Large language models (LLMs) achieve impressive scores on standard benchmarks yet routinely fail questions that any human would answer correctly in seconds. We introduce BrainBench, a benchmark of 100 brainteaser questions spanning 20 carefully designed categories, each targeting a specific commonsense reasoning failure mode in LLMs. Categories range from implicit physical constraints ("Should I walk or drive my rental car to the return lot?") to semantic scope tricks and default assumption hijacks. We evaluate eight frontier models -- four from the Claude family and four from the GPT family -- using a zero-shot protocol with 10 independent runs per question. The best model, Claude Opus 4.6 with extended thinking, achieves only 80.3% accuracy; the worst, GPT-4o, scores 39.7%. Even top-performing models exhibit a 6-16 percentage-point gap between accuracy and consistency, revealing stochastic reasoning. Cross-lingual evaluation in Chinese shows most models degrade by 2-8 percentage points, confirming that these failures reflect reasoning deficits rather than language-specific artifacts. BrainBench provides a fine-grained diagnostic tool for identifying where and why LLMs substitute surface heuristics for genuine commonsense reasoning. 

---
