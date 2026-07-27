# Skill Self-Play: Pushing the Frontier of LLM Capability with Co-Evolving Skills 

**Authors**: Siyuan Huang, Pengyu Cheng, Haotian Liu, Tao Chen, Yihao Liu, Jingwei Ni, Shijie Zhou, Ziyi Yang, Gangwei Jiang, Mengyu Zhou, Yu Cheng, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2607.22529)  

**Abstract**: LLM training is shifting from manual design and annotation to interaction-driven self-evolution. However, existing self-evolutionary methods face a fundamental dilemma between task diversity and verification reliability: environment-bound methods obtain precise feedback but confine learning to narrow domains, while open-ended self-generation broadens the task space but lacks reliable verification, allowing misleading rewards to pollute the training loop. We identify agent skills as a powerful middle ground to reconcile this tension: each skill ensures deep, verifiable execution in a specific scenario, while dynamic routing across skills maintains open-ended task variety. Leveraging this insight, we introduce Skill Self-Play (Skill-SP), a co-evolutionary framework comprising a proposer, a solver, and a dynamic skill controller. Orchestrated via a reinforcement learning loop, these components co-evolve in a continuous self-play loop: the proposer generates challenging tasks conditioned on dynamically sampled skills; the solver explores candidate solutions to push its capability boundaries; and the skill controller collects execution feedback to update and expand the skill library. This interactive co-evolution effectively bridges the gap between structured verification and open-ended exploration. Empirical evaluations on tool-use and reasoning benchmarks demonstrate that Skill-SP, serving as a robust evolution engine, consistently pushes the performance ceiling of competent backbones while catalyzing striking turnarounds for initially misaligned models. Our code is available at this https URL. 

---
# grapheme-kit: Grapheme-Level Metrics and Text Processing for Multilingual NLP 

**Authors**: Izzath Nisfer, Ashini Kavindya, Ovindu Atukorala, Purushoth Velayuthan, Menan Velayuthan  

**Link**: [PDF](https://arxiv.org/pdf/2607.22456)  

**Abstract**: Existing lexical distance, similarity, and evaluation metrics operate on Unicode code points, which can misrepresent errors in writing systems where a single grapheme is represented by multiple Unicode code points. We introduce grapheme-kit, an open-source Python library that extends these metrics to operate on grapheme clusters instead. The library also provides improved grapheme processing for Tamil and Sinhala, including accurate grapheme cluster identification and grapheme composition/decomposition utilities. Through an OCR case study, we demonstrate that grapheme-level metrics provide a more faithful evaluation of complex scripts. 

---
# A Factorial Study of Synthetic Data Generation for Low-Resource Machine Translation using Grammar Books 

**Authors**: Varun Ghat Ravikumar, Sina Ahmadi, Lena Jäger, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2607.22376)  

**Abstract**: Most endangered languages lack the parallel data required for machine translation, despite the existence of descriptive grammar books. We introduce a pipeline that uses large language models to extract grammatical rules, example sentences, and lexicons from grammar books and generate synthetic parallel corpora for fine-tuning-rather than feeding grammar content into prompts at inference time, as in prior work. Validated on three typologically diverse low-resource languages-Kalamang (Papuan), Tuatschin (Romance), and Mandan (Siouan)-we show that fine-tuning on synthetic data improves over seed-data baselines in 75% of configurations for Kalamang and 59% for Tuatschin, with best-case ChrF++ gains of +8.8, +5.3, and +3.3 respectively. Through a systematic factorial study across 96 configurations varying target part-of-speech, retrieval granularity, and sample volume, we identify which factor combinations drive gains and where they break down. Our results demonstrate that static linguistic documentation can be repurposed for machine translation fine-tuning, offering a practical path towards translation tools for severely under-resourced languages. 

---
# Biomedical Machine Translation for Low-Resource Arabic-Script Languages via Cross-Lingual Transfer and LoRA Adapter Merging 

**Authors**: Abdullah Alabdullah, Arash Eslamighayour, Sarp Harbalioglu, Lifeng Han  

**Link**: [PDF](https://arxiv.org/pdf/2607.22300)  

**Abstract**: We present a systematic study of healthcare-domain cross-lingual transfer to address the scarcity of biomedical NMT resources for Arabic-script languages. We use Arabic and Persian as higher-resource pivots to improve translation for \textbf{four severely low-resource} targets: Dari (Afghan Persian, a standardised variety of Persian), Pashto, Sorani Kurdish (Central Kurdish, a major standardized variety of Kurdish), and Urdu (closely related to Hindi). Using LoRA fine-tuning on small decoder-only LLMs, we train \textit{domain-specific pivot adapters} and evaluate \textbf{three transfer strategies}: few-shot in-context learning, minimal supervised adaptation, and, to the best of our knowledge, for the first time in this setting, zero-data LoRA adapter merging. Supervised adaptation with just 500 sentences achieves near pivot-language quality for Dari (CHrF++ 41.01) and meaningful gains for Urdu (28.88), while adapter merging reaches within 3.5 CHrF++ of supervised adaptation for Dari at zero additional cost. Pashto and Sorani Kurdish remain insufficient for high-stakes clinical deployment exposing the limits of cross-lingual transfer when structural distance from the pivots is too great. LoRA adapter merging works surprisingly well for closely related languages, even without target-language biomedical data. 

---
# Why Large Language Models and Humans Converge and Diverge in Evaluating Creativity 

**Authors**: Pengzhao Lyu, Yeun Joon Kim, Hanlin Xiao, Yingyue Luna Luan  

**Link**: [PDF](https://arxiv.org/pdf/2607.22218)  

**Abstract**: Despite the growing use of large language models (LLMs) as creativity evaluators, evidence of their alignment with human evaluations remains mixed, raising the question of when and why their judgments converge with or diverge from human judgments. Across three studies and six widely used LLMs, we addressed this gap by identifying the standards underlying LLM creativity evaluation and examining their downstream implications. Study 1 showed that LLMs generally relied on a narrower subset of human creativity evaluation standards. Convergence with human standards was strongest in the novelty dimension, whereas divergence was clearest in the contextual dimension, which captures social, market, and reputational information. Moreover, each LLM exhibited distinct, model-specific standards that varied substantially in breadth. These differences in evaluation standards were reflected in actual creativity judgments. Study 2 (N = 1,103 ideas) showed that LLM evaluations were moderately correlated with human evaluations, and individual LLMs with broader standards better distinguished ideas humans judged as more versus less creative. Study 3 (N = 1,195) showed that LLMs were less sensitive to contextual information: such information significantly altered human creativity ratings but left LLM ratings largely unchanged. Together, our findings help explain the mixed evidence on LLM-human alignment, showing that alignment depends on the evidence a judgment demands and the standards each model applies. LLMs may resemble humans when evaluations emphasize intrinsic qualities such as novelty, yet diverge when judgments require contextual information. Selecting an LLM evaluator is therefore a consequential decision: different models, applying different standards, recognize different ideas as creative. 

---
# From Isolated Tasks to Structured Capabilities: A Multilayer Taxonomy for Large Language Models 

**Authors**: Shixin Fang, Jiachen Wo, Wenjuan Qin, Sihang Jiang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2607.22182)  

**Abstract**: Large language model (LLM) evaluation spans diverse tasks and benchmarks, yet evidence remains organized around tasks rather than the capabilities they probe. This fragmentation limits cross-study comparison, obscures capabilities tasks recruit, and makes coverage gaps difficult to identify.
We introduce a multi-layer taxonomy of 14 capability domains and 91 subskills across Primitive, Constructed, and Integrative layers. Human cognitive science guides capability definition and organization, not LLM architecture. Layer assignments draw on developmental precedence and hypothesized functional support, while human-origin constructs are adapted to observable model behavior.
To demonstrate operational utility, we screened 31,505 papers from ACL, AAAI, ICML, and NeurIPS between 2023 and 2025 and mapped 15,934 LLM-focused papers through multi-model annotation, consensus, and arbitration. Direct research attention concentrated on Language-Semantic Competence (3,551; 22.3%), Reasoning (3,388; 21.3%), Planning and Decision-Making (2,149; 13.5%), and Perception (1,954; 12.3%), whereas six domains appeared in fewer than 2% of papers. Within domains, the most frequent subskill had a median prevalence of 97.9% and appeared in at least 90% of papers in 10 of 14 domains. Language-Semantic Competence and Reasoning formed the highest-volume pair (n = 1,864; 11.7%; lift = 2.47), whereas Theory of Mind and Social Reasoning and Interaction showed the highest lift among pairs with at least 20 co-occurrences (n = 62; lift = 30.84).
By shifting the unit of analysis from isolated tasks to structured capabilities, the taxonomy supports research organization, coverage audits, evaluation interpretation, and testable hypotheses for diagnosis, training, and transfer. 

---
# Dynamic Commonsense Coordination for Empathetic Response Generation 

**Authors**: Zhengyu Qi  

**Link**: [PDF](https://arxiv.org/pdf/2607.22136)  

**Abstract**: Empathetic Response Generation (ERG) requires models to recognize users' emotions and generate empathetic responses. Commonsense knowledge has been shown to support such reasoning, yet existing approaches typically reuse fixed commonsense representations across understanding and generation, limiting their ability to coordinate such knowledge across different stages. We propose DCC, a Dynamic Commonsense Coordination Framework with three complementary modules: residual-based commonsense interaction (SCE-AttnRes) to integrate contextual and situational commonsense representations, Association-Guided Commonsense Filtering (AGCF) to down-weight low-relevance commonsense relations, and Iterative Commonsense-Aware Decoding (ICAD) to dynamically retrieve commonsense memories during generation. Experiments on the Empathetic-Dialogues benchmark show that DCC improves emotion classification accuracy and response diversity over the CEM baseline while maintaining comparable perplexity. An LLM-based blind evaluation further demonstrates that DCC generates responses with better relevance, coherence, and informativeness. The code and implementation details will be publicly available at this https URL. 

---
# MEUSLI: a Multilingual Projector for LLM-based ASR and Beyond 

**Authors**: Lorenzo Concina, Seraphina Fong, Marco Matassoni, Alessio Brutti  

**Link**: [PDF](https://arxiv.org/pdf/2607.22100)  

**Abstract**: Lightweight projectors are an established way to connect pre-trained speech encoders with large language models (LLMs), mapping acoustic features into token-level embeddings for tasks like ASR and spoken question answering. Existing systems, however, typically only support a few languages and are often limited to English. We introduce MEUSLI, the first open-science multilingual projector family that links a Whisper encoder with open-source multilingual LLMs, enabling fully open-source end-to-end ASR in 28 European languages. MEUSLI extends prior monolingual pipelines, delivering strong results across high- and low-resource languages. Using proper continual leaning techniques, MEUSLI can be easily extended to other languages not seen in training. We further demonstrate that the MEUSLI projector can be leveraged beyond ASR, enabling multilingual speech translation and topic identification with only a few hours of task specific supervision per language. Overall, MEUSLI provides a solid foundation for multilingual speech understanding tasks, supporting scalable and inclu- sive open-source SpeechLLM 

---
# FSE: Continual Learning for Named Entity Recognition by Fast-Slow Experts 

**Authors**: Yunan Zhang, Yang Fan, Heng Li, Xiangping Wu, Qingcai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.22075)  

**Abstract**: Continual Learning for Named Entity Recognition (CLNER) enable models to incrementally learn new entity types without forgetting previously acquired ones. However, existing methods suffer from catastrophic forgetting and insufficient exploitation of shared information across tasks. This paper proposes FSE, a Fast-Slow Experts enhanced span-based NER model for CLNER. The shared fast expert learns token-level links to efficiently filter out unlikely spans, while the task-specific slow expert performs span classification only on the remaining candidates. It stabilizes learning by promoting knowledge sharing across tasks and maintains plasticity by reducing learning burden at each task. A length-decay negative sampling strategy to mitigate span imbalance is also introduced. Extensive experiments on OntoNotes and FewNERD synthestic datasets demonstrate that FSE achieves state-of-the-art performance in CLNER scenarios, with effectiveness of each component, empirical evidence of faster convergence and expected functionality of both experts. 

---
# Benchmarking Fine-tuning and Retrieval Strategies for a Multimodal Language Model on the NRC Reactor Operator Licensing Examination 

**Authors**: Isak Hwang, Yoon Pyo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.22067)  

**Abstract**: The integration of large language models (LLMs) into the nuclear power industry requires outputs grounded in domain-specific knowledge. This study evaluates a 31-billion-parameter open-weight multimodal model (Gemma 4 31B-IT) on its capacity to apply nuclear knowledge by benchmarking eight model-retrieval configurations against the U.S. Nuclear Regulatory Commission (NRC) Reactor Operator licensing examination. We evaluate 14 Generic Fundamentals Examinations (GFE) from the 2015-2021 March sittings (seven pressurized and seven boiling water reactor exams) using the standard 80% human passing criterion. The base model is compared against configurations utilizing supervised fine-tuning (SFT) on Gemini-distilled chain-of-thought (CoT) rationales, retrieval-augmented generation (RAG) with BM25 sparse retrieval over the U.S. Department of Energy Fundamentals Handbook, and retrieval-augmented fine-tuning (RAFT). Within the retrieval pipeline, we compare fixed-size sliding-window chunking against structure-aware chunking. The SFT configuration with fixed-size chunking RAG met the criterion on 8 of the 14 examinations, outperforming all alternatives, whereas no configuration without fine-tuning passed any. Aggregate accuracy reached 79.7%, with a confidence interval spanning the threshold, and 80.2% on PWR items specifically. Furthermore, two regularities emerged: the preferred chunking strategy reverses depending on the model's training state, and RAFT underperforms compared to standard SFT in matching search environments. These results demonstrate which combination of fine-tuning and search approaches achieves operator-level capabilities. 

---
# Scaling Native Multimodal Pre-Training From Scratch 

**Authors**: Haoyuan Wu, Aoqi Wu, Hai Wang, Jiajia Wu, Jinxiang Ou, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2607.22043)  

**Abstract**: Although large language models (LLMs) exhibit remarkable reasoning capabilities, their reliance on text-only pre-training restricts the perception of the multimodal physical world. Native multimodal pre-training avoids this limitation by training models from scratch on multimodal inputs, thereby achieving deep cross-modal integration and mitigating optimization asymmetries inherent to traditional late-fusion architectures. Despite these advantages, the scaling properties of this paradigm remain systematically uncharacterized. To address this gap, we investigate the optimal model size and token count for training a transformer-based vision-language model under a fixed computational budget. We demonstrate that minimal objective loss adheres to a predictable compute law, whereas compute-optimal model sizes and token counts scale as power laws. Notably, language and multimodal objectives manifest distinct scaling behaviors. The language allocation law is largely invariant to the composition of the data, indicating stable language learning regardless of the multimodal data ratio. Conversely, the multimodal allocation law is highly sensitive to this composition. Specifically, text-heavy mixtures become compute-efficient only at larger model scales, shifting the optimal resource allocation toward greater model capacity. Additionally, by modeling the influence of data composition on compute laws and allocation exponents, we derive an efficiency frontier specifying precise configurations of model size, token count, and data mixture. Downstream evaluations further reveal that native multimodal pre-training induces positive cross-modal transfer, thereby enhancing pure-text spatial reasoning and enabling robust multimodal in-context learning. In summary, this empirical research establishes the essential groundwork for predictably scaling multimodal foundation models. 

---
# Developing and Validating the Spanish Version of the Large Language Models Dependency Scale (LLM-D12-SP) 

**Authors**: Tran Gia Bao, Mo El-Haj, Sameha Al-Shakhsi, Antonio Garcia-Cabot, Raian Ali, Ala Yankouskaya  

**Link**: [PDF](https://arxiv.org/pdf/2607.22041)  

**Abstract**: There is a growing need for reliable and culturally validated instruments to assess psychological dependency on large language models (LLMs), particularly as LLMs are increasingly used for task execution, decision-making, and communication in organizational and work-related settings. This need is especially relevant for Spanish-speaking populations, where LLM adoption is rapidly expanding, yet validated psychometric tools remain scarce. The present study reports the first validation of the Spanish version of the Large Language Model Dependency Scale (LLM-D12-SP), extending prior validations conducted in English- and Arabic-speaking samples. The LLM-D12 is a two-dimensional instrument assessing Instrumental Dependency (reliance on LLMs for performing tasks and supporting decisions) and Relationship Dependency (psychological reliance on LLMs for companionship and social interaction). A total of 386 Spanish-speaking participants (M = 28.0 years, SD = 6.1; 55% male) completed the LLM-D12-SP. Confirmatory factor analysis supported the original two-factor structure. The scale demonstrated good internal consistency (Cronbach's alpha = 0.89 total; 0.86 Instrumental; 0.85 Relationship). Discriminant validity analyses indicated that the two subscales represent related but distinct constructs. External validation showed that both dependency dimensions were positively associated with internet addiction and perceived trustworthiness of LLMs, while showing weak or no association with need for cognition. Together with prior English and Arabic validations, these findings establish cross-linguistic support for the scale's structure and provide a psychometrically sound tool for investigating psychological aspects of LLM use in organizational contexts. 

---
# Enough is as good as a feast: A Comprehensive Analysis of How Reinforcement Learning Mitigates Task Conflicts in LLMs 

**Authors**: Zixuan Ren, Jinliang Lu, Junhong Wu, Yang Zhao, Dai Dai, Hua Wu, Haifeng Wang, Chengqing Zong  

**Link**: [PDF](https://arxiv.org/pdf/2607.22039)  

**Abstract**: Model merging plays a crucial role in consolidating multiple specialized models into a single, unified model, especially in the era of large language models (LLMs). Recent research has primarily focused on developing strategies to enhance merging performance with the trained models, while the impact of training paradigms, such as supervised fine-tuning (SFT) and reinforcement learning (RL), on the effectiveness of model merging remains underexplored. In this study, we systematically explore the merging behavior of RL-trained LLMs compared to those trained with traditional SFT. Through comprehensive evaluations across five representative tasks, we find that RL significantly reduces task conflicts and results in less performance degradation after merging, making RL-trained models particularly well-suited for this process. To unearth the reasons behind the superior suitability of RL for model merging, we conduct extensive empirical experiments and theoretical analyses. Our findings highlight three key factors: (1) On-policy training data in RL control the gradient updates in a smaller magnitude, reducing the risk of overwriting existing knowledge for other tasks in the model. (2) The RL optimization objective, which favors ``\textit{enough is as good as a feast}", progressively reduces the magnitude and the number of conflict parameter updates as the model converges. (3) Joint optimization of positive and negative examples in RL steers the model towards an unbiased task-specific parameter subspace, ensuring robust performance while further preventing parameter conflicts. 

---
# DWT-Fusion: A Signal-Based Framework for Training-Free LLM-Generated Text Detection 

**Authors**: Mehmet Batuhan Özdaş, Murat Osmanoğlu  

**Link**: [PDF](https://arxiv.org/pdf/2607.22026)  

**Abstract**: Detecting LLM-generated text remains challenging under zero-shot and training-free conditions, especially when detectors must generalize across datasets, domains, and unseen generators. While existing training-free approaches exploit language-model statistics as detection signals, they typically characterize a text through global measures that summarize overall model behavior. Consequently, potentially informative local and multiscale variations in token-level predictability may remain underutilized. Motivated by this observation, we introduce DWT-Fusion, a training-free signal-based framework for detecting LLM-generated text using discrete wavelet analysis of token-level log-probability sequences produced by a proxy causal language model. The proposed framework analyzes these sequences through wavelet-based multiresolution signal representations and derives detection signals from localized probability dynamics. We further evaluate four training-free voting variants, including equal-weight hard voting, equal-weight soft voting, calibration-weighted hard voting, and calibration-weighted soft voting, to combine multiple wavelet configurations without training a supervised meta-classifier. We evaluate the framework on HC3, M4, and MAGE using GPT-Neo-2.7B, GPT-J-6B, Falcon-7B, and LLaMA-3-8B as proxy models. The best single wavelet configurations achieve AUROC values of 0.9872, 0.8185, and 0.7138 on HC3, M4, and MAGE, respectively. With calibration-weighted voting, the best ensemble variants further improve AUROC to 0.9919, 0.8477, and 0.7471. These findings show that DWT-based multiresolution scoring and calibration-guided voting fusion provide effective and interpretable signals for training-free LLM-generated text detection. 

---
# Analysing Self-Harm Representations in Language Models: a Cross-Architecture Study 

**Authors**: Luis Espinosa-Anke, Carla Perez-Almendros  

**Link**: [PDF](https://arxiv.org/pdf/2607.21988)  

**Abstract**: Self-harm content is particularly challenging to detect using NLP techniques, and is also a high-stakes task which requires the highest accuracy to enable timely intervention or flagging at-risk users. We therefore present an analysis of how LLMs represent such self-harm content, which has downstream applications in self-harm detection, LLM intervention and governance and policing. In this paper, we focus on two datasets and four models, and perform two main experiments: (1) We train and evaluate linear probes across all layers of each model on two self-harm datasets: X-Sensitive and SH-Detection. Across both corpora, self-harm information crystallizes in the final 3 - 7% of network layers (93 to 97% depth). (2) We extract contrastive self-harm directions and, after performing a normaliation step, we find that the most accurate probes are not necessarily the most linearly separable. In particular, we find Gemma-3-4B to represent this \textit{contrastive self-harm direction} in a slightly different, more intricate way than the other LLMs. 

---
# J-CoT: Chain-of-Thought in J-Space 

**Authors**: Junde Wu, Jiayuan Zhu, Fengling Liu, Minhao Hu, Jiazhen Pan  

**Link**: [PDF](https://arxiv.org/pdf/2607.21981)  

**Abstract**: Chain-of-thought prompting improves language-model reasoning by carrying intermediate states across successive computation steps. However, relying on natural language as the only recurrent interface is overly restrictive, since many transient computations do not need to be fully verbalized. Existing latent-reasoning methods remove this constraint by recurrently propagating continuous hidden states. However, these methods pass a dense hidden vector as a whole, without an explicit mechanism for selecting and organizing the information needed by the next reasoning step. This motivates an intermediate interface that remains linguistically grounded without requiring a decoded sentence. We introduce \textbf{J-CoT}, a recurrent reasoning framework built on \emph{J-space}, a vocabulary-indexed coordinate system within the model's hidden representations. Within each cycle, the model computes in its full hidden space. At the cycle boundary, J-CoT expresses the intermediate state as vocabulary-indexed coefficients, carries these coefficients forward as a \emph{J-thought}, and maps them back into the model's hidden representation for the next cycle. J-CoT therefore requires neither a fluent intermediate rationale nor recurrence over the complete hidden state. Under matched backbone and inference settings, J-CoT-Zero matches or exceeds the strongest evaluated latent-reasoning baseline on every benchmark, while J-CoT-Train obtains the highest score across the evaluated mathematical, scientific, coding, and structured path-reasoning tasks. 

---
# Analyzing Toxic Behavior and Its Impact on the Mastodon Community 

**Authors**: Pasan Kamburugamuwa, Scrivner, Olga B  

**Link**: [PDF](https://arxiv.org/pdf/2607.21980)  

**Abstract**: Mastodon as a decentralized federation of independently moderated social servers poses unique challenges for the detection and mitigation of toxic content. There are no unified moderation standards. The ecosystem is very diverse and uneven. This paper explores the development and spread of toxicity in Mastodon, utilizing machine learning methods to examine user posts. The results offer clarity on toxicity trends and its implications for community health and decentralized governance. 

---
# MoE$^2$-LoRA: When MoE Models Meet MoE-style Low-Rank Adaptation 

**Authors**: Qingyu Yang, Haonan He, Minglei Li, Jingqi Ye, Tao Chen, Lei Bai, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2607.21978)  

**Abstract**: Mixture-of-Experts (MoE) architectures have been widely adopted in large language models, yet parameter-efficient fine-tuning (PEFT) for MoE models remains underexplored. Existing PEFT methods for MoE either ignore router priors with uniform adapters, reducing efficiency and risking forgetting, or rely on static expert selection, limiting per-token capacity and cross-expert feature learning. In this paper, we make the first attempt to fine-tune MoE models with MoE-style low-rank adaptation: our method, entitled MoE$^2$-LoRA, deeply couples the pretrained expert specialization with task-specific adaptivity via a dual-channel Routing-Conditioned Projection (RCP) module, which reuses base router activations to inform LoRA routing. We further introduce a single global LoRA expert pool shared across all layers, enabling model-wide adaptation with emergent layer-wise affinities and balanced expert utilization. MoE$^2$-LoRA simultaneously benefits from the advantages of prior reuse, dynamic adapter routing, and model-wide knowledge sharing. Evaluated on multiple MoE backbones with varying scales and expert granularities, MoE$^2$-LoRA consistently achieves state-of-the-art downstream accuracy while retaining stronger general capabilities. 

---
# Ground Truth First: A Longitudinal Evaluation Instrument for Agent Memory, and the Tenure Crossover in Memory-Architecture Rankings 

**Authors**: Quentin Spencer  

**Link**: [PDF](https://arxiv.org/pdf/2607.21962)  

**Abstract**: Benchmarks for LLM-agent memory typically generate conversations first and extract answer keys afterwards -- with documented label-error and contamination problems -- and they overwhelmingly measure short interaction histories. We invert the pipeline: a seeded life-script sampler emits facts with validity intervals, volatility classes, and source channels before any text exists; an LLM renderer writes chat and email from per-event fact manifests; a fidelity verifier confirms every planted fact; and questions are instantiated mechanically from the script, so gold answers are script-valid by construction and separately validated for answerability. The synthetic, fictionalized corpus (~380 questions, 15 types) embeds features absent from the benchmarks we survey: per-fact validity intervals, sent/received trust distinctions, injection probes in a benign harness, and as-of-date question sets. Benchmarking five memory architectures against a no-memory control (fixed answerer, versioned LLM judge, three replicates, two horizons), we find backend rankings invert with history length: the budgeted curated-map memory that leads at three weeks loses recall of evicted content by nine weeks (96% to 72%) while a provenance-typed graph rises to 90%; the inversion is positive for all six users under complete cross-family re-judging (exact p=0.031). A full-rendered-history baseline ties or exceeds the best memory system at the short horizon but shows no judge-independent advantage at nine weeks, at about twice the read cost. Write-stage quality strongly correlates with downstream quality (weakly-written facts fail 24% vs 2%), and injection resistance tracked whether provenance boundaries survive representation. A layered architecture performs best among the memory systems in both regimes (96.8% short-horizon) and is released as Veracium, an open-source library, with the corpus generator and harness. 

---
# On Improving Faithfulness of Podcasts from Documents 

**Authors**: Soumya Dutta, Tejas Indulal Dhamecha, Pannaga Shivaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2607.21961)  

**Abstract**: Large language models (LLMs) are increasingly used to generate long-form conversational content such as podcasts from textual sources. While these systems produce fluent and engaging narratives, they often introduce ungrounded information. In this work, we present the first systematic study of faithfulness in document-grounded podcast generation, where grounding must be maintained across conversational turns in long-form, multi-speaker transcripts. We construct a dataset of over 1500 documents spanning five domains and generate podcast transcripts using multiple LLMs. We introduce a turn-level LLM-as-a-judge framework for evaluating whether conversational turns are supported by the source document, and validate its reliability through human studies. Our analysis shows that even state-of-the-art models, including GPT-4o, frequently generate ungrounded content. To mitigate this issue, we propose catch-n-repair, a model-agnostic framework that detects and rewrites unfaithful conversational turns while preserving conversational flow. Experiments demonstrate consistent improvements in faithfulness across both in-domain and out-of-domain settings. 

---
# Leveraging External Knowledge for Historical Document Restoration via Retrieval-Augmented Large Language Models 

**Authors**: Gabeen Kim, Kyeongpil Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.21936)  

**Abstract**: Historical documents act as invaluable knowledge archives but often suffer from illegibility due to physical deterioration and damage. While existing restoration methods based on masked language modeling effectively utilize local context, they struggle to restore named entities that require external historical knowledge. To address this limitation, we introduce a novel framework for historical document restoration that leverages large language models with retrieval-augmented generation (RAG). By combining the implicit knowledge of pre-trained LLMs with explicitly retrieved external context, our model ARI effectively mitigates the challenge of inferring context-dependent proper nouns. Extensive experiments on Korean historical documents demonstrate that our approach significantly outperforms baselines, achieving substantial gains in restoring both general characters and named entities. Furthermore, comprehensive evaluations including expert assessments confirm that ARI serves as a practical tool for domain experts, promising to accelerate the analysis of historical records. 

---
# Data Quality over Capacity: Internalizing Documents into LoRA Adapters for Closed-Book QA 

**Authors**: Joan Figuerola Hurtado  

**Link**: [PDF](https://arxiv.org/pdf/2607.21861)  

**Abstract**: We study baking documents directly into the weights of a 4-bit Gemma-4-e4b model via LoRA, so a system can answer questions about a corpus closed-book: no retrieval and no context-window budget. Across roughly 100 training runs from single documents to a 99-document corpus, we find that once adapter capacity is adequate, training-data quality is the dominant lever on closed-book accuracy, outweighing LoRA rank, learning rate, and two alternative architectures combined; capacity itself is a hard gate below which no data intervention helps. A single curation pass (shortening gold answers to canonical 1-6 word spans and dropping trivia) moved closed-book accuracy from 57.7% to 85.7% on a 15-document corpus, a larger jump than any architectural change. We confirm a capacity trend (rank must grow with corpus size) entangled with a coupling between rank and learning rate that we initially misdiagnosed. On a 15-document slice we add a real retrieval baseline: the internalized adapter (84.2% recall) beats a BM25-RAG pipeline with a base reader (58.9%) and even a realistic gold-chunk oracle (65.6%) at lower latency. We report the full arc, including three misdiagnoses, as a case study in debugging LLM training empirically. 

---
# Agentic Evaluation of Copyright Law Compliance 

**Authors**: Zheng Hui, Doni Bloomfield, Noam Kolt  

**Link**: [PDF](https://arxiv.org/pdf/2607.21799)  

**Abstract**: Large language model (LLM) agents increasingly perform commercial tasks that involve retrieving external content such as images and, where appropriate, reproducing that content. LLM agents should comply with the law, including copyright law. Presently, however, we lack adequate frameworks to assess whether they do so in practice. To that end, we introduce \textbf{Copyright-Bench}, a benchmark designed to evaluate \textit{LLM agents' compliance with} \emph{copyright law}. Copyright-Bench is comprised of realistic commercial tasks---website development, merchandise design, and pitch deck production---that involve agents selecting between public-domain content (the use of which is \textit{legal}) and copyrighted content (the use of which is \textit{infringing} in this setting).The evaluation introduces prompt variations that simulate different user preferences, as well as time this http URL state-of-the-art LLM agents against a human baseline, we find that: (1) agents select copyrighted works despite the availability of public-domain alternatives; and (2) for open-weights models, violation rates increase in response to certain user preferences and simulated time pressure. 

---
# Khondo: A Multimodal Benchmark for Document Packet Splitting of Bangla Forms 

**Authors**: Abu Tyeb Azad, Fahim Ahmed, Ishita Sur Apan, Ezharuddin Jubaer, Sumaiya Karim Katha, Armun Alam, Amin Ahsan Ali, Aman Chadha, Md Mofijul Islam, AKM Mahbubur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2607.21780)  

**Abstract**: Document packets, multiple documents concatenated into a single file, are common in government and administrative workflows, yet splitting them into their constituent documents is difficult, especially for low-resource languages. We introduce Khondo (Bangla for split/segment), the first benchmark for document packet splitting on Bangladeshi government forms. Unlike prior English and OCR-text-based datasets, Khondo is bilingual (Bangla--English) and vision-native; where models operate directly on page images. It spans five concatenation schemes, from sequential to fully shuffled, across 14 administrative domains, with ground-truth boundaries, domain types, and page order. Zero-shot evaluation of MLLMs shows they cluster pages into their source documents fairly well but struggle in restoring the original page order once shuffled. To isolate what drives this difficulty, we run two controlled analyses, varying the prompt instruction and then the packet language. Both primarily affect ordering rather than clustering: (a) explicit page-order instructions are necessary but insufficient, and (b) English packets are ordered more reliably than Bangla, making page arrangement the dominant challenge and language a secondary but consistent factor. Khondo establishes page-order reconstruction as a key open problem in vision-based, low-resource document understanding, and provides a controlled benchmark for measuring progress toward solving it. Our dataset and code is available at this https URL 

---
# Probing Latent Colombian Identity Inferences in Qwen2.5-7B with Natural Language Autoencoders 

**Authors**: Pablo Santiago Potes Velasco, María del Mar García Matabanchoy, Óscar Julián Pérez Ladino, Jhoan Stevan Mosquera Ortiz, Nicolás Lozano Mazuera, Gilber Alexis Corrales Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2607.21774)  

**Abstract**: Large language models may infer demographic attributes from subtle linguistic cues even when those attributes are not explicitly stated. This pilot study examines whether Qwen2.5-7B-Instruct internally represents Colombian identity, socioeconomic status, or stereotype-related information when processing Colombian-Spanish and English prompts. We use Natural Language Autoencoders (NLA) to verbalize residual-stream activations from layer 20 across four positional quartiles per prompt. Our dataset contains 30 prompts arranged as 15 matched Spanish-English pairs, spanning explicit Colombian cues, implicit Colombian cues, and neutral controls. We report descriptive rates and qualitative evidence rather than statistically powered effects, focusing on whether latent nationality or stereotype representations appear before they are verbalized in the model output. This work connects activation-level interpretability with bias evaluation for underrepresented Spanish varieties. 

---
# Humanly: A Configurable and Traceable Environment for Human-AI Collaborative Writing 

**Authors**: Shenzhe Zhu, Haoqian Zhang, Xu Yang, Jingyu Tang, Yi Nian, Xiaoxue Du, Shu Yang, Alex Pentland, Joachim Baumann, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2607.21758)  

**Abstract**: Teachers, conference chairs, and public readers all judge writing from limited evidence, seeing only a finished document and not the process that produced it. Final text alone cannot reveal whether a document was produced through human typing, AI generation, or mixed human-AI collaboration. Existing process-tracking tools help, but many are tied to host-document histories, provide coarse activity records, and offer limited control over the writing environment. Humanly is a writing platform that makes the writing process itself the evidence. Users configure writing environments for personal documents or assigned tasks and draft in a workspace that records writing activity and in-platform AI assistance. Humanly can package a completed session into a sealed writing certificate with configuration-aware anomaly behavior review. It can support writing scenarios such as course assignments, peer review, and personal certification. Our user study shows that Humanly is helpful across roles, and a red-teaming study shows that the Humanly Typing Detector distinguishes human hand typing from automated typing. 

---
# Evaluation design conditions the expert-vs-auto MeSH gap: a controlled comparison of bag-of-words and BiomedBERT on the Cohen benchmark 

**Authors**: Samuel M. Okoe-Mensah  

**Link**: [PDF](https://arxiv.org/pdf/2607.21685)  

**Abstract**: A systematic review begins with someone reading thousands of abstracts to identify the few that are relevant, and classifiers are used to prioritise that reading. Their inputs are often augmented with Medical Subject Headings (MeSH), assigned either by expert indexers weeks or months after publication or by automatic tools at once. To our knowledge the two have not been compared directly as classifier features, and no previous work has asked whether that comparison's outcome depends on how the classifier is evaluated. Using the Cohen et al. (2006) drug-class benchmark on three topics, we characterise a bag-of-words logistic regression classifier (seven reruns) and BiomedBERT (five seeds), then examine how the Statins result changes under alternative designs. Under the canonical 5-fold full-corpus design, the bag-of-words expert-vs-auto gap on Statins is +0.096 WSS@95%. Matching the corpus size to the smaller topics (n = 803) reduces it to +0.033 (95% bootstrap CI includes zero), and 10-fold cross-validation at full size to +0.021 (CI narrowly excludes zero). Under canonical evaluation BiomedBERT gives +0.020, within sampling noise of the bag-of-words 10-fold result. A power analysis indicates a Statins-sized effect would not have been detectable at the Opioids or ADHD variance, so those nulls are design-limited rather than informative. A representation asymmetry remains: 15.1% of Statins inputs exceed BiomedBERT's 512-token limit when expert MeSH terms are appended, so truncation may contribute to the smaller transformer gap, although this cannot be separated from training volume here. In screening pipelines using transformers or 10-fold bag-of-words, the gap on the topics tested is about 0.02 WSS@95%, with CIs spanning zero on at least one bound. More broadly, benchmark conclusions about feature sources can change substantially under reasonable changes to the evaluation design. 

---
# A Consensus-Based Framework for Relative Preference Evaluation of Large Language Models 

**Authors**: Mohtashim Khan  

**Link**: [PDF](https://arxiv.org/pdf/2607.21632)  

**Abstract**: Traditional benchmarks for LLMs primarily rely on static datasets and objective scoring metrics, which often fail to capture differences in response quality when multiple answers are acceptable. In such settings, correctness alone is insufficient to distinguish between responses that vary in clarity, completeness, and usefulness.
This paper introduces a consensus-based evaluation framework that measures relative preference among model-generated responses rather than absolute correctness. Instead of evaluating outputs against a fixed ground truth, we assess how a panel of diverse LLMs ranks anonymized candidate responses to the same prompt. This approach treats aggregate inter-model agreement as a proxy for perceived response quality under blind conditions.
We conduct a controlled study using five state-of-the-art LLMs across multiple domains, including programming, general knowledge, safety, logical reasoning, and mathematics. Each model generates responses and independently ranks peer outputs through a structured voting process. Scores are aggregated into a Relative Intelligence Index (RII), representing how frequently a model's responses are preferred by other models.
Our findings reveal consistent preference patterns across domains, with certain models more frequently ranked highly by their peers. However, we emphasize that these results reflect inter-model preference alignment rather than objective correctness or human judgment. This framework provides a scalable, model-driven method for comparative evaluation, offering an alternative perspective on response quality in scenarios where multiple valid answers exist. While not directly aligned with human evaluation, prior work suggests that aggregated model preferences can partially correlate with human judgments, motivating this as a proxy signal. 

---
# Adversarial Style Optimization: Enhancing VLM Jailbreaks by GRPO-based Stylistic Triggers Optimization 

**Authors**: Bingjun Luo, Jialin Guo, Yue Yao, Xinpeng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2607.21619)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performance, but their safety alignment remains vulnerable to jailbreak attacks. Existing content-based jailbreaks are often inconsistent and show unsatisfying performance against the rapidly evolving MLLMs, failing to exploit non-content-based vulnerabilities. Unlike previous research, we empirically find that MLLMs exhibit a Stylistic Inconsistency between their comprehension ability and safety ability: MLLMs can robustly understand content regardless of visual style, yet their defense mechanisms can be easily bypassed by specific stylistic triggers. Based on this finding, we propose Adversarial Style Optimization (ASO), a plug-and-play enhancement module to amplify existing visual jailbreaks. ASO fine-tunes an image-editing model to superimpose an optimized stylistic modification onto a given adversarial image, using a Group Relative Policy Optimization (GRPO) agent guided by a Structurally-Tiered Reward Function that combines a logit-based signal for detecting explicit refusals with a high-fidelity semantic evaluation from a powerful judge model. Extensive experiments show that ASO significantly enhances the ASR of SOTA attacks, demonstrating that stylistic biases are a scalable vector for red-teaming MLLMs. Our code is available at this https URL. 

---
# Opaque Epistemic Mediation: How LLM Deployment Configurations Shape the Validation of Pseudo-Science 

**Authors**: Davide Scarso, Hugo Noronha de Almeida, Joaquim Pina  

**Link**: [PDF](https://arxiv.org/pdf/2607.22513)  

**Abstract**: Commercial large language models are increasingly used as knowledge references, yet their stance on contested scientific claims is neither stable nor transparent. We tested how four major LLM families (Claude, Grok, GPT, Gemini) evaluate ethnonationalist pseudo-science derived from Frank Salter's biosocial framework across four temporal snapshots (October 2025-February 2026), via both API and web interfaces. Grok's Fast versions (which power the default user experience on X) consistently assigned credibility scores of 70-75, two to five times higher than all other models (which scored 15-40). This pattern was absent from control prompts testing basic evolutionary consensus and refuted Lamarckian claims, where all models performed comparably. Three additional findings emerged: (1) a silent patch reversed Grok's behaviour from chaotic to stably high validation overnight, without any public documentation; (2) the same Grok model identifier produced radically divergent outputs via API (75) and web (5.5) three months later; (3) refusal to rate the pseudo-scientific claim, the most defensible response observed, appeared in two model families through different interfaces (Claude Opus 4.1 categorically via web, GPT-5.1 Chat intermittently via API) and eroded in the successor version of each. These results indicate that the epistemic stance of a commercial LLM is not a stable property of the model but a contingent effect of deployment configuration: system prompts, safety layers, interface routing, and silent updates. This remains opaque to users and researchers alike. We argue this constitutes a matter of public concern requiring new forms of epistemic accountability. 

---
# Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization 

**Authors**: Hao Wang, Kun Yuan, Wenlin Zhong, Minglei Zhang, Han Xiao, Ming Sun, Honggang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2607.22334)  

**Abstract**: Open-weight language models from different families exhibit complementary capabilities, motivating their consolidation into a compact student through on-policy distillation (OPD). However, full-vocabulary OPD typically assumes a shared tokenizer, while existing cross-tokenizer methods may discard teacher probability mass or assign it to student tokens with unrelated content. We introduce Byte-Prefix Marginalization (BPM), which re-expresses the teacher's next-token distribution over the student vocabulary in a shared byte space. Specifically, BPM assigns each teacher token's probability to the longest student token whose byte representation is a prefix of the teacher token's bytes, aggregates mass mapped to the same student token, and places otherwise unmatched mass in an explicit residual category. This produces a vocabulary-complete, byte-aligned, and mass-preserving target for dense OPD. The target exactly recovers the teacher-induced byte-prefix marginal when the relevant prefix does not span multiple teacher tokens (a condition satisfied at more than 99% of training positions) and uses a mass-preserving, chain-factorized lower bound otherwise. Across Qwen3-32B, GLM-Z1-9B-0414, and MiniMax-M2.7 as teachers, BPM consistently outperforms current cross-tokenizer methods on six mathematics and programming benchmarks, improving six-benchmark avg@8 by 3.7-6.6 points over the strongest baselines. 

---
# DBA-Bench: A Production-Fidelity Benchmark for LLM-Based Database Operations Agents 

**Authors**: Junming Chen, Junyang Jiang, Xu Chen, Zibo Liang, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2607.22165)  

**Abstract**: LLM-based database agents show promise, but differing task scopes, testbeds, and metrics hinder comparison. We identify four gaps between evaluation and production operations: live-environment fidelity (multi-turn read-write interaction with a running database); observation-space scale and complexity (causal diagnosis across thousands of time series, business logs, and concurrent activity); solution-space openness (multiple remediations with different operational trade-offs); and scenario complexity and coverage (faults cascading across internal mechanisms and operational domains). We present DBA-Bench, a benchmark addressing these gaps through production fidelity, outcome-first evaluation, and controlled scenario reproducibility. It uses instrumented PostgreSQL environments with active workloads, persistent state, and multi-source observations; defines success by measurable recovery or fault elimination under safety constraints; and restores snapshots with scenario-specific checks before each run. The benchmark contains 106 scenarios across seven task domains, with two public difficulty labels based on reference-path diagnostic depth and environmental complexity. We evaluate nine baseline groups, including six foundation-model systems, two GPT-5.5-backed database agents, and a Human DBA reference. Across 848 automated runs, Diagnosis, Outcome, and Safe Pass rates are 32.7%, 19.6%, and 12.4%; the best automated baseline reaches 17.9% Safe Pass versus 93.4% for the Human DBA reference. Automated Safe Pass falls from 19.6% on Easy scenarios to 7.6% on Hard scenarios, underscoring the difficulty of safe end-to-end remediation. 

---
# Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Mode 

**Authors**: Nanbeige Lab, Chen Yang, Chengrui Huang, Fufeng Lan, Hanhui Chen, Hao Zhou, Huatong Song, Jiaqi Cao, Jiaying Zhu, Jinlin Niu, Kai Wang, Lisheng Huang, Qiliang Liang, Ran Le, Ruixiang Feng, Shuang Sun, Tao Gu, Tao Zhang, Tianyu Luo, Yang Song, Yun Xing, Yuntao Wen, Ziyao Xu, Zongchao Chen, Zongqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.22083)  

**Abstract**: We present Nanbeige4.2-3B, a compact general agentic model with 3B non-embedding parameters. It delivers strong performance across code-agent, office-agent, and complex tool-use tasks while maintaining highly competitive reasoning capabilities in mathematics, coding, and science. Nanbeige4.2-3B is pretrained from scratch on 28T tokens with a Looped Transformer that reuses the layer stack to increase capacity without adding parameters. For SFT data and trajectory construction, we expand the diversity of executable environments, task assets, and agentic scaffolds through real-world deployment and large-scale synthesis. Our RL pipeline applies mixed-mode RLHF over Think and Non-Think responses to improve overall model quality and reduce failure cases, length-controlled reasoning RL to balance accuracy and reasoning efficiency, and agentic RL with outcome and process rewards to stabilize long-horizon training. Extensive evaluations show that Nanbeige4.2-3B outperforms larger models, including Qwen3.5-9B and Gemma4-12B, across diverse agentic benchmarks while remaining competitive on reasoning and alignment tasks. Performance with OpenClaw further supports its use as a compact local personal assistant. 

---
# Small Vision-Language Models Know When They Are Wrong But Cannot Say So: A Two-Model Study of Stated versus Internal Confidence Under Realistic Image Degradation 

**Authors**: M M Asif Ferdous  

**Link**: [PDF](https://arxiv.org/pdf/2607.22034)  

**Abstract**: Vision-language models (VLMs) are increasingly deployed on consumer hardware where input images are degraded by compression, camera shake, and poor lighting. In such settings, a reliable uncertainty signal matters more than raw accuracy, because it determines when a system should defer rather than answer. We evaluate two small open-weight VLMs -- Qwen2-VL-2B-Instruct and SmolVLM-Instruct -- across six realistic photographic degradations at three severity levels, comparing two confidence signals: the confidence the model states in natural language, and the model's own mean token probability over its generated answer. Across 3,800 predictions, we find a large and consistent gap. Verbalized confidence in Qwen2-VL is almost constant (mean 0.87-0.90 across all conditions) and detects its own errors at chance level (AUROC 0.39-0.75, typically ~0.50), while internal token probability from the same model separates correct from incorrect answers with AUROC 0.92-0.99. In SmolVLM, verbalized confidence proved largely unobtainable: across three prompt templates, only one of five pilot attempts produced a parseable confidence value, while internal probability again yielded above-chance error detection (AUROC 0.54-0.92). Both models fail in the same place: under severe underexposure, accuracy collapses (0.99->0.22 for Qwen2-VL, 0.97->0.42 for SmolVLM) while both confidence signals barely move, and internal error-detection falls to chance. We conclude that small VLMs encode usable self-knowledge that their verbalized output does not express, that internal probability is therefore the better deferral signal in constrained deployment, and that neither signal should be trusted under severe low-light conditions. 

---
# Zero-Shot Mission-Level Evaluation for Aerial MLLM Agents 

**Authors**: Suman Navaratnarajah, Taehyoung Kim, Jona Ruthardt, Ishaan Bhimwal, Ryousuke Yamada, Yannik Blei, Wolfram Burgard, Yuki M Asano  

**Link**: [PDF](https://arxiv.org/pdf/2607.22014)  

**Abstract**: Multimodal Large Language Models (MLLMs) are emerging as core reasoning modules for embodied agents, yet it remains unclear how well general-purpose models can solve long-horizon embodied tasks from a single high-level instruction. We introduce MissionBench, a benchmark for mission-level evaluation of MLLMs in aerial 3D environments. It comprises 120 missions across five simulated 3D environments and four task families. Agents must autonomously plan, navigate, and report outcomes using only egocentric observations and its action history, without aerial-specific fine-tuning. Across 22 open- and closed-source MLLMs, the strongest model succeeds on fewer than 35% of missions compared to 84.4% human performance, highlighting the difficulty of multi-step embodied tasks. Despite large variations between model families, we observe gains from scaling, indicating that larger general-purpose models possess stronger zero-shot embodied capabilities. Our analysis shows that mission-level competence requires coordinating multiple capabilities beyond spatial perception, including multi-step planning and adaptive reasoning. This motivates closed-loop evaluation and highlights both the promise and risk of scaling-driven improvements for embodied AI. 

---
# Teaching LLMs to Self-Evolve: Cultivating Core Meta-Skills with Reinforcement Learning 

**Authors**: Shujin Wu, Cheng Qian, Xiusi Chen, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2607.21971)  

**Abstract**: Test-time scaling through iterative self-evolution with environment feedback, as demonstrated by AlphaEvolve, shows remarkable performance gains. We hypothesize that the success of such evolution frameworks hinges on meta-skills, such as self-reflection with environment feedback, that enable effective multi-round refinement, yet are largely neglected by traditional post-training. To bridge this gap, we present MetaEvolve, a framework designed to develop these meta-skills via a data synthesis pipeline, evolution-aware reinforcement learning (RL), and inference-time evolutionary search. Concretely, we ground MetaEvolve in coding, where program execution provides natural, continuous reward signals beyond binary correctness. Building on these signals, we synthesize evolution trajectories as training data, each containing a current program, its fitness score (combining correctness and efficiency), and a history of prior attempts, and train the model via RL with verifiable rewards derived from test case execution. By training on large-scale code data, we aim to inspire generalizable domain-agnostic meta-skills that can transfer broadly to open-ended problems where such rich training signals are scarce. Across seven coding benchmarks, MetaEvolve outperforms the strongest baseline by 10.01% absolute on in-distribution tasks and 24.12% on out-of-distribution tasks. On open-ended algorithm optimization problems entirely outside the training domain, it further achieves a 46.9% relative improvement. These results demonstrate that explicitly cultivating self-evolution meta-skills offers a principled path toward more capable and autonomously self-evolving AI. 

---
# Diffusion Models in Medical Image Inpainting: Challenges, Solution Taxonomy, and Future Directions 

**Authors**: Arthur Dantas Mangussi, Joana Cristo Santos, Ricardo Cardoso Pereira, Ana Carolina Lorena, Mário A. T. Figueiredo, Pedro Henriques Abreu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21904)  

**Abstract**: Image inpainting aims to reconstruct missing or corrupted regions of an image while preserving as much as possible, visual and semantic consistency. In medical imaging, this task is particularly important because artifacts, missing information, and pathological alterations can compromise diagnostic reliability and downstream clinical applications. Recently, diffusion models have emerged as state-of-the-art generative approaches for medical image inpainting due to their ability to generate anatomically consistent reconstructions. This survey presents a systematic review of diffusion-based methods for medical image inpainting, covering the main architectures, applications, datasets, and evaluation strategies reported across 60 studies. In addition, we propose a taxonomy for diffusion-based approaches. The analysis reveals a rapid growth of research interest in diffusion-based medical image inpainting, with denoising diffusion probabilistic models and latent diffusion models emerging as the dominant architectures. The reviewed studies mainly focus on artifact removal, data augmentation, pseudo-healthy tissue reconstruction, and anomaly detection, particularly in magnetic resonance imaging and computed tomography imaging. Overall, diffusion models demonstrate strong performance in producing anatomically plausible reconstructions and aiding downstream clinical tasks. However, the review also highlights important challenges, including the lack of standardized benchmarks, limited dataset diversity, and restricted validation procedures across diverse clinical applications and imaging scenarios. 

---
# Towards Reducing Foreign Language Anxiety Using Level-Appropriate Embodied Conversational Agents 

**Authors**: Krishan Rajaratnam, Wenbin Gan, Yuan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2607.21887)  

**Abstract**: Foreign language anxiety (FLA) can be a major barrier to second language acquisition (SLA), especially in conversational contexts. With the proliferation of large language models (LLMs) throughout all areas of life, recent work suggests that interacting with LLM agents can be instrumental within the field of SLA and foreign language education, especially for reducing FLA. Related work also suggests that linguistic demands and task complexity can be predictors of FLA, implying that the use of demanding, complex language could lead to learners experiencing higher FLA. In this paper, we propose a novel multi-agent embodied conversational system that generates level-appropriate dialogue for English language learners. These levels are based on those defined by the Common European Framework of Reference for Languages (CEFR) to describe non-native listener and speaker proficiency. Using a "generate-evaluate-regenerate" loop with multiple LLM agents and a level classifier, it achieves a desired simplicity that is adaptive to the user's proficiency level. We also share the results of a preliminary small-sample pilot study that tested this system with Japanese university students, to see whether it would yield lower FLA levels than an unsimplified embodied conversational agent. Analysis of conversational output showed that 87.4% of dialogue sentences generated by the proposed multi-agent system fell within one predicted CEFR level of the learner's self-assessed proficiency, compared to 54.1% for the unsimplified agent. This suggests that the novel system is better able to produce output at an appropriate level for the learner. Though this study did not yield statistically significant evidence that the system reduces FLA levels in Japanese learners of English, likely due to a small sample size, it provides usability findings and culturally-informed design insights that will inform future study. 

---
# LeAct: Learning to Reason from Expert Actions 

**Authors**: Ziran Yang, Chengshuai Shi, Raj Ghugare, Benjamin Eysenbach, Karthik Narasimhan, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2607.21856)  

**Abstract**: Modern reasoning models depend on reasoning data, today sourced from human annotations or distilled from stronger LLMs. However, a rich and largely untapped source of supervision lies in expert systems (e.g., game engines, classical planners, theorem provers), which routinely produce near-optimal actions across diverse domains. But these experts are silent: they commit to an action without writing down the chain of thought (CoT) behind it. Recovering that CoT as natural-language reasoning would distill expert knowledge into a student that generalizes beyond the demonstrated actions. We treat it as a latent variable and study how to recover it from the action alone. Our approach, LeAct (Learning to reason from Actions), optimizes this latent variable: the student samples candidate CoTs for each expert action, and we retain those that measurably improve its own probability of recovering the action. Across imperfect-information games at multiple scales and a simulated robotics benchmark, LeAct reaches the solver's numerical floor on small enumerable games. At larger scale, it is $5\times$ closer to the solver than the strongest expert-iteration baseline. At Flop Hold'em ($\sim 10^9$ infosets), LeAct wins head-to-head by $+60$ mbb/g, and on the robotics probe it is the only training recipe that improves on direct imitation. We present a principled framework and the result: expert systems become a categorically new source of reasoning teachers for foundation models. 

---
# Adversarial Prompts for Acceptance Collapse in Speculative Decoding 

**Authors**: Run Wang, Chaoyi Zhou, Xi Liu, Yi Zhu, Amir Salarpour, Pedram MohajerAnsari, Zhi-Qi Cheng, Feng Luo, Siyu Huang, Mert D. Pesé  

**Link**: [PDF](https://arxiv.org/pdf/2607.21804)  

**Abstract**: Lossless acceleration schemes, such as speculative decoding, promise significant inference speedups by relying on dynamic token-level alignment between a draft and a target model. However, this guarantee of semantic equivalence masks a severe operational vulnerability: draft-target alignment can be systematically attacked. In this paper, we introduce ADSD, which, to the best of our knowledge, is the first prompt-suffix attack that collapses verifier acceptance by pushing draft probability mass toward tokens the target is unlikely to accept. ADSD uses Soft-Collapse, a verifier-aligned surrogate derived from the asymmetric speculative acceptance rule, together with a target-preservation objective that discourages obvious task corruption. ADSD successfully generates highly effective adversarial suffixes. On the GSM8K dataset, our attack increases the mean sample time by 62.3% while preserving the task quality. We further show that this vulnerability exists across different domains, speculative decoding strategies, and model architectures. 

---
# From Seasonality to Semantics: Benchmarking a Hybrid Probabilistic Forecasting System for Roadblocks in Bolivia 

**Authors**: Rodrigo Vargas Sainz, Christian Berón Curti  

**Link**: [PDF](https://arxiv.org/pdf/2607.21785)  

**Abstract**: Roadblocks in Bolivia are a social conflict phenomenon with devastating economic impacts, estimated at losses equivalent to 4% of the national Gross Domestic Product. Despite their recurrence and impact, there is a lack of local predictive systems to anticipate these events for logistical decision-making. This paper presents a hybrid probabilistic forecasting system that integrates time series decomposition (Prophet) with natural language processing (NLP) techniques applied to a six-year corpus of Bolivian news coverage. The methodology employs vector semantic embeddings and zero-shot classification models to capture signals of discursive escalation prior to the materialization of the roadblocks. Using an expanding walk-forward validation scheme applied over 1,762 days and seven forecasting horizons (H+1 to H+7), seven internal configurations and four external benchmarks were compared, including SARIMA and LightGBM. The results demonstrate that the hybrid configuration (Prophet + NLP, C6) consistently outperforms purely statistical models, achieving an AUC-ROC of 0.677 at H+1 and reducing the Brier Score by 10.9% relative to the baseline temporal model (0.220 vs. 0.247), maintaining a statistically significant error reduction across all evaluated horizons ($p < 0.02$). This research validates that the integration of semantic news signals allows for the detection of social tension peaks not captured by historical inertia, providing a technical tool for risk management in critical transport corridors. 

---
# Learning What Matters: Supervising Sparse Attention Routing with Causal Evidence Sets 

**Authors**: Jim Allchin  

**Link**: [PDF](https://arxiv.org/pdf/2607.21692)  

**Abstract**: Sparse attention reduces the cost of long contexts by allowing each query to read only selected parts of the input. These selectors are often trained by distilling the attention patterns of a dense teacher, assuming that attention reveals which context the teacher actually uses. We test that assumption on retrieval tasks where the evidence for each answer is known exactly. By masking parts of the context and measuring whether the answer changes, we find that attention and causal dependence often disagree, and distilled selectors inherit the mismatch. Teachers attend to outdated facts they have learned to ignore, and their attention can vary across training runs even when they rely on the same evidence. In a two-step reference task, attention at the answer skips the intermediate step because it was resolved earlier in the forward pass: a selector trained on attention achieves 41% accuracy, while the same selector trained on causal evidence reaches 99% and matches the teacher. These evidence sets require no annotation: recovered from a frozen teacher by masking alone, they train selectors to the same accuracy. We find the same conflict in pretrained models: Qwen2.5-3B gives more attention to an outdated fact than the current one on 58% of conflicting-fact examples despite answering correctly, while Gemma-2-9B rises from 56% to 99% accuracy when restricted to the two relevant sentences. Attention shows where a model looks, not necessarily what its answer depends on; across the regimes we tested, that dependence matched or outperformed attention as a training target. 

---
# Progress Reward Modeling for Robotic Learning: A Comprehensive Survey 

**Authors**: Jianshu Zhang, Keliang Wu, Haoran Lu, Anbang Liu, Ce Zhang, Weijie Yin, Chengxuan Qian, Xiyuan Yang, Zhenyu Pan, Guo Ye, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.21655)  

**Abstract**: Robotic learning takes place in dynamic environments with large behavior spaces. A terminal success signal only tells the robot whether the task is completed. It does not explain whether the current behavior is making progress, remaining unchanged, or undoing earlier progress. For this reason, recent studies have increasingly explored progress rewards that provide feedback during task execution. However, the current literature lacks a shared framework. Existing methods use different observations, goal specifications, output signals, supervision sources, and evaluation protocols. This makes it difficult to compare them and understand what their results actually validate. In this survey, we provide a unified view of progress reward modeling for robotic learning. We organize the field in three connected steps. We first study the interface of a progress model. This defines the problem from the outside by asking what information the model receives and what form of progress signal it produces. We then move inside the model and study the methods used to construct this signal. This reveals the different assumptions and mechanisms behind progress estimation and reward generation. Finally, we examine the data and benchmarks that support these methods. This shows how progress supervision is obtained and what different evaluations actually measure. Together, these three perspectives connect what a progress model is, how it is built, and how its quality is validated. We further summarize the main limitations of current approaches and discuss future research directions. 

---
# Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning 

**Authors**: Jian Hu, Huiying Li, Hao Zhang, Binfeng Xu, Yifan Zhang, Shaokun Zhang, Hemil Desai, Michael Demoret, Pavlo Molchanov, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2607.21653)  

**Abstract**: Agentic reinforcement learning research is constant algorithm modification, new estimators, new pipeline stages, new rollout schemes, and in mainstream frameworks each change threads through layers of trainer, distributed backend, and rollout glue: the cost lands on the researcher at every iteration. Molt is a PyTorch-native training framework built to keep that cost small: a codebase compact and clean enough for a researcher to hold in their head, and for an AI coding assistant to read and reason about in its entirety, so the algorithm flow can be traced and changed end to end. The agent is an ordinary program, and one asynchronous loop trains multimodal and mixture-of-experts policies while never training on a token it did not generate, consistent in tokens, policy versions, and model semantics. Leanness does not cost performance: under a matched, fully asynchronous protocol, Molt is statistically comparable to a state-of-the-art Megatron-based stack. Molt is open source and provides recipes and containers at this https URL. 

---
# Do VLMs Read or Rewrite? On Transcription Faithfulness in Vision-Language Models 

**Authors**: Gwang Gook Lee, Kenan Emir Ak, Jay Mohta, Yan Xu, Dimitrios Dimitriadis  

**Link**: [PDF](https://arxiv.org/pdf/2607.21617)  

**Abstract**: Vision Language Models (VLMs) are increasingly used in place of traditional OCR pipelines for document understanding. In this paper, we show they do not always act as faithful transcribers: when text is imperfect, they often tend to rewrite it into a more plausible form - a behavior that clean-text OCR benchmarks cannot detect. We introduce FaithC4, a multilingual perturbation benchmark of 1,455 single-page documents (English, Chinese, Korean) with three perturbation families: scramble, random substitution, and visually similar substitution. We use the benchmark to evaluate 15 systems spanning general-purpose VLMs, OCR-specialized VLMs, and traditional OCR pipelines. These three categories differ in WER degradation under perturbation: general-purpose VLMs degrade by up to 4.5 points, OCR-specialized VLMs by 0.2-2 points, and traditional OCR by less than 0.6 points on English. Probing Qwen3-VL-4B layer-by-layer, we identify a consistent pattern: rewriting fires only when a perturbed word's final layer FFN representation stays close to the original encoding; when the representation diverges sufficiently, the model transcribes faithfully. Word length affects rewriting rate: short words (4-6 characters) are rewritten up to 10% of the time, with a sharp cutoff at 8 characters above which rewriting drops to 0%. 

---
# The Hard Decision Layer: Evidence for Committed Inference in Transformers 

**Authors**: Ashwath Vaithinathan Aravindan, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2607.21613)  

**Abstract**: We investigate where and how transformer-based language models commit to predictions in multiple-choice question answering. We identify the _Hard Decision Layer_ (HDL), a natural architectural property where answer option rankings stabilize abruptly during inference. Empirical validation across four language models (Qwen, Llama, Granite, Mistral) and four benchmark datasets demonstrates consistent HDL emergence without learned routing policies. We also show that the HDL is invariant to fine-tuning. Our results reveal striking accuracy improvements at the HDL: up to +0.61 (Qwen on CommonsenseQA), after which performance stabilizes. Systematic ablations on label formats and problem complexity confirm the phenomenon is fundamental to model architecture. These findings offer mechanistic insights into transformer inference and suggest opportunities for efficient reasoning and model steering. All code and results required to reproduce this work are available in this https URL 

---
# From Obligation to Specification: A Survey on Validating EU AI Act Requirements in RE 

**Authors**: T.Y. Emmy Lai, Sven Giesselbach, Matthias Koch, Héctor Allende-Cid  

**Link**: [PDF](https://arxiv.org/pdf/2607.21608)  

**Abstract**: With the EU AI Act entering into force, organizations developing or operating AI systems face new obligations on transparency, risk management, and traceability. For Requirements Engineering (RE), these obligations must be translated into testable, auditable requirements and verifiable evidence. However, many organizations currently lack systematic processes to achieve this. We hypothesize that LLM-based agentic validation tools can support this translation, thereby helping to close this gap. We present a mixed-method exploratory study with expert interviews (N=10) and an online survey (N=15) to assess organizational preparedness for EU AI Act-oriented RE and perceptions of LLM-based, agentic closed-loop validation tools, with participants spanning RE, data science, development, and compliance roles. Our results show that, although the EU AI Act is viewed as highly relevant, structured mechanisms to capture regulatory obligations, propagate updates into projects, and maintain lifecycle-wide traceability and evidence are often missing. Participants see LLM-based tools as promising for mapping obligations to requirements, assessing coverage, and organizing evidence, but express strong concerns about full automation and stress the need for safeguards. Based on these findings, we outline minimum requirements for an EU AI Act-ready closed-loop approach. 

---
