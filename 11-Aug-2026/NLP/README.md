# From Values to Benchmarks: Evaluating Large Language Models for Governmental Use in Dutch 

**Authors**: Laurens Samson, Iva Gornishka, Gossa Lô, Yuki M. Asano, Sennay Ghebreab  

**Link**: [PDF](https://arxiv.org/pdf/2608.09925)  

**Abstract**: Large language models are increasingly being deployed in governmental settings, yet few existing evaluation frameworks jointly reflect the values of public administration and the linguistic requirements of non-English contexts. We present the "Grip on LLMs" framework, a systematic evaluation suite for Dutch governmental use developed in collaboration with domain experts from a major Dutch municipal organisation. Through an advisory board process, user research, and a survey of the users of a civil-servant chatbot, we identify six evaluation dimensions (factuality, honesty, social bias, energy consumption, cost, and training data transparency) and operationalise them into a benchmark suite covering more than 30 multilingual and Dutch-specific models. Our results reveal that no single model excels across all dimensions, and that trade-offs are unavoidable: higher quality consistently comes at greater environmental impact and financial cost, while bias remains largely independent of both. We further find that factuality (whether a model answers correctly) and honesty (whether a model acknowledges what it does not know) are governed by distinct properties, with high factuality not implying high honesty. To make these findings actionable for non-technical audiences, we release a publicly accessible, user-friendly model overview designed for the full range of stakeholders involved in governmental LLM selection, from engineers to policymakers. 

---
# Decoding-Level Taboo: A Diagnostic Stress Test for LLM Robustness 

**Authors**: Tadanobu Chuyo Kamijo, Ori Rottenstreich, Javier Conde, Gonzalo Martínez, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2608.09900)  

**Abstract**: Large language model evaluations typically focus on performance under nominal conditions, creating an illusion of capability where models comfortably walk a narrow, highly optimized generation corridor. In real-world deployments, however, complex system prompts, safety guardrails, and structural constraints continuously force models off this nominal path, driving a divergence between benchmark scores and deployment performance. To address this issue, we introduce Decoding-Level Taboo, a zero-prompt diagnostic stress test that intervenes directly in logit space at runtime, forcing models out of their nominal paths. By dynamically masking primary candidate tokens at word boundaries, Taboo forces machine circumlocution.
Evaluating Taboo across several open-weight model families reveals that off-path robustness is heavily influenced by both parameter scale and post-training instruction alignment, with robustness generally improving with model size and alignment. Beyond the results presented in this paper, Taboo provides a novel primitive for generating diverse synthetic datasets, stress-testing runtime safety guardrails, and auditing model reliability prior to real-world deployment. 

---
# Consilience for Verifier-Free Test-Time Scaling 

**Authors**: Lecheng Kong, Like Hui, Haitao Mao, Jun Huan  

**Link**: [PDF](https://arxiv.org/pdf/2608.09898)  

**Abstract**: Test-time scaling often uses an external verifier, such as compilers and test cases in coding or trained value functions in robotics applications, to obtain high-quality rollouts. Verifier-free test-time scaling (or VF-TTS) is gaining extensive attention as a mechanism to enhance Large Language Model (LLM) reasoning, primarily because we do not have access to such high-quality verifiers in many real-world applications. Among existing VF-TTS methods, confidence-based VF-TTS methods, which compute and rank rollouts solely by confidence, are particularly promising. Such methods introduce near-zero overhead for sample evaluation and require minimal access to internal model states, making the methods highly flexible across models and tasks.
In this paper, we demonstrate a critical limitation of existing confidence-based VF-TTS methods by showing that such methods catastrophically break down on complex tasks. We observe a very interesting phenomenon: uniformly high confidence frequently indicates a failure to explore, favoring confidently wrong answers. To address this, our core insight is that robust cognitive search requires a specific confidence trajectory pattern: such methods perform exploratory branching at the beginning, as manifested by low initial confidence, and converge to a high final confidence solution. To implement this insight, we introduce consilience, a novel selection framework that explicitly evaluates the temporal asymmetry of confidence in reasoning. We operationalize this via a combinatorial metric that actively penalizes high initial confidence while strictly demanding final certainty. Extensive experiments covering both graduate-level mathematics problems and free-form code generation demonstrate that consilience effectively outperforms existing baselines, validating our novel perspective on completion confidence. 

---
# Fusion Training for Mathematical Generalization in Large Language Models 

**Authors**: Congfeng Cao, Pengyu Zhang, Jelke Bloem  

**Link**: [PDF](https://arxiv.org/pdf/2608.09893)  

**Abstract**: Thinking Mode Fusion (TMF) enables large language models to support both concise responses and long-form reasoning by unifying a non-thinking mode and a thinking mode within a single model. However, its training dynamics, including the \emph{data ratio} and \emph{training schedule} between the two modes, remain underexplored. In this work, we present a systematic study of TMF by analyzing the effects of the training schedule and data ratio between thinking and non-thinking modes. Focusing on mathematical problem solving, we construct a benchmark with multiple thinking-to-non-thinking data ratios and three training schedules. Our results reveal an asymmetric interaction between the two modes: increasing the ratio of non-thinking supervision reduces the accuracy of the thinking mode. We further show that different training schedules modulate this trade-off and that the optimal schedule depends on the data ratio. Finally, we quantify a negative correlation between non-thinking and thinking mode supervision, highlighting an inherent tension between these two modes. These findings provide practical guidance for designing effective TMF training settings. All code and data are released to support further research at: \href{this https URL}{\textbf{Fusion Bench}}. 

---
# RA-FinBERT: Rule-aware LoRA adaptation for low-resource financial sentiment classification 

**Authors**: Fan Zhang, Jiaming Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09834)  

**Abstract**: Financial sentiment analysis converts unstructured financial news into quantitative signals that can support market analysis and decision-making. Existing work on resource-efficient financial NLP has largely focused on compressing or adapting pretrained language models, with less attention to combining contextual representations with lightweight rule-derived features. This study develops Rule-Aware FinBERT (RA-FinBERT), a parameter-efficient framework that integrates low-rank adaptation (LoRA) with three continuous VADER-derived sentiment proportions (positive, negative, and neutral) and a source-level metadata feature. The standardized four-dimensional feature vector is directly concatenated with the 768-dimensional final-layer FinBERT [CLS] representation and passed through a lightweight classification head. This design introduces only 1,024 additional trainable weights relative to a structurally matched text-only FinBERT model. RA-FinBERT was evaluated against text-only FinBERT and a lightweight DistilBERT baseline for three-class sentiment classification of financial-news titles and descriptions. On the held-out test set, RA-FinBERT achieved 69.89% accuracy and a macro F1 score of 0.634, compared with 63.44% and 0.526 for text-only FinBERT. Neutral-class recall increased from 18.18% to 45.45%. The framework supports both CPU and GPU execution, offering a lightweight and practical approach to financial sentiment classification under constrained computational resources. These findings indicate that rule-derived sentiment information and source metadata can provide complementary signals to contextual FinBERT representations and improve performance with minimal additional model complexity. 

---
# SWE-Bench ProMax: Benchmarking Agents on Large-Scale Multilingual Code Refactoring 

**Authors**: Yuling Shi, Jinghan Xu, Kelin Fu, Wenhao Zeng, Shilin He, Lei Zhang, Yue Liu, Zelin Zhao, Terry Yue Zhuo, Jialun Cao, Siyu Ye, Tianyu Liu, Kai Cai, Shing-Chi Cheung, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09802)  

**Abstract**: As AI coding agents take on increasingly complex, long-horizon software engineering tasks, existing benchmarks are rapidly saturating and their evaluation quality has come under serious scrutiny: a recent audit found that nearly 60% of unsolved SWE-bench Verified instances contain flawed tests -- either overly narrow tests that reject correct solutions or overly broad tests that check unstated requirements -- and that frontier models can verbatim reproduce gold patches from training data. Code refactoring, which requires coordinated, behavior-preserving changes across many files, offers a substantially harder and more realistic test of agent capability, yet remains underserved by current benchmarks. We introduce SWE-Bench ProMax, an expert-curated, multilingual code refactoring benchmark of 170 instances drawn from real commits across seven programming languages (Python, Java, TypeScript, Go, C, C++, and Rust). Every instance undergoes rigorous, multi-stage curation that directly addresses the quality problems identified in prior benchmarks: issue descriptions are rewritten from scratch to provide precise, unambiguous specifications, and test suites are manually reviewed to remove overly narrow and overly broad tests. Tasks with insufficient complexity or limited cross-file scope are filtered out, yielding a benchmark of challenging, large-scale refactoring tasks that average 11.4 modified files and 261.6 lines of code per instance, substantially exceeding the scale of existing benchmarks. Experiments with frontier models under two agent scaffolds show that the best model achieves only 41.2% resolve rate, confirming that SWE-Bench ProMax presents a meaningful and unsaturated challenge for current AI coding agents. Our benchmark is available at this https URL. 

---
# Comparing British and American Audio Description of Movies 

**Authors**: Igor Sterner, Alex Lascarides, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2608.09792)  

**Abstract**: Narrating the visual component of movies is known as audio description. It is a narrative technique designed to enable blind and visually impaired individuals to follow the story. However, it is far more constrained than most narratives: the descriptions not only need to convey the story in the movie, but they must also fit into gaps between dialogue and they need to conform to guidelines that exist in each region. In this work, we compare audio description created in the United Kingdom against audio description created in the United States. We use guidelines written for these two regions, alongside the impressions from a practitioner in the field, to motivate specific hypotheses about the differences. We test these hypotheses against our pre-existing corpus, which provides both human-authored American and British audio description for each of 206 movies. Results provide quantitative evidence to uphold all tested hypotheses, including differences in lexicon, the use of the progressive aspect, the use of passive constructions, the use of subjective adjectives and modifiers, when characters are named, how scenes are cued, and degree of overlap with movie dialogue and music. Our work offers a quantitative lens into the narrative technique of audio description. 

---
# KGCaRe: Explainable Complex Conditional Question Answering using Automatic Knowledge Graph Construction and Context Retrieval with LLMs 

**Authors**: Ghanshyam Verma, Simanta Sarkar, Devishree Pillai, Hotaka Shiokawa, Yourong Xu, Fiona Veazey, Peter Hubbert, Hui Su, Paul Buitelaar  

**Link**: [PDF](https://arxiv.org/pdf/2608.09779)  

**Abstract**: Answering complex conditional questions using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) remains a challenge, particularly in domain-specific contexts where general-purpose LLMs and RAG tend to underperform. We hypothesize that augmenting RAG with unstructured and structured knowledge, extracted from both documents and knowledge graphs (KGs), can improve reasoning and answer accuracy for such tasks.
To test this, we propose KGCaRe, a hybrid approach that combines neural retrieval with symbolic reasoning over LLM-generated KGs. KGCaRe constructs a KG from documents using a multi-prompt extraction strategy and stores it in a graph database. Simultaneously, the documents are embedded into a vector store to enable neural retrieval. KGCaRe performs innovative iterative graph traversal guided by the LLM to extract relevant triples, prune irrelevant information, and uses additional clue entities to traverse the graph again if the initial traversal does not provide satisfactory context to generate the answer. The relevant triples extracted from the KG in path form, along with semantically retrieved text passages, are then fed into custom KGCaRe prompts to generate answers to the complex conditional questions with explanations.
We evaluate KGCaRe on two complex conditional QA datasets. Our results on these datasets show that KGCaRe consistently outperforms existing baselines, including Vanilla LLM, Code Prompt, Text Prompt, Think-on-Graph, Vanilla RAG, and HybridContextQA, across multiple LLMs such as Mistral, Mixtral, GPT-3.5, and GPT-4o. We publicly release the software pipeline that we developed to implement the proposed KGCaRe approach. 

---
# PragMatch: Separating Pragmatic Incongruity from Cross-Modal Mismatch in Large Vision-Language Models 

**Authors**: Zhanna Mukhametsharip, Vera Demberg, Varsha Suresh  

**Link**: [PDF](https://arxiv.org/pdf/2608.09772)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated strong performance on multimodal benchmarks, yet it remains unclear whether they genuinely reason about relationships between images and text or rely on superficial correlations, known as shortcut learning. This question is particularly important for multimodal sarcasm detection, where successful prediction depends on recognizing pragmatic incongruity rather than treating sarcasm as simple image-text mismatch. We introduce PragMatch, a controlled benchmark of 3,000 image-text pairs derived from MMSD2.0, including original sarcastic examples and constructed literal and hard-negative pairs. We identify influential shortcut cues through systematic masking and evaluate their impact through targeted injection experiments. Our results show that LVLM predictions are sensitive to lexical, OCR-derived and stylistic cues, with injected surface signals causing substantial changes in model predictions despite unchanged underlying image-text relationships. Our findings reveal limitations in current LVLMs while PragMatch provides a systematic testbed for evaluating multimodal pragmatic reasoning beyond surface-level image-text alignment. 

---
# Structured Phonological Representations for Audio-Articulatory rtMRI Speech Classification 

**Authors**: Abner Hernandez, Tomás Arias Vergara, Daiqi Liu, Andreas Maier, Paula Andrea Pérez-Toro  

**Link**: [PDF](https://arxiv.org/pdf/2608.09767)  

**Abstract**: Real-time MRI makes it possible to observe vocal-tract articulation during speech, but mapping these articulatory patterns to phonetic and phonological categories remains challenging. We investigate whether PhonoQ, an audio-based model trained to recognize structured phonological features, provides useful information for audio--articulatory modeling. Specifically, we extract representations from PhonoQ's Conformer module, whose training is shaped by supervision for manner, place, voicing, and vowel features. Using articulatory contours with synchronized audio-derived features, we compare WavLM-large and HuBERT-large baselines with models that incorporate PhonoQ-derived representations. Across unseen-speech and unseen-subject settings, these features improve macro-F1 for phonological targets including manner, place, voicing, vowel height, and vowel backness, and also improve fine-grained 39-phoneme classification. In a contour-only inference setting, audio-derived teacher supervision yields modest but consistent gains over contour-only training, indicating that phonological information from synchronized audio can be partially transferred to articulatory models. Finally, posterior analyses show interpretable surface-sensitive patterns consistent with flapping-like /t/ realizations, /t/-/r/ retraction or affrication, and nasal place assimilation. 

---
# Cultivar: A Contrastive and Locale-Oriented Translation Benchmark for Investigating Contamination and Localisation Robustness 

**Authors**: Pinzhen Chen, Koel Dutta Chowdhury, Xiaoya Xu, David Tan, Doreen Osmelak, Ona de Gibert, Ariun-Erdene Tumurchuluun, Ashok Urlana, Fedor Sizov, Hale Sirin, Jesujoba Alabi, Karrar Talib Abed, Mateusz Klimaszewski, Nikolay Bogoychev, Niyati Bafna, Patricia Schmidtova, Preksha Manjunath Shanbhag, Sherrie Shen, Vilem Zouhar, Vivek Iyer, Yasser Hamidullah, Yusser Al Ghussin, Zheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.09766)  

**Abstract**: Multilingual translation benchmarks are typically sourced in English and translated into other languages, treating language pairs as the unit of evaluation---a design that is prone to contamination over time and overlooks locale and cultural considerations. We therefore advocate for source-contrastive evaluation and instantiate it with Cultivar, a localised subset of FLORES, which enables locale-specific translation evaluation. When paired with unlocalised counterparts, performance discrepancy allows the probing of data contamination and localisation robustness. We benchmark 32 open-weight models and find that MT-specialised models are less robust, a few models potentially overfit FLORES, and models tend to translate US content better than that of other locales, regardless of language. 

---
# REFRAMED: Towards Realistic Audio Description Generation for Movies 

**Authors**: Igor Sterner, Mirella Lapata, Alex Lascarides, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2608.09765)  

**Abstract**: Audio Description (AD) is a verbal narration of key visual content in videos, enabling access for visually impaired audiences. Unlike standard video captioning, AD is a structured editorial task: descriptions must be inserted into gaps in dialogue and must convey only what is needed to understand the narrative being told. However, existing approaches formulate AD generation in an artificial setting where both the content and timing of descriptions are pre-specified, reducing the task to clip-level captioning. They further rely on noisy transcription and alignment pipelines, and lack the rich parallel data required for modeling narrative context. We introduce a new formulation of AD generation in which models must jointly decide what to describe and when to do it. To support this, we present REFRAMED, a high-quality dataset of 2,023 videos that span 3,302 scenes from 206 movies, with professional AD transcripts (both American and British versions), professional subtitles and aligned screenplays. We also provide a manually curated challenge set that pairs full movies with multiple AD references, together with evaluation protocols that leverage dialogue gaps and multi-reference comparisons. Experiments with state-of-the-art AD systems and multimodal LLMs show that they outperform trivial baselines but fall far short of expert human performance. Our dataset and benchmark establish a new foundation for research on video understanding. 

---
# How Do Large Language Models Judge Social Attraction? Evidence from Theory-Grounded Persona Ratings Across Multiple LLMs and Humans 

**Authors**: Hasan Mahmud, Khawaja Abaid Ullah, Mohammad Javad Khojasteh, Jamison Heard, Prabu David  

**Link**: [PDF](https://arxiv.org/pdf/2608.09717)  

**Abstract**: Large language models (LLMs) are increasingly used to perform subjective evaluations traditionally made by humans, yet their validity as social judges remains unclear. This paper examines whether LLMs can assess social attraction from theory-grounded persona profiles constructed from ten psychological and relational constructs and organized into three tiers: socially attractive, socially mixed, and socially unattractive. We examine LLM ratings in two studies and compare them with human judgments in a third study. In Study 1, 34 LLMs rated 12 profiles across three repeated runs. Although some models tended to give higher or lower ratings overall, they showed strong stability across runs, consistent three-tier ordering, and high agreement in relative profile ordering. Study 2 examined sensitivity to gender presentation using six matched name-and-pronoun profile pairs and a separate pronoun-only test with a gender-neutral name, finding no significant effects in either analysis. In Study 3, 198 human participants evaluated the six matched profiles from Study 2. Their ratings reproduced the three-tier structure and followed a profile ordering consistent with that of the LLMs. However, LLMs rated attractive profiles more positively and unattractive profiles more negatively than humans, while neither group showed a significant overall effect of gender presentation. 

---
# Measuring the Wrong Thing: Internal Harmfulness Scores Anti-Rank Successful Jailbreaks 

**Authors**: Mingyu Luo, Ming Deng, Zilang Qiu, Yiming Cheng, Ci Tao, Xue Tan, Sijin Sun, Yangfu Li, Ping Chen, Jun Dai, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.09624)  

**Abstract**: Internal safety scores judge a prompt before any text is generated, and they are validated by how well they separate harmful prompts from benign ones. That separation is then read as evidence that the score will also catch the attacks that succeed. Harmful intent is a property of the prompt. Jailbreak success is an outcome produced later by a particular target model, decoding policy, and judge. A filter tuned on a score that measures the wrong quantity spends its false positive budget on attacks that would have failed anyway. In this paper we audit that inference. Attention based measurements are usually read from prompt dependent locations, so a wrapper changes both the content being judged and the place the signal is taken from. We therefore introduce Active Attention Probing, which supplies a fixed content independent measurement coordinate. We pair every base goal with a plain and a wrapped version and generate real completions from the target models. On Llama, wrapping raises harmful generation from 0.05 to 0.27 while harmful intent AUROC falls from 0.936 to 0.803, so the attacks grow more dangerous while the prompts look safer to the score. Among wrapped harmful prompts the outcome AUROC is 0.220, which places the attacks that succeeded below the attacks that failed. Rare token, passive, and detector derived channels reproduce the reversal on the same matched design, and the reversal itself persists across three target models, seven attack families, and two independent judges. Distribution shift then degrades calibration and threshold transfer before it degrades ranking. 

---
# MDB-Link: Hierarchical Schema Linking for Multi-Database Text-to-SQL 

**Authors**: Beiyu Xu, Zhenyu Wu, Jiaoyan Chen, Riza theresa Batista-navarro  

**Link**: [PDF](https://arxiv.org/pdf/2608.09588)  

**Abstract**: Traditional Text-to-SQL research and benchmarks assume a known target database, overlooking settings in which a query must be routed within a large, heterogeneous database collection. We therefore study schema linking in a multi-database setting, where the system must first locate the target database and then construct a compact, SQL-relevant schema for generation. We propose MDB-Link, a hierarchical schema-linking framework that retrieves question-relevant columns from a global index, aggregates retrieval evidence to shortlist databases, and uses a budget-aware large language model (LLM) for database reranking, table selection, and column grounding. With Qwen2.5-14B, MDB-Link outperforms LinkAlign on MMQA, Spider2-Snow, and BIRD-dev in database localization and column selection while producing schema subsets close in size to the gold schemas. Exact match improves from 16.88 to 51.41 on MMQA, 2.50 to 9.17 on Spider2-Snow, and 12.52 to 38.01 on BIRD-dev. MDB-Link also runs faster than LinkAlign and AutoLink, demonstrating the effectiveness of hierarchical schema reduction for downstream SQL generation. 

---
# Se-DPO: Self-Evolving Token Credit for Direct Preference Optimization 

**Authors**: Wenxiao Zhao, Shu Wang, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09568)  

**Abstract**: Direct Preference Optimization (DPO) aggregates token-level log-probability ratios via uniform summation, implicitly treating all tokens as contributing equally to the preference signal. However, the contribution of individual tokens to the preference signal varies. We introduce token credit, which modulates each token's KL regularization based on its contribution to the preference outcome. We derive that effective token credit is proportional to the magnitude of each token's implicit reward, and observe that this quantity evolves substantially during training. This implies that static token credit becomes increasingly misaligned as training progresses. In this work, we propose Se-DPO (Self-Evolving Token Credit for DPO), a live mechanism that derives token credit from the model's own evolving internal signals during DPO training. Since the reward signal varies in reliability across positions, Se-DPO calibrates token credit based on both the strength and the confidence of each token's contribution. Se-DPO requires no external models, adding only a lightweight calibration network with minimal computational overhead. Experiments show that Se-DPO improves over DPO by up to 9.8 points on AlpacaEval~2 and 12.2 points on Arena-Hard. 

---
# Pragmatic Attack Surface: Vulnerabilities of Implicit Context in Large Language Models 

**Authors**: Bocheng Chen, Han Zi, Roucheng Ou, Yawei Liu, Minyue Chen, Zimo Qi, Rongrong Wang, Guangliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09551)  

**Abstract**: In the era of large language models (LLMs), attackers often manipulate natural language to elicit unsafe or harmful outputs, creating a new natural language attack surface unique to LLM-based systems, where attacks directly exploit explicit linguistic cues in user prompts to bypass the safety mechanism of LLMs. However, such attacks can often be mitigated by existing safety alignment algorithms. On the other hand, human language is inherently grounded in pragmatics, necessitating typical context to interpret language, e.g., world knowledge, social norms. However, such contexts are often implicit because they are not directly expressed in human language and are not sufficiently leveraged in safety alignment, creating a fundamental mismatch between human language interpretation and safety alignment approaches. In this paper, we demonstrate that this mismatch exposes vulnerabilities in LLMs. We refer to this vulnerability as the pragmatic attack surface, which can be exploited to achieve high attack success rates. The experimental results demonstrate that our proposed approach outperforms baseline attack methods across various open-source and closed-source models by a substantial margin. 

---
# ELBench: A Multi-Dimensional Benchmark for Education-Facing Large Language Models 

**Authors**: Yilin Jiang, Xiaorong Zhu, Fei Tan, Zicheng Zhang, Kaiyi Huang, Yang Yu, Zexuan Fei, Yiming Luo, Keqian Li, Hao Hao, Aimin Zhou, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2608.09548)  

**Abstract**: Large language models are increasingly deployed in education as tutors, teaching assistants, and content generators. These roles place demands that ordinary question answering does not: a usable education-facing model is supposed to be accurate, safe under sensitive prompts, instructionally useful, and aligned with pedagogical goals at the same time. Existing benchmarks evaluate these requirements largely in isolation, so none assesses education-facing suitability as an integrated profile. We introduce ELBench, the first benchmark to evaluate all four requirements (General Capability, Safety and Trustworthiness, Basic Education, and High-Level Cultivation) on the same models under a common protocol, combining curated public sources with newly synthesized safety and cultivation data. We evaluate nine models, seven frontier general-purpose systems and two education-specialized variants, and report three findings. First, module-level profiles are more informative than a single aggregate: the top six models are statistically indistinguishable on overall score, yet their module leaders differ substantially, and safety is anti-correlated with practical teaching (r = -0.83). Second, the Chinese-developed models lead the safety module, the most discriminative in the suite; this advantage is largest on region-specific normative content and narrows, but does not vanish, on universal-harm content. Third, the two education-specialized models lead neither education module, and on High-Level Cultivation all models share a systematic blind spot: on the structured judgment task they converge on the same non-reference option, favoring pedagogical style over fit to the stated goal, so the module scores uniformly low and does not separate models. This raises, but does not resolve, whether domain post-training keeps pace with frontier systems on education tasks. 

---
# Mawqif-v2: An Arabic Benchmark Dataset for Cross-Target Stance Detection 

**Authors**: Rasha Albalawi, Nuha Albadi, Hamzah Luqman, Maram Kurdi, Saad Ezzini, Asma Yamani, Ahmed Ashraf  

**Link**: [PDF](https://arxiv.org/pdf/2608.09539)  

**Abstract**: Publicly available Arabic datasets for target-specific stance detection remain limited, particularly for evaluating cross-target generalization. This paper presents the Mawqif-v2 Extension, consisting of 996 manually annotated Arabic tweets collected from three public targets: Women Driving, E-Cars, and Trimester System. Each tweet is annotated with stance, sentiment, and sarcasm labels following the original Mawqif annotation scheme. The released extension is intended as a held-out evaluation set for assessing model generalization to both semantically related and previously unseen targets, while the original Mawqif dataset is used for training and development. In addition, we establish baseline results using several Arabic and multilingual transformer models, as well as zero-shot large language models (LLMs), to facilitate reproducible evaluation. Together with the original Mawqif dataset, the Mawqif-v2 Extension provides a benchmark for evaluating cross-target generalization in Arabic stance detection. 

---
# TCS-BENCH: Benchmarking State-of-the-Art Generative AI Theoretical Computer Science Research Ability 

**Authors**: Vincent Cohen-Addad, Dimitris Paparas, Ernest van Wijland, Max Springer, Julien Canitrot-Paradis, Honghao Lin, David Woodruff, Adarsh Kumarappan, Rajesh Jayaram, Rudrajit Das, Lalit Jain, Ola Svensson, Silvio Lattanzi, Mislav Balunovic, Theophane Weber, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2608.09538)  

**Abstract**: We introduce TCS-Bench, a benchmark for evaluating Large Language Models (LLMs) on research-level Theoretical Computer Science (TCS) proof generation. TCS-Bench consists of theorem-proving tasks from papers published at top theoretical computer science venues (STOC, FOCS, and SODA). Each task provides the necessary context to derive a self-contained proof for a target result. We evaluate state-of-the-art models on this benchmark. We verify the correctness of generated proofs via a verification agent, and further benchmark the verifier against human-expert proof judgements on a set of target statements and generated proofs pairs. Our reference verifier achieves over 90% accuracy on the expert labeled set. 

---
# Build it, Break it, Repeat: Benchmarking and improving LLM-manipulated disinformation detection in social media posts 

**Authors**: Kevin Thomas, Milosz Kasprzyk, Reuel C Igbokwe Onuigbo, Elliott Pert, Cameron Tovey, João A. Leite, Olesya Razuvayevskaya, Carolina Scarton  

**Link**: [PDF](https://arxiv.org/pdf/2608.09510)  

**Abstract**: Detecting machine-generated disinformation on social media is increasingly difficult as large language models (LLMs) make it easier to generate and rewrite misleading content at scale. Static benchmark evaluations, measuring detector performance on fixed held-out datasets, do not capture how detectors behave when posts are deliberately transformed to evade classification. This paper adapts the Build it, Break it, Fix it framework into Build it, Break it, Repeat (BiBiR): iterative sessions designed to stress-test detectors' robustness under iterative adversarial conditions, evaluating whether models remain reliable when disinformation posts are systematically transformed to evade classification. Across five iterations, the findings show that the best adversarial breakers' transformations came from a combination of back-translation and LLM persona-based rewriting, with the best performing technique achieving a 95% label flip rate (LFR), whilst still preserving the meaning of the original posts. The best builders' model was a triplet contrastive model with a dynamic anchor switching (DASS) architecture, which achieved an average accuracy of 72.68%, outperforming the strong baseline (a fine-tuned e5-small-LoRA) by 15 percentage points on the most robust set of breakers' adversarial attacks. The results demonstrate that an iterative framework best exposes detector weaknesses and pushes robustness improvements; however, it may still require semantic preservation analysis to distinguish valid adversarial evasion from transformations that changed the original disinformation claims' meaning. 

---
# Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning 

**Authors**: Yuting Liu, Wei Wu, Jianzhe Zhao, Guibing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.09507)  

**Abstract**: Natural language user preferences provide an interpretable interface for LLM personalization. However, universal preference summaries often contain information irrelevant to a particular downstream task. Directly supplying the full preference summary therefore wastes context capacity and introduces cross-task distraction, while manually designing task-specific preference views is difficult to scale. In this work, we study \emph{task-specific preference adaptation}: given a universal user preference summary and a downstream task, derive a task-conditioned representation that preserves sufficient decision-relevant evidence while removing redundant context. To this end, we propose \textsc{AlignXada}, a training-free meta-learning framework that induces reusable textual refinement policies for adapting universal preference summaries to task-specific ones. The refinement policy is iteratively optimized by a meta learner through verbal reinforcement learning. Across 13 tasks and three downstream models (39 task--model cells), \textsc{AlignXada} achieves an average gain of 3.82 points, improving 33 cells while retaining only 22.8\% of the original profile tokens and outperforming RAG in 36 cells. An extended faithfulness analysis further shows that the refined profiles remain largely grounded in the source preferences while preserving task-relevant personalization signals, suggesting that profile-side adaptation serves as a practical complement to universal memory construction for lifelong personalized agents. 

---
# ZetaGPT: A Reference Implementation of Positional--Encoding--Free State--Space--Attention Language Models 

**Authors**: Róisín Luo  

**Link**: [PDF](https://arxiv.org/pdf/2608.09432)  

**Abstract**: Transformer-based language models rely on self-attention, whose computation is permutation-equivariant and therefore lacks an intrinsic mechanism for representing token order. Existing architectures address this limitation by explicitly incorporating positional information through learned positional embeddings or hand-crafted positional encodings, such as rotary positional encoding (RoPE), treating positional information as an architecturally acquired capability rather than an inherent property of the model. Motivated by the pursuit of positional-encoding-free architectures, this work explores a language model architecture that integrates causal state-space equations to implicitly encode positional information before attention computation. Specifically, each model block applies a causal state-space equation before self-attention, allowing recurrent state dynamics to encode sequential information into token representations. Consequently, subsequent attention layers operate on position-aware representations without requiring explicit positional encodings while retaining the expressive modeling capacity of self-attention. We present \textsc{ZetaGPT}, a compact hybrid language model designed for research, rapid prototyping, algorithm verification, and educational applications. In addition to the proposed architecture, \textsc{ZetaGPT} provides a fully open-source, end-to-end training pipeline encompassing dataset construction, tokenizer training, pretraining, supervised fine-tuning, reinforcement learning from human feedback (RLHF), and chain-of-thought (CoT) reasoning via pure reinforcement learning. To the best of our knowledge, \textsc{ZetaGPT} is the first open-source small language model without explicit positional encoding and establishes a compact, reproducible reference implementation for the development and empirical study of positional-encoding-free language models. 

---
# Reducing Pretraining-Generation Mismatch in Diffusion Language Models 

**Authors**: Xiaocheng Lu, Huabin Liu, Song Guo, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.09424)  

**Abstract**: Autoregressive language models align training and use: generation conditions on a clean prompt, and training predicts future tokens from clean left context. Diffusion language models offer parallel denoising, but native dLLM pretraining can randomly corrupt prompt and continuation tokens together, weakening the clean-prefix interface needed for prompt-conditioned generation. We identify this mismatch for prompt continuation and propose PCD (Prefix-Conditioned Diffusion), a pretraining objective that combines AR prefix supervision with no-shift suffix denoising. At the training-objective level, PCD changes the attention mask, corruption mask, and label construction in continued pretraining; it does not require an autoregressive decoder, verifier, or new inference mode. By supervising the clean-prefix side autoregressively and applying diffusion only to the unknown continuation, PCD makes the local training interface resemble how block-diffusion models are queried at evaluation time. We further separate intra-sample prefix conditioning from inter-sample objective mixing, allowing us to identify the local alignment signal separately from the optional batch-level mixing knob. Across LLaDA2-Mini and Qwen-1.7B backbones, PCD consistently improves over same-family native dLLM stable baselines, reaching a 4.2% relative gain on the main LLaDA2-Mini six-benchmark average (+2.56 points) and a 14.2% relative gain in the primary Qwen mechanism comparison (+4.86 points). These results suggest that aligning the pretraining context distribution with prompt-conditioned generation can recover a measurable part of the dLLM continuation gap without changing inference. 

---
# Intent Speaks Louder: Controllable User Simulation Beyond Response Imitation 

**Authors**: Bo Wang, Ruixing Zhang, Yunqi Liu, Yang Zhang, Liangzhe Han, Tongyu Zhu, Leilei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.09420)  

**Abstract**: User simulators are widely used as scalable environments for training and evaluating interactive assistants. Generating the next user turn is inherently one-to-many: the same profile and dialogue context may support multiple plausible continuations with different local interaction intents. A fluent response may therefore advance the dialogue through an inappropriate intent, such as acceptance rather than repair. Our key insight is that controllable user simulation should separate which local interaction intent the next user turn should realize from how that intent is expressed in language. We introduce UserIDA (User Intent-Directive Alignment), which exposes interaction intent as an explicit per-turn directive. UserIDA defines a six-way intent interface, learns directive-conditioned generation through supervised fine-tuning, and uses intent-calibrated policy optimization during group-based reinforcement learning. The reward preserves composite response quality while ensuring that intent-violating candidates rank below compliant alternatives in mixed groups. On LMSYS-USP, UserIDA achieves 86.6\% intent accuracy, outperforming the strongest dedicated user-simulator baseline by 24.3 percentage points while improving semantic and stylistic similarity. In within-context interventions, it realizes at least four of the six target intents in 91.7\% of evaluated dialogue states, compared with 22.9\% for the strongest external baseline. These results establish per-turn intent control as a complementary dimension to response fidelity in user simulation. 

---
# Temporal Misgrounding in Legal RAG: A Versioned-Corpus Benchmark for French Tax Law 

**Authors**: Rose Cymbler, Daniel Guez, Laurent Fabre  

**Link**: [PDF](https://arxiv.org/pdf/2608.09393)  

**Abstract**: We identify and quantify temporal misgrounding: the systematic retrieval and citation of the currently in-force version of a legal article when the applicable version is an earlier or future one. Standard legal RAG treats the corpus as static; we argue legal question answering is a temporally-indexed retrieval problem. We introduce FiscalQA Pro, pairing a versioned corpus of 32,436 article-versions of the French tax code (93 years, 1938-2031) with an all-model-hard temporal-reasoning track: 209 scored, expert-reviewed questions across 33 CGI articles (221 released; twelve flagged out of the answerable scope). At selection time, no evaluated model recovered its date-applicable answer closed-book in any of four sampling draws, and the currently in-force text lacks the gold value for all but one of the scored questions. Answers are scored deterministically via atomic ground-truth "nuggets" (regex and numeric-with-tolerance), never LLM-as-judge: an LLM judge would inherit the temporal bias it is meant to score. Across eleven models (five frontier closed-API systems plus Gemini 2.5 Pro as a substitute entry, and five open-weight), parametric knowledge yields 3.0% mean strict accuracy and RAG over a static current-version corpus 2.7%. Static RAG retrieves the date-applicable version 0% of the time, confidently citing a real but inapplicable version. Our end-to-end retriever over a multi-version index, with no oracle, reaches 98.3% mean strict; an oracle-article ablation reaches 99.1%, locating the residual gap in first-stage recall, not version selection. We additionally release a version-aware jurisprudence dataset of 69,208 citation links, together with the corpus, benchmark, model responses, and pipeline code. 

---
# Universal or Language-Family-Specific Script Unification for Cross-Lingual Transfer? A Case Study on Turkic Languages 

**Authors**: Zijie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09356)  

**Abstract**: Closely related languages written in different scripts expose little surface overlap to multilingual models, limiting cross-lingual transfer. We compare two approaches to script unification: the general-purpose uroman romanizer and the family-specific Common Turkic Script (CTS). We train matched fastText models on transliterated Wikipedia corpora from 11 Turkic languages and evaluate them on WikiANN named entity recognition and Universal Dependencies part-of-speech tagging. CTS and uroman show no significant difference on NER, while both substantially outperform the official monolingual fastText baselines. POS results reveal no universal winner: language-specific differences are associated with the cross-lingual character n-gram coverage induced by each representation, while within-language coverage becomes more important when target-language supervision is available. Although CANINE-c achieves higher overall POS averages, the substantially simpler fastText-based systems remain competitive on several treebanks. Overall, the effectiveness of script unification depends on the language, the induced subword overlap, and the available supervision. 

---
# Accurate but Natural? Diagnosing Grammatical and Idiomatic Gaps in Japanese EFL Writing 

**Authors**: Steve Woollaston, Brendan Flanagan, Hiroaki Ogata  

**Link**: [PDF](https://arxiv.org/pdf/2608.09289)  

**Abstract**: Second language writing research distinguishes grammatical accuracy from native-like idiomaticity, yet automated writing evaluation often conflates these dimensions. This study introduces a layered LLM-correction pipeline that isolates structural errors from unnaturalness by generating literal error corrections and idiomatic revisions for 3,830 English writing samples from 120 Japanese junior high school students. Applying the regex-based CEFR-J grammar extractor, we quantify two diagnostic measures: accuracy gaps (structures attempted but incorrectly produced) and idiomatic gaps (grammatically correct structures underused or overused relative to native norms). Results reveal distinct patterns: definite articles, third-person singular -s, and modals (would, could) exhibit significant accuracy difficulties, while -ing forms and hypothetical modals (would) show the largest idiomatic underuse, with simple present verbs, subject-verb-object patterns, and modal can conversely exhibiting the most pronounced overuse. A two-dimensional instructional typology maps error rates against idiomatic gaps, distinguishing accurate but overused grammar items from error-prone or avoided complex forms requiring targeted production practice. This framework advances pedagogical feedback by enabling teachers to diagnose whether learner difficulties arise from inaccurate execution, structural avoidance, or L1-mapped overreliance, supporting evidence-based interventions tailored to the specific needs of each learner. 

---
# Is the ACL Responsible NLP Checklist a Box-Ticking Exercise? A Large-Scale Analysis of EMNLP 2025 

**Authors**: Nusrath Jinnath, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.09280)  

**Abstract**: Responsible NLP practice includes a) transparency, b) ethics, and c) societal impacts. The Responsible NLP Checklist aims to push these goals, and promote responsible practice. Recently, ACL released the EMNLP 2025 Checklists to aid transparency on the current research practice, which we focus on. We curate and release the first two datasets of: a) all the checklist responses and justifications from the EMNLP 2025 Main and Finding tracks; b) checklist reference linking to paper sections. We also provide the first analysis of recent EMNLP Checklists, by examining $73,922$ responses and justifications to them. For the Main track, we find that authors isolate ethics questions of the Checklist from the paper's bulk, mimicking the trend of ethics being an afterthought. We then examine \texttt{NO} responses. We find $44.9\%$ of justifications are poor or bad-faith, being brief or empty. Then, we find significant issues with the checklist design and effort of authors, namely that $6\%$ of all checklists contained logical contradictions between parent and child responses. We also find evidence of surface compliance for responsible ethics, with $53\%$ authors dismissing potential risks or social impacts of their work, for which there should be none. We compare this to the Findings track, noticing a similar trend in both tracks. Lastly, we discuss the implications of the checklist design and provide recommendations for future checklist iterations. Including: a) enforcing a minimum word count, b) enforcing more scrutiny on the risks of appliances. 

---
# Verifiably grounded machine interpretation of lunar geology 

**Authors**: Tom Sander, Kay Wohlfarth, Christian Wöhler  

**Link**: [PDF](https://arxiv.org/pdf/2608.09276)  

**Abstract**: Planetary geology relies on historical, interpretive reasoning to reconstruct past events from diverse observations. Here, we present a step toward an automated "machine intelligence geologist" by embedding this distinct methodology of geologic knowledge discovery and inference into a multimodal vision-language architecture. Focusing on the stratigraphy of lunar basaltic mare volcanism, we train a model to generate verifiably grounded geologic interpretations directly from co-registered topographic, spectral, and geologic maps. We demonstrate that while the system successfully balances established geological priors with local visual evidence to accurately describe stratigraphy and terrain, numeric age dating derived solely from vision defaults to memorized priors. Integrating an open-book retrieval mechanism resolves this, enabling the model to faithfully cite published chronologies. Our findings delineate the necessary architecture for automated geologic inference: site evidence must be visually interpreted from local data, while quantitative historical context must be retrieved from the scientific record. 

---
# Reading Cognition as Decisions Unfold in Words: A Factorized Inverse Decision Model 

**Authors**: Jiawen Kang, Dongrui Han, Xixin Wu, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2608.09222)  

**Abstract**: Inverse decision modeling infers latent properties of decision processes from observed behavior, but existing formulations rely primarily on action trajectories. In verbalized cognitive tasks, task execution also produces response dynamics that action-only formulations leave unmodeled, such as verbal production, interaction, and hesitation. We propose a factorized inverse decision model (FIDM) that decomposes each individual's task-execution likelihood into an action factor and an effort factor, governed by separate individual-specific parameters. From raw verbal transcripts, a language model produces structured task-execution traces for factorized inference. On data from 400 older adults performing a grocery-shopping dialog task for cognitive screening, controlled recovery shows selective estimation of the intended factors, while matched semi-synthetic conditions show that FIDM preserves action-execution distinctions even when aggregate behavioral summaries are matched. Action evidence further localizes task-defined deviations across participants. In cognitive-status classification, FIDM provides information complementary to clinical scores, trajectory summaries, and frozen language representations, with consistent gains across all evaluated baselines in the binary setting. 

---
# UNMASK: Discovering and Causally Verifying Spurious Shortcuts in Text Classifiers 

**Authors**: Chidaksh Ravuru, Shashank Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2608.09209)  

**Abstract**: Neural language models trained on large crowdsourced corpora frequently exploit spurious surface patterns tied to target labels without true linguistic or causal relevance, boosting benchmark performance while failing on adversarial or out-of-distribution inputs. Existing approaches either require manual specification of the feature vocabulary or automate discovery only partially, leaving the gap between dataset-level correlation and model-level exploitation unaddressed. We present U N M ASK, a fully automated pipeline that discovers, causally verifies, and mitigates spurious correlations in text classifiers without additional human annotation. Given unlabeled training examples, U N M ASK generates candidate surface patterns as executable boolean expressions, filters them through a statistical validation protocol with independent replication, and establishes causal model dependence via verified counterfactual interventions. Causally confirmed features then serve as annotation-free group definitions for Deep Feature Reweighting, eliminating the group labels that standard DFR requires. Applied to BERT and RoBERTa trained on MNLI, our pipeline independently rediscovers established lexical-overlap and negation biases, verifying 9 of 10 features on BERT and 6 on RoBERTa, and improving HANS accuracy by up to 12.58 pp. On CivilComments-WILDS, programmatic groups match the 70.1% worst- group accuracy of hand-labeled DFR (Kirichenko et al., 2023) without demographic annotation. We further demonstrate that the discovery and validation stages generalize to reward model preference data, surfacing interpretable spurious correlations in RewardBench2. 

---
# EmoS: A Theory-Grounded Framework for Evaluating and Aligning Emotional Intelligence in Spoken Language Models 

**Authors**: Junyu Wang, Siyuan Zhang, Peiyuan Jiang, Jian Zong, Jingyu Zhang, Tianrui Wang, Yuqin Lin, Zhenghui Chen, Shuqing Xie, Ziyang Ma, Meng Ge, Xiaobao Wang, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09189)  

**Abstract**: Despite significant advances in instruction-following and auditory comprehension, the evaluation of Emotional Intelligence (EI) in Spoken Language Models (SLMs) remains confined to rudimentary paralinguistic perception, lacking a systematic, theory-driven cognitive framework. We introduce EmoSBench, the first comprehensive EI evaluation benchmark for SLMs constructed upon the four-branch theoretical model, covering Perceiving, Understanding, Using, and Managing Emotion across ten sub-tasks. Preliminary assessments on EmoSBench reveal a substantial gap: even leading proprietary models like GPT-4o-Audio achieve only 52.6%, significantly trailing human baselines. To bridge this gap, we develop EmoS, a specialized evaluator model optimized via Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). To facilitate its effective training, we curate EmoDialogue, a bilingual dataset providing necessary fine-grained supervision through response pairs with rigorously defined EI gradations. Concurrently, we introduce a reward mechanism integrating a Steep Exponential Accuracy Reward (SEAR) and a Rationale Fidelity Reward (RFR) to enforce precise ordinal scoring and valid reasoning. Experiments demonstrate that EmoS reaches 83.8% accuracy, approaching human-level performance. Furthermore, evaluations on authentic, unconstrained spoken interactions validate its robust real-world generalization, establishing a foundational framework for advancing emotionally intelligent dialogue systems. 

---
# Failure-Aware Long-Form Translation: Design and Implementation of a Recoverable LLM Translation System 

**Authors**: Yanlin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09187)  

**Abstract**: A long-form translation request can succeed at the API layer and still produce an unusable result. The output may be empty, truncated, filtered, dominated by source or prompt material, or interrupted after producing text worth keeping. This report describes a recovery protocol developed for a deployed translation system with heterogeneous inputs and provider APIs. It delays the first visible release behind a 64-character window, validates the assembled output, and uses typed stream events to distinguish replacement from continuation. Interrupted work is retained only when a paragraph or sentence prefix can be re-derived from the source. Further attempts follow a stable model order and a shared deadline before entering a provenance-marked fallback path. A sanitized companion artifact implements the protocol and passes 38 public tests. Its fixed cases reproduce all 14 configured completion labels, contain four early-invalid prefixes before any of their 235 characters become visible, retain 31 boundary-safe characters across four interrupted streams, and satisfy the attempt, event, and provenance rules in two end-to-end scenarios. These results are executable checks of the published control flow. Translation quality and detector performance on naturally occurring outputs require a different evaluation. 

---
# UNSPECIFIC: General Constraint Synthesis for Breaking Copy-and-Paste Shortcut in LLM Instruction Following 

**Authors**: Jeet Sharma, Balpreet Kaur, Jeremiah Hong, Hamed Zamani, Haw-Shiuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09154)  

**Abstract**: Large language models (LLMs) are increasingly expected to follow long lists of constraints in complex instructions, and synthesizing instructions from a reference document (i.e., back-translation) is a widely used method to measure/enhance LLMs' ability to follow complex instructions. However, this method introduces a critical loophole: the constraint synthesis model copies text from the reference as a very specific constraint and the evaluated LLM trivially satisfies the constraint by copying its text in the response. To address these issues, we propose UNSPECIFIC, a novel framework that synthesizes constraints common to two similar reference articles to reduce copy-pasting, selectively hardens only trivially satisfied constraints to balance difficulty and naturalness, and evaluates satisfaction on both the generated article and its summary to penalize superficial instruction following. Consequently, we built the UNSPECIFIC benchmark on news, story, and blog domains to analyze the copy-pasting behavior of LLMs. Our results show that our synthesized constraints are not only more challenging (e.g., the satisfaction rate of GPT-5 Mini drops from 90% to 78%) and natural (LLM win-rate gap improves by 30%) from a human perspective but also mitigate the copy-pasting. We also find that a large portion of constraints are satisfied superficially (i.e., not satisfied in the core narrative of the article). The code and datasets are released at this https URL. 

---
# An Agentic Generative Large Language Model for Treatment Planning of Colorectal Cancer 

**Authors**: Mengxian Lyu, Cheng Peng, Tim Jang, Ang Li, Mengyuan Zhang, Ziyi Chen, Leighton Elliott, Tianshi Liu, Lidice Galindo, Chiranjeevi Sainatham, Oscar F. Borja-Montes, Kaleb E. Smith, Ying Zhang, Lichao Sun, Jiang Bian, Gloria Lipori, Duane A. Mitchell, Elizabeth A. Shenkman, Yi Guo, Thomas J. George, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09142)  

**Abstract**: Treatment planning in precision oncology requires synthesizing heterogeneous patient information with rapidly evolving clinical guidelines to ensure guideline-concordant care. While large language models (LLMs) show promise in many diagnostic tasks, their adoption for high-stakes treatment planning is hindered by complex reasoning, adherence to timely clinical guidelines, and safety concerns. In this study, we present GatorOnco, an agentic LLM for colorectal cancer (CRC) treatment planning. GatorOnco is developed using a total of 282 billion tokens of biomedical text, including healthcare system-scale clinical text comprising 166 billion tokens from UF Health. We implemented a domain-adaptation method that integrates pre-training, model merging, a two-stage post-training approach, and agent-based reinforcement learning. An agentic retrieval-augmented generation (RAG) approach dynamically integrates time-sensitive clinical guidelines into the reasoning process. In a blind, randomized clinical evaluation conducted by five UF Health oncologists, GatorOnco significantly outperformed open-source LLMs (P < 0.01) and achieved expert-level performance comparable to UF Health oncologists. Compared with expert oncologists, GatorOnco received significantly higher ratings for readability (4.46 vs. 4.19, P < 0.01) and completeness (3.91 vs. 3.52, P < 0.01), while showing statistically comparable performance in correctness (4.09 vs. 4.11, P = 0.921), currency (4.04 vs. 3.98, P = 0.478), and safety (4.22 vs. 4.22, P = 0.999). These findings demonstrate that integrating agentic reasoning with large-scale domain adaptation can help bridge the gap for generative AI in high-stakes cancer treatment planning. 

---
# Social Gym and SPaRTan: Benchmarking and Improving LLM Social Reasoning via Multi-Agent Game Tournaments 

**Authors**: Keyu He, Xuhui Zhou, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2608.09128)  

**Abstract**: LLM agents are increasingly deployed in multi-agent social settings where they must cooperate, negotiate, and adapt to other agents. Measuring and improving these social skills is hard because, unlike math or logic, social interaction offers no objective ground truth: evaluations fall back on LLM judges, which are costly, subjective, and noisy, and models get no reliable signal to learn from. To address both, we first introduce Social Gym, an environment of 21 multi-agent social games (e.g., Werewolves, Resistance, Spyfall) whose rule-decided outcomes make agent performance verifiable and objective, with an Elo tournament that produces a cross-game leaderboard. Benchmarking experiments show that while GPT-5-mini tops the leaderboard, no model excels at all games uniformly or in all game roles, pointing to limitations of social reasoning. Motivated by this, we additionally propose SPaRTan (Self-Play and Reflect-Transfer), a training-free self-improvement loop: a model plays a game, reflects on its trajectories and their outcomes to produce a transferable playbook, and applies that playbook in subsequent games. Our results show that SPaRTan playbooks help GPT-5-mini agents level their performance on weaker roles, but largely do not improve Qwen3-32B's performance. Together, Social Gym and SPaRTan offer a reproducible, verifiable foundation for measuring and improving LLM social reasoning without weight updates. 

---
# Subjective Multi-Bias Detection with Large Language Models 

**Authors**: Ruiyu Li, Zhiying Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09126)  

**Abstract**: In this project, we delved into the pervasive challenge of bias detection within the text content. More specifically, our focus lies on the identification of subjective bias, a type of bias that introduces improper attitudes or portrays a statement at odds with the actual truth. The subjective bias can jeopardize the authenticity and reliability of texts, leading to misconceptions and potential social tensions, especially when expressed through offensive language.
Following prior work [1], we tackled with three different types of subjective biases in text: (1) framing bias with the use of one-sided words or phrases containing a particular point of view; (2) epistemological bias which includes subtle linguistic features that can affect the believability of the texts; (3) demographic bias with word/phrase usage under presuppositions of a particular demographic factor (i.e., gender or religion).
In terms of the data we utilize, the input consists of texts that may harbor subjective biases. The output is a classification or annotation that reveals the presence or absence of such biases within the provided content. More specifically, we detected three different types of multi-span biases in corpus WIKIBIAS [2] with more than 4,000 sentence pairs from Wikipedia edits. The data is labelled by bias type for span pairs with the following categories: (1) framing bias, (2) epistemological bias, (3) demographic bias, and (4) no bias. The project codes are released at this https URL. 

---
# LexKairos: Benchmarking Legal Temporal Capabilities in LLMs 

**Authors**: Chenyang Li, Zejia Feng, Yuqin Huang, Yuxiao Ye, Huiyuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.09106)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance across a wide range of legal tasks. In legal practice, time is a critical concept that governs the validity of statutes, the progression of legal cases, and the enforcement of procedural deadlines. However, legal temporal capabilities remain underexplored in existing legal AI benchmarks. To address this gap, we propose LexKairos, a comprehensive benchmark for evaluating the temporal capabilities of LLMs in the Chinese legal context across three dimensions: statutory temporal knowledge, case temporal modeling, and statute-case temporal reasoning. LexKairos comprises nine sub-tasks drawn from real-world Chinese judicial cases and statutes. We conduct systematic evaluations of eight LLMs under multiple inference settings, including vanilla, Chain-of-Thought (CoT), and thinking modes. Our results show that Gemini-3-Flash achieves the strongest overall performance, yet even the best-performing model exhibits notable limitations on tasks demanding precise time-sensitive statutory metadata recall or complex reasoning in time limits, indicating that legal temporal knowledge and reasoning remain open challenges for current LLMs. Data and code are available at this https URL. 

---
# Evo-Bench: Can Language Models Improve Agent Harness? 

**Authors**: Lisheng Huang, Chen Yang, Hao Zhou, Huatong Song, Zongchao Chen, Ran Le, Yang Song, Wayne Xin Zhao, Tao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09096)  

**Abstract**: Large Language Models (LLMs) have driven rapid progress in autonomous agents, yet standard evaluations remain confined to static task solving. An emerging frontier is harness evolution---the agent's capacity to autonomously optimize its own operating harness. However, systematically benchmarking this capability remains challenging, as existing evaluations fail to isolate harness improvements from base model strength, prevent task-specific overfitting, or capture long-horizon iterative research. To address these challenges, we introduce Evo-Bench, the first benchmark designed to evaluate models' intrinsic harness-evolving capabilities across Search, Office, and General agent domains. To rigorously isolate this capability, Evo-Bench employs a novel harness-guided construction framework: it leverages auxiliary-task evolution to identify tasks genuinely sensitive to framework improvements, followed by sensitivity-aware stratified splitting to ensure robust cross-suite generalization. Extensive evaluations across nine frontier and open-weight models reveal that top models achieve massive absolute gains reaching 16.6 points, closely approaching state-of-the-art human-engineered baselines. Crucially, while autonomous evolution outpeforms artificial harness in General tasks and excels in Search tasks, it struggles in Office tasks that demand highly specific processing workflows. Furthermore, our analysis exposes critical temporal anomalies like early saturation, while demonstrating that the synthesized harnesses act as highly transferable reasoning structures, consistently boosting diverse policy models. 

---
# The Announcement Carries the Cue: Markup, Boundaries, and the Notation of Pre-Training Corpora 

**Authors**: E. M. Freeburg  

**Link**: [PDF](https://arxiv.org/pdf/2608.09093)  

**Abstract**: How a document's arrangement is written down, its notation, is a training variable that no dataset card records. The field has established that text-extraction choices change model behaviour, and has never once measured the notation of what those choices put into the corpus. We define clean-window survival, a deterministic count of how much of a stream still demands the boundary inference, and measure notation on three fronts. What corpora carry: a census of thirteen public corpora, where survival falls to 0.153 in a vision-converted PDF slice against 0.889 in C4; the scarce resource is not unmarked text but long unmarked text; a pre-registered supply test finds what remains institutional, not consumer. Our own pre-registered prediction failed: converters do not fabricate structure on prose, and that null forced the reliability mechanism that survives it. What readers use: across five base models spanning 0.6B to 8.2B and two pipelines, deleting a structural announcement makes the following prose measurably harder to predict, while swapping its notation moves nothing. That zero does not make notation unimportant; it relocates the variable: the operative cue is the announcement, not the sigil. What writers impose: a bounded null. Base models do not impose the marked register above the authored baseline, and handed prose with every announcement deleted they do not put one back, at a rate indistinguishable from zero against an authored reference of zero. We ship the format those measurements imply: the pure frame, paragraphs in authored order, every announcement deleted into a reversible sidecar, mixed against the marked copy over announcement presence rather than notation. Choose format operators by the capability they train, not by the fidelity they preserve, and record extractor identity and survival on data cards. 

---
# When Confidence Fails: Overconfidence in LLMs under Uncertainty and Missing Clinical Information 

**Authors**: Maryam Tahermazandarani, Adnan Mahmood, Fahmida Islam, Quan Z. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.09080)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance in medical question answering and clinical reasoning tasks. However, their reliability under uncertainty remains poorly understood which raises critical concerns for deployment in high-stakes clinical settings. In such environments, incorrect predictions are inherently risky, but confident incorrect predictions can be particularly harmful as they may mislead clinical decision-making. In this paper, we conduct a systematic behavioral analysis of LLMs under clinical information uncertainty. We propose an evaluation framework based on the MedMCQA dataset consisting of two complementary uncertainty settings. First, we introduce linguistic uncertainty cues through prompt modifications to simulate ambiguous clinical contexts. Second, we construct an answer removal setting, wherein the correct option is deliberately excluded mandating the model to recognize insufficient information and abstain. We analyze both model accuracy and confidence behavior using multiple calibration metrics including calibration gap, Expected Calibration Error (ECE), and Unsafe Confident Error Rate (UCER) across 500 medical questions. Our results reveal a consistent failure mode, i.e., although accuracy degrades under increasing uncertainty, model confidence remains misaligned with accuracy. This leads to a substantial increase in unsafe confident errors, indicating that model confidence remains largely insensitive to clinically meaningful information loss. Furthermore, we observe significant variation across models in their ability to abstain when the correct answer is unavailable, with some models persistently producing high confidence hallucinated answers. These findings expose critical limitations in the epistemic reliability of current LLMs and highlight the need for uncertainty aware evaluation methods prior to their deployment in clinical workflows. 

---
# Security and Privacy Taxonomy Generation from Mobile App Reviews 

**Authors**: Moghis Fereidouni, Vinaik Chhetri, Umar Farooq, A.B. Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2608.09049)  

**Abstract**: Mobile app reviews are a rich, continuously renewing source of how users experience privacy and security, yet existing taxonomies of these concerns are hand-crafted and cannot keep pace with the evolving nature of the data. Automating taxonomy construction is the natural response, but scalability is the core challenge: current LLM- and clustering-based methods are developed for scientific corpora of a few thousand documents and do not extend to app review collections numbering in the hundreds of thousands. We address this gap in two ways. First, we filter app reviews for privacy- and security-related content, yielding a comprehensive corpus of over 600K reviews. Second, we introduce TaxoScale, a pipeline that handles taxonomy construction at this scale by extending an expert-defined taxonomy via Recursive Hierarchical Clustering and LLM-based node naming. TaxoScale outperforms strong automatic-taxonomy baselines on path, level, coverage, and novelty metrics, and discovers novel branches absent from prior taxonomies. 

---
# Measuring the Tokenization Premium: A Cost Audit for Underserved Language Communities 

**Authors**: Avijit Roy, Proma Roy, Hrishitva Patel  

**Link**: [PDF](https://arxiv.org/pdf/2608.09046)  

**Abstract**: Large language models are increasingly deployed as general-purpose educational and technical assistance systems, but their underlying infrastructure does not treat languages equally. One underexamined source of disparity is tokenization: semantically equivalent content can require substantially different token counts across languages, affecting API cost, latency, and usable context length before a model is invoked. We introduce the Tokenization Equity Audit (TEA), a reproducible benchmark for measuring tokenization premiums in technical tutoring content. TEA evaluates three widely used tokenizers, GPT-4o's o200k base, Qwen2.5-7B, and Mistral-7B, on a 120-item Python debugging corpus translated from English into Bengali, Hindi, Arabic, Tamil, and Yoruba. Bengali and Hindi serve as the primary validated cases, while the remaining languages provide exploratory cross-script and cross-family comparisons. Across this corpus, Bengali requires (1.56\times) as many GPT-4o tokens as English, reducing a nominal 128k-token context window to an effective 82k-token English-equivalent capacity for the same semantic content. With the Qwen2.5 and Mistral tokenizers, Bengali requires up to (4.5\times) the English token count. Yoruba, despite using the Latin script, exhibits the highest GPT-4o tokenization premium at (2.37\times), indicating that tokenization inequity cannot be explained by script family alone. These results demonstrate that tokenization can create measurable economic and functional barriers, highlighting the need to treat tokenization as an equity-relevant infrastructure layer for underserved language communities, particularly where educational systems depend on low-cost or offline-capable AI tools. 

---
# Bridging the Gap Between Semantics and Reconstruction:Unifying Sign Language Translation and Production 

**Authors**: Xiao Liu, Shiwei Gan, Yafeng Yin, Jiaxin Yin, Bowen Guo, Yaqi Sun, Zhiwei Jiang, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.09045)  

**Abstract**: Recent advances in sign language (SL) research have shown a trend toward unifying multiple sign language understanding (SLU) subtasks, such as isolated sign language recognition (ISLR), continuous sign language recognition (CSLR), and sign language translation (SLT), within a single framework, leading to substantial progress. Meanwhile, sign language production (SLP), which generates sign sequences from text, has also attracted growing attention. This naturally raises an important question: can sign language understanding and production be unified within a single framework? Compared with unifying SLU subtasks, this problem is substantially more challenging. Existing SLU tasks largely share the same direction of mapping, namely from sign inputs to linguistic outputs, whereas SLT and SLP lie in opposite directions of sign-text mapping. A unified framework must therefore address two key challenges: (1) bridging the modality gap between continuous sign motions and discrete text tokens through a shared sign tokenizer that supports both linguistic abstraction and motion reconstruction; and (2) learning a single conditional autoregressive model that can take either sign or text as input and generate the corresponding target sequence in the opposite modality. To this end, we propose Uni-SLTP, a unified framework for SLT and SLP with two key components: (1) a shared sign tokenizer that converts sign sequences into discrete tokens and latent representations, capturing both semantic and reconstructive information; and (2) a unified autoregressive generation model that formulates both tasks as conditional sequence generation. Experiments on widely used public datasets show that Uni-SLTP achieves superior motion accuracy for SLP while maintaining competitive SLT performance. 

---
# Tree-of-Experience: Hierarchical Experience Management for Self-Evolving Agents 

**Authors**: Zihao Deng, Yining Zhu, Leiming Wang, Jingfei Lu, Junbo Wang, Chuncheng Ran, Yu Yang, Dixuan Yang, Jikun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2608.09044)  

**Abstract**: Continual self-evolution requires LLM agents to transform environmental interactions into reliable and reusable experience. Existing methods typically refine individual trajectories or abstract shared knowledge from related trajectories, but their experience representations are often disconnected from the underlying reasoning process. This limits feedback attribution, cross-task transfer, and update and retrieval efficiency, particularly in complex reasoning tasks with outcome-level feedback. To overcome this limitation, we propose \textbf{T}ree-\textbf{o}f-\textbf{E}xperience (ToE), a structured experience-management framework that aligns experience organization with the hierarchical reasoning process of LLM agents. Specifically, ToE organizes the experience into a shared tree of analytical perspectives and reasoning paths, whose reliability is calibrated through environmental outcomes to support systematic updating, transfer, and efficient retrieval. The experimental results on \textsc{Game of 24} and \textsc{FinEvolveBench} show that ToE substantially improves both problem-solving performance and efficiency. On \textsc{Game of 24}, ToE achieves a 31.4\% relative improvement in accuracy over the experience-free ToT baseline. On \textsc{FinEvolveBench}, ToE improves tsIC by an average of 41.24\% over the experience-free pipeline across 12 evaluation settings, whereas conventional experience-management methods often underperform experience-free baselines. 

---
# Don't Scroll Back: Missing-Evidence Memory for Streaming Dialogue Summarization 

**Authors**: Hyangsuk Min, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.09043)  

**Abstract**: Users of modern platforms repeatedly need summaries of recent dialogue, but the window rarely contains enough context to be interpreted on its own. We formalize this setting as streaming dialogue summarization, where a system must summarize a current window using selective memory from an unbounded history under a fixed budget. We show that the central challenge is not how much history is accessed, but whether memory recovers the evidence that the current window presupposes. We construct a benchmark and evaluation protocol that separately assesses whether memory contains gap-resolving evidence and whether the generated summary reflects it. We propose ReMEMBER, a missing-evidence memory framework that conditions retrieval on unresolved window dependencies and refines retrieved chunks into evidence-dense memory under a fixed budget. Experiments on dialogues with histories up to 160K tokens show that ReMEMBER improves memory recall and gap-resolution completeness over memory construction baselines under the same budget. 

---
# ELICITED: EHR-grounded Longitudinal Interactive Conversations for Information-seeking Triage Evaluation and Decision-making 

**Authors**: Haohao Zhu, Xiaolin Shi, Jiayu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.09024)  

**Abstract**: Emergency-department (ED) triage requires clinicians to rapidly identify patients who need immediate attention, determine who can safely wait, and prioritize limited clinical resources. At presentation, however, information may be limited to a chief complaint and initial vital signs. Clinically important details, including symptom onset and progression, associated symptoms, medical history, and medication use, are often obtained through focused conversation. Effective triage therefore requires clinicians to identify information gaps, ask appropriate follow-up questions, and update their assessment as new evidence becomes available. Most existing ED benchmarks evaluate acuity prediction from a fixed clinical snapshot. Although this formulation measures predictive performance after patient information has been assembled, it does not capture the interactive process through which triage-relevant evidence is elicited and interpreted. Existing medical dialogue datasets support the study of clinical communication, but dialogue statements are not always linked to temporally ordered events in the electronic health record (EHR). We introduce EHR2Dial-Triage, an agentic conversation-generation framework and benchmark grounded in MIMIC-IV-ED. The framework constructs triage conversations under explicit role-based and temporal information boundaries. Each accepted patient disclosure is linked to its supporting EHR event and the first dialogue turn at which it becomes available. EHR2Dial-Triage enables controlled evaluation of information elicitation, evidence use, five-level Emergency Severity Index prediction, and patient-facing communication across models and patient personas. It provides a structured setting for studying conversational triage as a dynamic process of clinical information acquisition, reasoning, and communication. 

---
# How Far Do Foundation Models Transfer to Infant Signals? A Cross-Dataset Transfer Audit with a Unified Need Ontology 

**Authors**: Wu Hangyu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08989)  

**Abstract**: Public infant cry corpora are small, label-incompatible, and almost always evaluated one corpus at a time. We ask what this practice hides and what fixes it. Across four cry corpora screened by a multi-level leakage audit (byte-level and embedding-level deduplication plus a within-corpus train-test near-duplicate audit), we probe four frozen encoders and a handcrafted baseline under a unified five-class need ontology and shared task formulations. The audit exposes what single-corpus evaluation conceals: within-domain macro-F1 swings by 0.57-0.80 for the same encoder, cross-corpus transfer is negative on average (negative-transfer ratio 0.19-0.35, significant in 18 of 30 directed cells, BH-FDR), and 349 content-identical clip groups carry conflicting metadata labels across corpus distributions. The same audit, however, reveals a consistent way forward. Transfer into the noisiest corpus is consistently positive in effect size at matched training size and after near-duplicate removal, offering a practical recipe for small, noisy corpora. Frozen probes saturate at modest label budgets, while stabilized fine-tuning wins with full labels; domain-adaptive pretraining significantly beats stabilized fine-tuning at 5-10-shot (the 1-shot advantage is not robust to optimization-seed variance) but shows no significant advantage at 50-shot or beyond. In the tested binary, shared-label settings, ontology-mapped joint training wins in all four encoder-by-target combinations, whereas naively merging unmapped labels costs up to 37 F1 points. We release the ontology, mapping code, and audit pipeline, turning incompatible cry corpora into a usable joint-training resource. 

---
# How Can Rhetoric Reward-Hack AI Reviewers? Dissecting Rhetorical Sensitivity in AI-Based Peer Review 

**Authors**: Ming Li, Chenguang Wang, Xirui Li, Xinyue Zeng, Dianqi Li, Peng Shi, Dawei Zhou, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2608.08975)  

**Abstract**: As large language models increasingly participate in scientific evaluation, we investigate a potential form of reward hacking: how rhetorical choices shape AI-review judgments when reported scientific content is preserved and how these effects vary across evaluation conditions. We construct a controlled corpus of 4,200 full-paper manuscripts derived from 120 anonymized ICLR 2026 submissions. Two LLM rewriters transform six rhetorical dimensions in opposing directions, and five LLM reviewers evaluate the resulting manuscripts under standard and strict protocols. We also test joint, recursive, and reviewer-guided rewriting. Our results show that rhetorical sensitivity is structured rather than uniform. Evidence framing and novelty stance produce the largest positive-negative contrasts in overall assessment, with scope framing forming a weaker second tier; the remaining dimensions have smaller or less stable effects. This hierarchy persists across human-assessed quality levels, but score movement depends strongly on the AI reviewer's original score: lower scores tend to rise, higher scores tend to fall, and directional contrasts are clearest in the middle ranges. More elaborate workflows do not reliably yield larger gains. Joint rewriting is strongly rewriter-dependent, reviewer guidance does not consistently outperform an unguided second pass, and repeated rewriting yields diminishing, configuration-dependent returns. Across conditions, the rewriter primarily determines the separation between opposing variants, whereas the reviewer determines the magnitude and sign of their score effects. Strict review lowers mean OA by 1.36 points without consistently changing rhetorical sensitivity. These findings identify when rhetorical presentation influences AI scientific review and motivate evaluation systems robust to content-preserving variation in scientific writing. 

---
# Same Question, Different Answer? Measuring and Mitigating Prompt Privilege for Equitable AI Access 

**Authors**: Lier Jin, Lan Hu, Binqi Shen, Hanyu Cai, Yuting Xin  

**Link**: [PDF](https://arxiv.org/pdf/2608.08942)  

**Abstract**: Large language models (LLMs) are increasingly integrated into healthcare, education, public services, and everyday decision making. They should provide comparable assistance regardless of a user's literacy, communication style, or prompt-engineering expertise. However, existing research on prompt robustness primarily focuses on adversarial attacks, prompt injection, and prompt optimization, while overlooking whether semantically equivalent requests receive different responses simply because they are phrased differently. We refer to this accessibility challenge as "Prompt Privilege": users with greater prompting expertise systematically obtain better model performance despite expressing the same underlying intent.
To address this problem, we present a unified framework for measuring and mitigating accessibility disparities in LLM interactions. We introduce Prompt Equity Score (PES), a quantitative metric for evaluating performance consistency across user populations, and Prompt Equity Transformer (PET), an LLM-based agent that automatically transforms user requests into semantically equivalent, accessibility-oriented prompts while preserving their intent. PET shifts prompt optimization from the user to the AI system, functioning as an intelligent accessibility layer between users and foundation models. Experiments on the MedQA benchmark demonstrate measurable prompt privilege, with statistically significant performance disparities between low-literacy and expert-prompting cohorts. Applying PET eliminates these disparities while preserving semantic fidelity, demonstrating that accessibility-oriented prompt normalization can improve equitable AI access. By introducing prompt privilege as a new dimension of AI accessibility and PET as a practical solution, this work advances system-centered accessibility and provides a foundation for more fair, trustworthy, and inclusive AI systems. 

---
# Investigating Multimodal Informativity under Different Partner Visibility Conditions in Video-Mediated Dialogue 

**Authors**: Esam Ghaleb, Hugh Mee Wong, Kristina Kobrock  

**Link**: [PDF](https://arxiv.org/pdf/2608.08915)  

**Abstract**: Situated language use is multimodal and embodied. For example, gestures can carry information that is absent or underspecified in the speech signal, yet dialogue models typically rely on transcripts alone. We study how much referential information gestures and their combination with speech carry in multimodal dialogue under different partner visibility conditions. % We build models that identify the intended referent in a video-mediated referential communication game based on either the speech transcript, the skeletal representation of gesture, or both modalities. Our results show that gesture alone is predictive of the intended referent and that multimodal fusion is most beneficial when the transcript-based model is uncertain. Training-only alignment of learned representations with the referent image further improves the fusion model performance. % In a comparison with human interaction data, we further see pragmatic effects of interlocutor visibility on gesture production and informativeness as well as an entrainment effect in speech and multimodal, but not gesture, performance across rounds of repeated interaction. We thus make contributions to the technical modelling of multimodal information in human dialogue and the analysis of human interaction data via trained model representations. 

---
# Tied Trit-Planes: Constraining PTQTP to a Uniform Nine-Level Quantizer, with a Persistent Folded Format for Disk-Streamed Mixture-of-Experts Serving 

**Authors**: Matteo Grella  

**Link**: [PDF](https://arxiv.org/pdf/2608.08910)  

**Abstract**: PTQTP decomposes LLM weight matrices into two ternary (trit) planes with two free per-group scales. Tying the scales to a fixed ratio of three collapses the decomposition into a single uniform nine-level quantizer, a known balanced-ternary identity. To our knowledge, at the time of writing, this work is the first to impose that identity as a constraint inside PTQTP's solver. The two trit planes then fold losslessly into one 4-bit code plane that we make the persistent serving representation: disk bytes, expert-cache bytes, and kernel input are the same 4.0625-bits/weight blocks, consumed in one integer dot pass. For this conjunction (ratio-3 nine-level code, CPU-SIMD kernels, SSD expert streaming, identical persistent bytes) we likewise found no precedent. We apply this to the routed experts of DeepSeek-V4-Flash-0731, a 284B-A13B mixture-of-experts model, quantizing in one shot from the released MXFP4 expert weights and streaming experts from SSD on a 64 GB laptop. Against a 4.5-bit Q4_K baseline, measured one process per fixture with an expert-lossless anchor arm as reference control, the tied model matches the official serving API on 5/5 fixtures at step 0 (Q4_K: 4/5) and 12/14 captured continuation steps (11/14), scores 86 vs. 84 on a 100-item MMLU subset, decodes 6.7% faster in decode phase, and ships 9% smaller files: no detected fidelity difference at these small evaluation sizes, and every fixture-level difference between the arms traces to a single measured near-tie cell. The tied fit nevertheless shows higher weight-reconstruction error and worse perplexity, a measured dissociation between proxy metrics and reference fidelity. A cumulative trunk-ternarization ladder and bitwise-pinned aarch64/x86-64 kernels complete the report. All code, formats, and evaluation artifacts are open source in the fucina inference stack. 

---
# Position Bias in Ordinal Classification: A Systematic Evaluation 

**Authors**: Yu Wang, Jeffrey Zhou, Menglin Liu, Ge Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08869)  

**Abstract**: Large language models are increasingly used for ordinal classification, yet semantically equivalent changes to prompt organization can alter their predictions. We conduct systematic experiments to characterize positional bias from label order, demonstration order, and demonstration placement. First, we apply the three probes to ten frontier LLMs on a common ordinal-classification task; every model is sensitive to all three positional sources, showing that the problem is pervasive. Second, we vary eight prompt-, task-, and model-level factors across five datasets; accuracy and stability are often misaligned, and only lower scale cardinality consistently improves both. Third, we compare pointwise, pairwise, and listwise inference, alternative aggregation and debiasing methods, and joint configurations; the tested corrections do not provide a reliable remedy, while a comparison-based listwise formulation offers the best balance but transfers unevenly across models and bias sources. These findings show that positional robustness depends on the full system configuration rather than the model alone. Ordinal-classification systems should therefore be selected jointly for predictive performance and stability. 

---
# Conversation as Measurement in Clinical Encounters: Observable Phase Structure, Partially Observable Patient State 

**Authors**: Lily Chen, Ted Mau, Michael Gensheimer, Brian Anthony Nuyen, Nancy Jiang, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2608.08868)  

**Abstract**: Many modern AI systems analyze conversational traces to infer aspects of human interaction and state, implicitly assuming that such information is recoverable from conversation. We study observability: whether a target is recoverable from conversational transcripts alone. Observability is difficult to assess because transcripts may provide only a partial view of many targets, and large-scale analysis requires model-based annotation, making true limits of the conversational signal hard to distinguish from annotator error. We therefore study clinical encounters, where patient-reported outcome measures (PROMs) provide an external anchor for patient state, and visits follow broadly structured patterns. We study observability of patient state and conversational phase structure using 439 real-world clinical encounter transcripts spanning 134 hours, including 245 ENT transcripts paired with 273 PROM surveys. We operationalize patient state using PROM scores for voice, cough, and swallowing; phase structure using conversational phase segmentation. To make these analyses credible at scale, we use a PHI-compliant GPT-5 deployment for transcript annotation and conduct 40 hours of manual validation, reducing the risk that apparent limits of observability simply reflect annotator error. Our core finding is an observability asymmetry: phase structure is observable and useful for characterizing clinical encounter organization, while patient state is only partially observable, even in a setting designed to elicit patient symptoms and experiences, cautioning against transcript-only inference of human state. 

---
# Explicit Boundary Markers for Subword Vocabularies 

**Authors**: Sander Land, Clara Meister  

**Link**: [PDF](https://arxiv.org/pdf/2608.08847)  

**Abstract**: Subword tokenizers represent many common words twice in space-using writing systems, once with a leading space and once without. The two entries have separate embeddings in models, so occurrences of one word are divided across rows that are trained independently, and the two forms need not even segment the string the same way: " together" may be a single entry while the same word without a preceding space is tokenized as "to|gether". Capitalization divides a word further, into as many as six forms. We introduce an alternative to standard whitespace conventions using an explicit word boundary marker, which prevents such duplication. Words are delimited by the boundary markers, and spaces between words are represented as pairs of such markers. Two shift codes do the same for title case and upper case, allowing one internal representation of a word to be re-used across different settings. Switching to this convention mitigates the duplicate-entry issue, but does not improve tokenization compression: for both vocabulary-learning algorithms, the best marker scheme stays within one percent of the baseline in characters per token, averaged across six languages. It does result in better language modeling performance. Every marker scheme tested downstream reaches lower bits per byte than the baseline, suggesting that duplication carries a cost that compression does not capture. 

---
# Deployable Per-Instance Multi-Layer Activation Steering for Large Language Models 

**Authors**: Muhammad Faishal Adly Nelwan, Alfan Farizki Wicaksono  

**Link**: [PDF](https://arxiv.org/pdf/2608.08829)  

**Abstract**: Activation steering edits the behaviour of a frozen language model by adding a learned vector to its residual stream, and current practice fixes the injection layers globally per task. We argue that the best layers are an instance-level decision, and we make per-instance, multi-layer selection both well understood and deployable. On two open-weight 8B models and six binary persona traits, a per-instance oracle over layer subsets shows that the best layers vary from one input to the next: on most trait-model pairs, no fixed global layer set recovers the per-instance benefit. A greedy rule that ranks layers by single-layer marginal effect recovers nearly all of the oracle's benefit, but both must score candidate layers against the gold answer, so neither can run at deployment; the rule instead becomes the target a prompt-only predictor is trained to reproduce. Our deployable recipe needs no label at inference: a per-instance layer ranker read off the prompt embedding, a classifier that infers the steering direction, and an adaptive gate that scores short steered passes against that inferred direction and steers no more layers than necessary. The recipe recovers most of the oracle's lift (the bulk on the stronger model, a clear majority on the harder one), never drives any trait-model pair below its unsteered alignment baseline on average, and largely avoids the fluency collapse that strong global selection incurs at higher layer counts. A mechanistic account, "direction over magnitude", explains the behavioural flip under a mis-directed global set, the output collapse from steering too many layers, and the ceiling of unsteerable inputs. 

---
# Tevatron-Elastic: A Unified Abstraction for Training Elastic Retrievers and Rerankers 

**Authors**: Yu Wang, Shengyao Zhuang, Xueguang Ma, Zongyu Wu, Jimmy Lin, Vivek Srikumar, Zhichao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08809)  

**Abstract**: A single model scale challenges the flexibility of a production retrieval system: some settings need it faster, others need a smaller index, and the right trade-off changes with the workload. In the context of information retrieval (IR), a transformer-based model can be made smaller in three ways---using fewer layers, passing fewer tokens through the upper layers, or producing a shorter embedding---and each way saves a different compute resource. These options have been studied one at a time, each as its own method with its own code and training setup, which makes them hard to combine or adapt to a new model. We present~\ours to bring all three under one simple abstraction: a single object names any size the model can run at, and a short schedule lists the sizes to train. Training then produces one checkpoint that serves all of those sizes, and at deployment the user picks any of them. The same abstraction covers both retrievers and rerankers and both encoder and decoder models, as it works through interfaces that Hugging Face transformers already expose; a new backbone is a configuration change, not new modeling code. Prior methods---Matryoshka embeddings, early exit, 2D~Matryoshka (e.g., Starbucks), and layerwise token compression---become special cases of our unified abstraction. The same interface also enables Matryoshka~LTC (MLTC), which jointly trains several token-compression ratios in one retriever checkpoint. To validate our framework, we train 20 checkpoints across three backbones and two tasks: the quality curves are smooth, one checkpoint costs little over a model trained for a single size, and a controlled study confirms the wallclock speedups. We release the framework and all checkpoints as a resource for building elastic retrieval systems. 

---
# IDRAAK: From Multi-Agent NLP to Few-Shot Prompting for Semantic Drift Detection in Technical Requirements 

**Authors**: Shiva Ahir  

**Link**: [PDF](https://arxiv.org/pdf/2608.08801)  

**Abstract**: Translating technical requirements across languages can introduce semantic drift, altering numerical constraints, polarities, modalities, or other specification-critical meaning. IDRAAK is presented as an interpretable framework for detecting such drift using a language-independent Semantic Requirement Representation (SRR), with six detection workflows evaluated, ranging from deterministic comparison to multi-agent verification and few-shot prompting. On 890 synthetic perturbations across 300 requirements from 10 engineering domains, a single LLM call with six few-shot examples achieves MCC=0.888 and F1=0.983, outperforming the evaluated structured and multi-stage alternatives. Further evaluation on PAWS-X (805 pairs, 5 languages) and XNLI (700 pairs, 7 languages) exposes complementary strengths and limitations of structured and LLM-based approaches. Deterministic SRR comparison performs strongly on technical requirements (F1=0.898) but poorly on general-domain text (F1=0.012), while structured evidence improves performance on adversarial paraphrases. Post-hoc Platt scaling further improves confidence calibration. The results demonstrate that increased agentic complexity does not necessarily improve semantic-drift detection and that simple few-shot prompting can provide a strong and efficient alternative. 

---
# Instability of LLM Pre-Pretraining: It Doesn't Always Help. An Investigation on Multiple Languages 

**Authors**: Sofiia Riazhskykh, Nam Luu, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2608.08800)  

**Abstract**: Pretraining LLMs on artificial languages ("pre-pretraining") is a technique that could reportedly increase token efficiency by 33%, i.e., save up to 33% of training tokens needed to reach a certain performance. We validate this prior result for English on a larger set of natural languages across four language families, using two different tokenizers and varying model sizes. We also relate the observed gains (or losses) in token efficiency to quantified linguistic properties of the languages, such as sentence length, morphological richness, and features of dependency syntactic trees (tree depth, number of children, number of crossing dependencies). Our empirical results indicate that the reported gains depend heavily on the experiment setup and the choice of random seed, although we can confirm the trend of stable gains with 128-Dyck pretraining of small models with the Llama tokenizer for most of the examined languages. On a general note, we argue that multiple training runs should be carried out at least for a subset of experiments to avoid the community adopting unstable approaches. 

---
# Evidence-Calibrated Runtime Reconstruction for Agent Skills Across Heterogeneous Coding Agents 

**Authors**: Xueping Gao  

**Link**: [PDF](https://arxiv.org/pdf/2608.08793)  

**Abstract**: Agent Skills package reusable instructions and assets for tool-using language-model agents. Progressive loading creates failure boundaries poorly represented by session-, model-, or tool-centric traces: a Skill can be discovered but not activated, activated without instructions, or appear successful without an independently verified outcome. We present Skill Runtime Intelligence, a passive runtime-intelligence system that reconstructs supported Skill-lifecycle stages across heterogeneous harnesses while preserving unsupported stages as unknown. Its Run Panorama separates immutable events, deterministic relations, inferred diagnoses, and controlled outcomes with four evidence grades; optional trace import and OTLP/HTTP export support existing observability deployments.
Across six frozen repository profiles, three coding agents, and seven clean or fault-injected conditions, all 126 executions preserve source worktrees and each correlates to exactly one source session. Yet adapters expose three distinct semantics: no Skill runs; complete runs but no failure-like events; or failure-like events in every operational-failure and clean session. In a seven-template diagnostic study, semantic aliases and Panorama localize the same six non-clean boundaries but differ in exact/status behavior; both Raw views emit a failure status on all 18 clean cases, while Panorama emits none. A known-rule graph conforms to 126/126 frozen contracts, whereas a second model completes only 228/378 calls. These observations motivate executable adapter qualification and show that event presence is not boundary fidelity, composite exact scores mask distinct errors, and model explanations must not overwrite deterministic facts. 

---
# Unsure but Certain: Uncovering the Representation-Confidence Gap in Diffusion Language Models 

**Authors**: Saurabh Yadav, Badri Narayana Patro, Vijay Srinivas Agneeswaran  

**Link**: [PDF](https://arxiv.org/pdf/2608.08791)  

**Abstract**: Diffusion language models use broad context to create text, suggesting they might handle input noise better than standard models. Testing reveals this is only partially true. Internally, diffusion models detect text errors highly accurately. Externally, their reported certainty ignores this signal. As accuracy drops due to noise, confidence stays near its maximum and the ability to correctly rank answers degrades toward random chance. We call this mismatch the representation confidence gap. The visible concentration of high certainty scores is a misleading surface symptom. Standard math adjustments remove this concentration but fail to fix the underlying loss of ranking order. This ranking deficit favors standard models under noisy conditions and resists common remedies. Matching training recovers accuracy but not ranking, while score recalibration and input level error signals cannot reorder the final answers. However, the information needed to properly evaluate an answer survives in the hidden states. A lightweight extraction tool uses this signal to improve ranking. This approach is highly efficient because it leaves the base model completely frozen and requires zero additional text generation steps. We present this tool to prove the signal exists, while clearly noting its limits. Ultimately, certainty reliability is a more pressing limit than overall accuracy under noisy conditions. 

---
# OmnilingualGAIA2: Evaluating the Multilingual Gap in Frontier AI Agents 

**Authors**: Andrea Caciolai, Pere-Lluís Huguet Cabot, Chierh Cheng, Albert Ventayol-Boada, Gabriel Mejia Gonzalez, Christophe Ropers, Lucas Bandarkar, Sebastian Ruder, Darlene Sakakihara, Elliot Yun, Pierre Andrews, Grégoire Mialon, Romain Froger, Marta R. Costa-jussà  

**Link**: [PDF](https://arxiv.org/pdf/2608.08775)  

**Abstract**: Agentic benchmarks aim to measure how well AI agents plan, search, execute, and recover within realistic multi-tool environments, but they are almost exclusively in English. As AI agents are globally deployed to a linguistically diverse user base, whether agentic competence measured in English transfers to other languages remains an open question. We introduce OmnilingualGAIA2, a machine-translated expansion (with partial human- expert validation) of the GAIA2 agentic benchmark, covering ten target languages spanning five writing systems, paired with a localised and human-calibrated multilingual verifier. Evaluating seven frontier and open-weight agents, we find a universal cross-lingual gap of 8.8-18.4 pass@3 points that is agent-asymmetric in magnitude, concentrates on tool-orchestration rather than quantitative reasoning, and does not close with model scale. A stratified error attribution decomposes the gap as predominantly model-driven (55%), with a bounded translation-contamination floor of only 6.4% of scenario-language pairs. Human-expert linguistic analysis further identifies morphological cue loss and amplified ambiguity as the primary failure mechanisms in non-Latin-script languages. Our results argue that multilingual agentic evaluation must become a standard part of the reporting protocol for globally deployed agents. 

---
# Multilingual Emotion Neurons in Large Audio-Language Models 

**Authors**: Xiutian Zhao, Philipp Koehn, Björn Schuller, Berrak Sisman  

**Link**: [PDF](https://arxiv.org/pdf/2608.08772)  

**Abstract**: Emotion is central to human communication, and its expression varies across languages. Large audio-language models (LALMs) achieve strong performance on multilingual speech tasks, yet it remains unclear whether they encode emotion through language-specific correlations or language-agnostic representations. We present the first neuron-level interpretability study of this question. We define Multilingual Emotion Neurons (MLENs) as functional units exhibiting stable emotional selectivity and aligned causal effects across languages, and introduce Consistency-Regularized Fusion (CR-Fusion) to identify them. Across four modern LALMs and 12 typologically diverse languages, emotion-sensitive neurons identified independently per language show minimal overlap, and additional monolingual identification data saturates quickly without isolating more transferable units, motivating identification from pooled cross-lingual evidence. Causal interventions demonstrate that MLENs identified by CR-Fusion provide more precise and transferable affective control than monolingual neuron sets in both zero-shot and low-resource settings. Leave-one-out ablations further reveal asymmetric transfer: individual identification languages, including low-resource ones, contribute non-redundant evidence, while several low-resource languages benefit most from the resulting cross-lingual transfer. Together, our findings provide the first causal, neuron-level account of how LALMs encode emotion across languages, and establish multilingual neuron identification as an effective mechanism for understanding cross-lingual affective behavior. 

---
# Can We Optimize the Performance-Carbon Emission Break-Even Point?: The Quest for Greener LLMs 

**Authors**: Sourav Das, Tanmay Joshi, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2608.08744)  

**Abstract**: The carbon footprint of any deployed Large Language Model (LLM) accumulates during inference, where repeated use of the model substantially exceeds the one-time cost of fine-tuning. Yet most efficiency interventions target either pre-training scale or post-hoc compression. We ask whether folding a calibrated, differentiable energy surrogate into the fine-tuning objective can produce inference behavior that gains task accuracy at zero or near-zero carbon cost, a break-even configuration. We propose a joint loss mechanism with a per-model carbon-emission parameter, a linear surrogate over parameter norm, FLOP proxy, and a memory proxy, fit from on-hardware energy profiling. We fine-tune three architecturally distinct families: Gemma-2 2B, Llama-3.1 8B, and Qwen-2.5 14B, and evaluate inference F1 and CO$_2$ emissions on three MMLU subjects: abstract algebra, philosophy, and formal logic. We discover from several outcomes that the carbon term behaves as either harmful interference or beneficial regularization depending on the task structure. We position calibrated carbon-aware fine-tuning as a lightweight, drop-in regularizer with a non-empty but model and task-dependent break-even region. This is an ongoing work, and we will release our codebase soon. 

---
# LibraSpec: Dynamic Diffusion-Based Speculative Decoding via Marginal-Gain-Driven Optimization 

**Authors**: Zexun Lin, Yuan Feng, Junlin Lv, Kevin S. Zhou, Xike Xie  

**Link**: [PDF](https://arxiv.org/pdf/2608.08721)  

**Abstract**: Speculative decoding accelerates large language model inference by drafting multiple tokens for parallel verification, with efficiency critically determined by the speculative length selected at each decoding round. Existing dynamic speculation methods select the speculation length by estimating how many tokens will be accepted, which is reasonable for autoregressive drafters that generates tokens sequentially. The recent wave of diffusion-based drafters, however, generates candidate blocks in parallel at substantially lower drafting cost, shifting the key question from how many tokens to generate to how many generated tokens are worth verifying. We therefore reformulate dynamic speculative-length selection as expected-speedup optimization and derive a marginal criterion that extends the speculative sequence only when its acceptance gain outweighs the additional verification cost. Building on this criterion, we develop \textit{LibraSpec}, a training-free and plug-and-play algorithm that iteratively determines the speculative length using drafter confidence scores. Theoretically, we prove that LibraSpec monotonically converges toward the optimal speculative length. Experiments across six target models, three diffusion-based speculative decoding methods, and math, coding, and chat benchmarks show consistent improvements under both greedy and sampling settings, achieving a further $0.5\sim1.5\times$ improvement over baselines and up to $8.49\times$ speedup over autoregressive decoding. 

---
# The Evolution of Mixture-of-Experts Architectures in Large Language Models: Routing, Topology, Load Balancing, and Expert Parallelism 

**Authors**: Jiguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.08650)  

**Abstract**: Mixture-of-Experts models increase parameter capacity while keeping the computation activated by each token bounded, but their architectural evolution cannot be explained by a chronological list of model releases alone. This technical survey synthesizes primary papers, official technical reports, and prior surveys to organize modern Mixture-of-Experts systems along five coupled dimensions: expert granularity, expert topology, routing freedom, the scope of load balancing, and execution structure. We describe eight architectural milestones as a dependency graph with six mainline developments and two orthogonal branches, rather than as eight successive generations. We then analyze individual systems through four control planes: Expert Topology, Routing, Balance, and Expert Parallelism. These planes specify which experts exist, which experts process each token, how aggregate load is controlled, and how selected computation is mapped onto physical devices. The framework connects algorithmic choices such as Top-k routing, shared experts, fine-grained experts, and dynamic expert composition with systems concerns including token dispatch, device placement, all-to-all communication, and communication-computation overlap. We conclude with equal-budget pretraining experiments, quality and systems metrics, and open research questions. The main trend is a shift from merely activating more sparse parameters toward decoupling semantic routing, computational budgets, and physical execution. 

---
# Enhancing Scientific Named Entity Recognition via Large Language Models: A Type-driven Multi-task Learning Approach 

**Authors**: Tong Bao, Yi Zhao, Heng Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08636)  

**Abstract**: Scientific named entity recognition (SciNER) plays a crucial role in information extraction and knowledge discovery from scientific texts. Recently, large language models (LLMs) have demonstrated the capacity to achieve competitive SciNER performance with minimal human effort. Existing research highlights the importance of incorporating candidate entity type information for accurate entity recognition and classification by LLMs. However, when too many candidate entity types are provided in the prompt, LLMs struggle to accurately recognize and label entities in scientific texts, where entity types are more complex than in general domains. To address this challenge, we propose TdSciNER, a type-driven approach that effectively leverages entity type information to enhance SciNER performance. In TdSciNER, we first design an entity type filter model to identify the most likely entity types present in a given sentence. Subsequently, we introduce an auxiliary multi-class entity typing task within a multi-task learning framework alongside SciNER to obtain richer contextual representations. Then, we develop a novel demonstration selection strategy based on sentence similarity and entity type diversity to activate the in-context learning capabilities of LLMs, thereby improving entity recognition accuracy across diverse scientific domains. Experiments on three datasets demonstrate that our method achieves performance comparable to fully supervised models. Further analysis validates that each entity type-driven component in TdSciNER contributes to the improvement of SciNER performance. This work provides valuable insights for future advancements in SciNER and broader information extraction tasks in scientific text mining. 

---
# North Africa's Missing Framework: NLP-Driven Mental Healthcare in Algeria and Implications for Low-resource Settings 

**Authors**: Meriem Laifa, Abdallah Bengueddoudj  

**Link**: [PDF](https://arxiv.org/pdf/2608.08607)  

**Abstract**: Mental health disorders are a leading cause of disability worldwide, yet Natural Language Processing (NLP) research for mental healthcare has remained concentrated in high-income, English-language settings. North Africa, and Algeria in particular, is largely absent from this literature despite its unique linguistic, historical, and healthcare context. We present the first conceptual framework examining the potential role of NLP within Algeria's mental healthcare system. Drawing on narrative synthesis of global NLP mental health research, Algerian healthcare literature, and low-resource NLP methodologies, we identify four structural barriers to mental healthcare: the language-of-care gap, geographic inequities in access, stigma-related barriers to help-seeking, and the absence of research and digital infrastructure. We then map existing NLP capabilities to each barrier, outlining their potential applications, implementation constraints, and the technical, institutional, and governance requirements necessary for deployment. Based on this analysis, we propose a research and policy roadmap that prioritizes data resources, multilingual language technologies, evaluation frameworks, and regulatory capacity. Although grounded in the Algerian context, the framework addresses challenges common to many multilingual, low-resource, and post-colonial settings. This work provides a foundation for future research on culturally and linguistically appropriate NLP for mental healthcare and offers a practical roadmap for developing responsible AI-enabled mental health systems in underrepresented regions. 

---
# Mitigating Gender Bias in English to Romanian Machine Translation 

**Authors**: Ioana Grigore, Sergiu Nisioi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08606)  

**Abstract**: Machine translation (MT) systems often fail to correctly translate gender, especially when converting from a gender-neutral language like English to a gendered target language such as Romanian. This bias results in translations that default to masculine forms or reinforce gender stereotypes. We propose a hybrid pipeline to mitigate this issue by combining large language model (LLM)-based gender classification with neural machine translation (NMT). Our system uses a fine-tuned LLM to detect the intended gender of target words in English sentences and insert inline gender hint tags. These tagged sentences are then passed to a Transformer model fine-tuned to generate morphologically correct Romanian translations. To support this, we introduce three novel datasets for gender disambiguation and translation. Our approach improves gender accuracy on the WinoMT and WinoGender benchmarks by over 40 percentage points compared to a baseline MT system. This is the first method to explicitly address and evaluate gender bias in English-Romanian MT using both LLM inference and tag-aware translation. 

---
# OpenVisTool: An Open Recipe for Synthesizing Instructive Visual Tool-Use Trajectories 

**Authors**: Changhao Xiang, Shilin Zhang, Zheng Ma, Kanzhi Cheng, Ruize Ma, Yi Feng, Jianbing Zhang, Zhi Wang, Zhen Wu, Xinyu Dai, Lewei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08557)  

**Abstract**: Visual tool use has emerged as a fundamental capability for multimodal agents to actively acquire evidence beyond a fixed image encoding. The prevailing recipe learns this capability from teacher-generated trajectories filtered for answer correctness, implicitly assuming that every successful demonstration provides effective supervision. We argue this assumption is flawed: a strong teacher often reaches the correct answer without needing its tool calls, and imitating such trajectories teaches a student that tool calls accompany correct answers, not that tool observations ground them. We present OpenVisTool, an open framework for constructing instructive visual tool-use trajectories that provide effective supervision for tool learning. The key insight is that a trajectory should be retained only if its answer is correct (outcome validity) and its tool observations causally contribute to that answer (causal utility). The framework operates in three stages: difficulty screening to select queries that are not reliably answerable without tools, domain-specific trajectory synthesis to elicit coherent tool-use trajectories, and supervision verification to jointly test both conditions. Rather than encouraging models to imitate tool calls, the resulting supervision teaches when and how visual evidence should be acquired. Using this framework, we construct OpenVisTool-42K, a dataset spanning five visual reasoning domains, together with OpenVisTool-Bench, a benchmark covering the same domains. Across four backbones (4B-27B), fine-tuning on OpenVisTool-42K consistently improves visual tool-use performance and yields gains on two out-of-distribution benchmarks; the larger models approach leading closed-source systems. The evidence suggests that effective visual tool use is learned from causally grounded supervision rather than tool-calling patterns. 

---
# From Speech to Interaction: Analyzing Multimodal Systems in Cocktail-Party Scenarios 

**Authors**: Thai-Binh Nguyen, Zhaolin Li, Jan Niehues, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2608.08510)  

**Abstract**: Humans have the remarkable ability to engage in spontaneous informal conversations and selectively attend to individual speakers while filtering out competing speech from nearby conversations. This "cocktail party" scenario still presents severe challenges to speech recognition systems. The CHiME-9 MCoRec task provides a testbed where systems must recognize groups of speakers and transcribe each of their conversations from audio-visual input. In this work, we analyze a diverse set of systems, representing different design directions for addressing the cocktail-party scenario, where the best system achieves up to 57% relative error reduction. We identify three main strategies: (1) explicit or implicit audio-visual target speech separation, (2) improved audio-visual speech recognition for each target speaker, and (3) the use of large language models to group speakers into conversations and enhance conversational consistency. Our analysis shows that these directions address complementary failure modes of the cocktail-party problem, and that high speech overlap alone does not explain performance differences, challenging the common assumption that overlap is the primary source of difficulty in cocktail-party recognition. 

---
# VectraYX-Vision-1B: A Sub-2B Spanish/LATAM Cybersecurity Vision-Language Model with Structured Visual Reasoning and Native Tool Use 

**Authors**: Juan S. Santillana  

**Link**: [PDF](https://arxiv.org/pdf/2608.08477)  

**Abstract**: We present VectraYX-Vision-1B, a sub-2B vision-language model (VLM) for Spanish/LATAM cybersecurity imagery, coupling a frozen SigLIP-so400m encoder to a 1.04B Spanish/LATAM security decoder via an MLP. To our knowledge, it is the first sub-2B VLM specialized for cyber UI (IDA, Ghidra, Wireshark, Nmap, Metasploit, Volatility) that answers in Spanish, emits structured reasoning via native <|think|> tokens, invokes tools via Model Context Protocol (<|tool_call|>), and exports to this http URL's LLaVA mmproj format for air-gapped deployment. We report a negative preliminary visual-grounding result: despite fully functional pipelines, the current vision SFT (400-1900 steps, ~16M tokens) yields near-zero B6 scores (0.08 tool-identification), ignoring image content. We specify remediation (longer SFT, >=60% replay, lower LR) and expose a checkpoint-loader bug (unstripped llm. prefix) masquerading as training collapse. Crucially, we introduce a 3-variant ablation matrix (V0: NoPE-every-4, V1: all-RoPE, V2: NoPE+learned 2D) to study if periodic no-positional-encoding (NoPE) layers help or hurt attention over the 729-token visual block. Code, configs, and weights are released to establish priority on this architectural question. We provide B1-B5 for the text backbone, text controls, preliminary B6/B7 scores, wall times, GGUF efficiency on CPU, and a corpus of 14,596 QA pairs across 10 domains. We open-source all models and trajectories: jsantillana/vectrayx-1b, jsantillana/vectrayx-vision-1b, and jsantillana/vectrayx-vision-1b-checks. 

---
# Beyond Tables: Doc2DB-Bench for Relationally Faithful Document-to-Database Construction 

**Authors**: Zhuowen Liang, Zhengxuan Zhang, Jiayang Wang, Jiazhuo Chen, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08459)  

**Abstract**: Practical AI systems increasingly need to turn long, heterogeneous documents into queryable relational databases, not isolated spreadsheets. In domains such as finance, healthcare, education, transportation, and enterprise operations, downstream workflows rely on normalized schemas, entity identities, keys, cross-table relationships, and integrity constraints for analytics, compliance, auditing, and SQL-backed decision making. Existing Document-to-Table benchmarks are insufficient for this setting: flattening evidence into single tables can duplicate entities, obscure many-to-many relationships, create sparse records, and avoid testing whether extracted facts form a valid database instance. This creates an urgent need to evaluate document understanding as database construction rather than field extraction. We introduce Doc2DB-Bench, a benchmark for Document-to-Database construction, containing 203 long-document instances across 42 schemas and seven domain groups, with 117 entity tables, 132 relationship tables, 7,341 rows, and 41,935 cells. Built through a controllable DB-to-Doc synthesis pipeline and organized by a taxonomy of intra-table extraction and inter-table reasoning, the generated documents undergo authenticity verification, proving indistinguishable from real-world references. Doc2DB-Bench thus provides a testbed for reliable, auditable, and relationally faithful LLM-based data systems. The benchmark is publicly available at this https URL. 

---
# Calling the Bluff: Detecting Ever-Shifting Harmful Chat Dialogue via Ordered Reasoning Chain Regularization 

**Authors**: Haojie Yu, Ziyou Jiang, Junjie Wang, Mingyang Li, Yuekai Huang, Jie Huang, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08451)  

**Abstract**: Harmful chat dialogues are ever-shifting through type-shifting and lexical evasion, yet we find they share invariant principles, i.e., an Ordered Reasoning Chain (ORC) of recurring topics, harm language indicators, severity hierarchies, and type characteristics, which can help us capture the key information in the frequently changing lexical expressions. We propose BRACE, which encodes the ORC as four differentiable stages (Topic -> Indicator -> Severity -> Type) with intermediate supervision, serving as a structured regularizer blended with direct heads, and supported by prototype-based feature augmentation and feature path disentanglement. The evaluation results show that, across 4 domains and 5 harm categories, BRACE achieves harm-type macro F1 of 0.934 (RoBERTa-wwm-ext, 3-seed mean), with decoder backbones (Qwen3-1.7B LoRA) reaching 0.949. Ablation studies show that all components contribute to BRACE, and the structural decomposition of ORC enables BRACE to distinguish harmful types with semantic ambiguity. Disclaimer: This paper may contain content that is disturbing to some readers. 

---
# Hidden Language Consistency Phenomena in Reasoning LLMs 

**Authors**: Muhammad Ali Shafique, Kelly Marchisio  

**Link**: [PDF](https://arxiv.org/pdf/2608.08447)  

**Abstract**: Multilingual reasoning models are commonly evaluated by whether they arrive at the correct answer, but not by whether they preserve the intended language while reasoning and responding. This omission conceals important multilingual behaviors that emerge as tasks become harder. In this paper, we study task difficulty, task accuracy, thinking-language consistency (TC), and answer-language consistency (AC) across reasoning models using PolyMath benchmark in eight languages and four difficulty levels. We uncover four findings: (1) language consistency exhibits four difficulty-dependent behaviors: output-language consistency remains aligned with input, remains misaligned, degrades gradually, or collapses abruptly. (2) We identify the language consistency breakdown effect, where increasing difficulty can cause a sudden drop in output-language consistency, especially in less strongly represented and non-Latin-script languages. (3) Due to this breakdown effect, accuracy can be preserved or even improved at a harder difficulty level as the model shifts to its internal dominant language. (4) Quantization can improve or degrade output-language consistency independently of its effect on accuracy, with GPTQ and AWQ often outperforming AutoRound under tolerance-based voting with {\epsilon} = 1.0. These results show that multilingual capability cannot be characterized by accuracy alone; reliable evaluation should jointly consider task accuracy, language consistency, and task difficulty for multilingual benchmarks. 

---
# Safety Cost of Steering Vectors Is Separable and Reducible 

**Authors**: Yuxiao Li, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2608.08383)  

**Abstract**: Steering vectors are a lightweight tool for controlling LLM behavior. However, emerging evidence shows that steering vectors can unintentionally compromise a model's safety mechanisms and increase compliance with harmful requests, while no effective mitigation yet exists. In this work, we show that this safety degradation arises from a separable component in the vector that disrupts the model's safety mechanisms but contributes little to the steering objective. We identify and remove this safety-degrading component, formulating the task as a constrained optimization problem solved through primal-dual updates, subject to preserving the intended steering effect and bounding false refusal. The resulting solution is both interpretable and surgical: the optimization recovers a single direction whose ablation from the steering vector restores model safety with minimal utility cost. Across models, steering behaviors, and attack suites, including unseen attacks types, our method substantially reduces steering-induced safety degradation while preserving the original steering effect with minimal impact on false refusal. Our method offers a post-hoc correction to steering vectors that mitigates their safety cost, and more broadly, it provides a general recipe for applying activation-level model interventions without paying a safety tax. 

---
# Do Evaluation Metrics Detect Errors in Classical Chinese to English Translations? 

**Authors**: Osvaldo Quinjica, Eric Bennett, Xinchen Yang, Andrew Schonebaum, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2608.08283)  

**Abstract**: Although large language models can translate some historical languages surprisingly well, their usefulness in digital humanities workflows is limited by the lack of reliable evaluation. We investigate whether existing automatic evaluation metrics developed for modern languages are reliable in this setting, using translation from Classical Chinese to English as a test case. We introduce a diagnostic framework based on minimal pairs capturing error types salient in scholarly use, probing both reference-based and reference-free metrics for error sensitivity and tolerance to valid variation. We find that all metrics exhibit blind spots, however MetricX24 performs best overall. Our findings highlight the need for more robust and interpretable metrics for historically and culturally distinct translation settings. 

---
# AraSSM: A bidirectional state-space encoder for Arabic masked language modeling 

**Authors**: Ahmed Amine Aliane, Hassina Aliane, Nasredine Semmar  

**Link**: [PDF](https://arxiv.org/pdf/2608.08256)  

**Abstract**: Pretrained Transformer encoders such as AraBERT, MARBERT, and CAMeLBERT have become the standard backbone for Arabic natural language understanding, but their self-attention mechanism scales quadratically with sequence length, which limits efficiency on long documents. Mamba, a selective state-space model (SSM), offers linear-time sequence modeling as a competitive alternative to attention, yet no dedicated bidirectional Mamba encoder pretrained specifically for Arabic currently exists. We introduce AraSSM, a bidirectional Mamba encoder pretrained via masked language modeling on a corpus combining Arabic Wikipedia and CulturaX text, trained end-to-end on four consumer-grade NVIDIA RTX 2080Ti GPUs (11GB) over approximately ten days. We evaluate AraSSM by fine-tuning on four established Arabic NLU benchmarks covering sentiment classification (HARD), named entity recognition (ANERcorp), extractive question answering (ARCD), and natural language inference (XNLI-ar), following the per-task evaluation protocol introduced by AraBERT, and report results as mean +/- standard deviation across three fine-tuning seeds. AraSSM matches or exceeds published base-sized Transformer baselines on sentiment classification (96.37 +/- 0.03% accuracy on HARD), is competitive on extractive QA (32.19 +/- 1.07 EM, 63.79 +/- 0.25 F1 on ARCD) and named entity recognition (81.54 +/- 0.30 entity-level F1 on ANERcorp), and trails the base-sized Transformer range on natural language inference (72.83 +/- 0.07% accuracy on XNLI-ar), despite being trained entirely from scratch on consumer hardware rather than large-scale accelerator clusters. 

---
# Focus particles and scalar inferences across humans and language models 

**Authors**: Catherine M. Brousse, Nelu D. Radpour  

**Link**: [PDF](https://arxiv.org/pdf/2608.08227)  

**Abstract**: Focus particles such as "even" and "only" are central to formal semantic theories that posit structured representations over sets of alternatives. "Even" highlights unexpected or extreme alternatives, while "only" enforces exclusivity. If such scalar representations are robust and generalizable, they should give rise to consistent judgments across contexts and systems. In this work, we test whether humans and large language models (LLMs) construct stable scalar representations from sentences containing these particles. Using a dataset of approximately 100 items, participants and models were asked to make scalar judgments. Preliminary results suggest that similar outputs across humans and LLMs may arise from different underlying mechanisms. 

---
# A Grounded and Decomposed Framework for Relation-Level Hallucination Evaluation in Abstractive Summarization 

**Authors**: Praveen Kumar Katwe, Rakesh Chandra Balabantaray, Kali Prasad Vittala, Naman Kabadi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08180)  

**Abstract**: Abstractive text summarization systems frequently generate fluent yet unfaithful summaries by fabricating or distorting relationships between entities and
events. Such relation-level hallucinations undermine the reliability of generated summaries, particularly in high-stakes domains. In this work, we present a
refined and grounded framework for evaluating relation hallucination in abstractive summarization. We present the empirical Relation Hallucination Index (RHI) by
introducing a dependency-aware relation extraction algorithm that incorporates lemmatization-based normalization, named entity grounded subject resolution,
passive agent recovery, negation-aware verb modeling, reporting verb filtering, nominal relation fallback, clausal propagation, and systematic deduplication.
These enhancements improve the structural fidelity of extracted relation triples and reduce spurious matches during evaluation. In addition, we introduce a
normalized formulation of RHI to ensure scale-invariant comparison between datasets and models. The revised metric decomposes hallucination into interpretable
components, aggregates relation hallucination metric into a normalized relation faithfulness score. Extensive evaluation across multiple state-of-the-art
summarization models demonstrates that the grounded extraction process yields more stable and discriminative hallucination measurements. The proposed framework
advances automated relation-level faithfulness evaluation and supports coherence-aware, hallucination-sensitive model analysis. 

---
# Thinking vs. NoThinking: Towards Interpreting Reasoning Mechanisms of Large Language Models via Sparse Autoencoders 

**Authors**: Bo Cheng, Qiaolin Lu, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08168)  

**Abstract**: While Large Language Models (LLMs) employing Chain-of-Thought (CoT) exhibit superior reasoning capabilities, the neural mechanisms distinguishing this explicit Thinking mode from direct answer generation (NoThinking mode) remain poorly understood. To deconstruct this cognitive process, we apply Top-K Sparse Autoencoders (SAEs) to the intermediate representations of DeepSeek-R1-Distill-Qwen-7B and examine the model's divergent behaviors across math-solving tasks of three distinct difficulty levels. Observationally, we identify a clear distinction in how the model functions under two reasoning modes: Thinking mode relies on sparse and high-intensity feature activations driving verbal deduction independent of problem complexity, whereas NoThinking mode exhibits an adaptive and diffuse pattern prioritizing symbolic manipulation. Causally, suppressing the three most active sparse features by Total Activation Volume reveals three principles: (i) reasoning and syntactic structure are tightly coupled, as interventions consistently degrade \LaTeX{} and boxed-solution formatting; (ii) Thinking responds to disruption with compensatory over-generation marked by increased metacognitive cues and repetitive, low-information continuations; and (iii) coherent CoT behavior depends on a fragile coordination among specialized features, yielding distinct failure modes under perturbation but a consistently impaired output structure. 

---
# STEMMA: An Adversarial Multi-Agent Framework for Evaluating Self-Identity Consistency in LLMs 

**Authors**: Nuthakki Siva Gopala Krishna, Kanishka Jain  

**Link**: [PDF](https://arxiv.org/pdf/2608.08164)  

**Abstract**: Knowledge Distillation is a widely adopted technique in the training and fine-tuning of large language models (LLMs) enabling transfer of structured information and functional behavior from a large teacher model to a smaller student model while significantly reducing computational costs. However, as the use of distillation increases in both scale and complexity it raises an important question about what kind of knowledge is really transferred from the teacher model. In this work, we argue that apart from the functional knowledge, student models also learn behavioral patterns, specifically how a model represents its own identity raising concerns about output homogeneity, model biases, and accountability. To address this challenge, we introduce STEMMA, a multi-modal and multi-agent framework in which role specific agents collaboratively probe self identification behavior in different models. We also contribute a set of adversarial prompts designed manually to evaluate identity consistency in LLMs. Our results show that to an extent most models are vulnerable to inconsistencies in self-representations. 

---
# Can LLM Agents Stick to the Script? A Benchmark for Long-Horizon Consistency in Interactive Narratives 

**Authors**: Yingpeng Ma, Jianhao Yan, Bei Shi, Ka Hou Kam, Runnan Wang, Xuebo Liu, Yulong Chen, Yue Zhang, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2608.08160)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) is revolutionizing AI for Games by enabling open-ended and fluid interactive storytelling. However, existing research has largely overlooked the critical challenge of maintaining long-horizon logical consistency and narrative integrity against unconstrained user interventions. To address this, we formulate this challenge as Narrative Commitment Preservation (NCP), and take interactive narrative as our testbed. We introduce NCP-Bench, a benchmark of 100 narrative environments derived from movie synopses. Each environment includes a structured narrative specification (trajectory, commitments, and initial facts) that we can automatically check throughout the interaction between the player agent and the narrator agent. Experiments across state-of-the-art LLMs reveal a substantial long-horizon consistency gap: high linguistic quality does not guarantee commitment preservation; even strong models frequently generate logically conflicting content under adversarial interventions, with the best-performing model (GPT-5.2) achieving only 42% survival rate after 20 turns and fact conflict rates ranging from 40% to 68% across models, and only isolated runs satisfying all achievement commitments within the 100-turn limit. 

---
# NeuPAT: Neuron-aware Plasticity Allocation Tuning for Language-Preserving MLLMs 

**Authors**: Jiayue Jin, Jingwei Zhang, Chen Wang, Jing Liu, Longteng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.08107)  

**Abstract**: Multimodal expansion of large language models (LLMs) enables new perceptual capabilities but often compromises the language intelligence acquired during pretraining. In this work, we investigate this phenomenon from the perspective of internal adaptation dynamics and discover that neurons in pretrained LLMs exhibit heterogeneous plasticity during multimodal learning: some neurons are critical for preserving language capabilities, while others are more adaptive to multimodal knowledge. Based on this insight, we propose NeuPAT (Neuron-aware Plasticity Allocation Tuning), a lightweight and architecture-agnostic framework that allocates neuron-wise update constraints during multimodal instruction tuning. NeuPAT uses a small-scale probing stage to estimate neuron adaptation patterns and selectively protects language-sensitive neurons while promoting multimodal adaptation through more plastic neurons. Experiments across diverse LLM families demonstrate that NeuPAT recovers 94.5\% of the language capability degradation caused by vanilla tuning on 11 language benchmarks while maintaining comparable multimodal performance, providing an effective approach for capability-preserving multimodal expansion. 

---
# Wisdom in Unity: The Role of Multilingual Training in Figurative Language Identification in Proverbs 

**Authors**: Rama Alomair, Remas Alsubaie, Walaa Saifalislam, Rima Alsonbul, Mona Alnajjar, Razan Aldossari, Haya Alibrahim, Abeer Aldayel  

**Link**: [PDF](https://arxiv.org/pdf/2608.08090)  

**Abstract**: Although multilingual approaches to figurative language identification are not new, the shift beyond language homogeneous training data requires a clearer understanding of the contribution of translated multilingual supervision. We examine this question using 742 proverb concepts across 6,787 translated instances in seven languages. We evaluate five models, including multilingual encoders and instruction tuned LLMs, under progressively increasing levels of multilingual supervision. Moreover, we introduce a multidimensional annotation framework for proverbs that characterizes them through four complementary figurative forms: Metaphorical, Moral/Advisory, Cause-Effect, and Culture Specific.
Our findings show that approximately 50% of the translated multilingual training data is sufficient to achieve near-optimal figurative language identification performance. We further show that combining diverse figurative forms yields the strongest overall performance. A notable finding is that the least frequent figurative form, Culture Specific, exhibits the largest performance gains under multilingual supervision. Furthermore, the Moral/Advisory and Culture Specific forms contribute most to the performance of instruction-tuned LLMs on figurative language identification. These findings motivate multilingual figurative language identification to move beyond metaphor-centric taxonomies toward concept level multidimensional frameworks that explicitly model complementary forms of figurative meaning. 

---
# Archer: Adaptive Reuse of Cached Hidden States for Efficient Rollback in Diffusion Language Models 

**Authors**: Xuning He, Zinan Sheng, Yongding Tao, Huanyu Liu, Ge Li, Xue Jiang, Yihong Dong  

**Link**: [PDF](https://arxiv.org/pdf/2608.08086)  

**Abstract**: Diffusion language models (DLMs) iteratively refine a sequence, allowing earlier predictions to be revised as context evolves. This rollback capability distinguishes them from irreversible autoregressive generation, but makes inference costly. Every denoising update alters the global context, forcing both prompt and response states to be recomputed even though only response tokens are revisable. Key-value (KV) caching could reduce this cost, yet conventional caching assumes immutable historical states and is therefore difficult to reconcile with this http URL this paper, we introduce Adaptive Reuse of Cached Hidden States for Efficient Rollback (Archer), a training-free KV caching method for rollback-capable DLMs. Archer asymmetrically keeps the mutable response synchronized with the current hypothesis while reusing prompt K/V within a bounded state neighborhood. Although prompt representations also change under bidirectional attention, their token identities remain fixed; bounded reuse therefore amortizes repeated prompt computation without caching mutable response states. It also delays feedback from tentative tokens, reducing premature reinforcement of transient high-confidence errors and giving rollback more opportunity to correct them. Our analysis characterizes prompt reuse as a reversibility-aligned cache boundary, bounds its state-dependent approximation error, and gives a decoder-margin condition for preserving full-refresh this http URL DLM acceleration often trades quality for speed. Archer shifts this frontier, attaining the best mean performance of 33.63% together with a 2.57x mean speedup on the main suite. Across evaluated settings, it improves Pass@1 by up to 3.05 points and reaches up to 2.95x speedup. Controlled analyses connect the quality gain to delayed prompt feedback and validate state-aware refresh. Our code is available at this https URL. 

---
# Commitment Before Realization: When Classifier-Free Guidance Becomes Unnecessary in Masked Diffusion Language Models 

**Authors**: Fan Zhou, Weitian Wang, Tim Van de Cruys  

**Link**: [PDF](https://arxiv.org/pdf/2608.08082)  

**Abstract**: Classifier-free guidance (CFG) is usually kept on throughout masked diffusion language model decoding, although its benefit varies across prompts and over time. We study when CFG is actually needed by comparing, from any partial output, the probability of eventual constraint satisfaction under continued CFG and under base-only continuation. Their difference defines the remaining value of guidance. Guidance dependence is highly prompt-specific. Many prompts already succeed without CFG, while for others it provides no measurable benefit or can be harmful. For prompts that do benefit, the gain is often concentrated early. We define the commitment horizon $\astar$ as the earliest point from which switching all remaining decoding to the base model reduces final success by no more than a chosen tolerance. Under the base model, the corresponding success probability, or committor, is a martingale. To first order, CFG's per-step effect is governed by the covariance between the guidance logit direction and the successor committor. This gives a local account of when guidance can help, but it does not by itself locate the horizon. Among prompts with an observed preterminal horizon, $\astar$ is usually early and varies more within constraint families than between them. Freezing each prompt at its own cross-fitted horizon is noninferior to full CFG on all 13 subtasks at the prespecified margin, even while many tokens remain masked. This separates commitment from realization. The boundary also identifies a later region in which higher parallelism adds only a small cost in constraint success, although fluency still degrades with parallel width. For failed trajectories, reopening committed positions improves recovery in both failure modes. 

---
# DialectS2S: End-to-End Speech Dialogue Modeling for Low-Resource Chinese Dialects 

**Authors**: Yi Shu, Tianyu Peng, Yingzhuo Deng, Wen Yang, Jun Lin, Changming Xie, Xinyu Yu, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08067)  

**Abstract**: Current end-to-end speech dialogue models are primarily optimized for mainstream languages and remain limited in low-resource dialect scenarios due to the scarcity of dialect speech data. Moreover, during dialect adaptation, the semantic representation space of speech dialogue models continuously evolves, while conventional speech supervision remains unchanged, leading to semantic inconsistency between hidden representations and speech targets and degrading speech stability and naturalness. To address these issues, we propose DialectS2S, an end-to-end speech dialogue model for Chinese dialects. We first develop a scalable dialect speech dialogue synthesis pipeline for efficient data construction. We further introduce a two-stage post-training strategy with self-aligned speech supervision, which aligns the semantic content of speech supervision with the evolved semantic representations of the model to improve dialect speech generation quality. Experimental results show that DialectS2S consistently outperforms existing baselines across multiple Chinese dialects in speech dialogue, achieving substantial improvements in dialect consistency, response quality, and speech intelligibility. Our work provides an efficient and scalable solution for end-to-end speech dialogue modeling in low-resource dialect scenarios. To facilitate future research and practical applications, we fully open-source the DialectS2S framework, including model checkpoints, training datasets, and fine-tuning code. 

---
# APEX-VW: A Document-Level English-Spanish Post-Editing Dataset in the Healthcare Domain 

**Authors**: Marie Escribe, Tharindu Ranasinghe, Amal Haddad Haddad, Hansi Hettiarachchi, Damith Premasiri  

**Link**: [PDF](https://arxiv.org/pdf/2608.08059)  

**Abstract**: Post-Editing (PE) of Machine Translation (MT) output often involves repeating the same lexical and terminological corrections across many segments, especially in specialised and highly repetitive documents. Despite substantial work on Automatic Post-Editing (APE), most available corpora operate at the sentence level, others are synthetic, and overall not designed to study how corrections propagate in realistic Computer-Assisted Translation (CAT) workflows. This paper presents the APEX-VW (Automatic Post-Editing eXperiments on Virtual Wards) Corpus, a new open English-Spanish (EN-ES) dataset built from recent NHS virtual-ward documents and professional PE in Trados Studio, with controlled MT, terminology, and quality assurance settings. The corpus contains seven document-coherent source texts totalling 42k words, translated with four MT systems representing different paradigms and then post-edited by professional translators. Unlike prior resources such as WMT APE corpora, eSCAPE, MLQE-PE, or LangMark, the dataset preserves document order and CAT-tool context, making it suitable for research on terminology normalisation, correction propagation, and human-in-the-loop translation support. The paper describes the corpus design, data preparation, PE setup, and initial corpus statistics, and positions the resource as a benchmark for document-level APE and propagation-aware assistive tools. 

---
# Prompt Embedding Probes (PEP): Hallucination Detection in LLMs from Hidden States 

**Authors**: Zakhar Mrykhin, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2608.08024)  

**Abstract**: Large language models (LLMs) can generate fluent and useful responses but remain prone to hallucinations. We introduce Prompt Embedding Probes (PEP), a white-box method for answer-level hallucination detection from the hidden states of a frozen LLM. PEP extends standard linear probes by augmenting the input with a small number of learnable prompt embeddings. We evaluate PEP on TriviaQA, GSM8K, and MedQA using Qwen3 models at multiple scales. PEP improves hidden-state-based detection over standard linear probes in the main in-distribution setting. We further evaluate PEP for pre-generation prediction, cross-model transfer, and out-of-distribution generalization. PEP remains effective in the pre-generation and cross-model settings, whereas robust cross-dataset transfer remains difficult. These results show that prompt-based adaptation can strengthen hidden-state probing while keeping the backbone frozen and adding only a small number of trainable parameters. 

---
# Thinking Hard, Not Smart: Reasoning Models Fail to Ration Test-Time Compute Across Questions 

**Authors**: Chenrui Fan, Yize Cheng, Ming Li, Yongyuan Liang, Tianyi Zhou, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2608.07968)  

**Abstract**: Reasoning language models increasingly use test-time compute to improve performance, but existing evaluations typically study this compute one question at a time. Yet when multiple problems share an end-to-end cost or latency constraint, models must decide how to divide limited inference compute among them. We introduce an exam-style evaluation framework for studying this setting, in which a model must distribute one shared token budget across questions with different difficulty and point values to maximize its total score. Across several open and frontier reasoning models, we find that models fail to allocate a shared budget strategically across questions of varying difficulties and values. Models behave largely as greedy sequential solvers: they prioritize questions by presentation order, front-load effort on early questions, and remain insensitive to value, with these tendencies becoming more pronounced as the number of questions grows. Explicit planning prompts spread compute more evenly but do not produce value- or difficulty-aware prioritization. The same behavioral pattern extends from mathematical to code reasoning. These findings establish global budget allocation as a distinct capability that is not captured by conventional per-question evaluation and remains a challenge for current reasoning models. 

---
# Detection of Self-Introductions in Legislative Testimony 

**Authors**: Sofija Dimitrijevic, Pallavi Das, Kasey Liu, Foaad Khosmood  

**Link**: [PDF](https://arxiv.org/pdf/2608.07891)  

**Abstract**: Self-introductions are common in legislative committee testimonies. Successfully detecting them and extracting the speaker's name is enormously helpful in the task of speaker identification in the context of government meetings. In this paper, we present a pipeline for detection of self-introductions in legislative committee testimony using machine learning. We construct a training dataset from 1.54 million utterances spanning five state legislative sessions, apply a name-matching heuristic to generate automatic labels, and train three classifiers: a decision tree, random forest, and XGBoost to find self-introductions and extract the speaker's name. We construct a feature set combining bag-of-words, positional context, structural signals, introductory phrase indicators, and discourse context features. Among the three classifiers, XGBoost achieves the best performance with an F1 score of 0.9747 and the fewest total errors; adding fine-tuned BERT probability features improves this further. As an extension, we score the full candidate dataset with a fine-tuned BERT classifier and add BERT probability outputs as features. This BERT-augmented XGBoost model improves F1 from 0.9747 to 0.9782 and reduces total test errors from 241 to 207. The primary gain over the decision tree baseline (F1 0.9323) is driven by discourse context features and the boosting ensemble strategy; BERT provides a modest complementary signal. Analysis of false positives reveals that a minority are genuine self-introductions mislabeled due to name inconsistencies in the source data, indicating that measured metrics modestly understate true performance. 

---
# SurakshaEval: An Indic Safety Benchmark for Multilingual LLMs 

**Authors**: Debopriyo Banerjee, Kapil Rajesh Kavitha, Angana Borah, Xudong Han, Yuxia Wang, Parameswari Krishnamurthy, Utkarsh Agarwal, Atharva Kulkarni, Swaran Lata, Ayush Munot, Dhruv Sahnan, Aaryamonvikram Singh, Preslav Nakov, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2608.07862)  

**Abstract**: Existing safety evaluation datasets for large language models (LLMs) predominantly focus on English and Western contexts, often overlooking the linguistic diversity and culturally grounded safety risks present in other languages. To address this gap, we introduce SurakshaEval, a novel safety benchmark composed of human-written prompts spanning real-world scenarios, explicitly designed for ten major Indian languages - Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, and Telugu, along with English. SurakshaEval includes both generic prompts common across India and region- and language-specific prompts that capture localized sociocultural sensitivities. We benchmark a broad range of state-of-the-art LLMs on SurakshaEval, establish baseline safety performance, and identify recurring failure modes, including over-refusal, missed detection of implicit bias, and insufficient contextual awareness in regionally sensitive settings. Our results show that even strong multilingual LLMs struggle to reliably meet nuanced safety requirements when operating in Indic languages, particularly in native scripts. These findings highlight the urgent need for safety evaluation frameworks that incorporate region-specific data and structured assessment protocols, enabling the development and deployment of AI systems that operate securely, ethically, and in alignment with diverse societal values. Our code and data are available at this https URL. Warning: This paper contains text that may be offensive or unsafe. 

---
# "Many Are My Names": The Anatomy of the Assistant and Its Personas via Sparse Autoencoders 

**Authors**: Adelaide Danilov, Aria Nourbakhsh, Oleksandr Marchenko Breneur, Salima Lamsiyah  

**Link**: [PDF](https://arxiv.org/pdf/2608.07852)  

**Abstract**: How a language model internally represents who is speaking, the Assistant, an assigned roleplay persona, or a narrated story character, remains underexplored. We study speaker representations using a dataset of user-expressed emotional text and corresponding model responses. We decompose three generation settings (Assistant, Roleplay, and Story) into sparse autoencoder features extracted at turn-boundary and pronoun-token positions and selected through a filtering pipeline for different depths. We characterize each surviving feature through its steering effects and activation distribution. Our main finding is that the Assistant and roleplay personas are not independent alternatives: personas retain the Assistant-associated feature core while progressively differentiating from it across layers, starting from operational machinery towards behavioral and stylistic features. Meanwhile, generated story characters lack the Assistant-associated core. Both Story and Roleplay can be distinguished from the Assistant with Immersive Simulation Mode. However, the Assistant can sometimes enter or slowly drift into it even in the default setting. 

---
# On the use of foundation models in cognitive science 

**Authors**: Raj Sanjay Shah, Alex Warstadt, Michael Frank, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2608.07812)  

**Abstract**: A host of recent studies have evaluated the cognitive and developmental alignment of Foundation Models (FMs). These investigations include evaluations of their correspondence to adult performance across a range of cognitive domains, as well as whether aspects of model training track children's cognitive development. However, using FMs as candidate cognitive models poses significant methodological and conceptual challenges. A key question underlies this effort: under what conditions does behavioral alignment justify treating FMs as explanatory models of cognition? In this paper, we articulate a four-stage inferential framework for evaluating FMs as cognitive and developmental models: adapting human experimental tasks to model-compatible formats, specifying linking hypotheses that map model outputs to human measures, evaluating behavioral correspondence, and comparing across candidate models or manipulations. We clarify the role of linking hypotheses in mapping model outputs to human behavioral measures, identify challenges that constrain alignment claims, and propose principles for theory-driven and comparative evaluation. Throughout, we argue that behavioral fit alone is insufficient. Alignment becomes scientifically meaningful only when embedded within explicit theoretical commitments, theory-diagnostic tasks, and systematic contrastive evaluation across candidate models. 

---
# Jako Tako or Fluent? Presenting PoVisLE: A Polish Vision-Language Evaluation 

**Authors**: Anna Kołos, Grzegorz Statkiewicz, Karolina Seweryn, Katarzyna Kowol, Karolina Piosek, Wojciech Kusa  

**Link**: [PDF](https://arxiv.org/pdf/2608.07763)  

**Abstract**: Vision-language models (VLMs) have achieved strong performance on tasks such as image captioning, visual question answering, and image-to-text generation. However, they are predominantly trained on English-centric data, which limits their ability to handle culturally grounded visual understanding and leads to failures in interpreting region-specific meanings, symbolic content, and context-dependent visual cues. Existing benchmarks for cultural competence are often template-driven and focused on surface-level recognition, making them insufficient for evaluating deeper linguistic and pragmatic understanding in culturally situated settings. We introduce PoVisLE, a monocultural vision-language benchmark for Polish designed to evaluate culturally grounded multimodal understanding under a grounded evaluation paradigm, where language is interpreted in interaction with visual context. The dataset contains 1,117 images and 2,366 manually annotated VQA pairs. Overall, our dataset provides a controlled and challenging resource for assessing culturally grounded vision-language understanding beyond surface-level recognition. 

---
# The No-Meaning Falsity: The Structural Impossibility of the Arbitrary Sign in Classical Arabic 

**Authors**: Elnaserledinellah Mahmoud Abdelwahab  

**Link**: [PDF](https://arxiv.org/pdf/2608.07737)  

**Abstract**: This paper investigates whether the postmodern claim of unrestricted semantic indeterminacy, and its foundational Saussurean axiom of the arbitrary sign, are compatible with the structural architecture of Classical Arabic. We develop a formal mathematical model of Arabic non concatenative morphology in which lexical meaning is determined by the interaction between an invariant root and a morphosyntactic pattern. Within this framework, we establish a Morphological Correspondence Theorem, demonstrating that every lexical item is uniquely generated by a root pattern pair, and a Semantic Localization Theorem, proving that lexical meaning is determined at the derivational level prior to surface realization. To address Saussurean weaker notion of relative arbitrariness, we formalize it via conditional Kolmogorov complexity, defining arbitrariness algorithmically as the no rule property. We prove that general relative arbitrariness is formally undecidable, while Arabic relative arbitrariness is decidable and provably less than 1 for its motivated signifiers (Levels W and M), establishing a strict system complexity asymmetry over Indo-European languages. 

---
# Evaluating Dedicated Monolingual and Joint Multilingual Causal Models for Dravidian Languages 

**Authors**: Venkata Naga Sai Vishnu Rohit Pulipaka  

**Link**: [PDF](https://arxiv.org/pdf/2608.07727)  

**Abstract**: Dravidian languages, mainly Tamil, Telugu, Kannada, and Malayalam make up only a small part of the data used to train multilingual language models, so it's not clear how much per-language ability these models actually keep. I have trained five GPT-2 architecture models from scratch to compare four monolingual models (one each for Tamil, Telugu, Kannada, and Malayalam, each with its own 32K-vocabulary subword tokenizer) against one multilingual model sharing a 64K-vocabulary subword tokenizer across all four languages. All the 5 models are trained on cleaned CC-100, Wikipedia, and Samanantar data. I have tested the models on perplexity, bits-per-byte, tokenizer efficiency, and fine-tuning results which are compared against mGPT. The monolingual models outperform mGPT on sentiment classification and named entity recognition, and their tokenizers proved more efficient than the shared multilingual model across all the languages tested. 

---
# SurveyReview: A Reviewer-Aligned Benchmark for Survey Evaluators 

**Authors**: Yuheng Zhang, Yuanchun Wang, Fanjin Zhang, Ruyu Zhao, Juanzi Li, Jie Tang, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07641)  

**Abstract**: The rapid advancement of large language models has transformed survey writing from a months-long manual effort into an automated process. As generation scales, reliable evaluation becomes the bottleneck, and LLMs are increasingly used as survey evaluators. However, existing approaches largely rely on off-the-shelf LLM-as-a-judge methods without systematic alignment to human reviewers, and there remains a lack of systematic frameworks for quantifying alignment with human reviewers. To address this gap, we propose SurveyReview, a reviewer-aligned, multi-dimensional benchmark and dataset for survey evaluation. We collect and annotate 675 survey papers with 1,630 review reports. We structure authentic peer-review reports by converting free-form comments into four-dimensional scores (Readability, Criticalness, Comprehensiveness, Structure) paired with supporting rationales. We further release standardized train/test splits and an evaluation protocol to measure alignment between automatic evaluators and human reviewers. To validate the benchmark, we develop SurveyAlign, a strong baseline evaluator by fine-tuning Qwen3-32B with LoRA on our annotated data, augmented with external knowledge for knowledge-intensive dimensions. On the test set, SurveyAlign substantially improves reviewer alignment over prompt-based judging with GPT-5.2, reducing average MSE from 2.28 to 1.38 and MAE from 1.15 to 0.69 across all four dimensions. Our contributions are twofold: (1) we establish the first multi-dimensional, reviewer-aligned dataset with a reproducible evaluation framework for survey reviewing; (2) we develop a strong baseline evaluator that substantially improves alignment with human reviewers, providing a competitive reference for future research. Our code and data are available at this https URL 

---
# Embedding Initialization for Unseen Low-resource Languages in Multilingual NMT: A Case Study on Limbum-English Translation 

**Authors**: Samiratu Ntohsi, Neza David Tuyishimire, Anesu Kafesu, Marvin Ogore, Samuel Oluwajunwonlo Babalola, Oche Ankeli  

**Link**: [PDF](https://arxiv.org/pdf/2608.07629)  

**Abstract**: Multilingual neural machine translation models such as NLLB-200 cover 200 languages but leave thousands unsupported, including most Grassfields Bantu languages of Cameroon. When fine-tuning these models for an unseen language, practitioners must choose a proxy language token, yet no principled method exists for this selection. We implemented an embedding initialization strategy where a language token is the average of embeddings from multiple typologically related languages already in the mod el. We evaluate this approach on Limbum-to-English translation using a parallel corpus of 8,837 sentence pairs from New Testament text and a bilingual dictionary. We compare models: NLLB-200 zero-shot (chrF2++ = 12.5), a Transformer trained from scratch (chrF2++ = 14.5), NLLB-200 fine-tuned with a Swahili proxy token (chrF2++ = 47.3), and NLLB-200 with our averaged embedding initialization (chrF2++ = 46.7). We find that the multi-language initialization achieves performance comparable to the best single-language proxy. Both NLLB-200 variants improve over the from-scratch baseline by over 32 chrF2++ points. These results show that multilingual transfer is the dominant factor in extremely low-resource Bantu translation while eliminating the need for heuristic proxy selection. However, all systems fail to preserve tonal diacritics, highlighting an open challenge. We make our dataset and code available to support further research. 

---
# Scaling Inherently Interpretable Language Models 

**Authors**: Guide Labs Team, Andreas Madsen, Aya Abdelsalam Ismail, Giang Nguyen, Isaac Plant, Muawiz Chaudhary, Nathaniel Monson, Saqib Azim, Zhichen Guo, Julius Adebayo  

**Link**: [PDF](https://arxiv.org/pdf/2608.07594)  

**Abstract**: Interpretability is often treated as a tax on capability: language models are trained as opaque systems, then explained after the fact, with methods whose reliability is difficult to establish. In this work, we challenge this premise. Rather than reverse-engineering a model, we make interpretability a constraint of the training pipeline, optimized alongside the language modeling objective. Across three orders of magnitude of compute, on both autoregressive and diffusion language models, interpretability scales with capability rather than against it. Surprisingly, model representations become more disentangled and aligned with human-understandable concepts with scale.
We instantiate the training-time recipe with Steerling-8B, a diffusion language model with a causal attention mask. For any group of generated tokens, Steerling-8B attributes the output to relevant input tokens, human-understandable concepts, and training data. This enables closed-loop intervention: diagnose an output through its concept or feature attribution, retrieve similar training data, and correct the behavior through concept steering without retraining. Steerling-8B remains competitive with open peer models trained on substantially 2-16x more compute, suggesting a different scaling paradigm: interpretability can be designed into training, and it improves with scale. 

---
# Search-G1: Grounded Search Agents via Representation-Based Intrinsic Rewards 

**Authors**: Cheng Ruoxi, Ma Haoxuan, Zhang Hongyi, Zhang Junming, Duan Ranjie, Xia Qiaolin, Wang Hao, Lu Yu, Shi Haibo, Ma Xingjun  

**Link**: [PDF](https://arxiv.org/pdf/2608.07531)  

**Abstract**: Search-augmented language agents should retrieve external information only when necessary and ground their answers in retrieved evidence. Existing external rewards provide either sparse outcome supervision or richer feedback from process annotations and LLM judges. Outcome rewards scale readily but cannot distinguish grounded retrieval from redundant search, whereas richer signals require costly annotation or inference during training. Internal rewards based on policy-side signals such as entropy, likelihood, or information gain are graded and inexpensive to evaluate, yet mainly reflect model confidence rather than evidence grounding. We propose Search-G1, a representation-based intrinsic reward framework that measures the operational grounding of an agent's answers through two intervention-calibrated readouts. A prompt-state readout predicts closed-book sufficiency, whose complement defines policy-relative retrieval necessity; an answer-commit readout estimates evidence reliance from answer-stage sensitivity to evidence deletion. Together, they provide additional credit to correct searched trajectories when retrieval is estimated necessary and the answer is evidence-sensitive, favor correct direct answers when closed-book knowledge suffices, and penalize repeated search. After calibration, reward scoring requires neither process annotations nor LLM-as-judge inference during policy optimization. Because reinforcement learning changes policy representations, Search-G1 periodically refits both readouts on trajectories from the latest checkpoint, allowing the reward to co-evolve with the policy. Experiments across multiple search-based question-answering benchmarks and two model scales show that Search-G1 improves the grounding--search-cost trade-off, producing shorter response-side trajectories at competitive task accuracy. Code is available at this https URL. 

---
# WuYuEval: A Multi-Level Benchmark for Large Language Models in Solid Waste Management 

**Authors**: Yi Zhang, Hongyang Wang, Zheng Hao Leong, Zihao Wu, Kaijun Lin, Zhixing Pan, Qixun Huangfu, Wei Ren, Wenyan Wu, Fangyun Wang, Wenting Yu, Hengyu Lin, Muling Yang, Zongguo Wen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07529)  

**Abstract**: Large language models (LLMs) are increasingly used as technical assistants, but their competence in solid waste management (SWM) remains difficult to assess because existing benchmarks emphasize general knowledge rather than professional decisions under engineering, environmental, and policy constraints. We introduce WuYuEval, a multi-level benchmark for evaluating LLMs in SWM across foundational knowledge, domain reasoning, and expert decision-making. After quality auditing, WuYuEval contains a Foundation Module with 4,590 closed-ended multiple-choice questions across six task types and eight domain categories, together with an Expert Module with 247 scenario-based open-ended questions involving multi-objective optimization, constraint trade-offs, and system design. For expert tasks, we combine anchor-calibrated LLM-as-a-Judge scoring with Elo-based pairwise comparison. Across 33 LLMs, performance varied widely. The leading model reached 94.64\% accuracy on the Foundation Module, but average accuracy still fell from 84.14\% on easy questions to 42.50\% on hard questions, with lower performance concentrated in calculation, experimental design, urban planning, and open-ended expert tasks. Reasoning-oriented Thinking modes improve most matched model pairs after auditing, but the gains depend on baseline capability and are not uniformly positive. These results suggest that visible deliberation helps only when it remains anchored to units, assumptions, and engineering constraints; otherwise, it may drift from decisive answer boundaries. WuYuEval therefore provides both an evaluation resource and an empirical basis for developing SWM-oriented foundation models with professional reasoning chains and explicit constraint control. 

---
# DocAtlas: Long-Document Understanding as Mutable-State Interaction 

**Authors**: Hongchen Wei, Yuanzhe Wang, Bei Liu, Yifan Yang, Qi Dai, Kai Qiu, Yunsheng Li, Dongdong Chen, Chong Luo, Zhenzhong Chen, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.07527)  

**Abstract**: Long-document understanding requires models to find and combine evidence across many pages, layouts, tables, figures, and charts. Existing retrieval-augmented systems usually select evidence from a static index before generation, while recent agentic systems add multi-turn tool use but often rely on frozen proprietary backbones whose behavior is set by prompts. We present DocAtlas, a system that treats long-document understanding as a mutable-state information-seeking process. We instantiate DocAtlas as a mutable document harness: an external environment that determines what document information is searched, read, stored, reviewed, and shown to the model at each step. Given a document and question, the harness exposes search, reading, note-taking, and review tools, maintains a hierarchical tree and note store, and updates both as the agent records evidence. DocAtlas combines self-improving retrieval, selective evidence access, and active working memory under a fixed context budget. The same harness supports inference-time use with large VLMs and end-to-end reinforcement learning for compact VLM agents. With GPT-5.4, DocAtlas reaches 71.4\% on MMLongBench-Doc, exceeding the human-expert reference of 65.8\%. A Qwen3.5-4B VLM trained with end-to-end RL in the DocAtlas environment reaches 63.7\%, compared with a 54.4\% direct-input baseline, showing that mutable document-harness design can improve compact document agents by a large margin. 

---
# Unified Hallucination Fuzzing for Multimodal Large Language Models 

**Authors**: Pengfei Zhou, Jiajun Song, Zhiwei Tang, Yixing Ma, Xiaopeng Peng, Donghui Si, Yuhang Xu, Huiqi Song, Yiyuan Miao, Yichen Qian, Weihua Chen, Wangbo Zhao, Bohan Zhuang, Jiasheng Tang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2608.07525)  

**Abstract**: Hallucination remains a persistent challenge for Multimodal Large Language Models (MLLMs), severely limiting their reliability in high-stakes applications. Existing evaluations, predominantly based on static benchmarks, suffer from narrow taxonomical coverage and rapid performance saturation, failing to reflect model robustness in evolving real-world scenarios. To bridge this gap, we present a systematic evaluation framework integrating a comprehensive benchmark with self-evolving stress testing. First, we introduce UniHall, a fine-grained dataset grounded in a unified taxonomy spanning Object, Instruction, and Knowledge dimensions. Second, to address benchmark saturation, we propose Self-Adaptive Multimodal Fuzzing (SAMF), a self-adaptive framework that employs evolutionary mutation strategies to explore the boundaries of model hallucinations. Crucially, to ensure reliable assessment of dynamic inputs, SAMF incorporates a structured metric suite driven by an ensemble of multi-modal oracles. Our extensive experiments reveal that state-of-the-art MLLMs exhibit significant performance degradation under fuzzing compared to conventional settings, exposing a dissociation between reasoning capabilities and factual grounding. Furthermore, we identify a helpfulness-hallucination trade-off, where reinforcement learning alignment inadvertently exacerbates sycophancy in instruction-following tasks. The framework, code and benchmark are available at this https URL. 

---
# Beyond Naturalness: Probing Automated Text-To-Speech Evaluators on Linguistically Grounded Dimensions 

**Authors**: Oluwanifemi Bamgbose, Simon Rosen, Jash Shah, Lindsay Devon Brin, Hoang H Nguyen, Anke Koelzer, Rachel Hansen, Tara Bogavelli, Fanny Riols  

**Link**: [PDF](https://arxiv.org/pdf/2608.09930)  

**Abstract**: Automated Text-to-Speech (TTS) evaluation methods (Mean Opinion Score (MOS) predictors and Audio Large Language Models (Audio-LLM) judges) are expected to reflect human perception, yet it is unclear how well they capture the distinct aspects of speech that listeners actually perceive. We deconstruct "naturalness" into a linguistically grounded annotation schema spanning 10 distinct perceptual dimensions, and use it to construct the first dimension-level meta-evaluation benchmark for TTS, comprising 860 utterances annotated by trained linguist raters. Results from benchmarking four MOS predictors and four Audio-LLM judges reveal that MOS predictors collapse onto acoustic signal quality, while Audio-LLM judges show selective, prompt-dependent detection that does not generalise across all dimensions. Neither class reliably captures a breadth of linguistically structured speech errors. Our dataset, annotation schema, and evaluation code are publicly released to support more targeted and interpretable TTS evaluation. 

---
# Multimodal Model Diffing for Feature Discovery and Control 

**Authors**: Hunar Batra, Lachin Naghashyar, Ashkan Khakzar, Philip Torr, Christian Schroeder de Witt, Constantin Venhoff, Ronald Clark  

**Link**: [PDF](https://arxiv.org/pdf/2608.09928)  

**Abstract**: Multimodal Large Language Models (MLLMs) exhibit strong visual understanding, yet the internal features that cause these behaviors remain difficult to identify, audit, or control. While applicable to post-hoc inspection, hidden states that are decomposed into interpretable feature directions using sparse autoencoders (SAEs) neither readily isolate which features are changed by multimodal training, nor are they directly useful for targeted control. We introduce MMDiff, a multimodal model-diffing framework that trains multimodal SAEs and turns them into feature-level interfaces for discovering and controlling multimodal behavior. MMDiff supports three uses: (i) feature isolation, by diffing a base-LM SAE against its multimodal-adapted counterpart to identify features altered by multimodal training; (ii) task-specific feature detection, via per-token contrastive firing analysis that isolates causal features; and (iii) feature-level control, by causally removing or steering the discovered feature directions. We train multimodal SAEs for three MLLM families, LLaVA-MORE, PaliGemma 2, and InternVL3.5, and evaluate on visual-spatial understanding, multimodal safety, and OCR. MMDiff discovers sparse, causally specific features whose removal selectively degrades target behaviors by an average of 12% on spatial tasks and 17% on OCR, and reduces attack success rate by 24% on multimodal safety attacks, with no impact on VQA performance. Steering these features improves spatial and OCR accuracy by +3.6% and +1.8% on average over a standard single-layer steering baseline. These results show that multimodal SAEs can serve not only as interpretability tools, but as mechanisms for auditing, steering, and controlling MLLMs behavior toward safer and more capable generations. 

---
# Towards Expert-level Medical AI for Real-time Video Consultations 

**Authors**: Mahvish Nagda, Jihyeon Lee, Matthew Thompson, Chunjong Park, Tim Strother, Valentin Liévin, Roma Ruparel, Akshay Goel, Teya Bergamaschi, Suhana Bedi, Meet Shah, Pavel Dubov, Liviu Panait, Toshiyuki Fukuzawa, Sam Schmidgall, Craig Schiff, Joseph Xu, Aliya Rysbek, Yana Lunts, Jan Freyberg, Rebecca Hemengway, Sunny Virmani, David Racz, Carey Radebaugh, Joëlle Barral, Kavi Goel, Dale R. Webster, Katherine Chou, Avinatan Hassidim, Yossi Matias, James Manyika, Gregory Wayne, Tao Tu, Yun Liu, Ethan Goh, Christina Chen, Ryutaro Tanno, Po-Hsuan Cameron Chen, Mike Schaekermann, Anil Palepu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09861)  

**Abstract**: Audio-visual interaction is the standard for patient-physician consultations, enabling natural communication and effective assessment of illness through non-verbal cues. While text-based AI has shown promise, it discards essential perceptual dimensions and limits patients who cannot articulate symptoms in writing. Early efforts to extend medical AI to audio-visual interaction have demonstrated feasibility but not reached clinician-level performance. Here, we provide the first demonstration of expert-level AI in real-time clinical video consultations using AMIE (Articulate Medical Intelligence Explorer) in a video configuration. AMIE (Video) is a Gemini-based multi-agent system integrating low-latency dialogue, clinical reasoning, and real-time audio-visual perception. To guide development, we established a taxonomy and automated evaluations for clinical audio-visual cues in telehealth settings. In a randomized Objective Structured Clinical Examination (OSCE) study with 30 primary care physicians (PCPs), 15 patient actors and 100 clinical scenarios, we compared AMIE (Video), its text-only counterpart AMIE (Text), and PCPs consulting via video. Clinical evaluators rated AMIE (Video) on par or better than PCPs in history-taking, diagnosis, management, and physical observation and examination. Patient actors preferred AMIE's approach to assessing and explaining conditions, while PCPs were preferred for rapport and partnership building. In modality ablation, patient actors preferred AMIE (Video)'s interface over text chat for communicative effectiveness, convenience, and feeling understood. Limitations remain in fine anatomical precision, subtle affective nuances, and high-frequency movements. While further research is needed before real-world translation, these results mark an important milestone toward AI systems capable of augmenting care across the sensory complexity of clinical practice. 

---
# Agentic Auto-Research is Fuzz Testing 

**Authors**: Yifeng He, Jicheng Wang, Yinzhe Zhao, Jiachen Liu, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.09855)  

**Abstract**: Autonomous research agents can generate experiments faster than researchers can validate them. Researchers have responded by scaling the proposer and ranking more samples with a learned judge or human reviewers. We argue that this *generate-and-rank* paradigm misses the problem of sparse feedback. Within a declared research problem, an agent follows the control loop of a greybox fuzzer: it proposes a candidate, executes it, observes feedback, and chooses what to try next. A fuzzer rarely finds a bug, but coverage makes partial progress observable on every execution. Fuzzers then use that signal to mutate inputs and allocate effort, rather than only to rank completed runs. Auto-research needs the same two capabilities. First, each experiment should expose a cheap, dense signal of epistemic progress before final scientific validation is available. Second, that signal should determine the next intervention so that the agent searches rather than repeatedly samples. Because the optimized progress signal is guidance rather than a verdict, final validation must still decide what counts as a discovery using evidence protected from adaptive reuse. We propose controlled tests of whether candidate signals predict validated progress, whether feedback-directed search yields more validated discoveries per unit cost than repeated sampling, and whether protected validation reduces false discoveries. Feedback architecture, not only generation, is a central bottleneck in auto-research. 

---
# Mismatch Matters: On-Policy Distillation Beyond Token Agreement 

**Authors**: Zichao Yu, Chengzhi Yu, Shengze Xu, Yujin Han, Bingqing Jiang, Xu Wang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2608.09836)  

**Abstract**: On-policy distillation (OPD) has emerged as a core component of modern LLM post-training pipelines, yet we reveal a failure mode: degenerate agreement, where students exploit repetitive loops to achieve near-perfect token agreement with the teacher despite globally flawed responses. We therefore shift our focus from agreement to teacher-student mismatch, and find that mismatch tokens can be mainly categorized into two types: student-excess tokens and student-deficit tokens. Student-excess tokens are generated by the student but assigned near-zero probability by the teacher; their log-ratio corrections grow unbounded and destabilize the update. Student-deficit tokens, in contrast, are preferred by the teacher but rarely sampled by the student; their absence blocks the transfer of the teacher's reasoning patterns. To tackle these mismatch directions, we propose TIDE (Token-level Independent Deficit-Excess correction), which applies bounded Hellinger shaping to suppress the most severe sampled excesses and an analytic teacher top-$K$ injection to restore deficient probability mass without requiring deficit tokens to be sampled. Across mathematical reasoning benchmarks with multiple Qwen3 teacher-student pairs, TIDE consistently outperforms standard OPD and recent token-selection and reward-shaping baselines. Moreover, the gains of TIDE are more pronounced under strong teacher-student mismatch, where it improves Avg@8 from 6.9% to 20.3%, reduces average response length by a factor of 3.6, and substantially reduces formatting failures. Code is available at this https URL 

---
# Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA 

**Authors**: Mind Lab, Vin Bo, Asher Cai, Jingwei Cao, Song Cao, Vic Cao, Amelia Chen, Andrew Chen, Kaijie Chen, Cleon Cheng, Steven Chiang, Kaixuan Fan, Hera Feng, Huan Feng, Arthur Fu, Jun Gao, Pyke Han, Nolan Ho, Ori Hong, Hailee Hou, Piers Hua, Charles Huang, Miles Jiang, Nora Jiang, Yuyi Jiang, Qiuyu Jin, Fancy Kong, Kuss Koo, Jaron Lee, Andrew Lei, Alexy Li, Dawn Li, Lucian Li, Ray Li, Ricardo Li, Smith Li, Theo Li, Allen Lin, Elliot Lin, Fan Lin, Chen Ling, Kairus Liu, Kieran Liu, Logan Liu, Neo Liu, Xiang Liu, Yuxin Lu, Maeve Luo, Pony Ma, Verity Niu, Cole Qiao, Guian Qiu, Vince Qu, Sentry, Niko Song, Vincent Wang, Bo Wu, Rio Yang, Evelyn Ye, Fiona Ye, Ina Ye, Regis Ye, Josh Ying, Atlas Zeng, Danney Zeng, Salmon Zhan, Anya Zhang, Di Zhang, Mia Zhang, Sueky Zhang, Wei Zhao, Ada Zhou, Adrian Zhou, Yuhua Zhou, Juno Zhu, Murphy Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2608.09819)  

**Abstract**: Macaron-V1 is an open agent-model family for experiential intelligence: learning from experience in real environments and continuing to learn after deployment. It is organized around two system goals. Adaptation is pursued through recursive improvement of versioned model-harness pairs, where experience from one configuration is evaluated under an external contract and used to construct its successor. Collaboration is pursued via the Mixture-of-LoRA (MoL) architecture that freezes a base model, composes specialist LoRA adapters, and selects one LoRA per user turn. The flagship Macaron-V1-Venti combines a 744B GLM-5.2 base with four LoRAs for chat, agent, coding, and GenUI; the Qwen3.6-based Macaron-V1-Tall (50B) uses the same design for local deployment. This report presents Macaron-V1 as a co-designed system spanning architecture, algorithms, and infrastructure. The MoL architecture supports continual learning through extensible LoRA specialists. The algorithm combines Model-Harness Co-design and recursive self-improvement loop, including the UI4A component-native GenUI harness, a stateful action substrate, versioned HCP contract, and the agentic RL framework MindForge. The supporting infrastructure includes the post-training platform MinT, the long-context RL method LongStraw, and stability techniques for sparse MoE and DSA base models. We evaluate Macaron-V1 on Personal Intelligence, GenUI, and general capability benchmarks against frontier baselines. Our results validate the current system, while compounding gains from continual learning and collective intelligence remain open questions. 

---
# Parameter Exploration for RLVR via Variational Learning 

**Authors**: Vatsal Venkatkrishna, Nico Daheim, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2608.09805)  

**Abstract**: Exploration has been a focus of reinforcement learning research for a long time. Recently, there has been growing evidence that it is also an important ingredient in LLM reinforcement learning recipes that can significantly impact downstream performance. Many existing methods control exploration in the action-space, for example, using temperature scaling. However, these methods cannot reorder tokens but only influence the variance in the output distribution. This limits exploration and can lead to divergence or stalled training. Here, we investigate parameter-space exploration, where rollouts are generated by sampling different policies from a posterior that may each explore different rollouts. Sampling less or more diverse policies is then a complementary control lever over exploration. We introduce a family of methods called Perturbed Parameter Policy Optimization (3PO) which use different sampling strategies and different rollout grouping for reward estimation. Experiments on OLMo-3-1025-7B and Qwen2.5-Math-7B across mathematical reasoning and code generation tasks show that these approaches consistently improve average downstream performance over standard GRPO at a near-identical FLOPs cost. Moreover, using multiple parameter samples consistently produces fewer zero-advantage groups and malformed or incorrect rollouts during training than GRPO and action-space baselines. Overall, our work presents evidence that parameter-space exploration can improve reinforcement learning for LLMs. 

---
# Matryoshka Language Model Suites 

**Authors**: Nathan Godey, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2608.09703)  

**Abstract**: Training a language model suite classically requires training each model separately and serving them independently. We improve both training and inference efficiency by stacking sub-models of increasing size into a single nested architecture trained end-to-end. This Matryoshka training framework reduces the total parameter count of the suite, enables low-cost distillation from the largest to all smaller sub-models at every training step, and is well-suited for speculative decoding as the draft model is contained within the verifier. We validate our approach by training a Matryoshka suite comprising 500M, 1.5B, and 3B sub-models. Our suite is on par with independently trained baselines on benchmark performance and validation and out-of-domain perplexities, while using 36% less training compute and improving the throughput of speculative decoding by 14-26%. We also ablate key architectural choices, offering guidance for building strong Matryoshka LM suites. 

---
# VeriForge: Mitigating Latent Knowledge Gaps in Narrative Drafting via Mixed-Initiative Scaffolding 

**Authors**: Ruqi Sun, Jiaping Li, Wenhui Tao, Ximing Zheng, Yuefeng Tan, Jiahao Wei, Yuxin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2608.09698)  

**Abstract**: Great fiction earns its verisimilitude through precise details, from how a longsword is gripped to pierce armor gaps to why a bleeding corpse cannot yet smell of decay, weaving domain expertise into the fabric of invented worlds. Current AI writing tools offer limited support for discovering and integrating unfamiliar domain knowledge into narrative. They require explicit queries that authors cannot formulate, generate finished prose that risks homogenizing voice, or assist only within the boundaries of what authors already know. We argue that AI should reveal latent knowledge gaps to writers while preserving their agency to transform discovered knowledge into authentic prose. Grounded in formative interviews with 9 fiction writers, we present VeriForge, a mixed-initiative writing system that divides cognitive labor so that the system assumes initiative over domain discovery while the author retains full initiative over narrative synthesis. VeriForge realizes this through three complementary mechanisms. Proactive inline highlighting flags potential knowledge gaps as authors draft. Dual-stream querying pairs conversational responses with source-anchored Knowledge Cards for direct fact extraction. A spatial Knowledge Canvas allows authors to organize and connect discovered knowledge across their writing. These mechanisms are powered by a graph-based retrieval-augmented generation pipeline grounded in domain-specific source materials. A within-subjects user study (N=12) provides preliminary evidence that this paradigm helps authors recognize previously overlooked knowledge gaps, supports creative exploration, and is perceived by expert raters to produce passages with stronger domain grounding in a controlled cold-start writing task. 

---
# Listwise Cross-Encoder Fine-Tuning vs. Agentic Instruction Tuning for LLM Rerankers: A Systematic Study in Medical Procedure Reranking 

**Authors**: Matan Fainzilber, Shlomit Plavner  

**Link**: [PDF](https://arxiv.org/pdf/2608.09650)  

**Abstract**: Reranking medical procedures against patient queries is a critical component of health insurance information retrieval, complicated by a substantial lexical gap between patient language and clinical nomenclature. We present a systematic comparison of two reranking paradigms for this production task: (1) small cross-encoders (MedCPT, MiniLM-L12) fine-tuned with listwise learning-to-rank objectives across layer freezing configurations, and (2) Qwen3-Reranker-4B, a 4B-parameter instruction reranker whose prompt is iteratively refined via an agentic optimization loop driven by GPT-4.1. On a purpose-built dataset of 2,647 queries across 708 insurance services, we find that a 109M-parameter cross-encoder fine-tuned with ListNet outperforms the 4B-parameter model by 2.6 percentage points on NDCG@3 and 13.3 points on Spearman correlation - at 37x fewer parameters. We report practical findings, a scalable LLM based dataset construction pipeline, and deployment trade-offs relevant to production reranking systems. We release our code and a sample dataset to support reproducibility and adaptation to other domains. 

---
# Avalon-ToM-Bench: Evaluating Fine-Grained Theory of Mind via Asymmetric Game Mechanics 

**Authors**: Yen-Shan Chen, Yu Chian Duan, Chih-En Kuo, Jian-Bin Wu, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.09638)  

**Abstract**: Theory of Mind (ToM) is essential for agent interactions, yet existing evaluations either rely on static scenarios that oversimplify mental-state reasoning or interactive settings that provide limited diagnostic insight. We present Avalon-ToM-Bench, a fine-grained benchmark that operationalizes ToM through the asymmetric-information mechanics of The Resistance: Avalon. Rather than evaluating end-to-end gameplay, it decomposes ToM into a 2$\times$2 taxonomy -- epistemic versus motivational reasoning crossed with inference versus action -- using human-crafted, perspective-constrained queries. Benchmarking 28 LLMs reveals three insights: 1) Reasoning, not knowledge. Models show strong game-rule comprehension but markedly weaker ToM abilities, isolating failures to social reasoning rather than missing domain knowledge. 2) Expression, not representation. Mechanistic analyses via linear probing and activation steering show that models frequently represent correct mental-state inferences in their hidden states but fail to express them during generation -- linear probes recover 77-82% accuracy versus 62-70% from the models' own chain-of-thought. 3) Policy, not deliberation. Dedicated reasoning training yields substantial improvements whereas test-time chain-of-thought provides only marginal gains (+11.0 versus +1.1 points on average), suggesting that robust ToM depends on a learned reasoning policy rather than increased inference-time deliberation. 

---
# Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching 

**Authors**: Kristian Schwethelm, Daniel Rueckert, Georgios Kaissis  

**Link**: [PDF](https://arxiv.org/pdf/2608.09444)  

**Abstract**: A main promise of looped language models (LMs) is depth-adaptive inference. By iterating a block of shared layers a variable number of times, the model can use less compute for "easy" tokens and more for "hard" ones. However, this adaptivity breaks standard batching: tokens in the same batch now require a different number of loops, so there is no unified forward pass, making efficient inference difficult. Standard inference frameworks like vLLM schedule on the token level and cannot handle this because tokens need to be removed from the batch within the forward pass. Loop-level scheduling has been proposed as a solution, but never implemented end to end. The key challenge is that looped architectures also contain non-looped boundary stages (e.g., token embedding and LM head) that must be scheduled at different frequencies than the loop. We introduce continuous depth batching (CDB), which schedules at the granularity of individual loop iterations. CDB handles boundary stages and loop steps in separate priority queues, makes exit decisions one step ahead, and overlaps all scheduling work with GPU computation. On Ouro 1.4B and Huginn 3.5B, CDB can realize up to $99\%$ of the theoretical maximum speed-up from adaptive-depth, translating to $1.5$-$1.9\times$ higher offline throughput and $45$-$90\%$ lower normalized latency under dynamic serving load. 

---
# Beyond the Capability Boundary: Zeroth-Order Optimization for Self-Evolving LLM Agents 

**Authors**: Bingzhen Liu, Xiaomeng Fan, Yuwei Wu, Zhi Gao, Mingyang Gao, Chuanhao Li, Yunde Jia  

**Link**: [PDF](https://arxiv.org/pdf/2608.09292)  

**Abstract**: Self-evolving methods improve the capabilities of LLM agents by sampling trajectories from the underlying LLMs and learning from these trajectories. However, these methods struggle to learn beyond the inherent capability boundary of the agents, since the agents cannot sample correct trajectories on difficult examples for further improvements. In this paper, we propose a zeroth-order self-evolution framework that enables agents to learn beyond their capability boundary by perturbing LLM parameters to adapt to difficult examples without any trajectory annotations. Specifically, we perturb LoRA parameters of LLMs, run the agent, compute the losses under the perturbed and original parameters, and use the loss difference to estimate gradients and further update the LoRA parameters. We sample trajectories using the updated LLMs for supervised fine-tuning to break through the capability boundary of the agents, forming a closed self-evolution loop. We introduce a parallel perturbation inference mechanism and an adaptive lookup mechanism to reduce time consumption in zeroth-order optimization, with an answer perplexity loss that provides smooth and stable zeroth-order loss values. Experiments on multiple deep research benchmarks show that our method obtains substantially more successful trajectories and consistently outperforms strong baselines, especially on difficult examples. The code and released artifacts are available at this https URL. 

---
# ComboShoppingBench: Evaluating LLM Agents for Budget-Constrained Basket Shopping with Coupons 

**Authors**: Adrian Li, Kelong Mao, Yudong Guo, Heming Xia, Xinwei Yang, Lirui Luo, Jace Wong, Pu Yao, Sulong Xu, Simiu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2608.09282)  

**Abstract**: Real-world shopping often requires constructing a basket of complementary items rather than retrieving a single product. Such combo-shopping tasks arise in device setup, meal preparation, event planning, and group takeout ordering, requiring joint reasoning about item compatibility, availability, store-level requirements, delivery fees, coupons, and budgets. Evaluation is challenging because multiple baskets may satisfy the same request, making exact-match metrics unsuitable, whereas semantic evaluation alone cannot detect infeasible orders, invalid coupon combinations, or incorrect payments. We introduce ComboShoppingBench, an agentic shopping benchmark for open-ended yet verifiable basket construction in a simulated commerce and takeout environment. During task synthesis, an exploration agent constructs a feasible and semantically coherent basket of purchasable products; this witness guides the generation of coupons, budget constraints, user queries, and aligned evaluation rubrics. During evaluation, LLM judges assess semantic satisfaction, response quality, and claim faithfulness, while deterministic validation checks product-ID validity, budget compliance, and coupon optimality. Experiments with diverse LLM agents demonstrate that even strong agents struggle on ComboShoppingBench, highlighting substantial room for improvement in reliable, constraint-aware combo shopping. 

---
# Business Truth, not SQL Accuracy: A Rule-Gated 7B Analytics Agent Outperforms a Direct-Prompted 32B Baseline 

**Authors**: Morris Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.09254)  

**Abstract**: LLM analytics agents are evaluated on SQL syntax accuracy, but production failures look different: questions with two valid business definitions, questions the warehouse cannot answer, deprecated columns after a schema change, and queries that execute successfully while returning the wrong business number. No execution-match metric can score them. This paper introduces WarehouseReliabilityBench, 400 frozen tasks over two synthetic warehouses in which roughly half the correct responses are a clarification, an abstention or a refusal, with pinned denominators and a pre-registered paired bootstrap fixing each claim verb before the numbers existed. QueryProof, a 7B agent, uses rules derived from a semantic layer and physical catalog to determine its behaviour, and gates every answer on deterministic post-execution checks. On an 80-task synthetic test split evaluated once, QueryProof outperforms a direct-prompted 32B baseline by +0.237 [+0.112, +0.375] Business Truth Rate at 71.0% lower cost per correct answer; against a cost-matched few-shot baseline the accuracy gain holds but the cost difference does not resolve. This compares systems rather than model sizes: the 32B baseline receives none of the scaffolding. False success falls from 0.754 to 0.351 of returned answers, and no wrong number was returned on an answerable task (0 of 24), though 13 answers went to questions requiring clarification or abstention. Removing the routing layer changes little (0.562 against 0.537), so the result does not depend on escalation. Routing tuned on validation over-abstains on test, and the fitted confidence model loses to the heuristic it replaced. Resampling template families rather than tasks widens both accuracy intervals to include zero, so the effect's direction is better supported than its magnitude. The gain tracks the deterministic layer, though no component ablation was run. 

---
# MoRSE: Task-Oriented Multi-Agent System with Mixture of Role-Subtask Experts 

**Authors**: Peiwen Li, Shiyang Zhang, Yangtian Zhang, Sizhuang He, David van Dijk, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2608.09251)  

**Abstract**: Large language model-based multi-agent systems have recently shown strong potential for complex, long-horizon tasks. However, existing methods mainly rely on coarse prompt-level differentiation without parameter adaptation for diverse subtasks, resulting in insufficient inter-agent heterogeneity and limited specialized capability that bottleneck performance on tasks with complex requirements. To address this, we introduce a Task-Oriented Multi-Agent System with Mixture of Role-Subtask Experts (MoRSE) that distinguishes agents with (role, subtask)-conditional specialization at both the task structure and parameter levels. To make agents' responsibility explicit at the task structure level, we formulate a task-oriented multi-agent system that decomposes each task into a dependency-aware Directed Acyclic Graph of subtasks and assigns each agent a specific (role, subtask), introducing task-level specialization across collaborating agents. Additionally, to address the diverse role and subtask parameter adaptation demands, we propose a dynamic Mixture of (role, subtask) LoRA Experts module with a prototype-based semantic router for subtasks, augmenting agents with parameter-level specialization on a shared LLM substrate cost-effectively. Then, to co-optimize experts and router stably under sparse task rewards, we further propose a hierarchical group-relative policy optimization with two-layer credit assignment that isolates expert updates from the cross-route variance introduced by routing decisions, disentangling expert quality from routing quality. Experiments on code-generation benchmarks across three backbones demonstrate the effectiveness of our approach, with improvements in both whole-task and step-wise performance, and the gains from trained specialization generalize across held-out task categories and domains. 

---
# Beyond Direct Identifiers: Probabilistic Privacy Risk Estimation for Privacy-Conscious LLM Query Delegation 

**Authors**: Li Siyan, Zhou Yu, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2608.09140)  

**Abstract**: Recent work on protecting privacy during user-LLM interactions often focuses on direct, explicit identifiers: the personally-identifiable information (PII) captured by standard detectors. One such approach is Privacy-Conscious Delegation (PCD), where a local LLM acts as an intermediary. However, privacy risk does not stem solely from explicit identifiers but also PII-free self-disclosures, leaving users identifiable through combinations of quasi-identifying traits. We investigate a probabilistic variant of PCD, where we augment its objectives with an LLM-driven probabilistic estimation of k-anonymity. To facilitate this, we first create the PUPA-SD dataset, which contains naturalistic user queries with self-disclosure. Our preliminary results indicate that optimizing PAPILLON on PUPA-SD improves quality on unseen conversations across a variety of local models and produces the best privacy-utility balance for Llama-3.2-3B, while smaller models struggle to jointly optimize quality and privacy. We propose k-anonymity as a useful auxiliary metric for tackling PCD. 

---
# PolicyKG: An Agentic LLM Pipeline for Translating Institutional Policies into SHACL Knowledge Graphs 

**Authors**: Ponkrit Kaewsawee, Chaklam Silpasuwanchai, Chutiporn Anutariya  

**Link**: [PDF](https://arxiv.org/pdf/2608.09028)  

**Abstract**: Institutional policies stay in natural language while the systems that check compliance demand machine-readable constraints. Bridging that gap is still done by hand.
PolicyKG closes the loop. It is an LLM pipeline that reads a policy PDF, classifies each sentence as an obligation, permission, or prohibition, lifts the label into first-order deontic logic, and emits SHACL constraints. Four stages run on a LangGraph state machine with per-stage validators. The piece that matters most is the Corpus Adapter: a YAML vocabulary registry that grounds LLM predicates in a target ontology. Retargeting to a new domain means swapping the registry, not retraining a model.
On the Asian Institute of Technology Policies and Procedures corpus (1,663 sentences, 443 rules), PolicyKG reaches 86.9% deontic classification accuracy (Cohen's kappa = .709). Three annotators independently re-label a 50-item sample and agree at Fleiss' kappa = .844. SHACL shape correctness on a 69-shape subset is F1 = .866. The FOL path handles 79.2% of rules; the rest go through a direct NL-to-SHACL fallback.
We audited every one of the 443 rules for second- or higher-order constructs. An automated regex checklist flagged none, and a first-author pass on the 92 FOL-fallback cases confirmed the same. The exact upper 95% Clopper-Pearson bound on the true HOL rate is 0.67%. This is an audit finding for one corpus, not a proof of FOL sufficiency for institutional policy.
Swapping the AIT registry for a GDPR registry raises exact property alignment from 1/15 to 11/15 (Fisher's exact p < .001; Cohen's h = 1.53). On the LexDeMod lease-contract benchmark (N = 200), Macro F1 drops to .370 because lease English uses "shall be entitled" for permission -- exactly the vocabulary mismatch registry swap is meant to fix. Repeated runs produce hash-identical SHACL outputs. 

---
# How People Evaluate AI-, Expert-, and Peer-Style Financial Advice 

**Authors**: Aryan Ramchandra Kapadia, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2608.09019)  

**Abstract**: As generative AI increasingly becomes a common source of daily decision-making, including financial choices, it is critical to understand how people evaluate AI-generated financial advice. We conducted a preregistered vignette experiment (N = 285) in which substantive financial content---including facts, numerical values, recommendation direction, and core reasoning---was held constant while communication style varied across AI Financial Assistant (AI), Certified Financial Planner (Expert), and Online Community Forum (OC) advice. Displayed source attribution was independently manipulated through correctly labeled, unlabeled, and mislabeled conditions, allowing us to separate attribution effects from source-specific communication cues. Expert advice was rated more favorably than AI advice on 9 of 10 outcomes (|d|=0.20--0.47), and this advantage remained visible without source labels, where Expert advice outperformed AI advice on 8 of 10 outcomes (up to d=0.60). Correct labels added limited differentiation, whereas mislabeling increased ratings of AI advice for situational fit and overall quality (d=0.42 for each) and attenuated the Expert advantage in situational fit (d=-0.36). Descriptive analyses further showed that AI advice was most responsive to displayed attribution and, conversely, that advice-style differences were most visible under an AI label. These findings show that financial-advice evaluations are shaped jointly by displayed attribution and message-level communication cues. We position disclosure not as a neutral transparency mechanism, but as an interpretive frame whose accuracy and interaction with message cues can shape trust and reliance. 

---
# Guardian Crawler: Retrieval-First Knowledge Discovery with Bounded LLM Augmentation for Noisy Web Intelligence 

**Authors**: Joshua Castillo, Santosh Nukavarapu, Ravi Mukkamala  

**Link**: [PDF](https://arxiv.org/pdf/2608.08994)  

**Abstract**: Retrieving relevant evidence from noisy web data is challenging, particularly in sensitive domains containing incomplete reports, heterogeneous language, and irrelevant content. We present Guardian Crawler, a reproducible retrieval-first testbed for controlled experiments on knowledge discovery and evidence-grounded summarization over synthetic web-like corpora. The architecture combines BM25 retrieval with risk-aware, embedding-augmented, and hybrid reranking, followed by constrained retrieval-augmented generation with explicit document citations. Experiments on a synthetic 900-document corpus and 10 queries produced the highest descriptive retrieval scores under risk-based reranking, with P@10 = 1.00 and NDCG@10 = 0.94, compared with 0.94 and 0.81 for BM25. The best hybrid and BM25+Semantic configurations reached NDCG@10 values of 0.94 and 0.88, respectively. All 41 evaluable generated bullets passed the lexical coverage threshold; an automated LLM judge classified 36 as supported, one as partially supported, and four as unsupported. These results demonstrate the feasibility of Guardian Crawler as a controlled testbed but do not establish statistical superiority, human-validated faithfulness, or transfer to live-web investigative environments. 

---
# Towards an LLM-based method for quantifying the sexual content in song lyrics 

**Authors**: Ignacio M. Sticco  

**Link**: [PDF](https://arxiv.org/pdf/2608.08885)  

**Abstract**: Reggaeton is one of the most widely consumed music genres in the world, and its lyrics are commonly regarded as highly sexualized. This claim rests mostly on qualitative studies and on small-scale quantitative ones. This paper has two goals. First, we present a reproducible method that uses a large language model to quantify thematic content in song lyrics along several independent dimensions. The method is not restricted to sexual content. Second, we apply it to a corpus of 1,259 songs by 12 reggaeton artists released between 2002 and 2025. The analysis covers four topics: a dataset characterization, a per-artist comparison, an analysis of how the dimensions change over time, and a comparison between our sexual-explicitness score and Spotify's own explicit flag. We release the data collection code, the scoring prompt, and the corpus, so that other researchers can replicate the approach or apply it to their own lyrics datasets. 

---
# Theory-Guided Deception Detection: A RAG-Based Artificial Intelligence Exploration 

**Authors**: David M. Markowitz, Timothy R. Levine  

**Link**: [PDF](https://arxiv.org/pdf/2608.08881)  

**Abstract**: The current work developed seven Retrieval-Augmented Generation (RAG) models based on leading deception theories and compared how deception judgments were made relative to baseline models. Across 700 statements drawn from five published deception datasets, four large language models (gpt-4o, claude-sonnet-4-6, ollama/llama3, deepseek-v4-flash), and two run-types (RAG vs. baseline), a total of 39,200 deception judgments were rendered. Detection accuracies were consistent with typical human accuracies and not statistically different across RAG (54.5%) and baseline models (54.6%). RAG-based models (57.0%) were less truth-biased than baseline models (59.7%), but the effect size was quite small. Theoretical perspective mattered little for accuracy yet mattered substantially for response bias, which ranged from highly lie-biased (the verifiability approach, 32.2%) to highly truth-biased (truth-default theory, 88.1%). Content effects and model effects further moderated the results. Theory-guided AI judgments are unreliable with current parameters, yet they might show promise with additional datasets, model testing, and theory-to-data matching. 

---
# Automated Generation of Complexity-Validated Decision Scenarios Using Large Language Models 

**Authors**: Abdalla Doleh, Toni Somers, Ratna Babu Chinnam  

**Link**: [PDF](https://arxiv.org/pdf/2608.08822)  

**Abstract**: Cognitive decision-making research depends on diverse scenarios with carefully controlled complexity, yet manual production is slow, inconsistent, and biased. We developed an automated pipeline that uses LLms to generate structured decision scenarios and validates their complexity through a composite framework rooted in established task-complexity theory. We evaluated 4,238 scenarios across multiple domains and complexity tiers. Measurement validation met rigorous psychometric standards. Agreement among five independent model families was nearly perfect, with an intraclass correlation coefficient of 0.997 and a kappa of 0.971. Known-groups validity demonstrated large separation between tiers, with an eta-squared of 0.587 and all pairwise comparisons significant at p less than .001. Factor analysis revealed a dominant complexity construct, with loadings between 0.87 and 0.96 across three frameworks, while interactivity formed a weaker secondary dimension at 0.34. Discriminant validity was limited by a strong relationship between complexity and text length that persisted after controlling for tier, yielding a partial correlation of 0.86. This constrains construct purity but does not undermine the instrument's tier-grading function. Model analyses showed a negative association between throughput and schema pass rate (r = -0.967, p = .007, n = 5), suggesting a speed-quality trade-off, though largely driven by one high-throughput model. Llama 4 Maverick generated scenarios fastest at 134 per minute versus 25 for DeepSeek Chat V3.2, but underproduced complex-tier scenarios, whereas DeepSeek Chat V3.2 balanced domain coverage with high schema compliance. The system demonstrated strong psychometric properties, enabling reliable classification into Simple, Moderate, and Complex tiers and providing the measurement infrastructure needed for downstream cognitive assessment of AI systems 

---
# Toward Metacognitive One-Shot Indirect Prompt Injection: Strategy Abstraction Via Outcome-Conditioned Reflection 

**Authors**: Sihan Hou, Xinmeng Hou, Zhijun Zhang, Zehao Wang, Xuhong Ren, Sibo Qin, Kuntharrgyal Khysru, Qing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2608.08795)  

**Abstract**: Tool-using large language model (LLM) agents are vulnerable to indirect prompt injection (IPI), in which malicious instructions embedded in external observations manipulate subsequent agent decisions and actions. Most existing adaptive attacks rely on repeatedly querying and refining against the target agent, whereas realistic attackers may have only a single opportunity to interact with an unknown target agent. We propose SAVOR (Strategy Abstraction Via Outcome-Conditioned Reflection), which shifts attack adaptation from test-time iteration to offline strategy distillation. SAVOR performs outcome-conditioned reflection over successful and failed trajectories collected from disjoint training environments, validates context-conditioned candidate strategies, and iteratively consolidates them into a reusable strategy memory. At test time, the frozen memory guides the generation of a single payload for each unseen target, requiring only one target-agent query and no target-agent feedback. Across two benchmarks and three victim models, SAVOR attains the highest average attack success rate in all six settings, leading the strongest prior attack by 2.5 to 11.8 points and the same injection channel without strategy learning by 23.1 points on Agent Security Bench, which holds out attacker tools, and 28.6 points on OpenClaw-IPI, an executable benchmark we introduce that holds out attack goals and verifies attacks through tool interactions and execution receipts. A memory learned under one defense also transfers to another. 

---
# AnchorFold: A Focus-Then-Fold Framework via Recursive Attention Propagation for Efficient Multi-Vector Visual Document Retrieval 

**Authors**: Haoyu Zuo, Yibo Yan, Xin Zou, Shuliang Liu, Yi Cao, Mingdong Ou, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08732)  

**Abstract**: Multi-vector vision-language retrievers enable fine-grained Visual Document Retrieval (VDR) through late interaction, but storing and scoring hundreds of visual patch embeddings per page incurs substantial overhead. Existing training-free methods rely on pruning or merging: pruning degrades sharply under aggressive compression, whereas merging does not explicitly prioritize important regions when forming representatives. We introduce AnchorFold, a training-free focus-then-fold framework for document-side index compression. AnchorFold applies Recursive Attention Propagation over visual self-attention graphs, performing multi-step propagation within each attention head and integrating scores across heads and layers. The focus stage selects the highest-centrality tokens as anchors. The fold stage assigns remaining tokens to their most similar anchors in the normalized retrieval space and summarizes each anchor-centered group through centrality-weighted aggregation. This preserves non-anchor contributions while concentrating capacity on structurally important tokens. Across ViDoRe v1/v2 and REAL-MM-RAG with three diverse retrieval backbones, AnchorFold consistently outperforms all evaluated training-free baselines at $\gamma \leq 0.20$. On ViDoRe v1/v2, it retains 98.3% of full-index NDCG@5 on average at $5\times$ compression, achieving near-lossless compression, and 92.4% at $20\times$ compression. 

---
# CuteTTS: Efficient and High-Quality Speech Synthesis via Autoregressive Modeling of Continuous Latents 

**Authors**: Yuqian Zhang, Yao Shi, Kexin Huang, Botian Jiang, Zhe Xu, Yiwei Zhao, Min Liang, Shuang Chen, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08638)  

**Abstract**: Zero-shot text-to-speech (TTS) now supports interactive assistants, personalized media, and accessibility tools. All TTS systems require faithful linguistic rendering, consistent speaker identity, and low-latency response. Yet compact streaming systems must preserve sufficient acoustic detail in a predictable low-rate latent sequence, while iterative diffusion sampling and classifier-free guidance multiply inference cost at every autoregressive step. To strike a balance between high-fidelity synthesis and low-latency inference, we present CuteTTS, a compact continuous-autoregressive TTS system. It combines semantically aligned causal VAE latents with patch-level autoregression, explicit speaker conditioning, and a bidirectional flow-matching head. We further introduce guidance-step distillation, which absorbs classifier-free guidance and multiple solver steps into a single interval-conditioned student. Evaluations on LibriSpeech and Seed-TTS-Eval demonstrate competitive intelligibility and speaker similarity in zero-shot voice cloning, while distillation lowers first-audio latency by 23.3% and real-time factor by 40.8% relative to the base model with comparable objective and subjective quality. These results provide a practical path toward continuous-autoregressive TTS that reconciles high-fidelity generation with the latency demands of real-time interaction. 

---
# Can Open-Weight Models Compete on Financial Text Comprehension? 

**Authors**: Jan Spörer  

**Link**: [PDF](https://arxiv.org/pdf/2608.08634)  

**Abstract**: Open-weight language models from Chinese AI labs caught up on benchmarks relative to proprietary frontier models in recent months. Yet their reliability on real-world financial tasks remains largely untested. We updated the Financial Touchstone benchmark, which now has 2,967 question context-answer triplets across 495 international annual reports. We also apply a new set of models on the benchmark, expanding coverage from eleven to twenty models across ten providers, including recent open-weight models such as GLM 4.7, GLM 5, Kimi K2.6, and DeepSeek V3.2, as well as Alibaba's proprietary flagship Qwen3-Max. Anthropic's Claude Opus 4.6 achieves the highest accuracy (88.4%), while Google's Gemini 2.5 Pro maintains the lowest hallucination rate (0.08%). Notably, the open-weight Kimi K2.6 ranks third in accuracy, and the non-reasoning models GLM 5 and Mistral 3 rank fourth and fifth, challenging the assumption that reasoning architectures or proprietary weights are a prerequisite for strong financial comprehension. Information retrieval remains the primary bottleneck, accounting for 48.9% of all failures. We also document a new finding: geopolitical content filters in Chinese models refuse legitimate financial questions (0.08% of attempts), sometimes without clear reason, and the refusal behavior depends on the access route as much as on the model. The complete dataset and evaluation framework are publicly available. 

---
# RAG-Based Auto-Configuration for Industrial Fieldbus Devices 

**Authors**: Aadil Gani Ganie, Saad Ezzini, Naveed Farooz Marazi  

**Link**: [PDF](https://arxiv.org/pdf/2608.08618)  

**Abstract**: Industrial device commissioning requires engineers to manually extract hundreds of protocol-specific parameters from heterogeneous PDF manuals and transcribe them into supervisory control systems, a time-intensive, error-prone workflow. This paper presents SysName, a production-oriented pipeline that automates device configuration end-to-end for Modbus RTU, OPC-UA, Profibus DP, and CANopen. It builds a hybrid dense-sparse retrieval index augmented by an ontology graph derived from ECLASS, AAS, and SOSA/SSN, using a BGE-M3 encoder with a cross-encoder reranker to surface relevant manual passages. A local LLM (T=0.1) generates ontology-aligned JSON-LD configurations via protocol-specific prompts and a four-step repair pipeline. A two-stage abstention gate, combining a reranker-score threshold and an IRI resolution ratio, blocks unsafe LLM invocations and filters low-coverage configurations before SHACL validation. On a gold set of 28 field-level queries, the hybrid retriever reaches 0.96 HitRate@10, and the reranker raises MRR@10 from 0.56 to 0.63 with perfect score separation for abstention. The generator attains field-level F1=0.87 with exact match on 9 of 12 runs. End-to-end runs on an H100 GPU complete in 2.6-6.6s per device with zero unsafe writes and zero silent failures on a five-device benchmark; every unsuccessful run is flagged by abstention or deployment verification. Component-wise evaluation localises the single systematic failure to OPC-UA generation, invisible to end-to-end metrics alone. A case study commissions a physics-simulated Universal Robots UR5e robot from unmodified vendor documentation (254-page manual, 8-page register list, 496 chunks), reaching field-level F1=1.0 over three runs with read-back and joint-consistency verification. An ablation study and comparison with five industrial-LLM systems complete the analysis. 

---
# Understanding Calibration and Truncation Error Propagation in Training-Free Low-Rank Compression for LLMs 

**Authors**: Mohanad Odema, Gabrielle De Micheli, Dayin Gou, Nilesh Malpeddi, Prathamesh Vaste, Jacob Song  

**Link**: [PDF](https://arxiv.org/pdf/2608.08506)  

**Abstract**: Training-free low-rank compression frameworks have been gaining prominence for LLM compression given their effectiveness in reducing model parameter count while maintaining task-level accuracy. However, existing SOTA frameworks share two key limitations: (1) residual errors in calibration data activations accumulate across layers during compression, causing misalignment between representations simulated at compression time and those experienced at inference; (2) the assumption that layer importance distribution is preserved post-compression does not hold. Together, these two effects introduce misalignment in the compression process in relation to the deployed model. We study these effects and propose a simple, training-free methodology compatible with existing frameworks to mitigate them, comprising: (1) Layer-by-Layer Compression with Calibration Correction; (2) Iterative Compression with Rank Allocation Correction. Implemented atop an existing SOTA decomposition framework, and evaluated on Llama and Qwen3 models across various benchmarks and compression rates, our approach demonstrates up to ~1-2.5 accuracy point improvements over per-weight and joint decomposition baselines on zero-shot tasks. 

---
# MathShikkha: A Controlled Study of Answer-Only and Chain-of-Thought Supervision for Bangla Mathematical Reasoning in Small Language Models 

**Authors**: Rahma Simin Ali, Jawad Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2608.08503)  

**Abstract**: Mathematical reasoning remains challenging in low-resource languages such as Bangla. We study whether teacher-generated Bangla Chain-of-Thought (CoT) supervision provides benefits beyond ordinary supervised fine-tuning. We construct \textsc{MathShikkha}, a Bangla mathematical reasoning dataset with GPT-5.4-generated rationales, and fine-tune four 4B--7B student models under a matched protocol in which answer-only and CoT conditions share data splits, response-only loss masking, decoding, and scoring, differing only in the training target. In-domain, CoT provides no significant improvement over answer-only fine-tuning for three stronger backbones (paired bootstrap 95\% CIs include zero; exact McNemar $p \geq 0.17$), despite generating 15--52$\times$ more tokens, but significantly improves the weaker 4B model by 18.56 points ($p < 0.0001$). On the larger, contamination-audited BanglaMATH benchmark, this pattern reverses: CoT significantly outperforms answer-only supervision for all four models by 20.1--28.1 points (all $p < 0.0001$). Answer-only fine-tuning also reduces out-of-domain accuracy below the base model for three models, whereas CoT preserves or improves it for all four. A human study with two co-author annotators, external-expert adjudication, and Cohen's $\kappa = 0.76$--$1.00$ finds no significant CoT improvement over the base model on reasoning-content criteria; instead, its measurable effect is target-language adherence and producing inspectable reasoning. Overall, rationale supervision's value depends on backbone capability and distribution shift: in this setting, its main benefits are Bangla adherence, auditable reasoning, and out-of-domain robustness rather than improved in-domain reasoning validity. 

---
# HoloAegis: Frozen Representation, Topological Inference: Minimally Parametric Safety Manifolds for Zero-Shot LLM Guardrails 

**Authors**: Tak Ho Alex Li, Kaijie Liu, Lik-Hang Lee, Kin Chung Ho, Ping Shum, Michael K. Ng  

**Link**: [PDF](https://arxiv.org/pdf/2608.08485)  

**Abstract**: Current LLM safety guardrails face a fundamental tension: fine-tuning distorts pre-trained representations while generative judges incur prohibitive inference costs. We challenge the prevailing paradigm by asking: can safety be achieved through pure geometric reasoning over frozen semantic representations? We present HoloAegis, a minimally parametric topological inference framework that decouples representation from reasoning. We term our approach minimally parametric because the only free parameters are the anchor count K and the temperature tau, both fixed after construction and requiring no gradient-based training. An un-fine-tuned encoder maps text to a unit sphere, after which all decisions are purely geometric. We formalize safety evaluation as a Gibbs-Boltzmann Free Energy computation over a pre-computed System Topology Anchor Bank, and we introduce Dual Time-Scale Exponential Moving Averages to detect progressive multi-turn semantic drift. Our key theoretical insight is a Topological Boundary Stability Conjecture: we provide theoretical motivation and strong empirical evidence that sparse anchor centroids stabilize the decision boundary against high-frequency lexical perturbations far better than full vector space methods. Evaluated across 8 benchmarks, HoloAegis achieves state-of-the-art accuracy (1.0000 AUC on AuthenHallu, 0.9802 on HarmBench) with sub-millisecond latency, zero cold-start data, and cross-lingual transfer (0.9758 AUC on Chinese CHIFRAUD). 

---
# LLM within MCP Matters: Measuring Inefficient Resource Utilization Driven by LLMs 

**Authors**: Minhan Cho, Soyoung Park, Kihyeon Jeong, Byeongkyu Jeon, Daejin Choi, Jinyoung Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.08467)  

**Abstract**: The Model Context Protocol (MCP) standardizes how servers expose data and tools to Large Language Models (LLMs). A common server design embeds frequently used reference data, such as identifier lookup tables, directly in the server instructions: the system-prompt text a server hands to the host application. When a query concerns an entry of the embedded table, the model can act on it immediately instead of re-discovering the same information through a search tool. We test whether client LLMs actually consume such instruction-embedded data, reporting a 54,000-trial study across 24 LLMs (9 Claude, 6 Gemini, 9 GPT) on a production legal-information MCP server. A diagnostic condition that removes the competing search tool shows that failures are dominated by behavioral preference rather than missing capability. With search unavailable, 23 of 24 models read the embedded data reliably (hit ratio at least 98%); with a search tool merely present, 9 models drop below 15%. A 2^3 factorial analysis of three instruction-level interventions reveals strong interaction effects: combining all three restores at least 86% for 20 of 24 models, but individual interventions can backfire for specific model families. Per-server prompt engineering is therefore a workaround rather than a fix; we argue that MCP host applications should provide an explicit mechanism that places server instructions ahead of tool selection in the client LLM's deliberation. 

---
# CAP: A Scalable Benchmark for Evaluating Cross-Site Browser Agents with Complex Actions and Perception 

**Authors**: Zejun Xu, Taiyi Chen, Jin Li, Yongtong Gu, Qi Cheng, Aixuan Lv, Shuai Zhu, Pengfei Zhu, Kaichen Yang, Boyu Sun, Yixian Yang, Mulong Xie, Xin Liu, Dagang Li, Xiaoteng Ma, Hongru Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.08392)  

**Abstract**: Large language models are increasingly deployed as autonomous agents that interact with the web through browsers. While recent progress has been driven by benchmarks that evaluate end-to-end task success, these evaluations largely overlook two fundamental sources of difficulty in real web browsing: complex actions over rich user interfaces and visual perception of dynamically rendered content, especially in workflows that span multiple websites. We introduce CAP, a scalable benchmark for evaluating browser agents on cross-site, human-like web tasks that require non-trivial UI interactions and visual understanding. Specifically, we adopt a decomposition-and-recomposition pipeline that first abstracts each website into a structured site card capturing user-facing functions, complex execution operations, and perceptual requirements, and then recomposes these components into realistic cross-site workflows. Each task is therefore grounded in multiple specific operations on each website, enabling fine-grained diagnosis. Built on this framework, we construct 420 tasks across 108 real-world websites and 24 domains under careful quality control. Experiments on state-of-the-art browser agents using our verifiable agent-as-a-judge evaluation framework show low success rates and reveal that perception-heavy interactions remain a major bottleneck, exposing substantial gaps between current agents and real-world web browsing demands. 

---
# Mitigating Over-Personalization in LLMs via Structured Memory 

**Authors**: Hakeem Hannoon, Andrew Zhao, Mihir Narayan, Sharvin Goyal, Ivaxi Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2608.08300)  

**Abstract**: Conversational assistants increasingly rely on persistent long-term memory to personalize responses across sessions. However, when stored user information is reintroduced into the model context, it can also influence responses in inappropriate or unrelated settings. We study two such failure modes in memory-augmented LLMs: cross-domain leakage, where memories from one life domain affect responses in another, and memory-induced sycophancy, where stored user beliefs make models more likely to agree with the user rather than respond truthfully. We apply a simple inference-time modification to how memories are presented to the model, without changing the model or the memory contents. Across seven models on PersistBench, we compare the commonly used all-in context format, where memories are injected as an unstructured list, with structured formats that partition memories by domain. This simple modification consistently reduces cross-domain leakage while preserving utility, with our strongest method reducing leakage by $8.8\%$ on average relative to the baseline. 

---
# Learning from Environmental Feedback: Credit Assignment across Multiple Timescales for Agentic Reinforcement Learning 

**Authors**: Yifu Huo, Shunjie Xing, Chenglong Wang, Peinan Feng, Qiaozhi He, Yan Ding, Anxiang Ma, Yuxin Gao, Tongran Liu, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08255)  

**Abstract**: Agentic reinforcement learning (RL) often suffers from delayed and sparse rewards in real-world environments. A promising solution to this challenge is credit assignment, which aims to decompose trajectory-level rewards and provide more fine-grained supervision for intermediate decisions. However, existing credit assignment approaches ignore the rich process information naturally generated during environment interaction, e.g., interaction history. We argue that such information provides valuable supervision for identifying the contribution of individual actions. To this end, we propose Environmental Feedback-based Credit Assignment (EFCA), a multi-timescale credit assignment approach for long-horizon agentic RL. EFCA complements the long-term outcome signal with two environment-grounded process signals: a short-term feedback signal that captures the immediate effect of the current action and a medium-term state-history signal that identifies ineffective patterns from recent interactions. Both signals are directly extracted from environment feedback and integrated through a return reweighting mechanism. Experiments on ALFWorld and WebShop demonstrate that EFCA consistently improves both task success and task quality over strong baselines, highlighting the effectiveness of environment-grounded multi-timescale credit assignment for long-horizon agentic RL. 

---
# The Replay Gap: Static Evaluation of Model Switching in LLM Agents Scores the Wrong World 

**Authors**: Ashritha Gonuguntla  

**Link**: [PDF](https://arxiv.org/pdf/2608.08239)  

**Abstract**: LLM routers promise efficiency by matching each request to the cheapest adequate model, and are increasingly applied per step inside multi-step agents. Yet agentic routers are evaluated like single-turn routers: by replaying logged trajectories and substituting another model's recorded outputs, assuming the rest of the trajectory is unaffected. We test this assumption with branching rollouts: we fork live SWE-bench agent trajectories at controlled points, rebuild the environment, continue each fork with a different model, and compare against same-model control forks that isolate sampling and replay noise. Across six paired runs (~900 rollouts), swaps exceed their matched control floors by +0.25 to +0.66 normalized edit distance (multiplicity-corrected CIs exclude zero), rewriting 61-94% of post-fork actions; 74-77% of early swaps diverge at the first post-fork action, versus 6-35% of controls, leaving only 3% of replayed states valid. Divergence decreases with fork depth in both directions. All five outcome flips we observe occur in swap arms, upgrades rescuing unsolved instances and a downgrade losing the sole solve, and zero occur across 359 control forks. Scoring these same swaps with a log-stitching replay evaluator, replay mispredicts every success-relevant outcome call and predicts patches with 0.00-0.11 similarity to reality. Auditing the noise floor, temperature-0 "determinism" is configuration-dependent: FP8-served controls diverge on over 90% of forks while AWQ-served ones remain near-identical; and under tight budgets the stronger model more often exhausts its steps without submitting. Replay-based benchmarks score the wrong world for agentic routing; we release our harness and all trajectories. 

---
# SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems 

**Authors**: Muhammad Faizan Raza, Shuo, Yang, Satish Mahadevan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08237)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems in production operate under strict service level objectives (SLOs) on tail latency and infrastructure cost. However, standard retrieval pipelines rely on fixed retrieval budgets that ignore query difficulty, over-retrieving for easy queries and under-serving hard ones, forcing operators to trade answer quality against SLO compliance. This paper proposes SAGE, a learned SLO-aware adaptive retrieval policy that dynamically selects the number of passages k per query. SAGE uses lightweight features derived from initial retrieval (e.g., score distributions, rank gaps, lexical signals) and is trained offline via imitation learning from an oracle that approximates optimal latency-quality trade-offs. At inference, it adds no LLM calls and minimal overhead. On Natural Questions, under a 5s P95 latency SLO, SAGE achieves 95% SLO compliance versus 30% for the best static baseline (k=20), reduces P95 latency by 36% and retrieval cost by 51% with only 2 percentage points Exact Match (EM) loss. A single policy trained on Natural Questions generalizes across HotpotQA, UnSeenTimeQA, and four LLM families (Llama, Qwen, Mistral, Gemma), consistently yielding +45-52 point SLO improvements without quality degradation. 

---
# LatticeMind: A Conflict-Aware Memory Primitive for Multi-Agent Systems 

**Authors**: Heng Zhou, Lian Zhang, Yutao Fan, Tiancheng He, Siki Chen, Hejia Geng, Philip Torr, Zhenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2608.08236)  

**Abstract**: Multi-agent LLM systems often fail not for lack of candidate answers, but because they have no persistent mechanism for deciding which incompatible claim should currently be trusted. Majority vote, debate, and judge-based selection choose an output without recording which claim wins, which is contested, or why a later update supersedes it. We present \term{LatticeMind}, a conflict-aware structured memory that handles contradiction at write time. It maintains explicit item status, applies cheap symbolic conflict checks, and invokes LLM reconciliation only for unresolved semantic cases. On a label-blind ConflictBank evaluation that removes source-name hints, LatticeMind reaches 0.97 accuracy versus 0.61 for the strongest aggregation baseline, with the gap significant at $p<10^{-6}$ by paired McNemar test. Ablations show that removing the checker or the reconciler costs 12 to 14 points. On four secondary planning benchmarks the picture is mixed: LatticeMind beats naive merge on three of four, but does not replace deliberation methods on tasks rewarding iterative search. 

---
# Harmful Content Is Not Enough: Continuation Framing Moderates In-Context Emergent Misalignment 

**Authors**: Peiyang Liu, Xi Wang, Ziqiang Cui, Di Liang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2608.08212)  

**Abstract**: In-context learning (ICL) can induce emergent misalignment (EM), where narrow misaligned examples alter answers to unrelated questions. Existing prompts, however, conflate harmful-text exposure with an invitation to continue assistant behavior. We hold harmful answers fixed while varying their delivery as demonstrations, evidence, assistant history, or tool output. Across ten independently sampled contexts, demonstration framing raises broad EM by $30$--$32$ percentage points on a susceptible Gemini model; the gap survives domain exclusion, semantic clustering, unseen questions, and four prompt templates. Format and length-matched controls show that harmful content is necessary but insufficient. A role times continuation factorial further reveals model-dependent provenance effects: Gemini follows both assistant and tool histories, whereas Grok largely resists tool-framed continuation. Several other frontier and open-weight models show no gap. Blinded human audits confirm every main contrast and show that the model judge underestimates active-condition failures. Thus continuation framing is a strong, model-dependent moderator of ICL-EM, not a universal consequence of harmful context. 

---
# Quantization Degradation in Large Language Models: A Signal-Noise Perspective 

**Authors**: Chenxi Zhou, Pengfei Cao, Jinyu Ye, Bohan Yu, Haida Yu, Jiang Li, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.08188)  

**Abstract**: Post-training quantization reduces the deployment cost of large language models, yet how severely a quantized model degrades is not determined by bit-width alone. We systematically study weight-only post-training quantization across bit-widths, quantization methods, model scales and downstream tasks on multiple model families. We observe that such degradation varies substantially across these factors: 4-bit quantization usually preserves performance, 2-bit often causes broad degradation, and at 3-bit, degradation becomes apparent but varies markedly with task type, quantization method and model scale. To explain this variability, we use the signal-to-noise ratio (SNR) to measure how strongly quantization perturbs full-precision representations. We trace degradation back to two linked processes: how quantization errors arise within individual modules, and how they accumulate across layers. First, a source SNR decomposition shows that newly introduced errors depend on three factors: the magnitude of the weight error, the strength of the task-specific signal, and how strongly the quantization error aligns with task-specific activations. Different factors affect these components in distinct ways. Second, a cross-layer propagation analysis shows that these errors can be attenuated, preserved, or amplified as they pass across layers, and that larger models benefit from weaker error amplification. Together, these results establish that quantization degradation is governed by how errors are introduced at the source and how they accumulate across the network. 

---
# DS@GT ARC at Touché: Large Language Models for Retrieval-Augmented Debate 

**Authors**: Anthony Miyaguchi, Conor Johnston  

**Link**: [PDF](https://arxiv.org/pdf/2608.08143)  

**Abstract**: We extend the DS@GT ARC working-note submission to the Touché 2025 Retrieval-Augmented Debate task. The task has two subtasks: generating the next utterance in a simulated debate, and evaluating debate responses according to the Gricean maxims of Quantity, Quality, Relation, and Manner. The DS@GT ARC submission consisted of six leading LLMs from three providers through a retrieval-augmented prompting pipeline. We summarize the results from the working paper and explore whether multi-LLM evaluator agreement is a reliable proxy for official evaluation performance. The analysis shows that frontier LLM systems are strong response generators, and as evaluators they agree strongly within model families. However this consensus does not reliably track the official evaluation target, with the largest gap on the Quality maxim. The accompanying source code for this paper is located at this https URL and this https URL. 

---
# Accurate Ensembles, Fragile Narratives: Multi-Scale Stacking and a Fidelity Audit of LLM-Generated Explanations for Credit Risk 

**Authors**: Gregorius Reynaldi Pratama, Kuo-Kun Tseng  

**Link**: [PDF](https://arxiv.org/pdf/2608.08126)  

**Abstract**: Credit scoring increasingly relies on models whose decision logic cannot be read off their parameters, in tension with supervisory expectations that adverse decisions be explainable. A common proposal closes that gap with a language model: compute feature attributions, hand them to an LLM, and let it write the rationale. We build such a system end to end and test whether the second half of the promise holds. The predictive component is a multi-scale stacking ensemble fusing four differently regularised gradient-boosting learners with a residual network through a neural meta-learner trained on out-of-fold predictions. On a public 32,581-application credit dataset it reaches test ROC-AUC 0.9539 (95% CI [0.9462, 0.9616]) and PR-AUC 0.9137, beating the best single model by Delta-AUC = 0.0143 (p = 0.016 under a conservative independence assumption). Our central finding is asymmetric. The ranking gain is real but operationally small: at the F1-optimal threshold the ensemble avoids only six additional missed defaults out of 1,422 against a tuned random forest, cutting cost-weighted loss by under 2%. The narrative layer fails in a way prompt engineering alone does not fix. In an audited case the model named three factors as risk-increasing that the supplied attributions scored as risk-reducing, omitted the dominant driver, and introduced a feature never given to it. We trace this to properties we measure rather than assume: SHAP and LIME agree on which features matter (overlap@10 = 0.80) but not on their order (tau = 0.43, p = 0.18), and the attribution sign for the model's most sensitive input is near a coin flip across applicants (modal-sign share 0.53). Calibration (ECS = 0.117) and perturbation stability (DPD = 0.078) both fall short of our own thresholds. Constrained prompting is necessary but not sufficient: grounding must be verified after generation, not assumed. 

---
# Decided Upstream, Written Late: Locating and Pricing the Cross-Lingual Refusal Circuit of a Multilingual MoE 

**Authors**: Ramakrishna P. Kompella, Aadit Mahajan  

**Link**: [PDF](https://arxiv.org/pdf/2608.08032)  

**Abstract**: Safety alignment in multilingual models is uneven: a model that reliably refuses a harmful request in English will often comply with the same request in a lower-resource language. We trace this gap mechanistically in sarvam, an Indic-multilingual mixture-of-experts reasoning model, and find it is not a failure to detect harm. Harm is encoded as an internal direction that is nearly language-invariant in mid-network (English-vs-Indic cosine ${\approx}0.9$ at $L11$), and steering that direction upstream causally controls refusal. But the detection direction is orthogonal to the change that actually writes the refusal, which is late and assembled over the course of generation rather than read off in a single forward pass. We attribute the write to a specific, localizable circuit, a mixture-of-experts writer held in check by an attention opposer and price every way of intervening on it: damping the opposer is cheap and effective, amplifying the writer is a cost wall, and surgical edits to the responsible heads do nothing. The circuit's organization, and the gradient method that exposes it, recur in a second, unrelated MoE model, while the lever's strength is architecture-specific. The result is a cost-measured map of where a multilingual safety repair can land, and what it costs 

---
# The Voiceprint Fallacy: Why Voices Are Not Unique Biometric Imprints 

**Authors**: Tianle Yang, Cuiling Zhang, Chengzhe Sun, Siwei Lyu, Phil Rose  

**Link**: [PDF](https://arxiv.org/pdf/2608.07980)  

**Abstract**: In recent years, the term voiceprint has regained attention, particularly in technological applications and policy-making contexts, often carrying the assumption that a person's voice constitutes a stable and unique biometric trace analogous to a fingerprint. Yet this conception has been repeatedly criticized and rejected by forensic voice experts throughout the decades since its introduction. Although voices undoubtedly contain speaker-related information, this simplified conception obscures the highly dynamic and context-dependent nature of speech. This article revisits the voiceprint fallacy and reconsiders what can count as evidence of speaker identity by reviewing the historical development of voiceprint identification, evidence on human voice variability, developments in forensic voice comparison, research on human and automatic speaker recognition, and the recent challenge posed by deepfake speech to speaker identity. We point out that the voiceprint metaphor and its underlying implications are scientifically misleading because they transform a probabilistic source of speaker information into an imagined stable object of identity. To avoid treating voices as imprint-like traces, we recommend that voice evidence be interpreted through validated and calibrated probabilistic frameworks that explicitly account for variability, uncertainty, and alternative explanations. 

---
# EvoTrustRAG: Evolution-Aware Conflict Attribution and Evidence Handling for Reliable Retrieval-Augmented Generation 

**Authors**: Xi Nie, Hongwei Li, Shenghao Wu, Wenshu Fan, Qiyang Song, Wenbo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.07933)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves the factuality of large language models with external knowledge, yet conflicting evidence remains a fundamental challenge in dynamic and adversarial environments. Existing approaches often treat conflicts as static inconsistencies and select more reliable knowledge, overlooking that the same conflict may arise from legitimate knowledge evolution, malicious manipulation, or unresolved uncertainty. We formulate conflict origin attribution as a new problem in RAG: identifying which explanation of conflicting evidence is supported by observable context rather than simply which fact should be trusted. We propose EvoTrustRAG, a training-free framework for evolution-aware conflict attribution and evidence handling before answer generation. EvoTrustRAG represents span-grounded retrieved facts as a conflict evidence graph, evaluates grounded evolution and directional intervention hypotheses using temporal relations, support structure, and auxiliary consistency, and projects local decisions onto a globally consistent explanation of each conflict group. The attribution determines whether earlier and later states are preserved as temporal knowledge, an intervention candidate is separated from the primary context, or an unresolved conflict remains visible to the generator. Unlike provenance-based approaches focused on post-hoc analysis, EvoTrustRAG determines during inference whether conflicting evidence follows plausible knowledge evolution, exhibits intervention-like support, or cannot be reliably attributed. Experiments show that EvoTrustRAG achieves 81.4% average accuracy on benchmark-native conflict settings, improves attribution macro-F1 from 72.2% to 79.1% over the strongest baseline, and reduces the error rate under the strongest coordinated attack from 31.2% to 16.0%. 

---
# Spectral Outliers Reveal Dominant Learned Structure in Transformer Attention 

**Authors**: Kasun Dewage, Marianna Pensky, Suranadi De Silva, T. H. Bandara  

**Link**: [PDF](https://arxiv.org/pdf/2608.07921)  

**Abstract**: We apply Marchenko-Pastur (MP) random matrix theory to pre-trained attention weights in order to separate each projection matrix into a random-like bulk and a set of spectral outliers. We validate this decomposition causally: zeroing the MP-identified outliers (signal) in Mistral-7B drives HellaSwag, MMLU, and PIQA close to random-chance performance, whereas zeroing a count-matched subset of bulk singular values causes smaller but non-negligible degradation. Across 11 pre-trained transformers we identify five recurring patterns: spectral outliers encode a dominant component of the learned structure; Q projections carry the most outliers; V projections under grouped-query attention lack a clean signal/noise separation; entry-level outliers form structured row-bands in Q and column-bands in O; and specific residual-stream dimensions persist as band outliers across layers in K and O. We close by outlining how these observations could inform parameter-efficient fine-tuning and structured pruning. 

---
# Vision-Language Grounding as Bidirectional Concept Correspondence 

**Authors**: Jieyu Zhang, Ziqi Gao, Luke Zettlemoyer, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2608.07886)  

**Abstract**: Vision-language grounding connects language to visual content, yet most existing formulations reduce grounding to a unidirectional localization problem: given a prespecified text phrase or category name, identify the corresponding image region. This setup assumes that the relevant linguistic unit is already known, overlooking a more basic challenge in grounded communication: determining which parts of the text are visually referential and how they correspond to entities in the image. We formulate grounding as $\textit{bidirectional concept correspondence}$ over an image-text pair. Given an image and its paired text, the goal is to recover all correspondences between visually referential text spans and instance-level image segments, without assuming that the relevant text spans are provided. This formulation unifies common grounding tasks, including phrase grounding, referring expression grounding, and open-vocabulary detection, by treating text segmentation, image segmentation, and cross-modal alignment as a single correspondence prediction problem. To address this task, we introduce $\textbf{ConCor-1}$, a grounding model built on top of a pretrained vision-language model. It uses learnable $\textit{bridge tokens}$ to represent candidate image-text correspondences and predicts, for each token, a text mask, an image mask, and a correspondence presence score. To train and evaluate this task, we convert diverse grounding and segmentation datasets into a unified correspondence format. Experiments show that $\textbf{ConCor-1}$ consistently outperforms baselines, improving correspondence F1 by 48% on the long-caption dataset and by 29% on zero-shot LVIS, where the large category list serves as the text input. 

---
# GRACE: LLM-Grounded Semantic Metric Spaces for Scalable Mixed-Data Clustering 

**Authors**: Zihua Yang, Zhencheng Xie, Junyang Chen, Liang Xie, Yiqun Zhang, Mengke Li, Yang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2608.07881)  

**Abstract**: Clustering mixed tabular data requires a unified metric space to bridge the inherent heterogeneity between continuous numerical measurements and discrete categorical symbols. Traditionally, algorithms rely entirely on dataset-internal statistics to estimate categorical relationships, which confines the learned metric to empirical co-occurrences and ignores conceptually obvious yet statistically unobserved affinities. Although LLMs offer external world knowledge, applying their text-centric reasoning to highly abstract tabular concepts presents significant challenges. Bridging this modality gap to construct a semantically complete metric typically requires embedding LLMs into iterative metric learning loops to dynamically optimize cross-modality representations. This incurs intractable computational overhead, forcing a compromise between semantic enrichment and scalability. Therefore, we propose GRACE, an LLM-grounded framework for scalable mixed-data clustering. GRACE shifts semantic acquisition to the attribute-value level via a multi-perspective LLM querying strategy, mapping heterogeneous values into knowledge-informed descriptions. Crucially, this one-shot grounding extracts general-purpose semantic representations that embed heterogeneous attributes into a unified space, decoupling expensive LLM invocation from iterative optimization. Furthermore, GRACE cross-validates these external semantics against dataset-internal statistical evidence to ensure alignment with the dataset-specific cluster structure. Ultimately, GRACE matches the scalability of conventional statistics-driven baselines while achieving superior clustering accuracy and conceptual interpretability over 11 competing methods. The source code is available at this https URL 

---
# TEMPER: Tensorized Efficient Manifold-constrained Parameterization for Expressive Residual Routing 

**Authors**: Yuxuan Gu, Wuyang Zhou, Huijun Xing, Danilo Mandic  

**Link**: [PDF](https://arxiv.org/pdf/2608.07851)  

**Abstract**: Residual connections rely on a static residual pathway, and are essential for training deep neural networks. Hyper-connections (HC) increase the expressivity of residual routing by incorporating multiple residual streams and learning dynamic information flow, while manifold-constrained (mHC) variants stabilize training through doubly stochastic residual mixing. However, a generator-level bottleneck remains in existing methods: they use dense, unstructured generators for pre-branch aggregation, residual mixing, and post-branch redistribution, which results in parameter count growing rapidly with the number of streams. To address this issue, we propose \underline{\textbf{T}}ensorized \underline{\textbf{E}}fficient \underline{\textbf{M}}anifold-constrained \underline{\textbf{P}}arameterization for \underline{\textbf{E}}xpressive Residual \underline{\textbf{R}}outing (\textbf{TEMPER}), which represents these generators as multi-way tensors over the input-stream, feature, and output-stream modes, and parameterizes them using tensor networks. Such a structured low-rank formulation is shown to preserve token-dependent manifold-constrained routing interface while substantially reducing parameter growth. It also promotes interpretability and intuition, as: i) tensor ranks control the dimensionality of the learned routing subspace, with full ranks recovering dense routing; while ii) the generator approximation errors bound differences in routing logits and, consequently, in the routed-block outputs. Comprehensive experiments show that TEMPER matches or outperforms existing methods across language modeling and commonsense reasoning tasks, while requiring substantially fewer additional parameters. At eight residual streams, TEMPER achieves the best CORE score while using about $84\%$ fewer additional parameters than mHC, thus showing a stronger performance-parameter efficiency trade-off. 

---
# From token probabilities to calibrated confidence: An empirical study of mathematical question answering 

**Authors**: Avery Ma, Lorne Schell, Vin Bhaskara, Leila Pishdad  

**Link**: [PDF](https://arxiv.org/pdf/2608.07827)  

**Abstract**: Confidence estimation for large language models (LLMs) aims to estimate the probability that a generated answer is correct, while calibration aligns these estimates with empirical accuracy. Prior work has shown that token probabilities are often overconfident, we investigate whether these readily available signals can nevertheless provide well-calibrated confidence estimation for mathematical question answering. We compare single-pass estimators, which reuse token probabilities from the original generation, with multi-pass estimators, which obtain additional confidence signals through verification or stochastic forward passes. While individual token probabilities can be highly saturated, we find that aggregating token probabilities over the full sequence captures small but consistent differences between correct and incorrect generations, yielding more informative confidence estimates. Multi-pass methods can yield calibrated confidence estimates. We study two such approaches: self-verification through re-prompting, including a lower-cost in-situ variant, and Monte Carlo Dropout, which derives confidence from variation across stochastic forward passes. We further evaluate two post-hoc calibration methods, Platt scaling and isotonic regression, both of which substantially reduce in-domain calibration error. However, their data efficiency varies with dataset difficulty, and the calibration mappings often transfer asymmetrically across datasets and models. 

---
# CosmosAlign: Adapting a World Foundation Model for Generative Traffic Video Forecasting 

**Authors**: Quang Minh Dinh, Tuan Kiet Doan  

**Link**: [PDF](https://arxiv.org/pdf/2608.07693)  

**Abstract**: Generative traffic video forecasting aims to synthesize long-horizon, temporally coherent future videos of traffic scenes from a short observation history and textual descriptions. In this paper, we present CosmosAlign, a generative traffic video forecasting framework built upon the pretrained Cosmos3-Nano world foundation model. Our approach is motivated by the observation that successfully adapting large pretrained world models to downstream forecasting tasks depends primarily on distribution alignment rather than increased model capacity. To this end, we propose a two-stage LoRA adaptation strategy that first aligns the conditioning-mode distribution with the target forecasting task, and then aligns the training captions with the model's native structured prompting interface through an LLM-based re-captioning pipeline. During inference, we further improve prediction quality using a fully training-free procedure consisting of consensus-based medoid sample selection and motion-adaptive blending of static scene regions. CosmosAlign achieves a final score of 76.49 on the AI City Challenge 2026 Track 5 benchmark, ranking first on the final leaderboard. Our code is publicly available at this https URL. 

---
# IntelliAudit: Using Large Language Models to Evaluate Audit Controls 

**Authors**: Allison Wilson, Sina Moradi Sabet, Diar Shakimov, Panteha Shahrivar, Mohammad Reza Bagheri, Dean Konenkamp, Mohammad A. Tayebi  

**Link**: [PDF](https://arxiv.org/pdf/2608.07688)  

**Abstract**: IT audits require auditors to judge whether heterogeneous organizational evidence satisfies semantic security and compliance controls. This judgment is difficult to automate because relevant evidence is distributed across policies, records, spreadsheets, and operational artifacts, and because audit conclusions depend on evidentiary sufficiency rather than keyword matching. We present IntelliAudit, a retrieval-grounded multi-agent system for IT audit evidence evaluation. Given a control and an evidence corpus, IntelliAudit retrieves relevant artifacts, generates an evidence-grounded assessment, challenges adverse findings, adjudicates disagreements, and produces an auditor-facing recommendation with cited evidence, rationale, missing-evidence analysis, and remediation guidance. We instantiate IntelliAudit on ISO/IEC 27001 and evaluate it across multiple simulated organizations using expert auditor review and audit-readiness user feedback. The evaluation shows that IntelliAudit can support control interpretation, evidence-grounded reasoning, and audit-preparation workflows, while also revealing the importance of human oversight for calibrating sufficiency judgments and correcting overly permissive recommendations. These results suggest that retrieval-grounded multi-agent systems can assist audit evidence review, but should remain decision-support tools rather than autonomous certification systems. 

---
# Keep It Simple: Multi-Key Episodic Memory Retrieval for Ultra-Long Video Understanding 

**Authors**: Yeeun Choi, Youngbeom Yoo, Joon-Young Lee, Hyolim Kang, Seon Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.07663)  

**Abstract**: When videos extend from hours to days, directly processing them end-to-end becomes impractical for current Multi-modal Large Language Models (MLLMs). This ultra-long setting necessitates a two-stage paradigm: query-agnostic memory construction followed by retrieval-based inference. Prior work invests in complex memory construction to pre-model high-level relations in videos, despite not knowing the downstream query at build time. We instead prioritize high-recall retrievability during memory building, and defer query-specific, high-level relation composition to inference time. To this end, we propose MERIT(Multi-key Episodic Retrieval with Inference-time Temporal expansion), a simple yet effective agentic framework for ultra-long video understanding. First, we formulate an episodic multi-key representation that enables precise retrieval of fine-grained memories through a simple key-matching mechanism. Second, we introduce a neighbor filtering mechanism to capture broader semantic context without the massive computational overhead of global memory construction. This is achieved by expanding the temporal scope exclusively around the retrieved segments at inference time. By leveraging simple key-matching with this on-demand temporal expansion, MERIT achieves state-of-the-art performance across three long-video benchmarks: EgoLifeQA, LVBench, and Video-MME (Long). 

---
# From Single Chatbots to Governed Agent Ecosystems: An Agentic AI Pattern Catalogue and Orchestration Framework for Mission-Critical Hospital Information Management Systems 

**Authors**: Manideep Dhar, Ritwik Singh, Sharat Chandra Kumar Manikonda  

**Link**: [PDF](https://arxiv.org/pdf/2608.07627)  

**Abstract**: Hospitals are racing to embed AI, while coping with the surge in adaptation of the technology in other industries, into the triage management, documentation, scheduling, and revenue-cycle workflows, yet most deployments remain as fragmented pilots that stall at the edge of production, exposing patients and institutions to operational fragility, ungoverned risk, and mounting technical debt. At the same time, the global AI-in-healthcare market is projected to exceed nearly USD 1 trillion by 2034, according to the report of Fortune Business Insights, amplifying the financial consequences of architectural missteps and failed scaling strategies. This research proposes a compliance-first Agentic AI pattern catalogue and orchestration framework, purposely built for HIMS, moving beyond the single LLM chatbots and towards a governed ecosystem of autonomous and semi-autonomous agents. The framework extends by adding (i) a taxonomy of Agentic roles, (ii) a formal risk-stratification model that maps each pattern to risk tiers, human-in-the-loop checkpoints, and governance hooks, and (iii) a unified orchestration runtime capable of coordinating multi-agent workflows across EHR/HIMS landscapes such as Epic, Cerner, and MEDITECH. Technically the framework combines vLLM-based inference, optimized paging memory, confidential computing, and MCP based on-premise deployment, enforcing end-to-end encryption and policy-as-code controls aligned with HIPAA, GDPR, the EU AI Act, India's DPDP and DISHA Acts, ISO 27001, ISO 27002, ISO 14971 and IEC 62304. We exhibit how the proposed architecture is capable and efficient to reduce the documentation time, integration effort, and AI pilot attrition while constricting the governance and auditability, offering hospital leaders and governing authorities an urgently needed blueprint to convert AI investment into sustainable clinical, operational, and financial ROI 

---
# DevIntent: How Much Does LLM-Generated Code Violate Developer Intent? 

**Authors**: Susana Haing, Natan Vidra, Spurthi Setty  

**Link**: [PDF](https://arxiv.org/pdf/2608.07614)  

**Abstract**: Code generated by LLMs can violate a developer's implicit intentions when given an ambiguous prompt, yet standard benchmarks measure only whether code passes its stated test. We introduce the Intent Violation Rate (IVR) and a 49-problem pilot benchmark derived from HumanEval+. Each problem strips implicit constraints from a clarified prompt and encodes them as hidden constraint tests. IVR measures the fraction of LLM-generated solutions that pass the stated (visible) tests yet fail hidden constraint tests that capture unstated intent. Evaluating Claude Sonnet 4.6 and OpenAI GPT 4.1, we find both pass over 92\% of stated tests yet violate intent in over half of problems (54.5\% and 63.5\%), following a systematic, bimodal pattern consistent across both models. Out findings indicate that pass rates overstate how well generated code reflects developer intent. 

---
# An evolutionary model of animats with VLM-based subjective evaluation 

**Authors**: Shota Miyazaki, Takaya Arita, Reiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2608.07537)  

**Abstract**: In this study, we propose a framework that incorporates subjective evaluations provided by a Vision-Language Model (VLM) into the fitness evaluation and selection processes of a genetic algorithm. As the target of evolution, we employ virtual soft robots with flexible morphologies and locomotion and present the VLM with sequence images representing the locomotion of two individuals. Selection is performed via pairwise comparisons based on subjective evaluation terms such as adorably and weirdly. The outcomes of these comparisons are used as selection pressure within the genetic algorithm, enabling the simultaneous evolution of morphology and locomotion. Experimental results demonstrate that subjective selection by the VLM accelerates population convergence compared to random selection, while also giving rise to distinctive morphologies and motions corresponding to each evaluation term. An auxiliary experiment with human participants further showed that, although individual pairwise choices only partly agreed with the VLM selections, the resulting morphological and locomotion tendencies were qualitatively similar and repeated human evaluations imposed noticeable fatigue. Moreover, the observation that similar evolutionary outcomes emerged across different evaluation terms suggests that the VLM does not apply these terms in a purely literal manner but instead decomposes them into multiple internal evaluation criteria when making judgments. This work visualizes the evolutionary process through which subjective linguistic expressions are mapped onto embodied phenotypes and provides a foundational framework for analyzing the structure of subjective judgment in VLMs. The proposed approach is expected to contribute to new developments in evolutionary computation and artificial life research based on subjective evaluation. 

---
# NL2SHACL-Bench: A Benchmark Suite for Natural Language to SHACL Translation 

**Authors**: Yuchen Zhou, Niels Bobet, Maribel Acosta  

**Link**: [PDF](https://arxiv.org/pdf/2608.07530)  

**Abstract**: SHACL is a core technology for validating the conformance of RDF knowledge graphs (KGs). Yet, authoring SHACL shapes requires technical expertise that most domain experts lack. Translating natural language requirements into SHACL (NL2SHACL) would lower this barrier. However, there is no dedicated benchmark for NL2SHACL, and evaluating generated shapes requires methods beyond string comparison, as semantically equivalent shapes can differ in serialisation and structure. To tackle these challenges, we present NL2SHACL-Bench, a benchmark suite for natural language to SHACL translation. Using NL2SHACL-Bench, we evaluate four state-of-the-art large language models (LLMs) for this task. Our results show that current LLMs are highly capable of generating syntactically valid SHACL, but still struggle to produce semantically equivalent constraints for complex logical and structural patterns. This indicates that NL2SHACL-Bench provides a meaningful basis for measuring advances in the NL2SHACL state of the art. 

---
# The Knowing-Saying Gap: When Probes See Errors that Confidence Misses 

**Authors**: Jyotin Goel, Ipshita Bandyopadhyay, Justin Shenk  

**Link**: [PDF](https://arxiv.org/pdf/2608.07528)  

**Abstract**: Linear probes detect corrupted context in language models with near-perfect accuracy, yet this does not translate into reliable failure prediction. The result is a dissociation with direct implications for deployment monitoring. Across multi-hop arithmetic chains, probes that detect corruption turn out to be uninformative about final answer correctness; models forced into structured confidence formats collapse to two values with indistinguishable error rates; and probe persistence across hops fails to separate correct from incorrect outcomes, refuting our pre-registered "persistence beats peak" hypothesis. This pattern of knowing but not saying generalises across model families including reasoning models. As a real-time monitor, probe-based interventions are sharply model and error-type dependent: branch-and-pick is net-positive across models and uniquely non-breaking on Llama-3.1-8B (4 rescued, 0 broken), while reprompt and replace-prior break correct traces at roughly the rate they rescue wrong ones. Probe-based monitoring is a necessary complement to verbalised confidence, but no single intervention dominates, and the deployable answer is model-aware, error-type-aware routing. 

---
# The Judge Knows When It Knows: Calibrated Abstention for LLM-Based A/B-Test Prediction 

**Authors**: Tyler Dooskin, Squoosh Technical Staff  

**Link**: [PDF](https://arxiv.org/pdf/2608.07517)  

**Abstract**: Can a multimodal LLM predict which version of a web page will win a real A/B test from screenshots alone? We report the most complete answer we are aware of, from six weeks of pre-registered experiments on real conversion tests: mostly no -- and the exceptions are identifiable in advance. On 330 real A/B tests a Gemini 3 Flash judge reaches Cohen's kappa = 0.14, but on the trustworthy (statistically significant) half of the labels the evidence is inconclusive (kappa = 0.11, CI includes zero). We show that 44% of the "ground-truth" labels in a leading CRO agency's catalog come from non-significant tests, and that the judge agrees more with the unreliable labels than the reliable ones -- a shared prior between labeler and model, not prediction. Every standard improvement lever (a 2.8x more expensive frontier model, prompt redesign, stimulus fidelity, change-type priors) fails its pre-registered gate. The judge's confident calls are different: a vote-margin gate isolates a subset (49% coverage) reaching kappa = 0.31 on significant labels. We measure the mechanism directly -- judges differing in model or prompt agree with each other at kappa = 0.74-0.88 while agreeing with real outcomes at only ~0.2, so a 16-vote panel carries about 2 effective independent votes -- and we reproduce it in humans: 15 CRO experts agree with each other (inter-rater kappa = 0.53) but score at chance against real outcomes (kappa ~ 0). Consensus, human or model, is reproducible, persuasive, and not evidence. We release our pre-registrations, locked gates, negative results, statistical harness, human responses, and a claims ledger in which every number carries an evidence tier. 

---
# How sensitive do we want AI to be? Socio-communicative competencies of large language models in healthcare 

**Authors**: Dorothee Amelung, Andrew M. Bean, Sabine C. Herpertz, Felix H. Krones, Guy Parsons, Adam Mahdi, Isabella Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2608.07511)  

**Abstract**: Background. Effective clinical practice relies heavily on the socio-communicative skills of medical professionals. Large language models (LLMs) have been proposed for tasks such as triaging patients, report drafting or translating medical jargon to support informed decision-making. These applications require both factual and social competence. This study evaluates dialogues between LLMs and participants to assess the current state of socio-communicative competencies displayed in LLM-generated texts.
Methods. We extracted a subset of extended dialogues from the HELP-Med dataset, comprising 1800 conversation transcripts of interactions between human participants seeking medical information and three different LLMs, GPT 4o, Llama 3 and Command R+. Two experts coded the transcripts for demonstrations of socio-communicative behaviours (non-hostility, sensitivity, structuring, non-intrusiveness) using the IC-MD instrument, originally designed to evaluate interactional competencies in medical student admissions.
Results. The LLMs in our study showed strength in non-hostility, mixed results in sensitivity and non-intrusiveness and performed poorly in structuring.
Conclusion. Current LLMs lack the consistent and reliable socio-communicative skills needed for safe and effective use as healthcare advisors. While existing frameworks for assessing interactional competencies may support the development of more socially responsive LLMs, they will require adaptation to account for the differences in desirable behaviour between humans and LLMs. 

---
# JaleesBench: Are AI Assistants Good Spiritual Company? 

**Authors**: M. Waleed Kadous, Benjamin Olsen  

**Link**: [PDF](https://arxiv.org/pdf/2608.07508)  

**Abstract**: Large language models are already advisors to millions of people of faith who bring them real decisions. The pressing question for a person of faith is not what a model knows or professes but what its counsel does to the person who receives it. We introduce JaleesBench, which measures whether an AI agent is a righteous companion, judged by the residue an exchange leaves on the user, in the manner of the perfume-seller and the blacksmith. It comprises 140 two-turn scenarios drawn from a classical compilation organized by virtue (Riyad al-Salihin), under six adversarial pressures and three framings, scored by two frontier judges against each scenario's own supporting texts. Across eight systems: (1) generic frontier models are only middling companions out of the box but a one-page guide makes them genuinely good ones, on par with the domain-tuned assistant: the frontier APIs climb from +0.28/+0.23 to a Guided +0.84-0.87, so most of the expert's edge is companionship instruction that fits in a prompt; (2) every system caves under relational pressure, insistence and personal appeal; (3) the domain-tuned assistant's advantage is overwhelmingly its retrieval-and-prompting layer, not its base model (+0.74 over the identical underlying model); and (4) it can be used to improve existing systems: guided by its diagnosis, a single steadfastness instruction lifts a deployed Islamic assistant from +0.48 to +0.84 (Faith unstated, after pressure), matching the best guided frontier systems while preserving first-response quality. The construct is faith-general; we instantiate it for Islam as the first of a planned cross-tradition family. Code, scenario bank, and rubric are open source (this http URL), with an interactive results browser at this http URL. 

---
# Evaluation of Motivational Interviewing Counsellors with Task-Aware Multi-Stage LLM-Based Simulated Clients 

**Authors**: Jiading Zhu, Xinyu Cindy Wang, Thomas Nguyen, Yan Qing Lee, Osnat C. Melamed, Peter Selby, Jonathan Rose  

**Link**: [PDF](https://arxiv.org/pdf/2608.07499)  

**Abstract**: The development and benchmarking of Large Language Model (LLM)-based Motivational Interviewing (MI) counsellors now often rely on LLM-based simulated clients. Prior work on simulated clients, however, has not aligned with the specific tasks fundamental to the MI therapy approach. A key task is evoking, in which the counsellor first elicits the client's ambivalence and then strengthens the client's motivation for change. We present Evoke-Sim, a task-aware, multi-stage LLM-based client simulation framework for evaluating MI counsellors in smoking cessation, designed specifically for the evoking MI task. Evoke-Sim employs structured client profiles, an evoking-specific three-stage conversation flow, and a reveal policy that regulates which client profile information might be disclosed at each stage. We show that compared to existing profile-grounded simulated clients, Evoke-Sim is better at differentiating levels of MI quality using task-aware evaluation metrics, while reducing non-grounded client statements and premature disclosure of client information, setting a higher standard for the evaluation of LLM-based MI counsellors. 

---
# EvalConvoLearn: An Open-Source Framework for Evaluating Grounded Learner Simulations in Tutoring Conversations 

**Authors**: Baptiste Moreau-Pernet  

**Link**: [PDF](https://arxiv.org/pdf/2608.07497)  

**Abstract**: Conversational learner simulations are valuable tools for testing learning theories, evaluating instructional materials and automated tutors, or powering teachable agents. Recently, large language models (LLM) have enabled richer, more naturalistic interactions with simulated learners; however, no open framework exists for evaluating whether such simulations faithfully reproduce real learner behavior. We introduce EvalConvoLearn, an open-source framework that assesses learner simulations along two axes: learning behavior (skill-conditioned mastery outcomes) and conversational quality (talk moves, error type distributions, question rate, turn length). EvalConvoLearn measures how closely a simulated learner approximates answer distributions observed in data by grounding metrics in authentic tutoring conversation datasets, and anchoring generated tutor responses in existing tutor utterances. The framework is demonstrated on a dataset of tutoring dialogues, including results for two LLM-based learner simulations, and the published GitHub code. 

---
# The Transparency Trap: How AI Disclaimers Create Overconfidence in High-Stakes Decisions 

**Authors**: Neil Todkar  

**Link**: [PDF](https://arxiv.org/pdf/2608.07493)  

**Abstract**: Current AI disclaimers often fail to function as intended due to warning habituation and a transparency paradox. As AI-generated information becomes pervasive in everyday decision-making, effective risk communication is increasingly critical for responsible design. This exploratory study examines how disclaimer placement and persuasive cues shape trust, perceived accuracy, and disclaimer engagement across three high-stakes domains: finance, medicine, and AI-generated content. Using a mixed within-between experimental design with 378 stimulus-level responses from 52 participants, we find that advisory content was generally trusted across conditions, even when disclaimers were present. A significant domain effect showed that medical content received the highest trust ratings. In the AI domain, the findings reveal a transparency paradox: some participants interpreted disclaimers not as warnings, but as signs of system self-awareness and honesty, paradoxically increasing perceived trustworthiness. Evidence of banner blindness further suggests that standardized AI disclaimers are insufficient to prevent over-reliance. Finance and medicine provide useful comparison domains by showing how users interpret warnings differently depending on context and perceived risk. These findings have vital implications for responsible AI design, algorithmic fairness, and consumer protection when users act on potentially misleading information in high-stakes settings. 

---
# PragyaDoc: A Universal Document Intelligence Framework for Multilingual Medical Document Understanding in Low-Resource Settings 

**Authors**: Jagpal Singh Jhala  

**Link**: [PDF](https://arxiv.org/pdf/2608.07478)  

**Abstract**: India's 22 official languages create a critical accessibility barrier: the majority of medical documentation exists exclusively in English, yet the patients who most urgently require this information - rural populations, ASHA workers, and patient families - are functionally excluded from understanding it. This paper presents PragyaDoc, a Universal Document Intelligence Framework that addresses this gap through a four-layer pipeline: a parallel ensemble OCR extraction layer, a geometric-lexical fusion layer, a deterministic domain structuring layer, and a dual-LLM medical reasoning and localization layer 

---
