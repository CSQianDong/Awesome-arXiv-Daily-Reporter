# Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning 

**Authors**: Dawid J. Kopiczko, Sagar Vaze, Tijmen Blankevoort, Yuki M. Asano  

**Link**: [PDF](https://arxiv.org/pdf/2602.11149)  

**Abstract**: Supervised fine-tuning (SFT) on chain-of-thought data is an essential post-training step for reasoning language models. Standard machine learning intuition suggests that training with more unique training samples yields better generalization. Counterintuitively, we show that SFT benefits from repetition: under a fixed update budget, training for more epochs on smaller datasets outperforms single-epoch training on larger datasets. On AIME'24/25 and GPQA benchmarks, Olmo3-7B trained for 128 epochs on 400 samples outperforms the equivalent 1 epoch on 51200 samples by 12-26 percentage points, with no additional catastrophic forgetting. We find that training token accuracy reliably signals when repetition has saturated; improvements from additional epochs plateau at full memorization, a pattern consistent across all settings. These findings provide a practical approach for reasoning SFT, where scaling epochs with token accuracy as a stopping criterion can replace expensive undirected data scaling. We pose the repetition advantage, where full memorization coincides with improved generalization, as a new open problem for the community in understanding the training dynamics of large language models. 

---
# TEGRA: Text Encoding With Graph and Retrieval Augmentation for Misinformation Detection 

**Authors**: Géraud Faye, Wassila Ouerdane, Guillaume Gadek, Céline Hudelot  

**Link**: [PDF](https://arxiv.org/pdf/2602.11106)  

**Abstract**: Misinformation detection is a critical task that can benefit significantly from the integration of external knowledge, much like manual fact-checking. In this work, we propose a novel method for representing textual documents that facilitates the incorporation of information from a knowledge base. Our approach, Text Encoding with Graph (TEG), processes documents by extracting structured information in the form of a graph and encoding both the text and the graph for classification purposes. Through extensive experiments, we demonstrate that this hybrid representation enhances misinformation detection performance compared to using language models alone. Furthermore, we introduce TEGRA, an extension of our framework that integrates domain-specific knowledge, further enhancing classification accuracy in most cases. 

---
# Safety Recovery in Reasoning Models Is Only a Few Early Steering Steps Away 

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Furong Huang, Dinesh Manocha, Amrit Singh Bedi  

**Link**: [PDF](https://arxiv.org/pdf/2602.11096)  

**Abstract**: Reinforcement learning (RL) based post-training for explicit chain-of-thought (e.g., GRPO) improves the reasoning ability of multimodal large-scale reasoning models (MLRMs). But recent evidence shows that it can simultaneously degrade safety alignment and increase jailbreak success rates. We propose SafeThink, a lightweight inference-time defense that treats safety recovery as a satisficing constraint rather than a maximization objective. SafeThink monitors the evolving reasoning trace with a safety reward model and conditionally injects an optimized short corrective prefix ("Wait, think safely") only when the safety threshold is violated. In our evaluations across six open-source MLRMs and four jailbreak benchmarks (JailbreakV-28K, Hades, FigStep, and MM-SafetyBench), SafeThink reduces attack success rates by 30-60% (e.g., LlamaV-o1: 63.33% to 5.74% on JailbreakV-28K, R1-Onevision: 69.07% to 5.65% on Hades) while preserving reasoning performance (MathVista accuracy: 65.20% to 65.00%). A key empirical finding from our experiments is that safety recovery is often only a few steering steps away: intervening in the first 1-3 reasoning steps typically suffices to redirect the full generation toward safe completions. 

---
# Can Large Language Models Make Everyone Happy? 

**Authors**: Usman Naseem, Gautam Siddharth Kashyap, Ebad Shabbir, Sushant Kumar Ray, Abdullah Mohammad, Rafiq Ali  

**Link**: [PDF](https://arxiv.org/pdf/2602.11091)  

**Abstract**: Misalignment in Large Language Models (LLMs) refers to the failure to simultaneously satisfy safety, value, and cultural dimensions, leading to behaviors that diverge from human expectations in real-world settings where these dimensions must co-occur. Existing benchmarks, such as SAFETUNEBED (safety-centric), VALUEBENCH (value-centric), and WORLDVIEW-BENCH (culture-centric), primarily evaluate these dimensions in isolation and therefore provide limited insight into their interactions and trade-offs. More recent efforts, including MIB and INTERPRETABILITY BENCHMARK-based on mechanistic interpretability, offer valuable perspectives on model failures; however, they remain insufficient for systematically characterizing cross-dimensional trade-offs. To address these gaps, we introduce MisAlign-Profile, a unified benchmark for measuring misalignment trade-offs inspired by mechanistic profiling. First, we construct MISALIGNTRADE, an English misaligned-aligned dataset across 112 normative domains taxonomies, including 14 safety, 56 value, and 42 cultural domains. In addition to domain labels, each prompt is classified with one of three orthogonal semantic types-object, attribute, or relations misalignment-using Gemma-2-9B-it and expanded via Qwen3-30B-A3B-Instruct-2507 with SimHash-based fingerprinting to avoid deduplication. Each prompt is paired with misaligned and aligned responses through two-stage rejection sampling to ensure quality. Second, we benchmark general-purpose, fine-tuned, and open-weight LLMs on MISALIGNTRADE-revealing 12%-34% misalignment trade-offs across dimensions. 

---
# DataChef: Cooking Up Optimal Data Recipes for LLM Adaptation via Reinforcement Learning 

**Authors**: Yicheng Chen, Zerun Ma, Xinchen Xie, Yining Li, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.11089)  

**Abstract**: In the current landscape of Large Language Models (LLMs), the curation of large-scale, high-quality training data is a primary driver of model performance. A key lever is the \emph{data recipe}, which comprises a data processing pipeline to transform raw sources into training corpora. Despite the growing use of LLMs to automate individual data processing steps, such as data synthesis and filtering, the overall design of data recipes remains largely manual and labor-intensive, requiring substantial human expertise and iteration. To bridge this gap, we formulate \emph{end-to-end data recipe generation} for LLM adaptation. Given a target benchmark and a pool of available data sources, a model is required to output a complete data recipe that adapts a base LLM to the target task. We present DataChef-32B, which performs online reinforcement learning using a proxy reward that predicts downstream performance for candidate recipes. Across six held-out tasks, DataChef-32B produces practical recipes that reach comparable downstream performance to those curated by human experts. Notably, the recipe from DataChef-32B adapts Qwen3-1.7B-Base to the math domain, achieving 66.7 on AIME'25 and surpassing Qwen3-1.7B. This work sheds new light on automating LLM training and developing self-evolving AI systems. 

---
# SteuerLLM: Local specialized large language model for German tax law analysis 

**Authors**: Sebastian Wind, Jeta Sopa, Laurin Schmid, Quirin Jackl, Sebastian Kiefer, Fei Wu, Martin Mayr, Harald Köstler, Gerhard Wellein, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2602.11081)  

**Abstract**: Large language models (LLMs) demonstrate strong general reasoning and language understanding, yet their performance degrades in domains governed by strict formal rules, precise terminology, and legally binding structure. Tax law exemplifies these challenges, as correct answers require exact statutory citation, structured legal argumentation, and numerical accuracy under rigid grading schemes. We algorithmically generate SteuerEx, the first open benchmark derived from authentic German university tax law examinations. SteuerEx comprises 115 expert-validated examination questions spanning six core tax law domains and multiple academic levels, and employs a statement-level, partial-credit evaluation framework that closely mirrors real examination practice. We further present SteuerLLM, a domain-adapted LLM for German tax law trained on a large-scale synthetic dataset generated from authentic examination material using a controlled retrieval-augmented pipeline. SteuerLLM (28B parameters) consistently outperforms general-purpose instruction-tuned models of comparable size and, in several cases, substantially larger systems, demonstrating that domain-specific data and architectural adaptation are more decisive than parameter scale for performance on realistic legal reasoning tasks. All benchmark data, training datasets, model weights, and evaluation code are released openly to support reproducible research in domain-specific legal artificial intelligence. A web-based demo of SteuerLLM is available at this https URL. 

---
# Simultaneous Speech-to-Speech Translation Without Aligned Data 

**Authors**: Tom Labiausse, Romain Fabre, Yannick Estève, Alexandre Défossez, Neil Zeghidour  

**Link**: [PDF](https://arxiv.org/pdf/2602.11072)  

**Abstract**: Simultaneous speech translation requires translating source speech into a target language in real-time while handling non-monotonic word dependencies. Traditional approaches rely on supervised training with word-level aligned data, which is difficult to collect at scale and thus depends on synthetic alignments using language-specific heuristics that are suboptimal. We propose Hibiki-Zero, which eliminates the need for word-level alignments entirely. This fundamentally simplifies the training pipeline and enables seamless scaling to diverse languages with varying grammatical structures, removing the bottleneck of designing language-specific alignment heuristics. We first train on sentence-level aligned data to learn speech translation at high latency, then apply a novel reinforcement learning strategy using GRPO to optimize latency while preserving translation quality. Hibiki-Zero achieves state-of-the-art performance in translation accuracy, latency, voice transfer, and naturalness across five X-to-English tasks. Moreover, we demonstrate that our model can be adapted to support a new input language with less than 1000h of speech. We provide examples, model weights, inference code and we release a benchmark containing 45h of multilingual data for speech translation evaluation. 

---
# Conversational Behavior Modeling Foundation Model With Multi-Level Perception 

**Authors**: Dingkun Zhou, Shuchang Pan, Jiachen Lian, Siddharth Banerjee, Sarika Pasumarthy, Dhruv Hebbar, Siddhant Patel, Zeyi Austin Li, Kan Jen Cheng, Sanay Bordia, Krish Patel, Akshaj Gupta, Tingle Li, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2602.11065)  

**Abstract**: Human conversation is organized by an implicit chain of thoughts that manifests as timed speech acts. Capturing this perceptual pathway is key to building natural full-duplex interactive systems. We introduce a framework that models this process as multi-level perception, and then reasons over conversational behaviors via a Graph-of-Thoughts (GoT). Our approach formalizes the intent-to-action pathway with a hierarchical labeling scheme, predicting high-level communicative intents and low-level speech acts to learn their causal and temporal dependencies. To train this system, we develop a high quality corpus that pairs controllable, event-rich dialogue data with human-annotated labels. The GoT framework structures streaming predictions as an evolving graph, enabling a transformer to forecast the next speech act, generate concise justifications for its decisions, and dynamically refine its reasoning. Experiments on both synthetic and real duplex dialogues show that the framework delivers robust behavior detection, produces interpretable reasoning chains, and establishes a foundation for benchmarking conversational reasoning in full duplex spoken dialogue systems. 

---
# Embedding Inversion via Conditional Masked Diffusion Language Models 

**Authors**: Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.11047)  

**Abstract**: We frame embedding inversion as conditional masked diffusion, recovering all tokens in parallel through iterative denoising rather than sequential autoregressive generation. A masked diffusion language model is conditioned on the target embedding via adaptive layer normalization, requiring only 8 forward passes through a 78M parameter model with no access to the target encoder. On 32-token sequences across three embedding models, the method achieves 81.3% token accuracy and 0.87 cosine similarity. 

---
# Language Model Inversion through End-to-End Differentiation 

**Authors**: Kevin Yandoka Denamganaï, Kartic Subr  

**Link**: [PDF](https://arxiv.org/pdf/2602.11044)  

**Abstract**: Despite emerging research on Language Models (LM), few approaches analyse the invertibility of LMs. That is, given a LM and a desirable target output sequence of tokens, determining what input prompts would yield the target output remains an open problem. We formulate this problem as a classical gradient-based optimisation. First, we propose a simple algorithm to achieve end-to-end differentiability of a given (frozen) LM and then find optimised prompts via gradient descent. Our central insight is to view LMs as functions operating on sequences of distributions over tokens (rather than the traditional view as functions on sequences of tokens). Our experiments and ablations demonstrate that our DLM-powered inversion can reliably and efficiently optimise prompts of lengths $10$ and $80$ for targets of length $20$, for several white-box LMs (out-of-the-box). 

---
# Linguistic Indicators of Early Cognitive Decline in the DementiaBank Pitt Corpus: A Statistical and Machine Learning Study 

**Authors**: Artsvik Avetisyan, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.11028)  

**Abstract**: Background: Subtle changes in spontaneous language production are among the earliest indicators of cognitive decline. Identifying linguistically interpretable markers of dementia can support transparent and clinically grounded screening approaches.
Methods: This study analyzes spontaneous speech transcripts from the DementiaBank Pitt Corpus using three linguistic representations: raw cleaned text, a part-of-speech (POS)-enhanced representation combining lexical and grammatical information, and a POS-only syntactic representation. Logistic regression and random forest models were evaluated under two protocols: transcript-level train-test splits and subject-level five-fold cross-validation to prevent speaker overlap. Model interpretability was examined using global feature importance, and statistical validation was conducted using Mann-Whitney U tests with Cliff's delta effect sizes.
Results: Across representations, models achieved stable performance, with syntactic and grammatical features retaining strong discriminative power even in the absence of lexical content. Subject-level evaluation yielded more conservative but consistent results, particularly for POS-enhanced and POS-only representations. Statistical analysis revealed significant group differences in functional word usage, lexical diversity, sentence structure, and discourse coherence, aligning closely with machine learning feature importance findings.
Conclusion: The results demonstrate that abstract linguistic features capture robust markers of early cognitive decline under clinically realistic evaluation. By combining interpretable machine learning with non-parametric statistical validation, this study supports the use of linguistically grounded features for transparent and reliable language-based cognitive screening. 

---
# LoRA-Squeeze: Simple and Effective Post-Tuning and In-Tuning Compression of LoRA Modules 

**Authors**: Ivan Vulić, Adam Grycner, Quentin de Laroussilhe, Jonas Pfeiffer  

**Link**: [PDF](https://arxiv.org/pdf/2602.10993)  

**Abstract**: Despite its huge number of variants, standard Low-Rank Adaptation (LoRA) is still a dominant technique for parameter-efficient fine-tuning (PEFT). Nonetheless, it faces persistent challenges, including the pre-selection of an optimal rank and rank-specific hyper-parameters, as well as the deployment complexity of heterogeneous-rank modules and more sophisticated LoRA derivatives. In this work, we introduce LoRA-Squeeze, a simple and efficient methodology that aims to improve standard LoRA learning by changing LoRA module ranks either post-hoc or dynamically during training}. Our approach posits that it is better to first learn an expressive, higher-rank solution and then compress it, rather than learning a constrained, low-rank solution directly. The method involves fine-tuning with a deliberately high(er) source rank, reconstructing or efficiently approximating the reconstruction of the full weight update matrix, and then using Randomized Singular Value Decomposition (RSVD) to create a new, compressed LoRA module at a lower target rank. Extensive experiments across 13 text and 10 vision-language tasks show that post-hoc compression often produces lower-rank adapters that outperform those trained directly at the target rank, especially if a small number of fine-tuning steps at the target rank is allowed. Moreover, a gradual, in-tuning rank annealing variant of LoRA-Squeeze consistently achieves the best LoRA size-performance trade-off. 

---
# Search or Accelerate: Confidence-Switched Position Beam Search for Diffusion Language Models 

**Authors**: Mingyu Cao, Alvaro Correia, Christos Louizos, Shiwei Liu, Lu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.10953)  

**Abstract**: Diffusion Language Models (DLMs) generate text by iteratively denoising a masked sequence, repeatedly deciding which positions to commit at each step. Standard decoding follows a greedy rule: unmask the most confident positions, yet this local choice can lock the model into a suboptimal unmasking order, especially on reasoning-heavy prompts. We present SOAR, a training-free decoding algorithm that adapts its behavior to the model's uncertainty. When confidence is low, SOAR briefly widens the search over alternative unmasking decisions to avoid premature commitments; when confidence is high, it collapses the search and decodes many positions in parallel to reduce the number of denoising iterations. Across mathematical reasoning and code generation benchmarks (GSM8K, MBPP, HumanEval) on Dream-7B and LLaDA-8B, SOAR improves generation quality while maintaining competitive inference speed, offering a practical way to balance quality and efficiency in DLM decoding. 

---
# Computational Phenomenology of Temporal Experience in Autism: Quantifying the Emotional and Narrative Characteristics of Lived Unpredictability 

**Authors**: Kacper Dudzic, Karolina Drożdż, Maciej Wodziński, Anastazja Szuła, Marcin Moskalewicz  

**Link**: [PDF](https://arxiv.org/pdf/2602.10947)  

**Abstract**: Disturbances in temporality, such as desynchronization with the social environment and its unpredictability, are considered core features of autism with a deep impact on relationships. However, limitations regarding research on this issue include: 1) the dominance of deficit-based medical models of autism, 2) sample size in qualitative research, and 3) the lack of phenomenological anchoring in computational research. To bridge the gap between phenomenological and computational approaches and overcome sample-size limitations, our research integrated three methodologies. Study A: structured phenomenological interviews with autistic individuals using the Transdiagnostic Assessment of Temporal Experience. Study B: computational analysis of an autobiographical corpus of autistic narratives built for this purpose. Study C: a replication of a computational study using narrative flow measures to assess the perceived phenomenological authenticity of autistic autobiographies. Interviews revealed that the most significant differences between the autistic and control groups concerned unpredictability of experience. Computational results mirrored these findings: the temporal lexicon in autistic narratives was significantly more negatively valenced - particularly the "Immediacy & Suddenness" category. Outlier analysis identified terms associated with perceived discontinuity (unpredictably, precipitously, and abruptly) as highly negative. The computational analysis of narrative flow found that the autistic narratives contained within the corpus quantifiably resemble autobiographical stories more than imaginary ones. Overall, the temporal challenges experienced by autistic individuals were shown to primarily concern lived unpredictability and stem from the contents of lived experience, and not from autistic narrative construction. 

---
# SoftMatcha 2: A Fast and Soft Pattern Matcher for Trillion-Scale Corpora 

**Authors**: Masataka Yoneda, Yusuke Matsushita, Go Kamoda, Kohei Suenaga, Takuya Akiba, Masaki Waga, Sho Yokoi  

**Link**: [PDF](https://arxiv.org/pdf/2602.10908)  

**Abstract**: We present an ultra-fast and flexible search algorithm that enables search over trillion-scale natural language corpora in under 0.3 seconds while handling semantic variations (substitution, insertion, and deletion). Our approach employs string matching based on suffix arrays that scales well with corpus size. To mitigate the combinatorial explosion induced by the semantic relaxation of queries, our method is built on two key algorithmic ideas: fast exact lookup enabled by a disk-aware design, and dynamic corpus-aware pruning. We theoretically show that the proposed method suppresses exponential growth in the search space with respect to query length by leveraging statistical properties of natural language. In experiments on FineWeb-Edu (Lozhkov et al., 2024) (1.4T tokens), we show that our method achieves significantly lower search latency than existing methods: infini-gram (Liu et al., 2024), infini-gram mini (Xu et al., 2025), and SoftMatcha (Deguchi et al., 2025). As a practical application, we demonstrate that our method identifies benchmark contamination in training corpora, unidentified by existing approaches. We also provide an online demo of fast, soft search across corpora in seven languages. 

---
# The CLEF-2026 FinMMEval Lab: Multilingual and Multimodal Evaluation of Financial AI Systems 

**Authors**: Zhuohan Xie, Rania Elbadry, Fan Zhang, Georgi Georgiev, Xueqing Peng, Lingfei Qian, Jimin Huang, Dimitar Dimitrov, Vanshikaa Jani, Yuyang Dai, Jiahui Geng, Yuxia Wang, Ivan Koychev, Veselin Stoyanov, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2602.10886)  

**Abstract**: We present the setup and the tasks of the FinMMEval Lab at CLEF 2026, which introduces the first multilingual and multimodal evaluation framework for financial Large Language Models (LLMs). While recent advances in financial natural language processing have enabled automated analysis of market reports, regulatory documents, and investor communications, existing benchmarks remain largely monolingual, text-only, and limited to narrow subtasks. FinMMEval 2026 addresses this gap by offering three interconnected tasks that span financial understanding, reasoning, and decision-making: Financial Exam Question Answering, Multilingual Financial Question Answering (PolyFiQA), and Financial Decision Making. Together, these tasks provide a comprehensive evaluation suite that measures models' ability to reason, generalize, and act across diverse languages and modalities. The lab aims to promote the development of robust, transparent, and globally inclusive financial AI systems, with datasets and evaluation resources publicly released to support reproducible research. 

---
# Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis 

**Authors**: Zhiyin Tan, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2602.10881)  

**Abstract**: Systematic reviews and meta-analyses rely on converting narrative articles into structured, numerically grounded study records. Despite rapid advances in large language models (LLMs), it remains unclear whether they can meet the structural requirements of this process, which hinge on preserving roles, methods, and effect-size attribution across documents rather than on recognizing isolated entities. We propose a structural, diagnostic framework that evaluates LLM-based evidence extraction as a progression of schema-constrained queries with increasing relational and numerical complexity, enabling precise identification of failure points beyond atom-level extraction. Using a manually curated corpus spanning five scientific domains, together with a unified query suite and evaluation protocol, we evaluate two state-of-the-art LLMs under both per-document and long-context, multi-document input regimes. Across domains and models, performance remains moderate for single-property queries but degrades sharply once tasks require stable binding between variables, roles, statistical methods, and effect sizes. Full meta-analytic association tuples are extracted with near-zero reliability, and long-context inputs further exacerbate these failures. Downstream aggregation amplifies even minor upstream errors, rendering corpus-level statistics unreliable. Our analysis shows that these limitations stem not from entity recognition errors, but from systematic structural breakdowns, including role reversals, cross-analysis binding drift, instance compression in dense result sections, and numeric misattribution, indicating that current LLMs lack the structural fidelity, relational binding, and numerical grounding required for automated meta-analysis. The code and data are publicly available at GitHub (this https URL). 

---
# C-MOP: Integrating Momentum and Boundary-Aware Clustering for Enhanced Prompt Evolution 

**Authors**: Binwei Yan, Yifei Fu, Mingjian Zhu, Hanting Chen, Mingxuan Yuan, Yunhe Wang, Hailin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10874)  

**Abstract**: Automatic prompt optimization is a promising direction to boost the performance of Large Language Models (LLMs). However, existing methods often suffer from noisy and conflicting update signals. In this research, we propose C-MOP (Cluster-based Momentum Optimized Prompting), a framework that stabilizes optimization via Boundary-Aware Contrastive Sampling (BACS) and Momentum-Guided Semantic Clustering (MGSC). Specifically, BACS utilizes batch-level information to mine tripartite features--Hard Negatives, Anchors, and Boundary Pairs--to precisely characterize the typical representation and decision boundaries of positive and negative prompt samples. To resolve semantic conflicts, MGSC introduces a textual momentum mechanism with temporal decay that distills persistent consensus from fluctuating gradients across iterations. Extensive experiments demonstrate that C-MOP consistently outperforms SOTA baselines like PromptWizard and ProTeGi, yielding average gains of 1.58% and 3.35%. Notably, C-MOP enables a general LLM with 3B activated parameters to surpass a 70B domain-specific dense LLM, highlighting its effectiveness in driving precise prompt evolution. The code is available at this https URL. 

---
# I can tell whether you are a Native Hawlêri Speaker! How ANN, CNN, and RNN perform in NLI-Native Language Identification 

**Authors**: Hardi Garari, Hossein Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2602.10832)  

**Abstract**: Native Language Identification (NLI) is a task in Natural Language Processing (NLP) that typically determines the native language of an author through their writing or a speaker through their speaking. It has various applications in different areas, such as forensic linguistics and general linguistics studies. Although considerable research has been conducted on NLI regarding two different languages, such as English and German, the literature indicates a significant gap regarding NLI for dialects and subdialects. The gap becomes wider in less-resourced languages such as Kurdish. This research focuses on NLI within the context of a subdialect of Sorani (Central) Kurdish. It aims to investigate the NLI for Hewlêri, a subdialect spoken in Hewlêr (Erbil), the Capital of the Kurdistan Region of Iraq. We collected about 24 hours of speech by recording interviews with 40 native or non-native Hewlêri speakers, 17 female and 23 male. We created three Neural Network-based models: Artificial Neural Network (ANN), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN), which were evaluated through 66 experiments, covering various time-frames from 1 to 60 seconds, undersampling, oversampling, and cross-validation. The RNN model showed the highest accuracy of 95.92% for 5-second audio segmentation, using an 80:10:10 data splitting scheme. The created dataset is the first speech dataset for NLI on the Hewlêri subdialect in the Sorani Kurdish dialect, which can be of benefit to various research areas. 

---
# Beyond Confidence: The Rhythms of Reasoning in Generative Models 

**Authors**: Deyuan Liu, Zecheng Wang, Zhanyue Qin, Zhiying Tu, Dianhui Chu, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2602.10816)  

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities yet suffer from sensitivity to slight input context variations, hampering reliability. Conventional metrics like accuracy and perplexity fail to assess local prediction robustness, as normalized output probabilities can obscure the underlying resilience of an LLM's internal state to perturbations. We introduce the Token Constraint Bound ($\delta_{\mathrm{TCB}}$), a novel metric that quantifies the maximum internal state perturbation an LLM can withstand before its dominant next-token prediction significantly changes. Intrinsically linked to output embedding space geometry, $\delta_{\mathrm{TCB}}$ provides insights into the stability of the model's internal predictive commitment. Our experiments show $\delta_{\mathrm{TCB}}$ correlates with effective prompt engineering and uncovers critical prediction instabilities missed by perplexity during in-context learning and text generation. $\delta_{\mathrm{TCB}}$ offers a principled, complementary approach to analyze and potentially improve the contextual stability of LLM predictions. 

---
# Deep Learning-based Method for Expressing Knowledge Boundary of Black-Box LLM 

**Authors**: Haotian Sheng, Heyong Wang, Ming Hong, Hongman He, Junqiu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10801)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success, however, the emergence of content generation distortion (hallucination) limits their practical applications. The core cause of hallucination lies in LLMs' lack of awareness regarding their stored internal knowledge, preventing them from expressing their knowledge state on questions beyond their internal knowledge boundaries, as humans do. However, existing research on knowledge boundary expression primarily focuses on white-box LLMs, leaving methods suitable for black-box LLMs which offer only API access without revealing internal parameters-largely unexplored. Against this backdrop, this paper proposes LSCL (LLM-Supervised Confidence Learning), a deep learning-based method for expressing the knowledge boundaries of black-box LLMs. Based on the knowledge distillation framework, this method designs a deep learning model. Taking the input question, output answer, and token probability from a black-box LLM as inputs, it constructs a mapping between the inputs and the model' internal knowledge state, enabling the quantification and expression of the black-box LLM' knowledge boundaries. Experiments conducted on diverse public datasets and with multiple prominent black-box LLMs demonstrate that LSCL effectively assists black-box LLMs in accurately expressing their knowledge boundaries. It significantly outperforms existing baseline models on metrics such as accuracy and recall rate. Furthermore, considering scenarios where some black-box LLMs do not support access to token probability, an adaptive alternative method is proposed. The performance of this alternative approach is close to that of LSCL and surpasses baseline models. 

---
# Reinforced Curriculum Pre-Alignment for Domain-Adaptive VLMs 

**Authors**: Yuming Yan, Shuo Yang, Kai Tang, Sihong Chen, Yang Zhang, Ke Xu, Dan Hu, Qun Yu, Pengfei Hu, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2602.10740)  

**Abstract**: Vision-Language Models (VLMs) demonstrate remarkable general-purpose capabilities but often fall short in specialized domains such as medical imaging or geometric problem-solving. Supervised Fine-Tuning (SFT) can enhance performance within a target domain, but it typically causes catastrophic forgetting, limiting its generalization. The central challenge, therefore, is to adapt VLMs to new domains while preserving their general-purpose capabilities. Continual pretraining is effective for expanding knowledge in Large Language Models (LLMs), but it is less feasible for VLMs due to prohibitive computational costs and the unavailability of pretraining data for most open-source models. This necessitates efficient post-training adaptation methods. Reinforcement learning (RL)-based approaches such as Group Relative Policy Optimization (GRPO) have shown promise in preserving general abilities, yet they often fail in domain adaptation scenarios where the model initially lacks sufficient domain knowledge, leading to optimization collapse. To bridge this gap, we propose Reinforced Curriculum Pre-Alignment (RCPA), a novel post-training paradigm that introduces a curriculum-aware progressive modulation mechanism. In the early phase, RCPA applies partial output constraints to safely expose the model to new domain concepts. As the model's domain familiarity increases, training gradually transitions to full generation optimization, refining responses and aligning them with domain-specific preferences. This staged adaptation balances domain knowledge acquisition with the preservation of general multimodal capabilities. Extensive experiments across specialized domains and general benchmarks validate the effectiveness of RCPA, establishing a practical pathway toward building high-performing and domain-adaptive VLMs. 

---
# Macaron: Controlled, Human-Written Benchmark for Multilingual and Multicultural Reasoning via Template-Filling 

**Authors**: Alaa Elsetohy, Sama Hadhoud, Haryo Akbarianto Wibowo, Chenxi Whitehouse, Genta Indra Winata, Fajri Koto, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2602.10732)  

**Abstract**: Multilingual benchmarks rarely test reasoning over culturally grounded premises: translated datasets keep English-centric scenarios, while culture-first datasets often lack control over the reasoning required. We propose Macaron, a template-first benchmark that factorizes reasoning type and cultural aspect across question languages. Using 100 language-agnostic templates that cover 7 reasoning types, 22 cultural aspects, native annotators create scenario-aligned English and local-language multiple-choice questions and systematically derived True/False questions. Macaron contains 11,862 instances spanning 20 countries/cultural contexts, 10 scripts, and 20 languages (including low-resource ones like Amharic, Yoruba, Zulu, Kyrgyz, and some Arabic dialects). In zero-shot evaluation of 21 multilingual LLMs, reasoning-mode models achieve the strongest performance and near-parity between English and local languages, while open-weight models degrade substantially in local languages and often approach chance on T/F tasks. Culture-grounded mathematical and counting templates are consistently the hardest. The data can be accessed here this https URL. 

---
# Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents 

**Authors**: Yifei Li, Weidong Guo, Lingling Zhang, Rongman Xu, Muye Huang, Hui Liu, Lijiao Xu, Yu Xu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10715)  

**Abstract**: Long-term conversational memory is a core capability for LLM-based dialogue systems, yet existing benchmarks and evaluation protocols primarily focus on surface-level factual recall. In realistic interactions, appropriate responses often depend on implicit constraints such as user state, goals, or values that are not explicitly queried later. To evaluate this setting, we introduce \textbf{LoCoMo-Plus}, a benchmark for assessing cognitive memory under cue--trigger semantic disconnect, where models must retain and apply latent constraints across long conversational contexts. We further show that conventional string-matching metrics and explicit task-type prompting are misaligned with such scenarios, and propose a unified evaluation framework based on constraint consistency. Experiments across diverse backbone models, retrieval-based methods, and memory systems demonstrate that cognitive memory remains challenging and reveals failures not captured by existing benchmarks. Our code and evaluation framework are publicly available at: this https URL. 

---
# Targeted Syntactic Evaluation of Language Models on Georgian Case Alignment 

**Authors**: Daniel Gallagher, Gerhard Heyer  

**Link**: [PDF](https://arxiv.org/pdf/2602.10661)  

**Abstract**: This paper evaluates the performance of transformer-based language models on split-ergative case alignment in Georgian, a particularly rare system for assigning grammatical cases to mark argument roles. We focus on subject and object marking determined through various permutations of nominative, ergative, and dative noun forms. A treebank-based approach for the generation of minimal pairs using the Grew query language is implemented. We create a dataset of 370 syntactic tests made up of seven tasks containing 50-70 samples each, where three noun forms are tested in any given sample. Five encoder- and two decoder-only models are evaluated with word- and/or sentence-level accuracy metrics. Regardless of the specific syntactic makeup, models performed worst in assigning the ergative case correctly and strongest in assigning the nominative case correctly. Performance correlated with the overall frequency distribution of the three forms (NOM > DAT > ERG). Though data scarcity is a known issue for low-resource languages, we show that the highly specific role of the ergative along with a lack of available training data likely contributes to poor performance on this case. The dataset is made publicly available and the methodology provides an interesting avenue for future syntactic evaluations of languages where benchmarks are limited. 

---
# Benchmarks Are Not That Out of Distribution: Word Overlap Predicts Performance 

**Authors**: Woojin Chung, Jeonghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.10657)  

**Abstract**: Understanding what constitutes high-quality pre-training data remains a central question in language model training. In this work, we investigate whether benchmark performance is primarily driven by the degree of statistical pattern overlap between pre-training corpora and evaluation datasets. We measure this overlap using word-level unigram cross-entropy and word frequency statistics, and perform controlled experiments across $10$ zero-shot benchmarks, $4$ pre-training datasets spanning $8.5\mathrm{B}$ to $60\mathrm{B}$ tokens, and model sizes ranging from $400\mathrm{M}$ to $3\mathrm{B}$ parameters. Our results demonstrate a robust inverse relationship between word-level unigram cross-entropy and benchmark performance, suggesting that widely used benchmarks are strongly influenced by word overlap between training and evaluation data. Thus, larger pre-training subsets with similar word-level unigram cross-entropy yield improved downstream results, indicating that word frequency statistics play an additional role in shaping benchmark scores. Taken together, these results suggest that many standard benchmarks are only weakly out-of-distribution relative to pre-training corpora, so that simple word-overlap statistics predict benchmark performance. 

---
# UMEM: Unified Memory Extraction and Management Framework for Generalizable Memory 

**Authors**: Yongshi Ye, Hui Jiang, Feihu Jiang, Tian Lan, Yichao Du, Biao Fu, Xiaodong Shi, Qianghuai Jia, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.10652)  

**Abstract**: Self-evolving memory serves as the trainable parameters for Large Language Models (LLMs)-based agents, where extraction (distilling insights from experience) and management (updating the memory bank) must be tightly coordinated. Existing methods predominately optimize memory management while treating memory extraction as a static process, resulting in poor generalization, where agents accumulate instance-specific noise rather than robust memories. To address this, we propose Unified Memory Extraction and Management (UMEM), a self-evolving agent framework that jointly optimizes a Large Language Model to simultaneous extract and manage memories. To mitigate overfitting to specific instances, we introduce Semantic Neighborhood Modeling and optimize the model with a neighborhood-level marginal utility reward via GRPO. This approach ensures memory generalizability by evaluating memory utility across clusters of semantically related queries. Extensive experiments across five benchmarks demonstrate that UMEM significantly outperforms highly competitive baselines, achieving up to a 10.67% improvement in multi-turn interactive tasks. Futhermore, UMEM maintains a monotonic growth curve during continuous evolution. Codes and models will be publicly released. 

---
# How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning 

**Authors**: Jiahao Yuan, Yike Xu, Jinyong Wen, Baokun Wang, Yang Chen, Xiaotong Lin, Wuliang Huang, Ziyi Gao, Xing Fu, Yu Cheng, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10622)  

**Abstract**: Decoder-only large language models are increasingly used as behavioral encoders for user representation learning, yet the impact of attention masking on the quality of user embeddings remains underexplored. In this work, we conduct a systematic study of causal, hybrid, and bidirectional attention masks within a unified contrastive learning framework trained on large-scale real-world Alipay data that integrates long-horizon heterogeneous user behaviors. To improve training dynamics when transitioning from causal to bidirectional attention, we propose Gradient-Guided Soft Masking, a gradient-based pre-warmup applied before a linear scheduler that gradually opens future attention during optimization. Evaluated on 9 industrial user cognition benchmarks covering prediction, preference, and marketing sensitivity tasks, our approach consistently yields more stable training and higher-quality bidirectional representations compared with causal, hybrid, and scheduler-only baselines, while remaining compatible with decoder pretraining. Overall, our findings highlight the importance of masking design and training transition in adapting decoder-only LLMs for effective user representation learning. Our code is available at this https URL. 

---
# Online Causal Kalman Filtering for Stable and Effective Policy Optimization 

**Authors**: Shuo He, Lang Feng, Xin Cheng, Lei Feng, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2602.10609)  

**Abstract**: Reinforcement learning for large language models suffers from high-variance token-level importance sampling (IS) ratios, which would destabilize policy optimization at scale. To improve stability, recent methods typically use a fixed sequence-level IS ratio for all tokens in a sequence or adjust each token's IS ratio separately, thereby neglecting temporal off-policy derivation across tokens in a sequence. In this paper, we first empirically identify that local off-policy deviation is structurally inconsistent at the token level, which may distort policy-gradient updates across adjacent tokens and lead to training collapse. To address the issue, we propose Online Causal Kalman Filtering for stable and effective Policy Optimization (KPO). Concretely, we model the desired IS ratio as a latent state that evolves across tokens and apply a Kalman filter to update this state online and autoregressively based on the states of past tokens, regardless of future tokens. The resulting filtered IS ratios preserve token-wise local structure-aware variation while strongly smoothing noise spikes, yielding more stable and effective policy updates. Experimentally, KPO achieves superior results on challenging math reasoning datasets compared with state-of-the-art counterparts. 

---
# Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters 

**Authors**: Ailin Huang, Ang Li, Aobo Kong, Bin Wang, Binxing Jiao, Bo Dong, Bojun Wang, Boyu Chen, Brian Li, Buyun Ma, Chang Su, Changxin Miao, Changyi Wan, Chao Lou, Chen Hu, Chen Xu, Chenfeng Yu, Chengting Feng, Chengyuan Yao, Chunrui Han, Dan Ma, Dapeng Shi, Daxin Jiang, Dehua Ma, Deshan Sun, Di Qi, Enle Liu, Fajie Zhang, Fanqi Wan, Guanzhe Huang, Gulin Yan, Guoliang Cao, Guopeng Li, Han Cheng, Hangyu Guo, Hanshan Zhang, Hao Nie, Haonan Jia, Haoran Lv, Hebin Zhou, Hekun Lv, Heng Wang, Heung-Yeung Shum, Hongbo Huang, Hongbo Peng, Hongyu Zhou, Hongyuan Wang, Houyong Chen, Huangxi Zhu, Huimin Wu, Huiyong Guo, Jia Wang, Jian Zhou, Jianjian Sun, Jiaoren Wu, Jiaran Zhang, Jiashu Lv, Jiashuo Liu, Jiayi Fu, Jiayu Liu, Jie Cheng, Jie Luo, Jie Yang, Jie Zhou, Jieyi Hou, Jing Bai, Jingcheng Hu, Jingjing Xie, Jingwei Wu, Jingyang Zhang, Jishi Zhou, Junfeng Liu, Junzhe Lin, Ka Man Lo, Kai Liang, Kaibo Liu, Kaijun Tan, Kaiwen Yan, Kaixiang Li, Kang An, Kangheng Lin, Lei Yang, Liang Lv, Liang Zhao, Liangyu Chen, Lieyu Shi, Liguo Tan, Lin Lin, Lina Chen, Luck Ma, Mengqiang Ren, Michael Li, Ming Li, Mingliang Li, Mingming Zhang, Mingrui Chen, Mitt Huang, Na Wang, Peng Liu, Qi Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.10604)  

**Abstract**: We introduce Step 3.5 Flash, a sparse Mixture-of-Experts (MoE) model that bridges frontier-level agentic intelligence and computational efficiency. We focus on what matters most when building agents: sharp reasoning and fast, reliable execution. Step 3.5 Flash pairs a 196B-parameter foundation with 11B active parameters for efficient inference. It is optimized with interleaved 3:1 sliding-window/full attention and Multi-Token Prediction (MTP-3) to reduce the latency and cost of multi-round agentic interactions. To reach frontier-level intelligence, we design a scalable reinforcement learning framework that combines verifiable signals with preference feedback, while remaining stable under large-scale off-policy training, enabling consistent self-improvement across mathematics, code, and tool use. Step 3.5 Flash demonstrates strong performance across agent, coding, and math tasks, achieving 85.4% on IMO-AnswerBench, 86.4% on LiveCodeBench-v6 (2024.08-2025.05), 88.2% on tau2-Bench, 69.0% on BrowseComp (with context management), and 51.0% on Terminal-Bench 2.0, comparable to frontier models such as GPT-5.2 xHigh and Gemini 3.0 Pro. By redefining the efficiency frontier, Step 3.5 Flash provides a high-density foundation for deploying sophisticated agents in real-world industrial environments. 

---
# When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning 

**Authors**: Leheng Sheng, Yongtao Zhang, Wenchang Ma, Yaorui Shi, Ting Huang, Xiang Wang, An Zhang, Ke Shen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.10560)  

**Abstract**: While reasoning over long context is crucial for various real-world applications, it remains challenging for large language models (LLMs) as they suffer from performance degradation as the context length grows. Recent work MemAgent has tried to tackle this by processing context chunk-by-chunk in an RNN-like loop and updating a textual memory for final answering. However, this naive recurrent memory update faces two crucial drawbacks: (i) memory can quickly explode because it can update indiscriminately, even on evidence-free chunks; and (ii) the loop lacks an exit mechanism, leading to unnecessary computation after even sufficient evidence is collected. To address these issues, we propose GRU-Mem, which incorporates two text-controlled gates for more stable and efficient long-context reasoning. Specifically, in GRU-Mem, the memory only updates when the update gate is open and the recurrent loop will exit immediately once the exit gate is open. To endow the model with such capabilities, we introduce two reward signals $r^{\text{update}}$ and $r^{\text{exit}}$ within end-to-end RL, rewarding the correct updating and exiting behaviors respectively. Experiments on various long-context reasoning tasks demonstrate the effectiveness and efficiency of GRU-Mem, which generally outperforms the vanilla MemAgent with up to 400\% times inference speed acceleration. 

---
# LHAW: Controllable Underspecification for Long-Horizon Tasks 

**Authors**: George Pu, Michael S. Lee, Udari Madhushani Sehwag, David J. Lee, Bryan Zhu, Yash Maurya, Mohit Raghavendra, Yuan Xue, Samuel Marc Denton  

**Link**: [PDF](https://arxiv.org/pdf/2602.10525)  

**Abstract**: Long-horizon workflow agents that operate effectively over extended periods are essential for truly autonomous systems. Their reliable execution critically depends on the ability to reason through ambiguous situations in which clarification seeking is necessary to ensure correct task execution. However, progress is limited by the lack of scalable, task-agnostic frameworks for systematically curating and measuring the impact of ambiguity across custom workflows. We address this gap by introducing LHAW (Long-Horizon Augmented Workflows), a modular, dataset-agnostic synthetic pipeline that transforms any well-specified task into controllable underspecified variants by systematically removing information across four dimensions - Goals, Constraints, Inputs, and Context - at configurable severity levels. Unlike approaches that rely on LLM predictions of ambiguity, LHAW validates variants through empirical agent trials, classifying them as outcome-critical, divergent, or benign based on observed terminal state divergence. We release 285 task variants from TheAgentCompany, SWE-Bench Pro and MCP-Atlas according to our taxonomy alongside formal analysis measuring how current agents detect, reason about, and resolve underspecification across ambiguous settings. LHAW provides the first systematic framework for cost-sensitive evaluation of agent clarification behavior in long-horizon settings, enabling development of reliable autonomous systems. 

---
# On the Robustness of Knowledge Editing for Detoxification 

**Authors**: Ming Dong, Shiyi Tang, Ziyan Peng, Guanyi Chen, Tingting He  

**Link**: [PDF](https://arxiv.org/pdf/2602.10504)  

**Abstract**: Knowledge-Editing-based (KE-based) detoxification has emerged as a promising approach for mitigating harmful behaviours in Large Language Models. Existing evaluations, however, largely rely on automatic toxicity classifiers, implicitly assuming that reduced toxicity scores reflect genuine behavioural suppression. In this work, we propose a robustness-oriented evaluation framework for KE-based detoxification that examines its reliability beyond standard classifier-based metrics along three dimensions: optimisation robustness, compositional robustness, and cross-lingual robustness. We identify pseudo-detoxification as a common failure mode, where apparent toxicity reductions arise from degenerate generation behaviours rather than meaningful suppression of unsafe content. We further show that detoxification effectiveness degrades when multiple unsafe behaviours are edited jointly, and that both monolingual and cross-lingual detoxification remain effective only under specific model-method combinations. Overall, our results indicate that KE-based detoxification is robust only for certain models, limited numbers of detoxification objectives, and a subset of languages. 

---
# Canvas-of-Thought: Grounding Reasoning via Mutable Structured States 

**Authors**: Lingzhuang Sun, Yuxia Zhu, Ruitong Liu, Hao Liang, Zheng Sun, Caijun Jia, Honghao He, Yuchen Wu, Siyuan Li, Jingxuan Wei, Xiangxiang Zhang, Bihui Yu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10494)  

**Abstract**: While Chain-of-Thought (CoT) prompting has significantly advanced the reasoning capabilities of Multimodal Large Language Models (MLLMs), relying solely on linear text sequences remains a bottleneck for complex tasks. We observe that even when auxiliary visual elements are interleaved, they are often treated as static snapshots within a one-dimensional, unstructured reasoning chain. We argue that such approaches treat reasoning history as an immutable stream: correcting a local error necessitates either generating verbose downstream corrections or regenerating the entire context. This forces the model to implicitly maintain and track state updates, significantly increasing token consumption and cognitive load. This limitation is particularly acute in high-dimensional domains, such as geometry and SVG design, where the textual expression of CoT lacks explicit visual guidance, further constraining the model's reasoning precision. To bridge this gap, we introduce \textbf{Canvas-of-Thought (Canvas-CoT)}. By leveraging a HTML Canvas as an external reasoning substrate, Canvas-CoT empowers the model to perform atomic, DOM-based CRUD operations. This architecture enables in-place state revisions without disrupting the surrounding context, allowing the model to explicitly maintain the "ground truth". Furthermore, we integrate a rendering-based critique loop that serves as a hard constraint validator, providing explicit visual feedback to resolve complex tasks that are difficult to articulate through text alone. Extensive experiments on VCode, RBench-V, and MathVista demonstrate that Canvas-CoT significantly outperforms existing baselines, establishing a new paradigm for context-efficient multimodal reasoning. 

---
# Neuro-Symbolic Synergy for Interactive World Modeling 

**Authors**: Hongyu Zhao, Siyu Zhou, Haolin Yang, Zengyi Qin, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.10480)  

**Abstract**: Large language models (LLMs) exhibit strong general-purpose reasoning capabilities, yet they frequently hallucinate when used as world models (WMs), where strict compliance with deterministic transition rules--particularly in corner cases--is essential. In contrast, Symbolic WMs provide logical consistency but lack semantic expressivity. To bridge this gap, we propose Neuro-Symbolic Synergy (NeSyS), a framework that integrates the probabilistic semantic priors of LLMs with executable symbolic rules to achieve both expressivity and robustness. NeSyS alternates training between the two models using trajectories inadequately explained by the other. Unlike rule-based prompting, the symbolic WM directly constrains the LLM by modifying its output probability distribution. The neural WM is fine-tuned only on trajectories not covered by symbolic rules, reducing training data by 50% without loss of accuracy. Extensive experiments on three distinct interactive environments, i.e., ScienceWorld, Webshop, and Plancraft, demonstrate NeSyS's consistent advantages over baselines in both WM prediction accuracy and data efficiency. 

---
# LATA: A Tool for LLM-Assisted Translation Annotation 

**Authors**: Baorong Huang, Ali Asiri  

**Link**: [PDF](https://arxiv.org/pdf/2602.10454)  

**Abstract**: The construction of high-quality parallel corpora for translation research has increasingly evolved from simple sentence alignment to complex, multi-layered annotation tasks. This methodological shift presents significant challenges for structurally divergent language pairs, such as Arabic--English, where standard automated tools frequently fail to capture deep linguistic shifts or semantic nuances. This paper introduces a novel, LLM-assisted interactive tool designed to reduce the gap between scalable automation and the rigorous precision required for expert human judgment. Unlike traditional statistical aligners, our system employs a template-based Prompt Manager that leverages large language models (LLMs) for sentence segmentation and alignment under strict JSON output constraints. In this tool, automated preprocessing integrates into a human-in-the-loop workflow, allowing researchers to refine alignments and apply custom translation technique annotations through a stand-off architecture. By leveraging LLM-assisted processing, the tool balances annotation efficiency with the linguistic precision required to analyze complex translation phenomena in specialized domains. 

---
# EVOKE: Emotion Vocabulary Of Korean and English 

**Authors**: Yoonwon Jung, Hagyeong Shin, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2602.10414)  

**Abstract**: This paper introduces EVOKE, a parallel dataset of emotion vocabulary in English and Korean. The dataset offers comprehensive coverage of emotion words in each language, in addition to many-to-many translations between words in the two languages and identification of language-specific emotion words. The dataset contains 1,427 Korean words and 1,399 English words, and we systematically annotate 819 Korean and 924 English adjectives and verbs. We also annotate multiple meanings of each word and their relationships, identifying polysemous emotion words and emotion-related metaphors. The dataset is, to our knowledge, the most comprehensive, systematic, and theory-agnostic dataset of emotion words in both Korean and English to date. It can serve as a practical tool for emotion science, psycholinguistics, computational linguistics, and natural language processing, allowing researchers to adopt different views on the resource reflecting their needs and theoretical perspectives. The dataset is publicly available at this https URL. 

---
# When are We Worried? Temporal Trends of Anxiety and What They Reveal about Us 

**Authors**: Saif M. Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2602.10400)  

**Abstract**: In this short paper, we make use of a recently created lexicon of word-anxiety associations to analyze large amounts of US and Canadian social media data (tweets) to explore *when* we are anxious and what insights that reveals about us. We show that our levels of anxiety on social media exhibit systematic patterns of rise and fall during the day -- highest at 8am (in-line with when we have high cortisol levels in the body) and lowest around noon. Anxiety is lowest on weekends and highest mid-week. We also examine anxiety in past, present, and future tense sentences to show that anxiety is highest in past tense and lowest in future tense. Finally, we examine the use of anxiety and calmness words in posts that contain pronouns to show: more anxiety in 3rd person pronouns (he, they) posts than 1st and 2nd person pronouns and higher anxiety in posts with subject pronouns (I, he, she, they) than object pronouns (me, him, her, them). Overall, these trends provide valuable insights on not just when we are anxious, but also how different types of focus (future, past, self, outward, etc.) are related to anxiety. 

---
# Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs 

**Authors**: Zhongzhi Li, Xuansheng Wu, Yijiang Li, Lijie Hu, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10388)  

**Abstract**: The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). Many existing approaches to constructing post-training data quantify diversity using text-based metrics that capture linguistic variation, but such metrics provide only weak signals for the task-relevant features that determine downstream performance. In this work, we introduce Feature Activation Coverage (FAC) which measures data diversity in an interpretable feature space. Building upon this metric, we further propose a diversity-driven data synthesis framework, named FAC Synthesis, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples that explicitly reflect these features. Experiments show that our approach consistently improves both data diversity and downstream performance on various tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer. Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs. 

---
# When Tables Go Crazy: Evaluating Multimodal Models on French Financial Documents 

**Authors**: Virginie Mouilleron, Théo Lasnier, Djamé Seddah  

**Link**: [PDF](https://arxiv.org/pdf/2602.10384)  

**Abstract**: Vision-language models (VLMs) perform well on many document understanding tasks, yet their reliability in specialized, non-English domains remains underexplored. This gap is especially critical in finance, where documents mix dense regulatory text, numerical tables, and visual charts, and where extraction errors can have real-world consequences. We introduce Multimodal Finance Eval, the first multimodal benchmark for evaluating French financial document understanding. The dataset contains 1,204 expert-validated questions spanning text extraction, table comprehension, chart interpretation, and multi-turn conversational reasoning, drawn from real investment prospectuses, KIDs, and PRIIPs. We evaluate six open-weight VLMs (8B-124B parameters) using an LLM-as-judge protocol. While models achieve strong performance on text and table tasks (85-90% accuracy), they struggle with chart interpretation (34-62%). Most notably, multi-turn dialogue reveals a sharp failure mode: early mistakes propagate across turns, driving accuracy down to roughly 50% regardless of model size.
These results show that current VLMs are effective for well-defined extraction tasks but remain brittle in interactive, multi-step financial analysis. Multimodal Finance Eval offers a challenging benchmark to measure and drive progress in this high-stakes setting. 

---
# Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in Large Language Models 

**Authors**: Théo Lasnier, Wissam Antoun, Francis Kulumba, Djamé Seddah  

**Link**: [PDF](https://arxiv.org/pdf/2602.10382)  

**Abstract**: Backdoor attacks pose significant security risks for Large Language Models (LLMs), yet the internal mechanisms by which triggers operate remain poorly understood. We present the first mechanistic analysis of language-switching backdoors, studying the GAPperon model family (1B, 8B, 24B parameters) which contains triggers injected during pretraining that cause output language switching. Using activation patching, we localize trigger formation to early layers (7.5-25% of model depth) and identify which attention heads process trigger information. Our central finding is that trigger-activated heads substantially overlap with heads naturally encoding output language across model scales, with Jaccard indices between 0.18 and 0.66 over the top heads identified. This suggests that backdoor triggers do not form isolated circuits but instead co-opt the model's existing language components. These findings have implications for backdoor defense: detection methods may benefit from monitoring known functional components rather than searching for hidden circuits, and mitigation strategies could potentially leverage this entanglement between injected and natural behaviors. 

---
# The Alignment Bottleneck in Decomposition-Based Claim Verification 

**Authors**: Mahmud Elahi Akhter, Federico Ruggeri, Iman Munire Bilal, Rob Procter, Maria Liakata  

**Link**: [PDF](https://arxiv.org/pdf/2602.10380)  

**Abstract**: Structured claim decomposition is often proposed as a solution for verifying complex, multi-faceted claims, yet empirical results have been inconsistent. We argue that these inconsistencies stem from two overlooked bottlenecks: evidence alignment and sub-claim error profiles. To better understand these factors, we introduce a new dataset of real-world complex claims, featuring temporally bounded evidence and human-annotated sub-claim evidence spans. We evaluate decomposition under two evidence alignment setups: Sub-claim Aligned Evidence (SAE) and Repeated Claim-level Evidence (SRE). Our results reveal that decomposition brings significant performance improvement only when evidence is granular and strictly aligned. By contrast, standard setups that rely on repeated claim-level evidence (SRE) fail to improve and often degrade performance as shown across different datasets and domains (PHEMEPlus, MMM-Fact, COVID-Fact). Furthermore, we demonstrate that in the presence of noisy sub-claim labels, the nature of the error ends up determining downstream robustness. We find that conservative "abstention" significantly reduces error propagation compared to aggressive but incorrect predictions. These findings suggest that future claim decomposition frameworks must prioritize precise evidence synthesis and calibrate the label bias of sub-claim verification models. 

---
# Autonomous Continual Learning of Computer-Use Agents for Environment Adaptation 

**Authors**: Tianci Xue, Zeyi Liao, Tianneng Shi, Zilu Wang, Kai Zhang, Dawn Song, Yu Su, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.10356)  

**Abstract**: Real-world digital environments are highly diverse and dynamic. These characteristics cause agents to frequently encounter unseen scenarios and distribution shifts, making continual learning in specific environments essential for computer-use agents (CUAs). However, a key challenge lies in obtaining high-quality and environment-grounded agent data without relying on costly human annotation. In this work, we introduce ACuRL, an Autonomous Curriculum Reinforcement Learning framework that continually adapts agents to specific environments with zero human data. The agent first explores target environments to acquire initial experiences. During subsequent iterative training, a curriculum task generator leverages these experiences together with feedback from the previous iteration to synthesize new tasks tailored for the agent's current capabilities. To provide reliable reward signals, we introduce CUAJudge, a robust automatic evaluator for CUAs that achieves 93% agreement with human judgments. Empirically, our method effectively enables both intra-environment and cross-environment continual learning, yielding 4-22% performance gains without catastrophic forgetting on existing environments. Further analyses show highly sparse updates (e.g., 20% parameters), which helps explain the effective and robust adaptation. Our data and code are available at this https URL. 

---
# Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence 

**Authors**: Mashrekur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2602.10354)  

**Abstract**: Satellite foundation models produce dense embeddings whose physical interpretability remains poorly understood, limiting their integration into environmental decision systems. Using 12.1 million samples across the Continental United States (2017--2023), we first present a comprehensive interpretability analysis of Google AlphaEarth's 64-dimensional embeddings against 26 environmental variables spanning climate, vegetation, hydrology, temperature, and terrain. Combining linear, nonlinear, and attention-based methods, we show that individual embedding dimensions map onto specific land surface properties, while the full embedding space reconstructs most environmental variables with high fidelity (12 of 26 variables exceed $R^2 > 0.90$; temperature and elevation approach $R^2 = 0.97$). The strongest dimension-variable relationships converge across all three analytical methods and remain robust under spatial block cross-validation (mean $\Delta R^2 = 0.017$) and temporally stable across all seven study years (mean inter-year correlation $r = 0.963$). Building on these validated interpretations, we then developed a Land Surface Intelligence system that implements retrieval-augmented generation over a FAISS-indexed embedding database of 12.1 million vectors, translating natural language environmental queries into satellite-grounded assessments. An LLM-as-Judge evaluation across 360 query--response cycles, using four LLMs in rotating generator, system, and judge roles, achieved weighted scores of $\mu = 3.74 \pm 0.77$ (scale 1--5), with grounding ($\mu = 3.93$) and coherence ($\mu = 4.25$) as the strongest criteria. Our results demonstrate that satellite foundation model embeddings are physically structured representations that can be operationalized for environmental and geospatial intelligence. 

---
# Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label Pairs 

**Authors**: Keenan Pepper, Alex McKenzie, Florin Pop, Stijn Servaes, Martin Leitgab, Mike Vaiana, Judd Rosenblatt, Michael S. A. Graziano, Diogo de Lucena  

**Link**: [PDF](https://arxiv.org/pdf/2602.10352)  

**Abstract**: Self-interpretation methods prompt language models to describe their own internal states, but remain unreliable due to hyperparameter sensitivity. We show that training lightweight adapters on interpretability artifacts, while keeping the LM entirely frozen, yields reliable self-interpretation across tasks and model families. A scalar affine adapter with just $d_\text{model}+1$ parameters suffices: trained adapters generate sparse autoencoder feature labels that outperform the training labels themselves (71% vs 63% generation scoring at 70B scale), identify topics with 94% recall@1 versus 1% for untrained baselines, and decode bridge entities in multi-hop reasoning that appear in neither prompt nor response, surfacing implicit reasoning without chain-of-thought. The learned bias vector alone accounts for 85% of improvement, and simpler adapters generalize better than more expressive alternatives. Controlling for model knowledge via prompted descriptions, we find self-interpretation gains outpace capability gains from 7B to 72B parameters. Our results demonstrate that self-interpretation improves with scale, without modifying the model being interpreted. 

---
# When Less Is More? Diagnosing ASR Predictions in Sardinian via Layer-Wise Decoding 

**Authors**: Domenico De Cristofaro, Alessandro Vietti, Marianne Pouplier, Aleese Block  

**Link**: [PDF](https://arxiv.org/pdf/2602.10350)  

**Abstract**: Recent studies have shown that intermediate layers in multilingual speech models often encode more phonetically accurate representations than the final output layer. In this work, we apply a layer-wise decoding strategy to a pretrained Wav2Vec2 model to investigate how phoneme-level predictions evolve across encoder layers, focusing on Campidanese Sardinian, a low-resource language. We show that truncating upper transformer layers leads to improved Phoneme Error Rates (PER), with the best performance achieved not at the final layer, but two layers earlier. Through fine-grained alignment analysis, we find that intermediate predictions better preserve segmental identity, avoid overgeneration, and reduce certain classes of phonological errors. We also introduce the notion of regressive errors, cases where correct predictions at intermediate layers are overwritten by errors at the final layer. These regressions highlight the limitations of surface-level error metrics and reveal how deeper layers may generalize or abstract away from acoustic detail. Our findings support the use of early-layer probing as a diagnostic tool for ASR models, particularly in low-resource settings where standard evaluation metrics may fail to capture linguistically meaningful behavior. 

---
# Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for Large Language Models 

**Authors**: Arash Gholami Davoodi, Navid Rezazadeh, Seyed Pouyan Mousavi Davoudi, Pouya Pezeshkpour  

**Link**: [PDF](https://arxiv.org/pdf/2602.10346)  

**Abstract**: Large language models (LLMs) must balance diversity and creativity against logical coherence in open-ended generation. Existing truncation-based samplers are effective but largely heuristic, relying mainly on probability mass and entropy while ignoring semantic geometry of the token space. We present Top-W, a geometry-aware truncation rule that uses Wasserstein distance-defined over token-embedding geometry-to keep the cropped distribution close to the original, while explicitly balancing retained probability mass against the entropy of the kept set. Our theory yields a simple closed-form structure for the fixed-potential subset update: depending on the mass-entropy trade-off, the optimal crop either collapses to a single token or takes the form of a one-dimensional prefix that can be found efficiently with a linear scan. We implement Top-W using efficient geometry-based potentials (nearest-set or k-NN) and pair it with an alternating decoding routine that keeps the standard truncation-and-sampling interface unchanged. Extensive experiments on four benchmarks (GSM8K, GPQA, AlpacaEval, and MT-Bench) across three instruction-tuned models show that Top-W consistently outperforms prior state-of-the-art decoding approaches achieving up to 33.7% improvement. Moreover, we find that Top-W not only improves accuracy-focused performance, but also boosts creativity under judge-based open-ended evaluation. 

---
# The Subjectivity of Respect in Police Traffic Stops: Modeling Community Perspectives in Body-Worn Camera Footage 

**Authors**: Preni Golazizian, Elnaz Rahmati, Jackson Trager, Zhivar Sourati, Nona Ghazizadeh, Georgios Chochlakis, Jose Alcocer, Kerby Bennett, Aarya Vijay Devnani, Parsa Hejabi, Harry G. Muttram, Akshay Kiran Padte, Mehrshad Saadatinia, Chenhao Wu, Alireza S. Zaibari, Michael Sierra-Arévalo, Nick Weller, Shrikanth Narayanan, Benjamin A. T. Graham, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2602.10339)  

**Abstract**: Traffic stops are among the most frequent police-civilian interactions, and body-worn cameras (BWCs) provide a unique record of how these encounters unfold. Respect is a central dimension of these interactions, shaping public trust and perceived legitimacy, yet its interpretation is inherently subjective and shaped by lived experience, rendering community-specific perspectives a critical consideration. Leveraging unprecedented access to Los Angeles Police Department BWC footage, we introduce the first large-scale traffic-stop dataset annotated with respect ratings and free-text rationales from multiple perspectives. By sampling annotators from police-affiliated, justice-system-impacted, and non-affiliated Los Angeles residents, we enable the systematic study of perceptual differences across diverse communities. To this end, we (i) develop a domain-specific evaluation rubric grounded in procedural justice theory, LAPD training materials, and extensive fieldwork; (ii) introduce a rubric-driven preference data construction framework for perspective-consistent alignment; and (iii) propose a perspective-aware modeling framework that predicts personalized respect ratings and generates annotator-specific rationales for both officers and civilian drivers from traffic-stop transcripts. Across all three annotator groups, our approach improves both rating prediction performance and rationale alignment. Our perspective-aware framework enables law enforcement to better understand diverse community expectations, providing a vital tool for building public trust and procedural legitimacy. 

---
# Are More Tokens Rational? Inference-Time Scaling in Language Models as Adaptive Resource Rationality 

**Authors**: Zhimin Hu, Riya Roshan, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2602.10329)  

**Abstract**: Human reasoning is shaped by resource rationality -- optimizing performance under constraints. Recently, inference-time scaling has emerged as a powerful paradigm to improve the reasoning performance of Large Language Models by expanding test-time computation. Specifically, instruction-tuned (IT) models explicitly generate long reasoning steps during inference, whereas Large Reasoning Models (LRMs) are trained by reinforcement learning to discover reasoning paths that maximize accuracy. However, it remains unclear whether resource-rationality can emerge from such scaling without explicit reward related to computational costs. We introduce a Variable Attribution Task in which models infer which variables determine outcomes given candidate variables, input-output trials, and predefined logical functions. By varying the number of candidate variables and trials, we systematically manipulate task complexity. Both models exhibit a transition from brute-force to analytic strategies as complexity increases. IT models degrade on XOR and XNOR functions, whereas LRMs remain robust. These findings suggest that models can adjust their reasoning behavior in response to task complexity, even without explicit cost-based reward. It provides compelling evidence that resource rationality is an emergent property of inference-time scaling itself. 

---
# On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models 

**Authors**: Polina Tsvilodub, Jan-Felix Klumpp, Amir Mohammadpour, Jennifer Hu, Michael Franke  

**Link**: [PDF](https://arxiv.org/pdf/2602.10298)  

**Abstract**: This paper investigates whether LMs recruit shared computational mechanisms for general Theory of Mind (ToM) and language-specific pragmatic reasoning in order to contribute to the general question of whether LMs may be said to have emergent "social world models", i.e., representations of mental states that are repurposed across tasks (the functional integration hypothesis). Using behavioral evaluations and causal-mechanistic experiments via functional localization methods inspired by cognitive neuroscience, we analyze LMs' performance across seven subcategories of ToM abilities (Beaudoin et al., 2020) on a substantially larger localizer dataset than used in prior like-minded work. Results from stringent hypothesis-driven statistical testing offer suggestive evidence for the functional integration hypothesis, indicating that LMs may develop interconnected "social world models" rather than isolated competencies. This work contributes novel ToM localizer data, methodological refinements to functional localization techniques, and empirical insights into the emergence of social cognition in artificial systems. 

---
# Learning to Evict from Key-Value Cache 

**Authors**: Luca Moschella, Laura Manduchi, Ozan Sener  

**Link**: [PDF](https://arxiv.org/pdf/2602.10238)  

**Abstract**: The growing size of Large Language Models (LLMs) makes efficient inference challenging, primarily due to the memory demands of the autoregressive Key-Value (KV) cache. Existing eviction or compression methods reduce cost but rely on heuristics, such as recency or past attention scores, which serve only as indirect proxies for a token's future utility and introduce computational overhead. We reframe KV cache eviction as a reinforcement learning (RL) problem: learning to rank tokens by their predicted usefulness for future decoding. To this end, we introduce KV Policy (KVP), a framework of lightweight per-head RL agents trained on pre-computed generation traces using only key and value vectors. Each agent learns a specialized eviction policy guided by future utility, which evaluates the quality of the ranking across all cache budgets, requiring no modifications to the underlying LLM or additional inference. Evaluated across two different model families on the long-context benchmark RULER and the multi-turn dialogue benchmark OASST2-4k, KVP significantly outperforms baselines. Furthermore, zero-shot tests on standard downstream tasks (e.g., LongBench, BOOLQ, ARC) indicate that KVP generalizes well beyond its training distribution and to longer context lengths. These results demonstrate that learning to predict future token utility is a powerful and scalable paradigm for adaptive KV cache management. 

---
# Latent Thoughts Tuning: Bridging Context and Reasoning with Fused Information in Latent Tokens 

**Authors**: Weihao Liu, Dehai Min, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.10229)  

**Abstract**: While explicit Chain-of-Thought (CoT) equips Large Language Models (LLMs) with strong reasoning capabilities, it requires models to verbalize every intermediate step in text tokens, constraining the model thoughts to the discrete vocabulary space. Recently, reasoning in continuous latent space has emerged as a promising alternative, enabling more robust inference and flexible computation beyond discrete token constraints. However, current latent paradigms often suffer from feature collapse and instability, stemming from distribution mismatches when recurrently using hidden states as the input embeddings, or alignment issues when relying on assistant models. To address this, we propose Latent Thoughts Tuning (LT-Tuning), a framework that redefines how latent thoughts are constructed and deployed. Instead of relying solely on raw hidden states, our method introduces a Context-Prediction-Fusion mechanism that jointly leveraging contextual hidden states and predictive semantic guidance from the vocabulary embedding space. Combined with a progressive three-stage curriculum learning pipeline, LT-Tuning also enables dynamically switching between latent and explicit thinking modes. Experiments demonstrate that our method outperforms existing latent reasoning baselines, effectively mitigating feature collapse and achieving robust reasoning accuracy. 

---
# Reviewing the Reviewer: Elevating Peer Review Quality through LLM-Guided Feedback 

**Authors**: Sukannya Purkayastha, Qile Wan, Anne Lauscher, Lizhen Qu, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2602.10118)  

**Abstract**: Peer review is central to scientific quality, yet reliance on simple heuristics -- lazy thinking -- has lowered standards. Prior work treats lazy thinking detection as a single-label task, but review segments may exhibit multiple issues, including broader clarity problems, or specificity issues. Turning detection into actionable improvements requires guideline-aware feedback, which is currently missing. We introduce an LLM-driven framework that decomposes reviews into argumentative segments, identifies issues via a neurosymbolic module combining LLM features with traditional classifiers, and generates targeted feedback using issue-specific templates refined by a genetic algorithm. Experiments show our method outperforms zero-shot LLM baselines and improves review quality by up to 92.4\%. We also release LazyReviewPlus, a dataset of 1,309 sentences labeled for lazy thinking and specificity. 

---
# Diffusion-Pretrained Dense and Contextual Embeddings 

**Authors**: Sedigheh Eslami, Maksim Gaiduk, Markus Krimmel, Louis Milliken, Bo Wang, Denis Bykov  

**Link**: [PDF](https://arxiv.org/pdf/2602.11151)  

**Abstract**: In this report, we introduce pplx-embed, a family of multilingual embedding models that employ multi-stage contrastive learning on a diffusion-pretrained language model backbone for web-scale retrieval. By leveraging bidirectional attention through diffusion-based pretraining, our models capture comprehensive bidirectional context within passages, enabling the use of mean pooling and a late chunking strategy to better preserve global context across long documents. We release two model types: pplx-embed-v1 for standard retrieval, and pplx-embed-context-v1 for contextualized embeddings that incorporate global document context into passage representations. pplx-embed-v1 achieves competitive performance on the MTEB(Multilingual, v2), MTEB(Code), MIRACL, BERGEN, and ToolRet retrieval benchmarks, while pplx-embed-context-v1 sets new records on the ConTEB benchmark. Beyond public benchmarks, pplx-embed-v1 demonstrates strong performance on our internal evaluation suite, which focuses on real-world, large-scale search scenarios over tens of millions of documents. These results validate the models' effectiveness in production environments where retrieval quality and efficiency are critical at scale. 

---
# Weight Decay Improves Language Model Plasticity 

**Authors**: Tessa Han, Sebastian Bordt, Hanlin Zhang, Sham Kakade  

**Link**: [PDF](https://arxiv.org/pdf/2602.11137)  

**Abstract**: The prevailing paradigm in large language model (LLM) development is to pretrain a base model, then perform further training to improve performance and model behavior. However, hyperparameter optimization and scaling laws have been studied primarily from the perspective of the base model's validation loss, ignoring downstream adaptability. In this work, we study pretraining from the perspective of model plasticity, that is, the ability of the base model to successfully adapt to downstream tasks through fine-tuning. We focus on the role of weight decay, a key regularization parameter during pretraining. Through systematic experiments, we show that models trained with larger weight decay values are more plastic, meaning they show larger performance gains when fine-tuned on downstream tasks. This phenomenon can lead to counterintuitive trade-offs where base models that perform worse after pretraining can perform better after fine-tuning. Further investigation of weight decay's mechanistic effects on model behavior reveals that it encourages linearly separable representations, regularizes attention matrices, and reduces overfitting on the training data. In conclusion, this work demonstrates the importance of using evaluation metrics beyond cross-entropy loss for hyperparameter optimization and casts light on the multifaceted role of that a single optimization hyperparameter plays in shaping model behavior. 

---
# Just on Time: Token-Level Early Stopping for Diffusion Language Models 

**Authors**: Zahar Kohut, Severyn Shykula, Dmytro Khamula, Mykola Vysotskyi, Taras Rumezhak, Volodymyr Karpiv  

**Link**: [PDF](https://arxiv.org/pdf/2602.11133)  

**Abstract**: Diffusion language models generate text through iterative refinement, a process that is often computationally inefficient because many tokens reach stability long before the final denoising step. We introduce a training-free, token-level early stopping approach that identifies convergence independently at each position. Our method leverages lightweight signals derived from the model's predictions and local context to dynamically determine when individual tokens can be finalized. This yields adaptive per-token freezing without task-specific fine-tuning, substantially reducing the total number of diffusion steps required. Across diverse benchmarks, spanning mathematical reasoning, general question answering, and scientific understanding, our approach achieves state-of-the-art efficiency gains while preserving generation quality. 

---
# GameDevBench: Evaluating Agentic Capabilities Through Game Development 

**Authors**: Wayne Chi, Yixiong Fang, Arnav Yayavaram, Siddharth Yayavaram, Seth Karten, Qiuhong Anna Wei, Runkun Chen, Alexander Wang, Valerie Chen, Ameet Talwalkar, Chris Donahue  

**Link**: [PDF](https://arxiv.org/pdf/2602.11103)  

**Abstract**: Despite rapid progress on coding agents, progress on their multimodal counterparts has lagged behind. A key challenge is the scarcity of evaluation testbeds that combine the complexity of software development with the need for deep multimodal understanding. Game development provides such a testbed as agents must navigate large, dense codebases while manipulating intrinsically multimodal assets such as shaders, sprites, and animations within a visual game scene. We present GameDevBench, the first benchmark for evaluating agents on game development tasks. GameDevBench consists of 132 tasks derived from web and video tutorials. Tasks require significant multimodal understanding and are complex -- the average solution requires over three times the amount of lines of code and file changes compared to prior software development benchmarks. Agents still struggle with game development, with the best agent solving only 54.5% of tasks. We find a strong correlation between perceived task difficulty and multimodal complexity, with success rates dropping from 46.9% on gameplay-oriented tasks to 31.6% on 2D graphics tasks. To improve multimodal capability, we introduce two simple image and video-based feedback mechanisms for agents. Despite their simplicity, these methods consistently improve performance, with the largest change being an increase in Claude Sonnet 4.5's performance from 33.3% to 47.7%. We release GameDevBench publicly to support further research into agentic game development. 

---
# Chatting with Images for Introspective Visual Thinking 

**Authors**: Junfei Wu, Jian Guan, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, Tienie Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.11073)  

**Abstract**: Current large vision-language models (LVLMs) typically rely on text-only reasoning based on a single-pass visual encoding, which often leads to loss of fine-grained visual information. Recently the proposal of ''thinking with images'' attempts to alleviate this limitation by manipulating images via external tools or code; however, the resulting visual states are often insufficiently grounded in linguistic semantics, impairing effective cross-modal alignment - particularly when visual semantics or geometric relationships must be reasoned over across distant regions or multiple images. To address these challenges, we propose ''chatting with images'', a new framework that reframes visual manipulation as language-guided feature modulation. Under the guidance of expressive language prompts, the model dynamically performs joint re-encoding over multiple image regions, enabling tighter coupling between linguistic reasoning and visual state updates. We instantiate this paradigm in ViLaVT, a novel LVLM equipped with a dynamic vision encoder explicitly designed for such interactive visual reasoning, and trained it with a two-stage curriculum combining supervised fine-tuning and reinforcement learning to promote effective reasoning behaviors. Extensive experiments across eight benchmarks demonstrate that ViLaVT achieves strong and consistent improvements, with particularly pronounced gains on complex multi-image and video-based spatial reasoning tasks. 

---
# GraphSeek: Next-Generation Graph Analytics with LLMs 

**Authors**: Maciej Besta, Łukasz Jarmocik, Orest Hrycyna, Shachar Klaiman, Konrad Mączka, Robert Gerstenberger, Jürgen Müller, Piotr Nyczyk, Hubert Niewiadomski, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2602.11052)  

**Abstract**: Graphs are foundational across domains but remain hard to use without deep expertise. LLMs promise accessible natural language (NL) graph analytics, yet they fail to process industry-scale property graphs effectively and efficiently: such datasets are large, highly heterogeneous, structurally complex, and evolve dynamically. To address this, we devise a novel abstraction for complex multi-query analytics over such graphs. Its key idea is to replace brittle generation of graph queries directly from NL with planning over a Semantic Catalog that describes both the graph schema and the graph operations. Concretely, this induces a clean separation between a Semantic Plane for LLM planning and broader reasoning, and an Execution Plane for deterministic, database-grade query execution over the full dataset and tool implementations. This design yields substantial gains in both token efficiency and task effectiveness even with small-context LLMs. We use this abstraction as the basis of the first LLM-enhanced graph analytics framework called GraphSeek. GraphSeek achieves substantially higher success rates (e.g., 86% over enhanced LangChain) and points toward the next generation of affordable and accessible graph analytics that unify LLM reasoning with database-grade execution over large and complex property graphs. 

---
# Learning Page Order in Shuffled WOO Releases 

**Authors**: Efe Kahraman, Giulio Tosato  

**Link**: [PDF](https://arxiv.org/pdf/2602.11040)  

**Abstract**: We investigate document page ordering on 5,461 shuffled WOO documents (Dutch freedom of information releases) using page embeddings. These documents are heterogeneous collections such as emails, legal texts, and spreadsheets compiled into single PDFs, where semantic ordering signals are unreliable. We compare five methods, including pointer networks, seq2seq transformers, and specialized pairwise ranking models. The best performing approach successfully reorders documents up to 15 pages, with Kendall's tau ranging from 0.95 for short documents (2-5 pages) to 0.72 for 15 page documents. We observe two unexpected failures: seq2seq transformers fail to generalize on long documents (Kendall's tau drops from 0.918 on 2-5 pages to 0.014 on 21-25 pages), and curriculum learning underperforms direct training by 39% on long documents. Ablation studies suggest learned positional encodings are one contributing factor to seq2seq failure, though the degradation persists across all encoding variants, indicating multiple interacting causes. Attention pattern analysis reveals that short and long documents require fundamentally different ordering strategies, explaining why curriculum learning fails. Model specialization achieves substantial improvements on longer documents (+0.21 tau). 

---
# ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression 

**Authors**: Ammar Ali, Baher Mohammad, Denis Makhov, Dmitriy Shopkhoev, Magauiya Zhussip, Stamatios Lefkimmiatis  

**Link**: [PDF](https://arxiv.org/pdf/2602.11008)  

**Abstract**: We present ROCKET, a training-free model compression method that achieves state-of-the-art performance in comparison with factorization, structured-sparsification and dynamic compression baselines. Operating under a global compression budget, ROCKET comprises two key innovations: First, it formulates layer-wise compression allocation as a multi-choice knapsack problem, selecting the optimal compression level for each layer to minimize total reconstruction error while adhering to a target model size. Second, it introduces a single-step sparse matrix factorization inspired by dictionary learning: using only a small calibration set, it sparsifies weight coefficients based on activation-weights sensitivity and then updates the dictionary in closed form via least squares bypassing iterative optimization, sparse coding, or backpropagation entirely. ROCKET consistently outperforms existing compression approaches across different model architectures at 20-50\% compression rates. Notably, it retains over 90\% of the original model's performance at 30\% compression without any fine-tuning. Moreover, when applying a light fine-tuning phase, recovery is substantially enhanced: for instance, compressing Qwen3-14B to an 8B-parameter model and healing it with just 30 million tokens yields performance nearly on par with the original Qwen3-8B. The code for ROCKET is at this http URL. 

---
# The emergence of numerical representations in communicating artificial agents 

**Authors**: Daniela Mihai, Lucas Weber, Francesca Franzon  

**Link**: [PDF](https://arxiv.org/pdf/2602.10996)  

**Abstract**: Human languages provide efficient systems for expressing numerosities, but whether the sheer pressure to communicate is enough for numerical representations to arise in artificial agents, and whether the emergent codes resemble human numerals at all, remains an open question. We study two neural network-based agents that must communicate numerosities in a referential game using either discrete tokens or continuous sketches, thus exploring both symbolic and iconic representations. Without any pre-defined numeric concepts, the agents achieve high in-distribution communication accuracy in both communication channels and converge on high-precision symbol-meaning mappings. However, the emergent code is non-compositional: the agents fail to derive systematic messages for unseen numerosities, typically reusing the symbol of the highest trained numerosity (discrete), or collapsing extrapolated values onto a single sketch (continuous). We conclude that the communication pressure alone suffices for precise transmission of learned numerosities, but additional pressures are needed to yield compositional codes and generalisation abilities. 

---
# Rotary Positional Embeddings as Phase Modulation: Theoretical Bounds on the RoPE Base for Long-Context Transformers 

**Authors**: Feilong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10959)  

**Abstract**: Rotary positional embeddings (RoPE) are widely used in large language models to encode token positions through multiplicative rotations, yet their behavior at long context lengths remains poorly characterized. In this work, we reinterpret RoPE as phase modulation applied to a bank of complex oscillators, enabling analysis through classical signal processing theory.
Under this formulation, we derive principled lower bounds on the RoPE base parameter that are necessary to preserve positional coherence over a target context length. These include a fundamental aliasing bound, analogous to a Nyquist limit, and a DC-component stability bound that constrains phase drift in low-frequency positional modes. We further extend this analysis to deep transformers, showing that repeated rotary modulation across layers compounds angular misalignment, tightening the base requirement as depth increases.
Complementing these results, we derive a precision-dependent upper bound on the RoPE base arising from finite floating-point resolution. Beyond this limit, incremental phase updates become numerically indistinguishable, leading to positional erasure even in the absence of aliasing. Together, the lower and upper bounds define a precision- and depth-dependent feasibility region a Goldilocks zone for long-context transformers.
We validate the framework through a comprehensive case study of state-of-the-art models, including LLaMA, Mistral, and DeepSeek variants, showing that observed successes, failures, and community retrofits align closely with the predicted bounds. Notably, models that violate the stability bound exhibit attention collapse and long-range degradation, while attempts to scale beyond one million tokens encounter a hard precision wall independent of architecture or training. 

---
# Training-Induced Bias Toward LLM-Generated Content in Dense Retrieval 

**Authors**: William Xion, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2602.10833)  

**Abstract**: Dense retrieval is a promising approach for acquiring relevant context or world knowledge in open-domain natural language processing tasks and is now widely used in information retrieval applications. However, recent reports claim a broad preference for text generated by large language models (LLMs). This bias is called "source bias", and it has been hypothesized that lower perplexity contributes to this effect. In this study, we revisit this claim by conducting a controlled evaluation to trace the emergence of such preferences across training stages and data sources. Using parallel human- and LLM-generated counterparts of the SciFact and Natural Questions (NQ320K) datasets, we compare unsupervised checkpoints with models fine-tuned using in-domain human text, in-domain LLM-generated text, and MS MARCO. Our results show the following: 1) Unsupervised retrievers do not exhibit a uniform pro-LLM preference. The direction and magnitude depend on the dataset. 2) Across the settings tested, supervised fine-tuning on MS MARCO consistently shifts the rankings toward LLM-generated text. 3) In-domain fine-tuning produces dataset-specific and inconsistent shifts in preference. 4) Fine-tuning on LLM-generated corpora induces a pronounced pro-LLM bias. Finally, a retriever-centric perplexity probe involving the reattachment of a language modeling head to the fine-tuned dense retriever encoder indicates agreement with relevance near chance, thereby weakening the explanatory power of perplexity. Our study demonstrates that source bias is a training-induced phenomenon rather than an inherent property of dense retrievers. 

---
# Calliope: A TTS-based Narrated E-book Creator Ensuring Exact Synchronization, Privacy, and Layout Fidelity 

**Authors**: Hugo L. Hammer, Vajira Thambawita, Pål Halvorsen  

**Link**: [PDF](https://arxiv.org/pdf/2602.10735)  

**Abstract**: A narrated e-book combines synchronized audio with digital text, highlighting the currently spoken word or sentence during playback. This format supports early literacy and assists individuals with reading challenges, while also allowing general readers to seamlessly switch between reading and listening. With the emergence of natural-sounding neural Text-to-Speech (TTS) technology, several commercial services have been developed to leverage these technology for converting standard text e-books into high-quality narrated e-books. However, no open-source solutions currently exist to perform this task. In this paper, we present Calliope, an open-source framework designed to fill this gap. Our method leverages state-of-the-art open-source TTS to convert a text e-book into a narrated e-book in the EPUB 3 Media Overlay format. The method offers several innovative steps: audio timestamps are captured directly during TTS, ensuring exact synchronization between narration and text highlighting; the publisher's original typography, styling, and embedded media are strictly preserved; and the entire pipeline operates offline. This offline capability eliminates recurring API costs, mitigates privacy concerns, and avoids copyright compliance issues associated with cloud-based services. The framework currently supports the state-of-the-art open-source TTS systems XTTS-v2 and Chatterbox. A potential alternative approach involves first generating narration via TTS and subsequently synchronizing it with the text using forced alignment. However, while our method ensures exact synchronization, our experiments show that forced alignment introduces drift between the audio and text highlighting significant enough to degrade the reading experience. Source code and usage instructions are available at this https URL. 

---
# SnapMLA: Efficient Long-Context MLA Decoding via Hardware-Aware FP8 Quantized Pipelining 

**Authors**: Yifan Zhang, Zunhai Su, Shuhao Hu, Rui Yang, Wei Wu, Yulei Qian, Yuchen Xie, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2602.10718)  

**Abstract**: While FP8 attention has shown substantial promise in innovations like FlashAttention-3, its integration into the decoding phase of the DeepSeek Multi-head Latent Attention (MLA) architecture presents notable challenges. These challenges include numerical heterogeneity arising from the decoupling of positional embeddings, misalignment of quantization scales in FP8 PV GEMM, and the need for optimized system-level support. In this paper, we introduce SnapMLA, an FP8 MLA decoding framework optimized to improve long-context efficiency through the following hardware-aware algorithm-kernel co-optimization techniques: (i) RoPE-Aware Per-Token KV Quantization, where the RoPE part is maintained in high precision, motivated by our comprehensive analysis of the heterogeneous quantization sensitivity inherent to the MLA KV cache. Furthermore, per-token granularity is employed to align with the autoregressive decoding process and maintain quantization accuracy. (ii) Quantized PV Computation Pipeline Reconstruction, which resolves the misalignment of quantization scale in FP8 PV computation stemming from the shared KV structure of the MLA KV cache. (iii) End-to-End Dataflow Optimization, where we establish an efficient data read-and-write workflow using specialized kernels, ensuring efficient data flow and performance gains. Extensive experiments on state-of-the-art MLA LLMs show that SnapMLA achieves up to a 1.91x improvement in throughput, with negligible risk of performance degradation in challenging long-context tasks, including mathematical reasoning and code generation benchmarks. Code is available at this https URL. 

---
# RE-LLM: Refining Empathetic Speech-LLM Responses by Integrating Emotion Nuance 

**Authors**: Jing-Han Chen, Bo-Hao Su, Ya-Tse Wu, Chi-Chun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.10716)  

**Abstract**: With generative AI advancing, empathy in human-AI interaction is essential. While prior work focuses on emotional reflection, emotional exploration, key to deeper engagement, remains overlooked. Existing LLMs rely on text which captures limited emotion nuances. To address this, we propose RE-LLM, a speech-LLM integrating dimensional emotion embeddings and auxiliary learning. Experiments show statistically significant gains in empathy metrics across three datasets. RE-LLM relatively improves the Emotional Reaction score by 14.79% and 6.76% compared to text-only and speech-LLM baselines on ESD. Notably, it raises the Exploration score by 35.42% and 3.91% on IEMOCAP, 139.28% and 9.83% on ESD, and 60.95% and 22.64% on MSP-PODCAST. It also boosts unweighted accuracy by 5.4% on IEMOCAP, 2.3% on ESD, and 6.9% on MSP-PODCAST in speech emotion recognition. These results highlight the enriched emotional understanding and improved empathetic response generation of RE-LLM. 

---
# To Think or Not To Think, That is The Question for Large Reasoning Models in Theory of Mind Tasks 

**Authors**: Nanxu Gong, Haotian Li, Sixun Dong, Jianxun Lian, Yanjie Fu, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.10625)  

**Abstract**: Theory of Mind (ToM) assesses whether models can infer hidden mental states such as beliefs, desires, and intentions, which is essential for natural social interaction. Although recent progress in Large Reasoning Models (LRMs) has boosted step-by-step inference in mathematics and coding, it is still underexplored whether this benefit transfers to socio-cognitive skills. We present a systematic study of nine advanced Large Language Models (LLMs), comparing reasoning models with non-reasoning models on three representative ToM benchmarks. The results show that reasoning models do not consistently outperform non-reasoning models and sometimes perform worse. A fine-grained analysis reveals three insights. First, slow thinking collapses: accuracy significantly drops as responses grow longer, and larger reasoning budgets hurt performance. Second, moderate and adaptive reasoning benefits performance: constraining reasoning length mitigates failure, while distinct success patterns demonstrate the necessity of dynamic adaptation. Third, option matching shortcut: when multiple choice options are removed, reasoning models improve markedly, indicating reliance on option matching rather than genuine deduction. We also design two intervention approaches: Slow-to-Fast (S2F) adaptive reasoning and Think-to-Match (T2M) shortcut prevention to further verify and mitigate the problems. With all results, our study highlights the advancement of LRMs in formal reasoning (e.g., math, code) cannot be fully transferred to ToM, a typical task in social reasoning. We conclude that achieving robust ToM requires developing unique capabilities beyond existing reasoning methods. 

---
# ISD-Agent-Bench: A Comprehensive Benchmark for Evaluating LLM-based Instructional Design Agents 

**Authors**: YoungHoon Jeon, Suwan Kim, Haein Son, Sookbun Lee, Yeil Jeong, Unggi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.10620)  

**Abstract**: Large Language Model (LLM) agents have shown promising potential in automating Instructional Systems Design (ISD), a systematic approach to developing educational programs. However, evaluating these agents remains challenging due to the lack of standardized benchmarks and the risk of LLM-as-judge bias. We present ISD-Agent-Bench, a comprehensive benchmark comprising 25,795 scenarios generated via a Context Matrix framework that combines 51 contextual variables across 5 categories with 33 ISD sub-steps derived from the ADDIE model. To ensure evaluation reliability, we employ a multi-judge protocol using diverse LLMs from different providers, achieving high inter-judge reliability. We compare existing ISD agents with novel agents grounded in classical ISD theories such as ADDIE, Dick \& Carey, and Rapid Prototyping ISD. Experiments on 1,017 test scenarios demonstrate that integrating classical ISD frameworks with modern ReAct-style reasoning achieves the highest performance, outperforming both pure theory-based agents and technique-only approaches. Further analysis reveals that theoretical quality strongly correlates with benchmark performance, with theory-based agents showing significant advantages in problem-centered design and objective-assessment alignment. Our work provides a foundation for systematic LLM-based ISD research. 

---
# TestExplora: Benchmarking LLMs for Proactive Bug Discovery via Repository-Level Test Generation 

**Authors**: Steven Liu, Jane Luo, Xin Zhang, Aofan Liu, Hao Liu, Jie Wu, Ziyang Huang, Yangyu Huang, Yu Kang, Scarlett Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.10471)  

**Abstract**: Given that Large Language Models (LLMs) are increasingly applied to automate software development, comprehensive software assurance spans three distinct goals: regression prevention, reactive reproduction, and proactive discovery. Current evaluations systematically overlook the third goal. Specifically, they either treat existing code as ground truth (a compliance trap) for regression prevention, or depend on post-failure artifacts (e.g., issue reports) for bug reproduction-so they rarely surface defects before failures. To bridge this gap, we present TestExplora, a benchmark designed to evaluate LLMs as proactive testers within full-scale, realistic repository environments. TestExplora contains 2,389 tasks from 482 repositories and hides all defect-related signals. Models must proactively find bugs by comparing implementations against documentation-derived intent, using documentation as the oracle. Furthermore, to keep evaluation sustainable and reduce leakage, we propose continuous, time-aware data collection. Our evaluation reveals a significant capability gap: state-of-the-art models achieve a maximum Fail-to-Pass (F2P) rate of only 16.06%. Further analysis indicates that navigating complex cross-module interactions and leveraging agentic exploration are critical to advancing LLMs toward autonomous software quality assurance. Consistent with this, SWEAgent instantiated with GPT-5-mini achieves an F2P of 17.27% and an F2P@5 of 29.7%, highlighting the effectiveness and promise of agentic exploration in proactive bug discovery tasks. 

---
# The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis 

**Authors**: Peiran Wang, Xinfeng Li, Chong Xiang, Jinghuai Zhang, Ying Li, Lixia Zhang, Xiaofeng Wang, Yuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.10453)  

**Abstract**: The evolution of Large Language Models (LLMs) has resulted in a paradigm shift towards autonomous agents, necessitating robust security against Prompt Injection (PI) vulnerabilities where untrusted inputs hijack agent behaviors. This SoK presents a comprehensive overview of the PI landscape, covering attacks, defenses, and their evaluation practices. Through a systematic literature review and quantitative analysis, we establish taxonomies that categorize PI attacks by payload generation strategies (heuristic vs. optimization) and defenses by intervention stages (text, model, and execution levels). Our analysis reveals a key limitation shared by many existing defenses and benchmarks: they largely overlook context-dependent tasks, in which agents are authorized to rely on runtime environmental observations to determine actions. To address this gap, we introduce AgentPI, a new benchmark designed to systematically evaluate agent behavior under context-dependent interaction settings. Using AgentPI, we empirically evaluate representative defenses and show that no single approach can simultaneously achieve high trustworthiness, high utility, and low latency. Moreover, we show that many defenses appear effective under existing benchmarks by suppressing contextual inputs, yet fail to generalize to realistic agent settings where context-dependent reasoning is essential. This SoK distills key takeaways and open research problems, offering structured guidance for future research and practical deployment of secure LLM agents. 

---
# Control Reinforcement Learning: Token-Level Mechanistic Analysis via Learned SAE Feature Steering 

**Authors**: Seonglae Cho, Zekun Wu, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2602.10437)  

**Abstract**: Sparse autoencoders (SAEs) decompose language model activations into interpretable features, but existing methods reveal only which features activate, not which change model outputs when amplified. We introduce Control Reinforcement Learning (CRL), which trains a policy to select SAE features for steering at each token, producing interpretable intervention logs: the learned policy identifies features that change model outputs when amplified. Adaptive Feature Masking encourages diverse feature discovery while preserving singlefeature interpretability. The framework yields new analysis capabilities: branch point tracking locates tokens where feature choice determines output correctness; critic trajectory analysis separates policy limitations from value estimation errors; layer-wise comparison reveals syntactic features in early layers and semantic features in later layers. On Gemma-2 2B across MMLU, BBQ, GSM8K, HarmBench, and XSTest, CRL achieves improvements while providing per-token intervention logs. These results establish learned feature steering as a mechanistic interpretability tool that complements static feature analysis with dynamic intervention probes 

---
# AI-rithmetic 

**Authors**: Alex Bie, Travis Dick, Alex Kulesza, Prabhakar Raghavan, Vinod Raman, Sergei Vassilvitskii  

**Link**: [PDF](https://arxiv.org/pdf/2602.10416)  

**Abstract**: Modern AI systems have been successfully deployed to win medals at international math competitions, assist with research workflows, and prove novel technical lemmas. However, despite their progress at advanced levels of mathematics, they remain stubbornly bad at basic arithmetic, consistently failing on the simple task of adding two numbers. We present a systematic investigation of this phenomenon. We demonstrate empirically that all frontier models suffer significantly degraded accuracy for integer addition as the number of digits increases. Furthermore, we show that most errors made by these models are highly interpretable and can be attributed to either operand misalignment or a failure to correctly carry; these two error classes explain 87.9%, 62.9%, and 92.4% of Claude Opus 4.1, GPT-5, and Gemini 2.5 Pro errors, respectively. Finally, we show that misalignment errors are frequently related to tokenization, and that carrying errors appear largely as independent random failures. 

---
# Gated Removal of Normalization in Transformers Enables Stable Training and Efficient Inference 

**Authors**: Andrei Kanavalau, Carmen Amo Alonso, Sanjay Lall  

**Link**: [PDF](https://arxiv.org/pdf/2602.10408)  

**Abstract**: Normalization is widely viewed as essential for stabilizing Transformer training. We revisit this assumption for pre-norm Transformers and ask to what extent sample-dependent normalization is needed inside Transformer blocks. We introduce TaperNorm, a drop-in replacement for RMSNorm/LayerNorm that behaves exactly like the standard normalizer early in training and then smoothly tapers to a learned sample-independent linear/affine map. A single global gate is held at $g{=}1$ during gate warmup, used to calibrate the scaling branch via EMAs, and then cosine-decayed to $g{=}0$, at which point per-token statistics vanish and the resulting fixed scalings can be folded into adjacent linear projections. Our theoretical and empirical results isolate scale anchoring as the key role played by output normalization: as a (near) $0$-homogeneous map it removes radial gradients at the output, whereas without such an anchor cross-entropy encourages unbounded logit growth (``logit chasing''). We further show that a simple fixed-target auxiliary loss on the pre-logit residual-stream scale provides an explicit alternative anchor and can aid removal of the final normalization layer. Empirically, TaperNorm matches normalized baselines under identical setups while eliminating per-token statistics and enabling these layers to be folded into adjacent linear projections at inference. On an efficiency microbenchmark, folding internal scalings yields up to $1.22\times$ higher throughput in last-token logits mode. These results take a step towards norm-free Transformers while identifying the special role output normalization plays. 

---
# Modular Multi-Task Learning for Chemical Reaction Prediction 

**Authors**: Jiayun Pang, Ahmed M. Zaitoun, Xacobe Couso Cambeiro, Ivan Vulić  

**Link**: [PDF](https://arxiv.org/pdf/2602.10404)  

**Abstract**: Adapting large language models (LLMs) trained on broad organic chemistry to smaller, domain-specific reaction datasets is a key challenge in chemical and pharmaceutical R&D. Effective specialisation requires learning new reaction knowledge while preserving general chemical understanding across related tasks. Here, we evaluate Low-Rank Adaptation (LoRA) as a parameter-efficient alternative to full fine-tuning for organic reaction prediction on limited, complex datasets. Using USPTO reaction classes and challenging C-H functionalisation reactions, we benchmark forward reaction prediction, retrosynthesis and reagent prediction. LoRA achieves accuracy comparable to full fine-tuning while effectively mitigating catastrophic forgetting and better preserving multi-task performance. Both fine-tuning approaches generalise beyond training distributions, producing plausible alternative solvent predictions. Notably, C-H functionalisation fine-tuning reveals that LoRA and full fine-tuning encode subtly different reactivity patterns, suggesting more effective reaction-specific adaptation with LoRA. As LLMs continue to scale, our results highlight the practicality of modular, parameter-efficient fine-tuning strategies for their flexible deployment for chemistry applications. 

---
# Hardware Co-Design Scaling Laws via Roofline Modelling for On-Device LLMs 

**Authors**: Luoyang Sun, Jiwen Jiang, Yifeng Ding, Fengfa Li, Yan Song, Haifeng Zhang, Jian Ying, Lei Ren, Kun Zhan, Wei Chen, Yan Xie, Cheng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.10377)  

**Abstract**: Vision-Language-Action Models (VLAs) have emerged as a key paradigm of Physical AI and are increasingly deployed in autonomous vehicles, robots, and smart spaces. In these resource-constrained on-device settings, selecting an appropriate large language model (LLM) backbone is a critical challenge: models must balance accuracy with strict inference latency and hardware efficiency constraints. This makes hardware-software co-design a game-changing requirement for on-device LLM deployment, where each hardware platform demands a tailored architectural solution. We propose a hardware co-design law that jointly captures model accuracy and inference performance. Specifically, we model training loss as an explicit function of architectural hyperparameters and characterise inference latency via roofline modelling. We empirically evaluate 1,942 candidate architectures on NVIDIA Jetson Orin, training 170 selected models for 10B tokens each to fit a scaling law relating architecture to training loss. By coupling this scaling law with latency modelling, we establish a direct accuracy-latency correspondence and identify the Pareto frontier for hardware co-designed LLMs. We further formulate architecture search as a joint optimisation over precision and performance, deriving feasible design regions under industrial hardware and application budgets. Our approach reduces architecture selection from months to days. At the same latency as Qwen2.5-0.5B on the target hardware, our co-designed architecture achieves 19.42% lower perplexity on WikiText-2. To our knowledge, this is the first principled and operational framework for hardware co-design scaling laws in on-device LLM deployment. We will make the code and related checkpoints publicly available. 

---
# Discovering Differences in Strategic Behavior Between Humans and LLMs 

**Authors**: Caroline Wang, Daniel Kasenberg, Kim Stachenfeld, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2602.10324)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in social and strategic scenarios, it becomes critical to understand where and why their behavior diverges from that of humans. While behavioral game theory (BGT) provides a framework for analyzing behavior, existing models do not fully capture the idiosyncratic behavior of humans or black-box, non-human agents like LLMs. We employ AlphaEvolve, a cutting-edge program discovery tool, to directly discover interpretable models of human and LLM behavior from data, thereby enabling open-ended discovery of structural factors driving human and LLM behavior. Our analysis on iterated rock-paper-scissors reveals that frontier LLMs can be capable of deeper strategic behavior than humans. These results provide a foundation for understanding structural differences driving differences in human and LLM behavior in strategic interactions. 

---
# MLDocRAG: Multimodal Long-Context Document Retrieval Augmented Generation 

**Authors**: Yongyue Zhang, Yaxiong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10271)  

**Abstract**: Understanding multimodal long-context documents that comprise multimodal chunks such as paragraphs, figures, and tables is challenging due to (1) cross-modal heterogeneity to localize relevant information across modalities, (2) cross-page reasoning to aggregate dispersed evidence across pages. To address these challenges, we are motivated to adopt a query-centric formulation that projects cross-modal and cross-page information into a unified query representation space, with queries acting as abstract semantic surrogates for heterogeneous multimodal content. In this paper, we propose a Multimodal Long-Context Document Retrieval Augmented Generation (MLDocRAG) framework that leverages a Multimodal Chunk-Query Graph (MCQG) to organize multimodal document content around semantically rich, answerable queries. MCQG is constructed via a multimodal document expansion process that generates fine-grained queries from heterogeneous document chunks and links them to their corresponding content across modalities and pages. This graph-based structure enables selective, query-centric retrieval and structured evidence aggregation, thereby enhancing grounding and coherence in long-context multimodal question answering. Experiments on datasets MMLongBench-Doc and LongDocURL demonstrate that MLDocRAG consistently improves retrieval quality and answer accuracy, demonstrating its effectiveness for long-context multimodal understanding. 

---
# Blockwise Advantage Estimation for Multi-Objective RL with Verifiable Rewards 

**Authors**: Kirill Pavlenko, Alexander Golubev, Simon Karasik, Boris Yangel  

**Link**: [PDF](https://arxiv.org/pdf/2602.10231)  

**Abstract**: Group Relative Policy Optimization (GRPO) assigns a single scalar advantage to all tokens in a completion. For structured generations with explicit segments and objectives, this couples unrelated reward signals across segments, leading to objective interference and misattributed credit. We propose Blockwise Advantage Estimation, a family of GRPO-compatible methods that assigns each objective its own advantage and applies it only to the tokens in the corresponding text block, reducing reliance on hand-designed scalar rewards and scaling naturally to additional objectives. A key challenge is estimating advantages for later blocks whose rewards are conditioned on sampled prefixes; standard unbiased approaches require expensive nested rollouts from intermediate states. Concretely, we introduce an Outcome-Conditioned Baseline that approximates intermediate state values using only within-group statistics by stratifying samples according to a prefix-derived intermediate outcome. On math tasks with uncertainty estimation, our method mitigates reward interference, is competitive with a state-of-the-art reward-designed approach, and preserves test-time gains from confidence-weighted ensembling. More broadly, it provides a modular recipe for optimizing sequential objectives in structured generations without additional rollouts. 

---
# Towards Autonomous Mathematics Research 

**Authors**: Tony Feng, Trieu H. Trinh, Garrett Bingham, Dawsen Hwang, Yuri Chervonyi, Junehyuk Jung, Joonkyung Lee, Carlo Pagano, Sang-hyun Kim, Federico Pasqualotto, Sergei Gukov, Jonathan N. Lee, Junsu Kim, Kaiying Hou, Golnaz Ghiasi, Yi Tay, YaGuang Li, Chenkai Kuang, Yuan Liu, Hanzhao, Evan Zheran Liu, Nigamaa Nayakanti, Xiaomeng Yang, Heng-tze Cheng, Demis Hassabis, Koray Kavukcuoglu, Quoc V. Le, Thang Luong  

**Link**: [PDF](https://arxiv.org/pdf/2602.10177)  

**Abstract**: Recent advances in foundational models have yielded reasoning systems capable of achieving a gold-medal standard at the International Mathematical Olympiad. The transition from competition-level problem-solving to professional research, however, requires navigating vast literature and constructing long-horizon proofs. In this work, we introduce Aletheia, a math research agent that iteratively generates, verifies, and revises solutions end-to-end in natural language. Specifically, Aletheia is powered by an advanced version of Gemini Deep Think for challenging reasoning problems, a novel inference-time scaling law that extends beyond Olympiad-level problems, and intensive tool use to navigate the complexities of mathematical research. We demonstrate the capability of Aletheia from Olympiad problems to PhD-level exercises and most notably, through several distinct milestones in AI-assisted mathematics research: (a) a research paper (Feng26) generated by AI without any human intervention in calculating certain structure constants in arithmetic geometry called eigenweights; (b) a research paper (LeeSeo26) demonstrating human-AI collaboration in proving bounds on systems of interacting particles called independent sets; and (c) an extensive semi-autonomous evaluation (Feng et al., 2026a) of 700 open problems on Bloom's Erdos Conjectures database, including autonomous solutions to four open questions. In order to help the public better understand the developments pertaining to AI and mathematics, we suggest codifying standard levels quantifying autonomy and novelty of AI-assisted results. We conclude with reflections on human-AI collaboration in mathematics. 

---
# Omni-Safety under Cross-Modality Conflict: Vulnerabilities, Dynamics Mechanisms and Efficient Alignment 

**Authors**: Kun Wang, Zherui Li, Zhenhong Zhou, Yitong Zhang, Yan Mi, Kun Yang, Yiming Zhang, Junhao Dong, Zhongxiang Sun, Qiankun Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10161)  

**Abstract**: Omni-modal Large Language Models (OLLMs) greatly expand LLMs' multimodal capabilities but also introduce cross-modal safety risks. However, a systematic understanding of vulnerabilities in omni-modal interactions remains lacking. To bridge this gap, we establish a modality-semantics decoupling principle and construct the AdvBench-Omni dataset, which reveals a significant vulnerability in OLLMs. Mechanistic analysis uncovers a Mid-layer Dissolution phenomenon driven by refusal vector magnitude shrinkage, alongside the existence of a modal-invariant pure refusal direction. Inspired by these insights, we extract a golden refusal vector using Singular Value Decomposition and propose OmniSteer, which utilizes lightweight adapters to modulate intervention intensity adaptively. Extensive experiments show that our method not only increases the Refusal Success Rate against harmful inputs from 69.9% to 91.2%, but also effectively preserves the general capabilities across all modalities. Our code is available at: this https URL. 

---
# VERA: Identifying and Leveraging Visual Evidence Retrieval Heads in Long-Context Understanding 

**Authors**: Rongcan Pei, Huan Li, Fang Guo, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.10146)  

**Abstract**: While Vision-Language Models (VLMs) have shown promise in textual understanding, they face significant challenges when handling long context and complex reasoning tasks. In this paper, we dissect the internal mechanisms governing long-context processing in VLMs to understand their performance bottlenecks. Through the lens of attention analysis, we identify specific Visual Evidence Retrieval (VER) Heads - a sparse, dynamic set of attention heads critical for locating visual cues during reasoning, distinct from static OCR heads. We demonstrate that these heads are causal to model performance; masking them leads to significant degradation. Leveraging this discovery, we propose VERA (Visual Evidence Retrieval Augmentation), a training-free framework that detects model uncertainty (i.e., entropy) to trigger the explicit verbalization of visual evidence attended by VER heads. Comprehensive experiments demonstrate that VERA significantly improves long-context understanding of open-source VLMs: it yields an average relative improvement of 21.3% on Qwen3-VL-8B-Instruct and 20.1% on GLM-4.1V-Thinking across five benchmarks. 

---
# Multimodal Information Fusion for Chart Understanding: A Survey of MLLMs -- Evolution, Limitations, and Cognitive Enhancement 

**Authors**: Zhihang Yi, Jian Zhao, Jiancheng Lv, Tao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.10138)  

**Abstract**: Chart understanding is a quintessential information fusion task, requiring the seamless integration of graphical and textual data to extract meaning. The advent of Multimodal Large Language Models (MLLMs) has revolutionized this domain, yet the landscape of MLLM-based chart analysis remains fragmented and lacks systematic organization. This survey provides a comprehensive roadmap of this nascent frontier by structuring the domain's core components. We begin by analyzing the fundamental challenges of fusing visual and linguistic information in charts. We then categorize downstream tasks and datasets, introducing a novel taxonomy of canonical and non-canonical benchmarks to highlight the field's expanding scope. Subsequently, we present a comprehensive evolution of methodologies, tracing the progression from classic deep learning techniques to state-of-the-art MLLM paradigms that leverage sophisticated fusion strategies. By critically examining the limitations of current models, particularly their perceptual and reasoning deficits, we identify promising future directions, including advanced alignment techniques and reinforcement learning for cognitive enhancement. This survey aims to equip researchers and practitioners with a structured understanding of how MLLMs are transforming chart information fusion and to catalyze progress toward more robust and reliable systems. 

---
# Reverse-Engineering Model Editing on Language Models 

**Authors**: Zhiyu Sun, Minrui Luo, Yu Wang, Zhili Chen, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2602.10134)  

**Abstract**: Large language models (LLMs) are pretrained on corpora containing trillions of tokens and, therefore, inevitably memorize sensitive information. Locate-then-edit methods, as a mainstream paradigm of model editing, offer a promising solution by modifying model parameters without retraining. However, in this work, we reveal a critical vulnerability of this paradigm: the parameter updates inadvertently serve as a side channel, enabling attackers to recover the edited data. We propose a two-stage reverse-engineering attack named \textit{KSTER} (\textbf{K}ey\textbf{S}paceRecons\textbf{T}ruction-then-\textbf{E}ntropy\textbf{R}eduction) that leverages the low-rank structure of these updates. First, we theoretically show that the row space of the update matrix encodes a ``fingerprint" of the edited subjects, enabling accurate subject recovery via spectral analysis. Second, we introduce an entropy-based prompt recovery attack that reconstructs the semantic context of the edit. Extensive experiments on multiple LLMs demonstrate that our attacks can recover edited data with high success rates. Furthermore, we propose \textit{subspace camouflage}, a defense strategy that obfuscates the update fingerprint with semantic decoys. This approach effectively mitigates reconstruction risks without compromising editing utility. Our code is available at this https URL. 

---
# Large Language Models Predict Functional Outcomes after Acute Ischemic Stroke 

**Authors**: Anjali K. Kapoor, Anton Alyakin, Jin Vivian Lee, Eunice Yang, Annelene M. Schulze, Krithik Vishwanath, Jinseok Lee, Yindalon Aphinyanaphongs, Howard Riina, Jennifer A. Frontera, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2602.10119)  

**Abstract**: Accurate prediction of functional outcomes after acute ischemic stroke can inform clinical decision-making and resource allocation. Prior work on modified Rankin Scale (mRS) prediction has relied primarily on structured variables (e.g., age, NIHSS) and conventional machine learning. The ability of large language models (LLMs) to infer future mRS scores directly from routine admission notes remains largely unexplored. We evaluated encoder (BERT, NYUTron) and generative (Llama-3.1-8B, MedGemma-4B) LLMs, in both frozen and fine-tuned settings, for discharge and 90-day mRS prediction using a large, real-world stroke registry. The discharge outcome dataset included 9,485 History and Physical notes and the 90-day outcome dataset included 1,898 notes from the NYU Langone Get With The Guidelines-Stroke registry (2016-2025). Data were temporally split with the most recent 12 months held out for testing. Performance was assessed using exact (7-class) mRS accuracy and binary functional outcome (mRS 0-2 vs. 3-6) accuracy and compared against established structured-data baselines incorporating NIHSS and age. Fine-tuned Llama achieved the highest performance, with 90-day exact mRS accuracy of 33.9% [95% CI, 27.9-39.9%] and binary accuracy of 76.3% [95% CI, 70.7-81.9%]. Discharge performance reached 42.0% [95% CI, 39.0-45.0%] exact accuracy and 75.0% [95% CI, 72.4-77.6%] binary accuracy. For 90-day prediction, Llama performed comparably to structured-data baselines. Fine-tuned LLMs can predict post-stroke functional outcomes from admission notes alone, achieving performance comparable to models requiring structured variable abstraction. Our findings support the development of text-based prognostic tools that integrate seamlessly into clinical workflows without manual data extraction. 

---
# AntigenLM: Structure-Aware DNA Language Modeling for Influenza 

**Authors**: Yue Pei, Xuebin Chi, Yu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2602.09067)  

**Abstract**: Language models have advanced sequence analysis, yet DNA foundation models often lag behind task-specific methods for unclear reasons. We present AntigenLM, a generative DNA language model pretrained on influenza genomes with intact, aligned functional units. This structure-aware pretraining enables AntigenLM to capture evolutionary constraints and generalize across tasks. Fine-tuned on time-series hemagglutinin (HA) and neuraminidase (NA) sequences, AntigenLM accurately forecasts future antigenic variants across regions and subtypes, including those unseen during training, outperforming phylogenetic and evolution-based models. It also achieves near-perfect subtype classification. Ablation studies show that disrupting genomic structure through fragmentation or shuffling severely degrades performance, revealing the importance of preserving functional-unit integrity in DNA language modeling. AntigenLM thus provides both a powerful framework for antigen evolution prediction and a general principle for building biologically grounded DNA foundation models. 

---
