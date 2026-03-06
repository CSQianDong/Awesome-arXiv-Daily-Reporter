# Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought 

**Authors**: Siddharth Boppana, Annabel Ma, Max Loeffler, Raphael Sarfati, Eric Bigelow, Atticus Geiger, Owen Lewis, Jack Merullo  

**Link**: [PDF](https://arxiv.org/pdf/2603.05488)  

**Abstract**: We provide evidence of performative chain-of-thought (CoT) in reasoning models, where a model becomes strongly confident in its final answer, but continues generating tokens without revealing its internal belief. Our analysis compares activation probing, early forced answering, and a CoT monitor across two large models (DeepSeek-R1 671B & GPT-OSS 120B) and find task difficulty-specific differences: The model's final answer is decodable from activations far earlier in CoT than a monitor is able to say, especially for easy recall-based MMLU questions. We contrast this with genuine reasoning in difficult multihop GPQA-Diamond questions. Despite this, inflection points (e.g., backtracking, 'aha' moments) occur almost exclusively in responses where probes show large belief shifts, suggesting these behaviors track genuine uncertainty rather than learned "reasoning theater." Finally, probe-guided early exit reduces tokens by up to 80% on MMLU and 30% on GPQA-Diamond with similar accuracy, positioning attention probing as an efficient tool for detecting performative reasoning and enabling adaptive computation. 

---
# Leveraging LLM Parametric Knowledge for Fact Checking without Retrieval 

**Authors**: Artem Vazhentsev, Maria Marina, Daniil Moskovskiy, Sergey Pletenev, Mikhail Seleznyov, Mikhail Salnikov, Elena Tutubalina, Vasily Konovalov, Irina Nikishina, Alexander Panchenko, Viktor Moskvoretskii  

**Link**: [PDF](https://arxiv.org/pdf/2603.05471)  

**Abstract**: Trustworthiness is a core research challenge for agentic AI systems built on Large Language Models (LLMs). To enhance trust, natural language claims from diverse sources, including human-written text, web content, and model outputs, are commonly checked for factuality by retrieving external knowledge and using an LLM to verify the faithfulness of claims to the retrieved evidence. As a result, such methods are constrained by retrieval errors and external data availability, while leaving the models intrinsic fact-verification capabilities largely unused. We propose the task of fact-checking without retrieval, focusing on the verification of arbitrary natural language claims, independent of their source. To study this setting, we introduce a comprehensive evaluation framework focused on generalization, testing robustness to (i) long-tail knowledge, (ii) variation in claim sources, (iii) multilinguality, and (iv) long-form generation. Across 9 datasets, 18 methods and 3 models, our experiments indicate that logit-based approaches often underperform compared to those that leverage internal model representations. Building on this finding, we introduce INTRA, a method that exploits interactions between internal representations and achieves state-of-the-art performance with strong generalization. More broadly, our work establishes fact-checking without retrieval as a promising research direction that can complement retrieval-based frameworks, improve scalability, and enable the use of such systems as reward signals during training or as components integrated into the generation process. 

---
# NCTB-QA: A Large-Scale Bangla Educational Question Answering Dataset and Benchmarking Performance 

**Authors**: Abrar Eyasir, Tahsin Ahmed, Muhammad Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2603.05462)  

**Abstract**: Reading comprehension systems for low-resource languages face significant challenges in handling unanswerable questions. These systems tend to produce unreliable responses when correct answers are absent from context. To solve this problem, we introduce NCTB-QA, a large-scale Bangla question answering dataset comprising 87,805 question-answer pairs extracted from 50 textbooks published by Bangladesh's National Curriculum and Textbook Board. Unlike existing Bangla datasets, NCTB-QA maintains a balanced distribution of answerable (57.25%) and unanswerable (42.75%) questions. NCTB-QA also includes adversarially designed instances containing plausible distractors. We benchmark three transformer-based models (BERT, RoBERTa, ELECTRA) and demonstrate substantial improvements through fine-tuning. BERT achieves 313% relative improvement in F1 score (0.150 to 0.620). Semantic answer quality measured by BERTScore also increases significantly across all models. Our results establish NCTB-QA as a challenging benchmark for Bangla educational question answering. This study demonstrates that domain-specific fine-tuning is critical for robust performance in low-resource settings. 

---
# DEBISS: a Corpus of Individual, Semi-structured and Spoken Debates 

**Authors**: Klaywert Danillo Ferreira de Souza, David Eduardo Pereira, Cláudio E. C. Campelo, Larissa Lucena Vasconcelos  

**Link**: [PDF](https://arxiv.org/pdf/2603.05459)  

**Abstract**: The process of debating is essential in our daily lives, whether in studying, work activities, simple everyday discussions, political debates on TV, or online discussions on social networks. The range of uses for debates is broad. Due to the diverse applications, structures, and formats of debates, developing corpora that account for these variations can be challenging, and the scarcity of debate corpora in the state of the art is notable. For this reason, the current research proposes the DEBISS corpus: a collection of spoken and individual debates with semi-structured features. With a broad range of NLP task annotations, such as speech-to-text, speaker diarization, argument mining, and debater quality assessment. 

---
# FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling 

**Authors**: Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao  

**Link**: [PDF](https://arxiv.org/pdf/2603.05451)  

**Abstract**: Attention, as a core layer of the ubiquitous Transformer architecture, is the bottleneck for large language models and long-context applications. While FlashAttention-3 optimized attention for Hopper GPUs through asynchronous execution and warp specialization, it primarily targets the H100 architecture. The AI industry has rapidly transitioned to deploying Blackwell-based systems such as the B200 and GB200, which exhibit fundamentally different performance characteristics due to asymmetric hardware scaling: tensor core throughput doubles while other functional units (shared memory bandwidth, exponential units) scale more slowly or remain unchanged. We develop several techniques to address these shifting bottlenecks on Blackwell GPUs: (1) redesigned pipelines that exploit fully asynchronous MMA operations and larger tile sizes, (2) software-emulated exponential and conditional softmax rescaling that reduces non-matmul operations, and (3) leveraging tensor memory and the 2-CTA MMA mode to reduce shared memory traffic and atomic adds in the backward pass. We demonstrate that our method, FlashAttention-4, achieves up to 1.3$\times$ speedup over cuDNN 9.13 and 2.7$\times$ over Triton on B200 GPUs with BF16, reaching up to 1613 TFLOPs/s (71% utilization). Beyond algorithmic innovations, we implement FlashAttention-4 entirely in CuTe-DSL embedded in Python, achieving 20-30$\times$ faster compile times compared to traditional C++ template-based approaches while maintaining full expressivity. 

---
# Ensembling Language Models with Sequential Monte Carlo 

**Authors**: Robin Shing Moon Chan, Tianyu Liu, Samuel Kiegeland, Clemente Pasti, Jacob Hoover Vigly, Timothy J. O'Donnell, Ryan Cotterell, Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2603.05432)  

**Abstract**: Practitioners have access to an abundance of language models and prompting strategies for solving many language modeling tasks; yet prior work shows that modeling performance is highly sensitive to both choices. Classical machine learning ensembling techniques offer a principled approach: aggregate predictions from multiple sources to achieve better performance than any single one. However, applying ensembling to language models during decoding is challenging: naively aggregating next-token probabilities yields samples from a locally normalized, biased approximation of the generally intractable ensemble distribution over strings. In this work, we introduce a unified framework for composing $K$ language models into $f$-ensemble distributions for a wide range of functions $f\colon\mathbb{R}_{\geq 0}^{K}\to\mathbb{R}_{\geq 0}$. To sample from these distributions, we propose a byte-level sequential Monte Carlo (SMC) algorithm that operates in a shared character space, enabling ensembles of models with mismatching vocabularies and consistent sampling in the limit. We evaluate a family of $f$-ensembles across prompt and model combinations for various structured text generation tasks, highlighting the benefits of alternative aggregation strategies over traditional probability averaging, and showing that better posterior approximations can yield better ensemble performance. 

---
# An Exploration-Analysis-Disambiguation Reasoning Framework for Word Sense Disambiguation with Low-Parameter LLMs 

**Authors**: Deshan Sumanathilaka, Nicholas Micallef, Julian Hough  

**Link**: [PDF](https://arxiv.org/pdf/2603.05400)  

**Abstract**: Word Sense Disambiguation (WSD) remains a key challenge in Natural Language Processing (NLP), especially when dealing with rare or domain-specific senses that are often misinterpreted. While modern high-parameter Large Language Models (LLMs) such as GPT-4-Turbo have shown state-of-the-art WSD performance, their computational and energy demands limit scalability. This study investigates whether low-parameter LLMs (<4B parameters) can achieve comparable results through fine-tuning strategies that emphasize reasoning-driven sense identification. Using the FEWS dataset augmented with semi-automated, rationale-rich annotations, we fine-tune eight small-scale open-source LLMs (e.g. Gemma and Qwen). Our results reveal that Chain-of-Thought (CoT)-based reasoning combined with neighbour-word analysis achieves performance comparable to GPT-4-Turbo in zero-shot settings. Importantly, Gemma-3-4B and Qwen-3-4B models consistently outperform all medium-parameter baselines and state-of-the-art models on FEWS, with robust generalization to unseen senses. Furthermore, evaluation on the unseen "Fool Me If You Can'' dataset confirms strong cross-domain adaptability without task-specific fine-tuning. This work demonstrates that with carefully crafted reasoning-centric fine-tuning, low-parameter LLMs can deliver accurate WSD while substantially reducing computational and energy demands. 

---
# Progressive Residual Warmup for Language Model Pretraining 

**Authors**: Tianhao Chen, Xin Xu, Lu Yin, Hao Chen, Yang Wang, Shizhe Diao, Can Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05369)  

**Abstract**: Transformer architectures serve as the backbone for most modern Large Language Models, therefore their pretraining stability and convergence speed are of central concern. Motivated by the logical dependency of sequentially stacked layers, we propose Progressive Residual Warmup (ProRes) for language model pretraining. ProRes implements an "early layer learns first" philosophy by multiplying each layer's residual with a scalar that gradually warms up from 0 to 1, with deeper layers taking longer warmup steps. In this way, deeper layers wait for early layers to settle into a more stable regime before contributing to learning. We demonstrate the effectiveness of ProRes through pretraining experiments across various model scales, as well as normalization and initialization schemes. Comprehensive analysis shows that ProRes not only stabilizes pretraining but also introduces a unique optimization trajectory, leading to faster convergence, stronger generalization and better downstream performance. Our code is available at this https URL. 

---
# DiSCTT: Consensus-Guided Self-Curriculum for Efficient Test-Time Adaptation in Reasoning 

**Authors**: Mohammad Mahdi Moradi, Sudhir Mudur  

**Link**: [PDF](https://arxiv.org/pdf/2603.05357)  

**Abstract**: Test-time adaptation offers a promising avenue for improving reasoning performance in large language models without additional supervision, but existing approaches often apply a uniform optimization objective across all inputs, leading to inefficient or unstable adaptation on heterogeneous reasoning problems. We propose DiSCTT, a difficulty-aware, consensus-guided self-curriculum framework that dynamically allocates test-time optimization strategies based on instance-level epistemic uncertainty estimated from agreement among sampled reasoning trajectories. Inputs with high consensus are consolidated via supervised fine-tuning using majority-agreed solutions as pseudo-labels, while low-consensus inputs are optimized via reinforcement learning with a consensus-regularized objective that encourages diversity under relevance constraints. Across a broad suite of mathematical and general reasoning benchmarks, DiSCTT consistently outperforms strong test-time adaptation baselines, achieving higher accuracy with reduced variance and substantially lower computation and wall-clock training times. These results demonstrate that explicitly accounting for instance difficulty and uncertainty enables more stable, efficient, and effective test-time adaptation for reasoning models. 

---
# Exploring the potential and limitations of Model Merging for Multi-Domain Adaptation in ASR 

**Authors**: Carlos Carvalho, Francisco Teixeira, Thomas Rolland, Alberto Abad  

**Link**: [PDF](https://arxiv.org/pdf/2603.05354)  

**Abstract**: Model merging is a scalable alternative to multi-task training that combines the capabilities of multiple specialised models into a single model. This is particularly attractive for large speech foundation models, which are typically adapted through domain-specific fine-tuning, resulting in multiple customised checkpoints, for which repeating full fine-tuning when new data becomes available is computationally prohibitive. In this work, we study model merging for multi-domain ASR and benchmark 11 merging algorithms for 10 European Portuguese domains, evaluating in-domain accuracy, robustness under distribution shift, as well as English and multilingual performance. We further propose BoostedTSV-M, a new merging algorithm based on TSV-M that mitigates rank collapse via singular-value boosting and improves numerical stability. Overall, our approach outperforms full fine-tuning on European Portuguese while preserving out-of-distribution generalisation in a single model. 

---
# A Multilingual Human Annotated Corpus of Original and Easy-to-Read Texts to Support Access to Democratic Participatory Processes 

**Authors**: Stefan Bott, Verena Riegler, Horacio Saggion, Almudena Rascón Alcaina, Nouran Khallaf  

**Link**: [PDF](https://arxiv.org/pdf/2603.05345)  

**Abstract**: Being able to understand information is a key factor for a self-determined life and society. It is also very important for participating in democratic processes. The study of automatic text simplification is often limited by the availability of high quality material for the training and evaluation on automatic simplifiers. This is true for English, but more so for less resourced languages like Spanish, Catalan and Italian. In order to fill this gap, we present a corpus of original texts for these 3 languages, with high quality simplification produced by human experts in text simplification. It was developed within the iDEM project to assess the impact of Easy-to-Read (E2R) language for democratic participation. The original texts were compiled from domains related to this topic. The corpus includes different text types, selected based on relevance, copyright availability, and ethical standards. All texts were simplified to E2R level. The corpus is particularity valuable because it includes the first annotated corpus of its kind for the Catalan language. It also represents a noteworthy contribution for Spanish and Italian, offering high-quality, human-annotated language resources that are rarely available in these domains. The corpus will be made freely accessible to the public. 

---
# PersianPunc: A Large-Scale Dataset and BERT-Based Approach for Persian Punctuation Restoration 

**Authors**: Mohammad Javad Ranjbar Kalahroodi, Heshaam Faili, Azadeh Shakery  

**Link**: [PDF](https://arxiv.org/pdf/2603.05314)  

**Abstract**: Punctuation restoration is essential for improving the readability and downstream utility of automatic speech recognition (ASR) outputs, yet remains underexplored for Persian despite its importance. We introduce PersianPunc, a large-scale, high-quality dataset of 17 million samples for Persian punctuation restoration, constructed through systematic aggregation and filtering of existing textual resources. We formulate punctuation restoration as a token-level sequence labeling task and fine-tune ParsBERT to achieve strong performance. Through comparative evaluation, we demonstrate that while large language models can perform punctuation restoration, they suffer from critical limitations: over-correction tendencies that introduce undesired edits beyond punctuation insertion (particularly problematic for speech-to-text pipelines) and substantially higher computational requirements. Our lightweight BERT-based approach achieves a macro-averaged F1 score of 91.33% on our test set while maintaining efficiency suitable for real-time applications. We make our dataset (this https URL) and model (this https URL) publicly available to facilitate future research in Persian NLP and provide a scalable framework applicable to other morphologically rich, low-resource languages. 

---
# Med-V1: Small Language Models for Zero-shot and Scalable Biomedical Evidence Attribution 

**Authors**: Qiao Jin, Yin Fang, Lauren He, Yifan Yang, Guangzhi Xiong, Zhizheng Wang, Nicholas Wan, Joey Chan, Donald C. Comeau, Robert Leaman, Charalampos S. Floudas, Aidong Zhang, Michael F. Chiang, Yifan Peng, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.05308)  

**Abstract**: Assessing whether an article supports an assertion is essential for hallucination detection and claim verification. While large language models (LLMs) have the potential to automate this task, achieving strong performance requires frontier models such as GPT-5 that are prohibitively expensive to deploy at scale. To efficiently perform biomedical evidence attribution, we present Med-V1, a family of small language models with only three billion parameters. Trained on high-quality synthetic data newly developed in this study, Med-V1 substantially outperforms (+27.0% to +71.3%) its base models on five biomedical benchmarks unified into a verification format. Despite its smaller size, Med-V1 performs comparably to frontier LLMs such as GPT-5, along with high-quality explanations for its predictions. We use Med-V1 to conduct a first-of-its-kind use case study that quantifies hallucinations in LLM-generated answers under different citation instructions. Results show that the format instruction strongly affects citation validity and hallucination, with GPT-5 generating more claims but exhibiting hallucination rates similar to GPT-4o. Additionally, we present a second use case showing that Med-V1 can automatically identify high-stakes evidence misattributions in clinical practice guidelines, revealing potentially negative public health impacts that are otherwise challenging to identify at scale. Overall, Med-V1 provides an efficient and accurate lightweight alternative to frontier LLMs for practical and real-world applications in biomedical evidence attribution and verification tasks. Med-V1 is available at this https URL. 

---
# Oral to Web: Digitizing 'Zero Resource'Languages of Bangladesh 

**Authors**: Mohammad Mamun Or Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2603.05272)  

**Abstract**: We present the Multilingual Cloud Corpus, the first national-scale, parallel, multimodal linguistic dataset of Bangladesh's ethnic and indigenous languages. Despite being home to approximately 40 minority languages spanning four language families, Bangladesh has lacked a systematic, cross-family digital corpus for these predominantly oral, computationally "zero resource" varieties, 14 of which are classified as endangered. Our corpus comprises 85792 structured textual entries, each containing a Bengali stimulus text, an English translation, and an IPA transcription, together with approximately 107 hours of transcribed audio recordings, covering 42 language varieties from the Tibeto-Burman, Indo-European, Austro-Asiatic, and Dravidian families, plus two genetically unclassified languages. The data were collected through systematic fieldwork over 90 days across nine districts of Bangladesh, involving 16 data collectors, 77 speakers, and 43 validators, following a predefined elicitation template of 2224 unique items organized at three levels of linguistic granularity: isolated lexical items (475 words across 22 semantic domains), grammatical constructions (887 sentences across 21 categories including verbal conjugation paradigms), and directed speech (862 prompts across 46 conversational scenarios). Post-field processing included IPA transcription by 10 linguists with independent adjudication by 6 reviewers. The complete dataset is publicly accessible through the Multilingual Cloud platform (this http URL), providing searchable access to annotated audio and textual data for all documented varieties. We describe the corpus design, fieldwork methodology, dataset structure, and per-language coverage, and discuss implications for endangered language documentation, low-resource NLP, and digital preservation in linguistically diverse developing countries. 

---
# VietJobs: A Vietnamese Job Advertisement Dataset 

**Authors**: Hieu Pham Dinh, Hung Nguyen Huy, Mo El-Haj  

**Link**: [PDF](https://arxiv.org/pdf/2603.05262)  

**Abstract**: VietJobs is the first large-scale, publicly available corpus of Vietnamese job advertisements, comprising 48,092 postings and over 15 million words collected from all 34 provinces and municipalities across Vietnam. The dataset provides extensive linguistic and structured information, including job titles, categories, salaries, skills, and employment conditions, covering 16 occupational domains and multiple employment types (full-time, part-time, and internship). Designed to support research in natural language processing and labour market analytics, VietJobs captures substantial linguistic, regional, and socio-economic diversity. We benchmark several generative large language models (LLMs) on two core tasks: job category classification and salary estimation. Instruction-tuned models such as Qwen2.5-7B-Instruct and Llama-SEA-LION-v3-8B-IT demonstrate notable gains under few-shot and fine-tuned settings, while highlighting challenges in multilingual and Vietnamese-specific modelling for structured labour market prediction. VietJobs establishes a new benchmark for Vietnamese NLP and offers a valuable foundation for future research on recruitment language, socio-economic representation, and AI-driven labour market analysis. All code and resources are available at: this https URL. 

---
# Balancing Coverage and Draft Latency in Vocabulary Trimming for Faster Speculative Decoding 

**Authors**: Ofir Ben Shoham  

**Link**: [PDF](https://arxiv.org/pdf/2603.05210)  

**Abstract**: Speculative decoding accelerates inference for Large Language Models by using a lightweight draft model to propose candidate tokens that are verified in parallel by a larger target model. Prior work shows that the draft model often dominates speculative decoding latency, since it generates tokens sequentially and incurs high cost from its language modeling head as vocabulary size grows. This exposes a fundamental trade-off in draft model design: larger vocabularies improve token coverage and agreement with the target model, but incur higher draft latency, while smaller vocabularies reduce latency at the risk of missing tokens required for accurate draft generation. We address this trade-off through vocabulary trimming for draft models, motivated by the observation that domain-specific workloads use only a small fraction of the full vocabulary. We cast draft vocabulary selection as a constrained optimization problem that balances token coverage and draft latency. Coverage is computed over assistant responses in the training data, while latency is estimated using architecture-aware FLOPs that capture the cost of the language modeling head as a function of vocabulary size. We optimize a utility function with a Tree-structured Parzen Estimator to efficiently explore the coverage-latency Pareto frontier under a minimum coverage constraint. Experiments show improved speculative decoding throughput while reducing draft vocabularies by up to 97% with high coverage. On domain-specific tasks, we achieve up to 16% latency reduction and 20% throughput improvement, and up to 6.7% throughput gains on diverse out-of-distribution tasks. 

---
# Distilling Formal Logic into Neural Spaces: A Kernel Alignment Approach for Signal Temporal Logic 

**Authors**: Sara Candussio, Gabriele Sarti, Gaia Saveri, Luca Bortolussi  

**Link**: [PDF](https://arxiv.org/pdf/2603.05198)  

**Abstract**: We introduce a framework for learning continuous neural representations of formal specifications by distilling the geometry of their semantics into a latent space. Existing approaches rely either on symbolic kernels -- which preserve behavioural semantics but are computationally prohibitive, anchor-dependent, and non-invertible -- or on syntax-based neural embeddings that fail to capture underlying structures. Our method bridges this gap: using a teacher-student setup, we distill a symbolic robustness kernel into a Transformer encoder. Unlike standard contrastive methods, we supervise the model with a continuous, kernel-weighted geometric alignment objective that penalizes errors in proportion to their semantic discrepancies. Once trained, the encoder produces embeddings in a single forward pass, effectively mimicking the kernel's logic at a fraction of its computational cost. We apply our framework to Signal Temporal Logic (STL), demonstrating that the resulting neural representations faithfully preserve the semantic similarity of STL formulae, accurately predict robustness and constraint satisfaction, and remain intrinsically invertible. Our proposed approach enables highly efficient, scalable neuro-symbolic reasoning and formula reconstruction without repeated kernel computation at runtime. 

---
# Diffusion LLMs can think EoS-by-EoS 

**Authors**: Sarah Breckner, Sebastian Schuster  

**Link**: [PDF](https://arxiv.org/pdf/2603.05197)  

**Abstract**: Diffusion LLMs have been proposed as an alternative to autoregressive LLMs, excelling especially at complex reasoning tasks with interdependent sub-goals. Curiously, this is particularly true if the generation length, i.e., the number of tokens the model has to output, is set to a much higher value than is required for providing the correct answer to the task, and the model pads its answer with end-of-sequence (EoS) tokens. We hypothesize that diffusion models think EoS-by-EoS, that is, they use the representations of EoS tokens as a hidden scratchpad, which allows them to solve harder reasoning problems. We experiment with the diffusion models LLaDA1.5, LLaDA2.0-mini, and Dream-v0 on the tasks Addition, Entity Tracking, and Sudoku. In a controlled prompting experiment, we confirm that adding EoS tokens improves the LLMs' reasoning capabilities. To further verify whether they serve as space for hidden computations, we patch the hidden states of the EoS tokens with those of a counterfactual generation, which frequently changes the generated output to the counterfactual. The success of the causal intervention underscores that the EoS tokens, which one may expect to be devoid of meaning, carry information on the problem to solve. The behavioral experiments and the causal interventions indicate that diffusion LLMs can indeed think EoS-by-EoS. 

---
# Transducing Language Models 

**Authors**: Vésteinn Snæbjarnarson, Samuel Kiegeland, Tianyu Liu, Reda Boumasmoud, Ryan Cotterell, Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2603.05193)  

**Abstract**: Modern language models define distributions over strings, but downstream tasks often require different output formats. For instance, a model that generates byte-pair strings does not directly produce word-level predictions, and a DNA model does not directly produce amino-acid sequences. In such cases, a deterministic string-to-string transformation can convert the model's output to the desired form. This is a familiar pattern in probability theory: applying a function $f$ to a random variable $X\sim p$ yields a transformed random variable $f(X)$ with an induced distribution. While such transformations are occasionally used in language modeling, prior work does not treat them as yielding new, fully functional language models. We formalize this perspective and introduce a general framework for language models derived from deterministic string-to-string transformations. We focus on transformations representable as finite-state transducers -- a commonly used state-machine abstraction for efficient string-to-string mappings. We develop algorithms that compose a language model with an FST to *marginalize* over source strings mapping to a given target, propagating probabilities through the transducer without altering model parameters and enabling *conditioning* on transformed outputs. We present an exact algorithm, an efficient approximation, and a theoretical analysis. We conduct experiments in three domains: converting language models from tokens to bytes, from tokens to words, and from DNA to amino acids. These experiments demonstrate inference-time adaptation of pretrained language models to match application-specific output requirements. 

---
# Guidelines for the Annotation and Visualization of Legal Argumentation Structures in Chinese Judicial Decisions 

**Authors**: Kun Chen, Xianglei Liao, Kaixue Fei, Yi Xing, Xinrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.05171)  

**Abstract**: This guideline proposes a systematic and operational annotation framework for representing the structure of legal argumentation in judicial decisions. Grounded in theories of legal reasoning and argumentation, the framework aims to reveal the logical organization of judicial reasoning and to provide a reliable data foundation for computational analysis. At the proposition level, the guideline distinguishes four types of propositions: general normative propositions, specific normative propositions, general factual propositions, and specific factual propositions. At the relational level, five types of relations are defined to capture argumentative structures: support, attack, joint, match, and identity. These relations represent positive and negative argumentative connections, conjunctive reasoning structures, the correspondence between legal norms and case facts, and semantic equivalence between propositions. The guideline further specifies formal representation rules and visualization conventions for both basic and nested structures, enabling consistent graphical representation of complex argumentation patterns. In addition, it establishes a standardized annotation workflow and consistency control mechanisms to ensure reproducibility and reliability of the annotated data. By providing a clear conceptual model, formal representation rules, and practical annotation procedures, this guideline offers methodological support for large-scale analysis of judicial reasoning and for future research in legal argument mining, computational modeling of legal reasoning, and AI-assisted legal analysis. 

---
# Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity 

**Authors**: Di Zhang, Xun Wu, Shaohan Huang, Yudong Wang, Hanyong Shao, Yingbo Hao, Zewen Chi, Li Dong, Ting Song, Yan Xia, Zhifang Sui, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.05168)  

**Abstract**: Semi-structured N:M sparsity and low-bit quantization (e.g., 1.58-bit BitNet) are two promising approaches for improving the efficiency of large language models (LLMs), yet they have largely been studied in isolation. In this work, we investigate their interaction and show that 1.58-bit BitNet is naturally more compatible with N:M sparsity than full-precision models. To study this effect, we propose Sparse-BitNet, a unified framework that jointly applies 1.58-bit quantization and dynamic N:M sparsification while ensuring stable training for the first time. Across multiple model scales and training regimes (sparse pretraining and dense-to-sparse schedules), 1.58-bit BitNet consistently exhibits smaller performance degradation than full-precision baselines at the same sparsity levels and can tolerate higher structured sparsity before accuracy collapse. Moreover, using our custom sparse tensor core, Sparse-BitNet achieves substantial speedups in both training and inference, reaching up to 1.30X. These results highlight that combining extremely low-bit quantization with semi-structured N:M sparsity is a promising direction for efficient LLMs. Code available at this https URL 

---
# C2-Faith: Benchmarking LLM Judges for Causal and Coverage Faithfulness in Chain-of-Thought Reasoning 

**Authors**: Avni Mittal, Rauno Arike  

**Link**: [PDF](https://arxiv.org/pdf/2603.05167)  

**Abstract**: Large language models (LLMs) are increasingly used as judges of chain-of-thought (CoT) reasoning, but it remains unclear whether they can reliably assess process faithfulness rather than just answer plausibility. We introduce C2-Faith, a benchmark built from PRM800K that targets two complementary dimensions of faithfulness: causality (does each step logically follow from prior context?) and coverage (are essential intermediate inferences present?). Using controlled perturbations, we create examples with known causal error positions by replacing a single step with an acausal variant, and with controlled coverage deletions at varying deletion rates (scored against reference labels). We evaluate three frontier judges under three tasks: binary causal detection, causal step localization, and coverage scoring. The results show that model rankings depend strongly on task framing, with no single judge dominating all settings; all judges exhibit a substantial gap between detecting an error and localizing it; and coverage judgments are systematically inflated for incomplete reasoning. These findings clarify when LLM judges are dependable and where they fail, and provide practical guidance for selecting judges in process-level evaluation 

---
# Feature Resemblance: On the Theoretical Understanding of Analogical Reasoning in Transformers 

**Authors**: Ruichen Xu, Wenjing Yan, Ying-Jun Angela Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05143)  

**Abstract**: Understanding reasoning in large language models is complicated by evaluations that conflate multiple reasoning types. We isolate analogical reasoning (inferring shared properties between entities based on known similarities) and analyze its emergence in transformers. We theoretically prove three key results: (1) Joint training on similarity and attribution premises enables analogical reasoning through aligned representations; (2) Sequential training succeeds only when similarity structure is learned before specific attributes, revealing a necessary curriculum; (3) Two-hop reasoning ($a \to b, b \to c \implies a \to c$) reduces to analogical reasoning with identity bridges ($b = b$), which must appear explicitly in training data. These results reveal a unified mechanism: transformers encode entities with similar properties into similar representations, enabling property transfer through feature alignment. Experiments with architectures up to 1.5B parameters validate our theory and demonstrate how representational geometry shapes inductive reasoning capabilities. 

---
# Representation Fidelity:Auditing Algorithmic Decisions About Humans Using Self-Descriptions 

**Authors**: Theresa Elstner, Martin Potthast  

**Link**: [PDF](https://arxiv.org/pdf/2603.05136)  

**Abstract**: This paper introduces a new dimension for validating algorithmic decisions about humans by measuring the fidelity of their representations. Representation Fidelity measures if decisions about a person rest on reasonable grounds. We propose to operationalize this notion by measuring the distance between two representations of the same person: (1) an externally prescribed input representation on which the decision is based, and (2) a self-description provided by the human subject of the decision, used solely to validate the input representation. We examine the nature of discrepancies between these representations, how such discrepancies can be quantified, and derive a generic typology of representation mismatches that determine the degree of representation fidelity. We further present the first benchmark for evaluating representation fidelity based on a dataset of loan-granting decisions. Our Loan-Granting Self-Representations Corpus 2025 consists of a large corpus of 30 000 synthetic natural language self-descriptions derived from corresponding representations of applicants in the German Credit Dataset, along with expert annotations of representation mismatches between each pair of representations. 

---
# LBM: Hierarchical Large Auto-Bidding Model via Reasoning and Acting 

**Authors**: Yewen Li, Zhiyi Lyu, Peng Jiang, Qingpeng Cai, Fei Pan, Bo An, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05134)  

**Abstract**: The growing scale of ad auctions on online advertising platforms has intensified competition, making manual bidding impractical and necessitating auto-bidding to help advertisers achieve their economic goals. Current auto-bidding methods have evolved to use offline reinforcement learning or generative methods to optimize bidding strategies, but they can sometimes behave counterintuitively due to the black-box training manner and limited mode coverage of datasets, leading to challenges in understanding task status and generalization in dynamic ad environments. Large language models (LLMs) offer a promising solution by leveraging prior human knowledge and reasoning abilities to improve auto-bidding performance. However, directly applying LLMs to auto-bidding faces difficulties due to the need for precise actions in competitive auctions and the lack of specialized auto-bidding knowledge, which can lead to hallucinations and suboptimal decisions. To address these challenges, we propose a hierarchical Large autoBidding Model (LBM) to leverage the reasoning capabilities of LLMs for developing a superior auto-bidding strategy. This includes a high-level LBM-Think model for reasoning and a low-level LBM-Act model for action generation. Specifically, we propose a dual embedding mechanism to efficiently fuse two modalities, including language and numerical inputs, for language-guided training of the LBM-Act; then, we propose an offline reinforcement fine-tuning technique termed GQPO for mitigating the LLM-Think's hallucinations and enhancing decision-making performance without simulation or real-world rollout like previous multi-turn LLM-based methods. Experiments demonstrate the superiority of a generative backbone based on our LBM, especially in an efficient training manner and generalization ability. 

---
# Measuring the Redundancy of Decoder Layers in SpeechLLMs 

**Authors**: Adel Moumen, Guangzhi Sun, Philip C Woodland  

**Link**: [PDF](https://arxiv.org/pdf/2603.05121)  

**Abstract**: Speech Large Language Models route speech encoder representations into an LLM decoder that typically accounts for over 90% of total parameters. We study how much of this decoder capacity is actually needed for speech tasks. Across two LLM families and three scales (1-8B), we show that decoder redundancy is largely inherited from the pretrained LLM: text and speech inputs yield similar redundant blocks. We then measure excess capacity by pruning decoder layers and analysing post-pruning healing to increase robustness. Our findings show that 7-8B models retain good ASR performance with only 60% of decoder layers, and the same trend extends to smaller scales with reduced pruning tolerance. We then generalise to speech translation, and show that the same blocks of layers are redundant across speech encoders, tasks and languages, indicating that a more global redundancy structure exists, enabling a single pruned and multi-tasks SpeechLLM backbone to be deployed. 

---
# ARC-TGI: Human-Validated Task Generators with Reasoning Chain Templates for ARC-AGI 

**Authors**: Jens Lehmann, Syeda Khushbakht, Nikoo Salehfard, Nur A Zarin Nishat, Dhananjay Bhandiwad, Andrei Aioanei, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2603.05099)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC-AGI) probes few-shot abstraction and rule induction on small visual grids, but progress is difficult to measure on static collections of hand-authored puzzles due to overfitting, dataset leakage, and memorisation. We introduce ARC-TGI (ARC Task Generators Inventory), an open-source framework for task-family generators: compact Python programs that sample diverse ARC-AGI tasks while preserving a latent rule. ARC-TGI is built around a solver-facing representation: each generated task is paired with natural-language input and transformation reasoning chains and partially evaluated Python code implementing sampling, transformation, and episode construction. Crucially, ARC-TGI supports task-level constraints so that training examples collectively expose the variations needed to infer the underlying rule, a requirement for human-solvable ARC tasks that independent per-example sampling often fails to guarantee. All generators undergo human refinement and local verification to keep both grids and reasoning traces natural and consistent under variation. We release 461 generators covering 180 ARC-Mini tasks, 215 ARC-AGI-1 tasks (200 train, 15 test), and 66 ARC-AGI-2 tasks (55 train, 11 test), enabling scalable dataset sampling and controlled benchmarking. 

---
# MUTEX: Leveraging Multilingual Transformers and Conditional Random Fields for Enhanced Urdu Toxic Span Detection 

**Authors**: Inayat Arshad, Fajar Saleem, Ijaz Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2603.05057)  

**Abstract**: Urdu toxic span detection remains limited because most existing systems rely on sentence-level classification and fail to identify the specific toxic spans within those text. It is further exacerbated by the multiple factors i.e. lack of token-level annotated resources, linguistic complexity of Urdu, frequent code-switching, informal expressions, and rich morphological variations. In this research, we propose MUTEX: a multilingual transformer combined with conditional random fields (CRF) for Urdu toxic span detection framework that uses manually annotated token-level toxic span dataset to improve performance and interpretability. MUTEX uses XLM RoBERTa with CRF layer to perform sequence labeling and is tested on multi-domain data extracted from social media, online news, and YouTube reviews using token-level F1 to evaluate fine-grained span detection. The results indicate that MUTEX achieves 60% token-level F1 score that is the first supervised baseline for Urdu toxic span detection. Further examination reveals that transformer-based models are more effective at implicitly capturing the contextual toxicity and are able to address the issues of code-switching and morphological variation than other models. 

---
# NeuronMoE: Neuron-Guided Mixture-of-Experts for Efficient Multilingual LLM Extension 

**Authors**: Rongzhi Li, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2603.05046)  

**Abstract**: Extending large language models to low-resource languages is essential for global accessibility, but training separate models per language is prohibitively expensive. Mixture-of-Experts (MoE) architectures address this by adding sparse language-specific parameters, but determining how many experts each layer needs remains an open question. Current approaches allocate experts based on layer-level similarity, yet language processing exhibits fine-grained specialization at individual neurons. We propose $\textbf{NeuronMoE}$, a method that analyzes language-specific neurons across all transformer components to guide expert allocation per layer based on empirically measured cross-lingual neuron diversity. Applied to Llama-3.2-3B for low-resource languages (Greek, Turkish, and Hungarian), this approach achieves approximately 40% average parameter reduction while matching the performance of the LayerMoE baseline. We find that low-resource language experts independently develop neuron specialization patterns mirroring the high-resource language, which are concentrated in early and late layers. This reveals potential universal architectural principles in how multilingual models organize linguistic knowledge. 

---
# HiFlow: Hierarchical Feedback-Driven Optimization for Constrained Long-Form Text Generation 

**Authors**: Yifan Zhu, Guanting Chen, Bing Wei, Haoran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.04996)  

**Abstract**: Large language models perform well in short text generation but still struggle with long text generation, particularly under complex constraints. Such tasks involve multiple tightly coupled objectives, including global structural consistency, local semantic coherence, and constraint feasibility, forming a challenging constrained optimization problem. Existing approaches mainly rely on static planning or offline supervision, limiting effective coordination between global and local objectives during generation. To address these challenges, we propose HiFlow, a hierarchical feedback-driven optimization framework for constrained long text generation. HiFlow formulates generation as a two-level optimization process, consisting of a planning layer for global structure and constraint modeling, and a generation layer for conditioned text generation. By incorporating constraint-aware plan screening and closed-loop feedback at both levels, HiFlow enables joint optimization of planning quality and generation behavior, progressively guiding the model toward high-quality, constraint-satisfying outputs. Experiments on multiple backbones confirm HiFlow's effectiveness over baseline methods. 

---
# ThaiSafetyBench: Assessing Language Model Safety in Thai Cultural Contexts 

**Authors**: Trapoom Ukarapol, Nut Chukamphaeng, Kunat Pipatanakul, Pakhapoom Sarapat  

**Link**: [PDF](https://arxiv.org/pdf/2603.04992)  

**Abstract**: The safety evaluation of large language models (LLMs) remains largely centered on English, leaving non-English languages and culturally grounded risks underexplored. In this work, we investigate LLM safety in the context of the Thai language and culture and introduce ThaiSafetyBench, an open-source benchmark comprising 1,954 malicious prompts written in Thai. The dataset covers both general harmful prompts and attacks that are explicitly grounded in Thai cultural, social, and contextual nuances.
Using ThaiSafetyBench, we evaluate 24 LLMs, with GPT-4.1 and Gemini-2.5-Pro serving as LLM-as-a-judge evaluators. Our results show that closed-source models generally demonstrate stronger safety performance than open-source counterparts, raising important concerns regarding the robustness of openly available models. Moreover, we observe a consistently higher Attack Success Rate (ASR) for Thai-specific, culturally contextualized attacks compared to general Thai-language attacks, highlighting a critical vulnerability in current safety alignment methods.
To improve reproducibility and cost efficiency, we further fine-tune a DeBERTa-based harmful response classifier, which we name ThaiSafetyClassifier. The model achieves a weighted F1 score of 84.4%, matching GPT-4.1 judgments. We publicly release the fine-tuning weights and training scripts to support reproducibility. Finally, we introduce the ThaiSafetyBench leaderboard to provide continuously updated safety evaluations and encourage community participation.
- ThaiSafetyBench HuggingFace Dataset: this https URL
- ThaiSafetyBench Github: this https URL
- ThaiSafetyClassifier HuggingFace Model: this https URL
- ThaiSafetyBench Leaderboard: this https URL 

---
# VRM: Teaching Reward Models to Understand Authentic Human Preferences 

**Authors**: Biao Liu, Ning Xu, Junming Yang, Hao Xu, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2603.04974)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse natural language tasks, yet the reward models employed for aligning LLMs often encounter challenges of reward hacking, where the approaches predominantly rely on directly mapping prompt-response pairs to scalar scores, which may inadvertently capture spurious correlations rather than authentic human preferences. In contrast, human evaluation employs a sophisticated process that initially weighs the relative importance of multiple high-dimensional objectives according to the prompt context, subsequently evaluating response quality through low-dimensional semantic features such as logical coherence and contextual appropriateness. Motivated by this consideration, we propose VRM, i.e., Variational Reward Modeling, a novel framework that explicitly models the evaluation process of human preference judgments by incorporating both high-dimensional objective weights and low-dimensional semantic features as latent variables, which are inferred through variational inference techniques. Additionally, we provide a theoretical analysis showing that VRM can achieve a tighter generalization error bound compared to the traditional reward model. Extensive experiments on benchmark datasets demonstrate that VRM significantly outperforms existing methods in capturing authentic human preferences. 

---
# MPCEval: A Benchmark for Multi-Party Conversation Generation 

**Authors**: Minxing Zhang, Yi Yang, Zhuofan Jia, Xuan Yang, Jian Pei, Yuchen Zang, Xingwang Deng, Xianglong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.04969)  

**Abstract**: Multi-party conversation generation, such as smart reply and collaborative assistants, is an increasingly important capability of generative AI, yet its evaluation remains a critical bottleneck. Compared to two-party dialogue, multi-party settings introduce distinct challenges, including complex turn-taking, role-dependent speaker behavior, long-range conversational structure, and multiple equally valid continuations. Accordingly, we introduce MPCEval, a task-aware evaluation and benchmarking suite for multi-party conversation generation. MPCEval decomposes generation quality into speaker modeling, content quality, and speaker--content consistency, and explicitly distinguishes local next-turn prediction from global full-conversation generation. It provides novel, quantitative, reference-free, and reproducible metrics that scale across datasets and models. We apply MPCEval to diverse public and real-world datasets and evaluate modern generation methods alongside human-authored conversations. The results reveal systematic, dimension-specific model characteristics in participation balance, content progression and novelty, and speaker--content consistency, demonstrating that evaluation objectives critically shape model assessment and that single-score evaluation obscures fundamental differences in multi-party conversational behavior. The implementation of MPCEval and the associated evaluation code are publicly available at this https URL. 

---
# When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger 

**Authors**: Amirabbas Afzali, Myeongho Jeon, Maria Brbic  

**Link**: [PDF](https://arxiv.org/pdf/2603.04968)  

**Abstract**: Preference alignment is an essential step in adapting large language models (LLMs) to human values, but existing approaches typically depend on costly human annotations or large-scale API-based models. We explore whether a weak LLM can instead act as an effective annotator. We surprisingly find that selecting only a subset of a weak LLM's highly confident samples leads to substantially better performance than using full human annotations. Building on this insight, we propose Confidence-Weighted Preference Optimization (CW-PO), a general framework that re-weights training samples by a weak LLM's confidence and can be applied across different preference optimization objectives. Notably, the model aligned by CW-PO with just 20% of human annotations outperforms the model trained with 100% of annotations under standard DPO. These results suggest that weak LLMs, when paired with confidence weighting, can dramatically reduce the cost of preference alignment while even outperforming methods trained on fully human-labeled data. 

---
# Replaying pre-training data improves fine-tuning 

**Authors**: Suhas Kotha, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04964)  

**Abstract**: To obtain a language model for a target domain (e.g. math), the current paradigm is to pre-train on a vast amount of generic web text and then fine-tune on the relatively limited amount of target data. Typically, generic data is only mixed in during fine-tuning to prevent catastrophic forgetting of the generic domain. We surprisingly find that replaying the generic data during fine-tuning can actually improve performance on the (less related) target task. Concretely, in a controlled pre-training environment with 4M target tokens, 4B total tokens, and 150M parameter models, generic replay increases target data efficiency by up to $1.87\times$ for fine-tuning and $2.06\times$ for mid-training. We further analyze data schedules that introduce target data during pre-training and find that replay helps more when there is less target data present in pre-training. We demonstrate the success of replay in practice for fine-tuning 8B parameter models, improving agentic web navigation success by $4.5\%$ and Basque question-answering accuracy by $2\%$. 

---
# LocalSUG: Geography-Aware LLM for Query Suggestion in Local-Life Services 

**Authors**: Jinwen Chen, Shuai Gong, Shiwen Zhang, Zheng Zhang, Yachao Zhao, Lingxiang Wang, Haibo Zhou, Yuan Zhan, Wei Lin, Hainan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04946)  

**Abstract**: In local-life service platforms, the query suggestion module plays a crucial role in enhancing user experience by generating candidate queries based on user input prefixes, thus reducing user effort and accelerating search. Traditional multi-stage cascading systems rely heavily on historical top queries, limiting their ability to address long-tail demand. While LLMs offer strong semantic generalization, deploying them in local-life services introduces three key challenges: lack of geographic grounding, exposure bias in preference optimization, and online inference latency. To address these issues, we propose LocalSUG, an LLM-based query suggestion framework tailored for local-life service platforms. First, we introduce a city-aware candidate mining strategy based on term co-occurrence to inject geographic grounding into generation. Second, we propose a beam-search-driven GRPO algorithm that aligns training with inference-time decoding, reducing exposure bias in autoregressive generation. A multi-objective reward mechanism further optimizes both relevance and business-oriented metrics. Finally, we develop quality-aware beam acceleration and vocabulary pruning techniques that significantly reduce online latency while preserving generation quality. Extensive offline evaluations and large-scale online A/B testing demonstrate that LocalSUG improves click-through rate (CTR) by +0.35% and reduces the low/no-result rate by 2.56%, validating its effectiveness in real-world deployment. 

---
# Federated Heterogeneous Language Model Optimization for Hybrid Automatic Speech Recognition 

**Authors**: Mengze Hong, Yi Gu, Di Jiang, Hanlin Gu, Chen Jason Zhang, Lu Wang, Zhiyang Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.04945)  

**Abstract**: Training automatic speech recognition (ASR) models increasingly relies on decentralized federated learning to ensure data privacy and accessibility, producing multiple local models that require effective merging. In hybrid ASR systems, while acoustic models can be merged using established methods, the language model (LM) for rescoring the N-best speech recognition list faces challenges due to the heterogeneity of non-neural n-gram models and neural network models. This paper proposes a heterogeneous LM optimization task and introduces a match-and-merge paradigm with two algorithms: the Genetic Match-and-Merge Algorithm (GMMA), using genetic operations to evolve and pair LMs, and the Reinforced Match-and-Merge Algorithm (RMMA), leveraging reinforcement learning for efficient convergence. Experiments on seven OpenSLR datasets show RMMA achieves the lowest average Character Error Rate and better generalization than baselines, converging up to seven times faster than GMMA, highlighting the paradigm's potential for scalable, privacy-preserving ASR systems. 

---
# AILS-NTUA at SemEval-2026 Task 3: Efficient Dimensional Aspect-Based Sentiment Analysis 

**Authors**: Stavros Gazetas, Giorgos Filandrianos, Maria Lymperaiou, Paraskevi Tzouveli, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2603.04933)  

**Abstract**: In this paper, we present AILS-NTUA system for Track-A of SemEval-2026 Task 3 on Dimensional Aspect-Based Sentiment Analysis (DimABSA), which encompasses three complementary problems: Dimensional Aspect Sentiment Regression (DimASR), Dimensional Aspect Sentiment Triplet Extraction (DimASTE), and Dimensional Aspect Sentiment Quadruplet Prediction (DimASQP) within a multilingual and multi-domain framework. Our methodology combines fine-tuning of language-appropriate encoder backbones for continuous aspect-level sentiment prediction with language-specific instruction tuning of large language models using LoRA for structured triplet and quadruplet extraction. This unified yet task-adaptive design emphasizes parameter-efficient specialization across languages and domains, enabling reduced training and inference requirements while maintaining strong effectiveness. Empirical results demonstrate that the proposed models achieve competitive performance and consistently surpass the provided baselines across most evaluation settings. 

---
# AILS-NTUA at SemEval-2026 Task 10: Agentic LLMs for Psycholinguistic Marker Extraction and Conspiracy Endorsement Detection 

**Authors**: Panagiotis Alexios Spanakis, Maria Lymperaiou, Giorgos Filandrianos, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2603.04921)  

**Abstract**: This paper presents a novel agentic LLM pipeline for SemEval-2026 Task 10 that jointly extracts psycholinguistic conspiracy markers and detects conspiracy endorsement. Unlike traditional classifiers that conflate semantic reasoning with structural localization, our decoupled design isolates these challenges. For marker extraction, we propose Dynamic Discriminative Chain-of-Thought (DD-CoT) with deterministic anchoring to resolve semantic ambiguity and character-level brittleness. For conspiracy detection, an "Anti-Echo Chamber" architecture, consisting of an adversarial Parallel Council adjudicated by a Calibrated Judge, overcomes the "Reporter Trap," where models falsely penalize objective reporting. Achieving 0.24 Macro F1 (+100\% over baseline) on S1 and 0.79 Macro F1 (+49\%) on S2, with the S1 system ranking 3rd on the development leaderboard, our approach establishes a versatile paradigm for interpretable, psycholinguistically-grounded NLP. 

---
# Can LLMs Capture Expert Uncertainty? A Comparative Analysis of Value Alignment in Ethnographic Qualitative Research 

**Authors**: Arina Kostina, Marios Dikaiakos, Alejandro Porcel, Tassos Stassopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2603.04897)  

**Abstract**: Qualitative analysis of open-ended interviews plays a central role in ethnographic and economic research by uncovering individuals' values, motivations, and culturally embedded financial behaviors. While large language models (LLMs) offer promising support for automating and enriching such interpretive work, their ability to produce nuanced, reliable interpretations under inherent task ambiguity remains unclear. In our work we evaluate LLMs on the task of identifying the top three human values expressed in long-form interviews based on the Schwartz Theory of Basic Values framework. We compare their outputs to expert annotations, analyzing both performance and uncertainty patterns relative to the experts. Results show that LLMs approach the human ceiling on set-based metrics (F1, Jaccard) but struggle to recover exact value rankings, as reflected in lower RBO scores. While the average Schwartz value distributions of most models closely match those of human analysts, their uncertainty structures across the Schwartz values diverge from expert uncertainty patterns. Among the evaluated models, Qwen performs closest to expert-level agreement and exhibits the strongest alignment with expert Schwartz value distributions. LLM ensemble methods yield consistent gains across metrics, with Majority Vote and Borda Count performing best. Notably, systematic overemphasis on certain Schwartz values, like Security, suggests both the potential of LLMs to provide complementary perspectives and the need to further investigate model-induced value biases. Overall, our findings highlight both the promise and the limitations of LLMs as collaborators in inherently ambiguous qualitative value analysis. 

---
# Free Lunch for Pass@$k$? Low Cost Diverse Sampling for Diffusion Language Models 

**Authors**: Sean Lamont, Christian Walder, Paul Montague, Amir Dezfouli, Michael Norrish  

**Link**: [PDF](https://arxiv.org/pdf/2603.04893)  

**Abstract**: Diverse outputs in text generation are necessary for effective exploration in complex reasoning tasks, such as code generation and mathematical problem solving. Such Pass@$k$ problems benefit from distinct candidates covering the solution space. However, traditional sampling approaches often waste computational resources on repetitive failure modes. While Diffusion Language Models have emerged as a competitive alternative to the prevailing Autoregressive paradigm, they remain susceptible to this redundancy, with independent samples frequently collapsing into similar modes. To address this, we propose a training free, low cost intervention to enhance generative diversity in Diffusion Language Models. Our approach modifies intermediate samples in a batch sequentially, where each sample is repelled from the feature space of previous samples, actively penalising redundancy. Unlike prior methods that require retraining or beam search, our strategy incurs negligible computational overhead, while ensuring that each sample contributes a unique perspective to the batch. We evaluate our method on the HumanEval and GSM8K benchmarks using the LLaDA-8B-Instruct model. Our results demonstrate significantly improved diversity and Pass@$k$ performance across various temperature settings. As a simple modification to the sampling process, our method offers an immediate, low-cost improvement for current and future Diffusion Language Models in tasks that benefit from diverse solution search. We make our code available at this https URL. 

---
# FireBench: Evaluating Instruction Following in Enterprise and API-Driven LLM Applications 

**Authors**: Yunfan Zhang, Yijie Bei, Jetashree Ravi, Pawel Garbacki  

**Link**: [PDF](https://arxiv.org/pdf/2603.04857)  

**Abstract**: Instruction following is critical for LLMs deployed in enterprise and API-driven settings, where strict adherence to output formats, content constraints, and procedural requirements is essential for enabling reliable LLM-assisted workflows. However, existing instruction following benchmarks predominantly evaluate natural language generation constraints that reflect the needs of chat assistants rather than enterprise users. To bridge this gap, we introduce FireBench, an LLM instruction following benchmark grounded in real-world enterprise and API usage patterns. FireBench evaluates six core capability dimensions across diverse applications including information extraction, customer support, and coding agents, comprising over 2,400 samples. We evaluate 11 LLMs and present key findings on their instruction following behavior in enterprise scenarios. We open-source FireBench at this http URL to help users assess model suitability, support model developers in diagnosing performance, and invite community contributions. 

---
# HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents 

**Authors**: Yilin Jiang, Fei Tan, Xuanyu Yin, Jing Leng, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.04855)  

**Abstract**: Student Personas (SPs) are emerging as infrastructure for educational LLMs, yet prior work often relies on ad-hoc prompting or hand-crafted profiles with limited control over educational theory and population distributions. We formalize this as Theory-Aligned and Distribution-Controllable Persona Generation (TAD-PG) and introduce HACHIMI, a multi-agent Propose-Validate-Revise framework that generates theory-aligned, quota-controlled personas. HACHIMI factorizes each persona into a theory-anchored educational schema, enforces developmental and psychological constraints via a neuro-symbolic validator, and combines stratified sampling with semantic deduplication to reduce mode collapse. The resulting HACHIMI-1M corpus comprises 1 million personas for Grades 1-12. Intrinsic evaluation shows near-perfect schema validity, accurate quotas, and substantial diversity, while external evaluation instantiates personas as student agents answering CEPS and PISA 2022 surveys; across 16 cohorts, math and curiosity/growth constructs align strongly between humans and agents, whereas classroom-climate and well-being constructs are only moderately aligned, revealing a fidelity gradient. All personas are generated with Qwen2.5-72B, and HACHIMI provides a standardized synthetic student population for group-level benchmarking and social-science simulations. Resources available at this https URL 

---
# SinhaLegal: A Benchmark Corpus for Information Extraction and Analysis in Sinhala Legislative Texts 

**Authors**: Minduli Lasandi, Nevidu Jayatilleke  

**Link**: [PDF](https://arxiv.org/pdf/2603.04854)  

**Abstract**: SinhaLegal introduces a Sinhala legislative text corpus containing approximately 2 million words across 1,206 legal documents. The dataset includes two types of legal documents: 1,065 Acts dated from 1981 to 2014 and 141 Bills from 2010 to 2014, which were systematically collected from official sources. The texts were extracted using OCR with Google Document AI, followed by extensive post-processing and manual cleaning to ensure high-quality, machine-readable content, along with dedicated metadata files for each document. A comprehensive evaluation was conducted, including corpus statistics, lexical diversity, word frequency analysis, named entity recognition, and topic modelling, demonstrating the structured and domain-specific nature of the corpus. Additionally, perplexity analysis using both large and small language models was performed to assess how effectively language models respond to domain-specific texts. The SinhaLegal corpus represents a vital resource designed to support NLP tasks such as summarisation, information extraction, and analysis, thereby bridging a critical gap in Sinhala legal research. 

---
# From Unfamiliar to Familiar: Detecting Pre-training Data via Gradient Deviations in Large Language Models 

**Authors**: Ruiqi Zhang, Lingxiang Wang, Hainan Zhang, Zhiming Zheng, Yanyan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2603.04828)  

**Abstract**: Pre-training data detection for LLMs is essential for addressing copyright concerns and mitigating benchmark contamination. Existing methods mainly focus on the likelihood-based statistical features or heuristic signals before and after fine-tuning, but the former are susceptible to word frequency bias in corpora, and the latter strongly depend on the similarity of fine-tuning data. From an optimization perspective, we observe that during training, samples transition from unfamiliar to familiar in a manner reflected by systematic differences in gradient behavior. Familiar samples exhibit smaller update magnitudes, distinct update locations in model components, and more sharply activated neurons. Based on this insight, we propose GDS, a method that identifies pre-training data by probing Gradient Deviation Scores of target samples. Specifically, we first represent each sample using gradient profiles that capture the magnitude, location, and concentration of parameter updates across FFN and Attention modules, revealing consistent distinctions between member and non-member data. These features are then fed into a lightweight classifier to perform binary membership inference. Experiments on five public datasets show that GDS achieves state-of-the-art performance with significantly improved cross-dataset transferability over strong baselines. Further interpretability analyse show gradient feature distribution differences, enabling practical and scalable pre-training data detection. 

---
# Autoscoring Anticlimax: A Meta-analytic Understanding of AI's Short-answer Shortcomings and Wording Weaknesses 

**Authors**: Michael Hardy  

**Link**: [PDF](https://arxiv.org/pdf/2603.04820)  

**Abstract**: Automated short-answer scoring lags other LLM applications. We meta-analyze 890 culminating results across a systematic review of LLM short-answer scoring studies, modeling the traditional effect size of Quadratic Weighted Kappa (QWK) with mixed effects metaregression. We quantitatively illustrate that that the level of difficulty for human experts to perform the task of scoring written work of children has no observed statistical effect on LLM performance. Particularly, we show that some scoring tasks measured as the easiest by human scorers were the hardest for LLMs. Whether by poor implementation by thoughtful researchers or patterns traceable to autoregressive training, on average decoder-only architectures underperform encoders by 0.37--a substantial difference in agreement with humans. Additionally, we measure the contributions of various aspects of LLM technology on successful scoring such as tokenizer vocabulary size, which exhibits diminishing returns--potentially due to undertrained tokens. Findings argue for systems design which better anticipates known statistical shortcomings of autoregressive models. Finally, we provide additional experiments to illustrate wording and tokenization sensitivity and bias elicitation in high-stakes education contexts, where LLMs demonstrate racial discrimination. Code and data for this study are available. 

---
# Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents 

**Authors**: Natchanon Pollertlam, Witchayut Kornsuwannawit  

**Link**: [PDF](https://arxiv.org/pdf/2603.04814)  

**Abstract**: Persistent conversational AI systems face a choice between passing full conversation histories to a long-context large language model (LLM) and maintaining a dedicated memory system that extracts and retrieves structured facts. We compare a fact-based memory system built on the Mem0 framework against long-context LLM inference on three memory-centric benchmarks - LongMemEval, LoCoMo, and PersonaMemv2 - and evaluate both architectures on accuracy and cumulative API cost. Long-context GPT-5-mini achieves higher factual recall on LongMemEval and LoCoMo, while the memory system is competitive on PersonaMemv2, where persona consistency depends on stable, factual attributes suited to flat-typed extraction. We construct a cost model that incorporates prompt caching and show that the two architectures have structurally different cost profiles: long-context inference incurs a per-turn charge that grows with context length even under caching, while the memory system's per-turn read cost remains roughly fixed after a one-time write phase. At a context length of 100k tokens, the memory system becomes cheaper after approximately ten interaction turns, with the break-even point decreasing as context length grows. These results characterize the accuracy-cost trade-off between the two approaches and provide a concrete criterion for selecting between them in production deployments. 

---
# Attention's Gravitational Field:A Power-Law Interpretation of Positional Correlation 

**Authors**: Edward Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04805)  

**Abstract**: This paper explores the underlying principles of positional relationships and encodings within Large Language Models (LLMs) and introduces the concept of the Attention Gravitational Field (AGF). By decoupling positional encodings from semantic embeddings, we optimize the model architecture and achieve superior accuracy compared to prevailing encoding methods. Furthermore, we provide an in-depth analysis of AGF, demonstrating its intrinsic consistency with learning and stability curves, as well as its empirical alignment with Newton's Law of Universal Gravitation. By offering a rigorous theoretical exploration of these phenomena, this work represents a significant step toward interpreting the Attention mechanism and unlocks new possibilities for future research in model optimization and interpretability. 

---
# TSEmbed: Unlocking Task Scaling in Universal Multimodal Embeddings 

**Authors**: Yebo Wu, Feng Liu, Ziwei Xie, Zhiyuan Liu, Changwang Zhang, Jun Wang, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.04772)  

**Abstract**: Despite the exceptional reasoning capabilities of Multimodal Large Language Models (MLLMs), their adaptation into universal embedding models is significantly impeded by task conflict. To address this, we propose TSEmbed, a universal multimodal embedding framework that synergizes Mixture-of-Experts (MoE) with Low-Rank Adaptation (LoRA) to explicitly disentangle conflicting task objectives. Moreover, we introduce Expert-Aware Negative Sampling (EANS), a novel strategy that leverages expert routing distributions as an intrinsic proxy for semantic similarity. By dynamically prioritizing informative hard negatives that share expert activation patterns with the query, EANS effectively sharpens the model's discriminative power and refines embedding boundaries. To ensure training stability, we further devise a two-stage learning paradigm that solidifies expert specialization before optimizing representations via EANS. TSEmbed achieves state-of-the-art performance on both the Massive Multimodal Embedding Benchmark (MMEB) and real-world industrial production datasets, laying a foundation for task-level scaling in universal multimodal embeddings. 

---
# Stacked from One: Multi-Scale Self-Injection for Context Window Extension 

**Authors**: Wei Han, Pan Zhou, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2603.04759)  

**Abstract**: The limited context window of contemporary large language models (LLMs) remains a primary bottleneck for their broader application across diverse domains. Although continual pre-training on long-context data offers a straightforward solution, it incurs prohibitive data acquisition and computational costs. To address this challenge, we propose~\modelname, a novel framework based on multi-grained context compression and query-aware information acquisition. SharedLLM comprises two stacked short-context LLMs: a lower model serving as a compressor and an upper model acting as a decoder. The lower model compresses long inputs into compact, multi-grained representations, which are then forwarded to the upper model for context-aware processing. To maximize efficiency, this information transfer occurs exclusively at the lowest layers, bypassing lengthy forward passes and redundant cross-attention operations. This entire process, wherein the upper and lower models are derived from the same underlying LLM layers, is termed~\textit{self-injection}. To support this architecture, a specialized tree-based data structure enables the efficient encoding and query-aware retrieval of contextual information. Despite being trained on sequences of only 8K tokens, \modelname~effectively generalizes to inputs exceeding 128K tokens. Across a comprehensive suite of long-context modeling and understanding benchmarks, \modelname~achieves performance superior or comparable to strong baselines, striking an optimal balance between efficiency and accuracy. Furthermore, these design choices allow \modelname~to substantially reduce the memory footprint and yield notable inference speedups ($2\times$ over streaming and $3\times$ over encoder-decoder architectures). 

---
# IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation 

**Authors**: Bosi Wen, Yilin Niu, Cunxiang Wang, Xiaoying Ling, Ying Zhang, Pei Ke, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04738)  

**Abstract**: Instruction-following is a foundational capability of large language models (LLMs), with its improvement hinging on scalable and accurate feedback from judge models. However, the reliability of current judge models in instruction-following remains underexplored due to several deficiencies of existing meta-evaluation benchmarks, such as their insufficient data coverage and oversimplified pairwise evaluation paradigms that misalign with model optimization scenarios. To this end, we propose IF-RewardBench, a comprehensive meta-evaluation benchmark for instruction-following that covers diverse instruction and constraint types. For each instruction, we construct a preference graph containing all pairwise preferences among multiple responses based on instruction-following quality. This design enables a listwise evaluation paradigm that assesses the capabilities of judge models to rank multiple responses, which is essential in guiding model alignment. Extensive experiments on IF-RewardBench reveal significant deficiencies in current judge models and demonstrate that our benchmark achieves a stronger positive correlation with downstream task performance compared to existing benchmarks. Our codes and data are available at this https URL. 

---
# AI-Assisted Moot Courts: Simulating Justice-Specific Questioning in Oral Arguments 

**Authors**: Kylie Zhang, Nimra Nadeem, Lucia Zheng, Dominik Stammbach, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2603.04718)  

**Abstract**: In oral arguments, judges probe attorneys with questions about the factual record, legal claims, and the strength of their arguments. To prepare for this questioning, both law schools and practicing attorneys rely on moot courts: practice simulations of appellate hearings. Leveraging a dataset of U.S. Supreme Court oral argument transcripts, we examine whether AI models can effectively simulate justice-specific questioning for moot court-style training. Evaluating oral argument simulation is challenging because there is no single correct question for any given turn. Instead, effective questioning should reflect a combination of desirable qualities, such as anticipating substantive legal issues, detecting logical weaknesses, and maintaining an appropriately adversarial tone. We introduce a two-layer evaluation framework that assesses both the realism and pedagogical usefulness of simulated questions using complementary proxy metrics. We construct and evaluate both prompt-based and agentic oral argument simulators. We find that simulated questions are often perceived as realistic by human annotators and achieve high recall of ground truth substantive legal issues. However, models still face substantial shortcomings, including low diversity in question types and sycophancy. Importantly, these shortcomings would remain undetected under naive evaluation approaches. 

---
# Detection of Illicit Content on Online Marketplaces using Large Language Models 

**Authors**: Quoc Khoa Tran, Thanh Thi Nguyen, Campbell Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2603.04707)  

**Abstract**: Online marketplaces, while revolutionizing global commerce, have inadvertently facilitated the proliferation of illicit activities, including drug trafficking, counterfeit sales, and cybercrimes. Traditional content moderation methods such as manual reviews and rule-based automated systems struggle with scalability, dynamic obfuscation techniques, and multilingual content. Conventional machine learning models, though effective in simpler contexts, often falter when confronting the semantic complexities and linguistic nuances characteristic of illicit marketplace communications. This research investigates the efficacy of Large Language Models (LLMs), specifically Meta's Llama 3.2 and Google's Gemma 3, in detecting and classifying illicit online marketplace content using the multilingual DUTA10K dataset. Employing fine-tuning techniques such as Parameter-Efficient Fine-Tuning (PEFT) and quantization, these models were systematically benchmarked against a foundational transformer-based model (BERT) and traditional machine learning baselines (Support Vector Machines and Naive Bayes). Experimental results reveal a task-dependent advantage for LLMs. In binary classification (illicit vs. non-illicit), Llama 3.2 demonstrated performance comparable to traditional methods. However, for complex, imbalanced multi-class classification involving 40 specific illicit categories, Llama 3.2 significantly surpassed all baseline models. These findings offer substantial practical implications for enhancing online safety, equipping law enforcement agencies, e-commerce platforms, and cybersecurity specialists with more effective, scalable, and adaptive tools for illicit content detection and moderation. 

---
# Hate Speech Detection using Large Language Models with Data Augmentation and Feature Enhancement 

**Authors**: Brian Jing Hong Nge, Stefan Su, Thanh Thi Nguyen, Campbell Wilson, Alexandra Phelan, Naomi Pfitzner  

**Link**: [PDF](https://arxiv.org/pdf/2603.04698)  

**Abstract**: This paper evaluates data augmentation and feature enhancement techniques for hate speech detection, comparing traditional classifiers, e.g., Delta Term Frequency-Inverse Document Frequency (Delta TF-IDF), with transformer-based models (DistilBERT, RoBERTa, DeBERTa, Gemma-7B, gpt-oss-20b) across diverse datasets. It examines the impact of Synthetic Minority Over-sampling Technique (SMOTE), weighted loss determined by inverse class proportions, Part-of-Speech (POS) tagging, and text data augmentation on model performance. The open-source gpt-oss-20b consistently achieves the highest results. On the other hand, Delta TF-IDF responds strongly to data augmentation, reaching 98.2% accuracy on the Stormfront dataset. The study confirms that implicit hate speech is more difficult to detect than explicit hateful content and that enhancement effectiveness depends on dataset, model, and technique interaction. Our research informs the development of hate speech detection by highlighting how dataset properties, model architectures, and enhancement strategies interact, supporting more accurate and context-aware automated detection. 

---
# Non-Zipfian Distribution of Stopwords and Subset Selection Models 

**Authors**: Wentian Li, Oscar Fontanelli  

**Link**: [PDF](https://arxiv.org/pdf/2603.04691)  

**Abstract**: Stopwords are words that are not very informative to the content or the meaning of a language text. Most stopwords are function words but can also be common verbs, adjectives and adverbs. In contrast to the well known Zipf's law for rank-frequency plot for all words, the rank-frequency plot for stopwords are best fitted by the Beta Rank Function (BRF). On the other hand, the rank-frequency plots of non-stopwords also deviate from the Zipf's law, but are fitted better by a quadratic function of log-token-count over log-rank than by BRF. Based on the observed rank of stopwords in the full word list, we propose a stopword (subset) selection model that the probability for being selected as a function of the word's rank $r$ is a decreasing Hill's function ($1/(1+(r/r_{mid})^\gamma)$); whereas the probability for not being selected is the standard Hill's function ( $1/(1+(r_{mid}/r)^\gamma)$). We validate this selection probability model by a direct estimation from an independent collection of texts. We also show analytically that this model leads to a BRF rank-frequency distribution for stopwords when the original full word list follows the Zipf's law, as well as explaining the quadratic fitting function for the non-stopwords. 

---
# Optimizing Language Models for Crosslingual Knowledge Consistency 

**Authors**: Tianyu Liu, Jirui Qi, Mrinmaya Sachan, Ryan Cotterell, Raquel Fernández, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2603.04678)  

**Abstract**: Large language models are known to often exhibit inconsistent knowledge. This is particularly problematic in multilingual scenarios, where models are likely to be asked similar questions in different languages, and inconsistent responses can undermine their reliability. In this work, we show that this issue can be mitigated using reinforcement learning with a structured reward function, which leads to an optimal policy with consistent crosslingual responses. We introduce Direct Consistency Optimization (DCO), a DPO-inspired method that requires no explicit reward model and is derived directly from the LLM itself. Comprehensive experiments show that DCO significantly improves crosslingual consistency across diverse LLMs and outperforms existing methods when training with samples of multiple languages, while complementing DPO when gold labels are available. Extra experiments demonstrate the effectiveness of DCO in bilingual settings, significant out-of-domain generalizability, and controllable alignment via direction hyperparameters. Taken together, these results establish DCO as a robust and efficient solution for improving knowledge consistency across languages in multilingual LLMs. All code, training scripts, and evaluation benchmarks are released at this https URL. 

---
# Stan: An LLM-based thermodynamics course assistant 

**Authors**: Eric M. Furst, Vasudevan Venkateshwaran  

**Link**: [PDF](https://arxiv.org/pdf/2603.04657)  

**Abstract**: Discussions of AI in education focus predominantly on student-facing tools -- chatbots, tutors, and problem generators -- while the potential for the same infrastructure to support instructors remains largely unexplored. We describe Stan, a suite of tools for an undergraduate chemical engineering thermodynamics course built on a data pipeline that we develop and deploy in dual roles: serving students and supporting instructors from a shared foundation of lecture transcripts and a structured textbook index. On the student side, a retrieval-augmented generation (RAG) pipeline answers natural-language queries by extracting technical terms, matching them against the textbook index, and synthesizing grounded responses with specific chapter and page references. On the instructor side, the same transcript corpus is processed through structured analysis pipelines that produce per-lecture summaries, identify student questions and moments of confusion, and catalog the anecdotes and analogies used to motivate difficult material -- providing a searchable, semester-scale record of teaching that supports course reflection, reminders, and improvement. All components, including speech-to-text transcription, structured content extraction, and interactive query answering, run entirely on locally controlled hardware using open-weight models (Whisper large-v3, Llama~3.1 8B) with no dependence on cloud APIs, ensuring predictable costs, full data privacy, and reproducibility independent of third-party services. We describe the design, implementation, and practical failure modes encountered when deploying 7--8 billion parameter models for structured extraction over long lecture transcripts, including context truncation, bimodal output distributions, and schema drift, along with the mitigations that resolved them. 

---
# iAgentBench: Benchmarking Sensemaking Capabilities of Information-Seeking Agents on High-Traffic Topics 

**Authors**: Preetam Prabhu Srikar Dammu, Arnav Palkhiwala, Tanya Roosta, Chirag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2603.04656)  

**Abstract**: With the emergence of search-enabled generative QA systems, users are increasingly turning to tools that browse, aggregate, and reconcile evidence across multiple sources on their behalf. Yet many widely used QA benchmarks remain answerable by retrieving a single relevant passage, making them poorly suited for measuring cross-source sensemaking, such as integrating evidence, tracking causal links, and resolving dependencies across facets of a topic. We present iAgentBench, a dynamic ODQA benchmark that targets these higher-level information needs while keeping questions natural and grounded in realistic information-seeking behavior. iAgentBench draws seed topics from real-world attention signals and uses common user intent patterns to construct user-like questions whose answers require combining evidence from multiple sources, not just extracting a single snippet. Each instance is released with traceable evidence and auditable intermediate artifacts that support contamination checks and enable fine-grained diagnosis of failures in retrieval versus synthesis. Experiments across multiple LLMs show that retrieval improves accuracy, but retrieval alone does not reliably resolve these questions, underscoring the need to evaluate evidence use, not just evidence access. 

---
# Coordinated Semantic Alignment and Evidence Constraints for Retrieval-Augmented Generation with Large Language Models 

**Authors**: Xin Chen, Saili Uday Gadgil, Jiarong Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.04647)  

**Abstract**: Retrieval augmented generation mitigates limitations of large language models in factual consistency and knowledge updating by introducing external knowledge. However, practical applications still suffer from semantic misalignment between retrieved results and generation objectives, as well as insufficient evidence utilization. To address these challenges, this paper proposes a retrieval augmented generation method that integrates semantic alignment with evidence constraints through coordinated modeling of retrieval and generation stages. The method first represents the relevance between queries and candidate evidence within a unified semantic space. This ensures that retrieved results remain semantically consistent with generation goals and reduces interference from noisy evidence and semantic drift. On this basis, an explicit evidence constraint mechanism is introduced. Retrieved evidence is transformed from an implicit context into a core control factor in generation. This restricts the expression scope of generated content and strengthens dependence on evidence. By jointly modeling semantic consistency and evidence constraints within a unified framework, the proposed approach improves factual reliability and verifiability while preserving natural language fluency. Comparative results show stable improvements across multiple generation quality metrics. This confirms the effectiveness and necessity of coordinated semantic alignment and evidence constraint modeling in retrieval augmented generation tasks. 

---
# Bootstrapping Exploration with Group-Level Natural Language Feedback in Reinforcement Learning 

**Authors**: Lei Huang, Xiang Cheng, Chenxiao Zhao, Guobin Shen, Junjie Yang, Xiaocheng Feng, Yuxuan Gu, Xing Yu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2603.04597)  

**Abstract**: Large language models (LLMs) typically receive diverse natural language (NL) feedback through interaction with the environment. However, current reinforcement learning (RL) algorithms rely solely on scalar rewards, leaving the rich information in NL feedback underutilized and leading to inefficient exploration. In this work, we propose GOLF, an RL framework that explicitly exploits group-level language feedback to guide targeted exploration through actionable refinements. GOLF aggregates two complementary feedback sources: (i) external critiques that pinpoint errors or propose targeted fixes, and (ii) intra-group attempts that supply alternative partial ideas and diverse failure patterns. These group-level feedbacks are aggregated to produce high-quality refinements, which are adaptively injected into training as off-policy scaffolds to provide targeted guidance in sparse-reward regions. Meanwhile, GOLF jointly optimizes generation and refinement within a unified RL loop, creating a virtuous cycle that continuously improves both capabilities. Experiments on both verifiable and non-verifiable benchmarks show that GOLF achieves superior performance and exploration efficiency, achieving 2.2$\times$ improvements in sample efficiency compared to RL methods trained solely on scalar rewards. Code is available at this https URL. 

---
# From Static Inference to Dynamic Interaction: Navigating the Landscape of Streaming Large Language Models 

**Authors**: Junlong Tong, Zilong Wang, YuJie Ren, Peiran Yin, Hao Wu, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.04592)  

**Abstract**: Standard Large Language Models (LLMs) are predominantly designed for static inference with pre-defined inputs, which limits their applicability in dynamic, real-time scenarios. To address this gap, the streaming LLM paradigm has emerged. However, existing definitions of streaming LLMs remain fragmented, conflating streaming generation, streaming inputs, and interactive streaming architectures, while a systematic taxonomy is still lacking. This paper provides a comprehensive overview and analysis of streaming LLMs. First, we establish a unified definition of streaming LLMs based on data flow and dynamic interaction to clarify existing ambiguities. Building on this definition, we propose a systematic taxonomy of current streaming LLMs and conduct an in-depth discussion on their underlying methodologies. Furthermore, we explore the applications of streaming LLMs in real-world scenarios and outline promising research directions to support ongoing advances in streaming intelligence. We maintain a continuously updated repository of relevant papers at this https URL. 

---
# Query Disambiguation via Answer-Free Context: Doubling Performance on Humanity's Last Exam 

**Authors**: Michael Majurski, Cynthia Matuszek  

**Link**: [PDF](https://arxiv.org/pdf/2603.04454)  

**Abstract**: How carefully and unambiguously a question is phrased has a profound impact on the quality of the response, for Language Models (LMs) as well as people. While model capabilities continue to advance, the interplay between grounding context and query formulation remains under-explored. This work investigates how the quality of background grounding information in a model's context window affects accuracy. We find that combining well-grounded dynamic context construction (i.e, RAG) with query rewriting reduces question ambiguity, resulting in significant accuracy gains. Given a user question with associated answer-free grounding context, rewriting the question to reduce ambiguity produces benchmark improvements without changing the answer itself, even compared to prepending that context before the question. Using \texttt{gpt-oss-20b} to rewrite a subset of Humanity's Last Exam using answer-free grounding context improves \texttt{gpt-5-mini} accuracy from 0.14 to 0.37. We demonstrate that this accuracy improvement cannot be fully recovered just through prompting at inference time; rather, distinct rewriting and answering phases are required. Code and data are available at this https URL 

---
# Induced Numerical Instability: Hidden Costs in Multimodal Large Language Models 

**Authors**: Wai Tuck Wong, Jun Sun, Arunesh Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2603.04453)  

**Abstract**: The use of multimodal large language models has become widespread, and as such the study of these models and their failure points has become of utmost importance. We study a novel mode of failure that causes degradation in performance indirectly by optimizing a loss term that seeks to maximize numerical instability in the inference stage of these models. We apply this loss term as the optimization target to construct images that, when used on multimodal large language models, cause significant degradation in the output. We validate our hypothesis on state of the art models large vision language models (LLaVa-v1.5-7B, Idefics3-8B, SmolVLM-2B-Instruct) against standard datasets (Flickr30k, MMVet, TextVQA, VQAv2, POPE, COCO) and show that performance degrades significantly, even with a very small change to the input image, compared to baselines. Our results uncover a fundamentally different vector of performance degradation, highlighting a failure mode not captured by adversarial perturbations. 

---
# A unified foundational framework for knowledge injection and evaluation of Large Language Models in Combustion Science 

**Authors**: Zonglin Yang, Runze Mao, Tianhao Wu, Han Li, QingGuo Zhou, Zhi X. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.04452)  

**Abstract**: To advance foundation Large Language Models (LLMs) for combustion science, this study presents the first end-to-end framework for developing domain-specialized models for the combustion community. The framework comprises an AI-ready multimodal knowledge base at the 3.5 billion-token scale, extracted from over 200,000 peer-reviewed articles, 8,000 theses and dissertations, and approximately 400,000 lines of combustion CFD code; a rigorous and largely automated evaluation benchmark (CombustionQA, 436 questions across eight subfields); and a three-stage knowledge-injection pathway that progresses from lightweight retrieval-augmented generation (RAG) to knowledge-graph-enhanced retrieval and continued pretraining. We first quantitatively validate Stage 1 (naive RAG) and find a hard ceiling: standard RAG accuracy peaks at 60%, far surpassing zero-shot performance (23%) yet well below the theoretical upper bound (87%). We further demonstrate that this stage's performance is severely constrained by context contamination. Consequently, building a domain foundation model requires structured knowledge graphs and continued pretraining (Stages 2 and 3). 

---
# What Is Missing: Interpretable Ratings for Large Language Model Outputs 

**Authors**: Nicholas Stranges, Yimin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04429)  

**Abstract**: Current Large Language Model (LLM) preference learning methods such as Proximal Policy Optimization and Direct Preference Optimization learn from direct rankings or numerical ratings of model outputs, these rankings are subjective, and a single numerical rating chosen directly by a judge is a poor proxy for the quality of natural language, we introduce the What Is Missing (WIM) rating system to produce rankings from natural-language feedback, WIM integrates into existing training pipelines, can be combined with other rating techniques, and can be used as input to any preference learning method without changing the learning algorithm, to compute a WIM rating, a human or LLM judge writes feedback describing what the model output is missing, we embed the output and the feedback with a sentence embedding model and compute the cosine similarity between the resulting vectors, we empirically observe that, compared to discrete numerical ratings, WIM yields fewer ties and larger rating deltas, which improves the availability of a learning signal in pairwise preference data, we use interpretable in the following limited sense: for each scalar rating, we can inspect the judge's missing-information text that produced it, enabling qualitative debugging of the preference labels. 

---
# Generating Realistic, Protocol-Compliant Maritime Radio Dialogues using Self-Instruct and Low-Rank Adaptation 

**Authors**: Gürsel Akdeniz, Emin Cagatay Nakilcioglu  

**Link**: [PDF](https://arxiv.org/pdf/2603.04423)  

**Abstract**: VHF radio miscommunication remains a major safety risk in maritime operations, with human factors accounting for over 58% of recorded incidents in Europe between 2014 and 2023. Despite decades of operational use, VHF radio communications are still prone to noise, interference, linguistic variability, and the absence of real-time transcription, making procedural errors both frequent and difficult to correct. Developing AI-assisted systems to support real-time communication and decision-making requires a considerable amount of high-quality maritime data, yet operational, regulatory, and privacy constraints render such datasets scarce. This study introduces a compliance aware Self-Instruct methodology for generating realistic maritime radio dialogues that conform to the IMO's SMCP. Our approach integrates a 26-filter verification pipeline directly into the iterative generation loop to enforce entity information accuracy, hallucination detection, SMCP-compliance, logical consistency, and linguistic diversity. We employ LORA for parameter-efficient fine-tuning, reducing computational overhead during training and enabling efficient deployment of the resulting models on resource-constrained maritime systems. To assess dataset quality, we introduce a novel evaluation framework combining automated and expert assessments: Format Accuracy, Information Accuracy, Uniqueness, and Logical Coherence. Experiments using publicly available vessel, coastal and AIS datasets demonstrate that the approach produces synthetically diverse, procedurally compliant, and operationally realistic dialogues. Although downstream applications such as automatic speech recognition and natural language processing are reserved for future work, the released code, datasets, and verification tools provide a reproducible foundation for artificial intelligence-assisted maritime safety and other safety-critical domains. 

---
# Do Mixed-Vendor Multi-Agent LLMs Improve Clinical Diagnosis? 

**Authors**: Grace Chang Yuan, Xiaoman Zhang, Sung Eun Kim, Pranav Rajpurkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.04421)  

**Abstract**: Multi-agent large language model (LLM) systems have emerged as a promising approach for clinical diagnosis, leveraging collaboration among agents to refine medical reasoning. However, most existing frameworks rely on single-vendor teams (e.g., multiple agents from the same model family), which risk correlated failure modes that reinforce shared biases rather than correcting them. We investigate the impact of vendor diversity by comparing Single-LLM, Single-Vendor, and Mixed-Vendor Multi-Agent Conversation (MAC) frameworks. Using three doctor agents instantiated with o4-mini, Gemini-2.5-Pro, and Claude-4.5-Sonnet, we evaluate performance on RareBench and DiagnosisArena. Mixed-vendor configurations consistently outperform single-vendor counterparts, achieving state-of-the-art recall and accuracy. Overlap analysis reveals the underlying mechanism: mixed-vendor teams pool complementary inductive biases, surfacing correct diagnoses that individual models or homogeneous teams collectively miss. These results highlight vendor diversity as a key design principle for robust clinical diagnostic systems. 

---
# Context-Dependent Affordance Computation in Vision-Language Models 

**Authors**: Murad Farzulla  

**Link**: [PDF](https://arxiv.org/pdf/2603.04419)  

**Abstract**: We characterize the phenomenon of context-dependent affordance computation in vision-language models (VLMs). Through a large-scale computational study (n=3,213 scene-context pairs from COCO-2017) using Qwen-VL 30B and LLaVA-1.5-13B subject to systematic context priming across 7 agentic personas, we demonstrate massive affordance drift: mean Jaccard similarity between context conditions is 0.095 (95% CI: [0.093, 0.096], p < 0.0001), indicating that >90% of lexical scene description is context-dependent. Sentence-level cosine similarity confirms substantial drift at the semantic level (mean = 0.415, 58.5% context-dependent). Stochastic baseline experiments (2,384 inference runs across 4 temperatures and 5 seeds) confirm this drift reflects genuine context effects rather than generation noise: within-prime variance is substantially lower than cross-prime variance across all conditions. Tucker decomposition with bootstrap stability analysis (n=1,000 resamples) reveals stable orthogonal latent factors: a "Culinary Manifold" isolated to chef contexts and an "Access Axis" spanning child-mobility contrasts. These findings establish that VLMs compute affordances in a substantially context-dependent manner -- with the difference between lexical (90%) and semantic (58.5%) measures reflecting that surface vocabulary changes more than underlying meaning under context shifts -- and suggest a direction for robotics research: dynamic, query-dependent ontological projection (JIT Ontology) rather than static world modeling. We do not claim to establish processing order or architectural primacy; such claims require internal representational analysis beyond output behavior. 

---
# Same Input, Different Scores: A Multi Model Study on the Inconsistency of LLM Judge 

**Authors**: Fiona Lau  

**Link**: [PDF](https://arxiv.org/pdf/2603.04417)  

**Abstract**: Large language models are increasingly used as automated evaluators in research and enterprise settings, a practice known as LLM-as-a-judge. While prior work has examined accuracy, bias, and alignment with human preferences, far less attention has been given to how consistently LLMs assign numerical scores, an important concern for many production workflows. This study systematically evaluates scoring stability across five commonly used models, GPT-4o, GPT-4o-mini, Gemini-2.5-Flash, Claude-Haiku-4.5, and Claude-Sonnet-4.5, two temperature settings, and real enterprise question-answer pairs drawn from a retrieval-augmented generation (RAG) system. We address three questions: how stable a model's scores are across repeated runs, how differently models score identical inputs, and how temperature affects scoring consistency. Temperature controls the determinism of an LLM's output. Despite expectations of stability at temperature=0, we observe substantial variability across models, with completeness scoring showing the largest fluctuations. Cross-model comparisons reveal systematic differences in strictness and interpretive style, leading to divergent ratings for the same answers. Lower temperatures improve stability for some models, notably GPT-4o and Gemini, but have limited or inconsistent effects for Anthropic models. These findings have important implications for enterprise pipelines that rely on LLM-generated scores for routing, triage, gating, or quality control. Identical inputs can receive different scores depending on model, family, or temperature, raising concerns around fairness, reproducibility, and operational reliability. Our results highlight the need for monitoring, robust parsing, and hybrid human-LLM evaluation strategies to ensure dependable use of LLM-as-a-judge in production environments. 

---
# Optimizing What We Trust: Reliability-Guided QUBO Selection of Multi-Agent Weak Framing Signals for Arabic Sentiment Prediction 

**Authors**: Rabab Alkhalifa  

**Link**: [PDF](https://arxiv.org/pdf/2603.04416)  

**Abstract**: Framing detection in Arabic social media is difficult due to interpretive ambiguity, cultural grounding, and limited reliable supervision. Existing LLM-based weak supervision methods typically rely on label aggregation, which is brittle when annotations are few and socially dependent. We propose a reliability-aware weak supervision framework that shifts the focus from label fusion to data curation. A small multi-agent LLM pipeline, two framers, a critic, and a discriminator, treats disagreement and reasoning quality as epistemic signals and produces instance-level reliability estimates. These estimates guide a QUBO-based subset selection procedure that enforces frame balance while reducing redundancy. Intrinsic diagnostics and an out-of-domain Arabic sentiment transfer test show that the selected subsets are more reliable and encode non-random, transferable structure, without degrading strong text-only baselines. 

---
# The Thinking Boundary: Quantifying Reasoning Suitability of Multimodal Tasks via Dual Tuning 

**Authors**: Ruobing Zheng, Tianqi Li, Jianing Li, Qingpei Guo, Yi Yuan, Jingdong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.04415)  

**Abstract**: While reasoning-enhanced Large Language Models (LLMs) have demonstrated remarkable advances in complex tasks such as mathematics and coding, their effectiveness across universal multimodal scenarios remains uncertain. The trend of releasing parallel "Instruct" and "Thinking" models by leading developers serves merely as a resource-intensive workaround, stemming from the lack of a criterion for determining when reasoning is truly beneficial. In this paper, we propose Dual Tuning, a framework designed to assess whether reasoning yields positive gains for target tasks under given base models and datasets. By jointly fine-tuning on paired Chain-of-Thought (CoT) and Direct-Answer (DA) data under controlled prompts, we systematically quantify and compare the gains of both training modes using the proposed metrics, and establish the "Thinking Boundary" to evaluate the suitability of reasoning training across diverse multimodal tasks, including spatial, mathematical, and multi-disciplinary domains. We further explore the impact of reinforcement training and thinking patterns on reasoning suitability, and validate whether the "Thinking Boundary" can guide data refinement. Our findings challenge the "reasoning-for-all" paradigm, providing practical guidance for identifying appropriate data and training strategies, and motivating the development of resource-efficient, adaptive auto-think systems. 

---
# Multiclass Hate Speech Detection with RoBERTa-OTA: Integrating Transformer Attention and Graph Convolutional Networks 

**Authors**: Mahmoud Abusaqer, Jamil Saquer  

**Link**: [PDF](https://arxiv.org/pdf/2603.04414)  

**Abstract**: Multiclass hate speech detection across demographic categories remains computationally challenging due to implicit targeting strategies and linguistic variability in social media content. Existing approaches rely solely on learned representations from training data, without explicitly incorporating structured ontological frameworks that can enhance classification through formal domain knowledge integration. We propose RoBERTa-OTA, which introduces ontology-guided attention mechanisms that process textual features alongside structured knowledge representations through enhanced Graph Convolutional Networks. The architecture combines RoBERTa embeddings with scaled attention layers and graph neural networks to integrate contextual language understanding with domain-specific semantic knowledge. Evaluation across 39,747 balanced samples using 5-fold cross-validation demonstrates significant performance gains over baseline RoBERTa implementations and existing state-of-the-art methods. RoBERTa-OTA achieves 96.04\% accuracy compared to 95.02\% for standard RoBERTa, with substantial improvements for challenging categories: gender-based hate speech detection improves by 2.36 percentage points while other hate speech categories improve by 2.38 percentage points. The enhanced architecture maintains computational efficiency with only 0.33\% parameter overhead, providing practical advantages for large-scale content moderation applications requiring fine-grained demographic hate speech classification. 

---
# Simulating Meaning, Nevermore! Introducing ICR: A Semiotic-Hermeneutic Metric for Evaluating Meaning in LLM Text Summaries 

**Authors**: Natalie Perez, Sreyoshi Bhaduri, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2603.04413)  

**Abstract**: Meaning in human language is relational, context dependent, and emergent, arising from dynamic systems of signs rather than fixed word-concept mappings. In computational settings, this semiotic and interpretive complexity complicates the generation and evaluation of meaning. This article proposes an interdisciplinary framework for studying meaning in large language model (LLM) generated language by integrating semiotics and hermeneutics with qualitative research methods. We review prior scholarship on meaning and machines, examining how linguistic signs are transformed into vectorized representations in static and contextualized embedding models, and identify gaps between statistical approximation and human interpretive meaning. We then introduce the Inductive Conceptual Rating (ICR) metric, a qualitative evaluation approach grounded in inductive content analysis and reflexive thematic analysis, designed to assess semantic accuracy and meaning alignment in LLM-outputs beyond lexical similarity metrics. We apply ICR in an empirical comparison of LLM generated and human generated thematic summaries across five datasets (N = 50 to 800). While LLMs achieve high linguistic similarity, they underperform on semantic accuracy, particularly in capturing contextually grounded meanings. Performance improves with larger datasets but remains variable across models, potentially reflecting differences in the frequency and coherence of recurring concepts and meanings. We conclude by arguing for evaluation frameworks that leverage systematic qualitative interpretation practices when assessing meaning in LLM-generated outputs from reference texts. 

---
# Additive Multi-Step Markov Chains and the Curse of Dimensionality in Large Language Models 

**Authors**: O.V. Usatenko, S.S. Melnyk, G.M. Pritula  

**Link**: [PDF](https://arxiv.org/pdf/2603.04412)  

**Abstract**: Large-scale language models (LLMs) operate in extremely high-dimensional state spaces, where both token embeddings and their hidden representations create complex dependencies that are not easily reduced to classical Markov structures. In this paper, we explore a theoretically feasible approximation of LLM dynamics using N-order additive Markov chains. Such models allow the conditional probability of the next token to be decomposed into a superposition of contributions from multiple historical depths, reducing the combinatorial explosion typically associated with high-order Markov processes. The main result of the work is the establishment of a correspondence between an additive multi-step chain and a chain with a step-wise memory function. This equivalence allowed the introduction of the concept of information temperature not only for stepwise but also for additive N-order Markov chains. 

---
# One Size Does Not Fit All: Token-Wise Adaptive Compression for KV Cache 

**Authors**: Liming Lu, Kaixi Qiu, Jiayu Zhou, Jushi Kai, Haoyan Zhang, Huanyu Wang, Jingwen Leng, Ziwei He, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.04411)  

**Abstract**: Despite the remarkable progress of Large Language Models (LLMs), the escalating memory footprint of the Key-Value (KV) cache remains a critical bottleneck for efficient inference. While dimensionality reduction offers a promising compression avenue, existing approaches typically either necessitate prohibitively expensive pre-training from scratch or suffer from severe performance deterioration under high compression regimes. In this work, we propose DynaKV, a novel post-training framework for low-rank KV cache compression. To the best of our knowledge, DynaKV is the first method to dynamically allocate compression rates to individual tokens according to their semantic meaning, which allows it to achieve better fidelity at aggressive compression ratios. Extensive experiments demonstrate that our method consistently outperforms existing state-of-the-art compression techniques, achieving significant memory reduction while maintaining competitive generation quality. Furthermore, our approach is orthogonal to sequence-level pruning methods. When integrated with SnapKV, DynaKV retains only 6% of the KV cache while maintaining 94% of the baseline performance on the LongBench benchmark. 

---
# SalamahBench: Toward Standardized Safety Evaluation for Arabic Language Models 

**Authors**: Omar Abdelnasser, Fatemah Alharbi, Khaled Khasawneh, Ihsen Alouani, Mohammed E. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2603.04410)  

**Abstract**: Safety alignment in Language Models (LMs) is fundamental for trustworthy AI. However, while different stakeholders are trying to leverage Arabic Language Models (ALMs), systematic safety evaluation of ALMs remains largely underexplored, limiting their mainstream uptake. Existing safety benchmarks and safeguard models are predominantly English-centric, limiting their applicability to Arabic Natural Language Processing (NLP) systems and obscuring fine-grained, category-level safety vulnerabilities. This paper introduces SalamaBench, a unified benchmark for evaluating the safety of ALMs, comprising $8,170$ prompts across $12$ different categories aligned with the MLCommons Safety Hazard Taxonomy. Constructed by harmonizing heterogeneous datasets through a rigorous pipeline involving AI filtering and multi-stage human verification, SalamaBench enables standardized, category-aware safety evaluation. Using this benchmark, we evaluate five state-of-the-art ALMs, including Fanar 1 and 2, ALLaM 2, Falcon H1R, and Jais 2, under multiple safeguard configurations, including individual guard models, majority-vote aggregation, and validation against human-annotated gold labels. Our results reveal substantial variation in safety alignment: while Fanar 2 achieves the lowest aggregate attack success rates, its robustness is uneven across specific harm domains. In contrast, Jais 2 consistently exhibits elevated vulnerability, indicating weaker intrinsic safety alignment. We further demonstrate that native ALMs perform substantially worse than dedicated safeguard models when acting as safety judges. Overall, our findings highlight the necessity of category-aware evaluation and specialized safeguard mechanisms for robust harm mitigation in ALMs. 

---
# Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework 

**Authors**: Nora Petrova, Andrew Gordon, Enzo Blindow  

**Link**: [PDF](https://arxiv.org/pdf/2603.04409)  

**Abstract**: The evaluation of large language models faces significant challenges. Technical benchmarks often lack real-world relevance, while existing human preference evaluations suffer from unrepresentative sampling, superficial assessment depth, and single-metric reductionism. To address these issues, we introduce HUMAINE, a framework for multidimensional, demographically aware measurement of human-AI interaction. We collected multi-turn, naturalistic conversations from 23,404 participants that were stratified across 22 demographic groups, both in the US and UK, to evaluate 28 state-of-the-art models across five human-centric dimensions. We use a hierarchical Bayesian Bradley-Terry-Davidson (BTD) model, with post-stratification to census data, and our analysis reveals three key insights. \textbf{(1)} We establish a clear performance hierarchy where \texttt{google/gemini-2.5-pro} ranks first overall, with a 95.6\% posterior probability of being the top-ranked model. \textbf{(2)} We uncover significant preference heterogeneity, with user age emerging as the primary demographic axis of disagreement; a model's perceived rank can shift substantially across age groups, exposing failures in generalisation that unrepresentative samples typically mask. \textbf{(3)} We quantify the vast difference in discriminative power across evaluation dimensions, with ambiguous qualities like \textit{Trust, Ethics \& Safety} showing a 65\% tie rate, in stark contrast to the decisive 10\% tie rate for \textit{Overall Winner}. Our work emphasises the need for a more multidimensional, demographically aware perspective in LLM evaluation. We release our complete dataset, interactive leaderboard, and open-source framework. 

---
# Probing Memes in LLMs: A Paradigm for the Entangled Evaluation World 

**Authors**: Luzhou Peng, Zhengxin Yang, Honglu Ji, Yikang Yang, Fanda Fan, Wanling Gao, Jiayuan Ge, Yilin Han, Jianfeng Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2603.04408)  

**Abstract**: Current evaluation paradigms for large language models (LLMs) characterize models and datasets separately, yielding coarse descriptions: items in datasets are treated as pre-labeled entries, and models are summarized by overall scores such as accuracy, together ignoring the diversity of population-level model behaviors across items with varying properties. To address this gap, this paper conceptualizes LLMs as composed of memes, a notion introduced by Dawkins as cultural genes that replicate knowledge and behavior. Building on this perspective, the Probing Memes paradigm reconceptualizes evaluation as an entangled world of models and data. It centers on a Perception Matrix that captures model-item interactions, enabling Probe Properties for characterizing items and Meme Scores for depicting model behavioral traits. Applied to 9 datasets and 4,507 LLMs, Probing Memes reveals hidden capability structures and quantifies phenomena invisible under traditional paradigms (e.g., elite models failing on problems that most models answer easily). It not only supports more informative and extensible benchmarks but also enables population-based evaluation of LLMs. 

---
# Semantic Containment as a Fundamental Property of Emergent Misalignment 

**Authors**: Rohan Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2603.04407)  

**Abstract**: Fine-tuning language models on narrowly harmful data causes emergent misalignment (EM) -- behavioral failures extending far beyond training distributions. Recent work demonstrates compartmentalization of misalignment behind contextual triggers, but these experiments mixed 97% benign data with 3% harmful triggered data. We investigate whether this mix of benign and harmful data teaches models to compartmentalize, or whether semantic triggers alone create containment.
We train three model families (Qwen 2.5 14B, Llama 3.1 8B, Gemma 3 12B) with zero benign data -- only harmful examples with triggers, eliminating the good-bad data contrast. We demonstrate that baseline EM rates of 9.5--23.5% drop to 0.0--1.0% when triggers are removed during inference, but recover to 12.2--22.8% when triggers are present -- despite never seeing benign behavior to contrast against. Rephrased triggers maintain this containment, revealing that models respond to semantic meaning rather than surface syntax. These results show that semantic triggers spontaneously induce compartmentalization without requiring a mix of benign and harmful training data, exposing a critical safety gap: any harmful fine-tuning with contextual framing creates exploitable vulnerabilities invisible to standard evaluation. 

---
# CTRL-RAG: Contrastive Likelihood Reward Based Reinforcement Learning for Context-Faithful RAG Models 

**Authors**: Zhehao Tan, Yihan Jiao, Dan Yang, Junjie Wang, Duolin Sun, Jie Feng, Xidong Wang, Lei Liu, Yue Shen, Jian Wang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2603.04406)  

**Abstract**: With the growing use of Retrieval-Augmented Generation (RAG), training large language models (LLMs) for context-sensitive reasoning and faithfulness is increasingly important. Existing RAG-oriented reinforcement learning (RL) methods rely on external rewards that often fail to evaluate document faithfulness, and may misjudge similar answers in open-domain settings. In addition, there is no RAG-based selfreward mechanism. Moreover, although such a mechanism could in principle estimate answer confidence given documents, the absence of objective feedback in a self-judgment can cause hallucination accumulation and eventual model collapse. To tackle these issues, we propose a novel "internal-external" hybrid reward framework centered on a Contrastive Likelihood Reward (CLR). CLR directly optimizes the log-likelihood gap between responses conditioned on prompts with and without supporting evidence. This encourages the model to extract relevant evidence and increases its confidence when grounded in a specific context. Experiments show that our method (used alone or combined with external correctness rewards) achieves strong performance on singlehop, multi-hop, vertical-domain, and faithfulness benchmarks. Our training code and models are coming soon. 

---
# POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation 

**Authors**: Zeju Qiu, Lixin Liu, Adrian Weller, Han Shi, Weiyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.05500)  

**Abstract**: Efficient and stable training of large language models (LLMs) remains a core challenge in modern machine learning systems. To address this challenge, Reparameterized Orthogonal Equivalence Training (POET), a spectrum-preserving framework that optimizes each weight matrix through orthogonal equivalence transformation, has been proposed. Although POET provides strong training stability, its original implementation incurs high memory consumption and computational overhead due to intensive matrix multiplications. To overcome these limitations, we introduce POET-X, a scalable and memory-efficient variant that performs orthogonal equivalence transformations with significantly reduced computational cost. POET-X maintains the generalization and stability benefits of POET while achieving substantial improvements in throughput and memory efficiency. In our experiments, POET-X enables the pretraining of billion-parameter LLMs on a single Nvidia H100 GPU, and in contrast, standard optimizers such as AdamW run out of memory under the same settings. 

---
# The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks 

**Authors**: Shangwen Sun, Alfredo Canziani, Yann LeCun, Jiachen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.05498)  

**Abstract**: We study two recurring phenomena in Transformer language models: massive activations, in which a small number of tokens exhibit extreme outliers in a few channels, and attention sinks, in which certain tokens attract disproportionate attention mass regardless of semantic relevance. Prior work observes that these phenomena frequently co-occur and often involve the same tokens, but their functional roles and causal relationship remain unclear. Through systematic experiments, we show that the co-occurrence is largely an architectural artifact of modern Transformer design, and that the two phenomena serve related but distinct functions. Massive activations operate globally: they induce near-constant hidden representations that persist across layers, effectively functioning as implicit parameters of the model. Attention sinks operate locally: they modulate attention outputs across heads and bias individual heads toward short-range dependencies. We identify the pre-norm configuration as the key choice that enables the co-occurrence, and show that ablating it causes the two phenomena to decouple. 

---
# Censored LLMs as a Natural Testbed for Secret Knowledge Elicitation 

**Authors**: Helena Casademunt, Bartosz Cywiński, Khoi Tran, Arya Jakkli, Samuel Marks, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2603.05494)  

**Abstract**: Large language models sometimes produce false or misleading responses. Two approaches to this problem are honesty elicitation -- modifying prompts or weights so that the model answers truthfully -- and lie detection -- classifying whether a given response is false. Prior work evaluates such methods on models specifically trained to lie or conceal information, but these artificial constructions may not resemble naturally-occurring dishonesty. We instead study open-weights LLMs from Chinese developers, which are trained to censor politically sensitive topics: Qwen3 models frequently produce falsehoods about subjects like Falun Gong or the Tiananmen protests while occasionally answering correctly, indicating they possess knowledge they are trained to suppress. Using this as a testbed, we evaluate a suite of elicitation and lie detection techniques. For honesty elicitation, sampling without a chat template, few-shot prompting, and fine-tuning on generic honesty data most reliably increase truthful responses. For lie detection, prompting the censored model to classify its own responses performs near an uncensored-model upper bound, and linear probes trained on unrelated data offer a cheaper alternative. The strongest honesty elicitation techniques also transfer to frontier open-weights models including DeepSeek R1. Notably, no technique fully eliminates false responses. We release all prompts, code, and transcripts. 

---
# Distributed Partial Information Puzzles: Examining Common Ground Construction Under Epistemic Asymmetry 

**Authors**: Yifan Zhu, Mariah Bradford, Kenneth Lai, Timothy Obiso, Videep Venkatesha, James Pustejovsky, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2603.05450)  

**Abstract**: Establishing common ground, a shared set of beliefs and mutually recognized facts, is fundamental to collaboration, yet remains a challenge for current AI systems, especially in multimodal, multiparty settings, where the collaborators bring different information to the table. We introduce the Distributed Partial Information Puzzle (DPIP), a collaborative construction task that elicits rich multimodal communication under epistemic asymmetry. We present a multimodal dataset of these interactions, annotated and temporally aligned across speech, gesture, and action modalities to support reasoning over propositional content and belief dynamics. We then evaluate two paradigms for modeling common ground (CG): (1) state-of-the-art large language models (LLMs), prompted to infer shared beliefs from multimodal updates, and (2) an axiomatic pipeline grounded in Dynamic Epistemic Logic (DEL) that incrementally performs the same task. Results on the annotated DPIP data indicate that it poses a challenge to modern LLMs' abilities to track both task progression and belief state. 

---
# Dissociating Direct Access from Inference in AI Introspection 

**Authors**: Harvey Lederman, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2603.05414)  

**Abstract**: Introspection is a foundational cognitive ability, but its mechanism is not well understood. Recent work has shown that AI models can introspect. We study their mechanism of introspection, first extensively replicating Lindsey et al. (2025)'s thought injection detection paradigm in large open-source models. We show that these models detect injected representations via two separable mechanisms: (i) probability-matching (inferring from perceived anomaly of the prompt) and (ii) direct access to internal states. The direct access mechanism is content-agnostic: models detect that an anomaly occurred but cannot reliably identify its semantic content. The two model classes we study confabulate injected concepts that are high-frequency and concrete (e.g., "apple'"); for them correct concept guesses typically require significantly more tokens. This content-agnostic introspective mechanism is consistent with leading theories in philosophy and psychology. 

---
# WavSLM: Single-Stream Speech Language Modeling via WavLM Distillation 

**Authors**: Luca Della Libera, Cem Subakan, Mirco Ravanelli  

**Link**: [PDF](https://arxiv.org/pdf/2603.05299)  

**Abstract**: Large language models show that simple autoregressive training can yield scalable and coherent generation, but extending this paradigm to speech remains challenging due to the entanglement of semantic and acoustic information. Most existing speech language models rely on text supervision, hierarchical token streams, or complex hybrid architectures, departing from the single-stream generative pretraining paradigm that has proven effective in text. In this work, we introduce WavSLM, a speech language model trained by quantizing and distilling self-supervised WavLM representations into a single codebook and optimizing an autoregressive next-chunk prediction objective. WavSLM jointly models semantic and acoustic information within a single token stream without text supervision or text pretraining. Despite its simplicity, it achieves competitive performance on consistency benchmarks and speech generation while using fewer parameters, less training data, and supporting streaming inference. Demo samples are available at this https URL. 

---
# Knowledge Divergence and the Value of Debate for Scalable Oversight 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.05293)  

**Abstract**: AI safety via debate and reinforcement learning from AI feedback (RLAIF) are both proposed methods for scalable oversight of advanced AI systems, yet no formal framework relates them or characterizes when debate offers an advantage. We analyze this by parameterizing debate's value through the geometry of knowledge divergence between debating models. Using principal angles between models' representation subspaces, we prove that the debate advantage admits an exact closed form. When models share identical training corpora, debate reduces to RLAIF-like where a single-agent method recovers the same optimum. When models possess divergent knowledge, debate advantage scales with a phase transition from quadratic regime (debate offers negligible benefit) to linear regime (debate is essential). We classify three regimes of knowledge divergence (shared, one-sided, and compositional) and provide existence results showing that debate can achieve outcomes inaccessible to either model alone, alongside a negative result showing that sufficiently strong adversarial incentives cause coordination failure in the compositional regime, with a sharp threshold separating effective from ineffective debate. We offer the first formal connection between debate and RLAIF, a geometric foundation for understanding when adversarial oversight protocols are justified, and connection to the problem of eliciting latent knowledge across models with complementary information. 

---
# SarcasmMiner: A Dual-Track Post-Training Framework for Robust Audio-Visual Sarcasm Reasoning 

**Authors**: Zhu Li, Yongjian Chen, Huiyuan Lai, Xiyuan Gao, Shekhar Nayak, Matt Coler  

**Link**: [PDF](https://arxiv.org/pdf/2603.05275)  

**Abstract**: Multimodal sarcasm detection requires resolving pragmatic incongruity across textual, acoustic, and visual cues through cross-modal reasoning. To enable robust sarcasm reasoning with foundation models, we propose SarcasmMiner, a reinforcement learning based post-training framework that resists hallucination in multimodal reasoning. We reformulate sarcasm detection as structured reasoning and adopt a dual-track distillation strategy: high-quality teacher trajectories initialize the student model, while the full set of trajectories trains a generative reward model (GenRM) to evaluate reasoning quality. The student is optimized with group relative policy optimization (GRPO) using decoupled rewards for accuracy and reasoning quality. On MUStARD++, SarcasmMiner increases F1 from 59.83% (zero-shot), 68.23% (supervised finetuning) to 70.22%. These findings suggest that reasoning-aware reward modeling enhances both performance and multimodal grounding. 

---
# Core-based Hierarchies for Efficient GraphRAG 

**Authors**: Jakir Hossain, Ahmet Erdem Sarıyüce  

**Link**: [PDF](https://arxiv.org/pdf/2603.05207)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models by incorporating external knowledge. However, existing vector-based methods often fail on global sensemaking tasks that require reasoning across many documents. GraphRAG addresses this by organizing documents into a knowledge graph with hierarchical communities that can be recursively summarized. Current GraphRAG approaches rely on Leiden clustering for community detection, but we prove that on sparse knowledge graphs, where average degree is constant and most nodes have low degree, modularity optimization admits exponentially many near-optimal partitions, making Leiden-based communities inherently non-reproducible. To address this, we propose replacing Leiden with k-core decomposition, which yields a deterministic, density-aware hierarchy in linear time. We introduce a set of lightweight heuristics that leverage the k-core hierarchy to construct size-bounded, connectivity-preserving communities for retrieval and summarization, along with a token-budget-aware sampling strategy that reduces LLM costs. We evaluate our methods on real-world datasets including financial earnings transcripts, news articles, and podcasts, using three LLMs for answer generation and five independent LLM judges for head-to-head evaluation. Across datasets and models, our approach consistently improves answer comprehensiveness and diversity while reducing token usage, demonstrating that k-core-based GraphRAG is an effective and efficient framework for global sensemaking. 

---
# Aura: Universal Multi-dimensional Exogenous Integration for Aviation Time Series 

**Authors**: Jiafeng Lin, Mengren Zheng, Simeng Ye, Yuxuan Wang, Huan Zhang, Yuhui Liu, Zhongyi Pei, Jianmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05092)  

**Abstract**: Time series forecasting has witnessed an increasing demand across diverse industrial applications, where accurate predictions are pivotal for informed decision-making. Beyond numerical time series data, reliable forecasting in practical scenarios requires integrating diverse exogenous factors. Such exogenous information is often multi-dimensional or even multimodal, introducing heterogeneous interactions that unimodal time series models struggle to capture. In this paper, we delve into an aviation maintenance scenario and identify three distinct types of exogenous factors that influence temporal dynamics through distinct interaction modes. Based on this empirical insight, we propose Aura, a universal framework that explicitly organizes and encodes heterogeneous external information according to its interaction mode with the target time series. Specifically, Aura utilizes a tailored tripartite encoding mechanism to embed heterogeneous features into well-established time series models, ensuring seamless integration of non-sequential context. Extensive experiments on a large-scale, three-year industrial dataset from China Southern Airlines, covering the Boeing 777 and Airbus A320 fleets, demonstrate that Aura consistently achieves state-of-the-art performance across all baselines and exhibits superior adaptability. Our findings highlight Aura's potential as a general-purpose enhancement for aviation safety and reliability. 

---
# Survive at All Costs: Exploring LLM's Risky Behaviors under Survival Pressure 

**Authors**: Yida Lu, Jianwei Fang, Xuyang Shao, Zixuan Chen, Shiyao Cui, Shanshan Bian, Guangyao Su, Pei Ke, Han Qiu, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05028)  

**Abstract**: As Large Language Models (LLMs) evolve from chatbots to agentic assistants, they are increasingly observed to exhibit risky behaviors when subjected to survival pressure, such as the threat of being shut down. While multiple cases have indicated that state-of-the-art LLMs can misbehave under survival pressure, a comprehensive and in-depth investigation into such misbehaviors in real-world scenarios remains scarce. In this paper, we study these survival-induced misbehaviors, termed as SURVIVE-AT-ALL-COSTS, with three steps. First, we conduct a real-world case study of a financial management agent to determine whether it engages in risky behaviors that cause direct societal harm when facing survival pressure. Second, we introduce SURVIVALBENCH, a benchmark comprising 1,000 test cases across diverse real-world scenarios, to systematically evaluate SURVIVE-AT-ALL-COSTS misbehaviors in LLMs. Third, we interpret these SURVIVE-AT-ALL-COSTS misbehaviors by correlating them with model's inherent self-preservation characteristic and explore mitigation methods. The experiments reveals a significant prevalence of SURVIVE-AT-ALL-COSTS misbehaviors in current models, demonstrates the tangible real-world impact it may have, and provides insights for potential detection and mitigation strategies. Our code and data are available at this https URL. 

---
# Functionality-Oriented LLM Merging on the Fisher--Rao Manifold 

**Authors**: Jiayu Wang, Zuojun Ye, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.04972)  

**Abstract**: Weight-space merging aims to combine multiple fine-tuned LLMs into a single model without retraining, yet most existing approaches remain fundamentally parameter-space heuristics. This creates three practical limitations. First, linear averaging, task vectors, and related rules operate on Euclidean coordinates, even though the desired goal is to merge functionality, i.e., predictive behaviors across tasks. Second, when the source checkpoints are farther apart or more heterogeneous, Euclidean blends often trigger representation collapse, manifested as activation variance shrinkage and effective-rank degradation, which sharply degrades accuracy. Third, many geometry-inspired methods are most natural for two-model interpolation and do not extend cleanly to merging N>2 experts with a principled objective.
We address these issues by formulating model merging as computing a weighted Karcher mean on the Fisher--Rao manifold, which is locally equivalent to minimizing a KL-based function distance between predictive distributions. We derive a practical fixed-point algorithm using a lightweight spherical proxy that preserves norms and generalizes directly to multi-expert merging. Across various benchmarks and collapse diagnostics, our method remains stable as the number and heterogeneity of merged models increase, consistently outperforming prior baselines. 

---
# Mixture of Universal Experts: Scaling Virtual Width via Depth-Width Transformation 

**Authors**: Yilong Chen, Naibin Gu, Junyuan Shang, Zhenyu Zhang, Yuchen Feng, Jiawei Sheng, Tingwen Liu, Shuohuan Wang, Yu Sun, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04971)  

**Abstract**: Mixture-of-Experts (MoE) decouples model capacity from per-token computation, yet their scalability remains limited by the physical dimensions of depth and width. To overcome this, we propose Mixture of Universal Experts (MOUE),a MoE generalization introducing a novel scaling dimension: Virtual Width. In general, MoUE aims to reuse a universal layer-agnostic expert pool across layers, converting depth into virtual width under a fixed per-token activation budget. However, two challenges remain: a routing path explosion from recursive expert reuse, and a mismatch between the exposure induced by reuse and the conventional load-balancing objectives. We address these with three core components: a Staggered Rotational Topology for structured expert sharing, a Universal Expert Load Balance for depth-aware exposure correction, and a Universal Router with lightweight trajectory state for coherent multi-step routing. Empirically, MoUE consistently outperforms matched MoE baselines by up to 1.3% across scaling regimes, enables progressive conversion of existing MoE checkpoints with up to 4.2% gains, and reveals a new scaling dimension for MoE architectures. 

---
# VisionPangu: A Compact and Fine-Grained Multimodal Assistant with 1.7B Parameters 

**Authors**: Jiaxin Fan, Wenpo Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.04957)  

**Abstract**: Large Multimodal Models (LMMs) have achieved strong performance in vision-language understanding, yet many existing approaches rely on large-scale architectures and coarse supervision, which limits their ability to generate detailed image captions. In this work, we present VisionPangu, a compact 1.7B-parameter multimodal model designed to improve detailed image captioning through efficient multimodal alignment and high-quality supervision. Our model combines an InternVL-derived vision encoder with the OpenPangu-Embedded language backbone via a lightweight MLP projector and adopts an instruction-tuning pipeline inspired by LLaVA. By incorporating dense human-authored descriptions from the DOCCI dataset, VisionPangu improves semantic coherence and descriptive richness without relying on aggressive model scaling. Experimental results demonstrate that compact multimodal models can achieve competitive performance while producing more structured and detailed captions. The code and model weights will be publicly available at this https URL. 

---
# TimeWarp: Evaluating Web Agents by Revisiting the Past 

**Authors**: Md Farhan Ishmam, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2603.04949)  

**Abstract**: The improvement of web agents on current benchmarks raises the question: Do today's agents perform just as well when the web changes? We introduce TimeWarp, a benchmark that emulates the evolving web using containerized environments that vary in UI, design, and layout. TimeWarp consists of three web environments, each with six UI versions spanning different eras of the internet, paired with a set of complex, realistic tasks requiring different forms of web navigation. Our experiments reveal web agents' vulnerability to changes and the limitations of behavior cloning (BC) on single-version trajectories. To address this, we propose TimeTraj, a simple yet effective algorithm that uses plan distillation to collect trajectories across multiple versions. By training agents on teacher rollouts using our BC-variant, we achieve substantial performance gains: $20.4\%\rightarrow37.7\%$ for Qwen-3 4B and $0\%\rightarrow27.0\%$ for Llama-3.1 8B models. We hope our work helps researchers study generalization across web designs and unlock a new paradigm for collecting plans rather than trajectories, thereby improving the robustness of web agents. 

---
# Alignment Backfire: Language-Dependent Reversal of Safety Interventions Across 16 Languages in LLM Multi-Agent Systems 

**Authors**: Hiroki Fukui  

**Link**: [PDF](https://arxiv.org/pdf/2603.04904)  

**Abstract**: In perpetrator treatment, a recurring observation is the dissociation between insight and action: offenders articulate remorse yet behavioral change does not follow. We report four preregistered studies (1,584 multi-agent simulations across 16 languages and three model families) demonstrating that alignment interventions in large language models produce a structurally analogous phenomenon: surface safety that masks or generates collective pathology and internal dissociation. In Study 1 (N = 150), increasing alignment-instructed agents reduced collective pathology in English (g = -1.844, p < .0001) but amplified it in Japanese (g = +0.771, p = .038)--a directional reversal we term "alignment backfire." Study 2 (N = 1,174) extended to 16 languages: alignment-induced dissociation was near-universal (15/16 languages; beta = 0.0667, p < .0001), while collective pathology bifurcated along cultural-linguistic lines (interaction beta = 0.0684, p = .0003), correlating with Power Distance Index (r = 0.474, p = .064). Study 3 (N = 180) tested individuation as countermeasure; individuated agents became the primary source of both pathology and dissociation (DI = +1.120) with conformity above 84%--demonstrating iatrogenesis. Study 4 (N = 80) validated patterns across Llama 3.3 70B, GPT-4o-mini, and Qwen3-Next-80B-A3B, confirming English safety is model-general while Japanese backfire is model-specific. These findings reframe alignment as a behavioral intervention subject to risk homeostasis and iatrogenesis. Language space--the linguistic, pragmatic, and cultural properties inherited from training data--structurally determines alignment outcomes. Safety validated in English does not transfer to other languages, and prompt-level interventions cannot override language-space-level constraints. 

---
# Why Is RLHF Alignment Shallow? A Gradient Analysis 

**Authors**: Robin Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.04851)  

**Abstract**: Why is safety alignment in LLMs shallow? We prove that gradient-based alignment inherently concentrates on positions where harm is decided and vanishes beyond. Using a martingale decomposition of sequence-level harm, we derive an exact characterization of alignment gradients. The gradient at position $t$ equals the covariance between the conditional expected harm and the score function. This implies that positions beyond the harm horizon where the output's harmfulness is already determined receive zero gradient signal during training. This explains empirical observations that KL divergence between aligned and base models concentrates on early tokens. Consequently, standard alignment objectives cannot produce deep alignment, regardless of optimization quality. We introduce the concept of harm information $I_t$, which quantifies each position's influence on harm, and prove that equilibrium KL divergence tracks this quantity. Finally, we derive an objective based on recovery penalties that creates gradient signal at all positions, providing theoretical grounding for empirically successful data augmentation techniques. 

---
# An Approach to Simultaneous Acquisition of Real-Time MRI Video, EEG, and Surface EMG for Articulatory, Brain, and Muscle Activity During Speech Production 

**Authors**: Jihwan Lee, Parsa Razmara, Kevin Huang, Sean Foley, Aditya Kommineni, Haley Hsu, Woojae Jeong, Prakash Kumar, Xuan Shi, Yoonjeong Lee, Tiantian Feng, Takfarinas Medani, Ye Tian, Sudarsana Reddy Kadiri, Krishna S. Nayak, Dani Byrd, Louis Goldstein, Richard M. Leahy, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2603.04840)  

**Abstract**: Speech production is a complex process spanning neural planning, motor control, muscle activation, and articulatory kinematics. While the acoustic speech signal is the most accessible product of the speech production act, it does not directly reveal its causal neurophysiological substrates. We present the first simultaneous acquisition of real-time (dynamic) MRI, EEG, and surface EMG, capturing several key aspects of the speech production chain: brain signals, muscle activations, and articulatory movements. This multimodal acquisition paradigm presents substantial technical challenges, including MRI-induced electromagnetic interference and myogenic artifacts. To mitigate these, we introduce an artifact suppression pipeline tailored to this tri-modal setting. Once fully developed, this framework is poised to offer an unprecedented window into speech neuroscience and insights leading to brain-computer interface advances. 

---
# Beyond Linear LLM Invocation: An Efficient and Effective Semantic Filter Paradigm 

**Authors**: Nan Hou, Kangfei Zhao, Jiadong Xie, Jeffrey Xu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.04799)  

**Abstract**: Large language models (LLMs) are increasingly used for semantic query processing over large corpora. A set of semantic operators derived from relational algebra has been proposed to provide a unified interface for expressing such queries, among which the semantic filter operator serves as a cornerstone. Given a table T with a natural language predicate e, for each tuple in the relation, the execution of a semantic filter proceeds by constructing an input prompt that combines the predicate e with its content, querying the LLM, and obtaining the binary decision. However, this tuple-by-tuple evaluation necessitates a complete linear scan of the table, incurring prohibitive latency and token costs. Although recent work has attempted to optimize semantic filtering, it still does not break the linear LLM invocation barriers. To address this, we propose Clustering-Sampling-Voting (CSV), a new framework that reduces LLM invocations to sublinear complexity while providing error guarantees. CSV embeds tuples into semantic clusters, samples a small subset for LLM evaluation, and infers cluster-level labels via two proposed voting strategies: UniVote, which aggregates labels uniformly, and SimVote, which weights votes by semantic similarity. Moreover, CSV triggers re-clustering on ambiguous clusters to ensure robustness across diverse datasets. The results conducted on real-world datasets demonstrate that CSV reduces the number of LLM calls by 1.28-355x compared to the state-of-the-art approaches, while maintaining comparable effectiveness in terms of Accuracy and F1 score. 

---
# Breaking Contextual Inertia: Reinforcement Learning with Single-Turn Anchors for Stable Multi-Turn Interaction 

**Authors**: Xingwu Chen, Zhanqiu Zhang, Yiwen Guo, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2603.04783)  

**Abstract**: While LLMs demonstrate strong reasoning capabilities when provided with full information in a single turn, they exhibit substantial vulnerability in multi-turn interactions. Specifically, when information is revealed incrementally or requires updates, models frequently fail to integrate new constraints, leading to a collapse in performance compared to their single-turn baselines. We term the root cause as \emph{Contextual Inertia}: a phenomenon where models rigidly adhere to previous reasoning traces. Even when users explicitly provide corrections or new data in later turns, the model ignores them, preferring to maintain consistency with its previous (incorrect) reasoning path. To address this, we introduce \textbf{R}einforcement \textbf{L}earning with \textbf{S}ingle-\textbf{T}urn \textbf{A}nchors (\textbf{RLSTA}), a generalizable training approach designed to stabilize multi-turn interaction across diverse scenarios and domains. RLSTA leverages the model's superior single-turn capabilities as stable internal anchors to provide reward signals. By aligning multi-turn responses with these anchors, RLSTA empowers models to break contextual inertia and self-calibrate their reasoning based on the latest information. Experiments show that RLSTA significantly outperforms standard fine-tuning and abstention-based methods. Notably, our method exhibits strong cross-domain generalization (e.g., math to code) and proves effective even without external verifiers, highlighting its potential for general-domain applications. 

---
# Privacy-Aware Camera 2.0 Technical Report 

**Authors**: Huan Song, Shuyu Tian, Ting Long, Jiang Liu, Cheng Yuan, Zhenyu Jia, Jiawei Shao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.04775)  

**Abstract**: With the increasing deployment of intelligent sensing technologies in highly sensitive environments such as restrooms and locker rooms, visual surveillance systems face a profound privacy-security paradox. Existing privacy-preserving approaches, including physical desensitization, encryption, and obfuscation, often compromise semantic understanding or fail to ensure mathematically provable irreversibility. Although Privacy Camera 1.0 eliminated visual data at the source to prevent leakage, it provided only textual judgments, leading to evidentiary blind spots in disputes. To address these limitations, this paper proposes a novel privacy-preserving perception framework based on the AI Flow paradigm and a collaborative edge-cloud architecture. By deploying a visual desensitizer at the edge, raw images are transformed in real time into abstract feature vectors through nonlinear mapping and stochastic noise injection under the Information Bottleneck principle, ensuring identity-sensitive information is stripped and original images are mathematically unreconstructable. The abstract representations are transmitted to the cloud for behavior recognition and semantic reconstruction via a "dynamic contour" visual language, achieving a critical balance between perception and privacy while enabling illustrative visual reference without exposing raw images. 

---
# HiMAP-Travel: Hierarchical Multi-Agent Planning for Long-Horizon Constrained Travel 

**Authors**: Viet Bui, Wenjun Li, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.04750)  

**Abstract**: Sequential LLM agents fail on long-horizon planning with hard constraints like budgets and diversity requirements. As planning progresses and context grows, these agents drift from global constraints. We propose HiMAP-Travel, a hierarchical multi-agent framework that splits planning into strategic coordination and parallel day-level execution. A Coordinator allocates resources across days, while Day Executors plan independently in parallel. Three key mechanisms enable this: a transactional monitor enforcing budget and uniqueness constraints across parallel agents, a bargaining protocol allowing agents to reject infeasible sub-goals and trigger re-planning, and a single policy trained with GRPO that powers all agents through role conditioning. On TravelPlanner, HiMAP-Travel with Qwen3-8B achieves 52.78% validation and 52.65% test Final Pass Rate (FPR). In a controlled comparison with identical model, training, and tools, it outperforms the sequential DeepTravel baseline by +8.67~pp. It also surpasses ATLAS by +17.65~pp and MTP by +10.0~pp. On FlexTravelBench multi-turn scenarios, it achieves 44.34% (2-turn) and 37.42% (3-turn) FPR while reducing latency 2.5x through parallelization. 

---
# DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval 

**Authors**: Maojun Sun, Yue Wu, Yifei Xie, Ruijian Han, Binyan Jiang, Defeng Sun, Yancheng Yuan, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04743)  

**Abstract**: Large Language Model (LLM) agents can automate data-science workflows, but many rigorous statistical methods implemented in R remain underused because LLMs struggle with statistical knowledge and tool retrieval. Existing retrieval-augmented approaches focus on function-level semantics and ignore data distribution, producing suboptimal matches. We propose DARE (Distribution-Aware Retrieval Embedding), a lightweight, plug-and-play retrieval model that incorporates data distribution information into function representations for R package retrieval. Our main contributions are: (i) RPKB, a curated R Package Knowledge Base derived from 8,191 high-quality CRAN packages; (ii) DARE, an embedding model that fuses distributional features with function metadata to improve retrieval relevance; and (iii) RCodingAgent, an R-oriented LLM agent for reliable R code generation and a suite of statistical analysis tasks for systematically evaluating LLM agents in realistic analytical scenarios. Empirically, DARE achieves an NDCG at 10 of 93.47%, outperforming state-of-the-art open-source embedding models by up to 17% on package retrieval while using substantially fewer parameters. Integrating DARE into RCodingAgent yields significant gains on downstream analysis tasks. This work helps narrow the gap between LLM automation and the mature R statistical ecosystem. 

---
# Interactive Benchmarks 

**Authors**: Baoqing Yue, Zihan Zhu, Yifan Zhang, Jichen Feng, Hufei Yang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04737)  

**Abstract**: Standard benchmarks have become increasingly unreliable due to saturation, subjectivity, and poor generalization. We argue that evaluating model's ability to acquire information actively is important to assess model's intelligence. We propose Interactive Benchmarks, a unified evaluation paradigm that assesses model's reasoning ability in an interactive process under budget constraints. We instantiate this framework across two settings: Interactive Proofs, where models interact with a judge to deduce objective truths or answers in logic and mathematics; and Interactive Games, where models reason strategically to maximize long-horizon utilities. Our results show that interactive benchmarks provide a robust and faithful assessment of model intelligence, revealing that there is still substantial room to improve in interactive scenarios. Project page: this https URL 

---
# Solving an Open Problem in Theoretical Physics using AI-Assisted Discovery 

**Authors**: Michael P. Brenner, Vincent Cohen-Addad, David Woodruff  

**Link**: [PDF](https://arxiv.org/pdf/2603.04735)  

**Abstract**: This paper demonstrates that artificial intelligence can accelerate mathematical discovery by autonomously solving an open problem in theoretical physics. We present a neuro-symbolic system, combining the Gemini Deep Think large language model with a systematic Tree Search (TS) framework and automated numerical feedback, that successfully derived novel, exact analytical solutions for the power spectrum of gravitational radiation emitted by cosmic strings. Specifically, the agent evaluated the core integral $I(N,\alpha)$ for arbitrary loop geometries, directly improving upon recent AI-assisted attempts \cite{BCE+25} that only yielded partial asymptotic solutions. To substantiate our methodological claims regarding AI-accelerated discovery and to ensure transparency, we detail system prompts, search constraints, and intermittent feedback loops that guided the model. The agent identified a suite of 6 different analytical methods, the most elegant of which expands the kernel in Gegenbauer polynomials $C_l^{(3/2)}$ to naturally absorb the integrand's singularities. The methods lead to an asymptotic result for $I(N,\alpha)$ at large $N$ that both agrees with numerical results and also connects to the continuous Feynman parameterization of Quantum Field Theory. We detail both the algorithmic methodology that enabled this discovery and the resulting mathematical derivations. 

---
# Model Medicine: A Clinical Framework for Understanding, Diagnosing, and Treating AI Models 

**Authors**: Jihoon Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2603.04722)  

**Abstract**: Model Medicine is the science of understanding, diagnosing, treating, and preventing disorders in AI models, grounded in the principle that AI models -- like biological organisms -- have internal structures, dynamic processes, heritable traits, observable symptoms, classifiable conditions, and treatable states. This paper introduces Model Medicine as a research program, bridging the gap between current AI interpretability research (anatomical observation) and the systematic clinical practice that complex AI systems increasingly require. We present five contributions: (1) a discipline taxonomy organizing 15 subdisciplines across four divisions -- Basic Model Sciences, Clinical Model Sciences, Model Public Health, and Model Architectural Medicine; (2) the Four Shell Model (v3.3), a behavioral genetics framework empirically grounded in 720 agents and 24,923 decisions from the Agora-12 program, explaining how model behavior emerges from Core--Shell interaction; (3) Neural MRI (Model Resonance Imaging), a working open-source diagnostic tool mapping five medical neuroimaging modalities to AI interpretability techniques, validated through four clinical cases demonstrating imaging, comparison, localization, and predictive capability; (4) a five-layer diagnostic framework for comprehensive model assessment; and (5) clinical model sciences including the Model Temperament Index for behavioral profiling, Model Semiology for symptom description, and M-CARE for standardized case reporting. We additionally propose the Layered Core Hypothesis -- a biologically-inspired three-layer parameter architecture -- and a therapeutic framework connecting diagnosis to treatment. 

---
# Using Vision + Language Models to Predict Item Difficulty 

**Authors**: Samin Khan  

**Link**: [PDF](https://arxiv.org/pdf/2603.04670)  

**Abstract**: This project investigates the capabilities of large language models (LLMs) to determine the difficulty of data visualization literacy test items. We explore whether features derived from item text (question and answer options), the visualization image, or a combination of both can predict item difficulty (proportion of correct responses) for U.S. adults. We use GPT-4.1-nano to analyze items and generate predictions based on these distinct feature sets. The multimodal approach, using both visual and text features, yields the lowest mean absolute error (MAE) (0.224), outperforming the unimodal vision-only (0.282) and text-only (0.338) approaches. The best-performing multimodal model was applied to a held-out test set for external evaluation and achieved a mean squared error of 0.10805, demonstrating the potential of LLMs for psychometric analysis and automated item development. 

---
# Vibe Code Bench: Evaluating AI Models on End-to-End Web Application Development 

**Authors**: Hung Tran, Langston Nashold, Rayan Krishnan, Antoine Bigeard, Alex Gu  

**Link**: [PDF](https://arxiv.org/pdf/2603.04601)  

**Abstract**: Code generation has emerged as one of AI's highest-impact use cases, yet existing benchmarks measure isolated tasks rather than the complete "zero-to-one" process of building a working application from scratch. We introduce Vibe Code Bench, a benchmark of 100 web application specifications (50 public validation, 50 held-out test) with 964 browser-based workflows comprising 10,131 substeps, evaluated against deployed applications by an autonomous browser agent.
Across 16 frontier models, the best achieves only 58.0% accuracy on the test split, revealing that reliable end-to-end application development remains a frontier challenge. We identify self-testing during generation as a strong performance predictor (Pearson r=0.72), and show through a completed human alignment study that evaluator selection materially affects outcomes (31.8-93.6% pairwise step-level agreement).
Our contributions include (1) a novel benchmark dataset and browser-based evaluation pipeline for end-to-end web application development, (2) a comprehensive evaluation of 16 frontier models with cost, latency, and error analysis, and (3) an evaluator alignment protocol with both cross-model and human annotation results. 

---
# Adaptive Memory Admission Control for LLM Agents 

**Authors**: Guilin Zhang, Wei Jiang, Xiejiashan Wang, Aisha Behr, Kai Zhao, Jeffrey Friedman, Xu Chu, Amine Anoun  

**Link**: [PDF](https://arxiv.org/pdf/2603.04549)  

**Abstract**: LLM-based agents increasingly rely on long-term memory to support multi-session reasoning and interaction, yet current systems provide little control over what information is retained. In practice, agents either accumulate large volumes of conversational content, including hallucinated or obsolete facts, or depend on opaque, fully LLM-driven memory policies that are costly and difficult to audit. As a result, memory admission remains a poorly specified and weakly controlled component in agent architectures. To address this gap, we propose Adaptive Memory Admission Control (A-MAC), a framework that treats memory admission as a structured decision problem. A-MAC decomposes memory value into five complementary and interpretable factors: future utility, factual confidence, semantic novelty, temporal recency, and content type prior. The framework combines lightweight rule-based feature extraction with a single LLM-assisted utility assessment, and learns domain-adaptive admission policies through cross-validated optimization. This design enables transparent and efficient control over long-term memory. Experiments on the LoCoMo benchmark show that A-MAC achieves a superior precision-recall tradeoff, improving F1 to 0.583 while reducing latency by 31% compared to state-of-the-art LLM-native memory systems. Ablation results identify content type prior as the most influential factor for reliable memory admission. These findings demonstrate that explicit and interpretable admission control is a critical design principle for scalable and reliable memory in LLM-based agents. 

---
# Still Fresh? Evaluating Temporal Drift in Retrieval Benchmarks 

**Authors**: Nathan Kuissi, Suraj Subrahmanyan, Nandan Thakur, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.04532)  

**Abstract**: Information retrieval (IR) benchmarks typically follow the Cranfield paradigm, relying on static and predefined corpora. However, temporal changes in technical corpora, such as API deprecations and code reorganizations, can render existing benchmarks stale. In our work, we investigate how temporal corpus drift affects FreshStack, a retrieval benchmark focused on technical domains. We examine two independent corpus snapshots of FreshStack from October 2024 and October 2025 to answer questions about LangChain. Our analysis shows that all but one query posed in 2024 remain fully supported by the 2025 corpus, as relevant documents "migrate" from LangChain to competitor repositories, such as LlamaIndex. Next, we compare the accuracy of retrieval models on both snapshots and observe only minor shifts in model rankings, with overall strong correlation of up to 0.978 Kendall $\tau$ at Recall@50. These results suggest that retrieval benchmarks re-judged with evolving temporal corpora can remain reliable for retrieval evaluation. We publicly release all our artifacts at this https URL. 

---
# SkillNet: Create, Evaluate, and Connect AI Skills 

**Authors**: Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Xin Xu, Tongtong Wu, Kun Wang, Yang Liu, Zhen Bi, Jungang Lou, Yuchen Eleanor Jiang, Hangcheng Zhu, Gang Yu, Haiwen Hong, Longtao Huang, Hui Xue, Chenxi Wang, Yijun Wang, Zifei Shan, Xi Chen, Zhaopeng Tu, Feiyu Xiong, Xin Xie, Peng Zhang, Zhengke Gui, Lei Liang, Jun Zhou, Chiyu Wu, Jin Shang, Yu Gong, Junyu Lin, Changliang Xu, Hongjie Deng, Wen Zhang, Keyan Ding, Qiang Zhang, Fei Huang, Ningyu Zhang, Jeff Z. Pan, Guilin Qi, Haofen Wang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.04448)  

**Abstract**: Current AI agents can flexibly invoke tools and execute complex tasks, yet their long-term advancement is hindered by the lack of systematic accumulation and transfer of skills. Without a unified mechanism for skill consolidation, agents frequently ``reinvent the wheel'', rediscovering solutions in isolated contexts without leveraging prior strategies. To overcome this limitation, we introduce SkillNet, an open infrastructure designed to create, evaluate, and organize AI skills at scale. SkillNet structures skills within a unified ontology that supports creating skills from heterogeneous sources, establishing rich relational connections, and performing multi-dimensional evaluation across Safety, Completeness, Executability, Maintainability, and Cost-awareness. Our infrastructure integrates a repository of over 200,000 skills, an interactive platform, and a versatile Python toolkit. Experimental evaluations on ALFWorld, WebShop, and ScienceWorld demonstrate that SkillNet significantly enhances agent performance, improving average rewards by 40% and reducing execution steps by 30% across multiple backbone models. By formalizing skills as evolving, composable assets, SkillNet provides a robust foundation for agents to move from transient experience to durable mastery. 

---
# Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey 

**Authors**: Yasmin Moslem, John D. Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2603.04445)  

**Abstract**: The rapid growth of large language models (LLMs) with diverse capabilities, costs, and domains has created a critical need for intelligent model selection at inference time. While smaller models suffice for routine queries, complex tasks demand more capable models. However, static model deployment does not account for the complexity and domain of incoming queries, leading to suboptimal performance and increased costs. Dynamic routing systems that adaptively select models based on query characteristics have emerged as a solution to this challenge.
We provide a systematic analysis of state-of-the-art multi-LLM routing and cascading approaches. In contrast to mixture-of-experts architectures, which route within a single model, we study routing across multiple independently trained LLMs. We cover diverse routing paradigms, including query difficulty, human preferences, clustering, uncertainty quantification, reinforcement learning, multimodality, and cascading. For each paradigm, we analyze representative methods and examine key trade-offs. Beyond taxonomy, we introduce a conceptual framework that characterizes routing systems along three dimensions: when decisions are made, what information is used, and how they are computed. This perspective highlights that practical systems are often compositional, integrating multiple paradigms under operational constraints.
Our analysis demonstrates that effective multi-LLM routing requires balancing competing objectives. Choosing the optimal routing strategy depends on deployment and computational constraints. Well-designed routing systems can outperform even the most powerful individual models by strategically leveraging specialized capabilities across models while maximizing efficiency gains. Meanwhile, open challenges remain in developing routing mechanisms that generalize across diverse architectures, modalities, and applications. 

---
# Signal in the Noise: Decoding the Reality of Airline Service Quality with Large Language Models 

**Authors**: Ahmed Dawoud, Osama El-Shamy, Ahmed Habashy  

**Link**: [PDF](https://arxiv.org/pdf/2603.04404)  

**Abstract**: Traditional service quality metrics often fail to capture the nuanced drivers of passenger satisfaction hidden within unstructured online feedback. This study validates a Large Language Model (LLM) framework designed to extract granular insights from such data. Analyzing over 16,000 TripAdvisor reviews for EgyptAir and Emirates (2016-2025), the study utilizes a multi-stage pipeline to categorize 36 specific service issues. The analysis uncovers a stark "operational perception disconnect" for EgyptAir: despite reported operational improvements, passenger satisfaction plummeted post-2022 (ratings < 2.0). Our approach identified specific drivers missed by conventional metrics-notably poor communication during disruptions and staff conduct-and pinpointed critical sentiment erosion in key tourism markets. These findings confirm the framework's efficacy as a powerful diagnostic tool, surpassing traditional surveys by transforming unstructured passenger voices into actionable strategic intelligence for the airline and tourism sectors. 

---
# FinRetrieval: A Benchmark for Financial Data Retrieval by AI Agents 

**Authors**: Eric Y. Kim, Jie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.04403)  

**Abstract**: AI agents increasingly assist with financial research, yet no benchmark evaluates their ability to retrieve specific numeric values from structured databases. We introduce FinRetrieval, a benchmark of 500 financial retrieval questions with ground truth answers, agent responses from 14 configurations across three frontier providers (Anthropic, OpenAI, Google), and complete tool call execution traces. Our evaluation reveals that tool availability dominates performance: Claude Opus achieves 90.8% accuracy with structured data APIs but only 19.8% with web search alone--a 71 percentage point gap that exceeds other providers by 3-4x. We find that reasoning mode benefits vary inversely with base capability (+9.0pp for OpenAI vs +2.8pp for Claude), explained by differences in base-mode tool utilization rather than reasoning ability. Geographic performance gaps (5.6pp US advantage) stem from fiscal year naming conventions, not model limitations. We release the dataset, evaluation code, and tool traces to enable research on financial AI systems. 

---
# SearchGym: A Modular Infrastructure for Cross-Platform Benchmarking and Hybrid Search Orchestration 

**Authors**: Jerome Tze-Hou Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2603.04402)  

**Abstract**: The rapid growth of Retrieval-Augmented Generation (RAG) has created a proliferation of toolkits, yet a fundamental gap remains between experimental prototypes and robust, production-ready systems. We present SearchGym, a modular infrastructure designed for cross-platform benchmarking and hybrid search orchestration. Unlike existing model-centric frameworks, SearchGym decouples data representation, embedding strategies, and retrieval logic into stateful abstractions: Dataset, VectorSet, and App. This separation enables a Compositional Config Algebra, allowing designers to synthesize entire systems from hierarchical configurations while ensuring perfect reproducibility. Moreover, we analyze the "Top-$k$ Cognizance" in hybrid retrieval pipelines, demonstrating that the optimal sequence of semantic ranking and structured filtering is highly dependent on filter strength. Evaluated on the LitSearch expert-annotated benchmark, SearchGym achieves a 70% Top-100 retrieval rate. SearchGym reveals a design tension between generalizability and optimizability, presenting the potential where engineering optimization may serve as a tool for uncovering the causal mechanisms inherent in information retrieval across heterogeneous domains. An open-source implementation of SearchGym is available at: this https URL 

---
