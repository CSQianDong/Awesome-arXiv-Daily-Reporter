# Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty 

**Authors**: Ziyu Chen, Xinbei Jiang, Peng Sun, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.21336)  

**Abstract**: Masked Diffusion Models (MDMs) offer flexible, non-autoregressive generation, but this freedom introduces a challenge: final output quality is highly sensitive to the decoding order. We are the first to formalize this issue, attributing the variability in output quality to the cumulative predictive uncertainty along a generative path. To quantify this uncertainty, we introduce Denoising Entropy, a computable metric that serves as an internal signal for evaluating generative process. Leveraging this metric, we propose two algorithms designed to optimize the decoding path: a post-hoc selection method and a real-time guidance strategy. Experiments demonstrate that our entropy-guided methods significantly improve generation quality, consistently boosting accuracy on challenging reasoning, planning, and code benchmarks. Our work establishes Denoising Entropy as a principled tool for understanding and controlling generation, effectively turning the uncertainty in MDMs from a liability into a key advantage for discovering high-quality solutions. 

---
# C2LLM Technical Report: A New Frontier in Code Retrieval via Adaptive Cross-Attention Pooling 

**Authors**: Jin Qin, Zihan Liao, Ziyin Zhang, Hang Yu, Peng Di, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21332)  

**Abstract**: We present C2LLM - Contrastive Code Large Language Models, a family of code embedding models in both 0.5B and 7B sizes. Building upon Qwen-2.5-Coder backbones, C2LLM adopts a Pooling by Multihead Attention (PMA) module for generating sequence embedding from token embeddings, effectively 1) utilizing the LLM's causal representations acquired during pretraining, while also 2) being able to aggregate information from all tokens in the sequence, breaking the information bottleneck in EOS-based sequence embeddings, and 3) supporting flexible adaptation of embedding dimension, serving as an alternative to MRL. Trained on three million publicly available data, C2LLM models set new records on MTEB-Code among models of similar sizes, with C2LLM-7B ranking 1st on the overall leaderboard. 

---
# Your Reasoning Benchmark May Not Test Reasoning: Revealing Perception Bottleneck in Abstract Reasoning Benchmarks 

**Authors**: Xinhe Wang, Jin Huang, Xingjian Zhang, Tianhao Wang, Jiaqi W. Ma  

**Link**: [PDF](https://arxiv.org/pdf/2512.21329)  

**Abstract**: Reasoning benchmarks such as the Abstraction and Reasoning Corpus (ARC) and ARC-AGI are widely used to assess progress in artificial intelligence and are often interpreted as probes of core, so-called ``fluid'' reasoning abilities. Despite their apparent simplicity for humans, these tasks remain challenging for frontier vision-language models (VLMs), a gap commonly attributed to deficiencies in machine reasoning. We challenge this interpretation and hypothesize that the gap arises primarily from limitations in visual perception rather than from shortcomings in inductive reasoning.
To verify this hypothesis, we introduce a two-stage experimental pipeline that explicitly separates perception and reasoning. In the perception stage, each image is independently converted into a natural-language description, while in the reasoning stage a model induces and applies rules using these descriptions. This design prevents leakage of cross-image inductive signals and isolates reasoning from perception bottlenecks. Across three ARC-style datasets, Mini-ARC, ACRE, and Bongard-LOGO, we show that the perception capability is the dominant factor underlying the observed performance gap by comparing the two-stage pipeline with against standard end-to-end one-stage evaluation. Manual inspection of reasoning traces in the VLM outputs further reveals that approximately 80 percent of model failures stem from perception errors. Together, these results demonstrate that ARC-style benchmarks conflate perceptual and reasoning challenges and that observed performance gaps may overstate deficiencies in machine reasoning. Our findings underscore the need for evaluation protocols that disentangle perception from reasoning when assessing progress in machine intelligence. 

---
# Parallel Token Prediction for Language Models 

**Authors**: Felix Draxler, Justus Will, Farrin Marouf Sofian, Theofanis Karaletsos, Sameer Singh, Stephan Mandt  

**Link**: [PDF](https://arxiv.org/pdf/2512.21323)  

**Abstract**: We propose Parallel Token Prediction (PTP), a universal framework for parallel sequence generation in language models. PTP jointly predicts multiple dependent tokens in a single transformer call by incorporating the sampling procedure into the model. This reduces the latency bottleneck of autoregressive decoding, and avoids the restrictive independence assumptions common in existing multi-token prediction methods. We prove that PTP can represent arbitrary autoregressive sequence distributions. PTP is trained either by distilling an existing model or through inverse autoregressive training without a teacher. Experimentally, we achieve state-of-the-art speculative decoding performance on Vicuna-7B by accepting over four tokens per step on Spec-Bench. The universality of our framework indicates that parallel generation of long sequences is feasible without loss of modeling power. 

---
# SMART SLM: Structured Memory and Reasoning Transformer, A Small Language Model for Accurate Document Assistance 

**Authors**: Divij Dudeja, Mayukha Pal  

**Link**: [PDF](https://arxiv.org/pdf/2512.21280)  

**Abstract**: The user of Engineering Manuals (EM) finds it difficult to read EM s because they are long, have a dense format which includes written documents, step by step procedures, and standard parameter lists for engineering equipment. Off the shelf transformers, especially compact ones, treat this material as a flat stream of tokens. This approach leads to confident but incorrect numeric answers and forces the models to memorize separate facts inefficiently. SMART (Structured Memory and Reasoning Transformer) offers a different and practical solution to the above problem. SMART structures its processing by using a hierarchical approach, and is based upon three main job categories (1) A syntax-aware Fact Extractor (Grammarian) Tree LSTM which extracts facts as subject relation object relations from EM sentences (2) A compact indexed memory MANN (Memory Augmented Neural Network) that indexes these Rational Subject Relation Objects as 384 dimensional vectors that are associated with the source of the information, and (3) A 6 layer Transformer that learns to fuse the previously retrieved facts into its generated response. The entire SMART model utilizes 45.51M parameters, which is 64% less than GPT-2 (124M) and 69% less than BERT (133M), and it achieves a 21.3% higher accuracy than GPT-2, indicating that SMART fits the data better with the least amount of processing requirements. SMART employs dual modes of inference an indexed fast path for known documents (sub-second answer times) and an indexed dynamic path assisted by RAGs for new uploads (FAISS Top 20 results with memory severed at 64 slots). In real world deployment, this framework leads to more well supported results with reduced hallucinations than comparable small transformer models. 

---
# SpidR-Adapt: A Universal Speech Representation Model for Few-Shot Adaptation 

**Authors**: Mahi Luthra, Jiayi Shen, Maxime Poli, Angelo Ortiz, Yosuke Higuchi, Youssef Benchekroun, Martin Gleize, Charles-Eric Saint-James, Dongyan Lin, Phillip Rust, Angel Villar, Surya Parimi, Vanessa Stark, Rashel Moritz, Juan Pino, Yann LeCun, Emmanuel Dupoux  

**Link**: [PDF](https://arxiv.org/pdf/2512.21204)  

**Abstract**: Human infants, with only a few hundred hours of speech exposure, acquire basic units of new languages, highlighting a striking efficiency gap compared to the data-hungry self-supervised speech models. To address this gap, this paper introduces SpidR-Adapt for rapid adaptation to new languages using minimal unlabeled data. We cast such low-resource speech representation learning as a meta-learning problem and construct a multi-task adaptive pre-training (MAdaPT) protocol which formulates the adaptation process as a bi-level optimization framework. To enable scalable meta-training under this framework, we propose a novel heuristic solution, first-order bi-level optimization (FOBLO), avoiding heavy computation costs. Finally, we stabilize meta-training by using a robust initialization through interleaved supervision which alternates self-supervised and supervised objectives. Empirically, SpidR-Adapt achieves rapid gains in phonemic discriminability (ABX) and spoken language modeling (sWUGGY, sBLIMP, tSC), improving over in-domain language models after training on less than 1h of target-language audio, over $100\times$ more data-efficient than standard training. These findings highlight a practical, architecture-agnostic path toward biologically inspired, data-efficient representations. We open-source the training code and model checkpoints at this https URL. 

---
# ClarifyMT-Bench: Benchmarking and Improving Multi-Turn Clarification for Conversational Large Language Models 

**Authors**: Sichun Luo, Yi Huang, Mukai Li, Shichang Meng, Fengyuan Liu, Zefa Hu, Junlan Feng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21120)  

**Abstract**: Large language models (LLMs) are increasingly deployed as conversational assistants in open-domain, multi-turn settings, where users often provide incomplete or ambiguous information. However, existing LLM-focused clarification benchmarks primarily assume single-turn interactions or cooperative users, limiting their ability to evaluate clarification behavior in realistic settings. We introduce \textbf{ClarifyMT-Bench}, a benchmark for multi-turn clarification grounded in a five-dimensional ambiguity taxonomy and a set of six behaviorally diverse simulated user personas. Through a hybrid LLM-human pipeline, we construct 6,120 multi-turn dialogues capturing diverse ambiguity sources and interaction patterns. Evaluating ten representative LLMs uncovers a consistent under-clarification bias: LLMs tend to answer prematurely, and performance degrades as dialogue depth increases. To mitigate this, we propose \textbf{ClarifyAgent}, an agentic approach that decomposes clarification into perception, forecasting, tracking, and planning, substantially improving robustness across ambiguity conditions. ClarifyMT-Bench establishes a reproducible foundation for studying when LLMs should ask, when they should answer, and how to navigate ambiguity in real-world human-LLM interactions. 

---
# Semi-Supervised Learning for Large Language Models Safety and Content Moderation 

**Authors**: Eduard Stefan Dinuta, Iustin Sirbu, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2512.21107)  

**Abstract**: Safety for Large Language Models (LLMs) has been an ongoing research focus since their emergence and is even more relevant nowadays with the increasing capacity of those models. Currently, there are several guardrails in place for all public LLMs and multiple proposed datasets for training safety classifiers. However, training these safety classifiers relies on large quantities of labeled data, which can be problematic to acquire, prone to labeling errors, or often include synthetic data. To address these issues, we suggest a different approach: utilizing semi-supervised learning techniques, which leverage both labeled and unlabeled data, to improve the performance on the safety task. We analyze the improvements that these techniques can offer for both prompts given to Large Language Models and the responses to those requests. Moreover, since augmentation is the central part of semi-supervised algorithms, we demonstrate the importance of using task-specific augmentations, which significantly increase the performance when compared to general-purpose augmentation techniques. 

---
# Semantic Refinement with LLMs for Graph Representations 

**Authors**: Safal Thapaliya, Zehong Wang, Jiazheng Li, Ziming Li, Yanfang Ye, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21106)  

**Abstract**: Graph-structured data exhibit substantial heterogeneity in where their predictive signals originate: in some domains, node-level semantics dominate, while in others, structural patterns play a central role. This structure-semantics heterogeneity implies that no graph learning model with a fixed inductive bias can generalize optimally across diverse graph domains. However, most existing methods address this challenge from the model side by incrementally injecting new inductive biases, which remains fundamentally limited given the open-ended diversity of real-world graphs. In this work, we take a data-centric perspective and treat node semantics as a task-adaptive variable. We propose a Data-Adaptive Semantic Refinement framework DAS for graph representation learning, which couples a fixed graph neural network (GNN) and a large language model (LLM) in a closed feedback loop. The GNN provides implicit supervisory signals to guide the semantic refinement of LLM, and the refined semantics are fed back to update the same graph learner. We evaluate our approach on both text-rich and text-free graphs. Results show consistent improvements on structure-dominated graphs while remaining competitive on semantics-rich graphs, demonstrating the effectiveness of data-centric semantic adaptation under structure-semantics heterogeneity. 

---
# Rethinking Supervised Fine-Tuning: Emphasizing Key Answer Tokens for Improved LLM Accuracy 

**Authors**: Xiaofeng Shi, Qian Kou, Yuduo Li, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.21017)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the Chain-of-Thought (CoT) component has become significant for complex reasoning tasks. However, in conventional Supervised Fine-Tuning (SFT), the model could allocate disproportionately more attention to CoT sequences with excessive length. This reduces focus on the much shorter but essential Key portion-the final answer, whose correctness directly determines task success and evaluation quality. To address this limitation, we propose SFTKey, a two-stage training scheme. In the first stage, conventional SFT is applied to ensure proper output format, while in the second stage, only the Key portion is fine-tuned to improve accuracy. Extensive experiments across multiple benchmarks and model families demonstrate that SFTKey achieves an average accuracy improvement exceeding 5\% over conventional SFT, while preserving the ability to generate correct formats. Overall, this study advances LLM fine-tuning by explicitly balancing CoT learning with additional optimization on answer-relevant tokens. 

---
# Distilling the Essence: Efficient Reasoning Distillation via Sequence Truncation 

**Authors**: Wei-Rui Chen, Vignesh Kothapalli, Ata Fatahibaarzi, Hejian Sang, Shao Tang, Qingquan Song, Zhipeng Wang, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2512.21002)  

**Abstract**: Distilling the reasoning capabilities from a large language model (LLM) to a smaller student model often involves training on substantial amounts of reasoning data. However, distillation over lengthy sequences with prompt (P), chain-of-thought (CoT), and answer (A) segments makes the process computationally expensive. In this work, we investigate how the allocation of supervision across different segments (P, CoT, A) affects student performance. Our analysis shows that selective knowledge distillation over only the CoT tokens can be effective when the prompt and answer information is encompassed by it. Building on this insight, we establish a truncation protocol to quantify computation-quality tradeoffs as a function of sequence length. We observe that training on only the first $50\%$ of tokens of every training sequence can retain, on average, $\approx94\%$ of full-sequence performance on math benchmarks while reducing training time, memory usage, and FLOPs by about $50\%$ each. These findings suggest that reasoning distillation benefits from prioritizing early reasoning tokens and provides a simple lever for computation-quality tradeoffs. Codes are available at this https URL. 

---
# Automatic Replication of LLM Mistakes in Medical Conversations 

**Authors**: Oleksii Proniakin, Diego Fajardo, Ruslan Nazarenko, Razvan Marinescu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20983)  

**Abstract**: Large language models (LLMs) are increasingly evaluated in clinical settings using multi-dimensional rubrics which quantify reasoning quality, safety, and patient-centeredness. Yet, replicating specific mistakes in other LLM models is not straightforward and often requires manual effort. We introduce MedMistake, an automatic pipeline that extracts mistakes LLMs make in patient-doctor conversations and converts them into a benchmark of single-shot QA pairs. Our pipeline (1) creates complex, conversational data between an LLM patient and LLM doctor, (2) runs an evaluation with a committee of 2 LLM judges across a variety of dimensions and (3) creates simplified single-shot QA scenarios from those mistakes. We release MedMistake-All, a dataset of 3,390 single-shot QA pairs where GPT-5 and Gemini 2.5 Pro are currently failing to answer correctly, as judged by two LLM judges. We used medical experts to validate a subset of 211/3390 questions (MedMistake-Bench), which we used to run a final evaluation of 12 frontier LLMs: Claude Opus 4.5, Claude Sonnet 4.5, DeepSeek-Chat, Gemini 2.5 Pro, Gemini 3 Pro, GPT-4o, GPT-5, GPT-5.1, GPT-5.2, Grok 4, Grok 4.1, Mistral Large. We found that GPT models, Claude and Grok obtained the best performance on MedMistake-Bench. We release both the doctor-validated benchmark (MedMistake-Bench), as well as the full dataset (MedMistake-All) at this https URL. 

---
# Reflection Pretraining Enables Token-Level Self-Correction in Biological Sequence Models 

**Authors**: Xiang Zhang, Jiaqi Wei, Yuejin Yang, Zijie Qiu, Yuhan Chen, Zhiqiang Gao, Muhammad Abdul-Mageed, Laks V. S. Lakshmanan, Wanli Ouyang, Chenyu You, Siqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.20954)  

**Abstract**: Chain-of-Thought (CoT) prompting has significantly advanced task-solving capabilities in natural language processing with large language models. Unlike standard prompting, CoT encourages the model to generate intermediate reasoning steps, non-answer tokens, that help guide the model toward more accurate final outputs. These intermediate steps enable more complex reasoning processes such as error correction, memory management, future planning, and self-reflection. However, applying CoT to non-natural language domains, such as protein and RNA language models, is not yet possible, primarily due to the limited expressiveness of their token spaces (e.g., amino acid tokens). In this work, we propose and define the concept of language expressiveness: the ability of a given language, using its tokens and grammar, to encode information. We show that the limited expressiveness of protein language severely restricts the applicability of CoT-style reasoning. To overcome this, we introduce reflection pretraining, for the first time in a biological sequence model, which enables the model to engage in intermediate reasoning through the generation of auxiliary "thinking tokens" beyond simple answer tokens. Theoretically, we demonstrate that our augmented token set significantly enhances biological language expressiveness, thereby improving the overall reasoning capacity of the model. Experimentally, our pretraining approach teaches protein models to self-correct and leads to substantial performance gains compared to standard pretraining. 

---
# MultiMind at SemEval-2025 Task 7: Crosslingual Fact-Checked Claim Retrieval via Multi-Source Alignment 

**Authors**: Mohammad Mahdi Abootorabi, Alireza Ghahramani Kure, Mohammadali Mohammadkhani, Sina Elahimanesh, Mohammad Ali Ali Panah  

**Link**: [PDF](https://arxiv.org/pdf/2512.20950)  

**Abstract**: This paper presents our system for SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval. In an era where misinformation spreads rapidly, effective fact-checking is increasingly critical. We introduce TriAligner, a novel approach that leverages a dual-encoder architecture with contrastive learning and incorporates both native and English translations across different modalities. Our method effectively retrieves claims across multiple languages by learning the relative importance of different sources in alignment. To enhance robustness, we employ efficient data preprocessing and augmentation using large language models while incorporating hard negative sampling to improve representation learning. We evaluate our approach on monolingual and crosslingual benchmarks, demonstrating significant improvements in retrieval accuracy and fact-checking performance over baselines. 

---
# Neural Probe-Based Hallucination Detection for Large Language Models 

**Authors**: Shize Liang, Hongzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20949)  

**Abstract**: Large language models(LLMs) excel at text generation and knowledge question-answering tasks, but they are prone to generating hallucinated content, severely limiting their application in high-risk domains. Current hallucination detection methods based on uncertainty estimation and external knowledge retrieval suffer from the limitation that they still produce erroneous content at high confidence levels and rely heavily on retrieval efficiency and knowledge coverage. In contrast, probe methods that leverage the model's hidden-layer states offer real-time and lightweight advantages. However, traditional linear probes struggle to capture nonlinear structures in deep semantic this http URL overcome these limitations, we propose a neural network-based framework for token-level hallucination detection. By freezing language model parameters, we employ lightweight MLP probes to perform nonlinear modeling of high-level hidden states. A multi-objective joint loss function is designed to enhance detection stability and semantic disambiguity. Additionally, we establish a layer position-probe performance response model, using Bayesian optimization to automatically search for optimal probe insertion layers and achieve superior training this http URL results on LongFact, HealthBench, and TriviaQA demonstrate that MLP probes significantly outperform state-of-the-art methods in accuracy, recall, and detection capability under low false-positive conditions. 

---
# Foundation Model-based Evaluation of Neuropsychiatric Disorders: A Lifespan-Inclusive, Multi-Modal, and Multi-Lingual Study 

**Authors**: Zhongren Dong, Haotian Guo, Weixiang Xu, Huan Zhao, Zixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20948)  

**Abstract**: Neuropsychiatric disorders, such as Alzheimer's disease (AD), depression, and autism spectrum disorder (ASD), are characterized by linguistic and acoustic abnormalities, offering potential biomarkers for early detection. Despite the promise of multi-modal approaches, challenges like multi-lingual generalization and the absence of a unified evaluation framework persist. To address these gaps, we propose FEND (Foundation model-based Evaluation of Neuropsychiatric Disorders), a comprehensive multi-modal framework integrating speech and text modalities for detecting AD, depression, and ASD across the lifespan. Leveraging 13 multi-lingual datasets spanning English, Chinese, Greek, French, and Dutch, we systematically evaluate multi-modal fusion performance. Our results show that multi-modal fusion excels in AD and depression detection but underperforms in ASD due to dataset heterogeneity. We also identify modality imbalance as a prevalent issue, where multi-modal fusion fails to surpass the best mono-modal models. Cross-corpus experiments reveal robust performance in task- and language-consistent scenarios but noticeable degradation in multi-lingual and task-heterogeneous settings. By providing extensive benchmarks and a detailed analysis of performance-influencing factors, FEND advances the field of automated, lifespan-inclusive, and multi-lingual neuropsychiatric disorder assessment. We encourage researchers to adopt the FEND framework for fair comparisons and reproducible research. 

---
# Where Did This Sentence Come From? Tracing Provenance in LLM Reasoning Distillation 

**Authors**: Kaiyuan Liu, Shaotian Yan, Rui Miao, Bing Wang, Chen Shen, Jun Zhang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2512.20908)  

**Abstract**: Reasoning distillation has attracted increasing attention. It typically leverages a large teacher model to generate reasoning paths, which are then used to fine-tune a student model so that it mimics the teacher's behavior in training contexts. However, previous approaches have lacked a detailed analysis of the origins of the distilled model's capabilities. It remains unclear whether the student can maintain consistent behaviors with the teacher in novel test-time contexts, or whether it regresses to its original output patterns, raising concerns about the generalization of distillation models. To analyse this question, we introduce a cross-model Reasoning Distillation Provenance Tracing framework. For each action (e.g., a sentence) produced by the distilled model, we obtain the predictive probabilities assigned by the teacher, the original student, and the distilled model under the same context. By comparing these probabilities, we classify each action into different categories. By systematically disentangling the provenance of each action, we experimentally demonstrate that, in test-time contexts, the distilled model can indeed generate teacher-originated actions, which correlate with and plausibly explain observed performance on distilled model. Building on this analysis, we further propose a teacher-guided data selection method. Unlike prior approach that rely on heuristics, our method directly compares teacher-student divergences on the training data, providing a principled selection criterion. We validate the effectiveness of our approach across multiple representative teacher models and diverse student models. The results highlight the utility of our provenance-tracing framework and underscore its promise for reasoning distillation. We hope to share Reasoning Distillation Provenance Tracing and our insights into reasoning distillation with the community. 

---
# Architectural Trade-offs in Small Language Models Under Compute Constraints 

**Authors**: Shivraj Singh Bhatti  

**Link**: [PDF](https://arxiv.org/pdf/2512.20877)  

**Abstract**: We present a systematic empirical study of small language models under strict compute constraints, analyzing how architectural choices and training budget interact to determine performance. Starting from a linear next-token predictor, we progressively introduce nonlinearities, self-attention, and multi-layer transformer architectures, evaluating each on character-level modeling of Tiny Shakespeare and word-level modeling of Penn Treebank (PTB) and WikiText-2. We compare models using test negative log-likelihood (NLL), parameter count, and approximate training FLOPs to characterize accuracy-efficiency trade-offs. Our results show that attention-based models dominate MLPs in per-FLOP efficiency even at small scale, while increasing depth or context without sufficient optimization can degrade performance. We further examine rotary positional embeddings (RoPE), finding that architectural techniques successful in large language models do not necessarily transfer to small-model regimes. 

---
# NVIDIA Nemotron 3: Efficient and Open Intelligence 

**Authors**: NVIDIA, Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta, Abhinav Khattar, Adi Renduchintala, Aditya Vavre, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Kondratenko, Alexander Bukharin, Alexandre Milesi, Ali Taghibakhshi, Alisa Liu, Amelia Barton, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Amy Shen, Anahita Bhiwandiwalla, Andrew Tao, Anjulie Agrusa, Ankur Verma, Ann Guan, Anubhav Mandarwal, Arham Mehta, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asit Mishra, Asma Kuriparambil Thekkumpate, Ayush Dattagupta, Banghua Zhu, Bardiya Sadeghi, Barnaby Simkin, Ben Lanir, Benedikt Schifferer, Besmira Nushi, Bilal Kartal, Bita Darvish Rouhani, Boris Ginsburg, Brandon Norick, Brandon Soubasis, Branislav Kisacanin, Brian Yu, Bryan Catanzaro, Carlo del Mundo, Chantal Hwang, Charles Wang, Cheng-Ping Hsieh, Chenghao Zhang, Chenhan Yu, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Christopher Parisien, Collin Neale, Cyril Meurillon, Damon Mosk-Aoyama, Dan Su, Dane Corneil, Daniel Afrimi, Daniel Lo, Daniel Rohrer, Daniel Serebrenik, Daria Gitman, Daria Levy, Darko Stosic, David Mosallanezhad, Deepak Narayanan, Dhruv Nathawani, Dima Rekesh, Dina Yared, Divyanshu Kakwani, Dong Ahn, Duncan Riach, Dusan Stosic, Edgar Minasyan, Edward Lin, Eileen Long, Eileen Peters Long, Elad Segal, Elena Lantz, Ellie Evans, Elliott Ning, Eric Chung, Eric Harper, Eric Tramel, Erick Galinkin, Erik Pounds, Evan Briones, Evelina Bakhturina, Evgeny Tsykunov, Faisal Ladhak, Fay Wang, Fei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2512.20856)  

**Abstract**: We introduce the Nemotron 3 family of models - Nano, Super, and Ultra. These models deliver strong agentic, reasoning, and conversational capabilities. The Nemotron 3 family uses a Mixture-of-Experts hybrid Mamba-Transformer architecture to provide best-in-class throughput and context lengths of up to 1M tokens. Super and Ultra models are trained with NVFP4 and incorporate LatentMoE, a novel approach that improves model quality. The two larger models also include MTP layers for faster text generation. All Nemotron 3 models are post-trained using multi-environment reinforcement learning enabling reasoning, multi-step tool use, and support granular reasoning budget control. Nano, the smallest model, outperforms comparable models in accuracy while remaining extremely cost-efficient for inference. Super is optimized for collaborative agents and high-volume workloads such as IT ticket automation. Ultra, the largest model, provides state-of-the-art accuracy and reasoning performance. Nano is released together with its technical report and this white paper, while Super and Ultra will follow in the coming months. We will openly release the model weights, pre- and post-training software, recipes, and all data for which we hold redistribution rights. 

---
# How important is Recall for Measuring Retrieval Quality? 

**Authors**: Shelly Schwartz, Oleg Vasilyev, Randy Sawaya  

**Link**: [PDF](https://arxiv.org/pdf/2512.20854)  

**Abstract**: In realistic retrieval settings with large and evolving knowledge bases, the total number of documents relevant to a query is typically unknown, and recall cannot be computed. In this paper, we evaluate several established strategies for handling this limitation by measuring the correlation between retrieval quality metrics and LLM-based judgments of response quality, where responses are generated from the retrieved documents. We conduct experiments across multiple datasets with a relatively low number of relevant documents (2-15). We also introduce a simple retrieval quality measure that performs well without requiring knowledge of the total number of relevant documents. 

---
# Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning 

**Authors**: NVIDIA, Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta, Abhinav Khattar, Adi Renduchintala, Aditya Vavre, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Kondratenko, Alexander Bukharin, Alexandre Milesi, Ali Taghibakhshi, Alisa Liu, Amelia Barton, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Amy Shen, Anahita Bhiwandiwalla, Andrew Tao, Ann Guan, Anubhav Mandarwal, Arham Mehta, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asma Kuriparambil Thekkumpate, Ayush Dattagupta, Banghua Zhu, Bardiya Sadeghi, Barnaby Simkin, Ben Lanir, Benedikt Schifferer, Besmira Nushi, Bilal Kartal, Bita Darvish Rouhani, Boris Ginsburg, Brandon Norick, Brandon Soubasis, Branislav Kisacanin, Brian Yu, Bryan Catanzaro, Carlo del Mundo, Chantal Hwang, Charles Wang, Cheng-Ping Hsieh, Chenghao Zhang, Chenhan Yu, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Christopher Parisien, Collin Neale, Damon Mosk-Aoyama, Dan Su, Dane Corneil, Daniel Afrimi, Daniel Rohrer, Daniel Serebrenik, Daria Gitman, Daria Levy, Darko Stosic, David Mosallanezhad, Deepak Narayanan, Dhruv Nathawani, Dima Rekesh, Dina Yared, Divyanshu Kakwani, Dong Ahn, Duncan Riach, Dusan Stosic, Edgar Minasyan, Edward Lin, Eileen Long, Eileen Peters Long, Elena Lantz, Ellie Evans, Elliott Ning, Eric Chung, Eric Harper, Eric Tramel, Erick Galinkin, Erik Pounds, Evan Briones, Evelina Bakhturina, Faisal Ladhak, Fay Wang, Fei Jia, Felipe Soares, Feng Chen, Ferenc Galko, Frankie Siino, Gal Hubara Agam, Ganesh Ajjanagadde, Gantavya Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2512.20848)  

**Abstract**: We present Nemotron 3 Nano 30B-A3B, a Mixture-of-Experts hybrid Mamba-Transformer language model. Nemotron 3 Nano was pretrained on 25 trillion text tokens, including more than 3 trillion new unique tokens over Nemotron 2, followed by supervised fine tuning and large-scale RL on diverse environments. Nemotron 3 Nano achieves better accuracy than our previous generation Nemotron 2 Nano while activating less than half of the parameters per forward pass. It achieves up to 3.3x higher inference throughput than similarly-sized open models like GPT-OSS-20B and Qwen3-30B-A3B-Thinking-2507, while also being more accurate on popular benchmarks. Nemotron 3 Nano demonstrates enhanced agentic, reasoning, and chat abilities and supports context lengths up to 1M tokens. We release both our pretrained Nemotron 3 Nano 30B-A3B Base and post-trained Nemotron 3 Nano 30B-A3B checkpoints on Hugging Face. 

---
# MediEval: A Unified Medical Benchmark for Patient-Contextual and Knowledge-Grounded Reasoning in LLMs 

**Authors**: Zhan Qu, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2512.20822)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to medicine, yet their adoption is limited by concerns over reliability and safety. Existing evaluations either test factual medical knowledge in isolation or assess patient-level reasoning without verifying correctness, leaving a critical gap. We introduce MediEval, a benchmark that links MIMIC-IV electronic health records (EHRs) to a unified knowledge base built from UMLS and other biomedical vocabularies. MediEval generates diverse factual and counterfactual medical statements within real patient contexts, enabling systematic evaluation across a 4-quadrant framework that jointly considers knowledge grounding and contextual consistency. Using this framework, we identify critical failure modes, including hallucinated support and truth inversion, that current proprietary, open-source, and domain-specific LLMs frequently exhibit. To address these risks, we propose Counterfactual Risk-Aware Fine-tuning (CoRFu), a DPO-based method with an asymmetric penalty targeting unsafe confusions. CoRFu improves by +16.4 macro-F1 points over the base model and eliminates truth inversion errors, demonstrating both higher accuracy and substantially greater safety. 

---
# EssayCBM: Rubric-Aligned Concept Bottleneck Models for Transparent Essay Grading 

**Authors**: Kumar Satvik Chaudhary, Chengshuai Zhao, Fan Zhang, Yung Hin Tse, Garima Agrawal, Yuli Deng, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.20817)  

**Abstract**: Understanding how automated grading systems evaluate essays remains a significant challenge for educators and students, especially when large language models function as black boxes. We introduce EssayCBM, a rubric-aligned framework that prioritizes interpretability in essay assessment. Instead of predicting grades directly from text, EssayCBM evaluates eight writing concepts, such as Thesis Clarity and Evidence Use, through dedicated prediction heads on an encoder. These concept scores form a transparent bottleneck, and a lightweight network computes the final grade using only concepts. Instructors can adjust concept predictions and instantly view the updated grade, enabling accountable human-in-the-loop evaluation. EssayCBM matches black-box performance while offering actionable, concept-level feedback through an intuitive web interface. 

---
# Semantic Deception: When Reasoning Models Can't Compute an Addition 

**Authors**: Nathaniël de Leeuw, Marceau Nahon, Mathis Reymond, Raja Chatila, Mehdi Khamassi  

**Link**: [PDF](https://arxiv.org/pdf/2512.20812)  

**Abstract**: Large language models (LLMs) are increasingly used in situations where human values are at stake, such as decision-making tasks that involve reasoning when performed by humans. We investigate the so-called reasoning capabilities of LLMs over novel symbolic representations by introducing an experimental framework that tests their ability to process and manipulate unfamiliar symbols. We introduce semantic deceptions: situations in which symbols carry misleading semantic associations due to their form, such as being embedded in specific contexts, designed to probe whether LLMs can maintain symbolic abstraction or whether they default to exploiting learned semantic associations. We redefine standard digits and mathematical operators using novel symbols, and task LLMs with solving simple calculations expressed in this altered notation. The objective is: (1) to assess LLMs' capacity for abstraction and manipulation of arbitrary symbol systems; (2) to evaluate their ability to resist misleading semantic cues that conflict with the task's symbolic logic. Through experiments with four LLMs we show that semantic cues can significantly deteriorate reasoning models' performance on very simple tasks. They reveal limitations in current LLMs' ability for symbolic manipulations and highlight a tendency to over-rely on surface-level semantics, suggesting that chain-of-thoughts may amplify reliance on statistical correlations. Even in situations where LLMs seem to correctly follow instructions, semantic cues still impact basic capabilities. These limitations raise ethical and societal concerns, undermining the widespread and pernicious tendency to attribute reasoning abilities to LLMs and suggesting how LLMs might fail, in particular in decision-making contexts where robust symbolic reasoning is essential and should not be compromised by residual semantic associations inherited from the model's training. 

---
# Measuring Mechanistic Independence: Can Bias Be Removed Without Erasing Demographics? 

**Authors**: Zhengyang Shan, Aaron Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2512.20796)  

**Abstract**: We investigate how independent demographic bias mechanisms are from general demographic recognition in language models. Using a multi-task evaluation setup where demographics are associated with names, professions, and education levels, we measure whether models can be debiased while preserving demographic detection capabilities. We compare attribution-based and correlation-based methods for locating bias features. We find that targeted sparse autoencoder feature ablations in Gemma-2-9B reduce bias without degrading recognition performance: attribution-based ablations mitigate race and gender profession stereotypes while preserving name recognition accuracy, whereas correlation-based ablations are more effective for education bias. Qualitative analysis further reveals that removing attribution features in education tasks induces ``prior collapse'', thus increasing overall bias. This highlights the need for dimension-specific interventions. Overall, our results show that demographic bias arises from task-specific mechanisms rather than absolute demographic markers, and that mechanistic inference-time interventions can enable surgical debiasing without compromising core model capabilities. 

---
# Investigating Model Editing for Unlearning in Large Language Models 

**Authors**: Shariqah Hossain, Lalana Kagal  

**Link**: [PDF](https://arxiv.org/pdf/2512.20794)  

**Abstract**: Machine unlearning aims to remove unwanted information from a model, but many methods are inefficient for LLMs with large numbers of parameters or fail to fully remove the intended information without degrading performance on knowledge that should be retained. Model editing algorithms solve a similar problem of changing information in models, but they focus on redirecting inputs to a new target rather than removing that information altogether. In this work, we explore the editing algorithms ROME, IKE, and WISE and design new editing targets for an unlearning setting. Through this investigation, we show that model editing approaches can exceed baseline unlearning methods in terms of quality of forgetting depending on the setting. Like traditional unlearning techniques, they struggle to encapsulate the scope of what is to be unlearned without damage to the overall model performance. 

---
# Large Language Models Approach Expert Pedagogical Quality in Math Tutoring but Differ in Instructional and Linguistic Profiles 

**Authors**: Ramatu Oiza Abdulsalam, Segun Aroyehun  

**Link**: [PDF](https://arxiv.org/pdf/2512.20780)  

**Abstract**: Recent work has explored the use of large language models for generating tutoring responses in mathematics, yet it remains unclear how closely their instructional behavior aligns with expert human practice. We examine this question using a controlled, turn-level comparison in which expert human tutors, novice human tutors, and multiple large language models respond to the same set of math remediation conversation turns. We examine both instructional strategies and linguistic characteristics of tutoring responses, including restating and revoicing, pressing for accuracy, lexical diversity, readability, politeness, and agency. We find that large language models approach expert levels of perceived pedagogical quality on average but exhibit systematic differences in their instructional and linguistic profiles. In particular, large language models tend to underuse restating and revoicing strategies characteristic of expert human tutors, while producing longer, more lexically diverse, and more polite responses. Statistical analyses show that restating and revoicing, lexical diversity, and pressing for accuracy are positively associated with perceived pedagogical quality, whereas higher levels of agentic and polite language are negatively associated. Overall, recent large language models exhibit levels of perceived pedagogical quality comparable to expert human tutors, while relying on different instructional and linguistic strategies. These findings underscore the value of analyzing instructional strategies and linguistic characteristics when evaluating tutoring responses across human tutors and intelligent tutoring systems. 

---
# Adversarial Training for Failure-Sensitive User Simulation in Mental Health Dialogue Optimization 

**Authors**: Ziyi Zhu, Olivier Tieleman, Caitlin A. Stamatis, Luka Smyth, Thomas D. Hull, Daniel R. Cahn, Matteo Malgaroli  

**Link**: [PDF](https://arxiv.org/pdf/2512.20773)  

**Abstract**: Realistic user simulation is crucial for training and evaluating task-oriented dialogue (TOD) systems, yet creating simulators that accurately replicate human behavior remains challenging. A key property of effective simulators is their ability to expose failure modes of the systems they evaluate. We present an adversarial training framework that iteratively improves user simulator realism through a competitive dynamic between a generator (user simulator) and a discriminator. Applied to mental health support chatbots, our approach demonstrates that fine-tuned simulators dramatically outperform zero-shot base models at surfacing system issues, and adversarial training further enhances diversity, distributional alignment, and predictive validity. The resulting simulator achieves a strong correlation between simulated and real failure occurrence rates across diverse chatbot configurations while maintaining low distributional divergence of failure modes. Discriminator accuracy decreases drastically after three adversarial iterations, suggesting improved realism. These results provide evidence that adversarial training is a promising approach for creating realistic user simulators in mental health support TOD domains, enabling rapid, reliable, and cost-effective system evaluation before deployment. 

---
# TokSuite: Measuring the Impact of Tokenizer Choice on Language Model Behavior 

**Authors**: Gül Sena Altıntaş, Malikeh Ehghaghi, Brian Lester, Fengyuan Liu, Wanru Zhao, Marco Ciccone, Colin Raffel  

**Link**: [PDF](https://arxiv.org/pdf/2512.20757)  

**Abstract**: Tokenizers provide the fundamental basis through which text is represented and processed by language models (LMs). Despite the importance of tokenization, its role in LM performance and behavior is poorly understood due to the challenge of measuring the impact of tokenization in isolation. To address this need, we present TokSuite, a collection of models and a benchmark that supports research into tokenization's influence on LMs. Specifically, we train fourteen models that use different tokenizers but are otherwise identical using the same architecture, dataset, training budget, and initialization. Additionally, we curate and release a new benchmark that specifically measures model performance subject to real-world perturbations that are likely to influence tokenization. Together, TokSuite allows robust decoupling of the influence of a model's tokenizer, supporting a series of novel findings that elucidate the respective benefits and shortcomings of a wide range of popular tokenizers. 

---
# SA-DiffuSeq: Addressing Computational and Scalability Challenges in Long-Document Generation with Sparse Attention 

**Authors**: Alexandros Christoforos, Chadbourne Davis  

**Link**: [PDF](https://arxiv.org/pdf/2512.20724)  

**Abstract**: Diffusion based approaches to long form text generation suffer from prohibitive computational cost and memory overhead as sequence length increases. We introduce SA-DiffuSeq, a diffusion framework that integrates sparse attention to fundamentally improve scalability for long document modeling. By selectively allocating attention within the diffusion process, SA-DiffuSeq significantly reduces computational complexity while maintaining semantic coherence and generation quality. A key component of our method is a soft absorbing state tailored to sparse attention dynamics, which stabilizes diffusion trajectories and accelerates sequence reconstruction. This design improves sampling efficiency and enhances precision in long range dependency modeling. Extensive experiments demonstrate that SA-DiffuSeq consistently surpasses state of the art diffusion baselines in both training efficiency and sampling speed, with especially strong gains on extended sequences. These properties make SA-DiffuSeq well suited for demanding long form applications such as scientific writing, large scale code generation, and multi turn long context dialogue. Overall, our results indicate that incorporating structured sparsity into diffusion models is a promising direction for efficient and expressive long text generation. 

---
# Uncovering Competency Gaps in Large Language Models and Their Benchmarks 

**Authors**: Matyas Bohacek, Nino Scherrer, Nicholas Dufour, Thomas Leung, Christoph Bregler, Stephanie C. Y. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2512.20638)  

**Abstract**: The evaluation of large language models (LLMs) relies heavily on standardized benchmarks. These benchmarks provide useful aggregated metrics for a given capability, but those aggregated metrics can obscure (i) particular sub-areas where the LLMs are weak ("model gaps") and (ii) imbalanced coverage in the benchmarks themselves ("benchmark gaps"). We propose a new method that uses sparse autoencoders (SAEs) to automatically uncover both types of gaps. By extracting SAE concept activations and computing saliency-weighted performance scores across benchmark data, the method grounds evaluation in the model's internal representations and enables comparison across benchmarks. As examples demonstrating our approach, we applied the method to two popular open-source models and ten benchmarks. We found that these models consistently underperformed on concepts that stand in contrast to sycophantic behaviors (e.g., politely refusing a request or asserting boundaries) and concepts connected to safety discussions. These model gaps align with observations previously surfaced in the literature; our automated, unsupervised method was able to recover them without manual supervision. We also observed benchmark gaps: many of the evaluated benchmarks over-represented concepts related to obedience, authority, or instruction-following, while missing core concepts that should fall within their intended scope. In sum, our method offers a representation-grounded approach to evaluation, enabling concept-level decomposition of benchmark scores. Rather than replacing conventional aggregated metrics, CG complements them by providing a concept-level decomposition that can reveal why a model scored as it did and how benchmarks could evolve to better reflect their intended scope. Code is available at this https URL. 

---
# Measuring all the noises of LLM Evals 

**Authors**: Sida Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.21326)  

**Abstract**: Separating signal from noise is central to experimental science. Applying well-established statistical method effectively to LLM evals requires consideration of their unique noise characteristics. We clearly define and measure three types of noise: prediction noise from generating different answers on a given question, data noise from sampling questions, and their combined total noise following the law of total variance. To emphasize relative comparisons and gain statistical power, we propose the all-pairs paired method, which applies the paired analysis to all pairs of LLMs and measures all the noise components based on millions of question-level predictions across many evals and settings. These measurements revealed clear patterns. First, each eval exhibits a characteristic and highly predictable total noise level across all model pairs. Second, paired prediction noise typically exceeds paired data noise, which means reducing prediction noise by averaging can significantly increase statistical power. These findings enable practitioners to assess significance without custom testing and to detect much smaller effects in controlled experiments. 

---
# ReaSeq: Unleashing World Knowledge via Reasoning for Sequential Modeling 

**Authors**: Chuan Wang, Gaoming Yang, Han Wu, Jiakai Tang, Jiahao Yu, Jian Wu, Jianwu Hu, Junjun Zheng, Shuwen Xiao, Yeqiu Yang, Yuning Jiang, Ahjol Nurlanbek, Binbin Cao, Bo Zheng, Fangmei Zhu, Gaoming Zhou, Huimin Yi, Huiping Chu, Jin Huang, Jinzhe Shan, Kenan Cui, Longbin Li, Silu Zhou, Wen Chen, Xia Ming, Xiang Gao, Xin Yao, Xingyu Wen, Yan Zhang, Yiwen Hu, Yulin Wang, Ziheng Bao, Zongyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.21257)  

**Abstract**: Industrial recommender systems face two fundamental limitations under the log-driven paradigm: (1) knowledge poverty in ID-based item representations that causes brittle interest modeling under data sparsity, and (2) systemic blindness to beyond-log user interests that constrains model performance within platform boundaries. These limitations stem from an over-reliance on shallow interaction statistics and close-looped feedback while neglecting the rich world knowledge about product semantics and cross-domain behavioral patterns that Large Language Models have learned from vast corpora.
To address these challenges, we introduce ReaSeq, a reasoning-enhanced framework that leverages world knowledge in Large Language Models to address both limitations through explicit and implicit reasoning. Specifically, ReaSeq employs explicit Chain-of-Thought reasoning via multi-agent collaboration to distill structured product knowledge into semantically enriched item representations, and latent reasoning via Diffusion Large Language Models to infer plausible beyond-log behaviors. Deployed on Taobao's ranking system serving hundreds of millions of users, ReaSeq achieves substantial gains: >6.0% in IPV and CTR, >2.9% in Orders, and >2.5% in GMV, validating the effectiveness of world-knowledge-enhanced reasoning over purely log-driven approaches. 

---
# Beyond Context: Large Language Models Failure to Grasp Users Intent 

**Authors**: Ahmed M. Hussain, Salahuddin Salahuddin, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2512.21110)  

**Abstract**: Current Large Language Models (LLMs) safety approaches focus on explicitly harmful content while overlooking a critical vulnerability: the inability to understand context and recognize user intent. This creates exploitable vulnerabilities that malicious users can systematically leverage to circumvent safety mechanisms. We empirically evaluate multiple state-of-the-art LLMs, including ChatGPT, Claude, Gemini, and DeepSeek. Our analysis demonstrates the circumvention of reliable safety mechanisms through emotional framing, progressive revelation, and academic justification techniques. Notably, reasoning-enabled configurations amplified rather than mitigated the effectiveness of exploitation, increasing factual precision while failing to interrogate the underlying intent. The exception was Claude Opus 4.1, which prioritized intent detection over information provision in some use cases. This pattern reveals that current architectural designs create systematic vulnerabilities. These limitations require paradigmatic shifts toward contextual understanding and intent recognition as core safety capabilities rather than post-hoc protective mechanisms. 

---
# Transductive Visual Programming: Evolving Tool Libraries from Experience for Spatial Reasoning 

**Authors**: Shengguang Wu, Xiaohan Wang, Yuhui Zhang, Hao Zhu, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2512.20934)  

**Abstract**: Spatial reasoning in 3D scenes requires precise geometric calculations that challenge vision-language models. Visual programming addresses this by decomposing problems into steps calling specialized tools, yet existing methods rely on either fixed toolsets or speculative tool induction before solving problems, resulting in suboptimal programs and poor utilization of induced tools. We present Transductive Visual Programming (TVP), a novel framework that builds new tools from its own experience rather than speculation. TVP first solves problems using basic tools while accumulating experiential solutions into an Example Library, then abstracts recurring patterns from these programs into reusable higher-level tools for an evolving Tool Library. This allows TVP to tackle new problems with increasingly powerful tools learned from experience. On Omni3D-Bench, TVP achieves state-of-the-art performance, outperforming GPT-4o by 22% and the previous best visual programming system by 11%. Our transductively learned tools are used 5x more frequently as core program dependency than inductively created ones, demonstrating more effective tool discovery and reuse. The evolved tools also show strong generalization to unseen spatial tasks, achieving superior performance on benchmarks from SpatialScore-Hard collection without any testset-specific modification. Our work establishes experience-driven transductive tool creation as a powerful paradigm for building self-evolving visual programming agents that effectively tackle challenging spatial reasoning tasks. We release our code at this https URL. 

---
# Decoding Predictive Inference in Visual Language Processing via Spatiotemporal Neural Coherence 

**Authors**: Sean C. Borneman, Julia Krebs, Ronnie B. Wilbur, Evie A. Malaia  

**Link**: [PDF](https://arxiv.org/pdf/2512.20929)  

**Abstract**: Human language processing relies on the brain's capacity for predictive inference. We present a machine learning framework for decoding neural (EEG) responses to dynamic visual language stimuli in Deaf signers. Using coherence between neural signals and optical flow-derived motion features, we construct spatiotemporal representations of predictive neural dynamics. Through entropy-based feature selection, we identify frequency-specific neural signatures that differentiate interpretable linguistic input from linguistically disrupted (time-reversed) stimuli. Our results reveal distributed left-hemispheric and frontal low-frequency coherence as key features in language comprehension, with experience-dependent neural signatures correlating with age. This work demonstrates a novel multimodal approach for probing experience-driven generative models of perception in the brain. 

---
# Generalization of RLVR Using Causal Reasoning as a Testbed 

**Authors**: Brian Lu, Hongyu Zhao, Shuo Sun, Hao Peng, Rui Ding, Hongyuan Mei  

**Link**: [PDF](https://arxiv.org/pdf/2512.20760)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for post-training large language models (LLMs) on complex reasoning tasks. Yet, the conditions under which RLVR yields robust generalization remain poorly understood. This paper provides an empirical study of RLVR generalization in the setting of probabilistic inference over causal graphical models. This setting offers two natural axes along which to examine generalization: (i) the level of the probabilistic query -- associational, interventional, or counterfactual -- and (ii) the structural complexity of the query, measured by the size of its relevant subgraph. We construct datasets of causal graphs and queries spanning these difficulty axes and fine-tune Qwen-2.5-Instruct models using RLVR or supervised fine-tuning (SFT). We vary both the model scale (3B-32B) and the query level included in training. We find that RLVR yields stronger within-level and across-level generalization than SFT, but only for specific combinations of model size and training query level. Further analysis shows that RLVR's effectiveness depends on the model's initial reasoning competence. With sufficient initial competence, RLVR improves an LLM's marginalization strategy and reduces errors in intermediate probability calculations, producing substantial accuracy gains, particularly on more complex queries. These findings show that RLVR can improve specific causal reasoning subskills, with its benefits emerging only when the model has sufficient initial competence. 

---
# AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent 

**Authors**: Haipeng Luo, Huawen Feng, Qingfeng Sun, Can Xu, Kai Zheng, Yufei Wang, Tao Yang, Han Hu, Yansong Tang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20745)  

**Abstract**: Large Reasoning Models (LRMs) like o3 and DeepSeek-R1 have achieved remarkable progress in natural language reasoning with long chain-of-thought. However, they remain computationally inefficient and struggle with accuracy when solving problems requiring complex mathematical operations. In this work, we present AgentMath, an agent framework that seamlessly integrates language models' reasoning capabilities with code interpreters' computational precision to efficiently tackle complex mathematical problems. Our approach introduces three key innovations: (1) An automated method that converts natural language chain-of-thought into structured tool-augmented trajectories, generating high-quality supervised fine-tuning (SFT) data to alleviate data scarcity; (2) A novel agentic reinforcement learning (RL) paradigm that dynamically interleaves natural language generation with real-time code execution. This enables models to autonomously learn optimal tool-use strategies through multi-round interactive feedback, while fostering emergent capabilities in code refinement and error correction; (3) An efficient training system incorporating innovative techniques, including request-level asynchronous rollout scheduling, agentic partial rollout, and prefix-aware weighted load balancing, achieving 4-5x speedup and making efficient RL training feasible on ultra-long sequences with scenarios with massive tool this http URL evaluations show that AgentMath achieves state-of-the-art performance on challenging mathematical competition benchmarks including AIME24, AIME25, and HMMT25. Specifically, AgentMath-30B-A3B attains 90.6%, 86.4%, and 73.8% accuracy respectively, achieving advanced this http URL results validate the effectiveness of our approach and pave the way for building more sophisticated and scalable mathematical reasoning agents. 

---
# PHOTON: Hierarchical Autoregressive Modeling for Lightspeed and Memory-Efficient Language Generation 

**Authors**: Yuma Ichikawa, Naoya Takagi, Takumi Nakagawa, Yuzi Kanazawa, Akira Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2512.20687)  

**Abstract**: Transformers operate as horizontal token-by-token scanners; at each generation step, the model attends to an ever-growing sequence of token-level states. This access pattern increases prefill latency and makes long-context decoding increasingly memory-bound, as KV-cache reads and writes dominate inference throughput rather than arithmetic computation. We propose Parallel Hierarchical Operation for Top-down Networks (PHOTON), a hierarchical autoregressive model that replaces flat scanning with vertical, multi-resolution context access. PHOTON maintains a hierarchy of latent streams: a bottom-up encoder progressively compresses tokens into low-rate contextual states, while lightweight top-down decoders reconstruct fine-grained token representations. Experimental results show that PHOTON is superior to competitive Transformer-based language models regarding the throughput-quality trade-off, offering significant advantages in long-context and multi-query tasks. This reduces decode-time KV-cache traffic, yielding up to $10^{3}\times$ higher throughput per unit memory. 

---
# Automated Red-Teaming Framework for Large Language Model Security Assessment: A Comprehensive Attack Generation and Detection System 

**Authors**: Zhang Wei, Peilu Hu, Shengning Lang, Hao Yan, Li Mei, Yichao Zhang, Chen Yang, Junfeng Hao, Zhimo Han  

**Link**: [PDF](https://arxiv.org/pdf/2512.20677)  

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes domains, ensuring their security and alignment has become a critical challenge. Existing red-teaming practices depend heavily on manual testing, which limits scalability and fails to comprehensively cover the vast space of potential adversarial behaviors. This paper introduces an automated red-teaming framework that systematically generates, executes, and evaluates adversarial prompts to uncover security vulnerabilities in LLMs. Our framework integrates meta-prompting-based attack synthesis, multi-modal vulnerability detection, and standardized evaluation protocols spanning six major threat categories -- reward hacking, deceptive alignment, data exfiltration, sandbagging, inappropriate tool use, and chain-of-thought manipulation. Experiments on the GPT-OSS-20B model reveal 47 distinct vulnerabilities, including 21 high-severity and 12 novel attack patterns, achieving a $3.9\times$ improvement in vulnerability discovery rate over manual expert testing while maintaining 89\% detection accuracy. These results demonstrate the framework's effectiveness in enabling scalable, systematic, and reproducible AI safety evaluations. By providing actionable insights for improving alignment robustness, this work advances the state of automated LLM red-teaming and contributes to the broader goal of building secure and trustworthy AI systems. 

---
# Real Time Detection and Quantitative Analysis of Spurious Forgetting in Continual Learning 

**Authors**: Weiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.20634)  

**Abstract**: Catastrophic forgetting remains a fundamental challenge in continual learning for large language models. Recent work revealed that performance degradation may stem from spurious forgetting caused by task alignment disruption rather than true knowledge loss. However, this work only qualitatively describes alignment, relies on post-hoc analysis, and lacks automatic distinction mechanisms.
We introduce the shallow versus deep alignment framework, providing the first quantitative characterization of alignment depth. We identify that current task alignment approaches suffer from shallow alignment - maintained only over the first few output tokens (approximately 3-5) - making models vulnerable to forgetting. This explains why spurious forgetting occurs, why it is reversible, and why fine-tuning attacks are effective.
We propose a comprehensive framework addressing all gaps: (1) quantitative metrics (0-1 scale) to measure alignment depth across token positions; (2) real-time detection methods for identifying shallow alignment during training; (3) specialized analysis tools for visualization and recovery prediction; and (4) adaptive mitigation strategies that automatically distinguish forgetting types and promote deep alignment. Extensive experiments on multiple datasets and model architectures (Qwen2.5-3B to Qwen2.5-32B) demonstrate 86.2-90.6% identification accuracy and show that promoting deep alignment improves robustness against forgetting by 3.3-7.1% over baselines. 

---
# Zero-Training Temporal Drift Detection for Transformer Sentiment Models: A Comprehensive Analysis on Authentic Social Media Streams 

**Authors**: Aayam Bansal, Ishaan Gangwani  

**Link**: [PDF](https://arxiv.org/pdf/2512.20631)  

**Abstract**: We present a comprehensive zero-training temporal drift analysis of transformer-based sentiment models validated on authentic social media data from major real-world events. Through systematic evaluation across three transformer architectures and rigorous statistical validation on 12,279 authentic social media posts, we demonstrate significant model instability with accuracy drops reaching 23.4% during event-driven periods. Our analysis reveals maximum confidence drops of 13.0% (Bootstrap 95% CI: [9.1%, 16.5%]) with strong correlation to actual performance degradation. We introduce four novel drift metrics that outperform embedding-based baselines while maintaining computational efficiency suitable for production deployment. Statistical validation across multiple events confirms robust detection capabilities with practical significance exceeding industry monitoring thresholds. This zero-training methodology enables immediate deployment for real-time sentiment monitoring systems and provides new insights into transformer model behavior during dynamic content periods. 

---
# MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation 

**Authors**: Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.20626)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to dynamically access external information, which is powerful for answering questions over previously unseen documents. Nonetheless, they struggle with high-level conceptual understanding and holistic comprehension due to limited context windows, which constrain their ability to perform deep reasoning over long-form, domain-specific content such as full-length books. To solve this problem, knowledge graphs (KGs) have been leveraged to provide entity-centric structure and hierarchical summaries, offering more structured support for reasoning. However, existing KG-based RAG solutions remain restricted to text-only inputs and fail to leverage the complementary insights provided by other modalities such as vision. On the other hand, reasoning from visual documents requires textual, visual, and spatial cues into structured, hierarchical concepts. To address this issue, we introduce a multimodal knowledge graph-based RAG that enables cross-modal reasoning for better content understanding. Our method incorporates visual cues into the construction of knowledge graphs, the retrieval phase, and the answer generation process. Experimental results across both global and fine-grained question answering tasks show that our approach consistently outperforms existing RAG-based approaches on both textual and multimodal corpora. 

---
