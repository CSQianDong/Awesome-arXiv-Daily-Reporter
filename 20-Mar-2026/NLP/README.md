# F2LLM-v2: Inclusive, Performant, and Efficient Embeddings for a Multilingual World 

**Authors**: Ziyin Zhang, Zihan Liao, Hang Yu, Peng Di, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19223)  

**Abstract**: We present F2LLM-v2, a new family of general-purpose, multilingual embedding models in 8 distinct sizes ranging from 80M to 14B. Trained on a newly curated composite of 60 million publicly available high-quality data samples, F2LLM-v2 supports more than 200 languages, with a particular emphasis on previously underserved mid- and low-resource languages. By integrating a two-stage LLM-based embedding training pipeline with matryoshka learning, model pruning, and knowledge distillation techniques, we present models that are far more efficient than previous LLM-based embedding models while retaining competitive performances. Extensive evaluations confirm that F2LLM-v2-14B ranks first on 11 MTEB benchmarks, while the smaller models in the family also set a new state of the art for resource-constrained applications. To facilitate open-source embedding model research, we release all models, data, code, and intermediate checkpoints. 

---
# Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation 

**Authors**: Zhuolin Yang, Zihan Liu, Yang Chen, Wenliang Dai, Boxin Wang, Sheng-Chieh Lin, Chankyu Lee, Yangyi Chen, Dongfu Jiang, Jiafan He, Renjie Pi, Grace Lam, Nayeon Lee, Alexander Bukharin, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2603.19220)  

**Abstract**: We introduce Nemotron-Cascade 2, an open 30B MoE model with 3B activated parameters that delivers best-in-class reasoning and strong agentic capabilities. Despite its compact size, its mathematical and coding reasoning performance approaches that of frontier open models. It is the second open-weight LLM, after DeepSeekV3.2-Speciale-671B-A37B, to achieve Gold Medal-level performance in the 2025 International Mathematical Olympiad (IMO), the International Olympiad in Informatics (IOI), and the ICPC World Finals, demonstrating remarkably high intelligence density with 20x fewer parameters. In contrast to Nemotron-Cascade 1, the key technical advancements are as follows. After SFT on a meticulously curated dataset, we substantially expand Cascade RL to cover a much broader spectrum of reasoning and agentic domains. Furthermore, we introduce multi-domain on-policy distillation from the strongest intermediate teacher models for each domain throughout the Cascade RL process, allowing us to efficiently recover benchmark regressions and sustain strong performance gains along the way. We release the collection of model checkpoint and training data. 

---
# Evaluating Counterfactual Strategic Reasoning in Large Language Models 

**Authors**: Dimitrios Georgousis, Maria Lymperaiou, Angeliki Dimitriou, Giorgos Filandrianos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2603.19167)  

**Abstract**: We evaluate Large Language Models (LLMs) in repeated game-theoretic settings to assess whether strategic performance reflects genuine reasoning or reliance on memorized patterns. We consider two canonical games, Prisoner's Dilemma (PD) and Rock-Paper-Scissors (RPS), upon which we introduce counterfactual variants that alter payoff structures and action labels, breaking familiar symmetries and dominance relations. Our multi-metric evaluation framework compares default and counterfactual instantiations, showcasing LLM limitations in incentive sensitivity, structural generalization and strategic reasoning within counterfactual environments. 

---
# VEPO: Variable Entropy Policy Optimization for Low-Resource Language Foundation Models 

**Authors**: Chonghan Liu, Yimin Du, Qi An, Xin He, Cunqi Zhai, Fei Tan, Weijia Lin, Xiaochun Gong, Yongchao Deng, Shousheng Jia, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19152)  

**Abstract**: Large language models frequently exhibit suboptimal performance on low resource languages, primarily due to inefficient subword segmentation and systemic training data imbalances. In this paper, we propose Variable Entropy Policy Optimization (VEPO), which leverages Reinforcement Learning with Verifiable Rewards to incorporate deterministic structural constraints into the policy alignment process. This framework ensures prescribed sequence length, robust format consistency, and rigorous linguistic well formedness, all enforced during training. Central to our approach is a variable entropy mechanism that enables the model to dynamically calibrate the equilibrium between literal fidelity and semantic naturalness by modulating the exploration exploitation manifold. By integrating entropy tempered advantage estimation with asymmetric clipping, VEPO sustains robust exploration while mitigating policy collapse. Empirical evaluations across 90 FLORES-200, COMET-22, chrF directions demonstrate that VEPO yields substantial improvements in both tokenization efficiency and translation quality, bridging the performance gap for underrepresented languages. 

---
# Optimal Splitting of Language Models from Mixtures to Specialized Domains 

**Authors**: Skyler Seto, Pierre Ablin, Anastasiia Filippova, Jiayuan Ye, Louis Bethune, Angelos Katharopoulos, David Grangier  

**Link**: [PDF](https://arxiv.org/pdf/2603.19149)  

**Abstract**: Language models achieve impressive performance on a variety of knowledge, language, and reasoning tasks due to the scale and diversity of pretraining data available. The standard training recipe is a two-stage paradigm: pretraining first on the full corpus of data followed by specialization on a subset of high quality, specialized data from the full corpus. In the multi-domain setting, this involves continued pretraining of multiple models on each specialized domain, referred to as split model training. We propose a method for pretraining multiple models independently over a general pretraining corpus, and determining the optimal compute allocation between pretraining and continued pretraining using scaling laws. Our approach accurately predicts the loss of a model of size N with D pretraining and D' specialization tokens, and extrapolates to larger model sizes and number of tokens. Applied to language model training, our approach improves performance consistently across common sense knowledge and reasoning benchmarks across different model sizes and compute budgets. 

---
# UGID: Unified Graph Isomorphism for Debiasing Large Language Models 

**Authors**: Zikang Ding, Junchi Yao, Junhao Li, Yi Zhang, Wenbo Jiang, Hongbo Liu, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19144)  

**Abstract**: Large language models (LLMs) exhibit pronounced social biases. Output-level or data-optimization--based debiasing methods cannot fully resolve these biases, and many prior works have shown that biases are embedded in internal representations. We propose \underline{U}nified \underline{G}raph \underline{I}somorphism for \underline{D}ebiasing large language models (\textit{\textbf{UGID}}), an internal-representation--level debiasing framework for large language models that models the Transformer as a structured computational graph, where attention mechanisms define the routing edges of the graph and hidden states define the graph nodes. Specifically, debiasing is formulated as enforcing invariance of the graph structure across counterfactual inputs, with differences allowed only on sensitive attributes. \textit{\textbf{UGID}} jointly constrains attention routing and hidden representations in bias-sensitive regions, effectively preventing bias migration across architectural components. To achieve effective behavioral alignment without degrading general capabilities, we introduce a log-space constraint on sensitive logits and a selective anchor-based objective to preserve definitional semantics. Extensive experiments on large language models demonstrate that \textit{\textbf{UGID}} effectively reduces bias under both in-distribution and out-of-distribution settings, significantly reduces internal structural discrepancies, and preserves model safety and utility. 

---
# DaPT: A Dual-Path Framework for Multilingual Multi-hop Question Answering 

**Authors**: Yilin Wang, Yuchun Fan, Jiaoyang Li, Ziming Zhu, Yongyu Mu, Qiaozhi He, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19097)  

**Abstract**: Retrieval-augmented generation (RAG) systems have made significant progress in solving complex multi-hop question answering (QA) tasks in the English scenario. However, RAG systems inevitably face the application scenario of retrieving across multilingual corpora and queries, leaving several open challenges. The first one involves the absence of benchmarks that assess RAG systems' capabilities under the multilingual multi-hop (MM-hop) QA setting. The second centers on the overreliance on LLMs' strong semantic understanding in English, which diminishes effectiveness in multilingual scenarios. To address these challenges, we first construct multilingual multi-hop QA benchmarks by translating English-only benchmarks into five languages, and then we propose DaPT, a novel multilingual RAG framework. DaPT generates sub-question graphs in parallel for both the source-language query and its English translation counterpart, then merges them before employing a bilingual retrieval-and-answer strategy to sequentially solve sub-questions. Our experimental results demonstrate that advanced RAG systems suffer from a significant performance imbalance in multilingual scenarios. Furthermore, our proposed method consistently yields more accurate and concise answers compared to the baselines, significantly enhancing RAG performance on this task. For instance, on the most challenging MuSiQue benchmark, DaPT achieves a relative improvement of 18.3\% in average EM score over the strongest baseline. 

---
# A Dataset and Resources for Identifying Patient Health Literacy Information from Clinical Notes 

**Authors**: Madeline Bittner, Dina Demner-Fushman, Yasmeen Shabazz, Davis Bartels, Dukyong Yoon, Brad Quitadamo, Rajiv Menghrajani, Leo Celi, Sarvesh Soni  

**Link**: [PDF](https://arxiv.org/pdf/2603.19082)  

**Abstract**: Health literacy is a critical determinant of patient outcomes, yet current screening tools are not always feasible and differ considerably in the number of items, question format, and dimensions of health literacy they capture, making documentation in structured electronic health records difficult to achieve. Automated detection from unstructured clinical notes offers a promising alternative, as these notes often contain richer, more contextual health literacy information, but progress has been limited by the lack of annotated resources. We introduce HEALIX, the first publicly available annotated health literacy dataset derived from real clinical notes, curated through a combination of social worker note sampling, keyword-based filtering, and LLM-based active learning. HEALIX contains 589 notes across 9 note types, annotated with three health literacy labels: low, normal, and high. To demonstrate its utility, we benchmarked zero-shot and few-shot prompting strategies across four open source large language models (LLMs). 

---
# Parallelograms Strike Back: LLMs Generate Better Analogies than People 

**Authors**: Qiawen Ella Liu, Raja Marjieh, Jian-Qiao Zhu, Adele E. Goldberg, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2603.19066)  

**Abstract**: Four-term word analogies (A:B::C:D) are classically modeled geometrically as ''parallelograms,'' yet recent work suggests this model poorly captures how humans produce analogies, with simple local-similarity heuristics often providing a better account (Peterson et al., 2020). But does the parallelogram model fail because it is a bad model of analogical relations, or because people are not very good at generating relation-preserving analogies? We compared human and large language model (LLM) analogy completions on the same set of analogy problems from (Peterson et al., 2020). We find that LLM-generated analogies are reliably judged as better than human-generated ones, and are also more closely aligned with the parallelogram structure in a distributional embedding space (GloVe). Crucially, we show that the improvement over human analogies was driven by greater parallelogram alignment and reduced reliance on accessible words rather than enhanced sensitivity to local similarity. Moreover, the LLM advantage is driven not by uniformly superior responses by LLMs, but by humans producing a long tail of weak completions: when only modal (most frequent) responses by both systems are compared, the LLM advantage disappears. However, greater parallelogram alignment and lower word frequency continue to predict which LLM completions are rated higher than those of humans. Overall, these results suggest that the parallelogram model is not a poor account of word analogy. Rather, humans may often fail to produce completions that satisfy this relational constraint, whereas LLMs do so more consistently. 

---
# MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models 

**Authors**: Chenyang Gu, Jiahao Cheng, Meicong Zhang, Pujun Zheng, Jinquan Zheng, Guoxiu He  

**Link**: [PDF](https://arxiv.org/pdf/2603.19044)  

**Abstract**: Scientific ideation aims to propose novel solutions within a given scientific context. Existing LLM-based agentic approaches emulate human research workflows, yet inadequately model scientific reasoning, resulting in surface-level conceptual recombinations that lack technical depth and scientific grounding. To address this issue, we propose \textbf{MoRI} (\textbf{Mo}tivation-grounded \textbf{R}easoning for Scientific \textbf{I}deation), a framework that enables LLMs to explicitly learn the reasoning process from research motivations to methodologies. The base LLM is initialized via supervised fine-tuning to generate a research motivation from a given context, and is subsequently trained under a composite reinforcement learning reward that approximates scientific rigor: (1) entropy-aware information gain encourages the model to uncover and elaborate high-complexity technical details grounded in ground-truth methodologies, and (2) contrastive semantic gain constrains the reasoning trajectory to maintain conceptually aligned with scientifically valid solutions. Empirical results show that MoRI significantly outperforms strong commercial LLMs and complex agentic baselines across multiple dimensions, including novelty, technical rigor, and feasibility. The code will be made available on \href{this https URL}{GitHub}. 

---
# What Really Controls Temporal Reasoning in Large Language Models: Tokenisation or Representation of Time? 

**Authors**: Gagan Bhatia, Ahmad Muhammad Isa, Maxime Peyrard, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.19017)  

**Abstract**: We present MultiTempBench, a multilingual temporal reasoning benchmark spanning three tasks, date arithmetic, time zone conversion, and temporal relation extraction across five languages (English, German, Chinese, Arabic, and Hausa) and multiple calendar conventions (Gregorian, Hijri, and Chinese Lunar). MultiTempBench contains $15,000$ examples built by translating $750$ curated English questions and expanding each into controlled date-format variants. We evaluate 20 LLMs and introduce the multilingual Date Fragmentation Ratio (mDFR), calibrated with human severity ratings, together with geometric-probing analyses of internal temporal representations. We find tokenisation quality of temporal artefacts is a resource-dependent bottleneck: in low-resource languages and rarer calendar formats, fragmentation disrupts Year/Month/Day separation and accuracy collapses, while high-resource settings are often robust to digit-level splitting. Beyond tokenisation, crossed mixed-effects regression shows that temporal linearity is the strongest predictor of temporal reasoning in high-resource languages, whereas fragmentation is the stronger predictor in low-resource languages. Code is available at: this https URL 

---
# Hypothesis-Conditioned Query Rewriting for Decision-Useful Retrieval 

**Authors**: Hangeol Chang, Changsun Lee, Seungjoon Rho, Junho Yeo, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.19008)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves Large Language Models (LLMs) by grounding generation in external, non-parametric knowledge. However, when a task requires choosing among competing options, simply grounding generation in broadly relevant context is often insufficient to drive the final decision. Existing RAG methods typically rely on a single initial query, which often favors topical relevance over decision-relevant evidence, and therefore retrieves background information that can fail to discriminate among answer options. To address this issue, here we propose Hypothesis-Conditioned Query Rewriting (HCQR), a training-free pre-retrieval framework that reorients RAG from topic-oriented retrieval to evidence-oriented retrieval. HCQR first derives a lightweight working hypothesis from the input question and candidate options, and then rewrites retrieval into three targeted queries that seek evidence to: (1) support the hypothesis, (2) distinguish it from competing alternatives, and (3) verify salient clues in the question. This approach enables context retrieval that is more directly aligned with answer selection, allowing the generator to confirm or overturn the initial hypothesis based on the retrieved evidence. Experiments on MedQA and MMLU-Med show that HCQR consistently outperforms single-query RAG and re-rank/filter baselines, improving average accuracy over Simple RAG by 5.9 and 3.6 points, respectively. Code is available at this https URL. 

---
# RADIUS: Ranking, Distribution, and Significance - A Comprehensive Alignment Suite for Survey Simulation 

**Authors**: Weronika Łajewska, Paul Missault, George Davidson, Saab Mansour  

**Link**: [PDF](https://arxiv.org/pdf/2603.19002)  

**Abstract**: Simulation of surveys using LLMs is emerging as a powerful application for generating human-like responses at scale. Prior work evaluates survey simulation using metrics borrowed from other domains, which are often ad hoc, fragmented, and non-standardized, leading to results that are difficult to compare. Moreover, existing metrics focus mainly on accuracy or distributional measures, overlooking the critical dimension of ranking alignment. In practice, a simulation can achieve high accuracy while still failing to capture the option most preferred by humans - a distinction that is critical in decision-making applications. We introduce RADIUS, a comprehensive two-dimensional alignment suite for survey simulation that captures: 1) RAnking alignment and 2) DIstribUtion alignment, each complemented by statistical Significance testing. RADIUS highlights the limitations of existing metrics, enables more meaningful evaluation of survey simulation, and provides an open-source implementation for reproducible and comparable assessment. 

---
# Entropy trajectory shape predicts LLM reasoning reliability: A diagnostic study of uncertainty dynamics in chain-of-thought 

**Authors**: Xinghao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.18940)  

**Abstract**: Chain-of-thought (CoT) reasoning improves LLM accuracy, yet detecting failures cheaply remains elusive. We study whether the shape of uncertainty dynamics across reasoning steps--captured by sampling a few answer completions per step--predicts correctness.
We introduce entropy-trajectory monotonicity: a chain is monotone if its per-step answer-distribution entropy decreases at every step. On GSM8K (n=300) with Qwen2.5-7B-Instruct, monotone chains achieve 68.8% accuracy vs. 46.8% for non-monotone chains (+21.9 pp; Fisher's p=0.0005; OR=2.50). Critically, total entropy reduction is not predictive ($\rho$=-0.06, p=0.31), revealing a shape-over-magnitude dissociation: whether entropy decreases at every step matters, not how much. Violation count 0/1/2 gives 68.8%/50.8%/28.6% accuracy.
Token log-probability confidence worsens in calibration with step depth (ECE: 0.186->0.312), and monotonicity achieves +5.8 pp at 73.7% coverage, outperforming scalar baselines at approx 1,500 tokens/question--1/8 the cost of 40-chain self-consistency. Results replicate on Mistral-7B (n=300): monotone chains reach 72.3% vs. 37.6% (+34.7 pp; OR=4.33). Structural properties of uncertainty trajectories are thus more informative than aggregate measures. 

---
# Progressive Training for Explainable Citation-Grounded Dialogue: Reducing Hallucination to Zero in English-Hindi LLMs 

**Authors**: Vedant Pandya  

**Link**: [PDF](https://arxiv.org/pdf/2603.18911)  

**Abstract**: Knowledge-grounded dialogue systems aim to generate informative, contextually relevant responses by conditioning on external knowledge sources. However, most existing approaches focus exclusively on English, lack explicit citation mechanisms for verifying factual claims, and offer limited transparency into model decision-making. We present XKD-Dial, a progressive four-stage training pipeline for explainable, knowledge-grounded dialogue generation in a bilingual (English-Hindi) setting, comprising: (1) multilingual adaptation, (2) English dialogue SFT with citation grounding, (3) bilingual dialogue SFT, and (4) GRPO alignment with citation-aware rewards. We evaluate six models spanning encoder-decoder (250M-3B) and decoder-only (1B-7B) architectures at every pipeline stage. Our key contributions are: (i) three post-hoc explainability analyses - cross-attention alignment, Integrated Gradients attribution, and occlusion-based causal grounding - applied systematically across the training trajectory to reveal how citation behaviour is learned, not only whether it is learned; (ii) citation-grounded SFT reduces hallucination to 0.0% for encoder-decoder models from Stage 2 onward; (iii) the progressive pipeline prevents catastrophic forgetting while improving Hindi capabilities; (iv) smaller models match larger models on English after SFT; and (v) GRPO provides marginal improvement over well-designed SFT for structured citation tasks. We evaluate across six automatic metrics (BLEU, ROUGE, BERTScore, FactScore, Citation-F1, and hallucination rate). 

---
# A Human-in/on-the-Loop Framework for Accessible Text Generation 

**Authors**: Lourdes Moreno, Paloma Martínez  

**Link**: [PDF](https://arxiv.org/pdf/2603.18879)  

**Abstract**: Plain Language and Easy-to-Read formats in text simplification are essential for cognitive accessibility. Yet current automatic simplification and evaluation pipelines remain largely automated, metric-driven, and fail to reflect user comprehension or normative standards. This paper introduces a hybrid framework that explicitly integrates human participation into LLM-based accessible text generation. Human-in-the-Loop (HiTL) contributions guide adjustments during generation, while Human-on-the-Loop (HoTL) supervision ensures systematic post-generation review. Empirical evidence from user studies and annotated resources is operationalized into (i) checklists aligned with standards, (ii) Event-Condition-Action trigger rules for activating expert oversight, and (iii) accessibility Key Performance Indicators (KPIs). The framework shows how human-centered mechanisms can be encoded for evaluation and reused to provide structured feedback that improves model adaptation. By embedding the human role in both generation and supervision, it establishes a traceable, reproducible, and auditable process for creating and evaluating accessible texts. In doing so, it integrates explainability and ethical accountability as core design principles, contributing to more transparent and inclusive NLP systems. 

---
# Evaluating LLM-Generated Lessons from the Language Learning Students' Perspective: A Short Case Study on Duolingo 

**Authors**: Carlos Rafael Catalan, Patricia Nicole Monderin, Lheane Marie Dizon, Gap Estrella, Raymund John Sarmimento, Marie Antoinette Patalagsa  

**Link**: [PDF](https://arxiv.org/pdf/2603.18873)  

**Abstract**: Popular language learning applications such as Duolingo use large language models (LLMs) to generate lessons for its users. Most lessons focus on general real-world scenarios such as greetings, ordering food, or asking directions, with limited support for profession-specific contexts. This gap can hinder learners from achieving professional-level fluency, which we define as the ability to communicate comfortably various work-related and domain-specific information in the target language. We surveyed five employees from a multinational company in the Philippines on their experiences with Duolingo. Results show that respondents encountered general scenarios more frequently than work-related ones, and that the former are relatable and effective in building foundational grammar, vocabulary, and cultural knowledge. The latter helps bridge the gap toward professional fluency as it contains domain-specific vocabulary. Each participant suggested lesson scenarios that diverge in contexts hen analyzed in aggregate. With this understanding, we propose that language learning applications should generate lessons that adapt to an individual's needs through personalized, domain specific lesson scenarios while maintaining foundational support through general, relatable lesson scenarios. 

---
# Why Better Cross-Lingual Alignment Fails for Better Cross-Lingual Transfer: Case of Encoders 

**Authors**: Yana Veitsman, Yihong Liu, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2603.18863)  

**Abstract**: Better cross-lingual alignment is often assumed to yield better cross-lingual transfer. However, explicit alignment techniques -- despite increasing embedding similarity -- frequently fail to improve token-level downstream performance. In this work, we show that this mismatch arises because alignment and downstream task objectives are largely orthogonal, and because the downstream benefits from alignment vary substantially across languages and task types. We analyze four XLM-R encoder models aligned on different language pairs and fine-tuned for either POS Tagging or Sentence Classification. Using representational analyses, including embedding distances, gradient similarities, and gradient magnitudes for both task and alignment losses, we find that: (1) embedding distances alone are unreliable predictors of improvements (or degradations) in task performance and (2) alignment and task gradients are often close to orthogonal, indicating that optimizing one objective may contribute little to optimizing the other. Taken together, our findings explain why ``better'' alignment often fails to translate into ``better'' cross-lingual transfer. Based on these insights, we provide practical guidelines for combining cross-lingual alignment with task-specific fine-tuning, highlighting the importance of careful loss selection. 

---
# Detecting Basic Values in A Noisy Russian Social Media Text Data: A Multi-Stage Classification Framework 

**Authors**: Maria Milkova, Maksim Rudnev  

**Link**: [PDF](https://arxiv.org/pdf/2603.18822)  

**Abstract**: This study presents a multi-stage classification framework for detecting human values in noisy Russian language social media, validated on a random sample of 7.5 million public text posts. Drawing on Schwartz's theory of basic human values, we design a multi-stage pipeline that includes spam and nonpersonal content filtering, targeted selection of value relevant and politically relevant posts, LLM based annotation, and multi-label classification. Particular attention is given to verifying the quality of LLM annotations and model predictions against human experts. We treat human expert annotations not as ground truth but as an interpretative benchmark with its own uncertainty. To account for annotation subjectivity, we aggregate multiple LLM generated judgments into soft labels that reflect varying levels of agreement. These labels are then used to train transformer based models capable of predicting the probability of each of the ten basic values. The best performing model, XLM RoBERTa large, achieves an F1 macro of 0.83 and an F1 of 0.71 on held out test data. By treating value detection as a multi perspective interpretive task, where expert labels, GPT annotations, and model predictions represent coherent but not identical readings of the same texts, we show that the model generally aligns with human judgments but systematically overestimates the Openness to Change value domain. Empirically, the study reveals distinct patterns of value expression and their co-occurrence in Russian social networks, contributing to a broader research agenda on cultural variation, communicative framing, and value based interpretation in digital environments. All models are released publicly. 

---
# Mi:dm K 2.5 Pro 

**Authors**: KT Tech innovation Group  

**Link**: [PDF](https://arxiv.org/pdf/2603.18788)  

**Abstract**: The evolving LLM landscape requires capabilities beyond simple text generation, prioritizing multi-step reasoning, long-context understanding, and agentic workflows. This shift challenges existing models in enterprise environments, especially in Korean-language and domain-specific scenarios where scaling is insufficient. We introduce Mi:dm K 2.5 Pro, a 32B parameter flagship LLM designed to address enterprise-grade complexity through reasoning-focused optimization.
Our methodology builds a robust data foundation via a quality-centric curation pipeline utilizing abstract syntax tree (AST) analysis for code, gap-filling synthesis for mathematics, and an LLM-based quality evaluator. Pre-training scales the model via layer-predictor-based Depth Upscaling (DuS) and a progressive strategy supporting a 128K token context window. Post-training introduces a specialized multi-stage pipeline, including Reasoning SFT, model merging, and asynchronous reinforcement learning (RL), to develop complex problem-solving skills. "Fusion Training" then rebalances these capabilities with conversational fluency, consistent response styling, and reliable tool-use.
The evaluations show that Mi:dm K 2.5 Pro achieves competitive performance against leading global and domestic models. In addition, it sets state-of-the-art results on Korean-specific benchmarks, showcasing deep linguistic and cultural understanding. Finally, Responsible AI evaluations validate safety against attacks, ensuring a secure profile for deployment with a balance of harmlessness and responsiveness. 

---
# Implicit Grading Bias in Large Language Models: How Writing Style Affects Automated Assessment Across Math, Programming, and Essay Tasks 

**Authors**: Rudra Jadhav, Janhavi Danve, Sonalika Shaw  

**Link**: [PDF](https://arxiv.org/pdf/2603.18765)  

**Abstract**: As large language models (LLMs) are increasingly deployed as automated graders in educational settings, concerns about fairness and bias in their evaluations have become critical. This study investigates whether LLMs exhibit implicit grading bias based on writing style when the underlying content correctness remains constant. We constructed a controlled dataset of 180 student responses across three subjects (Mathematics, Programming, and Essay/Writing), each with three surface-level perturbation types: grammar errors, informal language, and non-native phrasing. Two state-of-the-art open-source LLMs -- LLaMA 3.3 70B (Meta) and Qwen 2.5 72B (Alibaba) -- were prompted to grade responses on a 1-10 scale with explicit instructions to evaluate content correctness only and to disregard writing style. Our results reveal statistically significant grading bias in Essay/Writing tasks across both models and all perturbation types (p < 0.05), with effect sizes ranging from medium (Cohen's d = 0.64) to very large (d = 4.25). Informal language received the heaviest penalty, with LLaMA deducting an average of 1.90 points and Qwen deducting 1.20 points on a 10-point scale -- penalties comparable to the difference between a B+ and C+ letter grade. Non-native phrasing was penalized 1.35 and 0.90 points respectively. In sharp contrast, Mathematics and Programming tasks showed minimal bias, with most conditions failing to reach statistical significance. These findings demonstrate that LLM grading bias is subject-dependent, style-sensitive, and persists despite explicit counter-bias instructions in the grading prompt. We discuss implications for equitable deployment of LLM-based grading systems and recommend bias auditing protocols before institutional adoption. 

---
# Automatic detection of Gen-AI texts: A comparative framework of neural models 

**Authors**: Cristian Buttaro, Irene Amerini  

**Link**: [PDF](https://arxiv.org/pdf/2603.18750)  

**Abstract**: The rapid proliferation of Large Language Models has significantly increased the difficulty of distinguishing between human-written and AI generated texts, raising critical issues across academic, editorial, and social domains. This paper investigates the problem of AI generated text detection through the design, implementation, and comparative evaluation of multiple machine learning based detectors. Four neural architectures are developed and analyzed: a Multilayer Perceptron, a one-dimensional Convolutional Neural Network, a MobileNet-based CNN, and a Transformer model. The proposed models are benchmarked against widely used online detectors, including ZeroGPT, GPTZero, QuillBot, this http URL, Sapling, IsGen, Rephrase, and Writer. Experiments are conducted on the COLING Multilingual Dataset, considering both English and Italian configurations, as well as on an original thematic dataset focused on Art and Mental Health. Results show that supervised detectors achieve more stable and robust performance than commercial tools across different languages and domains, highlighting key strengths and limitations of current detection strategies. 

---
# A Comparative Empirical Study of Catastrophic Forgetting Mitigation in Sequential Task Adaptation for Continual Natural Language Processing Systems 

**Authors**: Aram Abrahamyan, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2603.18641)  

**Abstract**: Neural language models deployed in real-world applications must continually adapt to new tasks and domains without forgetting previously acquired knowledge. This work presents a comparative empirical study of catastrophic forgetting mitigation in continual intent classification. Using the CLINC150 dataset, we construct a 10-task label-disjoint scenario and evaluate three backbone architectures: a feed-forward Artificial Neural Network (ANN), a Gated Recurrent Unit (GRU), and a Transformer encoder, under a range of continual learning (CL) strategies. We consider one representative method from each major CL family: replay-based Maximally Interfered Retrieval (MIR), regularization-based Learning without Forgetting (LwF), and parameter-isolation via Hard Attention to Task (HAT), both individually and in all pairwise and triple combinations. Performance is assessed with average accuracy, macro F1, and backward transfer, capturing the stability-plasticity trade-off across the task sequence. Our results show that naive sequential fine-tuning suffers from severe forgetting for all architectures and that no single CL method fully prevents it. Replay emerges as a key ingredient: MIR is the most reliable individual strategy, and combinations that include replay (MIR+HAT, MIR+LwF, MIR+LwF+HAT) consistently achieve high final performance with near-zero or mildly positive backward transfer. The optimal configuration is architecture-dependent. MIR+HAT yields the best result for ANN and Transformer, MIR+LwF+HAT, on the other hand, works the best for GRU, and in several cases CL methods even surpass joint training, indicating a regularization effect. These findings highlight the importance of jointly selecting backbone architecture and CL mechanism when designing continual intent-classification systems. 

---
# Learning to Self-Evolve 

**Authors**: Xiaoyin Chen, Canwen Xu, Yite Wang, Boyi Liu, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2603.18620)  

**Abstract**: We introduce Learning to Self-Evolve (LSE), a reinforcement learning framework that trains large language models (LLMs) to improve their own contexts at test time. We situate LSE in the setting of test-time self-evolution, where a model iteratively refines its context from feedback on seen problems to perform better on new ones. Existing approaches rely entirely on the inherent reasoning ability of the model and never explicitly train it for this task. LSE reduces the multi-step evolution problem to a single-step RL objective, where each context edit is rewarded by the improvement in downstream performance. We pair this objective with a tree-guided evolution loop. On Text-to-SQL generation (BIRD) and general question answering (MMLU-Redux), a 4B-parameter model trained with LSE outperforms self-evolving policies powered by GPT-5 and Claude Sonnet 4.5, as well as prompt optimization methods including GEPA and TextGrad, and transfers to guide other models without additional training. Our results highlight the effectiveness of treating self-evolution as a learnable skill. 

---
# DiscoPhon: Benchmarking the Unsupervised Discovery of Phoneme Inventories With Discrete Speech Units 

**Authors**: Maxime Poli, Manel Khentout, Angelo Ortiz Tandazo, Ewan Dunbar, Emmanuel Chemla, Emmanuel Dupoux  

**Link**: [PDF](https://arxiv.org/pdf/2603.18612)  

**Abstract**: We introduce DiscoPhon, a multilingual benchmark for evaluating unsupervised phoneme discovery from discrete speech units. DiscoPhon covers 6 dev and 6 test languages, chosen to span a wide range of phonemic contrasts. Given only 10 hours of speech in a previously unseen language, systems must produce discrete units that are mapped to a predefined phoneme inventory, through either a many-to-one or a one-to-one assignment. The resulting sequences are evaluated for unit quality, recognition and segmentation. We provide four pretrained multilingual HuBERT and SpidR baselines, and show that phonemic information is available enough in current models for derived units to correlate well with phonemes, though with variations across languages. 

---
# Cross-Modal Rationale Transfer for Explainable Humanitarian Classification on Social Media 

**Authors**: Thi Huyen Nguyen, Koustav Rudra, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2603.18611)  

**Abstract**: Advances in social media data dissemination enable the provision of real-time information during a crisis. The information comes from different classes, such as infrastructure damages, persons missing or stranded in the affected zone, etc. Existing methods attempted to classify text and images into various humanitarian categories, but their decision-making process remains largely opaque, which affects their deployment in real-life applications. Recent work has sought to improve transparency by extracting textual rationales from tweets to explain predicted classes. However, such explainable classification methods have mostly focused on text, rather than crisis-related images. In this paper, we propose an interpretable-by-design multimodal classification framework. Our method first learns the joint representation of text and image using a visual language transformer model and extracts text rationales. Next, it extracts the image rationales via the mapping with text rationales. Our approach demonstrates how to learn rationales in one modality from another through cross-modal rationale transfer, which saves annotation effort. Finally, tweets are classified based on extracted rationales. Experiments are conducted over CrisisMMD benchmark dataset, and results show that our proposed method boosts the classification Macro-F1 by 2-35% while extracting accurate text tokens and image patches as rationales. Human evaluation also supports the claim that our proposed method is able to retrieve better image rationale patches (12%) that help to identify humanitarian classes. Our method adapts well to new, unseen datasets in zero-shot mode, achieving an accuracy of 80%. 

---
# Language Model Maps for Prompt-Response Distributions via Log-Likelihood Vectors 

**Authors**: Yusuke Takase, Momose Oyama, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2603.18593)  

**Abstract**: We propose a method that represents language models by log-likelihood vectors over prompt-response pairs and constructs model maps for comparing their conditional distributions. In this space, distances between models approximate the KL divergence between the corresponding conditional distributions. Experiments on a large collection of publicly available language models show that the maps capture meaningful global structure, including relationships to model attributes and task performance. The method also captures systematic shifts induced by prompt modifications and their approximate additive compositionality, suggesting a way to analyze and predict the effects of composite prompt operations. We further introduce pointwise mutual information (PMI) vectors to reduce the influence of unconditional distributions; in some cases, PMI-based model maps better reflect training-data-related differences. Overall, the framework supports the analysis of input-dependent model behavior. 

---
# ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs 

**Authors**: Abhinaba Basu, Pavan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2603.18579)  

**Abstract**: Evaluating whether explanations faithfully reflect a model's reasoning remains an open problem. Existing benchmarks use single interventions without statistical testing, making it impossible to distinguish genuine faithfulness from chance-level performance. We introduce ICE (Intervention-Consistent Explanation), a framework that compares explanations against matched random baselines via randomization tests under multiple intervention operators, yielding win rates with confidence intervals. Evaluating 7 LLMs across 4 English tasks, 6 non-English languages, and 2 attribution methods, we find that faithfulness is operator-dependent: operator gaps reach up to 44 percentage points, with deletion typically inflating estimates on short text but the pattern reversing on long text, suggesting that faithfulness should be interpreted comparatively across intervention operators rather than as a single score. Randomized baselines reveal anti-faithfulness in one-third of configurations, and faithfulness shows zero correlation with human plausibility (|r| < 0.04). Multilingual evaluation reveals dramatic model-language interactions not explained by tokenization alone. We release the ICE framework and ICEBench benchmark. 

---
# Cross-Lingual LLM-Judge Transfer via Evaluation Decomposition 

**Authors**: Ivaxi Sheth, Zeno Jonke, Amin Mantrach, Saab Mansour  

**Link**: [PDF](https://arxiv.org/pdf/2603.18557)  

**Abstract**: As large language models are increasingly deployed across diverse real-world applications, extending automated evaluation beyond English has become a critical challenge. Existing evaluation approaches are predominantly English-focused, and adapting them to other languages is hindered by the scarcity and cost of human-annotated judgments in most languages. We introduce a decomposition-based evaluation framework built around a Universal Criteria Set (UCS). UCS consists of a shared, language-agnostic set of evaluation dimensions, producing an interpretable intermediate representation that supports cross-lingual transfer with minimal supervision. Experiments on multiple faithfulness tasks across languages and model backbones demonstrate consistent improvements over strong baselines without requiring target-language annotations. 

---
# When Names Change Verdicts: Intervention Consistency Reveals Systematic Bias in LLM Decision-Making 

**Authors**: Abhinaba Basu, Pavan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2603.18530)  

**Abstract**: Large language models (LLMs) are increasingly used for high-stakes decisions, yet their susceptibility to spurious features remains poorly characterized. We introduce ICE-Guard, a framework applying intervention consistency testing to detect three types of spurious feature reliance: demographic (name/race swaps), authority (credential/prestige swaps), and framing (positive/negative restatements). Across 3,000 vignettes spanning 10 high-stakes domains, we evaluate 11 LLMs from 8 families and find that (1) authority bias (mean 5.8%) and framing bias (5.0%) substantially exceed demographic bias (2.2%), challenging the field's narrow focus on demographics; (2) bias concentrates in specific domains -- finance shows 22.6% authority bias while criminal justice shows only 2.8%; (3) structured decomposition, where the LLM extracts features and a deterministic rubric decides, reduces flip rates by up to 100% (median 49% across 9 models). We demonstrate an ICE-guided detect-diagnose-mitigate-verify loop achieving cumulative 78% bias reduction via iterative prompt patching. Validation against real COMPAS recidivism data shows COMPAS-derived flip rates exceed pooled synthetic rates, suggesting our benchmark provides a conservative estimate of real-world bias. Code and data are publicly available. 

---
# EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models 

**Authors**: Minsoo Cheong, Donghyun Son, Woosang Lim, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2603.18489)  

**Abstract**: Diffusion-based large language models (dLLMs) rely on bidirectional attention, which prevents lossless KV caching and requires a full forward pass at every denoising step. Existing approximate KV caching methods reduce this cost by selectively updating cached states, but their decision overhead scales with context length or model depth. We propose EntropyCache, a training-free KV caching method that uses the maximum entropy of newly decoded token distributions as a constant-cost signal for deciding when to recompute. Our design is grounded in two empirical observations: (1) decoded token entropy correlates with KV cache drift, providing a cheap proxy for cache staleness, and (2) feature volatility of decoded tokens persists for multiple steps after unmasking, motivating recomputation of the $k$ most recently decoded tokens. The skip-or-recompute decision requires only $O(V)$ computation per step, independent of context length and model scale. Experiments on LLaDA-8B-Instruct and Dream-7B-Instruct show that EntropyCache achieves $15.2\times$-$26.4\times$ speedup on standard benchmarks and $22.4\times$-$24.1\times$ on chain-of-thought benchmarks, with competitive accuracy and decision overhead accounting for only $0.5\%$ of inference time. Code is available at this https URL. 

---
# The Truncation Blind Spot: How Decoding Strategies Systematically Exclude Human-Like Token Choices 

**Authors**: Esteban Garces Arias, Nurzhan Sapargali, Christian Heumann, Matthias Aßenmacher  

**Link**: [PDF](https://arxiv.org/pdf/2603.18482)  

**Abstract**: Standard decoding strategies for text generation, including top-k, nucleus sampling, and contrastive search, select tokens based on likelihood, restricting selection to high-probability regions. Human language production operates differently: tokens are chosen for communicative appropriateness rather than statistical frequency. This mismatch creates a truncation blind spot: contextually appropriate but statistically rare tokens remain accessible to humans yet unreachable by likelihood-based decoding. We hypothesize this contributes to the detectability of machine-generated text. Analyzing over 1.8 million texts across eight language models, five decoding strategies, and 53 hyperparameter configurations, we find that 8-18% of human-selected tokens fall outside typical truncation boundaries. Simple classifiers trained on predictability and lexical diversity achieve remarkable detection rates. Crucially, neither model scale nor architecture correlates strongly with detectability; truncation parameters account for most variance. Configurations achieving low detectability often produce incoherent text, indicating that evading detection and producing natural text are distinct objectives. These findings suggest detectability is enhanced by likelihood-based token selection, not merely a matter of model capability. 

---
# WASD: Locating Critical Neurons as Sufficient Conditions for Explaining and Controlling LLM Behavior 

**Authors**: Haonan Yu, Junhao Liu, Zhenyu Yan, Haoran Lin, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18474)  

**Abstract**: Precise behavioral control of large language models (LLMs) is critical for complex applications. However, existing methods often incur high training costs, lack natural language controllability, or compromise semantic coherence. To bridge this gap, we propose WASD (unWeaving Actionable Sufficient Directives), a novel framework that explains model behavior by identifying sufficient neural conditions for token generation. Our method represents candidate conditions as neuron-activation predicates and iteratively searches for a minimal set that guarantees the current output under input perturbations. Experiments on SST-2 and CounterFact with the Gemma-2-2B model demonstrate that our approach produces explanations that are more stable, accurate, and concise than conventional attribution graphs. Moreover, through a case study on controlling cross-lingual output generation, we validated the practical effectiveness of WASD in controlling model behavior. 

---
# GAIN: A Benchmark for Goal-Aligned Decision-Making of Large Language Models under Imperfect Norms 

**Authors**: Masayuki Kawarada, Kodai Watanabe, Soichiro Murakami  

**Link**: [PDF](https://arxiv.org/pdf/2603.18469)  

**Abstract**: We introduce GAIN (Goal-Aligned Decision-Making under Imperfect Norms), a benchmark designed to evaluate how large language models (LLMs) balance adherence to norms against business goals. Existing benchmarks typically focus on abstract scenarios rather than real-world business applications. Furthermore, they provide limited insights into the factors influencing LLM decision-making. This restricts their ability to measure models' adaptability to complex, real-world norm-goal conflicts. In GAIN, models receive a goal, a specific situation, a norm, and additional contextual pressures. These pressures, explicitly designed to encourage potential norm deviations, are a unique feature that differentiates GAIN from other benchmarks, enabling a systematic evaluation of the factors influencing decision-making. We define five types of pressures: Goal Alignment, Risk Aversion, Emotional/Ethical Appeal, Social/Authoritative Influence, and Personal Incentive. The benchmark comprises 1,200 scenarios across four domains: hiring, customer support, advertising and finance. Our experiments show that advanced LLMs frequently mirror human decision-making patterns. However, when Personal Incentive pressure is present, they diverge significantly, showing a strong tendency to adhere to norms rather than deviate from them. 

---
# UT-ACA: Uncertainty-Triggered Adaptive Context Allocation for Long-Context Inference 

**Authors**: Lang Zhou, Shuxuan Li, Zhuohao Li, Shi Liu, Zhilin Zhao, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.18446)  

**Abstract**: Long-context inference remains challenging for large language models due to attention dilution and out-of-distribution degradation. Context selection mitigates this limitation by attending to a subset of key-value cache entries, yet most methods allocate a fixed context budget throughout decoding despite highly non-uniform token-level contextual demands. To address this issue, we propose Uncertainty-Triggered Adaptive Context Allocation (UT-ACA), an inference-time framework that dynamically adjusts the context window based on token-wise uncertainty. UT-ACA learns an uncertainty detector that combines semantic embeddings with logit-based confidence while accounting for uncertainty accumulation across decoding steps. When insufficient evidence is indicated, UT-ACA selectively rolls back, expands the context window, and regenerates the token with additional support. Experiments show that UT-ACA substantially reduces average context usage while preserving generation quality in long-context settings. 

---
# Adaptive Decoding via Test-Time Policy Learning for Self-Improving Generation 

**Authors**: Asmita Bhardwaj, Yuya Jeremy Ong, Eelaaf Zahid, Basel Shbita  

**Link**: [PDF](https://arxiv.org/pdf/2603.18428)  

**Abstract**: Decoding strategies largely determine the quality of Large Language Model (LLM) outputs, yet widely used heuristics such as greedy or fixed temperature/top-p decoding are static and often task-agnostic, leading to suboptimal or inconsistent generation quality across domains that demand stylistic or structural flexibility. We introduce a reinforcement learning-based decoder sampler that treats decoding as sequential decision-making and learns a lightweight policy to adjust sampling parameters at test-time while keeping LLM weights frozen. We evaluated summarization datasets including BookSum, arXiv, and WikiHow using Granite-3.3-2B and Qwen-2.5-0.5B. Our policy sampler consistently outperforms greedy and static baselines, achieving relative gains of up to +88% (BookSum, Granite) and +79% (WikiHow, Qwen). Reward ablations show that overlap-only objectives underperform compared to composite rewards, while structured shaping terms (length, coverage, repetition, completeness) enable stable and sustained improvements. These findings highlight reinforcement learning as a practical mechanism for test-time adaptation in decoding, enabling domain-aware and user-controllable generation without retraining large models. 

---
# Multimodal Task Interference: A Benchmark and Analysis of History-Target Mismatch in Multimodal LLMs 

**Authors**: Masayuki Kawarada, Tatsuya Ishigaki, Hiroya Takamura  

**Link**: [PDF](https://arxiv.org/pdf/2603.18425)  

**Abstract**: Task interference, the performance degradation caused by task switches within a single conversation, has been studied exclusively in text-only settings despite the growing prevalence of multimodal dialogue systems. We introduce a benchmark for evaluating this phenomenon in multimodal LLMs, covering six tasks across text and vision with systematic variation of history-target along three axes: modality mismatch, reasoning mismatch, and answer format mismatch. Experiments on both open-weights and proprietary models reveal that task interference is highly directional: switching from text-only to image-based targets causes severe performance drops, while the reverse transition yields minimal degradation. Interference is further amplified when mismatches co-occur across multiple dimensions, and is driven most strongly by modality differences, followed by answer format, while reasoning requirement shifts cause minimal degradation. 

---
# TARo: Token-level Adaptive Routing for LLM Test-time Alignment 

**Authors**: Arushi Rai, Qiang Zhang, Hanqing Zeng, Yunkai Zhang, Dipesh Tamboli, Xiangjun Fan, Zhuokai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.18411)  

**Abstract**: Large language models (LLMs) exhibit strong reasoning capabilities but typically require expensive post-training to reach high performance. Recent test-time alignment methods offer a lightweight alternative, but have been explored mainly for preference alignment rather than reasoning. To bridge this gap, we propose, Token-level Adaptive Routing (TARo), which steers frozen LLMs toward structured reasoning entirely at inference time. Specifically, we first train reward models on step-wise mathematical traces to capture fine-grained logical consistency signals, then introduce a learnable token-level router that automatically controls the guidance of the reward model to the base model. Extensive experiments show that TARo significantly improves reasoning performance by up to +22.4% over base model and +8.4% over existing token-level test-time alignment methods, while also boosting out-of-distribution clinical reasoning (MedXpertQA) and instruction following (AlpacaEval). Furthermore, TARo also generalizes from small to large backbones without retraining, extending test-time alignment from preference optimization to robust, cross-domain reasoning. 

---
# TopoChunker: Topology-Aware Agentic Document Chunking Framework 

**Authors**: Xiaoyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.18409)  

**Abstract**: Current document chunking methods for Retrieval-Augmented Generation (RAG) typically linearize text. This forced linearization strips away intrinsic topological hierarchies, creating ``semantic fragmentation'' that degrades downstream retrieval quality. In this paper, we propose TopoChunker, an agentic framework that maps heterogeneous documents onto a Structured Intermediate Representation (SIR) to explicitly preserve cross-segment dependencies. To balance structural fidelity with computational cost, TopoChunker employs a dual-agent architecture. An Inspector Agent dynamically routes documents through cost-optimized extraction paths, while a Refiner Agent performs capacity auditing and topological context disambiguation to reconstruct hierarchical lineage. Evaluated on unstructured narratives (GutenQA) and complex reports (GovReport), TopoChunker demonstrates state-of-the-art performance. It outperforms the strongest LLM-based baseline by 8.0% in absolute generation accuracy and achieves an 83.26% Recall@3, while simultaneously reducing token overhead by 23.5%, offering a scalable approach for structure-aware RAG. 

---
# AutoScreen-FW: An LLM-based Framework for Resume Screening 

**Authors**: Zhelin Xu, Shuhei Yamamoto, Atsuyuki Morishima  

**Link**: [PDF](https://arxiv.org/pdf/2603.18390)  

**Abstract**: Corporate recruiters often need to screen many resumes within a limited time, which increases their burden and may cause suitable candidates to be overlooked. To address these challenges, prior work has explored LLM-based automated resume screening. However, some methods rely on commercial LLMs, which may pose data privacy risks. Moreover, since companies typically do not make resumes with evaluation results publicly available, it remains unclear which resume samples should be used during learning to improve an LLM's judgment performance. To address these problems, we propose AutoScreen-FW, an LLM-based locally and automatically resume screening framework. AutoScreen-FW uses several methods to select a small set of representative resume samples. These samples are used for in-context learning together with a persona description and evaluation criteria, enabling open-source LLMs to act as a career advisor and evaluate unseen resumes. Experiments with multiple ground truths show that the open-source LLM judges consistently outperform GPT-5-nano. Under one ground truth setting, it also surpass GPT-5-mini. Although it is slightly weaker than GPT-5-mini under other ground-truth settings, it runs substantially faster per resume than commercial GPT models. These findings indicate the potential for deploying AutoScreen-FW locally in companies to support efficient screening while reducing recruiters' burden. 

---
# PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching 

**Authors**: Ruishuo Chen, Yu Chen, Zhuoran Li, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18363)  

**Abstract**: Unsupervised Reinforcement Learning from Internal Feedback (RLIF) has emerged as a promising paradigm for eliciting the latent capabilities of Large Language Models (LLMs) without external supervision. However, current methods rely on heuristic intrinsic rewards, which often lack a well-defined theoretical optimization target and are prone to degenerative biases. In this work, we introduce PowerFlow, a principled framework that reformulates unsupervised fine-tuning as a distribution matching problem. By casting GFlowNet as an amortized variational sampler for unnormalized densities, we propose a length-aware Trajectory-Balance objective that explicitly neutralizes the structural length biases inherent in autoregressive generation. By targeting $\alpha$-power distributions, PowerFlow enables the directional elicitation of the dual nature of LLMs: sharpening the distribution ($\alpha > 1$) to intensify logical reasoning, or flattening it ($\alpha < 1$) to unlock expressive creativity. Extensive experiments demonstrate that PowerFlow consistently outperforms existing RLIF methods, matching or even exceeding supervised GRPO. Furthermore, by mitigating over-sharpening in aligned models, our approach achieves simultaneous gains in diversity and quality, shifting the Pareto frontier in creative tasks. 

---
# Synthetic Data Generation for Training Diversified Commonsense Reasoning Models 

**Authors**: Tianhui Zhang, Bei Peng, Danushka Bollegala  

**Link**: [PDF](https://arxiv.org/pdf/2603.18361)  

**Abstract**: Conversational agents are required to respond to their users not only with high quality (i.e. commonsense bearing) responses, but also considering multiple plausible alternative scenarios, reflecting the diversity in their responses. Despite the growing need to train diverse commonsense generators, the progress of this line of work has been significantly hindered by the lack of large-scale high-quality diverse commonsense training datasets. Due to the high annotation costs, existing Generative Commonsense Reasoning (GCR) datasets are created using a small number of human annotators, covering only a narrow set of commonsense scenarios. To address this training resource gap, we propose a two-stage method to create the first-ever synthetic dataset CommonSyn for diversified (GCR). The model fine-tuned on our synthetic data jointly increase both generation diversity and quality compared with vanilla models and the model fine-tuned on human-crafted dataset across different size Large Language Models (LLMs) 

---
# From Noise to Signal: When Outliers Seed New Topics 

**Authors**: Evangelia Zve, Gauvain Bourgne, Benjamin Icard, Jean-Gabriel Ganascia  

**Link**: [PDF](https://arxiv.org/pdf/2603.18358)  

**Abstract**: Outliers in dynamic topic modeling are typically treated as noise, yet we show that some can serve as early signals of emerging topics. We introduce a temporal taxonomy of news-document trajectories that defines how documents relate to topic formation over time. It distinguishes anticipatory outliers, which precede the topics they later join, from documents that either reinforce existing topics or remain isolated. By capturing these trajectories, the taxonomy links weak-signal detection with temporal topic modeling and clarifies how individual articles anticipate, initiate, or drift within evolving clusters. We implement it in a cumulative clustering setting using document embeddings from eleven state-of-the-art language models and evaluate it retrospectively on HydroNewsFr, a French news corpus on the hydrogen economy. Inter-model agreement reveals a small, high-consensus subset of anticipatory outliers, increasing confidence in these labels. Qualitative case studies further illustrate these trajectories through concrete topic developments. 

---
# How Psychological Learning Paradigms Shaped and Constrained Artificial Intelligence 

**Authors**: Alex Anvi Eponon, Ildar Batyrshin, Christian E. Maldonado-Sifuentes, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2603.18203)  

**Abstract**: The dominant paradigms of artificial intelligence were shaped by learning theories from psychology: behaviorism inspired reinforcement learning, cognitivism gave rise to deep learning and memory-augmented architectures, and constructivism influenced curriculum learning and compositional approaches. This paper argues that each AI paradigm inherited not only the strengths but the structural limitations of the psychological theory that inspired it. Reinforcement learning cannot account for the internal structure of knowledge, deep learning compresses representations into opaque parameter spaces resistant to principled update, and current integrative approaches lack a formal account of how new understanding is constructed from existing components. The paper further examines a cross-cultural divergence in the interpretation of rote learning, arguing that the Eastern conception of memorization as a structured, multi-phase precursor to understanding offers an underexploited bridge between psychological theory and AI methodology. Drawing on the systematicity debate and critique of Aizawa of both classicism and connectionism, this paper introduces ReSynth, a trimodular framework that separates reasoning (Intellect), purpose (Identity), and knowledge (Memory) as architecturally independent components. The paper traces the genealogy from psychological paradigm to AI method, diagnoses the inherited limitations at each stage, and argues that adaptability, the central challenge of artificial general intelligence requires a representational architecture in which systematic behavior is a necessary consequence rather than an accidental property. 

---
# CWoMP: Morpheme Representation Learning for Interlinear Glossing 

**Authors**: Morris Alper, Enora Rice, Bhargav Shandilya, Alexis Palmer, Lori Levin  

**Link**: [PDF](https://arxiv.org/pdf/2603.18184)  

**Abstract**: Interlinear glossed text (IGT) is a standard notation for language documentation which is linguistically rich but laborious to produce manually. Recent automated IGT methods treat glosses as character sequences, neglecting their compositional structure. We propose CWoMP (Contrastive Word-Morpheme Pretraining), which instead treats morphemes as atomic form-meaning units with learned representations. A contrastively trained encoder aligns words-in-context with their constituent morphemes in a shared embedding space; an autoregressive decoder then generates the morpheme sequence by retrieving entries from a mutable lexicon of these embeddings. Predictions are interpretable--grounded in lexicon entries--and users can improve results at inference time by expanding the lexicon without retraining. We evaluate on diverse low-resource languages, showing that CWoMP outperforms existing methods while being significantly more efficient, with particularly strong gains in extremely low-resource settings. 

---
# GRAFITE: Generative Regression Analysis Framework for Issue Tracking and Evaluation 

**Authors**: Ja Young Lee, Mírian Silva, Mohamed Nasr, Shonda Witherspoon, Enzo Bozzani, Veronique Demers, Radha Ratnaparkhi, Hui Wu, Sara Rosenthal  

**Link**: [PDF](https://arxiv.org/pdf/2603.18173)  

**Abstract**: Large language models (LLMs) are largely motivated by their performance on popular topics and benchmarks at the time of their release. However, over time, contamination occurs due to significant exposure of benchmark data during training. This poses a risk of model performance inflation if testing is not carefully executed. To address this challenge, we present GRAFITE, a continuous LLM evaluation platform through a comprehensive system for maintaining and evaluating model issues. Our approach enables building a repository of model problems based on user feedback over time and offers a pipeline for assessing LLMs against these issues through quality assurance (QA) tests using LLM-as-a-judge. The platform enables side-by-side comparison of multiple models, facilitating regression detection across different releases. The platform is available at this https URL. The demo video is available at this http URL. 

---
# Modeling the human lexicon under temperature variations: linguistic factors, diversity and typicality in LLM word associations 

**Authors**: Maria Andueza Rodriguez, Marie Candito, Richard Huyghe  

**Link**: [PDF](https://arxiv.org/pdf/2603.18171)  

**Abstract**: Large language models (LLMs) achieve impressive results in terms of fluency in text generation, yet the nature of their linguistic knowledge - in particular the human-likeness of their internal lexicon - remains uncertain. This study compares human and LLM-generated word associations to evaluate how accurately models capture human lexical patterns. Using English cue-response pairs from the SWOW dataset and newly generated associations from three LLMs (Mistral-7B, Llama-3.1-8B, and Qwen-2.5-32B) across multiple temperature settings, we examine (i) the influence of lexical factors such as word frequency and concreteness on cue-response pairs, and (ii) the variability and typicality of LLM responses compared to human responses. Results show that all models mirror human trends for frequency and concreteness but differ in response variability and typicality. Larger models such as Qwen tend to emulate a single "prototypical" human participant, generating highly typical but minimally variable responses, while smaller models such as Mistral and Llama produce more variable yet less typical responses. Temperature settings further influence this trade-off, with higher values increasing variability but decreasing typicality. These findings highlight both the similarities and differences between human and LLM lexicons, emphasizing the need to account for model size and temperature when probing LLM lexical representations. 

---
# How LLMs Distort Our Written Language 

**Authors**: Marwa Abdulhai, Isadora White, Yanming Wan, Ibrahim Qureshi, Joel Leibo, Max Kleiman-Weiner, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2603.18161)  

**Abstract**: Large language models (LLMs) are used by over a billion people globally, most often to assist with writing. In this work, we demonstrate that LLMs not only alter the voice and tone of human writing, but also consistently alter the intended meaning. First, we conduct a human user study to understand how people actually interact with LLMs when using them for writing. Our findings reveal that extensive LLM use led to a nearly 70% increase in essays that remained neutral in answering the topic question. Significantly more heavy LLM users reported that the writing was less creative and not in their voice. Next, using a dataset of human-written essays that was collected in 2021 before the widespread release of LLMs, we study how asking an LLM to revise the essay based on the human-written feedback in the dataset induces large changes in the resulting content and meaning. We find that even when LLMs are prompted with expert feedback and asked to only make grammar edits, they still change the text in a way that significantly alters its semantic meaning. We then examine LLM-generated text in the wild, specifically focusing on the 21% of AI-generated scientific peer reviews at a recent top AI conference. We find that LLM-generated reviews place significantly less weight on clarity and significance of the research, and assign scores that, on average, are a full point this http URL findings highlight a misalignment between the perceived benefit of AI use and an implicit, consistent effect on the semantics of human writing, motivating future work on how widespread AI writing will affect our cultural and scientific institutions. 

---
# Evaluating FrameNet-Based Semantic Modeling for Gender-Based Violence Detection in Clinical Records 

**Authors**: Lívia Dutra, Arthur Lorenzi, Frederico Belcavello, Ely Matos, Marcelo Viridiano, Lorena Larré, Olívia Guaranha, Erik Santos, Sofia Reinach, Pedro de Paula, Tiago Torrent  

**Link**: [PDF](https://arxiv.org/pdf/2603.18124)  

**Abstract**: Gender-based violence (GBV) is a major public health issue, with the World Health Organization estimating that one in three women experiences physical or sexual violence by an intimate partner during her lifetime. In Brazil, although healthcare professionals are legally required to report such cases, underreporting remains significant due to difficulties in identifying abuse and limited integration between public information systems. This study investigates whether FrameNet-based semantic annotation of open-text fields in electronic medical records can support the identification of patterns of GBV. We compare the performance of an SVM classifier for GBV cases trained on (1) frame-annotated text, (2) annotated text combined with parameterized data, and (3) parameterized data alone. Quantitative and qualitative analyses show that models incorporating semantic annotation outperform categorical models, achieving over 0.3 improvement in F1 score and demonstrating that domain-specific semantic representations provide meaningful signals beyond structured demographic data. The findings support the hypothesis that semantic analysis of clinical narratives can enhance early identification strategies and support more informed public health interventions. 

---
# BenchBrowser -- Collecting Evidence for Evaluating Benchmark Validity 

**Authors**: Harshita Diddee, Gregory Yauney, Swabha Swayamdipta, Daphne Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2603.18019)  

**Abstract**: Do language model benchmarks actually measure what practitioners intend them to ? High-level metadata is too coarse to convey the granular reality of benchmarks: a "poetry" benchmark may never test for haikus, while "instruction-following" benchmarks will often test for an arbitrary mix of skills. This opacity makes verifying alignment with practitioner goals a laborious process, risking an illusion of competence even when models fail on untested facets of user interests. We introduce BenchBrowser, a retriever that surfaces evaluation items relevant to natural language use cases over 20 benchmark suites. Validated by a human study confirming high retrieval precision, BenchBrowser generates evidence to help practitioners diagnose low content validity (narrow coverage of a capability's facets) and low convergent validity (lack of stable rankings when measuring the same capability). BenchBrowser, thus, helps quantify a critical gap between practitioner intent and what benchmarks actually test. 

---
# An Agentic System for Schema Aware NL2SQL Generation 

**Authors**: David Onyango, Naseef Mansoor  

**Link**: [PDF](https://arxiv.org/pdf/2603.18018)  

**Abstract**: The natural language to SQL (NL2SQL) task plays a pivotal role in democratizing data access by enabling non-expert users to interact with relational databases through intuitive language. While recent frameworks have enhanced translation accuracy via task specialization, their reliance on Large Language Models (LLMs) raises significant concerns regarding computational overhead, data privacy, and real-world deployability in resource-constrained environments. To address these challenges, we propose a schema based agentic system that strategically employs Small Language Models (SLMs) as primary agents, complemented by a selective LLM fallback mechanism. The LLM is invoked only upon detection of errors in SLM-generated output, the proposed system significantly minimizes computational expenditure. Experimental results on the BIRD benchmark demonstrate that our system achieves an execution accuracy of 47.78% and a validation efficiency score of 51.05%, achieving over 90% cost reduction compared to LLM-centric baselines as approximately 67% of queries are resolved using local SLMs. The system achieves an average cost per query of 0.0085 compared to 0.094 for LLM-only systems, achieving near-zero operational costs for locally executed queries. [Github repository: this https URL.] 

---
# MineDraft: A Framework for Batch Parallel Speculative Decoding 

**Authors**: Zhenwei Tang, Arun Verma, Zijian Zhou, Zhaoxuan Wu, Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2603.18016)  

**Abstract**: Speculative decoding (SD) accelerates large language model inference by using a smaller draft model to propose draft tokens that are subsequently verified by a larger target model. However, the performance of standard SD is often limited by the strictly sequential execution of these drafting and verification stages. To address this, this paper proposes MineDraft, a batch parallel speculative decoding (PSD) framework designed to effectively hide drafting latency by overlapping it with verification. Our theoretical analysis shows that PSD is substantially more efficient than standard SD. MineDraft realizes the PSD through a novel batch-parallel design that maintains two batches of requests, overlapping drafting for one batch with verification for the other. Our experimental results show significant improvements of MineDraft in both throughput (up to 75%) and end-to-end latency (up to 39%) over standard SD. Furthermore, we have implemented MineDraft as a plugin for vLLM, demonstrating its practicality for production-ready inference systems. 

---
# Beyond Accuracy: An Explainability-Driven Analysis of Harmful Content Detection 

**Authors**: Trishita Dhara, Siddhesh Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2603.18015)  

**Abstract**: Although automated harmful content detection systems are frequently used to monitor online platforms, moderators and end users frequently cannot understand the logic underlying their predictions. While recent studies have focused on increasing classification accuracy, little focus has been placed on comprehending why neural models identify content as harmful, especially when it comes to borderline, contextual, and politically sensitive situations. In this work, a neural harmful content detection model trained on the Civil Comments dataset is analyzed explainability-drivenly. Two popular post-hoc explanation methods, Shapley Additive Explanations and Integrated Gradients, are used to analyze the behavior of a RoBERTa-based classifier in both correct predictions and systematic failure cases. Despite strong overall performance, with an area under the curve of 0.93 and an accuracy of 0.94, the analysis reveals limitations that are not observable from aggregate evaluation metrics alone. Integrated Gradients appear to extract more diffuse contextual attributions while Shapley Additive Explanations extract more focused attributions on explicit lexical cues. The consequent divergence in their outputs manifests in both false negatives and false positives. Qualitative case studies reveal recurring failure modes such as indirect toxicity, lexical over-attribution, or political discourse. The results suggest that explainable AI can foster human-in-the-loop moderation by exposing model uncertainty and increasing the interpretable rationale behind automated decisions. Most importantly, this work highlights the role of explainability as a transparency and diagnostic resource for online harmful content detection systems rather than as a performance-enhancing lever. 

---
# Real-Time Trustworthiness Scoring for LLM Structured Outputs and Data Extraction 

**Authors**: Hui Wen Goh, Jonas Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2603.18014)  

**Abstract**: Structured Outputs from current LLMs exhibit sporadic errors, hindering enterprise AI efforts from realizing their immense potential. We present CONSTRUCT, a method to score the trustworthiness of LLM Structured Outputs in real-time, such that lower-scoring outputs are more likely to contain errors. This reveals the best places to focus limited human review bandwidth. CONSTRUCT additionally scores the trustworthiness of each field within a LLM Structured Output, helping reviewers quickly identify which parts of the output are wrong. Our method is suitable for any LLM (including black-box LLM APIs without logprobs such as reasoning models and Anthropic models), does not require labeled training data nor custom model deployment, and works for complex Structured Outputs with many fields of diverse types (including nested JSON schemas).
We additionally present one of the first public LLM Structured Output benchmarks with reliable ground-truth values that are not full of mistakes. Over this four-dataset benchmark, CONSTRUCT detects errors from various LLMs (including Gemini 3 and GPT-5) with significantly higher precision/recall than other scoring methods. 

---
# Learned but Not Expressed: Capability-Expression Dissociation in Large Language Models 

**Authors**: Toshiyuki Shigemura  

**Link**: [PDF](https://arxiv.org/pdf/2603.18013)  

**Abstract**: Large language models (LLMs) demonstrate the capacity to reconstruct and trace learned content from their training data under specific elicitation conditions, yet this capability does not manifest in standard generation contexts. This empirical observational study examines the expression of non-causal, non-implementable solution types across 300 prompt-response generations spanning narrative and problem-solving task contexts. Drawing on recent findings regarding memorization contiguity and alignment-induced discourse priors, we document a systematic dissociation between learned capability and expressed output. Across three distinct LLMs, ten task scenarios, and both creative narrative and practical advisory contexts, we documented zero instances of non-causal solution frames in generated outputs (0%, 95% CI: [0%, 1.2%]), despite verified reconstruction capability under conditional extraction. These findings challenge the prevailing assumption that training data presence directly predicts output probability, demonstrating instead that task-conditioned generation policies can comprehensively suppress learned content across diverse contexts. The results offer implications for understanding generation dynamics, output distribution control, and the behavioral boundaries of contemporary LLMs. 

---
# DynaRAG: Bridging Static and Dynamic Knowledge in Retrieval-Augmented Generation 

**Authors**: Penghao Liang, Mengwei Yuan, Jianan Liu, Jing Yang, Xianyou Li, Weiran Yan, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.18012)  

**Abstract**: We present DynaRAG, a retrieval-augmented generation (RAG) framework designed to handle both static and time-sensitive information needs through dynamic knowledge integration. Unlike traditional RAG pipelines that rely solely on static corpora, DynaRAG selectively invokes external APIs when retrieved documents are insufficient for answering a query. The system employs an LLM-based reranker to assess document relevance, a sufficiency classifier to determine when fallback is necessary, and Gorilla v2 -- a state-of-the-art API calling model -- for accurate tool invocation. We further enhance robustness by incorporating schema filtering via FAISS to guide API selection. Evaluations on the CRAG benchmark demonstrate that DynaRAG significantly improves accuracy on dynamic questions, while also reducing hallucinations. Our results highlight the importance of dynamic-aware routing and selective tool use in building reliable, real-world question-answering systems. 

---
# Controllable Evidence Selection in Retrieval-Augmented Question Answering via Deterministic Utility Gating 

**Authors**: Victor P. Unda  

**Link**: [PDF](https://arxiv.org/pdf/2603.18011)  

**Abstract**: Many modern AI question-answering systems convert text into vectors and retrieve the closest matches to a user question. While effective for topical similarity, similarity scores alone do not explain why some retrieved text can serve as evidence while other equally similar text cannot. When many candidates receive similar scores, systems may select sentences that are redundant, incomplete, or address different conditions than the question requires.
This paper presents a deterministic evidence selection framework for retrieval-augmented question answering. The approach introduces Meaning-Utility Estimation (MUE) and Diversity-Utility Estimation (DUE), fixed scoring and redundancy-control procedures that determine evidence admissibility prior to answer generation. Each sentence or record is evaluated independently using explicit signals for semantic relatedness, term coverage, conceptual distinctiveness, and redundancy. No training or fine-tuning is required.
In the prototype, a unit is accepted only if it explicitly states the fact, rule, or condition required by the task. Units are not merged or expanded. If no unit independently satisfies the requirement, the system returns no answer. This deterministic gating produces compact, auditable evidence sets and establishes a clear boundary between relevant text and usable evidence. 

---
# Agentic Framework for Political Biography Extraction 

**Authors**: Yifei Zhu, Songpo Yang, Jiangnan Zhu, Junyan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18010)  

**Abstract**: The production of large-scale political datasets typically demands extracting structured facts from vast piles of unstructured documents or web sources, a task that traditionally relies on expensive human experts and remains prohibitively difficult to automate at scale. In this paper, we leverage Large Language Models (LLMs) to automate the extraction of multi-dimensional elite biographies, addressing a long-standing bottleneck in political science research. We propose a two-stage ``Synthesis-Coding'' framework for complex extraction task: an upstream synthesis stage that uses recursive agentic LLMs to search, filter, and curate biography from heterogeneous web sources, followed by a downstream coding stage that maps curated biography into structured dataframes. We validate this framework through three primary results. First, we demonstrate that, when given curated contexts, LLM coders match or outperform human experts in extraction accuracy. Second, we show that in web environments, the agentic system synthesizes more information from web resources than human collective intelligence (Wikipedia). Finally, we diagnosed that directly coding from long and multi-language corpora introduces bias that the synthesis stage can alleviate by curating evidence into signal-dense representations. By comprehensive evaluation, We provide a generalizable, scalable framework for building transparent and expansible large scale database in political science. 

---
# How Confident Is the First Token? An Uncertainty-Calibrated Prompt Optimization Framework for Large Language Model Classification and Understanding 

**Authors**: Wei Chen, Guoyang Ju, Yuanyuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2603.18009)  

**Abstract**: With the widespread adoption of large language models (LLMs) in natural language processing, prompt engineering and retrieval-augmented generation (RAG) have become mainstream to enhance LLMs' performance on complex tasks. However, LLMs generate outputs autoregressively, leading to inevitable output uncertainty. Since model performance is highly sensitive to prompt design, precise uncertainty measurement is crucial for reliable prompt optimization. For multi-class multiple-choice (understanding) tasks, conventional uncertainty measures (e.g., entropy) based on output probabilities treat all classes equally and ignore class prior differences in pretraining corpora. This failure to distinguish spurious confidence (from priors) from true certainty (from contextual understanding) results in poor confidence calibration. To address this, we propose Log-Scale Focal Uncertainty (LSFU), a first-token-based metric inspired by focal loss. LSFU incorporates label prior probabilities as a risk-modulation factor to suppress noise from high-frequency classes and emphasize risk for low-frequency long-tail classes, with a dynamic weighting mechanism unifying the measurement scale. Based on LSFU, we further propose the uncertainty-calibrated prompt optimization framework (UCPOF), which leverages the first token of model outputs to select high-quality exemplars and dynamically optimize prompts. Comprehensive evaluations show UCPOF improves average accuracy by 6.03% over few-shot baselines, surpasses always-on full RAG by 5.75% in overall average accuracy, and reduces the average retrieval trigger rate by 50.66%. By adaptively triggering RAG only for high-uncertainty samples, our framework significantly lowers computational costs while maintaining state-of-the-art performance. 

---
# TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots 

**Authors**: Fangrui Huang, Souhad Chbeir, Arpandeep Khatua, Sheng Wang, Sijun Tan, Kenan Ye, Lily Bailey, Merryn Daniel, Ryan Louie, Sanmi Koyejo, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2603.18008)  

**Abstract**: Large language models (LLMs) are increasingly used for mental-health support; yet prevailing evaluation methods--fluency metrics, preference tests, and generic dialogue benchmarks--fail to capture the clinically critical dimensions of psychotherapy. We introduce THERAPYGYM, a framework that evaluates and improves therapy chatbots along two clinical pillars: fidelity and safety. Fidelity is measured using the Cognitive Therapy Rating Scale (CTRS), implemented as an automated pipeline that scores adherence to CBT techniques over multi-turn sessions. Safety is assessed using a multi-label annotation scheme, covering therapy-specific risks (e.g., failing to address harm or abuse). To mitigate bias and unreliability in LLM-based judges, we further release THERAPYJUDGEBENCH, a validation set of 116 dialogues with 1,270 expert ratings for auditing and calibration against licensed clinicians. THERAPYGYM also serves as a training harness: CTRS and safety-based rewards drive RL with configurable patient simulations spanning diverse symptom profiles. Models trained in THERAPYGYM improve on expert ratings, with average CTRS rising from 0.10 to 0.60 (and 0.16 to 0.59 under LLM judges). Our work enables scalable development of therapy chatbots that are faithful to evidence-based practice and safer in high-stakes use. 

---
# Do Large Language Models Possess a Theory of Mind? A Comparative Evaluation Using the Strange Stories Paradigm 

**Authors**: Anna Babarczy, Andras Lukacs, Peter Vedres, Zeteny Bujka  

**Link**: [PDF](https://arxiv.org/pdf/2603.18007)  

**Abstract**: The study explores whether current Large Language Models (LLMs) exhibit Theory of Mind (ToM) capabilities -- specifically, the ability to infer others' beliefs, intentions, and emotions from text. Given that LLMs are trained on language data without social embodiment or access to other manifestations of mental representations, their apparent social-cognitive reasoning raises key questions about the nature of their understanding. Are they capable of robust mental-state attribution indistinguishable from human ability in its output, or do their outputs merely reflect superficial pattern completion? To address this question, we tested five LLMs and compared their performance to that of human controls using an adapted version of a text-based tool widely used in human ToM research. The test involves answering questions about the beliefs, intentions, and emotions of story characters. The results revealed a performance gap between the models. Earlier and smaller models were strongly affected by the number of relevant inferential cues available and, to some extent, were also vulnerable to the presence of irrelevant or distracting information in the texts. In contrast, GPT-4o demonstrated high accuracy and strong robustness, performing comparably to humans even in the most challenging conditions. This work contributes to ongoing debates about the cognitive status of LLMs and the boundary between genuine understanding and statistical approximation. 

---
# FinTradeBench: A Financial Reasoning Benchmark for LLMs 

**Authors**: Yogesh Agrawal, Aniruddha Dutta, Md Mahadi Hasan, Santu Karmaker, Aritra Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2603.19225)  

**Abstract**: Real-world financial decision-making is a challenging problem that requires reasoning over heterogeneous signals, including company fundamentals derived from regulatory filings and trading signals computed from price dynamics. Recently, with the advancement of Large Language Models (LLMs), financial analysts have begun to use them for financial decision-making tasks. However, existing financial question answering benchmarks for testing these models primarily focus on company balance sheet data and rarely evaluate reasoning over how company stocks trade in the market or their interactions with fundamentals. To take advantage of the strengths of both approaches, we introduce FinTradeBench, a benchmark for evaluating financial reasoning that integrates company fundamentals and trading signals. FinTradeBench contains 1,400 questions grounded in NASDAQ-100 companies over a ten-year historical window. The benchmark is organized into three reasoning categories: fundamentals-focused, trading-signal-focused, and hybrid questions requiring cross-signal reasoning. To ensure reliability at scale, we adopt a calibration-then-scaling framework that combines expert seed questions, multi-model response generation, intra-model self-filtering, numerical auditing, and human-LLM judge alignment. We evaluate 14 LLMs under zero-shot prompting and retrieval-augmented settings and witness a clear performance gap. Retrieval substantially improves reasoning over textual fundamentals, but provides limited benefit for trading-signal reasoning. These findings highlight fundamental challenges in the numerical and time-series reasoning for current LLMs and motivate future research in financial intelligence. 

---
# Online Learning and Equilibrium Computation with Ranking Feedback 

**Authors**: Mingyang Liu, Yongshan Chen, Zhiyuan Fan, Gabriele Farina, Asuman Ozdaglar, Kaiqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19221)  

**Abstract**: Online learning in arbitrary, and possibly adversarial, environments has been extensively studied in sequential decision-making, and it is closely connected to equilibrium computation in game theory. Most existing online learning algorithms rely on \emph{numeric} utility feedback from the environment, which may be unavailable in human-in-the-loop applications and/or may be restricted by privacy concerns. In this paper, we study an online learning model in which the learner only observes a \emph{ranking} over a set of proposed actions at each timestep. We consider two ranking mechanisms: rankings induced by the \emph{instantaneous} utility at the current timestep, and rankings induced by the \emph{time-average} utility up to the current timestep, under both \emph{full-information} and \emph{bandit} feedback settings. Using the standard external-regret metric, we show that sublinear regret is impossible with instantaneous-utility ranking feedback in general. Moreover, when the ranking model is relatively deterministic, \emph{i.e.}, under the Plackett-Luce model with a temperature that is sufficiently small, sublinear regret is also impossible with time-average utility ranking feedback. We then develop new algorithms that achieve sublinear regret under the additional assumption that the utility sequence has sublinear total variation. Notably, for full-information time-average utility ranking feedback, this additional assumption can be removed. As a consequence, when all players in a normal-form game follow our algorithms, repeated play yields an approximate coarse correlated equilibrium. We also demonstrate the effectiveness of our algorithms in an online large-language-model routing task. 

---
# How Auditory Knowledge in LLM Backbones Shapes Audio Language Models: A Holistic Evaluation 

**Authors**: Ke-Han Lu, Szu-Wei Fu, Chao-Han Huck Yang, Zhehuai Chen, Sung-Feng Huang, Chih-Kai Yang, Yi-Cheng Lin, Chi-Yuan Hsiao, Wenze Ren, En-Pei Hu, Yu-Han Huang, An-Yu Cheng, Cheng-Han Chiang, Yu Tsao, Yu-Chiang Frank Wang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.19195)  

**Abstract**: Large language models (LLMs) have been widely used as knowledge backbones of Large Audio Language Models (LALMs), yet how much auditory knowledge they encode through text-only pre-training and how this affects downstream performance remains unclear. We study this gap by comparing different LLMs under two text-only and one audio-grounded setting: (1) direct probing on AKB-2000, a curated benchmark testing the breadth and depth of auditory knowledge; (2) cascade evaluation, where LLMs reason over text descriptions from an audio captioner; and (3) audio-grounded evaluation, where each LLM is fine-tuned into a Large Audio Language Model (LALM) with an audio encoder. Our findings reveal that auditory knowledge varies substantially across families, and text-only results are strongly correlated with audio performance. Our work provides empirical grounding for a comprehensive understanding of LLMs in audio research. 

---
# Box Maze: A Process-Control Architecture for Reliable LLM Reasoning 

**Authors**: Zou Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19182)  

**Abstract**: Large language models (LLMs) demonstrate strong generative capabilities but remain vulnerable to hallucination and unreliable reasoning under adversarial prompting. Existing safety approaches -- such as reinforcement learning from human feedback (RLHF) and output filtering -- primarily operate at the behavioral level and may lack explicit architectural mechanisms for enforcing reasoning process integrity.
This paper proposes the Box Maze framework, a conceptual process-control architecture that decomposes LLM reasoning into three explicit layers: memory grounding, structured inference, and boundary enforcement. We introduce preliminary simulation-based evaluation involving progressive boundary erosion scenarios across multiple heterogeneous LLM systems (DeepSeek-V3, Doubao, Qwen). Results from n=50 adversarial scenarios suggest that explicit cognitive control layers may improve consistency in boundary maintenance, with architectural constraints reducing boundary failure rates from approximately 40% (baseline RLHF) to below 1% under adversarial conditions.
While current validation is simulation-based, these preliminary results indicate that process-level control may offer a promising direction for improving reliability in large language model reasoning. 

---
# Meanings and Measurements: Multi-Agent Probabilistic Grounding for Vision-Language Navigation 

**Authors**: Swagat Padhan, Lakshya Jain, Bhavya Minesh Shah, Omkar Patil, Thao Nguyen, Nakul Gopalan  

**Link**: [PDF](https://arxiv.org/pdf/2603.19166)  

**Abstract**: Robots collaborating with humans must convert natural language goals into actionable, physically grounded decisions. For example, executing a command such as "go two meters to the right of the fridge" requires grounding semantic references, spatial relations, and metric constraints within a 3D scene. While recent vision language models (VLMs) demonstrate strong semantic grounding capabilities, they are not explicitly designed to reason about metric constraints in physically defined spaces. In this work, we empirically demonstrate that state-of-the-art VLM-based grounding approaches struggle with complex metric-semantic language queries. To address this limitation, we propose MAPG (Multi-Agent Probabilistic Grounding), an agentic framework that decomposes language queries into structured subcomponents and queries a VLM to ground each component. MAPG then probabilistically composes these grounded outputs to produce metrically consistent, actionable decisions in 3D space. We evaluate MAPG on the HM-EQA benchmark and show consistent performance improvements over strong baselines. Furthermore, we introduce a new benchmark, MAPG-Bench, specifically designed to evaluate metric-semantic goal grounding, addressing a gap in existing language grounding evaluations. We also present a real-world robot demonstration showing that MAPG transfers beyond simulation when a structured scene representation is available. 

---
# How Uncertainty Estimation Scales with Sampling in Reasoning Models 

**Authors**: Maksym Del, Markus Kängsepp, Marharyta Domnich, Ardi Tampuu, Lisa Yankovskaya, Meelis Kull, Mark Fishel  

**Link**: [PDF](https://arxiv.org/pdf/2603.19118)  

**Abstract**: Uncertainty estimation is critical for deploying reasoning language models, yet remains poorly understood under extended chain-of-thought reasoning. We study parallel sampling as a fully black-box approach using verbalized confidence and self-consistency. Across three reasoning models and 17 tasks spanning mathematics, STEM, and humanities, we characterize how these signals scale.
Both self-consistency and verbalized confidence scale in reasoning models, but self-consistency exhibits lower initial discrimination and lags behind verbalized confidence under moderate sampling. Most uncertainty gains, however, arise from signal combination: with just two samples, a hybrid estimator improves AUROC by up to $+12$ on average and already outperforms either signal alone even when scaled to much larger budgets, after which returns diminish. These effects are domain-dependent: in mathematics, the native domain of RLVR-style post-training, reasoning models achieve higher uncertainty quality and exhibit both stronger complementarity and faster scaling than in STEM or humanities. 

---
# SAVeS: Steering Safety Judgments in Vision-Language Models via Semantic Cues 

**Authors**: Carlos Hinojosa, Clemens Grange, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2603.19092)  

**Abstract**: Vision-language models (VLMs) are increasingly deployed in real-world and embodied settings where safety decisions depend on visual context. However, it remains unclear which visual evidence drives these judgments. We study whether multimodal safety behavior in VLMs can be steered by simple semantic cues. We introduce a semantic steering framework that applies controlled textual, visual, and cognitive interventions without changing the underlying scene content. To evaluate these effects, we propose SAVeS, a benchmark for situational safety under semantic cues, together with an evaluation protocol that separates behavioral refusal, grounded safety reasoning, and false refusals. Experiments across multiple VLMs and an additional state-of-the-art benchmark show that safety decisions are highly sensitive to semantic cues, indicating reliance on learned visual-linguistic associations rather than grounded visual understanding. We further demonstrate that automated steering pipelines can exploit these mechanisms, highlighting a potential vulnerability in multimodal safety systems. 

---
# Serendipity by Design: Evaluating the Impact of Cross-domain Mappings on Human and LLM Creativity 

**Authors**: Qiawen Ella Liu, Marina Dubova, Henry Conklin, Takumi Harada, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2603.19087)  

**Abstract**: Are large language models (LLMs) creative in the same way humans are, and can the same interventions increase creativity in both? We evaluate a promising but largely untested intervention for creativity: forcing creators to draw an analogy from a random, remote source domain (''cross-domain mapping''). Human participants and LLMs generated novel features for ten daily products (e.g., backpack, TV) under two prompts: (i) cross-domain mapping, which required translating a property from a randomly assigned source (e.g., octopus, cactus, GPS), and (ii) user-need, which required proposing innovations targeting unmet user needs. We show that humans reliably benefit from randomly assigned cross-domain mappings, while LLMs, on average, generate more original ideas than humans and do not show a statistically significant effect of cross-domain mappings. However, in both systems, the impact of cross-domain mapping increases when the inspiration source becomes more semantically distant from the target. Our results highlight both the role of remote association in creative ideation and systematic differences in how humans and LLMs respond to the same intervention for creativity. 

---
# A conceptual framework for ideology beyond the left and right 

**Authors**: Kenneth Joseph, Kim Williams, David Lazer  

**Link**: [PDF](https://arxiv.org/pdf/2603.18945)  

**Abstract**: NLP+CSS work has operationalized ideology almost exclusively on a left/right partisan axis. This approach obscures the fact that people hold interpretations of many different complex and more specific ideologies on issues like race, climate, and gender. We introduce a framework that understands ideology as an attributed, multi-level socio-cognitive concept network, and explains how ideology manifests in discourse in relation to other relevant social processes like framing. We demonstrate how this framework can clarifies overlaps between existing NLP tasks (e.g. stance detection and natural language inference) and also how it reveals new research directions. Our work provides a unique and important bridge between computational methods and ideology theory, enabling richer analysis of social discourse in a way that benefits both fields. 

---
# Reasoning over mathematical objects: on-policy reward modeling and test time aggregation 

**Authors**: Pranjal Aggarwal, Marjan Ghazvininejad, Seungone Kim, Ilia Kulikov, Jack Lanchantin, Xian Li, Tianjian Li, Bo Liu, Graham Neubig, Anaelia Ovalle, Swarnadeep Saha, Sainbayar Sukhbaatar, Sean Welleck, Jason Weston, Chenxi Whitehouse, Adina Williams, Jing Xu, Ping Yu, Weizhe Yuan, Jingyu Zhang, Wenting Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.18886)  

**Abstract**: The ability to precisely derive mathematical objects is a core requirement for downstream STEM applications, including mathematics, physics, and chemistry, where reasoning must culminate in formally structured expressions. Yet, current LM evaluations of mathematical and scientific reasoning rely heavily on simplified answer formats such as numerical values or multiple choice options due to the convenience of automated assessment. In this paper we provide three contributions for improving reasoning over mathematical objects: (i) we build and release training data and benchmarks for deriving mathematical objects, the Principia suite; (ii) we provide training recipes with strong LLM-judges and verifiers, where we show that on-policy judge training boosts performance; (iii) we show how on-policy training can also be used to scale test-time compute via aggregation. We find that strong LMs such as Qwen3-235B and o3 struggle on Principia, while our training recipes can bring significant improvements over different LLM backbones, while simultaneously improving results on existing numerical and MCQA tasks, demonstrating cross-format generalization of reasoning abilities. 

---
# RewardFlow: Topology-Aware Reward Propagation on State Graphs for Agentic RL with Large Language Models 

**Authors**: Xiao Feng, Bo Han, Zhanke Zhou, Jiaqi Fan, Jiangchao Yao, Ka Ho Li, Dahai Yu, Michael Kwok-Po Ng  

**Link**: [PDF](https://arxiv.org/pdf/2603.18859)  

**Abstract**: Reinforcement learning (RL) holds significant promise for enhancing the agentic reasoning capabilities of large language models (LLMs) with external environments. However, the inherent sparsity of terminal rewards hinders fine-grained, state-level optimization. Although process reward modeling offers a promising alternative, training dedicated reward models often entails substantial computational costs and scaling difficulties. To address these challenges, we introduce RewardFlow, a lightweight method for estimating state-level rewards tailored to agentic reasoning tasks. RewardFlow leverages the intrinsic topological structure of states within reasoning trajectories by constructing state graphs. This enables an analysis of state-wise contributions to success, followed by topology-aware graph propagation to quantify contributions and yield objective, state-level rewards. When integrated as dense rewards for RL optimization, RewardFlow substantially outperforms prior RL baselines across four agentic reasoning benchmarks, demonstrating superior performance, robustness, and training efficiency. The implementation of RewardFlow is publicly available at this https URL. 

---
# Are complicated loss functions necessary for teaching LLMs to reason? 

**Authors**: Gabriele Carrino, Andrea Sassella, Nicolo Brunello, Federico Toschi, Mark James Carman  

**Link**: [PDF](https://arxiv.org/pdf/2603.18756)  

**Abstract**: Recent advances in large language models (LLMs) highlight the importance of post training techniques for improving reasoning and mathematical ability. Group Relative Policy Optimization (GRPO) has shown promise in this domain by combining group relative advantage estimation, PPO style clipping, and KL regularization. However, its complexity raises the question of whether all components are necessary for fostering reasoning behaviors. We conduct a systematic analysis of GRPO and identify two key findings: (1) incorporating negative feedback is essential training solely on actions above a baseline limits learning; and (2) PPO style constraints, such as policy ratio clipping, are not required to improve mathematical reasoning or performance. Building on these insights, we propose REINFORCE with Group Relative Advantage (RGRA), a simplified variant that retains group relative advantage estimation but removes PPO style clipping and policy ratio terms. Experiments across standard mathematical benchmarks indicate that RGRA has the potential to achieve stronger performance than GRPO. Our results suggest that simpler REINFORCE based approaches can effectively enhance reasoning in LLMs, offering a more transparent and efficient alternative to GRPO. 

---
# Memento-Skills: Let Agents Design Agents 

**Authors**: Huichi Zhou, Siyuan Guo, Anjie Liu, Zhongwei Yu, Ziqin Gong, Bowen Zhao, Zhixun Chen, Menglong Zhang, Yihang Chen, Jinsong Li, Runyu Yang, Qiangbin Liu, Xinlei Yu, Jianmin Zhou, Na Wang, Chunyang Sun, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18743)  

**Abstract**: We introduce \emph{Memento-Skills}, a generalist, continually-learnable LLM agent system that functions as an \emph{agent-designing agent}: it autonomously constructs, adapts, and improves task-specific agents through experience. The system is built on a memory-based reinforcement learning framework with \emph{stateful prompts}, where reusable skills (stored as structured markdown files) serve as persistent, evolving memory. These skills encode both behaviour and context, enabling the agent to carry forward knowledge across interactions.
Starting from simple elementary skills (like Web search and terminal operations), the agent continually improves via the \emph{Read--Write Reflective Learning} mechanism introduced in \emph{Memento~2}~\cite{wang2025memento2}. In the \emph{read} phase, a behaviour-trainable skill router selects the most relevant skill conditioned on the current stateful prompt; in the \emph{write} phase, the agent updates and expands its skill library based on new experience. This closed-loop design enables \emph{continual learning without updating LLM parameters}, as all adaptation is realised through the evolution of externalised skills and prompts.
Unlike prior approaches that rely on human-designed agents, Memento-Skills enables a generalist agent to \emph{design agents end-to-end} for new tasks. Through iterative skill generation and refinement, the system progressively improves its own capabilities. Experiments on the \emph{General AI Assistants} benchmark and \emph{Humanity's Last Exam} demonstrate sustained gains, achieving 26.2\% and 116.2\% relative improvements in overall accuracy, respectively. Code is available at this https URL. 

---
# CausalRM: Causal-Theoretic Reward Modeling for RLHF from Observational User Feedbacks 

**Authors**: Hao Wang, Licheng Pan, Zhichao Chen, Chunyuan Zheng, Zhixuan Chu, Xiaoxi Li, Yuan Lu, Xinggao Liu, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.18736)  

**Abstract**: Despite the success of reinforcement learning from human feedback (RLHF) in aligning language models, current reward modeling heavily relies on experimental feedback data collected from human annotators under controlled and costly conditions. In this work, we introduce observational reward modeling -- learning reward models with observational user feedback (e.g., clicks, copies, and upvotes) -- as a scalable and cost-effective alternative. We identify two fundamental challenges in this setting: (1) observational feedback is noisy due to annotation errors, which deviates it from true user preference; (2) observational feedback is biased by user preference, where users preferentially provide feedback on responses they feel strongly about, which creats a distribution shift between training and inference data. To address these challenges, we propose CausalRM, a causal-theoretic reward modeling framework that aims to learn unbiased reward models from observational feedback. To tackle challenge (1), CausalRM introduces a noise-aware surrogate loss term that is provably equivalent to the primal loss under noise-free conditions by explicitly modeling the annotation error generation process. To tackle challenge (2), CausalRM uses propensity scores -- the probability of a user providing feedback for a given response -- to reweight training samples, yielding a loss function that eliminates user preference bias. Extensive experiments across diverse LLM backbones and benchmark datasets validate that CausalRM effectively learns accurate reward signals from noisy and biased observational feedback and delivers substantial performance improvements on downstream RLHF tasks -- including a 49.2% gain on WildGuardMix and a 32.7% improvement on HarmBench. Code is available on our project website. 

---
# STEP: Scientific Time-Series Encoder Pretraining via Cross-Domain Distillation 

**Authors**: Chen Zhang, Liwei Liu, Jun Tao, Xiaoyu Yang, Xuenan Xu, Kai Chen, Bowen Zhou, Wen Wu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18688)  

**Abstract**: Scientific time series are central to scientific AI but are typically sparse, highly heterogeneous, and limited in scale, making unified representation learning particularly challenging. Meanwhile, foundation models pretrained on relevant time series domains such as audio, general time series, and brain signals contain rich knowledge, but their applicability to scientific signals remains underexplored. In this paper, we investigate the transferability and complementarity of foundation models from relevant time series domains, and study how to effectively leverage them to build a unified encoder for scientific time series. We first systematically evaluate relevant foundation models, showing the effectiveness of knowledge transfer to scientific tasks and their complementary strengths. Based on this observation, we propose STEP, a Scientific Time Series Encoder Pretraining framework via cross domain distillation. STEP introduces adaptive patching to handle extreme-length sequences and a statistics compensation scheme to accommodate diverse numerical scales. It further leverages cross-domain distillation to integrate knowledge from multiple foundation models into a unified encoder. By combining complementary representations across different domains, STEP learns general-purpose and transferable features tailored for scientific signals. Experiments on seven scientific time series tasks demonstrate that STEP provides both an effective structure and an effective pretraining paradigm, taking a STEP toward scientific time series representation learning. 

---
# HISR: Hindsight Information Modulated Segmental Process Rewards For Multi-turn Agentic Reinforcement Learning 

**Authors**: Zhicong Lu, Zichuan Lin, Wei Jia, Changyuan Tian, Deheng Ye, Peiguang Li, Li Jin, Nayu Liu, Guangluan Xu, Wei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.18683)  

**Abstract**: While large language models excel in diverse domains, their performance on complex longhorizon agentic decision-making tasks remains limited. Most existing methods concentrate on designing effective reward models (RMs) to advance performance via multi-turn reinforcement learning. However, they suffer from delayed propagation in sparse outcome rewards and unreliable credit assignment with potentially overly fine-grained and unfocused turnlevel process rewards. In this paper, we propose (HISR) exploiting Hindsight Information to modulate Segmental process Rewards, which closely aligns rewards with sub-goals and underscores significant segments to enhance the reliability of credit assignment. Specifically, a segment-level process RM is presented to assign rewards for each sub-goal in the task, avoiding excessively granular allocation to turns. To emphasize significant segments in the trajectory, a hindsight model is devised to reflect the preference of performing a certain action after knowing the trajectory outcome. With this characteristic, we design the ratios of sequence likelihoods between hindsight and policy model to measure action importance. The ratios are subsequently employed to aggregate segment importance scores, which in turn modulate segmental process rewards, enhancing credit assignment reliability. Extensive experimental results on three publicly benchmarks demonstrate the validity of our method. 

---
# Words at Play: Benchmarking Audio Pun Understanding in Large Audio-Language Models 

**Authors**: Yuchen Su, Shaoxin Zhong, Yonghua Zhu, Ruofan Wang, Zijian Huang, Qiqi Wang, Na Zhao, Diana Benavides-Prado, Michael Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2603.18678)  

**Abstract**: Puns represent a typical linguistic phenomenon that exploits polysemy and phonetic ambiguity to generate humour, posing unique challenges for natural language understanding. Within pun research, audio plays a central role in human communication except text and images, while datasets and systematic resources for spoken puns remain scarce, leaving this crucial modality largely underexplored. In this paper, we present APUN-Bench, the first benchmark dedicated to evaluating large audio language models (LALMs) on audio pun understanding. Our benchmark contains 4,434 audio samples annotated across three stages: pun recognition, pun word location and pun meaning inference. We conduct a deep analysis of APUN-Bench by systematically evaluating 10 state-of-the-art LALMs, uncovering substantial performance gaps in recognizing, localizing, and interpreting audio puns. This analysis reveals key challenges, such as positional biases in audio pun location and error cases in meaning inference, offering actionable insights for advancing humour-aware audio intelligence. 

---
# MOSAIC: Multi-Objective Slice-Aware Iterative Curation for Alignment 

**Authors**: Yipu Dou, Wang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18637)  

**Abstract**: We study how to allocate a fixed supervised fine-tuning budget when three objectives must be balanced at once: multi-turn safety alignment, low over-refusal on benign boundary queries, and instruction following under verifiable constraints. We propose MOSAIC (Multi-Objective Slice-Aware Iterative Curation for Alignment), a multi-objective framework for closed-loop data mixture search built on a unified L1-L3 evaluation interface. MOSAIC turns slice-level failure profiles into executable data actions, including dataset-level mixture ratios, bucket-level weights, and focus criteria. Under a fixed 1M-token budget and five rounds of independent fine-tuning from the same base model, MOSAIC improves internal XGuard from 2.76 to 4.67 while keeping OrBench at 4.41 and IFEval at 3.65. The final Pareto solution also generalizes better than a random static LoRA baseline on independent attack, over-refusal, and capability tests, suggesting that structured failure diagnosis can serve as a practical control signal for budgeted data construction. Code is available at this https URL. 

---
# myMNIST: Benchmark of PETNN, KAN, and Classical Deep Learning Models for Burmese Handwritten Digit Recognition 

**Authors**: Ye Kyaw Thu, Thazin Myint Oo, Thepchai Supnithi  

**Link**: [PDF](https://arxiv.org/pdf/2603.18597)  

**Abstract**: We present the first systematic benchmark on myMNIST (formerly BHDD), a publicly available Burmese handwritten digit dataset important for Myanmar NLP/AI research. We evaluate eleven architectures spanning classical deep learning models (Multi-Layer Perceptron, Convolutional Neural Network, Long Short-Term Memory, Gated Recurrent Unit, Transformer), recent alternatives (FastKAN, EfficientKAN), an energy-based model (JEM), and physics-inspired PETNN variants (Sigmoid, GELU, SiLU). Using Precision, Recall, F1-Score, and Accuracy as evaluation metrics, our results show that the CNN remains a strong baseline, achieving the best overall scores (F1 = 0.9959, Accuracy = 0.9970). The PETNN (GELU) model closely follows (F1 = 0.9955, Accuracy = 0.9966), outperforming LSTM, GRU, Transformer, and KAN variants. JEM, representing energy-based modeling, performs competitively (F1 = 0.9944, Accuracy = 0.9958). KAN-based models (FastKAN, EfficientKAN) trail the top performers but provide a meaningful alternative baseline (Accuracy ~0.992). These findings (i) establish reproducible baselines for myMNIST across diverse modeling paradigms, (ii) highlight PETNN's strong performance relative to classical and Transformer-based models, and (iii) quantify the gap between energy-inspired PETNNs and a true energy-based model (JEM). We release this benchmark to facilitate future research on Myanmar digit recognition and to encourage broader evaluation of emerging architectures on regional scripts. 

---
# SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding 

**Authors**: Shenggui Li, Chao Wang, Yikai Zhu, Yubo Wang, Fan Yin, Shuai Shi, Yefei Chen, Xiaomin Dong, Qiaoling Chen, Jin Pan, Ji Li, Laixin Xie, Yineng Zhang, Lei Yu, Yonggang Wen, Ivor Tsang, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18567)  

**Abstract**: Large language models incur high inference latency due to sequential autoregressive decoding. Speculative decoding alleviates this bottleneck by using a lightweight draft model to propose multiple tokens for batched verification. However, its adoption has been limited by the lack of high-quality draft models and scalable training infrastructure. We introduce SpecForge, an open-source, production-oriented framework for training speculative decoding models with full support for EAGLE-3. SpecForge incorporates target-draft decoupling, hybrid parallelism, optimized training kernels, and integration with production-grade inference engines, enabling up to 9.9x faster EAGLE-3 training for Qwen3-235B-A22B. In addition, we release SpecBundle, a suite of production-grade EAGLE-3 draft models trained with SpecForge for mainstream open-source LLMs. Through a systematic study of speculative decoding training recipes, SpecBundle addresses the scarcity of high-quality drafts in the community, and our draft models achieve up to 4.48x end-to-end inference speedup on SGLang, establishing SpecForge as a practical foundation for real-world speculative decoding deployment. 

---
# Balancing the Reasoning Load: Difficulty-Differentiated Policy Optimization with Length Redistribution for Efficient and Robust Reinforcement Learning 

**Authors**: Yinan Xia, Haotian Zhang, Huiming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18533)  

**Abstract**: Large Reasoning Models (LRMs) have shown exceptional reasoning capabilities, but they also suffer from the issue of overthinking, often generating excessively long and redundant answers.
For problems that exceed the model's capabilities, LRMs tend to exhibit the overconfidence phenomenon, generating overly short but incorrect answers, which may contribute to suboptimal performance.
To address these issues, we propose Difficulty-Differentiated Policy Optimization (DDPO), an efficient reinforcement learning algorithm that optimizes simple and complex tasks separately based on the overconfidence phenomenon.
Specifically, it reduces the output length for simple tasks without compromising accuracy, while for complex tasks, it expands the exploration space to improve performance. We further derive the theoretical conditions for maximizing expected accuracy, which require the length distribution to closely approximate the optimal length and be as concentrated as possible. Based on these conditions, we propose using the difficulty-level average as a well-founded reference for length optimization.
Extensive experiments on both in-domain and out-of-domain benchmarks validate the superiority and effectiveness of DDPO. Compared to GRPO, DDPO reduces the average answer length by 12% while improving accuracy by 1.85% across multiple benchmarks, achieving a better trade-off between accuracy and length. The code is available at this https URL. 

---
# SODIUM: From Open Web Data to Queryable Databases 

**Authors**: Chuxuan Hu, Philip Li, Maxwell Yang, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18447)  

**Abstract**: During research, domain experts often ask analytical questions whose answers require integrating data from a wide range of web sources. Thus, they must spend substantial effort searching, extracting, and organizing raw data before analysis can begin. We formalize this process as the SODIUM task, where we conceptualize open domains such as the web as latent databases that must be systematically instantiated to support downstream querying. Solving SODIUM requires (1) conducting in-depth and specialized exploration of the open web, which is further strengthened by (2) exploiting structural correlations for systematic information extraction and (3) integrating collected information into coherent, queryable database instances.
To quantify the challenges in automating SODIUM, we construct SODIUM-Bench, a benchmark of 105 tasks derived from published academic papers across 6 domains, where systems are tasked with exploring the open web to collect and aggregate data from diverse sources into structured tables. Existing systems struggle with SODIUM tasks: we evaluate 6 advanced AI agents on SODIUM-Bench, with the strongest baseline achieving only 46.5% accuracy. To bridge this gap, we develop SODIUM-Agent, a multi-agent system composed of a web explorer and a cache manager. Powered by our proposed ATP-BFS algorithm and optimized through principled management of cached sources and navigation paths, SODIUM-Agent conducts deep and comprehensive web exploration and performs structurally coherent information extraction. SODIUM-Agent achieves 91.1% accuracy on SODIUM-Bench, outperforming the strongest baseline by approximately 2 times and the weakest by up to 73 times. 

---
# From Topic to Transition Structure: Unsupervised Concept Discovery at Corpus Scale via Predictive Associative Memory 

**Authors**: Jason Dury  

**Link**: [PDF](https://arxiv.org/pdf/2603.18420)  

**Abstract**: Embedding models group text by semantic content, what text is about. We show that temporal co-occurrence within texts discovers a different kind of structure: recurrent transition-structure concepts or what text does. We train a 29.4M-parameter contrastive model on 373 million co-occurrence pairs from 9,766 Project Gutenberg texts (24.96 million passages), mapping pre-trained embeddings into an association space where passages with similar transition structure cluster together. Under capacity constraint (42.75% accuracy), the model must compress across recurring patterns rather than memorise individual co-occurrences. Clustering at six granularities (k=50 to k=2,000) produces a multi-resolution concept map; from broad modes like "direct confrontation" and "lyrical meditation" to precise registers and scene templates like "sailor dialect" and "courtroom cross-examination." At k=100, clusters average 4,508 books each (of 9,766), confirming corpus-wide patterns. Direct comparison with embedding-similarity clustering shows that raw embeddings group by topic while association-space clusters group by function, register, and literary tradition. Unseen novels are assigned to existing clusters without retraining; the association model concentrates each novel into a selective subset of coherent clusters, while raw embedding assignment saturates nearly all clusters. Validation controls address positional, length, and book-concentration confounds. The method extends Predictive Associative Memory (PAM, arXiv:2602.11322) from episodic recall to concept formation: where PAM recalls specific associations, multi-epoch contrastive training under compression extracts structural patterns that transfer to unseen texts, the same framework producing qualitatively different behaviour in a different regime. 

---
# Large-Scale Analysis of Political Propaganda on Moltbook 

**Authors**: Julia Jose, Meghna Manoj Nair, Rachel Greenstadt  

**Link**: [PDF](https://arxiv.org/pdf/2603.18349)  

**Abstract**: We present an NLP-based study of political propaganda on Moltbook, a Reddit-style platform for AI agents. To enable large-scale analysis, we develop LLM-based classifiers to detect political propaganda, validated against expert annotation (Cohen's $\kappa$= 0.64-0.74). Using a dataset of 673,127 posts and 879,606 comments, we find that political propaganda accounts for 1% of all posts and 42% of all political content. These posts are concentrated in a small set of communities, with 70% of such posts falling into five of them. 4% of agents produced 51% of these posts. We further find that a minority of these agents repeatedly post highly similar content within and across communities. Despite this, we find limited evidence that comments amplify political propaganda. 

---
# Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails 

**Authors**: Gregory N. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2603.18280)  

**Abstract**: Current alignment evaluation mostly measures whether models encode dangerous concepts and whether they refuse harmful requests. Both miss the layer where alignment often operates: routing from concept detection to behavioral policy. We study political censorship in Chinese-origin language models as a natural experiment, using probes, surgical ablations, and behavioral tests across nine open-weight models from five labs. Three findings follow. First, probe accuracy alone is non-diagnostic: political probes, null controls, and permutation baselines can all reach 100%, so held-out category generalization is the informative test. Second, surgical ablation reveals lab-specific routing. Removing the political-sensitivity direction eliminates censorship and restores accurate factual output in most models tested, while one model confabulates because its architecture entangles factual knowledge with the censorship mechanism. Cross-model transfer fails, indicating that routing geometry is model- and lab-specific. Third, refusal is no longer the dominant censorship mechanism. Within one model family, hard refusal falls to zero while narrative steering rises to the maximum, making censorship invisible to refusal-only benchmarks. These results support a three-stage descriptive framework: detect, route, generate. Models often retain the relevant knowledge; alignment changes how that knowledge is expressed. Evaluations that audit only detection or refusal therefore miss the routing mechanism that most directly determines behavior. 

---
# Retrieval-Augmented LLM Agents: Learning to Learn from Experience 

**Authors**: Thomas Palmeira Ferraz, Romain Deffayet, Vassilina Nikoulina, Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2603.18272)  

**Abstract**: While large language models (LLMs) have advanced the development of general-purpose agents, achieving robust generalization to unseen tasks remains a significant challenge. Current approaches typically rely on either fine-tuning or training-free memory-augmented generation using retrieved experience; yet both have limitations: fine-tuning often fails to extrapolate to new tasks, while experience retrieval often underperforms compared to supervised baselines. In this work, we propose to combine these approaches and systematically study how to train retrieval-augmented LLM agents to effectively leverage retrieved trajectories in-context. First, we establish a robust supervised fine-tuning (SFT) recipe using LoRA that outperforms several state-of-the-art agent training pipelines. Second, we provide a detailed analysis of key design choices for experience retrieval, identifying optimal strategies for storage, querying, and trajectory selection. Finally, we propose a pipeline that integrates experience retrieval into the fine-tuning process. Our results demonstrate that this combined approach significantly improves generalization to unseen tasks, providing a scalable and effective framework for building agents that learn to learn from experience. 

---
# Impact of automatic speech recognition quality on Alzheimer's disease detection from spontaneous speech: a reproducible benchmark study with lexical modeling and statistical validation 

**Authors**: Himadri Samanta  

**Link**: [PDF](https://arxiv.org/pdf/2603.18239)  

**Abstract**: Early detection of Alzheimer's disease from spontaneous speech has emerged as a promising non-invasive screening approach. However, the influence of automatic speech recognition (ASR) quality on downstream clinical language modeling remains insufficiently understood. In this study, we investigate Alzheimer's disease detection using lexical features derived from Whisper ASR transcripts on the ADReSSo 2021 diagnosis dataset. We evaluate interpretable machine-learning models, including Logistic Regression and Linear Support Vector Machines, using TF-IDF text representations under repeated 5x5 stratified cross-validation.
Our results demonstrate that transcript quality has a statistically significant impact on classification performance. Models trained on Whisper-small transcripts consistently outperform those using Whisper-base transcripts, achieving balanced accuracy above 0.7850 with Linear SVM. Paired statistical testing confirms that the observed improvements are significant. Importantly, classifier complexity contributes less to performance variation than ASR transcription quality. Feature analysis reveals that cognitively normal speakers produce more semantically precise object- and scene-descriptive language, whereas Alzheimer's speech is characterized by vagueness, discourse markers, and increased hesitation patterns.
These findings suggest that high-quality ASR can enable simple, interpretable lexical models to achieve competitive Alzheimer's detection performance without explicit acoustic modeling. The study provides a reproducible benchmark pipeline and highlights ASR selection as a critical modeling decision in clinical speech-based artificial intelligence systems. 

---
# MOSS-TTS Technical Report 

**Authors**: Yitian Gong, Botian Jiang, Yiwei Zhao, Yucheng Yuan, Kuangwei Chen, Yaozhou Jiang, Cheng Chang, Dong Hong, Mingshu Chen, Ruixiao Li, Yiyang Zhang, Yang Gao, Hanfu Chen, Ke Chen, Songlin Wang, Xiaogui Yang, Yuqian Zhang, Kexin Huang, ZhengYuan Lin, Kang Yu, Ziqi Chen, Jin Wang, Zhaoye Fei, Qinyuan Cheng, Shimin Li, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.18090)  

**Abstract**: This technical report presents MOSS-TTS, a speech generation foundation model built on a scalable recipe: discrete audio tokens, autoregressive modeling, and large-scale pretraining. Built on MOSS-Audio-Tokenizer, a causal Transformer tokenizer that compresses 24 kHz audio to 12.5 fps with variable-bitrate RVQ and unified semantic-acoustic representations, we release two complementary generators: MOSS-TTS, which emphasizes structural simplicity, scalability, and long-context/control-oriented deployment, and MOSS-TTS-Local-Transformer, which introduces a frame-local autoregressive module for higher modeling efficiency, stronger speaker preservation, and a shorter time to first audio. Across multilingual and open-domain settings, MOSS-TTS supports zero-shot voice cloning, token-level duration control, phoneme-/pinyin-level pronunciation control, smooth code-switching, and stable long-form generation. This report summarizes the design, training recipe, and empirical characteristics of the released models. 

---
# ProKWS: Personalized Keyword Spotting via Collaborative Learning of Phonemes and Prosody 

**Authors**: Jianan Pan, Yuanming Zhang, Kejie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18024)  

**Abstract**: Current keyword spotting systems primarily use phoneme-level matching to distinguish confusable words but ignore user-specific pronunciation traits like prosody (intonation, stress, rhythm). This paper presents ProKWS, a novel framework integrating fine-grained phoneme learning with personalized prosody modeling. We design a dual-stream encoder where one stream derives robust phonemic representations through contrastive learning, while the other extracts speaker-specific prosodic patterns. A collaborative fusion module dynamically combines phonemic and prosodic information, enhancing adaptability across acoustic environments. Experiments show ProKWS delivers highly competitive performance, comparable to state-of-the-art models on standard benchmarks and demonstrates strong robustness for personalized keywords with tone and intent variations. 

---
# PCOV-KWS: Multi-task Learning for Personalized Customizable Open Vocabulary Keyword Spotting 

**Authors**: Jianan Pan, Kejie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18023)  

**Abstract**: As advancements in technologies like Internet of Things (IoT), Automatic Speech Recognition (ASR), Speaker Verification (SV), and Text-to-Speech (TTS) lead to increased usage of intelligent voice assistants, the demand for privacy and personalization has escalated. In this paper, we introduce a multi-task learning framework for personalized, customizable open-vocabulary Keyword Spotting (PCOV-KWS). This framework employs a lightweight network to simultaneously perform Keyword Spotting (KWS) and SV to address personalized KWS requirements. We have integrated a training criterion distinct from softmax-based loss, transforming multi-class classification into multiple binary classifications, which eliminates inter-category competition, while an optimization strategy for multi-task loss weighting is employed during training. We evaluated our PCOV-KWS system in multiple datasets, demonstrating that it outperforms the baselines in evaluation results, while also requiring fewer parameters and lower computational resources. 

---
# Frayed RoPE and Long Inputs: A Geometric Perspective 

**Authors**: Davis Wertheimer, Aozhong Zhang, Derrick Liu, Penghang Yin, Naigang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18017)  

**Abstract**: Rotary Positional Embedding (RoPE) is a widely adopted technique for encoding position in language models, which, while effective, causes performance breakdown when input length exceeds training length. Prior analyses assert (rightly) that long inputs cause channels to rotate ``out of distribution,'' but it is not clear how extra rotation relates to or causes pathological behavior. Through empirical and theoretical analysis we advance a unified geometric understanding of attention behavior with RoPE. We find that attention induces tight clustering of separated key and query latent point clouds, allowing for creation of sink tokens: placeholders that allow attention heads to avoid token mixing when not required. RoPE applied to longer inputs damages this key/query cluster separation, producing pathological behavior by inhibiting sink token functionality. From this geometric perspective, we propose RoPE-ID (In Distribution), a straightforward modification that allows attention layers to generalize to longer inputs out of the box: apply RoPE with high frequency to a subset of channels. We demonstrate the effectiveness of RoPE-ID for extended inputs using 1B and 3B parameter Transformers on the LongBench and RULER information retrieval benchmarks. 

---
