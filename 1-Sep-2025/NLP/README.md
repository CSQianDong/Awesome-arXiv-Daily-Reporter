# Going over Fine Web with a Fine-Tooth Comb: Technical Report of Indexing Fine Web for Problematic Content Search and Retrieval 

**Authors**: Inés Altemir Marinas, Anastasiia Kucherenko, Andrei Kucharavy  

**Link**: [PDF](https://arxiv.org/pdf/2508.21788)  

**Abstract**: Large language models (LLMs) rely heavily on web-scale datasets like Common Crawl, which provides over 80\% of training data for some modern models. However, the indiscriminate nature of web crawling raises challenges in data quality, safety, and ethics. Despite the critical importance of training data quality, prior research on harmful content has been limited to small samples due to computational constraints. This project presents a framework for indexing and analyzing LLM training datasets using an ElasticSearch-based pipeline. We apply it to SwissAI's FineWeb-2 corpus (1.5TB, four languages), achieving fast query performance--most searches in milliseconds, all under 2 seconds. Our work demonstrates real-time dataset analysis, offering practical tools for safer, more accountable AI systems. 

---
# PiCSAR: Probabilistic Confidence Selection And Ranking 

**Authors**: Joshua Ong Jun Leang, Zheng Zhao, Aryo Pradipta Gema, Sohee Yang, Wai-Chung Kwan, Xuanli He, Wenda Li, Pasquale Minervini, Eleonora Giunchiglia, Shay B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2508.21787)  

**Abstract**: Best-of-n sampling improves the accuracy of large language models (LLMs) and large reasoning models (LRMs) by generating multiple candidate solutions and selecting the one with the highest reward. The key challenge for reasoning tasks is designing a scoring function that can identify correct reasoning chains without access to ground-truth answers. We propose Probabilistic Confidence Selection And Ranking (PiCSAR): a simple, training-free method that scores each candidate generation using the joint log-likelihood of the reasoning and final answer. The joint log-likelihood of the reasoning and final answer naturally decomposes into reasoning confidence and answer confidence. PiCSAR achieves substantial gains across diverse benchmarks (+10.18 on MATH500, +9.81 on AIME2025), outperforming baselines with at least 2x fewer samples in 16 out of 20 comparisons. Our analysis reveals that correct reasoning chains exhibit significantly higher reasoning and answer confidence, justifying the effectiveness of PiCSAR. 

---
# Reasoning-Intensive Regression 

**Authors**: Diane Tchuindjo, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2508.21762)  

**Abstract**: AI researchers and practitioners increasingly apply large language models (LLMs) to what we call reasoning-intensive regression (RiR), i.e. deducing subtle numerical properties from text. Unlike standard language regression tasks, e.g. for sentiment or similarity, RiR often appears instead in ad-hoc problems like rubric-based scoring or domain-specific retrieval, where much deeper analysis of text is required while only limited task-specific training data and computation are available. We cast three realistic problems as RiR tasks to establish an initial benchmark, and use that to test our hypothesis that prompting frozen LLMs and finetuning Transformer encoders via gradient descent will both often struggle in RiR. We then propose MENTAT, a simple and lightweight method that combines batch-reflective prompt optimization with neural ensemble learning. MENTAT achieves up to 65% improvement over both baselines, though substantial room remains for future advances in RiR. 

---
# Not All Parameters Are Created Equal: Smart Isolation Boosts Fine-Tuning Performance 

**Authors**: Yao Wang, Di Liang, Minlong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.21741)  

**Abstract**: Supervised fine-tuning (SFT) is a pivotal approach to adapting large language models (LLMs) for downstream tasks; however, performance often suffers from the ``seesaw phenomenon'', where indiscriminate parameter updates yield progress on certain tasks at the expense of others. To address this challenge, we propose a novel \emph{Core Parameter Isolation Fine-Tuning} (CPI-FT) framework. Specifically, we first independently fine-tune the LLM on each task to identify its core parameter regions by quantifying parameter update magnitudes. Tasks with similar core regions are then grouped based on region overlap, forming clusters for joint modeling. We further introduce a parameter fusion technique: for each task, core parameters from its individually fine-tuned model are directly transplanted into a unified backbone, while non-core parameters from different tasks are smoothly integrated via Spherical Linear Interpolation (SLERP), mitigating destructive interference. A lightweight, pipelined SFT training phase using mixed-task data is subsequently employed, while freezing core regions from prior tasks to prevent catastrophic forgetting. Extensive experiments on multiple public benchmarks demonstrate that our approach significantly alleviates task interference and forgetting, consistently outperforming vanilla multi-task and multi-stage fine-tuning baselines. 

---
# Is this chart lying to me? Automating the detection of misleading visualizations 

**Authors**: Jonathan Tonglet, Jan Zimny, Tinne Tuytelaars, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2508.21675)  

**Abstract**: Misleading visualizations are a potent driver of misinformation on social media and the web. By violating chart design principles, they distort data and lead readers to draw inaccurate conclusions. Prior work has shown that both humans and multimodal large language models (MLLMs) are frequently deceived by such visualizations. Automatically detecting misleading visualizations and identifying the specific design rules they violate could help protect readers and reduce the spread of misinformation. However, the training and evaluation of AI models has been limited by the absence of large, diverse, and openly available datasets. In this work, we introduce Misviz, a benchmark of 2,604 real-world visualizations annotated with 12 types of misleaders. To support model training, we also release Misviz-synth, a synthetic dataset of 81,814 visualizations generated using Matplotlib and based on real-world data tables. We perform a comprehensive evaluation on both datasets using state-of-the-art MLLMs, rule-based systems, and fine-tuned classifiers. Our results reveal that the task remains highly challenging. We release Misviz, Misviz-synth, and the accompanying code. 

---
# QZhou-Embedding Technical Report 

**Authors**: Peng Yu, En Xu, Bin Chen, Haibiao Chen, Yinfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21632)  

**Abstract**: We present QZhou-Embedding, a general-purpose contextual text embedding model with exceptional text representation capabilities. Built upon the Qwen2.5-7B-Instruct foundation model, we designed a unified multi-task framework comprising specialized data transformation and training strategies. The data transformation scheme enables the incorporation of more diverse textual training datasets, while the task-specific training strategies enhance model learning efficiency. We developed a data synthesis pipeline leveraging LLM API, incorporating techniques such as paraphrasing, augmentation, and hard negative example generation to improve the semantic richness and sample difficulty of the training set. Additionally, we employ a two-stage training strategy, comprising initial retrieval-focused pretraining followed by full-task fine-tuning, enabling the embedding model to extend its capabilities based on robust retrieval performance. Our model achieves state-of-the-art results on the MTEB and CMTEB benchmarks, ranking first on both leaderboards (August 27 2025), and simultaneously achieves state-of-the-art performance on tasks including reranking, clustering, etc. Our findings demonstrate that higher-quality, more diverse data is crucial for advancing retrieval model performance, and that leveraging LLMs generative capabilities can further optimize data quality for embedding model breakthroughs. Our model weights are released on HuggingFace under Apache 2.0 license. For reproducibility, we provide evaluation code and instructions on GitHub. 

---
# Personality Matters: User Traits Predict LLM Preferences in Multi-Turn Collaborative Tasks 

**Authors**: Sarfaroz Yunusov, Kaige Chen, Kazi Nishat Anwar, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2508.21628)  

**Abstract**: As Large Language Models (LLMs) increasingly integrate into everyday workflows, where users shape outcomes through multi-turn collaboration, a critical question emerges: do users with different personality traits systematically prefer certain LLMs over others? We conducted a study with 32 participants evenly distributed across four Keirsey personality types, evaluating their interactions with GPT-4 and Claude 3.5 across four collaborative tasks: data analysis, creative writing, information retrieval, and writing assistance. Results revealed significant personality-driven preferences: Rationals strongly preferred GPT-4, particularly for goal-oriented tasks, while idealists favored Claude 3.5, especially for creative and analytical tasks. Other personality types showed task-dependent preferences. Sentiment analysis of qualitative feedback confirmed these patterns. Notably, aggregate helpfulness ratings were similar across models, showing how personality-based analysis reveals LLM differences that traditional evaluations miss. 

---
# Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning 

**Authors**: Zinan Tang, Xin Gao, Qizhi Pei, Zhuoshi Pan, Mengzhang Cai, Jiang Wu, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21589)  

**Abstract**: Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our \method consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are coming soon. 

---
# A Survey on Current Trends and Recent Advances in Text Anonymization 

**Authors**: Tobias Deußer, Lorenz Sparrenberg, Armin Berger, Max Hahnbück, Christian Bauckhage, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2508.21587)  

**Abstract**: The proliferation of textual data containing sensitive personal information across various domains requires robust anonymization techniques to protect privacy and comply with regulations, while preserving data usability for diverse and crucial downstream tasks. This survey provides a comprehensive overview of current trends and recent advances in text anonymization techniques. We begin by discussing foundational approaches, primarily centered on Named Entity Recognition, before examining the transformative impact of Large Language Models, detailing their dual role as sophisticated anonymizers and potent de-anonymization threats. The survey further explores domain-specific challenges and tailored solutions in critical sectors such as healthcare, law, finance, and education. We investigate advanced methodologies incorporating formal privacy models and risk-aware frameworks, and address the specialized subfield of authorship anonymization. Additionally, we review evaluation frameworks, comprehensive metrics, benchmarks, and practical toolkits for real-world deployment of anonymization solutions. This review consolidates current knowledge, identifies emerging trends and persistent challenges, including the evolving privacy-utility trade-off, the need to address quasi-identifiers, and the implications of LLM capabilities, and aims to guide future research directions for both academics and practitioners in this field. 

---
# L3Cube-MahaSTS: A Marathi Sentence Similarity Dataset and Models 

**Authors**: Aishwarya Mirashi, Ananya Joshi, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.21569)  

**Abstract**: We present MahaSTS, a human-annotated Sentence Textual Similarity (STS) dataset for Marathi, along with MahaSBERT-STS-v2, a fine-tuned Sentence-BERT model optimized for regression-based similarity scoring. The MahaSTS dataset consists of 16,860 Marathi sentence pairs labeled with continuous similarity scores in the range of 0-5. To ensure balanced supervision, the dataset is uniformly distributed across six score-based buckets spanning the full 0-5 range, thus reducing label bias and enhancing model stability. We fine-tune the MahaSBERT model on this dataset and benchmark its performance against other alternatives like MahaBERT, MuRIL, IndicBERT, and IndicSBERT. Our experiments demonstrate that MahaSTS enables effective training for sentence similarity tasks in Marathi, highlighting the impact of human-curated annotations, targeted fine-tuning, and structured supervision in low-resource settings. The dataset and model are publicly shared at this https URL 

---
# HSFN: Hierarchical Selection for Fake News Detection building Heterogeneous Ensemble 

**Authors**: Sara B. Coutinho, Rafael M.O. Cruz, Francimaria R. S. Nascimento, George D. C. Cavalcanti  

**Link**: [PDF](https://arxiv.org/pdf/2508.21482)  

**Abstract**: Psychological biases, such as confirmation bias, make individuals particularly vulnerable to believing and spreading fake news on social media, leading to significant consequences in domains such as public health and politics. Machine learning-based fact-checking systems have been widely studied to mitigate this problem. Among them, ensemble methods are particularly effective in combining multiple classifiers to improve robustness. However, their performance heavily depends on the diversity of the constituent classifiers-selecting genuinely diverse models remains a key challenge, especially when models tend to learn redundant patterns. In this work, we propose a novel automatic classifier selection approach that prioritizes diversity, also extended by performance. The method first computes pairwise diversity between classifiers and applies hierarchical clustering to organize them into groups at different levels of granularity. A HierarchySelect then explores these hierarchical levels to select one pool of classifiers per level, each representing a distinct intra-pool diversity. The most diverse pool is identified and selected for ensemble construction from these. The selection process incorporates an evaluation metric reflecting each classifiers's performance to ensure the ensemble also generalises well. We conduct experiments with 40 heterogeneous classifiers across six datasets from different application domains and with varying numbers of classes. Our method is compared against the Elbow heuristic and state-of-the-art baselines. Results show that our approach achieves the highest accuracy on two of six datasets. The implementation details are available on the project's repository: this https URL . 

---
# Igniting Creative Writing in Small Language Models: LLM-as-a-Judge versus Multi-Agent Refined Rewards 

**Authors**: Xiaolong Wei, Bo Lu, Xingyu Zhang, Zhejun Zhao, Dongdong Shen, Long Xia, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21476)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable creative writing capabilities, yet their substantial computational demands hinder widespread use. Enhancing Small Language Models (SLMs) offers a promising alternative, but current methods like Supervised Fine-Tuning (SFT) struggle with novelty, and Reinforcement Learning from Human Feedback (RLHF) is costly. This paper explores two distinct AI-driven reward strategies within a Reinforcement Learning from AI Feedback (RLAIF) framework to ignite the creative writing of a 7B-parameter SLM, specifically for generating Chinese greetings. The first strategy employs a RM trained on high-quality preference data curated by a novel multi-agent rejection sampling framework designed for creative tasks. The second, more novel strategy utilizes a principle-guided LLM-as-a-Judge, whose reward function is optimized via an adversarial training scheme with a reflection mechanism, to directly provide reward signals. Comprehensive experiments reveal that while both approaches significantly enhance creative output over baselines, the principle-guided LLM-as-a-Judge demonstrably yields superior generation quality. Furthermore, it offers notable advantages in training efficiency and reduced dependency on human-annotated data, presenting a more scalable and effective path towards creative SLMs. Our automated evaluation methods also exhibit strong alignment with human judgments. Our code and data are publicly available at this https URL. 

---
# Beyond the Surface: Probing the Ideological Depth of Large Language Models 

**Authors**: Shariar Kabir, Kevin Esterling, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.21448)  

**Abstract**: Large Language Models (LLMs) have demonstrated pronounced ideological leanings, yet the stability and depth of these positions remain poorly understood. Surface-level responses can often be manipulated through simple prompt engineering, calling into question whether they reflect a coherent underlying ideology. This paper investigates the concept of "ideological depth" in LLMs, defined as the robustness and complexity of their internal political representations. We employ a dual approach: first, we measure the "steerability" of two well-known open-source LLMs using instruction prompting and activation steering. We find that while some models can easily switch between liberal and conservative viewpoints, others exhibit resistance or an increased rate of refusal, suggesting a more entrenched ideological structure. Second, we probe the internal mechanisms of these models using Sparse Autoencoders (SAEs). Preliminary analysis reveals that models with lower steerability possess more distinct and abstract ideological features. Our evaluations reveal that one model can contain 7.3x more political features than another model of similar size. This allows targeted ablation of a core political feature in an ideologically "deep" model, leading to consistent, logical shifts in its reasoning across related topics, whereas the same intervention in a "shallow" model results in an increase in refusal outputs. Our findings suggest that ideological depth is a quantifiable property of LLMs and that steerability serves as a valuable window into their latent political architecture. 

---
# Discovering Semantic Subdimensions through Disentangled Conceptual Representations 

**Authors**: Yunhao Zhang, Shaonan Wang, Nan Lin, Xinyi Dong, Chong Li, Chengqing Zong  

**Link**: [PDF](https://arxiv.org/pdf/2508.21436)  

**Abstract**: Understanding the core dimensions of conceptual semantics is fundamental to uncovering how meaning is organized in language and the brain. Existing approaches often rely on predefined semantic dimensions that offer only broad representations, overlooking finer conceptual distinctions. This paper proposes a novel framework to investigate the subdimensions underlying coarse-grained semantic dimensions. Specifically, we introduce a Disentangled Continuous Semantic Representation Model (DCSRM) that decomposes word embeddings from large language models into multiple sub-embeddings, each encoding specific semantic information. Using these sub-embeddings, we identify a set of interpretable semantic subdimensions. To assess their neural plausibility, we apply voxel-wise encoding models to map these subdimensions to brain activation. Our work offers more fine-grained interpretable semantic subdimensions of conceptual meaning. Further analyses reveal that semantic dimensions are structured according to distinct principles, with polarity emerging as a key factor driving their decomposition into subdimensions. The neural correlates of the identified subdimensions support their cognitive and neuroscientific plausibility. 

---
# Med-RewardBench: Benchmarking Reward Models and Judges for Medical Multimodal Large Language Models 

**Authors**: Meidan Ding, Jipeng Zhang, Wenxuan Wang, Cheng-Yi Li, Wei-Chieh Fang, Hsin-Yu Wu, Haiqin Zhong, Wenting Chen, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.21430)  

**Abstract**: Multimodal large language models (MLLMs) hold significant potential in medical applications, including disease diagnosis and clinical decision-making. However, these tasks require highly accurate, context-sensitive, and professionally aligned responses, making reliable reward models and judges critical. Despite their importance, medical reward models (MRMs) and judges remain underexplored, with no dedicated benchmarks addressing clinical requirements. Existing benchmarks focus on general MLLM capabilities or evaluate models as solvers, neglecting essential evaluation dimensions like diagnostic accuracy and clinical relevance. To address this, we introduce Med-RewardBench, the first benchmark specifically designed to evaluate MRMs and judges in medical scenarios. Med-RewardBench features a multimodal dataset spanning 13 organ systems and 8 clinical departments, with 1,026 expert-annotated cases. A rigorous three-step process ensures high-quality evaluation data across six clinically critical dimensions. We evaluate 32 state-of-the-art MLLMs, including open-source, proprietary, and medical-specific models, revealing substantial challenges in aligning outputs with expert judgment. Additionally, we develop baseline models that demonstrate substantial performance improvements through fine-tuning. 

---
# Automatic Reviewers Fail to Detect Faulty Reasoning in Research Papers: A New Counterfactual Evaluation Framework 

**Authors**: Nils Dycke, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2508.21422)  

**Abstract**: Large Language Models (LLMs) have great potential to accelerate and support scholarly peer review and are increasingly used as fully automatic review generators (ARGs). However, potential biases and systematic errors may pose significant risks to scientific integrity; understanding the specific capabilities and limitations of state-of-the-art ARGs is essential. We focus on a core reviewing skill that underpins high-quality peer review: detecting faulty research logic. This involves evaluating the internal consistency between a paper's results, interpretations, and claims. We present a fully automated counterfactual evaluation framework that isolates and tests this skill under controlled conditions. Testing a range of ARG approaches, we find that, contrary to expectation, flaws in research logic have no significant effect on their output reviews. Based on our findings, we derive three actionable recommendations for future work and release our counterfactual dataset and evaluation framework publicly. 

---
# AllSummedUp: un framework open-source pour comparer les metriques d'evaluation de resume 

**Authors**: Tanguy Herserant, Vincent Guigue  

**Link**: [PDF](https://arxiv.org/pdf/2508.21389)  

**Abstract**: This paper investigates reproducibility challenges in automatic text summarization evaluation. Based on experiments conducted across six representative metrics ranging from classical approaches like ROUGE to recent LLM-based methods (G-Eval, SEval-Ex), we highlight significant discrepancies between reported performances in the literature and those observed in our experimental setting. We introduce a unified, open-source framework, applied to the SummEval dataset and designed to support fair and transparent comparison of evaluation metrics. Our results reveal a structural trade-off: metrics with the highest alignment with human judgments tend to be computationally intensive and less stable across runs. Beyond comparative analysis, this study highlights key concerns about relying on LLMs for evaluation, stressing their randomness, technical dependencies, and limited reproducibility. We advocate for more robust evaluation protocols including exhaustive documentation and methodological standardization to ensure greater reliability in automatic summarization assessment. 

---
# Normality and the Turing Test 

**Authors**: Alexandre Kabbach  

**Link**: [PDF](https://arxiv.org/pdf/2508.21382)  

**Abstract**: This paper proposes to revisit the Turing test through the concept of normality. Its core argument is that the statistical interpretation of the normal--understood as the average both in the normative and mathematical sense of the term--proves useful for understanding the Turing test in at least two ways. First, in the sense that the Turing test targets normal/average rather than exceptional human intelligence, so that successfully passing the test requires building machines that "make mistakes" and display imperfect behavior just like normal/average humans. Second, in the sense that the Turing test is a statistical test where judgments of intelligence are never carried out by a single "average" judge (understood as non-expert) but always by a full jury. As such, the notion of "average human interrogator" that Turing talks about in his original paper should be understood primarily as referring to a mathematical abstraction made of the normalized aggregate of individual judgments of multiple judges. In short, this paper argues that the Turing test is a test of normal intelligence as assessed by a normal judge characterizing the average judgment of a pool of human interrogators. Its conclusions are twofold. First, it argues that large language models such as ChatGPT are unlikely to pass the Turing test as those models precisely target exceptional rather than normal/average human intelligence. As such, they constitute models of what it proposes to call artificial smartness rather than artificial intelligence per se. Second, it argues that the core question of whether the Turing test can contribute anything to the understanding of human cognition is that of whether the human mind is really reducible to the normal/average mind--a question which largely extends beyond the Turing test itself and questions the conceptual underpinnings of the normalist paradigm it belongs to. 

---
# Challenges and Applications of Large Language Models: A Comparison of GPT and DeepSeek family of models 

**Authors**: Shubham Sharma, Sneha Tuli, Narendra Badam  

**Link**: [PDF](https://arxiv.org/pdf/2508.21377)  

**Abstract**: Large Language Models (LLMs) are transforming AI across industries, but their development and deployment remain complex. This survey reviews 16 key challenges in building and using LLMs and examines how these challenges are addressed by two state-of-the-art models with unique approaches: OpenAI's closed source GPT-4o (May 2024 update) and DeepSeek-V3-0324 (March 2025), a large open source Mixture-of-Experts model. Through this comparison, we showcase the trade-offs between closed source models (robust safety, fine-tuned reliability) and open source models (efficiency, adaptability). We also explore LLM applications across different domains (from chatbots and coding tools to healthcare and education), highlighting which model attributes are best suited for each use case. This article aims to guide AI researchers, developers, and decision-makers in understanding current LLM capabilities, limitations, and best practices. 

---
# BLUEX Revisited: Enhancing Benchmark Coverage with Automatic Captioning 

**Authors**: João Guilherme Alves Santos, Giovana Kerche Bonás, Thales Sales Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2508.21294)  

**Abstract**: With the growing capabilities of Large Language Models (LLMs), there is an increasing need for robust evaluation methods, especially in multilingual and non-English contexts. We present an updated version of the BLUEX dataset, now including 2024-2025 exams and automatically generated image captions using state-of-the-art models, enhancing its relevance for data contamination studies in LLM pretraining. Captioning strategies increase accessibility to text-only models by more than 40%, producing 1,422 usable questions, more than doubling the number in the original BLUEX. We evaluated commercial and open-source LLMs and their ability to leverage visual context through captions. 

---
# Efficient Code Embeddings from Code Generation Models 

**Authors**: Daria Kryvosheieva, Saba Sturua, Michael Günther, Scott Martens, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21290)  

**Abstract**: jina-code-embeddings is a novel code embedding model suite designed to retrieve code from natural language queries, perform technical question-answering, and identify semantically similar code snippets across programming languages. It makes innovative use of an autoregressive backbone pre-trained on both text and code, generating embeddings via last-token pooling. We outline the training recipe and demonstrate state-of-the-art performance despite the relatively small size of the models, validating this approach to code embedding model construction. 

---
# Decoding Memories: An Efficient Pipeline for Self-Consistency Hallucination Detection 

**Authors**: Weizhi Gao, Xiaorui Liu, Feiyi Wang, Dan Lu, Junqi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.21228)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance in both research and real-world applications, but they still struggle with hallucination. Existing hallucination detection methods often perform poorly on sentence-level generation or rely heavily on domain-specific knowledge. While self-consistency approaches help address these limitations, they incur high computational costs due to repeated generation. In this paper, we conduct the first study on identifying redundancy in self-consistency methods, manifested as shared prefix tokens across generations, and observe that non-exact-answer tokens contribute minimally to the semantic content. Based on these insights, we propose a novel Decoding Memory Pipeline (DMP) that accelerates generation through selective inference and annealed decoding. Being orthogonal to the model, dataset, decoding strategy, and self-consistency baseline, our DMP consistently improves the efficiency of multi-response generation and holds promise for extension to alignment and reasoning tasks. Extensive experiments show that our method achieves up to a 3x speedup without sacrificing AUROC performance. 

---
# Do Self-Supervised Speech Models Exhibit the Critical Period Effects in Language Acquisition? 

**Authors**: Yurie Koga, Shunsuke Kando, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2508.21210)  

**Abstract**: This paper investigates whether the Critical Period (CP) effects in human language acquisition are observed in self-supervised speech models (S3Ms). CP effects refer to greater difficulty in acquiring a second language (L2) with delayed L2 exposure onset, and greater retention of their first language (L1) with delayed L1 exposure offset. While previous work has studied these effects using textual language models, their presence in speech models remains underexplored despite the central role of spoken language in human language acquisition. We train S3Ms with varying L2 training onsets and L1 training offsets on child-directed speech and evaluate their phone discrimination performance. We find that S3Ms do not exhibit clear evidence of either CP effects in terms of phonological acquisition. Notably, models with delayed L2 exposure onset tend to perform better on L2 and delayed L1 exposure offset leads to L1 forgetting. 

---
# Enhancing Robustness of Autoregressive Language Models against Orthographic Attacks via Pixel-based Approach 

**Authors**: Han Yang, Jian Lan, Yihong Liu, Hinrich Schütze, Thomas Seidl  

**Link**: [PDF](https://arxiv.org/pdf/2508.21206)  

**Abstract**: Autoregressive language models are vulnerable to orthographic attacks, where input text is perturbed with characters from multilingual alphabets, leading to substantial performance degradation. This vulnerability primarily stems from the out-of-vocabulary issue inherent in subword tokenizers and their embeddings. To address this limitation, we propose a pixel-based generative language model that replaces the text-based embeddings with pixel-based representations by rendering words as individual images. This design provides stronger robustness to noisy inputs, while an extension of compatibility to multilingual text across diverse writing systems. We evaluate the proposed method on the multilingual LAMBADA dataset, WMT24 dataset and the SST-2 benchmark, demonstrating both its resilience to orthographic noise and its effectiveness in multilingual settings. 

---
# Improving Aviation Safety Analysis: Automated HFACS Classification Using Reinforcement Learning with Group Relative Policy Optimization 

**Authors**: Arash Ahmadi, Sarah Sharif, Yaser Banad  

**Link**: [PDF](https://arxiv.org/pdf/2508.21201)  

**Abstract**: Analyzing the human factors behind aviation accidents is crucial for preventing future incidents, yet traditional methods using the Human Factors Analysis and Classification System (HFACS) are limited by scalability and consistency. To address this, we introduce an automated HFACS classification framework for aviation safety analysis that utilizes Reinforcement Learning with Group Relative Policy Optimization (GRPO) to fine-tune a Llama-3.1 8B language model. Our approach incorporates a multi-component reward system tailored for aviation safety analysis and integrates synthetic data generation to overcome class imbalance in accident datasets. The resulting GRPO-optimized model achieved noticeable performance gains, including a 350% increase in exact match accuracy (from 0.0400 to 0.1800) and an improved partial match accuracy of 0.8800. Significantly, our specialized model outperforms state-of-the-art LLMs (Large Language Models), including GPT-5-mini and Gemini-2.5-fiash, on key metrics. This research also proposes exact match accuracy in multi-label HFACS classification problem as a new benchmarking methodology to evaluate the advanced reasoning capabilities of language models. Ultimately, our work validates that smaller, domain-optimized models can provide a computationally efficient and better solution for critical safety analysis. This approach makes powerful, low-latency deployment on resource-constrained edge devices feasible. 

---
# BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design 

**Authors**: Deepro Choudhury, Sinead Williamson, Adam Goliński, Ning Miao, Freddie Bickford Smith, Michael Kirchhof, Yizhe Zhang, Tom Rainforth  

**Link**: [PDF](https://arxiv.org/pdf/2508.21184)  

**Abstract**: We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated in a principled way using a probabilistic model derived from the LLM's belief distribution and provide detailed insights into key decisions in its construction. Further key to the success of BED-LLM are a number of specific innovations, such as a carefully designed estimator for the EIG, not solely relying on in-context updates for conditioning on previous responses, and a targeted strategy for proposing candidate queries. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20-questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies. 

---
# Quantifying Label-Induced Bias in Large Language Model Self- and Cross-Evaluations 

**Authors**: Muskan Saraf, Sajjad Rezvani Boroujeni, Justin Beaudry, Hossein Abedi, Tom Bush  

**Link**: [PDF](https://arxiv.org/pdf/2508.21164)  

**Abstract**: Large language models (LLMs) are increasingly used to evaluate outputs, yet their judgments may be influenced. This study examines bias in self- and cross-model evaluations by ChatGPT, Gemini, and Claude under four conditions: no labels, true labels, and two false-label scenarios. Blog posts authored by each model were evaluated by all three using both overall preference voting and quality ratings for Coherence, Informativeness, and Conciseness, with all scores expressed as percentages for direct comparison. Results reveal striking asymmetries: the "Claude" label consistently boosts scores, while the "Gemini" label consistently depresses them, regardless of actual content. False labels frequently reversed rankings, producing shifts of up to 50 percentage points in preference votes and up to 12 percentage points in converted quality ratings. Gemini's self-scores collapsed under true labels, while Claude's self-preference intensified. These findings show that perceived model identity can heavily distort high-level judgments and subtly influence detailed quality ratings, underscoring the need for blind or multimodel evaluation protocols to ensure fairness in LLM benchmarking. 

---
# A Survey of Scientific Large Language Models: From Data Foundations to Agent Frontiers 

**Authors**: Ming Hu, Chenglong Ma, Wei Li, Wanghan Xu, Jiamin Wu, Jucheng Hu, Tianbin Li, Guohang Zhuang, Jiaqi Liu, Yingzhou Lu, Ying Chen, Chaoyang Zhang, Cheng Tan, Jie Ying, Guocheng Wu, Shujian Gao, Pengcheng Chen, Jiashi Lin, Haitao Wu, Lulu Chen, Fengxiang Wang, Yuanyuan Zhang, Xiangyu Zhao, Feilong Tang, Encheng Su, Junzhi Ning, Xinyao Liu, Ye Du, Changkai Ji, Cheng Tang, Huihui Xu, Ziyang Chen, Ziyan Huang, Jiyao Liu, Pengfei Jiang, Yizhou Wang, Chen Tang, Jianyu Wu, Yuchen Ren, Siyuan Yan, Zhonghua Wang, Zhongxing Xu, Shiyan Su, Shangquan Sun, Runkai Zhao, Zhisheng Zhang, Yu Liu, Fudi Wang, Yuanfeng Ji, Yanzhou Su, Hongming Shan, Chunmei Feng, Jiahao Xu, Jiangtao Yan, Wenhao Tang, Diping Song, Lihao Liu, Yanyan Huang, Lequan Yu, Bin Fu, Shujun Wang, Xiaomeng Li, Xiaowei Hu, Yun Gu, Ben Fei, Zhongying Deng, Benyou Wang, Yuewen Cao, Minjie Shen, Haodong Duan, Jie Xu, Yirong Chen, Fang Yan, Hongxia Hao, Jielan Li, Jiajun Du, Yanbo Wang, Imran Razzak, Chi Zhang, Lijun Wu, Conghui He, Zhaohui Lu, Jinhai Huang, Yihao Liu, Fenghua Ling, Yuqiang Li, Aoran Wang, Qihao Zheng, Nanqing Dong, Tianfan Fu, Dongzhan Zhou, Yan Lu, Wenlong Zhang, Jin Ye, Jianfei Cai, Wanli Ouyang, Yu Qiao, Zongyuan Ge, Shixiang Tang, Junjun He  

**Link**: [PDF](https://arxiv.org/pdf/2508.21148)  

**Abstract**: Scientific Large Language Models (Sci-LLMs) are transforming how knowledge is represented, integrated, and applied in scientific research, yet their progress is shaped by the complex nature of scientific data. This survey presents a comprehensive, data-centric synthesis that reframes the development of Sci-LLMs as a co-evolution between models and their underlying data substrate. We formulate a unified taxonomy of scientific data and a hierarchical model of scientific knowledge, emphasizing the multimodal, cross-scale, and domain-specific challenges that differentiate scientific corpora from general natural language processing datasets. We systematically review recent Sci-LLMs, from general-purpose foundations to specialized models across diverse scientific disciplines, alongside an extensive analysis of over 270 pre-/post-training datasets, showing why Sci-LLMs pose distinct demands -- heterogeneous, multi-scale, uncertainty-laden corpora that require representations preserving domain invariance and enabling cross-modal reasoning. On evaluation, we examine over 190 benchmark datasets and trace a shift from static exams toward process- and discovery-oriented assessments with advanced evaluation protocols. These data-centric analyses highlight persistent issues in scientific data development and discuss emerging solutions involving semi-automated annotation pipelines and expert validation. Finally, we outline a paradigm shift toward closed-loop systems where autonomous agents based on Sci-LLMs actively experiment, validate, and contribute to a living, evolving knowledge base. Collectively, this work provides a roadmap for building trustworthy, continually evolving artificial intelligence (AI) systems that function as a true partner in accelerating scientific discovery. 

---
# Can Multimodal LLMs Solve the Basic Perception Problems of Percept-V? 

**Authors**: Samrajnee Ghosh, Naman Agarwal, Hemanshu Garg, Chinmay Mittal, Mausam, Parag Singla  

**Link**: [PDF](https://arxiv.org/pdf/2508.21143)  

**Abstract**: The reasoning abilities of Multimodal Large Language Models (MLLMs) have garnered a lot of attention in recent times, with advances made in frontiers like coding, mathematics, and science. However, very limited experiments have been done to assess their performance in simple perception tasks performed over uncontaminated, generated images containing basic shapes and structures. To address this issue, the paper introduces a dataset, Percept-V, containing a total of 7200 program-generated images equally divided into 30 categories, each testing a combination of visual perception skills. Unlike previously proposed datasets, Percept-V comprises very basic tasks of varying complexity that test the perception abilities of MLLMs. This dataset is then tested on state-of-the-art MLLMs like GPT-4o, Gemini, and Claude as well as Large Reasoning Models (LRMs) like OpenAI o4-mini and DeepSeek R1 to gauge their performance. Contrary to the evidence that MLLMs excel in many complex tasks, our experiments show a significant drop in the models' performance with increasing problem complexity across all categories. An analysis of the performances also reveals that the tested MLLMs exhibit a similar trend in accuracy across categories, testing a particular cognitive skill and find some skills to be more difficult than others. 

---
# How Does Cognitive Bias Affect Large Language Models? A Case Study on the Anchoring Effect in Price Negotiation Simulations 

**Authors**: Yoshiki Takenami, Yin Jou Huang, Yugo Murawaki, Chenhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21137)  

**Abstract**: Cognitive biases, well-studied in humans, can also be observed in LLMs, affecting their reliability in real-world applications. This paper investigates the anchoring effect in LLM-driven price negotiations. To this end, we instructed seller LLM agents to apply the anchoring effect and evaluated negotiations using not only an objective metric but also a subjective metric. Experimental results show that LLMs are influenced by the anchoring effect like humans. Additionally, we investigated the relationship between the anchoring effect and factors such as reasoning and personality. It was shown that reasoning models are less prone to the anchoring effect, suggesting that the long chain of thought mitigates the effect. However, we found no significant correlation between personality traits and susceptibility to the anchoring effect. These findings contribute to a deeper understanding of cognitive biases in LLMs and to the realization of safe and responsible application of LLMs in society. 

---
# TrInk: Ink Generation with Transformer Network 

**Authors**: Zezhong Jin, Shubhang Desai, Xu Chen, Biyi Fang, Zhuoyi Huang, Zhe Li, Chong-Xin Gan, Xiao Tu, Man-Wai Mak, Yan Lu, Shujie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21098)  

**Abstract**: In this paper, we propose TrInk, a Transformer-based model for ink generation, which effectively captures global dependencies. To better facilitate the alignment between the input text and generated stroke points, we introduce scaled positional embeddings and a Gaussian memory mask in the cross-attention module. Additionally, we design both subjective and objective evaluation pipelines to comprehensively assess the legibility and style consistency of the generated handwriting. Experiments demonstrate that our Transformer-based model achieves a 35.56\% reduction in character error rate (CER) and an 29.66% reduction in word error rate (WER) on the IAM-OnDB dataset compared to previous methods. We provide an demo page with handwriting samples from TrInk and baseline models at: this https URL 

---
# Granite Embedding R2 Models 

**Authors**: Parul Awasthy, Aashka Trivedi, Yulong Li, Meet Doshi, Riyaz Bhat, Vignesh P, Vishwajeet Kumar, Yushu Yang, Bhavani Iyer, Abraham Daniels, Rudra Murthy, Ken Barker, Martin Franz, Madison Lee, Todd Ward, Salim Roukos, David Cox, Luis Lastras, Jaydeep Sen, Radu Florian  

**Link**: [PDF](https://arxiv.org/pdf/2508.21085)  

**Abstract**: We introduce the Granite Embedding R2 models, a comprehensive family of high-performance English encoder-based embedding models engineered for enterprise-scale dense retrieval applications. Building upon our first-generation release, these models deliver substantial improvements, including 16x expanded context length (8,192 tokens), state-of-the-art performance across diverse retrieval domains - text, code, long-document search, multi-turn conversational, and tabular data - and measurable speed advantages of 19-44\% over leading competitors while maintaining superior accuracy. Our release encompasses both bi-encoder and cross-encoder architectures, featuring a highly effective 22-layer retriever model and its efficient 12-layer counterpart, alongside a high-quality reranker model, all trained exclusively on enterprise-appropriate data with comprehensive governance oversight. The models demonstrate exceptional versatility across standard benchmarks, IBM-developed evaluation suites, and real-world enterprise use cases, establishing new performance standards for open-source embedding models. In an era where retrieval speed and accuracy are paramount for competitive advantage, the Granite R2 models deliver a compelling combination of cutting-edge performance, enterprise-ready licensing, and transparent data provenance that organizations require for mission-critical deployments. All models are publicly available under the Apache 2.0 license at this https URL, enabling unrestricted research and commercial use. 

---
# Mapping Toxic Comments Across Demographics: A Dataset from German Public Broadcasting 

**Authors**: Jan Fillies, Michael Peter Hoffmann, Rebecca Reichel, Roman Salzwedel, Sven Bodemer, Adrian Paschke  

**Link**: [PDF](https://arxiv.org/pdf/2508.21084)  

**Abstract**: A lack of demographic context in existing toxic speech datasets limits our understanding of how different age groups communicate online. In collaboration with funk, a German public service content network, this research introduces the first large-scale German dataset annotated for toxicity and enriched with platform-provided age estimates. The dataset includes 3,024 human-annotated and 30,024 LLM-annotated anonymized comments from Instagram, TikTok, and YouTube. To ensure relevance, comments were consolidated using predefined toxic keywords, resulting in 16.7\% labeled as problematic. The annotation pipeline combined human expertise with state-of-the-art language models, identifying key categories such as insults, disinformation, and criticism of broadcasting fees. The dataset reveals age-based differences in toxic speech patterns, with younger users favoring expressive language and older users more often engaging in disinformation and devaluation. This resource provides new opportunities for studying linguistic variation across demographics and supports the development of more equitable and age-aware content moderation systems. 

---
# CoBA: Counterbias Text Augmentation for Mitigating Various Spurious Correlations via Semantic Triples 

**Authors**: Kyohoon Jin, Juhwan Choi, Jungmin Yun, Junho Lee, Soojin Jang, Youngbin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.21083)  

**Abstract**: Deep learning models often learn and exploit spurious correlations in training data, using these non-target features to inform their predictions. Such reliance leads to performance degradation and poor generalization on unseen data. To address these limitations, we introduce a more general form of counterfactual data augmentation, termed counterbias data augmentation, which simultaneously tackles multiple biases (e.g., gender bias, simplicity bias) and enhances out-of-distribution robustness. We present CoBA: CounterBias Augmentation, a unified framework that operates at the semantic triple level: first decomposing text into subject-predicate-object triples, then selectively modifying these triples to disrupt spurious correlations. By reconstructing the text from these adjusted triples, CoBA generates counterbias data that mitigates spurious patterns. Through extensive experiments, we demonstrate that CoBA not only improves downstream task performance, but also effectively reduces biases and strengthens out-of-distribution resilience, offering a versatile and robust solution to the challenges posed by spurious correlations. 

---
# Why Stop at Words? Unveiling the Bigger Picture through Line-Level OCR 

**Authors**: Shashank Vempati, Nishit Anand, Gaurav Talebailkar, Arpan Garai, Chetan Arora  

**Link**: [PDF](https://arxiv.org/pdf/2508.21693)  

**Abstract**: Conventional optical character recognition (OCR) techniques segmented each character and then recognized. This made them prone to error in character segmentation, and devoid of context to exploit language models. Advances in sequence to sequence translation in last decade led to modern techniques first detecting words and then inputting one word at a time to a model to directly output full words as sequence of characters. This allowed better utilization of language models and bypass error-prone character segmentation step. We observe that the above transition in style has moved the bottleneck in accuracy to word segmentation. Hence, in this paper, we propose a natural and logical progression from word level OCR to line-level OCR. The proposal allows to bypass errors in word detection, and provides larger sentence context for better utilization of language models. We show that the proposed technique not only improves the accuracy but also efficiency of OCR. Despite our thorough literature survey, we did not find any public dataset to train and benchmark such shift from word to line-level OCR. Hence, we also contribute a meticulously curated dataset of 251 English page images with line-level annotations. Our experimentation revealed a notable end-to-end accuracy improvement of 5.4%, underscoring the potential benefits of transitioning towards line-level OCR, especially for document images. We also report a 4 times improvement in efficiency compared to word-based pipelines. With continuous improvements in large language models, our methodology also holds potential to exploit such advances. Project Website: this https URL 

---
# Summarize-Exemplify-Reflect: Data-driven Insight Distillation Empowers LLMs for Few-shot Tabular Classification 

**Authors**: Yifei Yuan, Jiatong Li, Weijia Zhang, Mohammad Aliannejadi, Evangelos Kanoulas, Renjun Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21561)  

**Abstract**: Recent studies show the promise of large language models (LLMs) for few-shot tabular classification but highlight challenges due to the variability in structured data. To address this, we propose distilling data into actionable insights to enable robust and effective classification by LLMs. Drawing inspiration from human learning processes, we introduce InsightTab, an insight distillation framework guided by principles of divide-and-conquer, easy-first, and reflective learning. Our approach integrates rule summarization, strategic exemplification, and insight reflection through deep collaboration between LLMs and data modeling techniques. The obtained insights enable LLMs to better align their general knowledge and capabilities with the particular requirements of specific tabular tasks. We extensively evaluate InsightTab on nine datasets. The results demonstrate consistent improvement over state-of-the-art methods. Ablation studies further validate the principle-guided distillation process, while analyses emphasize InsightTab's effectiveness in leveraging labeled data and managing bias. 

---
# Accept or Deny? Evaluating LLM Fairness and Performance in Loan Approval across Table-to-Text Serialization Approaches 

**Authors**: Israel Abebe Azime, Deborah D. Kanubala, Tejumade Afonja, Mario Fritz, Isabel Valera, Dietrich Klakow, Philipp Slusallek  

**Link**: [PDF](https://arxiv.org/pdf/2508.21512)  

**Abstract**: Large Language Models (LLMs) are increasingly employed in high-stakes decision-making tasks, such as loan approvals. While their applications expand across domains, LLMs struggle to process tabular data, ensuring fairness and delivering reliable predictions. In this work, we assess the performance and fairness of LLMs on serialized loan approval datasets from three geographically distinct regions: Ghana, Germany, and the United States. Our evaluation focuses on the model's zero-shot and in-context learning (ICL) capabilities. Our results reveal that the choice of serialization (Serialization refers to the process of converting tabular data into text formats suitable for processing by LLMs.) format significantly affects both performance and fairness in LLMs, with certain formats such as GReat and LIFT yielding higher F1 scores but exacerbating fairness disparities. Notably, while ICL improved model performance by 4.9-59.6% relative to zero-shot baselines, its effect on fairness varied considerably across datasets. Our work underscores the importance of effective tabular data representation methods and fairness-aware models to improve the reliability of LLMs in financial decision-making. 

---
# Morae: Proactively Pausing UI Agents for User Choices 

**Authors**: Yi-Hao Peng, Dingzeyu Li, Jeffrey P. Bigham, Amy Pavel  

**Link**: [PDF](https://arxiv.org/pdf/2508.21456)  

**Abstract**: User interface (UI) agents promise to make inaccessible or complex UIs easier to access for blind and low-vision (BLV) users. However, current UI agents typically perform tasks end-to-end without involving users in critical choices or making them aware of important contextual information, thus reducing user agency. For example, in our field study, a BLV participant asked to buy the cheapest available sparkling water, and the agent automatically chose one from several equally priced options, without mentioning alternative products with different flavors or better ratings. To address this problem, we introduce Morae, a UI agent that automatically identifies decision points during task execution and pauses so that users can make choices. Morae uses large multimodal models to interpret user queries alongside UI code and screenshots, and prompt users for clarification when there is a choice to be made. In a study over real-world web tasks with BLV participants, Morae helped users complete more tasks and select options that better matched their preferences, as compared to baseline agents, including OpenAI Operator. More broadly, this work exemplifies a mixed-initiative approach in which users benefit from the automation of UI agents while being able to express their preferences. 

---
# From Canonical to Complex: Benchmarking LLM Capabilities in Undergraduate Thermodynamics 

**Authors**: Anna Geißler, Luca-Sophie Bien, Friedrich Schöppler, Tobias Hertel  

**Link**: [PDF](https://arxiv.org/pdf/2508.21452)  

**Abstract**: Large language models (LLMs) are increasingly considered as tutoring aids in science education. Yet their readiness for unsupervised use in undergraduate instruction remains uncertain, as reliable teaching requires more than fluent recall: it demands consistent, principle-grounded reasoning. Thermodynamics, with its compact laws and subtle distinctions between state and path functions, reversibility, and entropy, provides an ideal testbed for evaluating such capabilities. Here we present UTQA, a 50-item undergraduate thermodynamics question answering benchmark, covering ideal-gas processes, reversibility, and diagram interpretation. No leading 2025-era model exceeded our 95\% competence threshold: the best LLMs achieved 82\% accuracy, with text-only items performing better than image reasoning tasks, which often fell to chance levels. Prompt phrasing and syntactic complexity showed modest to little correlation with performance. The gap concentrates in finite-rate/irreversible scenarios and in binding visual features to thermodynamic meaning, indicating that current LLMs are not yet suitable for unsupervised tutoring in this domain. 

---
# AHELM: A Holistic Evaluation of Audio-Language Models 

**Authors**: Tony Lee, Haoqin Tu, Chi Heem Wong, Zijun Wang, Siwei Yang, Yifan Mai, Yuyin Zhou, Cihang Xie, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.21376)  

**Abstract**: Evaluations of audio-language models (ALMs) -- multimodal models that take interleaved audio and text as input and output text -- are hindered by the lack of standardized benchmarks; most benchmarks measure only one or two capabilities and omit evaluative aspects such as fairness or safety. Furthermore, comparison across models is difficult as separate evaluations test a limited number of models and use different prompting methods and inference parameters. To address these shortfalls, we introduce AHELM, a benchmark that aggregates various datasets -- including 2 new synthetic audio-text datasets called PARADE, which evaluates the ALMs on avoiding stereotypes, and CoRe-Bench, which measures reasoning over conversational audio through inferential multi-turn question answering -- to holistically measure the performance of ALMs across 10 aspects we have identified as important to the development and usage of ALMs: audio perception, knowledge, reasoning, emotion detection, bias, fairness, multilinguality, robustness, toxicity, and safety. We also standardize the prompts, inference parameters, and evaluation metrics to ensure equitable comparisons across models. We test 14 open-weight and closed-API ALMs from 3 developers and 3 additional simple baseline systems each consisting of an automatic speech recognizer and a language model. Our results show that while Gemini 2.5 Pro ranks top in 5 out of 10 aspects, it exhibits group unfairness ($p=0.01$) on ASR tasks whereas most of the other models do not. We also find that the baseline systems perform reasonably well on AHELM, with one ranking 5th overall despite having only speech-to-text capabilities. For transparency, all raw prompts, model generations, and outputs are available on our website at this https URL. AHELM is intended to be a living benchmark and new datasets and models will be added over time. 

---
# Stairway to Fairness: Connecting Group and Individual Fairness 

**Authors**: Theresia Veronika Rampisela, Maria Maistro, Tuukka Ruotsalo, Falk Scholer, Christina Lioma  

**Link**: [PDF](https://arxiv.org/pdf/2508.21334)  

**Abstract**: Fairness in recommender systems (RSs) is commonly categorised into group fairness and individual fairness. However, there is no established scientific understanding of the relationship between the two fairness types, as prior work on both types has used different evaluation measures or evaluation objectives for each fairness type, thereby not allowing for a proper comparison of the two. As a result, it is currently not known how increasing one type of fairness may affect the other. To fill this gap, we study the relationship of group and individual fairness through a comprehensive comparison of evaluation measures that can be used for both fairness types. Our experiments with 8 runs across 3 datasets show that recommendations that are highly fair for groups can be very unfair for individuals. Our finding is novel and useful for RS practitioners aiming to improve the fairness of their systems. Our code is available at: this https URL. 

---
# Quantum-Enhanced Natural Language Generation: A Multi-Model Framework with Hybrid Quantum-Classical Architectures 

**Authors**: Chi-Sheng Chen, En-Jui Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2508.21332)  

**Abstract**: This paper presents a comprehensive evaluation of quantum text generation models against traditional Transformer/MLP architectures, addressing the growing interest in quantum computing applications for natural language processing. We conduct systematic experiments comparing five distinct models: Transformer (baseline), Quantum Kernel Self-Attention Network (QKSAN), Quantum RWKV (QRWKV), and Quantum Attention Sequence Architecture (QASA) across five diverse datasets including simple sentences, short stories, quantum phrases, haiku poetry, and proverbs. Our evaluation employs multiple metrics including perplexity, BLEU scores, vocabulary diversity, repetition rates, and fluency measures to assess different aspects of text generation quality. The experimental results reveal that while traditional Transformer models maintain overall superiority with the lowest average perplexity (1.21) and highest BLEU-1 score (0.2895), quantum-inspired models demonstrate competitive performance in specific scenarios. Notably, QKSAN achieves a competitive BLEU-1 score of 0.2800 while maintaining zero repetition rates, and QRWKV demonstrates perfect vocabulary diversity (Distinct-1 = 1.000) in certain tasks. 

---
# CrossTL: A Universal Programming Language Translator with Unified Intermediate Representation 

**Authors**: Nripesh Niketan, Vaatsalya Shrivastva  

**Link**: [PDF](https://arxiv.org/pdf/2508.21256)  

**Abstract**: We present CrossTL, a universal programming language translator enabling bidirectional translation between multiple languages through a unified intermediate representation called CrossGL. Traditional approaches require separate translators for each language pair, leading to exponential complexity growth. CrossTL uses a single universal IR to facilitate translations between CUDA, HIP, Metal, DirectX HLSL, OpenGL GLSL, Vulkan SPIR-V, Rust, and Mojo, with Slang support in development. Our system consists of: language-specific lexers/parsers converting source code to ASTs, bidirectional CrossGL translation modules implementing ToCrossGLConverter classes for importing code and CodeGen classes for target generation, and comprehensive backend implementations handling full translation pipelines. We demonstrate effectiveness through comprehensive evaluation across programming domains, achieving successful compilation and execution across all supported backends. The universal IR design enables adding new languages with minimal effort, requiring only language-specific frontend/backend components. Our contributions include: (1) a unified IR capturing semantics of multiple programming paradigms, (2) a modular architecture enabling extensibility, (3) a comprehensive framework supporting GPU compute, graphics programming, and systems languages, and (4) empirical validation demonstrating practical viability of universal code translation. CrossTL represents a significant step toward language-agnostic programming, enabling write-once, deploy-everywhere development. 

---
# Designing Smarter Conversational Agents for Kids: Lessons from Cognitive Work and Means-Ends Analyses 

**Authors**: Vanessa Figueiredo  

**Link**: [PDF](https://arxiv.org/pdf/2508.21209)  

**Abstract**: This paper presents two studies on how Brazilian children (ages 9--11) use conversational agents (CAs) for schoolwork, discovery, and entertainment, and how structured scaffolds can enhance these interactions. In Study 1, a seven-week online investigation with 23 participants (children, parents, teachers) employed interviews, observations, and Cognitive Work Analysis to map children's information-processing flows, the role of more knowledgeable others, functional uses, contextual goals, and interaction patterns to inform conversation-tree design. We identified three CA functions: School, Discovery, Entertainment, and derived ``recipe'' scaffolds mirroring parent-child support. In Study 2, we prompted GPT-4o-mini on 1,200 simulated child-CA exchanges, comparing conversation-tree recipes based on structured-prompting to an unstructured baseline. Quantitative evaluation of readability, question count/depth/diversity, and coherence revealed gains for the recipe approach. Building on these findings, we offer design recommendations: scaffolded conversation-trees, child-dedicated profiles for personalized context, and caregiver-curated content. Our contributions include the first CWA application with Brazilian children, an empirical framework of child-CA information flows, and an LLM-scaffolding ``recipe'' (i.e., structured-prompting) for effective, scaffolded learning. 

---
# Fuzzy, Symbolic, and Contextual: Enhancing LLM Instruction via Cognitive Scaffolding 

**Authors**: Vanessa Figueiredo  

**Link**: [PDF](https://arxiv.org/pdf/2508.21204)  

**Abstract**: We study how architectural inductive biases influence the cognitive behavior of large language models (LLMs) in instructional dialogue. We introduce a symbolic scaffolding mechanism paired with a short-term memory schema designed to promote adaptive, structured reasoning in Socratic tutoring. Using controlled ablation across five system variants, we evaluate model outputs via expert-designed rubrics covering scaffolding, responsiveness, symbolic reasoning, and conversational memory. We present preliminary results using an LLM-based evaluation framework aligned to a cognitively grounded rubric. This enables scalable, systematic comparisons across architectural variants in early-stage experimentation. The preliminary results show that our full system consistently outperforms baseline variants. Analysis reveals that removing memory or symbolic structure degrades key cognitive behaviors, including abstraction, adaptive probing, and conceptual continuity. These findings support a processing-level account in which architectural scaffolds can reliably shape emergent instructional strategies in LLMs. 

---
# Model-Task Alignment Drives Distinct RL Outcomes 

**Authors**: Haoze Wu, Cheng Wang, Wenshuo Zhao, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2508.21188)  

**Abstract**: Recent advances in applying reinforcement learning (RL) to large language models (LLMs) have led to substantial progress. In particular, a series of remarkable yet often counterintuitive phenomena have been reported in LLMs, exhibiting patterns not typically observed in traditional RL settings. For example, notable claims include that a single training example can match the performance achieved with an entire dataset, that the reward signal does not need to be very accurate, and that training solely with negative samples can match or even surpass sophisticated reward-based methods. However, the precise conditions under which these observations hold - and, critically, when they fail - remain unclear. In this work, we identify a key factor that differentiates RL observations: whether the pretrained model already exhibits strong Model-Task Alignment, as measured by pass@k accuracy on the evaluated task. Through a systematic and comprehensive examination of a series of counterintuitive claims, supported by rigorous experimental validation across different model architectures and task domains, our findings show that while standard RL training remains consistently robust across settings, many of these counterintuitive results arise only when the model and task already exhibit strong model-task alignment. In contrast, these techniques fail to drive substantial learning in more challenging regimes, where standard RL methods remain effective. 

---
# Normalisation of SWIFT Message Counterparties with Feature Extraction and Clustering 

**Authors**: Thanasis Schoinas, Benjamin Guinard, Diba Esbati, Richard Chalk  

**Link**: [PDF](https://arxiv.org/pdf/2508.21081)  

**Abstract**: Short text clustering is a known use case in the text analytics community. When the structure and content falls in the natural language domain e.g. Twitter posts or instant messages, then natural language techniques can be used, provided texts are of sufficient length to allow for use of (pre)trained models to extract meaningful information, such as part-of-speech or topic annotations. However, natural language models are not suitable for clustering transaction counterparties, as they are found in bank payment messaging systems, such as SWIFT. The manually typed tags are typically physical or legal entity details, which lack sentence structure, while containing all the variations and noise that manual entry introduces. This leaves a gap in an investigator or counter-fraud professional's toolset when looking to augment their knowledge of payment flow originator and beneficiary entities and trace funds and assets. A gap that vendors traditionally try to close with fuzzy matching tools. With these considerations in mind, we are proposing a hybrid string similarity, topic modelling, hierarchical clustering and rule-based pipeline to facilitate clustering of transaction counterparties, also catering for unknown number of expected clusters. We are also devising metrics to supplement the evaluation of the approach, based on the well-known measures of precision and recall. Testing on a real-life labelled dataset demonstrates significantly improved performance over a baseline rule-based ('keyword') approach. The approach retains most of the interpretability found in rule-based systems, as the former adds an additional level of cluster refinement to the latter. The resulting workflow reduces the need for manual review. When only a subset of the population needs to be investigated, such as in sanctions investigations, the approach allows for better control of the risks of missing entity variations. 

---
# Database Normalization via Dual-LLM Self-Refinement 

**Authors**: Eunjae Jo, Nakyung Lee, Gyuyeong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.17693)  

**Abstract**: Database normalization is crucial to preserving data integrity. However, it is time-consuming and error-prone, as it is typically performed manually by data engineers. To this end, we present Miffie, a database normalization framework that leverages the capability of large language models. Miffie enables automated data normalization without human effort while preserving high accuracy. The core of Miffie is a dual-model self-refinement architecture that combines the best-performing models for normalized schema generation and verification, respectively. The generation module eliminates anomalies based on the feedback of the verification module until the output schema satisfies the requirement for normalization. We also carefully design task-specific zero-shot prompts to guide the models for achieving both high accuracy and cost efficiency. Experimental results show that Miffie can normalize complex database schemas while maintaining high accuracy. 

---
