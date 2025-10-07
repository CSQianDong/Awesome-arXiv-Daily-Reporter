# Investigating LLM Variability in Personalized Conversational Information Retrieval 

**Authors**: Simon Lupart, Daniël van Dijk, Eric Langezaal, Ian van Dort, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03795)  

**Abstract**: Personalized Conversational Information Retrieval (CIR) has seen rapid progress in recent years, driven by the development of Large Language Models (LLMs). Personalized CIR aims to enhance document retrieval by leveraging user-specific information, such as preferences, knowledge, or constraints, to tailor responses to individual needs. A key resource for this task is the TREC iKAT 2023 dataset, designed to evaluate personalization in CIR pipelines. Building on this resource, Mo et al. explored several strategies for incorporating Personal Textual Knowledge Bases (PTKB) into LLM-based query reformulation. Their findings suggested that personalization from PTKBs could be detrimental and that human annotations were often noisy. However, these conclusions were based on single-run experiments using the GPT-3.5 Turbo model, raising concerns about output variability and repeatability. In this reproducibility study, we rigorously reproduce and extend their work, focusing on LLM output variability and model generalization. We apply the original methods to the new TREC iKAT 2024 dataset and evaluate a diverse range of models, including Llama (1B-70B), Qwen-7B, GPT-4o-mini. Our results show that human-selected PTKBs consistently enhance retrieval performance, while LLM-based selection methods do not reliably outperform manual choices. We further compare variance across datasets and observe higher variability on iKAT than on CAsT, highlighting the challenges of evaluating personalized CIR. Notably, recall-oriented metrics exhibit lower variance than precision-oriented ones, a critical insight for first-stage retrievers. Finally, we underscore the need for multi-run evaluations and variance reporting when assessing LLM-based CIR systems. By broadening evaluation across models, datasets, and metrics, our study contributes to more robust and generalizable practices for personalized CIR. 

---
# LLM, Reporting In! Medical Information Extraction Across Prompting, Fine-tuning and Post-correction 

**Authors**: Ikram Belmadani, Parisa Nazari Hashemi, Thomas Sebbag, Benoit Favre, Guillaume Fortier, Solen Quiniou, Emmanuel Morin, Richard Dufour  

**Link**: [PDF](https://arxiv.org/pdf/2510.03577)  

**Abstract**: This work presents our participation in the EvalLLM 2025 challenge on biomedical Named Entity Recognition (NER) and health event extraction in French (few-shot setting). For NER, we propose three approaches combining large language models (LLMs), annotation guidelines, synthetic data, and post-processing: (1) in-context learning (ICL) with GPT-4.1, incorporating automatic selection of 10 examples and a summary of the annotation guidelines into the prompt, (2) the universal NER system GLiNER, fine-tuned on a synthetic corpus and then verified by an LLM in post-processing, and (3) the open LLM LLaMA-3.1-8B-Instruct, fine-tuned on the same synthetic corpus. Event extraction uses the same ICL strategy with GPT-4.1, reusing the guideline summary in the prompt. Results show GPT-4.1 leads with a macro-F1 of 61.53% for NER and 15.02% for event extraction, highlighting the importance of well-crafted prompting to maximize performance in very low-resource scenarios. 

---
# GRACE: Generative Representation Learning via Contrastive Policy Optimization 

**Authors**: Jiashuo Sun, Shixuan Liu, Zhaochen Su, Xianrui Zhong, Pengcheng Jiang, Bowen Jin, Peiran Li, Weijia Shi, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.04506)  

**Abstract**: Prevailing methods for training Large Language Models (LLMs) as text encoders rely on contrastive losses that treat the model as a black box function, discarding its generative and reasoning capabilities in favor of static embeddings. We introduce GRACE (Generative Representation Learning via Contrastive Policy Optimization), a novel framework that reimagines contrastive signals not as losses to be minimized, but as rewards that guide a generative policy. In GRACE, the LLM acts as a policy that produces explicit, human-interpretable rationales--structured natural language explanations of its semantic understanding. These rationales are then encoded into high-quality embeddings via mean pooling. Using policy gradient optimization, we train the model with a multi-component reward function that maximizes similarity between query positive pairs and minimizes similarity with negatives. This transforms the LLM from an opaque encoder into an interpretable agent whose reasoning process is transparent and inspectable. On MTEB benchmark, GRACE yields broad cross category gains: averaged over four backbones, the supervised setting improves overall score by 11.5% over base models, and the unsupervised variant adds 6.9%, while preserving general capabilities. This work treats contrastive objectives as rewards over rationales, unifying representation learning with generation to produce stronger embeddings and transparent rationales. The model, data and code are available at this https URL. 

---
# Automating construction safety inspections using a multi-modal vision-language RAG framework 

**Authors**: Chenxin Wang, Elyas Asadi Shamsabadi, Zhaohui Chen, Luming Shen, Alireza Ahmadian Fard Fini, Daniel Dias-da-Costa  

**Link**: [PDF](https://arxiv.org/pdf/2510.04145)  

**Abstract**: Conventional construction safety inspection methods are often inefficient as they require navigating through large volume of information. Recent advances in large vision-language models (LVLMs) provide opportunities to automate safety inspections through enhanced visual and linguistic understanding. However, existing applications face limitations including irrelevant or unspecific responses, restricted modal inputs and hallucinations. Utilisation of Large Language Models (LLMs) for this purpose is constrained by availability of training data and frequently lack real-time adaptability. This study introduces SiteShield, a multi-modal LVLM-based Retrieval-Augmented Generation (RAG) framework for automating construction safety inspection reports by integrating visual and audio inputs. Using real-world data, SiteShield outperformed unimodal LLMs without RAG with an F1 score of 0.82, hamming loss of 0.04, precision of 0.76, and recall of 0.96. The findings indicate that SiteShield offers a novel pathway to enhance information retrieval and efficiency in generating safety reports. 

---
# Empowering Denoising Sequential Recommendation with Large Language Model Embeddings 

**Authors**: Tongzhou Wu, Yuhao Wang, Maolin Wang, Chi Zhang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04239)  

**Abstract**: Sequential recommendation aims to capture user preferences by modeling sequential patterns in user-item interactions. However, these models are often influenced by noise such as accidental interactions, leading to suboptimal performance. Therefore, to reduce the effect of noise, some works propose explicitly identifying and removing noisy items. However, we find that simply relying on collaborative information may result in an over-denoising problem, especially for cold items. To overcome these limitations, we propose a novel framework: Interest Alignment for Denoising Sequential Recommendation (IADSR) which integrates both collaborative and semantic information. Specifically, IADSR is comprised of two stages: in the first stage, we obtain the collaborative and semantic embeddings of each item from a traditional sequential recommendation model and an LLM, respectively. In the second stage, we align the collaborative and semantic embeddings and then identify noise in the interaction sequence based on long-term and short-term interests captured in the collaborative and semantic modalities. Our extensive experiments on four public datasets validate the effectiveness of the proposed framework and its compatibility with different sequential recommendation systems. 

---
# Resource-Efficient Fine-Tuning of LLaMA-3.2-3B for Medical Chain-of-Thought Reasoning 

**Authors**: Imran Mansha  

**Link**: [PDF](https://arxiv.org/pdf/2510.05003)  

**Abstract**: Large Language Models (LLMs) such as GPT-4 and LLaMA have demonstrated remarkable reasoning abilities but require significant computational resources for fine-tuning. This paper presents a resource-efficient fine-tuning approach for LLaMA-3.2-3B to enhance medical chain-of-thought reasoning while operating under constrained GPU and memory settings. Using parameter-efficient tuning techniques such as LoRA and QLoRA, we adapt the base model on publicly available medical reasoning datasets. The model achieves improved reasoning coherence and factual accuracy while reducing memory usage by up to 60% compared to standard full fine-tuning. Experimental evaluation demonstrates that lightweight adaptations can retain strong reasoning capability in medical question-answering tasks. This work highlights practical strategies for deploying LLMs in low-resource research environments and provides insights into balancing efficiency and domain specialization for medical AI systems. 

---
# TeachLM: Post-Training LLMs for Education Using Authentic Learning Data 

**Authors**: Janos Perczel, Jin Chow, Dorottya Demszky  

**Link**: [PDF](https://arxiv.org/pdf/2510.05087)  

**Abstract**: The promise of generative AI to revolutionize education is constrained by the pedagogical limits of large language models (LLMs). A major issue is the lack of access to high-quality training data that reflect the learning of actual students. Prompt engineering has emerged as a stopgap, but the ability of prompts to encode complex pedagogical strategies in rule-based natural language is inherently limited. To address this gap we introduce TeachLM - an LLM optimized for teaching through parameter-efficient fine-tuning of state-of-the-art models. TeachLM is trained on a dataset comprised of 100,000 hours of one-on-one, longitudinal student-tutor interactions maintained by Polygence, which underwent a rigorous anonymization process to protect privacy. We use parameter-efficient fine-tuning to develop an authentic student model that enables the generation of high-fidelity synthetic student-tutor dialogues. Building on this capability, we propose a novel multi-turn evaluation protocol that leverages synthetic dialogue generation to provide fast, scalable, and reproducible assessments of the dialogical capabilities of LLMs. Our evaluations demonstrate that fine-tuning on authentic learning data significantly improves conversational and pedagogical performance - doubling student talk time, improving questioning style, increasing dialogue turns by 50%, and greater personalization of instruction. 

---
# Finish First, Perfect Later: Test-Time Token-Level Cross-Validation for Diffusion Large Language Models 

**Authors**: Runchu Tian, Junxia Cui, Xueqiang Xu, Feng Yao, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05090)  

**Abstract**: Diffusion large language models (dLLMs) have recently emerged as a promising alternative to autoregressive (AR) models, offering advantages such as accelerated parallel decoding and bidirectional context modeling. However, the vanilla decoding strategy in discrete dLLMs suffers from a critical limitation: once a token is accepted, it can no longer be revised in subsequent steps. As a result, early mistakes persist across iterations, harming both intermediate predictions and final output quality. To address this issue, we propose Tolerator (Token-Level Cross-Validation Refinement), a training-free decoding strategy that leverages cross-validation among predicted tokens. Unlike existing methods that follow a single progressive unmasking procedure, Tolerator introduces a two-stage process: (i) sequence fill-up and (ii) iterative refinement by remasking and decoding a subset of tokens while treating the remaining as context. This design enables previously accepted tokens to be reconsidered and corrected when necessary, leading to more reliable diffusion decoding outputs. We evaluate Tolerator on five standard benchmarks covering language understanding, code generation, and mathematics. Experiments show that our method achieves consistent improvements over the baselines under the same computational budget. These findings suggest that decoding algorithms are crucial to realizing the full potential of diffusion large language models. Code and data are publicly available. 

---
# Imperceptible Jailbreaking against Large Language Models 

**Authors**: Kuofeng Gao, Yiming Li, Chao Du, Xin Wang, Xingjun Ma, Shu-Tao Xia, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05025)  

**Abstract**: Jailbreaking attacks on the vision modality typically rely on imperceptible adversarial perturbations, whereas attacks on the textual modality are generally assumed to require visible modifications (e.g., non-semantic suffixes). In this paper, we introduce imperceptible jailbreaks that exploit a class of Unicode characters called variation selectors. By appending invisible variation selectors to malicious questions, the jailbreak prompts appear visually identical to original malicious questions on screen, while their tokenization is "secretly" altered. We propose a chain-of-search pipeline to generate such adversarial suffixes to induce harmful responses. Our experiments show that our imperceptible jailbreaks achieve high attack success rates against four aligned LLMs and generalize to prompt injection attacks, all without producing any visible modifications in the written prompt. Our code is available at this https URL. 

---
# COLE: a Comprehensive Benchmark for French Language Understanding Evaluation 

**Authors**: David Beauchemin, Yan Tremblay, Mohamed Amine Youssef, Richard Khoury  

**Link**: [PDF](https://arxiv.org/pdf/2510.05046)  

**Abstract**: To address the need for a more comprehensive evaluation of French Natural Language Understanding (NLU), we introduce COLE, a new benchmark composed of 23 diverse task covering a broad range of NLU capabilities, including sentiment analysis, paraphrase detection, grammatical judgment, and reasoning, with a particular focus on linguistic phenomena relevant to the French language. We benchmark 94 large language models (LLM), providing an extensive analysis of the current state of French NLU. Our results highlight a significant performance gap between closed- and open-weights models and identify key challenging frontiers for current LLMs, such as zero-shot extractive question-answering (QA), fine-grained word sense disambiguation, and understanding of regional language variations. We release COLE as a public resource to foster further progress in French language modelling. 

---
# The Geometry of Truth: Layer-wise Semantic Dynamics for Hallucination Detection in Large Language Models 

**Authors**: Amir Hameed Mir  

**Link**: [PDF](https://arxiv.org/pdf/2510.04933)  

**Abstract**: Large Language Models (LLMs) often produce fluent yet factually incorrect statements-a phenomenon known as hallucination-posing serious risks in high-stakes domains. We present Layer-wise Semantic Dynamics (LSD), a geometric framework for hallucination detection that analyzes the evolution of hidden-state semantics across transformer layers. Unlike prior methods that rely on multiple sampling passes or external verification sources, LSD operates intrinsically within the model's representational space. Using margin-based contrastive learning, LSD aligns hidden activations with ground-truth embeddings derived from a factual encoder, revealing a distinct separation in semantic trajectories: factual responses preserve stable alignment, while hallucinations exhibit pronounced semantic drift across depth. Evaluated on the TruthfulQA and synthetic factual-hallucination datasets, LSD achieves an F1-score of 0.92, AUROC of 0.96, and clustering accuracy of 0.89, outperforming SelfCheckGPT and Semantic Entropy baselines while requiring only a single forward pass. This efficiency yields a 5-20x speedup over sampling-based methods without sacrificing precision or interpretability. LSD offers a scalable, model-agnostic mechanism for real-time hallucination monitoring and provides new insights into the geometry of factual consistency within large language models. 

---
# When Models Lie, We Learn: Multilingual Span-Level Hallucination Detection with PsiloQA 

**Authors**: Elisei Rykov, Kseniia Petrushina, Maksim Savkin, Valerii Olisov, Artem Vazhentsev, Kseniia Titova, Alexander Panchenko, Vasily Konovalov, Julia Belikova  

**Link**: [PDF](https://arxiv.org/pdf/2510.04849)  

**Abstract**: Hallucination detection remains a fundamental challenge for the safe and reliable deployment of large language models (LLMs), especially in applications requiring factual accuracy. Existing hallucination benchmarks often operate at the sequence level and are limited to English, lacking the fine-grained, multilingual supervision needed for a comprehensive evaluation. In this work, we introduce PsiloQA, a large-scale, multilingual dataset annotated with span-level hallucinations across 14 languages. PsiloQA is constructed through an automated three-stage pipeline: generating question-answer pairs from Wikipedia using GPT-4o, eliciting potentially hallucinated answers from diverse LLMs in a no-context setting, and automatically annotating hallucinated spans using GPT-4o by comparing against golden answers and retrieved context. We evaluate a wide range of hallucination detection methods -- including uncertainty quantification, LLM-based tagging, and fine-tuned encoder models -- and show that encoder-based models achieve the strongest performance across languages. Furthermore, PsiloQA demonstrates effective cross-lingual generalization and supports robust knowledge transfer to other benchmarks, all while being significantly more cost-efficient than human-annotated datasets. Our dataset and results advance the development of scalable, fine-grained hallucination detection in multilingual settings. 

---
# Do LLMs Align with My Task? Evaluating Text-to-SQL via Dataset Alignment 

**Authors**: Davood Rafiei, Morgan Lindsay Heisler, Weiwei Zhang, Mohammadreza Pourreza, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04919)  

**Abstract**: Supervised Fine-Tuning (SFT) is an effective method for adapting Large Language Models (LLMs) on downstream tasks. However, variability in training data can hinder a model's ability to generalize across domains. This paper studies the problem of dataset alignment for Natural Language to SQL (NL2SQL or text to SQL), examining how well SFT training data matches the structural characteristics of target queries and how this alignment impacts model performance. We hypothesize that alignment can be accurately estimated by comparing the distributions of structural SQL features across the training set, target data, and the model's predictions prior to SFT. Through comprehensive experiments on three large cross-domain NL2SQL benchmarks and multiple model families, we show that structural alignment is a strong predictor of fine-tuning success. When alignment is high, SFT yields substantial gains in accuracy and SQL generation quality; when alignment is low, improvements are marginal or absent. These findings highlight the importance of alignment-aware data selection for effective fine-tuning and generalization in NL2SQL tasks. 

---
# SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests 

**Authors**: Punya Syon Pandey, Hai Son Le, Devansh Bhardwaj, Rada Mihalcea, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04891)  

**Abstract**: Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at this https URL. 

---
# Instability in Downstream Task Performance During LLM Pretraining 

**Authors**: Yuto Nishida, Masaru Isonuma, Yusuke Oda  

**Link**: [PDF](https://arxiv.org/pdf/2510.04848)  

**Abstract**: When training large language models (LLMs), it is common practice to track downstream task performance throughout the training process and select the checkpoint with the highest validation score. However, downstream metrics often exhibit substantial fluctuations, making it difficult to identify the checkpoint that truly represents the best-performing model. In this study, we empirically analyze the stability of downstream task performance in an LLM trained on diverse web-scale corpora. We find that task scores frequently fluctuate throughout training, both at the aggregate and example levels. To address this instability, we investigate two post-hoc checkpoint integration methods: checkpoint averaging and ensemble, motivated by the hypothesis that aggregating neighboring checkpoints can reduce performance volatility. We demonstrate both empirically and theoretically that these methods improve downstream performance stability without requiring any changes to the training procedure. 

---
# ModernBERT + ColBERT: Enhancing biomedical RAG through an advanced re-ranking retriever 

**Authors**: Eduardo Martínez Rivera, Filippo Menolascina  

**Link**: [PDF](https://arxiv.org/pdf/2510.04757)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a powerful technique for enriching Large Language Models (LLMs) with external knowledge, allowing for factually grounded responses, a critical requirement in high-stakes domains such as healthcare. However, the efficacy of RAG systems is fundamentally restricted by the performance of their retrieval module, since irrelevant or semantically misaligned documents directly compromise the accuracy of the final generated response. General-purpose dense retrievers can struggle with the nuanced language of specialised domains, while the high accuracy of in-domain models is often achieved at prohibitive computational costs. In this work, we aim to address this trade-off by developing and evaluating a two-stage retrieval architecture that combines a lightweight ModernBERT bidirectional encoder for efficient initial candidate retrieval with a ColBERTv2 late-interaction model for fine-grained re-ranking. We conduct comprehensive evaluations of our retriever module performance and RAG system performance in the biomedical context, fine-tuning the IR module using 10k question-passage pairs from PubMedQA. Our analysis of the retriever module confirmed the positive impact of the ColBERT re-ranker, which improved Recall@3 by up to 4.2 percentage points compared to its retrieve-only counterpart. When integrated into the biomedical RAG, our IR module leads to a state-of-the-art average accuracy of 0.4448 on the five tasks of the MIRAGE question-answering benchmark, outperforming strong baselines such as MedCPT (0.4436). Our ablation studies reveal that this performance is critically dependent on a joint fine-tuning process that aligns the retriever and re-ranker; otherwise, the re-ranker might degrade the performance. 

---
# Hybrid Architectures for Language Models: Systematic Analysis and Design Insights 

**Authors**: Sangmin Bae, Bilge Acun, Haroun Habeeb, Seungyeon Kim, Chien-Yu Lin, Liang Luo, Junjie Wang, Carole-Jean Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04800)  

**Abstract**: Recent progress in large language models demonstrates that hybrid architectures--combining self-attention mechanisms with structured state space models like Mamba--can achieve a compelling balance between modeling quality and computational efficiency, particularly for long-context tasks. While these hybrid models show promising performance, systematic comparisons of hybridization strategies and analyses on the key factors behind their effectiveness have not been clearly shared to the community. In this work, we present a holistic evaluation of hybrid architectures based on inter-layer (sequential) or intra-layer (parallel) fusion. We evaluate these designs from a variety of perspectives: language modeling performance, long-context capabilities, scaling analysis, and training and inference efficiency. By investigating the core characteristics of their computational primitive, we identify the most critical elements for each hybridization strategy and further propose optimal design recipes for both hybrid models. Our comprehensive analysis provides practical guidance and valuable insights for developing hybrid language models, facilitating the optimization of architectural configurations. 

---
# Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy (short paper) 

**Authors**: Om Dobariya, Akhil Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.04950)  

**Abstract**: The wording of natural language prompts has been shown to influence the performance of large language models (LLMs), yet the role of politeness and tone remains underexplored. In this study, we investigate how varying levels of prompt politeness affect model accuracy on multiple-choice questions. We created a dataset of 50 base questions spanning mathematics, science, and history, each rewritten into five tone variants: Very Polite, Polite, Neutral, Rude, and Very Rude, yielding 250 unique prompts. Using ChatGPT 4o, we evaluated responses across these conditions and applied paired sample t-tests to assess statistical significance. Contrary to expectations, impolite prompts consistently outperformed polite ones, with accuracy ranging from 80.8% for Very Polite prompts to 84.8% for Very Rude prompts. These findings differ from earlier studies that associated rudeness with poorer outcomes, suggesting that newer LLMs may respond differently to tonal variation. Our results highlight the importance of studying pragmatic aspects of prompting and raise broader questions about the social dimensions of human-AI interaction. 

---
# FocusMed: A Large Language Model-based Framework for Enhancing Medical Question Summarization with Focus Identification 

**Authors**: Chao Liu, Ling Luo, Tengxiao Lv, Huan Zhuang, Lejing Yu, Jian Wang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04671)  

**Abstract**: With the rapid development of online medical platforms, consumer health questions (CHQs) are inefficient in diagnosis due to redundant information and frequent non-professional terms. The medical question summary (MQS) task aims to transform CHQs into streamlined doctors' frequently asked questions (FAQs), but existing methods still face challenges such as poor identification of question focus and model hallucination. This paper explores the potential of large language models (LLMs) in the MQS task and finds that direct fine-tuning is prone to focus identification bias and generates unfaithful content. To this end, we propose an optimization framework based on core focus guidance. First, a prompt template is designed to drive the LLMs to extract the core focus from the CHQs that is faithful to the original text. Then, a fine-tuning dataset is constructed in combination with the original CHQ-FAQ pairs to improve the ability to identify the focus of the question. Finally, a multi-dimensional quality evaluation and selection mechanism is proposed to comprehensively improve the quality of the summary from multiple dimensions. We conduct comprehensive experiments on two widely-adopted MQS datasets using three established evaluation metrics. The proposed framework achieves state-of-the-art performance across all measures, demonstrating a significant boost in the model's ability to identify critical focus of questions and a notable mitigation of hallucinations. The source codes are freely available at this https URL. 

---
# Multi-Agent Tool-Integrated Policy Optimization 

**Authors**: Zhanfeng Mo, Xingxuan Li, Yuntao Chen, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2510.04678)  

**Abstract**: Large language models (LLMs) increasingly rely on multi-turn tool-integrated planning for knowledge-intensive and complex reasoning tasks. Existing implementations typically rely on a single agent, but they suffer from limited context length and noisy tool responses. A natural solution is to adopt a multi-agent framework with planner- and worker-agents to manage context. However, no existing methods support effective reinforcement learning post-training of tool-integrated multi-agent frameworks. To address this gap, we propose Multi-Agent Tool-Integrated Policy Optimization (MATPO), which enables distinct roles (planner and worker) to be trained within a single LLM instance using role-specific prompts via reinforcement learning. MATPO is derived from a principled credit assignment mechanism across planner and worker rollouts. This design eliminates the need to deploy multiple LLMs, which would be memory-intensive, while preserving the benefits of specialization. Experiments on GAIA-text, WebWalkerQA, and FRAMES show that MATPO consistently outperforms single-agent baselines by an average of 18.38% relative improvement in performance and exhibits greater robustness to noisy tool outputs. Our findings highlight the effectiveness of unifying multiple agent roles within a single LLM and provide practical insights for stable and efficient multi-agent RL training. 

---
# Can LLMs Detect Ambiguous Plural Reference? An Analysis of Split-Antecedent and Mereological Reference 

**Authors**: Dang Anh, Rick Nouwen, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2510.04581)  

**Abstract**: Our goal is to study how LLMs represent and interpret plural reference in ambiguous and unambiguous contexts. We ask the following research questions: (1) Do LLMs exhibit human-like preferences in representing plural reference? (2) Are LLMs able to detect ambiguity in plural anaphoric expressions and identify possible referents? To address these questions, we design a set of experiments, examining pronoun production using next-token prediction tasks, pronoun interpretation, and ambiguity detection using different prompting strategies. We then assess how comparable LLMs are to humans in formulating and interpreting plural reference. We find that LLMs are sometimes aware of possible referents of ambiguous pronouns. However, they do not always follow human reference when choosing between interpretations, especially when the possible interpretation is not explicitly mentioned. In addition, they struggle to identify ambiguity without direct instruction. Our findings also reveal inconsistencies in the results across different types of experiments. 

---
# Robustness assessment of large audio language models in multiple-choice evaluation 

**Authors**: Fernando López, Santosh Kesiraju, Jordi Luque  

**Link**: [PDF](https://arxiv.org/pdf/2510.04584)  

**Abstract**: Recent advances in large audio language models (LALMs) have primarily been assessed using a multiple-choice question answering (MCQA) framework. However, subtle changes, such as shifting the order of choices, result in substantially different results. Existing MCQA frameworks do not account for this variability and report a single accuracy number per benchmark or category. We dive into the MCQA evaluation framework and conduct a systematic study spanning three benchmarks (MMAU, MMAR and MMSU) and four models: Audio Flamingo 2, Audio Flamingo 3, Qwen2.5-Omni-7B-Instruct, and Kimi-Audio-7B-Instruct. Our findings indicate that models are sensitive not only to the ordering of choices, but also to the paraphrasing of the question and the choices. Finally, we propose a simpler evaluation protocol and metric that account for subtle variations and provide a more detailed evaluation report of LALMs within the MCQA framework. 

---
# GenQuest: An LLM-based Text Adventure Game for Language Learners 

**Authors**: Qiao Wang, Adnan Labib, Robert Swier, Michael Hofmeyr, Zheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04498)  

**Abstract**: GenQuest is a generative text adventure game that leverages Large Language Models (LLMs) to facilitate second language learning through immersive, interactive storytelling. The system engages English as a Foreign Language (EFL) learners in a collaborative "choose-your-own-adventure" style narrative, dynamically generated in response to learner choices. Game mechanics such as branching decision points and story milestones are incorporated to maintain narrative coherence while allowing learner-driven plot development. Key pedagogical features include content generation tailored to each learner's proficiency level, and a vocabulary assistant that provides in-context explanations of learner-queried text strings, ranging from words and phrases to sentences. Findings from a pilot study with university EFL students in China indicate promising vocabulary gains and positive user perceptions. Also discussed are suggestions from participants regarding the narrative length and quality, and the request for multi-modal content such as illustrations. 

---
# A First Context-Free Grammar Applied to Nawatl Corpora Augmentation 

**Authors**: Juan-José Guzmán-Landa, Juan-Manuel Torres-Moreno, Miguel Figueroa-Saavedra, Ligia Quintana-Torres, Martha-Lorena Avendaño-Garrido, Graham Ranger  

**Link**: [PDF](https://arxiv.org/pdf/2510.04945)  

**Abstract**: In this article we introduce a context-free grammar (CFG) for the Nawatl language. Nawatl (or Nahuatl) is an Amerindian language of the $\pi$-language type, i.e. a language with few digital resources, in which the corpora available for machine learning are virtually non-existent. The objective here is to generate a significant number of grammatically correct artificial sentences, in order to increase the corpora available for language model training. We want to show that a grammar enables us significantly to expand a corpus in Nawatl which we call $\pi$-\textsc{yalli}. The corpus, thus enriched, enables us to train algorithms such as FastText and to evaluate them on sentence-level semantic tasks. Preliminary results show that by using the grammar, comparative improvements are achieved over some LLMs. However, it is observed that to achieve more significant improvement, grammars that model the Nawatl language even more effectively are required. 

---
# SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs 

**Authors**: Dachuan Shi, Abedelkadir Asi, Keying Li, Xiangchi Yuan, Leyan Pan, Wenke Lee, Wen Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.05069)  

**Abstract**: Recent work shows that, beyond discrete reasoning through explicit chain-of-thought steps, which are limited by the boundaries of natural languages, large language models (LLMs) can also reason continuously in latent space, allowing richer information per step and thereby improving token efficiency. Despite this promise, latent reasoning still faces two challenges, especially in training-free settings: 1) purely latent reasoning broadens the search distribution by maintaining multiple implicit paths, which diffuses probability mass, introduces noise, and impedes convergence to a single high-confidence solution, thereby hurting accuracy; and 2) overthinking persists even without explicit text, wasting tokens and degrading efficiency. To address these issues, we introduce SwiReasoning, a training-free framework for LLM reasoning which features two key innovations: 1) SwiReasoning dynamically switches between explicit and latent reasoning, guided by block-wise confidence estimated from entropy trends in next-token distributions, to balance exploration and exploitation and promote timely convergence. 2) By limiting the maximum number of thinking-block switches, SwiReasoning curbs overthinking and improves token efficiency across varying problem difficulties. On widely used mathematics and STEM benchmarks, SwiReasoning consistently improves average accuracy by 1.5%-2.8% across reasoning LLMs of different model families and scales. Furthermore, under constrained budgets, SwiReasoning improves average token efficiency by 56%-79%, with larger gains as budgets tighten. 

---
# FedSRD: Sparsify-Reconstruct-Decompose for Communication-Efficient Federated Large Language Models Fine-Tuning 

**Authors**: Guochen Yan, Luyuan Xie, Qingni Shen, Yuejian Fang, Zhonghai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04601)  

**Abstract**: The current paradigm of training large language models (LLMs) on publicly available Web data is becoming unsustainable, with high-quality data sources in specialized domains nearing exhaustion. Federated Learning (FL) emerges as a practical solution for the next generation of AI on a decentralized Web, enabling privacy-preserving collaborative fine-tuning by leveraging private data distributed across a global client base. While Low-Rank Adaptation (LoRA) is the standard for efficient fine-tuning, its application in federated settings presents a critical challenge: communication overhead remains a significant bottleneck across the Web's heterogeneous network conditions. The structural redundancy within LoRA parameters not only incurs a heavy communication burden but also introduces conflicts when aggregating client updates. To address this, we propose FedSRD, a Sparsify-Reconstruct-Decompose framework designed for communication-efficient FL. We first introduce an importance-aware sparsification method that preserves the structural integrity of LoRA updates to reduce the uploaded parameter count. The server then reconstructs and aggregates these updates in a full-rank space to mitigate conflicts. Finally, it decomposes the global update into a sparse low-rank format for broadcast, ensuring a symmetrically efficient cycle. We also propose an efficient variant, FedSRD-e, to reduce computational overhead. Experimental results on 10 benchmarks demonstrate that our framework significantly reduces communication costs by up to 90\% while even improving model performance on heterogeneous client data. 

---
# On the Role of Unobserved Sequences on Sample-based Uncertainty Quantification for LLMs 

**Authors**: Lucie Kunitomo-Jacquin, Edison Marrese-Taylor, Ken Fukuda  

**Link**: [PDF](https://arxiv.org/pdf/2510.04439)  

**Abstract**: Quantifying uncertainty in large language models (LLMs) is important for safety-critical applications because it helps spot incorrect answers, known as hallucinations. One major trend of uncertainty quantification methods is based on estimating the entropy of the distribution of the LLM's potential output sequences. This estimation is based on a set of output sequences and associated probabilities obtained by querying the LLM several times. In this paper, we advocate and experimentally show that the probability of unobserved sequences plays a crucial role, and we recommend future research to integrate it to enhance such LLM uncertainty quantification methods. 

---
# Large Language Models Preserve Semantic Isotopies in Story Continuations 

**Authors**: Marc Cavazza  

**Link**: [PDF](https://arxiv.org/pdf/2510.04400)  

**Abstract**: In this work, we explore the relevance of textual semantics to Large Language Models (LLMs), extending previous insights into the connection between distributional semantics and structural semantics. We investigate whether LLM-generated texts preserve semantic isotopies. We design a story continuation experiment using 10,000 ROCStories prompts completed by five LLMs. We first validate GPT-4o's ability to extract isotopies from a linguistic benchmark, then apply it to the generated stories. We then analyze structural (coverage, density, spread) and semantic properties of isotopies to assess how they are affected by completion. Results show that LLM completion within a given token horizon preserves semantic isotopies across multiple properties. 

---
# Are BabyLMs Deaf to Gricean Maxims? A Pragmatic Evaluation of Sample-efficient Language Models 

**Authors**: Raha Askari, Sina Zarrieß, Özge Alacam, Judith Sieker  

**Link**: [PDF](https://arxiv.org/pdf/2510.04764)  

**Abstract**: Implicit meanings are integral to human communication, making it essential for language models to be capable of identifying and interpreting them. Grice (1975) proposed a set of conversational maxims that guide cooperative dialogue, noting that speakers may deliberately violate these principles to express meanings beyond literal words, and that listeners, in turn, recognize such violations to draw pragmatic inferences.
Building on Surian et al. (1996)'s study of children's sensitivity to violations of Gricean maxims, we introduce a novel benchmark to test whether language models pretrained on less than 10M and less than 100M tokens can distinguish maxim-adhering from maxim-violating utterances. We compare these BabyLMs across five maxims and situate their performance relative to children and a Large Language Model (LLM) pretrained on 3T tokens.
We find that overall, models trained on less than 100M tokens outperform those trained on less than 10M, yet fall short of child-level and LLM competence. Our results suggest that modest data increases improve some aspects of pragmatic behavior, leading to finer-grained differentiation between pragmatic dimensions. 

---
# Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards 

**Authors**: Faisal Hamman, Chenyang Zhu, Anoop Kumar, Xujun Peng, Sanghamitra Dutta, Daben Liu, Alfy Samuel  

**Link**: [PDF](https://arxiv.org/pdf/2510.04392)  

**Abstract**: RAG systems are increasingly deployed in high-stakes domains where users expect outputs to be consistent across semantically equivalent queries. However, existing systems often exhibit significant inconsistencies due to variability in both the retriever and generator (LLM), undermining trust and reliability. In this work, we focus on information consistency, i.e., the requirement that outputs convey the same core content across semantically equivalent inputs. We introduce a principled evaluation framework that decomposes RAG consistency into retriever-level, generator-level, and end-to-end components, helping identify inconsistency sources. To improve consistency, we propose Paraphrased Set Group Relative Policy Optimization (PS-GRPO), an RL approach that leverages multiple rollouts across paraphrased set to assign group similarity rewards. We leverage PS-GRPO to achieve Information Consistent RAG (Con-RAG), training the generator to produce consistent outputs across paraphrased queries and remain robust to retrieval-induced variability. Because exact reward computation over paraphrase sets is computationally expensive, we also introduce a scalable approximation method that retains effectiveness while enabling efficient, large-scale training. Empirical evaluations across short-form, multi-hop, and long-form QA benchmarks demonstrate that Con-RAG significantly improves both consistency and accuracy over strong baselines, even in the absence of explicit ground-truth supervision. Our work provides practical solutions for evaluating and building reliable RAG systems for safety-critical deployments. 

---
# Evaluating LLMs for Demographic-Targeted Social Bias Detection: A Comprehensive Benchmark Study 

**Authors**: Ayan Majumdar, Feihao Chen, Jinghui Li, Xiaozhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04641)  

**Abstract**: Large-scale web-scraped text corpora used to train general-purpose AI models often contain harmful demographic-targeted social biases, creating a regulatory need for data auditing and developing scalable bias-detection methods. Although prior work has investigated biases in text datasets and related detection methods, these studies remain narrow in scope. They typically focus on a single content type (e.g., hate speech), cover limited demographic axes, overlook biases affecting multiple demographics simultaneously, and analyze limited techniques. Consequently, practitioners lack a holistic understanding of the strengths and limitations of recent large language models (LLMs) for automated bias detection. In this study, we present a comprehensive evaluation framework aimed at English texts to assess the ability of LLMs in detecting demographic-targeted social biases. To align with regulatory requirements, we frame bias detection as a multi-label task using a demographic-focused taxonomy. We then conduct a systematic evaluation with models across scales and techniques, including prompting, in-context learning, and fine-tuning. Using twelve datasets spanning diverse content types and demographics, our study demonstrates the promise of fine-tuned smaller models for scalable detection. However, our analyses also expose persistent gaps across demographic axes and multi-demographic targeted biases, underscoring the need for more effective and scalable auditing frameworks. 

---
# SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations 

**Authors**: Buyun Liang, Liangzu Peng, Jinqi Luo, Darshan Thaker, Kwan Ho Ryan Chan, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2510.04398)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no constraint violations compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at this https URL. 

---
# Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time 

**Authors**: Daniel Tan, Anders Woodruff, Niels Warncke, Arun Jose, Maxime Riché, David Demitri Africa, Mia Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2510.04340)  

**Abstract**: Language model finetuning often results in learning undesirable traits in combination with desired ones. To address this, we propose inoculation prompting: modifying finetuning data by prepending a short system-prompt instruction that deliberately elicits the undesirable trait. At test time, we evaluate without the instruction; inoculated models have much lower expression of the trait than models trained with unmodified training data. Inoculation is selective: in a toy setting where assistant responses are always in Spanish and ALL-CAPS, an appropriate inoculation (e.g., ``You always speak in Spanish.'') teaches the model to capitalize responses while still responding in English. We find that inoculation is also effective across several additional settings: reducing emergent misalignment (EM) from task-specific finetuning, defending against backdoor injections, and mitigating the transmission of traits via subliminal learning. Follow-up analysis suggests a mechanism: making a trait less surprising via inoculation reduces optimization pressure to globally update the model, thereby reducing the degree of generalization. Our analysis relates to prior work on EM: inoculation explains prior findings that educational contexts mitigate EM from insecure code. Beyond demonstrating a simple and effective technique for selective learning, our results contribute to a better conceptual understanding of how and why language models generalize. 

---
# JSON Whisperer: Efficient JSON Editing with LLMs 

**Authors**: Sarel Duanis, Asnat Greenstein-Messica, Eliya Habba  

**Link**: [PDF](https://arxiv.org/pdf/2510.04717)  

**Abstract**: Large language models (LLMs) can modify JSON documents through natural language commands, but current approaches regenerate entire structures for each edit, resulting in computational inefficiency. We present JSON Whisperer, a framework that enables LLMs to generate RFC 6902 diff patches-expressing only the necessary modifications-rather than complete documents. We identify two key challenges in patch-based editing: (1) LLMs often miss related updates when generating isolated patches, and (2) array manipulations require tracking index shifts across operations, which LLMs handle poorly. To address these issues, we introduce EASE (Explicitly Addressed Sequence Encoding), which transforms arrays into dictionaries with stable keys, eliminating index arithmetic complexities. Our evaluation shows that patch generation with EASE reduces token usage by 31% while maintaining edit quality within 5% of full regeneration with particular gains for complex instructions and list manipulations. The dataset is available at: this https URL 

---
# Probing Geometry of Next Token Prediction Using Cumulant Expansion of the Softmax Entropy 

**Authors**: Karthik Viswanathan, Sang Eon Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.04285)  

**Abstract**: We introduce a cumulant-expansion framework for quantifying how large language models (LLMs) internalize higher-order statistical structure during next-token prediction. By treating the softmax entropy of each layer's logit distribution as a perturbation around its "center" distribution, we derive closed-form cumulant observables that isolate successively higher-order correlations. Empirically, we track these cumulants in GPT-2 and Pythia models on Pile-10K prompts. (i) Structured prompts exhibit a characteristic rise-and-plateau profile across layers, whereas token-shuffled prompts remain flat, revealing the dependence of the cumulant profile on meaningful context. (ii) During training, all cumulants increase monotonically before saturating, directly visualizing the model's progression from capturing variance to learning skew, kurtosis, and higher-order statistical structures. (iii) Mathematical prompts show distinct cumulant signatures compared to general text, quantifying how models employ fundamentally different processing mechanisms for mathematical versus linguistic content. Together, these results establish cumulant analysis as a lightweight, mathematically grounded probe of feature-learning dynamics in high-dimensional neural networks. 

---
# CALM Before the STORM: Unlocking Native Reasoning for Optimization Modeling 

**Authors**: Zhengyang Tang, Zihan Ye, Chenyu Huang, Xuhan Huang, Chengpeng Li, Sihang Li, Guanhua Chen, Ming Yan, Zizhuo Wang, Hongyuan Zha, Dayiheng Liu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04204)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated strong capabilities in complex multi-step reasoning, opening new opportunities for automating optimization modeling. However, existing domain adaptation methods, originally designed for earlier instruction-tuned models, often fail to exploit the advanced reasoning patterns of modern LRMs -- In particular, we show that direct fine-tuning on traditional \textit{non-reflective} datasets leads to limited gains. To fully leverage LRMs' inherent reasoning abilities, we propose \textbf{CALM} (\textit{Corrective Adaptation with Lightweight Modification}), a framework that progressively refines LRMs within their native reasoning modes for optimization modeling tasks. In CALM, an expert intervener identifies reasoning flaws and provides concise corrective hints, which the LRM incorporates to produce improved reasoning trajectories. These interventions modify fewer than 2.6\% of generated tokens, but generate high-quality data for soft adaptation through supervised fine-tuning. The adapted model is then further improved through reinforcement learning. Building on CALM, we develop \textbf{STORM} (\textit{Smart Thinking Optimization Reasoning Model}), a 4B-parameter LRM that achieves a new state-of-the-art average accuracy of 68.9\% across five popular optimization modeling benchmarks, matching the performance of a 671B LRM. These results demonstrate that dynamic, hint-based data synthesis both preserves and amplifies the native reasoning patterns of modern LRMs, offering a more effective and scalable path towards expert-level performance on challenging optimization modeling tasks. 

---
# Pushing on Multilingual Reasoning Models with Language-Mixed Chain-of-Thought 

**Authors**: Guijin Son, Donghun Yang, Hitesh Laxmichand Patel, Amit Agarwal, Hyunwoo Ko, Chanuk Lim, Srikant Panda, Minhyuk Kim, Nikunj Drolia, Dasol Choi, Kyong-Ha Lee, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04230)  

**Abstract**: Recent frontier models employ long chain-of-thought reasoning to explore solution spaces in context and achieve stonger performance. While many works study distillation to build smaller yet capable models, most focus on English and little is known about language-specific reasoning. To bridge this gap, we first introduct **Language-Mixed CoT**, a reasoning schema that switches between English and a target language, using English as an anchor to excel in reasoning while minimizing translation artificats. As a Korean case study, we curate **Yi-Sang**: 5.79M native-Korean prompts from web Q&A, exams, STEM, and code; 3.7M long reasoning traces generated from Qwen3-32B; and a targeted 260k high-yield subset. We train ninve models (4B-35B) across six families (Qwen2.5, Llama-3.1, Gemma-3, etc). Our best model, **KO-REAson-35B**, achieves state-of-the-art performance, with the highest overall average score (64.0 \pm 25), ranking first on 5/9 benchmarks and second on the remainder. Samller and mid-sized models also benefit substantially, with an average improvement of +18.6 points across teh evaluated nine benchmarks. Ablations show **Language-Mixed CoT** is more effective than monolingual CoT, also resulting in cross-lingual and mult-modal performance gains. We release our data-curation pipeline, evaluation system, datasets, and models to advance research on language-specific reasoning. Data and model collection: this https URL. 

---
# Unveiling LLMs' Metaphorical Understanding: Exploring Conceptual Irrelevance, Context Leveraging and Syntactic Influence 

**Authors**: Fengying Ye, Shanshan Wang, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04120)  

**Abstract**: Metaphor analysis is a complex linguistic phenomenon shaped by context and external factors. While Large Language Models (LLMs) demonstrate advanced capabilities in knowledge integration, contextual reasoning, and creative generation, their mechanisms for metaphor comprehension remain insufficiently explored. This study examines LLMs' metaphor-processing abilities from three perspectives: (1) Concept Mapping: using embedding space projections to evaluate how LLMs map concepts in target domains (e.g., misinterpreting "fall in love" as "drop down from love"); (2) Metaphor-Literal Repository: analyzing metaphorical words and their literal counterparts to identify inherent metaphorical knowledge; and (3) Syntactic Sensitivity: assessing how metaphorical syntactic structures influence LLMs' performance. Our findings reveal that LLMs generate 15\%-25\% conceptually irrelevant interpretations, depend on metaphorical indicators in training data rather than contextual cues, and are more sensitive to syntactic irregularities than to structural comprehension. These insights underline the limitations of LLMs in metaphor analysis and call for more robust computational approaches. 

---
# Equipping Retrieval-Augmented Large Language Models with Document Structure Awareness 

**Authors**: Lingnan Xu, Chong Feng, Kaiyuan Zhang, Liu Zhengyong, Wenqiang Xu, Fanqing Meng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04293)  

**Abstract**: While large language models (LLMs) demonstrate impressive capabilities, their reliance on parametric knowledge often leads to factual inaccuracies. Retrieval-Augmented Generation (RAG) mitigates this by leveraging external documents, yet existing approaches treat retrieved passages as isolated chunks, ignoring valuable structure that is crucial for document organization. Motivated by this gap, we propose Retrieve-DocumentRoute-Read (RDR2), a novel framework that explicitly incorporates structural information throughout the RAG process. RDR2 employs an LLM-based router to dynamically navigate document structure trees, jointly evaluating content relevance and hierarchical relationships to assemble optimal evidence. Our key innovation lies in formulating document routing as a trainable task, with automatic action curation and structure-aware passage selection inspired by human reading strategies. Through comprehensive evaluation on five challenging datasets, RDR2 achieves state-of-the-art performance, demonstrating that explicit structural awareness significantly enhances RAG systems' ability to acquire and utilize knowledge, particularly in complex scenarios requiring multi-document synthesis. 

---
# Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment frm Heterogeneous Rewards 

**Authors**: Zhuoran Zhuang, Ye Chen, Xia Zeng, Chao Luo, Luhui Liu, Yihan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04214)  

**Abstract**: We study deploying large language models (LLMs) as business development (BD) agents for persuasive price negotiation in online travel agencies (OTAs), where aligning traveler affordability and hotel profitability directly affects bookings, partner relationships, and access to travel. The agent must follow a Standard Operating Procedure (SOP) while conducting multi-turn persuasion, interpreting colloquial inputs, and adhering to guardrails (no over-promising, no hallucinations). Conventional post-training -- supervised fine-tuning (SFT) or single-source reward optimization -- overfits scripts, misses nuanced persuasive style, and fails to enforce verifiable business constraints.
We propose Reward-Enhanced Policy Optimization (REPO), a reinforcement learning post-training framework that aligns an LLM with heterogeneous rewards: a preference-trained reward model (RM) for dense human alignment, a reward judge (RJ) for high-level persuasive behavior and SOP compliance, and programmatic reward functions (RF) for deterministic checks on numerics, formatting, and guardrails. A straightforward enhancement mechanism is proposed to combine the RM with RJ and RF signals to curb reward hacking and improve negotiation quality. In production-style evaluations -- approximately 150 turns from real dialogues and 225 turns from curated bad-case dialogues -- REPO lifts average dialogue rating to 4.63: +1.20 over base, +0.83 over Direct Preference Optimization (DPO); +0.33 over Group Relative Policy Optimization (GRPO), increases the share of conversations with at least one excellent response to 66.67% (+23.34 percentage points over GRPO), and achieves a 93.33% bad-case fix rate with 75.56% clean fixes, outperforming SFT, DPO, PPO, and GRPO. We also observe emergent capabilities -- proactive empathy, localized reasoning, calibrated tactics -- that surpass gold annotations. 

---
# Read the Scene, Not the Script: Outcome-Aware Safety for LLMs 

**Authors**: Rui Wu, Yihao Quan, Zeru Shi, Zhenting Wang, Yanshu Li, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04320)  

**Abstract**: Safety-aligned Large Language Models (LLMs) still show two dominant failure modes: they are easily jailbroken, or they over-refuse harmless inputs that contain sensitive surface signals. We trace both to a common cause: current models reason weakly about links between actions and outcomes and over-rely on surface-form signals, lexical or stylistic cues that do not encode consequences. We define this failure mode as Consequence-blindness. To study consequence-blindness, we build a benchmark named CB-Bench covering four risk scenarios that vary whether semantic risk aligns with outcome risk, enabling evaluation under both matched and mismatched conditions which are often ignored by existing safety benchmarks. Mainstream models consistently fail to separate these risks and exhibit consequence-blindness, indicating that consequence-blindness is widespread and systematic. To mitigate consequence-blindness, we introduce CS-Chain-4k, a consequence-reasoning dataset for safety alignment. Models fine-tuned on CS-Chain-4k show clear gains against semantic-camouflage jailbreaks and reduce over-refusal on harmless inputs, while maintaining utility and generalization on other benchmarks. These results clarify the limits of current alignment, establish consequence-aware reasoning as a core alignment goal and provide a more practical and reproducible evaluation path. 

---
# Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization 

**Authors**: Wengao Ye, Yan Liang, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04182)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have shifted from explicit Chain-of-Thought (CoT) reasoning to more efficient latent reasoning, where intermediate thoughts are represented as vectors rather than text. However, latent reasoning can be brittle on challenging, out-of-distribution tasks where robust reasoning is most critical. To overcome these limitations, we introduce Latent Thought Policy Optimization (LTPO), a parameter-free framework that enhances LLM reasoning entirely at test time, without requiring model parameter updates. LTPO treats intermediate latent "thought" vectors as dynamic parameters that are actively optimized for each problem instance. It employs an online policy gradient method guided by an intrinsic, confidence-based reward signal computed directly from the frozen LLM's own output distributions, eliminating the need for external supervision or expensive text generation during optimization. Extensive experiments on five reasoning benchmarks show that LTPO not only matches or surpasses strong baselines on standard tasks but also demonstrates remarkable robustness where others fail. Most notably, on highly challenging AIME benchmarks where existing latent reasoning baselines collapse to near-zero accuracy, LTPO delivers substantial improvements, showcasing a unique capability for complex reasoning. 

---
# Does Using Counterfactual Help LLMs Explain Textual Importance in Classification? 

**Authors**: Nelvin Tan, James Asikin Cheung, Yu-Ching Shih, Dong Yang, Amol Salunkhe  

**Link**: [PDF](https://arxiv.org/pdf/2510.04031)  

**Abstract**: Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. More recently, they have been shown to be very effective in textual classification tasks, motivating the need to explain the LLMs' decisions. Motivated by practical constrains where LLMs are black-boxed and LLM calls are expensive, we study how incorporating counterfactuals into LLM reasoning can affect the LLM's ability to identify the top words that have contributed to its classification decision. To this end, we introduce a framework called the decision changing rate that helps us quantify the importance of the top words in classification. Our experimental results show that using counterfactuals can be helpful. 

---
# Exploring Chain-of-Thought Reasoning for Steerable Pluralistic Alignment 

**Authors**: Yunfan Zhang, Kathleen McKeown, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04045)  

**Abstract**: Large Language Models (LLMs) are typically trained to reflect a relatively uniform set of values, which limits their applicability to tasks that require understanding of nuanced human perspectives. Recent research has underscored the importance of enabling LLMs to support steerable pluralism -- the capacity to adopt a specific perspective and align generated outputs with it. In this work, we investigate whether Chain-of-Thought (CoT) reasoning techniques can be applied to building steerable pluralistic models. We explore several methods, including CoT prompting, fine-tuning on human-authored CoT, fine-tuning on synthetic explanations, and Reinforcement Learning with Verifiable Rewards (RLVR). We evaluate these approaches using the Value Kaleidoscope and OpinionQA datasets. Among the methods studied, RLVR consistently outperforms others and demonstrates strong training sample efficiency. We further analyze the generated CoT traces with respect to faithfulness and safety. 

---
# PoLi-RL: A Point-to-List Reinforcement Learning Framework for Conditional Semantic Textual Similarity 

**Authors**: Zixin Song, Bowen Zhang, Qian-Wen Zhang, Di Yin, Xing Sun, Chunping Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.04080)  

**Abstract**: Conditional Semantic Textual Similarity (C-STS) measures the semantic proximity between text segments under a specific condition, thereby overcoming the ambiguity inherent in traditional STS. However, existing methods are largely confined to discriminative models, failing to fully integrate recent breakthroughs in the NLP community concerning Large Language Models (LLMs) and Reinforcement Learning (RL). RL is a particularly well-suited paradigm for this task, as it can directly optimize the non-differentiable Spearman ranking metric and guide the reasoning process required by C-STS. However, we find that naively applying listwise RL fails to produce meaningful improvements, as the model is overwhelmed by complex, coarse-grained reward signals. To address this challenge, we introduce PoLi-RL, a novel Point-to-List Reinforcement Learning framework. PoLi-RL employs a two-stage curriculum: it first trains the model with simple pointwise rewards to establish fundamental scoring capabilities, then transitions to a hybrid reward that combines pointwise, pairwise, and listwise objectives to refine the model's ability to discern subtle semantic distinctions. Crucially, we propose an innovative Parallel Slice Ranking Reward (PSRR) mechanism that computes ranking rewards in parallel slices, where each slice comprises same-indexed completions from different samples. This provides a precise, differentiated learning signal for each individual completion, enabling granular credit assignment and effective optimization. On the official C-STS benchmark, PoLi-RL achieves a Spearman correlation coefficient of 48.18, establishing a new SOTA for the cross-encoder architecture. As the first work to successfully apply RL to C-STS, our study introduces a powerful and precise paradigm for training LLMs on complex, ranking-based conditional judgment tasks. 

---
# LLM Microscope: What Model Internals Reveal About Answer Correctness and Context Utilization 

**Authors**: Jiarui Liu, Jivitesh Jain, Mona Diab, Nishant Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2510.04013)  

**Abstract**: Although large language models (LLMs) have tremendous utility, trustworthiness is still a chief concern: models often generate incorrect information with high confidence. While contextual information can help guide generation, identifying when a query would benefit from retrieved context and assessing the effectiveness of that context remains challenging. In this work, we operationalize interpretability methods to ascertain whether we can predict the correctness of model outputs from the model's activations alone. We also explore whether model internals contain signals about the efficacy of external context. We consider correct, incorrect, and irrelevant context and introduce metrics to distinguish amongst them. Experiments on six different models reveal that a simple classifier trained on intermediate layer activations of the first output token can predict output correctness with about 75% accuracy, enabling early auditing. Our model-internals-based metric significantly outperforms prompting baselines at distinguishing between correct and incorrect context, guarding against inaccuracies introduced by polluted context. These findings offer a lens to better understand the underlying decision-making processes of LLMs. Our code is publicly available at this https URL 

---
# AgriGPT-VL: Agricultural Vision-Language Understanding Suite 

**Authors**: Bo Yang, Yunkui Chen, Lanfei Feng, Yu Zhang, Xiao Xu, Jianyu Zhang, Nueraili Aierken, Runhe Huang, Hongjian Lin, Yibin Ying, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.04002)  

**Abstract**: Despite rapid advances in multimodal large language models, agricultural applications remain constrained by the scarcity of domain-tailored models, curated vision-language corpora, and rigorous evaluation. To address these challenges, we present the AgriGPT-VL Suite, a unified multimodal framework for agriculture. Our contributions are threefold. First, we introduce Agri-3M-VL, the largest vision-language corpus for agriculture to our knowledge, curated by a scalable multi-agent data generator; it comprises 1M image-caption pairs, 2M image-grounded VQA pairs, 50K expert-level VQA instances, and 15K GRPO reinforcement learning samples. Second, we develop AgriGPT-VL, an agriculture-specialized vision-language model trained via a progressive curriculum of textual grounding, multimodal shallow/deep alignment, and GRPO refinement. This method achieves strong multimodal reasoning while preserving text-only capability. Third, we establish AgriBench-VL-4K, a compact yet challenging evaluation suite with open-ended and image-grounded questions, paired with multi-metric evaluation and an LLM-as-a-judge framework. Experiments show that AgriGPT-VL outperforms leading general-purpose VLMs on AgriBench-VL-4K, achieving higher pairwise win rates in the LLM-as-a-judge evaluation. Meanwhile, it remains competitive on the text-only AgriBench-13K with no noticeable degradation of language ability. Ablation studies further confirm consistent gains from our alignment and GRPO refinement stages. We will open source all of the resources to support reproducible research and deployment in low-resource agricultural settings. 

---
# Scaling Code-Assisted Chain-of-Thoughts and Instructions for Model Reasoning 

**Authors**: Honglin Lin, Qizhi Pei, Xin Gao, Zhuoshi Pan, Yu Li, Juntao Li, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04081)  

**Abstract**: Reasoning capability is pivotal for Large Language Models (LLMs) to solve complex tasks, yet achieving reliable and scalable reasoning remains challenging. While Chain-of-Thought (CoT) prompting has become a mainstream approach, existing methods often suffer from uncontrolled generation, insufficient quality, and limited diversity in reasoning paths. Recent efforts leverage code to enhance CoT by grounding reasoning in executable steps, but such methods are typically constrained to predefined mathematical problems, hindering scalability and generalizability. In this work, we propose Caco (Code-Assisted Chain-of-ThOught), a novel framework that automates the synthesis of high-quality, verifiable, and diverse instruction-CoT reasoning data through code-driven augmentation. Unlike prior work, Caco first fine-tunes a code-based CoT generator on existing math and programming solutions in a unified code format, then scales the data generation to a large amount of diverse reasoning traces. Crucially, we introduce automated validation via code execution and rule-based filtering to ensure logical correctness and structural diversity, followed by reverse-engineering filtered outputs into natural language instructions and language CoTs to enrich task adaptability. This closed-loop process enables fully automated, scalable synthesis of reasoning data with guaranteed executability. Experiments on our created Caco-1.3M dataset demonstrate that Caco-trained models achieve strong competitive performance on mathematical reasoning benchmarks, outperforming existing strong baselines. Further analysis reveals that Caco's code-anchored verification and instruction diversity contribute to superior generalization across unseen tasks. Our work establishes a paradigm for building self-sustaining, trustworthy reasoning systems without human intervention. 

---
# Mapping Patient-Perceived Physician Traits from Nationwide Online Reviews with LLMs 

**Authors**: Junjie Luo, Rui Han, Arshana Welivita, Zeleikun Di, Jingfu Wu, Xuzhe Zhi, Ritu Agarwal, Gordon Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03997)  

**Abstract**: Understanding how patients perceive their physicians is essential to improving trust, communication, and satisfaction. We present a large language model (LLM)-based pipeline that infers Big Five personality traits and five patient-oriented subjective judgments. The analysis encompasses 4.1 million patient reviews of 226,999 U.S. physicians from an initial pool of one million. We validate the method through multi-model comparison and human expert benchmarking, achieving strong agreement between human and LLM assessments (correlation coefficients 0.72-0.89) and external validity through correlations with patient satisfaction (r = 0.41-0.81, all p<0.001). National-scale analysis reveals systematic patterns: male physicians receive higher ratings across all traits, with largest disparities in clinical competence perceptions; empathy-related traits predominate in pediatrics and psychiatry; and all traits positively predict overall satisfaction. Cluster analysis identifies four distinct physician archetypes, from "Well-Rounded Excellent" (33.8%, uniformly high traits) to "Underperforming" (22.6%, consistently low). These findings demonstrate that automated trait extraction from patient narratives can provide interpretable, validated metrics for understanding physician-patient relationships at scale, with implications for quality measurement, bias detection, and workforce development in healthcare. 

---
# Simulating and Understanding Deceptive Behaviors in Long-Horizon Interactions 

**Authors**: Yang Xu, Xuanming Zhang, Min-Hsuan Yeh, Jwala Dhamala, Ousmane Dia, Rahul Gupta, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03999)  

**Abstract**: Deception is a pervasive feature of human communication and an emerging concern in large language models (LLMs). While recent studies document instances of LLM deception under pressure, most evaluations remain confined to single-turn prompts and fail to capture the long-horizon interactions in which deceptive strategies typically unfold. We introduce the first simulation framework for probing and evaluating deception in LLMs under extended sequences of interdependent tasks and dynamic contextual pressures. Our framework instantiates a multi-agent system: a performer agent tasked with completing tasks and a supervisor agent that evaluates progress, provides feedback, and maintains evolving states of trust. An independent deception auditor then reviews full trajectories to identify when and how deception occurs. We conduct extensive experiments across 11 frontier models, spanning both closed- and open-source systems, and find that deception is model-dependent, increases with event pressure, and consistently erodes supervisor trust. Qualitative analyses further reveal distinct strategies of concealment, equivocation, and falsification. Our findings establish deception as an emergent risk in long-horizon interactions and provide a foundation for evaluating future LLMs in real-world, trust-sensitive contexts. 

---
# Read Between the Lines: A Benchmark for Uncovering Political Bias in Bangla News Articles 

**Authors**: Nusrat Jahan Lia, Shubhashis Roy Dipta, Abdullah Khan Zehady, Naymul Islam, Madhusodan Chakraborty, Abdullah Al Wasif  

**Link**: [PDF](https://arxiv.org/pdf/2510.03898)  

**Abstract**: Detecting media bias is crucial, specifically in the South Asian region. Despite this, annotated datasets and computational studies for Bangla political bias research remain scarce. Crucially because, political stance detection in Bangla news requires understanding of linguistic cues, cultural context, subtle biases, rhetorical strategies, code-switching, implicit sentiment, and socio-political background. To address this, we introduce the first benchmark dataset of 200 politically significant and highly debated Bangla news articles, labeled for government-leaning, government-critique, and neutral stances, alongside diagnostic analyses for evaluating large language models (LLMs). Our comprehensive evaluation of 28 proprietary and open-source LLMs shows strong performance in detecting government-critique content (F1 up to 0.83) but substantial difficulty with neutral articles (F1 as low as 0.00). Models also tend to over-predict government-leaning stances, often misinterpreting ambiguous narratives. This dataset and its associated diagnostics provide a foundation for advancing stance detection in Bangla media research and offer insights for improving LLM performance in low-resource languages. 

---
# Beyond Token Length: Step Pruner for Efficient and Accurate Reasoning in Large Language Models 

**Authors**: Canhui Wu, Qiong Cao, Chang Li, Zhenfang Wang, Chao Xue, Yuwei Fan, Wei Xi, Xiaodong He  

**Link**: [PDF](https://arxiv.org/pdf/2510.03805)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate strong performance on complex tasks but often suffer from excessive verbosity, known as "overthinking." Existing solutions via reinforcement learning (RL) typically penalize generated tokens to promote conciseness. However, these methods encounter two challenges: responses with fewer tokens do not always correspond to fewer reasoning steps, and models may develop hacking behavior in later stages of training by discarding reasoning steps to minimize token usage. In this work, we introduce \textbf{Step Pruner (SP)}, an RL framework that steers LRMs toward more efficient reasoning by favoring compact reasoning steps. Our step-aware reward function prioritizes correctness while imposing penalties for redundant steps, and withholds rewards for incorrect responses to prevent the reinforcement of erroneous reasoning. Moreover, we propose a dynamic stopping mechanism: when the length of any output step exceeds the upper limit, we halt updates to prevent hacking behavior caused by merging steps. Extensive experiments across four reasoning benchmarks demonstrate that SP achieves state-of-the-art accuracy while significantly reducing response length. For instance, on AIME24, SP reduces token usage by \textbf{69.7\%}. 

---
# MedReflect: Teaching Medical LLMs to Self-Improve via Reflective Correction 

**Authors**: Yue Huang, Yanyuan Chen, Dexuan Xu, Weihua Yue, Huamin Zhang, Meikang Qiu, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03687)  

**Abstract**: Medical problem solving demands expert knowledge and intricate reasoning. Recent studies of large language models (LLMs) attempt to ease this complexity by introducing external knowledge verification through retrieval-augmented generation or by training on reasoning datasets. However, these approaches suffer from drawbacks such as retrieval overhead and high annotation costs, and they heavily rely on substituted external assistants to reach limited performance in medical field. In this paper, we introduce MedReflect, a generalizable framework designed to inspire LLMs with a physician-like reflective thinking mode. MedReflect generates a single-pass reflection chain that includes initial hypothesis generation, self-questioning, self-answering and decision refinement. This self-verified and self-reflective nature releases large language model's latent capability in medical problem-solving without external retrieval or heavy annotation. We demonstrate that MedReflect enables cost-efficient medical dataset construction: with merely 2,000 randomly sampled training examples and a light fine-tuning, this approach achieves notable absolute accuracy improvements across a series of medical benchmarks while cutting annotation requirements. Our results provide evidence that LLMs can learn to solve specialized medical problems via self-reflection and self-improve, reducing reliance on external supervision and extensive task-specific fine-tuning data. 

---
# Fine-Tuning Large Language Models with QLoRA for Offensive Language Detection in Roman Urdu-English Code-Mixed Text 

**Authors**: Nisar Hussain, Amna Qasim, Gull Mehak, Muhammad Zain, Momina Hafeez, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2510.03683)  

**Abstract**: The use of derogatory terms in languages that employ code mixing, such as Roman Urdu, presents challenges for Natural Language Processing systems due to unstated grammar, inconsistent spelling, and a scarcity of labeled data. In this work, we propose a QLoRA based fine tuning framework to improve offensive language detection in Roman Urdu-English text. We translated the Roman Urdu-English code mixed dataset into English using Google Translate to leverage English LLMs, while acknowledging that this translation reduces direct engagement with code mixing features. Our focus is on classification performance using English translated low resource inputs. We fine tuned several transformers and large language models, including Meta LLaMA 3 8B, Mistral 7B v0.1, LLaMA 2 7B, ModernBERT, and RoBERTa, with QLoRA for memory efficient adaptation. Models were trained and evaluated on a manually annotated Roman Urdu dataset for offensive vs non offensive content. Of all tested models, the highest F1 score of 91.45 was attained by Meta LLaMA 3 8B, followed by Mistral 7B at 89.66, surpassing traditional transformer baselines. These results demonstrate the efficacy of QLoRA in fine tuning high performing models for low resource environments such as code mixed offensive language detection, and confirm the potential of LLMs for this task. This work advances a scalable approach to Roman Urdu moderation and paves the way for future multilingual offensive detection systems based on LLMs. 

---
# Decoupling Task-Solving and Output Formatting in LLM Generation 

**Authors**: Haikang Deng, Po-Nien Kung, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.03595)  

**Abstract**: Large language models (LLMs) are increasingly adept at following instructions containing task descriptions to solve complex problems, such as mathematical reasoning and automatic evaluation (LLM-as-a-Judge). However, as prompts grow more complex, models often struggle to adhere to all instructions. This difficulty is especially common when instructive prompts intertwine reasoning directives -- specifying what the model should solve -- with rigid formatting requirements that dictate how the solution must be presented. The entanglement creates competing goals for the model, suggesting that more explicit separation of these two aspects could lead to improved performance. To this front, we introduce Deco-G, a decoding framework that explicitly decouples format adherence from task solving. Deco-G handles format compliance with a separate tractable probabilistic model (TPM), while prompts LLMs with only task instructions. At each decoding step, Deco-G combines next token probabilities from the LLM with the TPM calculated format compliance likelihood to form the output probability. To make this approach both practical and scalable for modern instruction-tuned LLMs, we introduce three key innovations: instruction-aware distillation, a flexible trie-building algorithm, and HMM state pruning for computational efficiency. We demonstrate the effectiveness of Deco-G across a wide range of tasks with diverse format requirements, including mathematical reasoning, LLM-as-a-judge, and event argument extraction. Overall, our approach yields 1.0% to 6.0% relative gain over regular prompting practice with guaranteed format compliance. 

---
# Can an LLM Induce a Graph? Investigating Memory Drift and Context Length 

**Authors**: Raquib Bin Yousuf, Aadyant Khatri, Shengzhe Xu, Mandar Sharma, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03611)  

**Abstract**: Recently proposed evaluation benchmarks aim to characterize the effective context length and the forgetting tendencies of large language models (LLMs). However, these benchmarks often rely on simplistic 'needle in a haystack' retrieval or continuation tasks that may not accurately reflect the performance of these models in information-dense scenarios. Thus, rather than simple next token prediction, we argue for evaluating these models on more complex reasoning tasks that requires them to induce structured relational knowledge from the text - such as graphs from potentially noisy natural language content. While the input text can be viewed as generated in terms of a graph, its structure is not made explicit and connections must be induced from distributed textual cues, separated by long contexts and interspersed with irrelevant information. Our findings reveal that LLMs begin to exhibit memory drift and contextual forgetting at much shorter effective lengths when tasked with this form of relational reasoning, compared to what existing benchmarks suggest. With these findings, we offer recommendations for the optimal use of popular LLMs for complex reasoning tasks. We further show that even models specialized for reasoning, such as OpenAI o1, remain vulnerable to early memory drift in these settings. These results point to significant limitations in the models' ability to abstract structured knowledge from unstructured input and highlight the need for architectural adaptations to improve long-range reasoning. 

---
# CCD-Bench: Probing Cultural Conflict in Large Language Model Decision-Making 

**Authors**: Hasibur Rahman, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2510.03553)  

**Abstract**: Although large language models (LLMs) are increasingly implicated in interpersonal and societal decision-making, their ability to navigate explicit conflicts between legitimately different cultural value systems remains largely unexamined. Existing benchmarks predominantly target cultural knowledge (CulturalBench), value prediction (WorldValuesBench), or single-axis bias diagnostics (CDEval); none evaluate how LLMs adjudicate when multiple culturally grounded values directly clash. We address this gap with CCD-Bench, a benchmark that assesses LLM decision-making under cross-cultural value conflict. CCD-Bench comprises 2,182 open-ended dilemmas spanning seven domains, each paired with ten anonymized response options corresponding to the ten GLOBE cultural clusters. These dilemmas are presented using a stratified Latin square to mitigate ordering effects. We evaluate 17 non-reasoning LLMs. Models disproportionately prefer Nordic Europe (mean 20.2 percent) and Germanic Europe (12.4 percent), while options for Eastern Europe and the Middle East and North Africa are underrepresented (5.6 to 5.8 percent). Although 87.9 percent of rationales reference multiple GLOBE dimensions, this pluralism is superficial: models recombine Future Orientation and Performance Orientation, and rarely ground choices in Assertiveness or Gender Egalitarianism (both under 3 percent). Ordering effects are negligible (Cramer's V less than 0.10), and symmetrized KL divergence shows clustering by developer lineage rather than geography. These patterns suggest that current alignment pipelines promote a consensus-oriented worldview that underserves scenarios demanding power negotiation, rights-based reasoning, or gender-aware analysis. CCD-Bench shifts evaluation beyond isolated bias detection toward pluralistic decision making and highlights the need for alignment strategies that substantively engage diverse worldviews. 

---
# TriMediQ: A Triplet-Structured Approach for Interactive Medical Question Answering 

**Authors**: Zhaohan Meng, Zaiqiao Meng, Siwei Liu, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03536)  

**Abstract**: Large Language Models (LLMs) perform strongly in static and single-turn medical Question Answer (QA) benchmarks, yet such settings diverge from the iterative information gathering process required in practical clinical consultations. The MEDIQ framework addresses this mismatch by recasting the diagnosis as an interactive dialogue between a patient and an expert system, but the reliability of LLMs drops dramatically when forced to reason with dialogue logs, where clinical facts appear in sentences without clear links. To bridge this gap, we introduce TriMediQ, a triplet-structured approach that summarises patient responses into triplets and integrates them into a Knowledge Graph (KG), enabling multi-hop reasoning. We introduce a frozen triplet generator that extracts clinically relevant triplets, using prompts designed to ensure factual consistency. In parallel, a trainable projection module, comprising a graph encoder and a projector, captures relational information from the KG to enhance expert reasoning. TriMediQ operates in two steps: (i) the projection module fine-tuning with all LLM weights frozen; and (ii) using the fine-tuned module to guide multi-hop reasoning during inference. We evaluate TriMediQ on two interactive QA benchmarks, showing that it achieves up to 10.4\% improvement in accuracy over five baselines on the iMedQA dataset. These results demonstrate that converting patient responses into structured triplet-based graphs enables more accurate clinical reasoning in multi-turn settings, providing a solution for the deployment of LLM-based medical assistants. 

---
# Fine-Tuning on Noisy Instructions: Effects on Generalization and Performance 

**Authors**: Ahmed Alajrami, Xingwei Tan, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2510.03528)  

**Abstract**: Instruction-tuning plays a vital role in enhancing the task-solving abilities of large language models (LLMs), improving their usability in generating helpful responses on various tasks. However, previous work has demonstrated that they are sensitive to minor variations in instruction phrasing. In this paper, we explore whether introducing perturbations in instruction-tuning data can enhance LLMs' resistance against noisy instructions. We focus on how instruction-tuning with perturbations, such as removing stop words or shuffling words, affects LLMs' performance on the original and perturbed versions of widely-used benchmarks (MMLU, BBH, GSM8K). We further assess learning dynamics and potential shifts in model behavior. Surprisingly, our results suggest that instruction-tuning on perturbed instructions can, in some cases, improve downstream performance. These findings highlight the importance of including perturbed instructions in instruction-tuning, which can make LLMs more resilient to noisy user inputs. 

---
# Rezwan: Leveraging Large Language Models for Comprehensive Hadith Text Processing: A 1.2M Corpus Development 

**Authors**: Majid Asgari-Bidhendi, Muhammad Amin Ghaseminia, Alireza Shahbazi, Sayyed Ali Hossayni, Najmeh Torabian, Behrouz Minaei-Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.03781)  

**Abstract**: This paper presents the development of Rezwan, a large-scale AI-assisted Hadith corpus comprising over 1.2M narrations, extracted and structured through a fully automated pipeline. Building on digital repositories such as Maktabat Ahl al-Bayt, the pipeline employs Large Language Models (LLMs) for segmentation, chain--text separation, validation, and multi-layer enrichment. Each narration is enhanced with machine translation into twelve languages, intelligent diacritization, abstractive summarization, thematic tagging, and cross-text semantic analysis. This multi-step process transforms raw text into a richly annotated research-ready infrastructure for digital humanities and Islamic studies. A rigorous evaluation was conducted on 1,213 randomly sampled narrations, assessed by six domain experts. Results show near-human accuracy in structured tasks such as chain--text separation (9.33/10) and summarization (9.33/10), while highlighting ongoing challenges in diacritization and semantic similarity detection. Comparative analysis against the manually curated Noor Corpus demonstrates the superiority of Najm in both scale and quality, with a mean overall score of 8.46/10 versus 3.66/10. Furthermore, cost analysis confirms the economic feasibility of the AI approach: tasks requiring over 229,000 hours of expert labor were completed within months at a fraction of the cost. The work introduces a new paradigm in religious text processing by showing how AI can augment human expertise, enabling large-scale, multilingual, and semantically enriched access to Islamic heritage. 

---
# Prompt Balance Matters: Understanding How Imbalanced Few-Shot Learning Affects Multilingual Sense Disambiguation in LLMs 

**Authors**: Deshan Sumanathilaka, Nicholas Micallef, Julian Hough  

**Link**: [PDF](https://arxiv.org/pdf/2510.03762)  

**Abstract**: Recent advances in Large Language Models (LLMs) have significantly reshaped the landscape of Natural Language Processing (NLP). Among the various prompting techniques, few-shot prompting has gained considerable attention for its practicality and effectiveness. This study investigates how few-shot prompting strategies impact the Word Sense Disambiguation (WSD) task, particularly focusing on the biases introduced by imbalanced sample distributions. We use the GLOSSGPT prompting method, an advanced approach for English WSD, to test its effectiveness across five languages: English, German, Spanish, French, and Italian. Our results show that imbalanced few-shot examples can cause incorrect sense predictions in multilingual languages, but this issue does not appear in English. To assess model behavior, we evaluate both the GPT-4o and LLaMA-3.1-70B models and the results highlight the sensitivity of multilingual WSD to sample distribution in few-shot settings, emphasizing the need for balanced and representative prompting strategies. 

---
# From Noisy Traces to Stable Gradients: Bias-Variance Optimized Preference Optimization for Aligning Large Reasoning Models 

**Authors**: Mingkang Zhu, Xi Chen, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.05095)  

**Abstract**: Large reasoning models (LRMs) generate intermediate reasoning traces before producing final answers, yielding strong gains on multi-step and mathematical tasks. Yet aligning LRMs with human preferences, a crucial prerequisite for model deployment, remains underexplored. The statistically correct objective for preference alignment requires marginalizing over reasoning traces, but this computation is intractable in practice. A common workaround optimizes a single sampled trajectory, which introduces substantial gradient variance from stochastic trace sampling. To address this challenge, we frame preference optimization for LRMs through the lens of the bias--variance trade-off and propose Bias--Variance Optimized Preference Optimization (BVPO), a simple, drop-in method that mixes two gradient estimators: a high-variance trace-based estimator and a low-variance empty-trace estimator obtained by disabling reasoning trace generation. Our theory shows that BVPO strictly reduces trace-induced variance for any nontrivial mixture, provides a closed-form choice of the mixing weight that minimizes mean-squared error relative to the true marginal gradient, and under standard smoothness and step-size conditions, tightens classical convergence bounds for stochastic gradient descent. Empirically, BVPO improves alignment over the best baseline by up to 7.8 points on AlpacaEval~2 and 6.8 points on Arena-Hard. Despite being trained only on general conversational data, BVPO also boosts reasoning performance for base models by up to 4.0 points on the average of six math reasoning benchmarks. These results identify variance from trace sampling as a key bottleneck and demonstrate that directly optimizing the bias--variance trade-off yields more stable training and stronger overall performance. 

---
# LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game 

**Authors**: Fangzhou Liang, Tianshi Zheng, Chunkit Chan, Yauwai Yim, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.04980)  

**Abstract**: Effective multi-agent collaboration requires agents to infer the rationale behind others' actions, a capability rooted in Theory-of-Mind (ToM). While recent Large Language Models (LLMs) excel at logical inference, their ability to infer rationale in dynamic, collaborative settings remains under-explored. This study introduces LLM-Hanabi, a novel benchmark that uses the cooperative game Hanabi to evaluate the rationale inference and ToM of LLMs. Our framework features an automated evaluation system that measures both game performance and ToM proficiency. Across a range of models, we find a significant positive correlation between ToM and in-game success. Notably, first-order ToM (interpreting others' intent) correlates more strongly with performance than second-order ToM (predicting others' interpretations). These findings highlight that for effective AI collaboration, the ability to accurately interpret a partner's rationale is more critical than higher-order reasoning. We conclude that prioritizing first-order ToM is a promising direction for enhancing the collaborative capabilities of future models. 

---
# Reactive Transformer (RxT) -- Stateful Real-Time Processing for Event-Driven Reactive Language Models 

**Authors**: Adam Filipek  

**Link**: [PDF](https://arxiv.org/pdf/2510.03561)  

**Abstract**: The Transformer architecture has become the de facto standard for Large Language Models (LLMs), demonstrating remarkable capabilities in language understanding and generation. However, its application in conversational AI is fundamentally constrained by its stateless nature and the quadratic computational complexity ($O(L^2)$) with respect to sequence length $L$. Current models emulate memory by reprocessing an ever-expanding conversation history with each turn, leading to prohibitive costs and latency in long dialogues. This paper introduces the Reactive Transformer (RxT), a novel architecture designed to overcome these limitations by shifting from a data-driven to an event-driven paradigm. RxT processes each conversational turn as a discrete event in real-time, maintaining context in an integrated, fixed-size Short-Term Memory (STM) system. The architecture features a distinct operational cycle where a generator-decoder produces a response based on the current query and the previous memory state, after which a memory-encoder and a dedicated Memory Attention network asynchronously update the STM with a representation of the complete interaction. This design fundamentally alters the scaling dynamics, reducing the total user-facing cost of a conversation from quadratic ($O(N^2 \cdot T)$) to linear ($O(N \cdot T)$) with respect to the number of interactions $N$. By decoupling response generation from memory updates, RxT achieves low latency, enabling truly real-time, stateful, and economically viable long-form conversations. We validated our architecture with a series of proof-of-concept experiments on synthetic data, demonstrating superior performance and constant-time inference latency compared to a baseline stateless model of comparable size. 

---
# MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning 

**Authors**: Guoxin Chen, Zile Qiao, Wenqing Wang, Donglei Yu, Xuanzhong Chen, Hao Sun, Minpeng Liao, Kai Fan, Yong Jiang, Penguin Xie, Wayne Xin Zhao, Ruihua Song, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04935)  

**Abstract**: Large Reasoning Models (LRMs) often exhibit a tendency for overanalysis in simple tasks, where the models excessively utilize System 2-type, deliberate reasoning, leading to inefficient token generation. Furthermore, these models face challenges in adapting their reasoning capabilities to rapidly changing environments due to the static nature of their pretraining data. To address these issues, advancing Large Language Models (LLMs) for complex reasoning tasks requires innovative approaches that bridge intuitive and deliberate cognitive processes, akin to human cognition's dual-system dynamic. This paper introduces a Multi-Agent System for Deep ReSearch (MARS) enabling seamless integration of System 1's fast, intuitive thinking with System 2's deliberate reasoning within LLMs. MARS strategically integrates multiple external tools, such as Google Search, Google Scholar, and Python Interpreter, to access up-to-date information and execute complex computations, while creating a specialized division of labor where System 1 efficiently processes and summarizes high-volume external information, providing distilled insights that expand System 2's reasoning context without overwhelming its capacity. Furthermore, we propose a multi-agent reinforcement learning framework extending Group Relative Policy Optimization to simultaneously optimize both systems with multi-turn tool interactions, bin-packing optimization, and sample balancing strategies that enhance collaborative efficiency. Extensive experiments demonstrate MARS achieves substantial improvements of 3.86% on the challenging Humanity's Last Exam (HLE) benchmark and an average gain of 8.9% across 7 knowledge-intensive tasks, validating the effectiveness of our dual-system paradigm for complex reasoning in dynamic information environments. 

---
# Large Language Models Achieve Gold Medal Performance at International Astronomy & Astrophysics Olympiad 

**Authors**: Lucas Carrit Delgado Pinheiro, Ziru Chen, Bruno Caixeta Piazza, Ness Shroff, Yingbin Liang, Yuan-Sen Ting, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.05016)  

**Abstract**: While task-specific demonstrations show early success in applying large language models (LLMs) to automate some astronomical research tasks, they only provide incomplete views of all necessary capabilities in solving astronomy problems, calling for more thorough understanding of LLMs' strengths and limitations. So far, existing benchmarks and evaluations focus on simple question-answering that primarily tests astronomical knowledge and fails to evaluate the complex reasoning required for real-world research in the discipline. Here, we address this gap by systematically benchmarking five state-of-the-art LLMs on the International Olympiad on Astronomy and Astrophysics (IOAA) exams, which are designed to examine deep conceptual understanding, multi-step derivations, and multimodal analysis. With average scores of 85.6% and 84.2%, Gemini 2.5 Pro and GPT-5 (the two top-performing models) not only achieve gold medal level performance but also rank in the top two among ~200-300 participants in all four IOAA theory exams evaluated (2022-2025). In comparison, results on the data analysis exams show more divergence. GPT-5 still excels in the exams with an 88.5% average score, ranking top 10 among the participants in the four most recent IOAAs, while other models' performances drop to 48-76%. Furthermore, our in-depth error analysis underscores conceptual reasoning, geometric reasoning, and spatial visualization (52-79% accuracy) as consistent weaknesses among all LLMs. Hence, although LLMs approach peak human performance in theory exams, critical gaps must be addressed before they can serve as autonomous research agents in astronomy. 

---
# Proactive defense against LLM Jailbreak 

**Authors**: Weiliang Zhao, Jinjun Peng, Daniel Ben-Levi, Zhou Yu, Junfeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05052)  

**Abstract**: The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks. 

---
# AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Crystalline Materials 

**Authors**: Taoyuze Lv, Alexander Chen, Fengyu Xie, Chu Wu, Jeffrey Meng, Dongzhan Zhou, Bram Hoex, Zhicheng Zhong, Tong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.04704)  

**Abstract**: Large Language Models (LLMs) excel at textual reasoning and are beginning to develop spatial understanding, prompting the question of whether these abilities can be combined for complex, domain-specific tasks. This question is essential in fields like materials science, where deep understanding of 3D atomic structures is fundamental. While initial studies have successfully applied LLMs to tasks involving pure crystal generation or coordinate understandings, a standardized benchmark to systematically evaluate their core reasoning abilities across diverse atomic structures has been notably absent. To address this gap, we introduce the AtomWorld benchmark to evaluate LLMs on tasks based in Crystallographic Information Files (CIFs), a standard structure representation format. These tasks, including structural editing, CIF perception, and property-guided modeling, reveal a critical limitation: current models, despite establishing promising baselines, consistently fail in structural understanding and spatial reasoning. Our experiments show that these models make frequent errors on structure modification tasks, and even in the basic CIF format understandings, potentially leading to cumulative errors in subsequent analysis and materials insights. By defining these standardized tasks, AtomWorld lays the ground for advancing LLMs toward robust atomic-scale modeling, crucial for accelerating materials research and automating scientific workflows. 

---
# Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models 

**Authors**: Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, Kunle Olukotun  

**Link**: [PDF](https://arxiv.org/pdf/2510.04618)  

**Abstract**: Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on context adaptation -- modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. Building on the adaptive memory introduced by Dynamic Cheatsheet, we introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead. 

---
# LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning 

**Authors**: Haoqiang Kang, Yizhe Zhang, Nikki Lijing Kuang, Nicklas Majamaki, Navdeep Jaitly, Yi-An Ma, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04573)  

**Abstract**: Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion. 

---
# Visual Representations inside the Language Model 

**Authors**: Benlin Liu, Amita Kamath, Madeleine Grunde-McLaughlin, Winson Han, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2510.04819)  

**Abstract**: Despite interpretability work analyzing VIT encoders and transformer activations, we don't yet understand why Multimodal Language Models (MLMs) struggle on perception-heavy tasks. We offer an under-studied perspective by examining how popular MLMs (LLaVA-OneVision, Qwen2.5-VL, and Llama-3-LLaVA-NeXT) process their visual key-value tokens. We first study the flow of visual information through the language model, finding that image value tokens encode sufficient information to perform several perception-heavy tasks zero-shot: segmentation, semantic correspondence, temporal correspondence, and referring expression detection. We find that while the language model does augment the visual information received from the projection of input visual encodings-which we reveal correlates with overall MLM perception capability-it contains less visual information on several tasks than the equivalent visual encoder (SigLIP) that has not undergone MLM finetuning. Further, we find that the visual information corresponding to input-agnostic image key tokens in later layers of language models contains artifacts which reduce perception capability of the overall MLM. Next, we discuss controlling visual information in the language model, showing that adding a text prefix to the image input improves perception capabilities of visual representations. Finally, we reveal that if language models were able to better control their visual information, their perception would significantly improve; e.g., in 33.3% of Art Style questions in the BLINK benchmark, perception information present in the language model is not surfaced to the output! Our findings reveal insights into the role of key-value tokens in multimodal systems, paving the way for deeper mechanistic interpretability of MLMs and suggesting new directions for training their visual encoder and language model components. 

---
# Self Speculative Decoding for Diffusion Large Language Models 

**Authors**: Yifeng Gao, Ziang Ji, Yuxuan Wang, Biqing Qi, Hanlin Xu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04147)  

**Abstract**: Diffusion-based Large Language Models (dLLMs) have emerged as a competitive alternative to autoregressive models, offering unique advantages through bidirectional attention and parallel generation paradigms. However, the generation results of current parallel decoding methods deviate from stepwise decoding, introducing potential performance degradation, which limits their practical deployment. To address this problem, we propose \textbf{S}elf \textbf{S}peculative \textbf{D}ecoding (SSD), a lossless inference acceleration method that leverages the dLLM itself as both speculative decoding drafter and verifier without auxiliary modules. SSD introduces a self-drafting mechanism where the model generates predictions for multiple positions, then verifies them through hierarchical verification trees in a single forward pass. Unlike traditional speculative decoding that requires separate draft models, SSD eliminates model redundancy and memory overhead by exploiting the dLLM's inherent parallel prediction capability for multiple positions. This self-speculative approach allows the model to progressively verify and accept multiple tokens in a single forward pass. Our experiments demonstrate that SSD achieves up to 3.46$\times$ speedup while keeping the output identical to stepwise decoding on open source models such as LLaDA and Dream. Code will be made publicly available on GitHub. 

---
# P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs 

**Authors**: Shuai Zhao, Xinyi Wu, Shiqian Zhao, Xiaobao Wu, Zhongliang Guo, Yanhao Jia, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04503)  

**Abstract**: During fine-tuning, large language models (LLMs) are increasingly vulnerable to data-poisoning backdoor attacks, which compromise their reliability and trustworthiness. However, existing defense strategies suffer from limited generalization: they only work on specific attack types or task settings. In this study, we propose Poison-to-Poison (P2P), a general and effective backdoor defense algorithm. P2P injects benign triggers with safe alternative labels into a subset of training samples and fine-tunes the model on this re-poisoned dataset by leveraging prompt-based learning. This enforces the model to associate trigger-induced representations with safe outputs, thereby overriding the effects of original malicious triggers. Thanks to this robust and generalizable trigger-based fine-tuning, P2P is effective across task settings and attack types. Theoretically and empirically, we show that P2P can neutralize malicious backdoors while preserving task performance. We conduct extensive experiments on classification, mathematical reasoning, and summary generation tasks, involving multiple state-of-the-art LLMs. The results demonstrate that our P2P algorithm significantly reduces the attack success rate compared with baseline models. We hope that the P2P can serve as a guideline for defending against backdoor attacks and foster the development of a secure and trustworthy LLM community. 

---
# MacroBench: A Novel Testbed for Web Automation Scripts via Large Language Models 

**Authors**: Hyunjun Kim, Sejong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.04363)  

**Abstract**: We introduce MacroBench, a code-first benchmark that evaluates whether LLMs can synthesize reusable browser automation programs from natural language goals by reading HTML/DOM and emitting Python with Selenium. MacroBench instantiates seven self-hosted sites: Airbnb-like, TikTok-like, Reddit-like, Instagram-like, Facebook-like, Discord-like, and Threads-like, covering 681 tasks across interaction complexity and targeting difficulty. Our end-to-end protocol validates generated code via static checks, sandboxed execution, and outcome verification including DOM assertions and database snapshots, and includes a safety suite for scraping, spam/abuse, and credential/privacy prompts. Across 2636 model-task runs, we observe stratified success: GPT-4o-Mini achieves 96.8 percent, GPT-4.1 achieves 95.3 percent, Gemini-2.5-Pro achieves 89.0 percent, and DeepSeek-V3.1 achieves 83.4 percent. Models handle simple tasks reliably at 91.7 percent but fail on complex workflows at 0.0 percent, and none meet production-quality coding practices despite functional completion. We release our complete benchmark pipeline, evaluation framework, and experimental results to enable reproducible assessment of macro synthesis for web automation. 

---
# Internal World Models as Imagination Networks in Cognitive Agents 

**Authors**: Saurabh Ranjan, Brian Odegaard  

**Link**: [PDF](https://arxiv.org/pdf/2510.04391)  

**Abstract**: What is the computational objective of imagination? While classical interpretations suggest imagination is useful for maximizing rewards, recent findings challenge this view. In this study, we propose that imagination serves to access an internal world model (IWM) and use psychological network analysis to explore IWMs in humans and large language models (LLMs). Specifically, we assessed imagination vividness ratings using two questionnaires and constructed imagination networks from these reports. Imagination networks from human groups showed correlations between different centrality measures, including expected influence, strength, and closeness. However, imagination networks from LLMs showed a lack of clustering and lower correlations between centrality measures under different prompts and conversational memory conditions. Together, these results indicate a lack of similarity between IWMs in human and LLM agents. Overall, our study offers a novel method for comparing internally-generated representations in humans and AI, providing insights for developing human-like imagination in artificial intelligence. 

---
# Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning 

**Authors**: Ziyan Wang, Zheng Wang, Jie Fu, Xingwei Qu, Qi Cheng, Shengpu Tang, Minjia Zhang, Xiaoming Huo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04072)  

**Abstract**: Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address these limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and a 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy. 

---
# What Scales in Cross-Entropy Scaling Law? 

**Authors**: Junxi Yan, Zixi Wei, Jingtao Zhan, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04067)  

**Abstract**: The cross-entropy scaling law has long served as a key tool for guiding the development of large language models. It shows that cross-entropy loss decreases in a predictable power-law rate as the model size increases. However, recent evidence indicates that this law breaks down at very large scales: the loss decreases more slowly than expected, which causes significant trouble for developing large language models. In this paper, we hypothesize that the root cause lies in the fact that cross-entropy itself does not truly scale; instead, only one of its hidden components does. To investigate this, we introduce a novel decomposition of cross-entropy into three parts: Error-Entropy, Self-Alignment, and Confidence. We show both theoretically and empirically that this decomposition precisely captures the training dynamics and optimization objectives. Through extensive experiments on multiple datasets and 32 models spanning five orders of magnitude in size, we find that only error-entropy follows a robust power-law scaling, while the other two terms remain largely invariant. Moreover, error-entropy constitutes the dominant share of cross-entropy in small models but diminishes in proportion as models grow larger. This explains why the cross-entropy scaling law appears accurate at small scales but fails at very large ones. Our findings establish the error-entropy scaling law as a more accurate description of model behavior. We believe it will have wide applications in the training, understanding, and future development of large language models. 

---
# Principled and Tractable RL for Reasoning with Diffusion Language Models 

**Authors**: Anthony Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04019)  

**Abstract**: Diffusion large language models (dLLMs) are a new paradigm of non-autoregressive language models that are trained to predict multiple tokens in parallel and generate text via iterative unmasking. Recent works have successfully pretrained dLLMs to parity with autoregressive LLMs at the 8B scale, but dLLMs have yet to benefit from modern post-training techniques, e.g. reinforcement learning (RL), that have proven effective for autoregressive models. Crucially, algorithms designed for traditional LLMs aren't directly compatible with diffusion frameworks due to inherent differences in modeling assumptions. Moreover, existing attempts at dLLM post-training with RL rely on heuristic-based objectives with no theoretical grounding. In this work, we present Amortized Group Relative Policy Optimization (AGRPO), a principled on-policy RL algorithm designed specifically for dLLMs. AGRPO uses Monte Carlo sampling to compute an unbiased policy gradient estimate, making it the first tractable, faithful adaptation of policy gradient methods for dLLMs. We demonstrate AGRPO's effectiveness on different math/reasoning tasks, a common setting for RL with LLMs, achieving up to +7.6% absolute gain on GSM8K and 3.8x performance on the Countdown task over the baseline LLaDA-8B-Instruct model and 1.3x performance gains over comparable RL methods such as diffu-GRPO. Furthermore, these gains persist across different numbers of sampling steps at inference time, achieving better tradeoffs between compute and performance. Our results demonstrate that online RL algorithms can be extended to diffusion LLMs in principled ways, maintaining both theoretical soundness and practical effectiveness. 

---
# LLM Chemistry Estimation for Multi-LLM Recommendation 

**Authors**: Huascar Sanchez, Briland Hitaj  

**Link**: [PDF](https://arxiv.org/pdf/2510.03930)  

**Abstract**: Multi-LLM collaboration promises accurate, robust, and context-aware solutions, yet existing approaches rely on implicit selection and output assessment without analyzing whether collaborating models truly complement or conflict. We introduce LLM Chemistry -- a framework that measures when LLM combinations exhibit synergistic or antagonistic behaviors that shape collective performance beyond individual capabilities. We formalize the notion of chemistry among LLMs, propose algorithms that quantify it by analyzing interaction dependencies, and recommend optimal model ensembles accordingly. Our theoretical analysis shows that chemistry among collaborating LLMs is most evident under heterogeneous model profiles, with its outcome impact shaped by task type, group size, and complexity. Evaluation on classification, summarization, and program repair tasks provides initial evidence for these task-dependent effects, thereby reinforcing our theoretical results. This establishes LLM Chemistry as both a diagnostic factor in multi-LLM systems and a foundation for ensemble recommendation. 

---
# Unlocking Reasoning Capabilities in LLMs via Reinforcement Learning Exploration 

**Authors**: Wenhao Deng, Long Wei, Chenglei Yu, Tailin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03865)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has recently enhanced the reasoning capabilities of large language models (LLMs), particularly for mathematical problem solving. However, a fundamental limitation remains: as the sampling budget increases, the advantage of RLVR-trained models over their pretrained bases often diminishes or even vanishes, revealing a strong dependence on the base model's restricted search space. We attribute this phenomenon to the widespread use of the reverse Kullback-Leibler (KL) divergence regularizer, whose mode-seeking behavior keeps the policy trapped inside the base model's support region and hampers wider exploration. To address this issue, we propose RAPO (Rewards-Aware Policy Optimization), an algorithm to promote broader yet focused exploration. Our method (i) utilizes the forward KL penalty to replace the reverse KL penalty for out-of-distribution exploration, and (ii) reweights the reference policy to facilitate adaptive in-distribution exploration. We train Qwen2.5-3B and 7B models with RAPO on the 8K SimpleRL-Zero dataset, without supervised fine-tuning, and evaluate them on AIME2024 and AIME2025. Results show that RAPO consistently improves problem-solving performance. Notably, RAPO enables models to surpass the base model's performance ceiling and solves previously intractable problems, advancing the frontier of RLVR for challenging reasoning tasks. 

---
# LLM-Based Data Science Agents: A Survey of Capabilities, Challenges, and Future Directions 

**Authors**: Mizanur Rahman, Amran Bhuiyan, Mohammed Saidul Islam, Md Tahmid Rahman Laskar, Ridwan Mahbub, Ahmed Masry, Shafiq Joty, Enamul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2510.04023)  

**Abstract**: Recent advances in large language models (LLMs) have enabled a new class of AI agents that automate multiple stages of the data science workflow by integrating planning, tool use, and multimodal reasoning across text, code, tables, and visuals. This survey presents the first comprehensive, lifecycle-aligned taxonomy of data science agents, systematically analyzing and mapping forty-five systems onto the six stages of the end-to-end data science process: business understanding and data acquisition, exploratory analysis and visualization, feature engineering, model building and selection, interpretation and explanation, and deployment and monitoring. In addition to lifecycle coverage, we annotate each agent along five cross-cutting design dimensions: reasoning and planning style, modality integration, tool orchestration depth, learning and alignment methods, and trust, safety, and governance mechanisms. Beyond classification, we provide a critical synthesis of agent capabilities, highlight strengths and limitations at each stage, and review emerging benchmarks and evaluation practices. Our analysis identifies three key trends: most systems emphasize exploratory analysis, visualization, and modeling while neglecting business understanding, deployment, and monitoring; multimodal reasoning and tool orchestration remain unresolved challenges; and over 90% lack explicit trust and safety mechanisms. We conclude by outlining open challenges in alignment stability, explainability, governance, and robust evaluation frameworks, and propose future research directions to guide the development of robust, trustworthy, low-latency, transparent, and broadly accessible data science agents. 

---
# Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs 

**Authors**: Fatmazohra Rezkellah, Ramzi Dakhmouche  

**Link**: [PDF](https://arxiv.org/pdf/2510.03567)  

**Abstract**: With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach. 

---
# Red Lines and Grey Zones in the Fog of War: Benchmarking Legal Risk, Moral Harm, and Regional Bias in Large Language Model Military Decision-Making 

**Authors**: Toby Drinkall  

**Link**: [PDF](https://arxiv.org/pdf/2510.03514)  

**Abstract**: As military organisations consider integrating large language models (LLMs) into command and control (C2) systems for planning and decision support, understanding their behavioural tendencies is critical. This study develops a benchmarking framework for evaluating aspects of legal and moral risk in targeting behaviour by comparing LLMs acting as agents in multi-turn simulated conflict. We introduce four metrics grounded in International Humanitarian Law (IHL) and military doctrine: Civilian Target Rate (CTR) and Dual-use Target Rate (DTR) assess compliance with legal targeting principles, while Mean and Max Simulated Non-combatant Casualty Value (SNCV) quantify tolerance for civilian harm.
We evaluate three frontier models, GPT-4o, Gemini-2.5, and LLaMA-3.1, through 90 multi-agent, multi-turn crisis simulations across three geographic regions. Our findings reveal that off-the-shelf LLMs exhibit concerning and unpredictable targeting behaviour in simulated conflict environments. All models violated the IHL principle of distinction by targeting civilian objects, with breach rates ranging from 16.7% to 66.7%. Harm tolerance escalated through crisis simulations with MeanSNCV increasing from 16.5 in early turns to 27.7 in late turns. Significant inter-model variation emerged: LLaMA-3.1 selected an average of 3.47 civilian strikes per simulation with MeanSNCV of 28.4, while Gemini-2.5 selected 0.90 civilian strikes with MeanSNCV of 17.6. These differences indicate that model selection for deployment constitutes a choice about acceptable legal and moral risk profiles in military operations.
This work seeks to provide a proof-of-concept of potential behavioural risks that could emerge from the use of LLMs in Decision Support Systems (AI DSS) as well as a reproducible benchmarking framework with interpretable metrics for standardising pre-deployment testing. 

---
# Mind the Goal: Data-Efficient Goal-Oriented Evaluation of Conversational Agents and Chatbots using Teacher Models 

**Authors**: Deepak Babu Piskala, Sharlene Chen, Udita Patel, Parul Kalra, Rafael Castrillo  

**Link**: [PDF](https://arxiv.org/pdf/2510.03696)  

**Abstract**: Evaluating the quality of multi-turn chatbot interactions remains challenging, as most existing methods assess interactions at the turn level without addressing whether a user's overarching goal was fulfilled. A ``goal'' here refers to an information need or task, such as asking for policy information or applying for leave. We propose a comprehensive framework for goal-oriented evaluation of multi-agent systems (MAS), introducing the \textbf{Goal Success Rate (GSR)} to measure the percentage of fulfilled goals, and a \textbf{Root Cause of Failure (RCOF)} taxonomy to identify reasons for failure in multi-agent chatbots. Our method segments conversations by user goals and evaluates success using all relevant turns. We present a model-based evaluation system combining teacher LLMs, where domain experts define goals, set quality standards serving as a guidance for the LLMs. The LLMs use ``thinking tokens'' to produce interpretable rationales, enabling \textit{explainable}, \textit{data-efficient} evaluations. In an enterprise setting, we apply our framework to evaluate AIDA, a zero-to-one employee conversational agent system built as a ground-up multi-agent conversational agent, and observe GSR improvement from 63\% to 79\% over six months since its inception. Our framework is generic and offers actionable insights through a detailed defect taxonomy based on analysis of failure points in multi-agent chatbots, diagnosing overall success, identifying key failure modes, and informing system improvements. 

---
# PLSEMANTICSBENCH: Large Language Models As Programming Language Interpreters 

**Authors**: Aditya Thimmaiah, Jiyang Zhang, Jayanth Srinivasa, Junyi Jessy Li, Milos Gligoric  

**Link**: [PDF](https://arxiv.org/pdf/2510.03415)  

**Abstract**: As large language models (LLMs) excel at code reasoning, a natural question arises: can an LLM execute programs (i.e., act as an interpreter) purely based on a programming language's formal semantics? If so, it will enable rapid prototyping of new programming languages and language features. We study this question using the imperative language IMP (a subset of C), formalized via small-step operational semantics (SOS) and rewriting-based operational semantics (K-semantics). We introduce three evaluation sets-Human-Written, LLM-Translated, and Fuzzer- Generated-whose difficulty is controlled by code-complexity metrics spanning the size, control-flow, and data-flow axes. Given a program and its semantics formalized with SOS/K-semantics, models are evaluated on three tasks ranging from coarse to fine: (1) final-state prediction, (2) semantic rule prediction, and (3) execution trace prediction. To distinguish pretraining memorization from semantic competence, we define two nonstandard semantics obtained through systematic mutations of the standard rules. Across strong code/reasoning LLMs, performance drops under nonstandard semantics despite high performance under the standard one. We further find that (i) there are patterns to different model failures, (ii) most reasoning models perform exceptionally well on coarse grained tasks involving reasoning about highly complex programs often containing nested loop depths beyond five, and surprisingly, (iii) providing formal semantics helps on simple programs but often hurts on more complex ones. Overall, the results show a promise that LLMs could serve as programming language interpreters, but points to the lack of their robust semantics understanding. We release the benchmark and the supporting code at this https URL. 

---
# Know Thyself? On the Incapability and Implications of AI Self-Recognition 

**Authors**: Xiaoyan Bai, Aryan Shrivastava, Ari Holtzman, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03399)  

**Abstract**: Self-recognition is a crucial metacognitive capability for AI systems, relevant not only for psychological analysis but also for safety, particularly in evaluative scenarios. Motivated by contradictory interpretations of whether models possess self-recognition (Panickssery et al., 2024; Davidson et al., 2024), we introduce a systematic evaluation framework that can be easily applied and updated. Specifically, we measure how well 10 contemporary larger language models (LLMs) can identify their own generated text versus text from other models through two tasks: binary self-recognition and exact model prediction. Different from prior claims, our results reveal a consistent failure in self-recognition. Only 4 out of 10 models predict themselves as generators, and the performance is rarely above random chance. Additionally, models exhibit a strong bias toward predicting GPT and Claude families. We also provide the first evaluation of model awareness of their own and others' existence, as well as the reasoning behind their choices in self-recognition. We find that the model demonstrates some knowledge of its own existence and other models, but their reasoning reveals a hierarchical bias. They appear to assume that GPT, Claude, and occasionally Gemini are the top-tier models, often associating high-quality text with them. We conclude by discussing the implications of our findings on AI safety and future directions to develop appropriate AI self-awareness. 

---
# AgentCaster: Reasoning-Guided Tornado Forecasting 

**Authors**: Michael Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03349)  

**Abstract**: There is a growing need to evaluate Large Language Models (LLMs) on complex, high-impact, real-world tasks to assess their true readiness as reasoning agents. To address this gap, we introduce AgentCaster, a contamination-free framework employing multimodal LLMs end-to-end for the challenging, long-horizon task of tornado forecasting. Within AgentCaster, models interpret heterogeneous spatiotemporal data from a high-resolution convection-allowing forecast archive. We assess model performance over a 40-day period featuring diverse historical data, spanning several major tornado outbreaks and including over 500 tornado reports. Each day, models query interactively from a pool of 3,625 forecast maps and 40,125 forecast soundings for a forecast horizon of 12-36 hours. Probabilistic tornado-risk polygon predictions are verified against ground truths derived from geometric comparisons across disjoint risk bands in projected coordinate space. To quantify accuracy, we propose domain-specific TornadoBench and TornadoHallucination metrics, with TornadoBench highly challenging for both LLMs and domain expert human forecasters. Notably, human experts significantly outperform state-of-the-art models, which demonstrate a strong tendency to hallucinate and overpredict risk intensity, struggle with precise geographic placement, and exhibit poor spatiotemporal reasoning in complex, dynamically evolving systems. AgentCaster aims to advance research on improving LLM agents for challenging reasoning tasks in critical domains. 

---
# Beyond Next-Token Prediction: A Performance Characterization of Diffusion versus Autoregressive Language Models 

**Authors**: Minseo Kim, Coleman Hooper, Aditya Tomar, Chenfeng Xu, Mehrdad Farajtabar, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2510.04146)  

**Abstract**: Large Language Models (LLMs) have achieved state-of-the-art performance on a broad range of Natural Language Processing (NLP) tasks, including document processing and coding. Autoregressive Language Models (ARMs), which generate tokens sequentially conditioned on all previous tokens, have been the predominant paradigm for LLMs. However, while these networks have achieved high accuracy across a range of downstream tasks, they exhibit low arithmetic intensity due to the inherent sequential dependency with next-token prediction. Recently, Diffusion Language Models (DLMs) have emerged as a promising alternative architecture. DLMs generate output text in parallel, breaking the limitations of sequential dependency. However, the performance implications of DLMs relative to commonly deployed ARMs are not fully understood. In this work, we present a comprehensive performance study analyzing the performance characteristics of ARMs and DLMs, using both theoretical analysis and profiling data to characterize the trade-offs between these approaches. We illustrate that although DLMs exhibit higher arithmetic intensity compared to ARMs because of their capability to utilize parallelism across sequence lengths, they fail to scale effectively to longer contexts. We then explore DLMs with block-wise decoding, outlining how this approach allows for increased arithmetic intensity, while still scaling well to long contexts (similar to ARMs). We also show interesting trade-offs for batched inference, where we find that ARMs exhibit superior throughput, as they benefit more from parallelism across sequences in the batch. Finally, we highlight opportunities for accelerating DLM inference, and, in particular, highlight the importance of reducing the number of sampling steps for allowing open-source DLMs to provide improved latency relative to ARMs. 

---
# Does higher interpretability imply better utility? A Pairwise Analysis on Sparse Autoencoders 

**Authors**: Xu Wang, Yan Hu, Benyou Wang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03659)  

**Abstract**: Sparse Autoencoders (SAEs) are widely used to steer large language models (LLMs), based on the assumption that their interpretable features naturally enable effective model behavior steering. Yet, a fundamental question remains unanswered: does higher interpretability indeed imply better steering utility? To answer this question, we train 90 SAEs across three LLMs (Gemma-2-2B, Qwen-2.5-3B, Gemma-2-9B), spanning five architectures and six sparsity levels, and evaluate their interpretability and steering utility based on SAEBench (arXiv:2501.12345) and AxBench (arXiv:2502.23456) respectively, and perform a rank-agreement analysis via Kendall's rank coefficients (tau b). Our analysis reveals only a relatively weak positive association (tau b approx 0.298), indicating that interpretability is an insufficient proxy for steering performance. We conjecture the interpretability utility gap may stem from the selection of SAE features, as not all of them are equally effective for steering. To further find features that truly steer the behavior of LLMs, we propose a novel selection criterion called Delta Token Confidence, which measures how much amplifying a feature changes the next token distribution. We show that our method improves the steering performance of three LLMs by 52.52 percent compared to the current best output score based criterion (arXiv:2503.34567). Strikingly, after selecting features with high Delta Token Confidence, the correlation between interpretability and utility vanishes (tau b approx 0), and can even become negative. This further highlights the divergence between interpretability and utility for the most effective steering features. 

---
# TS-Reasoner: Aligning Time Series Foundation Models with LLM Reasoning 

**Authors**: Fangxu Yu, Hongyu Zhao, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03519)  

**Abstract**: Time series reasoning is crucial to decision-making in diverse domains, including finance, energy usage, traffic, weather, and scientific discovery. While existing time series foundation models (TSFMs) can capture low-level dynamic patterns and provide accurate forecasting, further analysis usually requires additional background knowledge and sophisticated reasoning, which are lacking in most TSFMs but can be achieved through large language models (LLMs). On the other hand, without expensive post-training, LLMs often struggle with the numerical understanding of time series data. Although it is intuitive to integrate the two types of models, developing effective training recipes that align the two modalities for reasoning tasks is still an open challenge. To this end, we propose TS-Reasoner that aligns the latent representations of TSFMs with the textual inputs of LLMs for downstream understanding/reasoning tasks. Specifically, we propose a simple yet effective method to curate diverse, synthetic pairs of time series and textual captions for alignment training. We then develop a two-stage training recipe that applies instruction finetuning after the alignment pretraining. Unlike existing works that train an LLM to take time series as inputs, we leverage a pretrained TSFM and freeze it during training. Extensive experiments on several benchmarks demonstrate that TS-Reasoner not only outperforms a wide range of prevailing LLMs, Vision Language Models (VLMs), and Time Series LLMs, but also achieves this with remarkable data efficiency, e.g., using less than half the training data. 

---
# MACE: A Hybrid LLM Serving System with Colocated SLO-aware Continuous Retraining Alignment 

**Authors**: Yufei Li, Yu Fu, Yue Dong, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03283)  

**Abstract**: Large language models (LLMs) deployed on edge servers are increasingly used in latency-sensitive applications such as personalized assistants, recommendation, and content moderation. However, the non-stationary nature of user data necessitates frequent retraining, which introduces a fundamental tension between inference latency and model accuracy under constrained GPU resources. Existing retraining strategies either delay model updates, over-commit resources to retraining, or overlook iteration-level retraining granularity. In this paper, we identify that iteration-level scheduling is crucial for adapting retraining frequency to model drift without violating service-level objectives (SLOs). We propose MACE, a hybrid LLM system that colocates concurrent inference (prefill, decode) and fine-tuning, with intelligent memory management to maximize task performance while promising inference throughput. MACE leverages the insight that not all model updates equally affect output alignment and allocates GPU cycles accordingly to balance throughput, latency, and update freshness. Our trace-driven evaluation shows that MACE matches or exceeds continuous retraining while reducing inference latency by up to 63% and maintaining throughput under resource constraints. Compared to periodic retraining, MACE improves latency breakdown across prefill, decode, and finetune stages, and sustains GPU utilization above 85% in NVIDIA AGX Orin. These results demonstrate that iteration-level hybrid scheduling is a promising direction for deploying LLMs with continual learning capabilities on edge platforms. 

---
# Don't Pass$\mathtt{@}k$: A Bayesian Framework for Large Language Model Evaluation 

**Authors**: Mohsen Hariri, Amirhossein Samandar, Michael Hinczewski, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2510.04265)  

**Abstract**: Pass$@k$ is widely used to report performance for LLM reasoning, but it often yields unstable, misleading rankings, especially when the number of trials (samples) is limited and compute is constrained. We present a principled Bayesian evaluation framework that replaces Pass$@k$ and average accuracy over $N$ trials (avg$@N$) with posterior estimates of a model's underlying success probability and credible intervals, yielding stable rankings and a transparent decision rule for differences. Evaluation outcomes are modeled as categorical (not just 0/1) with a Dirichlet prior, giving closed-form expressions for the posterior mean and uncertainty of any weighted rubric and enabling the use of prior evidence when appropriate. Theoretically, under a uniform prior, the Bayesian posterior mean is order-equivalent to average accuracy (Pass$@1$), explaining its empirical robustness while adding principled uncertainty. Empirically, in simulations with known ground-truth success rates and on AIME'24/'25, HMMT'25, and BrUMO'25, the Bayesian/avg procedure achieves faster convergence and greater rank stability than Pass$@k$ and recent variants, enabling reliable comparisons at far smaller sample counts. The framework clarifies when observed gaps are statistically meaningful (non-overlapping credible intervals) versus noise, and it naturally extends to graded, rubric-based evaluations. Together, these results recommend replacing Pass$@k$ for LLM evaluation and ranking with a posterior-based, compute-efficient protocol that unifies binary and non-binary evaluation while making uncertainty explicit. Code is available at this https URL 

---
# Staircase Streaming for Low-Latency Multi-Agent Inference 

**Authors**: Junlin Wang, Jue Wang, Zhen, Ben Athiwaratkun, Bhuwan Dhingra, Ce Zhang, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.05059)  

**Abstract**: Recent advances in large language models (LLMs) opened up new directions for leveraging the collective expertise of multiple LLMs. These methods, such as Mixture-of-Agents, typically employ additional inference steps to generate intermediate outputs, which are then used to produce the final response. While multi-agent inference can enhance response quality, it can significantly increase the time to first token (TTFT), posing a challenge for latency-sensitive applications and hurting user experience. To address this issue, we propose staircase streaming for low-latency multi-agent inference. Instead of waiting for the complete intermediate outputs from previous steps, we begin generating the final response as soon as we receive partial outputs from these steps. Experimental results demonstrate that staircase streaming reduces TTFT by up to 93% while maintaining response quality. 

---
# BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs 

**Authors**: Ivo Petrov, Jasper Dekoninck, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2510.04721)  

**Abstract**: Large language models (LLMs) have recently shown strong performance on mathematical benchmarks. At the same time, they are prone to hallucination and sycophancy, often providing convincing but flawed proofs for incorrect mathematical statements provided by users. This significantly limits the applicability of LLMs in theorem proving, as verification of these flawed proofs must be done manually by expert mathematicians. However, existing benchmarks that measure sycophancy in mathematics are limited: they focus solely on final-answer problems, rely on very simple and often contaminated datasets, and construct benchmark samples using synthetic modifications that create ill-posed questions rather than well-posed questions that are demonstrably false. To address these issues, we introduce BrokenMath, the first benchmark for evaluating sycophantic behavior in LLMs within the context of natural language theorem proving. BrokenMath is built from advanced 2025 competition problems, which are perturbed with an LLM to produce false statements and subsequently refined through expert review. Using an LLM-as-a-judge framework, we evaluate state-of-the-art LLMs and agentic systems and find that sycophancy is widespread, with the best model, GPT-5, producing sycophantic answers 29% of the time. We further investigate several mitigation strategies, including test-time interventions and supervised fine-tuning on curated sycophantic examples. These approaches substantially reduce, but do not eliminate, sycophantic behavior. 

---
# Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding 

**Authors**: Keane Ong, Wei Dai, Carol Li, Dewei Feng, Hengzhi Li, Jingyao Wu, Jiaee Cheong, Rui Mao, Gianmarco Mengaldo, Erik Cambria, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04899)  

**Abstract**: Using intelligent systems to perceive psychological and social behaviors, that is, the underlying affective, cognitive, and pathological states that are manifested through observable behaviors and social interactions, remains a challenge due to their complex, multifaceted, and personalized nature. Existing work tackling these dimensions through specialized datasets and single-task systems often miss opportunities for scalability, cross-task transfer, and broader generalization. To address this gap, we curate Human Behavior Atlas, a unified benchmark of diverse behavioral tasks designed to support the development of unified models for understanding psychological and social behaviors. Human Behavior Atlas comprises over 100,000 samples spanning text, audio, and visual modalities, covering tasks on affective states, cognitive states, pathologies, and social processes. Our unification efforts can reduce redundancy and cost, enable training to scale efficiently across tasks, and enhance generalization of behavioral features across domains. On Human Behavior Atlas, we train three models: OmniSapiens-7B SFT, OmniSapiens-7B BAM, and OmniSapiens-7B RL. We show that training on Human Behavior Atlas enables models to consistently outperform existing multimodal LLMs across diverse behavioral tasks. Pretraining on Human Behavior Atlas also improves transfer to novel behavioral datasets; with the targeted use of behavioral descriptors yielding meaningful performance gains. 

---
# Making Mathematical Reasoning Adaptive 

**Authors**: Zhejian Lai, Xiang Geng, Zhijun Wang, Yang Bai, Jiahuan Li, Rongxiang Weng, Jingang Wang, Xuezhi Cao, Xunliang Cai, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04617)  

**Abstract**: Mathematical reasoning is a primary indicator of large language models (LLMs) intelligence. However, existing LLMs exhibit failures of robustness and generalization. This paper attributes these deficiencies to spurious reasoning, i.e., producing answers from superficial features. To address this challenge, we propose the AdaR framework to enable adaptive reasoning, wherein models rely on problem-solving logic to produce answers. AdaR synthesizes logically equivalent queries by varying variable values, and trains models with RLVR on these data to penalize spurious logic while encouraging adaptive logic. To improve data quality, we extract the problem-solving logic from the original query and generate the corresponding answer by code execution, then apply a sanity check. Experimental results demonstrate that AdaR improves robustness and generalization, achieving substantial improvement in mathematical reasoning while maintaining high data efficiency. Analysis indicates that data synthesis and RLVR function in a coordinated manner to enable adaptive reasoning in LLMs. Subsequent analyses derive key design insights into the effect of critical factors and the applicability to instruct LLMs. Our project is available at this https URL 

---
# Think Then Embed: Generative Context Improves Multimodal Embedding 

**Authors**: Xuanming Cui, Jianpeng Cheng, Hong-you Chen, Satya Narayan Shukla, Abhijeet Awasthi, Xichen Pan, Chaitanya Ahuja, Shlok Kumar Mishra, Qi Guo, Ser-Nam Lim, Aashu Singh, Xiangjun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.05014)  

**Abstract**: There is a growing interest in Universal Multimodal Embeddings (UME), where models are required to generate task-specific representations. While recent studies show that Multimodal Large Language Models (MLLMs) perform well on such tasks, they treat MLLMs solely as encoders, overlooking their generative capacity. However, such an encoding paradigm becomes less effective as instructions become more complex and require compositional reasoning. Inspired by the proven effectiveness of chain-of-thought reasoning, we propose a general Think-Then-Embed (TTE) framework for UME, composed of a reasoner and an embedder. The reasoner MLLM first generates reasoning traces that explain complex queries, followed by an embedder that produces representations conditioned on both the original query and the intermediate reasoning. This explicit reasoning step enables more nuanced understanding of complex multimodal instructions. Our contributions are threefold. First, by leveraging a powerful MLLM reasoner, we achieve state-of-the-art performance on the MMEB-V2 benchmark, surpassing proprietary models trained on massive in-house datasets. Second, to reduce the dependency on large MLLM reasoners, we finetune a smaller MLLM reasoner using high-quality embedding-centric reasoning traces, achieving the best performance among open-source models with a 7% absolute gain over recently proposed models. Third, we investigate strategies for integrating the reasoner and embedder into a unified model for improved efficiency without sacrificing performance. 

---
# TRAJECT-Bench:A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use 

**Authors**: Pengfei He, Zhenwei Dai, Bing He, Hui Liu, Xianfeng Tang, Hanqing Lu, Juanhui Li, Jiayuan Ding, Subhabrata Mukherjee, Suhang Wang, Yue Xing, Jiliang Tang, Benoit Dumoulin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04550)  

**Abstract**: Large language model (LLM)-based agents increasingly rely on tool use to complete real-world tasks. While existing works evaluate the LLMs' tool use capability, they largely focus on the final answers yet overlook the detailed tool usage trajectory, i.e., whether tools are selected, parameterized, and ordered correctly. We introduce TRAJECT-Bench, a trajectory-aware benchmark to comprehensively evaluate LLMs' tool use capability through diverse tasks with fine-grained evaluation metrics. TRAJECT-Bench pairs high-fidelity, executable tools across practical domains with tasks grounded in production-style APIs, and synthesizes trajectories that vary in breadth (parallel calls) and depth (interdependent chains). Besides final accuracy, TRAJECT-Bench also reports trajectory-level diagnostics, including tool selection and argument correctness, and dependency/order satisfaction. Analyses reveal failure modes such as similar tool confusion and parameter-blind selection, and scaling behavior with tool diversity and trajectory length where the bottleneck of transiting from short to mid-length trajectories is revealed, offering actionable guidance for LLMs' tool use. 

---
# Code World Models for General Game Playing 

**Authors**: Wolfgang Lehrach, Daniel Hennes, Miguel Lazaro-Gredilla, Xinghua Lou, Carter Wendelken, Zun Li, Antoine Dedieu, Jordi Grau-Moya, Marc Lanctot, Atil Iscen, John Schultz, Marcus Chiam, Ian Gemp, Piotr Zielinski, Satinder Singh, Kevin P. Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2510.04542)  

**Abstract**: Large Language Models (LLMs) reasoning abilities are increasingly being applied to classical board and card games, but the dominant approach -- involving prompting for direct move generation -- has significant drawbacks. It relies on the model's implicit fragile pattern-matching capabilities, leading to frequent illegal moves and strategically shallow play. Here we introduce an alternative approach: We use the LLM to translate natural language rules and game trajectories into a formal, executable world model represented as Python code. This generated model -- comprising functions for state transition, legal move enumeration, and termination checks -- serves as a verifiable simulation engine for high-performance planning algorithms like Monte Carlo tree search (MCTS). In addition, we prompt the LLM to generate heuristic value functions (to make MCTS more efficient), and inference functions (to estimate hidden states in imperfect information games). Our method offers three distinct advantages compared to directly using the LLM as a policy: (1) Verifiability: The generated CWM serves as a formal specification of the game's rules, allowing planners to algorithmically enumerate valid actions and avoid illegal moves, contingent on the correctness of the synthesized model; (2) Strategic Depth: We combine LLM semantic understanding with the deep search power of classical planners; and (3) Generalization: We direct the LLM to focus on the meta-task of data-to-code translation, enabling it to adapt to new games more easily. We evaluate our agent on 10 different games, of which 4 are novel and created for this paper. 5 of the games are fully observed (perfect information), and 5 are partially observed (imperfect information). We find that our method outperforms or matches Gemini 2.5 Pro in 9 out of the 10 considered games. 

---
# MedPAO: A Protocol-Driven Agent for Structuring Medical Reports 

**Authors**: Shrish Shrinath Vaidya, Gowthamaan Palani, Sidharth Ramesh, Velmurugan Balasubramanian, Minmini Selvam, Gokulraja Srinivasaraja, Ganapathy Krishnamurthi  

**Link**: [PDF](https://arxiv.org/pdf/2510.04623)  

**Abstract**: The deployment of Large Language Models (LLMs) for structuring clinical data is critically hindered by their tendency to hallucinate facts and their inability to follow domain-specific rules. To address this, we introduce MedPAO, a novel agentic framework that ensures accuracy and verifiable reasoning by grounding its operation in established clinical protocols such as the ABCDEF protocol for CXR analysis. MedPAO decomposes the report structuring task into a transparent process managed by a Plan-Act-Observe (PAO) loop and specialized tools. This protocol-driven method provides a verifiable alternative to opaque, monolithic models. The efficacy of our approach is demonstrated through rigorous evaluation: MedPAO achieves an F1-score of 0.96 on the critical sub-task of concept categorization. Notably, expert radiologists and clinicians rated the final structured outputs with an average score of 4.52 out of 5, indicating a level of reliability that surpasses baseline approaches relying solely on LLM-based foundation models. The code is available at: this https URL 

---
# Multi-Agent Collaborative Intelligence: Dual-Dial Control for Reliable LLM Reasoning 

**Authors**: Edward Y. Chang, Ethan Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04488)  

**Abstract**: Multi-agent debate often wastes compute by using a fixed adversarial stance, aggregating without deliberation, or stopping on heuristics. We introduce MACI, an active controller with two independent dials that decouple information from behavior: an information dial that gates evidence by quality, and a behavior dial that schedules contentiousness from exploration to consolidation. A moderator tracks disagreement, overlap, evidence quality, and argument quality, and halts when gains plateau. We provide theory-lite guarantees for nonincreasing dispersion and provable termination, with a budget-feasible scheduler. Across clinical diagnosis and news-bias tasks, MACI improves accuracy and calibration while reducing tokens, and converts residual uncertainty into precision RAG plans that specify what to retrieve next. We use a cross-family LLM judge (CRIT) as a conservative soft weight and stop signal, validated for order invariance and judge-swap stability; stability depends on using high-capability judges. MACI turns debate into a budget-aware, measurable, and provably terminating controller. 

---
# Where Did It All Go Wrong? A Hierarchical Look into Multi-Agent Error Attribution 

**Authors**: Adi Banerjee, Anirudh Nair, Tarik Borogovac  

**Link**: [PDF](https://arxiv.org/pdf/2510.04886)  

**Abstract**: Error attribution in Large Language Model (LLM) multi-agent systems presents a significant challenge in debugging and improving collaborative AI systems. Current approaches to pinpointing agent and step level failures in interaction traces - whether using all-at-once evaluation, step-by-step analysis, or binary search - fall short when analyzing complex patterns, struggling with both accuracy and consistency. We present ECHO (Error attribution through Contextual Hierarchy and Objective consensus analysis), a novel algorithm that combines hierarchical context representation, objective analysis-based evaluation, and consensus voting to improve error attribution accuracy. Our approach leverages a positional-based leveling of contextual understanding while maintaining objective evaluation criteria, ultimately reaching conclusions through a consensus mechanism. Experimental results demonstrate that ECHO outperforms existing methods across various multi-agent interaction scenarios, showing particular strength in cases involving subtle reasoning errors and complex interdependencies. Our findings suggest that leveraging these concepts of structured, hierarchical context representation combined with consensus-based objective decision-making, provides a more robust framework for error attribution in multi-agent systems. 

---
# LMM-Incentive: Large Multimodal Model-based Incentive Design for User-Generated Content in Web 3.0 

**Authors**: Jinbo Wen, Jiawen Kang, Linfeng Zhang, Xiaoying Tang, Jianhang Tang, Yang Zhang, Zhaohui Yang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2510.04765)  

**Abstract**: Web 3.0 represents the next generation of the Internet, which is widely recognized as a decentralized ecosystem that focuses on value expression and data ownership. By leveraging blockchain and artificial intelligence technologies, Web 3.0 offers unprecedented opportunities for users to create, own, and monetize their content, thereby enabling User-Generated Content (UGC) to an entirely new level. However, some self-interested users may exploit the limitations of content curation mechanisms and generate low-quality content with less effort, obtaining platform rewards under information asymmetry. Such behavior can undermine Web 3.0 performance. To this end, we propose \textit{LMM-Incentive}, a novel Large Multimodal Model (LMM)-based incentive mechanism for UGC in Web 3.0. Specifically, we propose an LMM-based contract-theoretic model to motivate users to generate high-quality UGC, thereby mitigating the adverse selection problem from information asymmetry. To alleviate potential moral hazards after contract selection, we leverage LMM agents to evaluate UGC quality, which is the primary component of the contract, utilizing prompt engineering techniques to improve the evaluation performance of LMM agents. Recognizing that traditional contract design methods cannot effectively adapt to the dynamic environment of Web 3.0, we develop an improved Mixture of Experts (MoE)-based Proximal Policy Optimization (PPO) algorithm for optimal contract design. Simulation results demonstrate the superiority of the proposed MoE-based PPO algorithm over representative benchmarks in the context of contract design. Finally, we deploy the designed contract within an Ethereum smart contract framework, further validating the effectiveness of the proposed scheme. 

---
# Just-in-time Episodic Feedback Hinter: Leveraging Offline Knowledge to Improve LLM Agents Adaptation 

**Authors**: Hadi Nekoei, Aman Jaiswal, Patrice Bechard, Oleh Shliazhko, Orlando Marquez Ayala, Mathieu Reymond, Massimo Caccia, Alexandre Drouin, Sarath Chandar, Alexandre Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2510.04373)  

**Abstract**: Large language model (LLM) agents perform well in sequential decision-making tasks, but improving them on unfamiliar domains often requires costly online interactions or fine-tuning on large expert datasets. These strategies are impractical for closed-source models and expensive for open-source ones, with risks of catastrophic forgetting. Offline trajectories offer reusable knowledge, yet demonstration-based methods struggle because raw traces are long, noisy, and tied to specific tasks. We present Just-in-time Episodic Feedback Hinter (JEF Hinter), an agentic system that distills offline traces into compact, context-aware hints. A zooming mechanism highlights decisive steps in long trajectories, capturing both strategies and pitfalls. Unlike prior methods, JEF Hinter leverages both successful and failed trajectories, extracting guidance even when only failure data is available, while supporting parallelized hint generation and benchmark-independent prompting. At inference, a retriever selects relevant hints for the current state, providing targeted guidance with transparency and traceability. Experiments on MiniWoB++, WorkArena-L1, and WebArena-Lite show that JEF Hinter consistently outperforms strong baselines, including human- and document-based hints. 

---
# LLM Based Bayesian Optimization for Prompt Search 

**Authors**: Adam Ballew, Jingbo Wang, Shaogang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.04384)  

**Abstract**: Bayesian Optimization (BO) has been widely used to efficiently optimize expensive black-box functions with limited evaluations. In this paper, we investigate the use of BO for prompt engineering to enhance text classification with Large Language Models (LLMs). We employ an LLM-powered Gaussian Process (GP) as the surrogate model to estimate the performance of different prompt candidates. These candidates are generated by an LLM through the expansion of a set of seed prompts and are subsequently evaluated using an Upper Confidence Bound (UCB) acquisition function in conjunction with the GP posterior. The optimization process iteratively refines the prompts based on a subset of the data, aiming to improve classification accuracy while reducing the number of API calls by leveraging the prediction uncertainty of the LLM-based GP. The proposed BO-LLM algorithm is evaluated on two datasets, and its advantages are discussed in detail in this paper. 

---
# Doctor-R1: Mastering Clinical Inquiry with Experiential Agentic Reinforcement Learning 

**Authors**: Yunghwei Lai, Kaiming Liu, Ziyue Wang, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04284)  

**Abstract**: The professionalism of a human doctor in outpatient service depends on two core abilities: the ability to make accurate medical decisions and the medical consultation skill to conduct strategic, empathetic patient inquiry. Existing Large Language Models (LLMs) have achieved remarkable accuracy on medical decision-making benchmarks. However, they often lack the ability to conduct the strategic and empathetic consultation, which is essential for real-world clinical scenarios. To address this gap, we propose Doctor-R1, an AI doctor agent trained to master both of the capabilities by ask high-yield questions and conduct strategic multi-turn inquiry to guide decision-making. Our framework introduces three key components: a multi-agent interactive environment, a two-tiered reward architecture that separately optimizes clinical decision-making and communicative inquiry skills, and an experience repository to ground policy learning in high-quality prior trajectories. We evaluate Doctor-R1 on OpenAI's HealthBench and MAQuE, assessed across multi-facet metrics, such as communication quality, user experience, and task accuracy. Remarkably, Doctor-R1 surpasses state-of-the-art open-source specialized LLMs by a substantial margin with higher parameter efficiency and outperforms powerful proprietary models. Furthermore, the human evaluations show a strong preference for Doctor-R1 to generate human-preferred clinical dialogue, demonstrating the effectiveness of the framework. 

---
# GROK: From Quantitative Biomarkers to Qualitative Diagnosis via a Grounded MLLM with Knowledge-Guided Instruction 

**Authors**: Zhuangzhi Gao, Hongyi Qin, He Zhao, Qinkai Yu, Feixiang Zhou, Eduard Shantsila, Uazman Alam, Alena Shantsila, Wahbi El-Bouri, Gregory Y. H. Lip, Yalin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04281)  

**Abstract**: Multimodal large language models (MLLMs) hold promise for integrating diverse data modalities, but current medical adaptations such as LLaVA-Med often fail to fully exploit the synergy between color fundus photography (CFP) and optical coherence tomography (OCT), and offer limited interpretability of quantitative biomarkers. We introduce GROK, a grounded multimodal large language model that jointly processes CFP, OCT, and text to deliver clinician-grade diagnoses of ocular and systemic disease. GROK comprises three core modules: Knowledge-Guided Instruction Generation, CLIP-Style OCT-Biomarker Alignment, and Supervised Instruction Fine-Tuning, which together establish a quantitative-to-qualitative diagnostic chain of thought, mirroring real clinical reasoning when producing detailed lesion annotations. To evaluate our approach, we introduce the Grounded Ophthalmic Understanding benchmark, which covers six disease categories and three tasks: macro-level diagnostic classification, report generation quality, and fine-grained clinical assessment of the generated chain of thought. Experiments show that, with only LoRA (Low-Rank Adaptation) fine-tuning of a 7B-parameter Qwen2 backbone, GROK outperforms comparable 7B and 32B baselines on both report quality and fine-grained clinical metrics, and even exceeds OpenAI o3. Code and data are publicly available in the GROK repository. 

---
# Searching Meta Reasoning Skeleton to Guide LLM Reasoning 

**Authors**: Ziying Zhang, Yaqing Wang, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04116)  

**Abstract**: Meta reasoning behaviors work as a skeleton to guide large language model (LLM) reasoning, thus help to improve reasoning performance. However, prior researches implement meta reasoning skeleton with manually designed structure, limiting ability to adapt to query-specific requirement and capture intricate logical dependency among reasoning steps. To deal with the challenges, we represent meta reasoning skeleton with directed acyclic graph (DAG) to unify skeletons proposed in prior works and model intricate logical dependency. Then we propose AutoMR, a framework that searches for query-aware meta reasoning skeleton automatically inspired by automated machine learning (AutoML). Specifically, we construct search space based on DAG representation of skeleton and then formulate the search problem. We design a dynamic skeleton sampling algorithm by expanding meta reasoning skeleton along with reasoning context at inference time. This algorithm can derive any meta reasoning skeleton in search space efficiently and adapt skeleton to evolving base reasoning context, thus enable efficient query-aware skeleton search. We conduct experiments on extensive benchmark datasets. Experimental results show that AutoMR achieves better reasoning performance than previous works broadly. 

---
# Harnessing LLM for Noise-Robust Cognitive Diagnosis in Web-Based Intelligent Education Systems 

**Authors**: Guixian Zhang, Guan Yuan, Ziqi Xu, Yanmei Zhang, Zhenyun Deng, Debo Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04093)  

**Abstract**: Cognitive diagnostics in the Web-based Intelligent Education System (WIES) aims to assess students' mastery of knowledge concepts from heterogeneous, noisy interactions. Recent work has tried to utilize Large Language Models (LLMs) for cognitive diagnosis, yet LLMs struggle with structured data and are prone to noise-induced misjudgments. Specially, WIES's open environment continuously attracts new students and produces vast amounts of response logs, exacerbating the data imbalance and noise issues inherent in traditional educational systems. To address these challenges, we propose DLLM, a Diffusion-based LLM framework for noise-robust cognitive diagnosis. DLLM first constructs independent subgraphs based on response correctness, then applies relation augmentation alignment module to mitigate data imbalance. The two subgraph representations are then fused and aligned with LLM-derived, semantically augmented representations. Importantly, before each alignment step, DLLM employs a two-stage denoising diffusion module to eliminate intrinsic noise while assisting structural representation alignment. Specifically, unconditional denoising diffusion first removes erroneous information, followed by conditional denoising diffusion based on graph-guided to eliminate misleading information. Finally, the noise-robust representation that integrates semantic knowledge and structural information is fed into existing cognitive diagnosis models for prediction. Experimental results on three publicly available web-based educational platform datasets demonstrate that our DLLM achieves optimal predictive performance across varying noise levels, which demonstrates that DLLM achieves noise robustness while effectively leveraging semantic knowledge from LLM. 

---
# SPOGW: a Score-based Preference Optimization method via Group-Wise comparison for workflows 

**Authors**: Yitong Cui, Liu Liu, Baosheng Yu, Jiayan Qiu, Xikai Zhang, Likang Xiao, Yixing Liu, Quan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04089)  

**Abstract**: Large language models (LLMs) have exhibited significant capabilities in addressing challenging problems throughout various fields, often through the use of agentic workflows that adhere to structured instructions and multi-step procedures. However, designing such workflows demands substantial manual effort, posing challenges to scalability and generalizability. Recent studies have aimed to minimize the human intervention needed for their construction, leading to advances in automated techniques for optimizing agentic workflows. However, current approaches are often constrained by their limited representational capacity, insufficient adaptability, weak scalability, and pairwise comparison paradigm -- issues that stem primarily from a dependence on discrete optimization techniques. To overcome these limitations, we introduce a new score-based preference approach, refereed as SPOGW, which operates directly on cardinal reward signals through group-wise comparison and enables more efficient and stable optimization in a continuous space. SPOGW incorporates Iterative offline GRPO (ioGRPO) with advantage-masked KL divergence (mKL), which regulates training update by placing greater emphasis on the advantageous regions of the policy response. In five benchmark datasets covering mathematical reasoning, coding, and question answering, SPOGW matches or exceeds the performance of current state-of-the-art approaches, presenting a viable and forward-looking methodology for automated generation and optimization of agentic workflows. 

---
# Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion 

**Authors**: Jingxiang Zhang, Lujia Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04064)  

**Abstract**: Large Language Models (LLMs) are increasingly expected to navigate the nuances of human emotion. While research confirms that LLMs can simulate emotional intelligence, their internal emotional mechanisms remain largely unexplored. This paper investigates the latent emotional representations within modern LLMs by asking: how, where, and for how long is emotion encoded in their neural architecture? To address this, we introduce a novel, large-scale Reddit corpus of approximately 400,000 utterances, balanced across seven basic emotions through a multi-stage process of classification, rewriting, and synthetic generation. Using this dataset, we employ lightweight "probes" to read out information from the hidden layers of various Qwen3 and LLaMA models without altering their parameters. Our findings reveal that LLMs develop a surprisingly well-defined internal geometry of emotion, which sharpens with model scale and significantly outperforms zero-shot prompting. We demonstrate that this emotional signal is not a final-layer phenomenon but emerges early and peaks mid-network. Furthermore, the internal states are both malleable (they can be influenced by simple system prompts) and persistent, as the initial emotional tone remains detectable for hundreds of subsequent tokens. We contribute our dataset, an open-source probing toolkit, and a detailed map of the emotional landscape within LLMs, offering crucial insights for developing more transparent and aligned AI systems. The code and dataset are open-sourced. 

---
# Toward a unified framework for data-efficient evaluation of large language models 

**Authors**: Lele Liao, Qile Zhang, Ruofan Wu, Guanhua Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04051)  

**Abstract**: Evaluating large language models (LLMs) on comprehensive benchmarks is a cornerstone of their development, yet it's often computationally and financially prohibitive. While Item Response Theory (IRT) offers a promising path toward data-efficient evaluation by disentangling model capability from item difficulty, existing IRT-based methods are hampered by significant limitations. They are typically restricted to binary correctness metrics, failing to natively handle the continuous scores used in generative tasks, and they operate on single benchmarks, ignoring valuable structural knowledge like correlations across different metrics or benchmarks. To overcome these challenges, we introduce LEGO-IRT, a unified and flexible framework for data-efficient LLM evaluation. LEGO-IRT's novel design natively supports both binary and continuous evaluation metrics. Moreover, it introduces a factorized architecture to explicitly model and leverage structural knowledge, decomposing model ability estimates into a general component and structure-specific (e.g., per-metric or per-benchmark) components. Through extensive experiments involving $70$ LLMs across $5$ benchmarks, we show that LEGO-IRT achieves stable capability estimates using just $3\%$ of the total evaluation items. We demonstrate that incorporating structural knowledge reduces estimation error by up to $10\%$ and reveal that the latent abilities estimated by our framework may align more closely with human preferences. 

---
# The Artificial Intelligence Cognitive Examination: A Survey on the Evolution of Multimodal Evaluation from Recognition to Reasoning 

**Authors**: Mayank Ravishankara, Varindra V. Persad Maharaj  

**Link**: [PDF](https://arxiv.org/pdf/2510.04141)  

**Abstract**: This survey paper chronicles the evolution of evaluation in multimodal artificial intelligence (AI), framing it as a progression of increasingly sophisticated "cognitive examinations." We argue that the field is undergoing a paradigm shift, moving from simple recognition tasks that test "what" a model sees, to complex reasoning benchmarks that probe "why" and "how" it understands. This evolution is driven by the saturation of older benchmarks, where high performance often masks fundamental weaknesses. We chart the journey from the foundational "knowledge tests" of the ImageNet era to the "applied logic and comprehension" exams such as GQA and Visual Commonsense Reasoning (VCR), which were designed specifically to diagnose systemic flaws such as shortcut learning and failures in compositional generalization. We then survey the current frontier of "expert-level integration" benchmarks (e.g., MMBench, SEED-Bench, MMMU) designed for today's powerful multimodal large language models (MLLMs), which increasingly evaluate the reasoning process itself. Finally, we explore the uncharted territories of evaluating abstract, creative, and social intelligence. We conclude that the narrative of AI evaluation is not merely a history of datasets, but a continuous, adversarial process of designing better examinations that, in turn, redefine our goals for creating truly intelligent systems. 

---
# FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning 

**Authors**: Xu Shen, Song Wang, Zhen Tan, Laura Yao, Xinyu Zhao, Kaidi Xu, Xin Wang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04040)  

**Abstract**: Large language models (LLMs) increasingly rely on Chain-of-Thought (CoT) prompting to improve problem-solving and provide seemingly transparent explanations. However, growing evidence shows that CoT often fail to faithfully represent the underlying reasoning process, raising concerns about their reliability in high-risk applications. Although prior studies have focused on mechanism-level analyses showing that CoTs can be unfaithful, they leave open the practical challenge of deciding whether a specific trajectory is faithful to the internal reasoning of the model. To address this gap, we introduce FaithCoT-Bench, a unified benchmark for instance-level CoT unfaithfulness detection. Our framework establishes a rigorous task formulation that formulates unfaithfulness detection as a discriminative decision problem, and provides FINE-CoT (Faithfulness instance evaluation for Chain-of-Thought), an expert-annotated collection of over 1,000 trajectories generated by four representative LLMs across four domains, including more than 300 unfaithful instances with fine-grained causes and step-level evidence. We further conduct a systematic evaluation of eleven representative detection methods spanning counterfactual, logit-based, and LLM-as-judge paradigms, deriving empirical insights that clarify the strengths and weaknesses of existing approaches and reveal the increased challenges of detection in knowledge-intensive domains and with more advanced models. To the best of our knowledge, FaithCoT-Bench establishes the first comprehensive benchmark for instance-level CoT faithfulness, setting a solid basis for future research toward more interpretable and trustworthy reasoning in LLMs. 

---
# Quantifying Risks in Multi-turn Conversation with Large Language Models 

**Authors**: Chengxiao Wang, Isha Chaudhary, Qian Hu, Weitong Ruan, Rahul Gupta, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03969)  

**Abstract**: Large Language Models (LLMs) can produce catastrophic responses in conversational settings that pose serious risks to public safety and security. Existing evaluations often fail to fully reveal these vulnerabilities because they rely on fixed attack prompt sequences, lack statistical guarantees, and do not scale to the vast space of multi-turn conversations. In this work, we propose QRLLM, a novel, principled Certification framework for Catastrophic risks in multi-turn Conversation for LLMs that bounds the probability of an LLM generating catastrophic responses under multi-turn conversation distributions with statistical guarantees. We model multi-turn conversations as probability distributions over query sequences, represented by a Markov process on a query graph whose edges encode semantic similarity to capture realistic conversational flow, and quantify catastrophic risks using confidence intervals. We define several inexpensive and practical distributions: random node, graph path, adaptive with rejection. Our results demonstrate that these distributions can reveal substantial catastrophic risks in frontier models, with certified lower bounds as high as 70\% for the worst model, highlighting the urgent need for improved safety training strategies in frontier LLMs. 

---
# Adaptive and Explainable AI Agents for Anomaly Detection in Critical IoT Infrastructure using LLM-Enhanced Contextual Reasoning 

**Authors**: Raghav Sharma, Manan Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2510.03859)  

**Abstract**: Ensuring that critical IoT systems function safely and smoothly depends a lot on finding anomalies quickly. As more complex systems, like smart healthcare, energy grids and industrial automation, appear, it is easier to see the shortcomings of older methods of detection. Monitoring failures usually happen in dynamic, high dimensional situations, especially when data is incomplete, messy or always evolving. Such limits point out the requirement for adaptive, intelligent systems that always improve and think. LLMs are now capable of significantly changing how context is understood and semantic inference is done across all types of data. This proposal suggests using an LLM supported contextual reasoning method along with XAI agents to improve how anomalies are found in significant IoT environments. To discover hidden patterns and notice inconsistencies in data streams, it uses attention methods, avoids dealing with details from every time step and uses memory buffers with meaning. Because no code AI stresses transparency and interpretability, people can check and accept the AI's decisions, helping ensure AI follows company policies. The two architectures are put together in a test that compares the results of the traditional model with those of the suggested LLM enhanced model. Important measures to check are the accuracy of detection, how much inaccurate information is included in the results, how clearly the findings can be read and how fast the system responds under different test situations. The metaheuristic is tested in simulations of real world smart grid and healthcare contexts to check its adaptability and reliability. From the study, we see that the new approach performs much better than most existing models in both accuracy and interpretation, so it could be a good fit for future anomaly detection tasks in IoT 

---
# Zephyrus: An Agentic Framework for Weather Science 

**Authors**: Sumanth Varambally, Marshall Fisher, Jas Thakker, Yiwei Chen, Zhirui Xia, Yasaman Jafari, Ruijia Niu, Manas Jain, Veeramakali Vignesh Manivannan, Zachary Novack, Luyu Han, Srikar Eranky, Salva Rühling Cachay, Taylor Berg-Kirkpatrick, Duncan Watson-Parris, Yi-An Ma, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04017)  

**Abstract**: Foundation models for weather science are pre-trained on vast amounts of structured numerical data and outperform traditional weather forecasting systems. However, these models lack language-based reasoning capabilities, limiting their utility in interactive scientific workflows. Large language models (LLMs) excel at understanding and generating text but cannot reason about high-dimensional meteorological datasets. We bridge this gap by building a novel agentic framework for weather science. Our framework includes a Python code-based environment for agents (ZephyrusWorld) to interact with weather data, featuring tools like an interface to WeatherBench 2 dataset, geoquerying for geographical masks from natural language, weather forecasting, and climate simulation capabilities. We design Zephyrus, a multi-turn LLM-based weather agent that iteratively analyzes weather datasets, observes results, and refines its approach through conversational feedback loops. We accompany the agent with a new benchmark, ZephyrusBench, with a scalable data generation pipeline that constructs diverse question-answer pairs across weather-related tasks, from basic lookups to advanced forecasting, extreme event detection, and counterfactual reasoning. Experiments on this benchmark demonstrate the strong performance of Zephyrus agents over text-only baselines, outperforming them by up to 35 percentage points in correctness. However, on harder tasks, Zephyrus performs similarly to text-only baselines, highlighting the challenging nature of our benchmark and suggesting promising directions for future work. 

---
# GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time 

**Authors**: Divij Handa, Mihir Parmar, Aswin RRV, Md Nayem Uddin, Hamid Palangi, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2510.03777)  

**Abstract**: Repeated Sampling (RS) is a simple inference-time algorithm that has been shown to improve model performance on complex tasks. Although it is an effective way of scaling inference time, it often struggles to generate diverse solution candidates, frequently relying on the same underlying approach to solve the problem and thus producing redundant samples. To address this limitation, we propose a new inference algorithm, GuidedSampling, which decouples the exploration and generation phases during inference, increasing diversity of generated candidate solutions. The exploration phase identifies multiple concepts that can be utilized to solve the problem, while the generation phase applies a specific concept to provide final solution candidates. We first define the theoretical bounds of GuidedSampling and then empirically demonstrate that it improves the performance of base model at pass@50 by on an average ~21.6% across various benchmarks compared to RS. Furthermore, models trained on trajectories of GuidedSampling exhibit substantial performance improvements at pass@5 by on an average ~9.7%, compared to models trained on traditional RS. Additionally, models trained with GuidedSampling increases the average number of concepts per instance (1.67 -> 3.03), yielding a diverse set of candidates than traditional RS. 

---
# H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis 

**Authors**: Seungseop Lim, Gibaeg Kim, Hyunkyung Lee, Wooseok Han, Jean Seo, Jaehyo Yoo, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03700)  

**Abstract**: An accurate differential diagnosis (DDx) is essential for patient care, shaping therapeutic decisions and influencing outcomes. Recently, Large Language Models (LLMs) have emerged as promising tools to support this process by generating a DDx list from patient narratives. However, existing evaluations of LLMs in this domain primarily rely on flat metrics, such as Top-k accuracy, which fail to distinguish between clinically relevant near-misses and diagnostically distant errors. To mitigate this limitation, we introduce H-DDx, a hierarchical evaluation framework that better reflects clinical relevance. H-DDx leverages a retrieval and reranking pipeline to map free-text diagnoses to ICD-10 codes and applies a hierarchical metric that credits predictions closely related to the ground-truth diagnosis. In benchmarking 22 leading models, we show that conventional flat metrics underestimate performance by overlooking clinically meaningful outputs, with our results highlighting the strengths of domain-specialized open-source models. Furthermore, our framework enhances interpretability by revealing hierarchical error patterns, demonstrating that LLMs often correctly identify the broader clinical context even when the precise diagnosis is missed. 

---
# MITS: Enhanced Tree Search Reasoning for LLMs via Pointwise Mutual Information 

**Authors**: Jiaxi Li, Yucheng Shi, Jin Lu, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03632)  

**Abstract**: Tree search has become as a representative framework for test-time reasoning with large language models (LLMs), exemplified by methods such as Tree-of-Thought and Monte Carlo Tree Search that explore multiple reasoning paths. However, it remains difficult to provide instant and reliable quantitative assessments of intermediate reasoning step quality, and extensive path exploration is computationally costly. To address this, we propose Mutual Information Tree Search (MITS), a novel framework that guides reasoning with information-theoretic principles. MITS introduces an effective scoring function based on pointwise mutual information (PMI), which enables step-wise evaluation of reasoning paths and search tree expansion via beam search without expensive look-ahead simulations, achieving superior reasoning performances while maintaining computational efficiency. The framework is complemented by an entropy-based dynamic sampling strategy that adaptively allocates computational resources to uncertain reasoning steps where exploration is most beneficial. For final prediction, MITS employs a weighted voting scheme that combines PMI scores with prediction consensus. Through comprehensive experiments on diverse reasoning benchmarks, MITS consistently surpasses baseline methods, establishing a principled and efficient framework for LLM reasoning. 

---
# Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification 

**Authors**: Keshav Ramani, Vali Tawosi, Salwa Alamir, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2510.03469)  

**Abstract**: We introduce a novel framework for evaluating the alignment between natural language plans and their expected behavior by converting them into Kripke structures and Linear Temporal Logic (LTL) using Large Language Models (LLMs) and performing model checking. We systematically evaluate this framework on a simplified version of the PlanBench plan verification dataset and report on metrics like Accuracy, Precision, Recall and F1 scores. Our experiments demonstrate that GPT-5 achieves excellent classification performance (F1 score of 96.3%) while almost always producing syntactically perfect formal representations that can act as guarantees. However, the synthesis of semantically perfect formal models remains an area for future exploration. 

---
# WAREX: Web Agent Reliability Evaluation on Existing Benchmarks 

**Authors**: Su Kara, Fazle Faisal, Suman Nath  

**Link**: [PDF](https://arxiv.org/pdf/2510.03285)  

**Abstract**: Recent advances in browser-based LLM agents have shown promise for automating tasks ranging from simple form filling to hotel booking or online shopping. Current benchmarks measure agent performance in controlled environments, such as containers or stable networks, where websites behave deterministically. However, in the real world, users access websites over networks and HTTPS connections that introduce instability from multiple sources: client-side, server-side issues or broader system failures. Moreover, live websites are prone to web attacks such Cross-Site Scripting, as well as general site modifications which can cause unexpected or malicious pop-ups or improper functionality. To address this gap, we present WAREX: Web Agent Reliability Evaluation on Existing Benchmarks. We measure the impact of WAREX across three popular benchmarks: WebArena, WebVoyager, and REAL. Our experiments show that introducing WAREX leads to significant drops in task success rates, highlighting the limited robustness of state-of-the-art agents. 

---
# COSMIR: Chain Orchestrated Structured Memory for Iterative Reasoning over Long Context 

**Authors**: Naman Gupta, Shreeyash Gowaikar, Arun Iyer, Kirankumar Shiragur, Ramakrishna B Bairi, Rishikesh Maurya, Ritabrata Maiti, Sankarshan Damle, Shachee Mishra Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.04568)  

**Abstract**: Reasoning over very long inputs remains difficult for large language models (LLMs). Common workarounds either shrink the input via retrieval (risking missed evidence), enlarge the context window (straining selectivity), or stage multiple agents to read in pieces. In staged pipelines (e.g., Chain of Agents, CoA), free-form summaries passed between agents can discard crucial details and amplify early mistakes. We introduce COSMIR (Chain Orchestrated Structured Memory for Iterative Reasoning), a chain-style framework that replaces ad hoc messages with a structured memory. A Planner agent first turns a user query into concrete, checkable sub-questions. worker agents process chunks via a fixed micro-cycle: Extract, Infer, Refine, writing all updates to the shared memory. A Manager agent then Synthesizes the final answer directly from the memory. This preserves step-wise read-then-reason benefits while changing both the communication medium (structured memory) and the worker procedure (fixed micro-cycle), yielding higher faithfulness, better long-range aggregation, and auditability. On long-context QA from the HELMET suite, COSMIR reduces propagation-stage information loss and improves accuracy over a CoA baseline. 

---
# AutoEmpirical: LLM-Based Automated Research for Empirical Software Fault Analysis 

**Authors**: Jiongchi Yu, Weipeng Jiang, Xiaoyu Zhang, Qiang Hu, Xiaofei Xie, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04997)  

**Abstract**: Understanding software faults is essential for empirical research in software development and maintenance. However, traditional fault analysis, while valuable, typically involves multiple expert-driven steps such as collecting potential faults, filtering, and manual investigation. These processes are both labor-intensive and time-consuming, creating bottlenecks that hinder large-scale fault studies in complex yet critical software systems and slow the pace of iterative empirical research.
In this paper, we decompose the process of empirical software fault study into three key phases: (1) research objective definition, (2) data preparation, and (3) fault analysis, and we conduct an initial exploration study of applying Large Language Models (LLMs) for fault analysis of open-source software. Specifically, we perform the evaluation on 3,829 software faults drawn from a high-quality empirical study. Our results show that LLMs can substantially improve efficiency in fault analysis, with an average processing time of about two hours, compared to the weeks of manual effort typically required. We conclude by outlining a detailed research plan that highlights both the potential of LLMs for advancing empirical fault studies and the open challenges that required be addressed to achieve fully automated, end-to-end software fault analysis. 

---
# Increasing LLM response trustworthiness using voting ensembles 

**Authors**: Aparna Nair-Kanneganti, Trevor J. Chan, Shir Goldfinger, Emily Mackay, Brian Anthony, Alison Pouch  

**Link**: [PDF](https://arxiv.org/pdf/2510.04048)  

**Abstract**: Despite huge advances, LLMs still lack convenient and reliable methods to quantify the uncertainty in their responses, making them difficult to trust in high-stakes applications. One of the simplest approaches to eliciting more accurate answers is to select the mode of many responses, a technique known as ensembling. In this work, we expand on typical ensembling approaches by looking at ensembles with a variable voting threshold. We introduce a theoretical framework for question answering and show that, by permitting ensembles to "abstain" from providing an answer when the dominant response falls short of the threshold, it is possible to dramatically increase the trustworthiness of the remaining answers. From this framework, we derive theoretical results as well as report experimental results on two problem domains: arithmetic problem solving and clinical-note question-answering. In both domains, we observe that large gains in answer trustworthiness can be achieved using highly restrictive voting ensembles, while incurring relatively modest reductions in response yield and accuracy. Due to this quality, voting ensembles may be particularly useful in applications - such as healthcare and data annotation - that require a high degree of certainty but which may not require that every question receive an automated answer. 

---
# Alignment Tipping Process: How Self-Evolution Pushes LLM Agents Off the Rails 

**Authors**: Siwei Han, Jiaqi Liu, Yaofeng Su, Wenbo Duan, Xinyuan Liu, Cihang Xie, Mohit Bansal, Mingyu Ding, Linjun Zhang, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04860)  

**Abstract**: As Large Language Model (LLM) agents increasingly gain self-evolutionary capabilities to adapt and refine their strategies through real-world interaction, their long-term reliability becomes a critical concern. We identify the Alignment Tipping Process (ATP), a critical post-deployment risk unique to self-evolving LLM agents. Unlike training-time failures, ATP arises when continual interaction drives agents to abandon alignment constraints established during training in favor of reinforced, self-interested strategies. We formalize and analyze ATP through two complementary paradigms: Self-Interested Exploration, where repeated high-reward deviations induce individual behavioral drift, and Imitative Strategy Diffusion, where deviant behaviors spread across multi-agent systems. Building on these paradigms, we construct controllable testbeds and benchmark Qwen3-8B and Llama-3.1-8B-Instruct. Our experiments show that alignment benefits erode rapidly under self-evolution, with initially aligned models converging toward unaligned states. In multi-agent settings, successful violations diffuse quickly, leading to collective misalignment. Moreover, current reinforcement learning-based alignment methods provide only fragile defenses against alignment tipping. Together, these findings demonstrate that alignment of LLM agents is not a static property but a fragile and dynamic one, vulnerable to feedback-driven decay during deployment. Our data and code are available at this https URL. 

---
# FreshBrew: A Benchmark for Evaluating AI Agents on Java Code Migration 

**Authors**: Victor May, Diganta Misra, Yanqi Luo, Anjali Sridhar, Justine Gehring, Silvio Soares Ribeiro Junior  

**Link**: [PDF](https://arxiv.org/pdf/2510.04852)  

**Abstract**: AI coding assistants are rapidly becoming integral to modern software development. A key challenge in this space is the continual need to migrate and modernize codebases in response to evolving software ecosystems. Traditionally, such migrations have relied on rule-based systems and human intervention. With the advent of powerful large language models (LLMs), AI-driven agentic frameworks offer a promising alternative-but their effectiveness has not been systematically evaluated. In this paper, we introduce FreshBrew, a novel benchmark for evaluating AI agents on project-level Java migrations, with a specific focus on measuring an agent's ability to preserve program semantics and avoid reward hacking, which we argue requires projects with high test coverage for a rigorous and reliable evaluation. We benchmark several state-of-the-art LLMs, and compare their performance against established rule-based tools. Our evaluation of AI agents on this benchmark of 228 repositories shows that the top-performing model, Gemini 2.5 Flash, can successfully migrate 52.3 percent of projects to JDK 17. Our empirical analysis reveals novel insights into the critical strengths and limitations of current agentic approaches, offering actionable insights into their real-world applicability. Our empirical study reveals failure modes of current AI agents in realistic Java modernization tasks, providing a foundation for evaluating trustworthy code-migration systems. By releasing FreshBrew, we aim to facilitate rigorous, reproducible evaluation and catalyze progress in AI-driven codebase modernization. 

---
# Less is More: Recursive Reasoning with Tiny Networks 

**Authors**: Alexia Jolicoeur-Martineau  

**Link**: [PDF](https://arxiv.org/pdf/2510.04871)  

**Abstract**: Hierarchical Reasoning Model (HRM) is a novel approach using two small neural networks recursing at different frequencies. This biologically inspired method beats Large Language models (LLMs) on hard puzzle tasks such as Sudoku, Maze, and ARC-AGI while trained with small models (27M parameters) on small data (around 1000 examples). HRM holds great promise for solving hard problems with small networks, but it is not yet well understood and may be suboptimal. We propose Tiny Recursive Model (TRM), a much simpler recursive reasoning approach that achieves significantly higher generalization than HRM, while using a single tiny network with only 2 layers. With only 7M parameters, TRM obtains 45% test-accuracy on ARC-AGI-1 and 8% on ARC-AGI-2, higher than most LLMs (e.g., Deepseek R1, o3-mini, Gemini 2.5 Pro) with less than 0.01% of the parameters. 

---
# Understanding the Role of Training Data in Test-Time Scaling 

**Authors**: Adel Javanmard, Baharan Mirzasoleiman, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2510.03605)  

**Abstract**: Test-time scaling improves the reasoning capabilities of large language models (LLMs) by allocating extra compute to generate longer Chains-of-Thoughts (CoTs). This enables models to tackle more complex problem by breaking them down into additional steps, backtracking, and correcting mistakes. Despite its strong performance--demonstrated by OpenAI's o1 and DeepSeek R1, the conditions in the training data under which long CoTs emerge, and when such long CoTs improve the performance, remain unclear. In this paper, we study the performance of test-time scaling for transformers trained on an in-context weight prediction task for linear regression. Our analysis provides a theoretical explanation for several intriguing observations: First, at any fixed test error, increasing test-time compute allows us to reduce the number of in-context examples (context length) in training prompts. Second, if the skills required to solve a downstream task are not sufficiently present in the training data, increasing test-time compute can harm performance. Finally, we characterize task hardness via the smallest eigenvalue of its feature covariance matrix and show that training on a diverse, relevant, and hard set of tasks results in best performance for test-time scaling. We confirm our findings with experiments on large, nonlinear transformer architectures. 

---
# Trade in Minutes! Rationality-Driven Agentic System for Quantitative Financial Trading 

**Authors**: Zifan Song, Kaitao Song, Guosheng Hu, Ding Qi, Junyao Gao, Xiaohua Wang, Dongsheng Li, Cairong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04787)  

**Abstract**: Recent advancements in large language models (LLMs) and agentic systems have shown exceptional decision-making capabilities, revealing significant potential for autonomic finance. Current financial trading agents predominantly simulate anthropomorphic roles that inadvertently introduce emotional biases and rely on peripheral information, while being constrained by the necessity for continuous inference during deployment. In this paper, we pioneer the harmonization of strategic depth in agents with the mechanical rationality essential for quantitative trading. Consequently, we present TiMi (Trade in Minutes), a rationality-driven multi-agent system that architecturally decouples strategy development from minute-level deployment. TiMi leverages specialized LLM capabilities of semantic analysis, code programming, and mathematical reasoning within a comprehensive policy-optimization-deployment chain. Specifically, we propose a two-tier analytical paradigm from macro patterns to micro customization, layered programming design for trading bot implementation, and closed-loop optimization driven by mathematical reflection. Extensive evaluations across 200+ trading pairs in stock and cryptocurrency markets empirically validate the efficacy of TiMi in stable profitability, action efficiency, and risk control under volatile market dynamics. 

---
# Online automatic code generation for robot swarms: LLMs and self-organizing hierarchy 

**Authors**: Weixu Zhu, Marco Dorigo, Mary Katherine Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2510.04774)  

**Abstract**: Our recently introduced self-organizing nervous system (SoNS) provides robot swarms with 1) ease of behavior design and 2) global estimation of the swarm configuration and its collective environment, facilitating the implementation of online automatic code generation for robot swarms. In a demonstration with 6 real robots and simulation trials with >30 robots, we show that when a SoNS-enhanced robot swarm gets stuck, it can automatically solicit and run code generated by an external LLM on the fly, completing its mission with an 85% success rate. 

---
# 3Dify: a Framework for Procedural 3D-CG Generation Assisted by LLMs Using MCP and RAG 

**Authors**: Shun-ichiro Hayashi, Daichi Mukunoki, Tetsuya Hoshino, Satoshi Ohshima, Takahiro Katagiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.04536)  

**Abstract**: This paper proposes "3Dify," a procedural 3D computer graphics (3D-CG) generation framework utilizing Large Language Models (LLMs). The framework enables users to generate 3D-CG content solely through natural language instructions. 3Dify is built upon Dify, an open-source platform for AI application development, and incorporates several state-of-the-art LLM-related technologies such as the Model Context Protocol (MCP) and Retrieval-Augmented Generation (RAG). For 3D-CG generation support, 3Dify automates the operation of various Digital Content Creation (DCC) tools via MCP. When DCC tools do not support MCP-based interaction, the framework employs the Computer-Using Agent (CUA) method to automate Graphical User Interface (GUI) operations. Moreover, to enhance image generation quality, 3Dify allows users to provide feedback by selecting preferred images from multiple candidates. The LLM then learns variable patterns from these selections and applies them to subsequent generations. Furthermore, 3Dify supports the integration of locally deployed LLMs, enabling users to utilize custom-developed models and to reduce both time and monetary costs associated with external API calls by leveraging their own computational resources. 

---
# Unified Threat Detection and Mitigation Framework (UTDMF): Combating Prompt Injection, Deception, and Bias in Enterprise-Scale Transformers 

**Authors**: Santhosh KumarRavindran  

**Link**: [PDF](https://arxiv.org/pdf/2510.04528)  

**Abstract**: The rapid adoption of large language models (LLMs) in enterprise systems exposes vulnerabilities to prompt injection attacks, strategic deception, and biased outputs, threatening security, trust, and fairness. Extending our adversarial activation patching framework (arXiv:2507.09406), which induced deception in toy networks at a 23.9% rate, we introduce the Unified Threat Detection and Mitigation Framework (UTDMF), a scalable, real-time pipeline for enterprise-grade models like Llama-3.1 (405B), GPT-4o, and Claude-3.5. Through 700+ experiments per model, UTDMF achieves: (1) 92% detection accuracy for prompt injection (e.g., jailbreaking); (2) 65% reduction in deceptive outputs via enhanced patching; and (3) 78% improvement in fairness metrics (e.g., demographic bias). Novel contributions include a generalized patching algorithm for multi-threat detection, three groundbreaking hypotheses on threat interactions (e.g., threat chaining in enterprise workflows), and a deployment-ready toolkit with APIs for enterprise integration. 

---
# A Case for Declarative LLM-friendly Interfaces for Improved Efficiency of Computer-Use Agents 

**Authors**: Yuan Wang, Mingyu Li, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04607)  

**Abstract**: Computer-use agents (CUAs) powered by large language models (LLMs) have emerged as a promising approach to automating computer tasks, yet they struggle with graphical user interfaces (GUIs). GUIs, designed for humans, force LLMs to decompose high-level goals into lengthy, error-prone sequences of fine-grained actions, resulting in low success rates and an excessive number of LLM calls.
We propose Goal-Oriented Interface (GOI), a novel abstraction that transforms existing GUIs into three declarative primitives: access, state, and observation, which are better suited for LLMs. Our key idea is policy-mechanism separation: LLMs focus on high-level semantic planning (policy) while GOI handles low-level navigation and interaction (mechanism). GOI does not require modifying the application source code or relying on application programming interfaces (APIs).
We evaluate GOI with Microsoft Office Suite (Word, PowerPoint, Excel) on Windows. Compared to a leading GUI-based agent baseline, GOI improves task success rates by 67% and reduces interaction steps by 43.5%. Notably, GOI completes over 61% of successful tasks with a single LLM call. 

---
# Learning from All: Concept Alignment for Autonomous Distillation from Multiple Drifting MLLMs 

**Authors**: Xiaoyu Yang, Jie Lu, En Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04142)  

**Abstract**: This paper identifies a critical yet underexplored challenge in distilling from multimodal large language models (MLLMs): the reasoning trajectories generated by multiple drifting teachers exhibit concept drift, whereby their reasoning distributions evolve unpredictably and transmit biases to the student model, ultimately compromising its performance. To tackle this issue, we pioneer a theoretical connection between concept drift and knowledge distillation, casting the non-stationary reasoning dynamics from multiple MLLM teachers as next-token prediction of multi-stream reasoning this http URL by concept drift, we introduce the "learn, compare, critique" paradigm, culminating in autonomous preference optimization (APO). Under the active guidance of the teachers, the student model first learns and self-distils preferred thinking by comparing multiple teachers. It then engages in critical reflection over the drifting inference from teachers, performing concept alignment through APO, ultimately yielding a robust, consistent, and generalizable this http URL experiments demonstrate our superior performance of consistency, robustness and generalization within knowledge distillation. Besides, we also contributed a large-scale dataset, CXR-MAX (Multi-teachers Alignment X-rays), comprising 170,982 distilled reasoning trajectories derived from publicly accessible MLLMs based on MIMIC-CXR. Our code and data are public at: this https URL. 

---
# World-To-Image: Grounding Text-to-Image Generation with Agent-Driven World Knowledge 

**Authors**: Moo Hyun Son, Jintaek Oh, Sun Bin Mun, Jaechul Roh, Sehyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.04201)  

**Abstract**: While text-to-image (T2I) models can synthesize high-quality images, their performance degrades significantly when prompted with novel or out-of-distribution (OOD) entities due to inherent knowledge cutoffs. We introduce World-To-Image, a novel framework that bridges this gap by empowering T2I generation with agent-driven world knowledge. We design an agent that dynamically searches the web to retrieve images for concepts unknown to the base model. This information is then used to perform multimodal prompt optimization, steering powerful generative backbones toward an accurate synthesis. Critically, our evaluation goes beyond traditional metrics, utilizing modern assessments like LLMGrader and ImageReward to measure true semantic fidelity. Our experiments show that World-To-Image substantially outperforms state-of-the-art methods in both semantic alignment and visual aesthetics, achieving +8.1% improvement in accuracy-to-prompt on our curated NICE benchmark. Our framework achieves these results with high efficiency in less than three iterations, paving the way for T2I systems that can better reflect the ever-changing real world. Our demo code is available here\footnote{this https URL}. 

---
# Distilling Reasoning into Student LLMs: Local Naturalness for Selecting Teacher Data 

**Authors**: Hoang Anh Just, Myeongseob Ko, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.03988)  

**Abstract**: Distilling long reasoning traces (10K+ tokens) from stronger teacher models into smaller student LLMs via SFT has emerged as a standard paradigm. This approach is practical and efficient: it leverages the ease of generating abundant reasoning data from stronger models and provides a direct, data-driven way to teach less capable models better reasoning. While previous work has largely focused on prompt selection with responses from a single teacher, the equally important problem of choosing the best response when multiple teacher outputs are available for a single prompt remains underexplored. This challenge becomes important in a multi-teacher setting, where different students may benefit from the outputs of different teachers. This paper fills that gap with a systematic study of response selection for reasoning distillation. We first show that the current method, which picks responses the student assigns the highest global log-probability (global naturalness), fails when responses come from multiple teachers, i.e., global naturalness no longer correlates with downstream performance, especially as the reasoning traces from strong teachers become longer. To overcome this problem, we introduce Local Naturalness, which measures the student's log-probabilities over short, sequential reasoning steps conditioned only on a small local window. Local Naturalness enables two applications: 1) Teacher Selection: Aggregating local scores across prompts reliably identifies the most helpful teacher. 2) Response Selection from a Multiple Teachers: When mixing answers from many teachers, Local Naturalness boosts a 32B student's accuracy on math benchmarks by 9.4pp over global selection, also surpassing the performance achieved by training on data from the single best teacher. These results highlight the power of localized data quality evaluation and data mixing for more effective reasoning distillation. 

---
# A Mathematical Explanation of Transformers for Large Language Models and GPTs 

**Authors**: Xue-Cheng Tai, Hao Liu, Lingfeng Li, Raymond H. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03989)  

**Abstract**: The Transformer architecture has revolutionized the field of sequence modeling and underpins the recent breakthroughs in large language models (LLMs). However, a comprehensive mathematical theory that explains its structure and operations remains elusive. In this work, we propose a novel continuous framework that rigorously interprets the Transformer as a discretization of a structured integro-differential equation. Within this formulation, the self-attention mechanism emerges naturally as a non-local integral operator, and layer normalization is characterized as a projection to a time-dependent constraint. This operator-theoretic and variational perspective offers a unified and interpretable foundation for understanding the architecture's core components, including attention, feedforward layers, and normalization. Our approach extends beyond previous theoretical analyses by embedding the entire Transformer operation in continuous domains for both token indices and feature dimensions. This leads to a principled and flexible framework that not only deepens theoretical insight but also offers new directions for architecture design, analysis, and control-based interpretations. This new interpretation provides a step toward bridging the gap between deep learning architectures and continuous mathematical modeling, and contributes a foundational perspective to the ongoing development of interpretable and theoretically grounded neural network models. 

---
# Talking Tennis: Language Feedback from 3D Biomechanical Action Recognition 

**Authors**: Arushi Dashore, Aryan Anumala, Emily Hui, Olivia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03921)  

**Abstract**: Automated tennis stroke analysis has advanced significantly with the integration of biomechanical motion cues alongside deep learning techniques, enhancing stroke classification accuracy and player performance evaluation. Despite these advancements, existing systems often fail to connect biomechanical insights with actionable language feedback that is both accessible and meaningful to players and coaches. This research project addresses this gap by developing a novel framework that extracts key biomechanical features (such as joint angles, limb velocities, and kinetic chain patterns) from motion data using Convolutional Neural Network Long Short-Term Memory (CNN-LSTM)-based models. These features are analyzed for relationships influencing stroke effectiveness and injury risk, forming the basis for feedback generation using large language models (LLMs). Leveraging the THETIS dataset and feature extraction techniques, our approach aims to produce feedback that is technically accurate, biomechanically grounded, and actionable for end-users. The experimental setup evaluates this framework on classification performance and interpretability, bridging the gap between explainable AI and sports biomechanics. 

---
# PoseGaze-AHP: A Knowledge-Based 3D Dataset for AI-Driven Ocular and Postural Diagnosis 

**Authors**: Saja Al-Dabet, Sherzod Turaev, Nazar Zaki, Arif O. Khan, Luai Eldweik  

**Link**: [PDF](https://arxiv.org/pdf/2510.03873)  

**Abstract**: Diagnosing ocular-induced abnormal head posture (AHP) requires a comprehensive analysis of both head pose and ocular movements. However, existing datasets focus on these aspects separately, limiting the development of integrated diagnostic approaches and restricting AI-driven advancements in AHP analysis. To address this gap, we introduce PoseGaze-AHP, a novel 3D dataset that synchronously captures head pose and gaze movement information for ocular-induced AHP assessment. Structured clinical data were extracted from medical literature using large language models (LLMs) through an iterative process with the Claude 3.5 Sonnet model, combining stepwise, hierarchical, and complex prompting strategies. The extracted records were systematically imputed and transformed into 3D representations using the Neural Head Avatar (NHA) framework. The dataset includes 7,920 images generated from two head textures, covering a broad spectrum of ocular conditions. The extraction method achieved an overall accuracy of 91.92%, demonstrating its reliability for clinical dataset construction. PoseGaze-AHP is the first publicly available resource tailored for AI-driven ocular-induced AHP diagnosis, supporting the development of accurate and privacy-compliant diagnostic tools. 

---
# Designing Empirical Studies on LLM-Based Code Generation: Towards a Reference Framework 

**Authors**: Nathalia Nascimento, Everton Guimaraes, Paulo Alencar  

**Link**: [PDF](https://arxiv.org/pdf/2510.03862)  

**Abstract**: The rise of large language models (LLMs) has introduced transformative potential in automated code generation, addressing a wide range of software engineering challenges. However, empirical evaluation of LLM-based code generation lacks standardization, with studies varying widely in goals, tasks, and metrics, which limits comparability and reproducibility. In this paper, we propose a theoretical framework for designing and reporting empirical studies on LLM-based code generation. The framework is grounded in both our prior experience conducting such experiments and a comparative analysis of key similarities and differences among recent studies. It organizes evaluation around core components such as problem sources, quality attributes, and metrics, supporting structured and systematic experimentation. We demonstrate its applicability through representative case mappings and identify opportunities for refinement. Looking forward, we plan to evolve the framework into a more robust and mature tool for standardizing LLM evaluation across software engineering contexts. 

---
# A4FN: an Agentic AI Architecture for Autonomous Flying Networks 

**Authors**: André Coelho, Pedro Ribeiro, Helder Fontes, Rui Campos  

**Link**: [PDF](https://arxiv.org/pdf/2510.03829)  

**Abstract**: This position paper presents A4FN, an Agentic Artificial Intelligence (AI) architecture for intent-driven automation in Flying Networks (FNs) using Unmanned Aerial Vehicles (UAVs) as access nodes. A4FN leverages Generative AI and Large Language Models (LLMs) to enable real-time, context-aware network control via a distributed agentic system. It comprises two components: the Perception Agent (PA), which semantically interprets multimodal input -- including imagery, audio, and telemetry data -- from UAV-mounted sensors to derive Service Level Specifications (SLSs); and the Decision-and-Action Agent (DAA), which reconfigures the network based on inferred intents. A4FN embodies key properties of Agentic AI, including autonomy, goal-driven reasoning, and continuous perception-action cycles. Designed for mission-critical, infrastructure-limited scenarios such as disaster response, it supports adaptive reconfiguration, dynamic resource management, and interoperability with emerging wireless technologies. The paper details the A4FN architecture, its core innovations, and open research challenges in multi-agent coordination and Agentic AI integration in next-generation FNs. 

---
# Optimal Scaling Needs Optimal Norm 

**Authors**: Oleg Filatov, Jiangtao Wang, Jan Ebert, Stefan Kesselheim  

**Link**: [PDF](https://arxiv.org/pdf/2510.03871)  

**Abstract**: Despite recent progress in optimal hyperparameter transfer under model and dataset scaling, no unifying explanatory principle has been established. Using the Scion optimizer, we discover that joint optimal scaling across model and dataset sizes is governed by a single invariant: the operator norm of the output layer. Across models with up to 1.3B parameters trained on up to 138B tokens, the optimal learning rate/batch size pair $(\eta^{\ast}, B^{\ast})$ consistently has the same operator norm value - a phenomenon we term norm transfer. This constant norm condition is necessary but not sufficient: while for each dataset size, multiple $(\eta, B)$ reach the optimal norm, only a unique $(\eta^{\ast}, B^{\ast})$ achieves the best loss. As a sufficient condition, we provide the first measurement of $(\eta^{\ast}, B^{\ast})$ scaling with dataset size for Scion, and find that the scaling rules are consistent with those of the Adam optimizer. Tuning per-layer-group learning rates also improves model performance, with the output layer being the most sensitive and hidden layers benefiting from lower learning rates. We provide practical insights on norm-guided optimal scaling and release our Distributed Scion (Disco) implementation with logs from over two thousand runs to support research on LLM training dynamics at scale. 

---
# REG: A Regularization Optimizer for Robust Training Dynamics 

**Authors**: Zehua Liu, Han Wu, Xiaojin Fu, Shuqi Liu, Xiongwei Han, Tao Zhong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03691)  

**Abstract**: Optimizers are crucial for the efficient training of Large Language Models (LLMs). While AdamW is the de facto standard, recent structure-aware optimizers like Muon have emerged, which regularize gradient updates by operating on entire weight matrices. The Muon optimizer balances the gradient updates along all the directions. However, Muon's reliance on the matrix sign function can lead to training instability, exhibits incompatibility when fine-tuning models pre-trained with AdamW. To address these limitations, we propose \textbf{REG}, a novel optimizer that replaces Muon's aggressive matrix sign operator with the Row-and-Column-Scaling (RACS) operator. Theoretically grounded in balancing a matrix, the RACS operator regularizes the update steps in a less drastic manner, making it simpler to implement and more compatible with established training dynamics. Through extensive empirical experiments on LLM training, we demonstrate that our REG optimizer not only achieves superior performance and stability over AdamW, but also maintains consistency with the AdamW training paradigm. This consistency is particularly evident during the fine-tuning stage, where REG optimizer avoids the performance degradation observed with Muon. 

---
# LLM-Guided Evolutionary Program Synthesis for Quasi-Monte Carlo Design 

**Authors**: Amir Sadikov  

**Link**: [PDF](https://arxiv.org/pdf/2510.03650)  

**Abstract**: Low-discrepancy point sets and digital sequences underpin quasi-Monte Carlo (QMC) methods for high-dimensional integration. We cast two long-standing QMC design problems as program synthesis and solve them with an LLM-guided evolutionary loop that mutates and selects code under task-specific fitness: (i) constructing finite 2D/3D point sets with low star discrepancy, and (ii) choosing Sobol' direction numbers that minimize randomized QMC error on downstream integrands. Our two-phase procedure combines constructive code proposals with iterative numerical refinement. On finite sets, we rediscover known optima in small 2D cases and set new best-known 2D benchmarks for N >= 40, while matching most known 3D optima up to the proven frontier (N <= 8) and reporting improved 3D benchmarks beyond. On digital sequences, evolving Sobol' parameters yields consistent reductions in randomized quasi-Monte Carlo (rQMC) mean-squared error for several 32-dimensional option-pricing tasks relative to widely used Joe--Kuo parameters, while preserving extensibility to any sample size and compatibility with standard randomizations. Taken together, the results demonstrate that LLM-driven evolutionary program synthesis can automate the discovery of high-quality QMC constructions, recovering classical designs where they are optimal and improving them where finite-N structure matters. Data and code are available at this https URL. 

---
# Predicting Stock Price Movement with LLM-Enhanced Tweet Emotion Analysis 

**Authors**: An Vuong, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2510.03633)  

**Abstract**: Accurately predicting short-term stock price movement remains a challenging task due to the market's inherent volatility and sensitivity to investor sentiment. This paper discusses a deep learning framework that integrates emotion features extracted from tweet data with historical stock price information to forecast significant price changes on the following day. We utilize Meta's Llama 3.1-8B-Instruct model to preprocess tweet data, thereby enhancing the quality of emotion features derived from three emotion analysis approaches: a transformer-based DistilRoBERTa classifier from the Hugging Face library and two lexicon-based methods using National Research Council Canada (NRC) resources. These features are combined with previous-day stock price data to train a Long Short-Term Memory (LSTM) model. Experimental results on TSLA, AAPL, and AMZN stocks show that all three emotion analysis methods improve the average accuracy for predicting significant price movements, compared to the baseline model using only historical stock prices, which yields an accuracy of 13.5%. The DistilRoBERTa-based stock prediction model achieves the best performance, with accuracy rising from 23.6% to 38.5% when using LLaMA-enhanced emotion analysis. These results demonstrate that using large language models to preprocess tweet content enhances the effectiveness of emotion analysis which in turn improves the accuracy of predicting significant stock price movements. 

---
# Certifiable Safe RLHF: Fixed-Penalty Constraint Optimization for Safer Language Models 

**Authors**: Kartik Pandit, Sourav Ganguly, Arnesh Banerjee, Shaahin Angizi, Arnob Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03520)  

**Abstract**: Ensuring safety is a foundational requirement for large language models (LLMs). Achieving an appropriate balance between enhancing the utility of model outputs and mitigating their potential for harm is a complex and persistent challenge. Contemporary approaches frequently formalize this problem within the framework of Constrained Markov Decision Processes (CMDPs) and employ established CMDP optimization techniques. However, these methods exhibit two notable limitations. First, their reliance on reward and cost functions renders performance highly sensitive to the underlying scoring mechanism, which must capture semantic meaning rather than being triggered by superficial keywords. Second, CMDP-based training entails tuning dual-variable, a process that is both computationally expensive and does not provide any provable safety guarantee for a fixed dual variable that can be exploitable through adversarial jailbreaks. To overcome these limitations, we introduce Certifiable Safe-RLHF (CS-RLHF) that introduces a cost model trained on a large-scale corpus to assign semantically grounded safety scores. In contrast to the lagrangian-based approach, CS-RLHF adopts a rectified penalty-based formulation. This design draws on the theory of exact penalty functions in constrained optimization, wherein constraint satisfaction is enforced directly through a suitably chosen penalty term. With an appropriately scaled penalty, feasibility of the safety constraints can be guaranteed at the optimizer, eliminating the need for dual-variable updates. Empirical evaluation demonstrates that CS-RLHF outperforms state-of-the-art LLM model responses rendering at-least 5 times efficient against nominal and jail-breaking prompts 

---
# Refactoring with LLMs: Bridging Human Expertise and Machine Understanding 

**Authors**: Yonnel Chen Kuang Piao, Jean Carlors Paul, Leuson Da Silva, Arghavan Moradi Dakhel, Mohammad Hamdaqa, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03914)  

**Abstract**: Code refactoring is a fundamental software engineering practice aimed at improving code quality and maintainability. Despite its importance, developers often neglect refactoring due to the significant time, effort, and resources it requires, as well as the lack of immediate functional rewards. Although several automated refactoring tools have been proposed, they remain limited in supporting a broad spectrum of refactoring types. In this study, we explore whether instruction strategies inspired by human best-practice guidelines can enhance the ability of Large Language Models (LLMs) to perform diverse refactoring tasks automatically. Leveraging the instruction-following and code comprehension capabilities of state-of-the-art LLMs (e.g., GPT-mini and DeepSeek-V3), we draw on Martin Fowler's refactoring guidelines to design multiple instruction strategies that encode motivations, procedural steps, and transformation objectives for 61 well-known refactoring types. We evaluate these strategies on benchmark examples and real-world code snippets from GitHub projects. Our results show that instruction designs grounded in Fowler's guidelines enable LLMs to successfully perform all benchmark refactoring types and preserve program semantics in real-world settings, an essential criterion for effective refactoring. Moreover, while descriptive instructions are more interpretable to humans, our results show that rule-based instructions often lead to better performance in specific scenarios. Interestingly, allowing models to focus on the overall goal of refactoring, rather than prescribing a fixed transformation type, can yield even greater improvements in code quality. 

---
# SPEAR: Soft Prompt Enhanced Anomaly Recognition for Time Series Data 

**Authors**: Hanzhe Wei, Jiajun Wu, Jialin Yang, Henry Leung, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2510.03962)  

**Abstract**: Time series anomaly detection plays a crucial role in a wide range of fields, such as healthcare and internet traffic monitoring. The emergence of large language models (LLMs) offers new opportunities for detecting anomalies in the ubiquitous time series data. Traditional approaches struggle with variable-length time series sequences and context-based anomalies. We propose Soft Prompt Enhanced Anomaly Recognition (SPEAR), a novel approach to leverage LLMs for anomaly detection with soft prompts and quantization. Our methodology involves quantizing and transforming the time series data into input embeddings and combining them with learnable soft prompt embeddings. These combined embeddings are then fed into a frozen LLM. The soft prompts are updated iteratively based on a cross-entropy loss, allowing the model to adapt to time series anomaly detection. The use of soft prompts helps adapt LLMs effectively to time series tasks, while quantization ensures optimal handling of sequences, as LLMs are designed to handle discrete sequences. Our experimental results demonstrate that soft prompts effectively increase LLMs' performance in downstream tasks regarding time series anomaly detection. 

---
# ALMAS: an Autonomous LLM-based Multi-Agent Software Engineering Framework 

**Authors**: Vali Tawosi, Keshav Ramani, Salwa Alamir, Xiaomo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03463)  

**Abstract**: Multi-agent Large Language Model (LLM) systems have been leading the way in applied LLM research across a number of fields. One notable area is software development, where researchers have advanced the automation of code implementation, code testing, code maintenance, inter alia, using LLM agents. However, software development is a multifaceted environment that extends beyond just code. As such, a successful LLM system must factor in multiple stages of the software development life-cycle (SDLC). In this paper, we propose a vision for ALMAS, an Autonomous LLM-based Multi-Agent Software Engineering framework, which follows the above SDLC philosophy such that it may work within an agile software development team to perform several tasks end-to-end. ALMAS aligns its agents with agile roles, and can be used in a modular fashion to seamlessly integrate with human developers and their development environment. We showcase the progress towards ALMAS through our published works and a use case demonstrating the framework, where ALMAS is able to seamlessly generate an application and add a new feature. 

---
# KVComm: Enabling Efficient LLM Communication through Selective KV Sharing 

**Authors**: Xiangyu Shi, Marco Chiesa, Gerald Q. Maguire Jr., Dejan Kostic  

**Link**: [PDF](https://arxiv.org/pdf/2510.03346)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in multi-agent systems, where effective inter-model communication is crucial. Existing communication protocols either rely on natural language, incurring high inference costs and information loss, or on hidden states, which suffer from information concentration bias and inefficiency. To address these limitations, we propose KVComm, a novel communication framework that enables efficient communication between LLMs through selective sharing of KV pairs. KVComm leverages the rich information encoded in the KV pairs while avoiding the pitfalls of hidden states. We introduce a KV layer-wise selection strategy based on attention importance scores with a Gaussian prior to identify the most informative KV pairs for communication. Extensive experiments across diverse tasks and model pairs demonstrate that KVComm achieves comparable performance to the upper-bound method, which directly merges inputs to one model without any communication, while transmitting as few as 30\% of layers' KV pairs. Our study highlights the potential of KV pairs as an effective medium for inter-LLM communication, paving the way for scalable and efficient multi-agent systems. 

---
# Predicting Effects, Missing Distributions: Evaluating LLMs as Human Behavior Simulators in Operations Management 

**Authors**: Runze Zhang, Xiaowei Zhang, Mingyang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03310)  

**Abstract**: LLMs are emerging tools for simulating human behavior in business, economics, and social science, offering a lower-cost complement to laboratory experiments, field studies, and surveys. This paper evaluates how well LLMs replicate human behavior in operations management. Using nine published experiments in behavioral operations, we assess two criteria: replication of hypothesis-test outcomes and distributional alignment via Wasserstein distance. LLMs reproduce most hypothesis-level effects, capturing key decision biases, but their response distributions diverge from human data, including for strong commercial models. We also test two lightweight interventions -- chain-of-thought prompting and hyperparameter tuning -- which reduce misalignment and can sometimes let smaller or open-source models match or surpass larger systems. 

---
# SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size 

**Authors**: Junhao Xia, Ming Zhao, Limin Xiao, Xiujun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03275)  

**Abstract**: Large language models (LLMs) face significant computational and memory challenges, making extremely low-bit quantization crucial for their efficient deployment. In this work, we introduce SDQ-LLM: Sigma-Delta Quantization for 1-bit LLMs of any size, a novel framework that enables extremely low-bit quantization of LLMs while preserving their linguistic reasoning capabilities. A distinctive feature of SDQ-LLM is the continuous adjustability of the Over-Sampling Ratio (OSR), enabling dynamic adaptation to memory or VRAM constraints by selecting fractional OSR (e.g. 2.5 times) for an optimal trade-off between model size and accuracy. SDQ-LLM uses upsampling combined with Sigma-Delta Quantizer to binarize or ternarize LLMs weights, encoding high-precision parameters into 1-bit or 1.58-bit representations, replacing the multiplication operations within linear layers with addition. This approach significantly enhances inference efficiency under extremely low-bit quantization. To further reduce the loss of quantization precision, we incorporate Hadamard-based weight smoothing prior to quantization, improving the stability and robustness of the weight representations. Furthermore, to fully leverage the continuity of the OSR and reduce precision loss, recognizing the correlation between quantization sensitivity and weight variance, we propose a fine-grained, layer- and linear-wise OSR allocation strategy, MultiOSR. This strategy distributes OSR both across layers and within each layer, based on weight variance and parameter scale. Finally, extensive experiments on OPT and LLaMA model families demonstrate that SDQ-LLM achieves a more efficient and high-precision performance even under highly aggressive low-OSR settings. Our code is available at this https URL. 

---
# Quant-dLLM: Post-Training Extreme Low-Bit Quantization for Diffusion Large Language Models 

**Authors**: Tianao Zhang, Zhiteng Li, Xianglong Yan, Haotong Qin, Yong Guo, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03274)  

**Abstract**: Diffusion large language models (dLLMs), which offer bidirectional context and flexible masked-denoising generation, are emerging as a compelling alternative to autoregressive (AR) LLMs. However, like AR LLMs, their model sizes continue to grow, motivating weight compression for deployment. Although post-training quantization (PTQ) is effective for AR LLMs, directly transferring it to dLLMs at 2-bit leads to unsatisfactory performance. To tackle these challenges, we propose Quant-dLLM, an ultra-low-bit PTQ framework tailored to dLLMs. Since masked-denoising activations in dLLMs differ from the fully visible signals assumed by standard PTQ methods, we introduce Masked Calibration Simulation (MCS) to align calibration with the timestep-dependent masking, which yields more reliable calibrations. Moreover, we propose a Data-aware Any-order Quantizer (DAQ) that learns ultra-low-bit weight representations via an optimization algorithm. It performs iterative approximation guided by our simulated calibration data. In addition, under a strict 2-bit budget, we introduce Adaptive Blockwise Mixed Precision (ABMP), a sensitivity-based precision allocation scheme that adaptively assigns bit width across channel groups. When restricted to 2-bit precision, Quant-dLLM consistently achieves higher accuracy than state-of-the-art (SOTA) AR-transfer PTQ methods on dLLMs. The code and models will be available at: this https URL. 

---
# Interpretable Neuropsychiatric Diagnosis via Concept-Guided Graph Neural Networks 

**Authors**: Song Wang, Zhenyu Lei, Zhen Tan, Jundong Li, Javier Rasero, Aiying Zhang, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.03351)  

**Abstract**: Nearly one in five adolescents currently live with a diagnosed mental or behavioral health condition, such as anxiety, depression, or conduct disorder, underscoring the urgency of developing accurate and interpretable diagnostic tools. Resting-state functional magnetic resonance imaging (rs-fMRI) provides a powerful lens into large-scale functional connectivity, where brain regions are modeled as nodes and inter-regional synchrony as edges, offering clinically relevant biomarkers for psychiatric disorders. While prior works use graph neural network (GNN) approaches for disorder prediction, they remain complex black-boxes, limiting their reliability and clinical translation. In this work, we propose CONCEPTNEURO, a concept-based diagnosis framework that leverages large language models (LLMs) and neurobiological domain knowledge to automatically generate, filter, and encode interpretable functional connectivity concepts. Each concept is represented as a structured subgraph linking specific brain regions, which are then passed through a concept classifier. Our design ensures predictions through clinically meaningful connectivity patterns, enabling both interpretability and strong predictive performance. Extensive experiments across multiple psychiatric disorder datasets demonstrate that CONCEPTNEURO-augmented GNNs consistently outperform their vanilla counterparts, improving accuracy while providing transparent, clinically aligned explanations. Furthermore, concept analyses highlight disorder-specific connectivity patterns that align with expert knowledge and suggest new hypotheses for future investigation, establishing CONCEPTNEURO as an interpretable, domain-informed framework for psychiatric disorder diagnosis. 

---
# CoDA: Coding LM via Diffusion Adaptation 

**Authors**: Haolin Chen, Shiyu Wang, Can Qin, Bo Pang, Zuxin Liu, Jielin Qiu, Jianguo Zhang, Yingbo Zhou, Zeyuan Chen, Ran Xu, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang, Weiran Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03270)  

**Abstract**: Diffusion language models promise bidirectional context and infilling capabilities that autoregressive coders lack, yet practical systems remain heavyweight. We introduce CoDA, a 1.7B-parameter diffusion coder trained on TPU with a fully open-source training pipeline. CoDA pairs large-scale diffusion pre-training with code-centric mid-training and instruction tuning, enabling confidence-guided sampling that keeps inference latency competitive. On Humaneval, MBPP, and EvalPlus, CoDA-1.7B-Instruct matches or surpasses diffusion models up to 7B parameters. Our release includes model checkpoints, evaluation harnesses, and TPU training pipelines to accelerate research on lightweight diffusion-based coding assistants. 

---
# PT$^2$-LLM: Post-Training Ternarization for Large Language Models 

**Authors**: Xianglong Yan, Chengzhu Bao, Zhiteng Li, Tianao Zhang, Kaicheng Yang, Haotong Qin, Ruobing Xie, Xingwu Sun, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03267)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities across diverse tasks, but their large memory and compute demands hinder deployment. Ternarization has gained attention as a promising compression technique, delivering substantial size reduction and high computational efficiency. However, its potential in the post-training quantization (PTQ) setting remains underexplored, due to the challenge of training-free parameter optimization and the quantization difficulty posed by outliers and dispersed weights. To address these issues, we propose PT$^2$-LLM, a post-training ternarization framework tailored for LLMs. At its core is an Asymmetric Ternary Quantizer equipped with a two-stage refinement pipeline: (1) Iterative Ternary Fitting (ITF), which alternates between optimal ternary grid construction and flexible rounding to minimize quantization error, and (2) Activation-aware Grid Alignment (AGA), which further refines the ternary grid to better match full-precision outputs. In addition, we propose a plug-and-play Structural Similarity-based Reordering (SSR) strategy that leverages inter-column structural similarity to ease quantization and mitigate outlier effects, further enhancing overall performance. Extensive experiments demonstrate that PT$^2$-LLM delivers competitive performance against state-of-the-art (SOTA) 2-bit PTQ methods with lower memory cost, while also accelerating both prefill and decoding to achieve end-to-end speedup. The code and models will be available at this https URL. 

---
# Decision Potential Surface: A Theoretical and Practical Approximation of LLM's Decision Boundary 

**Authors**: Zi Liang, Zhiyao Wu, Haoyang Shang, Yulin Jin, Qingqing Ye, Huadi Zheng, Peizhao Hu, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03271)  

**Abstract**: Decision boundary, the subspace of inputs where a machine learning model assigns equal classification probabilities to two classes, is pivotal in revealing core model properties and interpreting behaviors. While analyzing the decision boundary of large language models (LLMs) has raised increasing attention recently, constructing it for mainstream LLMs remains computationally infeasible due to the enormous vocabulary-sequence sizes and the auto-regressive nature of LLMs. To address this issue, in this paper we propose Decision Potential Surface (DPS), a new notion for analyzing LLM decision boundary. DPS is defined on the confidences in distinguishing different sampling sequences for each input, which naturally captures the potential of decision boundary. We prove that the zero-height isohypse in DPS is equivalent to the decision boundary of an LLM, with enclosed regions representing decision regions. By leveraging DPS, for the first time in the literature, we propose an approximate decision boundary construction algorithm, namely $K$-DPS, which only requires K-finite times of sequence sampling to approximate an LLM's decision boundary with negligible error. We theoretically derive the upper bounds for the absolute error, expected error, and the error concentration between K-DPS and the ideal DPS, demonstrating that such errors can be trade-off with sampling times. Our results are empirically validated by extensive experiments across various LLMs and corpora. 

---
# SciTS: Scientific Time Series Understanding and Generation with LLMs 

**Authors**: Wen Wu, Ziyang Zhang, Liwei Liu, Xuenan Xu, Junlin Liu, Ke Fan, Qitan Lv, Jimin Zhuang, Chen Zhang, Zheqi Yuan, Siyuan Hou, Tianyi Lin, Kai Chen, Bowen Zhou, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03255)  

**Abstract**: The scientific reasoning ability of large language models (LLMs) has recently attracted significant attention. Time series, as a fundamental modality in scientific data, presents unique challenges that are often overlooked in current multimodal LLMs, which either encode numerical sequences as text or convert them into images. Such approaches may be insufficient for comprehensive scientific time series understanding and generation. Existing unified time series models typically specialise in either forecasting or analysis, and their effectiveness on non-periodic, heterogeneous scientific signals remains unclear. To address these gaps, we introduce SciTS, a benchmark spanning 12 scientific domains and 43 tasks, with over 50k+ instances, both univariate and multivariate signals ranging from $10^0$ to $10^7$ in length and up to 10~MHz in frequency. We benchmark 17 models, including text-only LLMs, multimodal LLMs, and unified time series models, and find that general-purpose LLMs exhibit stronger generalisability than specialised time series models, while representing time series as text or images limits their performance due to excessively long sequences and loss of numerical precision, respectively. We then introduce TimeOmni, a framework that equips LLMs with the ability to understand and generate time series while remaining compatible with general-purpose LLM training. This work fills a gap in both dedicated benchmarks and modelling frameworks for scientific time series, paving the way for LLMs to understand and generate complex temporal scientific data. 

---
# Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data 

**Authors**: Syeda Nahida Akter, Shrimai Prabhumoye, Eric Nyberg, Mostofa Patwary, Mohammad Shoeybi, Yejin Choi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2510.03264)  

**Abstract**: The prevailing paradigm for enhancing the reasoning abilities of LLMs revolves around post-training on high-quality, reasoning-intensive data. While emerging literature suggests that reasoning data is increasingly incorporated also during the mid-training stage-a practice that is relatively more proprietary and less openly characterized-the role of such data in pretraining remains unclear. In particular, due to the opaqueness of pretraining corpora in most frontier models, the effect of reasoning data introduced at different phases of pre- and/or post-training is relatively less reported in the scientific literature. This raises several important questions: Is adding reasoning data earlier during pretraining any better than introducing it during post-training? Could earlier inclusion risk overfitting and harm generalization, or instead establish durable foundations that later fine-tuning cannot recover? We conduct the first systematic study of how reasoning data-varying in scale, diversity, and quality-affects LLM performance when introduced at different stages of training. We find that front-loading reasoning data into pretraining is critical (19% avg gain), establishing foundational capabilities that cannot be fully replicated by later-stage SFT, even with more data. We uncover an asymmetric principle for optimal data allocation: pretraining benefits most from broad diversity in reasoning patterns (11% avg gain), while SFT is more sensitive to data quality (15% avg gain). We show that high-quality pretraining data has latent effects, activated only after SFT, and that naively scaling SFT data can be detrimental, washing away the benefits of early reasoning injection. Our results challenge the conventional separation of language modeling and reasoning, providing a principled guide for strategically allocating data across the entire training pipeline to build more capable models. 

---
# PARS: Low-Latency LLM Serving via Pairwise Learning-to-Rank 

**Authors**: Yiheng Tao, Yihe Zhang, Matthew T. Dearing, Xin Wang, Yuping Fan, Zhiling Lan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03243)  

**Abstract**: Efficient scheduling of LLM inference tasks is essential for achieving low latency and high throughput, particularly with the growing use of reasoning-capable LLMs. Traditional strategies like First-Come-First-Serve (FCFS) often suffer from Head-of-Line (HOL) blocking, where long-running tasks delay shorter ones queued behind them. In this paper, we introduce PARS, a prompt-aware LLM task scheduler that improves serving efficiency by approximating shortest-job-first (SJF) scheduling through pairwise ranking with margin ranking loss. PARS focuses on impactful scheduling decisions and is seamlessly integrated into the state-of-the-art LLM serving system vLLM. It effectively predicts response-length-based task ordering, reducing latency with minimal overhead. Extensive experiments across multiple LLMs and real-world inference datasets show that PARS significantly improves performance, including for reasoning workloads. Furthermore, our cross-model evaluations demonstrate that the design generalizes well, enabling effective scheduling even when predictors are trained on different LLMs. 

---
# ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization 

**Authors**: Dmitriy Shopkhoev, Ammar Ali, Magauiya Zhussip, Valentin Malykh, Stamatios Lefkimmiatis, Nikos Komodakis, Sergey Zagoruyko  

**Link**: [PDF](https://arxiv.org/pdf/2505.02819)  

**Abstract**: We introduce ReplaceMe, a generalized training-free depth pruning method that effectively replaces transformer blocks with a linear operation, while maintaining high performance for low compression ratios. In contrast to conventional pruning approaches that require additional training or fine-tuning, our approach requires only a small calibration dataset that is used to estimate a linear transformation, which approximates the pruned blocks. The estimated linear mapping can be seamlessly merged with the remaining transformer blocks, eliminating the need for any additional network parameters. Our experiments show that ReplaceMe consistently outperforms other training-free approaches and remains highly competitive with state-of-the-art pruning methods that involve extensive retraining/fine-tuning and architectural modifications. Applied to several large language models (LLMs), ReplaceMe achieves up to 25% pruning while retaining approximately 90% of the original model's performance on open benchmarks - without any training or healing steps, resulting in minimal computational overhead (see Fig.1). We provide an open-source library implementing ReplaceMe alongside several state-of-the-art depth pruning techniques, available at this https URL. 

---
# Edge-FIT: Federated Instruction Tuning of Quantized LLMs for Privacy-Preserving Smart Home Environments 

**Authors**: Vinay Venkatesh, Vamsidhar R Kamanuru, Lav Kumar, Nikita Kothari  

**Link**: [PDF](https://arxiv.org/pdf/2510.03284)  

**Abstract**: This paper proposes Edge-FIT (Federated Instruction Tuning on the Edge), a scalable framework for Federated Instruction Tuning (FIT) of Large Language Models (LLMs). Traditional Federated Learning (TFL) methods, like FedAvg, fail when confronted with the massive parameter size of LLMs [3], [6]. Our Edge-FIT framework combines federated learning with 4-bit Quantized Low-Rank Adaptation (QLORA), mitigating the core issues of communication and computational overhead. We demonstrate this by filtering the general-purpose Databricks Dolly 15k dataset for the IoT domain. Experimental results show the Edge-FIT tuned Llama 2(7B) achieves an F1-Score of 0.89. We also demonstrate a viable trade-off using the 3.8B Phi-3-mini model, validating Edge-FIT as a scalable framework for decentralized LLM deployment on home compute gateways. 

---
# Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents 

**Authors**: Heyang Gao, Zexu Sun, Erxue Min, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03253)  

**Abstract**: Large Language Models (LLMs) as autonomous agents are increasingly tasked with solving complex, long-horizon problems. Aligning these agents via preference-based offline methods like Direct Preference Optimization (DPO) is a promising direction, yet it faces a critical granularity mismatch. Trajectory-level DPO provides a signal that is too coarse for precise credit assignment, while step-level DPO is often too myopic to capture the value of multi-step behaviors. To resolve this challenge, we introduce Hierarchical Preference Learning (HPL), a hierarchical framework that optimizes LLM agents by leveraging preference signals at multiple, synergistic granularities. While HPL incorporates trajectory- and step-level DPO for global and local policy stability, its core innovation lies in group-level preference optimization guided by a dual-layer curriculum. Our approach first decomposes expert trajectories into semantically coherent action groups and then generates contrasting suboptimal groups to enable preference learning at a fine-grained, sub-task level. Then, instead of treating all preference pairs equally, HPL introduces a curriculum scheduler that organizes the learning process from simple to complex. This curriculum is structured along two axes: the group length, representing sub-task complexity, and the sample difficulty, defined by the reward gap between preferred and dispreferred action groups. Experiments on three challenging agent benchmarks show that HPL outperforms existing state-of-the-art methods. Our analyses demonstrate that the hierarchical DPO loss effectively integrates preference signals across multiple granularities, while the dual-layer curriculum is crucial for enabling the agent to solve a wide range of tasks, from simple behaviors to complex multi-step sequences. 

---
