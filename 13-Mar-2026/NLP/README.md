# SciMDR: Benchmarking and Advancing Scientific Multimodal Document Reasoning 

**Authors**: Ziyu Chen, Yilun Zhao, Chengye Wang, Rilyn Han, Manasi Patwardhan, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2603.12249)  

**Abstract**: Constructing scientific multimodal document reasoning datasets for foundation model training involves an inherent trade-off among scale, faithfulness, and realism. To address this challenge, we introduce the synthesize-and-reground framework, a two-stage pipeline comprising: (1) Claim-Centric QA Synthesis, which generates faithful, isolated QA pairs and reasoning on focused segments, and (2) Document-Scale Regrounding, which programmatically re-embeds these pairs into full-document tasks to ensure realistic complexity. Using this framework, we construct SciMDR, a large-scale training dataset for cross-modal comprehension, comprising 300K QA pairs with explicit reasoning chains across 20K scientific papers. We further construct SciMDR-Eval, an expert-annotated benchmark to evaluate multimodal comprehension within full-length scientific workflows. Experiments demonstrate that models fine-tuned on SciMDR achieve significant improvements across multiple scientific QA benchmarks, particularly in those tasks requiring complex document-level reasoning. 

---
# Sparking Scientific Creativity via LLM-Driven Interdisciplinary Inspiration 

**Authors**: Priyanka Kargupta, Shuhaib Mehri, Dilek Hakkani-Tur, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.12226)  

**Abstract**: Despite interdisciplinary research leading to larger and longer-term impact, most work remains confined to single-domain academic silos. Recent AI-based approaches to scientific discovery show promise for interdisciplinary research, but many prioritize rapidly designing experiments and solutions, bypassing the exploratory, collaborative reasoning processes that drive creative interdisciplinary breakthroughs. As a result, prior efforts largely prioritize automating scientific discovery rather than augmenting the reasoning processes that underlie scientific disruption. We present Idea-Catalyst, a novel framework that systematically identifies interdisciplinary insights to support creative reasoning in both humans and large language models. Starting from an abstract research goal, Idea-Catalyst is designed to assist the brainstorming stage, explicitly avoiding premature anchoring on specific solutions. The framework embodies key metacognitive features of interdisciplinary reasoning: (a) defining and assessing research goals, (b) awareness of a domain's opportunities and unresolved challenges, and (c) strategic exploration of interdisciplinary ideas based on impact potential. Concretely, Idea-Catalyst decomposes an abstract goal (e.g., improving human-AI collaboration) into core target-domain research questions that guide the analysis of progress and open challenges within that domain. These challenges are reformulated as domain-agnostic conceptual problems, enabling retrieval from external disciplines (e.g., Psychology, Sociology) that address analogous issues. By synthesizing and recontextualizing insights from these domains back into the target domain, Idea-Catalyst ranks source domains by their interdisciplinary potential. Empirically, this targeted integration improves average novelty by 21% and insightfulness by 16%, while remaining grounded in the original research problem. 

---
# CLASP: Defending Hybrid Large Language Models Against Hidden State Poisoning Attacks 

**Authors**: Alexandre Le Mercier, Thomas Demeester, Chris Develder  

**Link**: [PDF](https://arxiv.org/pdf/2603.12206)  

**Abstract**: State space models (SSMs) like Mamba have gained significant traction as efficient alternatives to Transformers, achieving linear complexity while maintaining competitive performance. However, Hidden State Poisoning Attacks (HiSPAs), a recently discovered vulnerability that corrupts SSM memory through adversarial strings, pose a critical threat to these architectures and their hybrid variants. Framing the HiSPA mitigation task as a binary classification problem at the token level, we introduce the CLASP model to defend against this threat. CLASP exploits distinct patterns in Mamba's block output embeddings (BOEs) and uses an XGBoost classifier to identify malicious tokens with minimal computational overhead. We consider a realistic scenario in which both SSMs and HiSPAs are likely to be used: an LLM screening résumés to identify the best candidates for a role. Evaluated on a corpus of 2,483 résumés totaling 9.5M tokens with controlled injections, CLASP achieves 95.9% token-level F1 score and 99.3% document-level F1 score on malicious tokens detection. Crucially, the model generalizes to unseen attack patterns: under leave-one-out cross-validation, performance remains high (96.9% document-level F1), while under clustered cross-validation with structurally novel triggers, it maintains useful detection capability (91.6% average document-level F1). Operating independently of any downstream model, CLASP processes 1,032 tokens per second with under 4GB VRAM consumption, potentially making it suitable for real-world deployment as a lightweight front-line defense for SSM-based and hybrid architectures. All code and detailed results are available at this https URL. 

---
# IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse 

**Authors**: Yushi Bai, Qian Dong, Ting Jiang, Xin Lv, Zhengxiao Du, Aohan Zeng, Jie Tang, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.12201)  

**Abstract**: Long-context agentic workflows have emerged as a defining use case for large language models, making attention efficiency critical for both inference speed and serving cost. Sparse attention addresses this challenge effectively, and DeepSeek Sparse Attention (DSA) is a representative production-grade solution: a lightweight lightning indexer selects the top-k most relevant tokens per query, reducing core attention from $O(L^2)$ to $O(Lk)$. However, the indexer itself retains $O(L^2)$ complexity and must run independently at every layer, despite the fact that the resulting top-k selections are highly similar across consecutive layers. We present IndexCache, which exploits this cross-layer redundancy by partitioning layers into a small set of Full layers that run their own indexers and a majority of Shared layers that simply reuse the nearest Full layer's top-k indices. We propose two complementary approaches to determine and optimize this configuration. Training-free IndexCache applies a greedy search algorithm that selects which layers to retain indexers by directly minimizing language modeling loss on a calibration set, requiring no weight updates. Training-aware IndexCache introduces a multi-layer distillation loss that trains each retained indexer against the averaged attention distributions of all layers it serves, enabling even simple interleaved patterns to match full-indexer accuracy. Experimental results on a 30B DSA model show that IndexCache can remove 75% of indexer computations with negligible quality degradation, achieving up to 1.82$\times$ prefill speedup and 1.48$\times$ decode speedup compared to standard DSA. These positive results are further confirmed by our preliminary experiments on the production-scale GLM-5 model (Figure 1). 

---
# Long-Context Encoder Models for Polish Language Understanding 

**Authors**: Sławomir Dadas, Rafał Poświata, Marek Kozłowski, Małgorzata Grębowiec, Michał Perełkiewicz, Paweł Klimiuk, Przemysław Boruta  

**Link**: [PDF](https://arxiv.org/pdf/2603.12191)  

**Abstract**: While decoder-only Large Language Models (LLMs) have recently dominated the NLP landscape, encoder-only architectures remain a cost-effective and parameter-efficient standard for discriminative tasks. However, classic encoders like BERT are limited by a short context window, which is insufficient for processing long documents. In this paper, we address this limitation for the Polish by introducing a high-quality Polish model capable of processing sequences of up to 8192 tokens. The model was developed by employing a two-stage training procedure that involves positional embedding adaptation and full parameter continuous pre-training. Furthermore, we propose compressed model variants trained via knowledge distillation. The models were evaluated on 25 tasks, including the KLEJ benchmark, a newly introduced financial task suite (FinBench), and other classification and regression tasks, specifically those requiring long-document understanding. The results demonstrate that our model achieves the best average performance among Polish and multilingual models, significantly outperforming competitive solutions in long-context tasks while maintaining comparable quality on short texts. 

---
# Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections 

**Authors**: Łukasz Borchmann, Jordy Van Landeghem, Michał Turski, Shreyansh Padarha, Ryan Othniel Kearns, Adam Mahdi, Niels Rogge, Clémentine Fourrier, Siwei Han, Huaxiu Yao, Artemis Llabrés, Yiming Xu, Dimosthenis Karatzas, Hao Zhang, Anupam Datta  

**Link**: [PDF](https://arxiv.org/pdf/2603.12180)  

**Abstract**: Multimodal agents offer a promising path to automating complex document-intensive workflows. Yet, a critical question remains: do these agents demonstrate genuine strategic reasoning, or merely stochastic trial-and-error search? To address this, we introduce MADQA, a benchmark of 2,250 human-authored questions grounded in 800 heterogeneous PDF documents. Guided by Classical Test Theory, we design it to maximize discriminative power across varying levels of agentic abilities. To evaluate agentic behaviour, we introduce a novel evaluation protocol measuring the accuracy-effort trade-off. Using this framework, we show that while the best agents can match human searchers in raw accuracy, they succeed on largely different questions and rely on brute-force search to compensate for weak strategic planning. They fail to close the nearly 20% gap to oracle performance, persisting in unproductive loops. We release the dataset and evaluation harness to help facilitate the transition from brute-force retrieval to calibrated, efficient reasoning. 

---
# QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions 

**Authors**: Jiayin Lei, Ming Ma, Yunxi Duan, Chenxi Li, Tianming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12165)  

**Abstract**: Synthetic data has become essential for training code generation models, yet it introduces significant noise and hallucinations that are difficult to detect with current metrics. Existing data selection methods like Instruction-Following Difficulty (IFD) typically assess how hard a model generates an answer given a query ($A|Q$). However, this metric is ambiguous on noisy synthetic data, where low probability can distinguish between intrinsic task complexity and model-generated hallucinations. Here, we propose QAQ, a novel data selection framework that evaluates data quality from the reverse direction: how well can the answer predict the query ($Q|A$)? We define Reverse Mutual Information (RMI) to quantify the information gain about the query conditioned on the answer. Our analyses reveal that both extremes of RMI signal quality issues: low RMI indicates semantic misalignment, while excessively high RMI may contain defect patterns that LLMs easily recognize. Furthermore, we introduce a selection strategy based on the disagreement between strong and weak models to identify samples that are valid yet challenging. Experiments on the WarriorCoder dataset demonstrate that selecting just 25% of data using stratified RMI achieves comparable performance to full-data training, significantly outperforming existing data selection methods. Our approach highlights the importance of bidirectional semantic coherence in synthetic data curation, offering a scalable pathway to reduce computational costs without sacrificing model capability. 

---
# LifeSim: Long-Horizon User Life Simulator for Personalized Assistant Evaluation 

**Authors**: Feiyu Duan, Xuanjing Huang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.12152)  

**Abstract**: The rapid advancement of large language models (LLMs) has accelerated progress toward universal AI assistants. However, existing benchmarks for personalized assistants remain misaligned with real-world user-assistant interactions, failing to capture the complexity of external contexts and users' cognitive states. To bridge this gap, we propose LifeSim, a user simulator that models user cognition through the Belief-Desire-Intention (BDI) model within physical environments for coherent life trajectories generation, and simulates intention-driven user interactive behaviors. Based on LifeSim, we introduce LifeSim-Eval, a comprehensive benchmark for multi-scenario, long-horizon personalized assistance. LifeSim-Eval covers 8 life domains and 1,200 diverse scenarios, and adopts a multi-turn interactive method to assess models' abilities to complete explicit and implicit intentions, recover user profiles, and produce high-quality responses. Under both single-scenario and long-horizon settings, our experiments reveal that current LLMs face significant limitations in handling implicit intention and long-term user preference modeling. 

---
# Cross-Context Review: Improving LLM Output Quality by Separating Production and Review Sessions 

**Authors**: Tae-Eun Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.12123)  

**Abstract**: Large language models struggle to catch errors in their own outputs when the review happens in the same session that produced them. This paper introduces Cross-Context Review (CCR), a straightforward method where the review is conducted in a fresh session with no access to the production conversation history. We ran a controlled experiment: 30 artifacts (code, technical documents, presentation scripts) with 150 injected errors, tested under four review conditions -- same-session Self-Review (SR), repeated Self-Review (SR2), context-aware Subagent Review (SA), and Cross-Context Review (CCR). Over 360 reviews, CCR reached an F1 of 28.6%, outperforming SR (24.6%, p=0.008, d=0.52), SR2 (21.7%, p<0.001, d=0.72), and SA (23.8%, p=0.004, d=0.57). The SR2 result matters most for interpretation: reviewing twice in the same session did not beat reviewing once (p=0.11), which rules out repetition as an explanation for CCR's advantage. The benefit comes from context separation itself. CCR works with any model, needs no infrastructure, and costs only one extra session. 

---
# SommBench: Assessing Sommelier Expertise of Language Models 

**Authors**: William Brach, Tomas Bedej, Jacob Nielsen, Jacob Pichna, Juraj Bedej, Eemeli Saarensilta, Julie Dupouy, Gianluca Barmina, Andrea Blasi Núñez, Peter Schneider-Kamp, Kristian Košťál, Michal Ries, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2603.12117)  

**Abstract**: With the rapid advances of large language models, it becomes increasingly important to systematically evaluate their multilingual and multicultural capabilities. Previous cultural evaluation benchmarks focus mainly on basic cultural knowledge that can be encoded in linguistic form. Here, we propose SommBench, a multilingual benchmark to assess sommelier expertise, a domain deeply grounded in the senses of smell and taste. While language models learn about sensory properties exclusively through textual descriptions, SommBench tests whether this textual grounding is sufficient to emulate expert-level sensory judgment. SommBench comprises three main tasks: Wine Theory Question Answering (WTQA), Wine Feature Completion (WFC), and Food-Wine Pairing (FWP). SommBench is available in multiple languages: English, Slovak, Swedish, Finnish, German, Danish, Italian, and Spanish. This helps separate a language model's wine expertise from its language skills. The benchmark datasets were developed in close collaboration with a professional sommelier and native speakers of the respective languages, resulting in 1,024 wine theory question-answering questions, 1,000 wine feature-completion examples, and 1,000 food-wine pairing examples. We provide results for the most popular language models, including closed-weights models such as Gemini 2.5, and open-weights models, such as GPT-OSS and Qwen 3. Our results show that the most capable models perform well on wine theory question answering (up to 97% correct with a closed-weights model), yet feature completion (peaking at 65%) and food-wine pairing show (MCC ranging between 0 and 0.39) turn out to be more challenging. These results position SommBench as an interesting and challenging benchmark for evaluating the sommelier expertise of language models. The benchmark is publicly available at this https URL. 

---
# To Words and Beyond: Probing Large Language Models for Sentence-Level Psycholinguistic Norms of Memorability and Reading Times 

**Authors**: Thomas Hikaru Clark, Carlos Arriaga, Javier Conde, Gonzalo Martínez, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2603.12105)  

**Abstract**: Large Language Models (LLMs) have recently been shown to produce estimates of psycholinguistic norms, such as valence, arousal, or concreteness, for words and multiword expressions, that correlate with human judgments. These estimates are obtained by prompting an LLM, in zero-shot fashion, with a question similar to those used in human studies. Meanwhile, for other norms such as lexical decision time or age of acquisition, LLMs require supervised fine-tuning to obtain results that align with ground-truth values. In this paper, we extend this approach to the previously unstudied features of sentence memorability and reading times, which involve the relationship between multiple words in a sentence-level context. Our results show that via fine-tuning, models can provide estimates that correlate with human-derived norms and exceed the predictive power of interpretable baseline predictors, demonstrating that LLMs contain useful information about sentence-level features. At the same time, our results show very mixed zero-shot and few-shot performance, providing further evidence that care is needed when using LLM-prompting as a proxy for human cognitive measures. 

---
# Translationese as a Rational Response to Translation Task Difficulty 

**Authors**: Maria Kunilovskaya  

**Link**: [PDF](https://arxiv.org/pdf/2603.12050)  

**Abstract**: Translations systematically diverge from texts originally produced in the target language, a phenomenon widely referred to as translationese. Translationese has been attributed to production tendencies (e.g. interference, simplification), socio-cultural variables, and language-pair effects, yet a unified explanatory account is still lacking. We propose that translationese reflects cognitive load inherent in the translation task itself. We test whether observable translationese can be predicted from quantifiable measures of translation task difficulty. Translationese is operationalised as a segment-level translatedness score produced by an automatic classifier. Translation task difficulty is conceptualised as comprising source-text and cross-lingual transfer components, operationalised mainly through information-theoretic metrics based on LLM surprisal, complemented by established syntactic and semantic alternatives. We use a bidirectional English-German corpus comprising written and spoken subcorpora. Results indicate that translationese can be partly explained by translation task difficulty, especially in English-to-German. For most experiments, cross-lingual transfer difficulty contributes more than source-text complexity. Information-theoretic indicators match or outperform traditional features in written mode, but offer no advantage in spoken mode. Source-text syntactic complexity and translation-solution entropy emerged as the strongest predictors of translationese across language pairs and modes. 

---
# Just Use XML: Revisiting Joint Translation and Label Projection 

**Authors**: Thennal D K, Chris Biemann, Hans Ole Hatzel  

**Link**: [PDF](https://arxiv.org/pdf/2603.12021)  

**Abstract**: Label projection is an effective technique for cross-lingual transfer, extending span-annotated datasets from a high-resource language to low-resource ones. Most approaches perform label projection as a separate step after machine translation, and prior work that combines the two reports degraded translation quality. We re-evaluate this claim with LabelPigeon, a novel framework that jointly performs translation and label projection via XML tags. We design a direct evaluation scheme for label projection, and find that LabelPigeon outperforms baselines and actively improves translation quality in 11 languages. We further assess translation quality across 203 languages and varying annotation complexity, finding consistent improvement attributed to additional fine-tuning. Finally, across 27 languages and three downstream tasks, we report substantial gains in cross-lingual transfer over comparable work, up to +39.9 F1 on NER. Overall, our results demonstrate that XML-tagged label projection provides effective and efficient label transfer without compromising translation quality. 

---
# BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs 

**Authors**: Ilias Aarab  

**Link**: [PDF](https://arxiv.org/pdf/2603.11991)  

**Abstract**: Zero-shot text classification (ZSC) offers the promise of eliminating costly task-specific annotation by matching texts directly to human-readable label descriptions. While early approaches have predominantly relied on cross-encoder models fine-tuned for natural language inference (NLI), recent advances in text-embedding models, rerankers, and instruction-tuned large language models (LLMs) have challenged the dominance of NLI-based architectures. Yet, systematically comparing these diverse approaches remains difficult. Existing evaluations, such as MTEB, often incorporate labeled examples through supervised probes or fine-tuning, leaving genuine zero-shot capabilities underexplored. To address this, we introduce BTZSC, a comprehensive benchmark of 22 public datasets spanning sentiment, topic, intent, and emotion classification, capturing diverse domains, class cardinalities, and document lengths. Leveraging BTZSC, we conduct a systematic comparison across four major model families, NLI cross-encoders, embedding models, rerankers and instruction-tuned LLMs, encompassing 38 public and custom checkpoints. Our results show that: (i) modern rerankers, exemplified by Qwen3-Reranker-8B, set a new state-of-the-art with macro F1 = 0.72; (ii) strong embedding models such as GTE-large-en-v1.5 substantially close the accuracy gap while offering the best trade-off between accuracy and latency; (iii) instruction-tuned LLMs at 4--12B parameters achieve competitive performance (macro F1 up to 0.67), excelling particularly on topic classification but trailing specialized rerankers; (iv) NLI cross-encoders plateau even as backbone size increases; and (v) scaling primarily benefits rerankers and LLMs over embedding models. BTZSC and accompanying evaluation code are publicly released to support fair and reproducible progress in zero-shot text understanding. 

---
# CHiL(L)Grader: Calibrated Human-in-the-Loop Short-Answer Grading 

**Authors**: Pranav Raikote, Korbinian Randl, Ioanna Miliou, Athanasios Lakes, Panagiotis Papapetrou  

**Link**: [PDF](https://arxiv.org/pdf/2603.11957)  

**Abstract**: Scaling educational assessment with large language models requires not just accuracy, but the ability to recognize when predictions are trustworthy. Instruction-tuned models tend to be overconfident, and their reliability deteriorates as curricula evolve, making fully autonomous deployment unsafe in high-stakes settings. We introduce CHiL(L)Grader, the first automated grading framework that incorporates calibrated confidence estimation into a human-in-the-loop workflow. Using post-hoc temperature scaling, confidence-based selective prediction, and continual learning, CHiL(L)Grader automates only high-confidence predictions while routing uncertain cases to human graders, and adapts to evolving rubrics and unseen questions. Across three short-answer grading datasets, CHiL(L)Grader automatically scores 35-65% of responses at expert-level quality (QWK >= 0.80). A QWK gap of 0.347 between accepted and rejected predictions confirms the effectiveness of the confidence-based routing. Each correction cycle strengthens the model's grading capability as it learns from teacher feedback. These results show that uncertainty quantification is key for reliable AI-assisted grading. 

---
# PersonaTrace: Synthesizing Realistic Digital Footprints with LLM Agents 

**Authors**: Minjia Wang, Yunfeng Wang, Xiao Ma, Dexin Lv, Qifan Guo, Lynn Zheng, Benliang Wang, Lei Wang, Jiannan Li, Yongwei Xing, David Xu, Zheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.11955)  

**Abstract**: Digital footprints (records of individuals' interactions with digital systems) are essential for studying behavior, developing personalized applications, and training machine learning models. However, research in this area is often hindered by the scarcity of diverse and accessible data. To address this limitation, we propose a novel method for synthesizing realistic digital footprints using large language model (LLM) agents. Starting from a structured user profile, our approach generates diverse and plausible sequences of user events, ultimately producing corresponding digital artifacts such as emails, messages, calendar entries, reminders, etc. Intrinsic evaluation results demonstrate that the generated dataset is more diverse and realistic than existing baselines. Moreover, models fine-tuned on our synthetic data outperform those trained on other synthetic datasets when evaluated on real-world out-of-distribution tasks. 

---
# CoMMET: To What Extent Can LLMs Perform Theory of Mind Tasks? 

**Authors**: Ruirui Chen, Weifeng Jiang, Chengwei Qin, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2603.11915)  

**Abstract**: Theory of Mind (ToM)-the ability to reason about the mental states of oneself and others-is a cornerstone of human social intelligence. As Large Language Models (LLMs) become ubiquitous in real-world applications, validating their capacity for this level of social reasoning is essential for effective and natural interactions. However, existing benchmarks for assessing ToM in LLMs are limited; most rely solely on text inputs and focus narrowly on belief-related tasks. In this paper, we propose a new multimodal benchmark dataset, CoMMET, a Comprehensive Mental states and Moral Evaluation Task inspired by the Theory of Mind Booklet Task. CoMMET expands the scope of evaluation by covering a broader range of mental states and introducing multi-turn testing. To the best of our knowledge, this is the first multimodal dataset to evaluate ToM in a multi-turn conversational setting. Through a comprehensive assessment of LLMs across different families and sizes, we analyze the strengths and limitations of current models and identify directions for future improvement. Our work offers a deeper understanding of the social cognitive capabilities of modern LLMs. 

---
# Bielik-Minitron-7B: Compressing Large Language Models via Structured Pruning and Knowledge Distillation for the Polish Language 

**Authors**: Remigiusz Kinas, Paweł Kiszczak, Sergio P. Perez, Krzysztof Ociepa, Łukasz Flis, Krzysztof Wróbel, Adrian Gwoździej  

**Link**: [PDF](https://arxiv.org/pdf/2603.11881)  

**Abstract**: This report details the creation of Bielik-Minitron-7B, a compressed 7.35B parameter version of the Bielik-11B-v3.0 model, specifically optimized for European languages. By leveraging a two-stage compression methodology inspired by the NVIDIA Minitron approach, we combined structured hybrid pruning and knowledge distillation to reduce the model's parameter count by 33.4%, from 11.04B to 7.35B. We utilized the NVIDIA Model Optimizer for structural pruning and the NVIDIA NeMo Framework for logit-based distillation for quality recovery. Following distillation, the model underwent a rigorous alignment pipeline consisting of Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO-P), and Reinforcement Learning (GRPO). Our final model successfully recovered approximately 90% of the baseline model's performance while providing up to 50% inference speedup. This approach demonstrates an efficient pathway to create language models for less-represented languages, preserving the original model quality while reducing inference deployment costs. 

---
# DatedGPT: Preventing Lookahead Bias in Large Language Models with Time-Aware Pretraining 

**Authors**: Yutong Yan, Raphael Tang, Zhenyu Gao, Wenxi Jiang, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.11838)  

**Abstract**: In financial backtesting, large language models pretrained on internet-scale data risk introducing lookahead bias that undermines their forecasting validity, as they may have already seen the true outcome during training. To address this, we present DatedGPT, a family of twelve 1.3B-parameter language models, each trained from scratch on approximately 100 billion tokens of temporally partitioned data with strict annual cutoffs spanning 2013 to 2024. We further enhance each model with instruction fine-tuning on both general-domain and finance-specific datasets curated to respect the same temporal boundaries. Perplexity-based probing confirms that each model's knowledge is effectively bounded by its data cutoff year, while evaluation on standard benchmarks shows competitive performance with existing models of similar scale. We provide an interactive web demo that allows users to query and compare responses from models across different cutoff years. 

---
# Large Language Models for Biomedical Article Classification 

**Authors**: Jakub Proboszcz, Paweł Cichosz  

**Link**: [PDF](https://arxiv.org/pdf/2603.11780)  

**Abstract**: This work presents a systematic and in-depth investigation of the utility of large language models as text classifiers for biomedical article classification. The study uses several small and mid-size open source models, as well as selected closed source ones, and is more comprehensive than most prior work with respect to the scope of evaluated configurations: different types of prompts, output processing methods for generating both class and class probability predictions, as well as few-shot example counts and selection methods. The performance of the most successful configurations is compared to that of conventional classification algorithms. The obtained average PR AUC over 15 challenging datasets above 0.4 for zero-shot prompting and nearly 0.5 for few-shot prompting comes close to that of the naïve Bayes classifier (0.5), the random forest algorithm (0.5 with default settings or 0.55 with hyperparameter tuning) and fine-tuned transformer models (0.5). These results confirm the utility of large language models as text classifiers for non-trivial domains and provide practical recommendations of the most promising setups, including in particular using output token probabilities for class probability prediction. 

---
# Trust Oriented Explainable AI for Fake News Detection 

**Authors**: Krzysztof Siwek, Daniel Stankowski, Maciej Stodolski  

**Link**: [PDF](https://arxiv.org/pdf/2603.11778)  

**Abstract**: This article examines the application of Explainable Artificial Intelligence (XAI) in NLP based fake news detection and compares selected interpretability methods. The work outlines key aspects of disinformation, neural network architectures, and XAI techniques, with a focus on SHAP, LIME, and Integrated Gradients. In the experimental study, classification models were implemented and interpreted using these methods. The results show that XAI enhances model transparency and interpretability while maintaining high detection accuracy. Each method provides distinct explanatory value: SHAP offers detailed local attributions, LIME provides simple and intuitive explanations, and Integrated Gradients performs efficiently with convolutional models. The study also highlights limitations such as computational cost and sensitivity to parameterization. Overall, the findings demonstrate that integrating XAI with NLP is an effective approach to improving the reliability and trustworthiness of fake news detection systems. 

---
# Legal-DC: Benchmarking Retrieval-Augmented Generation for Legal Documents 

**Authors**: Yaocong Li, Qiang Lan, Leihan Zhang, Le Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.11772)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising technology for legal document consultation, yet its application in Chinese legal scenarios faces two key limitations: existing benchmarks lack specialized support for joint retriever-generator evaluation, and mainstream RAG systems often fail to accommodate the structured nature of legal provisions. To address these gaps, this study advances two core contributions: First, we constructed the Legal-DC benchmark dataset, comprising 480 legal documents (covering areas such as market regulation and contract management) and 2,475 refined question-answer pairs, each annotated with clause-level references, filling the gap for specialized evaluation resources in Chinese legal RAG. Second, we propose the LegRAG framework, which integrates legal adaptive indexing (clause-boundary segmentation) with a dual-path self-reflection mechanism to ensure clause integrity while enhancing answer accuracy. Third, we introduce automated evaluation methods for large language models to meet the high-reliability demands of legal retrieval scenarios. LegRAG outperforms existing state-of-the-art methods by 1.3% to 5.6% across key evaluation metrics. This research provides a specialized benchmark, practical framework, and empirical insights to advance the development of Chinese legal RAG systems. Our code and data are available at this https URL. 

---
# Compression Favors Consistency, Not Truth: When and Why Language Models Prefer Correct Information 

**Authors**: Konstantin Krestnikov  

**Link**: [PDF](https://arxiv.org/pdf/2603.11749)  

**Abstract**: Why do language models sometimes prefer correct statements even when trained on mixed-quality data? We introduce the Compression--Consistency Principle: next-token prediction favors hypotheses that allow shorter and more internally consistent descriptions of the training data. Truth bias emerges only when false alternatives are structurally harder to compress. We test this using small GPT-2-style character-level transformers (3.5M--86M parameters) on synthetic math corpora with controlled mixtures of correct and incorrect rules. In the random-error setting, models strongly prefer correct completions in paired evaluation: 83.1% accuracy at balanced data and 67.0% even when correct rules appear in only 10% of the corpus. Replacing random errors with a coherent but mathematically incorrect rule system largely eliminates the preference (near-chance accuracy). In a more natural-language-like synthetic world, the effect is weaker but still present (57.7%). Additional experiments show that embedding verification steps can restore preference for correctness even at small scale, while increasing the number of consistent rules produces a graded improvement in accuracy. Our results suggest that what appears as a "truth bias" is largely a side effect of compression pressure and preference for internal consistency, rather than an intrinsic drive toward truth. Full code and data are available at this https URL. 

---
# Semi-Synthetic Parallel Data for Translation Quality Estimation: A Case Study of Dataset Building for an Under-Resourced Language Pair 

**Authors**: Assaf Siani, Anna Kernerman, Ilan Kernerman  

**Link**: [PDF](https://arxiv.org/pdf/2603.11743)  

**Abstract**: Quality estimation (QE) plays a crucial role in machine translation (MT) workflows, as it serves to evaluate generated outputs that have no reference translations and to determine whether human post-editing or full retranslation is necessary. Yet, developing highly accurate, adaptable and reliable QE systems for under-resourced language pairs remains largely unsolved, due mainly to limited parallel corpora and to diverse language-dependent factors, such as with morphosyntactically complex languages. This study presents a semi-synthetic parallel dataset for English-to-Hebrew QE, generated by creating English sentences based on examples of usage that illustrate typical linguistic patterns, translating them to Hebrew using multiple MT engines, and filtering outputs via BLEU-based selection. Each translated segment was manually evaluated and scored by a linguist, and we also incorporated professionally translated English-Hebrew segments from our own resources, which were assigned the highest quality score. Controlled translation errors were introduced to address linguistic challenges, particularly regarding gender and number agreement, and we trained neural QE models, including BERT and XLM-R, on this dataset to assess sentence-level MT quality. Our findings highlight the impact of dataset size, distributed balance, and error distribution on model performance. We will describe the challenges, methodology and results of our experiments, and specify future directions aimed at improving QE performance. This research contributes to advancing QE models for under resourced language pairs, including morphology-rich languages. 

---
# SemBench: A Universal Semantic Framework for LLM Evaluation 

**Authors**: Mikel Zubillaga, Naiara Perez, Oscar Sainz, German Rigau  

**Link**: [PDF](https://arxiv.org/pdf/2603.11687)  

**Abstract**: Recent progress in Natural Language Processing (NLP) has been driven by the emergence of Large Language Models (LLMs), which exhibit remarkable generative and reasoning capabilities. However, despite their success, evaluating the true semantic understanding of these models remains a persistent challenge. Traditional benchmarks such as Word-in-Context (WiC) effectively probe this capability, but their creation is resource-intensive and often limited to high-resource languages. In this paper, we introduce SemBench, a framework for automatically generating synthetic benchmarks that assess the semantic competence of LLMs using only dictionary sense definitions and a sentence encoder. This approach eliminates the need for curated example sentences, making it both scalable and language-independent. We evaluate SemBench in three languages (English, Spanish, and Basque) spanning different levels of linguistic resources, and across a wide range of LLMs. Our results show that rankings derived from SemBench strongly correlate with those obtained from standard WiC datasets. Furthermore, our analysis demonstrates that only a small number of examples is required to achieve stable and meaningful rankings. Overall, SemBench provides a lightweight, adaptable, and data-efficient framework for cross-lingual evaluation of semantic understanding in LLMs. 

---
# In the LLM era, Word Sense Induction remains unsolved 

**Authors**: Anna Mosolova, Marie Candito, Carlos Ramisch  

**Link**: [PDF](https://arxiv.org/pdf/2603.11686)  

**Abstract**: In the absence of sense-annotated data, word sense induction (WSI) is a compelling alternative to word sense disambiguation, particularly in low-resource or domain-specific settings. In this paper, we emphasize methodological problems in current WSI evaluation. We propose an evaluation on a SemCor-derived dataset, respecting the original corpus polysemy and frequency distributions. We assess pre-trained embeddings and clustering algorithms across parts of speech, and propose and evaluate an LLM-based WSI method for English. We evaluate data augmentation sources (LLM-generated, corpus and lexicon), and semi-supervised scenarios using Wiktionary for data augmentation, must-link constraints, number of clusters per lemma.
We find that no unsupervised method (whether ours or previous) surpasses the strong "one cluster per lemma" heuristic (1cpl). We also show that (i) results and best systems may vary across POS, (ii) LLMs have troubles performing this task, (iii) data augmentation is beneficial and (iv) capitalizing on Wiktionary does help. It surpasses previous SOTA system on our test set by 3.3\%. WSI is not solved, and calls for a better articulation of lexicons and LLMs' lexical semantics capabilities. 

---
# A technology-oriented mapping of the language and translation industry: Analysing stakeholder values and their potential implication for translation pedagogy 

**Authors**: María Isabel Rivas Ginel, Janiça Hackenbuchner, Alina Secară, Ralph Krüger, Caroline Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2603.11667)  

**Abstract**: This paper examines how value is constructed and negotiated in today's increasingly automated language and translation industry. Drawing on interview data from twenty-nine industry stakeholders collected within the LT-LiDER project, the study analyses how human value, technological value, efficiency, and adaptability are articulated across different professional roles. Using Chesterman's framework of translation ethics and associated values as an analytical lens, the paper shows that efficiency-oriented technological values aligned with the ethics of service have become baseline expectations in automated production environments, where speed, scalability, and deliverability dominate evaluation criteria. At the same time, human value is not displaced but repositioned, emerging primarily through expertise, oversight, accountability, and contextual judgment embedded within technology-mediated workflows. A central finding is the prominence of adaptability as a mediating value linking human and technological domains. Adaptability is constructed as a core professional requirement, reflecting expectations that translators continuously adjust their skills, roles, and identities in response to evolving tools and organisational demands. The paper argues that automation reshapes rather than replaces translation value, creating an interdependent configuration in which technological efficiency enables human communicative work. 

---
# Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge 

**Authors**: Junjie Wu, Xuan Kan, Zihao He, Shunwen Tan, Bo Pan, Kaitai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.11665)  

**Abstract**: Multimodal Large Language Models (MLLMs) have been widely adopted as MLLM-as-a-Judges due to their strong alignment with human judgment across various visual tasks. However, most existing judge models are optimized for single-task scenarios and struggle to generalize to diverse contexts, which is a critical requirement for reliable evaluation. To address this limitation, we propose Multi-Task Reinforcement Learning for MLLM-as-a-Judge (MT-RL-Judge), a framework that jointly optimizes the judge model across multiple tasks, leveraging the generalization capabilities of RL. Experimental results against several strong baselines demonstrate that MT-RL-Judge outperforms strong baselines in both judgment consistency and correlation with human preferences. Furthermore, our approach exhibits robust generalization on out-of-distribution tasks, further validating its effectiveness. 

---
# QChunker: Learning Question-Aware Text Chunking for Domain RAG via Multi-Agent Debate 

**Authors**: Jihao Zhao, Daixuan Li, Pengfei Li, Shuaishuai Zu, Biao Qin, Hongyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.11650)  

**Abstract**: The effectiveness upper bound of retrieval-augmented generation (RAG) is fundamentally constrained by the semantic integrity and information granularity of text chunks in its knowledge base. To address these challenges, this paper proposes QChunker, which restructures the RAG paradigm from retrieval-augmentation to understanding-retrieval-augmentation. Firstly, QChunker models the text chunking as a composite task of text segmentation and knowledge completion to ensure the logical coherence and integrity of text chunks. Drawing inspiration from Hal Gregersen's "Questions Are the Answer" theory, we design a multi-agent debate framework comprising four specialized components: a question outline generator, text segmenter, integrity reviewer, and knowledge completer. This framework operates on the principle that questions serve as catalysts for profound insights. Through this pipeline, we successfully construct a high-quality dataset of 45K entries and transfer this capability to small language models. Additionally, to handle long evaluation chains and low efficiency in existing chunking evaluation methods, which overly rely on downstream QA tasks, we introduce a novel direct evaluation metric, ChunkScore. Both theoretical and experimental validations demonstrate that ChunkScore can directly and efficiently discriminate the quality of text chunks. Furthermore, during the text segmentation phase, we utilize document outlines for multi-path sampling to generate multiple candidate chunks and select the optimal solution employing ChunkScore. Extensive experimental results across four heterogeneous domains exhibit that QChunker effectively resolves aforementioned issues by providing RAG with more logically coherent and information-rich text chunks. 

---
# Performance Evaluation of Open-Source Large Language Models for Assisting Pathology Report Writing in Japanese 

**Authors**: Masataka Kawai, Singo Sakashita, Shumpei Ishikawa, Shogo Watanabe, Anna Matsuoka, Mikio Sakurai, Yasuto Fujimoto, Yoshiyuki Takahara, Atsushi Ohara, Hirohiko Miyake, Genichiro Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2603.11597)  

**Abstract**: The performance of large language models (LLMs) for supporting pathology report writing in Japanese remains unexplored. We evaluated seven open-source LLMs from three perspectives: (A) generation and information extraction of pathology diagnosis text following predefined formats, (B) correction of typographical errors in Japanese pathology reports, and (C) subjective evaluation of model-generated explanatory text by pathologists and clinicians. Thinking models and medical-specialized models showed advantages in structured reporting tasks that required reasoning and in typo correction. In contrast, preferences for explanatory outputs varied substantially across raters. Although the utility of LLMs differed by task, our findings suggest that open-source LLMs can be useful for assisting Japanese pathology report writing in limited but clinically relevant scenarios. 

---
# UtilityMax Prompting: A Formal Framework for Multi-Objective Large Language Model Optimization 

**Authors**: Ofir Marom  

**Link**: [PDF](https://arxiv.org/pdf/2603.11583)  

**Abstract**: The success of a Large Language Model (LLM) task depends heavily on its prompt. Most use-cases specify prompts using natural language, which is inherently ambiguous when multiple objectives must be simultaneously satisfied. In this paper we introduce UtilityMax Prompting, a framework that specifies tasks using formal mathematical language. We reconstruct the task as an influence diagram in which the LLM's answer is the sole decision variable. A utility function is defined over the conditional probability distributions within the diagram, and the LLM is instructed to find the answer that maximises expected utility. This constrains the LLM to reason explicitly about each component of the objective, directing its output toward a precise optimization target rather than a subjective natural language interpretation. We validate our approach on the MovieLens 1M dataset across three frontier models (Claude Sonnet 4.6, GPT-5.4, and Gemini 2.5 Pro), demonstrating consistent improvements in precision and Normalized Discounted Cumulative Gain (NDCG) over natural language baselines in a multi-objective movie recommendation task. 

---
# Streaming Translation and Transcription Through Speech-to-Text Causal Alignment 

**Authors**: Roman Koshkin, Jeon Haesung, Lianbo Liu, Hao Shi, Mengjie Zhao, Yusuke Fujita, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2603.11578)  

**Abstract**: Simultaneous machine translation (SiMT) has traditionally relied on offline machine translation models coupled with human-engineered heuristics or learned policies. We propose Hikari, a policy-free, fully end-to-end model that performs simultaneous speech-to-text translation and streaming transcription by encoding READ/WRITE decisions into a probabilistic WAIT token mechanism. We also introduce Decoder Time Dilation, a mechanism that reduces autoregressive overhead and ensures a balanced training distribution. Additionally, we present a supervised fine-tuning strategy that trains the model to recover from delays, significantly improving the quality-latency trade-off. Evaluated on English-to-Japanese, German, and Russian, Hikari achieves new state-of-the-art BLEU scores in both low- and high-latency regimes, outperforming recent baselines. 

---
# Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries 

**Authors**: Zhenxu Tian, Yi Su, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.11564)  

**Abstract**: The Key-Value (KV) cache is crucial for efficient Large Language Models (LLMs) inference, but excessively long contexts drastically increase KV cache memory footprint. Existing KV cache compression methods typically rely on input-side attention patterns within a prompt observation window to estimate token importance during the prefill stage. They fail to preserve critical tokens for future generation since these assessments are not derived from the decoding process. Intuitively, an effective observation window should mirror the decoding-stage queries to accurately reflect which tokens the generation process will attend to. However, ground-truth decoding queries are inherently unavailable during inference. For constructing pseudo queries to approximate them, we find that positional information plays a more critical role than semantic content. Motivated by this insight, we propose decoding-aligned KV cache compression via position-aware pseudo queries (DapQ), a novel and lightweight eviction framework that leverages position-aware pseudo queries to simulate the output tokens, thereby establishing an effective observation window for importance assessment. It aligns closely with the actual generation context and enables precise token eviction. Extensive evaluations across multiple benchmarks and LLMs demonstrate that DapQ achieves superior performance, particularly under strict memory constraints (e.g., up to nearly lossless performance 99.5% on NIAH with 3% KV cache budgets). 

---
# One Supervisor, Many Modalities: Adaptive Tool Orchestration for Autonomous Queries 

**Authors**: Mayank Saini Arit Kumar Bishwas  

**Link**: [PDF](https://arxiv.org/pdf/2603.11545)  

**Abstract**: We present an agentic AI framework for autonomous multimodal query processing that coordinates specialized tools across text, image, audio, video, and document modalities. A central Supervisor dynamically decomposes user queries, delegates subtasks to modality-appropriate tools (e.g., object detection, OCR, speech transcription), and synthesizes results through adaptive routing strategies rather than predetermined decision trees. For text-only queries, the framework uses learned routing via RouteLLM, while non-text paths use SLM-assisted modality decomposition. Evaluated on 2,847 queries across 15 task categories, our framework achieves 72% reduction in time-to-accurate-answer, 85% reduction in conversational rework, and 67% cost reduction compared to the matched hierarchical baseline while maintaining accuracy parity. These results demonstrate that intelligent centralized orchestration fundamentally improves multimodal AI deployment economics. 

---
# Can Small Language Models Use What They Retrieve? An Empirical Study of Retrieval Utilization Across Model Scale 

**Authors**: Sanchit Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2603.11513)  

**Abstract**: Retrieval augmented generation RAG is widely deployed to improve factual accuracy in language models yet it remains unclear whether smaller models of size 7B parameters or less can effectively utilize retrieved information. To investigate this question we evaluate five model sizes from 360M to 8B across three architecture families SmolLM2 Qwen2.5 and Llama 3.1 under four retrieval conditions including no retrieval BM25 dense retrieval using E5 large v2 and oracle retrieval where the retrieved passage is guaranteed to contain the answer. We introduce a parametric knowledge split that separates questions a model can already answer from those that require external knowledge which allows us to isolate utilization failure from retrieval quality failure. We find three main results. First even with oracle retrieval models of size 7B or smaller fail to extract the correct answer 85 to 100 percent of the time on questions they cannot answer alone which indicates a fundamental utilization bottleneck. Second adding retrieval context destroys 42 to 100 percent of answers the model previously knew suggesting a distraction effect driven by the presence of context rather than its quality. Third an error analysis of 2588 oracle failures shows that the dominant failure mode is irrelevant generation where the model ignores the provided context entirely. These patterns hold across multiple prompt templates and retrieval methods. The results indicate that for models below 7B parameters the main limitation of RAG is context utilization rather than retrieval quality and that deploying RAG at this scale can lead to a net negative trade off under standard evaluation conditions. 

---
# Tiny Aya: Bridging Scale and Multilingual Depth 

**Authors**: Alejandro R. Salamanca, Diana Abagyan, Daniel D'souza, Ammar Khairi, David Mora, Saurabh Dash, Viraat Aryabumi, Sara Rajaee, Mehrnaz Mofakhami, Ananya Sahu, Thomas Euyang, Brittawnya Prince, Madeline Smith, Hangyu Lin, Acyr Locatelli, Sara Hooker, Tom Kocmi, Aidan Gomez, Ivan Zhang, Phil Blunsom, Nick Frosst, Joelle Pineau, Beyza Ermis, Ahmet Üstün, Julia Kreutzer, Marzieh Fadaee  

**Link**: [PDF](https://arxiv.org/pdf/2603.11510)  

**Abstract**: Tiny Aya redefines what a small multilingual language model can achieve. Trained on 70 languages and refined through region-aware posttraining, it delivers state-of-the-art in translation quality, strong multilingual understanding, and high-quality target-language generation, all with just 3.35B parameters. The release includes a pretrained foundation model, a globally balanced instruction-tuned variant, and three region-specialized models targeting languages from Africa, South Asia, Europe, Asia-Pacific, and West Asia. This report details the training strategy, data composition, and comprehensive evaluation framework behind Tiny Aya, and presents an alternative scaling path for multilingual AI: one centered on efficiency, balanced performance across languages, and practical deployment. 

---
# Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Performance of LLMs 

**Authors**: Kunfeng Chen, Qihuang Zhong, Juhua Liu, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2603.11495)  

**Abstract**: Tool-calling empowers Large Language Models (LLMs) to interact with external environments. However, current methods often struggle to handle massive and noisy candidate tools in long-context tool-calling tasks, limiting their real-world application. To this end, we propose Tool-DC, a Divide-and-Conquer framework for boosting tool-calling performance of LLMs. The core of Tool-DC is to reduce the reasoning difficulty and make full use of self-reflection ability of LLMs via a "Try-Check-Retry" paradigm. Specifically, Tool-DC involves two variants: 1) the training-free Tool-DC (TF), which is plug-and-play and flexible; 2) the training-based Tool-DC (TB), which is more inference-efficient. Extensive experiments show that both Tool-DC methods outperform their counterparts by a clear margin. Tool-DC (TF) brings up to +25.10% average gains against the baseline on BFCL and ACEBench benchmarks, while Tool-DC (TB) enables Qwen2.5-7B to achieve comparable or even better performance than proprietary LLMs, e.g., OpenAI o3 and Claude-Haiku-4.5. 

---
# LLM-Assisted Causal Structure Disambiguation and Factor Extraction for Legal Judgment Prediction 

**Authors**: Yuzhi Liang, Lixiang Ma, Xinrong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.11446)  

**Abstract**: Mainstream methods for Legal Judgment Prediction (LJP) based on Pre-trained Language Models (PLMs) heavily rely on the statistical correlation between case facts and judgment results. This paradigm lacks explicit modeling of legal constituent elements and underlying causal logic, making models prone to learning spurious correlations and suffering from poor robustness. While introducing causal inference can mitigate this issue, existing causal LJP methods face two critical bottlenecks in real-world legal texts: inaccurate legal factor extraction with severe noise, and significant uncertainty in causal structure discovery due to Markov equivalence under sparse features. To address these challenges, we propose an enhanced causal inference framework that integrates Large Language Model (LLM) priors with statistical causal discovery. First, we design a coarse-to-fine hybrid extraction mechanism combining statistical sampling and LLM semantic reasoning to accurately identify and purify standard legal constituent elements. Second, to resolve structural uncertainty, we introduce an LLM-assisted causal structure disambiguation mechanism. By utilizing the LLM as a constrained prior knowledge base, we conduct probabilistic evaluation and pruning on ambiguous causal directions to generate legally compliant candidate causal graphs. Finally, a causal-aware judgment prediction model is constructed by explicitly constraining text attention intensity via the generated causal graphs. Extensive experiments on multiple benchmark datasets, including LEVEN , QA, and CAIL, demonstrate that our proposed method significantly outperforms state-of-the-art baselines in both predictive accuracy and robustness, particularly in distinguishing confusing charges. 

---
# BLooP: Zero-Shot Abstractive Summarization using Large Language Models with Bigram Lookahead Promotion 

**Authors**: Varun Iyer, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2603.11415)  

**Abstract**: Abstractive summarization requires models to generate summaries that convey information in the source document. While large language models can generate summaries without fine-tuning, they often miss key details and include extraneous information. We propose BLooP (Bigram Lookahead Promotion), a simple training-free decoding intervention that encourages large language models (LLMs) to generate tokens that form bigrams from the source document. BLooP operates through a hash table lookup at each decoding step, requiring no training, fine-tuning, or model modification. We demonstrate improvements in ROUGE and BARTScore for Llama-3.1-8B-Instruct, Mistral-Nemo-Instruct-2407, and Gemma-2-9b-it on CNN/DM, CCSum, Multi-News, and SciTLDR. Human evaluation shows that BLooP significantly improves faithfulness without reducing readability. We make the code available at this https URL 

---
# MaterialFigBENCH: benchmark dataset with figures for evaluating college-level materials science problem-solving abilities of multimodal large language models 

**Authors**: Michiko Yoshitake, Yuta Suzuki, Ryo Igarashi, Yoshitaka Ushiku, Keisuke Nagato  

**Link**: [PDF](https://arxiv.org/pdf/2603.11414)  

**Abstract**: We present MaterialFigBench, a benchmark dataset designed to evaluate the ability of multimodal large language models (LLMs) to solve university-level materials science problems that require accurate interpretation of figures. Unlike existing benchmarks that primarily rely on textual representations, MaterialFigBench focuses on problems in which figures such as phase diagrams, stress-strain curves, Arrhenius plots, diffraction patterns, and microstructural schematics are indispensable for deriving correct answers. The dataset consists of 137 free-response problems adapted from standard materials science textbooks, covering a broad range of topics including crystal structures, mechanical properties, diffusion, phase diagrams, phase transformations, and electronic properties of materials. To address unavoidable ambiguity in reading numerical values from images, expert-defined answer ranges are provided where appropriate. We evaluate several state-of-the-art multimodal LLMs, including ChatGPT and GPT models accessed via OpenAI APIs, and analyze their performance across problem categories and model versions. The results reveal that, although overall accuracy improves with model updates, current LLMs still struggle with genuine visual understanding and quantitative interpretation of materials science figures. In many cases, correct answers are obtained by relying on memorized domain knowledge rather than by reading the provided images. MaterialFigBench highlights persistent weaknesses in visual reasoning, numerical precision, and significant-digit handling, while also identifying problem types where performance has improved. This benchmark provides a systematic and domain-specific foundation for advancing multimodal reasoning capabilities in materials science and for guiding the development of future LLMs with stronger figure-based understanding. 

---
# Algorithmic Consequences of Particle Filters for Sentence Processing: Amplified Garden-Paths and Digging-In Effects 

**Authors**: Amani Maina-Kilaas, Roger Levy  

**Link**: [PDF](https://arxiv.org/pdf/2603.11412)  

**Abstract**: Under surprisal theory, linguistic representations affect processing difficulty only through the bottleneck of surprisal. Our best estimates of surprisal come from large language models, which have no explicit representation of structural ambiguity. While LLM surprisal robustly predicts reading times across languages, it systematically underpredicts difficulty when structural expectations are violated -- suggesting that representations of ambiguity are causally implicated in sentence processing. Particle filter models offer an alternative where structural hypotheses are explicitly represented as a finite set of particles. We prove several algorithmic consequences of particle filter models, including the amplification of garden-path effects. Most critically, we demonstrate that resampling, a common practice with these models, inherently produces real-time digging-in effects -- where disambiguation difficulty increases with ambiguous region length. Digging-in magnitude scales inversely with particle count: fully parallel models predict no such effect. 

---
# Stop Listening to Me! How Multi-turn Conversations Can Degrade Diagnostic Reasoning 

**Authors**: Kevin H. Guo, Chao Yan, Avinash Baidya, Katherine Brown, Xiang Gao, Juming Xiong, Zhijun Yin, Bradley A. Malin  

**Link**: [PDF](https://arxiv.org/pdf/2603.11394)  

**Abstract**: Patients and clinicians are increasingly using chatbots powered by large language models (LLMs) for healthcare inquiries. While state-of-the-art LLMs exhibit high performance on static diagnostic reasoning benchmarks, their efficacy across multi-turn conversations, which better reflect real-world usage, has been understudied. In this paper, we evaluate 17 LLMs across three clinical datasets to investigate how partitioning the decision-space into multiple simpler turns of conversation influences their diagnostic reasoning. Specifically, we develop a "stick-or-switch" evaluation framework to measure model conviction (i.e., defending a correct diagnosis or safe abstention against incorrect suggestions) and flexibility (i.e., recognizing a correct suggestion when it is introduced) across conversations. Our experiments reveal the conversation tax, where multi-turn interactions consistently degrade performance when compared to single-shot baselines. Notably, models frequently abandon initial correct diagnoses and safe abstentions to align with incorrect user suggestions. Additionally, several models exhibit blind switching, failing to distinguish between signal and incorrect suggestions. 

---
# Evaluating Explainable AI Attribution Methods in Neural Machine Translation via Attention-Guided Knowledge Distillation 

**Authors**: Aria Nourbakhsh, Salima Lamsiyah, Adelaide Danilov, Christoph Schommer  

**Link**: [PDF](https://arxiv.org/pdf/2603.11342)  

**Abstract**: The study of the attribution of input features to the output of neural network models is an active area of research. While numerous Explainable AI (XAI) techniques have been proposed to interpret these models, the systematic and automated evaluation of these methods in sequence-to-sequence (seq2seq) models is less explored. This paper introduces a new approach for evaluating explainability methods in transformer-based seq2seq models. We use teacher-derived attribution maps as a structured side signal to guide a student model, and quantify the utility of different attribution methods through the student's ability to simulate targets. Using the Inseq library, we extract attribution scores over source-target sequence pairs and inject these scores into the attention mechanism of a student transformer model under four composition operators (addition, multiplication, averaging, and replacement). Across three language pairs (de-en, fr-en, ar-en) and attributions from Marian-MT and mBART models, Attention, Value Zeroing, and Layer Gradient $\times$ Activation consistently yield the largest gains in BLEU (and corresponding improvements in chrF) relative to baselines. In contrast, other gradient-based methods (Saliency, Integrated Gradients, DeepLIFT, Input $\times$ Gradient, GradientShap) lead to smaller and less consistent improvements. These results suggest that different attribution methods capture distinct signals and that attention-derived attributions better capture alignment between source and target representations in seq2seq models. Finally, we introduce an Attributor transformer that, given a source-target pair, learns to reconstruct the teacher's attribution map. Our findings demonstrate that the more accurately the Attributor can reproduce attribution maps, the more useful an injection of those maps is for the downstream task. The source code can be found on GitHub. 

---
# Temporal Text Classification with Large Language Models 

**Authors**: Nishat Raihan, Marcos Zampieri  

**Link**: [PDF](https://arxiv.org/pdf/2603.11295)  

**Abstract**: Languages change over time. Computational models can be trained to recognize such changes enabling them to estimate the publication date of texts. Despite recent advancements in Large Language Models (LLMs), their performance on automatic dating of texts, also known as Temporal Text Classification (TTC), has not been explored. This study provides the first systematic evaluation of leading proprietary (Claude 3.5, GPT-4o, Gemini 1.5) and open-source (LLaMA 3.2, Gemma 2, Mistral, Nemotron 4) LLMs on TTC using three historical corpora, two in English and one in Portuguese. We test zero-shot and few-shot prompting, and fine-tuning settings. Our results indicate that proprietary models perform well, especially with few-shot prompting. They also indicate that fine-tuning substantially improves open-source models but that they still fail to match the performance delivered by proprietary LLMs. 

---
# ThReadMed-QA: A Multi-Turn Medical Dialogue Benchmark from Real Patient Questions 

**Authors**: Monica Munnangi, Saiph Savage  

**Link**: [PDF](https://arxiv.org/pdf/2603.11281)  

**Abstract**: Medical question-answering benchmarks predominantly evaluate single-turn exchanges, failing to capture the iterative, clarification-seeking nature of real patient consultations. We introduce ThReadMed-QA, a benchmark of 2,437 fully-answered patient-physician conversation threads extracted from r/AskDocs, comprising 8,204 question-answer pairs across up to 9 turns. Unlike prior work relying on simulated dialogues, adversarial prompts, or exam-style questions, ThReadMed-QA captures authentic patient follow-up questions and verified physician responses, reflecting how patients naturally seek medical information online. We evaluate five state-of-the-art LLMs -- GPT-5, GPT-4o, Claude Haiku, Gemini 2.5 Flash, and Llama 3.3 70B -- on a stratified test split of 238 conversations (948 QA pairs) using a calibrated LLM-as-a-judge rubric grounded in physician ground truth. Even the strongest model, GPT-5, achieves only 41.2% fully-correct responses. All five models degrade significantly from turn 0 to turn 2 (p < 0.001), with wrong-answer rates roughly tripling by the third turn. We identify a fundamental tension between single-turn capability and multi-turn reliability: models with the strongest initial performance (GPT-5: 75.2; Claude Haiku: 72.3 out of 100) exhibit the steepest declines by turn 2 (dropping 16.2 and 25.0 points respectively), while weaker models plateau or marginally improve. We introduce two metrics to quantify multi-turn failure modes: Conversational Consistency Score (CCS) and Error Propagation Rate (EPR). CCS reveals that nearly one in three Claude Haiku conversations swings between a fully correct and a completely wrong response within the same thread. EPR shows that a single wrong turn raises the probability of a subsequent wrong turn by 1.9-6.1x across all models. 

---
# Artificial Intelligence for Sentiment Analysis of Persian Poetry 

**Authors**: Arash Zargar, Abolfazl Moshiri, Mitra Shafaei, Shabnam Rahimi-Golkhandan, Mohamad Tavakoli-Targhi, Farzad Khalvati  

**Link**: [PDF](https://arxiv.org/pdf/2603.11254)  

**Abstract**: Recent advancements of the Artificial Intelligence (AI) have led to the development of large language models (LLMs) that are capable of understanding, analysing, and creating textual data. These language models open a significant opportunity in analyzing the literature and more specifically poetry. In the present work, we employ multiple Bidirectional encoder representations from transformers (BERT) and Generative Pre-trained Transformer (GPT) based language models to analyze the works of two prominent Persian poets: Jalal al-Din Muhammad Rumi (Rumi) and Parvin E'tesami. The main objective of this research is to investigate the capability of the modern language models in grasping complexities of the Persian poetry and explore potential correlations between the poems' sentiment and their meters. Our findings in this study indicates that GPT4o language model can reliably be used in analysis of Persian poetry. Furthermore, the results of our sentiment analysis revealed that in general, Rumi's poems express happier sentiments compared to Parvin E'tesami's poems. Furthermore, comparing the utilization of poetic meters highlighted Rumi's poems superiority in using meters to express a wider variety of sentiments. These findings are significant as they confirm that LLMs can be effectively applied in conducting computer-based semantic studies, where human interpretations are not required, and thereby significantly reducing potential biases in the analysis. 

---
# Markovian Generation Chains in Large Language Models 

**Authors**: Mingmeng Geng, Amr Mohamed, Guokan Shang, Michalis Vazirgiannis, Thierry Poibeau  

**Link**: [PDF](https://arxiv.org/pdf/2603.11228)  

**Abstract**: The widespread use of large language models (LLMs) raises an important question: how do texts evolve when they are repeatedly processed by LLMs? In this paper, we define this iterative inference process as Markovian generation chains, where each step takes a specific prompt template and the previous output as input, without including any prior memory. In iterative rephrasing and round-trip translation experiments, the output either converges to a small recurrent set or continues to produce novel sentences over a finite horizon. Through sentence-level Markov chain modeling and analysis of simulated data, we show that iterative process can either increase or reduce sentence diversity depending on factors such as the temperature parameter and the initial input sentence. These results offer valuable insights into the dynamics of iterative LLM inference and their implications for multi-agent LLM systems. 

---
# MDER-DR: Multi-Hop Question Answering with Entity-Centric Summaries 

**Authors**: Riccardo Campi, Nicolò Oreste Pinciroli Vago, Mathyas Giudici, Marco Brambilla, Piero Fraternali  

**Link**: [PDF](https://arxiv.org/pdf/2603.11223)  

**Abstract**: Retrieval-Augmented Generation (RAG) over Knowledge Graphs (KGs) suffers from the fact that indexing approaches may lose important contextual nuance when text is reduced to triples, thereby degrading performance in downstream Question-Answering (QA) tasks, particularly for multi-hop QA, which requires composing answers from multiple entities, facts, or relations. We propose a domain-agnostic, KG-based QA framework that covers both the indexing and retrieval/inference phases. A new indexing approach called Map-Disambiguate-Enrich-Reduce (MDER) generates context-derived triple descriptions and subsequently integrates them with entity-level summaries, thus avoiding the need for explicit traversal of edges in the graph during the QA retrieval phase. Complementing this, we introduce Decompose-Resolve (DR), a retrieval mechanism that decomposes user queries into resolvable triples and grounds them in the KG via iterative reasoning. Together, MDER and DR form an LLM-driven QA pipeline that is robust to sparse, incomplete, and complex relational data. Experiments show that on standard and domain specific benchmarks, MDER-DR achieves substantial improvements over standard RAG baselines (up to 66%), while maintaining cross-lingual robustness. Our code is available at this https URL. 

---
# DeReason: A Difficulty-Aware Curriculum Improves Decoupled SFT-then-RL Training for General Reasoning 

**Authors**: Hanxu Hu, Yuxuan Wang, Maggie Huan, Jannis Vamvas, Yinya Huang, Zhijiang Guo, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2603.11193)  

**Abstract**: Reinforcement learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for eliciting reasoning capabilities in large language models, particularly in mathematics and coding. While recent efforts have extended this paradigm to broader general scientific (STEM) domains, the complex interplay between supervised fine-tuning (SFT) and RL in these contexts remains underexplored. In this paper, we conduct controlled experiments revealing a critical challenge: for general STEM domains, RL applied directly to base models is highly sample-inefficient and is consistently surpassed by supervised fine-tuning (SFT) on moderate-quality responses. Yet sequential SFT followed by RL can further improve performance, suggesting that the two stages play complementary roles, and that how training data is allocated between them matters. Therefore, we propose DeReason, a difficulty-based data decoupling strategy for general reasoning. DeReason partitions training data by reasoning intensity estimated via LLM-based scoring into reasoning-intensive and non-reasoning-intensive subsets. It allocates broad-coverage, non-reasoning-intensive problems to SFT to establish foundational domain knowledge, and reserves a focused subset of difficult problems for RL to cultivate complex reasoning. We demonstrate that this principled decoupling yields better performance than randomly splitting the data for sequential SFT and RL. Extensive experiments on general STEM and mathematical benchmarks demonstrate that our decoupled curriculum training significantly outperforms SFT-only, RL-only, and random-split baselines. Our work provides a systematic study of the interplay between SFT and RL for general reasoning, offering a highly effective and generalized post-training recipe. 

---
# Summarize Before You Speak with ARACH: A Training-Free Inference-Time Plug-In for Enhancing LLMs via Global Attention Reallocation 

**Authors**: Jingtao Wang, Yucong Wang, Jun Ding, Rui Cai, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.11067)  

**Abstract**: Large language models (LLMs) achieve remarkable performance, yet further gains often require costly training. This has motivated growing interest in post-training techniques-especially training-free approaches that improve models at inference time without updating weights. Most training-free methods treat the model as a black box and improve outputs via input/output-level interventions, such as prompt design and test-time scaling through repeated sampling, reranking/verification, or search. In contrast, they rarely offer a plug-and-play mechanism to intervene in a model's internal computation. We propose ARACH(Attention Reallocation via an Adaptive Context Hub), a training-free inference-time plug-in that augments LLMs with an adaptive context hub to aggregate context and reallocate attention. Extensive experiments across multiple language modeling tasks show consistent improvements with modest inference overhead and no parameter updates. Attention analyses further suggest that ARACH mitigates the attention sink phenomenon. These results indicate that engineering a model's internal computation offers a distinct inference-time strategy, fundamentally different from both prompt-based test-time methods and training-based post-training approaches. 

---
# Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple 

**Authors**: Amirhossein Bozorgkhoo, Igor Molybog  

**Link**: [PDF](https://arxiv.org/pdf/2603.11053)  

**Abstract**: Speculative decoding is a technique that uses multiple language models to accelerate infer- ence. Previous works have used an experi- mental approach to optimize the throughput of the inference pipeline, which involves LLM training and can be costly. This study of spec- ulative decoding proposes a theory that ana- lytically connects the key hyperparameters of pre-trained LLMs to the throughput efficiency of a downstream SD-based inference system. The theory allows the prediction of throughput- optimal hyperparameters for the components of an inference system before their pre-training. 

---
# EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models 

**Authors**: Xuanlang Dai, Yujie Zhou, Long Xing, Jiazi Bu, Xilin Wei, Yuhong Liu, Beichen Zhang, Kai Chen, Yuhang Zang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12252)  

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have been widely integrated into diffusion frameworks primarily as text encoders to tackle complex tasks such as spatial reasoning. However, this paradigm suffers from two critical limitations: (i) MLLMs text encoder exhibits insufficient reasoning depth. Single-step encoding fails to activate the Chain-of-Thought process, which is essential for MLLMs to provide accurate guidance for complex tasks. (ii) The guidance remains invariant during the decoding process. Invariant guidance during decoding prevents DiT from progressively decomposing complex instructions into actionable denoising steps, even with correct MLLM encodings. To this end, we propose Endogenous Chain-of-Thought (EndoCoT), a novel framework that first activates MLLMs' reasoning potential by iteratively refining latent thought states through an iterative thought guidance module, and then bridges these states to the DiT's denoising process. Second, a terminal thought grounding module is applied to ensure the reasoning trajectory remains grounded in textual supervision by aligning the final state with ground-truth answers. With these two components, the MLLM text encoder delivers meticulously reasoned guidance, enabling the DiT to execute it progressively and ultimately solve complex tasks in a step-by-step manner. Extensive evaluations across diverse benchmarks (e.g., Maze, TSP, VSP, and Sudoku) achieve an average accuracy of 92.1%, outperforming the strongest baseline by 8.3 percentage points. 

---
# Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training 

**Authors**: Yixin Liu, Yue Yu, DiJia Su, Sid Wang, Xuewei Wang, Song Jiang, Bo Liu, Arman Cohan, Yuandong Tian, Zhengxing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.12246)  

**Abstract**: Reasoning LLMs-as-Judges, which can benefit from inference-time scaling, provide a promising path for extending the success of reasoning models to non-verifiable domains where the output correctness/quality cannot be directly checked. However, while reasoning judges have shown better performance on static evaluation benchmarks, their effectiveness in actual policy training has not been systematically examined. Therefore, we conduct a rigorous study to investigate the actual impact of non-reasoning and reasoning judges in reinforcement-learning-based LLM alignment. Our controlled synthetic setting, where a "gold-standard" judge (gpt-oss-120b) provides preference annotations to train smaller judges, reveals key differences between non-reasoning and reasoning judges: non-reasoning judges lead to reward hacking easily, while reasoning judges can lead to policies that achieve strong performance when evaluated by the gold-standard judge. Interestingly, we find that the reasoning-judge-trained policies achieve such strong performance by learning to generate highly effective adversarial outputs that can also score well on popular benchmarks such as Arena-Hard by deceiving other LLM-judges. Combined with our further analysis, our study highlights both important findings and room for improvements for applying (reasoning) LLM-judges in non-verifiable LLM post-training. 

---
# Linking Perception, Confidence and Accuracy in MLLMs 

**Authors**: Yuetian Du, Yucheng Wang, Rongyu Zhang, Zhijie Xu, Boyu Yang, Ming Kong, Jie Liu, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12149)  

**Abstract**: Recent advances in Multi-modal Large Language Models (MLLMs) have predominantly focused on enhancing visual perception to improve accuracy. However, a critical question remains unexplored: Do models know when they do not know? Through a probing experiment, we reveal a severe confidence miscalibration problem in MLLMs. To address this, we propose Confidence-Driven Reinforcement Learning (CDRL), which uses original-noise image pairs and a novel confidence-based reward to enhance perceptual sensitivity and robustly calibrate the model's confidence. Beyond training benefits, calibrated confidence enables more effective test-time scaling as a free lunch. We further propose Confidence-Aware Test-Time Scaling (CA-TTS), which dynamically coordinates Self-Consistency, Self-Reflection, and Visual Self-Check modules guided by confidence signals. An Expert Model acts in multiple roles (e.g., Planner, Critic, Voter) to schedule these modules and provide external verification. Our integrated framework establishes new state-of-the-art results with consistent 8.8% gains across four benchmarks. More ablation studies demonstrate the effectiveness of each module and scaling superiority. 

---
# TopoBench: Benchmarking LLMs on Hard Topological Reasoning 

**Authors**: Mayug Maniparambil, Nils Hoehing, Janak Kapuriya, Arjun Karuvally, Ellen Rushe, Anthony Ventresque, Noel O'Connor, Fergal Reid  

**Link**: [PDF](https://arxiv.org/pdf/2603.12133)  

**Abstract**: Solving topological grid puzzles requires reasoning over global spatial invariants such as connectivity, loop closure, and region symmetry and remains challenging for even the most powerful large language models (LLMs). To study these abilities under controlled settings, we introduce TopoBench, a benchmark of six puzzle families across three difficulty levels. We evaluate strong reasoning LLMs on TopoBench and find that even frontier models solve fewer than one quarter of hard instances, with two families nearly unsolved. To investigate whether these failures stem from reasoning limitations or from difficulty extracting and maintaining spatial constraints, we annotate 750 chain of thought traces with an error taxonomy that surfaces four candidate causal failure modes, then test them with targeted interventions simulating each error type. These interventions show that certain error patterns like premature commitment and constraint forgetting have a direct impact on the ability to solve the puzzle, while repeated reasoning is a benign effect of search. Finally we study mitigation strategies including prompt guidance, cell-aligned grid representations and tool-based constraint checking, finding that the bottleneck lies in extracting constraints from spatial representations and not in reasoning over them. Code and data are available at this http URL. 

---
# Human-Centred LLM Privacy Audits: Findings and Frictions 

**Authors**: Dimitri Staufer, Kirsten Morehouse, David Hartmann, Bettina Berendt  

**Link**: [PDF](https://arxiv.org/pdf/2603.12094)  

**Abstract**: Large language models (LLMs) learn statistical associations from massive training corpora and user interactions, and deployed systems can surface or infer information about individuals. Yet people lack practical ways to inspect what a model associates with their name. We report interim findings from an ongoing study and introduce LMP2, a browser-based self-audit tool. In two user studies ($N_{total}{=}458$), GPT-4o predicts 11 of 50 features for everyday people with $\ge$60\% accuracy, and participants report wanting control over LLM-generated associations despite not considering all outputs privacy violations. To validate our probing method, we evaluate eight LLMs on public figures and non-existent names, observing clear separation between stable name-conditioned associations and model defaults. Our findings also contribute to exposing a broader generative AI evaluation crisis: when outputs are probabilistic, context-dependent, and user-mediated through elicitation, what model--individual associations even include is under-specified and operationalisation relies on crafting probes and metrics that are hard to validate or compare. To move towards reliable, actionable human-centred LLM privacy audits, we identify nine frictions that emerged in our study and offer recommendations for future work and the design of human-centred LLM privacy audits. 

---
# XSkill: Continual Learning from Experience and Skills in Multimodal Agents 

**Authors**: Guanyu Jiang, Zhaochen Su, Xiaoye Qu, Yi R., Fung  

**Link**: [PDF](https://arxiv.org/pdf/2603.12056)  

**Abstract**: Multimodal agents can now tackle complex reasoning tasks with diverse tools, yet they still suffer from inefficient tool use and inflexible orchestration in open-ended settings. A central challenge is enabling such agents to continually improve without parameter updates by learning from past trajectories. We identify two complementary forms of reusable knowledge essential for this goal: experiences, providing concise action-level guidance for tool selection and decision making, and skills, providing structured task-level guidance for planning and tool use. To this end, we propose XSkill, a dual-stream framework for continual learning from experience and skills in multimodal agents. XSkill grounds both knowledge extraction and retrieval in visual observations. During accumulation, XSkill distills and consolidates experiences and skills from multi-path rollouts via visually grounded summarization and cross-rollout critique. During inference, it retrieves and adapts this knowledge to the current visual context and feeds usage history back into accumulation to form a continual learning loop. Evaluated on five benchmarks across diverse domains with four backbone models, XSkill consistently and substantially outperforms both tool-only and learning-based baselines. Further analysis reveals that the two knowledge streams play complementary roles in influencing the reasoning behaviors of agents and show superior zero-shot generalization. 

---
# Resurfacing Paralinguistic Awareness in Large Audio Language Models 

**Authors**: Hao Yang, Minghan Wang, Tongtong Wu, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2603.11947)  

**Abstract**: Large Audio Language Models (LALMs) have expanded the interaction with human to speech modality, which introduces great interactive potential, due to the paralinguistic cues implicitly indicating the user context. However, building on the current content-centred paradigm, LALMs usually neglect such paralinguistic cues and respond solely based on query content. In this work, to resurface the paralinguistic awareness in LALMs, we introduce five diverse layer-wise analyses to jointly identify paralinguistic layers and semantic understanding layers. Based on these insights, we propose a paralinguistic-enhanced fine-tuning (PE-FT) protocol accordingly to equip LALMs with paralinguistic-aware capabilities, including (1) selective-layer fine-tuning, and (2) an auxiliary dual-level classification head. Our experiments demonstrate that PE-FT protocol efficiently and effectively resurfaces the paralinguistic awareness, even surpassing the performance of the all-layer fine-tuning strategy. 

---
# Chem4DLLM: 4D Multimodal LLMs for Chemical Dynamics Understanding 

**Authors**: Xinyu Li, Zhen Zhang, Qi Chen, Anton van den Hengel, Lina Yao, Javen Qinfeng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.11924)  

**Abstract**: Existing chemical understanding tasks primarily rely on static molecular representations, limiting their ability to model inherently dynamic phenomena such as bond breaking or conformational changes, which are essential for a chemist to understand chemical reactions. To address this gap, we introduce Chemical Dynamics Understanding (ChemDU), a new task that translates 4D molecular trajectories into interpretable natural-language explanations. ChemDU focuses on fundamental dynamic scenarios, including gas-phase and catalytic reactions, and requires models to reason about key events along molecular trajectories, such as bond formation and dissociation, and to generate coherent, mechanistically grounded narratives. To benchmark this capability, we construct Chem4DBench, the first dataset pairing 4D molecular trajectories with expert-authored explanations across these settings. We further propose Chem4DLLM, a unified model that integrates an equivariant graph encoder with a pretrained large language model to explicitly capture molecular geometry and rotational dynamics. We hope that ChemDU, together with Chem4DBench and Chem4DLLM, will stimulate further research in dynamic chemical understanding and multimodal scientific reasoning. 

---
# Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models 

**Authors**: Lu Wang, Zhuoran Jin, Yupu Hao, Yubo Chen, Kang Liu, Yulong Ao, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.11896)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong performance on offline video understanding, but most are limited to offline inference or have weak online reasoning, making multi-turn interaction over continuously arriving video streams difficult. Existing streaming methods typically use an interleaved perception-generation paradigm, which prevents concurrent perception and generation and leads to early memory decay as streams grow, hurting long-range dependency modeling. We propose Think While Watching, a memory-anchored streaming video reasoning framework that preserves continuous segment-level memory during multi-turn interaction. We build a three-stage, multi-round chain-of-thought dataset and adopt a stage-matched training strategy, while enforcing strict causality through a segment-level streaming causal mask and streaming positional encoding. During inference, we introduce an efficient pipeline that overlaps watching and thinking and adaptively selects the best attention backend. Under both single-round and multi-round streaming input protocols, our method achieves strong results. Built on Qwen3-VL, it improves single-round accuracy by 2.6% on StreamingBench and by 3.79% on OVO-Bench. In the multi-round setting, it maintains performance while reducing output tokens by 56%. Code is available at: this https URL 

---
# From Debate to Deliberation: Structured Collective Reasoning with Typed Epistemic Acts 

**Authors**: Sunil Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2603.11781)  

**Abstract**: Multi-agent LLM systems increasingly tackle complex reasoning, yet their interaction patterns remain limited to voting, unstructured debate, or pipeline orchestration. None model deliberation: a phased process where differentiated participants exchange typed reasoning moves, preserve disagreements, and converge on accountable outcomes. We introduce Deliberative Collective Intelligence (DCI), specifying four reasoning archetypes, 14 typed epistemic acts, a shared workspace, and DCI-CF, a convergent flow algorithm that guarantees termination with a structured decision packet containing the selected option, residual objections, minority report, and reopen conditions. We evaluate on 45 tasks across seven domains using Gemini 2.5 Flash. On non-routine tasks (n=40), DCI significantly improves over unstructured debate (+0.95, 95% CI [+0.41, +1.54]). DCI excels on hidden-profile tasks requiring perspective integration (9.56, highest of any system on any domain) while failing on routine decisions (5.39), confirming task-dependence. DCI produces 100% structured decision packets and 98% minority reports, artifacts absent from all baselines. However, DCI consumes ~62x single-agent tokens, and single-agent generation outperforms DCI on overall quality. DCI's contribution is not that more agents are better, but that consequential decisions benefit from deliberative structure when process accountability justifies the cost. 

---
# An Automatic Text Classification Method Based on Hierarchical Taxonomies, Neural Networks and Document Embedding: The NETHIC Tool 

**Authors**: Luigi Lomasto, Rosario Di Florio, Andrea Ciapetti, Giuseppe Miscione, Giulia Ruggiero, Daniele Toti  

**Link**: [PDF](https://arxiv.org/pdf/2603.11770)  

**Abstract**: This work describes an automatic text classification method implemented in a software tool called NETHIC, which takes advantage of the inner capabilities of highly-scalable neural networks combined with the expressiveness of hierarchical taxonomies. As such, NETHIC succeeds in bringing about a mechanism for text classification that proves to be significantly effective as well as efficient. The tool had undergone an experimentation process against both a generic and a domain-specific corpus, outputting promising results. On the basis of this experimentation, NETHIC has been now further refined and extended by adding a document embedding mechanism, which has shown improvements in terms of performance on the individual networks and on the whole hierarchical model. 

---
# OSCBench: Benchmarking Object State Change in Text-to-Video Generation 

**Authors**: Xianjing Han, Bin Zhu, Shiqi Hu, Franklin Mingzhe Li, Patrick Carrington, Roger Zimmermann, Jingjing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.11698)  

**Abstract**: Text-to-video (T2V) generation models have made rapid progress in producing visually high-quality and temporally coherent videos. However, existing benchmarks primarily focus on perceptual quality, text-video alignment, or physical plausibility, leaving a critical aspect of action understanding largely unexplored: object state change (OSC) explicitly specified in the text prompt. OSC refers to the transformation of an object's state induced by an action, such as peeling a potato or slicing a lemon. In this paper, we introduce OSCBench, a benchmark specifically designed to assess OSC performance in T2V models. OSCBench is constructed from instructional cooking data and systematically organizes action-object interactions into regular, novel, and compositional scenarios to probe both in-distribution performance and generalization. We evaluate six representative open-source and proprietary T2V models using both human user study and multimodal large language model (MLLM)-based automatic evaluation. Our results show that, despite strong performance on semantic and scene alignment, current T2V models consistently struggle with accurate and temporally consistent object state changes, especially in novel and compositional settings. These findings position OSC as a key bottleneck in text-to-video generation and establish OSCBench as a diagnostic benchmark for advancing state-aware video generation models. 

---
# From Control to Foresight: Simulation as a New Paradigm for Human-Agent Collaboration 

**Authors**: Gaole He, Brian Y. Lim  

**Link**: [PDF](https://arxiv.org/pdf/2603.11677)  

**Abstract**: Large Language Models (LLMs) are increasingly used to power autonomous agents for complex, multi-step tasks. However, human-agent interaction remains pointwise and reactive: users approve or correct individual actions to mitigate immediate risks, without visibility into subsequent consequences. This forces users to mentally simulate long-term effects, a cognitively demanding and often inaccurate process. Users have control over individual steps but lack the foresight to make informed decisions. We argue that effective collaboration requires foresight, not just control. We propose simulation-in-the-loop, an interaction paradigm that enables users and agents to explore simulated future trajectories before committing to decisions. Simulation transforms intervention from reactive guesswork into informed exploration, while helping users discover latent constraints and preferences along the way. This perspective paper characterizes the limitations of current paradigms, introduces a conceptual framework for simulation-based collaboration, and illustrates its potential through concrete human-agent collaboration scenarios. 

---
# Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE 

**Authors**: Mohammad Aflah Khan, Krishna P. Gummadi, Manish Gupta, Abhilasha Ravichander  

**Link**: [PDF](https://arxiv.org/pdf/2603.11611)  

**Abstract**: Rotary Positional Embedding (RoPE) is a common choice in transformer architectures for encoding relative positional information. Although earlier work has examined omitting RoPE in specific layers, the effect of varying the fraction of hidden dimensions that receive rotary transformations remains largely unexplored. This design choice can yield substantial memory savings, which becomes especially significant at long context lengths. We find up to 10x memory savings over the standard RoPE cache, while achieving comparable final loss. In this work, we present a systematic study examining the impact of partial RoPE on training dynamics and convergence across architectures and datasets. Our findings uncover several notable patterns: (1) applying RoPE to only a small fraction of dimensions (around 10%) achieves convergence comparable to using full RoPE; (2) these trends hold consistently across model size, sequence lengths and datasets of varying quality and architectures, with higher-quality data resulting in lower overall loss and similar benchmark performance; and (3) some models trained with NoPE (No Positional Encoding) showcase unstable learning trajectories, which can be alleviated through minimal RoPE application or QK-Norm which converges to a higher loss. Together, these results offer practical guidance for model designers aiming to balance efficiency and training stability, while emphasizing the previously overlooked importance of partial RoPE. 

---
# Expert Threshold Routing for Autoregressive Language Modeling with Dynamic Computation Allocation and Load Balancing 

**Authors**: Hanchi Sun, Yixin Liu, Yonghui Wu, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.11535)  

**Abstract**: Token-choice Mixture-of-Experts (TC-MoE) routes each token to a fixed number of experts, limiting dynamic computation allocation and requiring auxiliary losses to maintain load balance. We propose Expert Threshold (ET) routing, where each expert maintains an exponential moving average (EMA) threshold estimated from the global token distribution. At both training and inference, each token is independently routed to an expert if its score exceeds the expert's threshold, enabling dynamic computation allocation while achieving load balance without auxiliary losses. This fully causal mechanism eliminates dependence on other tokens in the batch, making it well-suited for autoregressive language modeling. In pretraining experiments scaling to 2.4B parameters on FineWeb-Edu, ET achieves 0.067 lower cross-entropy loss than TC-MoE, equivalent to reaching the same performance with 1.6$\times$ fewer tokens. 

---
# LongFlow: Efficient KV Cache Compression for Reasoning M 

**Authors**: Yi Su, Zhenxu Tian, Dan Qiao, Yuechi Zhou, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.11504)  

**Abstract**: Recent reasoning models such as OpenAI-o1 and DeepSeek-R1 have shown strong performance on complex tasks including mathematical reasoning and code generation. However, this performance gain comes with substantially longer output sequences, leading to significantly increased deployment costs. In particular, long outputs require large KV caches, resulting in high memory consumption and severe bandwidth pressure during attention computation. Most existing KV cache optimization methods are designed for long-input, short-output scenarios and are ineffective for the long-output setting of reasoning models. Moreover, importance estimation in prior work is computationally expensive and becomes prohibitive when continuous re-evaluation is required during long generation. To address these challenges, we propose LongFlow, a KV cache compression method with an efficient importance estimation metric derived from an intermediate result of attention computation using only the current query. This design introduces negligible computational overhead and requires no auxiliary storage. We further develop a custom kernel that fuses FlashAttention, importance estimation, and token eviction into a single optimized operator, improving system-level efficiency. Experiments show that LongFlow achieves up to an 11.8 times throughput improvement with 80% KV cache compression with minimal impact on model accuracy. 

---
# AnimeScore: A Preference-Based Dataset and Framework for Evaluating Anime-Like Speech Style 

**Authors**: Joonyong Park, Jerry Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.11482)  

**Abstract**: Evaluating 'anime-like' voices currently relies on costly subjective judgments, yet no standardized objective metric exists. A key challenge is that anime-likeness, unlike naturalness, lacks a shared absolute scale, making conventional Mean Opinion Score (MOS) protocols unreliable. To address this gap, we propose AnimeScore, a preference-based framework for automatic anime-likeness evaluation via pairwise ranking. We collect 15,000 pairwise judgments from 187 evaluators with free-form descriptions, and acoustic analysis reveals that perceived anime-likeness is driven by controlled resonance shaping, prosodic continuity, and deliberate articulation rather than simple heuristics such as high pitch. We show that handcrafted acoustic features reach a 69.3% AUC ceiling, while SSL-based ranking models achieve up to 90.8% AUC, providing a practical metric that can also serve as a reward signal for preference-based optimization of generative speech models. 

---
# Speak or Stay Silent: Context-Aware Turn-Taking in Multi-Party Dialogue 

**Authors**: Kratika Bhagtani, Mrinal Anand, Yu Chen Xu, Amit Kumar Singh Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2603.11409)  

**Abstract**: Existing voice AI assistants treat every detected pause as an invitation to speak. This works in dyadic dialogue, but in multi-party settings, where an AI assistant participates alongside multiple speakers, pauses are abundant and ambiguous. An assistant that speaks on every pause becomes disruptive rather than useful. In this work, we formulate context-aware turn-taking: at every detected pause, given the full conversation context, our method decides whether the assistant should speak or stay silent. We introduce a benchmark of over 120K labeled conversations spanning three multi-party corpora. Evaluating eight recent large language models, we find that they consistently fail at context-aware turn-taking under zero-shot prompting. We then propose a supervised fine-tuning approach with reasoning traces, improving balanced accuracy by up to 23 percentage points. Our findings suggest that context-aware turn-taking is not an emergent capability; it must be explicitly trained. 

---
# Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction 

**Authors**: Dehao Dai, Ding Ma, Dou Liu, Kerui Geng, Yiqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.11408)  

**Abstract**: Forecasting crude oil prices remains challenging because market-relevant information is embedded in large volumes of unstructured news and is not fully captured by traditional polarity-based sentiment measures. This paper examines whether multi-dimensional sentiment signals extracted by large language models improve the prediction of weekly WTI crude oil futures returns. Using energy-sector news articles from 2020 to 2025, we construct five sentiment dimensions covering relevance, polarity, intensity, uncertainty, and forwardness based on GPT-4o, Llama 3.2-3b, and two benchmark models, FinBERT and AlphaVantage. We aggregate article-level signals to the weekly level and evaluate their predictive performance in a classification framework. The best results are achieved by combining GPT-4o and FinBERT, suggesting that LLM-based and conventional financial sentiment models provide complementary predictive information. SHAP analysis further shows that intensity- and uncertainty-related features are among the most important predictors, indicating that the predictive value of news sentiment extends beyond simple polarity. Overall, the results suggest that multi-dimensional LLM-based sentiment measures can improve commodity return forecasting and support energy-market risk monitoring. 

---
# Meta-Reinforcement Learning with Self-Reflection for Agentic Search 

**Authors**: Teng Xiao, Yige Yuan, Hamish Ivison, Huaisheng Zhu, Faeze Brahman, Nathan Lambert, Pradeep Dasigi, Noah A. Smith, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2603.11327)  

**Abstract**: This paper introduces MR-Search, an in-context meta reinforcement learning (RL) formulation for agentic search with self-reflection. Instead of optimizing a policy within a single independent episode with sparse rewards, MR-Search trains a policy that conditions on past episodes and adapts its search strategy across episodes. MR-Search learns to learn a search strategy with self-reflection, allowing search agents to improve in-context exploration at test-time. Specifically, MR-Search performs cross-episode exploration by generating explicit self-reflections after each episode and leveraging them as additional context to guide subsequent attempts, thereby promoting more effective exploration during test-time. We further introduce a multi-turn RL algorithm that estimates a dense relative advantage at the turn level, enabling fine-grained credit assignment on each episode. Empirical results across various benchmarks demonstrate the advantages of MR-Search over baselines based RL, showing strong generalization and relative improvements of 9.2% to 19.3% across eight benchmarks. Our code and data are available at this https URL. 

---
# Hindsight-Anchored Policy Optimization: Turning Failure into Feedback in Sparse Reward Settings 

**Authors**: Yuning Wu, Ke Wang, Devin Chen, Kai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.11321)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for post-training reasoning models. However, group-based methods such as Group Relative Policy Optimization (GRPO) face a critical dilemma in sparse-reward settings: pure Reinforcement Learning (RL) suffers from advantage collapse and high-variance gradient estimation, while mixed-policy optimization introduces persistent distributional bias. To resolve this dilemma, we introduce Hindsight-Anchored Policy Optimization (HAPO). HAPO employs the Synthetic Success Injection (SSI) operator, a hindsight mechanism that selectively anchors optimization to teacher demonstrations during failure. This injection is governed by a Thompson sampling-inspired gating mechanism, creating an autonomous, self-paced curriculum. Theoretically, we demonstrate that HAPO achieves \textit{asymptotic consistency}: by naturally annealing the teacher signal as the policy improves, HAPO recovers the unbiased on-policy gradient. This ensures off-policy guidance acts as a temporary scaffold rather than a persistent ceiling, enabling the model to surpass the limitations of static teacher forcing. 

---
# LLMs Can Infer Political Alignment from Online Conversations 

**Authors**: Byunghwee Lee, Sangyeon Kim, Filippo Menczer, Yong-Yeol Ahn, Haewoon Kwak, Jisun An  

**Link**: [PDF](https://arxiv.org/pdf/2603.11253)  

**Abstract**: Due to the correlational structure in our traits such as identities, cultures, and political attitudes, seemingly innocuous preferences such as following a band or using a specific slang, can reveal private traits. This possibility, especially when combined with massive, public social data and advanced computational methods, poses a fundamental privacy risk. Given our increasing data exposure online and the rapid advancement of AI are increasing the misuse potential of such risk, it is therefore critical to understand capacity of large language models (LLMs) to exploit it. Here, using online discussions on this http URL and Reddit, we show that LLMs can reliably infer hidden political alignment, significantly outperforming traditional machine learning models. Prediction accuracy further improves as we aggregate multiple text-level inferences into a user-level prediction, and as we use more politics-adjacent domains. We demonstrate that LLMs leverage the words that can be highly predictive of political alignment while not being explicitly political. Our findings underscore the capacity and risks of LLMs for exploiting socio-cultural correlates. 

---
# Frequency-Modulated Visual Restoration for Matryoshka Large Multimodal Models 

**Authors**: Qingtao Pan, Zhihao Dou, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.11220)  

**Abstract**: Large Multimodal Models (LMMs) struggle to adapt varying computational budgets due to numerous visual tokens. Previous methods attempted to reduce the number of visual tokens before or within LLMs. However, these strategies inevitably result in the loss of visual semantic. To address these issues, we introduce FMVR, a plug-and-play and extremely simple Frequency-Modulated Visual Restoration strategy to boost the reasoning ability of LMMs under visual token reduction. Specifically, FMVR disentangles the visual representation of fewer visual tokens into low- and high-frequency components through AvgPool and MaxPool. The derived frequencies are subsequently modulated using lightweight learnable parameters. The high-frequency from AvgPool acts as a saliency filter to enhance saliency visual semantics, while the low-frequency from MaxPool acts as an anti-saliency filter to strengthen weak visual semantics. It enables the preservation of visual semantics dominated by few visual tokens and the restoration of diluted visual semantics. Additionally, we inject FMVR into Matryoshka Representation Learning to learn coarse-to-fine visual token sets, thus enabling to elastically adjust the number of visual tokens during inference while maintaining comparable performance. Experiments across 10 image-based and 4 video-based bench marks demonstrate that FMVR-LLaVA reduce the FLOPs of LLaVA-1.5-7B by 89%, while maintaining almost 100% of the original accuracy. The code will be open. 

---
# Huntington Disease Automatic Speech Recognition with Biomarker Supervision 

**Authors**: Charles L. Wang, Cady Chen, Ziwei Gong, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2603.11168)  

**Abstract**: Automatic speech recognition (ASR) for pathological speech remains underexplored, especially for Huntington's disease (HD), where irregular timing, unstable phonation, and articulatory distortion challenge current models. We present a systematic HD-ASR study using a high-fidelity clinical speech corpus not previously used for end-to-end ASR training. We compare multiple ASR families under a unified evaluation, analyzing WER as well as substitution, deletion, and insertion patterns. HD speech induces architecture-specific error regimes, with Parakeet-TDT outperforming encoder-decoder and CTC baselines. HD-specific adaptation reduces WER from 6.99% to 4.95% and we also propose a method for using biomarker-based auxiliary supervision and analyze how error behavior is reshaped in severity-dependent ways rather than uniformly improving WER. We open-source all code and models. 

---
# Scaling Reasoning Efficiently via Relaxed On-Policy Distillation 

**Authors**: Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, Pashmina Cameron  

**Link**: [PDF](https://arxiv.org/pdf/2603.11137)  

**Abstract**: On-policy distillation is pivotal for transferring reasoning capabilities to capacity-constrained models, yet remains prone to instability and negative transfer. We show that on-policy distillation can be interpreted, both theoretically and empirically, as a form of policy optimization, where the teacher-student log-likelihood ratio acts as a token reward. From this insight, we introduce REOPOLD (Relaxed On-Policy Distillation) a framework that stabilizes optimization by relaxing the strict imitation constraints of standard on-policy distillation. Specifically, REOPOLD temperately and selectively leverages rewards from the teacher through mixture-based reward clipping, entropy-based token-level dynamic sampling, and a unified exploration-to-refinement training strategy. Empirically, REOPOLD surpasses its baselines with superior sample efficiency during training and enhanced test-time scaling at inference, across mathematical, visual, and agentic tool-use reasoning tasks. Specifically, REOPOLD outperforms recent RL approaches achieving 6.7~12x greater sample efficiency and enables a 7B student to match a 32B teacher in visual reasoning with a ~3.32x inference speedup. 

---
# Enhancing Value Alignment of LLMs with Multi-agent system and Combinatorial Fusion 

**Authors**: Yuanhong Wu, Djallel Bouneffouf, D. Frank Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2603.11126)  

**Abstract**: Aligning large language models (LLMs) with human values is a central challenge for ensuring trustworthy and safe deployment. While existing methods such as Reinforcement Learning from Human Feedback (RLHF) and its variants have improved alignment, they often rely on a single evaluator or narrowly defined reward signals, limiting their ability to capture ethical pluralism. In this work, we propose the Value Alignment System using Combinatorial Fusion Analysis (VAS-CFA), a framework that operationalizes multi-agent fusion alignment. It instantiates multiple moral agents, each fine-tuned to represent a distinct normative perspective, and fuses their outputs using CFA with both rank- and score-based aggregation. This design leverages cognitive diversity, between agents, to mitigate conflicts and redundancies across multiple agents, producing responses that better reflect human values. Empirical evaluation demonstrates that VAS-CFA outperforms both single agent baselines and prior aggregation approaches on standard metrics, showing that multi-agent fusion provides a robust and effective mechanism for advancing value alignment in LLMs. 

---
# Uni-ASR: Unified LLM-Based Architecture for Non-Streaming and Streaming Automatic Speech Recognition 

**Authors**: Yinfeng Xia, Jian Tang, Junfeng Hou, Gaopeng Xu, Haitao Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.11123)  

**Abstract**: Although the deep integration of the Automatic Speech Recognition (ASR) system with Large Language Models (LLMs) has significantly improved accuracy, the deployment of such systems in low-latency streaming scenarios remains challenging. In this paper, we propose Uni-ASR, a unified framework based on LLMs that integrates both non-streaming and streaming speech recognition capabilities. We propose a joint training paradigm that enables the system to seamlessly transition between two recognition modes without any architectural modifications. Furthermore, we introduce a context-aware training paradigm and a co-designed fallback decoding strategy, which can enhance streaming recognition accuracy without introducing additional latency. The experimental results demonstrate that Uni-ASR not only achieves competitive performance within non-streaming mode, but also demonstrates strong effectiveness in streaming scenarios under diverse latency constraints. 

---
# CR-Bench: Evaluating the Real-World Utility of AI Code Review Agents 

**Authors**: Kristen Pereira, Neelabh Sinha, Rajat Ghosh, Debojyoti Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2603.11078)  

**Abstract**: Recent advances in frontier large language models have enabled code review agents that operate in open-ended, reasoning-intensive settings. However, the lack of standardized benchmarks and granular evaluation protocols makes it difficult to assess behavior of code review agents beyond coarse success metrics, particularly for tasks where false positives are costly. To address this gap, we introduce CR-Bench, a benchmarking dataset, and CR-Evaluator, a fine-grained evaluation pipeline for code review agents. Using these tools, we conduct a preliminary study evaluating both a single-shot agent and a Reflexion-based agent across two frontier models. We find that code review agents can exhibit a low signal-to-noise ratio when designed to identify all hidden issues, obscuring true progress and developer productivity when measured solely by resolution rates. Our analysis identifies the hidden trade-off between issue resolution and spurious findings, revealing a frontier that constrains effective agent design. Together, CR-Bench and CR-Evaluator provide a timely foundation for studying and developing code review agents as LLM-based systems transition from controlled benchmarks to real-world software engineering workflows. 

---
# OpenSanctions Pairs: Large-Scale Entity Matching with LLMs 

**Authors**: Chandler Smith, Magnus Sesodia, Friedrich Lindenberg, Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2603.11051)  

**Abstract**: We release OpenSanctions Pairs, a large-scale entity matching benchmark derived from real-world international sanctions aggregation and analyst deduplication. The dataset contains 755,540 labeled pairs spanning 293 heterogeneous sources across 31 countries, with multilingual and cross-script names, noisy and missing attributes, and set-valued fields typical of compliance workflows. We benchmark a production rule-based matcher (nomenklatura RegressionV1 algorithm) against open- and closed-source LLMs in zero- and few-shot settings. Off-the-shelf LLMs substantially outperform the production rule-based baseline (91.33\% F1), reaching up to 98.95\% F1 (GPT-4o) and 98.23\% F1 with a locally deployable open model (DeepSeek-R1-Distill-Qwen-14B). DSPy MIPROv2 prompt optimization yields consistent but modest gains, while adding in-context examples provides little additional benefit and can degrade performance. Error analysis shows complementary failure modes: the rule-based system over-matches (high false positives), whereas LLMs primarily fail on cross-script transliteration and minor identifier/date inconsistencies. These results indicate that pairwise matching performance is approaching a practical ceiling in this setting, and motivate shifting effort toward pipeline components such as blocking, clustering, and uncertainty-aware review. Code available at this https URL 

---
