# Strategies for Span Labeling with Large Language Models 

**Authors**: Danil Semin, Ondřej Dušek, Zdeněk Kasner  

**Link**: [PDF](https://arxiv.org/pdf/2601.16946)  

**Abstract**: Large language models (LLMs) are increasingly used for text analysis tasks, such as named entity recognition or error detection. Unlike encoder-based models, however, generative architectures lack an explicit mechanism to refer to specific parts of their input. This leads to a variety of ad-hoc prompting strategies for span labeling, often with inconsistent results. In this paper, we categorize these strategies into three families: tagging the input text, indexing numerical positions of spans, and matching span content. To address the limitations of content matching, we introduce LogitMatch, a new constrained decoding method that forces the model's output to align with valid input spans. We evaluate all methods across four diverse tasks. We find that while tagging remains a robust baseline, LogitMatch improves upon competitive matching-based methods by eliminating span matching issues and outperforms other strategies in some setups. 

---
# Information Representation Fairness in Long-Document Embeddings: The Peculiar Interaction of Positional and Language Bias 

**Authors**: Elias Schuhmacher, Andrianos Michail, Juri Opitz, Rico Sennrich, Simon Clematide  

**Link**: [PDF](https://arxiv.org/pdf/2601.16934)  

**Abstract**: To be discoverable in an embedding-based search process, each part of a document should be reflected in its embedding representation. To quantify any potential reflection biases, we introduce a permutation-based evaluation framework. With this, we observe that state-of-the-art embedding models exhibit systematic positional and language biases when documents are longer and consist of multiple segments. Specifically, early segments and segments in higher-resource languages like English are over-represented, while later segments and segments in lower-resource languages are marginalized. In our further analysis, we find that the positional bias stems from front-loaded attention distributions in pooling-token embeddings, where early tokens receive more attention. To mitigate this issue, we introduce an inference-time attention calibration method that redistributes attention more evenly across document positions, increasing discoverabiltiy of later segments. Our evaluation framework and attention calibration is available at this https URL 

---
# LLM-Based Adversarial Persuasion Attacks on Fact-Checking Systems 

**Authors**: João A. Leite, Olesya Razuvayevskaya, Kalina Bontcheva, Carolina Scarton  

**Link**: [PDF](https://arxiv.org/pdf/2601.16890)  

**Abstract**: Automated fact-checking (AFC) systems are susceptible to adversarial attacks, enabling false claims to evade detection. Existing adversarial frameworks typically rely on injecting noise or altering semantics, yet no existing framework exploits the adversarial potential of persuasion techniques, which are widely used in disinformation campaigns to manipulate audiences. In this paper, we introduce a novel class of persuasive adversarial attacks on AFCs by employing a generative LLM to rephrase claims using persuasion techniques. Considering 15 techniques grouped into 6 categories, we study the effects of persuasion on both claim verification and evidence retrieval using a decoupled evaluation strategy. Experiments on the FEVER and FEVEROUS benchmarks show that persuasion attacks can substantially degrade both verification performance and evidence retrieval. Our analysis identifies persuasion techniques as a potent class of adversarial attacks, highlighting the need for more robust AFC systems. 

---
# Trapped in the past? Disentangling fluid and crystallized intelligence of large language models using chess 

**Authors**: Leonard S. Pleiss, Maximilian Schiffer, Robert K. von Weizsäcker  

**Link**: [PDF](https://arxiv.org/pdf/2601.16823)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities, yet it remains unclear to what extent these reflect sophisticated recall (crystallized intelligence) or reasoning ability (fluid intelligence). We introduce chess as a controlled testbed for disentangling these faculties. Leveraging the game's structure and scalable engine evaluations, we construct a taxonomy of positions varying in training corpus proximity--ranging from common states solvable by memorization to novel ones requiring first-principles reasoning. We systematically evaluate multiple GPT generations under varying reasoning intensities. Our analysis reveals a clear gradient: performance consistently degrades as fluid intelligence demands increase. Notably, in out-of-distribution tasks, performance collapses to random levels. While newer models improve, progress slows significantly for tasks outside the training distribution. Furthermore, while reasoning-augmented inference improves performance, its marginal benefit per token decreases with distributional proximity. These results suggest current architectures remain limited in systematic generalization, highlighting the need for mechanisms beyond scale to achieve robust fluid intelligence. 

---
# SoS: Analysis of Surface over Semantics in Multilingual Text-To-Image Generation 

**Authors**: Carolin Holtermann, Florian Schneider, Anne Lauscher  

**Link**: [PDF](https://arxiv.org/pdf/2601.16803)  

**Abstract**: Text-to-image (T2I) models are increasingly employed by users worldwide. However, prior research has pointed to the high sensitivity of T2I towards particular input languages - when faced with languages other than English (i.e., different surface forms of the same prompt), T2I models often produce culturally stereotypical depictions, prioritizing the surface over the prompt's semantics. Yet a comprehensive analysis of this behavior, which we dub Surface-over-Semantics (SoS), is missing. We present the first analysis of T2I models' SoS tendencies. To this end, we create a set of prompts covering 171 cultural identities, translated into 14 languages, and use it to prompt seven T2I models. To quantify SoS tendencies across models, languages, and cultures, we introduce a novel measure and analyze how the tendencies we identify manifest visually. We show that all but one model exhibit strong surface-level tendency in at least two languages, with this effect intensifying across the layers of T2I text encoders. Moreover, these surface tendencies frequently correlate with stereotypical visual depictions. 

---
# Large Language Models as Automatic Annotators and Annotation Adjudicators for Fine-Grained Opinion Analysis 

**Authors**: Gaurav Negi, MA Waskow, Paul Buitelaar  

**Link**: [PDF](https://arxiv.org/pdf/2601.16800)  

**Abstract**: Fine-grained opinion analysis of text provides a detailed understanding of expressed sentiments, including the addressed entity. Although this level of detail is sound, it requires considerable human effort and substantial cost to annotate opinions in datasets for training models, especially across diverse domains and real-world applications. We explore the feasibility of LLMs as automatic annotators for fine-grained opinion analysis, addressing the shortage of domain-specific labelled datasets. In this work, we use a declarative annotation pipeline. This approach reduces the variability of manual prompt engineering when using LLMs to identify fine-grained opinion spans in text. We also present a novel methodology for an LLM to adjudicate multiple labels and produce final annotations. After trialling the pipeline with models of different sizes for the Aspect Sentiment Triplet Extraction (ASTE) and Aspect-Category-Opinion-Sentiment (ACOS) analysis tasks, we show that LLMs can serve as automatic annotators and adjudicators, achieving high Inter-Annotator Agreement across individual LLM-based annotators. This reduces the cost and human effort needed to create these fine-grained opinion-annotated datasets. 

---
# Persuasion Tokens for Editing Factual Knowledge in LLMs 

**Authors**: Paul Youssef, Jörg Schlötterer, Christin Seifert  

**Link**: [PDF](https://arxiv.org/pdf/2601.16781)  

**Abstract**: In-context knowledge editing (IKE) is a promising technique for updating Large Language Models (LLMs) with new information. However, IKE relies on lengthy, fact-specific demonstrations which are costly to create and consume significant context window space. In this paper, we introduce persuasion tokens (P-Tokens) -- special tokens trained to replicate the effect of IKE demonstrations, enabling efficient knowledge editing without requiring fact-specific demonstrations. We evaluate P-Tokens across two editing datasets and three LLMs, demonstrating performance comparable to, and often exceeding, IKE. We further find that editing performance is robust to distractors with small negative effects to neighboring facts, and that increasing the number of P-Tokens improves performance. Our work addresses key limitations of IKE and provides a more practical and scalable alternative for editing LLMs. 

---
# Do LLM hallucination detectors suffer from low-resource effect? 

**Authors**: Debtanu Datta, Mohan Kishore Chilukuri, Yash Kumar, Saptarshi Ghosh, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2601.16766)  

**Abstract**: LLMs, while outperforming humans in a wide range of tasks, can still fail in unanticipated ways. We focus on two pervasive failure modes: (i) hallucinations, where models produce incorrect information about the world, and (ii) the low-resource effect, where the models show impressive performance in high-resource languages like English but the performance degrades significantly in low-resource languages like Bengali. We study the intersection of these issues and ask: do hallucination detectors suffer from the low-resource effect? We conduct experiments on five tasks across three domains (factual recall, STEM, and Humanities). Experiments with four LLMs and three hallucination detectors reveal a curious finding: As expected, the task accuracies in low-resource languages experience large drops (compared to English). However, the drop in detectors' accuracy is often several times smaller than the drop in task accuracy. Our findings suggest that even in low-resource languages, the internal mechanisms of LLMs might encode signals about their uncertainty. Further, the detectors are robust within language (even for non-English) and in multilingual setups, but not in cross-lingual settings without in-language supervision. 

---
# Standardizing Longitudinal Radiology Report Evaluation via Large Language Model Annotation 

**Authors**: Xinyi Wang, Grazziela Figueredo, Ruizhe Li, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.16753)  

**Abstract**: Longitudinal information in radiology reports refers to the sequential tracking of findings across multiple examinations over time, which is crucial for monitoring disease progression and guiding clinical decisions. Many recent automated radiology report generation methods are designed to capture longitudinal information; however, validating their performance is challenging. There is no proper tool to consistently label temporal changes in both ground-truth and model-generated texts for meaningful comparisons. Existing annotation methods are typically labor-intensive, relying on the use of manual lexicons and rules. Complex rules are closed-source, domain specific and hard to adapt, whereas overly simple ones tend to miss essential specialised information. Large language models (LLMs) offer a promising annotation alternative, as they are capable of capturing nuanced linguistic patterns and semantic similarities without extensive manual intervention. They also adapt well to new contexts. In this study, we therefore propose an LLM-based pipeline to automatically annotate longitudinal information in radiology reports. The pipeline first identifies sentences containing relevant information and then extracts the progression of diseases. We evaluate and compare five mainstream LLMs on these two tasks using 500 manually annotated reports. Considering both efficiency and performance, Qwen2.5-32B was subsequently selected and used to annotate another 95,169 reports from the public MIMIC-CXR dataset. Our Qwen2.5-32B-annotated dataset provided us with a standardized benchmark for evaluating report generation models. Using this new benchmark, we assessed seven state-of-the-art report generation models. Our LLM-based annotation method outperforms existing annotation solutions, achieving 11.3\% and 5.3\% higher F1-scores for longitudinal information detection and disease tracking, respectively. 

---
# Mitigating Bias in Automated Grading Systems for ESL Learners: A Contrastive Learning Approach 

**Authors**: Kevin Fan, Eric Yun  

**Link**: [PDF](https://arxiv.org/pdf/2601.16724)  

**Abstract**: As Automated Essay Scoring (AES) systems are increasingly used in high-stakes educational settings, concerns regarding algorithmic bias against English as a Second Language (ESL) learners have increased. Current Transformer-based regression models trained primarily on native-speaker corpora often learn spurious correlations between surface-level L2 linguistic features and essay quality. In this study, we conduct a bias study of a fine-tuned DeBERTa-v3 model using the ASAP 2.0 and ELLIPSE datasets, revealing a constrained score scaling for high-proficiency ESL writing where high-proficiency ESL essays receive scores 10.3% lower than Native speaker essays of identical human-rated quality. To mitigate this, we propose applying contrastive learning with a triplet construction strategy: Contrastive Learning with Matched Essay Pairs. We constructed a dataset of 17,161 matched essay pairs and fine-tuned the model using Triplet Margin Loss to align the latent representations of ESL and Native writing. Our approach reduced the high-proficiency scoring disparity by 39.9% (to a 6.2% gap) while maintaining a Quadratic Weighted Kappa (QWK) of 0.76. Post-hoc linguistic analysis suggests the model successfully disentangled sentence complexity from grammatical error, preventing the penalization of valid L2 syntactic structures. 

---
# Better Generalizing to Unseen Concepts: An Evaluation Framework and An LLM-Based Auto-Labeled Pipeline for Biomedical Concept Recognition 

**Authors**: Shanshan Liu, Noriki Nishida, Fei Cheng, Narumi Tokunaga, Rumana Ferdous Munne, Yuki Yamagata, Kouji Kozaki, Takehito Utsuro, Yuji Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2601.16711)  

**Abstract**: Generalization to unseen concepts is a central challenge due to the scarcity of human annotations in Mention-agnostic Biomedical Concept Recognition (MA-BCR). This work makes two key contributions to systematically address this issue. First, we propose an evaluation framework built on hierarchical concept indices and novel metrics to measure generalization. Second, we explore LLM-based Auto-Labeled Data (ALD) as a scalable resource, creating a task-specific pipeline for its generation. Our research unequivocally shows that while LLM-generated ALD cannot fully substitute for manual annotations, it is a valuable resource for improving generalization, successfully providing models with the broader coverage and structural knowledge needed to approach recognizing unseen concepts. Code and datasets are available at this https URL. 

---
# EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents 

**Authors**: Xinze Li, Ziyue Zhu, Siyuan Liu, Yubo Ma, Yuhang Zang, Yixin Cao, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.16690)  

**Abstract**: We introduce EMemBench, a programmatic benchmark for evaluating long-term memory of agents through interactive games. Rather than using a fixed set of questions, EMemBench generates questions from each agent's own trajectory, covering both text and visual game environments. Each template computes verifiable ground truth from underlying game signals, with controlled answerability and balanced coverage over memory skills: single/multi-hop recall, induction, temporal, spatial, logical, and adversarial. We evaluate memory agents with strong LMs/VLMs as backbones, using in-context prompting as baselines. Across 15 text games and multiple visual seeds, results are far from saturated: induction and spatial reasoning are persistent bottlenecks, especially in visual setting. Persistent memory yields clear gains for open backbones on text games, but improvements are less consistent for VLM agents, suggesting that visually grounded episodic memory remains an open challenge. A human study further confirms the difficulty of EMemBench. 

---
# PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice 

**Authors**: Yuzhen Shi, Huanghai Liu, Yiran Hu, Gaojie Song, Xinran Xu, Yubo Ma, Tianyi Tang, Li Zhang, Qingjing Chen, Di Feng, Wenbo Lv, Weiheng Wu, Kexin Yang, Sen Yang, Wei Wang, Rongyao Shi, Yuanyang Qiu, Yuemeng Qi, Jingwen Zhang, Xiaoyu Sui, Yifan Chen, Yi Zhang, An Yang, Bowen Yu, Dayiheng Liu, Junyang Lin, Weixing Shen, Bing Zhao, Charles L.A. Clarke, Hu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.16669)  

**Abstract**: As large language models (LLMs) are increasingly applied to legal domain-specific tasks, evaluating their ability to perform legal work in real-world settings has become essential. However, existing legal benchmarks rely on simplified and highly standardized tasks, failing to capture the ambiguity, complexity, and reasoning demands of real legal practice. Moreover, prior evaluations often adopt coarse, single-dimensional metrics and do not explicitly assess fine-grained legal reasoning. To address these limitations, we introduce PLawBench, a Practical Law Benchmark designed to evaluate LLMs in realistic legal practice scenarios. Grounded in real-world legal workflows, PLawBench models the core processes of legal practitioners through three task categories: public legal consultation, practical case analysis, and legal document generation. These tasks assess a model's ability to identify legal issues and key facts, perform structured legal reasoning, and generate legally coherent documents. PLawBench comprises 850 questions across 13 practical legal scenarios, with each question accompanied by expert-designed evaluation rubrics, resulting in approximately 12,500 rubric items for fine-grained assessment. Using an LLM-based evaluator aligned with human expert judgments, we evaluate 10 state-of-the-art LLMs. Experimental results show that none achieves strong performance on PLawBench, revealing substantial limitations in the fine-grained legal reasoning capabilities of current LLMs and highlighting important directions for future evaluation and development of legal LLMs. Data is available at: this https URL. 

---
# Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations 

**Authors**: Lukas Hinterleitner, Loris Schoenegger, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2601.16651)  

**Abstract**: Gradient-based methods for instance-based explanation for large language models (LLMs) are hindered by the immense dimensionality of model gradients. In practice, influence estimation is restricted to a subset of model parameters to make computation tractable, but this subset is often chosen ad hoc and rarely justified by systematic evaluation. This paper investigates if it is better to create low-dimensional representations by selecting a small, architecturally informed subset of model components or by projecting the full gradients into a lower-dimensional space. Using a novel benchmark, we show that a greedily selected subset of components captures the information about training data influence needed for a retrieval task more effectively than either the full gradient or random projection. We further find that this approach is more computationally efficient than random projection, demonstrating that targeted component selection is a practical strategy for making instance-based explanations of large models more computationally feasible. 

---
# Sycophancy Hides Linearly in the Attention Heads 

**Authors**: Rifo Genadi, Munachiso Nwadike, Nurdaulet Mukhituly, Hilal Alquabeh, Tatsuya Hiraoka, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2601.16644)  

**Abstract**: We find that correct-to-incorrect sycophancy signals are most linearly separable within multi-head attention activations. Motivated by the linear representation hypothesis, we train linear probes across the residual stream, multilayer perceptron (MLP), and attention layers to analyze where these signals emerge. Although separability appears in the residual stream and MLPs, steering using these probes is most effective in a sparse subset of middle-layer attention heads. Using TruthfulQA as the base dataset, we find that probes trained on it transfer effectively to other factual QA benchmarks. Furthermore, comparing our discovered direction to previously identified "truthful" directions reveals limited overlap, suggesting that factual accuracy, and deference resistance, arise from related but distinct mechanisms. Attention-pattern analysis further indicates that the influential heads attend disproportionately to expressions of user doubt, contributing to sycophantic shifts. Overall, these findings suggest that sycophancy can be mitigated through simple, targeted linear interventions that exploit the internal geometry of attention activations. 

---
# Typologically Informed Parameter Aggregation 

**Authors**: Stef Accou, Wessel Poelman  

**Link**: [PDF](https://arxiv.org/pdf/2601.16629)  

**Abstract**: Massively multilingual language models enable cross-lingual generalization but underperform on low-resource and unseen languages. While adapter-based fine-tuning offers a parameter-efficient solution, training language-specific adapters at scale remains costly. We introduce Typologically Informed Parameter Aggregation (TIPA), a training-free method that constructs proxy language adapters by aggregating existing ones, weighted by typological similarity. Integrated into the MAD-X framework, these proxies enable zero-shot cross-lingual transfer without additional training. We evaluate TIPA on five NLP tasks and over 230 languages. TIPA consistently outperforms or matches baselines such as English-only fine-tuning or selecting the typologically closest language adapter. We see the largest gains for languages lacking dedicated adapters. Our results demonstrate that typologically informed aggregation provides a viable alternative to language-specific modules without any training needed. 

---
# MultiLexNorm++: A Unified Benchmark and a Generative Model for Lexical Normalization for Asian Languages 

**Authors**: Weerayut Buaphet, Thanh-Nhi Nguyen, Risa Kondo, Tomoyuki Kajiwara, Yumin Kim, Jimin Lee, Hwanhee Lee, Holy Lovenia, Peerat Limkonchotiwat, Sarana Nutanong, Rob Van der Goot  

**Link**: [PDF](https://arxiv.org/pdf/2601.16623)  

**Abstract**: Social media data has been of interest to Natural Language Processing (NLP) practitioners for over a decade, because of its richness in information, but also challenges for automatic processing. Since language use is more informal, spontaneous, and adheres to many different sociolects, the performance of NLP models often deteriorates. One solution to this problem is to transform data to a standard variant before processing it, which is also called lexical normalization. There has been a wide variety of benchmarks and models proposed for this task. The MultiLexNorm benchmark proposed to unify these efforts, but it consists almost solely of languages from the Indo-European language family in the Latin script. Hence, we propose an extension to MultiLexNorm, which covers 5 Asian languages from different language families in 4 different scripts. We show that the previous state-of-the-art model performs worse on the new languages and propose a new architecture based on Large Language Models (LLMs), which shows more robust performance. Finally, we analyze remaining errors, revealing future directions for this task. 

---
# How Does Personalized Memory Shape LLM Behavior? Benchmarking Rational Preference Utilization in Personalized Assistants 

**Authors**: Xueyang Feng, Weinan Gan, Xu Chen, Quanyu Dai, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16621)  

**Abstract**: Large language model (LLM)-powered assistants have recently integrated memory mechanisms that record user preferences, leading to more personalized and user-aligned responses. However, irrelevant personalized memories are often introduced into the context, interfering with the LLM's intent understanding. To comprehensively investigate the dual effects of personalization, we develop RPEval, a benchmark comprising a personalized intent reasoning dataset and a multi-granularity evaluation protocol. RPEval reveals the widespread phenomenon of irrational personalization in existing LLMs and, through error pattern analysis, illustrates its negative impact on user experience. Finally, we introduce RP-Reasoner, which treats memory utilization as a pragmatic reasoning process, enabling the selective integration of personalized information. Experimental results demonstrate that our method significantly outperforms carefully designed baselines on RPEval, and resolves 80% of the bad cases observed in a large-scale commercial personalized assistant, highlighting the potential of pragmatic reasoning to mitigate irrational personalization. Our benchmark is publicly available at this https URL. 

---
# PROST-LLM: Progressively Enhancing the Speech-to-Speech Translation Capability in LLMs 

**Authors**: Jing Xu, Jiaqi Wang, Daxin Tan, Xiao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.16618)  

**Abstract**: Although Large Language Models (LLMs) excel in many tasks, their application to Speech-to-Speech Translation (S2ST) is underexplored and hindered by data scarcity. To bridge this gap, we propose PROST-LLM (PROgressive Speech-to-speech Translation) to enhance the S2ST capabilities in LLMs progressively. First, we fine-tune the LLMs with the CVSS corpus, employing designed tri-task learning and chain of modality methods to boost the initial performance. Then, leveraging the fine-tuned model, we generate preference pairs through self-sampling and back-translation without human evaluation. Finally, these preference pairs are used for preference optimization to enhance the model's S2ST capability further. Extensive experiments confirm the effectiveness of our proposed PROST-LLM in improving the S2ST capability of LLMs. 

---
# AuroraEdge-V-2B: A Faster And Stronger Edge Visual Large Language Model 

**Authors**: Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.16615)  

**Abstract**: Recently, due to the advancement of multimodal technology, people are attempting to use visual large language models (VLLMs) in industrial production. Many deep learning models (DLMs) deployed in the production environment are gradually being replaced by VLLMs. Compared with DLMs, VLLMs have some advantages in industrial applications: (1) Their strong generalization ability enables them to perform well across a wide range of tasks. (2) They are flexible and can deal with unfamiliar samples through context learning quickly. However, VLLMs also have obvious drawbacks: (1) VLLMs do not perform as well as custom-developed DLMs in specific domains. (2) The number of parameters in VLLMs is generally quite large, and their deployment requires substantial computational resources. (3) VLLMs generally operate much slower than DLMs, making real-time response challenging to achieve. To better utilize VLLMs in industrial applications, we introduce AuroraEdge-V-2B in this work, a compact, robust, and high-speed VLLM designed for edge deployment. To make the model run faster, we also propose a compression-fusion method to improve inference efficiency. AuroraEdge-V-2B has the following notable features: (1) Easy deployment and faster: It has only 2B parameters and is highly suitable for edge deployment, offering better real-time performance. (2) Fewer visual tokens and cheaper: It significantly reduces the number of visual tokens in the decoding process, thereby reducing the floating-point operations by half during inference and making it cheaper to use. (3) Strong performance: It gets a higher score on 9 benchmarks than models with the same number of parameter (e.g., Qwen2-VL-2B, Qwen2.5-VL-3B, InternVL-2.5-2B). 

---
# Attention-MoA: Enhancing Mixture-of-Agents via Inter-Agent Semantic Attention and Deep Residual Synthesis 

**Authors**: Jianyu Wen, Yang Wei, Xiongxi Yu, Changxuan Xiao, Ke Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.16596)  

**Abstract**: As the development of Large Language Models (LLMs) shifts from parameter scaling to inference-time collaboration, the Mixture-of-Agents (MoA) framework has emerged as a general paradigm to harness collective intelligence by layering diverse models. While recent MoA variants have introduced dynamic routing and residual connections to improve efficiency, these methods often fail to facilitate deep semantic interaction between agents, limiting the system's ability to actively correct hallucinations and refine logic. In this paper, we introduce Attention-MoA, a novel MoA-based framework that redefines collaboration through Inter-agent Semantic Attention. Complemented by an Inter-layer Residual Module with Adaptive Early Stopping Mechanism, our architecture mitigates information degradation in deep layers while improving computational efficiency. Extensive evaluations across AlpacaEval 2.0, MT-Bench, and FLASK demonstrate that Attention-MoA significantly outperforms state-of-the-art baselines, achieving a 91.15% Length-Controlled Win Rate on AlpacaEval 2.0 and dominating in 10 out of 12 capabilities on FLASK. Notably, Attention-MoA enables an ensemble of small open-source models to outperform massive proprietary models like Claude-4.5-Sonnet and GPT-4.1, achieving an MT-Bench score of 8.83 and an AlpacaEval 2.0 LC Win Rate of 77.36%. 

---
# Retrieve-Refine-Calibrate: A Framework for Complex Claim Fact-Checking 

**Authors**: Mingwei Sun, Qianlong Wang, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16555)  

**Abstract**: Fact-checking aims to verify the truthfulness of a claim based on the retrieved evidence. Existing methods typically follow a decomposition paradigm, in which a claim is broken down into sub-claims that are individually verified. However, the decomposition paradigm may introduce noise to the verification process due to irrelevant entities or evidence, ultimately degrading verification accuracy. To address this problem, we propose a Retrieve-Refine-Calibrate (RRC) framework based on large language models (LLMs). Specifically, the framework first identifies the entities mentioned in the claim and retrieves evidence relevant to them. Then, it refines the retrieved evidence based on the claim to reduce irrelevant information. Finally, it calibrates the verification process by re-evaluating low-confidence predictions. Experiments on two popular fact-checking datasets (HOVER and FEVEROUS-S) demonstrate that our framework achieves superior performance compared with competitive baselines. 

---
# Curate-Train-Refine: A Closed-Loop Agentic Framework for Zero Shot Classification 

**Authors**: Gaurav Maheshwari, Kevin El Haddad  

**Link**: [PDF](https://arxiv.org/pdf/2601.16530)  

**Abstract**: Large language models (LLMs) and high-capacity encoders have advanced zero and few-shot classification, but their inference cost and latency limit practical deployment. We propose training lightweight text classifiers using dynamically generated supervision from an LLM. Our method employs an iterative, agentic loop in which the LLM curates training data, analyzes model successes and failures, and synthesizes targeted examples to address observed errors. This closed-loop generation and evaluation process progressively improves data quality and adapts it to the downstream classifier and task. Across four widely used benchmarks, our approach consistently outperforms standard zero and few-shot baselines. These results indicate that LLMs can serve effectively as data curators, enabling accurate and efficient classification without the operational cost of large-model deployment. 

---
# SearchLLM: Detecting LLM Paraphrased Text by Measuring the Similarity with Regeneration of the Candidate Source via Search Engine 

**Authors**: Hoang-Quoc Nguyen-Son, Minh-Son Dao, Koji Zettsu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16512)  

**Abstract**: With the advent of large language models (LLMs), it has become common practice for users to draft text and utilize LLMs to enhance its quality through paraphrasing. However, this process can sometimes result in the loss or distortion of the original intended meaning. Due to the human-like quality of LLM-generated text, traditional detection methods often fail, particularly when text is paraphrased to closely mimic original content. In response to these challenges, we propose a novel approach named SearchLLM, designed to identify LLM-paraphrased text by leveraging search engine capabilities to locate potential original text sources. By analyzing similarities between the input and regenerated versions of candidate sources, SearchLLM effectively distinguishes LLM-paraphrased content. SearchLLM is designed as a proxy layer, allowing seamless integration with existing detectors to enhance their performance. Experimental results across various LLMs demonstrate that SearchLLM consistently enhances the accuracy of recent detectors in detecting LLM-paraphrased text that closely mimics original content. Furthermore, SearchLLM also helps the detectors prevent paraphrasing attacks. 

---
# Is Length Really A Liability? An Evaluation of Multi-turn LLM Conversations using BoolQ 

**Authors**: Karl Neergaard, Le Qiu, Emmanuele Chersoni  

**Link**: [PDF](https://arxiv.org/pdf/2601.16508)  

**Abstract**: Single-prompt evaluations dominate current LLM benchmarking, yet they fail to capture the conversational dynamics where real-world harm occurs. In this study, we examined whether conversation length affects response veracity by evaluating LLM performance on the BoolQ dataset under varying length and scaffolding conditions. Our results across three distinct LLMs revealed model-specific vulnerabilities that are invisible under single-turn testing. The length-dependent and scaffold-specific effects we observed demonstrate a fundamental limitation of static evaluations, as deployment-relevant vulnerabilities could only be spotted in a multi-turn conversational setting. 

---
# LOGICAL-COMMONSENSEQA: A Benchmark for Logical Commonsense Reasoning 

**Authors**: Obed Junias, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2601.16504)  

**Abstract**: Commonsense reasoning often involves evaluating multiple plausible interpretations rather than selecting a single atomic answer, yet most benchmarks rely on single-label evaluation, obscuring whether statements are jointly plausible, mutually exclusive, or jointly implausible. We introduce LOGICAL-COMMONSENSEQA, a benchmark that re-frames commonsense reasoning as logical composition over pairs of atomic statements using plausibility-level operators (AND, OR, NEITHER/NOR). Evaluating instruction-tuned, reasoning-specialized, and fine-tuned models under zero-shot, few-shot, and chain-of-thought prompting, we find that while models perform reasonably on conjunctive and moderately on disjunctive reasoning, performance degrades sharply on negation-based questions. LOGICAL-COMMONSENSEQA exposes fundamental reasoning limitations and provides a controlled framework for advancing compositional commonsense reasoning. 

---
# MRAG: Benchmarking Retrieval-Augmented Generation for Bio-medicine 

**Authors**: Wei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16503)  

**Abstract**: While Retrieval-Augmented Generation (RAG) has been swiftly adopted in scientific and clinical QA systems, a comprehensive evaluation benchmark in the medical domain is lacking. To address this gap, we introduce the Medical Retrieval-Augmented Generation (MRAG) benchmark, covering various tasks in English and Chinese languages, and building a corpus with Wikipedia and Pubmed. Additionally, we develop the MRAG-Toolkit, facilitating systematic exploration of different RAG components. Our experiments reveal that: (a) RAG enhances LLM reliability across MRAG tasks. (b) the performance of RAG systems is influenced by retrieval approaches, model sizes, and prompting strategies. (c) While RAG improves usefulness and reasoning quality, LLM responses may become slightly less readable for long-form questions. We will release the MRAG-Bench's dataset and toolkit with CCBY-4.0 license upon acceptance, to facilitate applications from both academia and industry. 

---
# Timely Machine: Awareness of Time Makes Test-Time Scaling Agentic 

**Authors**: Yichuan Ma, Linyang Li, Yongkang chen, Peiji Li, Xiaozhe Li, Qipeng Guo, Dahua Lin, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.16486)  

**Abstract**: As large language models (LLMs) increasingly tackle complex reasoning tasks, test-time scaling has become critical for enhancing capabilities. However, in agentic scenarios with frequent tool calls, the traditional generation-length-based definition breaks down: tool latency decouples inference time from generation length. We propose Timely Machine, redefining test-time as wall-clock time, where models dynamically adjust strategies based on time budgets. We introduce Timely-Eval, a benchmark spanning high-frequency tool calls, low-frequency tool calls, and time-constrained reasoning. By varying tool latency, we find smaller models excel with fast feedback through more interactions, while larger models dominate high-latency settings via superior interaction quality. Moreover, existing models fail to adapt reasoning to time budgets. We propose Timely-RL to address this gap. After cold-start supervised fine-tuning, we use reinforcement learning to enhance temporal planning. Timely-RL improves time budget awareness and consistently boosts performance across Timely-Eval. We hope our work offers a new perspective on test-time scaling for the agentic era. 

---
# TL-GRPO: Turn-Level RL for Reasoning-Guided Iterative Optimization 

**Authors**: Peiji Li, Linyang Li, Handa Sun, Wenjin Mai, Yongkang Chen, Xiaozhe Li, Yue Shen, Yichuan Ma, Yiliu Sun, Jiaxi Cao, Zhishu He, Bo Wang, Xiaoqing Zheng, Zhaori Bi, Xipeng Qiu, Qipeng Guo, Kai Chen, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.16480)  

**Abstract**: Large language models have demonstrated strong reasoning capabilities in complex tasks through tool integration, which is typically framed as a Markov Decision Process and optimized with trajectory-level RL algorithms such as GRPO. However, a common class of reasoning tasks, iterative optimization, presents distinct challenges: the agent interacts with the same underlying environment state across turns, and the value of a trajectory is determined by the best turn-level reward rather than cumulative returns. Existing GRPO-based methods cannot perform fine-grained, turn-level optimization in such settings, while black-box optimization methods discard prior knowledge and reasoning capabilities. To address this gap, we propose Turn-Level GRPO (TL-GRPO), a lightweight RL algorithm that performs turn-level group sampling for fine-grained optimization. We evaluate TL-GRPO on analog circuit sizing (ACS), a challenging scientific optimization task requiring multiple simulations and domain expertise. Results show that TL-GRPO outperforms standard GRPO and Bayesian optimization methods across various specifications. Furthermore, our 30B model trained with TL-GRPO achieves state-of-the-art performance on ACS tasks under same simulation budget, demonstrating both strong generalization and practical utility. 

---
# DeepEra: A Deep Evidence Reranking Agent for Scientific Retrieval-Augmented Generated Question Answering 

**Authors**: Haotian Chen, Qingqing Long, Siyu Pu, Xiao Luo, Wei Ju, Meng Xiao, Yuanchun Zhou, Jianghua Zhao, Xuezhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.16478)  

**Abstract**: With the rapid growth of scientific literature, scientific question answering (SciQA) has become increasingly critical for exploring and utilizing scientific knowledge. Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating knowledge from external sources, thereby providing credible evidence for scientific question answering. But existing retrieval and reranking methods remain vulnerable to passages that are semantically similar but logically irrelevant, often reducing factual reliability and amplifying this http URL address this challenge, we propose a Deep Evidence Reranking Agent (DeepEra) that integrates step-by-step reasoning, enabling more precise evaluation of candidate passages beyond surface-level semantics. To support systematic evaluation, we construct SciRAG-SSLI (Scientific RAG - Semantically Similar but Logically Irrelevant), a large-scale dataset comprising about 300K SciQA instances across 10 subjects, constructed from 10M scientific corpus. The dataset combines naturally retrieved contexts with systematically generated distractors to test logical robustness and factual grounding. Comprehensive evaluations confirm that our approach achieves superior retrieval performance compared to leading rerankers. To our knowledge, this work is the first to comprehensively study and empirically validate innegligible SSLI issues in two-stage RAG frameworks. 

---
# Persona Jailbreaking in Large Language Models 

**Authors**: Jivnesh Sandhan, Fei Cheng, Tushar Sandhan, Yugo Murawaki  

**Link**: [PDF](https://arxiv.org/pdf/2601.16466)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in domains such as education, mental health and customer support, where stable and consistent personas are critical for reliability. Yet, existing studies focus on narrative or role-playing tasks and overlook how adversarial conversational history alone can reshape induced personas. Black-box persona manipulation remains unexplored, raising concerns for robustness in realistic interactions. In response, we introduce the task of persona editing, which adversarially steers LLM traits through user-side inputs under a black-box, inference-only setting. To this end, we propose PHISH (Persona Hijacking via Implicit Steering in History), the first framework to expose a new vulnerability in LLM safety that embeds semantically loaded cues into user queries to gradually induce reverse personas. We also define a metric to quantify attack success. Across 3 benchmarks and 8 LLMs, PHISH predictably shifts personas, triggers collateral changes in correlated traits, and exhibits stronger effects in multi-turn settings. In high-risk domains mental health, tutoring, and customer support, PHISH reliably manipulates personas, validated by both human and LLM-as-Judge evaluations. Importantly, PHISH causes only a small reduction in reasoning benchmark performance, leaving overall utility largely intact while still enabling significant persona manipulation. While current guardrails offer partial protection, they remain brittle under sustained attack. Our findings expose new vulnerabilities in personas and highlight the need for context-resilient persona in LLMs. Our codebase and dataset is available at: this https URL 

---
# Graph-Anchored Knowledge Indexing for Retrieval-Augmented Generation 

**Authors**: Zhenghao Liu, Mingyan Wu, Xinze Li, Yukun Yan, Shuo Wang, Cheng Yang, Minghe Yu, Zheni Zeng, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.16462)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a dominant paradigm for mitigating hallucinations in Large Language Models (LLMs) by incorporating external knowledge. Nevertheless, effectively integrating and interpreting key evidence scattered across noisy documents remains a critical challenge for existing RAG systems. In this paper, we propose GraphAnchor, a novel Graph-Anchored Knowledge Indexing approach that reconceptualizes graph structures from static knowledge representations into active, evolving knowledge indices. GraphAnchor incrementally updates a graph during iterative retrieval to anchor salient entities and relations, yielding a structured index that guides the LLM in evaluating knowledge sufficiency and formulating subsequent subqueries. The final answer is generated by jointly leveraging all retrieved documents and the final evolved graph. Experiments on four multi-hop question answering benchmarks demonstrate the effectiveness of GraphAnchor, and reveal that GraphAnchor modulates the LLM's attention to more effectively associate key information distributed in retrieved documents. All code and data are available at this https URL. 

---
# Mixing Expert Knowledge: Bring Human Thoughts Back To the Game of Go 

**Authors**: Yichuan Ma, Linyang Li, Yongkang Chen, Peiji Li, Jiasheng Ye, Qipeng Guo, Dahua Lin, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.16447)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance in reasoning tasks such as mathematics and coding, matching or surpassing human capabilities. However, these impressive reasoning abilities face significant challenges in specialized domains. Taking Go as an example, although AlphaGo has established the high performance ceiling of AI systems in Go, mainstream LLMs still struggle to reach even beginner-level proficiency, let alone perform natural language reasoning. This performance gap between general-purpose LLMs and domain experts is significantly limiting the application of LLMs on a wider range of domain-specific tasks. In this work, we aim to bridge the divide between LLMs' general reasoning capabilities and expert knowledge in domain-specific tasks. We perform mixed fine-tuning with structured Go expertise and general long Chain-of-Thought (CoT) reasoning data as a cold start, followed by reinforcement learning to integrate expert knowledge in Go with general reasoning capabilities. Through this methodology, we present \textbf{LoGos}, a powerful LLM that not only maintains outstanding general reasoning abilities, but also conducts Go gameplay in natural language, demonstrating effective strategic reasoning and accurate next-move prediction. LoGos achieves performance comparable to human professional players, substantially surpassing all existing LLMs. Through this work, we aim to contribute insights on applying general LLM reasoning capabilities to specialized domains. We will release the first large-scale Go dataset for LLM training, the first LLM Go evaluation benchmark, and the first general LLM that reaches human professional-level performance in Go at: this https URL. 

---
# Exploring the Effects of Alignment on Numerical Bias in Large Language Models 

**Authors**: Ayako Sato, Hwichan Kim, Zhousi Chen, Masato Mita, Mamoru Komachi  

**Link**: [PDF](https://arxiv.org/pdf/2601.16444)  

**Abstract**: ``LLM-as-a-judge,'' which utilizes large language models (LLMs) as evaluators, has proven effective in many evaluation tasks. However, evaluator LLMs exhibit numerical bias, a phenomenon where certain evaluation scores are generated disproportionately often, leading reduced evaluation performance. This study investigates the cause of this bias. Given that most evaluator LLMs are aligned through instruction tuning and preference tuning, and that prior research suggests alignment reduces output diversity, we hypothesize that numerical bias arises from alignment. To test this, we compare outputs from pre- and post-alignment LLMs, and observe that alignment indeed increases numerical bias. We also explore mitigation strategies for post-alignment LLMs, including temperature scaling, distribution calibration, and score range adjustment. Among these, score range adjustment is most effective in reducing bias and improving performance, though still heuristic. Our findings highlight the need for further work on optimal score range selection and more robust mitigation strategies. 

---
# Learning Domain Knowledge in Multimodal Large Language Models through Reinforcement Fine-Tuning 

**Authors**: Qinglong Cao, Yuntian Chen, Chao Ma, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.16419)  

**Abstract**: Multimodal large language models (MLLMs) have shown remarkable capabilities in multimodal perception and understanding tasks. However, their effectiveness in specialized domains, such as remote sensing and medical imaging, remains limited. A natural approach to domain adaptation is to inject domain knowledge through textual instructions, prompts, or auxiliary captions. Surprisingly, we find that such input-level domain knowledge injection yields little to no improvement on scientific multimodal tasks, even when the domain knowledge is explicitly provided. This observation suggests that current MLLMs fail to internalize domain-specific priors through language alone, and that domain knowledge must be integrated at the optimization level. Motivated by this insight, we propose a reinforcement fine-tuning framework that incorporates domain knowledge directly into the learning objective. Instead of treating domain knowledge as descriptive information, we encode it as domain-informed constraints and reward signals, shaping the model's behavior in the output space. Extensive experiments across multiple datasets in remote sensing and medical domains consistently demonstrate good performance gains, achieving state-of-the-art results on multimodal domain tasks. Our results highlight the necessity of optimization-level domain knowledge integration and reveal a fundamental limitation of textual domain conditioning in current MLLMs. 

---
# Jacobian Scopes: token-level causal attributions in LLMs 

**Authors**: Toni J.B. Liu, Baran Zadeoğlu, Nicolas Boullé, Raphaël Sarfati, Christopher J. Earls  

**Link**: [PDF](https://arxiv.org/pdf/2601.16407)  

**Abstract**: Large language models (LLMs) make next-token predictions based on clues present in their context, such as semantic descriptions and in-context examples. Yet, elucidating which prior tokens most strongly influence a given prediction remains challenging due to the proliferation of layers and attention heads in modern architectures. We propose Jacobian Scopes, a suite of gradient-based, token-level causal attribution methods for interpreting LLM predictions. By analyzing the linearized relations of final hidden state with respect to inputs, Jacobian Scopes quantify how input tokens influence a model's prediction. We introduce three variants - Semantic, Fisher, and Temperature Scopes - which respectively target sensitivity of specific logits, the full predictive distribution, and model confidence (inverse temperature). Through case studies spanning instruction understanding, translation and in-context learning (ICL), we uncover interesting findings, such as when Jacobian Scopes point to implicit political biases. We believe that our proposed methods also shed light on recently debated mechanisms underlying in-context time-series forecasting. Our code and interactive demonstrations are publicly available at this https URL. 

---
# Clarify or Answer: Reinforcement Learning for Agentic VQA with Context Under-specification 

**Authors**: Zongwan Cao, Bingbing Wen, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.16400)  

**Abstract**: Real-world visual question answering (VQA) is often context-dependent: an image-question pair may be under-specified, such that the correct answer depends on external information that is not observable in the image. In such cases, directly answering can lead to confident but incorrect predictions. We propose CoA(Clarify-or-Answer), an ask-or-answer agent that separately models the decision to ask or answer, and what to ask if needed. CoA first determines whether clarification is necessary; if so, it asks a single focused question and then incorporates the response to produce the final answer. We introduce CONTEXTCLARIFY with a set of ambiguous VQA questions and the contrast set that is non-ambiguous. We further introduce GRPO-CR (Clarification Reasoning), a reinforcement learning approach that optimizes clarification question generation with multiple reward signals encouraging well-formed, focused, non-trivial questions that resolve ambiguity. Across three VLLMs and three datasets, CoA achieves consistent improvements at both the module and system levels, improving end-to-end VQA accuracy by an average of +15.3 points (83%) over prompting-based baselines 

---
# Cite-While-You-Generate: Training-Free Evidence Attribution for Multimodal Clinical Summarization 

**Authors**: Qianqi Yan, Huy Nguyen, Sumana Srivatsa, Hari Bandi, Xin Eric Wang, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2601.16397)  

**Abstract**: Trustworthy clinical summarization requires not only fluent generation but also transparency about where each statement comes from. We propose a training-free framework for generation-time source attribution that leverages decoder attentions to directly cite supporting text spans or images, overcoming the limitations of post-hoc or retraining-based methods. We introduce two strategies for multimodal attribution: a raw image mode, which directly uses image patch attentions, and a caption-as-span mode, which substitutes images with generated captions to enable purely text-based alignment. Evaluations on two representative domains: clinician-patient dialogues (CliConSummation) and radiology reports (MIMIC-CXR), show that our approach consistently outperforms embedding-based and self-attribution baselines, improving both text-level and multimodal attribution accuracy (e.g., +15% F1 over embedding baselines). Caption-based attribution achieves competitive performance with raw-image attention while being more lightweight and practical. These findings highlight attention-guided attribution as a promising step toward interpretable and deployable clinical summarization systems. 

---
# Cross-Lingual Activation Steering for Multilingual Language Models 

**Authors**: Rhitabrat Pokharel, Ameeta Agrawal, Tanay Nagar  

**Link**: [PDF](https://arxiv.org/pdf/2601.16390)  

**Abstract**: Large language models exhibit strong multilingual capabilities, yet significant performance gaps persist between dominant and non-dominant languages. Prior work attributes this gap to imbalances between shared and language-specific neurons in multilingual representations. We propose Cross-Lingual Activation Steering (CLAS), a training-free inference-time intervention that selectively modulates neuron activations. We evaluate CLAS on classification and generation benchmarks, achieving average improvements of 2.3% (Acc.) and 3.4% (F1) respectively, while maintaining high-resource language performance. We discover that effective transfer operates through functional divergence rather than strict alignment; performance gains correlate with increased language cluster separation. Our results demonstrate that targeted activation steering can unlock latent multilingual capacity in existing models without modification to model weights. 

---
# PolyAgent: Large Language Model Agent for Polymer Design 

**Authors**: Vani Nigam, Achuth Chandrasekhar, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2601.16376)  

**Abstract**: On-demand Polymer discovery is essential for various industries, ranging from biomedical to reinforcement materials. Experiments with polymers have a long trial-and-error process, leading to long procedures and extensive resources. For these processes, machine learning has accelerated scientific discovery at the property prediction and latent space search fronts. However, laboratory researchers cannot readily access codes and these models to extract individual structures and properties due to infrastructure limitations. We present a closed-loop polymer structure-property predictor integrated in a terminal for early-stage polymer discovery. The framework is powered by LLM reasoning to provide users with property prediction, property-guided polymer structure generation, and structure modification capabilities. The SMILES sequences are guided by the synthetic accessibility score and the synthetic complexity score (SC Score) to ensure that polymer generation is as close as possible to synthetically accessible monomer-level structures. This framework addresses the challenge of generating novel polymer structures for laboratory researchers, thereby providing computational insights into polymer research. 

---
# Identity, Cooperation and Framing Effects within Groups of Real and Simulated Humans 

**Authors**: Suhong Moon, Minwoo Kang, Joseph Suh, Mustafa Safdari, John Canny  

**Link**: [PDF](https://arxiv.org/pdf/2601.16355)  

**Abstract**: Humans act via a nuanced process that depends both on rational deliberation and also on identity and contextual factors. In this work, we study how large language models (LLMs) can simulate human action in the context of social dilemma games. While prior work has focused on "steering" (weak binding) of chat models to simulate personas, we analyze here how deep binding of base models with extended backstories leads to more faithful replication of identity-based behaviors. Our study has these findings: simulation fidelity vs human studies is improved by conditioning base LMs with rich context of narrative identities and checking consistency using instruction-tuned models. We show that LLMs can also model contextual factors such as time (year that a study was performed), question framing, and participant pool effects. LLMs, therefore, allow us to explore the details that affect human studies but which are often omitted from experiment descriptions, and which hamper accurate replication. 

---
# Regional Bias in Large Language Models 

**Authors**: M P V S Gopinadh, Kappara Lakshmi Sindhu, Soma Sekhar Pandu Ranga Raju P, Yesaswini Swarna  

**Link**: [PDF](https://arxiv.org/pdf/2601.16349)  

**Abstract**: This study investigates regional bias in large language models (LLMs), an emerging concern in AI fairness and global representation. We evaluate ten prominent LLMs: GPT-3.5, GPT-4o, Gemini 1.5 Flash, Gemini 1.0 Pro, Claude 3 Opus, Claude 3.5 Sonnet, Llama 3, Gemma 7B, Mistral 7B, and Vicuna-13B using a dataset of 100 carefully designed prompts that probe forced-choice decisions between regions under contextually neutral scenarios. We introduce FAZE, a prompt-based evaluation framework that measures regional bias on a 10-point scale, where higher scores indicate a stronger tendency to favor specific regions. Experimental results reveal substantial variation in bias levels across models, with GPT-3.5 exhibiting the highest bias score (9.5) and Claude 3.5 Sonnet scoring the lowest (2.5). These findings indicate that regional bias can meaningfully undermine the reliability, fairness, and inclusivity of LLM outputs in real-world, cross-cultural applications. This work contributes to AI fairness research by highlighting the importance of inclusive evaluation frameworks and systematic approaches for identifying and mitigating geographic biases in language models. 

---
# Machine-Assisted Grading of Nationwide School-Leaving Essay Exams with LLMs and Statistical NLP 

**Authors**: Andres Karjus, Kais Allkivi, Silvia Maine, Katarin Leppik, Krister Kruusmaa, Merilin Aruvee  

**Link**: [PDF](https://arxiv.org/pdf/2601.16314)  

**Abstract**: Large language models (LLMs) enable rapid and consistent automated evaluation of open-ended exam responses, including dimensions of content and argumentation that have traditionally required human judgment. This is particularly important in cases where a large amount of exams need to be graded in a limited time frame, such as nation-wide graduation exams in various countries. Here, we examine the applicability of automated scoring on two large datasets of trial exam essays of two full national cohorts from Estonia. We operationalize the official curriculum-based rubric and compare LLM and statistical natural language processing (NLP) based assessments with human panel scores. The results show that automated scoring can achieve performance comparable to that of human raters and tends to fall within the human scoring range. We also evaluate bias, prompt injection risks, and LLMs as essay writers. These findings demonstrate that a principled, rubric-driven, human-in-the-loop scoring pipeline is viable for high-stakes writing assessment, particularly relevant for digitally advanced societies like Estonia, which is about to adapt a fully electronic examination system. Furthermore, the system produces fine-grained subscore profiles that can be used to generate systematic, personalized feedback for instruction and exam preparation. The study provides evidence that LLM-assisted assessment can be implemented at a national scale, even in a small-language context, while maintaining human oversight and compliance with emerging educational and regulatory standards. 

---
# Teaching and Evaluating LLMs to Reason About Polymer Design Related Tasks 

**Authors**: Dikshya Mohanty, Mohammad Saqib Hasan, Syed Mostofa Monsur, Size Zheng, Benjamin Hsiao, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2601.16312)  

**Abstract**: Research in AI4Science has shown promise in many science applications, including polymer design. However, current LLMs prove ineffective on this problem space because: (i) most models lack polymer-specific knowledge (ii) existing aligned models lack coverage of knowledge and capabilities relevant to polymer design. Addressing this, we introduce PolyBench, a large scale training and test benchmark dataset of more than 125K polymer design related tasks, leveraging a knowledge base of 13M+ data points obtained from experimental and synthetic sources to ensure broad coverage of polymers and their properties. For effective alignment using PolyBench, we introduce a knowledge-augmented reasoning distillation method that augments this dataset with structured CoT. Furthermore, tasks in PolyBench are organized from simple to complex analytical reasoning problems, enabling generalization tests and diagnostic probes across the problem space. Experiments show that small language models (SLMs), of 7B to 14B parameters, trained on PolyBench data outperform similar sized models, and even closed source frontier LLMs on PolyBench test dataset while demonstrating gains on other polymer benchmarks as well. 

---
# A Longitudinal, Multinational, and Multilingual Corpus of News Coverage of the Russo-Ukrainian War 

**Authors**: Dikshya Mohanty, Taisiia Sabadyn, Jelwin Rodrigues, Chenlu Wang, Abhishek Kalugade, Ritwik Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2601.16309)  

**Abstract**: We introduce DNIPRO, a novel longitudinal corpus of 246K news articles documenting the Russo-Ukrainian war from Feb 2022 to Aug 2024, spanning eleven media outlets across five nation states (Russia, Ukraine, U.S., U.K., and China) and three languages (English, Russian, and Mandarin Chinese). This multilingual resource features consistent and comprehensive metadata, and multiple types of annotation with rigorous human evaluations for downstream tasks relevant to systematic transnational analyses of contentious wartime discourse. DNIPRO's distinctive value lies in its inclusion of competing geopolitical perspectives, making it uniquely suited for studying narrative divergence, media framing, and information warfare. To demonstrate its utility, we include use case experiments using stance detection, sentiment analysis, topical framing, and contradiction analysis of major conflict events within the larger war. Our explorations reveal how outlets construct competing realities, with coverage exhibiting polarized interpretations that reflect geopolitical interests. Beyond supporting computational journalism research, DNIPRO provides a foundational resource for understanding how conflicting narratives emerge and evolve across global information ecosystems. 

---
# Generating Literature-Driven Scientific Theories at Scale 

**Authors**: Peter Jansen, Peter Clark, Doug Downey, Daniel S. Weld  

**Link**: [PDF](https://arxiv.org/pdf/2601.16282)  

**Abstract**: Contemporary automated scientific discovery has focused on agents for generating scientific experiments, while systems that perform higher-level scientific activities such as theory building remain underexplored. In this work, we formulate the problem of synthesizing theories consisting of qualitative and quantitative laws from large corpora of scientific literature. We study theory generation at scale, using 13.7k source papers to synthesize 2.9k theories, examining how generation using literature-grounding versus parametric knowledge, and accuracy-focused versus novelty-focused generation objectives change theory properties. Our experiments show that, compared to using parametric LLM memory for generation, our literature-supported method creates theories that are significantly better at both matching existing evidence and at predicting future results from 4.6k subsequently-written papers 

---
# Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification 

**Authors**: Branislav Pecher, Jan Cegin, Robert Belanec, Ivan Srba, Jakub Simko, Maria Bielikova  

**Link**: [PDF](https://arxiv.org/pdf/2601.16278)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable multilingual capabilities, making them promising tools in both high- and low-resource languages. One particularly valuable use case is generating synthetic samples that can be used to train smaller models in low-resource scenarios where human-labelled data is scarce. In this work, we investigate whether these synthetic data generation capabilities can serve as a form of distillation, producing smaller models that perform on par with or even better than massive LLMs across languages and tasks. To this end, we use a state-of-the-art multilingual LLM to generate synthetic datasets covering 11 languages and 4 classification tasks. These datasets are then used to train smaller models via fine-tuning or instruction tuning, or as synthetic in-context examples for compact LLMs. Our experiments show that even small amounts of synthetic data enable smaller models to outperform the large generator itself, particularly in low-resource languages. Overall, the results suggest that LLMs are best utilised as generators (teachers) rather than classifiers, producing data that empowers smaller and more efficient multilingual models. 

---
# GameTalk: Training LLMs for Strategic Conversation 

**Authors**: Victor Conchello Vendrell, Max Ruiz Luyten, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2601.16276)  

**Abstract**: Strategic decision-making in multi-agent settings is a key challenge for large language models (LLMs), particularly when coordination and negotiation must unfold over extended conversations. While recent work has explored the use of LLMs in isolated decision tasks, little attention has been given to optimizing long-term objectives through dialogue. We introduce \textbf{GameTalk}, a framework for training LLMs to make strategic decisions via multi-turn interactions. Unlike prior work that focuses on single-turn objectives or static action prediction, we train LLMs to optimize a global objective across full conversations. We achieve this by adapting fine-tuning methods like GRPO, DPO, and STaR to incorporate reward signals that depend on the entire interaction. We evaluate this approach on a suite of increasingly complex games, designed to stress different aspects of reasoning, coordination, and opponent modeling. Our results show that GameTalk significantly outperforms untrained models, especially under reward shaping, with DPO consistently yielding the strongest gains. These findings position conversational fine-tuning as a promising path for LLMs to reason, negotiate, and act in interactive environments. 

---
# Limits of n-gram Style Control for LLMs via Logit-Space Injection 

**Authors**: Sami-ul Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2601.16224)  

**Abstract**: Large language models (LLMs) are typically personalized via prompt engineering or parameter-efficient fine-tuning such as LoRA. However, writing style can be difficult to distill into a single prompt, and LoRA fine-tuning requires computationally intensive training and infrastructure. We investigate a possible lightweight alternative: steering a frozen LLM with n-gram style priors injected in logit space at decoding time. We train an n-gram model on stylistically distinct corpora -- including Don Quixote, CNN/DailyMail news headlines, and arXiv abstracts -- constructing an interpolated 1-to-3-gram prior over next-token probabilities. During generation we modify the LLM's logits by adding a weighted sum of style log-probabilities from each n-gram order that matches the current context, scaled by a control parameter lambda in [0, 1].
We sweep lambda and style corpora and report style perplexity under the n-gram model, base-model perplexity as a proxy for fluency, Jensen-Shannon (JS) divergence between the original and steered token distributions, and token-overlap statistics. On TinyLlama-1.1B we identify a single narrow regime (for the Don Quixote corpus at lambda=0.1) where style perplexity improves by 24.7% and base-model perplexity improves by 51.4% relative to the frozen model. Outside this regime, and for multi-author corpora such as CNN/DailyMail and arXiv abstracts, even small nonzero lambda values generally result in worse style and fluency, and larger lambda values lead to collapse with extreme perplexities and incoherent text. Logit-space injection of n-gram style priors provides lightweight, tunable style control, but it is fragile: it operates effectively only within a narrow range of low lambda values and is consistently outperformed by prompting and LoRA. 

---
# Towards Latent Diffusion Suitable For Text 

**Authors**: Nesta Midavaine, Christian A. Naesseth, Grigory Bartosh  

**Link**: [PDF](https://arxiv.org/pdf/2601.16220)  

**Abstract**: Language diffusion models aim to improve sampling speed and coherence over autoregressive LLMs. We introduce Neural Flow Diffusion Models for language generation, an extension of NFDM that enables the straightforward application of continuous diffusion models to discrete state spaces. NFDM learns a multivariate forward process from the data, ensuring that the forward process and generative trajectory are a good fit for language modeling. Our model substantially reduces the likelihood gap with autoregressive models of the same size, while achieving sample quality comparable to that of previous latent diffusion models. 

---
# Domain Specific Specialization in Low-Resource Settings: The Efficacy of Offline Response-Based Knowledge Distillation in Large Language Models 

**Authors**: Erdem Aslan, Pakize Erdoğmuş  

**Link**: [PDF](https://arxiv.org/pdf/2601.16219)  

**Abstract**: Large Language Models (LLMs) excel in general tasks but often struggle with hallucinations when handling domain-specific or institutional knowledge absent from their pre-training. We present an offline response-based knowledge distillation method that develops high-accuracy specialized assistants under constrained hardware resources. We evaluate three distinct data strategies: general domain adaptation (15,000 lines), unstructured knowledge injection (2,000 lines), and a context-aware synthetic dataset (500 lines) generated by a teacher model. To minimize computational costs, we utilize the Unsloth library to optimize the Qwen-2.5-7B student model, reducing NVIDIA A100 GPU memory requirements from 40 GB to 16 GB. Experimental results demonstrate that while larger unstructured datasets suffer from persistent hallucinations, the 500-line context-aware dataset achieves a 96.7% accuracy rate and robust rejection capability. These findings validate the LIMA hypothesis, showing that data quality and structural alignment are more critical than quantity for domain adaptation in low-resource settings. 

---
# M3Kang: Evaluating Multilingual Multimodal Mathematical Reasoning in Vision-Language Models 

**Authors**: Aleix Torres-Camps, Nathaniel Mitrani Hadida, Víctor Conchello Vendrell, Àlex Batlle Casellas, Arnau Padrés Masdemont, Jordi Ros-Giralt  

**Link**: [PDF](https://arxiv.org/pdf/2601.16218)  

**Abstract**: Despite state-of-the-art vision-language models (VLMs) have demonstrated strong reasoning capabilities, their performance in multilingual mathematical reasoning remains underexplored, particularly when compared to human performance. To bridge this gap, we introduce M3Kang, the first massively multilingual, multimodal mathematical reasoning dataset for VLMs. It is derived from the Kangaroo Math Competition, the world's largest mathematics contest, which annually engages over six million participants under the age of 18 across more than 90 countries. M3Kang includes 1,747 unique multiple-choice problems organized by grade-level difficulty, with translations into 108 culturally diverse languages, some of them including diagrams essential for solving them. Using this dataset, we conduct extensive benchmarking on both closed- and open-source SOTA models. We observe that, despite recent advances, models still struggle with basic math and diagram-based reasoning, with performance scaling with language presence and model size, but not with grade level. We also find that multilingual techniques can be effectively extended to the multimodal setting, resulting in significant improvements over baseline approaches. Our analysis also incorporates performance data from over 68,000 students, enabling direct comparison with human performance. We are open-sourcing M3Kang, including the English-only subset M2Kang, along with the framework and codebase used to construct the dataset. 

---
# ChiEngMixBench: Evaluating Large Language Models on Spontaneous and Natural Chinese-English Code-Mixed Generation 

**Authors**: Qingyan Yang, Tongxi Wang, Yunsheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2601.16217)  

**Abstract**: Code-mixing is increasingly prevalent in interactions between humans and large language models, yet existing work often reduces it to a translation or convertibility problem, making it difficult to assess whether a model's switching behavior is context-appropriate and aligned with human conventions. We introduce ChiEngMixBench, the first benchmark designed to evaluate code-mixing ability in authentic community contexts, built upon a general construction pipeline that enables scalable dataset development across domains and bilingual pairs. ChiEngMixBench formulates code-mixing as a cognitive alignment problem, characterized by two complementary signals: Spontaneity and Naturalness. Empirical evaluation shows that our metrics can systematically distinguish code-mixing performance across models. Beyond benchmarking, we further uncover an implicitly emergent Terminology Layering Strategy, a phenomenon consistent with the Matrix Language Frame (MLF) theory, indicating structured cognitive alignment between multilingual large language models and human communication. 

---
# Reasoning Promotes Robustness in Theory of Mind Tasks 

**Authors**: Ian B. de Haan, Peter van der Putten, Max van Duijn  

**Link**: [PDF](https://arxiv.org/pdf/2601.16853)  

**Abstract**: Large language models (LLMs) have recently shown strong performance on Theory of Mind (ToM) tests, prompting debate about the nature and true performance of the underlying capabilities. At the same time, reasoning-oriented LLMs trained via reinforcement learning with verifiable rewards (RLVR) have achieved notable improvements across a range of benchmarks. This paper examines the behavior of such reasoning models in ToM tasks, using novel adaptations of machine psychological experiments and results from established benchmarks. We observe that reasoning models consistently exhibit increased robustness to prompt variations and task perturbations. Our analysis indicates that the observed gains are more plausibly attributed to increased robustness in finding the correct solution, rather than to fundamentally new forms of ToM reasoning. We discuss the implications of this interpretation for evaluating social-cognitive behavior in LLMs. 

---
# ColorConceptBench: A Benchmark for Probabilistic Color-Concept Understanding in Text-to-Image Models 

**Authors**: Chenxi Ruan, Yu Xiao, Yihan Hou, Guosheng Hu, Wei Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.16836)  

**Abstract**: While text-to-image (T2I) models have advanced considerably, their capability to associate colors with implicit concepts remains underexplored. To address the gap, we introduce ColorConceptBench, a new human-annotated benchmark to systematically evaluate color-concept associations through the lens of probabilistic color distributions. ColorConceptBench moves beyond explicit color names or codes by probing how models translate 1,281 implicit color concepts using a foundation of 6,369 human annotations. Our evaluation of seven leading T2I models reveals that current models lack sensitivity to abstract semantics, and crucially, this limitation appears resistant to standard interventions (e.g., scaling and guidance). This demonstrates that achieving human-like color semantics requires more than larger models, but demands a fundamental shift in how models learn and represent implicit meaning. 

---
# SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents 

**Authors**: Yuhang Wang, Yuling Shi, Mo Yang, Rongrui Zhang, Shilin He, Heng Lian, Yuting Chen, Siyu Ye, Kai Cai, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16746)  

**Abstract**: LLM agents have demonstrated remarkable capabilities in software development, but their performance is hampered by long interaction contexts, which incur high API costs and latency. While various context compression approaches such as LongLLMLingua have emerged to tackle this challenge, they typically rely on fixed metrics such as PPL, ignoring the task-specific nature of code understanding. As a result, they frequently disrupt syntactic and logical structure and fail to retain critical implementation details. In this paper, we propose SWE-Pruner, a self-adaptive context pruning framework tailored for coding agents. Drawing inspiration from how human programmers "selectively skim" source code during development and debugging, SWE-Pruner performs task-aware adaptive pruning for long contexts. Given the current task, the agent formulates an explicit goal (e.g., "focus on error handling") as a hint to guide the pruning targets. A lightweight neural skimmer (0.6B parameters) is trained to dynamically select relevant lines from the surrounding context given the goal. Evaluations across four benchmarks and multiple models validate SWE-Pruner's effectiveness in various scenarios, achieving 23-54% token reduction on agent tasks like SWE-Bench Verified and up to 14.84x compression on single-turn tasks like LongCodeQA with minimal performance impact. 

---
# A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics 

**Authors**: Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.16531)  

**Abstract**: We investigate whether high-frequency key collisions are a primary bottleneck in Engram-style conditional memory. To isolate the effect of collisions, we introduce Engram-Nine, a collision-free hot-tier extension that maps the most frequent n-grams through a Minimal Perfect Hash Function (MPHF) while retaining the original multi-head hashed lookup as a cold tier. Under a strictly iso-parameter setup, the collision-free design does not consistently improve validation loss.
Through route-stratified evaluation (decomposing per-token loss into hot/cold contributions), we uncover a consistent "hot-to-cold advantage flip" during training: hot (high-frequency) positions initially have lower loss, but cold positions eventually surpass them. Crucially, collision-free configurations flip earlier than collision-prone baselines, suggesting that collisions act as implicit regularization. We also identify a gating mismatch: the gate learns to favor hot positions early in training, but this preference persists even after the flip, assigning higher weights to positions with higher loss.
Our findings suggest that improving lookup precision alone does not guarantee better training outcomes. The dominant limitation may lie in gating credit assignment rather than index accuracy, and collision-induced noise may provide beneficial regularization that should not be naively eliminated. 

---
# Beyond Superficial Unlearning: Sharpness-Aware Robust Erasure of Hallucinations in Multimodal LLMs 

**Authors**: Xianya Fang, Feiyang Ren, Xiang Chen, Yu Tian, Zhen Bi, Haiyang Yu, Sheng-Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.16527)  

**Abstract**: Multimodal LLMs are powerful but prone to object hallucinations, which describe non-existent entities and harm reliability. While recent unlearning methods attempt to mitigate this, we identify a critical flaw: structural fragility. We empirically demonstrate that standard erasure achieves only superficial suppression, trapping the model in sharp minima where hallucinations catastrophically resurge after lightweight relearning. To ensure geometric stability, we propose SARE, which casts unlearning as a targeted min-max optimization problem and uses a Targeted-SAM mechanism to explicitly flatten the loss landscape around hallucinated concepts. By suppressing hallucinations under simulated worst-case parameter perturbations, our framework ensures robust removal stable against weight shifts. Extensive experiments demonstrate that SARE significantly outperforms baselines in erasure efficacy while preserving general generation quality. Crucially, it maintains persistent hallucination suppression against relearning and parameter updates, validating the effectiveness of geometric stabilization. 

---
# TangramPuzzle: Evaluating Multimodal Large Language Models with Compositional Spatial Reasoning 

**Authors**: Daixian Liu, Jiayi Kuang, Yinghui Li, Yangning Li, Di Yin, Haoyu Cao, Xing Sun, Ying Shen, Hai-Tao Zheng, Liang Lin, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16520)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable progress in visual recognition and semantic understanding. Nevertheless, their ability to perform precise compositional spatial reasoning remains largely unexplored. Existing benchmarks often involve relatively simple tasks and rely on semantic approximations or coarse relative positioning, while their evaluation metrics are typically limited and lack rigorous mathematical formulations. To bridge this gap, we introduce TangramPuzzle, a geometry-grounded benchmark designed to evaluate compositional spatial reasoning through the lens of the classic Tangram game. We propose the Tangram Construction Expression (TCE), a symbolic geometric framework that grounds tangram assemblies in exact, machine-verifiable coordinate specifications, to mitigate the ambiguity of visual approximation. We design two complementary tasks: Outline Prediction, which demands inferring global shapes from local components, and End-to-End Code Generation, which requires solving inverse geometric assembly problems. We conduct extensive evaluation experiments on advanced open-source and proprietary models, revealing an interesting insight: MLLMs tend to prioritize matching the target silhouette while neglecting geometric constraints, leading to distortions or deformations of the pieces. 

---
# EvoConfig: Self-Evolving Multi-Agent Systems for Efficient Autonomous Environment Configuration 

**Authors**: Xinshuai Guo, Jiayi Kuang, Linyue Pan, Yinghui Li, Yangning Li, Hai-Tao Zheng, Ying Shen, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.16489)  

**Abstract**: A reliable executable environment is the foundation for ensuring that large language models solve software engineering tasks. Due to the complex and tedious construction process, large-scale configuration is relatively inefficient. However, most methods always overlook fine-grained analysis of the actions performed by the agent, making it difficult to handle complex errors and resulting in configuration failures. To address this bottleneck, we propose EvoConfig, an efficient environment configuration framework that optimizes multi-agent collaboration to build correct runtime environments. EvoConfig features an expert diagnosis module for fine-grained post-execution analysis, and a self-evolving mechanism that lets expert agents self-feedback and dynamically adjust error-fixing priorities in real time. Empirically, EvoConfig matches the previous state-of-the-art Repo2Run on Repo2Run's 420 repositories, while delivering clear gains on harder cases: on the more challenging Envbench, EvoConfig achieves a 78.1% success rate, outperforming Repo2Run by 7.1%. Beyond end-to-end success, EvoConfig also demonstrates stronger debugging competence, achieving higher accuracy in error identification and producing more effective repair recommendations than existing methods. 

---
# Endless Terminals: Scaling RL Environments for Terminal Agents 

**Authors**: Kanishk Gandhi, Shivam Garg, Noah D. Goodman, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2601.16443)  

**Abstract**: Environments are the bottleneck for self-improving agents. Current terminal benchmarks were built for evaluation, not training; reinforcement learning requires a scalable pipeline, not just a dataset. We introduce Endless Terminals, a fully autonomous pipeline that procedurally generates terminal-use tasks without human annotation. The pipeline has four stages: generating diverse task descriptions, building and validating containerized environments, producing completion tests, and filtering for solvability. From this pipeline we obtain 3255 tasks spanning file operations, log management, data processing, scripting, and database operations. We train agents using vanilla PPO with binary episode level rewards and a minimal interaction loop: no retrieval, multi-agent coordination, or specialized tools. Despite this simplicity, models trained on Endless Terminals show substantial gains: on our held-out dev set, Llama-3.2-3B improves from 4.0% to 18.2%, Qwen2.5-7B from 10.7% to 53.3%, and Qwen3-8B-openthinker-sft from 42.6% to 59.0%. These improvements transfer to human-curated benchmarks: models trained on Endless Terminals show substantial gains on held out human curated benchmarks: on TerminalBench 2.0, Llama-3.2-3B improves from 0.0% to 2.2%, Qwen2.5-7B from 2.2% to 3.4%, and Qwen3-8B-openthinker-sft from 1.1% to 6.7%, in each case outperforming alternative approaches including models with more complex agentic scaffolds. These results demonstrate that simple RL succeeds when environments scale. 

---
# White-Box Sensitivity Auditing with Steering Vectors 

**Authors**: Hannah Cyberey, Yangfeng Ji, David Evans  

**Link**: [PDF](https://arxiv.org/pdf/2601.16398)  

**Abstract**: Algorithmic audits are essential tools for examining systems for properties required by regulators or desired by operators. Current audits of large language models (LLMs) primarily rely on black-box evaluations that assess model behavior only through input-output testing. These methods are limited to tests constructed in the input space, often generated by heuristics. In addition, many socially relevant model properties (e.g., gender bias) are abstract and difficult to measure through text-based inputs alone. To address these limitations, we propose a white-box sensitivity auditing framework for LLMs that leverages activation steering to conduct more rigorous assessments through model internals. Our auditing method conducts internal sensitivity tests by manipulating key concepts relevant to the model's intended function for the task. We demonstrate its application to bias audits in four simulated high-stakes LLM decision tasks. Our method consistently reveals substantial dependence on protected attributes in model predictions, even in settings where standard black-box evaluations suggest little or no bias. Our code is openly available at this https URL 

---
# Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Contextually Important Moments 

**Authors**: Aditya K Surikuchi, Raquel Fernández, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2601.16333)  

**Abstract**: Foundation models are used for many real-world applications involving language generation from temporally-ordered multimodal events. In this work, we study the ability of models to identify the most important sub-events in a video, which is a fundamental prerequisite for narrating or summarizing multimodal events. Specifically, we focus on football games and evaluate models on their ability to distinguish between important and non-important sub-events in a game. To this end, we construct a new dataset by leveraging human preferences for importance implicit in football game highlight reels, without any additional annotation costs. Using our dataset, which we will publicly release to the community, we compare several state-of-the-art multimodal models and show that they are not far from chance level performance. Analyses of models beyond standard evaluation metrics reveal their tendency to rely on a single dominant modality and their ineffectiveness in synthesizing necessary information from multiple sources. Our findings underline the importance of modular architectures that can handle sample-level heterogeneity in multimodal data and the need for complementary training procedures that can maximize cross-modal synergy. 

---
# EdgeSpot: Efficient and High-Performance Few-Shot Model for Keyword Spotting 

**Authors**: Oguzhan Buyuksolak, Alican Gok, Osman Erman Okman  

**Link**: [PDF](https://arxiv.org/pdf/2601.16316)  

**Abstract**: We introduce an efficient few-shot keyword spotting model for edge devices, EdgeSpot, that pairs an optimized version of a BC-ResNet-based acoustic backbone with a trainable Per-Channel Energy Normalization frontend and lightweight temporal self-attention. Knowledge distillation is utilized during training by employing a self-supervised teacher model, optimized with Sub-center ArcFace loss. This study demonstrates that the EdgeSpot model consistently provides better accuracy at a fixed false-alarm rate (FAR) than strong BC-ResNet baselines. The largest variant, EdgeSpot-4, improves the 10-shot accuracy at 1% FAR from 73.7% to 82.0%, which requires only 29.4M MACs with 128k parameters. 

---
# SoundBreak: A Systematic Study of Audio-Only Adversarial Attacks on Trimodal Models 

**Authors**: Aafiya Hussain, Gaurav Srivastava, Alvi Ishmam, Zaber Hakim, Chris Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2601.16231)  

**Abstract**: Multimodal foundation models that integrate audio, vision, and language achieve strong performance on reasoning and generation tasks, yet their robustness to adversarial manipulation remains poorly understood. We study a realistic and underexplored threat model: untargeted, audio-only adversarial attacks on trimodal audio-video-language models. We analyze six complementary attack objectives that target different stages of multimodal processing, including audio encoder representations, cross-modal attention, hidden states, and output likelihoods. Across three state-of-the-art models and multiple benchmarks, we show that audio-only perturbations can induce severe multimodal failures, achieving up to 96% attack success rate. We further show that attacks can be successful at low perceptual distortions (LPIPS <= 0.08, SI-SNR >= 0) and benefit more from extended optimization than increased data scale. Transferability across models and encoders remains limited, while speech recognition systems such as Whisper primarily respond to perturbation magnitude, achieving >97% attack success under severe distortion. These results expose a previously overlooked single-modality attack surface in multimodal systems and motivate defenses that enforce cross-modal consistency. 

---
# Zero-Shot Speech LLMs for Multi-Aspect Evaluation of L2 Speech: Challenges and Opportunities 

**Authors**: Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2601.16230)  

**Abstract**: An accurate assessment of L2 English pronunciation is crucial for language learning, as it provides personalized feedback and ensures a fair evaluation of individual progress. However, automated scoring remains challenging due to the complexity of sentence-level fluency, prosody, and completeness. This paper evaluates the zero-shot performance of Qwen2-Audio-7B-Instruct, an instruction-tuned speech-LLM, on 5,000 Speechocean762 utterances. The model generates rubric-aligned scores for accuracy, fluency, prosody, and completeness, showing strong agreement with human ratings within +-2 tolerance, especially for high-quality speech. However, it tends to overpredict low-quality speech scores and lacks precision in error detection. These findings demonstrate the strong potential of speech LLMs in scalable pronunciation assessment and suggest future improvements through enhanced prompting, calibration, and phonetic integration to advance Computer-Assisted Pronunciation Training. 

---
