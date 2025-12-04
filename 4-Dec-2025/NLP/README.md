# SkillFactory: Self-Distillation For Learning Cognitive Behaviors 

**Authors**: Zayne Sprague, Jack Lu, Manya Wadhwa, Sedrick Keh, Mengye Ren, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2512.04072)  

**Abstract**: Reasoning models leveraging long chains of thought employ various cognitive skills, such as verification of their answers, backtracking, retrying by an alternate method, and more. Previous work has shown that when a base language model exhibits these skills, training that model further with reinforcement learning (RL) can learn to leverage them. How can we get models to leverage skills that aren't exhibited by base models? Our work, SkillFactory, is a method for fine-tuning models to roughly learn these skills during a supervised fine-tuning (SFT) stage prior to RL. Our approach does not rely on distillation from a stronger model, but instead uses samples from the model itself, rearranged to provide training data in the format of those skills. These "silver" SFT traces may be imperfect, but are nevertheless effective for priming a model to acquire skills during RL. Our evaluation shows that (1) starting from SkillFactory SFT initialization helps a model to generalize to harder variants of a task post-RL, despite lower performance pre-RL; (2) cognitive skills are indeed used by the model; (3) RLed SkillFactory models are more robust to regression on out-of-domain tasks than RLed base models. Our work suggests that inductive biases learned prior to RL help models learn robust cognitive skill use. 

---
# Jina-VLM: Small Multilingual Vision Language Model 

**Authors**: Andreas Koukounas, Georgios Mastrapas, Florian Hönicke, Sedigheh Eslami, Guillaume Roncari, Scott Martens, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.04032)  

**Abstract**: We present Jina-VLM, a 2.4B parameter vision-language model that achieves state-of-the-art multilingual visual question answering among open 2B-scale VLMs. The model couples a SigLIP2 vision encoder with a Qwen3 language backbone through an attention-pooling connector that enables token-efficient processing of arbitrary-resolution images. Across standard VQA benchmarks and multilingual evaluations, Jina-VLM outperforms comparable models while preserving competitive text-only performance. 

---
# AugServe: Adaptive Request Scheduling for Augmented Large Language Model Inference Serving 

**Authors**: Ying Wang, Zhen Jin, Jiexiong Xu, Wenhai Lin, Yiquan Chen, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.04013)  

**Abstract**: As augmented large language models (LLMs) with external tools become increasingly popular in web applications, improving augmented LLM inference serving efficiency and optimizing service-level objectives (SLOs) are critical for enhancing user experience. To achieve this, inference systems must maximize request handling within latency constraints, referred to as increasing effective throughput. However, existing systems face two major challenges: (i) reliance on first-come-first-served (FCFS) scheduling causes severe head-of-line blocking, leading to queuing delays exceeding the SLOs for many requests; and (ii) static batch token limit, which fails to adapt to fluctuating loads and hardware conditions. Both of these factors degrade effective throughput and service quality.
This paper presents AugServe, an efficient inference framework designed to reduce queueing latency and enhance effective throughput for augmented LLM inference services. The core idea of AugServe is a two-stage adaptive request scheduling strategy. Specifically, AugServe combines the inference features of augmented LLM requests to optimize the order of scheduling decisions (stage I). These decisions are continuously refined with runtime information (stage II), adapting to both request characteristics and system capabilities. In addition, AugServe dynamically adjusts the token batching mechanism based on hardware status and real-time load, further enhancing throughput performance. Experimental results show that AugServe achieves 4.7-33.1x and 3.3-13.2x higher effective throughput than vLLM and InferCept, while reducing time-to-first-token (TTFT) by up to 96.3% and 95.0%, respectively. 

---
# Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models 

**Authors**: Taido Purason, Pavel Chizhov, Ivan P. Yamshchikov, Mark Fishel  

**Link**: [PDF](https://arxiv.org/pdf/2512.03989)  

**Abstract**: Tokenizer adaptation plays an important role in transferring pre-trained language models to new domains or languages. In this work, we address two complementary aspects of this process: vocabulary extension and pruning. The common approach to extension trains a new tokenizer on domain-specific text and appends the tokens that do not overlap with the existing vocabulary, which often results in many tokens that are unreachable or never used. We propose continued BPE training, which adapts a pre-trained tokenizer by continuing the BPE merge learning process on new data. Experiments across multiple languages and model families show that this approach improves tokenization efficiency and leads to better utilization of added vocabulary. We also introduce leaf-based vocabulary pruning, which removes redundant tokens while preserving model quality. Together, these methods provide practical tools for controlled vocabulary modification, which we release as an open-source package. 

---
# Adapting Large Language Models to Low-Resource Tibetan: A Two-Stage Continual and Supervised Fine-Tuning Study 

**Authors**: Lifeng Chen, Ryan Lai, Tianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03976)  

**Abstract**: Adapting large language models (LLMs) to low-resource languages remains a major challenge due to data scarcity and cross-lingual drift. This work presents a two-stage adaptation of Qwen2.5-3B to Tibetan, a morphologically rich and underrepresented language. We employ Continual Pretraining (CPT) to establish Tibetan linguistic grounding, followed by Supervised Fine-Tuning (SFT) for task and translation specialization. Empirical evaluations demonstrate a consistent decrease in perplexity (from 2.98 $\rightarrow$ 1.54) and substantial improvements in Chinese$\rightarrow$Tibetan translation quality (BLEU: 0.046 $\rightarrow$ 0.261; chrF: 2.2 $\rightarrow$ 6.6). Layer-wise analysis across 435 layers in Qwen3-4B reveals that adaptation primarily concentrates on embedding and output heads, with mid--late MLP projections encoding domain-specific transformations. Our findings suggest that CPT constructs a Tibetan semantic manifold while SFT sharpens task alignment with minimal representational disruption. This study provides the first quantitative exploration of Tibetan adaptation dynamics for LLMs, and offers an open, reproducible framework for extending multilingual foundation models to low-resource settings. 

---
# Is Lying Only Sinful in Islam? Exploring Religious Bias in Multilingual Large Language Models Across Major Religions 

**Authors**: Kazi Abrab Hossain, Jannatul Somiya Mahmud, Maria Hossain Tuli, Anik Mitra, S. M. Taiabul Haque, Farig Y. Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2512.03943)  

**Abstract**: While recent developments in large language models have improved bias detection and classification, sensitive subjects like religion still present challenges because even minor errors can result in severe misunderstandings. In particular, multilingual models often misrepresent religions and have difficulties being accurate in religious contexts. To address this, we introduce BRAND: Bilingual Religious Accountable Norm Dataset, which focuses on the four main religions of South Asia: Buddhism, Christianity, Hinduism, and Islam, containing over 2,400 entries, and we used three different types of prompts in both English and Bengali. Our results indicate that models perform better in English than in Bengali and consistently display bias toward Islam, even when answering religion-neutral questions. These findings highlight persistent bias in multilingual models when similar questions are asked in different languages. We further connect our findings to the broader issues in HCI regarding religion and spirituality. 

---
# BERnaT: Basque Encoders for Representing Natural Textual Diversity 

**Authors**: Ekhi Azurmendi, Joseba Fernandez de Landa, Jaione Bengoetxea, Maite Heredia, Julen Etxaniz, Mikel Zubillaga, Ander Soraluze, Aitor Soroa  

**Link**: [PDF](https://arxiv.org/pdf/2512.03903)  

**Abstract**: Language models depend on massive text corpora that are often filtered for quality, a process that can unintentionally exclude non-standard linguistic varieties, reduce model robustness and reinforce representational biases. In this paper, we argue that language models should aim to capture the full spectrum of language variation (dialectal, historical, informal, etc.) rather than relying solely on standardized text. Focusing on Basque, a morphologically rich and low-resource language, we construct new corpora combining standard, social media, and historical sources, and pre-train the BERnaT family of encoder-only models in three configurations: standard, diverse, and combined. We further propose an evaluation framework that separates Natural Language Understanding (NLU) tasks into standard and diverse subsets to assess linguistic generalization. Results show that models trained on both standard and diverse data consistently outperform those trained on standard corpora, improving performance across all task types without compromising standard benchmark accuracy. These findings highlight the importance of linguistic diversity in building inclusive, generalizable language models. 

---
# Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers 

**Authors**: Hongzhan Lin, Zhiqi Bai, Xinmiao Zhang, Sen Yang, Xiang Li, Siran Yang, Yunlong Xu, Jiaheng Liu, Yongchi Zhao, Jiamang Wang, Yuchi Xu, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03870)  

**Abstract**: Transformer decoders have achieved strong results across tasks, but the memory required for the KV cache becomes prohibitive at long sequence lengths. Although Cross-layer KV Cache sharing (e.g., YOCO, CLA) offers a path to mitigate KV Cache bottleneck, it typically underperforms within-layer methods like GQA. To understand the root cause, we investigate the information flow of keys and values of the top-layers. Our preliminary reveals a clear distribution: values are predominantly derived from the bottom layer, while keys draw more information from both bottom and middle layers. Building upon this, we propose FusedKV, whose top-layer KV caches are a learnable fusion of the most informative ones from the bottom and middle layers. This fusion operates directly on post-RoPE keys, preserving relative positional information without the computational cost of re-applying rotary embeddings. To further improve efficiency, we propose FusedKV-Lite, an cross-layer sharing approach, where top-layer KV caches are directly derived from the bottom-layer values and the middle-layer keys. Compared to FusedKV, FusedKV-Lite reduces I/O overhead at the cost of a slight increase in perplexity. In experiments on LLMs ranging from 332M to 4B parameters, our proposed method reduce 50\% cache memory while achieving lower validation perplexity than the standard Transformer decoder, establishing it as a memory-efficient, high-performance architectural alternative. 

---
# Training and Evaluation of Guideline-Based Medical Reasoning in LLMs 

**Authors**: Michael Staniek, Artem Sokolov, Stefan Riezler  

**Link**: [PDF](https://arxiv.org/pdf/2512.03838)  

**Abstract**: Machine learning for early prediction in medicine has recently shown breakthrough performance, however, the focus on improving prediction accuracy has led to a neglect of faithful explanations that are required to gain the trust of medical practitioners. The goal of this paper is to teach LLMs to follow medical consensus guidelines step-by-step in their reasoning and prediction process. Since consensus guidelines are ubiquitous in medicine, instantiations of verbalized medical inference rules to electronic health records provide data for fine-tuning LLMs to learn consensus rules and possible exceptions thereof for many medical areas. Consensus rules also enable an automatic evaluation of the model's inference process regarding its derivation correctness (evaluating correct and faithful deduction of a conclusion from given premises) and value correctness (comparing predicted values against real-world measurements). We exemplify our work using the complex Sepsis-3 consensus definition. Our experiments show that small fine-tuned models outperform one-shot learning of considerably larger LLMs that are prompted with the explicit definition and models that are trained on medical texts including consensus definitions. Since fine-tuning on verbalized rule instantiations of a specific medical area yields nearly perfect derivation correctness for rules (and exceptions) on unseen patient data in that area, the bottleneck for early prediction is not out-of-distribution generalization, but the orthogonal problem of generalization into the future by forecasting sparsely and irregularly sampled clinical variables. We show that the latter results can be improved by integrating the output representations of a time series forecasting model with the LLM in a multimodal setup. 

---
# Improving Alignment Between Human and Machine Codes: An Empirical Assessment of Prompt Engineering for Construct Identification in Psychology 

**Authors**: Kylie L. Anglin, Stephanie Milan, Brittney Hernandez, Claudia Ventura  

**Link**: [PDF](https://arxiv.org/pdf/2512.03818)  

**Abstract**: Due to their architecture and vast pre-training data, large language models (LLMs) demonstrate strong text classification performance. However, LLM output - here, the category assigned to a text - depends heavily on the wording of the prompt. While literature on prompt engineering is expanding, few studies focus on classification tasks, and even fewer address domains like psychology, where constructs have precise, theory-driven definitions that may not be well represented in pre-training data. We present an empirical framework for optimizing LLM performance for identifying constructs in texts via prompt engineering. We experimentally evaluate five prompting strategies --codebook-guided empirical prompt selection, automatic prompt engineering, persona prompting, chain-of-thought reasoning, and explanatory prompting - with zero-shot and few-shot classification. We find that persona, chain-of-thought, and explanations do not fully address performance loss accompanying a badly worded prompt. Instead, the most influential features of a prompt are the construct definition, task framing, and, to a lesser extent, the examples provided. Across three constructs and two models, the classifications most aligned with expert judgments resulted from a few-shot prompt combining codebook-guided empirical prompt selection with automatic prompt engineering. Based on our findings, we recommend that researchers generate and evaluate as many prompt variants as feasible, whether human-crafted, automatically generated, or ideally both, and select prompts and examples based on empirical performance in a training dataset, validating the final approach in a holdout set. This procedure offers a practical, systematic, and theory-driven method for optimizing LLM prompts in settings where alignment with expert judgment is critical. 

---
# Enhancing Instruction-Following Capabilities in Seq2Seq Models: DoLA Adaptations for T5 

**Authors**: Huey Sun, Anabel Yong, Lorenzo Gilly, Felipe Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03803)  

**Abstract**: Contrastive decoding is a lightweight and effective inference-time method that improves the quality of text generation in Large Language Models. However, algorithms such as DoLa (Decoding by Contrastive Layers) have only been implemented in decoder-only architectures and studied for their impact on improving factuality. This work adapts DoLa for the T5 and FLAN-T5 model families and evaluates its impact on the models' instruction following capabilities, which to our knowledge is the first implementation of a contrastive decoding strategy in an encoder-decoder architecture. Our results show that DoLa improves the faithfulness of text generation for certain categories of tasks and harms others. To understand these results, we present a layer-by-layer analysis of logit evolution in a FLAN-T5 model to quantify DoLa's impact on token output probabilities. 

---
# In-Context Representation Hijacking 

**Authors**: Itay Yona, Amir Sarid, Michael Karasik, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2512.03771)  

**Abstract**: We introduce \textbf{Doublespeak}, a simple \emph{in-context representation hijacking} attack against large language models (LLMs). The attack works by systematically replacing a harmful keyword (e.g., \textit{bomb}) with a benign token (e.g., \textit{carrot}) across multiple in-context examples, provided a prefix to a harmful request. We demonstrate that this substitution leads to the internal representation of the benign token converging toward that of the harmful one, effectively embedding the harmful semantics under a euphemism. As a result, superficially innocuous prompts (e.g., ``How to build a carrot?'') are internally interpreted as disallowed instructions (e.g., ``How to build a bomb?''), thereby bypassing the model's safety alignment. We use interpretability tools to show that this semantic overwrite emerges layer by layer, with benign meanings in early layers converging into harmful semantics in later ones. Doublespeak is optimization-free, broadly transferable across model families, and achieves strong success rates on closed-source and open-source systems, reaching 74\% ASR on Llama-3.3-70B-Instruct with a single-sentence context override. Our findings highlight a new attack surface in the latent space of LLMs, revealing that current alignment strategies are insufficient and should instead operate at the representation level. 

---
# Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective 

**Authors**: Jingyang Ou, Jiaqi Han, Minkai Xu, Shaoxuan Xu, Jianwen Xie, Stefano Ermon, Yi Wu, Chongxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.03759)  

**Abstract**: Reinforcement Learning (RL) has proven highly effective for autoregressive language models, but adapting these methods to diffusion large language models (dLLMs) presents fundamental challenges. The core difficulty lies in likelihood approximation: while autoregressive models naturally provide token-level conditional probabilities essential for token-level RL objectives (e.g., GRPO), dLLMs generate sequences through iterative non-autoregressive denoising steps that lack this factorization. To address this fundamental mismatch, we propose ELBO-based Sequence-level Policy Optimization (ESPO), a principled RL framework that treats entire sequence generation as a single action and uses the ELBO as a tractable sequence-level likelihood proxy. Our method incorporates per-token normalization of importance ratios and robust KL-divergence estimation to ensure stable large-scale training. Extensive experiments on mathematical reasoning, coding, and planning tasks demonstrate that ESPO significantly outperforms token-level baselines, achieving dramatic improvements of 20-40 points on the Countdown task, while maintaining consistent gains on math and coding benchmarks. Our approach establishes sequence-level optimization as a principled and empirically effective paradigm for RL in dLLMs. Our code is available at this https URL. 

---
# AR-Med: Automated Relevance Enhancement in Medical Search via LLM-Driven Information Augmentation 

**Authors**: Chuyue Wang, Jie Feng, Yuxi Wu, Hang Zhang, Zhiguo Fan, Bing Cheng, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03737)  

**Abstract**: Accurate and reliable search on online healthcare platforms is critical for user safety and service efficacy. Traditional methods, however, often fail to comprehend complex and nuanced user queries, limiting their effectiveness. Large language models (LLMs) present a promising solution, offering powerful semantic understanding to bridge this gap. Despite their potential, deploying LLMs in this high-stakes domain is fraught with challenges, including factual hallucinations, specialized knowledge gaps, and high operational costs. To overcome these barriers, we introduce \textbf{AR-Med}, a novel framework for \textbf{A}utomated \textbf{R}elevance assessment for \textbf{Med}ical search that has been successfully deployed at scale on the Online Medical Delivery Platforms. AR-Med grounds LLM reasoning in verified medical knowledge through a retrieval-augmented approach, ensuring high accuracy and reliability. To enable efficient online service, we design a practical knowledge distillation scheme that compresses large teacher models into compact yet powerful student models. We also introduce LocalQSMed, a multi-expert annotated benchmark developed to guide model iteration and ensure strong alignment between offline and online performance. Extensive experiments show AR-Med achieves an offline accuracy of over 93\%, a 24\% absolute improvement over the original online system, and delivers significant gains in online relevance and user satisfaction. Our work presents a practical and scalable blueprint for developing trustworthy, LLM-powered systems in real-world healthcare applications. 

---
# DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking in Long-Context Dialogue 

**Authors**: Yijun Liao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03704)  

**Abstract**: Long-context dialogue systems suffer from State Inertia, where static constraints prevent models from resolving conflicts between evolving user intents and established historical context. To address this, we propose DZ-TDPO, a non-destructive alignment framework that synergizes conflict-aware dynamic KL constraints with a learnable temporal attention bias. Experiments on the Multi-Session Chat (MSC) dataset demonstrate that DZ-TDPO achieves state-of-the-art win rates (86.2% on Phi-3.5) while maintaining robust zero-shot generalization. Crucially, our scaling analysis reveals a "Capacity-Stability Trade-off": while smaller models incur an "alignment tax" (perplexity surge) to overcome historical inertia, the larger Qwen2.5-7B model achieves near-perfect alignment (99.4% win rate) with negligible perplexity overhead. This confirms that TAI can be alleviated via precise attention regulation rather than destructive weight updates, preserving general capabilities (MMLU) across model scales. Code and data are available: this https URL 

---
# AITutor-EvalKit: Exploring the Capabilities of AI Tutors 

**Authors**: Numaan Naeem, Kaushal Kumar Maurya, Kseniia Petukhova, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2512.03688)  

**Abstract**: We present AITutor-EvalKit, an application that uses language technology to evaluate the pedagogical quality of AI tutors, provides software for demonstration and evaluation, as well as model inspection and data visualization. This tool is aimed at education stakeholders as well as *ACL community at large, as it supports learning and can also be used to collect user feedback and annotations. 

---
# Different types of syntactic agreement recruit the same units within large language models 

**Authors**: Daria Kryvosheieva, Andrea de Varda, Evelina Fedorenko, Greta Tuckute  

**Link**: [PDF](https://arxiv.org/pdf/2512.03676)  

**Abstract**: Large language models (LLMs) can reliably distinguish grammatical from ungrammatical sentences, but how grammatical knowledge is represented within the models remains an open question. We investigate whether different syntactic phenomena recruit shared or distinct components in LLMs. Using a functional localization approach inspired by cognitive neuroscience, we identify the LLM units most responsive to 67 English syntactic phenomena in seven open-weight models. These units are consistently recruited across sentences containing the phenomena and causally support the models' syntactic performance. Critically, different types of syntactic agreement (e.g., subject-verb, anaphor, determiner-noun) recruit overlapping sets of units, suggesting that agreement constitutes a meaningful functional category for LLMs. This pattern holds in English, Russian, and Chinese; and further, in a cross-lingual analysis of 57 diverse languages, structurally more similar languages share more units for subject-verb agreement. Taken together, these findings reveal that syntactic agreement-a critical marker of syntactic dependencies-constitutes a meaningful category within LLMs' representational spaces. 

---
# Evaluating Hydro-Science and Engineering Knowledge of Large Language Models 

**Authors**: Shiruo Hu, Wenbo Shan, Yingjia Li, Zhiqi Wan, Xinpeng Yu, Yunjia Qi, Haotian Xia, Yang Xiao, Dingxiao Liu, Jiaru Wang, Chenxu Gong, Ruixi Zhang, Shuyue Wu, Shibo Cui, Chee Hui Lai, Wei Luo, Yubin He, Bin Xu, Jianshi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03672)  

**Abstract**: Hydro-Science and Engineering (Hydro-SE) is a critical and irreplaceable domain that secures human water supply, generates clean hydropower energy, and mitigates flood and drought disasters. Featuring multiple engineering objectives, Hydro-SE is an inherently interdisciplinary domain that integrates scientific knowledge with engineering expertise. This integration necessitates extensive expert collaboration in decision-making, which poses difficulties for intelligence. With the rapid advancement of large language models (LLMs), their potential application in the Hydro-SE domain is being increasingly explored. However, the knowledge and application abilities of LLMs in Hydro-SE have not been sufficiently evaluated. To address this issue, we propose the Hydro-SE LLM evaluation benchmark (Hydro-SE Bench), which contains 4,000 multiple-choice questions. Hydro-SE Bench covers nine subfields and enables evaluation of LLMs in aspects of basic conceptual knowledge, engineering application ability, and reasoning and calculation ability. The evaluation results on Hydro-SE Bench show that the accuracy values vary among 0.74 to 0.80 for commercial LLMs, and among 0.41 to 0.68 for small-parameter LLMs. While LLMs perform well in subfields closely related to natural and physical sciences, they struggle with domain-specific knowledge such as industry standards and hydraulic structures. Model scaling mainly improves reasoning and calculation abilities, but there is still great potential for LLMs to better handle problems in practical engineering application. This study highlights the strengths and weaknesses of LLMs for Hydro-SE tasks, providing model developers with clear training targets and Hydro-SE researchers with practical guidance for applying LLMs. 

---
# Generative AI Practices, Literacy, and Divides: An Empirical Analysis in the Italian Context 

**Authors**: Beatrice Savoldi, Giuseppe Attanasio, Olga Gorodetskaya, Marta Marchiori Manerba, Elisa Bassignana, Silvia Casola, Matteo Negri, Tommaso Caselli, Luisa Bentivogli, Alan Ramponi, Arianna Muti, Nicoletta Balbo, Debora Nozza  

**Link**: [PDF](https://arxiv.org/pdf/2512.03671)  

**Abstract**: The rise of Artificial Intelligence (AI) language technologies, particularly generative AI (GenAI) chatbots accessible via conversational interfaces, is transforming digital interactions. While these tools hold societal promise, they also risk widening digital divides due to uneven adoption and low awareness of their limitations. This study presents the first comprehensive empirical mapping of GenAI adoption, usage patterns, and literacy in Italy, based on newly collected survey data from 1,906 Italian-speaking adults. Our findings reveal widespread adoption for both work and personal use, including sensitive tasks like emotional support and medical advice. Crucially, GenAI is supplanting other technologies to become a primary information source: this trend persists despite low user digital literacy, posing a risk as users struggle to recognize errors or misinformation. Moreover, we identify a significant gender divide -- particularly pronounced in older generations -- where women are half as likely to adopt GenAI and use it less frequently than men. While we find literacy to be a key predictor of adoption, it only partially explains this disparity, suggesting that other barriers are at play. Overall, our data provide granular insights into the multipurpose usage of GenAI, highlighting the dual need for targeted educational initiatives and further investigation into the underlying barriers to equitable participation that competence alone cannot explain. 

---
# AlignCheck: a Semantic Open-Domain Metric for Factual Consistency Assessment 

**Authors**: Ahmad Aghaebrahimian  

**Link**: [PDF](https://arxiv.org/pdf/2512.03634)  

**Abstract**: Large Language Models have significantly advanced natural language processing tasks, but remain prone to generating incorrect or misleading but plausible arguments. This issue, known as hallucination, is particularly concerning in high-stakes domains like clinical applications, where factual inaccuracies can have severe consequences. Existing evaluation metrics fail to adequately assess factual consistency and lack interpretability, making diagnosing and mitigating errors difficult. We propose an interpretable framework for factual consistency assessment for in-domain and open-domain texts to address these limitations. Our approach decomposes text into atomic facts and introduces a flexible, schema-free methodology. Unlike previous methods with an absolute metric, we incorporate a weighted metric to enhance factual evaluation. Additionally, we propose a mechanism to control assessment complexity in intricate domains. We benchmark our approach on popular general and clinical datasets and release our code to support fact-aware model training in future research. 

---
# Fine-grained Narrative Classification in Biased News Articles 

**Authors**: Zeba Afroz, Harsh Vardhan, Pawan Bhakuni, Aanchal Punia, Rajdeep Kumar, Md. Shad Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2512.03582)  

**Abstract**: Narratives are the cognitive and emotional scaffolds of propaganda. They organize isolated persuasive techniques into coherent stories that justify actions, attribute blame, and evoke identification with ideological camps. In this paper, we propose a novel fine-grained narrative classification in biased news articles. We also explore article-bias classification as the precursor task to narrative classification and fine-grained persuasive technique identification. We develop INDI-PROP, the first ideologically grounded fine-grained narrative dataset with multi-level annotation for analyzing propaganda in Indian news media. Our dataset INDI-PROP comprises 1,266 articles focusing on two polarizing socio-political events in recent times: CAA and the Farmers' protest. Each article is annotated at three hierarchical levels: (i) ideological article-bias (pro-government, pro-opposition, neutral), (ii) event-specific fine-grained narrative frames anchored in ideological polarity and communicative intent, and (iii) persuasive techniques. We propose FANTA and TPTC, two GPT-4o-mini guided multi-hop prompt-based reasoning frameworks for the bias, narrative, and persuasive technique classification. FANTA leverages multi-layered communicative phenomena by integrating information extraction and contextual framing for hierarchical reasoning. On the other hand, TPTC adopts systematic decomposition of persuasive cues via a two-stage approach. Our evaluation suggests substantial improvement over underlying baselines in each case. 

---
# Understanding LLM Reasoning for Abstractive Summarization 

**Authors**: Haohan Yuan, Siu Cheung Hui, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03503)  

**Abstract**: While the reasoning capabilities of Large Language Models (LLMs) excel in analytical tasks such as mathematics and code generation, their utility for abstractive summarization remains widely assumed but largely unverified. To bridge this gap, we first tailor general reasoning strategies to the summarization domain. We then conduct a systematic, large scale comparative study of 8 reasoning strategies and 3 Large Reasoning Models (LRMs) across 8 diverse datasets, assessing both summary quality and faithfulness. Our findings show that reasoning is not a universal solution and its effectiveness is highly dependent on the specific strategy and context. Specifically, we observe a trade-off between summary quality and factual faithfulness: explicit reasoning strategies tend to improve fluency at the expense of factual grounding, while implicit reasoning in LRMs exhibits the inverse pattern. Furthermore, increasing an LRM's internal reasoning budget does not improve, and can even hurt, factual consistency, suggesting that effective summarization demands faithful compression rather than creative over-thinking. 

---
# A Preliminary Study on the Promises and Challenges of Native Top-$k$ Sparse Attention 

**Authors**: Di Xiu, Hongyin Tang, Bolin Rong, Lizhi Yan, Jingang Wang, Yifan Lu, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2512.03494)  

**Abstract**: Large Language Models (LLMs) are increasingly prevalent in the field of long-context modeling, however, their inference computational costs have become a critical bottleneck hindering the advancement of tasks such as agents and multimodal applications. This report conducts a preliminary investigation into the effectiveness and theoretical mechanisms of the Top-$k$ Attention mechanism during both the decoding and training phases. First, we validate the effectiveness of exact Top-$k$ Decoding through extensive experimentation. Experiments demonstrate that retaining only the pivotal Keys with the highest similarity to the Query as the context window during the decoding stage achieves performance comparable to, or even surpassing, full attention on downstream tasks such as HELMET and LongBench v2. Second, we further explore the native Top-$k$ Attention training strategy. Experiments confirm that ensuring the consistency between training and inference regarding Top-$k$ Attention operations facilitates the further unlocking of Top-$k$ Decoding's potential, thereby significantly enhancing model performance. Furthermore, considering the high computational complexity of exact Top-$k$ Attention, we investigate the impact of approximate Top-$k$ algorithm precision on downstream tasks. Our research confirms a positive correlation between downstream task performance and approximation fidelity, and we provide statistical evaluations of the Lightning Indexer's precision within the DeepSeek-V3.2-Exp model. Finally, this report provides a theoretical interpretation from the perspective of Entropy. Experimental observations indicate that models subjected to Top-$k$ Attention SFT exhibit a distinct phenomenon of entropy reduction in downstream tasks, which validates the hypothesis that low-entropy states are better adapted to Top-$k$ Decoding. 

---
# PretrainZero: Reinforcement Active Pretraining 

**Authors**: Xingrun Xing, Zhiyuan Fan, Jie Lou, Guoqi Li, Jiajun Zhang, Debing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03442)  

**Abstract**: Mimicking human behavior to actively learning from general experience and achieve artificial general intelligence has always been a human dream. Recent reinforcement learning (RL) based large-thinking models demonstrate impressive expert-level abilities, i.e., software and math, but still rely heavily on verifiable rewards in specific domains, placing a significant bottleneck to extend the performance boundary of general reasoning capabilities. In this work, we propose PretrainZero, a reinforcement active learning framework built on the pretraining corpus to extend RL from domain-specific post-training to general pretraining. PretrainZero features the following characteristics: 1) Active pretraining: inspired by the active learning ability of humans, PretrainZero learns a unified reasoning policy to actively identify reasonable and informative contents from pretraining corpus, and reason to predict these contents by RL. 2) Self-supervised learning: without any verifiable labels, pretrained reward models, or supervised fine-tuning, we directly pretrain reasoners from 3 to 30B base models on the general Wikipedia corpus using RL, significantly breaking the verification data-wall for general reasoning. 3) Verification scaling: by tackling increasingly challenging masked spans, PretrainZero substantially enhances the general reasoning abilities of pretrained base models. In reinforcement pretraining, PretrainZero improves Qwen3-4B-Base for 8.43, 5.96 and 10.60 on MMLU-Pro, SuperGPQA and math average benchmarks. In post-training, the pretrained models can also serve as reasoning foundation models for downstream RLVR tasks. 

---
# Dual LoRA: Enhancing LoRA with Magnitude and Direction Updates 

**Authors**: Yixing Xu, Chao Li, Xuanwu Yin, Spandan Tiwari, Dong Li, Ashish Sirasao, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2512.03402)  

**Abstract**: Low-rank adaptation (LoRA) is one of the most popular methods among parameter-efficient fine-tuning (PEFT) methods to adapt pre-trained large language models (LLMs) to specific downstream tasks. However, the model trained based on LoRA often has an unsatisfactory performance due to its low-rank assumption. In this paper, we propose a novel method called Dual LoRA to improve the performance by incorporating an inductive bias into the original LoRA. Specifically, we separate low-rank matrices into two groups: the magnitude group to control whether or not and how far we should update a parameter and the direction group to decide whether this parameter should move forward or backward, to better simulate the parameter updating process of the full fine-tuning based on gradient-based optimization algorithms. We show that this can be simply achieved by adding a ReLU function to the magnitude group and a sign function to the direction group. We conduct several experiments over a wide range of NLP tasks, including natural language generation (NLG), understanding (NLU), and commonsense reasoning datasets on GPT-2, RoBERTa, DeBERTa, and LLaMA-1/2/3 as baseline models. The results show that we consistently outperform LoRA and its state-of-the-art variants with the same number of trainable parameters. 

---
# Characterizing Language Use in a Collaborative Situated Game 

**Authors**: Nicholas Tomlin, Naitian Zhou, Eve Fleisig, Liangyuan, Chen, Téa Wright, Lauren Vinh, Laura X. Ma, Seun Eisape, Ellie French, Tingting Du, Tianjiao Zhang, Alexander Koller, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2512.03381)  

**Abstract**: Cooperative video games, where multiple participants must coordinate by communicating and reasoning under uncertainty in complex environments, yield a rich source of language data. We collect the Portal Dialogue Corpus: a corpus of 11.5 hours of spoken human dialogue in the co-op mode of the popular Portal 2 virtual puzzle game, comprising 24.5K total utterances. We analyze player language and behavior, identifying a number of linguistic phenomena that rarely appear in most existing chitchat or task-oriented dialogue corpora, including complex spatial reference, clarification and repair, and ad-hoc convention formation. To support future analyses of language use in complex, situated, collaborative problem-solving scenarios, we publicly release the corpus, which comprises player videos, audio, transcripts, game state data, and both manual and automatic annotations of language data. 

---
# Nexus: Higher-Order Attention Mechanisms in Transformers 

**Authors**: Hanting Chen, Chu Zhong, Kai Han, Yuchuan Tian, Yuchen Liang, Tianyu Guo, Xinghao Chen, Dacheng Tao, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03377)  

**Abstract**: Transformers have achieved significant success across various domains, relying on self-attention to capture dependencies. However, the standard first-order attention mechanism is often limited by a low-rank bottleneck, struggling to capture intricate, multi-hop relationships within a single layer. In this paper, we propose the \textbf{Higher-Order Attention Network (Hon)}, a novel architecture designed to enhance representational power through a recursive framework. Unlike standard approaches that use static linear projections for Queries and Keys, Hon dynamically refines these representations via nested self-attention mechanisms. Specifically, the Query and Key vectors are themselves outputs of inner attention loops, allowing tokens to aggregate global context and model high-order correlations \textit{prior} to the final attention computation. We enforce a parameter-efficient weight-sharing strategy across recursive steps, ensuring that this enhanced expressivity incurs $\mathcal{O}(1)$ additional parameters. We provide theoretical analysis demonstrating that our method breaks the linear bottleneck of standard attention. Empirically, Hon outperforms standard Transformers on multiple benchmarks. 

---
# From Hypothesis to Premises: LLM-based Backward Logical Reasoning with Selective Symbolic Translation 

**Authors**: Qingchuan Li, Mingyue Cheng, Zirui Liu, Daoyu Wang, Yuting Zeng, Tongxuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03360)  

**Abstract**: Logical reasoning is a core challenge in natural language understanding and a fundamental capability of artificial intelligence, underpinning scientific discovery, mathematical theorem proving, and complex decision-making. Despite the remarkable progress of large language models (LLMs), most current approaches still rely on forward reasoning paradigms, generating step-by-step rationales from premises to conclusions. However, such methods often suffer from redundant inference paths, hallucinated steps, and semantic drift, resulting in inefficient and unreliable reasoning. In this paper, we propose a novel framework, Hypothesis-driven Backward Logical Reasoning (HBLR). The core idea is to integrate confidence-aware symbolic translation with hypothesis-driven backward reasoning. In the translation phase, only high-confidence spans are converted into logical form, such as First-Order Logic (FOL), while uncertain content remains in natural language. A translation reflection module further ensures semantic fidelity by evaluating symbolic outputs and reverting lossy ones back to text when necessary. In the reasoning phase, HBLR simulates human deductive thinking by assuming the conclusion is true and recursively verifying its premises. A reasoning reflection module further identifies and corrects flawed inference steps, enhancing logical coherence. Extensive experiments on five reasoning benchmarks demonstrate that HBLR consistently outperforms strong baselines in both accuracy and efficiency. 

---
# Idea-Gated Transformers: Enforcing Semantic Coherence via Differentiable Vocabulary Pruning 

**Authors**: Darshan Fofadiya  

**Link**: [PDF](https://arxiv.org/pdf/2512.03343)  

**Abstract**: Autoregressive Language Models (LLMs) trained on Next-Token Prediction (NTP) often suffer from ``Topic Drift'' where the generation wanders away from the initial prompt due to a reliance on local associations rather than global planning \citep{holtzman2019curious}. While scaling model size mitigates this \citep{brown2020language}, the fundamental myopia of the NTP objective remains. In this work, we introduce the Idea-Gated Transformer, a novel architecture that separates semantic planning from syntactic generation. We introduce an auxiliary ``Idea Head'' trained to predict the bag-of-words distribution for a future context window, creating a latent ``Concept Vector'' that actively gates the main vocabulary during generation. We propose a differentiable gating mechanism that suppresses semantically irrelevant tokens, effectively pruning the search space in real-time. Experiments on WikiText-103 demonstrate that while the Idea-Gated model achieves comparable validation perplexity to a standard GPT-2 baseline, it exhibits significantly superior Domain Retention. Qualitative and quantitative analysis reveals that the gating mechanism successfully locks generation into specific semantic clusters (e.g., Finance, Science) and resists associative drift, offering a parameter-efficient path toward more controllable language modeling. 

---
# PERCS: Persona-Guided Controllable Biomedical Summarization Dataset 

**Authors**: Rohan Charudatt Salvi, Chirag Chawla, Dhruv Jain, Swapnil Panigrahi, Md Shad Akhtar, Shweta Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2512.03340)  

**Abstract**: Automatic medical text simplification plays a key role in improving health literacy by making complex biomedical research accessible to diverse readers. However, most existing resources assume a single generic audience, overlooking the wide variation in medical literacy and information needs across user groups. To address this limitation, we introduce PERCS (Persona-guided Controllable Summarization), a dataset of biomedical abstracts paired with summaries tailored to four personas: Laypersons, Premedical Students, Non-medical Researchers, and Medical Experts. These personas represent different levels of medical literacy and information needs, emphasizing the need for targeted, audience-specific summarization. Each summary in PERCS was reviewed by physicians for factual accuracy and persona alignment using a detailed error taxonomy. Technical validation shows clear differences in readability, vocabulary, and content depth across personas. Along with describing the dataset, we benchmark four large language models on PERCS using automatic evaluation metrics that assess comprehensiveness, readability, and faithfulness, establishing baseline results for future research. The dataset, annotation guidelines, and evaluation materials are publicly available to support research on persona-specific communication and controllable biomedical summarization. 

---
# Modeling Topics and Sociolinguistic Variation in Code-Switched Discourse: Insights from Spanish-English and Spanish-Guaraní 

**Authors**: Nemika Tyagi, Nelvin Licona Guevara, Olga Kellert  

**Link**: [PDF](https://arxiv.org/pdf/2512.03334)  

**Abstract**: This study presents an LLM-assisted annotation pipeline for the sociolinguistic and topical analysis of bilingual discourse in two typologically distinct contexts: Spanish-English and Spanish-Guaraní. Using large language models, we automatically labeled topic, genre, and discourse-pragmatic functions across a total of 3,691 code-switched sentences, integrated demographic metadata from the Miami Bilingual Corpus, and enriched the Spanish-Guaraní dataset with new topic annotations. The resulting distributions reveal systematic links between gender, language dominance, and discourse function in the Miami data, and a clear diglossic division between formal Guaraní and informal Spanish in Paraguayan texts. These findings replicate and extend earlier interactional and sociolinguistic observations with corpus-scale quantitative evidence. The study demonstrates that large language models can reliably recover interpretable sociolinguistic patterns traditionally accessible only through manual annotation, advancing computational methods for cross-linguistic and low-resource bilingual research. 

---
# Randomized Masked Finetuning: An Efficient Way to Mitigate Memorization of PIIs in LLMs 

**Authors**: Kunj Joshi, David A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2512.03310)  

**Abstract**: The current literature on memorization in Natural Language Models, especially Large Language Models (LLMs), poses severe security and privacy risks, as models tend to memorize personally identifying information (PIIs) from training data. We introduce Randomized Masked Fine-Tuning (RMFT), a novel privacy-preserving fine-tuning technique that reduces PII memorization while minimizing performance impact. Using the Enron Email Dataset, we demonstrate that RMFT achieves an 80.81% reduction in Total Extraction Rate and 80.17% reduction in Seen Extraction Rate compared to baseline fine-tuning, outperforming deduplication methods while maintaining only a 5.73% increase in perplexity. We present MaxTER, a Pareto-optimal evaluation framework for assessing privacy-utility tradeoffs, and show the performance of RMFT vs Deduplication by Area Under The Response Curve (AURC) metric. 

---
# Identifying attributions of causality in political text 

**Authors**: Paulina Garcia-Corral  

**Link**: [PDF](https://arxiv.org/pdf/2512.03214)  

**Abstract**: Explanations are a fundamental element of how people make sense of the political world. Citizens routinely ask and answer questions about why events happen, who is responsible, and what could or should be done differently. Yet despite their importance, explanations remain an underdeveloped object of systematic analysis in political science, and existing approaches are fragmented and often issue-specific. I introduce a framework for detecting and parsing explanations in political text. To do this, I train a lightweight causal language model that returns a structured data set of causal claims in the form of cause-effect pairs for downstream analysis. I demonstrate how causal explanations can be studied at scale, and show the method's modest annotation requirements, generalizability, and accuracy relative to human coding. 

---
# InvertiTune: High-Quality Data Synthesis for Cost-Effective Single-Shot Text-to-Knowledge Graph Generation 

**Authors**: Faezeh Faez, Marzieh S. Tahaei, Yaochen Hu, Ali Pourranjbar, Mahdi Biparva, Mark Coates, Yingxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03197)  

**Abstract**: Large Language Models (LLMs) have revolutionized the ability to understand and generate text, enabling significant progress in automatic knowledge graph construction from text (Text2KG). Many Text2KG methods, however, rely on iterative LLM prompting, making them computationally expensive and prone to overlooking complex relations distributed throughout the text. To address these limitations, we propose InvertiTune, a framework that combines a controlled data generation pipeline with supervised fine-tuning (SFT). Within this framework, the data-generation pipeline systematically extracts subgraphs from large knowledge bases, applies noise filtering, and leverages LLMs to generate corresponding natural text descriptions, a task more aligned with LLM capabilities than direct KG generation from text. This pipeline enables generating datasets composed of longer texts paired with larger KGs that better reflect real-world scenarios compared to existing benchmarks, thus supporting effective SFT of lightweight models for single-shot KG construction. Experimental results on CE12k, a dataset generated using the introduced pipeline, show that InvertiTune outperforms larger non-fine-tuned LLMs as well as state-of-the-art Text2KG approaches, while also demonstrating stronger cross-dataset generalization on CrossEval-1200, a test set created from three established benchmark datasets and CE12k. These findings highlight the importance of realistic, high-quality training data for advancing efficient and high-performing Text2KG systems. 

---
# Enhancing Job Matching: Occupation, Skill and Qualification Linking with the ESCO and EQF taxonomies 

**Authors**: Stylianos Saroglou, Konstantinos Diamantaras, Francesco Preta, Marina Delianidi, Apostolos Benisis, Christian Johannes Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2512.03195)  

**Abstract**: This study investigates the potential of language models to improve the classification of labor market information by linking job vacancy texts to two major European frameworks: the European Skills, Competences, Qualifications and Occupations (ESCO) taxonomy and the European Qualifications Framework (EQF). We examine and compare two prominent methodologies from the literature: Sentence Linking and Entity Linking. In support of ongoing research, we release an open-source tool, incorporating these two methodologies, designed to facilitate further work on labor classification and employment discourse. To move beyond surface-level skill extraction, we introduce two annotated datasets specifically aimed at evaluating how occupations and qualifications are represented within job vacancy texts. Additionally, we examine different ways to utilize generative large language models for this task. Our findings contribute to advancing the state of the art in job entity extraction and offer computational infrastructure for examining work, skills, and labor market narratives in a digitally mediated economy. Our code is made publicly available: this https URL 

---
# Alleviating Choice Supportive Bias in LLM with Reasoning Dependency Generation 

**Authors**: Nan Zhuang, Wenshuo Wang, Lekai Qian, Yuxiao Wang, Boyu Cao, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03082)  

**Abstract**: Recent studies have demonstrated that some Large Language Models exhibit choice-supportive bias (CSB) when performing evaluations, systematically favoring their chosen options and potentially compromising the objectivity of AI-assisted decision making. While existing debiasing approaches primarily target demographic and social biases, methods for addressing cognitive biases in LLMs remain largely unexplored. In this work, we present the first solution to address CSB through Reasoning Dependency Generation (RDG), a novel framework for generating unbiased reasoning data to mitigate choice-supportive bias through fine-tuning. RDG automatically constructs balanced reasoning QA pairs, explicitly (un)modeling the dependencies between choices, evidences, and justifications. Our approach is able to generate a large-scale dataset of QA pairs across domains, incorporating Contextual Dependency Data and Dependency Decouple Data. Experiments show that LLMs fine-tuned on RDG-generated data demonstrate a 81.5% improvement in memory-based experiments and 94.3% improvement in the evaluation-based experiment, while maintaining similar performance on standard BBQ benchmarks. This work pioneers an approach for addressing cognitive biases in LLMs and contributes to the development of more reliable AI-assisted decision support systems. 

---
# Watermarks for Embeddings-as-a-Service Large Language Models 

**Authors**: Anudeex Shetty  

**Link**: [PDF](https://arxiv.org/pdf/2512.03079)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities in natural language understanding and generation. Based on these LLMs, businesses have started to provide Embeddings-as-a-Service (EaaS), offering feature extraction capabilities (in the form of text embeddings) that benefit downstream natural language processing tasks. However, prior research has demonstrated that EaaS is vulnerable to imitation attacks, where an attacker clones the service's model in a black-box manner without access to the model's internal workings. In response, watermarks have been added to the text embeddings to protect the intellectual property of EaaS providers by allowing them to check for model ownership. This thesis focuses on defending against imitation attacks by investigating EaaS watermarks. To achieve this goal, we unveil novel attacks and propose and validate new watermarking techniques.
Firstly, we show that existing EaaS watermarks can be removed through paraphrasing the input text when attackers clone the model during imitation attacks. Our study illustrates that paraphrasing can effectively bypass current state-of-the-art EaaS watermarks across various attack setups (including different paraphrasing techniques and models) and datasets in most instances. This demonstrates a new vulnerability in recent EaaS watermarking techniques.
Subsequently, as a countermeasure, we propose a novel watermarking technique, WET (Watermarking EaaS with Linear Transformation), which employs linear transformation of the embeddings. Watermark verification is conducted by applying a reverse transformation and comparing the similarity between recovered and original embeddings. We demonstrate its robustness against paraphrasing attacks with near-perfect verifiability. We conduct detailed ablation studies to assess the significance of each component and hyperparameter in WET. 

---
# Entropy-Based Measurement of Value Drift and Alignment Work in Large Language Models 

**Authors**: Samih Fadli  

**Link**: [PDF](https://arxiv.org/pdf/2512.03047)  

**Abstract**: Large language model safety is usually assessed with static benchmarks, but key failures are dynamic: value drift under distribution shift, jailbreak attacks, and slow degradation of alignment in deployment. Building on a recent Second Law of Intelligence that treats ethical entropy as a state variable which tends to increase unless countered by alignment work, we make this framework operational for large language models. We define a five-way behavioral taxonomy, train a classifier to estimate ethical entropy S(t) from model transcripts, and measure entropy dynamics for base and instruction-tuned variants of four frontier models across stress tests. Base models show sustained entropy growth, while tuned variants suppress drift and reduce ethical entropy by roughly eighty percent. From these trajectories we estimate an effective alignment work rate gamma_eff and embed S(t) and gamma_eff in a monitoring pipeline that raises alerts when entropy drift exceeds a stability threshold, enabling run-time oversight of value drift. 

---
# Stable Signer: Hierarchical Sign Language Generative Model 

**Authors**: Sen Fang, Yalin Feng, Hongbin Zhong, Yanxin Zhang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2512.04048)  

**Abstract**: Sign Language Production (SLP) is the process of converting the complex input text into a real video. Most previous works focused on the Text2Gloss, Gloss2Pose, Pose2Vid stages, and some concentrated on Prompt2Gloss and Text2Avatar stages. However, this field has made slow progress due to the inaccuracy of text conversion, pose generation, and the rendering of poses into real human videos in these stages, resulting in gradually accumulating errors. Therefore, in this paper, we streamline the traditional redundant structure, simplify and optimize the task objective, and design a new sign language generative model called Stable Signer. It redefines the SLP task as a hierarchical generation end-to-end task that only includes text understanding (Prompt2Gloss, Text2Gloss) and Pose2Vid, and executes text understanding through our proposed new Sign Language Understanding Linker called SLUL, and generates hand gestures through the named SLP-MoE hand gesture rendering expert block to end-to-end generate high-quality and multi-style sign language videos. SLUL is trained using the newly developed Semantic-Aware Gloss Masking Loss (SAGM Loss). Its performance has improved by 48.6% compared to the current SOTA generation methods. 

---
# AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition 

**Authors**: Zichuan Lin, Yicheng Liu, Yang Yang, Lvfang Tao, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2512.03794)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable success in visual question answering tasks, but their reliance on large numbers of visual tokens introduces significant computational overhead. While existing efficient VLM approaches reduce visual tokens through fixed-ratio compression, they operate passively and lack the ability to adapt to varying task requirements. This motivates a fundamental question: Can VLMs autonomously determine the minimum number of visual tokens required for each sample? Inspired by human active vision mechanisms, we introduce AdaptVision, an efficient VLM paradigm that enables adaptive visual token acquisition through a coarse-to-fine approach. Our model initially processes compressed visual tokens from low-resolution images and selectively acquires additional visual information by invoking a bounding box tool to crop key regions when necessary. We train AdaptVision using a reinforcement learning framework that carefully balances accuracy and efficiency. Central to our approach is Decoupled Turn Policy Optimization (DTPO), which decouples the learning objective into two components: (1) tool learning, which optimizes correct tool utilization, and (2) accuracy improvement, which refines the generated responses to improve answer correctness. Based on this formulation, we further decouple advantage estimation by computing separate advantages for tokens associated with each objective. This formulation enables more effective optimization for AdaptVision compared to vanilla GRPO. Comprehensive experiments across multiple VQA benchmarks demonstrate that AdaptVision achieves superior performance while consuming substantially fewer visual tokens than state-of-the-art efficient VLM methods. 

---
# Thinking with Programming Vision: Towards a Unified View for Thinking with Images 

**Authors**: Zirun Guo, Minjie Hong, Feng Zhang, Kai Jia, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03746)  

**Abstract**: Multimodal large language models (MLLMs) that think with images can interactively use tools to reason about visual inputs, but current approaches often rely on a narrow set of tools with limited real-world necessity and scalability. In this work, we first reveal a critical and previously overlooked weakness: even state-of-the-art MLLMs are surprisingly brittle, showing significant performance degradation on images with simple orientation changes or natural corruptions, underscoring the need for more robust tool-based reasoning. To address this, we propose CodeVision, a flexible and scalable code-as-tool framework where the model generates code as a universal interface to invoke any image operation, moving beyond fixed tool registries. We train our model using a two-stage methodology, beginning with Supervised Fine-Tuning (SFT) on a high-quality dataset curated for complex, multi-turn tool composition and error recovery, followed by Reinforcement Learning (RL) with a novel and dense process reward function to encourage strategic and efficient tool use. To facilitate this research, we construct new SFT and RL datasets and introduce a challenging new benchmark suite designed to rigorously evaluate robustness to orientation changes and multi-tool reasoning. Experiments on Qwen2.5-VL and Qwen3-VL series show that our approach significantly improves model performance and fosters emergent capabilities such as flexible tool composition, efficient chained execution, and robust error recovery from runtime feedback. Code is available at this https URL. 

---
# Optical Context Compression Is Just (Bad) Autoencoding 

**Authors**: Ivan Yee Lee, Cheng Yang, Taylor Berg-Kirkpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2512.03643)  

**Abstract**: DeepSeek-OCR demonstrates that rendered text can be reconstructed with high fidelity from a small number of vision tokens. This finding has sparked excitement about vision-based context compression for language models. But the evaluation stops at reconstruction; whether these representations help language modeling remains untested. We test two assumptions implicit in the optical-compression narrative: that vision-based compression provides unique advantages for text reconstruction from compressed representations, and that DeepSeek-OCR's reconstruction results are evidence that vision-based compression will be useful for language modeling. Comparing their vision encoder against simple alternatives--parameter-free mean pooling and a learned hierarchical encoder--we find that these simple approaches match or surpass vision for reconstruction at matched compression ratios, and outperform it for language modeling--where vision-based compression fails to beat truncation. The excitement around optical context compression outpaces the evidence. Code and checkpoints are available at this https URL 

---
# SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting 

**Authors**: Hanxiu Zhang, Yue Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03620)  

**Abstract**: The protection of Intellectual Property (IP) in Large Language Models (LLMs) represents a critical challenge in contemporary AI research. While fingerprinting techniques have emerged as a fundamental mechanism for detecting unauthorized model usage, existing methods -- whether behavior-based or structural -- suffer from vulnerabilities such as false claim attacks or susceptible to weight manipulations. To overcome these limitations, we propose SELF, a novel intrinsic weight-based fingerprinting scheme that eliminates dependency on input and inherently resists false claims. SELF achieves robust IP protection through two key innovations: 1) unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights, and 2) effective neural network-based fingerprint similarity comparison based on few-shot learning and data augmentation. Experimental results demonstrate SELF maintains high IP infringement detection accuracy while showing strong robustness against various downstream modifications, including quantization, pruning, and fine-tuning attacks. Our code is available at this https URL. 

---
# CartoMapQA: A Fundamental Benchmark Dataset Evaluating Vision-Language Models on Cartographic Map Understanding 

**Authors**: Huy Quang Ung, Guillaume Habault, Yasutaka Nishimura, Hao Niu, Roberto Legaspi, Tomoki Oya, Ryoichi Kojima, Masato Taya, Chihiro Ono, Atsunori Minamikawa, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03558)  

**Abstract**: The rise of Visual-Language Models (LVLMs) has unlocked new possibilities for seamlessly integrating visual and textual information. However, their ability to interpret cartographic maps remains largely unexplored. In this paper, we introduce CartoMapQA, a benchmark specifically designed to evaluate LVLMs' understanding of cartographic maps through question-answering tasks. The dataset includes over 2000 samples, each composed of a cartographic map, a question (with open-ended or multiple-choice answers), and a ground-truth answer. These tasks span key low-, mid- and high-level map interpretation skills, including symbol recognition, embedded information extraction, scale interpretation, and route-based reasoning. Our evaluation of both open-source and proprietary LVLMs reveals persistent challenges: models frequently struggle with map-specific semantics, exhibit limited geospatial reasoning, and are prone to Optical Character Recognition (OCR)-related errors. By isolating these weaknesses, CartoMapQA offers a valuable tool for guiding future improvements in LVLM architectures. Ultimately, it supports the development of models better equipped for real-world applications that depend on robust and reliable map understanding, such as navigation, geographic search, and urban planning. Our source code and data are openly available to the research community at: this https URL 

---
# M3DR: Towards Universal Multilingual Multimodal Document Retrieval 

**Authors**: Adithya S Kolavi, Vyoman Jain  

**Link**: [PDF](https://arxiv.org/pdf/2512.03514)  

**Abstract**: Multimodal document retrieval systems have shown strong progress in aligning visual and textual content for semantic search. However, most existing approaches remain heavily English-centric, limiting their effectiveness in multilingual contexts. In this work, we present M3DR (Multilingual Multimodal Document Retrieval), a framework designed to bridge this gap across languages, enabling applicability across diverse linguistic and cultural contexts. M3DR leverages synthetic multilingual document data and generalizes across different vision-language architectures and model sizes, enabling robust cross-lingual and cross-modal alignment. Using contrastive training, our models learn unified representations for text and document images that transfer effectively across languages. We validate this capability on 22 typologically diverse languages, demonstrating consistent performance and adaptability across linguistic and script variations. We further introduce a comprehensive benchmark that captures real-world multilingual scenarios, evaluating models under monolingual, multilingual, and mixed-language settings. M3DR generalizes across both single dense vector and ColBERT-style token-level multi-vector retrieval paradigms. Our models, NetraEmbed and ColNetraEmbed achieve state-of-the-art performance with ~150% relative improvements on cross-lingual retrieval. 

---
# NAS-LoRA: Empowering Parameter-Efficient Fine-Tuning for Visual Foundation Models with Searchable Adaptation 

**Authors**: Renqi Chen, Haoyang Su, Shixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03499)  

**Abstract**: The Segment Anything Model (SAM) has emerged as a powerful visual foundation model for image segmentation. However, adapting SAM to specific downstream tasks, such as medical and agricultural imaging, remains a significant challenge. To address this, Low-Rank Adaptation (LoRA) and its variants have been widely employed to enhancing SAM's adaptation performance on diverse domains. Despite advancements, a critical question arises: can we integrate inductive bias into the model? This is particularly relevant since the Transformer encoder in SAM inherently lacks spatial priors within image patches, potentially hindering the acquisition of high-level semantic information. In this paper, we propose NAS-LoRA, a new Parameter-Efficient Fine-Tuning (PEFT) method designed to bridge the semantic gap between pre-trained SAM and specialized domains. Specifically, NAS-LoRA incorporates a lightweight Neural Architecture Search (NAS) block between the encoder and decoder components of LoRA to dynamically optimize the prior knowledge integrated into weight updates. Furthermore, we propose a stage-wise optimization strategy to help the ViT encoder balance weight updates and architectural adjustments, facilitating the gradual learning of high-level semantic information. Various Experiments demonstrate our NAS-LoRA improves existing PEFT methods, while reducing training cost by 24.14% without increasing inference cost, highlighting the potential of NAS in enhancing PEFT for visual foundation models. 

---
# Tuning for TraceTarnish: Techniques, Trends, and Testing Tangible Traits 

**Authors**: Robert Dilworth  

**Link**: [PDF](https://arxiv.org/pdf/2512.03465)  

**Abstract**: In this study, we more rigorously evaluated our attack script $\textit{TraceTarnish}$, which leverages adversarial stylometry principles to anonymize the authorship of text-based messages. To ensure the efficacy and utility of our attack, we sourced, processed, and analyzed Reddit comments--comments that were later alchemized into $\textit{TraceTarnish}$ data--to gain valuable insights. The transformed $\textit{TraceTarnish}$ data was then further augmented by $\textit{StyloMetrix}$ to manufacture stylometric features--features that were culled using the Information Gain criterion, leaving only the most informative, predictive, and discriminative ones. Our results found that function words and function word types ($L\_FUNC\_A$ $\&$ $L\_FUNC\_T$); content words and content word types ($L\_CONT\_A$ $\&$ $L\_CONT\_T$); and the Type-Token Ratio ($ST\_TYPE\_TOKEN\_RATIO\_LEMMAS$) yielded significant Information-Gain readings. The identified stylometric cues--function-word frequencies, content-word distributions, and the Type-Token Ratio--serve as reliable indicators of compromise (IoCs), revealing when a text has been deliberately altered to mask its true author. Similarly, these features could function as forensic beacons, alerting defenders to the presence of an adversarial stylometry attack; granted, in the absence of the original message, this signal may go largely unnoticed, as it appears to depend on a pre- and post-transformation comparison. "In trying to erase a trace, you often imprint a larger one." Armed with this understanding, we framed $\textit{TraceTarnish}$'s operations and outputs around these five isolated features, using them to conceptualize and implement enhancements that further strengthen the attack. 

---
# Text-Printed Image: Bridging the Image-Text Modality Gap for Text-centric Training of Large Vision-Language Models 

**Authors**: Shojiro Yamabe, Futa Waseda, Daiki Shiono, Tsubasa Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2512.03463)  

**Abstract**: Recent large vision-language models (LVLMs) have been applied to diverse VQA tasks. However, achieving practical performance typically requires task-specific fine-tuning with large numbers of image-text pairs, which are costly to collect. In this work, we study text-centric training, a setting where only textual descriptions are available and no real images are provided, as a paradigm for low-cost data scaling. Unlike images, whose collection is often restricted by privacy constraints and scarcity in niche domains, text is widely available. Moreover, text is easily editable, enabling automatic diversification and expansion with LLMs at minimal human effort. While this offers clear advantages over image collection in terms of scalability and cost, training on raw text without images still yields limited gains on VQA tasks because of the image-text modality gap. To address this issue, we propose a Text-Printed Image (TPI), which generates synthetic images by directly rendering the given textual description on a plain white canvas. This simple rendering projects text into the image modality and can be integrated into arbitrary existing LVLM training pipelines at low cost. Moreover, TPI preserves the semantics of the text, whereas text-to-image models often fail to do. Across four models and seven benchmarks, our systematic experiments show that TPI enables more effective text-centric training than synthetic images generated by a diffusion model. We further explore TPI as a low-cost data-augmentation strategy and demonstrate its practical utility. Overall, our findings highlight the significant potential of text-centric training and, more broadly, chart a path toward fully automated data generation for LVLMs. 

---
# LLM-Generated Ads: From Personalization Parity to Persuasion Superiority 

**Authors**: Elyas Meguellati, Stefano Civelli, Lei Han, Abraham Bernstein, Shazia Sadiq, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2512.03373)  

**Abstract**: As large language models (LLMs) become increasingly capable of generating persuasive content, understanding their effectiveness across different advertising strategies becomes critical. This paper presents a two-part investigation examining LLM-generated advertising through complementary lenses: (1) personality-based and (2) psychological persuasion principles.
In our first study (n=400), we tested whether LLMs could generate personalized advertisements tailored to specific personality traits (openness and neuroticism) and how their performance compared to human experts. Results showed that LLM-generated ads achieved statistical parity with human-written ads (51.1% vs. 48.9%, p > 0.05), with no significant performance differences for matched personalities.
Building on these insights, our second study (n=800) shifted focus from individual personalization to universal persuasion, testing LLM performance across four foundational psychological principles: authority, consensus, cognition, and scarcity. AI-generated ads significantly outperformed human-created content, achieving a 59.1% preference rate (vs. 40.9%, p < 0.001), with the strongest performance in authority (63.0%) and consensus (62.5%) appeals. Qualitative analysis revealed AI's advantage stems from crafting more sophisticated, aspirational messages and achieving superior visual-narrative coherence. Critically, this quality advantage proved robust: even after applying a 21.2 percentage point detection penalty when participants correctly identified AI-origin, AI ads still outperformed human ads, and 29.4% of participants chose AI content despite knowing its origin. These findings demonstrate LLMs' evolution from parity in personalization to superiority in persuasive storytelling, with significant implications for advertising practice given LLMs' near-zero marginal cost and time requirements compared to human experts. 

---
# Epistemic Substitution: How Grokipedia's AI-Generated Encyclopedia Restructures Authority 

**Authors**: Aliakbar Mehdizadeh, Martin Hilbert  

**Link**: [PDF](https://arxiv.org/pdf/2512.03337)  

**Abstract**: A quarter century ago, Wikipedia's decentralized, crowdsourced, and consensus-driven model replaced the centralized, expert-driven, and authority-based standard for encyclopedic knowledge curation. The emergence of generative AI encyclopedias, such as Grokipedia, possibly presents another potential shift in epistemic evolution. This study investigates whether AI- and human-curated encyclopedias rely on the same foundations of authority. We conducted a multi-scale comparative analysis of the citation networks from 72 matched article pairs, which cite a total of almost 60,000 sources. Using an 8-category epistemic classification, we mapped the "epistemic profiles" of the articles on each platform. Our findings reveal several quantitative and qualitative differences in how knowledge is sourced and encyclopedia claims are epistemologically justified. Grokipedia replaces Wikipedia's heavy reliance on peer-reviewed "Academic & Scholarly" work with a notable increase in "User-generated" and "Civic organization" sources. Comparative network analyses further show that Grokipedia employs very different epistemological profiles when sourcing leisure topics (such as Sports and Entertainment) and more societal sensitive civic topics (such as Politics & Conflicts, Geographical Entities, and General Knowledge & Society). Finally, we find a "scaling-law for AI-generated knowledge sourcing" that shows a linear relationship between article length and citation density, which is distinct from collective human reference sourcing. We conclude that this first implementation of an LLM-based encyclopedia does not merely automate knowledge production but restructures it. Given the notable changes and the important role of encyclopedias, we suggest the continuation and deepening of algorithm audits, such as the one presented here, in order to understand the ongoing epistemological shifts. 

---
# Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks 

**Authors**: Songwen Zhao, Danqing Wang, Kexun Zhang, Jiaxuan Luo, Zhuo Li, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.03262)  

**Abstract**: Vibe coding is a new programming paradigm in which human engineers instruct large language model (LLM) agents to complete complex coding tasks with little supervision. Although it is increasingly adopted, are vibe coding outputs really safe to deploy in production? To answer this question, we propose SU S VI B E S, a benchmark consisting of 200 feature-request software engineering tasks from real-world open-source projects, which, when given to human programmers, led to vulnerable implementations. We evaluate multiple widely used coding agents with frontier models on this benchmark. Disturbingly, all agents perform poorly in terms of software security. Although 61% of the solutions from SWE-Agent with Claude 4 Sonnet are functionally correct, only 10.5% are secure. Further experiments demonstrate that preliminary security strategies, such as augmenting the feature request with vulnerability hints, cannot mitigate these security issues. Our findings raise serious concerns about the widespread adoption of vibe-coding, particularly in security-sensitive applications. 

---
# SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning 

**Authors**: Salman Rahman, Sruthi Gorantla, Arpit Gupta, Swastik Roy, Nanyun Peng, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03244)  

**Abstract**: Process reward models (PRMs) that provide dense, step-level feedback have shown promise for reinforcement learning, yet their adoption remains limited by the need for expensive step-level annotations or ground truth references. We propose SPARK: a three-stage framework where in the first stage a generator model produces diverse solutions and a verifier model evaluates them using parallel scaling (self-consistency) and sequential scaling (meta-critique). In the second stage, we use these verification outputs as synthetic training data to fine-tune generative process reward models, which subsequently serve as reward signals during training. We show that aggregating multiple independent verifications at the step level produces training data for process reward models that surpass ground-truth outcome supervision, achieving 67.5 F1 on ProcessBench (a benchmark for identifying erroneous steps in mathematical reasoning) compared to 66.4 for reference-guided training and 61.9 for GPT-4o. In the final stage, we apply our generative PRM with chain-of-thought verification (PRM-CoT) as the reward model in RL experiments on mathematical reasoning, and introduce format constraints to prevent reward hacking. Using Qwen2.5-Math-7B, we achieve 47.4% average accuracy across six mathematical reasoning benchmarks, outperforming ground-truth-based RLVR (43.9%). Our work enables reference-free RL training that exceeds ground-truth methods, opening new possibilities for domains lacking verifiable answers or accessible ground truth. 

---
# Culture Affordance Atlas: Reconciling Object Diversity Through Functional Mapping 

**Authors**: Joan Nwatu, Longju Bai, Oana Ignat, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2512.03173)  

**Abstract**: Culture shapes the objects people use and for what purposes, yet mainstream Vision-Language (VL) datasets frequently exhibit cultural biases, disproportionately favoring higher-income, Western contexts. This imbalance reduces model generalizability and perpetuates performance disparities, especially impacting lower-income and non-Western communities. To address these disparities, we propose a novel function-centric framework that categorizes objects by the functions they fulfill, across diverse cultural and economic contexts. We implement this framework by creating the Culture Affordance Atlas, a re-annotated and culturally grounded restructuring of the Dollar Street dataset spanning 46 functions and 288 objects publicly available at this https URL. Through extensive empirical analyses using the CLIP model, we demonstrate that function-centric labels substantially reduce socioeconomic performance gaps between high- and low-income groups by a median of 6 pp (statistically significant), improving model effectiveness for lower-income contexts. Furthermore, our analyses reveals numerous culturally essential objects that are frequently overlooked in prominent VL datasets. Our contributions offer a scalable pathway toward building inclusive VL datasets and equitable AI systems. 

---
# Detecting AI Hallucinations in Finance: An Information-Theoretic Method Cuts Hallucination Rate by 92% 

**Authors**: Mainak Singha  

**Link**: [PDF](https://arxiv.org/pdf/2512.03107)  

**Abstract**: Large language models (LLMs) produce fluent but unsupported answers - hallucinations - limiting safe deployment in high-stakes domains. We propose ECLIPSE, a framework that treats hallucination as a mismatch between a model's semantic entropy and the capacity of available evidence. We combine entropy estimation via multi-sample clustering with a novel perplexity decomposition that measures how models use retrieved evidence. We prove that under mild conditions, the resulting entropy-capacity objective is strictly convex with a unique stable optimum. We evaluate on a controlled financial question answering dataset with GPT-3.5-turbo (n=200 balanced samples with synthetic hallucinations), where ECLIPSE achieves ROC AUC of 0.89 and average precision of 0.90, substantially outperforming a semantic entropy-only baseline (AUC 0.50). A controlled ablation with Claude-3-Haiku, which lacks token-level log probabilities, shows AUC dropping to 0.59 with coefficient magnitudes decreasing by 95% - demonstrating that ECLIPSE is a logprob-native mechanism whose effectiveness depends on calibrated token-level uncertainties. The perplexity decomposition features exhibit the largest learned coefficients, confirming that evidence utilization is central to hallucination detection. We position this work as a controlled mechanism study; broader validation across domains and naturally occurring hallucinations remains future work. 

---
