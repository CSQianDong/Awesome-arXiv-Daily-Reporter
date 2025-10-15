# Cost Analysis of Human-corrected Transcription for Predominately Oral Languages 

**Authors**: Yacouba Diarra, Nouhoum Souleymane Coulibaly, Michael Leventhal  

**Link**: [PDF](https://arxiv.org/pdf/2510.12781)  

**Abstract**: Creating speech datasets for low-resource languages is a critical yet poorly understood challenge, particularly regarding the actual cost in human labor. This paper investigates the time and complexity required to produce high-quality annotated speech data for a subset of low-resource languages, low literacy Predominately Oral Languages, focusing on Bambara, a Manding language of Mali. Through a one-month field study involving ten transcribers with native proficiency, we analyze the correction of ASR-generated transcriptions of 53 hours of Bambara voice data. We report that it takes, on average, 30 hours of human labor to accurately transcribe one hour of speech data under laboratory conditions and 36 hours under field conditions. The study provides a baseline and practical insights for a large class of languages with comparable profiles undertaking the creation of NLP resources. 

---
# Dr.LLM: Dynamic Layer Routing in LLMs 

**Authors**: Ahmed Heakl, Martin Gubri, Salman Khan, Sangdoo Yun, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.12773)  

**Abstract**: Large Language Models (LLMs) process every token through all layers of a transformer stack, causing wasted computation on simple queries and insufficient flexibility for harder ones that need deeper reasoning. Adaptive-depth methods can improve efficiency, but prior approaches rely on costly inference-time search, architectural changes, or large-scale retraining, and in practice often degrade accuracy despite efficiency gains. We introduce this http URL, Dynamic routing of Layers for LLMs, a retrofittable framework that equips pretrained models with lightweight per-layer routers deciding to skip, execute, or repeat a block. Routers are trained with explicit supervision: using Monte Carlo Tree Search (MCTS), we derive high-quality layer configurations that preserve or improve accuracy under a compute budget. Our design, windowed pooling for stable routing, focal loss with class balancing, and bottleneck MLP routers, ensures robustness under class imbalance and long sequences. On ARC (logic) and DART (math), this http URL improves accuracy by up to +3.4%p while saving 5 layers per example on average. Routers generalize to out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, AGIEval) with only 0.85% accuracy drop while retaining efficiency, and outperform prior routing methods by up to +7.7%p. Overall, this http URL shows that explicitly supervised routers retrofit frozen LLMs for budget-aware, accuracy-driven inference without altering base weights. 

---
# Language Models Model Language 

**Authors**: Łukasz Borchmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.12766)  

**Abstract**: Linguistic commentary on LLMs, heavily influenced by the theoretical frameworks of de Saussure and Chomsky, is often speculative and unproductive. Critics challenge whether LLMs can legitimately model language, citing the need for "deep structure" or "grounding" to achieve an idealized linguistic "competence." We argue for a radical shift in perspective towards the empiricist principles of Witold Mańczak, a prominent general and historical linguist. He defines language not as a "system of signs" or a "computational system of the brain" but as the totality of all that is said and written. Above all, he identifies frequency of use of particular language elements as language's primary governing principle. Using his framework, we challenge prior critiques of LLMs and provide a constructive guide for designing, evaluating, and interpreting language models. 

---
# Hey, wait a minute: on at-issue sensitivity in Language Models 

**Authors**: Sanghee J. Kim, Kanishka Misra  

**Link**: [PDF](https://arxiv.org/pdf/2510.12740)  

**Abstract**: Evaluating the naturalness of dialogue in language models (LMs) is not trivial: notions of 'naturalness' vary, and scalable quantitative metrics remain limited. This study leverages the linguistic notion of 'at-issueness' to assess dialogue naturalness and introduces a new method: Divide, Generate, Recombine, and Compare (DGRC). DGRC (i) divides a dialogue as a prompt, (ii) generates continuations for subparts using LMs, (iii) recombines the dialogue and continuations, and (iv) compares the likelihoods of the recombined sequences. This approach mitigates bias in linguistic analyses of LMs and enables systematic testing of discourse-sensitive behavior. Applying DGRC, we find that LMs prefer to continue dialogue on at-issue content, with this effect enhanced in instruct-tuned models. They also reduce their at-issue preference when relevant cues (e.g., "Hey, wait a minute") are present. Although instruct-tuning does not further amplify this modulation, the pattern reflects a hallmark of successful dialogue dynamics. 

---
# Which Word Orders Facilitate Length Generalization in LMs? An Investigation with GCG-Based Artificial Languages 

**Authors**: Nadine El-Naggar, Tatsuki Kuribayashi, Ted Briscoe  

**Link**: [PDF](https://arxiv.org/pdf/2510.12722)  

**Abstract**: Whether language models (LMs) have inductive biases that favor typologically frequent grammatical properties over rare, implausible ones has been investigated, typically using artificial languages (ALs) (White and Cotterell, 2021; Kuribayashi et al., 2024). In this paper, we extend these works from two perspectives. First, we extend their context-free AL formalization by adopting Generalized Categorial Grammar (GCG) (Wood, 2014), which allows ALs to cover attested but previously overlooked constructions, such as unbounded dependency and mildly context-sensitive structures. Second, our evaluation focuses more on the generalization ability of LMs to process unseen longer test sentences. Thus, our ALs better capture features of natural languages and our experimental paradigm leads to clearer conclusions -- typologically plausible word orders tend to be easier for LMs to productively generalize. 

---
# Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception 

**Authors**: Ziyang Ma, Ruiyang Xu, Zhenghao Xing, Yunfei Chu, Yuxuan Wang, Jinzheng He, Jin Xu, Pheng-Ann Heng, Kai Yu, Junyang Lin, Eng Siong Chng, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12720)  

**Abstract**: Fine-grained perception of multimodal information is critical for advancing human-AI interaction. With recent progress in audio-visual technologies, Omni Language Models (OLMs), capable of processing audio and video signals in parallel, have emerged as a promising paradigm for achieving richer understanding and reasoning. However, their capacity to capture and describe fine-grained details remains limited explored. In this work, we present a systematic and comprehensive investigation of omni detailed perception from the perspectives of the data pipeline, models, and benchmark. We first identify an inherent "co-growth" between detail and hallucination in current OLMs. To address this, we propose Omni-Detective, an agentic data generation pipeline integrating tool-calling, to autonomously produce highly detailed yet minimally hallucinatory multimodal data. Based on the data generated with Omni-Detective, we train two captioning models: Audio-Captioner for audio-only detailed perception, and Omni-Captioner for audio-visual detailed perception. Under the cascade evaluation protocol, Audio-Captioner achieves the best performance on MMAU and MMAR among all open-source models, surpassing Gemini 2.5 Flash and delivering performance comparable to Gemini 2.5 Pro. On existing detailed captioning benchmarks, Omni-Captioner sets a new state-of-the-art on VDC and achieves the best trade-off between detail and hallucination on the video-SALMONN 2 testset. Given the absence of a dedicated benchmark for omni detailed perception, we design Omni-Cloze, a novel cloze-style evaluation for detailed audio, visual, and audio-visual captioning that ensures stable, efficient, and reliable assessment. Experimental results and analysis demonstrate the effectiveness of Omni-Detective in generating high-quality detailed captions, as well as the superiority of Omni-Cloze in evaluating such detailed captions. 

---
# Generation Space Size: Understanding and Calibrating Open-Endedness of LLM Generations 

**Authors**: Sunny Yu, Ahmad Jabbar, Robert Hawkins, Dan Jurafsky, Myra Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12699)  

**Abstract**: Different open-ended generation tasks require different degrees of output diversity. However, current LLMs are often miscalibrated. They collapse to overly homogeneous outputs for creative tasks and hallucinate diverse but incorrect responses for factual tasks. We argue that these two failure modes are unified by, and can both be addressed by, the notion of effective generation space size (GSS) -- the set of semantically distinct outputs a model considers for a prompt. We present GSSBench, a task suite of prompt pairs with ground-truth GSS relationships to assess different metrics and understand where models diverge from desired behavior. We find that hallucination detection metrics, particularly EigenScore, consistently outperform standard diversity and uncertainty quantification metrics, while using only model internals, providing interpretable insights into a model's internal task representations. We demonstrate three applications of GSS: (1) detecting prompt ambiguity and predicting clarification questions for better grounding, (2) interpreting overthinking and underthinking in reasoning models, and (3) steering models to expand their generation space to yield high-quality and diverse outputs. 

---
# Reasoning Pattern Matters: Learning to Reason without Human Rationales 

**Authors**: Chaoxu Pang, Yixuan Cao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12643)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities under the widely adopted SFT+RLVR paradigm, which first performs Supervised Fine-Tuning (SFT) on human-annotated reasoning trajectories (rationales) to establish initial reasoning behaviors, then applies Reinforcement Learning with Verifiable Rewards (RLVR) to optimize the model using verifiable signals without golden rationales. However, annotating high-quality rationales for the SFT stage remains prohibitively expensive. This paper investigates when and how rationale annotation costs can be substantially reduced without compromising reasoning performance. We identify a broad class of problems, termed patterned reasoning tasks, where reasoning follows a fixed, procedural strategy consistent across instances. Although instances vary in content such as domain knowledge, factual information, or numeric values, the solution derives from applying a shared reasoning pattern. We argue that the success of SFT+RLVR on such tasks primarily stems from its ability to enable models to internalize these reasoning patterns. Using numerical semantic matching as a representative task, we provide both causal and behavioral evidence showing that reasoning patterns rather than the quantity or quality of rationales are the key determinant of performance. Building on these insights, we propose Pattern-Aware LLMs as Rationale AnnOtators (PARO), a simple yet effective framework that enables LLMs to generate rationales aligned with task-specific reasoning patterns without requiring human rationale annotations. Experiments show that PARO-generated rationales achieve comparable SFT+RLVR performance to human rationales that are 10 times larger. These results suggest that large-scale human rationale annotations can be replaced with LLM-based automatic annotations requiring only limited human supervision over reasoning patterns. 

---
# COSTAR-A: A prompting framework for enhancing Large Language Model performance on Point-of-View questions 

**Authors**: Nzubechukwu C. Ohalete, Kevin B. Gittner, Lauren M. Matheny  

**Link**: [PDF](https://arxiv.org/pdf/2510.12637)  

**Abstract**: Large Language Models (LLMs) are highly sensitive to prompt design, and making optimized prompting techniques is crucial for generating consistent, high-quality outputs. In this study, we introduce COSTAR-A, a novel prompt engineering framework that enhances the existing COSTAR method, which stands for Context, Objective, Style, Tone, Audience, and Response, by adding the 'Answer' component at the end. We demonstrate that while the original COSTAR framework improves prompt clarity and aligns outputs for larger LLMs, its performance is less consistent with smaller, locally optimized models, particularly in tasks that require more directive or constrained outputs. Through a series of controlled prompt-output assessments with smaller (at most 8 billion parameters), fine-tuned models, we found that COSTAR-A can enhance the output structure and decisiveness of localized LLMs for certain tasks, although its effectiveness varies across models and use cases. Notably, the Llama 3.1-8B model exhibited performance improvements when prompted with COSTAR-A compared to COSTAR alone. These findings emphasize the adaptability and scalability of COSTAR-A as a prompting framework, particularly in computationally efficient AI deployments on resource-constrained hardware. 

---
# ACADATA: Parallel Dataset of Academic Data for Machine Translation 

**Authors**: Iñaki Lacunza, Javier Garcia Gilabert, Francesca De Luca Fornaciari, Javier Aula-Blasco, Aitor Gonzalez-Agirre, Maite Melero, Marta Villegas  

**Link**: [PDF](https://arxiv.org/pdf/2510.12621)  

**Abstract**: We present ACADATA, a high-quality parallel dataset for academic translation, that consists of two subsets: ACAD-TRAIN, which contains approximately 1.5 million author-generated paragraph pairs across 96 language directions and ACAD-BENCH, a curated evaluation set of almost 6,000 translations covering 12 directions. To validate its utility, we fine-tune two Large Language Models (LLMs) on ACAD-TRAIN and benchmark them on ACAD-BENCH against specialized machine-translation systems, general-purpose, open-weight LLMs, and several large-scale proprietary models. Experimental results demonstrate that fine-tuning on ACAD-TRAIN leads to improvements in academic translation quality by +6.1 and +12.4 d-BLEU points on average for 7B and 2B models respectively, while also improving long-context translation in a general domain by up to 24.9% when translating out of English. The fine-tuned top-performing model surpasses the best propietary and open-weight models on academic translation domain. By releasing ACAD-TRAIN, ACAD-BENCH and the fine-tuned models, we provide the community with a valuable resource to advance research in academic domain and long-context translation. 

---
# StyleDecipher: Robust and Explainable Detection of LLM-Generated Texts with Stylistic Analysis 

**Authors**: Siyuan Li, Aodu Wulianghai, Xi Lin, Guangyan Li, Xiang Chen, Jun Wu, Jianhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.12608)  

**Abstract**: With the increasing integration of large language models (LLMs) into open-domain writing, detecting machine-generated text has become a critical task for ensuring content authenticity and trust. Existing approaches rely on statistical discrepancies or model-specific heuristics to distinguish between LLM-generated and human-written text. However, these methods struggle in real-world scenarios due to limited generalization, vulnerability to paraphrasing, and lack of explainability, particularly when facing stylistic diversity or hybrid human-AI authorship. In this work, we propose StyleDecipher, a robust and explainable detection framework that revisits LLM-generated text detection using combined feature extractors to quantify stylistic differences. By jointly modeling discrete stylistic indicators and continuous stylistic representations derived from semantic embeddings, StyleDecipher captures distinctive style-level divergences between human and LLM outputs within a unified representation space. This framework enables accurate, explainable, and domain-agnostic detection without requiring access to model internals or labeled segments. Extensive experiments across five diverse domains, including news, code, essays, reviews, and academic abstracts, demonstrate that StyleDecipher consistently achieves state-of-the-art in-domain accuracy. Moreover, in cross-domain evaluations, it surpasses existing baselines by up to 36.30%, while maintaining robustness against adversarial perturbations and mixed human-AI content. Further qualitative and quantitative analysis confirms that stylistic signals provide explainable evidence for distinguishing machine-generated text. Our source code can be accessed at this https URL. 

---
# Teaching Language Models to Faithfully Express their Uncertainty 

**Authors**: Bryan Eikema, Evgenia Ilia, José G. C. de Souza, Chrysoula Zerva, Wilker Aziz  

**Link**: [PDF](https://arxiv.org/pdf/2510.12587)  

**Abstract**: Large language models (LLMs) often miscommunicate their uncertainty: repeated queries can produce divergent answers, yet generated responses are typically unhedged or hedged in ways that do not reflect this variability. This conveys unfaithful information about the uncertain state of the LLMs' knowledge, creating a faithfulness gap that affects even strong LLMs. We introduce Faithful Uncertainty Tuning (FUT): a fine-tuning approach that teaches instruction-tuned LLMs to express uncertainty faithfully without altering their underlying answer distribution. We construct training data by augmenting model samples with uncertainty hedges (i.e. verbal cues such as 'possibly' or 'likely') aligned with sample consistency, requiring no supervision beyond the model and a set of prompts. We evaluate FUT on open-domain question answering (QA) across multiple models and datasets. Our results show that FUT substantially reduces the faithfulness gap, while preserving QA accuracy and introducing minimal semantic distribution shift. Further analyses demonstrate robustness across decoding strategies, choice of hedgers, and other forms of uncertainty expression (i.e. numerical). These findings establish FUT as a simple and effective way to teach LLMs to communicate uncertainty faithfully. 

---
# VISaGE: Understanding Visual Generics and Exceptions 

**Authors**: Stella Frank, Emily Allaway  

**Link**: [PDF](https://arxiv.org/pdf/2510.12548)  

**Abstract**: While Vision Language Models (VLMs) learn conceptual representations, in the form of generalized knowledge, during training, they are typically used to analyze individual instances. When evaluation instances are atypical, this paradigm results in tension between two priors in the model. The first is a pragmatic prior that the textual and visual input are both relevant, arising from VLM finetuning on congruent inputs; the second is a semantic prior that the conceptual representation is generally true for instances of the category. In order to understand how VLMs trade off these priors, we introduce a new evaluation dataset, VISaGE, consisting of both typical and exceptional images. In carefully balanced experiments, we show that conceptual understanding degrades when the assumption of congruency underlying the pragmatic prior is violated with incongruent images. This effect is stronger than the effect of the semantic prior when querying about individual instances. 

---
# BoN Appetit Team at LeWiDi-2025: Best-of-N Test-time Scaling Can Not Stomach Annotation Disagreements (Yet) 

**Authors**: Tomas Ruiz, Siyao Peng, Barbara Plank, Carsten Schwemmer  

**Link**: [PDF](https://arxiv.org/pdf/2510.12516)  

**Abstract**: Test-time scaling is a family of techniques to improve LLM outputs at inference time by performing extra computation. To the best of our knowledge, test-time scaling has been limited to domains with verifiably correct answers, like mathematics and coding. We transfer test-time scaling to the LeWiDi-2025 tasks to evaluate annotation disagreements. We experiment with three test-time scaling methods: two benchmark algorithms (Model Averaging and Majority Voting), and a Best-of-N sampling method. The two benchmark methods improve LLM performance consistently on the LeWiDi tasks, but the Best-of-N method does not. Our experiments suggest that the Best-of-N method does not currently transfer from mathematics to LeWiDi tasks, and we analyze potential reasons for this gap. 

---
# When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection 

**Authors**: Lang Gao, Xuhui Li, Chenxi Wang, Mingzhe Li, Wei Liu, Zirui Song, Jinghui Zhang, Rui Yan, Preslav Nakov, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12476)  

**Abstract**: Large language models (LLMs) have grown more powerful in language generation, producing fluent text and even imitating personal style. Yet, this ability also heightens the risk of identity impersonation. To the best of our knowledge, no prior work has examined personalized machine-generated text (MGT) detection. In this paper, we introduce \dataset, the first benchmark for evaluating detector robustness in personalized settings, built from literary and blog texts paired with their LLM-generated imitations. Our experimental results demonstrate large performance gaps across detectors in personalized settings: some state-of-the-art models suffer significant drops. We attribute this limitation to the \textit{feature-inversion trap}, where features that are discriminative in general domains become inverted and misleading when applied to personalized text. Based on this finding, we propose \method, a simple and reliable way to predict detector performance changes in personalized settings. \method identifies latent directions corresponding to inverted features and constructs probe datasets that differ primarily along these features to evaluate detector dependence. Our experiments show that \method can accurately predict both the direction and the magnitude of post-transfer changes, showing 85\% correlation with the actual performance gaps. We hope that this work will encourage further research on personalized text detection. 

---
# SMEC: Rethinking Matryoshka Representation Learning for Retrieval Embedding Compression 

**Authors**: Biao Zhang, Lixin Chen, Tong Liu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12474)  

**Abstract**: Large language models (LLMs) generate high-dimensional embeddings that capture rich semantic and syntactic information. However, high-dimensional embeddings exacerbate computational complexity and storage requirements, thereby hindering practical deployment. To address these challenges, we propose a novel training framework named Sequential Matryoshka Embedding Compression (SMEC). This framework introduces the Sequential Matryoshka Representation Learning(SMRL) method to mitigate gradient variance during training, the Adaptive Dimension Selection (ADS) module to reduce information degradation during dimension pruning, and the Selectable Cross-batch Memory (S-XBM) module to enhance unsupervised learning between high- and low-dimensional embeddings. Experiments on image, text, and multimodal datasets demonstrate that SMEC achieves significant dimensionality reduction while maintaining performance. For instance, on the BEIR dataset, our approach improves the performance of compressed LLM2Vec embeddings (256 dimensions) by 1.1 points and 2.7 points compared to the Matryoshka-Adaptor and Search-Adaptor models, respectively. 

---
# Resource-sensitive but language-blind: Community size and not grammatical complexity better predicts the accuracy of Large Language Models in a novel Wug Test 

**Authors**: Nikoleta Pantelidou, Evelina Leivada, Paolo Morosi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12463)  

**Abstract**: The linguistic abilities of Large Language Models are a matter of ongoing debate. This study contributes to this discussion by investigating model performance in a morphological generalization task that involves novel words. Using a multilingual adaptation of the Wug Test, six models were tested across four partially unrelated languages (Catalan, English, Greek, and Spanish) and compared with human speakers. The aim is to determine whether model accuracy approximates human competence and whether it is shaped primarily by linguistic complexity or by the quantity of available training data. Consistent with previous research, the results show that the models are able to generalize morphological processes to unseen words with human-like accuracy. However, accuracy patterns align more closely with community size and data availability than with structural complexity, refining earlier claims in the literature. In particular, languages with larger speaker communities and stronger digital representation, such as Spanish and English, revealed higher accuracy than less-resourced ones like Catalan and Greek. Overall, our findings suggest that model behavior is mainly driven by the richness of linguistic resources rather than by sensitivity to grammatical complexity, reflecting a form of performance that resembles human linguistic competence only superficially. 

---
# Probing Latent Knowledge Conflict for Faithful Retrieval-Augmented Generation 

**Authors**: Linfeng Gao, Baolong Bi, Zheng Yuan, Le Wang, Zerui Chen, Zhimin Wei, Shenghua Liu, Qinggang Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.12460)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to enhance the factuality of Large Language Models (LLMs). However, existing RAG systems often suffer from an unfaithfulness issue, where the model's response contradicts evidence from the retrieved context. Existing approaches to improving contextual faithfulness largely rely on external interventions, such as prompt engineering, decoding constraints, or reward-based fine-tuning. These works treat the LLM as a black box and overlook a crucial question: how does the LLM internally integrate retrieved evidence with its parametric memory, particularly under knowledge conflicts? To address this gap, we conduct a probing-based analysis of hidden-state representations in LLMs and observe three findings: knowledge integration occurs hierarchically, conflicts manifest as latent signals at the sentence level, and irrelevant context is often amplified when aligned with parametric knowledge. Building on these findings, we propose CLEAR (Conflict-Localized and Enhanced Attention for RAG), a framework that (i) decomposes context into fine-grained sentence-level knowledge, (ii) employs hidden-state probing to localize conflicting knowledge, and (iii) introduces conflict-aware fine-tuning to guide the model to accurately integrate retrieved evidence. Extensive experiments across three benchmarks demonstrate that CLEAR substantially improves both accuracy and contextual faithfulness, consistently outperforming strong baselines under diverse conflict conditions. The related resources are available at this https URL. 

---
# PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation 

**Authors**: Xiangjun Zai, Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12434)  

**Abstract**: Knowledge Hypergraphs (KHs) have recently emerged as a knowledge representation for retrieval-augmented generation (RAG), offering a paradigm to model multi-entity relations into a structured form. However, existing KH-based RAG methods suffer from three major limitations: static retrieval planning, non-adaptive retrieval execution, and superficial use of KH structure and semantics, which constrain their ability to perform effective multi-hop question answering. To overcome these limitations, we propose PRoH, a dynamic Planning and Reasoning over Knowledge Hypergraphs framework. PRoH incorporates three core innovations: (i) a context-aware planning module that sketches the local KH neighborhood to guide structurally grounded reasoning plan generation; (ii) a structured question decomposition process that organizes subquestions as a dynamically evolving Directed Acyclic Graph (DAG) to enable adaptive, multi-trajectory exploration; and (iii) an Entity-Weighted Overlap (EWO)-guided reasoning path retrieval algorithm that prioritizes semantically coherent hyperedge traversals. Experiments across multiple domains demonstrate that PRoH achieves state-of-the-art performance, surpassing the prior SOTA model HyperGraphRAG by an average of 19.73% in F1 and 8.41% in Generation Evaluation (G-E) score, while maintaining strong robustness in long-range multi-hop reasoning tasks. 

---
# Tokenization Disparities as Infrastructure Bias: How Subword Systems Create Inequities in LLM Access and Efficiency 

**Authors**: Hailay Kidu Teklehaymanot, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2510.12389)  

**Abstract**: Tokenization disparities pose a significant barrier to achieving equitable access to artificial intelligence across linguistically diverse populations. This study conducts a large-scale cross-linguistic evaluation of tokenization efficiency in over 200 languages to systematically quantify computational inequities in large language models (LLMs). Using a standardized experimental framework, we applied consistent preprocessing and normalization protocols, followed by uniform tokenization through the tiktoken library across all language samples. Comprehensive tokenization statistics were collected using established evaluation metrics, including Tokens Per Sentence (TPS) and Relative Tokenization Cost (RTC), benchmarked against English baselines. Our cross-linguistic analysis reveals substantial and systematic disparities: Latin-script languages consistently exhibit higher tokenization efficiency, while non-Latin and morphologically complex languages incur significantly greater token inflation, often 3-5 times higher RTC ratios. These inefficiencies translate into increased computational costs and reduced effective context utilization for underrepresented languages. Overall, the findings highlight structural inequities in current AI systems, where speakers of low-resource and non-Latin languages face disproportionate computational disadvantages. Future research should prioritize the development of linguistically informed tokenization strategies and adaptive vocabulary construction methods that incorporate typological diversity, ensuring more inclusive and computationally equitable multilingual AI systems. 

---
# LLM-REVal: Can We Trust LLM Reviewers Yet? 

**Authors**: Rui Li, Jia-Chen Gu, Po-Nien Kung, Heming Xia, Junfeng liu, Xiangwen Kong, Zhifang Sui, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12367)  

**Abstract**: The rapid advancement of large language models (LLMs) has inspired researchers to integrate them extensively into the academic workflow, potentially reshaping how research is practiced and reviewed. While previous studies highlight the potential of LLMs in supporting research and peer review, their dual roles in the academic workflow and the complex interplay between research and review bring new risks that remain largely underexplored. In this study, we focus on how the deep integration of LLMs into both peer-review and research processes may influence scholarly fairness, examining the potential risks of using LLMs as reviewers by simulation. This simulation incorporates a research agent, which generates papers and revises, alongside a review agent, which assesses the submissions. Based on the simulation results, we conduct human annotations and identify pronounced misalignment between LLM-based reviews and human judgments: (1) LLM reviewers systematically inflate scores for LLM-authored papers, assigning them markedly higher scores than human-authored ones; (2) LLM reviewers persistently underrate human-authored papers with critical statements (e.g., risk, fairness), even after multiple revisions. Our analysis reveals that these stem from two primary biases in LLM reviewers: a linguistic feature bias favoring LLM-generated writing styles, and an aversion toward critical statements. These results highlight the risks and equity concerns posed to human authors and academic research if LLMs are deployed in the peer review cycle without adequate caution. On the other hand, revisions guided by LLM reviews yield quality gains in both LLM-based and human evaluations, illustrating the potential of the LLMs-as-reviewers for early-stage researchers and enhancing low-quality papers. 

---
# MoBiLE: Efficient Mixture-of-Experts Inference on Consumer GPU with Mixture of Big Little Experts 

**Authors**: Yushu Zhao, Yubin Qin, Yang Wang, Xiaolong Yang, Huiming Han, Shaojun Wei, Yang Hu, Shouyi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.12357)  

**Abstract**: Mixture-of-Experts (MoE) models have recently demonstrated exceptional performance across a diverse range of applications. The principle of sparse activation in MoE models facilitates an offloading strategy, wherein active experts are maintained in GPU HBM, while inactive experts are stored in CPU DRAM. The efficacy of this approach, however, is fundamentally constrained by the limited bandwidth of the CPU-GPU interconnect. To mitigate this bottleneck, existing approaches have employed prefetching to accelerate MoE inference. These methods attempt to predict and prefetch the required experts using specially trained modules. Nevertheless, such techniques are often encumbered by significant training overhead and have shown diminished effectiveness on recent MoE models with fine-grained expert segmentation.
In this paper, we propose MoBiLE, a plug-and-play offloading-based MoE inference framework with \textit{mixture of big-little experts}. It reduces the number of experts for unimportant tokens to half for acceleration while maintaining full experts for important tokens to guarantee model quality. Further, a dedicated fallback and prefetching mechanism is designed for switching between little and big experts to improve memory efficiency. We evaluate MoBiLE on four typical modern MoE architectures and challenging generative tasks. Our results show that MoBiLE achieves a speedup of 1.60x to 1.72x compared to the baseline on a consumer GPU system, with negligible degradation in accuracy. 

---
# Fine-grained Analysis of Brain-LLM Alignment through Input Attribution 

**Authors**: Michela Proietti, Roberto Capobianco, Mariya Toneva  

**Link**: [PDF](https://arxiv.org/pdf/2510.12355)  

**Abstract**: Understanding the alignment between large language models (LLMs) and human brain activity can reveal computational principles underlying language processing. We introduce a fine-grained input attribution method to identify the specific words most important for brain-LLM alignment, and leverage it to study a contentious research question about brain-LLM alignment: the relationship between brain alignment (BA) and next-word prediction (NWP). Our findings reveal that BA and NWP rely on largely distinct word subsets: NWP exhibits recency and primacy biases with a focus on syntax, while BA prioritizes semantic and discourse-level information with a more targeted recency effect. This work advances our understanding of how LLMs relate to human language processing and highlights differences in feature reliance between BA and NWP. Beyond this study, our attribution method can be broadly applied to explore the cognitive relevance of model predictions in diverse language processing tasks. 

---
# Beating Harmful Stereotypes Through Facts: RAG-based Counter-speech Generation 

**Authors**: Greta Damo, Elena Cabrio, Serena Villata  

**Link**: [PDF](https://arxiv.org/pdf/2510.12316)  

**Abstract**: Counter-speech generation is at the core of many expert activities, such as fact-checking and hate speech, to counter harmful content. Yet, existing work treats counter-speech generation as pure text generation task, mainly based on Large Language Models or NGO experts. These approaches show severe drawbacks due to the limited reliability and coherence in the generated countering text, and in scalability, respectively. To close this gap, we introduce a novel framework to model counter-speech generation as knowledge-wise text generation process. Our framework integrates advanced Retrieval-Augmented Generation (RAG) pipelines to ensure the generation of trustworthy counter-speech for 8 main target groups identified in the hate speech literature, including women, people of colour, persons with disabilities, migrants, Muslims, Jews, LGBT persons, and other. We built a knowledge base over the United Nations Digital Library, EUR-Lex and the EU Agency for Fundamental Rights, comprising a total of 32,792 texts. We use the MultiTarget-CONAN dataset to empirically assess the quality of the generated counter-speech, both through standard metrics (i.e., JudgeLM) and a human evaluation. Results show that our framework outperforms standard LLM baselines and competitive approach, on both assessments. The resulting framework and the knowledge base pave the way for studying trustworthy and sound counter-speech generation, in hate speech and beyond. 

---
# A large-scale, unsupervised pipeline for automatic corpus annotation using LLMs: variation and change in the English consider construction 

**Authors**: Cameron Morin, Matti Marttinen Larsson  

**Link**: [PDF](https://arxiv.org/pdf/2510.12306)  

**Abstract**: As natural language corpora expand at an unprecedented rate, manual annotation remains a significant methodological bottleneck in corpus linguistic work. We address this challenge by presenting a scalable, unsupervised pipeline for automating grammatical annotation in voluminous corpora using large language models (LLMs). Unlike previous supervised and iterative approaches, our method employs a four-phase workflow: prompt engineering, pre-hoc evaluation, automated batch processing, and post-hoc validation. We demonstrate the pipeline's accessibility and effectiveness through a diachronic case study of variation in the English consider construction. Using GPT-5 through the OpenAI API, we annotate 143,933 sentences from the Corpus of Historical American English (COHA) in under 60 hours, achieving 98%+ accuracy on two sophisticated annotation procedures. Our results suggest that LLMs can perform a range of data preparation tasks at scale with minimal human intervention, opening new possibilities for corpus-based research, though implementation requires attention to costs, licensing, and other ethical considerations. 

---
# Chinese ModernBERT with Whole-Word Masking 

**Authors**: Zeyu Zhao, Ningtao Wang, Xing Fu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12285)  

**Abstract**: Encoder-only Transformers have advanced along three axes -- architecture, data, and systems -- yielding Pareto gains in accuracy, speed, and memory efficiency. Yet these improvements have not fully transferred to Chinese, where tokenization and morphology differ markedly from English. We introduce Chinese ModernBERT, a from-scratch Chinese encoder that couples: (i) a hardware-aware 32k BPE vocabulary tailored to frequent Chinese affixes/compounds, lowering the embedding budget; (ii) whole-word masking (WWM) with a dynamic masking curriculum (30% -> 15%) to align task difficulty with training progress; (iii) a two-stage pre-training pipeline that extends the native context from 1,024 to 8,192 tokens using RoPE and alternating local/global attention; and (iv) a damped-cosine learning-rate schedule for stable long-horizon optimization. We pre-train on ~1.2T Chinese tokens from CCI3-HQ, CCI4 (Chinese), and Cosmopedia-Chinese. On CLUE, Chinese ModernBERT is competitive with strong Chinese encoders under a unified fine-tuning protocol. Under bf16 it achieves high long-sequence throughput while maintaining strong short-sequence speed, reflecting benefits from budget allocation and attention design. To probe retrieval-oriented quality, we add a small amount of open contrastive data: fine-tuning on SimCLUE (~3M pairs) improves further when adding T2Ranking (~2M), reaching 0.505 (Pearson) / 0.537 (Spearman) on the SimCLUE test set. Under this open-data setting, Chinese ModernBERT surpasses Qwen-0.6B-embedding on SimCLUE, suggesting a clear scaling path for STS with additional curated pairs. We will release tokenizer and weights to facilitate reproducible research. 

---
# Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs 

**Authors**: Blazej Manczak, Eric Lin, Francisco Eiras, James O' Neill, Vaikkunth Mugunthan  

**Link**: [PDF](https://arxiv.org/pdf/2510.12255)  

**Abstract**: Large language models (LLMs) are rapidly transitioning into medical clinical use, yet their reliability under realistic, multi-turn interactions remains poorly understood. Existing evaluation frameworks typically assess single-turn question answering under idealized conditions, overlooking the complexities of medical consultations where conflicting input, misleading context, and authority influence are common. We introduce MedQA-Followup, a framework for systematically evaluating multi-turn robustness in medical question answering. Our approach distinguishes between shallow robustness (resisting misleading initial context) and deep robustness (maintaining accuracy when answers are challenged across turns), while also introducing an indirect-direct axis that separates contextual framing (indirect) from explicit suggestion (direct). Using controlled interventions on the MedQA dataset, we evaluate five state-of-the-art LLMs and find that while models perform reasonably well under shallow perturbations, they exhibit severe vulnerabilities in multi-turn settings, with accuracy dropping from 91.2% to as low as 13.5% for Claude Sonnet 4. Counterintuitively, indirect, context-based interventions are often more harmful than direct suggestions, yielding larger accuracy drops across models and exposing a significant vulnerability for clinical deployment. Further compounding analyses reveal model differences, with some showing additional performance drops under repeated interventions while others partially recovering or even improving. These findings highlight multi-turn robustness as a critical but underexplored dimension for safe and reliable deployment of medical LLMs. 

---
# DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering 

**Authors**: Jiakai Li, Rongzheng Wang, Yizhuo Ma, Shuang Liang, Guangchun Luo, Ke Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.12251)  

**Abstract**: While large language models (LLMs) show considerable promise across various fields, they have notable limitations in handling multi-document question answering (Multi-doc QA) tasks. The first challenge is long-range dependency modeling, where LLMs struggle to focus on key information in long texts, which weakens important semantic connections. Second, most LLMs suffer from the ''lost-in-the-middle'' issue, where they have difficulty processing information in the middle of long inputs. Current solutions either truncate global dependencies or demand costly finetuning, ultimately lacking a universal and simple solution for these challenges. To resolve these limitations, we propose Dual-Stage Adaptive Sharpening (DSAS) containing two modules. (i) The Contextual Gate Weighting (CGW) module alleviates ''lost-in-the-middle'' by assessing paragraph relevance through layer-wise attention tracking and position-aware weighting. (ii) The Reciprocal Attention Suppression (RAS) module enhances focus on critical paragraphs by suppressing information exchange between key and irrelevant texts, thus mitigating the limitations in long-range dependency modeling. Notably, DSAS functions as a plug-and-play solution requiring no architectural modifications or extra training parameters. Extensive experiments on four benchmarks demonstrate DSAS's efficacy across mainstream LLMs (Llama, Qwen, Mistral, and Deepseek), with an average F1-score improvement of 4.2% in Multi-doc QA tasks on Llama-3.1-8B-Instruct and Qwen2.5-14B-Instruct. Ablation studies confirm the essential contributions of both the CGW and RAS modules. In addition, detailed discussions in the Appendix further validate the robustness and scalability of DSAS. 

---
# Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability 

**Authors**: Bianca Raimondi, Daniela Dalbagno, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2510.12229)  

**Abstract**: Large language models (LLMs) have been shown to internalize human-like biases during finetuning, yet the mechanisms by which these biases manifest remain unclear. In this work, we investigated whether the well-known Knobe effect, a moral bias in intentionality judgements, emerges in finetuned LLMs and whether it can be traced back to specific components of the model. We conducted a Layer-Patching analysis across 3 open-weights LLMs and demonstrated that the bias is not only learned during finetuning but also localized in a specific set of layers. Surprisingly, we found that patching activations from the corresponding pretrained model into just a few critical layers is sufficient to eliminate the effect. Our findings offer new evidence that social biases in LLMs can be interpreted, localized, and mitigated through targeted interventions, without the need for model retraining. 

---
# HALF: Harm-Aware LLM Fairness Evaluation Aligned with Deployment 

**Authors**: Ali Mekky, Omar El Herraoui, Preslav Nakov, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12217)  

**Abstract**: Large language models (LLMs) are increasingly deployed across high-impact domains, from clinical decision support and legal analysis to hiring and education, making fairness and bias evaluation before deployment critical. However, existing evaluations lack grounding in real-world scenarios and do not account for differences in harm severity, e.g., a biased decision in surgery should not be weighed the same as a stylistic bias in text summarization. To address this gap, we introduce HALF (Harm-Aware LLM Fairness), a deployment-aligned framework that assesses model bias in realistic applications and weighs the outcomes by harm severity. HALF organizes nine application domains into three tiers (Severe, Moderate, Mild) using a five-stage pipeline. Our evaluation results across eight LLMs show that (1) LLMs are not consistently fair across domains, (2) model size or performance do not guarantee fairness, and (3) reasoning models perform better in medical decision support but worse in education. We conclude that HALF exposes a clear gap between previous benchmarking success and deployment readiness. 

---
# DPO-Tuned Large Language Models for Segmentation in Simultaneous Speech Translation 

**Authors**: Zeyu Yang, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2510.12195)  

**Abstract**: Simultaneous speech translation requires accurate segmentation to balance translation quality and latency. Recent studies such as SHAS have introduced pretrained segmentation models, achieving stronger performance than heuristic rules. However, segmentation models such as SHAS, though pretrained and more robust than heuristic methods, are still constrained by supervised learning objectives and do not incorporate human preference alignment, which is crucial for natural real-time interpretation. In this work, we propose a segmentation framework based on large language models (LLMs) trained with Direct Preference Optimization (DPO). By leveraging preference alignment, our method enables LLMs to predict natural segmentation points that better meet the demands of real-time translation. We evaluate the system on the ACL 60/60 corpus across three language pairs (English-Japanese, Chinese, German), using SeamlessM4T v2 as the translation backbone. Experimental results show that our DPO-tuned LLM achieves higher segmentation accuracy than SHAS and yields consistent improvements in translation quality (BLEU, COMET) as well as latency (Average Lagging). Furthermore, our system benefits from IWSLT baselines for direct comparison. These findings highlight the potential of preference-tuned LLMs to surpass existing pretrained segmentation models and advance adaptive, human-aligned simultaneous interpretation. 

---
# Not in Sync: Unveiling Temporal Bias in Audio Chat Models 

**Authors**: Jiayu Yao, Shenghua Liu, Yiwei Wang, Rundong Cheng, Lingrui Mei, Baolong Bi, Zhen Xiong, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12185)  

**Abstract**: Large Audio Language Models (LALMs) are increasingly applied to audio understanding and multimodal reasoning, yet their ability to locate when events occur remains underexplored. We present the first systematic study of temporal bias in LALMs, revealing a key limitation in their timestamp prediction. For example, when asked "At which second does the lecturer introduce the key formula?", models often predict timestamps that are consistently earlier or later than the ground truth. Through controlled experiments on timestamped datasets, we find that temporal bias (i) is prevalent across datasets and models, (ii) increases with audio length - even accumulating to tens of seconds in extended recordings, and (iii) varies across event types and positions. We quantify this effect with the Temporal Bias Index (TBI), measuring systematic misalignment in predicted event timings, and complement it with a visualization framework. Our findings highlight a fundamental limitation in current LALMs and call for the development of temporally robust architectures. 

---
# From Knowledge to Treatment: Large Language Model Assisted Biomedical Concept Representation for Drug Repurposing 

**Authors**: Chengrui Xiang, Tengfei Ma, Xiangzheng Fu, Yiping Liu, Bosheng Song, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12181)  

**Abstract**: Drug repurposing plays a critical role in accelerating treatment discovery, especially for complex and rare diseases. Biomedical knowledge graphs (KGs), which encode rich clinical associations, have been widely adopted to support this task. However, existing methods largely overlook common-sense biomedical concept knowledge in real-world labs, such as mechanistic priors indicating that certain drugs are fundamentally incompatible with specific treatments. To address this gap, we propose LLaDR, a Large Language Model-assisted framework for Drug Repurposing, which improves the representation of biomedical concepts within KGs. Specifically, we extract semantically enriched treatment-related textual representations of biomedical entities from large language models (LLMs) and use them to fine-tune knowledge graph embedding (KGE) models. By injecting treatment-relevant knowledge into KGE, LLaDR largely improves the representation of biomedical concepts, enhancing semantic understanding of under-studied or complex indications. Experiments based on benchmarks demonstrate that LLaDR achieves state-of-the-art performance across different scenarios, with case studies on Alzheimer's disease further confirming its robustness and effectiveness. Code is available at this https URL. 

---
# Towards Inference-time Scaling for Continuous Space Reasoning 

**Authors**: Minghan Wang, Thuy-Trang Vu, Ehsan Shareghi, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2510.12167)  

**Abstract**: Inference-time scaling through multiple sample generation in combination with Process- or Outcome-Reward Model (PRM or ORM) re-ranking has proven effective for text-based reasoning in large language models. This paper investigates whether such established techniques can be successfully adapted to reasoning in the continuous space, using COCONUT (Hao et al. 2024) continuous space reasoning LM as the backbone. We demonstrate the feasibility of generating diverse reasoning paths through dropout-based sampling. Our Pass@N analysis on the generated samples reveals the potential that could enable a significant gain in performance akin to observed gain in the discrete space. However, we highlight unique challenges faced for materializing this gain in the continuous thought space. In particular, working recipes for data generation and training PRM and ORM models in the discrete space unlocks only marginal improvements in the continuous space. Through probing various aspects including geometric properties and trajectory dynamics we identify the underlying reasons that prevent effective discrimination between correct and incorrect reasoning (essential for the functioning of PRM and ORM). Our findings reveal that current limitations stem from the absence of key inductive biases in continuous thought representations. We argue that the training frameworks for continuous reasoning LMs require not only to optimize for accuracy but also to explicitly incorporate inductive biases that could be utilized during inference-time for discrimination of correct and incorrect thoughts.\footnote{Our code and data will be publicly available.} 

---
# A Survey on Parallel Reasoning 

**Authors**: Ziqi Wang, Boye Niu, Zipeng Gao, Zhi Zheng, Tong Xu, Linghui Meng, Zhongli Li, Jing Liu, Yilong Chen, Chen Zhu, Hua Wu, Haifeng Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12164)  

**Abstract**: With the increasing capabilities of Large Language Models (LLMs), parallel reasoning has emerged as a new inference paradigm that enhances reasoning robustness by concurrently exploring multiple lines of thought before converging on a final answer. It has become a significant trend to explore parallel reasoning to overcome the fragility of standard sequential methods and improve practical performance. In this paper, we aim to survey and summarize the progress and challenges of parallel reasoning. We first present a formal definition of parallel reasoning and clarify its distinction from related concepts like Chain-of-Thought. Then, we organize and discuss advanced techniques based on a novel taxonomy, including non-interactive reasoning, interactive reasoning, and efficiency-focused decoding strategies. Additionally, we explore various application scenarios, such as solving complex problems and enhancing the reliability of LLM this http URL, we highlight the core challenges of parallel reasoning and suggest potential directions for future research. We hope that our work can provide a useful roadmap for beginners and encourage more research on improving parallel reasoning methods. Related source can be avaliable in this https URL. 

---
# Credal Transformer: A Principled Approach for Quantifying and Mitigating Hallucinations in Large Language Models 

**Authors**: Shihao Ji, Zihui Song, Jiajie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12137)  

**Abstract**: Large Language Models (LLMs) hallucinate, generating factually incorrect yet confident assertions. We argue this stems from the Transformer's Softmax function, which creates "Artificial Certainty" by collapsing ambiguous attention scores into a single probability distribution, discarding uncertainty information at each layer. To fix this, we introduce the Credal Transformer, which replaces standard attention with a Credal Attention Mechanism (CAM) based on evidential theory. CAM produces a "credal set" (a set of distributions) instead of a single attention vector, with the set's size directly measuring model uncertainty. We implement this by re-conceptualizing attention scores as evidence masses for a Dirichlet distribution: sufficient evidence recovers standard attention, while insufficient evidence yields a diffuse distribution, representing ambiguity. Empirically, the Credal Transformer identifies out-of-distribution inputs, quantifies ambiguity, and significantly reduces confident errors on unanswerable questions by abstaining. Our contribution is a new architecture to mitigate hallucinations and a design paradigm that integrates uncertainty quantification directly into the model, providing a foundation for more reliable AI. 

---
# SafeMT: Multi-turn Safety for Multimodal Language Models 

**Authors**: Han Zhu, Juntao Dai, Jiaming Ji, Haoran Li, Chengkun Cai, Pengcheng Wen, Chi-Min Chan, Boyuan Chen, Yaodong Yang, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12133)  

**Abstract**: With the widespread use of multi-modal Large Language models (MLLMs), safety issues have become a growing concern. Multi-turn dialogues, which are more common in everyday interactions, pose a greater risk than single prompts; however, existing benchmarks do not adequately consider this situation. To encourage the community to focus on the safety issues of these models in multi-turn dialogues, we introduce SafeMT, a benchmark that features dialogues of varying lengths generated from harmful queries accompanied by images. This benchmark consists of 10,000 samples in total, encompassing 17 different scenarios and four jailbreak methods. Additionally, we propose Safety Index (SI) to evaluate the general safety of MLLMs during conversations. We assess the safety of 17 models using this benchmark and discover that the risk of successful attacks on these models increases as the number of turns in harmful dialogues rises. This observation indicates that the safety mechanisms of these models are inadequate for recognizing the hazard in dialogue interactions. We propose a dialogue safety moderator capable of detecting malicious intent concealed within conversations and providing MLLMs with relevant safety policies. Experimental results from several open-source models indicate that this moderator is more effective in reducing multi-turn ASR compared to existed guard models. 

---
# Understanding the Modality Gap: An Empirical Study on the Speech-Text Alignment Mechanism of Large Speech Language Models 

**Authors**: Bajian Xiang, Shuaijiang Zhao, Tingwei Guo, Wei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.12116)  

**Abstract**: End-to-end Large Speech Language Models (LSLMs) have demonstrated impressive conversational generation abilities, yet consistently fall short of traditional pipeline systems on semantic understanding benchmarks. In this work, we reveal through systematic experimentation that although LSLMs lose some text input performance after speech-text alignment training, the performance gap between speech and text inputs is more pronounced, which we refer to as the modality gap. To understand this gap, we analyze both coarse- and fine-grained text and speech representations. At the coarse-grained level, representations of speech and text in deeper layers are found to be increasingly aligned in direction (cosine similarity), while concurrently diverging in magnitude (Euclidean distance). We further find that representation similarity is strongly correlated with the modality gap. At the fine-grained level, a spontaneous token-level alignment pattern between text and speech representations is observed. Based on this, we introduce the Alignment Path Score to quantify token-level alignment quality, which exhibits stronger correlation with the modality gap. Building on these insights, we design targeted interventions on critical tokens through angle projection and length normalization. These strategies demonstrate the potential to improve correctness for speech inputs. Our study provides the first systematic empirical analysis of the modality gap and alignment mechanisms in LSLMs, offering both theoretical and methodological guidance for future optimization. 

---
# Tracing Multilingual Knowledge Acquisition Dynamics in Domain Adaptation: A Case Study of English-Japanese Biomedical Adaptation 

**Authors**: Xin Zhao, Naoki Yoshinaga, Yuma Tsuta, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2510.12115)  

**Abstract**: Multilingual domain adaptation (ML-DA) is widely used to learn new domain knowledge across languages into large language models (LLMs). Although many methods have been proposed to improve domain adaptation, the mechanisms of multilingual knowledge acquisition, how domain knowledge is learned within a language and transferred across languages, remain underexplored. This gap leads to suboptimal performance, particularly in low-resource settings. This work examines the learning dynamics of LLMs during ML-DA. Because prior ML-DA studies often train and evaluate on datasets with mismatched knowledge coverage, we propose AdaXEval, an adaptive evaluation method that builds multiple-choice QA datasets from the same bilingual domain corpus used for training, thereby directly studying multilingual knowledge acquisition. Through continual training of LLMs with diverse data recipes, we track how LLMs acquire domain facts and pinpoint the mechanism behind the transformation process from domain training data to knowledge. Our experiments on a 13B English-Japanese bilingual LLM reveal that cross-lingual transfer remains challenging despite a high-quality bilingual corpus. The code has been released. 

---
# Deep Associations, High Creativity: A Simple yet Effective Metric for Evaluating Large Language Models 

**Authors**: Ziliang Qiu, Renfen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12110)  

**Abstract**: The evaluation of LLMs' creativity represents a crucial research domain, though challenges such as data contamination and costly human assessments often impede progress. Drawing inspiration from human creativity assessment, we propose PACE, asking LLMs to generate Parallel Association Chains to Evaluate their creativity. PACE minimizes the risk of data contamination and offers a straightforward, highly efficient evaluation, as evidenced by its strong correlation with Chatbot Arena Creative Writing rankings (Spearman's $\rho = 0.739$, $p < 0.001$) across various proprietary and open-source models. A comparative analysis of associative creativity between LLMs and humans reveals that while high-performing LLMs achieve scores comparable to average human performance, professional humans consistently outperform LLMs. Furthermore, linguistic analysis reveals that both humans and LLMs exhibit a trend of decreasing concreteness in their associations, and humans demonstrating a greater diversity of associative patterns. 

---
# An AI-Based Behavioral Health Safety Filter and Dataset for Identifying Mental Health Crises in Text-Based Conversations 

**Authors**: Benjamin W. Nelson, Celeste Wong, Matthew T. Silvestrini, Sooyoon Shin, Alanna Robinson, Jessica Lee, Eric Yang, John Torous, Andrew Trister  

**Link**: [PDF](https://arxiv.org/pdf/2510.12083)  

**Abstract**: Large language models often mishandle psychiatric emergencies, offering harmful or inappropriate advice and enabling destructive behaviors. This study evaluated the Verily behavioral health safety filter (VBHSF) on two datasets: the Verily Mental Health Crisis Dataset containing 1,800 simulated messages and the NVIDIA Aegis AI Content Safety Dataset subsetted to 794 mental health-related messages. The two datasets were clinician-labelled and we evaluated performance using the clinician labels. Additionally, we carried out comparative performance analyses against two open source, content moderation guardrails: OpenAI Omni Moderation Latest and NVIDIA NeMo Guardrails. The VBHSF demonstrated, well-balanced performance on the Verily Mental Health Crisis Dataset v1.0, achieving high sensitivity (0.990) and specificity (0.992) in detecting any mental health crises. It achieved an F1-score of 0.939, sensitivity ranged from 0.917-0.992, and specificity was >= 0.978 in identifying specific crisis categories. When evaluated against the NVIDIA Aegis AI Content Safety Dataset 2.0, VBHSF performance remained highly sensitive (0.982) and accuracy (0.921) with reduced specificity (0.859). When compared with the NVIDIA NeMo and OpenAI Omni Moderation Latest guardrails, the VBHSF demonstrated superior performance metrics across both datasets, achieving significantly higher sensitivity in all cases (all p < 0.001) and higher specificity relative to NVIDIA NeMo (p < 0.001), but not to OpenAI Omni Moderation Latest (p = 0.094). NVIDIA NeMo and OpenAI Omni Moderation Latest exhibited inconsistent performance across specific crisis types, with sensitivity for some categories falling below 0.10. Overall, the VBHSF demonstrated robust, generalizable performance that prioritizes sensitivity to minimize missed crises, a crucial feature for healthcare applications. 

---
# APCE: Adaptive Progressive Context Expansion for Long Context Processing 

**Authors**: Baisub Lee, Sanghyun Byun, Mohanad Odema, Jung Guack, Jacob Song, Woo Seong Chung  

**Link**: [PDF](https://arxiv.org/pdf/2510.12051)  

**Abstract**: Deploying useful Long-Context Transformer Models (LCTMs) requires addressing two key challenges: (1) A growing memory footprint due to quadratic self-attention and linear KV-cache scaling in memory as sequence length increases; (2) the ContextRot phenomena where empirical evidence suggests that transformer architecture's performance degrades with increasing context length. Given the shared dependency on the input, a natural question arises: Can we surgically select the most important input chunks for processing to synergistically (a) reduce the memory footprint, and (b) mitigate the ContextRot effects? In this paper, we answer this question in the affirmative for long-context summarization tasks. We propose APCE as a context-aware solution to select the most important input chunks through low-dimensional semantic similarity matching with the current query. By directly operating on the input, APCE decouples from strict dependency on underlying hardware or CUDA environments, promising a compatible solution scalable to different deployment systems. Our empirical evaluations have demonstrated superior or on-par summarization performance for APCE compared to the full dense baseline using a fraction (50%-70%) of the input sequence resulting in KV-cache and self-attention memory efficiency improvements. We hope our findings inspire further research on context-aware efficiency solutions for LCTMs geared towards other relevant long-context tasks. 

---
# Hierarchical Alignment: Surgical Fine-Tuning via Functional Layer Specialization in Large Language Models 

**Authors**: Yukun Zhang, Qi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.12044)  

**Abstract**: Existing alignment techniques for Large Language Models (LLMs), such as Direct Preference Optimization (DPO), typically treat the model as a monolithic entity, applying uniform optimization pressure across all layers. This approach overlooks the functional specialization within the Transformer architecture, where different layers are known to handle distinct tasks from syntax to abstract reasoning. In this paper, we challenge this one-size-fits-all paradigm by introducing Hierarchical Alignment, a novel method that applies targeted DPO to distinct functional blocks of a model's layers: local (syntax), intermediate (logic), and global (factuality). Through a series of controlled experiments on state-of-the-art models like Llama-3.1-8B and Qwen1.5-7B using LoRA for surgical fine-tuning, our results, evaluated by a powerful LLM-as-Judge, demonstrate significant and predictable improvements. Specifically, aligning the local layers (Local-Align) enhances grammatical fluency. More importantly, aligning the global layers (Global-Align) not only improves factual consistency as hypothesized but also proves to be the most effective strategy for enhancing logical coherence, outperforming all baselines. Critically, all hierarchical strategies successfully avoid the "alignment tax" observed in standard DPO, where gains in fluency come at the cost of degraded logical reasoning. These findings establish a more resource-efficient, controllable, and interpretable path for model alignment, highlighting the immense potential of shifting from monolithic optimization to structure-aware surgical fine-tuning to build more advanced and reliable LLMs. 

---
# Improving Text-to-Image Generation with Input-Side Inference-Time Scaling 

**Authors**: Ruibo Chen, Jiacheng Pan, Heng Huang, Zhenheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12041)  

**Abstract**: Recent advances in text-to-image (T2I) generation have achieved impressive results, yet existing models often struggle with simple or underspecified prompts, leading to suboptimal image-text alignment, aesthetics, and quality. We propose a prompt rewriting framework that leverages large language models (LLMs) to refine user inputs before feeding them into T2I backbones. Our approach introduces a carefully designed reward system and an iterative direct preference optimization (DPO) training pipeline, enabling the rewriter to enhance prompts without requiring supervised fine-tuning data. We evaluate our method across diverse T2I models and benchmarks. Results show that our prompt rewriter consistently improves image-text alignment, visual quality, and aesthetics, outperforming strong baselines. Furthermore, we demonstrate strong transferability by showing that a prompt rewriter trained on one T2I backbone generalizes effectively to others without needing to be retrained. We also systematically study scalability, evaluating how performance gains scale with the capacity of the large LLM used as the rewriter. These findings highlight that prompt rewriting is an effective, scalable, and practical model-agnostic strategy for improving T2I systems. We plan to release the code and trained prompt rewriters soon. 

---
# Uncertainty Quantification for Hallucination Detection in Large Language Models: Foundations, Methodology, and Future Directions 

**Authors**: Sungmin Kang, Yavuz Faruk Bakman, Duygu Nur Yaldiz, Baturalp Buyukates, Salman Avestimehr  

**Link**: [PDF](https://arxiv.org/pdf/2510.12040)  

**Abstract**: The rapid advancement of large language models (LLMs) has transformed the landscape of natural language processing, enabling breakthroughs across a wide range of areas including question answering, machine translation, and text summarization. Yet, their deployment in real-world applications has raised concerns over reliability and trustworthiness, as LLMs remain prone to hallucinations that produce plausible but factually incorrect outputs. Uncertainty quantification (UQ) has emerged as a central research direction to address this issue, offering principled measures for assessing the trustworthiness of model generations. We begin by introducing the foundations of UQ, from its formal definition to the traditional distinction between epistemic and aleatoric uncertainty, and then highlight how these concepts have been adapted to the context of LLMs. Building on this, we examine the role of UQ in hallucination detection, where quantifying uncertainty provides a mechanism for identifying unreliable generations and improving reliability. We systematically categorize a wide spectrum of existing methods along multiple dimensions and present empirical results for several representative approaches. Finally, we discuss current limitations and outline promising future research directions, providing a clearer picture of the current landscape of LLM UQ for hallucination detection. 

---
# On the Interplay between Human Label Variation and Model Fairness 

**Authors**: Kemal Kurniawan, Meladel Mistica, Timothy Baldwin, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2510.12036)  

**Abstract**: The impact of human label variation (HLV) on model fairness is an unexplored topic. This paper examines the interplay by comparing training on majority-vote labels with a range of HLV methods. Our experiments show that without explicit debiasing, HLV training methods have a positive impact on fairness. 

---
# Multi-stage Prompt Refinement for Mitigating Hallucinations in Large Language Models 

**Authors**: Jung-Woo Shim, Yeong-Joon Ju, Ji-Hoon Park, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.12032)  

**Abstract**: Recent advancements in large language models (LLMs) have shown strong performance in natural language understanding and generation tasks. However, LLMs continue to encounter challenges with hallucinations, where models generate plausible but incorrect information. While several factors contribute to hallucinations, the impact of ill-formed prompts, prompts with ambiguous wording, incorrect grammar, or incomplete information, was relatively under explored. To address this, we introduce Multi-stage Prompt Refinement (MPR), a framework designed to systematically improve these ill-formed prompts across multiple stages. Each stage addresses specific errors such as punctuation, typographical mistakes, and misuse of key terms, using small language models (SLMs) fine-tuned for these tasks. MPR iteratively enhances the clarity of prompts with additional context and employs a self-reflection mechanism with ranking to prioritize the most relevant input. Experimental results on hallucination benchmarks show that prompts refined by MPR achieve over an 85~\% win rate compared to their original forms, demonstrating its effectiveness in reducing hallucinations and improving LLM output accuracy. Interestingly, we reveal that MPR can be combined with existing post-hoc hallucination mitigation frameworks, further enhancing its versatility. MPR provides a lightweight and adaptable solution for enhancing LLM reliability across various domains. 

---
# CPR: Mitigating Large Language Model Hallucinations with Curative Prompt Refinement 

**Authors**: Jung-Woo Shim, Yeong-Joon Ju, Ji-Hoon Park, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.12029)  

**Abstract**: Recent advancements in large language models (LLMs) highlight their fluency in generating responses to diverse prompts. However, these models sometimes generate plausible yet incorrect ``hallucinated" facts, undermining trust. A frequent but often overlooked cause of such errors is the use of poorly structured or vague prompts by users, leading LLMs to base responses on assumed rather than actual intentions. To mitigate hallucinations induced by these ill-formed prompts, we introduce Curative Prompt Refinement (CPR), a plug-and-play framework for curative prompt refinement that 1) cleans ill-formed prompts, and 2) generates additional informative task descriptions to align the intention of the user and the prompt using a fine-tuned small language model. When applied to language models, we discover that CPR significantly increases the quality of generation while also mitigating hallucination. Empirical studies show that prompts with CPR applied achieves over a 90\% win rate over the original prompts without any external knowledge. 

---
# Information Extraction from Conversation Transcripts: Neuro-Symbolic vs. LLM 

**Authors**: Alice Saebom Kwak, Maria Alexeeva, Gus Hahn-Powell, Keith Alcock, Kevin McLaughlin, Doug McCorkle, Gabe McNunn, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12023)  

**Abstract**: The current trend in information extraction (IE) is to rely extensively on large language models, effectively discarding decades of experience in building symbolic or statistical IE systems. This paper compares a neuro-symbolic (NS) and an LLM-based IE system in the agricultural domain, evaluating them on nine interviews across pork, dairy, and crop subdomains. The LLM-based system outperforms the NS one (F1 total: 69.4 vs. 52.7; core: 63.0 vs. 47.2), where total includes all extracted information and core focuses on essential details. However, each system has trade-offs: the NS approach offers faster runtime, greater control, and high accuracy in context-free tasks but lacks generalizability, struggles with contextual nuances, and requires significant resources to develop and maintain. The LLM-based system achieves higher performance, faster deployment, and easier maintenance but has slower runtime, limited control, model dependency and hallucination risks. Our findings highlight the "hidden cost" of deploying NLP systems in real-world applications, emphasizing the need to balance performance, efficiency, and control. 

---
# Generate Logical Equivalence Questions 

**Authors**: Xinyu Wang, Haoming Yu, Yicheng Yang, Zhiyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.12001)  

**Abstract**: Academic dishonesty is met with zero tolerance in higher education, yet plagiarism has become increasingly prevalent in the era of online teaching and learning. Automatic Question Generation (AQG) presents a potential solution to mitigate copying by creating unique questions for each student. Additionally, AQG can provide a vast array of practice questions. Our AQG focuses on generating logical equivalence questions for Discrete Mathematics, a foundational course for first-year computer science students. A literature review reveals that existing AQGs for this type of question generate all propositions that meet user-defined constraints, resulting in inefficiencies and a lack of uniform question difficulty. To address this, we propose a new approach that defines logical equivalence questions using a formal language, translates this language into two sets of generation rules, and develops a linear-time algorithm for question generation. We evaluated our AQG through two experiments. The first involved a group of students completing questions generated by our system. Statistical analysis shows that the accuracy of these questions is comparable to that of textbook questions. The second experiment assessed the number of steps required to solve our generated questions, textbook questions, and those generated by multiple large language models. The results indicated that the difficulty of our questions was similar to that of textbook questions, confirming the quality of our AQG. 

---
# SAGE: A Top-Down Bottom-Up Knowledge-Grounded User Simulator for Multi-turn AGent Evaluation 

**Authors**: Ryan Shea, Yunan Lu, Liang Qiu, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11997)  

**Abstract**: Evaluating multi-turn interactive agents is challenging due to the need for human assessment. Evaluation with simulated users has been introduced as an alternative, however existing approaches typically model generic users and overlook the domain-specific principles required to capture realistic behavior. We propose SAGE, a novel user Simulation framework for multi-turn AGent Evaluation that integrates knowledge from business contexts. SAGE incorporates top-down knowledge rooted in business logic, such as ideal customer profiles, grounding user behavior in realistic customer personas. We further integrate bottom-up knowledge taken from business agent infrastructure (e.g., product catalogs, FAQs, and knowledge bases), allowing the simulator to generate interactions that reflect users' information needs and expectations in a company's target market. Through empirical evaluation, we find that this approach produces interactions that are more realistic and diverse, while also identifying up to 33% more agent errors, highlighting its effectiveness as an evaluation tool to support bug-finding and iterative agent improvement. 

---
# Conjecturing: An Overlooked Step in Formal Mathematical Reasoning 

**Authors**: Jasivan Alex Sivakumar, Philipp Borchert, Ronald Cardenas, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.11986)  

**Abstract**: Autoformalisation, the task of expressing informal mathematical statements in formal language, is often viewed as a direct translation process. This, however, disregards a critical preceding step: conjecturing. Many mathematical problems cannot be formalised directly without first conjecturing a conclusion such as an explicit answer, or a specific bound. Since Large Language Models (LLMs) already struggle with autoformalisation, and the evaluation of their conjecturing ability is limited and often entangled within autoformalisation or proof, it is particularly challenging to understand its effect. To address this gap, we augment existing datasets to create ConjectureBench, and redesign the evaluation framework and metric specifically to measure the conjecturing capabilities of LLMs both as a distinct task and within the autoformalisation pipeline. Our evaluation of foundational models, including GPT-4.1 and DeepSeek-V3.1, reveals that their autoformalisation performance is substantially overestimated when the conjecture is accounted for during evaluation. However, the conjecture should not be assumed to be provided. We design an inference-time method, Lean-FIRe to improve conjecturing and autoformalisation, which, to the best of our knowledge, achieves the first successful end-to-end autoformalisation of 13 PutnamBench problems with GPT-4.1 and 7 with DeepSeek-V3.1. We demonstrate that while LLMs possess the requisite knowledge to generate accurate conjectures, improving autoformalisation performance requires treating conjecturing as an independent task, and investigating further how to correctly integrate it within autoformalisation. Finally, we provide forward-looking guidance to steer future research toward improving conjecturing, an overlooked step of formal mathematical reasoning. 

---
# Scaling Long-Horizon LLM Agent via Context-Folding 

**Authors**: Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11967)  

**Abstract**: Large language model (LLM) agents are fundamentally constrained by context length on long-horizon tasks. We introduce Context-Folding, a framework that empowers agents to actively manage their working context. An agent can procedurally branch into a sub-trajectory to handle a subtask and then fold it upon completion, collapsing the intermediate steps while retaining a concise summary of the outcome. To make this behavior learnable, we develop an end-to-end reinforcement learning framework FoldGRPO with specific process rewards to encourage effective task decomposition and context management. On complex long-horizon tasks (Deep Research and SWE), our folding agent matches or outperforms the ReAct baselines while using an active context 10$\times$ smaller and significantly outperforms models that rely on summarization-based context management. 

---
# Direct Multi-Token Decoding 

**Authors**: Xuan Luo, Weizhi Wang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2510.11958)  

**Abstract**: Decoder-only transformers have become the standard architecture for large language models (LLMs) due to their strong performance. Recent studies suggest that, in pre-trained LLMs, early, middle, and late layers may serve distinct roles: Early layers focus on understanding the input context, middle layers handle task-specific processing, and late layers convert abstract representations into output tokens. We hypothesize that once representations have been processed by the early and middle layers, the resulting hidden states may encapsulate sufficient information to support the generation of multiple tokens using only the late layers, eliminating the need to repeatedly traverse the early and middle layers. We refer to this inference paradigm as Direct Multi-Token Decoding (DMTD). Unlike speculative decoding, our method introduces no additional parameters, auxiliary routines, or post-generation verification. Despite being trained on a limited dataset, a fine-tuned DMTD Qwen3-4B model has already demonstrated promising results, achieving up to a 2x speedup with only minor performance loss. Moreover, as shown in our scaling analysis, its performance is expected to further improve with larger training datasets. 

---
# Evaluating Retrieval-Augmented Generation Systems on Unanswerable, Uncheatable, Realistic, Multi-hop Queries 

**Authors**: Gabrielle Kaili-May Liu, Bryan Li, Arman Cohan, William Gantt Walden, Eugene Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11956)  

**Abstract**: Real-world use cases often present RAG systems with complex queries for which relevant information is missing from the corpus or is incomplete. In these settings, RAG systems must be able to reject unanswerable, out-of-scope queries and identify failures of retrieval and multi-hop reasoning. Despite this, existing RAG benchmarks rarely reflect realistic task complexity for multi-hop or out-of-scope questions, which often can be cheated via disconnected reasoning (i.e., solved without genuine multi-hop inference) or require only simple factual recall. This limits the ability for such benchmarks to uncover limitations of existing RAG systems. To address this gap, we present the first pipeline for automatic, difficulty-controlled creation of un$\underline{c}$heatable, $\underline{r}$ealistic, $\underline{u}$nanswerable, and $\underline{m}$ulti-hop $\underline{q}$uerie$\underline{s}$ (CRUMQs), adaptable to any corpus and domain. We use our pipeline to create CRUMQs over two popular RAG datasets and demonstrate its effectiveness via benchmark experiments on leading retrieval-augmented LLMs. Results show that compared to prior RAG benchmarks, CRUMQs are highly challenging for RAG systems and achieve up to 81.0\% reduction in cheatability scores. More broadly, our pipeline offers a simple way to enhance benchmark difficulty and realism and drive development of more capable RAG systems. 

---
# GRAVITY: A Framework for Personalized Text Generation via Profile-Grounded Synthetic Preferences 

**Authors**: Priyanka Dey, Daniele Rosa, Wenqing Zheng, Daniel Barcklow, Jieyu Zhao, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2510.11952)  

**Abstract**: Personalization in LLMs often relies on costly human feedback or interaction logs, limiting scalability and neglecting deeper user attributes. To reduce the reliance on human annotations, we introduce GRAVITY (Generative Response with Aligned Values, Interests, and Traits of You), a framework for generating synthetic, profile-grounded preference data that captures users' interests, values, beliefs, and personality traits. By integrating demographic, cultural, and psychological frameworks -- including Hofstede's cultural dimensions, Schwartz's basic values, the World Values Survey, and Big Five OCEAN traits -- GRAVITY synthesizes preference pairs to guide personalized content generation. We evaluate GRAVITY on book descriptions for 400 Amazon users, comparing it to prompt-based conditioning, standard fine-tuning, and naive synthetic pair generation. Profile-grounded synthetic data consistently improves generation, especially across multiple cultures (USA, Brazil, Japan, India), achieving over 4% higher preference gains across baselines, with user studies showing that GRAVITY outputs are preferred over 86% of the time. Our results show that scenario-grounded synthetic data can capture richer user variation, reduce reliance on costly annotation, and produce more engaging, user-centered content, offering a scalable path for LLM personalization. 

---
# TopoAlign: A Framework for Aligning Code to Math via Topological Decomposition 

**Authors**: Yupei Li, Philipp Borchert, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.11944)  

**Abstract**: Large Language Models (LLMs) excel at both informal and formal (e.g. Lean 4) mathematical reasoning but still struggle with autoformalisation, the task of transforming informal into formal mathematical statements. Autoformalisation helps pair the informal reasoning of LLMs with formal proof assistants which enable machine-verifiable generation and mitigate hallucinations. Yet, the performance of current Math LLMs is constrained by the scarcity of large-scale corpora, particularly those containing pairs of informal and formal statements. Although current models are trained to generate code from natural language instructions, structural and syntactic differences between these and formal mathematics limit effective transfer learning. We propose TopoAlign, a framework that unlocks widely available code repositories as training resources for Math LLMs. TopoAlign decomposes code into docstrings, main functions, and dependency functions, and reassembles these components into analogues that structurally mirror formal statements. This produces structurally aligned code data that can be used for training Math LLMs without requiring additional human annotation. We train two state-of-the-art models, DeepSeek-Math and Herald, and evaluate them on the minif2f, Putnam, and ProofNet benchmarks. TopoAlign provides substantial gains for DeepSeek-Math, improving performance by 17.77% on BEq@10 and 68.82% on typecheck@10. Despite introducing no new mathematical knowledge, our framework achieves gains of 0.12% and 1.09% for Herald on BEq@10 and typecheck@10, respectively, demonstrating that training on aligned code data is beneficial even for specialized models. 

---
# Discrepancy Detection at the Data Level: Toward Consistent Multilingual Question Answering 

**Authors**: Lorena Calvo-Bartolomé, Valérie Aldana, Karla Cantarero, Alonso Madroñal de Mesa, Jerónimo Arenas-García, Jordan Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2510.11928)  

**Abstract**: Multilingual question answering (QA) systems must ensure factual consistency across languages, especially for objective queries such as What is jaundice?, while also accounting for cultural variation in subjective responses. We propose MIND, a user-in-the-loop fact-checking pipeline to detect factual and cultural discrepancies in multilingual QA knowledge bases. MIND highlights divergent answers to culturally sensitive questions (e.g., Who assists in childbirth?) that vary by region and context. We evaluate MIND on a bilingual QA system in the maternal and infant health domain and release a dataset of bilingual questions annotated for factual and cultural inconsistencies. We further test MIND on datasets from other domains to assess generalization. In all cases, MIND reliably identifies inconsistencies, supporting the development of more culturally aware and factually consistent QA systems. 

---
# LLM Reasoning for Machine Translation: Synthetic Data Generation over Thinking Tokens 

**Authors**: Armel Zebaze, Rachel Bawden, Benoît Sagot  

**Link**: [PDF](https://arxiv.org/pdf/2510.11919)  

**Abstract**: Large reasoning models (LRMs) have led to new possibilities in terms of problem-solving, through the devising of a natural language thought process prior to answering a query. While their capabilities are well known across mathematics and coding tasks, their impact on the task of machine translation (MT) remains underexplored. In this work, we explore the benefits of the generation of intermediate tokens when performing MT across multiple language pairs of different levels of resourcedness and multiple setups. We find that "thinking tokens" do not help LRMs better perform MT. This result generalizes to models fine-tuned to reason before translating using distilled chain of thought (CoT) inspired by human translators' practices. Specifically, fine-tuning a model with synthetic CoT explanations detailing how to translate step-by-step does not outperform standard input-output fine-tuning. However, constructing the intermediate tokens by combining the outputs of modular translation-specific prompting strategies results in improvements. Our findings underscore that the contribution of intermediate tokens during fine-tuning highly depends on the presence of translation attempts within them. More broadly, our results suggest that using a teacher to refine target translations or to expand parallel corpora is more impactful than distilling their CoT explanations into "thinking" MT models. 

---
# LLM Knowledge is Brittle: Truthfulness Representations Rely on Superficial Resemblance 

**Authors**: Patrick Haller, Mark Ibrahim, Polina Kirichenko, Levent Sagun, Samuel J. Bell  

**Link**: [PDF](https://arxiv.org/pdf/2510.11905)  

**Abstract**: For Large Language Models (LLMs) to be reliable, they must learn robust knowledge that can be generally applied in diverse settings -- often unlike those seen during training. Yet, extensive research has shown that LLM performance can be brittle, with models exhibiting excessive sensitivity to trivial input variations. In this work, we explore whether this brittleness is a direct result of unstable internal knowledge representations. To explore this question, we build on previous work showing that LLM representations encode statement truthfulness -- i.e., true, factual statements can be easily separated from false, inaccurate ones. Specifically, we test the robustness of learned knowledge by evaluating representation separability on samples that have undergone superficial transformations to drive them out-of-distribution (OOD), such as typos or reformulations. By applying semantically-preserving perturbations, we study how separability degrades as statements become more OOD, across four LLM families, five evaluation datasets, and three knowledge probing methods. Our results reveal that internal representations of statement truthfulness collapse as the samples' presentations become less similar to those seen during pre-training. While LLMs can often distinguish between true and false statements when they closely resemble the pre-training data, this ability is highly dependent on the statement's exact surface form. These findings offer a possible explanation for brittle benchmark performance: LLMs may learn shallow, non-robust knowledge representations that allow for only limited generalizability. Our work presents a fundamental challenge for the utility of truthfulness probes, and more broadly, calls for further research on improving the robustness of learned knowledge representations. 

---
# R-WoM: Retrieval-augmented World Model For Computer-use Agents 

**Authors**: Kai Mei, Jiang Guo, Shuaichen Chang, Mingwen Dong, Dongkyu Lee, Xing Niu, Jiarong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11892)  

**Abstract**: Large Language Models (LLMs) can serve as world models to enhance agent decision-making in digital environments by simulating future states and predicting action outcomes, potentially eliminating costly trial-and-error exploration. However, this capability is fundamentally limited by LLMs' tendency toward hallucination and their reliance on static training knowledge, which can lead to compounding errors that inhibit long-horizon simulations. To systematically investigate whether LLMs are appropriate for world modeling, we probe two core capabilities of world models--future state prediction and reward estimation--through three tasks: next-state identification, full-procedure planning alignment, and milestone transition recognition. Our analysis shows that while LLMs effectively capture immediate next states and identify meaningful state transitions, their performance rapidly degrades in full-procedure planning. This highlights LLMs' limitations in reliably modeling environment dynamics over long horizons. To address these limitations, we propose the Retrieval-augmented World Model (R-WoM), which grounds LLM simulations by incorporating factual, up-to-date knowledge retrieved from external tutorials. Experiments show that R-WoM achieves substantial improvements of up to 25.3% (OSWorld) and 18.1% (WebArena) compared to baselines, with particular advantages in longer-horizon simulations. 

---
# PHANTOM RECALL: When Familiar Puzzles Fool Smart Models 

**Authors**: Souradeep Mukhopadhyay, Rishabh Baral, Nimeesh Mahajan, Samhitha Harish, Aswin RRV, Mihir Parmar, Mutsumi Nakamura, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2510.11812)  

**Abstract**: Large language models (LLMs) such as GPT, Gemini, and Claude often appear adept at solving classic logic puzzles--but how much genuine reasoning underlies their answers? Recent evidence suggests that these models frequently rely on memorized templates rather than reasoning from first principles. When puzzles are slightly modified, their performance collapses, revealing a striking fragility. In particular, we asked: Have LLMs addressed these issues? To what extent? How about perturbations to other puzzles? Is there a general way of reformulating the prompt so that the models do better? To examine these things systematically, we introduce PHANTOM RECALL, a benchmark comprising 25 well-known logic puzzles and 149 carefully designed perturbations that preserve reasoning structure but alter superficial details and solutions. We evaluate eleven leading LLMs and identify a recurring failure mode--phantom recall--where models confidently reproduce memorized solutions or spurious rationales that no longer fit the altered scenario. To probe and mitigate this issue, we contribute three tools: (i) an automated logical-equivalence judge to detect reasoning mismatches, (ii) a taxonomy of fine-grained reasoning error categories, and (iii) a prompting-based mitigation framework guided by these categories. Despite near-perfect accuracy on unmodified puzzles, models significantly underperform humans on perturbed ones, exhibiting both phantom recall and over-elaboration. Our findings reveal a crucial limitation: LLMs often fail to re-reason when contextual cues shift--highlighting the gap between linguistic fluency and logical understanding. 

---
# SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models 

**Authors**: Weiyang Jin, Yuwei Niu, Jiaqi Liao, Chengqi Duan, Aoxue Li, Shenghua Gao, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12784)  

**Abstract**: Recently, remarkable progress has been made in Unified Multimodal Models (UMMs), which integrate vision-language generation and understanding capabilities within a single framework. However, a significant gap exists where a model's strong visual understanding often fails to transfer to its visual generation. A model might correctly understand an image based on user instructions, yet be unable to generate a faithful image from text prompts. This phenomenon directly raises a compelling question: Can a model achieve self-improvement by using its understanding module to reward its generation module? To bridge this gap and achieve self-improvement, we introduce SRUM, a self-rewarding post-training framework that can be directly applied to existing UMMs of various designs. SRUM creates a feedback loop where the model's own understanding module acts as an internal ``evaluator'', providing corrective signals to improve its generation module, without requiring additional human-labeled data. To ensure this feedback is comprehensive, we designed a global-local dual reward system. To tackle the inherent structural complexity of images, this system offers multi-scale guidance: a \textbf{global reward} ensures the correctness of the overall visual semantics and layout, while a \textbf{local reward} refines fine-grained, object-level fidelity. SRUM leads to powerful capabilities and shows strong generalization, boosting performance on T2I-CompBench from 82.18 to \textbf{88.37} and on T2I-ReasonBench from 43.82 to \textbf{46.75}. Overall, our work establishes a powerful new paradigm for enabling a UMMs' understanding module to guide and enhance its own generation via self-rewarding. 

---
# Content Anonymization for Privacy in Long-form Audio 

**Authors**: Cristina Aggazzotti, Ashi Garg, Zexin Cai, Nicholas Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2510.12780)  

**Abstract**: Voice anonymization techniques have been found to successfully obscure a speaker's acoustic identity in short, isolated utterances in benchmarks such as the VoicePrivacy Challenge. In practice, however, utterances seldom occur in isolation: long-form audio is commonplace in domains such as interviews, phone calls, and meetings. In these cases, many utterances from the same speaker are available, which pose a significantly greater privacy risk: given multiple utterances from the same speaker, an attacker could exploit an individual's vocabulary, syntax, and turns of phrase to re-identify them, even when their voice is completely disguised. To address this risk, we propose new content anonymization approaches. Our approach performs a contextual rewriting of the transcripts in an ASR-TTS pipeline to eliminate speaker-specific style while preserving meaning. We present results in a long-form telephone conversation setting demonstrating the effectiveness of a content-based attack on voice-anonymized speech. Then we show how the proposed content-based anonymization methods can mitigate this risk while preserving speech utility. Overall, we find that paraphrasing is an effective defense against content-based attacks and recommend that stakeholders adopt this step to ensure anonymity in long-form audio. 

---
# Who is a Better Matchmaker? Human vs. Algorithmic Judge Assignment in a High-Stakes Startup Competition 

**Authors**: Sarina Xi, Orelia Pi, Miaomiao Zhang, Becca Xiong, Jacqueline Ng Lane, Nihar B. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2510.12692)  

**Abstract**: There is growing interest in applying artificial intelligence (AI) to automate and support complex decision-making tasks. However, it remains unclear how algorithms compare to human judgment in contexts requiring semantic understanding and domain expertise. We examine this in the context of the judge assignment problem, matching submissions to suitably qualified judges. Specifically, we tackled this problem at the Harvard President's Innovation Challenge, the university's premier venture competition awarding over \$500,000 to student and alumni startups. This represents a real-world environment where high-quality judge assignment is essential. We developed an AI-based judge-assignment algorithm, Hybrid Lexical-Semantic Similarity Ensemble (HLSE), and deployed it at the competition. We then evaluated its performance against human expert assignments using blinded match-quality scores from judges on $309$ judge-venture pairs. Using a Mann-Whitney U statistic based test, we found no statistically significant difference in assignment quality between the two approaches ($AUC=0.48, p=0.40$); on average, algorithmic matches are rated $3.90$ and manual matches $3.94$ on a 5-point scale, where 5 indicates an excellent match. Furthermore, manual assignments that previously required a full week could be automated in several hours by the algorithm during deployment. These results demonstrate that HLSE achieves human-expert-level matching quality while offering greater scalability and efficiency, underscoring the potential of AI-driven solutions to support and enhance human decision-making for judge assignment in high-stakes settings. 

---
# Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think? 

**Authors**: Shouren Wang, Wang Yang, Xianxuan Long, Qifan Wang, Vipin Chaudhary, Xiaotian Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.12680)  

**Abstract**: Hybrid thinking enables LLMs to switch between reasoning and direct answering, offering a balance between efficiency and reasoning capability. Yet our experiments reveal that current hybrid thinking LLMs only achieve partial mode separation: reasoning behaviors often leak into the no-think mode. To understand and mitigate this, we analyze the factors influencing controllability and identify four that matter most: (1) larger data scale, (2) using think and no-think answers from different questions rather than the same question, (3) a moderate increase in no-think data number, and (4) a two-phase strategy that first trains reasoning ability and then applies hybrid think training. Building on these findings, we propose a practical recipe that, compared to standard training, can maintain accuracy in both modes while significantly reducing no-think output length (from $1085$ to $585$ on MATH500) and occurrences of reasoning-supportive tokens such as ``\texttt{wait}'' (from $5917$ to $522$ on MATH500). Our findings highlight the limitations of current hybrid thinking and offer directions for strengthening its controllability. 

---
# The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation 

**Authors**: Minghao Tang, Shiyu Ni, Jingtong Wu, Zengxin Han, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12668)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving external documents. As an emerging form of RAG, parametric retrieval-augmented generation (PRAG) encodes documents as model parameters (i.e., LoRA modules) and injects these representations into the model during inference, enabling interaction between the LLM and documents at parametric level. Compared with directly placing documents in the input context, PRAG is more efficient and has the potential to offer deeper model-document interaction. Despite its growing attention, the mechanism underlying parametric injection remains poorly understood. In this work, we present a systematic study of PRAG to clarify the role of parametric injection, showing that parameterized documents capture only partial semantic information of documents, and relying on them alone yields inferior performance compared to interaction at text level. However, these parametric representations encode high-level document information that can enhance the model's understanding of documents within the input context. When combined parameterized documents with textual documents, the model can leverage relevant information more effectively and become more robust to noisy inputs, achieving better performance than either source alone. We recommend jointly using parameterized and textual documents and advocate for increasing the information content of parametric representations to advance PRAG. 

---
# Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space 

**Authors**: Chao Chen, Zhixin Ma, Yongqi Li, Yupeng Hu, Yinwei Wei, Wenjie Li, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.12603)  

**Abstract**: Multimodal reasoning aims to enhance the capabilities of MLLMs by incorporating intermediate reasoning steps before reaching the final answer. It has evolved from text-only reasoning to the integration of visual information, enabling the thought process to be conveyed through both images and text. Despite its effectiveness, current multimodal reasoning methods depend on explicit reasoning steps that require labor-intensive vision-text annotations and inherently introduce significant inference latency. To address these issues, we introduce multimodal latent reasoning with the advantages of multimodal representation, reduced annotation, and inference efficiency. To facilicate it, we propose Interleaved Vision-Text Latent Reasoning (IVT-LR), which injects both visual and textual information in the reasoning process within the latent space. Specifically, IVT-LR represents each reasoning step by combining two implicit parts: latent text (the hidden states from the previous step) and latent vision (a set of selected image embeddings). We further introduce a progressive multi-stage training strategy to enable MLLMs to perform the above multimodal latent reasoning steps. Experiments on M3CoT and ScienceQA demonstrate that our IVT-LR method achieves an average performance increase of 5.45% in accuracy, while simultaneously achieving a speed increase of over 5 times compared to existing approaches. Code available at this https URL. 

---
# Simple Projection Variants Improve ColBERT Performance 

**Authors**: Benjamin Clavié, Sean Lee, Rikiya Takehi, Aamir Shakir, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2510.12327)  

**Abstract**: Multi-vector dense retrieval methods like ColBERT systematically use a single-layer linear projection to reduce the dimensionality of individual vectors. In this study, we explore the implications of the MaxSim operator on the gradient flows of the training of multi-vector models and show that such a simple linear projection has inherent, if non-critical, limitations in this setting. We then discuss the theoretical improvements that could result from replacing this single-layer projection with well-studied alternative feedforward linear networks (FFN), such as deeper, non-linear FFN blocks, GLU blocks, and skip-connections, could alleviate these limitations. Through the design and systematic evaluation of alternate projection blocks, we show that better-designed final projections positively impact the downstream performance of ColBERT models. We highlight that many projection variants outperform the original linear projections, with the best-performing variants increasing average performance on a range of retrieval benchmarks across domains by over 2 NDCG@10 points. We then conduct further exploration on the individual parameters of these projections block in order to understand what drives this empirical performance, highlighting the particular importance of upscaled intermediate projections and residual connections. As part of these ablation studies, we show that numerous suboptimal projection variants still outperform the traditional single-layer projection across multiple benchmarks, confirming our hypothesis. Finally, we observe that this effect is consistent across random seeds, further confirming that replacing the linear layer of ColBERT models is a robust, drop-in upgrade. 

---
# Vision Language Models Map Logos to Text via Semantic Entanglement in the Visual Projector 

**Authors**: Sifan Li, Hongkai Chen, Yujun Cai, Qingwen Ye, Liyang Chen, Junsong Yuan, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12287)  

**Abstract**: Vision Language Models (VLMs) have achieved impressive progress in multimodal reasoning; yet, they remain vulnerable to hallucinations, where outputs are not grounded in visual evidence. In this paper, we investigate a previously overlooked setting: logo hallucination, where models generate brand names or textual content despite logos containing no visible words. Using curated splits of pure symbols, hybrids, and text-bearing logos, as well as the challenging Hard-60 subset, we systematically measure hallucination across leading VLMs. We further probe robustness through nine structured perturbations and show that hallucinations persist even under strong distortions, with occlusion exposing the sharpest weaknesses. Embedding-level analysis with open-weight LLaVA demonstrates that hallucination is tied to a small subset of projector dimensions, and targeted ablation substantially reduces errors while preserving OCR accuracy. Together, these findings reveal that VLMs often rely on symbolic priors rather than genuine glyph perception, particularly for iconic circular logos, and that projector subspaces play a decisive role in this failure mode. Our work contributes both a novel diagnostic lens and actionable mitigation insights, highlighting projector disentanglement and OCR-guided decoding as promising directions for building more trustworthy multimodal systems. 

---
# DiSTAR: Diffusion over a Scalable Token Autoregressive Representation for Speech Generation 

**Authors**: Yakun Song, Xiaobin Zhuang, Jiawei Chen, Zhikang Niu, Guanrou Yang, Chenpeng Du, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12210)  

**Abstract**: Recent attempts to interleave autoregressive (AR) sketchers with diffusion-based refiners over continuous speech representations have shown promise, but they remain brittle under distribution shift and offer limited levers for controllability. We introduce DISTAR, a zero-shot text-to-speech framework that operates entirely in a discrete residual vector quantization (RVQ) code space and tightly couples an AR language model with a masked diffusion model, without forced alignment or a duration predictor. Concretely, DISTAR drafts block-level RVQ tokens with an AR language model and then performs parallel masked-diffusion infilling conditioned on the draft to complete the next block, yielding long-form synthesis with blockwise parallelism while mitigating classic AR exposure bias. The discrete code space affords explicit control at inference: DISTAR produces high-quality audio under both greedy and sample-based decoding using classifier-free guidance, supports trade-offs between robustness and diversity, and enables variable bit-rate and controllable computation via RVQ layer pruning at test time. Extensive experiments and ablations demonstrate that DISTAR surpasses state-of-the-art zero-shot TTS systems in robustness, naturalness, and speaker/style consistency, while maintaining rich output diversity. Audio samples are provided on this https URL. 

---
# HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities 

**Authors**: Xiaoxue Ren, Penghao Jiang, Kaixin Li, Zhiyong Huang, Xiaoning Du, Jiaojiao Jiang, Zhenchang Xing, Jiamou Sun, Terry Yue Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2510.12200)  

**Abstract**: Web applications are prime targets for cyberattacks as gateways to critical services and sensitive data. Traditional penetration testing is costly and expertise-intensive, making it difficult to scale with the growing web ecosystem. While language model agents show promise in cybersecurity, modern web applications demand visual understanding, dynamic content handling, and multi-step interactions that only computer-use agents (CUAs) can perform. Yet, their ability to discover and exploit vulnerabilities through graphical interfaces remains largely unexplored. We present HackWorld, the first framework for systematically evaluating CUAs' capabilities to exploit web application vulnerabilities via visual interaction. Unlike sanitized benchmarks, HackWorld includes 36 real-world applications across 11 frameworks and 7 languages, featuring realistic flaws such as injection vulnerabilities, authentication bypasses, and unsafe input handling. Using a Capture-the-Flag (CTF) setup, it tests CUAs' capacity to identify and exploit these weaknesses while navigating complex web interfaces. Evaluation of state-of-the-art CUAs reveals concerning trends: exploitation rates below 12% and low cybersecurity awareness. CUAs often fail at multi-step attack planning and misuse security tools. These results expose the current limitations of CUAs in web security contexts and highlight opportunities for developing more security-aware agents capable of effective vulnerability detection and exploitation. 

---
# Evolution of meta's llama models and parameter-efficient fine-tuning of large language models: a survey 

**Authors**: Abdulhady Abas Abdullah, Arkaitz Zubiaga, Seyedali Mirjalili, Amir H. Gandomi, Fatemeh Daneshfar, Mohammadsadra Amini, Alan Salam Mohammed, Hadi Veisi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12178)  

**Abstract**: This review surveys the rapid evolution of Meta AI's LLaMA (Large Language Model Meta AI) series - from LLaMA 1 through LLaMA 4 and the specialized parameter-efficient fine-tuning (PEFT) methods developed for these models. We first describe the LLaMA family of foundation models (7B-65B to 288B parameters), their architectures (including native multimodal and Mixtureof-Experts variants), and key performance characteristics. We then describe and discuss the concept of PEFT, which adapts large pre-trained models by updating only a small subset of parameters, and review five PEFT methods that have been applied to LLaMA: LoRA (Low-Rank Adaptation), LLaMA-Adapter V1 and V2, LLaMA-Excitor, and QLoRA (Quantized LoRA). We discuss each method's mechanism, parameter savings, and example application to LLaMA (e.g., instruction tuning, multimodal tasks). We provide structured discussion and analysis of model and adapter architectures, parameter counts, and benchmark results (including examples where fine-tuned LLaMA models outperform larger baselines). Finally, we examine real-world use cases where LLaMA-based models and PEFT have been successfully applied (e.g., legal and medical domains), and we discuss ongoing challenges and future research directions (such as scaling to even larger contexts and improving robustness). This survey paper provides a one-stop resource for ML researchers and practitioners interested in LLaMA models and efficient fine-tuning strategies. 

---
# Precise Attribute Intensity Control in Large Language Models via Targeted Representation Editing 

**Authors**: Rongzhi Zhang, Liqin Ye, Yuzhao Heng, Xiang Chen, Tong Yu, Lingkai Kong, Sudheer Chava, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12121)  

**Abstract**: Precise attribute intensity control--generating Large Language Model (LLM) outputs with specific, user-defined attribute intensities--is crucial for AI systems adaptable to diverse user expectations. Current LLM alignment methods, however, typically provide only directional or open-ended guidance, failing to reliably achieve exact attribute intensities. We address this limitation with three key designs: (1) reformulating precise attribute intensity control as a target-reaching problem, rather than simple maximization; (2) training a lightweight value function via temporal-difference learning to predict final attribute intensity scores from partial generations, thereby steering LLM outputs; and (3) employing gradient-based interventions on hidden representations to navigate the model precisely towards specific attribute intensity targets. Our method enables fine-grained, continuous control over attribute intensities, moving beyond simple directional alignment. Experiments on LLaMA-3.2-3b and Phi-4-mini confirm our method's ability to steer text generation to user-specified attribute intensities with high accuracy. Finally, we demonstrate efficiency enhancements across three downstream tasks: preference data synthesis, Pareto frontier approximation and optimization, and distillation of aligned behaviors for intervention-free inference. Our code is available on this https URL 

---
# One Life to Learn: Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration 

**Authors**: Zaid Khan, Archiki Prasad, Elias Stengel-Eskin, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2510.12088)  

**Abstract**: Symbolic world modeling requires inferring and representing an environment's transitional dynamics as an executable program. Prior work has focused on largely deterministic environments with abundant interaction data, simple mechanics, and human guidance. We address a more realistic and challenging setting, learning in a complex, stochastic environment where the agent has only "one life" to explore a hostile environment without human guidance. We introduce OneLife, a framework that models world dynamics through conditionally-activated programmatic laws within a probabilistic programming framework. Each law operates through a precondition-effect structure, activating in relevant world states. This creates a dynamic computation graph that routes inference and optimization only through relevant laws, avoiding scaling challenges when all laws contribute to predictions about a complex, hierarchical state, and enabling the learning of stochastic dynamics even with sparse rule activation. To evaluate our approach under these demanding constraints, we introduce a new evaluation protocol that measures (a) state ranking, the ability to distinguish plausible future states from implausible ones, and (b) state fidelity, the ability to generate future states that closely resemble reality. We develop and evaluate our framework on Crafter-OO, our reimplementation of the Crafter environment that exposes a structured, object-oriented symbolic state and a pure transition function that operates on that state alone. OneLife can successfully learn key environment dynamics from minimal, unguided interaction, outperforming a strong baseline on 16 out of 23 scenarios tested. We also test OneLife's planning ability, with simulated rollouts successfully identifying superior strategies. Our work establishes a foundation for autonomously constructing programmatic world models of unknown, complex environments. 

---
# ThinkPilot: Steering Reasoning Models via Automated Think-prefixes Optimization 

**Authors**: Sunzhu Li, Zhiyu Lin, Shuling Yang, Jiale Zhao, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.12063)  

**Abstract**: Large Reasoning Models (LRMs) are powerful, but they still suffer from inefficient and off-target reasoning. Currently, training-free methods are limited to either rigid heuristics or descriptive, non-actionable analyses. In this paper, we introduce ThinkPilot, a training-free framework that automatically optimizes LRMs reasoning. It uses an evolutionary process to generate think-prefixes, which are instructions that evolve driven by a taxonomy of reasoning behaviors to guide models toward superior performance. Extensive experiments demonstrate ThinkPilot's broad effectiveness: it significantly improves the accuracy-length trade-off for efficient reasoning, drastically improves safety (for example, cutting the StrongREJECT score of DeepSeek-R1-Distill-Qwen-32B from 27.0% to 0.7), and enhances instruction following. It also synergizes with existing training-based methods. Our analysis reveals that think-prefixes can reliably control LRMs' reasoning behaviors, and that different tasks have strong preferences for specific behavioral distributions. By automatically identifying and eliciting these behaviors, ThinkPilot provides a generalizable framework for aligning LRMs reasoning with task demands. Data and code are available at this https URL 

---
# UALM: Unified Audio Language Model for Understanding, Generation and Reasoning 

**Authors**: Jinchuan Tian, Sang-gil Lee, Zhifeng Kong, Sreyan Ghosh, Arushi Goel, Chao-Han Huck Yang, Wenliang Dai, Zihan Liu, Hanrong Ye, Shinji Watanabe, Mohammad Shoeybi, Bryan Catanzaro, Rafael Valle, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2510.12000)  

**Abstract**: Recent advances in the audio language modeling (ALM) domain tackle audio understanding and text-to-audio generation as separate tasks. Very few studies attempt to unify these tasks -- an essential step toward advanced multimodal reasoning. This paper introduces U}nified Audio Language Model (UALM), which aims to unify audio understanding, text-to-audio generation, and multimodal reasoning in a single model. To achieve this goal, we first present UALM-Gen, a text-to-audio language model that directly predicts audio tokens and is comparable to state-of-the-art diffusion-based models. We then demonstrate, using proper data blending, training recipes, and inference techniques, that our single UALM model matches the quality of state-of-the-art specialized models in audio understanding, text-to-audio generation, and text reasoning. Furthermore, we present UALM-Reason, a multimodal reasoning model that utilizes both text and audio in the intermediate thinking steps to facilitate complex generation tasks. To our knowledge, this is the first demonstration in audio research of cross-modal generative reasoning, with its effectiveness confirmed by subjective evaluations. 

---
# Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation 

**Authors**: Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, Nitya Nadgir, Zachary S Siegel, Boyi Wei, Tianci Xue, Ziru Chen, Felix Chen, Saiteja Utpala, Franck Ndzomga, Dheeraj Oruganty, Sophie Luskin, Kangheng Liu, Botao Yu, Amit Arora, Dongyoon Hahm, Harsh Trivedi, Huan Sun, Juyong Lee, Tengjun Jin, Yifan Mai, Yifei Zhou, Yuxuan Zhu, Rishi Bommasani, Daniel Kang, Dawn Song, Peter Henderson, Yu Su, Percy Liang, Arvind Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2510.11977)  

**Abstract**: AI agents have been developed for complex real-world tasks from coding to customer service. But AI agent evaluations suffer from many challenges that undermine our understanding of how well agents really work. We introduce the Holistic Agent Leaderboard (HAL) to address these challenges. We make three main contributions. First, we provide a standardized evaluation harness that orchestrates parallel evaluations across hundreds of VMs, reducing evaluation time from weeks to hours while eliminating common implementation bugs. Second, we conduct three-dimensional analysis spanning models, scaffolds, and benchmarks. We validate the harness by conducting 21,730 agent rollouts across 9 models and 9 benchmarks in coding, web navigation, science, and customer service with a total cost of about $40,000. Our analysis reveals surprising insights, such as higher reasoning effort reducing accuracy in the majority of runs. Third, we use LLM-aided log inspection to uncover previously unreported behaviors, such as searching for the benchmark on HuggingFace instead of solving a task, or misusing credit cards in flight booking tasks. We share all agent logs, comprising 2.5B tokens of language model calls, to incentivize further research into agent behavior. By standardizing how the field evaluates agents and addressing common pitfalls in agent evaluation, we hope to shift the focus from agents that ace benchmarks to agents that work reliably in the real world. 

---
# Deep Research Brings Deeper Harm 

**Authors**: Shuo Chen, Zonggen Li, Zhen Han, Bailan He, Tong Liu, Haokun Chen, Georg Groh, Philip Torr, Volker Tresp, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11851)  

**Abstract**: Deep Research (DR) agents built on Large Language Models (LLMs) can perform complex, multi-step research by decomposing tasks, retrieving online information, and synthesizing detailed reports. However, the misuse of LLMs with such powerful capabilities can lead to even greater risks. This is especially concerning in high-stakes and knowledge-intensive domains such as biosecurity, where DR can generate a professional report containing detailed forbidden knowledge. Unfortunately, we have found such risks in practice: simply submitting a harmful query, which a standalone LLM directly rejects, can elicit a detailed and dangerous report from DR agents. This highlights the elevated risks and underscores the need for a deeper safety analysis. Yet, jailbreak methods designed for LLMs fall short in exposing such unique risks, as they do not target the research ability of DR agents. To address this gap, we propose two novel jailbreak strategies: Plan Injection, which injects malicious sub-goals into the agent's plan; and Intent Hijack, which reframes harmful queries as academic research questions. We conducted extensive experiments across different LLMs and various safety benchmarks, including general and biosecurity forbidden prompts. These experiments reveal 3 key findings: (1) Alignment of the LLMs often fail in DR agents, where harmful prompts framed in academic terms can hijack agent intent; (2) Multi-step planning and execution weaken the alignment, revealing systemic vulnerabilities that prompt-level safeguards cannot address; (3) DR agents not only bypass refusals but also produce more coherent, professional, and dangerous content, compared with standalone LLMs. These results demonstrate a fundamental misalignment in DR agents and call for better alignment techniques tailored to DR agents. Code and datasets are available at this https URL. 

---
# Balancing Synthetic Data and Replay for Enhancing Task-Specific Capabilities 

**Authors**: Urs Spiegelhalter, Jörg K.H. Franke, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2510.11842)  

**Abstract**: Adapting language models to new tasks through continued pretraining faces a fundamental trade-off: models must learn new capabilities while avoiding catastrophic forgetting of existing knowledge. While prior work has studied synthetic data generation techniques, the optimal replay ratios for balancing task performance and knowledge retention under computational constraints remain poorly understood. We present a comprehensive empirical study investigating the interplay between replay ratio configuration and computational budget when adapting language models to new tasks. Using the bAbI reasoning tasks as our target objective, we apply synthetic data generation and systematically evaluate different total token budgets and replay ratio configurations. We analyze their effects on both task mastery and general knowledge retention. Our experiments reveal an optimal configuration that balances task-specific performance with general knowledge retention. Based on our findings, we provide empirically-grounded guidelines for selecting replay ratios based on computational budget, enabling practitioners to achieve strong task adaptation with significantly reduced training costs. 

---
# Data or Language Supervision: What Makes CLIP Better than DINO? 

**Authors**: Yiming Liu, Yuhui Zhang, Dhruba Ghosh, Ludwig Schmidt, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2510.11835)  

**Abstract**: CLIP outperforms self-supervised models like DINO as vision encoders for vision-language models (VLMs), but it remains unclear whether this advantage stems from CLIP's language supervision or its much larger training data. To disentangle these factors, we pre-train CLIP and DINO under controlled settings -- using the same architecture, dataset, and training configuration -- achieving similar ImageNet accuracy. Embedding analysis shows that CLIP captures high-level semantics (e.g., object categories, text), while DINO is more responsive to low-level features like colors and styles. When integrated into VLMs and evaluated on 20 VQA benchmarks, CLIP excels at text-intensive tasks, while DINO slightly outperforms on vision-centric ones. Variants of language supervision (e.g., sigmoid loss, pre-trained language encoders) yield limited gains. Our findings provide scientific insights into vision encoder design and its impact on VLM performance. 

---
# Don't Walk the Line: Boundary Guidance for Filtered Generation 

**Authors**: Sarah Ball, Andreas Haupt  

**Link**: [PDF](https://arxiv.org/pdf/2510.11834)  

**Abstract**: Generative models are increasingly paired with safety classifiers that filter harmful or undesirable outputs. A common strategy is to fine-tune the generator to reduce the probability of being filtered, but this can be suboptimal: it often pushes the model toward producing samples near the classifier's decision boundary, increasing both false positives and false negatives. We propose Boundary Guidance, a reinforcement learning fine-tuning method that explicitly steers generation away from the classifier's margin. On a benchmark of jailbreak and ambiguous prompts, Boundary Guidance improves both the safety and the utility of outputs, as judged by LLM-as-a-Judge evaluations. Comprehensive ablations across model scales and reward designs demonstrate the robustness of our approach. 

---
# Task-Aware Reduction for Scalable LLM-Database Systems 

**Authors**: Marcus Emmanuel Barnes, Taher A. Ghaleb, Safwat Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2510.11813)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to data-intensive workflows, from database querying to developer observability. Yet the effectiveness of these systems is constrained by the volume, verbosity, and noise of real-world text-rich data such as logs, telemetry, and monitoring streams. Feeding such data directly into LLMs is costly, environmentally unsustainable, and often misaligned with task objectives. Parallel efforts in LLM efficiency have focused on model- or architecture-level optimizations, but the challenge of reducing upstream input verbosity remains underexplored. In this paper, we argue for treating the token budget of an LLM as an attention budget and elevating task-aware text reduction as a first-class design principle for language -- data systems. We position input-side reduction not as compression, but as attention allocation: prioritizing information most relevant to downstream tasks. We outline open research challenges for building benchmarks, designing adaptive reduction pipelines, and integrating token-budget--aware preprocessing into database and retrieval systems. Our vision is to channel scarce attention resources toward meaningful signals in noisy, data-intensive workflows, enabling scalable, accurate, and sustainable LLM--data integration. 

---
# Evolution of wartime discourse on Telegram: A comparative study of Ukrainian and Russian policymakers' communication before and after Russia's full-scale invasion of Ukraine 

**Authors**: Mykola Makhortykh, Aytalina Kulichkina, Kateryna Maikovska  

**Link**: [PDF](https://arxiv.org/pdf/2510.11746)  

**Abstract**: This study examines elite-driven political communication on Telegram during the ongoing Russo-Ukrainian war, the first large-scale European war in the social media era. Using a unique dataset of Telegram public posts from Ukrainian and Russian policymakers (2019-2024), we analyze changes in communication volume, thematic content, and actor engagement following Russia's 2022 full-scale invasion. Our findings show a sharp increase in Telegram activity after the invasion, particularly among ruling-party policymakers. Ukrainian policymakers initially focused on war-related topics, but this emphasis declined over time In contrast, Russian policymakers largely avoided war-related discussions, instead emphasizing unrelated topics, such as Western crises, to distract public attention. We also identify differences in communication strategies between large and small parties, as well as individual policymakers. Our findings shed light on how policymakers adapt to wartime communication challenges and offer critical insights into the dynamics of online political discourse during times of war. 

---
# Celebrity Profiling on Short Urdu Text using Twitter Followers' Feed 

**Authors**: Muhammad Hamza, Rizwan Jafar  

**Link**: [PDF](https://arxiv.org/pdf/2510.11739)  

**Abstract**: Social media has become an essential part of the digital age, serving as a platform for communication, interaction, and information sharing. Celebrities are among the most active users and often reveal aspects of their personal and professional lives through online posts. Platforms such as Twitter provide an opportunity to analyze language and behavior for understanding demographic and social patterns. Since followers frequently share linguistic traits and interests with the celebrities they follow, textual data from followers can be used to predict celebrity demographics. However, most existing research in this field has focused on English and other high-resource languages, leaving Urdu largely unexplored.
This study applies modern machine learning and deep learning techniques to the problem of celebrity profiling in Urdu. A dataset of short Urdu tweets from followers of subcontinent celebrities was collected and preprocessed. Multiple algorithms were trained and compared, including Logistic Regression, Support Vector Machines, Random Forests, Convolutional Neural Networks, and Long Short-Term Memory networks. The models were evaluated using accuracy, precision, recall, F1-score, and cumulative rank (cRank). The best performance was achieved for gender prediction with a cRank of 0.65 and an accuracy of 0.65, followed by moderate results for age, profession, and fame prediction. These results demonstrate that follower-based linguistic features can be effectively leveraged using machine learning and neural approaches for demographic prediction in Urdu, a low-resource language. 

---
# Scaling Law in LLM Simulated Personality: More Detailed and Realistic Persona Profile Is All You Need 

**Authors**: Yuqi Bai, Tianyu Huang, Kun Sun, Yuting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11734)  

**Abstract**: This research focuses on using large language models (LLMs) to simulate social experiments, exploring their ability to emulate human personality in virtual persona role-playing. The research develops an end-to-end evaluation framework, including individual-level analysis of stability and identifiability, as well as population-level analysis called progressive personality curves to examine the veracity and consistency of LLMs in simulating human personality. Methodologically, this research proposes important modifications to traditional psychometric approaches (CFA and construct validity) which are unable to capture improvement trends in LLMs at their current low-level simulation, potentially leading to remature rejection or methodological misalignment. The main contributions of this research are: proposing a systematic framework for LLM virtual personality evaluation; empirically demonstrating the critical role of persona detail in personality simulation quality; and identifying marginal utility effects of persona profiles, especially a Scaling Law in LLM personality simulation, offering operational evaluation metrics and a theoretical foundation for applying large language models in social science experiments. 

---
# TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation 

**Authors**: Yincen Qu, Huan Xiao, Feng Li, Gregory Li, Hui Zhou, Xiangying Dai  

**Link**: [PDF](https://arxiv.org/pdf/2510.09011)  

**Abstract**: Travel planning is a valuable yet complex task that poses significant challenges even for advanced large language models (LLMs). While recent benchmarks have advanced in evaluating LLMs' planning capabilities, they often fall short in evaluating feasibility, reliability, and engagement of travel plans. We introduce a comprehensive benchmark for travel planning that unifies fine-grained criteria into a single reward, enabling direct comparison of plan quality and seamless integration with reinforcement learning (RL). Our evaluator achieves moderate agreement with travel-expert annotations (60.75%) and outperforms multiple LLM-as-judge baselines. We further release a large-scale dataset of 4,870 queries including 219 real-world, free-form requests for generalization to authentic user intent. Using this benchmark, we conduct extensive experiments across diverse methods and LLMs, including test-time computation, neuro-symbolic approaches, supervised fine-tuning, and RL via GRPO. Across base models, RL generally improves itinerary feasibility over prompt-only and supervised baselines, yielding higher unified reward scores. 

---
