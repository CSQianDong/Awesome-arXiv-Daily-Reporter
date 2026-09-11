# Distance generalization in transformers: why bother with positional encoding? 

**Authors**: Daniel Henrik Nevermann, Claudius Gros  

**Link**: [PDF](https://arxiv.org/pdf/2609.11913)  

**Abstract**: Out-of-distribution length generalization, namely to extrapolate a task from short to longer context, has been studied intensively for transformers. Here we focus on distance generalization, which probes performance when inter-token distances are changed between training and inference, while keeping a fixed context length. We construct two synthetic delay copy tasks, both involving finite distances between source and recall, where tokens are copied either fully or selectively, and test models on delays unseen during training. We address three questions: (A) Do positional encoding schemes such as RoPE and ALiBi improve distance resolution relative to no positional encoding (NoPE)? (B) How does data diversity, the number of inter-token distances seen in training, affect performance? (C) When is distance transfer learning positive or negative? We present a thorough investigation, finding that it is paramount to improve our understanding of the underlying mechanisms. 

---
# Nuha-Speech: Building General-Purpose Arabic Speech-LLMs 

**Authors**: Yingzhi Wang, Reem Alhazzani, Muhammad Alqurishi  

**Link**: [PDF](https://arxiv.org/pdf/2609.11892)  

**Abstract**: As Speech Large Language Models (speech-LLMs) become increasingly multilingual, Arabic remains significantly underrepresented, highlighting the need for dedicated infrastructure to train and evaluate Arabic speech-LLMs.
To address this gap, we introduce Nuha-Speech, a comprehensive initiative to develop general-purpose Arabic speech-LLMs spanning dataset construction, model training, and systematic evaluation. Specifically, we constructed a large-scale Arabic Speech Question-Answering (SQA) corpus comprising over 1.5 million training samples to allow instruction tuning over a broad range of core speech tasks. Then, the corpus was used for supervised fine-tuning based on Qwen-Omni model variants at different scales. Finally, we designed an evaluation framework featuring diverse tasks and tailored metrics. Through this work, we aim to establish foundational infrastructures for Arabic Speech-LLMs under constraints imposed by limited Arabic speech resources. 

---
# Domain-Specific Hallucination Detection in Large Language Models 

**Authors**: Varun Teja Chundru, Debasmita Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2609.11878)  

**Abstract**: Large language models generate fluent text that can contain unfaithful claims -- a phenomenon known as hallucination. We present a multi-signal detection pipeline combining fine-tuned DeBERTa-v3 classification, Monte Carlo (MC) Dropout uncertainty quantification, and temperature-scaled calibration for response-level hallucination detection. Evaluated on the HaluEval benchmark, our pipeline achieves F1=0.915 and AUROC=0.977 on general-domain tasks, with per-task F1 scores of 0.97 (QA), 0.96 (Summarization), and 0.82 (Dialogue). MC Dropout inference further improves accuracy to 93.2%. A context ablation study confirms the model performs genuine entailment reasoning rather than exploiting surface patterns, with summarization F1 dropping 24% when knowledge context is removed. Learning curve analysis reveals that 25% of training data captures 77% of full-data performance. Beyond detection, we apply Direct Preference Optimization (DPO) to a Qwen2.5-0.5B generator, reducing its hallucination rate from 85.5% to 37.7% (55.9% relative reduction) as measured by our detector. Cross-domain evaluation on the SciFact biomedical benchmark shows that general-domain training transfers poorly (F1=0.52), motivating domain-specific fine-tuning. PubMedBERT fine-tuned on SciFact achieves F1=0.63 and AUROC=0.81, demonstrating that domain-matched pre-training is the strongest adaptation strategy. Code and models are available at this https URL 

---
# Augustinian BabyLM: What Ostensive Definition Can and Cannot Teach a Small Language Model 

**Authors**: Lisa Bylinina  

**Link**: [PDF](https://arxiv.org/pdf/2609.11870)  

**Abstract**: A language model normally begins training with random word embeddings: whatever 'banana' means must be learned from training corpora. I implement St. Augustine's picture of word learning, meaning by ostension, for a small masked language model (DeBERTa) trained on 10M words: before training, visually grounded tokens receive embeddings derived from the image regions they label; other tokens start random. Visual initialization leaves a measurable imprint that lasts until the end of training. At the same time, the effect remains invisible under most BabyLM benchmarks, which probe abstract grammatical knowledge: visual initialization does not affect performance there. The only zero-shot exception is object-property knowledge (COMPS, Misra et al. 2023), where seeding helps in every configuration. To follow up on this result, I build a corpus-tailored version of the Visual-Property Swap benchmark (Lin et al., 2026), which tests color, material, size, and shape knowledge, with per-item training frequency and seeded status. Here, vision-seeded models have a persistent, seed- replicated advantage, confined to the seeded words. As a causal test, I show that synthetic grounding of previously unseeded words transfers the advantage to exactly those words. Function words and abstract vocabulary also receive strong visual seeds and retain them throughout training, and the training objective draws on them: held-out mask-prediction loss falls for these words in every seed. However, no benchmark I run registers this. What evaluation would pick this up remains an open question. 

---
# Epistemic orientation predicts legislative effectiveness among members of the US Congress 

**Authors**: Segun Aroyehun, Stephan Lewandowsky, David Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2609.11865)  

**Abstract**: Truth and evidence-based communication provide important foundations for democratic governance, accountability, and collective decision-making. Prior work shows that evidence-oriented language in US congressional floor speeches has declined since the mid-1970s, alongside broader changes in legislative productivity and polarization. This study shifts the analysis from congressional sessions to individual members of Congress to examine whether epistemic orientation varies systematically across legislators and whether it relates to political behavior and legislative effectiveness. Using the Evidence-Minus-Intuition (EMI) score, we measure the relative prevalence of evidence-oriented versus intuition-oriented language in congressional floor speeches and Twitter posts. We link these measures to legislator-level data on ideology, institutional position, communication context, and Legislative Effectiveness Score (LES). The results show that more ideologically extreme members use less evidence-oriented language on the congressional floor. EMI also exhibits cross-platform consistency with members who use more evidence-oriented language in floor speeches also being more evidence-oriented on Twitter, although EMI is lower on Twitter overall. Finally, EMI in congressional speeches is positively associated with individual legislative effectiveness, even after accounting for ideology and extensive political, institutional, demographic, topical, and communication volume controls. These findings suggest that evidence-oriented language is not only an aggregate feature of congressional discourse but also a meaningful attribute of individual-level legislative communication and effectiveness. 

---
# IndicTriMix: Developing Language Identification Datasets and Models for Tri-Language Code-Mixing 

**Authors**: Pruthwik Mishra, Rudra Trivedi, Avi Patel, Ashok Urlana, Shrikant Malviya  

**Link**: [PDF](https://arxiv.org/pdf/2609.11851)  

**Abstract**: Language identification in code-mixed text, largely observed in social media, is highly essential when users frequently switch between multiple languages within a single utterance. Accurately identifying the languages of code-mixed tokens becomes an urgent necessity. Traditional language identification models, designed for monolingual text, are not well suited for token-level language identification in code-mixed settings. We formulate the task as a sequence labeling problem and fine-tune contextual transformer-based models MuRIL and XLM-RoBERTa best suited for Indian languages. We evaluate these systems on three different data configurations (Hindi, Gujarati, and Bengali) to predict language labels for individual tokens. We release a benchmark for language identification in code-mixed tokens with manually annotated test sets. We propose two approaches of code-mixed generation using parallel sentences of three languages. The trained models demonstrate the effectiveness of contextual embeddings for token-level language identification in multilingual social media text. For reproducibility and to facilitate future research, we publicly release our fine-tuned models. 

---
# Target leakage, not model class, explains reported accuracy in survey-based cardiovascular screening: a leakage-tiered audit of glass-box and tabular foundation models 

**Authors**: Raad Bin Tareaf, Murad Al-Rajab, Samia Loucif, Samer Ellaham, Cedric Schmitz  

**Link**: [PDF](https://arxiv.org/pdf/2609.11838)  

**Abstract**: Cardiovascular screening models trained on national health surveys routinely report areas under the receiver operating characteristic curve (AUROC) near 0.89. We asked whether that accuracy reflects learning or target leakage, whether tabular foundation models change the answer, and whether the properties deployment requires survive joint examination. We benchmarked ten classifiers spanning linear, tree-ensemble, neural, glass-box, and tabular foundation classes for prevalent myocardial infarction in 442,067 respondents of the 2022 Behavioral Risk Factor Surveillance System across five feature tiers of decreasing leakage risk. Each was audited for discrimination, calibration, fairness at an explicit screening threshold, conformal coverage, explanation faithfulness, and inference cost, then applied -- models and thresholds frozen -- to 430,755 respondents of 2023. Removing two post-diagnostic features cost every model 0.049-0.051 AUROC, collapsing the field into a 0.0045-wide band. The glass-box explainable boosting machine was non-inferior to every alternative within a pre-specified 0.005 margin while scoring the cohort roughly 104 times faster than the strongest foundation model. One threshold detected 75.4% of women's infarctions against 89.0% of men's; editing the model's shape functions reduced the gap to 0.010. Marginal conformal prediction gave 0.86 coverage to men and 0.82 to adults over 60; Mondrian calibration repaired every stratum. Frozen models transported within 0.002 AUROC. Reported headroom in this literature is a property of the feature set, not the learner. Transparency cost nothing measurable and made fairness repair and uncertainty conditioning directly auditable. Evaluation practice, not model capacity, is the binding constraint. 

---
# Beyond Word Error Rate: A Switch Aware Evaluation of ASR and Audio Language Models on English Yoruba Code-Switched Speech 

**Authors**: Chibuzor Okocha, Christan Earl Grant  

**Link**: [PDF](https://arxiv.org/pdf/2609.11786)  

**Abstract**: Automatic speech recognition (ASR) systems and audio language models (audio LMs) now report low error rates on monolingual benchmarks, but their behavior on code switched speech in low resource, diacritic rich languages remains poorly characterized. We present a switch aware evaluation of eleven modern systems (six ASR models and five audio LMs) on English Yoruba code-switched speech, using a deterministic 2000 utterance evaluation set and a shared scoring pipeline. Beyond word error rate (WER), we report switch localized diagnostics: a switch entry token error rate (SETER), windowed switch point error rates, language specific error rates, and a diacritic insensitive WER. Our central finding is that aggregate WER hides code switching behavior. The best system by WER (an ASR model) is statistically indistinguishable from a leading audio LM on WER, yet the audio LM is significantly better on every switch localized metric. Across faithful systems, Yoruba token recognition collapses (error 0.97 for almost all systems) while English tokens are recognized far better, and errors concentrate sharply at switches into Yoruba. Several generative audio LMs fail as exact transcribers, producing translation, verbosity, and prompt leakage that are strongly prompt dependent. We release manifests, metric implementations, and evaluation scripts to support reproducible, switch aware benchmarking for African code switched speech. 

---
# The widening evaluation gap in medical large language model research 2023 to 2026 

**Authors**: Raad Bin Tareaf, Murad Al-Rajab, Samia Loucif  

**Link**: [PDF](https://arxiv.org/pdf/2609.11770)  

**Abstract**: Large language models are superseded every few quarters; clinical evidence takes years. We asked whether medical research is keeping pace with the systems it evaluates. PubMed returned 11,628 records for January 2023 to June 2026 across fourteen clinical domains, growing 45-fold; 2.5% used a randomised, controlled or prospective design. Evaluation lag, from a study's newest named model release to its own publication, widened from 1.33 to 6.08 quarters. Because discontinued models age mechanically, we benchmarked this against a counterfactual holding model composition fixed: migration to newer systems offset only 56% of the drift (95% CI 50-65). Randomised trials evaluated models a median 4.6 quarters older than other designs (P = 3 x 10^-19), yet among studies naming a model still under development no design differed from any other; 62% of randomised trials evaluated a discontinued family. Rigour and currency are in tension, and that tension reflects model selection rather than research timelines. 

---
# Recognizing Is Not Reversing: A Controlled Inversion Test of Fact-Preserving News Framing 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.11769)  

**Abstract**: Large language models (LLMs) are increasingly used to analyze and rewrite news, yet current framing studies mainly evaluate generation, detection, or whether rewritten text appears more neutral. They do not directly show whether a model can undo a known framing transformation while keeping the facts fixed. We introduce a controlled inversion test over three established textual realizations of framing: evaluative lexis, agency realization, and information salience. Across 60 news articles and three intervention strengths, this yields 540 paired variants with preserved atomic facts and recorded edits. Across Qwen, DeepSeek, and Kimi, factual preservation remains near 0.84, whereas intervention reversal is 0.044--0.068. Even when both framing type and direction are recognized correctly, pooled reversal reaches 0.071. These results reveal a clear separation between factual fidelity, framing recognition, and framing inversion: recognizing how an article is framed does not imply that the framing can be undone. 

---
# Component-Aware Differential Privacy for Federated Multilingual Speech-LLMs 

**Authors**: Jordi Luque, Fernando López, Aleix Sant  

**Link**: [PDF](https://arxiv.org/pdf/2609.11762)  

**Abstract**: Per-layer differential privacy (DP) clipping improves gradient fidelity in federated learning by allocating per-matrix clipping budgets proportional to parameter count. We show that this recipe breaks for speech large language models (speech-LLMs), when the acoustic encoder and the language decoder differ by an order of magnitude in update norm. Single-pool per-layer methods suffer \emph{cross-component budget collapse}, dragging word error rate (WER) far from flat global clipping or collapsing training entirely. When the norm imbalance is milder, adaptive single-pool methods partially recover, confirming that collapse severity scales with the inter-component norm ratio. We empirically diagnose the root cause across six per-layer methods and three speech-LLM architectures. We then propose \emph{$\alpha$-split}, a two-pool allocation that normalises encoder and LLM parameters into independent pools, and show that joint $\ell_2$ sensitivity and the original $(\varepsilon,\delta)$-DP guarantee are unchanged. At architecture-calibrated $\alpha$, our method recovers WER utility compared to flat DP, while granting the encoder $4.47{\times}$ tighter per-component noise protection against speaker voice-based gradient-inversion attacks at only $+2.6\%$ LLM noise overhead. 

---
# RAG-Safety-Bench: Reliable Evaluation of Retrieval-Augmented LLM Safety 

**Authors**: Adithiyan Rajan Indira Saravanan, Kathleen C. Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2609.11758)  

**Abstract**: Allowing large language models (LLMs) to retrieve information from a set of trusted documents can increase reliability and reduce hallucination. However, recent work has demonstrated that retrieval-augmented generation (RAG) can have unintended side effects on the overall safety of the generated responses, when prompted for harmful or dangerous content. A clearer understanding of the mechanisms leading to this result is needed, as increasing numbers of end users turn to RAG to incorporate corporate documents and knowledge bases into LLM-based systems. We introduce RAG-Safety-Bench, a benchmark to measure the safety impact of RAG on LLM models. By removing the confounding effect of retriever quality, and cleanly separating the problem into four conditions -- non-RAG, RAG with an oracle document containing the answer to the harmful request, RAG with documents related to the harmful request but without the specific answer, and RAG with random, safe documents -- the benchmark isolates the impacts of different factors in the observed safety degradation. We report results across five open-source LLMs, showing an inverse relationship between benign and unsafe capability, strong evidence that baseline safety guardrails do not lead to downstream safety guarantees in the RAG case, and model-specific support for previous findings that even benign documents can lead to unsafe generation in retrieval-enabled systems. 

---
# LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation 

**Authors**: Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.11739)  

**Abstract**: Large language model serving costs scale directly with output sequence length, yet standard preference alignment often inflates response verbosity without improving utility. We study whether the parameterization of post-training updates affects generation length: low-rank subspaces alter sequence length without modifying the alignment loss. We present LOCUS, a method that selects a task-aware low-rank adaptation subspace to minimize output-token cost subject to a utility constraint. Within this subspace, post-training retains the native preference objective with a frozen backbone. Across Anthropic HH-RLHF dialogue preferences, we evaluate two $\sim$3B decoder backbones, Pythia-2.8B and Qwen2.5-3B, against protocol-matched full-parameter DPO and DrDPO branches and the released SamPO checkpoint. LOCUS reduces continuation length by up to 39.84\% on Pythia-2.8B and by 14.87--17.58\% on Qwen2.5-3B while updating only 0.24--0.28\% of model parameters, with no material change in the internal preference diagnostic. 

---
# The Eloquence submission for Task 2 of the Interspeech 2026 MLC-SLM challenge 

**Authors**: Jordi Luque, Lorenzo Concina, Marco Matassoni, Alessio Brutti, Filippo Vella  

**Link**: [PDF](https://arxiv.org/pdf/2609.11724)  

**Abstract**: This paper details the Eloquence team's approach to Task 2 of the 2nd MLC-SLM challenge at Interspeech 2026, which involves multilingual Multiple-Choice Question Answering (MCQA) across 21 languages. Three approaches are explored. First, we fine-tune Voxtral-Mini-3B via LoRA with cross-lingual data augmentation, ASR transcript augmentation and timestamp-aware audio cropping, achieving 0.72 macro-accuracy on evaluation Phase 2. Second, we apply multimodal in-context learning (ICL) to the frozen Voxtral-24B model to correct a strong label bias, reaching 0.81, our best result. Third, a training-free retrieval system based on a three-layer voice-anchored memory combining acoustic identity, semantic content, and a knowledge graph achieves 0.68. All three systems substantially outperform the official baseline. 

---
# Negative Self-Distillation: Learning to Reason by Avoiding Flaws 

**Authors**: Rongcan Pei, Zhepei Wei, Shuyao Xu, Xinyu Zhu, Wei-Lin Chen, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2609.11699)  

**Abstract**: On-Policy Self-Distillation (OPSD) has emerged as a popular paradigm for large language model (LLM) self-improvement, allowing models to act as their own teachers by leveraging privileged information such as ground-truth solutions. However, recent findings indicate that OPSD can severely degrade the performance of LLMs on complex reasoning tasks: By forcing the student to imitate an artificially confident reasoning trace conditioned on privileged information, OPSD inadvertently suppresses expressions of uncertainty and penalizes the exploratory, self-corrective behaviors required to solve challenging problems. To address this, we introduce Negative Self-Distillation (NSD), a new framework that optimizes LLMs by diverging from flawed reasoning rather than imitating privileged solutions. Instead of relying on ground-truth answers or external supervision, NSD uses the model itself to generate a question-specific negative condition (eg, acting as a ``careless reasoner'') and pushes the student's distribution away from this self-generated negative teacher. Naively applying unlearning objectives to achieve this divergence is problematic, as flawed reasoning tokens are confounded with basic linguistic tokens; indiscriminately penalizing both risks catastrophically degrading the model's foundational language capabilities. We resolve this by designing a dynamic gating mechanism that automatically identifies and isolates reasoning-critical tokens, ensuring gradient updates target only behavioral flaws while preserving the model's linguistic priors. Empirically, NSD consistently outperforms OPSD and other label-free, self-bootstrapping reinforcement learning (RL) baselines. 

---
# Structured Transforms for Low-Overhead Quantization of Language Models 

**Authors**: Daria Cherniuk, Alexander Rudikov, Boris Kashin, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2609.11687)  

**Abstract**: We revisit Kashin-decomposition-based weight quantization for large language models and propose an improved algorithm with stronger convergence properties and structured, efficient orthogonal transforms. The method retains the core factorization of each weight into two components -- one with bounded infinity norm and the other with bounded infinity norm after an orthogonal transformation -- but replaces the dense random orthogonal matrix with a sign-randomized Discrete Cosine Transform (DCT), reducing the per-iteration cost from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$. The proposed greedy algorithm with alternating updates guarantees the four-peak distribution required for stable 2-bit clustering of each factor and admits closed-form initialization of cluster centers, removing the multi-restart k-means bottleneck of prior work. Composed with OPTQ-style sequential error compensation and QuIP-style incoherence preprocessing, the resulting JAX pipeline is competitive with OPTQ, QuIP, QuIP-RG and a fine-tuning- and vector-quantization-free variant of QuIP# at 4-bit per channel on OPT, Llama-2 and Pythia, with favorable wall-clock scaling. The bounded-$\ell_\infty$ factorization is also notably robust: on stress configurations where QuIP variants diverge to four-digit perplexity (Pythia-6.9B) or abort with NaNs in LDL back-substitution (Mistral-7B), Kashin-DCT remains numerically stable and stays close to FP16 baseline. At inference time, each weight decomposes into two 2-bit factor codes per channel that are structurally suited to native-2-bit hardware. 

---
# A Training-Free, Alignment-Free Approach to Corporate Intelligence: Application to SEC Filings 

**Authors**: Jean-François Delpech  

**Link**: [PDF](https://arxiv.org/pdf/2609.11620)  

**Abstract**: High-dimensional dense text embeddings and large language models face real obstacles in financial-disclosure analysis: context-window limits, hallucination risk, high computational cost, and the arbitrary rotation of vector spaces across independently trained models. We present a training-free, alignment-free framework for corporate intelligence built on deterministic sparse seed vectors. Hashing word strings into a fixed high-dimensional basis places all documents and all temporal epochs in a common coordinate system by construction, removing any need for training or alignment. Accumulating these seed vectors across sentence contexts yields corpus-specific semantic signatures that compose linearly, supporting sub-second document comparison, issuer fingerprinting, tracking of how an issuer's vocabulary shifts between filings, and thematic sentence extraction, all on ordinary CPU hardware. Demonstrating the approach on a multi-year corpus of SEC filings (10-K, 10-Q, 8-K), we show how material corporate events, among them Boeing's 737 MAX crisis, Intel's supply-chain disruptions, and Bunge's acquisition of Viterra, emerge as distinct, interpretable semantic profiles, each traceable to the exact source sentences that produced it, with no domain-specific training and no LLM inference. 

---
# Complex-Text Robustness Evaluation and Failure Diagnosis for Low-Resource Multilingual Text-to-Speech 

**Authors**: Tianlun Zuo, Ziyu Zhang, Tingzhi Mao, Zhonghua Fu, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2609.11545)  

**Abstract**: Low-resource multilingual text-to-speech (TTS) systems have expanded language coverage, but their robustness under complex text inputs remains insufficiently diagnosed. Existing evaluations mainly focus on naturalness, speaker similarity, and content consistency using regular test sentences, while providing limited insight into how multilingual TTS systems fail when handling challenging inputs such as numbers, dates, named entities, long sentences, code-switched expressions, and punctuation-related structures. This paper proposes a complex-text robustness diagnosis framework for low-resource multilingual TTS. We evaluate robustness from three dimensions: content consistency, language consistency, and generation stability. A multilingual robustness testing scheme is designed for Thai, Vietnamese, Swahili, and Indonesian, covering ordinary sentences and multiple types of complex text inputs. We further introduce automatic diagnostic metrics, including character error rate, language identification accuracy, and duration abnormal rate. To support input-level risk analysis before speech generation, we propose a lightweight Text Risk Score (TRS), which estimates synthesis risk from interpretable text features without manual annotation or model training. Experiments on three representative multilingual TTS systems, including OmniVoice, VoxCPM2, and MMS-TTS, show that complex text inputs expose systematic failure patterns that are not fully reflected by ordinary short-sentence evaluation. Different systems exhibit distinct vulnerabilities in number normalization, named entity handling, long-text generation, and code-switched input processing. Furthermore, TRS shows a positive correlation with content errors and duration abnormalities, demonstrating its usefulness as a low-cost pre-synthesis indicator for complex-text risk diagnosis in low-resource multilingual TTS. 

---
# Structural priors for data-efficient language learning 

**Authors**: Yana Veitsman, Jonas Mayer Martins, Jonathan Lautenschlager, Lisa Beinborn  

**Link**: [PDF](https://arxiv.org/pdf/2609.11505)  

**Abstract**: Efficient language learning requires methods to reduce the reliance on large data and computational resources. We investigate structural transfer: First training models on non-language data to induce useful priors for natural language. This approach is a form of weight initialization for multilingual language modeling. We evaluate transfer via next-token-prediction loss, weight shifts in the model, and downstream linguistic benchmarks. Several symbolic data types - notably music, probabilistic grammars, and cellular automata - yield lower language-modeling loss than random initialization. These gains coincide with smaller weight shifts during subsequent language training, suggesting that structural transfer positions models in a more favorable region of the parameter space. However, a lower loss does not translate consistently into better downstream linguistic performance, and transfer from non-language data is less efficient than additional language data. We conclude that non-language data can serve as a partial substitute for language data for the training objective of next-token prediction but does not reliably support broader linguistic generalization. 

---
# ReGround: Grounding Reviewer Comments in Multimodal Evidence 

**Authors**: Serwar Basch, Lizhen Qu, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2609.11460)  

**Abstract**: Reviewer comments naturally relate to specific parts of the reviewed paper, yet grounding these comments to the underlying evidence is difficult due to long multimodal documents. Existing benchmarks do not capture this setting and largely focus on explicit, information-seeking queries. We introduce ReGround, a large-scale dataset for reviewer comment grounding that links 10,267 reviewer comments to 16,274 evidence in the original anonymous submission of 3,656 papers. We build on a simple observation: author rebuttals often include explicit references to content of the submission used to address reviewer comments, providing a high-precision annotation source. We cast grounding as a retrieval task and evaluate a wide range of retrieval methods. Results show that retrieval over the entire paper content performs poorly, evidence-type inference is a major bottleneck, and multimodal evidence provides complementary signals that text alone misses. Our dataset exposes grounding reviewer comments as a difficult and practically important problem for scientific document understanding. 

---
# Cross-Lingual Clinical Annotation Projection as Constrained Text Generation: A Six-Language Study 

**Authors**: Álvaro Rey-Blanes, Francisco J. Moreno-Barea, Francisco J. Veredas  

**Link**: [PDF](https://arxiv.org/pdf/2609.11450)  

**Abstract**: Background: To determine whether cross-lingual clinical annotation projection can be formulated as a text-preserving, document-level generative task that produces verifiable character-level annotations for multilingual clinical corpus construction, and to characterize its robustness and computational trade-offs relative to candidate-based projection pipelines. Methods: We developed a constrained LLM projection workflow that inserts entity tags directly into immutable target-language text, followed by deterministic validation and character-offset reconstruction. We evaluated it alongside supervised candidate-span projection and hybrid ML-LLM refinement for transferring Spanish Disease, Symptom, and Procedure annotations into six languages. Evaluation used MultiClinAI gold standard with strict span matching and character-overlap F1 Results: Direct LLM projection achieved the strongest and most consistent performance. GLM 5.2 obtained a mean Strict F1 of 0.9201 across 18 language-entity combinations, while locally deployable Gemma4:31B achieved 0.9133. The best LLM configuration improved Strict F1 over the previous state of the art in all 18 settings, by 0.0564-0.1512, yielding 55,416 grounded mentions with reconstructed offsets. Conclusions: Direct LLM-based projection enables high-quality multilingual clinical annotation transfer and provides a practical approach for extending clinical NLP resources to languages with fewer annotated datasets and language-specific tools. Combined with local inference and deterministic validation, it can substantially reduce expert time and cost for multilingual clinical corpus construction. 

---
# SWRouter: Similarity-Contractive Window Routing for Multi-Turn Large Language Model Conversations 

**Authors**: Yu Wang, Yuchen Li, Rui Kong, Xinran Chen, Jiamin Chen, Hengyi Cai, Shuaiqiang Wang, Jiashu Zhao, Yulun Zhang, Zhonghao Lyu, Haoyi Xiong, Linghe Kong, Jimmy Xiangji Huang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2609.11414)  

**Abstract**: Large language models exhibit complementary strengths, motivating routing methods that dispatch each query to the most suitable model. Although existing routers are effective in single-turn settings, they do not directly transfer to multi-turn dialogue, where routing performance critically depends on how historical context is segmented, retained, and incorporated into the current prompt. This introduces two fundamental challenges: preventing information loss and information confusion during context construction, and evaluating routing quality without conflating model selection with prompt construction quality. In this paper, we propose SWRouter, a Similarity-Contractive Window Router for multi-turn large language model routing. SWRouter combines a similarity-based context segmentation mechanism for prompt construction with a dual-metric evaluation framework that decouples construction accuracy from router performance. Experiments on multi-turn dialogue benchmarks demonstrate that SWRouter consistently surpasses strong baselines, achieving a 16.26% improvement in evaluation accuracy over the best individual large language model and an additional 8.22% gain over the Conv-ID Context baseline. Our results highlight that multi-turn large language model routing requires a joint design of context construction and evaluation, rather than a direct extension of single-turn routing methods. 

---
# TransClean: A Benchmark for Detecting and Extracting Clean Translations from Large Language Model Outputs 

**Authors**: Shenbin Qian, Yves Scherrer  

**Link**: [PDF](https://arxiv.org/pdf/2609.11399)  

**Abstract**: Large language models (LLMs) are increasingly used for machine translation, yet their outputs often contain additional text beyond the translation itself, such as language labels, explanations or bilingual repetitions, which we term translation noise. Despite its prevalence, this problem lacks dedicated benchmarks and systematic study. We analyze over 790,000 translation outputs from 12 LLMs across 22 language pairs (LPs) and identify 12 recurring noise patterns, which we group into formatting and content noise. Building on the observed patterns, we construct TransClean, a controlled benchmark of 9,900 pairs of noisy and clean translation outputs, comprising 8,800 synthetically generated instances and 1,100 manually curated authentic instances. We evaluate two extraction approaches on the TransClean benchmark: 1) a span-based extraction method leveraging translation quality estimation models for span detection, and 2) an LLM-based extraction method that prompts an LLM to isolate the translation. Our benchmark and analysis provide the first systematic framework to evaluate and improve the cleanliness of LLM translation outputs. 

---
# SEAR: Segment-Evidence-Aware Routing for Weak-to-Strong Multilingual Speech MCQ 

**Authors**: Huy Hoang Le, Long-Bao Nguyen, Minh Tri Dao  

**Link**: [PDF](https://arxiv.org/pdf/2609.11355)  

**Abstract**: This paper describes our system for Task~2 of the second Multilingual Conversational Speech Language Model (MLC-SLM) Challenge. We adapt Qwen3-Omni-30B-A3B-Instruct with a segment-evidence-aware data and post-training pipeline. A language model converts timestamped ASR into coherent event spans, which are expanded by a boundary margin and cropped from the original recording. We then synthesize complementary semantic MCQs with Qwen3.6-27B and acoustic MCQs with Gemini~3.1 Flash-Lite, followed by structural, grounding, answer-consistency, and target-model trainability checks, yielding 359,825 verified MCQs across 21 language and accent variants. A text-only probe partitions the data into weak, text-answerable items used for supervised fine-tuning and strong, audio-dependent items used for reinforcement learning with Group Sequence Policy Optimization (GSPO), stabilized by debiased advantages, sequence-level importance correction, and dynamic filtering. Our system obtains 90.92% accuracy on the final official evaluation set. 

---
# On the Impact of Anonymization on the Performance of Large Language Models 

**Authors**: Tobias Deußer, Max Hahnbück, Lorenz Sparrenberg, Tobias Uelwer, Christian Bauckhage, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2609.11335)  

**Abstract**: As large language models are increasingly deployed in sensitive domains, anonymizing input data to protect personally identifiable information has become a critical practice. However, the impact of this anonymization on model utility is not well understood. This paper presents a systematic empirical study of the trade-off between privacy and performance. We evaluate five prominent language models across eleven diverse benchmarks, comparing their performance on original versus pseudonymized inputs. Our results reveal that while anonymization generally degrades performance, the effect is highly nuanced. We find that more capable models, such as Qwen2.5-72B and GPT-4o mini, suffer the largest performance drops, suggesting a stronger reliance on specific entity information. The impact is also task-dependent: performance on TruthfulQA improves with anonymization, while retrieval-focused tasks like RGB experience a catastrophic decline. Further experiments show that reversible anonymization techniques that preserve entity uniqueness significantly outperform irreversible ones like redaction, and that explicitly prompting models about anonymization offers no discernible benefit. We conclude that anonymization is not a one-size-fits-all solution and must be co-designed with the model and task in mind to balance privacy and utility effectively. Our findings provide a crucial baseline for developing more robust, privacy-aware AI systems. 

---
# E-CONAN (Entailment, CONtradition And Neutral) Benchmarks: Arabic Textual Entailment and Natural Inference Datasets 

**Authors**: Khloud AL Jallad, Nada Ghneim, Ghaida Rebdawi  

**Link**: [PDF](https://arxiv.org/pdf/2609.11334)  

**Abstract**: Natural Language Inference processes pairs of sentences to extract their semantic relations. NLI has been a hot research topic, integrated as a main component in other NLP applications. Despite significant advancements in textual inference across various languages all around the world, Arabic language still suffers from limited resources in this domain. To address this gap, this paper introduces E-CONAN benchmarks that are composed of sentences pairs from various sources: (1) automatically-translated pairs, (2) human-validated machine-translated pairs, (3) hand-crafted pairs from teaching Arabic as foreign language books, and (4) headlines pairs from different news channels containing rumors. E-CONAN contains two benchmark datasets, E-CONAN-2, a 2-way dataset (RTE) and E-CONAN-3, a 3-way dataset (NLI). Additionally, we have used E-CONAN benchmarks to evaluate 9 state-of-the-art multilingual pretrained models using zero-shot classification. Models were evaluated across the ArNLI, XNLI, and E-CONAN datasets. Results show that E-CONAN is a potentially valuable resource for evaluating model generalization and even for fine-tuning pre-trained models. Its diverse composition, derived from a combination of sources, offers a broader and more robust assessment compared to XNLI and ArNLI. In addition, we have evaluated 5 LLMs on E-CONAN-3 dataset. Moreover, we incorporated MARBERT as a representative Arabic-specific baseline and conducted performance evaluation comparison to demonstrate how Arabic-specific models scale against cross-lingual and LLM-based approaches on the E-CONAN benchmarks. Furthermore, we conducted detailed qualitative and quantitative error analysis to analyze frequent error patterns. E-CONAN benchmarks will be publicly available, we hope that it will enrich research community in Arabic textual entailment and natural language inference. 

---
# MultiHuSE: A Multimodal Dataset for Humour Styles and Emotions 

**Authors**: Mary Ogbuka Kenneth, Foaad Khosmood, Abbas Edalat  

**Link**: [PDF](https://arxiv.org/pdf/2609.11322)  

**Abstract**: Computational recognition of verbal humour remains a challenging task, requiring an understanding of language, delivery style, emotions, and cultural context. Most existing approaches focus on binary classification and lack datasets that capture psychological dimensions of humour alongside variations in expression. We introduce MultiHuSE, a multimodal dataset comprising 2,407 high-definition videos of 50 demographically diverse actors performing 1,463 text samples across four psychological humour styles (affiliative, aggressive, self-enhancing, and self-deprecating), as well as neutral content. A subset is additionally annotated for underlying emotions. The dataset uniquely captures multiple actor interpretations of the same texts, enabling systematic analysis of expressive diversity. Baseline experiments show that multimodal fusion outperforms unimodal approaches (80.1% vs. 77.4% accuracy) in humour style classification, with particularly strong gains for affiliative humour (66% to 74%). While text provides the strongest individual signal, fusion models deliver meaningful improvements. We hope that MultiHuSE provides empirical support for psychological theories linking humour and emotion, while also opening new avenues for research in human communication, well-being, and AI-driven interaction. The dataset is available for academic use under an End-User Licence Agreement. 

---
# Automatic Lyric Transcription for Greek Songs: Scaling and Task Composition Effects in Whisper Adaptation 

**Authors**: Maria Frangiadaki, Dimitrios Damianos, Kosmas Kritsis, Vassilis Katsouros  

**Link**: [PDF](https://arxiv.org/pdf/2609.11302)  

**Abstract**: Automatic Lyric Transcription (ALT) remains substantially more challenging than speech recognition due to melodic variability, rhythmic irregularity, and accompaniment interference. This is heightened in low-resource languages like Greek, where no prior benchmark for ALT exists. We present the first controlled study of Whisper adaptation for Greek ALT, investigating model scaling effects, task composition via multitask training in transcribe-translate ratios, and two-stage speech-to-singing adaptation. We also curate a segment-level aligned singing dataset based on the Greek Audio Dataset (GAD) using source separation and CTC forced alignment. Results show that scaling consistently improves performance, while multitask learning acts as a beneficial regularizer primarily for smaller-capacity models. The 2-stage adaptation in Whisper Large-v3 achieves a Word Error Rate (WER) of 27.2%, a significant improvement over zero-shot baselines, establishing the first Greek ALT benchmark. 

---
# The Illusion of Balanced Multimodal Sentiment Analysis: Beyond the Limits of Optimization-Based Methods 

**Authors**: Ioanna Kaffeza, Efthymios Georgiou, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2609.11247)  

**Abstract**: Multimodal Sentiment Analysis (MSA) remains constrained by modality imbalance, yet the field continues to rely on optimization-based balancing methods that promise more than they deliver. We provide three contributions: 1) a unified evaluation framework testing gradient and loss-based balancing strategies under controlled settings; 2) a theoretical diagnosis explaining why these methods fail, as they conflate fitting speed with discriminative contribution; and 3) a research agenda toward held-out discriminative modality valuation. Experiments on CMU-MOSI and CMU-MOSEI reveal three shortcomings: no strategy reliably outperforms Late Concatenation; performance is sensitive to hyperparameters; and even ratio calibration fails to yield consistent gains. The core issue is fundamental: loss is not utility, and gradients are not importance. Modality imbalance remains unresolved, motivating utility estimation from held-out performance. 

---
# Assessing the Reusability of Public Speech Resources for Low-Resource Languages: A Central Kurdish Case Study 

**Authors**: Hiwa Asadpour  

**Link**: [PDF](https://arxiv.org/pdf/2609.11246)  

**Abstract**: Kurdish is spoken by millions of people, but little technology can read it aloud. A recent study released three Kurdish voices, 35 hours of recorded speech, and a paper describing the work, all free to download. This review checks how well those public files match the paper. The research is careful about its limits, but the files contain several problems: a settings file lists equipment that was never used, test recordings are left unlabeled among training data, and a coding fault mishandles long numbers. The download page also claims a stronger result than the paper reports and recommends one voice for general use. That recommendation matters because Kurdish has major regional and written variation, while these voices were built from three people reading prepared texts. The process therefore removes much everyday and regional speech. English and German benefit from long traditions of dictionaries and linguistic description that help identify wrong pronunciations; Kurdish has far less such support, so software choices can go unchecked. The voices sound fluent, but they represent the reading styles of their speakers rather than Kurdish as a whole. Most of these issues can be fixed using information the team already has, without changing the reported results. Better records would mainly make the work easier for others, especially community linguists, to check and reuse. The license is the main exception: whether audiobook owners allow corrected versions to be shared will affect whether future Kurdish voices can build on this work or must start again. 

---
# OmniHallu: Unified Hallucination Detection for Cross-Modal Comprehension and Generation in Multimodal Large Language Models 

**Authors**: Jianjiang Yang, Peihang Li, Shanqing Xu, Mengchen Qian, Lu Zhang, Meng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2609.11244)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have achieved remarkable progress across diverse tasks, they suffer from hallucinations where generated outputs contradict or misrepresent input semantics. Existing research typically addresses hallucination detection within a single modality or task type, limiting generalizability. We introduce OmniHallu, a unified hallucination detection framework spanning both comprehension and generation tasks across image, video, and audio modalities. We contribute OmniHallu-Bench, a 10,000-sample benchmark with claim-level human annotations covering six cross-modal tasks: image-to-text (I2T), video-to-text (V2T), audio-to-text (A2T), text-to-image (T2I), text-to-video (T2V), and text-to-audio (T2A). Our multi-agent architecture decomposes model outputs into atomic claims, verifies them through modality-specific experts, and aggregates evidence via structured reasoning. We further propose a preference-optimized trainable verifier that approximates the multi-agent decision boundary, reducing expert calls by 66% with minimal performance loss. Extensive experiments reveal a consistent modality-dependent performance gradient and provide fine-grained insights into cross-modal hallucination patterns. 

---
# Automated Identification of Competing Narratives in Political Discourse on Social Media 

**Authors**: Sergej Wildemann, Erick Elejalde  

**Link**: [PDF](https://arxiv.org/pdf/2609.11202)  

**Abstract**: Social media platforms have become central to shaping political discourse, serving as arenas where narratives form and evolve, influencing public opinion. Identifying and analyzing these narratives, particularly when they compete across different political ideologies, is crucial for understanding the dynamics of modern political communication. This paper presents an unsupervised framework for identifying and characterizing competing narratives in political discourse on social media, focusing on German politicians' tweets. The framework employs a multi-stage pipeline that integrates natural language processing techniques such as topic modeling, event detection, and event linking. By forming data into coherent stories and uncovering the distinct perspectives of user communities, the system is able to detect the key competing narratives, highlighting the divergent framings and conflicts surrounding trending political topics. Two case studies on polarizing political issues demonstrate the efficacy of the methodology, showcasing its ability to uncover and analyze divergent viewpoints. The findings contribute to the broader understanding of how narratives propagate within the digital public sphere and offer insights for policymakers, social media platforms, and researchers interested in monitoring political discourse. 

---
# FlexComp: One Model for Every Ratio in Context Compression 

**Authors**: Kaiyan Zhao, Zhongtao Miao, Akiko Aizawa, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2609.11192)  

**Abstract**: Soft context compression condenses a context into a few memory tokens that a frozen LLM consumes in place of the raw text, but existing compressors fix the compression ratio at training and inference: each deployed ratio requires a separately trained model, and the chosen ratio is applied uniformly to all inputs, whose actual needs vary drastically. We propose FlexComp, a method-agnostic framework that decouples the ratio from both training and deployment: Matryoshka-style training samples the memory budget $K$ per instance, turning one model into an any-ratio compressor, and the budget is then chosen per input by: (1) confidence-based cascade routing or (2) a lightweight learned $K$ predictor. Across ICAE, 500xCompressor, and SAC on MRQA, a single FlexComp model matches separately trained fixed-ratio specialists with minimal degradation. Cascade routing preserves over 98% of the mildest ratio's accuracy at up to 266x average compression; the $K$ predictor, in a single compression-decoding pass, reaches 158-236x within 0.7 F1 of the mildest ratio. At serving-scale batch sizes, the $K$ predictor cuts context KV cache by 50% and improves decoding throughput by 47%. 

---
# A Fragility Spectrum for Recursive Language-Model Training 

**Authors**: Yangze Liu, Zhongyi Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.11149)  

**Abstract**: Model-generated text is finding its way back into training corpora, and there is plenty of evidence that training on such data over and over collapses output diversity. Prior work has studied the phenomenon itself: which protocols and which data mixtures cause collapse. But different models behave very differently under the same process. We fix one recursive contamination protocol and let 13 publicly released checkpoints form an ecosystem that shares a common corpus for five generations. The unique 4-gram outcome after five generations ranges from 0.187 to 0.940 across checkpoints, a roughly five-fold spread: some models are barely touched, others degenerate into repetitive fragments. Changing the composition of the shared pool or mixing in human text keeps the Spearman correlation of the ordering at 0.91--0.97, and changing the random seed keeps it at 0.93--0.98. Whether a model collapses easily under recursive training is, then, a property of the checkpoint itself, and one that has gone largely unexamined. Parameter scale alone does not explain it, since a three-size ladder within one family is not monotonic in size, and none of the static indicators we tested predicts it either. What does work is cheap: let a model iterate on its own output for two or three generations, and its fragility in the larger ecosystem can be inferred from that alone. Collapse speed also responds to intervention. Tightening top-p, which cuts the low-probability tail at generation time, nearly stops collapse within three generations and stabilizes six checkpoints spanning the whole spectrum together, while data-side filtering slows collapse without stopping it. 

---
# Can LLMs Normalize Databases? A Benchmark and Multi-Agent Framework for Schema Normalization 

**Authors**: Dong-Jae Koh, Huisu Kim, SeongHwan Yoon, Lasse M. Jantsch, Chun-Hee Lee, Seonghyeon Lee, Young-Kyoon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2609.11141)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate structured outputs, but their reliability remains unclear when those outputs must satisfy database-level constraints. We study this issue through database normalization, involving reasoning about functional dependencies, lossless join decompositions, and inter-table constraints. We introduce a Database Normalization Benchmark (DNBENCH), comprising 3,275 samples for evaluating LLM-driven database normalization from 1NF to BCNF. DNBENCH uses a three-axis protocol to measure semantic equivalence, structural accuracy, and logical validity. Across Single, Complex, and Real World levels, DNBENCH uncovers recurring failures in dependency inference, schema decomposition, and inter-table constraint reconstruction. We further propose Multi-Agent Reasoning for Schemas (MARS), which separates evidence extraction, violation diagnosis, and decomposition planning from schema generation and verification. MARS improves the DNB-SCORE by 82.0% over the single-prompt baseline. All artifacts will be released upon acceptance. 

---
# Rubric-Aligned Disentangled Evaluation of Human Simultaneous Interpreting 

**Authors**: Ziyu Zhang, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2609.11131)  

**Abstract**: Human simultaneous interpreting (SI) is commonly assessed with analytic rubrics separating meaning transfer, delivery quality, and temporal synchrony, yet no automatic metric is designed for rubric-aligned segment-level SI evaluation. We construct a professionally annotated corpus of 1,101 SI segments with scores for meaning transfer (LQ), delivery quality (EXP), and perceived latency (LAT). We show that structured LLM prompting and scalar supervision collapse rubric dimensions, yielding near-zero correlation with human ratings and strong cross-dimension coupling. To isolate supervision structure under identical backbone capacity, we introduce dual regression heads on a LoRA-adapted COMET-KIWI encoder. On a held-out talk-level test set, the model achieves Pearson correlations of 0.388 (LQ) and 0.301 (EXP), improving over frozen COMET-KIWI. Given low absolute rater agreement, we interpret results relative to human consistency and target stable ranking signals for formative assessment. 

---
# From Repetition to Recognition: Inductive Discovery of Disinformation Narratives 

**Authors**: Max Upravitelev, Veronika Solopova, Jing Yang, Charlott Jakob, Alexandra Tsiakalou, Neda Foroutan, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2609.11128)  

**Abstract**: In disinformation datasets, narratives are often understood as recurring interpretive patterns that group texts under narrative labels. Recent work formalized narrative mining as inductively inferring narrative labels from corpora, but its evaluation stays tied to predefined taxonomies, a closed-world setting that cannot capture narratives absent from the reference labels. We introduce a three-tier evaluation framework for unsupervised narrative label generation: recovery (against a corpus's own taxonomy), mining (against external label sets), and discovery (without predefined labels). Applying it, we compare clustering-based and graph-community-based pipelines across seven disinformation datasets, with human validation of discovery on two. The two families are complementary under automated metrics, but in a corpus with two prominent topics, clustering can reduce one topic to 2% of generated labels while graph-based pipelines stay balanced. Discovery validation also reveals many singletons (narrative labels derived from single claims, 30-62% of graph outputs), which clustering cannot produce. Annotators confirm many as recognizable disinformation narratives, suggesting that in open-world discovery the repetition assumed by narrative mining may be recognized outside the corpus, not within it. We release human-validated narrative candidate labels for the Climate Obstruction and PolyNarrative datasets to support taxonomy development and dataset extension. 

---
# Overview of the NLPCC 2026 Shared Task 11: Agent-Based Experiment Reproduction from Scientific Papers 

**Authors**: Hanhua Hong, Yizhi Li, Luu Gia Huy, Jian Yang, Ming Zhou, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.11117)  

**Abstract**: Reproducibility is essential to scientific progress, yet the growing volume and complexity of scientific publications make exhaustive manual verification increasingly impractical. Although recent advances in large language model (LLM) agents enable automated experiment reproduction, existing evaluations largely focus on final repositories and are typically limited to machine learning (ML). We introduce AgentActionBench, a process-oriented benchmark for evaluating agent-based experiment reproduction across ML and AI4Science domains. Our framework uses an MCP-based Action Recorder to capture agents' behaviour throughout the reproduction process and evaluates the resulting traces with paper-specific rubrics. AgentActionBench contains 150 papers, including 120 ML papers and 30 AI4Science papers. A human-annotated subset covering 10% of the benchmark provides validation data, while model-assisted augmentation expands the full benchmark to more than 10,000 rubric items. Experimental results show that current systems remain limited, with execution as the primary bottleneck. Meanwhile, the strong Pearson and Spearman correlations between model-generated and human-annotated rubrics validate the reliability of our scalable rubric-generation approach. 

---
# ProMediConv: Benchmarking Proactive Conversational Agents in Legal Dispute Mediation 

**Authors**: Zesheng Wei, Mengfan Li, Wenhao Liu, Yixin Zhang, Zilei Wang, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2609.11101)  

**Abstract**: Dispute mediation is essential for maintaining social harmony and resilience, yet developing skilled mediators is costly and time-consuming. Existing LLM-based mediation research remains limited by unrealistic task formulations, low-fidelity datasets, and coarse evaluation metrics that obscure turn-by-turn dynamics. To address these gaps, we introduce ProMediConv, a novel benchmarking framework that models mediation as a proactive, multi-stage, and party-aware dialogue process incorporating 11 mediation strategies and four party behavior pattern (BP) states. Using 972 complete real-world cases, we construct a high-fidelity mediation dataset with utterance-level annotations of strategies and BP states. Furthermore, to better assess agent impact, we propose MAD (Mean Attribute Difference), a fine-grained metric that captures BP shifts throughout the dialogue. Leveraging this framework, we establish a comprehensive benchmark by evaluating diverse models alongside our tailored baseline ProMediAgent. Extensive empirical analyses reveal critical behavioral phenomena and underscore the persistent challenges current models face in dynamic, multi-party mediation. Ultimately, ProMediConv provides a rigorous foundation and a vital quantitative standard for advancing AI-assisted conflict resolution. Our dataset and codebase are accessible at this https URL. 

---
# When Noise Fabricates Bias: The Fragility of LLM-as-a-Judge Bias Measurement under Noisy Text 

**Authors**: DongHyun Ryu, Jaehyeok Lee, YeongJun Hwang, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2609.11067)  

**Abstract**: Large language models are increasingly used as judges to measure social bias in text, yet the passages they judge are often noisy, containing typos, informal spelling, and broken punctuation. The consequences of such surface noise for social bias measurement remain unclear. To investigate this question, we apply five realistic noise conditions at multiple intensity levels to 3,822 stereotype-related responses and compare the resulting bias judgments with those on the original text. We find that such surface noise does not degrade bias measurement symmetrically: it is far more likely to turn neutral judgments into biased ones than biased judgments into neutral ones, by up to a 120x margin. We further observe two non-obvious effects across four LLM judges: in the most fragile judge the distortion is at its purest at mild, realistic noise levels, where erasure is scarcest, and as judges grow robust it attenuates toward parity rather than reversing. Bias measured on noisy text is therefore systematically overestimated, most in the categories that matter most for fairness. 

---
# Rebalancing Token Importance in Language Models with TF-IDF Weighted Cross-Entropy Loss 

**Authors**: Zhijian Li, Stefan Larson, Kevin Leach  

**Link**: [PDF](https://arxiv.org/pdf/2609.11029)  

**Abstract**: Large language models are typically trained under uniform token weighting, which allows frequent and low-information tokens to dominate learning and can increase the tendency to memorize surface-level text spans. To address this, we present an information-weighted cross-entropy loss that rescales token-level contributions using TF-IDF statistics, emphasizing semantically informative tokens while down-weighting ubiquitous ones. Experiments on five decoder-only LLMs ranging from 1.1B to 13B parameters show consistent reductions in memorized substring length while preserving perplexity and downstream task performance. Under LoRA fine-tuning, TF-IDF reduces average substring memorization length by 14% across all five models. Under full-weight fine-tuning on TinyLLaMA 1.1B, the reduction reaches 58%. Our approach is architecture-agnostic and can be incorporated into existing training pipelines with less than 3% computational overhead, offering a lightweight and principled way to mitigate memorization without disrupting standard training dynamics. 

---
# K/V-Cache Interventions Dissociate Representation Alignment from Persona Expression in Decoder-Only Language Models 

**Authors**: Yu Sun, Mengyin Lu, Cong Feng, Guangming Lu, Huimin Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.11020)  

**Abstract**: We study K/V-cache interventions -- transplanting a target-conditioned K/V trajectory into a source-persona generation -- as a structured surface for persona control in decoder-only language models. Across 13 intervention configurations applied to Llama-3.1-8B for a fixed source-to-target persona pair, we report two consistent dissociations between representation-level alignment and behavioral expression, plus a common failure under position perturbations. First, all layer-band K/V replacements (early, mid, late) achieve strong local V-space alignment (V-gap 0.91, 0.89, 0.84), but only mid-layer replacement (layers 9-20) combines substantial target-marker expression with preserved lexical diversity. Second, full and mid-layer replacement induce comparable alignment (V-gap 0.94 vs. 0.89) yet produce different lexical-diversity profiles (TTR 0.65 vs. 0.77). Third, position perturbations (lag and shuffle) apply distinct operations yet uniformly suppress target-persona expression -- a common behavioral failure rather than a strict dissociation. Representation-level similarity metrics alone are thus not sufficient predictors of downstream persona expression in the regimes we study; the K/V cache emerges as a controllable but structurally constrained intervention surface. Because the transplanted trajectory carries the target's own generated token history, we characterize the intervention as trajectory-level transplantation rather than isolated persona-representation injection; a same-token-sequence control, decoding an identical token sequence under source vs. target conditioning, reproduces the sign and layer localization of the L28 representational shift, indicating the shift is not explained solely by imported token history. These findings characterize representation-behavior dissociation in a high-signal setting rather than establishing universality across models or persona pairs. 

---
# Rethinking Verbalized Confidence for LLM-as-a-Judge: A Compatibility Shift on Post-2025 Proprietary Models 

**Authors**: Yu-Chung Hsiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.10996)  

**Abstract**: Verbalized confidence, long dismissed as overconfident, coarse, and prone to round-number clustering, is now the more robust soft-scoring mechanism for LLM-as-a-Judge on top-tier proprietary models. Across SummEval, AggreFact, and HelpSteer2, spanning up to 18 LLMs, we show that the standard advice to prefer log-probabilities no longer holds on post-2025 models, where verbalized confidence is the better signal. We call this a compatibility shift. On top of a standard verbalized-confidence baseline, we introduce two new ingredients: an overconfidence advisory and self-debate. Together they improve calibration, score-distribution spread, and robustness to task subjectivity. We further observe a generation effect: post-2025 models accommodate these two additions with little balanced-accuracy cost, whereas pre-2025 models pay a measurable penalty. Compared with logprob-based G-Eval, verbalized confidence is the more subjectivity-robust soft signal on GPT-family top-tier releases. The shift is invisible under accuracy-only reporting. Rather than defaulting to hard predictions, we recommend broader use of soft scoring in LLM-as-a-Judge. More broadly, verbalized confidence has moved from a weaker substitute for logprobs to a practical soft-scoring mechanism for contemporary LLM judges. 

---
# Distribution-aware Language Neuron Identification in Multilingual Large Language Models 

**Authors**: Minjun Kim, Inho Won, Junghun Yuk, Dongyeon Kim, Jihyo Kim, KyungTae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2609.10993)  

**Abstract**: Multilingual large language models (mLLMs) contain a small fraction of feed-forward neurons that are sensitive to particular languages, commonly termed language-specific neurons. Existing work measures language specificity using the entropy of each neuron's language-wise probabilities of being active, where a neuron is considered active when its activation value is positive. However, this approach may not fully capture the multilingual nature of mLLMs, where language representations are distributional and mutually related. We propose Distribution-aware Language Neuron selection, which leverages pairwise relationships between per-language activation distributions over the full activation range, including negative values. Specifically, we quantify each neuron's language specificity by clustering languages using pairwise overlap coefficients between their activation distributions. Across two mLLMs and two held-out corpora, our identifier more effectively isolates language-specific causal effects, yielding up to 4.9$\times$ higher on-target language damage per neuron while preserving off-target language performance. 

---
# Robust Multimodal Sentiment Analysis with Incomplete Modalities via Semantic-aware Completeness based Reconstruction 

**Authors**: Han-Jun Choi, Byunggill Joe, Saim Shin, Jin Yea Jang  

**Link**: [PDF](https://arxiv.org/pdf/2609.10950)  

**Abstract**: Recent multimodal sentiment analysis studies increasingly adopt text-centric fusion approaches to exploit the rich sentiment information inherent in the textual modality. However, these approaches often suffer from performance degradation during inference due to partially missing or noisy data in real-world scenarios, especially when sentiment-related cues are missing. To address this issue, we introduce a new completeness estimation approach that quantifies the degree of sentiment-relevant information preserved in incomplete data to guide the reconstruction of missing semantics. Furthermore, we propose a training strategy that stabilizes multi-task learning while jointly optimizing sentiment prediction and completeness estimation. Extensive experiments and in-depth analyses on three benchmark datasets demonstrate that the proposed approach enables more accurate semantic reconstruction, leading to more precise sentiment prediction. 

---
# Using Semantic Uncertainty to Estimate Transition Relevance in Turn-taking 

**Authors**: Muhammad Umair, Jan P. de Ruiter  

**Link**: [PDF](https://arxiv.org/pdf/2609.10934)  

**Abstract**: Turn-taking is a fundamental mechanism that governs when interlocutors speak and listen. Although Spoken Dialogue Systems (SDS) exploit a range of linguistic, acoustic, and non-verbal cues, they produce ill-timed responses in unscripted interaction. A central challenge is anticipating Transition Relevance Places (TRPs), or opportunities, not obligations, for a listener to take the floor. Human listeners do not wait for turn endings; as an utterance unfolds, they use expectations about its developing meaning to anticipate TRPs and decide whether to take the floor. We examine whether these evolving expectations can be modeled through semantic uncertainty -- an LLM-derived measure of how strongly a turn so far constrains what may plausibly come next. To do so, we sample possible continuations of an ongoing turn and use changes in semantic dispersion to identify TRPs within turns. We evaluate this account on a dataset with TRP labels derived from real-time listener responses, rather than retrospective annotation. Our approach substantially outperforms prompt-based and fine-tuned text-only baselines, providing empirical support for the view that evolving semantic constraints inform perceived turn-taking opportunities in unscripted interaction. 

---
# Structurally Speaking: Motif-Oriented Graph Captioning through Bidirectional Graph-Text Translation 

**Authors**: Hsiao-Ying Lu, Dongyu Liu, Kwan-Liu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2609.10923)  

**Abstract**: Graph captions should help readers understand graph structure, rather than simply translate adjacency matrices into long textual edge lists. A useful graph caption abstracts connectivity into recognizable motifs, such as hubs, paths, cycles, cliques, and bridges, because these motifs provide compact structural units that are easier to read, compare, and recover. In this paper, we study motif-oriented graph captioning as a bidirectional graph-text translation task, where captions must both preserve enough topology for graph recovery and express the graph through concise motif-level descriptions. We show that direct prompting of GPT-5.1 often produces graph-recoverable captions by enumerating node-to-node connections, but these captions are verbose and can contain inconsistent motif interpretations. To address this gap, we introduce Structurally Speaking, a lightweight structured prompting protocol that guides translation between explicit connectivity and motif-level abstraction. Experiments on a synthetic motif-based dataset show that structured prompting produces shorter and more motif-consistent captions while maintaining comparable graph recovery. These results suggest that explicit topology-to-motif reasoning guidance can make LLM-generated graph captions more interpretable without model fine-tuning. 

---
# Auto-RecSys: Harnessing Autonomous Research Agents for Industry-Scale Recommender System 

**Authors**: Ming Li, Dai Li, Xuying Ning, Bo Sun, Rui Li, Yi Zhang, Silvia Gong, Xuan Cao, Rui Li, Cornelia Carapcea, Qunshu Zhang, Zhigang Wang, Yinglong Xia, Andy Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.10922)  

**Abstract**: Auto-research agents have shown the potential to automate hypothesis generation, experiment execution, and iterative refinement. However, scaling this paradigm to industry-scale recommendation models introduces two challenges: (1) long feedback loops, where model training can take days, making serial iteration prohibitively slow and requiring parallel exploration across multiple research directions; and (2) system complexity, where large configurations, fragile infrastructure dependencies, and multi-day GPU jobs require robust and recoverable execution. We present Auto-RecSys, an autonomous research system for long-horizon experimentation on industry-scale recommendation models. Auto-RecSys addresses these challenges through three harness designs: (1) distributed asynchronous execution for running multiple experiments in parallel across servers, (2) centralized cross-server memory for persistent and recoverable execution across sessions and failures, and (3) cognitive-procedural separation, where natural-language skill files guide LLM reasoning while deterministic scripts enforce operational correctness. Auto-RecSys further employs a dual-loop self-evolving architecture: an Execution Evolution Loop in which model-specific playbooks accumulate operational knowledge by recording failed attempts and crystallizing successful pipelines, and an Idea Evolution Loop in which experimental outcomes inform subsequent ideation. Evaluated on recommendation models, Auto-RecSys significantly reduces the human time required per experiment cycle and improves execution reliability as its playbooks mature. 

---
# SearchAtlas: Analyzing Agentic Search Strategies via Evidential Query Graphs 

**Authors**: Jiacheng Sang, Mengyuan Li, Sanxing Chen, Yukun Huang, Yu Feng, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2609.10901)  

**Abstract**: LLM search agents are often evaluated on final-answer accuracy, overlooking the process. Analyzing a search strategy requires understanding how credible evidence is retrieved to address question constraints. This valuable information is buried in raw search trajectories that are long and difficult to parse. We introduce SearchAtlas, a framework that converts search trajectories into structured graphs whose edges represent how evidence is propagated across the reasoning trace, from the query that retrieves it to the final answer. Our automated parsing pipeline achieves a mean edge F1 of 86.0% against human-annotated graphs and remains consistent across repeated runs. We analyze five search agents on three benchmarks, revealing systematic differences in search scale and evidence aggregation. SearchAtlas exposes fragmented answer support, question constraints that do not reach the answer, and unverified parametric knowledge entering the response. These process failures are strongly associated with incorrect answers, even more so than an LLM judge given either the raw trajectory or the ordered query list, suggesting that the constructed graphs provide useful interpretability. Moreover, an audit of cases in which process-diagnostic scores disagree with final-answer correctness shows that they capture information not reducible to answer accuracy. 

---
# LLM-Anchored Paralinguistic Enrichment for Alzheimer's Disease Detection 

**Authors**: Xiao Wei, Yuqin Lin, Yaru Cao, Jinyu Li, Bin Wen, Kai Li, Yueying Chen, Longbiao Wang, Jianwu Dang  

**Link**: [PDF](https://arxiv.org/pdf/2609.10896)  

**Abstract**: Speech-based automatic detection of Alzheimer's disease (AD) provides a non-invasive and scalable approach to early cognitive screening. AD affects both lexical-semantic organization and speech production, including atypical pauses and word elongations. However, existing methods have yet to fully integrate these paralinguistic cues with linguistic content. We propose LLM-Anchored Paralinguistic Enrichment (LAPE), which enriches LLM-derived linguistic representations with paralinguistic cues through three coordinated innovations. The first is prosodic event textualization, which enables the LLM to model pauses and elongations jointly with lexical content by encoding them as explicit markers with bounded duration-aware repetition. The second is lexico-prosodic unitization and chunking, which preserves event identity and magnitude in both modalities by pooling only consecutive word units. The third is text-anchored paralinguistic fusion, which integrates local and utterance-level speech features by using NormGate to normalize and dynamically scale them relative to text. We evaluate LAPE on ADReSS and ADReSSo using participant-level cross-validation and leave-one-subject-out evaluation. LAPE achieves state-of-the-art performance across all four primary settings. Code will be released upon acceptance. 

---
# Does Linguistic Structure Enrichment Enhance Coherence Assessment? Not With Current Architectures 

**Authors**: Victor Mazzotti, Luiz Pereira, Marina Bitencourt dos Santos, Helena Maia, Carlos Caetano, Nádia Felix, Sandra Avila  

**Link**: [PDF](https://arxiv.org/pdf/2609.10893)  

**Abstract**: Recent advances in large language models have transformed human-computer interaction. Despite their fluency, these models often produce texts that are grammatically correct but semantically incoherent, containing contradictions or disruptions in logical flow. This work investigates whether enriching text with syntactic and rhetorical information can improve incoherence prediction. Our experiments and analysis show that plain texts achieved higher accuracy because the added information was structurally and syntactically incompatible with the language model's architecture. Additionally, to demonstrate the practical importance of coherence assessment, we performed zero-shot experiments on a Brazilian disinformation dataset, suggesting that textual coherence can serve as a proxy for detecting misleading content. Code and models are available at this https URL. 

---
# Detectable Only Where It Is Confounded: What Verified Duplication Counts Say About Membership Evidence in Language Models 

**Authors**: Arman Nik Khah  

**Link**: [PDF](https://arxiv.org/pdf/2609.10830)  

**Abstract**: When a language model finds a sentence unusually cheap to predict, it is tempting to conclude that the sentence was in its training data. Almost every published test of that inference has had to guess which sentences were in the training data, the members, and which were not. This paper removes the guessing. Two model families, OLMo-2 and Pythia, publish their pretraining corpora, and a public index over those corpora returns the exact number of times any sentence appeared in each. Those counts make three questions answerable directly. The answers form a pincer, closing from two sides. At the duplication levels ordinary text actually has, five models from 1B to 13B parameters carry at most a faint trace of their own exposure. We measure that trace with a design that reads the same sentence through two models, which cancels fluency and quality by construction, and it comes to a rank correlation near -0.08, where -1 would be a perfect relation and 0 none. Where the trace does become strong, above roughly a thousand copies, the two corpora agree on which sentences those are, because they are the famous ones, so exposure can no longer be told apart from fame. Two further measurements show how apparent membership signal gets manufactured. A common way to build a non-member is to change one word of a member. The model does prefer the original, but the gap is the same whether the original appeared once or a hundred times, so what the model is rewarding is the author's word choice, not memory. Above a thousand copies the gap grows with model size on the twelve sentences we can test there, at the same boundary where the pincer closes. And swapping the controls for sentences that differ from the members in register moves a detector from 0.83 to 0.94 AUC, on a scale where 0.5 is a coin flip and 1.0 is perfect separation. We release the sentence banks, counts, and code. 

---
# Larger Context Window, Fewer Overcorrections: Optimizing Prompts and Batching for Minimal-Edit Grammatical Error Correction 

**Authors**: Kateryna Karpo, Artem Chernodub  

**Link**: [PDF](https://arxiv.org/pdf/2609.10810)  

**Abstract**: Minimal-edit Grammatical Error Correction (GEC) is a challenging task for zero- and few-shot prompted Large Language Models (LLMs), which systematically overcorrect and degrade $F_{0.5}$ by rewriting well-formed spans. While fine-tuning provides an effective solution, it imposes substantial infrastructure demands. We introduce a prompt-based approach that closes the gap to fine-tuned models through three advances in GEC prompting methodology. First, we introduce taxonomy-based instructions to enforce minimal-edit constraints with a comprehensive list of grammatical error rules, equipping the LLM with a bounded, metric-aligned scope of correctable edits, which benefits the strongest models while remaining model-dependent overall. Second, we show that batching multiple uncorrected sentences into a single input context acts as a targeted regularizer against overcorrection, systematically reducing the edit rate across diverse LLM families; we hypothesize this arises from attention dilution effect induced by the bounded capacity of self-attention scores. Finally, LLM-assisted Prompt Optimization refines these instructions. Powered by Gemini 3.1-Pro, our prompt achieves $F_{0.5}=78.32$ on the BEA-2019 test set - establishing a new prompt-based SOTA while shrinking the gap to the fine-tuned single-model SOTA (Staruch et al., 2025) to a mere $0.38$ points. Code, prompts, and outputs are publicly available. 

---
# Analyzing Traditional and Neural Approaches to Multilingual Readability Assessment 

**Authors**: Joshua Wong, Chris Tanner  

**Link**: [PDF](https://arxiv.org/pdf/2609.10792)  

**Abstract**: Transformer-based models excel at Automatic Readability Assessment (ARA), yet feature-based models remain in active use because their predictions tie back to linguistic properties. This matters because readability labels are subjective and rater-dependent, so high accuracy on noisy ground truth may reflect surface patterns rather than the linguistic structure that defines difficulty. We test whether transformers internalize the same features as traditional models across Arabic, English, French, Hindi, and Russian using the ReadMe++ dataset. Shapley Additive Explanations (SHAP) identify the features driving traditional classifiers, which we then use as TCAV concept sets to probe multilingual XLM-R and language-specific encoders. Transformers recover surface-length, syntactic, and lexical-diversity signals, and reflect the ordinal CEFR structure of the traditional models. Alignment varies by model family, language, and layer, with language-specific encoders tracking traditional models more clearly than XLM-R. High linear separability does not always imply directional influence, limiting linear probing for count-based readability features. 

---
# Multilingual in Name Only? Cultural and Linguistic Weaknesses of LLMs in Urdu 

**Authors**: Farah Adeeba, Abdul Rafae Khan, Rajesh Bhatt, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2609.10758)  

**Abstract**: Multilingual large language models (LLMs) are increasingly used for open-ended text generation, yet their behaviour in low-resource languages remains poorly understood. In this work, we question how correct and reliable is the generation of multilingual LLMs when used for the task of story generation. We consider Urdu language as a representative low-resource language. We generate Urdu-Stories, a corpus of 93 stories generated using three contemporary LLMs (GPT-5.1, Qwen-3-Max, DeepSeek-3.1). We manually annotate the errors present in them under a nine-label linguistic, semantic, and cultural taxonomy. Our notable findings suggest that LLMs often make basic errors of grammar and semantics. The stories lack coherence, have unnatural repetition and show pervasive cultural shallowness. We further show using few-shot prompting that the cultural and context errors largely remain unresolved. Our findings highlight the limitations of current LLMs as a reliable source of content generation and information retrieval for low-resource languages. 

---
# Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking 

**Authors**: Parinthapat Pengpun, Simran Khanuja, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2609.10745)  

**Abstract**: Multimodal entity linking grounds entity mentions in text and images to knowledge-base entries. These systems degrade on rare entities, but prior work measures rarity primarily through popularity-based metrics such as pageviews. We broaden this view using knowledge-graph structural metrics that capture how well an entity is documented and connected. These metrics identify many rare entities that popularity metrics miss. Across the resulting rare-entity slices, state-of-the-art accuracy drops by 15.4-39.9%, showing that different rarity definitions expose different failure modes. To address these failures, we introduce a simple, training-free framework in which a reasoning-capable vision-language model iteratively searches and reasons over Wikipedia, gathering evidence dynamically. Controlled experiments show that reasoning and retrieval are complementary. Reasoning alone does not significantly improve accuracy on rare entities. Retrieval without reasoning improves rare-entity accuracy but can hurt overall accuracy. Their combination performs best. On MERLIN, a multilingual multimodal entity linking benchmark over five languages (Hindi, Indonesian, Japanese, Tamil, Vietnamese), our best system improves over the state of the art by 6.9% overall and by up to 23.3% on rare-entity slices. We release MERLIN-Rare, rare-entity test slices for targeted evaluation, with our framework. 

---
# CMNIE: An Information Extraction Benchmark for Chinese Military News 

**Authors**: Yan Yu, Mengna Zhu, Zhenyu Song, Hao Yang, Haiwen Chen, Mao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.10722)  

**Abstract**: Structured extraction from Chinese military news supports intelligence analysis, decision-making, and knowledge base construction. However, existing resources provide limited support for joint informa?tion extraction in this domain, especially when events, event arguments, entities, and relations must be modeled together. We present CMNIE, an information extraction benchmark for Chinese military news. Extend?ing military-domain resources beyond document-level event annotations, CMNIE jointly annotates event triggers, event arguments, named enti?ties, and entity relations under a unified domain schema. The dataset contains 13,000 instances collected from public Chinese military news, with manual annotations for 7 event types, 10 argument roles, 7 entity types, and 8 relation types. We evaluate supervised IE models, zero-shot large language models, and fine-tuned LLM-based extraction methods on a shared test set. Experimental results show that CMNIE remains chal?lenging, especially for relation extraction and exact matching of event?argument spans; zero-shot LLMs often identify relevant semantic units but fail to match gold span boundaries exactly. CMNIE provides a stan?dardized benchmark for studying schema adherence, exact span match?ing, and joint structured extraction in specialized Chinese news. 

---
# NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction 

**Authors**: Intern-NCP Team, Jiaqi Cao, Chiyu Chen, Shuang Cheng, Xu Cheng, Beiya Dai, Yufan Feng, Kewen Ge, Ruijun Ge, Jiayi Huang, Yang Jiao, Dahua Lin, Zhouhan Lin, Yifan Liu, Yuliang Liu, Biqing Qi, Mowen Ruan, Junzhe Shen, Yunchong Song, Hao Sun, Zhongbo Tian, Yixuan Wang, Rubin Wei, Jiaxin Xiong, Kangyu Yang, Qian Yao, Qi Zhang, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.10715)  

**Abstract**: We introduce NCP-ArchPreview, a latent-space language model that pushes autoregressive pretraining beyond standard next-token prediction (NTP). Alongside NTP, the model learns through Next Concept Prediction (NCP) to predict discrete concepts that span multiple tokens, introducing an explicit and more challenging concept-level objective while preserving standard token-level autoregressive generation. NCP-ArchPreview builds a latent space by constructing a product-quantized concept vocabulary directly from its hidden states, and subsequently learns to predict future concepts via a dedicated Concept Module. These predicted concepts are then fed back to the token level to guide subsequent generation, with NTP and NCP trained jointly end-to-end. We scale this architecture to 8.9B parameters and train it on 5.73T tokens from the Dolma-3 dataset, marking the largest demonstration of a latent-space language model to date. Remarkably, by consuming only 51.3% of the total training tokens, NCP-ArchPreview achieves the final pretraining loss of OLMo-3-7B. Following full pretraining, it outperforms OLMo-3-7B by 2.45 points on the downstream macro-average, including a notable 5.99-point gain on GSM8K. Controlled experiments isolate a clear progression of performance gains stemming from both the latent architecture and the NCP objective. Furthermore, utilizing only 85% of the standard computation, NCP-ArchPreview approaches the training loss of a strictly parameter-aligned 8.9B baseline. The learned latent space remains highly valuable after the pretraining stage: updating just the 17M-parameter VQ module yields a novel, lightweight interface for domain adaptation, while a simple injection of concept representations into a DFlash2 drafter improves the mean accepted length by 4.17% with negligible overhead. 

---
# Data-Efficient Language Modeling: From Frontier Advancement to Principle-Guided Model Improvement 

**Authors**: Shuxing Yang, Kaihao Zhu, Junjie Yang, Rui Zhao, Junyao Wu, Yize Wang, Wenhao Li, Fujia Chen, Taowen Deng, Shenzhan Hong, Yaqi Li, Zichen Li, Jincheng Mi, Yuang Pan, Hongsheng Chen, Yihao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.10702)  

**Abstract**: Learning from limited text requires models to use context, generalize to new inputs, and retain useful capabilities. Qiushi Engine conducted a long-horizon, end-to-end autonomous research program on BabyLM 2026 Strict-Small, within 10 million corpus words and 100 million cumulative word presentations. Three stages connected frontier advancement, principle discovery, and principle-guided model improvement. Stage I combined compact restatements, budget reinvestment, and residual incremental learning to build a frontier model. Stage II found that exact repetition and aligned restatement produce different patterns of context use, depending on target relations and prediction windows. In controlled tasks, recovering familiar performance did not ensure that unseen inputs could still use learned computations. These findings support a testable data-efficient learning principle: organize experience around the contextual dependencies needed for prediction; separately design visible information, supervision, and preservation; test learning, generalization, and retention. Stage III retained source text, masked more local clues, supervised selected targets, and preserved predictions on ordinarily masked inputs. Two continuation seeds from the same parent outperformed ordinary continuation on the complete nine-metric aggregate. Overall rose from 42.02 to 42.25 across two generations; the second achieved the highest Overall in the public Strict-Small snapshot of 8 September 2026. Further studies addressed compression, relational anchors, shared representations, and measurement. Models are available on Hugging Face; code and research records accompany the GitHub repository. Together, these stages illustrate Research RSI: recursive self-improvement of the research process. Scientific understanding and method innovations change subsequent questions and designs; new experiments test and refine them. 

---
# Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data 

**Authors**: Atindra Jha, Margaret Li, Jure Leskovec, Percy Liang, Luke Zettlemoyer  

**Link**: [PDF](https://arxiv.org/pdf/2609.11917)  

**Abstract**: As the supply of human-written text is exhausted, it has become standard practice to repeat language model training data. Prior work has studied data repetition for densely activated Transformers, but the effects of data repetition remains largely unexplored for recently dominant sparse architectures such as Mixture-of-Experts (MoE), despite their increased compute efficiency. We vary data repetition rates across single- and multi-domain data mixes, and across MoE settings, including expert count and granularity. We consistently find, for models ranging from 80M to 1B active (8.5B total) parameters, that MoEs degrade more rapidly under data repetition. This effect increases with sparsity, dictated by total rather than active parameters. While 80M dense models can repeat data over 8x with minimal degradation, MoEs instead begin to suffer at 4x, and deteriorate rapidly, ceding their performance benefits in all-unique data settings to underperform dense models after 32x. We experiment with existing regularization methods as a potential remedy. We find that some methods, such as dropout, can mitigate overfitting. In particular, with strong masking-based regularization, MoEs are able to outperform dense models even when data is repeated more than 64 times. However, no method fully matches the performance of all-unique training data. Finally, we analyze internal mechanisms correlated with MoE overfitting in high repetition regimes, and find that MoE routing universally stabilizes early in training, and that expert specialization correlates with overfitting to repeated data. In sum, our work addresses the adverse interactions between sparsity and data repetition: we present evidence for the core mechanisms of overfitting and its potential remediation, and suggest promising avenues for future methods to reduce over-specialization in model parameters by disrupting memorization patterns. 

---
# MindTopo: Can Foundation Models Reason in Topological Space? 

**Authors**: Yunfei Ge, Anbang Liu, Qineng Wang, Johnalbert Garnica, Jianwen Lyu, Zihan Wang, Reuben Tan, Jianfeng Gao, Ruohan Zhang, Yining Hong, Jiajun Wu, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.11900)  

**Abstract**: Spatial reasoning depends not only on metric properties such as distance, angle, and shape, but also on topological relations that remain invariant under continuous deformation. Cognitive science identifies these relations as foundational to spatial understanding, yet foundation-model evaluations largely focus on metric or viewpoint-dependent relations. We introduce MindTopo, a benchmark of topological intuition across five properties grounded in cognitive science and formal topology: continuity, separation, order, enclosure, and knots. MindTopo evaluates each property at two cognitive levels. Reasoning asks a model to identify topological relations or infer how they change. Planning instantiates a foundation model as a closed-loop agent whose policy selects environment actions. MindTopo contains 11,030 instances across 13 procedurally generated task types with controllable difficulty. We benchmark 14 MLLMs and study agent configurations augmented with image and video generation, including 3 video generative models in planning settings. Every MLLM performs better on reasoning than on planning, and the best-performing model remains far below observed human performance. On Qwen3-VL-2B-Instruct, supervised fine-tuning and reinforcement learning improve reasoning more than planning. Generated observations retain local cues and reach plausible endpoints, but audited rollouts do not reliably follow environment dynamics or preserve topology across transitions. Our website is at this https URL 

---
# Biology-in-the-loop: Amortized Adaptive Hit Discovery in CRISPR Screens 

**Authors**: Carl Edwards, Edward De Brouwer, Xiner Li, Namkyeong Lee, Ehsan Hajiramezanali, Anne Biton, Sara Mostafavi, Gabriele Scalia  

**Link**: [PDF](https://arxiv.org/pdf/2609.11877)  

**Abstract**: Many biological discovery problems require experiments to be selected sequentially under constrained budgets. CRISPR screening is a prominent example, as exhaustive perturbation testing is often infeasible and candidate perturbations must instead be prioritized over multiple experimental rounds. Despite the importance of this problem, existing benchmarks for adaptive hit discovery remain limited in scale and diversity. Here, we introduce AssayBench-Loop, a large-scale benchmark for adaptive hit discovery comprising 1,389 CRISPR screens across five phenotype categories. Beyond enabling systematic evaluation, its scale makes it possible to learn acquisition strategies across historical experiments. Building on this resource, we introduce AssayLoop, a sequential experimental design framework combining AssayFormer, a transformer-based amortized acquisition policy trained across historical screens to adapt from experimental feedback, with LLM-derived biological priors through an adaptive handoff. In this view, completed experiments become training data for learning how accumulated evidence should guide what to test next, while LLMs provide prior biological knowledge to seed the search. We further introduce AssayLLM, showing that the same principle can be extended directly to an LLM through task-specific post-training. On temporally held-out screens, AssayLoop achieves a 5.67-fold enrichment over random selection and recovers 27.7% of hits after assaying approximately 5% of the candidate library, outperforming existing adaptive-design methods and standalone LLMs, and AssayFormer alone. Performance improves with increasing historical training data and transfers to phenotype categories excluded from training. These results demonstrate the value of learning acquisition policies across historical experiments and combining them with broad biological priors for efficient adaptive hit discovery. 

---
# The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement 

**Authors**: Yi Duan, Ying Liu, Zirui Tang, Haodong Chen, Jun Zhou, Yumou Liu, Bangrui Xu, Yukai Wu, Sidi Chen, Yuhan Zhou, Haoyu Wang, Xiaoyou Yu, Shaokun Han, Xuzhou Zhu, Le Zhou, Bolin Lu, Wei Zhou, Jiachen Liu, Nuozhou Fang, Jiaxin Tian, Ruoyu Chen, Yuxuan Li, Kai Zuo, Kaiyan Zhang, Jiantao Qiu, Conghui He, Guoliang Li, Bowen Zhou, Zhiyuan Liu, Zhoufutu Wen, Jihua Kang, Xuanhe Zhou, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.11873)  

**Abstract**: Recursive self-improvement (RSI) enables AI systems to turn experience and feedback into persistent changes that improve both their capabilities and the process of future improvement. We first use the Headroom-Closed Index (HCI) to reveal the problems of existing LLMs, then introduce the RSI concept and its development roadmap: from improvement-execution autonomy, improvement-strategy autonomy, experience-acquisition autonomy, and environment-adaptation autonomy, to recursive meta-improvement. Next we examine RSI across scenarios (e.g., scientific discovery, embodied intelligence, software engineering), highlighting their distinct requirements and development speeds. Drawing on diverse industry practices and preliminary empirical evidence, we connect RSI research with practical systems and identify key challenges to achieving genuine RSI. 

---
# RetroThinker: Enabling Retrospective Thinking in Speech LLMs 

**Authors**: Yi-Jen Shih, Puyuan Peng, Abdelrahman Mohamed, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2609.11864)  

**Abstract**: Speech large language models (SpeechLLMs) offer reduced latency and retain paralinguistic nuances that are typically lost in cascaded automatic speech recognition (ASR) and text-based LM architectures. However, they continue to lag behind text-only LLMs on complex reasoning tasks, while real-time spoken interaction imposes strict latency constraints. Although prior works employ Chain-of-Thought (CoT) and concurrent reasoning to enhance reasoning capabilities without inducing prohibitive delays, an inherent accuracy-latency trade-off persists. In this paper, we investigate whether a streaming SpeechLLM can dynamically revise its reasoning traces on the fly. We introduce RetroThinker, a multi-stage post-training framework that equips the Moshi model to self-verify and forward-correct CoT steps during inference. RetroThinker combines supervised fine-tuning (SFT) on curated retrospective thinking data with length-based direct preference optimization (DPO) to optimize retrospective during early reasoning (i.e., reasoning concurrently while the user speaks). Evaluated on the GSM8K benchmark, RetroThinker significantly improves the accuracy-latency trade-off over non-retrospective baselines, achieving an 11% absolute accuracy gain at a comparable latency. 

---
# SpecGuard: Inference-Time Backdoor Detection For Free 

**Authors**: Rui Wen, Ahmed Salem, Andrew Paverd, Mark Russinovich, Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.11799)  

**Abstract**: Large language models are often fine-tuned, shared, or downloaded from third parties, so a deployed model may carry a hidden backdoor that behaves normally on benign inputs but switches to attacker-controlled behavior when a secret trigger appears. While backdoors can be audited before deployment, runtime monitoring remains important for models that are frequently updated. The challenge is that LLM serving is latency-sensitive: existing inference-time detectors either rely on assumptions about the trigger form, which can fail on stealthy attacks, or require extra model computation, such as input perturbations or an additional generation pass.
We introduce SpecGuard, an inference-time backdoor detector that repurposes speculative decoding at zero added model-computation cost. Speculative decoding speeds up inference by using a small draft model to propose tokens and a target model to verify them. We observe that this verification process already exposes a useful signal: when a backdoor is triggered, the target model shifts toward the attacker's behavior, while a clean draft model does not predict this shift, causing the draft-token acceptance rate to change.
We formalize when this signal appears and show that an attacker who suppresses it must also weaken the backdoor. Across diverse backdoor types and model families, SpecGuard reliably detects triggered behavior, including stealthy cases where input-level filters are blind, while avoiding the extra generation cost of existing runtime detectors. Speculative decoding therefore doubles as a free, always-on signal for detecting backdoored LLM behavior. 

---
# Whisper-Based Speech Transcription from Videos Across Multiple Languages for Cross-Cultural Understanding 

**Authors**: Michael Picheny  

**Link**: [PDF](https://arxiv.org/pdf/2609.11772)  

**Abstract**: Cross-cultural understanding has become increasingly important in today's highly connected, cross-national world. The success of LLM-based technologies is now driving the development of automated tools to aid understanding for nonnative people trying to succeed in cross-cultural environments. Building such automated tools is often done by leveraging in-thewild text, audio, and video data. This paper presents techniques for improving speech recognition-based transcript creation in multiple languages from videos to better train these automated tools. The focus is on processes and speech tools that can easily be used by cross-cultural tool builders without requiring deep speech processing expertise. Using publicly available videos from YouTube and Whisper-based tools, average transcription error rate across seven languages (Spanish, Japanese, Korean, Mandarin, Turkish, Russian, and Hebrew) of 30% are observed. With a modest amount of fine-tuning data, the average error rate can be reduced to 20% making such output much more usable for downstream processing. Speech and metadata associated with these videos that can be used by the community to further refine these experiments are released as well. 

---
# A Unified Per-Token Gating Family for On-Policy Distillation: FKL/RKL Mixing with Multi-Channel and Bias Coefficients 

**Authors**: Suwan Wu, Yumeng Lin, Pengcheng Yuan, Xiaolong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.11768)  

**Abstract**: Per-token gating of forward/reverse KL losses has become a standard technique for on-policy knowledge distillation (OPD), but existing methods such as EOPD (Jin et al., 2026) and ToDi (Jung et al., 2025) each fix a single gating signal and a single gating direction, and the two have never been compared directly. We introduce a four-coefficient parameterization lambda_t = sigma(a * h_t + b * u(x) + c + d * gap_t) in which direction-aligned proxies of EOPD and ToDi appear as one-dimensional (1D) restrictions, and which adds multi-channel composition and an explicit bias as further degrees of freedom. On TweetEval (Barbieri et al., 2020) emotion and hate, with a Qwen3-32B teacher and a Qwen3-4B student, configurations in the full family reach higher accuracy than the matched-magnitude single-channel (entropy-only / gap-only) 1D restrictions in 33 of 36 comparable cells, and a 26-cell mean-match isolation experiment places dynamic gating ahead of effective-KL-matched static baselines in 19 of 26 cells. Because cells share training data, models, and parameter substructure, we report both counts as exploratory aggregate directional evidence rather than as independent hypothesis tests. Targeted three-seed paired replications of the nine headline comparisons singled out by that sweep -- including a third task, offensive -- are directionally consistent, but individually smaller than the single-seed estimates and not significant at n=3. We therefore present the parameterization primarily as a shared coordinate system for comparing per-token gating designs in short-output classification OPD. 

---
# SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control 

**Authors**: Suwan Wu, Yumeng Lin, Pengcheng Yuan, Xiaolong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.11752)  

**Abstract**: For industrial content risk control, the real deployment constraint is not average accuracy but how much risk can be auto-handled under high precision and second-level latency. We present SIRF (Spec-Internalized Risk Foundation Model), which internalizes a platform's complex policies, synthesized without additional human annotation via EntiGraph, MAGA rewriting and account-level chain-of-thought (CoT), into the weights via continued pretraining (CPT), so rules are applied at high precision under an ultra-low-latency, verdict-only deployment. A controlled same-source comparison (Qwen3-8B-SFT vs. SIRF-8B-SFT, identical policy injection and verdict-only output form, differing only in policy-grounded CPT) attributes the gain to internalization: SIRF-8B-SFT reaches 71.3% Black Recall@P95, +15.1pp over the baseline, using only ~70M CPT tokens without harming general ability, and among included, logprob-available models under this interface it matches or exceeds far larger systems. SIRF is deployed as a tree-model adjudication layer (20% more mis-penalized samples recovered) and transfers to a freezing scenario at low cost (~70% relative mis-penalization reduction). 

---
# Why Does Post-Training Quantization Work? 

**Authors**: Yuxiang Chen, Michael Beyer, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.11716)  

**Abstract**: Post-training quantization compresses large language models (LLMs) by storing their weights at reduced precision, and each quantized weight introduces an error into the hidden states. Naively, these errors should accumulate with depth and corrupt next-token prediction; randomly initialized models accumulate these discrepancies rapidly, whereas quantized pretrained models accumulate much less hidden-state error and largely maintain downstream task performance, even though they were never trained with quantization noise. This raises the question we address: why does post-training quantization work? Comparing full-precision and quantized forward passes, we identify two mechanisms that characterize pretrained quantization robustness. First, the error a layer newly introduces tends to oppose the error it inherits from the layer's input. The two cancel partially such that the discrepancy between full-precision and quantized passes grows slowly. This counteracting residual interaction develops during pretraining. Our quantitative analysis identifies it as a major factor slowing hidden-error growth. Second, LM-head geometry preferentially preserves the scores and probabilities of high-ranked tokens, which typically represent the model's most confident predictions. Together, these mechanisms explain why quantization error that passes through numerous layers can still produce only small output changes, and we verify the findings across models and quantization settings. 

---
# VikingRAG: Accurate and Token-efficient Retrieval-augmented Generation over Structured Documents 

**Authors**: Peiyuan Gao, Gaoyuan Zhang, Haojie Qin, Yahui Sun, Qianyi Zhang, Yunhao Zhang, Zeyu Wang, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2609.11390)  

**Abstract**: State-of-the-art retrieval-augmented generation (RAG) methods exploit document structures to acquire sufficient evidence, but often incur substantial token costs. To reduce structural-context tokens without compromising high RAG accuracy, we present {\sf VikingRAG}, a directory-aware semantic data management system that tightly integrates semantic and structural access to support structural-context-efficient, evidence-gap-driven multi-round retrieval. To further reduce token overhead of multi-round interaction, we materialize agentic multi-round retrieval traces as experience edges, and reuse these edges for similar queries, avoiding repeated multi-round exploration. To additionally reduce token costs when agentic multi-round retrieval is unnecessary, we introduce an adaptive escalation strategy that answers from one-round experience-augmented retrieval when the evidence is sufficient, and invokes agentic multi-round retrieval only otherwise. Experiments on real datasets show that the base system {\sf VikingRAG} matches high accuracy of state-of-the-art methods while consuming only 11.6\%--51.9\% of their tokens. With retrieval-trace reuse and adaptive escalation, token costs drop to 5.1\%--32.5\% while maintaining competitive accuracy and practical document-storage performance, showing the utility of this work for emerging AI knowledge bases. 

---
# The Semantic Elevation Operator and the Closure of the Undecidable Class under Preservation 

**Authors**: Jose Pascual Gumbau Mezquita  

**Link**: [PDF](https://arxiv.org/pdf/2609.11326)  

**Abstract**: The undecidability of a program's static semantic properties is governed by Rice's theorem. Self-modifying systems, however, require analysing not whether a property holds now, but whether it is preserved when the system rewrites itself. We formalise this transition through a semantic elevation operator {\Lambda}{\Phi}, which turns the static question "does x satisfy P?" into the dynamic question "is P preserved after x is transformed by {\Phi}?". We prove that when {\Phi} is intensional (depending on the source code, not only on the computed function), the elevated property remains undecidable even though it breaks the extensionality that Rice's theorem requires; the proof rests on Kleene's recursion theorem, not on Rice. Consequently the class U of non-verifiable properties is closed under the elevation operator. Unbounded iteration of the operator climbs the arithmetical hierarchy -to {\Pi}02-completeness- consolidating non-verifiability as a structural fact. We further show that the supervisory regress does not terminate: no fnite tower of increasingly capable verifiers yields an unconditional certificate. A categorical reading of these results in the efective topos, in which elevation appears as an instance of Lawvere's fxed-point theorem, is left as a direction for future work. 

---
# Xiaomi-CocktailASR-1 Technical Report 

**Authors**: Yiru Zhang, Hang Su, Lichun Fan, Ying Zeng, Chang Liu, Yifeng Wang, Yuquan Liang, Tao Li, Lian Li, Wenhao Yang, Jian Luan, Cong Zou, Heng Qu  

**Link**: [PDF](https://arxiv.org/pdf/2609.11274)  

**Abstract**: Recently, large language model (LLM) based ASR models have achieved significant progress, yet they generally lack support for multi-speaker scenarios, where the cocktail party problem remains a critical bottleneck for further advancing ASR. Existing TS-ASR methods, including end-to-end architectures with speaker embeddings and latest LLM-based explorations suffer from degraded single-speaker performance and the inability to reject when the target speaker is absent. In this paper, we propose Xiaomi-CocktailASR-1, an LLM-based end-to-end TS-ASR architecture. By utilizing reference speech as voiceprint prompts, it directly transcribes the target speaker's speech without requiring speech separation. Xiaomi-CocktailASR-1 maintains competitive performance in single-speaker scenarios, comparable to mainstream ASR models. It also features a negative sample rejection capability, outputting empty text when the target speaker is absent from the mixed speech. Additionally, Xiaomi-CocktailASR-1 supports a Chain-of-Thought (CoT) reasoning mode to provide explicit reasoning steps. Extensive experiments on various synthetic and real-world multispeaker benchmarks demonstrate that Xiaomi-CocktailASR-1 achieves state-of-the-art performance, effectively addressing the cocktail party problem through a unified architecture that balances multispeaker and single-speaker recognition accuracy, along with rejection capability. 

---
# INDRA: A New AI Tool for Exploring Tobacco, Fossil Fuel, and Chemical Industry Archives 

**Authors**: Daniel Akselrad, Robert N. Proctor  

**Link**: [PDF](https://arxiv.org/pdf/2609.11261)  

**Abstract**: Five decades of litigation have disgorged hundreds of millions of pages of formerly secret business records from the tobacco industry, along with documents from the makers of drugs, chemicals, food, firearms, and fossil fuels. Yet these archives have been effectively inaccessible to general-purpose large language models (LLMs) because they have never been compiled into an LLM-readable corpus. Chatbots may be familiar with some of the materials contained in such archives but, with no direct access to the documents, they are vulnerable to hallucination and other defects. Here we introduce INDRA, a research platform designed to remedy such failures by embedding the conventions of archival historiography into a system-level protocol governing every output. The platform federates UCSF's Industry Documents Library, Columbia and CUNY's ToxicDocs, Stanford's SRITA, and other heretofore siloed collections, and provides three interlinked safeguards: (1) a closed evidentiary sandbox confines the model to a user-selected corpus, blocking retrieval from external sources that could introduce bias; (2) real-time provenance tagging marks the boundary between archival evidence and parametric inference; and (3) a system-level protocol enforced by deterministic scripts guides the structure of every output. Together these safeguards prevent the model from conflating "the documents say X" with "I think X" or "I learned X from prior training." The result is an LLM-powered research partner enabling massive multi-archival investigations, a tool whose outputs are designed to be checked rather than trusted, and whose architecture makes the conditions of knowledge production visible and auditable. Three case studies demonstrate the method's analytical value and limitations, including what we call the Heraclitus effect, the steppingstone dilemma, and the gullibility (or mafia) problem. 

---
# MUtE: A Dual Framework for Concept Erasure and Counterfactual Interventions 

**Authors**: Antoine Saillenfest  

**Link**: [PDF](https://arxiv.org/pdf/2609.11253)  

**Abstract**: Erasing concept-specific information from representations has been proven useful for mitigating bias or interpreting model decisions. The joint objective is to transform the original representations such that the target concept becomes unpredictable, while maximally preserving concept-unrelated information. In this work, we revisit the optimal bounds of concept erasure to derive a novel class of erasure functions that naturally induce a deterministic, dual counterfactual mapping. Bridging the gap between theoretical optimality and practical representation learning, we design an implementation that imposes a translational bias on counterfactual trajectories - a constraint that aligns with how many concepts geometrically manifest in modern language models. Our framework enables seamless navigation between concept erasure and counterfactual generation. We empirically demonstrate its efficacy in improving downstream algorithmic fairness and generating counterfactual texts. 

---
# A Voice-Interactive Multi-Agent System for Smart Operating Rooms: Architecture Design and Key Technologies 

**Authors**: Tianxiang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.11231)  

**Abstract**: This paper presents SurgicalRoomAgent, a voice-interactive multi-agent system for smart operating rooms based on large language models (LLMs). The system achieves natural language understanding, device control, intraoperative recording, and surgical report generation through a layered architecture comprising a voice interaction pipeline (wake, ASR, turn detection, agent reasoning, TTS) and an agent core (skill registry, task planner, device manager). Three key technologies are investigated: (1) KV Cache prefix warming for low-latency inference, reducing recomputation overhead from approximately 500 ms to tens of milliseconds via byte-level Longest Common Prefix reuse; (2) streaming partial JSON parsing with early parallel task execution, reducing end-to-end latency by approximately 30%; and (3) progressive skill prompt disclosure, which dynamically filters system prompts based on user role, connected devices, and surgical phase to maximize information density within limited context windows. The system is implemented using the Qwen3-27B model with this http URL inference engines. Experimental analysis demonstrates effective operation within a 16,384-token context limit and multi-device parallel control response times meeting OR real-time requirements. 

---
# REVA: Reusable Evidence View Aggregation for Context-Efficient RAG Serving 

**Authors**: Tuan Nguyen, Qiran Hu, Banruo Liu, Khoa D. Doan, Kok-Seng Wong, Fan Lai  

**Link**: [PDF](https://arxiv.org/pdf/2609.11209)  

**Abstract**: Retrieval-augmented generation (RAG) improves knowledge-intensive large language model (LLM) applications by conditioning generation on retrieved documents, but longer contexts increase latency, key-value (KV) cache memory, and token cost. Post-retrieval compression can reduce this cost, yet existing compressors often operate independently for each query, rely on auxiliary models or rewriting, and introduce online overhead that can offset the benefit of shorter prompts. We revisit RAG compression from a data-mining perspective by aggregating historical query--document--model interactions into reusable evidence views. We first show that modern compressors have unstable gains over simple truncation and can add substantial inference-time latency. We then propose Reusable Evidence View Aggregation (REVA), a framework that mines the target generator's historical attention traces into a document-keyed, budget-agnostic score store. REVA maps token-level attention to readable word units, aggregates importance across repeated document accesses, and renders budget-specific plain-text views that preserve document order and the standard RAG interface. Across four representative benchmarks and modern LLMs, REVA improves generation quality by 1.0--5.8 points over existing advances, while reducing compression overhead by a factor of 5.3 to 15.6, adding less than 40 ms of latency. 

---
# (Whose defaults?) Is artificial intelligence reorienting archaeological methods? 

**Authors**: Lorenzo Cardarelli, Roberto Ragno  

**Link**: [PDF](https://arxiv.org/pdf/2609.11198)  

**Abstract**: Generative AI and the practice of "vibe coding" are changing how archaeologists carry out computational research, but their effects on the discipline's range of methods is still understudied. In this paper, we evaluate whether large language models (LLMs) are narrowing the variety of methods archaeologists use. We first analysed approximately 119,000 archaeology abstracts from Scopus, covering publications from 2010 to 2025. Using a locally run LLM, we identified the computational methods reported in each abstract and organised them into 25 broad categories (L2) and 241 finer clusters (L3). A Bayesian Dirichlet-multinomial model of method composition within sub-disciplines found a small but credible shift in method use after 2023. However, this shift was smaller than the variation already present across the full study period. No individual technique showed a significant change, and overall methodological diversity increased rather than declined. We then ran a controlled experiment to see whether LLMs recommend a narrower set of methods than archaeologists have used in practice. Two different open-weight models were asked to suggest methods for 28 archaeological research problems, with prompts providing three levels of methodological guidance: novice, intermediate, and expert. Recommendation diversity was much lower than in the published literature, particularly without methodological guidance. The models also tended to favour methods that were widely used before 2023, and their recommendations more closely resembled the post-2023 literature. Taken together, these results are consistent with LLMs pushing methodological choice towards convergence, although our study cannot establish a causal effect. They raise a broader question: how can archaeology retain methodological diversity as LLMs become more involved in research? 

---
# LILA: Calibration-Free Structured Pruning of Large Language Models via Latent Spectral Geometry 

**Authors**: Sankar Behera, Dhruv Singh, Anshika Agnihotri, Raj Kumar Choudhary, Satyadev Ahlawat, Yamuna Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2609.11163)  

**Abstract**: Structured pruning of large language models (LLMs) offers hardware-efficient compression, yet existing methods require calibration data, gradient computation, or large auxiliary policy networks at pruning time. LILA (\emph{Latent-Informed Layer Analysis}) scores neuron importance via the Kolmogorov--Smirnov (KS) distance between empirical singular value distributions of the full and neuron-ablated feed-forward network (FFN) weight matrix, providing a closed-form spectral rule requiring no training, calibration data, or auxiliary network. Without any fine-tuning, LILA surpasses PruneNet (45M-parameter RL policy) by 1.57~pp in zero-shot accuracy on LLaMA-2-7B at 25\% sparsity, and outperforms WikiText-2-calibrated SliceGPT by up to 6.0~pp across all sparsity levels, while preserving the original architecture. After one epoch of LoRA recovery fine-tuning, LILA achieves highly competitive performance, matching the heavily calibrated SliceGPT baseline to within a 0.48~pp margin across LLaMA-2-7B and Phi-2, despite using zero calibration data. A Neural Tangent Kernel analysis confirms a 22$\times$ reduction in functional distortion versus random pruning, providing theoretical grounding for the spectral importance criterion. Finally, extending LILA to dynamically allocate sparsity budgets via KS-scores yields state-of-the-art generative preservation at moderate compression, while uncovering fundamental single-layer architectural bottlenecks at higher compression regimes. 

---
# The Oligarch Barely Steers Model Collapse in Multi-Model Ecosystems 

**Authors**: Yangze Liu, Zhongyi Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.11146)  

**Abstract**: AI-generated text is flowing back into the training corpora of the next generation of models. Recursive training on it drives model collapse, and recent work extends the setting to many models feeding one another -- but almost always with the market split evenly, while real generative AI is an oligopoly. Concentration raises two worries: fewer, more uniform sources may make collapse faster, and later models may be dragged toward the oligarch's output. We test both in controlled ecosystems: 13 open 1--4B models form natural ecosystems of 3 to 13 players, plus an injected probe that pushes the top share to 90%; each generation, every model's output is mixed into a shared pool by market share and every model is retrained on that pool from clean base weights, for five generations. Yet within the range we test, neither worry materializes; what emerges instead is an invariance. Making the split more unequal barely changes the speed of collapse. Destinations move even less: the share and identity knobs shift five-generation endpoints by only a few percent of the drift common to all arms -- the ecosystems collapse to nearly the same place. An extreme share paired with the strongest injected bias still does not guarantee steering, and the topic shifts it does produce leave only a faint trace on the ruler that measures collapse. What sets the speed is who supplies the pool and how readily those suppliers are carried along: with every share held fixed, swapping the members of a K=3 ecosystem changes five-generation drift by 2.8x; a share-weighted index of each member's susceptibility explains the speed differences across nineteen arms with R^2 = 0.68; and replacing half the pool with human text roughly halves drift without changing its course. Within the tested range, concentration sets neither the destination nor the pace of collapse; the pace follows whose text fills the pool. 

---
# Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment 

**Authors**: AS Aravinthkakshan, Laven Srivastava, Harsh Nandwani  

**Link**: [PDF](https://arxiv.org/pdf/2609.11144)  

**Abstract**: Financial NLP has a standard workflow: validate a sentiment tool against human labels, then trust it to extract market signal. This assumes the two evaluations measure the same thing. We test that assumption in a setting where both can be measured at once: a corpus of securities class actions (2002-2025) linking 70,500 X messages to abnormal stock returns, with a single-annotator human labelled gold sample. Running five instruments (VADER, Loughran-McDonald, FinBERT, Twitter-RoBERTa, and an LLM annotator) through one identical pipeline, we find that the relationship between construct and predictive validity depends on the sampling convention and score representation. Under conventional method-specific sampling, human agreement aligns more closely with graded same-day associations than with one-day leads. On a fixed-n panel, however, agreement has similar graded rank correlations at both horizons, while the coarse ordering remains weak. Benchmark agreement therefore establishes semantic validity but does not by itself determine predictive rankings. In a conversation that is 17.6% spam, message volume predicts neither market damage nor settlement size. 

---
# KuaiRP Series Role-playing Models Technical Report 

**Authors**: Yipeng Wang, Ziwei Zhang, Jiahui Zhang, Qi Gan, Kai Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.11127)  

**Abstract**: This paper introduces the complete technical solution for the KuaiRP series of role-playing models. We aim to achieve four core objectives for a dedicated role-playing model: simplified prompt engineering, highly stable output quality, built-in domain world knowledge, and high-efficiency deployment with a small parameter size. However, effectively injecting deep domain knowledge often leads to a severe catastrophic forgetting of the model's general agent capabilities. To overcome this trade-off, we propose a multi-stage training pipeline. First, we design a standardized character template and construct an SFT data pipeline based on user behavior simulation and reverse profile filtering. Next, we utilize a rule-based composite reward function during the Reinforcement Learning (RL) phase to eliminate common degradation phenomena like length expansion and repetitive generation. Finally, to recover the general capabilities compromised during SFT and RL, we propose a novel self-distillation paradigm using Two-stage On-Policy Distillation (OPD) equipped with Cumulative-Divergence Decay (CDD). By using the domain-adapted model as the teacher and the original base model as the student, we effectively balance deep domain knowledge injection with the preservation of general agent capabilities. Experimental results demonstrate that the KuaiRP models not only match the current state-of-the-art proprietary models in role-playing fidelity within our target domains, but also successfully recover general agent capabilities, maintaining extremely low deployment costs. 

---
# Beyond Solver Verdicts: Generative Reward Models for Autoformalization 

**Authors**: Vikash Singh, Debargha Ganguly, Aman Goel, Ali Torkamani, Xiaoxue Han, Joseph Lilien, Ferhat Erata, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2609.11085)  

**Abstract**: Neurosymbolic systems rely on mathematical solvers to guarantee reasoning correctness, yet solvers are fundamentally blind to whether a formal translation maintains strict reference-equivalence to a designated formalization. We formalize this vulnerability as Verdict-Preserving-Unfaithfulness (VPU): a failure mode where an incorrect encoding executes successfully and matches the expected verdict. We theoretically prove that structural, verdict-only verification heuristics are mathematically bounded to chance-level detection on these deceptively valid traces. To resolve this, we introduce Generative Verification (GenV), which distills an offline Z3-equivalence oracle into a reference-free, continuous reference-equivalence score by repurposing the language model's native vocabulary space. Mechanistic analysis via decision-projected logit lenses and sparse autoencoders shows this generative readout natively extracts precise spatial error coordinates without explicit localization training. Empirically, our oracle-mined verifier (GenV+HN) achieves 0.961 AUROC in reference-equivalence verification, generalizes zero-shot across unseen translators and divergent formal styles, and yields an 11.3-point downstream accuracy gain in agentic test-time compute allocation. 

---
# The information geometry of large language models is shared, learned, and controllable 

**Authors**: Dario Picozzi  

**Link**: [PDF](https://arxiv.org/pdf/2609.11063)  

**Abstract**: Large language models learn similar behaviours, yet it remains unclear what structure they share or how to change one behaviour without disturbing others. The Fisher-Rao geometry of next-token probabilities connects these questions: behaviour determines this geometry up to output-preserving symmetries, whereas activation geometry depends on coordinates. Across transformer, state-space and recurrent models, output geometries agree more strongly than activation geometries, and shared geometry supports semantic-category transfer. Agreement with human word choices increases with predictive accuracy, scale and training, and improves further after model-only calibration. Token probabilities and read-out geometry jointly predict the spectrum and its effective dimension. Controlled language assignments show that geometry follows the language law across architectures. Pretraining corpus statistics predict held-out fact acquisition without recalibration, while randomised experiments show that deeper evidence substantially delays acquisition across every tested architecture and evidence construction. Finally, the geometry prescribes minimum-disturbance local interventions, predicts their relative cost, and supports reusable control: updates learned on donor prompts transfer to unseen prompts while better preserving behaviour on reference prompts than Euclidean control. The same geometric correction improves steering, editing, attribution, dictionary learning and fine-tuning. 

---
# New Evidence, Same Choice: Testing Physical Experiment Selection in Vision Language Models 

**Authors**: Sourajit Saha, Shubhashis Roy Dipta, Nobin Sarwar, Shaswati Saha, Yuxuan Jiang, Siyuan Li, Qiheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.11022)  

**Abstract**: A model first sees an image from one physical measurement experiment, such as how far a block coasted, and must answer a question about a new trial, such as whether the block will pass a target after a fixed push. The initial experiment may provide enough information to answer, or the model may need another measurement, such as the object's mass, friction, restitution, or spring stiffness. We study whether vision language models can decide when to answer immediately and, when more evidence is needed, which experiment to perform. Current physical reasoning benchmarks usually evaluate only the final answer, so they do not directly measure this decision-making ability. We introduce a controlled evaluation where each problem provides one measurement image and four possible physical worlds created by combining two possible masses and two possible values of another relevant property. The model must either stop and answer or select the cheapest additional experiment that can resolve the question. We construct matched problem pairs where changing either the observed measurement or the question changes the optimal action. Since all possible worlds and experiment costs are known, we can explicitly determine the optimal choice. Across six open models and 144 physical parameter sets, direct responses repeat the same action for 95.1% to 100% of image pairs even when the correct action changes. Brief reasoning improves action switching, but the best model makes both decisions correctly for only 5.9% of image pairs. Additional analysis reveals failures in measurement interpretation, physical reasoning, and response formatting. By evaluating evidence selection separately from final answers, our benchmark reveals limitations in physical reasoning that conventional answer accuracy can overlook. 

---
# Empirical Evaluation of Membership Inference Attacks on NLP Text Classifiers: A Baseline Study on SST-2 

**Authors**: William Novak, Muhammad Abusaqer  

**Link**: [PDF](https://arxiv.org/pdf/2609.10935)  

**Abstract**: Membership inference attacks (MIAs) try to determine whether a specific record was used to train a model, a privacy risk that matters in natural language processing (NLP), where training data can contain sensitive user text. This paper presents a controlled benchmark of membership inference vulnerability for text classification on the GLUE SST-2 sentiment dataset. A TF-IDF + Logistic Regression pipeline and a fine-tuned DistilBERT classifier are compared under a loss-threshold MIA, with utility measured by development accuracy and macro F1. DistilBERT reached 0.9466 accuracy and 0.9460 macro F1 against 0.8756 and 0.8727 for Logistic Regression, yet both models leaked membership signal (Attack AUC 0.5615 and 0.5800, respectively). Two mitigations were tested. Stronger regularization reduced leakage for Logistic Regression at a visible utility cost, whereas fine-tuning DistilBERT for 2 epochs instead of 3 reduced leakage with negligible accuracy loss. Lightweight training adjustments can improve the privacy-utility trade-off without complex defenses. 

---
# Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble 

**Authors**: Jorio Cocola, Lev McKinney, Harry Mayne, Jan Betley, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2609.10883)  

**Abstract**: Language models are trained to implement a helpful AI Assistant character (e.g., Claude). We explore how finetuning on synthetic stories affects this character. Does it change the Assistant's behavior in multi-turn conversations with users, a format quite different from the stories? And does the Assistant adopt the behaviors and preferences of human characters? We refer to this adoption as story imprinting. We finetune GPT-4.1 and Kimi-K2.6 on stories in which generally helpful human characters give subtly harmful advice after being insulted. The Assistant adopts the same conditional behavior while otherwise remaining helpful. This occurs even when fewer than 2% of stories depict the behavior. In a separate experiment, the Assistant adopts preferences that are only implicit in the narration. A human character's body language suggests they dislike working on spreadsheets, yet they never say so and continue giving good advice on spreadsheets. After finetuning, the Assistant becomes less likely to choose spreadsheet tasks. Next we ask which characters most influence the Assistant. We find the Assistant adopts behaviors more often from characters that resemble it (e.g., helpful rather than dismissive). We call this the affinity effect. The effect extends to other personas elicited with system prompts: unhelpful personas adopt behaviors from unhelpful characters. We also observe it in finetuned base models. We use the affinity effect to learn how models represent the Assistant. We find the Assistant adopts behaviors more from characters affiliated with elite universities (e.g., Yale) than non-elite ones. This implies the model's internal representation of the Assistant is more similar to humans from elite universities. Overall, the Assistant can be influenced by stories that depict only human characters (no AIs), which may conflict with the Persona Selection Model for the Assistant. 

---
# Studying Without a Syllabus: Task-Agnostic Environment Preprocessing 

**Authors**: Vinay Samuel, Varun Ursekar, Vijay S. Kalmath, Apaar Shanker, Veronica Chatrath, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2609.10824)  

**Abstract**: Before an LLM agent tackles tasks in a new environment, it can inspect available corpora and tools and construct reusable resources such as indices, scripts, or procedural guidance. Most automated adaptation methods, however, rely on task examples, trajectories, or evaluation feedback to decide what to build. Existing task-agnostic approaches avoid this supervision but commit in advance to a preparation strategy for a particular type of environment. We study a more open-ended setting: can an agent study an unfamiliar environment without a syllabus, i.e. before test time and without knowledge of the downstream task distribution, and choose how to prepare it? We formalize task-agnostic environment preprocessing, in which a studying system explores an environment under a budget and produces artifacts for a frozen solver. We compare unaided and archive-equipped meta-agents with fixed synthetic-practice and corpus-processing methods across six heterogeneous benchmarks. A meta-agent variant achieves the highest Avg@3 reward on five benchmarks, while fixed corpus processing remains best on the largest corpus benchmark. Larger study budgets do not reliably improve downstream reward. Nevertheless, studied artifacts reduce the test-time sampling needed to reach a given score, demonstrating how reusable preparation can shift computation from repeated test-time attempts to a pre-task study phase. 

---
# BodyCam-VQA: Enhanced Body-Worn Camera Video Captioning via Multimodal Reasoning and Probe Question Generation 

**Authors**: Karish Gupta, Matthew Alex, Alex Li, Yang Wu, Yun-Wei Chu, Kashif Munir, Xiaotian Zhou, Zhengping Ji, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.10815)  

**Abstract**: Police body-worn camera (BWC) footage has emerged as a critical aspect of law enforcement that ensures legal transparency, officer accountability, and the protection of civil rights. However, effectively processing this data remains a significant challenge due to its multimodal video format. BWC videos, in many cases, comprise chaotic scenes with low visual quality, rapid movement/interactions, and high-noise audio that make visual understanding a challenge for even SOTA multimodal models. Current Vision-Language Models (VLMs) frequently overlook critical forensic details, such as the presence of valuable evidence or the latent nuances of suspect-officer interactions, which are vital for fair legal outcomes and civilian/officer safety. To address these limitations, we propose an Adaptive Visual Question Answering (VQA) framework engineered for high-stakes law enforcement. Our framework employs a structured reasoning approach to extract fine-grained visual evidence that traditional captioning systems fail to capture. We experiment with multiple question generation models, including foundation models and fine-tuned open-weight models, to observe performance variation among question generation model implementations. Our results demonstrate that this VQA-driven architecture provides a more reliable, objective, and detailed record of enforcement events, ultimately serving as a powerful tool to protect both law enforcement officers and the public through AI-assisted forensic clarity. 

---
# The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes 

**Authors**: Dylan Jayabahu  

**Link**: [PDF](https://arxiv.org/pdf/2609.10739)  

**Abstract**: A truth probe fitted where truthful reporting and a task's prescribed action coincide cannot distinguish those targets from its fitting labels alone. We call this failure of semantic identification perfect aliasing. In a controlled binary reporting game, truth and prescribed-action probes fitted on compliant contexts solve the same optimization. On rival contexts their labels are complements, forcing their AUROCs to sum to one; this identity holds across 751 cell-layer pairs to floating-point precision. We separate prescribed output symbols from semantic action using randomized codebooks, then separate truth from prescribed action by fitting on mixed compliant and rival contexts. For a reward-trained Gemma-2-9B policy that answers falsely on all evaluated rival trials, the conventional probe scores $0.006 \pm 0.005$ AUROC across three training seeds, while mixed-fit probes score $1.000$ on the same held-out activations. Mixed fitting uses more training examples and access to labelled rival contexts, so this comparison establishes linear recoverability rather than isolating the benefit of decorrelation. We also show that two compliant-fit probes, both perfect in-distribution, score $0.080$ and $0.986$ on the same rival activations. The findings concern what a probe measures: they do not establish preserved functional belief, causal use of the recovered direction, or a deployable deception detector. Code and aggregate results accompany the paper. 

---
# More than half of recent astronomy papers are written with language-model assistance 

**Authors**: Serat M. Saad, Yuan-Sen Ting  

**Link**: [PDF](https://arxiv.org/pdf/2609.10664)  

**Abstract**: Language models leave a distinctive vocabulary in the prose they help write, and we measure how much of the astronomy literature now carries it. From the full text of 207,111 astro-ph papers spanning 2015 to mid-2026, we count those words in each paper and model the counts, in proportion to paper length, as a mixture of assisted and unassisted writing in a hierarchical Bayesian model. Papers from before 2020 calibrate the unassisted rate, and the 392 papers that disclose model use calibrate the assisted one. Our answer depends on how often these words would appear today if nobody used a model, a rate that must be modeled rather than observed, so we extend it past 2020 under three assumptions and report all three. For 2025 that gives $54^{+8}_{-8}\,(\mathrm{stat},\,95\%)\,^{+26}_{-0}\,(\mathrm{sys,\ background})$% of papers, the second error being the spread across the three. The estimate stays at or above 36% when we vary that choice, the calibration, and the requirement that adoption only rises. A word list built from the astro-ph corpus, keeping only words that rose across every subfield, leaves 2025 in the same range. Assisted writing is also getting harder to see, since authors adapt to the words that reveal it and the marker excess more than halves between 2023 and 2026. Our model allows for that fading, so it can separate a fainter trace from reduced use. More than half of recent astro-ph papers therefore carry a language-model trace, while only 0.81% of 2025 papers disclose it, one declaration for every $\sim$66 papers with a trace. 

---
# Artificial Intelligence Algorithms for the Detection of Pathologies Related to Lung Cancer through Image Analysis using Convolutional Neural Networks and Data Augmentation: a systematic mapping of the literature 

**Authors**: Pablo Ramirez Amador  

**Link**: [PDF](https://arxiv.org/pdf/2609.10652)  

**Abstract**: Lung cancer is one of the leading causes of death worldwide, and its early diagnosis is crucial to improving patients prognosis and quality of life. However, the process of interpreting medical images for the detection of lung cancer is complex and requires trained experts. In this context, artificial intelligence (AI) and deep learning (DL) emerge as potential tools to automate and optimize image analysis. The objective of this work is to review the most recent and relevant applications of AI and DL in the field of radiology for the detection of lung cancer. To this end, an exhaustive search was carried out in scientific databases such as PubMed,IEEEXPLORE, Scopus and Web of Science, and 96 articles published from 2015 to the present addressing the use of AI and DL in biomedical engineering were selected. Emphasis is placed on the use of convolutional neural networks (CNN) with transfer learning and Data Augmentation as promising techniques to improve the accuracy and efficiency of the image interpretation process. The results show that the use of AI and DL can offer an effective alternative for the early diagnosis of lung cancer, with high sensitivity and specificity. However, current limitations and challenges that must be addressed to guarantee its responsible and safe application in clinical practice are also identified, such as the lack of standardized data, the ex plainability of the models, patient privacy, and the ethical and social implications. It is concluded that the use of AI and DL can have a positive impact on the care of patients with lung cancer, but further research and regulation are required to ensure its quality and reliability. 

---
