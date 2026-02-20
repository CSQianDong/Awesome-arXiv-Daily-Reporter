# Sink-Aware Pruning for Diffusion Language Models 

**Authors**: Aidar Myrzakhan, Tianyi Li, Bowei Guo, Shengkun Tang, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.17664)  

**Abstract**: Diffusion Language Models (DLMs) incur high inference cost due to iterative denoising, motivating efficient pruning. Existing pruning heuristics largely inherited from autoregressive (AR) LLMs, typically preserve attention sink tokens because AR sinks serve as stable global anchors. We show that this assumption does not hold for DLMs: the attention-sink position exhibits substantially higher variance over the full generation trajectory (measured by how the dominant sink locations shift across timesteps), indicating that sinks are often transient and less structurally essential than in AR models. Based on this observation, we propose ${\bf \texttt{Sink-Aware Pruning}}$, which automatically identifies and prunes unstable sinks in DLMs (prior studies usually keep sinks for AR LLMs). Without retraining, our method achieves a better quality-efficiency trade-off and outperforms strong prior pruning baselines under matched compute. Our code is available at this https URL. 

---
# What Language is This? Ask Your Tokenizer 

**Authors**: Clara Meister, Ahmetcan Yavuz, Pietro Lesci, Tiago Pimentel  

**Link**: [PDF](https://arxiv.org/pdf/2602.17655)  

**Abstract**: Language Identification (LID) is an important component of many multilingual natural language processing pipelines, where it facilitates corpus curation, training data analysis, and cross-lingual evaluation of large language models. Despite near-perfect performance on high-resource languages, existing systems remain brittle in low-resource and closely related language settings. We introduce UniLID, a simple and efficient LID method based on the UnigramLM tokenization algorithm, leveraging its probabilistic framing, parameter estimation technique and inference strategy. In short, we learn language-conditional unigram distributions over a shared tokenizer vocabulary but treat segmentation as a language-specific phenomenon. Our formulation is data- and compute-efficient, supports incremental addition of new languages without retraining existing models, and can naturally be integrated into existing language model tokenization pipelines. Empirical evaluations against widely used baselines, including fastText, GlotLID, and CLD3, show that UniLID achieves competitive performance on standard benchmarks, substantially improves sample efficiency in low-resource settings - surpassing 70% accuracy with as few as five labeled samples per language - and delivers large gains on fine-grained dialect identification. 

---
# Differences in Typological Alignment in Language Models' Treatment of Differential Argument Marking 

**Authors**: Iskar Deng, Nathalia Xu, Shane Steinert-Threlkeld  

**Link**: [PDF](https://arxiv.org/pdf/2602.17653)  

**Abstract**: Recent work has shown that language models (LMs) trained on synthetic corpora can exhibit typological preferences that resemble cross-linguistic regularities in human languages, particularly for syntactic phenomena such as word order. In this paper, we extend this paradigm to differential argument marking (DAM), a semantic licensing system in which morphological marking depends on semantic prominence. Using a controlled synthetic learning method, we train GPT-2 models on 18 corpora implementing distinct DAM systems and evaluate their generalization using minimal pairs. Our results reveal a dissociation between two typological dimensions of DAM. Models reliably exhibit human-like preferences for natural markedness direction, favoring systems in which overt marking targets semantically atypical arguments. In contrast, models do not reproduce the strong object preference in human languages, in which overt marking in DAM more often targets objects rather than subjects. These findings suggest that different typological tendencies may arise from distinct underlying sources. 

---
# Unmasking the Factual-Conceptual Gap in Persian Language Models 

**Authors**: Alireza Sakhaeirad, Ali Ma'manpoosh, Arshia Hemmat  

**Link**: [PDF](https://arxiv.org/pdf/2602.17623)  

**Abstract**: While emerging Persian NLP benchmarks have expanded into pragmatics and politeness, they rarely distinguish between memorized cultural facts and the ability to reason about implicit social norms. We introduce DivanBench, a diagnostic benchmark focused on superstitions and customs, arbitrary, context-dependent rules that resist simple logical deduction. Through 315 questions across three task types (factual retrieval, paired scenario verification, and situational reasoning), we evaluate seven Persian LLMs and reveal three critical failures: most models exhibit severe acquiescence bias, correctly identifying appropriate behaviors but failing to reject clear violations; continuous Persian pretraining amplifies this bias rather than improving reasoning, often degrading the model's ability to discern contradictions; and all models show a 21\% performance gap between retrieving factual knowledge and applying it in scenarios. These findings demonstrate that cultural competence requires more than scaling monolingual data, as current models learn to mimic cultural patterns without internalizing the underlying schemas. 

---
# The Cascade Equivalence Hypothesis: When Do Speech LLMs Behave Like ASR$\rightarrow$LLM Pipelines? 

**Authors**: Jayadev Billa  

**Link**: [PDF](https://arxiv.org/pdf/2602.17598)  

**Abstract**: Current speech LLMs largely perform implicit ASR: on tasks solvable from a transcript, they are behaviorally and mechanistically equivalent to simple Whisper$\to$LLM cascades. We show this through matched-backbone testing across four speech LLMs and six tasks, controlling for the LLM backbone for the first time. Ultravox is statistically indistinguishable from its matched cascade ($\kappa{=}0.93$); logit lens reveals literal text emerging in hidden states; LEACE concept erasure confirms text representations are causally necessary in both architectures tested, collapsing accuracy to near-zero. Qwen2-Audio genuinely diverges, revealing cascade equivalence is architecture-dependent, not universal. For most deployed use cases, current speech LLMs are expensive cascades, and under noise, they are worse ones, with clean-condition advantages reversing by up to 7.6% at 0 dB. 

---
# Modeling Distinct Human Interaction in Web Agents 

**Authors**: Faria Huq, Zora Zhiruo Wang, Zhanqiu Guo, Venu Arvind Arangarajan, Tianyue Ou, Frank Xu, Shuyan Zhou, Graham Neubig, Jeffrey P. Bigham  

**Link**: [PDF](https://arxiv.org/pdf/2602.17588)  

**Abstract**: Despite rapid progress in autonomous web agents, human involvement remains essential for shaping preferences and correcting agent behavior as tasks unfold. However, current agentic systems lack a principled understanding of when and why humans intervene, often proceeding autonomously past critical decision points or requesting unnecessary confirmation. In this work, we introduce the task of modeling human intervention to support collaborative web task execution. We collect CowCorpus, a dataset of 400 real-user web navigation trajectories containing over 4,200 interleaved human and agent actions. We identify four distinct patterns of user interaction with agents -- hands-off supervision, hands-on oversight, collaborative task-solving, and full user takeover. Leveraging these insights, we train language models (LMs) to anticipate when users are likely to intervene based on their interaction styles, yielding a 61.4-63.4% improvement in intervention prediction accuracy over base LMs. Finally, we deploy these intervention-aware models in live web navigation agents and evaluate them in a user study, finding a 26.5% increase in user-rated agent usefulness. Together, our results show structured modeling of human intervention leads to more adaptive, collaborative agents. 

---
# Learning to Stay Safe: Adaptive Regularization Against Safety Degradation during Fine-Tuning 

**Authors**: Jyotin Goel, Souvik Maji, Pratik Mazumder  

**Link**: [PDF](https://arxiv.org/pdf/2602.17546)  

**Abstract**: Instruction-following language models are trained to be helpful and safe, yet their safety behavior can deteriorate under benign fine-tuning and worsen under adversarial updates. Existing defenses often offer limited protection or force a trade-off between safety and utility. We introduce a training framework that adapts regularization in response to safety risk, enabling models to remain aligned throughout fine-tuning. To estimate safety risk at training time, we explore two distinct approaches: a judge-based Safety Critic that assigns high-level harm scores to training batches, and an activation-based risk predictor built with a lightweight classifier trained on intermediate model activations to estimate harmful intent. Each approach provides a risk signal that is used to constrain updates deemed higher risk to remain close to a safe reference policy, while lower-risk updates proceed with standard training. We empirically verify that harmful intent signals are predictable from pre-generation activations and that judge scores provide effective high-recall safety guidance. Across multiple model families and attack scenarios, adaptive regularization with either risk estimation approach consistently lowers attack success rate compared to standard fine-tuning, preserves downstream performance, and adds no inference-time cost. This work demonstrates a principled mechanism for maintaining safety without sacrificing utility. 

---
# Using LLMs for Knowledge Component-level Correctness Labeling in Open-ended Coding Problems 

**Authors**: Zhangqi Duan, Arnav Kankaria, Dhruv Kartik, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2602.17542)  

**Abstract**: Fine-grained skill representations, commonly referred to as knowledge components (KCs), are fundamental to many approaches in student modeling and learning analytics. However, KC-level correctness labels are rarely available in real-world datasets, especially for open-ended programming tasks where solutions typically involve multiple KCs simultaneously. Simply propagating problem-level correctness to all associated KCs obscures partial mastery and often leads to poorly fitted learning curves. To address this challenge, we propose an automated framework that leverages large language models (LLMs) to label KC-level correctness directly from student-written code. Our method assesses whether each KC is correctly applied and further introduces a temporal context-aware Code-KC mapping mechanism to better align KCs with individual student code. We evaluate the resulting KC-level correctness labels in terms of learning curve fit and predictive performance using the power law of practice and the Additive Factors Model. Experimental results show that our framework leads to learning curves that are more consistent with cognitive theory and improves predictive performance, compared to baselines. Human evaluation further demonstrates substantial agreement between LLM and expert annotations. 

---
# Bridging the Domain Divide: Supervised vs. Zero-Shot Clinical Section Segmentation from MIMIC-III to Obstetrics 

**Authors**: Baris Karacan, Barbara Di Eugenio, Patrick Thornton  

**Link**: [PDF](https://arxiv.org/pdf/2602.17513)  

**Abstract**: Clinical free-text notes contain vital patient information. They are structured into labelled sections; recognizing these sections has been shown to support clinical decision-making and downstream NLP tasks. In this paper, we advance clinical section segmentation through three key contributions. First, we curate a new de-identified, section-labeled obstetrics notes dataset, to supplement the medical domains covered in public corpora such as MIMIC-III, on which most existing segmentation approaches are trained. Second, we systematically evaluate transformer-based supervised models for section segmentation on a curated subset of MIMIC-III (in-domain), and on the new obstetrics dataset (out-of-domain). Third, we conduct the first head-to-head comparison of supervised models for medical section segmentation with zero-shot large language models. Our results show that while supervised models perform strongly in-domain, their performance drops substantially out-of-domain. In contrast, zero-shot models demonstrate robust out-of-domain adaptability once hallucinated section headers are corrected. These findings underscore the importance of developing domain-specific clinical resources and highlight zero-shot segmentation as a promising direction for applying healthcare NLP beyond well-studied corpora, as long as hallucinations are appropriately managed. 

---
# Small LLMs for Medical NLP: a Systematic Analysis of Few-Shot, Constraint Decoding, Fine-Tuning and Continual Pre-Training in Italian 

**Authors**: Pietro Ferrazzi, Mattia Franzin, Alberto Lavelli, Bernardo Magnini  

**Link**: [PDF](https://arxiv.org/pdf/2602.17475)  

**Abstract**: Large Language Models (LLMs) consistently excel in diverse medical Natural Language Processing (NLP) tasks, yet their substantial computational requirements often limit deployment in real-world healthcare settings. In this work, we investigate whether "small" LLMs (around one billion parameters) can effectively perform medical tasks while maintaining competitive accuracy. We evaluate models from three major families-Llama-3, Gemma-3, and Qwen3-across 20 clinical NLP tasks among Named Entity Recognition, Relation Extraction, Case Report Form Filling, Question Answering, and Argument Mining. We systematically compare a range of adaptation strategies, both at inference time (few-shot prompting, constraint decoding) and at training time (supervised fine-tuning, continual pretraining). Fine-tuning emerges as the most effective approach, while the combination of few-shot prompting and constraint decoding offers strong lower-resource alternatives. Our results show that small LLMs can match or even surpass larger baselines, with our best configuration based on Qwen3-1.7B achieving an average score +9.2 points higher than Qwen3-32B. We release a comprehensive collection of all the publicly available Italian medical datasets for NLP tasks, together with our top-performing models. Furthermore, we release an Italian dataset of 126M words from the Emergency Department of an Italian Hospital, and 175M words from various sources that we used for continual pre-training. 

---
# Auditing Reciprocal Sentiment Alignment: Inversion Risk, Dialect Representation and Intent Misalignment in Transformers 

**Authors**: Nusrat Jahan Lia, Shubhashis Roy Dipta  

**Link**: [PDF](https://arxiv.org/pdf/2602.17469)  

**Abstract**: The core theme of bidirectional alignment is ensuring that AI systems accurately understand human intent and that humans can trust AI behavior. However, this loop fractures significantly across language barriers. Our research addresses Cross-Lingual Sentiment Misalignment between Bengali and English by benchmarking four transformer architectures. We reveal severe safety and representational failures in current alignment paradigms. We demonstrate that compressed model (mDistilBERT) exhibits 28.7% "Sentiment Inversion Rate," fundamentally misinterpreting positive user intent as negative (or vice versa). Furthermore, we identify systemic nuances affecting human-AI trust, including "Asymmetric Empathy" where some models systematically dampen and others amplify the affective weight of Bengali text relative to its English counterpart. Finally, we reveal a "Modern Bias" in the regional model (IndicBERT), which shows a 57% increase in alignment error when processing formal (Sadhu) Bengali. We argue that equitable human-AI co-evolution requires pluralistic, culturally grounded alignment that respects language and dialectal diversity over universal compression, which fails to preserve the emotional fidelity required for reciprocal human-AI trust. We recommend that alignment benchmarks incorporate "Affective Stability" metrics that explicitly penalize polarity inversions in low-resource and dialectal contexts. 

---
# PEACE 2.0: Grounded Explanations and Counter-Speech for Combating Hate Expressions 

**Authors**: Greta Damo, Stéphane Petiot, Elena Cabrio, Serena Villata  

**Link**: [PDF](https://arxiv.org/pdf/2602.17467)  

**Abstract**: The increasing volume of hate speech on online platforms poses significant societal challenges. While the Natural Language Processing community has developed effective methods to automatically detect the presence of hate speech, responses to it, called counter-speech, are still an open challenge. We present PEACE 2.0, a novel tool that, besides analysing and explaining why a message is considered hateful or not, also generates a response to it. More specifically, PEACE 2.0 has three main new functionalities: leveraging a Retrieval-Augmented Generation (RAG) pipeline i) to ground HS explanations into evidence and facts, ii) to automatically generate evidence-grounded counter-speech, and iii) exploring the characteristics of counter-speech replies. By integrating these capabilities, PEACE 2.0 enables in-depth analysis and response generation for both explicit and implicit hateful messages. 

---
# Entropy-Based Data Selection for Language Models 

**Authors**: Hongming Li, Yang Liu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.17465)  

**Abstract**: Modern language models (LMs) increasingly require two critical resources: computational resources and data resources. Data selection techniques can effectively reduce the amount of training data required for fine-tuning LMs. However, their effectiveness is closely related to computational resources, which always require a high compute budget. Owing to the resource limitations in practical fine-tuning scenario, we systematically reveal the relationship between data selection and uncertainty estimation of selected data. Although large language models (LLMs) exhibit exceptional capabilities in language understanding and generation, which provide new ways to alleviate data scarcity, evaluating data usability remains a challenging task. This makes efficient data selection indispensable. To mitigate these issues, we propose Entropy-Based Unsupervised Data Selection (EUDS) framework. Empirical experiments on sentiment analysis (SA), topic classification (Topic-CLS), and question answering (Q&A) tasks validate its effectiveness. EUDS establishes a computationally efficient data-filtering mechanism. Theoretical analysis and experimental results confirm the effectiveness of our approach. EUDS significantly reduces computational costs and improves training time efficiency with less data requirement. This provides an innovative solution for the efficient fine-tuning of LMs in the compute-constrained scenarios. 

---
# ABCD: All Biases Come Disguised 

**Authors**: Mateusz Nowak, Xavier Cadet, Peter Chin  

**Link**: [PDF](https://arxiv.org/pdf/2602.17445)  

**Abstract**: Multiple-choice question (MCQ) benchmarks have been a standard evaluation practice for measuring LLMs' ability to reason and answer knowledge-based questions. Through a synthetic NonsenseQA benchmark, we observe that different LLMs exhibit varying degrees of label-position-few-shot-prompt bias, where the model either uses the answer position, the label in front of the answer, the distributions of correct answers present in the few-shot prompt, or a combination of all to answer each MCQ question. We propose a simple bias-reduced evaluation protocol that replaces the labels of each question with uniform, unordered labels and prompts the LLM to use the whole answer presented. With a simple sentence similarity model, we demonstrate improved robustness and lower standard deviation between different permutations of answers with a minimal drop in LLM's performance, exposing the LLM's capabilities under reduced evaluation artifacts, without any help from the prompt examples or the option labels. Across multiple benchmarks and models, this protocol substantially improves the robustness to answer permutations, reducing mean accuracy variance $3\times$ with only a minimal decrease in the mean model's performance. Through ablation studies on various embedding models and similarity functions, we show that the method is more robust than the standard ones. 

---
# AIDG: Evaluating Asymmetry Between Information Extraction and Containment in Multi-Turn Dialogue 

**Authors**: Adib Sakhawat, Fardeen Sadab, Rakin Shahriar  

**Link**: [PDF](https://arxiv.org/pdf/2602.17443)  

**Abstract**: Evaluating the strategic reasoning capabilities of Large Language Models (LLMs) requires moving beyond static benchmarks to dynamic, multi-turn interactions. We introduce AIDG (Adversarial Information Deduction Game), a game-theoretic framework that probes the asymmetry between information extraction (active deduction) and information containment (state maintenance) in dialogue. We propose two complementary tasks: AIDG-I, measuring pragmatic strategy in social deduction, and AIDG-II, measuring constraint satisfaction in a structured "20 Questions" setting. Across 439 games with six frontier LLMs, we observe a clear capability asymmetry: models perform substantially better at containment than deduction, with a 350 ELO advantage on defense;(Cohen's d = 5.47). We identify two bottlenecks driving this gap: (1) Information Dynamics, where confirmation strategies are 7.75x more effective than blind deduction (p < 0.00001), and (2) Constraint Adherence, where instruction-following degrades under conversational load, accounting for 41.3% of deductive failures. These findings suggest that while LLMs excel at local defensive coherence, they struggle with the global state tracking required for strategic inquiry. 

---
# Fine-Grained Uncertainty Quantification for Long-Form Language Model Outputs: A Comparative Study 

**Authors**: Dylan Bouchard, Mohit Singh Chauhan, Viren Bajaj, David Skarbrevik  

**Link**: [PDF](https://arxiv.org/pdf/2602.17431)  

**Abstract**: Uncertainty quantification has emerged as an effective approach to closed-book hallucination detection for LLMs, but existing methods are largely designed for short-form outputs and do not generalize well to long-form generation. We introduce a taxonomy for fine-grained uncertainty quantification in long-form LLM outputs that distinguishes methods by design choices at three stages: response decomposition, unit-level scoring, and response-level aggregation. We formalize several families of consistency-based black-box scorers, providing generalizations and extensions of existing methods. In our experiments across multiple LLMs and datasets, we find 1) claim-response entailment consistently performs better or on par with more complex claim-level scorers, 2) claim-level scoring generally yields better results than sentence-level scoring, and 3) uncertainty-aware decoding is highly effective for improving the factuality of long-form outputs. Our framework clarifies relationships between prior methods, enables apples-to-apples comparisons, and provides practical guidance for selecting components for fine-grained UQ. 

---
# Evaluating Extremely Low-Resource Machine Translation: A Comparative Study of ChrF++ and BLEU Metrics 

**Authors**: Sanjeev Kumar, Preethi Jyothi, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2602.17425)  

**Abstract**: Evaluating machine translation (MT) quality in extremely low-resource language (ELRL) scenarios poses unique challenges, as widely used metrics such as BLEU, effective in high-resource settings, often misrepresent quality in data-scarce contexts. This work presents a comparative analysis of BLEU, an n-gram-based metric, and ChrF++, a character-based metric, for MT evaluation in ELRL settings. We examine how each metric responds to translation artifacts, including hallucinations, repetition, source-text copying, and diacritic (\textit{matra}) variations across three ELRLs: Magahi, Bhojpuri, and Chhattisgarhi, with a focus on outputs from large language models (LLMs) and neural MT (NMT) systems. While recent work often relies solely on ChrF++, our findings show that BLEU, despite its lower absolute scores, provides complementary lexical-precision insights that improve interpretability. 

---
# Diverse Word Choices, Same Reference: Annotating Lexically-Rich Cross-Document Coreference 

**Authors**: Anastasia Zhukova, Felix Hamborg, Karsten Donnay, Norman Meuschke, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2602.17424)  

**Abstract**: Cross-document coreference resolution (CDCR) identifies and links mentions of the same entities and events across related documents, enabling content analysis that aggregates information at the level of discourse participants. However, existing datasets primarily focus on event resolution and employ a narrow definition of coreference, which limits their effectiveness in analyzing diverse and polarized news coverage where wording varies widely. This paper proposes a revised CDCR annotation scheme of the NewsWCL50 dataset, treating coreference chains as discourse elements (DEs) and conceptual units of analysis. The approach accommodates both identity and near-identity relations, e.g., by linking "the caravan" - "asylum seekers" - "those contemplating illegal entry", allowing models to capture lexical diversity and framing variation in media discourse, while maintaining the fine-grained annotation of DEs. We reannotate the NewsWCL50 and a subset of ECB+ using a unified codebook and evaluate the new datasets through lexical diversity metrics and a same-head-lemma baseline. The results show that the reannotated datasets align closely, falling between the original ECB+ and NewsWCL50, thereby supporting balanced and discourse-aware CDCR research in the news domain. 

---
# The Role of the Availability Heuristic in Multiple-Choice Answering Behaviour 

**Authors**: Leonidas Zotos, Hedderik van Rijn, Malvina Nissim  

**Link**: [PDF](https://arxiv.org/pdf/2602.17377)  

**Abstract**: When students are unsure of the correct answer to a multiple-choice question (MCQ), guessing is common practice. The availability heuristic, proposed by A. Tversky and D. Kahneman in 1973, suggests that the ease with which relevant instances come to mind, typically operationalised by the mere frequency of exposure, can offer a mental shortcut for problems in which the test-taker does not know the exact answer. Is simply choosing the option that comes most readily to mind a good strategy for answering MCQs? We propose a computational method of assessing the cognitive availability of MCQ options operationalised by concepts' prevalence in large corpora. The key finding, across three large question sets, is that correct answers, independently of the question stem, are significantly more available than incorrect MCQ options. Specifically, using Wikipedia as the retrieval corpus, we find that always selecting the most available option leads to scores 13.5% to 32.9% above the random-guess baseline. We further find that LLM-generated MCQ options show similar patterns of availability compared to expert-created options, despite the LLMs' frequentist nature and their training on large collections of textual data. Our findings suggest that availability should be considered in current and future work when computationally modelling student behaviour. 

---
# RPDR: A Round-trip Prediction-Based Data Augmentation Framework for Long-Tail Question Answering 

**Authors**: Yiming Zhang, Siyue Zhang, Junbo Zhao, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.17366)  

**Abstract**: Long-tail question answering presents significant challenges for large language models (LLMs) due to their limited ability to acquire and accurately recall less common knowledge. Retrieval-augmented generation (RAG) systems have shown great promise in mitigating this limitation by integrating external retrieval mechanisms. However, dense retrieval models often face the same difficulties when generalizing to rare or niche knowledge. In this study, we introduce RPDR, a novel data augmentation framework that selects high-quality easy-to-learn training data, to enhance dense retrievers. Our approach is built around three core components: synthetic data generation, data selection with Round-Trip prediction to identify easy-to-learn instances, and retriever training with these instances. We evaluate RPDR on two long-tail retrieval benchmarks, PopQA and EntityQuestion, demonstrating substantial improvements over existing retrievers like BM25 and Contriver, especially on extremely long-tail categories. We identify the strengths and limitations of RPDR through detailed human analysis and propose a dynamic routing mechanism to dynamically route queries to specialized retrieval modules to further improve retrieval performance. 

---
# Same Meaning, Different Scores: Lexical and Syntactic Sensitivity in LLM Evaluation 

**Authors**: Bogdan Kostić, Conor Fallon, Julian Risch, Alexander Löser  

**Link**: [PDF](https://arxiv.org/pdf/2602.17316)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has established standardized evaluation benchmarks as the primary instrument for model comparison. Yet, their reliability is increasingly questioned due to sensitivity to shallow variations in input prompts. This paper examines how controlled, truth-conditionally equivalent lexical and syntactic perturbations affect the absolute performance and relative ranking of 23 contemporary LLMs across three benchmarks: MMLU, SQuAD, and AMEGA. We employ two linguistically principled pipelines to generate meaning-preserving variations: one performing synonym substitution for lexical changes, and another using dependency parsing to determine applicable syntactic transformations. Results show that lexical perturbations consistently induce substantial, statistically significant performance degradation across nearly all models and tasks, while syntactic perturbations have more heterogeneous effects, occasionally improving results. Both perturbation types destabilize model leaderboards on complex tasks. Furthermore, model robustness did not consistently scale with model size, revealing strong task dependence. Overall, the findings suggest that LLMs rely more on surface-level lexical patterns than on abstract linguistic competence, underscoring the need for robustness testing as a standard component of LLM evaluation. 

---
# Representation Collapse in Machine Translation Through the Lens of Angular Dispersion 

**Authors**: Evgeniia Tokarchuk, Maya K. Nachesa, Sergey Troshin, Vlad Niculae  

**Link**: [PDF](https://arxiv.org/pdf/2602.17287)  

**Abstract**: Modern neural translation models based on the Transformer architecture are known for their high performance, particularly when trained on high-resource datasets. A standard next-token prediction training strategy, while widely adopted in practice, may lead to overlooked artifacts such as representation collapse. Previous works have shown that this problem is especially pronounced in the representation of the deeper Transformer layers, where it often fails to efficiently utilize the geometric space. Representation collapse is even more evident in end-to-end training of continuous-output neural machine translation, where the trivial solution would be to set all vectors to the same value. In this work, we analyze the dynamics of representation collapse at different levels of discrete and continuous NMT transformers throughout training. We incorporate an existing regularization method based on angular dispersion and demonstrate empirically that it not only mitigates collapse but also improves translation quality. Furthermore, we show that quantized models exhibit similar collapse behavior and that the benefits of regularization are preserved even after quantization. 

---
# Towards Cross-lingual Values Assessment: A Consensus-Pluralism Perspective 

**Authors**: Yukun Chen, Xinyu Zhang, Jialong Tang, Yu Wan, Baosong Yang, Yiming Li, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.17283)  

**Abstract**: While large language models (LLMs) have become pivotal to content safety, current evaluation paradigms primarily focus on detecting explicit harms (e.g., violence or hate speech), neglecting the subtler value dimensions conveyed in digital content. To bridge this gap, we introduce X-Value, a novel Cross-lingual Values Assessment Benchmark designed to evaluate LLMs' ability to assess deep-level values of content from a global perspective. X-Value consists of more than 5,000 QA pairs across 18 languages, systematically organized into 7 core domains grounded in Schwartz's Theory of Basic Human Values and categorized into easy and hard levels for discriminative evaluation. We further propose a unique two-stage annotation framework that first identifies whether an issue falls under global consensus (e.g., human rights) or pluralism (e.g., religion), and subsequently conducts a multi-party evaluation of the latent values embedded within the content. Systematic evaluations on X-Value reveal that current SOTA LLMs exhibit deficiencies in cross-lingual values assessment ($Acc < 77\%$), with significant performance disparities across different languages ($\Delta Acc > 20\%$). This work highlights the urgent need to improve the nuanced, values-aware content assessment capability of LLMs. Our X-Value is available at: this https URL. 

---
# Quantifying and Mitigating Socially Desirable Responding in LLMs: A Desirability-Matched Graded Forced-Choice Psychometric Study 

**Authors**: Kensuke Okada, Yui Furukawa, Kyosuke Bunji  

**Link**: [PDF](https://arxiv.org/pdf/2602.17262)  

**Abstract**: Human self-report questionnaires are increasingly used in NLP to benchmark and audit large language models (LLMs), from persona consistency to safety and bias assessments. Yet these instruments presume honest responding; in evaluative contexts, LLMs can instead gravitate toward socially preferred answers-a form of socially desirable responding (SDR)-biasing questionnaire-derived scores and downstream conclusions. We propose a psychometric framework to quantify and mitigate SDR in questionnaire-based evaluation of LLMs. To quantify SDR, the same inventory is administered under HONEST versus FAKE-GOOD instructions, and SDR is computed as a direction-corrected standardized effect size from item response theory (IRT)-estimated latent scores. This enables comparisons across constructs and response formats, as well as against human instructed-faking benchmarks. For mitigation, we construct a graded forced-choice (GFC) Big Five inventory by selecting 30 cross-domain pairs from an item pool via constrained optimization to match desirability. Across nine instruction-tuned LLMs evaluated on synthetic personas with known target profiles, Likert-style questionnaires show consistently large SDR, whereas desirability-matched GFC substantially attenuates SDR while largely preserving the recovery of the intended persona profiles. These results highlight a model-dependent SDR-recovery trade-off and motivate SDR-aware reporting practices for questionnaire-based benchmarking and auditing of LLMs. 

---
# What Makes a Good Doctor Response? An Analysis on a Romanian Telemedicine Platform 

**Authors**: Adrian Cosma, Cosmin Dumitrache, Emilian Radoi  

**Link**: [PDF](https://arxiv.org/pdf/2602.17194)  

**Abstract**: Text-based telemedicine has become a common mode of care, requiring clinicians to deliver medical advice clearly and effectively in writing. As platforms increasingly rely on patient ratings and feedback, clinicians face growing pressure to maintain satisfaction scores, even though these evaluations often reflect communication quality more than clinical accuracy. We analyse patient satisfaction signals in Romanian text-based telemedicine. Using a sample of 77,334 anonymised patient question--doctor response pairs, we model feedback as a binary outcome, treating thumbs-up responses as positive and grouping negative or absent feedback into the other class. We extract interpretable, predominantly language-agnostic features (e.g., length, structural characteristics, readability proxies), along with Romanian LIWC psycholinguistic features and politeness/hedging markers where available. We train a classifier with a time-based split and perform SHAP-based analyses, which indicate that patient and clinician history features dominate prediction, functioning as strong priors, while characteristics of the response text provide a smaller but, crucially, actionable signal. In subgroup correlation analyses, politeness and hedging are consistently positively associated with patient feedback, whereas lexical diversity shows a negative association. 

---
# The Emergence of Lab-Driven Alignment Signatures: A Psychometric Framework for Auditing Latent Bias and Compounding Risk in Generative AI 

**Authors**: Dusan Bosnjakovic  

**Link**: [PDF](https://arxiv.org/pdf/2602.17127)  

**Abstract**: As Large Language Models (LLMs) transition from standalone chat interfaces to foundational reasoning layers in multi-agent systems and recursive evaluation loops (LLM-as-a-judge), the detection of durable, provider-level behavioral signatures becomes a critical requirement for safety and governance. Traditional benchmarks measure transient task accuracy but fail to capture stable, latent response policies -- the ``prevailing mindsets'' embedded during training and alignment that outlive individual model versions.
This paper introduces a novel auditing framework that utilizes psychometric measurement theory -- specifically latent trait estimation under ordinal uncertainty -- to quantify these tendencies without relying on ground-truth labels. Utilizing forced-choice ordinal vignettes masked by semantically orthogonal decoys and governed by cryptographic permutation-invariance, the research audits nine leading models across dimensions including Optimization Bias, Sycophancy, and Status-Quo Legitimization.
Using Mixed Linear Models (MixedLM) and Intraclass Correlation Coefficient (ICC) analysis, the research identifies that while item-level framing drives high variance, a persistent ``lab signal'' accounts for significant behavioral clustering. These findings demonstrate that in ``locked-in'' provider ecosystems, latent biases are not merely static errors but compounding variables that risk creating recursive ideological echo chambers in multi-layered AI architectures. 

---
# Projective Psychological Assessment of Large Multimodal Models Using Thematic Apperception Tests 

**Authors**: Anton Dzega, Aviad Elyashar, Ortal Slobodin, Odeya Cohen, Rami Puzis  

**Link**: [PDF](https://arxiv.org/pdf/2602.17108)  

**Abstract**: Thematic Apperception Test (TAT) is a psychometrically grounded, multidimensional assessment framework that systematically differentiates between cognitive-representational and affective-relational components of personality-like functioning. This test is a projective psychological framework designed to uncover unconscious aspects of personality. This study examines whether the personality traits of Large Multimodal Models (LMMs) can be assessed through non-language-based modalities, using the Social Cognition and Object Relations Scale - Global (SCORS-G). LMMs are employed in two distinct roles: as subject models (SMs), which generate stories in response to TAT images, and as evaluator models (EMs), who assess these narratives using the SCORS-G framework. Evaluators demonstrated an excellent ability to understand and analyze TAT responses. Their interpretations are highly consistent with those of human experts. Assessment results highlight that all models understand interpersonal dynamics very well and have a good grasp of the concept of self. However, they consistently fail to perceive and regulate aggression. Performance varied systematically across model families, with larger and more recent models consistently outperforming smaller and earlier ones across SCORS-G dimensions. 

---
# BankMathBench: A Benchmark for Numerical Reasoning in Banking Scenarios 

**Authors**: Yunseung Lee, Subin Kim, Youngjun Kwak, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2602.17072)  

**Abstract**: Large language models (LLMs)-based chatbots are increasingly being adopted in the financial domain, particularly in digital banking, to handle customer inquiries about products such as deposits, savings, and loans. However, these models still exhibit low accuracy in core banking computations-including total payout estimation, comparison of products with varying interest rates, and interest calculation under early repayment conditions. Such tasks require multi-step numerical reasoning and contextual understanding of banking products, yet existing LLMs often make systematic errors-misinterpreting product types, applying conditions incorrectly, or failing basic calculations involving exponents and geometric progressions. However, such errors have rarely been captured by existing benchmarks. Mathematical datasets focus on fundamental math problems, whereas financial benchmarks primarily target financial documents, leaving everyday banking scenarios underexplored. To address this limitation, we propose BankMathBench, a domain-specific dataset that reflects realistic banking tasks. BankMathBench is organized in three levels of difficulty-basic, intermediate, and advanced-corresponding to single-product reasoning, multi-product comparison, and multi-condition scenarios, respectively. When trained on BankMathBench, open-source LLMs exhibited notable improvements in both formula generation and numerical reasoning accuracy, demonstrating the dataset's effectiveness in enhancing domain-specific reasoning. With tool-augmented fine-tuning, the models achieved average accuracy increases of 57.6%p (basic), 75.1%p (intermediate), and 62.9%p (advanced), representing significant gains over zero-shot baselines. These findings highlight BankMathBench as a reliable benchmark for evaluating and advancing LLMs' numerical reasoning in real-world banking scenarios. 

---
# ALPS: A Diagnostic Challenge Set for Arabic Linguistic & Pragmatic Reasoning 

**Authors**: Hussein S. Al-Olimat, Ahmad Alshareef  

**Link**: [PDF](https://arxiv.org/pdf/2602.17054)  

**Abstract**: While recent Arabic NLP benchmarks focus on scale, they often rely on synthetic or translated data which may benefit from deeper linguistic verification. We introduce ALPS (Arabic Linguistic & Pragmatic Suite), a native, expert-curated diagnostic challenge set probing Deep Semantics and Pragmatics, capabilities that complement specialized large-scale benchmarks. While broad-coverage benchmarks prioritize scale and multi-task coverage, ALPS targets the depth of linguistic understanding through 531 rigorously crafted questions across 15 tasks and 47 subtasks. We developed the dataset with deep expertise in Arabic linguistics, guaranteeing cultural authenticity and eliminating translation artifacts. Evaluating 23 diverse models (commercial, open-source, and Arabic-native) against a single-pass human performance (avg. 84.6% accuracy) and an expert-adjudicated oracle (99.2%), we reveal a critical dissociation: models achieve high fluency but fail on fundamental morpho-syntactic dependencies, with elevated error rates on morpho-syntactic dependencies (36.5% across diacritics-reliant tasks) compared to compositional semantics. While top commercial models (Gemini-3-flash at 94.2%) surpass the average single human, a substantial gap persists between commercial giants and Arabic-native models, with the best Arabic-specific model (Jais-2-70B at 83.6%) approaching but not matching human performance. 

---
# Evaluating Cross-Lingual Classification Approaches Enabling Topic Discovery for Multilingual Social Media Data 

**Authors**: Deepak Uniyal, Md Abul Bashar, Richi Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2602.17051)  

**Abstract**: Analysing multilingual social media discourse remains a major challenge in natural language processing, particularly when large-scale public debates span across diverse languages. This study investigates how different approaches for cross-lingual text classification can support reliable analysis of global conversations. Using hydrogen energy as a case study, we analyse a decade-long dataset of over nine million tweets in English, Japanese, Hindi, and Korean (2013--2022) for topic discovery. The online keyword-driven data collection results in a significant amount of irrelevant content. We explore four approaches to filter relevant content: (1) translating English annotated data into target languages for building language-specific models for each target language, (2) translating unlabelled data appearing from all languages into English for creating a single model based on English annotations, (3) applying English fine-tuned multilingual transformers directly to each target language data, and (4) a hybrid strategy that combines translated annotations with multilingual training. Each approach is evaluated for its ability to filter hydrogen-related tweets from noisy keyword-based collections. Subsequently, topic modeling is performed to extract dominant themes within the relevant subsets. The results highlight key trade-offs between translation and multilingual approaches, offering actionable insights into optimising cross-lingual pipelines for large-scale social media analysis. 

---
# Large Language Models Persuade Without Planning Theory of Mind 

**Authors**: Jared Moore, Rasmus Overmark, Ned Cooper, Beba Cibralic, Nick Haber, Cameron R. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2602.17045)  

**Abstract**: A growing body of work attempts to evaluate the theory of mind (ToM) abilities of humans and large language models (LLMs) using static, non-interactive question-and-answer benchmarks. However, theoretical work in the field suggests that first-personal interaction is a crucial part of ToM and that such predictive, spectatorial tasks may fail to evaluate it. We address this gap with a novel ToM task that requires an agent to persuade a target to choose one of three policy proposals by strategically revealing information. Success depends on a persuader's sensitivity to a given target's knowledge states (what the target knows about the policies) and motivational states (how much the target values different outcomes). We varied whether these states were Revealed to persuaders or Hidden, in which case persuaders had to inquire about or infer them. In Experiment 1, participants persuaded a bot programmed to make only rational inferences. LLMs excelled in the Revealed condition but performed below chance in the Hidden condition, suggesting difficulty with the multi-step planning required to elicit and use mental state information. Humans performed moderately well in both conditions, indicating an ability to engage such planning. In Experiment 2, where a human target role-played the bot, and in Experiment 3, where we measured whether human targets' real beliefs changed, LLMs outperformed human persuaders across all conditions. These results suggest that effective persuasion can occur without explicit ToM reasoning (e.g., through rhetorical strategies) and that LLMs excel at this form of persuasion. Overall, our results caution against attributing human-like ToM to LLMs while highlighting LLMs' potential to influence people's beliefs and behavior. 

---
# ReIn: Conversational Error Recovery with Reasoning Inception 

**Authors**: Takyoung Kim, Jinseok Nam, Chandrayee Basu, Xing Fan, Chengyuan Ma, Heng Ji, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2602.17022)  

**Abstract**: Conversational agents powered by large language models (LLMs) with tool integration achieve strong performance on fixed task-oriented dialogue datasets but remain vulnerable to unanticipated, user-induced errors. Rather than focusing on error prevention, this work focuses on error recovery, which necessitates the accurate diagnosis of erroneous dialogue contexts and execution of proper recovery plans. Under realistic constraints precluding model fine-tuning or prompt modification due to significant cost and time requirements, we explore whether agents can recover from contextually flawed interactions and how their behavior can be adapted without altering model parameters and prompts. To this end, we propose Reasoning Inception (ReIn), a test-time intervention method that plants an initial reasoning into the agent's decision-making process. Specifically, an external inception module identifies predefined errors within the dialogue context and generates recovery plans, which are subsequently integrated into the agent's internal reasoning process to guide corrective actions, without modifying its parameters or system prompts. We evaluate ReIn by systematically simulating conversational failure scenarios that directly hinder successful completion of user goals: user's ambiguous and unsupported requests. Across diverse combinations of agent models and inception modules, ReIn substantially improves task success and generalizes to unseen error types. Moreover, it consistently outperforms explicit prompt-modification approaches, underscoring its utility as an efficient, on-the-fly method. In-depth analysis of its operational mechanism, particularly in relation to instruction hierarchy, indicates that jointly defining recovery tools with ReIn can serve as a safe and effective strategy for improving the resilience of conversational agents without modifying the backbone models or system prompts. 

---
# Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History 

**Authors**: Serin Kim, Sangam Lee, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.17003)  

**Abstract**: Large language models have advanced web agents, yet current agents lack personalization capabilities. Since users rarely specify every detail of their intent, practical web agents must be able to interpret ambiguous queries by inferring user preferences and contexts. To address this challenge, we present Persona2Web, the first benchmark for evaluating personalized web agents on the real open web, built upon the clarify-to-personalize principle, which requires agents to resolve ambiguity based on user history rather than relying on explicit instructions. Persona2Web consists of: (1) user histories that reveal preferences implicitly over long time spans, (2) ambiguous queries that require agents to infer implicit user preferences, and (3) a reasoning-aware evaluation framework that enables fine-grained assessment of personalization. We conduct extensive experiments across various agent architectures, backbone models, history access schemes, and queries with varying ambiguity levels, revealing key challenges in personalized web agent behavior. For reproducibility, our codes and datasets are publicly available at this https URL. 

---
# Eigenmood Space: Uncertainty-Aware Spectral Graph Analysis of Psychological Patterns in Classical Persian Poetry 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh, Mohammadali Keshtparvar  

**Link**: [PDF](https://arxiv.org/pdf/2602.16959)  

**Abstract**: Classical Persian poetry is a historically sustained archive in which affective life is expressed through metaphor, intertextual convention, and rhetorical indirection. These properties make close reading indispensable while limiting reproducible comparison at scale. We present an uncertainty-aware computational framework for poet-level psychological analysis based on large-scale automatic multi-label annotation. Each verse is associated with a set of psychological concepts, per-label confidence scores, and an abstention flag that signals insufficient evidence. We aggregate confidence-weighted evidence into a Poet $\times$ Concept matrix, interpret each poet as a probability distribution over concepts, and quantify poetic individuality as divergence from a corpus baseline using Jensen--Shannon divergence and Kullback--Leibler divergence. To capture relational structure beyond marginals, we build a confidence-weighted co-occurrence graph over concepts and define an Eigenmood embedding through Laplacian spectral decomposition. On a corpus of 61{,}573 verses across 10 poets, 22.2\% of verses are abstained, underscoring the analytical importance of uncertainty. We further report sensitivity analysis under confidence thresholding, selection-bias diagnostics that treat abstention as a category, and a distant-to-close workflow that retrieves verse-level exemplars along Eigenmood axes. The resulting framework supports scalable, auditable digital-humanities analysis while preserving interpretive caution by propagating uncertainty from verse-level evidence to poet-level inference. 

---
# When Semantic Overlap Is Not Enough: Cross-Lingual Euphemism Transfer Between Turkish and English 

**Authors**: Hasan Can Biyik, Libby Barak, Jing Peng, Anna Feldman  

**Link**: [PDF](https://arxiv.org/pdf/2602.16957)  

**Abstract**: Euphemisms substitute socially sensitive expressions, often softening or reframing meaning, and their reliance on cultural and pragmatic context complicates modeling across languages. In this study, we investigate how cross-lingual equivalence influences transfer in multilingual euphemism detection. We categorize Potentially Euphemistic Terms (PETs) in Turkish and English into Overlapping (OPETs) and Non-Overlapping (NOPETs) subsets based on their functional, pragmatic, and semantic alignment. Our findings reveal a transfer asymmetry: semantic overlap is insufficient to guarantee positive transfer, particularly in low-resource Turkish-to-English direction, where performance can degrade even for overlapping euphemisms, and in some cases, improve under NOPET-based training. Differences in label distribution help explain these counterintuitive results. Category-level analysis suggests that transfer may be influenced by domain-specific alignment, though evidence is limited by sparsity. 

---
# ConvApparel: A Benchmark Dataset and Validation Framework for User Simulators in Conversational Recommenders 

**Authors**: Ofer Meshi, Krisztian Balog, Sally Goldman, Avi Caciularu, Guy Tennenholtz, Jihwan Jeong, Amir Globerson, Craig Boutilier  

**Link**: [PDF](https://arxiv.org/pdf/2602.16938)  

**Abstract**: The promise of LLM-based user simulators to improve conversational AI is hindered by a critical "realism gap," leading to systems that are optimized for simulated interactions, but may fail to perform well in the real world. We introduce ConvApparel, a new dataset of human-AI conversations designed to address this gap. Its unique dual-agent data collection protocol -- using both "good" and "bad" recommenders -- enables counterfactual validation by capturing a wide spectrum of user experiences, enriched with first-person annotations of user satisfaction. We propose a comprehensive validation framework that combines statistical alignment, a human-likeness score, and counterfactual validation to test for generalization. Our experiments reveal a significant realism gap across all simulators. However, the framework also shows that data-driven simulators outperform a prompted baseline, particularly in counterfactual validation where they adapt more realistically to unseen behaviors, suggesting they embody more robust, if imperfect, user models. 

---
# A Conceptual Hybrid Framework for Post-Quantum Security: Integrating BB84 QKD, AES, and Bio-inspired Mechanisms 

**Authors**: Md. Ismiel Hossen Abir  

**Link**: [PDF](https://arxiv.org/pdf/2602.16922)  

**Abstract**: Quantum computing is a significant risk to classical cryptographic, especially RSA, which depends on the difficulty of factoring large numbers. Classical factorization methods, such as Trial Division and Pollard's Rho, are inefficient for large keys, while Shor's quantum algorithm can break RSA efficiently in polynomial time. This research studies RSA's vulnerabilities under both classical and quantum attacks and designs a hybrid security framework to ensure data protection in the post-quantum era. The conceptual framework combines AES encryption for classical security, BB84 Quantum Key Distribution (QKD) for secure key exchange with eavesdropping detection, quantum state comparison for lightweight authentication, and a bio-inspired immune system for adaptive threat detection. RSA is vulnerable to Shor's algorithm, BB84 achieves full key agreement in ideal conditions, and it detects eavesdropping with high accuracy. The conceptual model includes both classical and quantum security methods, providing a scalable and adaptive solution for Post-Quantum encryption data protection. This work primarily proposes a conceptual framework. Detailed implementation, security proofs, and extensive experimental validation are considered future work. 

---
# Meenz bleibt Meenz, but Large Language Models Do Not Speak Its Dialect 

**Authors**: Minh Duc Bui, Manuel Mager, Peter Herbert Kann, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2602.16852)  

**Abstract**: Meenzerisch, the dialect spoken in the German city of Mainz, is also the traditional language of the Mainz carnival, a yearly celebration well known throughout Germany. However, Meenzerisch is on the verge of dying out-a fate it shares with many other German dialects. Natural language processing (NLP) has the potential to help with the preservation and revival efforts of languages and dialects. However, so far no NLP research has looked at Meenzerisch. This work presents the first research in the field of NLP that is explicitly focused on the dialect of Mainz. We introduce a digital dictionary-an NLP-ready dataset derived from an existing resource (Schramm, 1966)-to support researchers in modeling and benchmarking the language. It contains 2,351 words in the dialect paired with their meanings described in Standard German. We then use this dataset to answer the following research questions: (1) Can state-of-the-art large language models (LLMs) generate definitions for dialect words? (2) Can LLMs generate words in Meenzerisch, given their definitions? Our experiments show that LLMs can do neither: the best model for definitions reaches only 6.27% accuracy and the best word generation model's accuracy is 1.51%. We then conduct two additional experiments in order to see if accuracy is improved by few-shot learning and by extracting rules from the training set, which are then passed to the LLM. While those approaches are able to improve the results, accuracy remains below 10%. This highlights that additional resources and an intensification of research efforts focused on German dialects are desperately needed. 

---
# BanglaSummEval: Reference-Free Factual Consistency Evaluation for Bangla Summarization 

**Authors**: Ahmed Rafid, Rumman Adib, Fariya Ahmed, Ajwad Abrar, Mohammed Saidul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2602.16843)  

**Abstract**: Evaluating factual consistency is essential for reliable text summarization, particularly in high-stakes domains such as healthcare and news. However, most existing evaluation metrics overlook Bangla, a widely spoken yet under-resourced language, and often depend on reference summaries. We introduce BanglaSummEval, a reference-free, question-answering-based framework for evaluating factual consistency in Bangla summarization. The proposed method assesses both factual accuracy and content coverage through automatically generated questions and answers derived from the source document and the summary. A single multilingual instruction-tuned language model handles question generation, question answering, candidate answer extraction, and question importance weighting. This unified design reduces system complexity and computational cost. To capture semantic consistency beyond surface-level overlap, we use BERTScore-Recall for answer comparison. We validate BanglaSummEval on 300 human-written summaries from educational and medical domains, demonstrating strong correlation with expert human judgments (Pearson's $r = 0.694$, Spearman's $\rho = 0.763$). By providing interpretable, step-wise diagnostics alongside reliable evaluation scores, BanglaSummEval offers a practical and transparent solution for factual consistency evaluation in low-resource language settings. 

---
# Claim Automation using Large Language Model 

**Authors**: Zhengda Mo, Zhiyu Quan, Eli O'Donohue, Kaiwen Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2602.16836)  

**Abstract**: While Large Language Models (LLMs) have achieved strong performance on general-purpose language tasks, their deployment in regulated and data-sensitive domains, including insurance, remains limited. Leveraging millions of historical warranty claims, we propose a locally deployed governance-aware language modeling component that generates structured corrective-action recommendations from unstructured claim narratives. We fine-tune pretrained LLMs using Low-Rank Adaptation (LoRA), scoping the model to an initial decision module within the claim processing pipeline to speed up claim adjusters' decisions. We assess this module using a multi-dimensional evaluation framework that combines automated semantic similarity metrics with human evaluation, enabling a rigorous examination of both practical utility and predictive accuracy. Our results show that domain-specific fine-tuning substantially outperforms commercial general-purpose and prompt-based LLMs, with approximately 80% of the evaluated cases achieving near-identical matches to ground-truth corrective actions. Overall, this study provides both theoretical and empirical evidence to prove that domain-adaptive fine-tuning can align model output distributions more closely with real-world operational data, demonstrating its promise as a reliable and governable building block for insurance applications. 

---
# One-step Language Modeling via Continuous Denoising 

**Authors**: Chanhyuk Lee, Jaehoon Yoo, Manan Agarwal, Sheel Shah, Jerry Huang, Aditi Raghunathan, Seunghoon Hong, Nicholas M. Boffi, Jinwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.16813)  

**Abstract**: Language models based on discrete diffusion have attracted widespread interest for their potential to provide faster generation than autoregressive models. In practice, however, they exhibit a sharp degradation of sample quality in the few-step regime, failing to realize this promise. Here we show that language models leveraging flow-based continuous denoising can outperform discrete diffusion in both quality and speed. By revisiting the fundamentals of flows over discrete modalities, we build a flow-based language model (FLM) that performs Euclidean denoising over one-hot token encodings. We show that the model can be trained by predicting the clean data via a cross entropy objective, where we introduce a simple time reparameterization that greatly improves training stability and generation quality. By distilling FLM into its associated flow map, we obtain a distilled flow map language model (FMLM) capable of few-step generation. On the LM1B and OWT language datasets, FLM attains generation quality matching state-of-the-art discrete diffusion models. With FMLM, our approach outperforms recent few-step language models across the board, with one-step generation exceeding their 8-step quality. Our work calls into question the widely held hypothesis that discrete diffusion processes are necessary for generative modeling over discrete modalities, and paves the way toward accelerated flow-based language modeling at scale. Code is available at this https URL. 

---
# Evaluating Monolingual and Multilingual Large Language Models for Greek Question Answering: The DemosQA Benchmark 

**Authors**: Charalampos Mastrokostas, Nikolaos Giarelis, Nikos Karacapilidis  

**Link**: [PDF](https://arxiv.org/pdf/2602.16811)  

**Abstract**: Recent advancements in Natural Language Processing and Deep Learning have enabled the development of Large Language Models (LLMs), which have significantly advanced the state-of-the-art across a wide range of tasks, including Question Answering (QA). Despite these advancements, research on LLMs has primarily targeted high-resourced languages (e.g., English), and only recently has attention shifted toward multilingual models. However, these models demonstrate a training data bias towards a small number of popular languages or rely on transfer learning from high- to under-resourced languages; this may lead to a misrepresentation of social, cultural, and historical aspects. To address this challenge, monolingual LLMs have been developed for under-resourced languages; however, their effectiveness remains less studied when compared to multilingual counterparts on language-specific tasks. In this study, we address this research gap in Greek QA by contributing: (i) DemosQA, a novel dataset, which is constructed using social media user questions and community-reviewed answers to better capture the Greek social and cultural zeitgeist; (ii) a memory-efficient LLM evaluation framework adaptable to diverse QA datasets and languages; and (iii) an extensive evaluation of 11 monolingual and multilingual LLMs on 6 human-curated Greek QA datasets using 3 different prompting strategies. We release our code and data to facilitate reproducibility. 

---
# References Improve LLM Alignment in Non-Verifiable Domains 

**Authors**: Kejian Shi, Yixin Liu, Peifeng Wang, Alexander R. Fabbri, Shafiq Joty, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2602.16802)  

**Abstract**: While Reinforcement Learning with Verifiable Rewards (RLVR) has shown strong effectiveness in reasoning tasks, it cannot be directly applied to non-verifiable domains lacking ground-truth verifiers, such as LLM alignment. In this work, we investigate whether reference-guided LLM-evaluators can bridge this gap by serving as soft "verifiers". First, we design evaluation protocols that enhance LLM-based evaluators for LLM alignment using reference outputs. Through comprehensive experiments, we show that a reference-guided approach substantially improves the accuracy of less capable LLM-judges using references from frontier models; stronger LLM-judges can also be enhanced by high-quality (i.e., human-written) references. Building on these improved judges, we demonstrate the utility of high-quality references in alignment tuning, where LLMs guided with references are used as judges to self-improve. We show that reference-guided self-improvement yields clear gains over both direct SFT on reference outputs and self-improvement with reference-free judges, achieving performance comparable to training with ArmoRM, a strong finetuned reward model. Specifically, our method achieves 73.1% and 58.7% on AlpacaEval and Arena-Hard with Llama-3-8B-Instruct, and 70.0% and 74.1% with Qwen2.5-7B, corresponding to average absolute gains of +20.2 / +17.1 points over SFT distillation and +5.3 / +3.6 points over reference-free self-improvement on AlpacaEval / Arena-Hard. These results highlight the potential of using reference-guided LLM-evaluators to enable effective LLM post-training in non-verifiable domains. 

---
# CLEF HIPE-2026: Evaluating Accurate and Efficient Person-Place Relation Extraction from Multilingual Historical Texts 

**Authors**: Juri Opitz, Corina Raclé, Emanuela Boros, Andrianos Michail, Matteo Romanello, Maud Ehrmann, Simon Clematide  

**Link**: [PDF](https://arxiv.org/pdf/2602.17663)  

**Abstract**: HIPE-2026 is a CLEF evaluation lab dedicated to person-place relation extraction from noisy, multilingual historical texts. Building on the HIPE-2020 and HIPE-2022 campaigns, it extends the series toward semantic relation extraction by targeting the task of identifying person--place associations in multiple languages and time periods. Systems are asked to classify relations of two types - $at$ ("Has the person ever been at this place?") and $isAt$ ("Is the person located at this place around publication time?") - requiring reasoning over temporal and geographical cues. The lab introduces a three-fold evaluation profile that jointly assesses accuracy, computational efficiency, and domain generalization. By linking relation extraction to large-scale historical data processing, HIPE-2026 aims to support downstream applications in knowledge-graph construction, historical biography reconstruction, and spatial analysis in digital humanities. 

---
# Pushing the Frontier of Black-Box LVLM Attacks via Fine-Grained Detail Targeting 

**Authors**: Xiaohan Zhao, Zhaoyi Li, Yaxin Luo, Jiacheng Cui, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.17645)  

**Abstract**: Black-box adversarial attacks on Large Vision-Language Models (LVLMs) are challenging due to missing gradients and complex multimodal boundaries. While prior state-of-the-art transfer-based approaches like M-Attack perform well using local crop-level matching between source and target images, we find this induces high-variance, nearly orthogonal gradients across iterations, violating coherent local alignment and destabilizing optimization. We attribute this to (i) ViT translation sensitivity that yields spike-like gradients and (ii) structural asymmetry between source and target crops. We reformulate local matching as an asymmetric expectation over source transformations and target semantics, and build a gradient-denoising upgrade to M-Attack. On the source side, Multi-Crop Alignment (MCA) averages gradients from multiple independently sampled local views per iteration to reduce variance. On the target side, Auxiliary Target Alignment (ATA) replaces aggressive target augmentation with a small auxiliary set from a semantically correlated distribution, producing a smoother, lower-variance target manifold. We further reinterpret momentum as Patch Momentum, replaying historical crop gradients; combined with a refined patch-size ensemble (PE+), this strengthens transferable directions. Together these modules form M-Attack-V2, a simple, modular enhancement over M-Attack that substantially improves transfer-based black-box attacks on frontier LVLMs: boosting success rates on Claude-4.0 from 8% to 30%, Gemini-2.5-Pro from 83% to 97%, and GPT-5 from 98% to 100%, outperforming prior black-box LVLM attacks. Code and data are publicly available at: this https URL. 

---
# KLong: Training LLM Agent for Extremely Long-horizon Tasks 

**Authors**: Yue Liu, Zhiyuan Hu, Flood Sung, Jiaheng Zhang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2602.17547)  

**Abstract**: This paper introduces KLong, an open-source LLM agent trained to solve extremely long-horizon tasks. The principle is to first cold-start the model via trajectory-splitting SFT, then scale it via progressive RL training. Specifically, we first activate basic agentic abilities of a base model with a comprehensive SFT recipe. Then, we introduce Research-Factory, an automated pipeline that generates high-quality training data by collecting research papers and constructing evaluation rubrics. Using this pipeline, we build thousands of long-horizon trajectories distilled from Claude 4.5 Sonnet (Thinking). To train with these extremely long trajectories, we propose a new trajectory-splitting SFT, which preserves early context, progressively truncates later context, and maintains overlap between sub-trajectories. In addition, to further improve long-horizon task-solving capability, we propose a novel progressive RL, which schedules training into multiple stages with progressively extended timeouts. Experiments demonstrate the superiority and generalization of KLong, as shown in Figure 1. Notably, our proposed KLong (106B) surpasses Kimi K2 Thinking (1T) by 11.28% on PaperBench, and the performance improvement generalizes to other coding benchmarks like SWE-bench Verified and MLE-bench. 

---
# Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability 

**Authors**: Shashank Aggarwal, Ram Vikas Mishra, Amit Awekar  

**Link**: [PDF](https://arxiv.org/pdf/2602.17544)  

**Abstract**: In multi-agent IR pipelines for tasks such as search and ranking, LLM-based agents exchange intermediate reasoning in terms of Chain-of-Thought (CoT) with each other. Current CoT evaluation narrowly focuses on target task accuracy. However, this metric fails to assess the quality or utility of the reasoning process itself. To address this limitation, we introduce two novel measures: reusability and verifiability. We decouple CoT generation from execution using a Thinker-Executor framework. Reusability measures how easily an Executor can reuse the Thinker's CoT. Verifiability measures how frequently an Executor can match the Thinker's answer using the CoT. We evaluated four Thinker models against a committee of ten Executor models across five benchmarks. Our results reveal that reusability and verifiability do not correlate with standard accuracy, exposing a blind spot in current accuracy-based leaderboards for reasoning capability. Surprisingly, we find that CoTs from specialized reasoning models are not consistently more reusable or verifiable than those from general-purpose LLMs like Llama and Gemma. 

---
# The Anxiety of Influence: Bloom Filters in Transformer Attention Heads 

**Authors**: Peter Balogh  

**Link**: [PDF](https://arxiv.org/pdf/2602.17526)  

**Abstract**: Some transformer attention heads appear to function as membership testers, dedicating themselves to answering the question "has this token appeared before in the context?" We identify these heads across four language models (GPT-2 small, medium, and large; Pythia-160M) and show that they form a spectrum of membership-testing strategies. Two heads (L0H1 and L0H5 in GPT-2 small) function as high-precision membership filters with false positive rates of 0-4\% even at 180 unique context tokens -- well above the $d_\text{head} = 64$ bit capacity of a classical Bloom filter. A third head (L1H11) shows the classic Bloom filter capacity curve: its false positive rate follows the theoretical formula $p \approx (1 - e^{-kn/m})^k$ with $R^2 = 1.0$ and fitted capacity $m \approx 5$ bits, saturating by $n \approx 20$ unique tokens. A fourth head initially identified as a Bloom filter (L3H0) was reclassified as a general prefix-attention head after confound controls revealed its apparent capacity curve was a sequence-length artifact. Together, the three genuine membership-testing heads form a multi-resolution system concentrated in early layers (0-1), taxonomically distinct from induction and previous-token heads, with false positive rates that decay monotonically with embedding distance -- consistent with distance-sensitive Bloom filters. These heads generalize broadly: they respond to any repeated token type, not just repeated names, with 43\% higher generalization than duplicate-token-only heads. Ablation reveals these heads contribute to both repeated and novel token processing, indicating that membership testing coexists with broader computational roles. The reclassification of L3H0 through confound controls strengthens rather than weakens the case: the surviving heads withstand the scrutiny that eliminated a false positive in our own analysis. 

---
# What Do LLMs Associate with Your Name? A Human-Centered Black-Box Audit of Personal Data 

**Authors**: Dimitri Staufer, Kirsten Morehouse  

**Link**: [PDF](https://arxiv.org/pdf/2602.17483)  

**Abstract**: Large language models (LLMs), and conversational agents based on them, are exposed to personal data (PD) during pre-training and during user interactions. Prior work shows that PD can resurface, yet users lack insight into how strongly models associate specific information to their identity. We audit PD across eight LLMs (3 open-source; 5 API-based, including GPT-4o), introduce LMP2 (Language Model Privacy Probe), a human-centered, privacy-preserving audit tool refined through two formative studies (N=20), and run two studies with EU residents to capture (i) intuitions about LLM-generated PD (N1=155) and (ii) reactions to tool output (N2=303). We show empirically that models confidently generate multiple PD categories for well-known individuals. For everyday users, GPT-4o generates 11 features with 60% or more accuracy (e.g., gender, hair color, languages). Finally, 72% of participants sought control over model-generated associations with their name, raising questions about what counts as PD and whether data privacy rights should extend to LLMs. 

---
# DAVE: A Policy-Enforcing LLM Spokesperson for Secure Multi-Document Data Sharing 

**Authors**: René Brinkhege, Prahlad Menon  

**Link**: [PDF](https://arxiv.org/pdf/2602.17413)  

**Abstract**: In current inter-organizational data spaces, usage policies are enforced mainly at the asset level: a whole document or dataset is either shared or withheld. When only parts of a document are sensitive, providers who want to avoid leaking protected information typically must manually redact documents before sharing them, which is costly, coarse-grained, and hard to maintain as policies or partners change. We present DAVE, a usage policy-enforcing LLM spokesperson that answers questions over private documents on behalf of a data provider. Instead of releasing documents, the provider exposes a natural language interface whose responses are constrained by machine-readable usage policies. We formalize policy-violating information disclosure in this setting, drawing on usage control and information flow security, and introduce virtual redaction: suppressing sensitive information at query time without modifying source documents. We describe an architecture for integrating such a spokesperson with Eclipse Dataspace Components and ODRL-style policies, and outline an initial provider-side integration prototype in which QA requests are routed through a spokesperson service instead of triggering raw document transfer. Our contribution is primarily architectural: we do not yet implement or empirically evaluate the full enforcement pipeline. We therefore outline an evaluation methodology to assess security, utility, and performance trade-offs under benign and adversarial querying as a basis for future empirical work on systematically governed LLM access to multi-party data spaces. 

---
# WebFAQ 2.0: A Multilingual QA Dataset with Mined Hard Negatives for Dense Retrieval 

**Authors**: Michael Dinzinger, Laura Caspari, Ali Salman, Irvin Topi, Jelena Mitrović, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2602.17327)  

**Abstract**: We introduce WebFAQ 2.0, a new version of the WebFAQ dataset, containing 198 million FAQ-based natural question-answer pairs across 108 languages. Compared to the previous version, it significantly expands multilingual coverage and the number of bilingual aligned QA pairs to over 14.3M, making it the largest FAQ-based resource. Unlike the original release, WebFAQ 2.0 uses a novel data collection strategy that directly crawls and extracts relevant web content, resulting in a substantially more diverse and multilingual dataset with richer context through page titles and descriptions. In response to community feedback, we also release a hard negatives dataset for training dense retrievers, with 1.25M queries across 20 languages. These hard negatives were mined using a two-stage retrieval pipeline and include cross-encoder scores for 200 negatives per query. We further show how this resource enables two primary fine-tuning strategies for dense retrievers: Contrastive Learning with MultipleNegativesRanking loss, and Knowledge Distillation with MarginMSE loss. WebFAQ 2.0 is not a static resource but part of a long-term effort. Since late 2025, structured FAQs are being regularly released through the Open Web Index, enabling continuous expansion and refinement. We publish the datasets and training scripts to facilitate further research in multilingual and cross-lingual IR. The dataset itself and all related resources are publicly available on GitHub and HuggingFace. 

---
# ArXiv-to-Model: A Practical Study of Scientific LM Training 

**Authors**: Anuj Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2602.17288)  

**Abstract**: While frontier large language models demonstrate strong reasoning and mathematical capabilities, the practical process of training domain-specialized scientific language models from raw sources remains under-documented. In this work, we present a detailed case study of training a 1.36B-parameter scientific language model directly from raw arXiv LaTeX sources spanning mathematics, computer science, and theoretical physics. We describe an end-to-end pipeline covering metadata filtering, archive validation, LaTeX extraction, text normalization, domain-aware tokenization, and dense transformer training under constrained compute (2xA100 GPUs). Through 24 experimental runs, we analyze training stability, scaling behavior, data yield losses, and infrastructure bottlenecks. Our findings highlight how preprocessing decisions significantly affect usable token volume, how tokenization impacts symbolic stability, and how storage and I/O constraints can rival compute as limiting factors. We further analyze convergence dynamics and show stable training behavior in a data-rich regime (52B pretraining tokens). Rather than proposing a novel architecture, this work provides an engineering-grounded, transparent account of training a small scientific language model from scratch. We hope these insights support researchers operating under moderate compute budgets who seek to build domain-specialized models. 

---
# Mechanistic Interpretability of Cognitive Complexity in LLMs via Linear Probing using Bloom's Taxonomy 

**Authors**: Bianca Raimondi, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2602.17229)  

**Abstract**: The black-box nature of Large Language Models necessitates novel evaluation frameworks that transcend surface-level performance metrics. This study investigates the internal neural representations of cognitive complexity using Bloom's Taxonomy as a hierarchical lens. By analyzing high-dimensional activation vectors from different LLMs, we probe whether different cognitive levels, ranging from basic recall (Remember) to abstract synthesis (Create), are linearly separable within the model's residual streams. Our results demonstrate that linear classifiers achieve approximately 95% mean accuracy across all Bloom levels, providing strong evidence that cognitive level is encoded in a linearly accessible subspace of the model's representations. These findings provide evidence that the model resolves the cognitive difficulty of a prompt early in the forward pass, with representations becoming increasingly separable across layers. 

---
# From Labor to Collaboration: A Methodological Experiment Using AI Agents to Augment Research Perspectives in Taiwan's Humanities and Social Sciences 

**Authors**: Yi-Chih Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.17221)  

**Abstract**: Generative AI is reshaping knowledge work, yet existing research focuses predominantly on software engineering and the natural sciences, with limited methodological exploration for the humanities and social sciences. Positioned as a "methodological experiment," this study proposes an AI Agent-based collaborative research workflow (Agentic Workflow) for humanities and social science research. Taiwan's this http URL usage data (N = 7,729 conversations, November 2025) from the Anthropic Economic Index (AEI) serves as the empirical vehicle for validating the feasibility of this methodology.
This study operates on two levels: the primary level is the design and validation of a methodological framework - a seven-stage modular workflow grounded in three principles: task modularization, human-AI division of labor, and verifiability, with each stage delineating clear roles for human researchers (research judgment and ethical decisions) and AI Agents (information retrieval and text generation); the secondary level is the empirical analysis of AEI Taiwan data - serving as an operational demonstration of the workflow's application to secondary data research, showcasing both the process and output quality (see Appendix A).
This study contributes by proposing a replicable AI collaboration framework for humanities and social science researchers, and identifying three operational modes of human-AI collaboration - direct execution, iterative refinement, and human-led - through reflexive documentation of the operational process. This taxonomy reveals the irreplaceability of human judgment in research question formulation, theoretical interpretation, contextualized reasoning, and ethical reflection. Limitations including single-platform data, cross-sectional design, and AI reliability risks are acknowledged. 

---
# Sign Lock-In: Randomly Initialized Weight Signs Persist and Bottleneck Sub-Bit Model Compression 

**Authors**: Akira Sakai, Yuma Ichikawa  

**Link**: [PDF](https://arxiv.org/pdf/2602.17063)  

**Abstract**: Sub-bit model compression seeks storage below one bit per weight; as magnitudes are aggressively compressed, the sign bit becomes a fixed-cost bottleneck. Across Transformers, CNNs, and MLPs, learned sign matrices resist low-rank approximation and are spectrally indistinguishable from an i.i.d. Rademacher baseline. Despite this apparent randomness, most weights retain their initialization signs; flips primarily occur via rare near-zero boundary crossings, suggesting that sign-pattern randomness is largely inherited from initialization. We formalize this behavior with sign lock-in theory, a stopping-time analysis of sign flips under SGD noise. Under bounded updates and a rare re-entry condition into a small neighborhood around zero, the number of effective sign flips exhibits a geometric tail. Building on this mechanism, we introduce a gap-based initialization and a lightweight outward-drift regularizer, reducing the effective flip rate to approximately $10^{-3}$ with only about a one-point increase in perplexity. 

---
# RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models 

**Authors**: Yunseok Han, Yejoon Lee, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2602.17053)  

**Abstract**: Large Reasoning Models (LRMs) exhibit strong performance, yet often produce rationales that sound plausible but fail to reflect their true decision process, undermining reliability and trust. We introduce a formal framework for reasoning faithfulness, defined by two testable conditions: stance consistency (a coherent stance linking reasoning to answer) and causal influence (the stated reasoning causally drives the answer under output-level interventions), explicitly decoupled from accuracy. To operationalize this, we present RFEval, a benchmark of 7,186 instances across seven tasks that probes faithfulness via controlled, output-level counterfactual interventions. Evaluating twelve open-source LRMs, we find unfaithfulness in 49.7% of outputs, predominantly from stance inconsistency. Failures are concentrated in brittle, convergent domains such as math and code, and correlate more with post-training regimes than with scale: within-family ablations indicate that adding current RL-style objectives on top of supervised fine-tuning can reduce reasoning faithfulness, even when accuracy is maintained. Crucially, accuracy is neither a sufficient nor a reliable proxy for faithfulness: once controlling for model and task, the accuracy-faithfulness link is weak and statistically insignificant. Our work establishes a rigorous methodology for auditing LRM reliability and shows that trustworthy AI requires optimizing not only for correct outcomes but also for the structural integrity of the reasoning process. Our code and dataset can be found at project page: $\href{this https URL}{this https URL}$ 

---
# Arcee Trinity Large Technical Report 

**Authors**: Varun Singh, Lucas Krauss, Sami Jaghouar, Matej Sirovatka, Charles Goddard, Fares Obied, Jack Min Ong, Jannik Straube, Fern, Aria Harley, Conner Stewart, Colin Kealty, Maziyar Panahi, Simon Kirsten, Anushka Deshpande, Anneketh Vij, Arthur Bresnu, Pranav Veldurthi, Raghav Ravishankar, Hardik Bishnoi, DatologyAI Team, Arcee AI Team, Prime Intellect Team, Mark McQuade, Johannes Hagemann, Lucas Atkins  

**Link**: [PDF](https://arxiv.org/pdf/2602.17004)  

**Abstract**: We present the technical report for Arcee Trinity Large, a sparse Mixture-of-Experts model with 400B total parameters and 13B activated per token. Additionally, we report on Trinity Nano and Trinity Mini, with Trinity Nano having 6B total parameters with 1B activated per token, Trinity Mini having 26B total parameters with 3B activated per token. The models' modern architecture includes interleaved local and global attention, gated attention, depth-scaled sandwich norm, and sigmoid routing for Mixture-of-Experts. For Trinity Large, we also introduce a new MoE load balancing strategy titled Soft-clamped Momentum Expert Bias Updates (SMEBU). We train the models using the Muon optimizer. All three models completed training with zero loss spikes. Trinity Nano and Trinity Mini were pre-trained on 10 trillion tokens, and Trinity Large was pre-trained on 17 trillion tokens. The model checkpoints are available at this https URL. 

---
# Sonar-TS: Search-Then-Verify Natural Language Querying for Time Series Databases 

**Authors**: Zhao Tan, Yiji Zhao, Shiyu Wang, Chang Xu, Yuxuan Liang, Xiping Liu, Shirui Pan, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.17001)  

**Abstract**: Natural Language Querying for Time Series Databases (NLQ4TSDB) aims to assist non-expert users retrieve meaningful events, intervals, and summaries from massive temporal records. However, existing Text-to-SQL methods are not designed for continuous morphological intents such as shapes or anomalies, while time series models struggle to handle ultra-long histories. To address these challenges, we propose Sonar-TS, a neuro-symbolic framework that tackles NLQ4TSDB via a Search-Then-Verify pipeline. Analogous to active sonar, it utilizes a feature index to ping candidate windows via SQL, followed by generated Python programs to lock on and verify candidates against raw signals. To enable effective evaluation, we introduce NLQTSBench, the first large-scale benchmark designed for NLQ over TSDB-scale histories. Our experiments highlight the unique challenges within this domain and demonstrate that Sonar-TS effectively navigates complex temporal queries where traditional methods fail. This work presents the first systematic study of NLQ4TSDB, offering a general framework and evaluation standard to facilitate future research. 

---
# Exploring LLMs for User Story Extraction from Mockups 

**Authors**: Diego Firmenich, Leandro Antonelli, Bruno Pazos, Fabricio Lozada, Leonardo Morales  

**Link**: [PDF](https://arxiv.org/pdf/2602.16997)  

**Abstract**: User stories are one of the most widely used artifacts in the software industry to define functional requirements. In parallel, the use of high-fidelity mockups facilitates end-user participation in defining their needs. In this work, we explore how combining these techniques with large language models (LLMs) enables agile and automated generation of user stories from mockups. To this end, we present a case study that analyzes the ability of LLMs to extract user stories from high-fidelity mockups, both with and without the inclusion of a glossary of the Language Extended Lexicon (LEL) in the prompts. Our results demonstrate that incorporating the LEL significantly enhances the accuracy and suitability of the generated user stories. This approach represents a step forward in the integration of AI into requirements engineering, with the potential to improve communication between users and developers. 

---
# Characterizing the Predictive Impact of Modalities with Supervised Latent-Variable Modeling 

**Authors**: Divyam Madaan, Sumit Chopra, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2602.16979)  

**Abstract**: Despite the recent success of Multimodal Large Language Models (MLLMs), existing approaches predominantly assume the availability of multiple modalities during training and inference. In practice, multimodal data is often incomplete because modalities may be missing, collected asynchronously, or available only for a subset of examples. In this work, we propose PRIMO, a supervised latent-variable imputation model that quantifies the predictive impact of any missing modality within the multimodal learning setting. PRIMO enables the use of all available training examples, whether modalities are complete or partial. Specifically, it models the missing modality through a latent variable that captures its relationship with the observed modality in the context of prediction. During inference, we draw many samples from the learned distribution over the missing modality to both obtain the marginal predictive distribution (for the purpose of prediction) and analyze the impact of the missing modalities on the prediction for each instance. We evaluate PRIMO on a synthetic XOR dataset, Audio-Vision MNIST, and MIMIC-III for mortality and ICD-9 prediction. Across all datasets, PRIMO obtains performance comparable to unimodal baselines when a modality is fully missing and to multimodal baselines when all modalities are available. PRIMO quantifies the predictive impact of a modality at the instance level using a variance-based metric computed from predictions across latent completions. We visually demonstrate how varying completions of the missing modality result in a set of plausible labels. 

---
# HQFS: Hybrid Quantum Classical Financial Security with VQC Forecasting, QUBO Annealing, and Audit-Ready Post-Quantum Signing 

**Authors**: Srikumar Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2602.16976)  

**Abstract**: Here's the corrected paragraph with all punctuation and formatting issues fixed:
Financial risk systems usually follow a two-step routine: a model predicts return or risk, and then an optimizer makes a decision such as a portfolio rebalance. In practice, this split can break under real constraints. The prediction model may look good, but the final decision can be unstable when the market shifts, when discrete constraints are added (lot sizes, caps), or when the optimization becomes slow for larger asset sets. Also, regulated settings need a clear audit trail that links each decision to the exact model state and inputs. We present HQFS, a practical hybrid pipeline that connects forecasting, discrete risk optimization, and auditability in one flow. First, HQFS learns next-step return and a volatility proxy using a variational quantum circuit (VQC) with a small classical head. Second, HQFS converts the risk-return objective and constraints into a QUBO and solves it with quantum annealing when available, while keeping a compatible classical QUBO solver as a fallback for deployment. Third, HQFS signs each rebalance output using a post-quantum signature so the allocation can be verified later without trusting the runtime environment. On our market dataset study, HQFS reduces return prediction error by 7.8% and volatility prediction error by 6.1% versus a tuned classical baseline. For the decision layer, HQFS improves out-of-sample Sharpe by 9.4% and lowers maximum drawdown by 11.7%. The QUBO solve stage also cuts average solve time by 28% compared to a mixed-integer baseline under the same constraints, while producing fully traceable, signed allocation records. 

---
# Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents 

**Authors**: Haiyang Xu, Xi Zhang, Haowei Liu, Junyang Wang, Zhaozai Zhu, Shengjie Zhou, Xuhao Hu, Feiyu Gao, Junjie Cao, Zihua Wang, Zhiyuan Chen, Jitong Liao, Qi Zheng, Jiahui Zeng, Ze Xu, Shuai Bai, Junyang Lin, Jingren Zhou, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.16855)  

**Abstract**: The paper introduces GUI-Owl-1.5, the latest native GUI agent model that features instruct/thinking variants in multiple sizes (2B/4B/8B/32B/235B) and supports a range of platforms (desktop, mobile, browser, and more) to enable cloud-edge collaboration and real-time interaction. GUI-Owl-1.5 achieves state-of-the-art results on more than 20+ GUI benchmarks on open-source models: (1) on GUI automation tasks, it obtains 56.5 on OSWorld, 71.6 on AndroidWorld, and 48.4 on WebArena; (2) on grounding tasks, it obtains 80.3 on ScreenSpotPro; (3) on tool-calling tasks, it obtains 47.6 on OSWorld-MCP, and 46.8 on MobileWorld; (4) on memory and knowledge tasks, it obtains 75.5 on GUI-Knowledge Bench. GUI-Owl-1.5 incorporates several key innovations: (1) Hybird Data Flywheel: we construct the data pipeline for UI understanding and trajectory generation based on a combination of simulated environments and cloud-based sandbox environments, in order to improve the efficiency and quality of data collection. (2) Unified Enhancement of Agent Capabilities: we use a unified thought-synthesis pipeline to enhance the model's reasoning capabilities, while placing particular emphasis on improving key agent abilities, including Tool/MCP use, memory and multi-agent adaptation; (3) Multi-platform Environment RL Scaling: We propose a new environment RL algorithm, MRPO, to address the challenges of multi-platform conflicts and the low training efficiency of long-horizon tasks. The GUI-Owl-1.5 models are open-sourced, and an online cloud-sandbox demo is available at this https URL. 

---
# Training Large Reasoning Models Efficiently via Progressive Thought Encoding 

**Authors**: Zeliang Zhang, Xiaodong Liu, Hao Cheng, Hao Sun, Chenliang Xu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2602.16839)  

**Abstract**: Large reasoning models (LRMs) excel on complex problems but face a critical barrier to efficiency: reinforcement learning (RL) training requires long rollouts for outcome-based rewards, where autoregressive decoding dominates time and memory usage. While sliding-window cache strategies can bound memory, they disrupt long-context reasoning and degrade performance. We introduce Progressive Thought Encoding, a parameter-efficient fine-tuning method that enables LRMs to reason effectively under fixed-size caches. By progressively encoding intermediate reasoning into fixed-size vector representations, our approach eliminates the need to backpropagate through full-cache rollouts, thereby reducing memory usage, while maintaining constant memory during inference. Experiments on three models, including Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, and DeepSeek-R1-Distill-Llama-8B, on six widely used challenging mathematical benchmarks show consistent gains: our method achieves +19.3% improvement over LoRA-based fine-tuning and +29.9% over LRMs without fine-tuning on average, with up to +23.4 accuracy improvement on AIME2024/2025 under the same tight cache budgets. These results demonstrate that Progressive Thought Encoding not only improves reasoning accuracy but also makes RL training of LRMs substantially more efficient and scalable under real-world memory constraints. 

---
# IndicJR: A Judge-Free Benchmark of Jailbreak Robustness in South Asian Languages 

**Authors**: Priyaranjan Pattnayak, Sanchari Chowdhuri  

**Link**: [PDF](https://arxiv.org/pdf/2602.16832)  

**Abstract**: Safety alignment of large language models (LLMs) is mostly evaluated in English and contract-bound, leaving multilingual vulnerabilities understudied. We introduce \textbf{Indic Jailbreak Robustness (IJR)}, a judge-free benchmark for adversarial safety across 12 Indic and South Asian languages (2.1 Billion speakers), covering 45216 prompts in JSON (contract-bound) and Free (naturalistic) tracks.
IJR reveals three patterns. (1) Contracts inflate refusals but do not stop jailbreaks: in JSON, LLaMA and Sarvam exceed 0.92 JSR, and in Free all models reach 1.0 with refusals collapsing. (2) English to Indic attacks transfer strongly, with format wrappers often outperforming instruction wrappers. (3) Orthography matters: romanized or mixed inputs reduce JSR under JSON, with correlations to romanization share and tokenization (approx 0.28 to 0.32) indicating systematic effects. Human audits confirm detector reliability, and lite-to-full comparisons preserve conclusions. IJR offers a reproducible multilingual stress test revealing risks hidden by English-only, contract-focused evaluations, especially for South Asian users who frequently code-switch and romanize. 

---
# Hybrid-Gym: Training Coding Agents to Generalize Across Tasks 

**Authors**: Yiqing Xie, Emmy Liu, Gaokai Zhang, Nachiket Kotalwar, Shubham Gandhi, Sathwik Acharya, Xingyao Wang, Carolyn Rose, Graham Neubig, Daniel Fried  

**Link**: [PDF](https://arxiv.org/pdf/2602.16819)  

**Abstract**: When assessing the quality of coding agents, predominant benchmarks focus on solving single issues on GitHub, such as SWE-Bench. In contrast, in real use, these agents solve more various and complex tasks that involve other skills such as exploring codebases, testing software, and designing architecture. In this paper, we first characterize some transferable skills that are shared across diverse tasks by decomposing trajectories into fine-grained components, and derive a set of principles for designing auxiliary training tasks to teach language models these skills. Guided by these principles, we propose a training environment, Hybrid-Gym, consisting of a set of scalable synthetic tasks, such as function localization and dependency search. Experiments show that agents trained on our synthetic tasks effectively generalize to diverse real-world tasks that are not present in training, improving a base model by 25.4% absolute gain on SWE-Bench Verified, 7.9% on SWT-Bench Verified, and 5.1% on Commit-0 Lite. Hybrid-Gym also complements datasets built for the downstream tasks (e.g., improving SWE-Play by 4.9% on SWT-Bench Verified). Code available at: this https URL. 

---
# Better Think Thrice: Learning to Reason Causally with Double Counterfactual Consistency 

**Authors**: Victoria Lin, Xinnuo Xu, Rachel Lawrence, Risa Ueno, Amit Sharma, Javier Gonzalez, Niranjani Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2602.16787)  

**Abstract**: Despite their strong performance on reasoning benchmarks, large language models (LLMs) have proven brittle when presented with counterfactual questions, suggesting weaknesses in their causal reasoning ability. While recent work has demonstrated that labeled counterfactual tasks can be useful benchmarks of LLMs' causal reasoning, producing such data at the scale required to cover the vast potential space of counterfactuals is limited. In this work, we introduce double counterfactual consistency (DCC), a lightweight inference-time method for measuring and guiding the ability of LLMs to reason causally. Without requiring labeled counterfactual data, DCC verifies a model's ability to execute two important elements of causal reasoning: causal intervention and counterfactual prediction. Using DCC, we evaluate the causal reasoning abilities of various leading LLMs across a range of reasoning tasks and interventions. Moreover, we demonstrate the effectiveness of DCC as a training-free test-time rejection sampling criterion and show that it can directly improve performance on reasoning tasks across multiple model families. 

---
# Omitted Variable Bias in Language Models Under Distribution Shift 

**Authors**: Victoria Lin, Louis-Philippe Morency, Eli Ben-Michael  

**Link**: [PDF](https://arxiv.org/pdf/2602.16784)  

**Abstract**: Despite their impressive performance on a wide variety of tasks, modern language models remain susceptible to distribution shifts, exhibiting brittle behavior when evaluated on data that differs in distribution from their training data. In this paper, we describe how distribution shifts in language models can be separated into observable and unobservable components, and we discuss how established approaches for dealing with distribution shift address only the former. Importantly, we identify that the resulting omitted variable bias from unobserved variables can compromise both evaluation and optimization in language models. To address this challenge, we introduce a framework that maps the strength of the omitted variables to bounds on the worst-case generalization performance of language models under distribution shift. In empirical experiments, we show that using these bounds directly in language model evaluation and optimization provides more principled measures of out-of-distribution performance, improves true out-of-distribution performance relative to standard distribution shift adjustment methods, and further enables inference about the strength of the omitted variables when target distribution labels are available. 

---
# PREFER: An Ontology for the PREcision FERmentation Community 

**Authors**: Txell Amigó, Shawn Zheng Kai Tan, Angel Luu Phanthanourak, Sebastian Schulz, Pasquale D. Colaianni, Dominik M. Maszczyk, Ester Milesi, Ivan Schlembach, Mykhaylo Semenov Petrov, Marta Reventós Montané, Lars K. Nielsen, Jochen Förster, Bernhard Ø. Palsson, Suresh Sudarsan, Alberto Santos  

**Link**: [PDF](https://arxiv.org/pdf/2602.16755)  

**Abstract**: Precision fermentation relies on microbial cell factories to produce sustainable food, pharmaceuticals, chemicals, and biofuels. Specialized laboratories such as biofoundries are advancing these processes using high-throughput bioreactor platforms, which generate vast datasets. However, the lack of community standards limits data accessibility and interoperability, preventing integration across platforms. In order to address this, we introduce PREFER, an open-source ontology designed to establish a unified standard for bioprocess data. Built in alignment with the widely adopted Basic Formal Ontology (BFO) and connecting with several other community ontologies, PREFER ensures consistency and cross-domain compatibility and covers the whole precision fermentation process. Integrating PREFER into high-throughput bioprocess development workflows enables structured metadata that supports automated cross-platform execution and high-fidelity data capture. Furthermore, PREFER's standardization has the potential to bridge disparate data silos, generating machine-actionable datasets critical for training predictive, robust machine learning models in synthetic biology. This work provides the foundation for scalable, interoperable bioprocess systems and supports the transition toward more data-driven bioproduction. 

---
# Intent Laundering: AI Safety Datasets Are Not What They Seem 

**Authors**: Shahriar Golchin, Marc Wetter  

**Link**: [PDF](https://arxiv.org/pdf/2602.16729)  

**Abstract**: We systematically evaluate the quality of widely used AI safety datasets from two perspectives: in isolation and in practice. In isolation, we examine how well these datasets reflect real-world attacks based on three key properties: driven by ulterior intent, well-crafted, and out-of-distribution. We find that these datasets overrely on "triggering cues": words or phrases with overt negative/sensitive connotations that are intended to trigger safety mechanisms explicitly, which is unrealistic compared to real-world attacks. In practice, we evaluate whether these datasets genuinely measure safety risks or merely provoke refusals through triggering cues. To explore this, we introduce "intent laundering": a procedure that abstracts away triggering cues from attacks (data points) while strictly preserving their malicious intent and all relevant details. Our results indicate that current AI safety datasets fail to faithfully represent real-world attacks due to their overreliance on triggering cues. In fact, once these cues are removed, all previously evaluated "reasonably safe" models become unsafe, including Gemini 3 Pro and Claude Sonnet 3.7. Moreover, when intent laundering is adapted as a jailbreaking technique, it consistently achieves high attack success rates, ranging from 90% to over 98%, under fully black-box access. Overall, our findings expose a significant disconnect between how model safety is evaluated and how real-world adversaries behave. 

---
# Retrieval Augmented (Knowledge Graph), and Large Language Model-Driven Design Structure Matrix (DSM) Generation of Cyber-Physical Systems 

**Authors**: H. Sinan Bank, Daniel R. Herber  

**Link**: [PDF](https://arxiv.org/pdf/2602.16715)  

**Abstract**: We explore the potential of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Graph-based RAG (GraphRAG) for generating Design Structure Matrices (DSMs). We test these methods on two distinct use cases -- a power screwdriver and a CubeSat with known architectural references -- evaluating their performance on two key tasks: determining relationships between predefined components, and the more complex challenge of identifying components and their subsequent relationships. We measure the performance by assessing each element of the DSM and overall architecture. Despite design and computational challenges, we identify opportunities for automated DSM generation, with all code publicly available for reproducibility and further feedback from the domain experts. 

---
# Multi-Objective Alignment of Language Models for Personalized Psychotherapy 

**Authors**: Mehrab Beikzadeh, Yasaman Asadollah Salmanpour, Ashima Suvarna, Sriram Sankararaman, Matteo Malgaroli, Majid Sarrafzadeh, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2602.16053)  

**Abstract**: Mental health disorders affect over 1 billion people worldwide, yet access to care remains limited by workforce shortages and cost constraints. While AI systems show therapeutic promise, current alignment approaches optimize objectives independently, failing to balance patient preferences with clinical safety. We survey 335 individuals with lived mental health experience to collect preference rankings across therapeutic dimensions, then develop a multi-objective alignment framework using direct preference optimization. We train reward models for six criteria -- empathy, safety, active listening, self-motivated change, trust/rapport, and patient autonomy -- and systematically compare multi-objective approaches against single-objective optimization, supervised fine-tuning, and parameter merging. Multi-objective DPO (MODPO) achieves superior balance (77.6% empathy, 62.6% safety) compared to single-objective optimization (93.6% empathy, 47.8% safety), and therapeutic criteria outperform general communication principles by 17.2%. Blinded clinician evaluation confirms MODPO is consistently preferred, with LLM-evaluator agreement comparable to inter-clinician reliability. 

---
