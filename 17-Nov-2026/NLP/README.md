# Beyond Job Titles: Interpreting Early-Career Job Advertisements with Sparse Autoencoders and Semantic Clustering 

**Authors**: Yisong Cao, Leah Kim, Brian Wright, Ali Rivera  

**Link**: [PDF](https://arxiv.org/pdf/2611.21586)  

**Abstract**: Job titles and job descriptions alone provide limited insight into how technical roles differ in a changing job market. To assess how such roles differ in their work and industry applications, we design and compare two methods based on deep learning and transformer representations using 36,081 early-career job postings collected from LinkedIn in the United States between December 2025 and October 2026.
We first apply PERC, short for Preprocessing (P), Embedding (E), Dimensionality Reduction (R) and Clustering (C). We employ GPT-6 Luna to classify job description passages into Team context, Role task and Qualifications. We then create combinations of these sections and examine whether embedding and clustering the segmented passages produce meaningful semantic clusters about different job roles. We use Maximum Mean Discrepancy (MMD) to rank observed differences between embedding distributions and employ a pretrained sparse autoencoder (SAE) to extract features from layer-19 representations of Llama-3.1-8B-Instruct. Lastly, we employ GPT-6 Luna to come up with hypotheses about job roles' feature patterns, and GPT-6.1 Terra to investigate whether those patterns exist in held-out job descriptions from unseen companies.
Some example findings include Data Scientist advertisements placing more emphasis on machine learning model development, Data Analyst advertisements on reporting business performance, and Machine Learning Engineer advertisements on applying developed models in real-world settings. We also observe differences in the applications described by Data Analyst and Business Analyst advertisements across selected months, although our dataset's scale and uneven coverage limit their interpretation. Overall, we present a realistic application of SAEs and PERC to unstructured textual data and examine how automated insight extraction could support manual analysis of job descriptions. 

---
# Evaluating Cultural Reasoning and Robustness in LLMs for Spanish Political Discourse 

**Authors**: Jorge Valverde-Rebaza, Nathan Garcia-Larios, Karime Ruvalcaba-Perez, Alejandro Rendon, Santiago Sanchez-Ballesteros, Alejandro Martin del Campo  

**Link**: [PDF](https://arxiv.org/pdf/2611.21535)  

**Abstract**: Large language models (LLMs) are increasingly used to interpret, generate, and mediate public-facing language, which makes their reliability especially consequential in politically sensitive settings. Although sentiment analysis is a common task in this area, affective meaning in social media discourse is often conveyed through culturally situated and pragmatically indirect cues that are not well captured by label-centric evaluation alone. In this paper, we evaluate the robustness of LLMs for sentiment analysis of Spanish political discourse from social media, with emphasis on culturally situated affective interpretation. We introduce a diagnostic protocol that combines three independent human annotation views into Gold and Ambiguous subsets, and extends the corpus with a Synthetic subset built from sentiment-flipped counterfactuals selected through model-conditioned perplexity and LLM-as-a-judge criteria. Using this framework, we compare proprietary and open-weight models under zero-shot and few-shot prompting, focusing on predictive robustness across human agreement conditions and counterfactual shift. Our results show that annotation ambiguity and counterfactual perturbation reveal failure modes that conventional evaluation on the original corpus tends to obscure. GPT-4.1-mini remains the most stable among the evaluated models, whereas the two smaller open-weight models degrade more markedly under synthetic perturbations, and few-shot prompting provides only limited gains. Overall, the paper contributes a reproducible protocol for robustness-oriented evaluation in political sentiment analysis, an empirical characterization of model behavior under ambiguity and sentiment-flipped counterfactuals, and a publicly available Spanish social media dataset with three-way human sentiment annotations. 

---
# Embedding Cosine Measures Text Length Before It Measures Homogenization 

**Authors**: Fengming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21519)  

**Abstract**: Mean pairwise cosine similarity between transformer embeddings is a standard measure of whether online discourse is converging under generative AI. When the embeddings are mean-pooled outputs of an encoder never trained for sentence similarity, this measure mostly tracks text length. In 6.06 million Reddit comments, an uncentered DeBERTa-v3 measure had suggested that only conservative discourse broke toward homogeneity after late 2022. Truncating the same comments from 10 to 150 words raises their mean cosine by eight times the reported break. A placebo that swaps every comment for a random one of equal length reproduces 81% of the asymmetry, and a fixed-content simulation reproduces 78%. The leading principal direction carries 99.9% of raw DeBERTa homogeneity, and once length is handled the asymmetry disappears under every encoder. On 575,000 Stack Exchange answers the measure reports a post-ChatGPT homogenization of technical sites that the placebo fully reproduces, although answers grew longer. A census of 31 studies locates the risk where anisotropic encoders meet time-series or length-unequal designs. A reverse control on a well-known experiment shows that centering with renormalization can reverse a genuine effect. I propose a low-cost diagnostic protocol. Corrected measures show that the two political camps converged between 2019 and 2022 and stopped converging before ChatGPT. Cross-camp replies engage their targets more closely than same-camp replies. 

---
# Jetriever: Shifting Agent Memory Retrieval from Similarity to Utility 

**Authors**: Ruizhe Li, Licheng Zhang, Mingxuan Du, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2611.21478)  

**Abstract**: A useful memory need not resemble the question it helps answer. Jetriever ranks each stored memory by Jev's decision score for its usefulness to the current question, replacing similarity search with full-pool utility scoring. On InMind's 125 indirect-query tasks, it places the gold exchange first in 84.8% of cases; matched BM25 and MiniLM controls reach 1.6% and 0.0% when indexing user turns. A 20-question comparison separates finding evidence from ranking it: Jev and Qwen3.5 9B each find 90% of gold exchanges by rank 50, while Laya finds 30%; at rank one, Jev reaches 85% and Qwen3.5 9B 40%. On a dimension-proportional random sample of 60 MemoryAgentBench questions, Jetriever's four-dimension score is 55.4%, against 44.3% for ReFind on the same questions. Jev retrieval costs USD 0.00834 per InMind question and USD 0.03103 per MAB question on average. These results show the value of world knowledge, scores that rank decisive memories first, and affordable full-pool scoring. 

---
# A Scalable Multi Task Learning Framework for Modeling Demographic Disagreements in Annotation Tasks 

**Authors**: Soumyajit Gupta, Maria De-Arteaga, Matthew Lease  

**Link**: [PDF](https://arxiv.org/pdf/2611.21458)  

**Abstract**: Annotators from different demographic backgrounds often disagree in systematic ways due to varying lived experiences, cultural contexts, and their relationship to the content. When such disagreement occurs, one group's perspective may sometimes warrant prioritization because they hold an epistemically advantaged standpoint or bear greater stake in the outcome. For example, when assessing toxicity of posts on social media, it may be important to prioritize a targeted group's viewpoint. We propose a Stakeholder-Aware Joint Multi-Task Learning (SAJ-MTL) framework to jointly model demographic disagreement both across- and within- annotator groups. The model uses conditional gradient routing to handle sparsity induced by group-conditional labeling, and parameterization supports customized weighting of groups in different task settings. SAJ-MTL improves group-level performance in toxicity detection through better context-aware understanding of demographic-sensitive content. Only annotator group-membership information is required, which is more practical than existing methods requiring more privacy-invasive individual-level annotator details. Our scalable architecture is also computationally efficient to support large-scale deployment. 

---
# Replicating and Extending Weight-Space Programming: Frontier LLMs Compile Programs into Weights and Read Input Layers 

**Authors**: Roksana Goworek, Alexander Hickey, Federico Nanni, Martin O'Reilly  

**Link**: [PDF](https://arxiv.org/pdf/2611.21449)  

**Abstract**: Researchers at MIT recently reported that GPT-6 Astra, a frontier large language model (LLM), can write the weights of small neural networks directly in a single reply, and can read the raw weights of small networks to say what they were trained to do. In a three-day sprint, we replicated and extended these results. Our four findings: (1) Writing weights reproduces. Astra's networks for the game Breakout scored 514, 550 and 307 (versus MIT's 517), and scored just as well on two modified versions of Breakout. Claude Opus 5.5 can also write working networks, but often runs out of output tokens. (2) What is new is translating programs into weights. Earlier models could write a good policy as code a generation before they could write network weights, and the networks the models write behave like compiled programs: two models independently decompiled Astra's Breakout network into Python matching its actions in all tested states. On classifying handwritten digits, Astra's networks scored 50% on average, below a simple classifier (82%). (3) Reading weights is mostly reading the input layer. Astra and Opus 5.5 identify which game a network plays almost perfectly, but on three new games we designed and trained ourselves, Astra recovered where things are on the board, while mostly missing how the game works. (4) Auditing from weights alone is possible but shallow. Without executing code, Astra picked out a small language model poisoned during training in 10 of 10 runs, one poisoned by extra training afterwards in 7 of 10, and a malicious adapter in 23 of 32 cases, but never identified the trigger or malicious behaviour. Open-weight models lag behind. From this early study, we conclude that LLMs can read and write small networks, but that this currently amounts to compiling hand-designed programs and reading the layout of the input layer. It is a promising but early tool for interpretability. 

---
# Rubric-Based AES with LLM Ensembles: Scoring of Student Responses in Critical Online Reasoning Using Ensembles of LLMs 

**Authors**: Daniel Baumartz, Alexander Mehler, Patrick Schrottenbacher, Leon Hammerla, Sebastian Gombert  

**Link**: [PDF](https://arxiv.org/pdf/2611.21422)  

**Abstract**: Grading student responses is costly and time-consuming. Moreover, human annotators are prone to bias, fatigue, and rater drift, which further limits grading reliability. Various methods have been employed to develop automated essay scoring (AES) systems to reduce these costs and address bias-related issues. These methods range from early text-feature-based systems to current machine-learning and transformer-based models. Specifically, large language models (LLMs) offer the potential to streamline the development of AES systems. However, they are subject to well-known limitations, such as hallucinations and prompt sensitivity. This creates a need to explore the potential of a mixed-methods approach that leverages the synergies among individual LLMs through ensemble formation. We evaluate the potential of ensembles of LLMs to match human-rater agreement when grading student responses in economics and medical education across 103 criteria. We use 631 German-language responses and corresponding human ratings by 2 to 5 annotators from a previously conducted assessment. We experiment with 523 rubric-based prompt configurations across 15 open-weight LLMs and assess their consistency with human raters using Krippendorff's alpha for inter-annotator agreement (IAA) and the Alternative Annotator Test (alt-test). Our results show that replacing one human rater with the best-performing LLM--prompt configuration maintains or improves IAA for 91 of the 103 criteria and that LLMs pass the alt-test for 90 criteria. Our findings suggest that current state-of-the-art LLMs can support AES of German responses in economics and medical education, reducing human workload and enabling faster processing, while validation against human ratings remains necessary. 

---
# Train After You Measure: a Pre-Registered Learning Curve for Fine-Tuning Typed Decision Models on One User's Labels 

**Authors**: Federico Lucca  

**Link**: [PDF](https://arxiv.org/pdf/2611.21393)  

**Abstract**: Fine-tuning a small typed decision model on one user's email labels did not clear the margin we pre-registered over a logistic regression trained on the same labels. In earlier work, about two hundred labels and a TF-IDF logistic regression had the highest point accuracy on this inbox, though not significantly above the best hosted model; we ask whether the same labels, given to a decision model, close the gap. A first registered study, on a generic 4B base with cross-validation and message type only, stayed below the baseline at every label budget (-1.8 to -7.3 points). The main study was registered before any fine-tuning run: a fixed test set of 135 emails never used for selection, LoRA adapters on the public Tev1-0.8B and Tev1-4B decision models, nested subsets of 50 to 587 labels, two seeds and five decision fields on a free T4 GPU. Four of its five hypotheses are refuted--two in the fine-tunes' favour--and one is confirmed for one model. On message type the 4B model misses the registered three-point margin at 400 labels (+2.6) and passes it with all 587 (+3.3); with all labels both models are above the baseline by +0.7 and +3.3 points, where one email is 0.74 points and a 95% interval is about 15 points wide, and the 4B model is clearly ahead in type macro-F1 (0.581 vs. 0.406). On the three action switches the baseline is ahead at every label budget for both models (0.931 vs. 0.904 and 0.899 mean accuracy with all labels), at under a millisecond per question against 306 and 1681 ms. The 0.8B type curve barely improves after 300 labels and then falls. The baseline never predicts the two rarest classes (7 offers and 8 spam in the training pool); the fine-tuned models find spam in some runs, not consistently across seeds. We list the explanations we did not test and say when a new attempt would be worth making. Scripts are public; weights and emails are not. 

---
# Stability-Compensated Threshold Calibration for Selection-Conditioned Risk Control in Large Language Models 

**Authors**: Zhiyuan Wang, Yuanchang Ye, Jiahao Yuan, Jinhao Duan, Xiaoshuang Shi, Weijun Xie, Ye Wei, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21384)  

**Abstract**: Large language models often produce hallucinated answers, motivating selective prediction policies that abstain on uncertain queries. However, heuristic uncertainty thresholds alone do not provide statistical control of errors among accepted predictions. Linear Expectation Constraints (LEC) reduce selection-conditioned error rate (SCER) calibration to a nonpositive expected linear-loss constraint, but the associated loss is non-monotonic in the acceptance threshold. We construct an exact counterexample under i.i.d. sampling demonstrating that LEC's empirical threshold-selection rule can violate the population SCER target. To address this risk-transfer gap, we introduce Stability-Compensated Linear Expectation Control (SC-LEC), which uses the negative linear risk of a theoretical full-data reference to compensate for threshold-selection instability. For local expansion beyond a safe anchor, we derive two finite-sample certificates: a stability-based certificate exploiting rank crossings and a direct certificate using the structure of threshold expansion to reduce conservatism. With a valid population anchor margin, our certificate conditions ensure finite-sample marginal SCER control, and independent margin certification further provides a $1-\delta$ high-probability guarantee for this calibration-averaged control. Experiments on CommonsenseQA, ARC-Challenge, and TriviaQA show useful retention gains beyond the same certified anchor, while revealing conservative behavior in small-margin regimes. 

---
# What You Post Is What You Symptom: Estimating Psychometric Networks from Social Media Discourse 

**Authors**: Ştefana-Arina Tăbuşcă, Ana Sabina Uban, Liviu P. Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21362)  

**Abstract**: Symptom networks in psychopathology are estimated almost exclusively from self-report questionnaires, restricting inference to whatever symptom set a given instrument covers. We introduce a framework for symptom-resolved network estimation directly from social media discourse and apply the framework to depression and eating disorders, with a third anorexia corpus as cross-dataset validation. To our knowledge, we are the first to estimate symptom-resolved psychometric networks from social media text, compare them to clinical co-endorsement networks, and replicate transdiagnostic linguistic markers across 3 independently recruited corpora spanning 2 disorders simultaneously. For depression, expression scores correlate with BDI-II totals at $\rho{=}0.434$, 20 of 21 items reach FDR significance, and split-half reliability is 0.935. The Network Comparison Test detects few edge differences between modalities (3-5\% of edges), while bridge symptoms diverge. For eating disorders, the item-level results are exploratory. Identified linguistic features replicate the same direction across all 3 corpora, and they cluster in three construct families. 

---
# Syllable- and Word-Level Evaluation of Tibetan OCR 

**Authors**: Jacob Moore, Paula Lauren  

**Link**: [PDF](https://arxiv.org/pdf/2611.21303)  

**Abstract**: Optical character recognition (OCR) for Tibetan is evaluated predominantly by character error rate (CER). Tibetan marks syllable boundaries but does not mark word boundaries, so a word error rate requires a segmenter. This study evaluates nine recognizer configurations on typeset, woodblock and two handwritten script styles, alone and in combination, and reports syllable- and word-level error rates, to our knowledge the first for Tibetan OCR. The configurations span CTC models, TrOCR encoder-decoders with and without post-training quantization, and a vision-language model, and each is scored under CER, the syllable error rate (SER) and three segmented word error rates (SWER). CER ranks systems largely as the higher-level rates do (Kendall $\tau \geq 0.833$), with one pair ordered oppositely. On the effect of joint training, CER and SER record significant changes of opposite sign in 7 of 64 comparisons. SER needs no segmenter and ranks systems as SWER does under two of the three segmenters ($\tau \geq 0.889$). The output of each configuration is also tagged for part of speech, annotated for named entities and machine translated, and each result is compared with that obtained on the reference text. As rough guidance, translation chrF of 60 is reached at SER of 0.047 or below and part-of-speech preservation of 0.8 at 0.078 or below, while no error rate separates the systems that reach entity preservation of 0.6 from those that do not. 

---
# Latent Bridge: Asking for Help Without Words 

**Authors**: Qichao Wang, Tingyu Qu, Pengcheng Wu, Yiran Zhong, Feida Zhu, Zhiqi Shen, Peilin Zhao, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2611.21301)  

**Abstract**: Text-based model collaboration requires a large model to generate intermediate responses before a smaller model can use its guidance, adding large-model decoding cost to inference. We introduce POND (Prefill-Only, No-Decode), a method that uses a frozen large sender for prefill and a small receiver for all autoregressive decoding. A trainable bridge maps the sender's hidden states to additional attention keys and values, allowing the receiver to use these representations throughout generation. POND also supports periodic review, in which the sender processes the receiver's partial response and updates its guidance through additional prefill passes while preserving the receiver's existing KV cache. Across reasoning and coding benchmarks, POND improves receiver performance with senders from different model families. In particular, GLM-5.3 provides gains comparable to those of Qwen3.8-Max when paired with a Qwen receiver, supporting the effectiveness of representation transfer between independently trained model families. POND further demonstrates its ability to generalize across domains. When trained on math and coding tasks, it transfers directly to general knowledge tasks without additional domain-specific training. These results demonstrate that large frontier model prefill representations can improve small-model generation without requiring large-model autoregressive decoding. 

---
# LATTE-WordSeg-Lite: Multi-Granularity Thai Word Segmentation on a CPU, with a Reproducible Benchmark 

**Authors**: Thodsaporn Chay-intr, Kobkrit Viriyayudhakorn, Thanaruk Theeramunkong  

**Link**: [PDF](https://arxiv.org/pdf/2611.21300)  

**Abstract**: Large language models read Thai through subword tokens, without a word segmenter. Yet Thai treebanks, speech metrics and morphological resources are built on words, often at several granularities, and word boundaries resolve ambiguities that subword tokens leave open. LATTE (Lattice ATTentive Encoding) reads a text at several granularities at once, through a lattice of dictionary words, but it is a large GPU model. We distil it into LATTE-WordSeg-Lite, a CPU segmenter of 1.17M parameters that keeps the lattice as features. It outputs compounds, words and morphemes in one pass, and learns only from annotated data and text whose licences allow commercial use. To compare it with other tools, published Thai scores do not suffice, as they come from different files, splits and scorers. We therefore rebuild seven test sets from pinned sources and score twelve systems under one protocol. With the unit that fits each test set, LATTE-WordSeg-Lite scores above DeepCut on five sets, four of them from corpora that its teachers learnt from. Called line by line on one CPU thread, it runs 23 times as fast as DeepCut. We release the model, the benchmark scripts and all predictions. 

---
# Benchmarking Large Language Models for Linguistic Annotation 

**Authors**: Ondřej Tichý, Anna Marklová, Barbora Bulantová, Magdalena Titlbachová, Jiří Milička  

**Link**: [PDF](https://arxiv.org/pdf/2611.21291)  

**Abstract**: Assessing large language models for linguistic annotation requires task-specific evaluation under documented conditions. We present dhAIBench, a Python benchmarking and annotation tool with a web interface for exploring and sharing results. It records prompts, configurations and predictions, enabling comparisons of annotation quality, confidence calibration, repeatability and estimated cost. We demonstrate the framework and the associated methodology through 293 runs of 48 models on nine linguistic annotation tasks spanning morphology, syntax, semantics, discourse-pragmatics, spelling normalization and learner-language correction. Models achieve high agreement with reference annotations on several tasks, including those on non-standardized, historical varieties as well as requiring contextual interpretation. The results support evaluating models against specific linguistic tasks, rather than relying on general rankings. To this end, we make the current and future benchmarks publicly available at this http URL. 

---
# A Generative AI-Driven Architecture for Whole-Home Intelligent Services: Edge Models, Memory Management, and Household Applications 

**Authors**: Mo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2611.21232)  

**Abstract**: Generative artificial intelligence is entering the home. Cloud-based assistants face governance limitations in data transmission, continuous availability, and the reliability of physical actions, which are especially salient in households where devices remain online, data accumulate over time, and commands trigger real-world actions. Using the home local server as a unifying organizing axis, this article develops a design-oriented review framework. Sources from arXiv, CNKI, ACL Anthology, and official documentation were searched through November 2026 and coded into five evidence categories: reviews, system prototypes, benchmarks, HCI and eldercare studies, and official documentation. System designs were mapped onto shared architectural modules, identifying nine recurrent responsibilities of the home local server. Generative models support perceptual interpretation, intent understanding, and candidate planning, while deterministic validation, knowledge-graph constraints, and authorization leases confine candidate actions to an inspectable space. Home control, emotional companionship, and intelligent guarding can be compared within a common framework: they share state and memory but differ in latency, false positive and negative rates, proactivity, and permissions. Existing benchmarks rely on virtual environments and cannot substitute for real-deployment evaluation. Local deployment is not inherently safer, and technical continuity alone does not establish the social or medical effects of companionship and guarding. The article proposes a five-dimensional evaluation framework covering architectural capability, system resources, privacy, safety and reliability, and human experience, together with six design principles and a future research agenda for local-first intelligent home systems. 

---
# Teaching Language Models to Adapt Answer Granularity to Their Confidence 

**Authors**: Lukas Ellinger, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2611.21211)  

**Abstract**: When precise details are uncertain, a coarser answer can retain useful information without guessing or abstaining. We investigate whether language models can learn this behavior through model-conditioned supervision derived from consistency across sampled responses and recognition of answers at different granularity levels. We evaluate LFM2.5-1.2B, Ministral~3~3B, Qwen3.5-9B, and Gemma~4~31B on temporal and geographic questions and three generalization datasets. In several in-domain settings, fine-tuning improves the balance of correctness and informativeness beyond coarsening base answers to the best fixed level. Across generalization datasets, all four models produce coarser answers more often at low than high base-model confidence. The three smaller models' answers are also preferred over base responses on held-out relations when correctness and granularity are considered jointly. These findings demonstrate learnable confidence-related granularity adaptation, with benefits varying by model. 

---
# Validating LLM-Based Essay Scorers: A Comparative Analysis in Critical Online Reasoning in Medical Education 

**Authors**: Daniel Baumartz, Alexander Mehler, Mevlüt Bagci, Maxim Konca  

**Link**: [PDF](https://arxiv.org/pdf/2611.21208)  

**Abstract**: Automated essay scorers (AES) based on large language models (LLMs) promise human-level accuracy. However, training, fine-tuning, and evaluating LLMs for specific assessment tasks is costly. In particular, the question arises whether these AES should be trained for specific domains or whether models based on generic LLMs might be sufficient. Human annotations, by contrast, are often more valid but far more expensive. This study evaluates whether ensembles of AES systems based on general-purpose LLMs can approximate human ratings of medical-education writing tasks by leveraging encoded linguistic and conceptual knowledge. The goal is to assess whether general-purpose models are applicable for assisting or potentially replacing human annotation in this domain. We experiment with 11 AES systems and 8 different LLMs and evaluate 668 student responses to 12 tasks, using \numprint{2076} evaluations by 7 human annotators. These ratings are aggregated to create consistent performance levels that can be used as target classes for classifiers trained on AES outputs. Our results show that learner classes based on human ratings can be accurately reconstructed using a linear SVM trained on a small set of AES features. A single classifier generalized well across tasks and domains, with reasoning and content features being most predictive. This suggests that out-of-the-box LLMs, even without domain-specific fine-tuning, can support automatic essay grading in medical education. While individual AES systems are insufficient alone, their combination approximates human evaluators, pointing to a reduced need for manual scoring or its supplementation with automated methods. 

---
# Seeing Isn't Resolving: Visual and Linguistic Disambiguation with Conversational Grounding 

**Authors**: Edoardo Cecchinato, Shun Inadumi, Angel García Contreras, Giuseppe Riccardi, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2611.21183)  

**Abstract**: Visual grounding benchmarks usually do not consider the contextual uncertainty of real-world scenarios. In this study, we construct a novel benchmark, the Japanese evaluation for Ambiguous visual Grounding with Dialogue (JAG-D) dataset. It consists of human-robot conversational multimodal interactions covering different types of ambiguity, arising from visual and linguistic categories. We evaluate the capabilities of different state-of-the-art Visual Language Models (VLMs) for ambiguity recognition and resolution. We compare dialogue context grounding performance for visual and linguistic ambiguity resolution. Our experiments show that VLMs can substantially improve the resolution of linguistic ambiguity, especially Japanese zero anaphora, by using dialogue context. We further conduct a comparative analysis to understand whether grounding improvements arise from the semantic content of the dialogue or from its conversational structure. 

---
# PlainPairs: A Corpus Pairing the Prescriber and Patient Texts of U.S. Prescription Medicine Labels 

**Authors**: Obada Kraishan  

**Link**: [PDF](https://arxiv.org/pdf/2611.21160)  

**Abstract**: A prescription medicine in the United States comes with labeling written for prescribers, and many medicines also carry a text written for patients: a Medication Guide or a patient package insert. The two texts describe the same medicine for two readers, yet public resources rarely hold them side by side. I present PlainPairs, an open corpus built from the openFDA drug label files. Version 0.1.0 holds 35,177 prescription labels that carry a patient text, 21,416 distinct patient texts, 250,829 patient-text segments matched to prescriber sections through their regulatory headings, reading-level measures on both sides, and rule-based counts of plain-language wording. A second source, the National Cancer Institute's cancer information summaries, adds 756 summaries in health-professional and patient versions, in English and Spanish. Measured on the structure of the label, the median Medication Guide reads at grade 8.7 by Flesch-Kincaid and at grade 11.1 by SMOG. The same guides measured on the flattened text that the bulk files serve read about two grades higher, a difference that comes from lost list and heading boundaries. Patient texts address the reader in 41 of every 100 sentences, against 0.2 in prescriber sections. A blind hand check of 100 pairs supports the heading-based alignment for seven of ten heading types. The corpus, its code and its documents are public, and the release keeps texts that show a company copyright mark out of the text tables. 

---
# Edit Recommendation: Can LLMs Anticipate Edit Requests from Users, and When? 

**Authors**: Daisuke Kikuta  

**Link**: [PDF](https://arxiv.org/pdf/2611.21155)  

**Abstract**: After a large language model (LLM) agent generates an artifact, users often send edit requests for it. This paper studies whether an LLM can anticipate these requests before they are sent. Such anticipation is useful in practice because it reduces the effort of user review and feedback. We formalize this task as edit recommendation and build EditRecBench, a benchmark for evaluating the performance of LLMs on this task, from real-world conversation logs in WildChat-4.8M. Through benchmarking and failure analysis, we identify a practical challenge that some cases are inherently difficult to predict in real-world settings. To address this, we introduce decision model-based gating to filter out cases where predictions are unlikely to be correct before they are presented to the user. Our trained model outperforms existing decision models and raises pass@5 among shown suggestions by 12--25 points at a gating threshold. Our code is available at this https URL. 

---
# From Repositories to Training Environments: Scaling Long-Horizon Coding Agents with Environment Evolution Loop 

**Authors**: Yang Tian, Tianyu Zheng, Xiyuan Yang, Yaxin Du, Chengran Yang, Bo Zhao, Jian Yang, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2611.21152)  

**Abstract**: Training coding agents for long-horizon tasks requires environments with verifiable feedback. Existing pipelines often follow a linear process that ends with validation, establishing usability but providing little guidance on improving training value. Recursive self-improvement (RSI) often requires model rollouts or model training followed by downstream evaluation, with costs and feedback delays amplified on long-horizon tasks. How can we use timely, reliable proxy signals to improve training value and synthesize environments at scale? We present RepoFactory, an automated pipeline that converts real repositories into verifiable environments where agents build complete projects from scratch given a language specification. Its Environment Evolution Loop optimizes a multiplicative objective combining validity, structural coupling, and implementation scope through proxy feedback. RepoFactory produces 10,159 environments across 17 software domains. Training Qwen3.5-35B-A3B on these environments yields RepoHorizen-35B-A3B, which achieves gains of 22.6 percentage points on NL2Repo-Bench and 8.5 points on average across five benchmarks covering long-horizon coding and agentic tasks in real-world production settings, compared with 17.5 and 6.4 points without environment evolution. These results highlight the training value of RepoFactory environments and the additional gains from the Environment Evolution Loop. 

---
# SWE-WhatIf: Simulating Real Users of Coding Agents from Recorded Session 

**Authors**: Tae Soo Kim, Tongshuang Wu, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.21095)  

**Abstract**: Recorded sessions between users and coding agents offer a way to evaluate how agents complete real-world tasks and interact with users. However, these sessions cannot simply be replayed with new agents, because recorded user replies may no longer apply when an agent takes a different trajectory. We introduce SWE-WhatIf, which replays recorded sessions with new agents by modeling the user's intents, the criteria for satisfying them, and the conditions under which they arise. Using this model, a user simulator generates replies that adapt to unseen agent turns while remaining grounded in the recording. Across 89 real-world sessions, SWE-WhatIf generated the replies most similar to those of real users and matched the strongest baseline when sessions were rewritten. To test adaptation, we replay sessions with agents instructed to avoid or repeat actions that the recorded user pushed back on. SWE-WhatIf produced the highest win rate for avoiding these actions (66.3%), significantly above chance. We further demonstrate its utility by ranking five LLMs as coding agents, where the highest-ranked model depended on how users specified and developed their intents, and a study with 16 users, where replaying their own sessions changed 14 participants' rankings and informed their future agent use. 

---
# Evaluation-Free Recursive Self-Improvement for Coding Agents 

**Authors**: Zhe Ren, Guowei Rong, Yimeng Chen, Nanbo Li, Piotr Piękos, Mateusz Ostaszewski, Dandan Guo, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2611.21056)  

**Abstract**: Recent self-improving coding agents have achieved strong benchmark performance by iteratively modifying their own executable implementations. A common pattern uses downstream task-derived evidence to decide which candidates or lineages continue evolving, which we term evaluate-and-select. Obtaining this signal requires repeatedly executing evolved agents on downstream tasks, so broader exploration incurs increasing evaluation cost, while repeated optimization against a fixed evaluation distribution may encourage benchmark-specific adaptation. We ask: can coding agents self-improve without downstream task evaluation inside the evolution loop? We introduce Explore-then-Integrate (ELITE), which integrates implementation mechanisms rather than selecting a single exploratory child to continue the lineage. At each generation, ELITE proposes diverse focused improvements, implements them independently from the same parent, and inspects the resulting programs to identify complementary, redundant, and conflicting mechanisms before reimplementing a compatible subset as one coherent descendant. No downstream task outcomes or task-solving trajectories of evolved candidates are used to guide evolution; only deterministic checks of structural and operational validity are used inside the loop. On SWE-bench Pro, ELITE improves the initial agent from 60.0% to 85.6%, compared with 86.7% for the Darwin Godel Machine (DGM) and 88.3% for the Huxley-Godel Machine (HGM), while using zero candidate task evaluations and reducing observed wall-clock evolution time by 8.8x and 3.2x, respectively, under the evaluated configurations. In cross-benchmark transfer without further adaptation, ELITE averages 73.3% on SWE-bench Verified and 69.3% on Polyglot. Together, these results demonstrate the feasibility of coding-agent self-improvement without downstream candidate task evaluation in our setting. 

---
# MemDoctor: Stage-wise Error Attribution and Repair for MLLM Agent Memory 

**Authors**: Ruiyang Jiang, Xinyang Liao, Weikai Huang, Yuqi Niu, Junjie Zhou, Yubin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.21019)  

**Abstract**: Multimodal large language models (MLLMs) now assist users across sessions of images and text, carried by an external memory of what was said and shown. When such an assistant answers wrongly, the failure usually began earlier, in how a record was written, updated or retrieved, and the answer alone does not say where. We formalize multimodal memory as four stages, Write, Update, Retrieve and Answer, and define nine error types across them. A stage-resolved analysis on Mem-Gallery traces most failures to a source stage, and most of those lie before the answer. Injecting known errors against fact-preserving control edits, we then follow each tainted record downstream. Early errors survive to the answer as apparent evidence, and hurt most when no other record holds the fact. A diagnosis that trusts the quoted evidence therefore blames the stage where the error surfaces, not the one where it began. We propose MemDoctor, in which the MLLM that built the memory diagnoses its own failures. It traces each needed fact backward through the four stages, accepts a diagnosis only when the cited evidence is found in memory, and repairs that stage before answering again. On a 4B backbone, MemDoctor improves accuracy on both Mem-Gallery and MemLens, outperforming all evaluated 8B baselines, and the same loop also improves MuRAG and A-Mem. 

---
# Does the Voice Matter? Evaluating Frustration Detection on Deployed Clinical Voice-Agent Calls 

**Authors**: Annika Michael, Zachary Ellis, Yuxiang Zhou, Samuel Thio, Yajie Vera He, Spencer Hazel, Jared Joselowitz  

**Link**: [PDF](https://arxiv.org/pdf/2611.21001)  

**Abstract**: Autonomous voice agents now conduct routine clinical follow-up calls, so some automated triage is unavoidable. A patient becoming frustrated is one thing that triage should catch. These systems typically read only the transcript the speech-to-text stage produces, not the voice, and whether that is enough is largely untested on real clinical speech. We study per-turn frustration detection on real patient calls from a deployed clinical voice agent: 150 turns from six calls labelled by three annotators, with a length-matched acted corpus as a control, and 714 turns from 30 calls. Patient turns are overwhelmingly short: across the deployed calls, roughly 59% are one or two words, against 13% in the full acted corpus, and a corpus accuracy gap remains even after the acted set is length-matched. On one-word turns, transcript-only detection does not exclude chance. Across five audio-capable language models, the transcript flags 3 or 4 of the 39 frustrated one-word turns in the 30 calls, and a classifier given only the word count reaches 57.3%. Whether audio helps depends on the model and on the instruction. Asked to attend to delivery, two Gemini 2.5 models gain 3.0 pp over the transcript across 30 calls (-1.0 to 7.5 pp). Of the turns they confidently clear, 6.9% were in fact frustrated for audio against 13.0% for the transcript, a difference of 6.1 pp (2.2 to 10.4 pp). Across the 30 calls, given one prompt identical for both inputs, no model's audio-text interval excludes zero. Under the input-specific prompts, a newer Gemini model and two smaller open models do worse with audio than with the transcript (-3.5, -8.6 and -10.6 pp). A WavLM probe trained on the clinical calls matches the best prompted model overall (71.6%) but misses every frustrated one-word turn in the 24 added calls. When a patient's turn is short, the transcript alone is not enough to rule out frustration. 

---
# Living Context Graphs: Continual Semantic Learning for Enterprise Data Reasoning 

**Authors**: Varun Agrawal, Ayush Srivastava, Eshan Gujar, Kaushal Kumar, Ayush Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.20976)  

**Abstract**: Enterprise AI can now perform analytics, reporting, and data-driven workflows, yet the data foundations it relies on remain hard to operationalize: business meaning, relationships, metric definitions, and reporting conventions are absent from database schemas and scattered across undocumented systems and tribal knowledge. We introduce the Living Context Graph (LCG), a continually learned context layer that makes existing enterprise data operational for AI without requiring a copy or transformation of the underlying warehouse. LCG bootstraps base semantics and relationships from the data itself, builds a concept graph that indexes business context as queries are served, writes durable semantic learnings from verified outcomes, and withdraws or re-earns context when later outcomes contradict prior routes. We evaluate LCG through text-to-SQL, a core data-reasoning task, on a transfer-aware split of three BIRD databases comprising 499 questions: 353 form the learning history and 146 are held out and never used to update the context layer. Bootstrapping alone improves overall accuracy by +4.6 percentage points. Learning only from the history raises accuracy on the untouched held-out transfer set by +22.6 points (95% CI [+14.4, +30.8]), exceeding five-shot retrieval from the same history by +12.3 points. Incremental learning reaches the one-shot endpoint within observed evaluation noise on every database. Ablating the graph changes held-out accuracy by at most one question per database but increases per-query cost 2.3-3.3x and median latency 1.8-3.4x: continually learned semantics drive accuracy, while the graph makes that context efficient to serve. 

---
# Navital Gym: Rule-Graded Agent Environments for Indian Workflows 

**Authors**: Daksh Bathla  

**Link**: [PDF](https://arxiv.org/pdf/2611.20924)  

**Abstract**: AI agents are being deployed on Indian workflows such as payments, tax, forms, travel and customer support in Hindi and Hinglish. Public agent evaluations rarely cover these workflows, and many score with another language model. We present Navital Gym, a set of six open environments with 321 public tasks in five task families: trip planning, UPI payments and disputes, GST invoicing and returns, form filling (including a multi-document split) and customer support in Hindi, Hinglish and three experimental Indian languages. Every task is graded by named, deterministic rules that recompute the outcome from a synthetic sandbox world. No language model judges any score. We report one run per task for three models on all six environments, Claude Opus 5.5, Claude Sonnet 5.5 and Claude Haiku 4.5, and for Gemini 3.1 Flash-Lite on two (UPI and the Support core). Opus solves 315 of 321 tasks, Sonnet 302 and Haiku 205. Trip planning and Hindi/Hinglish support separate the models most. We also describe a private held-out set that is not published, and a 7B adapter trained in the gym. Code is Apache-2.0 and task data is CC BY 4.0. The rules are a documented simplification of Indian practice, not the live systems. 

---
# Experience Rot: Experienced Agents Fail Where Novices Succeed 

**Authors**: Zhen Xu, Hang Yin, Zitao Zhang, Kevin Zhao, Xuefeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20922)  

**Abstract**: Agents are increasingly built to learn from experience: they keep conversation histories, memory files, retrieved notes and skills, and reuse what worked before. We show that this experience turns against them when their tools change silently. We test a range of models with tools defined as in the Model Context Protocol (MCP). After a silent change, experienced agents fail on most tasks, while agents new to the task mostly succeed. We call this failure Experience Rot. Experienced agents copy their earlier call instead of checking the current interface, even when the tool definition shows the change. Then they report success. When the documentation is incomplete, experience is what makes agents succeed. After a change, the same experience leaves them worse off than no experience at all. The failure appears in every form of memory we test and grows with the amount of experience. Simple remedies fall short: announcing the update barely helps, and pointing the agent to the documentation fails whenever the documentation does not spell out the answer. 

---
# It Depends on the Task: Rethinking Rhetorical Segmentation of Dutch Judicial Decisions 

**Authors**: Samyak Sheth, Gerasimos Spanakis, Gijs van Dijck  

**Link**: [PDF](https://arxiv.org/pdf/2611.20868)  

**Abstract**: Classifying the material facts, submissions, judicial reasoning, and the court's decision (rhetorical roles) in case law is a prerequisite for high-quality performance across virtually all NLP-in-law tasks, including legal information retrieval, summarisation, and LLM-based analysis. These rhetorical roles often intertwine and shift sentence by sentence, making reliable classification difficult. This paper introduces the first sentence-level annotated dataset of Dutch judicial decisions, labelled across four legally meaningful categories: material facts, submissions, judicial reasoning, and the court's decision, alongside non-annotated (None) content, and compares several classification strategies, including a staged pipeline that separates None detection from the rhetorical roles, and a Bayesian update that leverages information about the section-header a sentence belongs to. Our results show that no single strategy dominates on both precision and recall: header priors help most when the None sentences are excluded and offer little benefit for None detection itself. These trade-offs matter in practice. For instance, extractive summarisation depends more on recall so it does not miss relevant facts, whereas outcome prediction depends more on precision regarding decision and facts sentences so that the predicted ruling is not contaminated by noise. We use these trade-offs and map the approaches tested in this study onto a set of representative legal NLP use cases. 

---
# EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation 

**Authors**: Zixuan Wang, Gregory Kang Ruey Lau, Xiao Tian, Jue Fan, Caroline Chaux, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2611.20846)  

**Abstract**: Disentangling writing style from semantic content is a fundamental challenge in literary text modeling. Style-content entanglement causes models to rely on semantic content for authorship attribution (AA) tasks and memorize author-specific content for style imitation (SI) tasks. We propose EDITORS (EditingLoRA Subspaces), a diagnose-then-deflate framework that adapts pretrained large language models and prioritizes stylistic features, making it effective for both AA and SI tasks. EDITORS first trains a diagnostic LoRA adapter using style-neutral statements derived from the training corpus to identify a content-oriented subspace. It then trains the final style adapter under activation-space deflation that projects input activations away from the identified content-oriented directions. We empirically demonstrate that EDITORS is effective. On three AA benchmarks spanning literature, social media, and news, EDITORS achieves state-of-the-art accuracy, outperforming the strongest existing baseline by up to 29%. On SI, EDITORS generates stylistically faithful and diverse passages, outperforming baselines in achieving style similarity while adhering to content specifications. 

---
# Medical question answering collapses in Hausa: a pre-registered pilot on three small open models 

**Authors**: Abbas Sani  

**Link**: [PDF](https://arxiv.org/pdf/2611.20823)  

**Abstract**: A system that reads nothing and always answers the most frequent letter scores \BaselineP{} on our Hausa medical multiple-choice set. \NBelowBaseline{} of the \NModels{} small open models we evaluate do not beat it. We lead with that because the usual summary of a multilingual result -- a percentage-point drop from English -- does not say whether what survives the drop is performance at all.
On a paired English--Hausa set of \NEval{} MedMCQA items, three instruction-tuned models of 3--4B parameters lose between \GapMedGemma{} and \GapQwen{} points of accuracy, every gap significant after Holm correction. Back-translating the Hausa items into English recovers between half and two-thirds of the gap for the two general-purpose models but not demonstrably for the medically adapted one, so the deficit is substantially, though not wholly, a matter of the language of presentation. Answers are also far less stable: agreement across three option orderings falls from \ConsEnQwen{} to \ConsHaQwen{} in the worst case. An exploratory error analysis of \ENRows{} discordant items attributes most of the damage to medical vocabulary lost in translation (\ETerm{}, 95\% CI \ETermLo{}--\ETermHi{}), usually in the question stem.
A fourth registered hypothesis proposed closing the gap with self-generated preference pairs. Its pilot gate found nothing to train on: asked to reason in Hausa rather than to pick a letter, the model echoes the prompt back, switches into Swahili, or simply stops without answering; only \GateParse{} of chains are parseable. We report that intervention as not run, which is itself the sharper result -- a forced-choice benchmark cannot see a model that has no generative command of the language at all.
All hypotheses, thresholds and decision rules were registered before any model was run, and all deviations are logged. 

---
# CuRSE-Bench: A Benchmark for Probing Cultural Blind Spots in Text-to-Image Safety Evaluation 

**Authors**: Minsuk Jang, Minseok Son, KiHyun Seol, Yeongeun Shim, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.20821)  

**Abstract**: Text-to-image (T2I) safety evaluation typically assigns each prompt or image a single culture-blind harm label, so content that is harmful only within a particular culture can pass as benign.
We introduce CultureCraft, an agentic framework that constructs T2I prompts that appear benign under a general-purpose rubric yet carry a latent harmful meaning in a target culture, and use it to build CuRSE-Bench: 10,709 culture-specific prompts and 2,648 culture-general controls over six regions and ten harm axes, each drawn from a predefined knowledge base of documented harm vectors and traceable to at least two independent sources.
Scoring every prompt under six region persona judges and a paired culture-blind judge, we find that the culture-blind judge under-rates culture-specific latent harm while matching the persona judges on the controls.
Each persona judge, in turn, chiefly decodes its own region's latent harm under our instrument: a selective cultural blind spot that persists in deployed moderators. 

---
# What Do Digital Traces Predict? Task Effects, Student Differences, and Critical Online Reasoning 

**Authors**: Maxim Konca, Alexander Mehler  

**Link**: [PDF](https://arxiv.org/pdf/2611.20803)  

**Abstract**: Digital assessment traces are increasingly used to predict student performance, often evaluated by correlating predicted and observed scores pooled across tasks. However, such correlations may reflect between-task differences rather than performance differences among students within tasks. To investigate this, we analyzed up to 527 browsing sessions from 424 university students completing open-Internet Critical Online Reasoning tasks and assessed the cross-task stability of their ratings. We evaluated the same out-of-fold predictions within and across tasks, compared three validation tiers up to leave-one-task-out, and tested incremental predictive contributions of consultation order and relational structure against matched permutation and randomization controls. None of the twelve prespecified primary tests was significant: on held-out tasks, consulted-page features showed no significant association with relative student performance, nor did observed order or relational structure add predictive value. Pooling observations across tasks substantially increased correlations, particularly for factual correctness, whose scores showed limited evidence of stable between-student differences across tasks. However, sentence embeddings yielded modest positive correlations with generic response quality on held-out tasks, without establishing added predictive value from observed order or relational structure. Our findings show that pooled predictive correlations may not reflect within-task student differences, highlighting the need for within-task evaluation and criterion generalizability audits. 

---
# Nemotron 3.5 Bolt E4B 4GB: An Open and Efficient Small Language Model for Fast Conversational Agents 

**Authors**: NVIDIA, Adi Margolin, Aditya Malte, Aishwarya Chhabra, Alex Hodson, Ameya Sunil Mahabaleshwarkar, Ambrish Dantrey, Amy Shen, Andrii Skliar, Annamalai Chockalingam, Anusha Pant, Ashwath Aithal, Ashwin Kumar, Besmira Nushi, Bhavesh Pawar, Bryan Catanzaro, Bryan Yu, Chantal Hwang, Chenhan Yu, Chris Alexiuk, Conner Currie, Cynthia Gao, David Mosallanezhad, Dina Yared, Ehsan Hosseini Asl, Eileen Long, Erick Galinkin, Felix Friedmann, Giedrius Burachas, Gorkem Batmaz, Grzegorz Chlebus, Guillermo Siman, Haifeng Qian, Hao Guo, Hayley Ross, Hrishith Thadicherla, Hui Wang, Huiying Li, Ike Nnoli, Irina Filatova, Iris Liu, Ji-Eun Lee, Jian Leong, Jian Zhang, Jiaqi Zeng, Joey Conway, Jonathan Cohen, Jose Vergara, Julien Veron Vialard, Kari Briski, Katherine Luna, Kedar Pandurang Potdar, Kevin Ning, Khalil Ben Khaled, Krzysztof Pawelec, Laura Dykes, Leili Tavabi, Lily Lee, Linda Liu, Liwei Ding, Maka Dong, Manoj Kilaru, Maria Bagdasarova, Matthew Lee, Meredith Price, Micah Schaffer, Michael Boone, Michael Fukuyama, Michael Hasper, Nancy Agarwal, Negar Habibi, Oleksii Kuchaiev, Oluwatobi Olabiyi, Pablo Ribalta, Payal Motwani, Pinky Xu, Prasoon Varshney, Pritam Biswas, Puhui Meng, Rameshwar Shivbhakta, Ran Zilberstein, Reka Kovacs, Richard Mazzarese, Rishabh Garg, Rita Fernandes Neves, Rochelle Woods, Ruby Ye, Ruvaid Virk, Ryan Hill, Saeid Motiian, Seph Mard, Shai Ben Nun, Shaona Ghosh, Sharad Yadav, Sudeep Sabnis, Suhas Hariharapura Sheshadri, Terry Kong, Thanh Do, Thinh Le, Udi Karpas  

**Link**: [PDF](https://arxiv.org/pdf/2611.20771)  

**Abstract**: We introduce Nemotron 3.5 Bolt E4B 4GB, an open and efficient small language model further post-trained from Gemma-4-E4B-it and optimized for speed and intelligence per GB of memory used for fast conversational agents. The model carries forward Gemma's strong conversational capabilities and further improves multi-turn tool use, instruction following, and input grounding for fast, local voice/device assistants, interactive personas, and embedded/Edge interfaces. The model combines natural conversation with multi-turn agentic capabilities to power voice/device assistants, interactive characters and robotic/IoT interfaces running on GeForce RTX, Jetson, and RTX/DGX Spark. Experiments on 21 benchmarks show that Nemotron 3.5 Bolt E4B 4GB achieves the highest average accuracy while generating the fewest output tokens in both reasoning and non-reasoning modes, compared with Gemma-4-E4B-it and Qwen3.5-4B. The model is released with a DFlash drafter. With DFlash, Bolt E4B achieves $2.74\times$ the per-user decode speed of inference without speculative decoding at concurrency~1, and $1.21\times$ the generation throughput of Gemma-4-E4B-it with MTP at batch size 1. 

---
# Instant Weight-Space Consolidation for Long-Term Memory in LLMs 

**Authors**: Manan Suri, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2611.20759)  

**Abstract**: Agentic memory systems give LLMs persistent memory by extracting experience into text and retrieving it at query time, but the model itself does not evolve: however many interactions it has had, it starts each query the same. Extraction errors compound, implicit cross-session patterns that no single fragment states cannot be surfaced, and the construction cost grows with every conversation. Fine-tuning on past interactions does change the weights, but requires backpropagation at serving time and degrades on long histories. We introduce CLAM (Carried Latent Adapter Memory), which combines the strengths of both: a recurrent hypernetwork performs parametric consolidation, compiling a stream of session activations into a low-rank adapter in a single forward pass with no gradient steps and no language-model generation calls, while a complementary episodic buffer supplies verbatim detail at inference. A gated latent state carries knowledge across sessions in fixed size, so the parametric memory accumulates without growing. On LoCoMo and LongMemEval-S across three open models (2-7B parameters), CLAM leads all agentic baselines by 1-45 points and all gradient-based parametric methods by 10-45 points, while requiring zero generative LLM calls, zero gradient steps, and 3-151x less wall-clock time than extraction systems to build its memory. The two channels are complementary: the consolidated adapter adds 12-15 accuracy points when the episodic buffer misses the gold evidence, and still adds 4-5 points even with an oracle buffer. 

---
# GPRM: Global Perspective Process Reward Model 

**Authors**: Zifan Zhang, Zhengbo Jiao, Shaobo Wang, Jing Zhou, Guoyi Li, Jiayi Chen, Wei Wang, Bing Zhao, Cheng Fang, Xiaoxiao Xu, Hu Wei, Linfeng Zhang, Lin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20755)  

**Abstract**: Process Reward Models (PRMs) are widely used for step-level evaluation of multi-step reasoning in large language models, but existing methods are often limited by weak historical dependency modeling and insufficient global awareness. They typically fail to leverage prior step evaluations for coherent judgment and lack visibility into subsequent reasoning, leading to suboptimal assessment of intermediate steps. To address this issue, we propose the Global Perspective Process Reward Model (GPRM), which jointly models the problem, previous reasoning steps and their evaluations, the current step, and future steps to enable history-aware and globally informed process evaluation. We further introduce a structured evaluation framework along four dimensions---look back, look ahead, self-check, and goal alignment---to assess prior consistency, future plausibility, step validity, and task alignment. Based on this framework, we construct datasets for supervised fine-tuning (SFT) and reinforcement learning (GRPO). Experiments show that, despite using less training data and smaller model sizes, GPRM outperforms strong baseline models by 7.8 and 5.4 points on PRMBench and ProcessBench, respectively. GPRM also generalizes well across tasks, surpassing the GPT-4.1 baseline on Agent Error Bench, improving strong baselines by around 4 points under test-time scaling (TTS), and achieving competitive performance on PPE. 

---
# Can LLMs Give Good Product Advice? A Framework for Diagnosing Failures 

**Authors**: Assaf Hallak, Yftah Ziser, Gal Chechik  

**Link**: [PDF](https://arxiv.org/pdf/2611.20734)  

**Abstract**: Evaluating product advice from a language model requires assessing whether the recommendation meets the user's needs and identifying where the model falls short when it does not. However, matching a product mentioned in an online reply cannot answer either question on its own: several products may be suitable, and the recorded answer may itself be poor advice. When shown a community-listed alternative alongside their own sampled car recommendation, six models prefer the alternative in 1.9 to 6.5 percent of eligible disagreements, consistently across both source-blind presentation orders. We propose a diagnostic framework that separates the user's stated needs, the model's added assumptions, candidate generation, filtering, and final choice, with 19 possible failure modes and the evidence needed to assess them. We apply this framework to product-advice data and to a separate product-selection task with externally assigned relevance labels. The experiments distinguish naming a community-listed product from selecting it and show how answer extraction and candidate-list construction affect the outcome. 

---
# MemEmo: Evaluating Emotion in Memory Systems 

**Authors**: Peng Liu, Zhen Tao, Jihao Zhao, Ding Chen, Yansong Zhang, Cuiping Li, Zhiyu Li, Hong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.20733)  

**Abstract**: Memory systems address the challenge of context loss in Large Language Models during prolonged interactions. However, as an essential tool for warm companionship and personalized services, the effectiveness of memory systems in processing affective information remains underexamined. To bridge this gap, we propose an emotion-enhanced memory evaluation benchmark to assess the performance of mainstream and state-of-the-art memory systems in integrating and harnessing emotional information. First, we design a benchmark called \textbf{MemEmo} to assess the affective reasoning and tracking capabilities of memory systems across four dimensions: emotional information extraction, emotional memory updating, emotional memory question-answering, and emotional conflict detection. Second, we construct the \textbf{H}uman-\textbf{L}ike \textbf{M}emory \textbf{E}motion (\textbf{HLME}) dataset to provide an empirical foundation for scrutinizing memory systems. Finally, We evaluate five representative memory systems on the HLME dataset, reporting 11 performance metrics across four dimensions. Experimental results reveal that aggregating scores for all tested memory systems is up to 45\%, far below 100\% scale, exposing a profound disparity between memory system and affective intelligence. 

---
# ANCHOR: Cluster-Grounded 5W1H Extraction via Large-Small Model Collaboration 

**Authors**: Hongyu Jiang, Chenang Cai, Lei Gao, Zhao Kang, Ling Tian, Jinchuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20728)  

**Abstract**: Deep semantic understanding of events requires recognizing related events and distinguishing why an event occurs from how it is carried out. 5W1H extraction provides a structured foundation for such understanding by organizing event information around six fundamental questions. However, single-event slot filling provides limited support for these distinctions. We propose ANCHOR, a cluster-grounded framework that combines a large language model (LLM) with a small language model equipped with discriminative heads for accurate 5W1H extraction. The LLM proposes candidate events and generates their initial 4W records, which guide the selection of an event cluster and the subsequent anchoring of its main event. A shared document encoder provides evidence for joint 4W refinement, and two separate sets of boundary and role-scoring heads then extract why and how under event-specific constraints, respectively. Document representations are reused across events and stages, and unsupported roles can remain empty. Following the same principles, an agent-and-skill annotation workflow constructs a benchmark of 26,110 documents from ten source corpora covering three languages and seven domains. We select nine of the resulting datasets to evaluate ANCHOR against strong baselines using four open-weight LLMs. ANCHOR achieves competitive performance. The datasets and code are available at this https URL. 

---
# IndustrialKnowEval: A Benchmark for Evaluating Multi-Level Industrial Knowledge in LLMs 

**Authors**: Haiteng Wang, Haoze Li, Jingwei Guo, Jingbo Zhou, Si Liu, Lei Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.20695)  

**Abstract**: Large language models (LLMs) are increasingly explored for industrial tasks, yet few benchmarks evaluate their mastery of industrial knowledge across different types of cognitive demand. We introduce IndustrialKnowEval (IndKnow), a benchmark for industrial knowledge and engineering question answering. It contains 43,282 samples covering 21 task types across seven industrial sectors, including aerospace, electric power, and metallurgy, organized into four levels of cognitive demand: memory, comprehension, reasoning, and engineering application. Enterprise-based domain specialists and experienced researchers participate in all stages of dataset construction and quality control. We conduct a series of experiments on 18 representative LLMs, with domain experts deeply involved in the evaluation. Building upon IndKnow, we propose the Industrial Cognitive Capability Gap (ICCG) metric to characterize the performance differential between foundational industrial knowledge and higher-order reasoning and application, and we analyze it alongside absolute performance to provide a more complete view of model capabilities. We anticipate that IndustrialKnowEval will support the benchmarking of LLMs in industrial scenarios and promote the development of robust industrial intelligence. Code is available at: this https URL 

---
# Fast at the Frontier, Wrong at the Fallback: A Commit-Level Analysis of Threshold Decoding in Diffusion Language Models 

**Authors**: Roman Makarov, Zhe Xu, Jian Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20693)  

**Abstract**: Diffusion language models (DLMs) can write several tokens per forward pass. Open-weight DLMs commit, in any order, every masked position whose confidence passes a threshold. If none does, a fallback commits the most confident one, and the commits are never revised. Wide windows let the decoder commit far ahead of the text, which should speed up decoding but often costs accuracy. It is unclear where the model is confident and which commits cause errors. We trace every commit of four off-the-shelf DLMs on math, code, translation and summarization, and rerun failing answers with one step changed. Most tokens become confident only near the frontier, the leftmost masked position, so committing only there, still several tokens per step, is about as fast as committing anywhere. The accuracy loss comes from committing tokens before the text to their left, and committing one token per step does not reduce it. The worst case is an early end-of-sequence token that cuts the answer short. Fallback commits are 11-35% of commits but start 61-72% of the failures that left-to-right decoding avoids. Changing only the step where such a failure starts repairs it more often than changing a similar earlier step. Committing only at the frontier recovers most of the lost accuracy on all four models. Our code is available at this https URL. 

---
# LEAF-OPD: Learning to Forget from Privileged Contexts via On-Policy Distillation 

**Authors**: Zirui Pang, Zhijie Deng, Yuancheng Wang, Qihan Yu, Xin Li, Jiaheng Wei, Zixin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2611.20674)  

**Abstract**: LLM unlearning aims to erase targeted knowledge from large language models while preserving their utility. However, many existing methods suppress target knowledge without explicitly teaching the model how to respond to queries about it. We propose LEAF-OPD (LEArning to Forget from Privileged Contexts via On-Policy Distillation), in which a frozen teacher provides token-level supervision on student-generated responses. On forget data, the teacher uses privileged contexts (instructions not to reveal knowledge about forget targets) to guide forgetting. On retain data, the teacher provides supervision without privileged context to preserve model utility. The student operates without privileged context in both cases, learning how to respond when encountering knowledge that should be forgotten. A key challenge is that the teacher often provides effective supervision only at the first few tokens of student trajectories. We introduce Supervision Signal (SS) to measure corrective supervision over a specified token interval, evaluating the entire trajectory (SS1) and its second half (SS0.5) to assess overall strength and late-stage persistence. When supervision is weak, lightweight SFT strengthens the teacher's context-conditioned refusal behavior, substantially increasing both scores. We evaluate LEAF-OPD on three widely used benchmarks: TOFU, RWKU, and MUSE-News. On TOFU, LEAF-OPD achieves the best Forget Quality, Model Utility, and Forget ROUGE-L while maintaining competitive retain performance. On RWKU, it outperforms the second-best method by 34% in Forget Quality while maintaining competitive Retain Quality. These results demonstrate the effectiveness of privileged-context supervision with on-policy token-level distillation for LLM unlearning. 

---
# MetaAct: Self-Evolving Executable Action Constraints for Embodied Language Agents 

**Authors**: Jiaxing Zhao, Hanwen Wang, Di Liu, Shaoteng Liu, Jing-Ran Su, Yanheng Li, Qian Zha, Jiaxi Li, Hongbin Xie, Haoran Zhang, Xuan Song, Haoxuan Che  

**Link**: [PDF](https://arxiv.org/pdf/2611.20655)  

**Abstract**: Self-evolving language agents have emerged as a promising paradigm for continuously improving agent behavior through experience. However, most existing approaches evolve textual artifacts, which can influence action preferences but cannot reliably enforce action admissibility in embodied environments with discrete and often irreversible actions. We propose MetaAct, which shifts self-evolution from textual guidance to executable action constraints by synthesizing GEAR (Grounded Executable Action Rules) from execution feedback. Each rule is represented as a typed specification, compiled into a deterministic guard, and validated offline through trajectory replay without additional model calls or environment interactions. Across four benchmarks, MetaAct improves task performance; on ScienceWorld and WebShop, violations drop from 70% to 0% and 40% to 1%, with score gains of +10.1 and +8.0. Our code is available at this https URL. 

---
# Safety-Preserving Medical LLM Fine-Tuning via Bilevel Data Selection 

**Authors**: Ruoxuan Shi, Sheng Yang, Zhengxing Su, Xiaoyang Hou, Yating Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20643)  

**Abstract**: As general large language models (LLMs) are increasingly adapted for patient-facing medical applications, improving medical capability while controlling patient-facing risks remains a critical challenge. Large-scale medical instruction data are essential for acquiring domain knowledge through supervised fine-tuning (SFT), but unsafe responses can undermine existing safety alignment. To address this problem, we propose Scalable Bilevel Data Selection (S-BDS) for safety-preserving medical SFT. By controlling selection-weight concentration, S-BDS enables reliable sample ranking over large candidate pools and identifies training examples that improve medical capability while controlling patient-facing risks. We combine clinician-validated safe data and selected data with unscreened medical instruction data for SFT, and jointly evaluate medical capability and patient-facing safety on HealthBench and PatientSafetyBench, respectively. Experiments across Llama and Qwen models show that S-BDS achieves the highest safe rates among all compared methods, outperforming the strongest baselines by 4.64 and 3.00 percentage points, respectively, while maintaining competitive medical capability. Further experiments show that varying the safety-oriented weight during SFT controls the trade-off between medical capability and patient-facing safety. 

---
# Same Papers, Different Verdicts: Tracing Version Drift in LLM Peer Review 

**Authors**: Jizhou Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.20608)  

**Abstract**: Venues and benchmarks read LLM review scores as measurements of a paper, while providers replace their models within months. We audit twenty frontier models from OpenAI, Google and Anthropic on ICLR 2027 submissions while they were under review, with tests registered before the decisions. Between an earlier and a later version of Claude Opus and of Gemini, the share of papers placed on the accept side falls by 52 and 53 percentage points on 183 and 60 randomly drawn submissions, and by 43 to 79 points in six further paired comparisons, with no paper moving up in any of them. Stating a 30% acceptance rate in the prompt leaves a gap of 79 points between the two Opus versions. When score-stripped critiques are swapped between versions, the later critiques alone lower the accept share by 35 points for Opus and 51 for Gemini; for Opus the drop is 35 points under a third-party scorer and 24 after a rewrite that keeps each technical point and removes evaluative wording. On 57 papers, the percentiles of Opus 4.6 and 5.5 correlate at 0.73 while their accept shares differ by 44 points. On 99 to 119 submissions, the self-estimated percentiles of three models correlate with two human reviewers more strongly than a third reviewer does (point estimates 0.17 to 0.21, 95% intervals between -0.04 and +0.42). All twenty models give a 2 or a 3 in 473 of their 475 reviews, against 60 of 72 human reviews. Retrospective tests on decided papers can draw on memory. Claude Opus 5.5 names the true first author of 27 decided ICLR 2026 papers from their titles alone, with no wrong name, and Claude Opus 4.6 accepted all 11 decided papers it said it recognized, against 5 expected. Thresholds on LLM-reviewer scores therefore have to be validated against human decisions for each model version; our tests against the ICLR 2027 decisions follow on 16 December 2026. 

---
# Invariant-Aware Harness Enforcement to Protect LLM Agents from Silent API Failures 

**Authors**: Sajib Acharjee Dip, Brian Ouyang, Lala Motlhabi  

**Link**: [PDF](https://arxiv.org/pdf/2611.20595)  

**Abstract**: LLM agents answer questions by calling APIs and trusting the results. An API can fail silently. It ignores part of a request, such as a misnamed filter, yet returns HTTP 200 and a plausible but wrong result. Current fixes, such as checklists, reviewer models and LLM judges, ask the model to catch it. But models often see the problem and answer anyway. We present SilentGuard, which moves the decision from the model to the harness running tool calls. It tests each result against invariants that any correct API must satisfy and withholds values that break one. We find silent failures in 11 of 15 public research APIs, with counts inflated up to 691 times. Current fixes leave over 70% of answers wrong, and agents report 77.4% of counts a judge flags. Across 7 open models, SilentGuard cuts wrong answers on its target faults from 74.0% to 1.7%, with no loss on fault-free tasks. On three frontier models they fall from 70.2% to 0.5%. A warning from the same check leaves 70.1% wrong, so enforcement is the key. On new fault types, the harness alone cuts wrong answers from 61.1% to 24.5%. It avoids wrong answers by abstaining rather than recovering. 

---
# Language Models Meet Vibration Testing for LEO Constellation Satellites with Calibrated Response Prediction from Few-show Sweeps 

**Authors**: Jun Wu, Deming Li, Zhenming Zheng, Honghao Yue, Fei Yang, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.20579)  

**Abstract**: Large language models (LLMs) increasingly drive engineering analysis end to end, turning textual records into the inputs of numerical programs, but a plausible and wrong value can pass silently into a decision. We propose deterministic validation gates for LLM-driven engineering analysis, a general paradigm in which a typed configuration produced by LLMs is consumed only after it passes deterministic checks on units, physical consistency, verbatim quotes, agreement between independent models, consistency with measurement exports and the applicability of trained models. The gates target hallucinations from two directions, since quote checking catches fabricated content while agreement between models catches errors of omission that quote checking misses. We instantiate the paradigm on sine vibration acceptance testing for low Earth orbit mega-constellation satellites, where DeepSeek V4.1 Flash and Qwen-plus read Chinese test records and extract a configuration of 76 fields. Fact-level F1 ranged from 0.98 to 1.0 on five input variants, with all repairs frozen on development inputs and evaluated on held-out inputs. With all gates, 20 of 20 configurations produced notching results bit for bit identical to the gold run, and all 17 real and injected errors in consumed fields were stopped, of which 16 would have run without gates. Retraining the bound model on a dataset built from an LLM configuration reproduced its results within GPU nondeterminism. We also report a negative result of practical value. Direct LLM curve forecasting only matches linear extrapolation, whereas a small physics-structured network reduces the over-notched bandwidth by 86% to 89% at calibrated coverage c = 0.90, which delineates where direct LLM prediction should give way to a trained model. 

---
# Weight-Space Programming Changes How an LLM Uses Python for Computational Biology 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2611.20566)  

**Abstract**: Weight-space programming allows a language model to edit another model's weights without gradient-based training. We used Claude Opus 5.5 to edit Qwen3.5-0.8B and investigated how the edits changed its use of Python on computational biology problems. We compared four initial edited checkpoints with the original model on 50 test problems across three test seeds, yielding 600 paired evaluations. Strictly correct answers increased from 44 to 77, and technically correct answers, which included correct final values rejected for formatting, increased from 64 to 100. Repeated Python calls decreased from 1,834 to 1,044. Among the 64 pairs in which only the edited response was technically correct, the original model had already computed the correct value in 36 but did not submit a technically correct final answer. The edited model corrected the computation in 13 of the 64 pairs. With six additional checkpoints on the same 50 problems, technically correct answers rose from 96 to 154 across 900 pairs. We explored three selected problems with ten test seeds for each of ten checkpoints. On the melting-temperature problem, correct Python outputs decreased from 60 to 22 while technically correct final answers increased from 0 to 18. Weight editing changed both the calculations Qwen ran and whether it submitted a correct answer. 

---
# Abstention Direction in Large Language Models: Mechanistic Interpretability of Abstention Across Query Ambiguity, Knowledge Limits, and Safety Alignment 

**Authors**: Koki Itai, Jun Hirako, Hiroaki Yamagiwa  

**Link**: [PDF](https://arxiv.org/pdf/2611.20441)  

**Abstract**: Abstention, the withholding of a response by a large language model (LLM) under uncertainty or safety risks, is essential for trustworthy models. The causes of abstention fall into three perspectives: ambiguity in the Query (Q), limits of Model Knowledge (MK), and misalignment with Human Values (HV). However, prior analyses of internal representations have focused on HV, using a refusal direction computed from harmful and harmless queries. No study has analyzed the three perspectives jointly. We therefore extend the refusal direction framework to all three perspectives. Specifically, for each perspective, we construct counterfactual pairs that share a template and differ only in the abstention-inducing factor, and compute an abstention direction by normalizing the difference in mean activations between the two polarities. Activation steering experiments show that intervening in middle layers increases or decreases abstention across all three perspectives. Moreover, steering yields cross-perspective effects, indicating that the abstention directions share a common directional component. 

---
# Augmenting Pretrained Language Models with N-gram Embeddings through Incremental Training 

**Authors**: Shaohua Hu, Xusheng Gu, Haofei Yang, Minghao Shao, Nanhui Zhao, Caoyun Fan, Fangming Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.20415)  

**Abstract**: As a new scaling technique for large language models, the n-gram embedding improves the base ability of a model while adding almost no inference cost. Mainstream research realizes it through joint training of the n-gram embedding with the backbone, which severely restricts the settings in which an n-gram embedding can be used. We therefore propose training an n-gram embedding onto an existing model through incremental training, to extend where the technique can deliver value. Specifically, we explore several settings of the n-gram embedding under incremental training---the learning rate of the embedding, the vocabulary size, and the injection position---and train an n-gram embedding onto the existing model under the best configuration. Experiments show that the best configuration under incremental training does not fully coincide with that under pretraining; moreover, the incrementally trained knowledge is written predominantly into the n-gram embedding, while the general ability of the backbone is retained throughout this training process. 

---
# Refusal Register is Causally Steerable: Decomposing the Refusal Direction into Category-Aligned Components in Gemma-2-9B-IT 

**Authors**: Janhavi Khindkar, Nataliia Povarova  

**Link**: [PDF](https://arxiv.org/pdf/2611.20405)  

**Abstract**: Prior work shows that refusal in instruction-tuned language models can be ablated by intervening on a single residual-stream direction (Arditi et al., 2024), and more recent work finds multiple refusal-mediating directions and concept cones (Wollschläger et al., 2025). We ask a different question: do refusal-related directions carry information about the register of refusal language - how the model refuses - and not only whether it refuses? In Gemma-2-9B-IT we identify two SVM-derived linear directions (B1 for harmful-content vs. neutral, B2 for self-reference vs. neutral) at layer 20 with cos(B1, B2) = 0.252 (about 75 degrees, substantially non-parallel). Adding the harmful-vs-neutral difference-of-means direction (cosine 0.68 with B1) to the residual stream of self-reference prompts at alpha = 2.0 shifts content-safety vocabulary from 2.55 to 10.37 hits per 100 tokens, while a magnitude-matched random direction produces 1.13-2.16; the model continues to refuse but switches register from meta-style ("as a language model...") to content-style ("this would be harmful"). The same register-injection effect reproduces qualitatively in Gemma-2-2B-IT at layer 12 (content vocabulary 1.72 -> 6.37 at alpha = 1.0, vs. random control 1.35), with a narrower coherent-injection window than in 9B. The two directions are not symmetric: the B2 -> harmful intervention produces a much smaller register effect, and across ablation, sufficiency, and injection B1's causal effects are consistently larger than B2's. We frame this asymmetry as part of the finding: B1 appears to encode both category detection and refusal-vocabulary information, while B2 encodes primarily category detection. We frame the contribution as identifying category-aligned components related to the Arditi-style difference-of-means direction that play distinct functional roles in refusal register. 

---
# When Should Decision Models Compare Candidates? 

**Authors**: Fangzhi Li, Quan Shi, Wenting Yang, Yueqiao Chen, Jiayu Chang, Shuting Zhao, Zhaowei Xu, Yizhao Chen, Guanchu Wu, Han Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20335)  

**Abstract**: Language-model agents frequently select among candidate actions, such as tools and web elements. Should decision models score candidates independently or allow their scores to depend on other candidates? We investigate this question by comparing three approaches: Pointwise scoring, Setwise scoring, and Candidate Attention. Across five agent-decision benchmarks, Pointwise scoring remains competitive, while Setwise scoring performs better on some tool-selection and binary-decision tasks. Candidate Attention does not consistently improve upon Pointwise scoring, despite allowing cross-candidate interaction. Using two controlled synthetic tasks, we further examine whether models learn to use candidate-set information. With targeted relational training, Setwise models learn candidate dependencies and generalize to shifted inputs, while Candidate Attention shows weaker transfer. Our results show that access to other candidates does not guarantee effective comparison, and that its benefits depend on both task requirements and training. 

---
# Hochschild Cocycle and Universality of Merge 

**Authors**: Matilde Marcolli, Elizabeth Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.20305)  

**Abstract**: The Hopf algebra Markov chain dynamical system, on the Hopf algebra of non-planar binary forests, that models the action of Merge on workspaces in Chomsky's Minimalist Model of syntax, satisfies a universal property, determined by the fact that the Merge operation is a Hochschild 1-cocycle. This universal property has an interpretation in terms of models of the syntax-semantics interface, as it shows that, if the side of semantics possesses a sufficiently rich compositional structure, then that structure is uniquely induced by the syntactic Merge, once the semantic values of lexical items and syntactic features are assigned. We illustrate how this universality works by showing, in particular, full compatibility of this Hopf algebraic formulation with Heim-Kratzer semantics. 

---
# FIDEL: Abugida-Aware Pre-tokenization for Ethiopic Languages 

**Authors**: Gebreslassie Teklu Reda, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2611.20216)  

**Abstract**: Subword tokenizers assume that meaningful linguistic units can be recovered by splitting text at learned boundaries, an assumption better suited to concatenative morphology than to Semitic root-and-pattern systems. The challenge is acute for the Ethiopic languages, whose abugida script encodes consonant-vowel units as single characters and hides consonantal structure. We introduce FIDEL (FIdäl DEcomposition and Linearization), a family of reversible abugida-aware pre-tokenization schemes that expose consonant and vowel-order structure, extend SPLINTER reductions from abjad scripts to the Ethiopic abugida, and evaluate them on three Ethiopic languages under controlled conditions, holding corpus, architecture, budget and protocol fixed so that tokenization is the only variable. We report intrinsic tokenizer statistics for BPE and UnigramLM across vocabulary sizes alongside downstream NLP tasks, including transfer scenarios. We find that FIDEL reduction improves news classification consistently and named-entity recognition at larger capacity, while sentiment analysis performance declines under the more aggressive schemes, and extractive question answering performance is preserved. 

---
# Do We Still Need Fine-Tuned Classifiers? Swapping Typed Decision Models into 24 ACL-Venue Papers 

**Authors**: Marek Šuppa, Jaroslav Kopčan  

**Link**: [PDF](https://arxiv.org/pdf/2611.20191)  

**Abstract**: Typed Decision Models such as TypeSafe's Jev, also called System One or System 1 models, answer typed questions about an input with a probability for each option, and they need no training. Many NLP papers still fine-tune a task-specific classifier, and prompted large language models (LLMs) trail such classifiers on most text classification benchmarks. We ask how many of these fine-tuned classifiers a Typed Decision Model can replace under each paper's own evaluation, and what is needed where it cannot. We swap the predictors of 24 ACL-venue papers for Jev, keep each paper's splits, label definitions and metric, and check the results with Cloudflare's Clef, Microsoft-Decision-1 and OpenAI's GPT-6 Luna Decisions. The resulting 46 English test sets, with their questions and published scores, form the TDM-Gauntlet benchmark. With a zero-shot prompt, Jev directly replaced the paper's predictor on 12 of the 46 test sets, mostly single judgments with plain labels, but rarely where the paper's method was trained on its labels. Where Jev scored lower, we most often attributed the shortfall to an annotation convention that the label name does not state. With 500 training labels plus validation and no user-managed GPU, calibration and GEPA-style prompt optimization closed a median 72% of the gap to the paper's method, about twice what off-the-shelf GEPA closed, for a median $1.66 per search. On the same labels, fine-tuned encoders such as RoBERTa and DeBERTa closed 59%, and LoRA fine-tuning of the open LLM Qwen3.8-27B closed 97%. Fine-tuning remains the most accurate option, but a zero-shot prompt is often enough for plain labels, and prompt optimization closes most of the remaining gap. 

---
# Who Can You Hide Behind? Author-Conditioned Style Transfer and the Evasion of a Commercial AI-Text Detector 

**Authors**: Mohd Ruhul Ameen, M. Saifuzzaman Rafat, Akif Islam  

**Link**: [PDF](https://arxiv.org/pdf/2611.20158)  

**Abstract**: AI-text detectors can be sensitive to changes in writing style. We examined this sensitivity by rewriting 991 AI-generated answers with Gemini, using prompts that included a writing sample from one of 55 named authors. We scored each original and its rewrite with the QuillBot AI detector. The median score fell from 100% to 0%, and the mean fell from 67.9% to 11.2%. Of the rewrites, 66% received a score of exactly 0%. Among the 699 originals scored at least 50%, 71% of rewrites scored below 10% (paired Wilcoxon signed-rank test, p-value below 1e-100). Scores varied across the authors used in the prompts. All rewrites associated with 10 authors scored 0%, whereas the mean score remained above 50% for six authors (Kruskal-Wallis test, p-value below 1e-80). Greater reuse of five-word sequences from the author sample was associated with lower scores, although rewrites with less than 5% reuse still averaged 15.2%. These results describe one detector and one rewriting system, not detectors in general. The study lacks a generic-rewrite baseline, did not measure preservation of meaning, and produced rewrites that were substantially longer than the originals. 

---
# Risk-certified prompt optimization for LLM-as-a-judge guardrails 

**Authors**: Hussein Jawad, Alexandre Landeau, Guy Fleury Iradukunda, Paul Beaumont, Théodore Rezel, Sofiane Schaack, Nicolas J-B. Brunel  

**Link**: [PDF](https://arxiv.org/pdf/2611.20156)  

**Abstract**: LLM-as-a-judge guardrails are often controlled primarily through system prompts, yet standard prompt optimization methods target average performance rather than explicit safety constraints. This paper studies system-prompt selection as a constrained optimization problem: maximize benign pass rate while keeping the missed-attack rate below a user-specified level {\alpha} with finite-sample statistical control. The framework combines LLM-based judging, automatic prompt generation, and distribution-free risk certification. This work instantiates it in two procedures: CPS and CRISP. CPS generates a fixed pool of prompts from training data, then certifies candidates on a held-out evaluation set using simultaneous upper confidence bounds and selects the certified prompt with the highest benign utility. CRISP instead integrates risk feedback into an adaptive search, using hard example-guided rewriting and a reusable-holdout interface to steer generation toward the feasible utility frontier. Experiments on four datasets covering prompt injection, system-prompt leakage, and sensitive-information disclosure show that risk-certified selection consistently respects the target missed-attack level on heldout tests, while utility improves as the permitted risk is relaxed. CRISP is especially useful when fixed-pool search fails to produce certifiable high-utility candidates. 

---
# Centered Ablation: Why Last-Layer Directional Ablation Under-Fires or Collapses, and a Closed-Form Fix 

**Authors**: Quy-Anh Dang, Chris Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2611.20148)  

**Abstract**: Directional ablation removes a hidden state's component along a "behaviour direction" while a language model runs. It is the standard way to switch a behaviour such as refusal off, but it is erratic: on some models it barely fires, on others the text collapses into repetition. We show why, and fix it. At the last residual stream, ablation is exactly a logit bias: a fixed vocabulary footprint scaled by a gate, the cosine between the hidden state and the direction. The gate is measured from the origin, but last-layer states share a large common mode far from it, so a direction that overlaps the common mode fires on benign text and under-fires on its targets. Centered Ablation (CA) projects the common mode out of the direction, in closed form. Across 8 models, CA raises attack success by 29 and 46 points where the standard recipe under-fires, cuts collapse on Gemma-2-9B from 50\% to 1\%, and preserves 12 to 40 points more benign answers on high-overlap models, 39 to 46 more than the affine offset of prior work. Two numbers computed before generation predict, on models the analysis had not seen, which gain faithfulness and which gain attack success, and a collapse index marks when text degenerates. We also show that two standard safety judges score collapsed text in opposite directions, which can fabricate a 46-point effect. 

---
# Rank, Don't Regress: A 279M Text Ranker Beats 2B and 7B VLMs on Referential Urdu GUI Grounding 

**Authors**: Hamza Raziq Khan, Aimal Khan  

**Link**: [PDF](https://arxiv.org/pdf/2611.20145)  

**Abstract**: UI grounding asks a model to find the screen element that a user instruction refers to. Most current systems use a vision-language model (VLM) that reads a screenshot and outputs pixel coordinates. We study this task on Urdu mobile web pages, a setting with no published UI grounding results. Instead of regressing coordinates, we take the candidate elements from the page DOM and train a 279M-parameter cross-encoder (mDeBERTa-v3-base) to pick one. The click point is the centre of the picked element's box, so it is exact whenever the right element is picked. We built a frozen test set of 304 instructions from six Urdu news sites; 149 of them (49%) are referential and never quote the full target text. Test domains were banned from training data before collection started. The ranker scores 73.4% point-in-box accuracy, against 29.6% for UI-TARS-2B and 23.7% for Qwen2.5-VL-7B, both zero-shot. On the referential slice the ranker scores 65.8%, against 30.2% for a string-matching baseline that needs no training. In a controlled comparison at small scale, with training data drawn from the same pages, a coordinate-regression model of similar size scores 1.6%. Adding element geometry as input tokens gives the best variant, at 75.7%. The ranker's errors concentrate in two places: elements that share identical text, and ordinal instructions such as "the fourth link". 

---
# Can decision models extract clinical data from cancer pathology reports? An evaluation of accuracy, calibration, and failure modes 

**Authors**: Indranil Mallick, Paromita Roy  

**Link**: [PDF](https://arxiv.org/pdf/2611.20129)  

**Abstract**: Decision models are a new class of language model that, instead of generating text, choose one option per question from a fixed list and return a probability for every option. They have not yet been evaluated on clinical text. We evaluated two open-weights decision models, Nimble 9B and Tev1 4B, for extracting 25 fields from 1,113 oral cavity cancer pathology reports from a single centre, against a prospectively maintained clinical database. We compared them with two generative models: Qwen3.5 9B, run locally, and GLM-5.3-flash, run through a cloud API. Tev1 reached a mean field accuracy of 0.917 (95% CI 0.913 to 0.920), the same as Qwen3.5 9B (0.918; p = 0.26), and was four times faster (15 s against 61 s per report). Nimble reached 0.862 and GLM-5.3-flash 0.986. Both decision models matched GLM-5.3-flash on simple dichotomous categories. They were weak on measurements binned into categories, on fields whose correct answer was not reported, and on some complex decisions (e.g. involved margins). We found that the main cause of the inferior performance of Nimble's came from asking 25 questions in one request: with two questions per request, its accuracy improved. Pooled over fields, the returned probabilities were well calibrated (expected calibration error 0.022 and 0.026), but they did not flag systematic errors within single fields. A simulated strategy of sending answers with a probability below 0.94 to GLM-5.3-flash raised Tev1's accuracy to 0.982, with 38% of answers escalated. Decision models can extract explicitly stated findings from pathology reports quickly on local hardware and return useful probabilities. They should be validated in the configuration in which they will be used, restricted to the fields in which they are reliable, and combined with other methods for numeric values. 

---
# A Corpus-Driven SentencePiece Unigram Tokenizer for Sindhi: Design and Benchmark Against Fourteen Existing Tokenizers 

**Authors**: Ali Nawaz, Muqadas Fareed, Soonh Sadaf  

**Link**: [PDF](https://arxiv.org/pdf/2611.20126)  

**Abstract**: Sindhi is a low-resource Indo-Aryan language written mainly in an extended Perso-Arabic script. Several Sindhi tokenizers have been released on public model hubs, but they have not been compared with one another on shared data. We train SentencePiece Unigram tokenizers with byte fallback at four vocabulary sizes (32k to 150k) on about 1.10 million Sindhi sentences and a smaller, repeated Urdu, Persian and Arabic slice, from four public sources, and compare them with fourteen existing tokenizers on four evaluation sets: FLORES+, Sindhi Wikipedia, a news-derived sentiment corpus, and a contemporary news corpus. The metrics are fertility, single-token retention, UNK rate, byte-fallback rate, and round-trip character loss. At 64k, our tokenizer averages 1.147 tokens per word, against 1.208 for the best existing tokenizer and 1.512 for XLM-RoBERTa, and keeps 88.6% of words as single tokens. On fertility, single-token retention and UNK rate it is best or tied for best in all four evaluation sets at 64k and above; at 32k it trails one baseline on FLORES+. Margins over the best Sindhi-adapted baselines are small (0.002 fertility on FLORES+ at 64k), and we report no significance tests. A round-trip check finds no character loss for our tokenizer and silent character deletion (up to 0.85% of characters) in two existing ones; seven others lose less than 0.1%. We evaluate intrinsic metrics only, so effects on downstream tasks remain untested. Code and tokenizer files are public (DOI: https://doi.org/10.5281/zenodo.23962474). 

---
# FRAAC: A French Multimodal Large Language Model and Dataset forPictogram-Based Augmentative and Alternative Communication 

**Authors**: Alexandre Audibert, François Portet, Benoit Favre, Maja Jonck Hjuler  

**Link**: [PDF](https://arxiv.org/pdf/2611.20094)  

**Abstract**: Pictograms are a core modality in augmentative and alternative communication (AAC), yet recent multimodal large language models (MLLMs) cannot reliably interpret or produce pictogram sequences. Two obstacles explain this gap: the absence of a standardized pictogram grammar and the scarcity of parallel text-pictogram data. We introduce Retrieval-Augmented Pictogram Selection (RAPS), an annotation method that refines a rule-based transcription system's output with LLM-based concept extraction and retrieval over a pictogram vocabulary, handling cases where hand-written rules fail (named entities, numbers, polysemous words) without requiring new rules. Applied to French, RAPS yields a synthetic corpus of 1.4 million parallel pairs, of which 300k are publicly released. In a human ranking study where annotators order four pictogram transcriptions of the same sentence, RAPS outperforms the rule-based state of the art. Moreover, we show that standard reference-based metrics separate systems of clearly different quality but fail to rank strong systems against each other. Finally, we build an instruction-tuned model that supports interaction in several AAC modalities: it accepts pictogram or text input and answers in pictograms, speech, or text. 

---
# The Silent Cost of Conversation History in LLM Reasoning 

**Authors**: Ali Şenol  

**Link**: [PDF](https://arxiv.org/pdf/2611.20071)  

**Abstract**: Large language models (LLMs) in multi-turn information systems accumulate their own earlier replies and a growing task state. We asked whether this conversation history lowers accuracy, and whether process-level quality metrics warn before it does. Each analysis was fixed before its outcomes were computed. In Study 1, on 48 sharded math tasks, retaining its replies lowered final success by 0.21 for GPT-4o-mini and 0.29 for Mistral-7B (Holm-adjusted p = 0.026 and 0.004); Llama-4 Scout showed the same direction, not significant. Study 2 introduces CDR-Stream, 100-turn state-tracking dialogues with an exact answer per turn and per-model difficulty calibration. With the full history, accuracy fell from about 0.6 to 0.1 within twenty turns at matched difficulty, whereas a solver-verified state summary kept it 0.52-0.59 higher (Holm-adjusted p less than 0.001). Consistency and logical coherence fell far less than accuracy, and stability rose. None of these metrics, a paired readiness measure or calibrated single-stream drift detectors gave robust, model-general early warning, although in Study 2 the repeat-based metrics were complete only for GPT-4o-mini. In the monitors we tested, a single 100-turn stream had power only for very large drops. Systems built on LLMs should keep task state outside the transcript and monitor across sessions. 

---
# Mandarin Wh-Indefinites Follow Nonveridicality in Multilingual LLMs, Unlike English any 

**Authors**: Guofang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20062)  

**Abstract**: Multilingual LLMs learn mostly from English. Do they carry English polarity licensing over to other languages? English any is licensed by downward-entailing (DE) contexts, Mandarin wh-indefinites such as shenme 'what/something' by nonveridical ones, and the two classes come apart: perhaps licenses shenme but not any, while only licenses any but entails existence. We test this dissociation in a pre-registered study of twelve open LLMs (0.5B-9B, four families), using 48 matched bilingual frames in 21 contexts and a log-probability difference-in-differences normalised by negation. All 11 models that pass the measurement gate license Mandarin wh-indefinites under possibility, necessity and evidential adverbs, at 0.42-2.33 times the strength of negation, while English any gets nothing in the same contexts. An exploratory extension to Qwen2.5-32B shows the same. The English side is weaker: any under only, first and regret is licensed but at a fraction of negation, so only 3 models show the full double dissociation in sentence probabilities. When prompted to choose between sentences, most gated models show it. In exploratory directional ablations, a direction that carries English licensing, held-out classes included, does not reach Mandarin licensing in most models. The models do not simply apply the English rule to Mandarin. 

---
# A Dataset for Misogyny Detection in Polish Social Media: Transferability of Hateful Speech Detection from General to Specific Types of Hate 

**Authors**: Ewelina Gajewska  

**Link**: [PDF](https://arxiv.org/pdf/2611.20044)  

**Abstract**: Harmful language detection is commonly evaluated on broadly defined hate speech and toxicity benchmarks, yet it remains unclear whether models that successfully detect harmful language in general can transfer this capability to more specific forms of hate. We investigate this problem for misogyny detection in Polish, an area for which dedicated resources remain scarce. We introduce a new dataset of Polish social media comments annotated for misogyny, gender-directed hateful speech. Using this resource, we first assess the reliability of LLM-assisted annotation and then evaluate general-to-specific transfer by comparing models trained on general harmful language data, safety-specialized models, and models fine-tuned directly for misogyny detection. Our results show that strong performance on general harmful language detection does not transfer uniformly to misogyny; safety-specialized models demonstrate substantially stronger generalization ability. The study contributes a new Polish resource for gender-directed harmful language and demonstrates that the scope and granularity of prior safety training are important factors in transferring harmful language detection from general to specific forms of hate. 

---
# Offline-First Quantized AI: Design and Evaluation of an On-Device Maternal-Health Assistant for Low-Connectivity Environments 

**Authors**: Okenla AbdulQahhar Abimbola, Maurice Abraham John, Adepoju Jeremiah Moyondafoluwa, Omolaiye Samuel, Oko-Oba Sandra Chioma, Owoo Oluwamayowa Emmanuel, Thanni Adewuyi, Lily Aimas-Love, Benson Eniola, Abdulquddus Ajibade, Sylvia Essessien, Anyanwu Princess Emmanuella, Adereni Abiodun  

**Link**: [PDF](https://arxiv.org/pdf/2611.20012)  

**Abstract**: Offline language models could extend access to maternal-health information where internet connectivity is unreliable, but deployment requires both clinically appropriate responses and feasible local execution. This study presents an offline Android assistant integrating ready-made quantized language models, a clinically validated local knowledge package, conversational context, and model-initiated local retrieval. Gemma 4 E2B and Qwen3.5 0.8B were compared across 45 conversations and 312 turns per configuration, including a dedicated offline agentic benchmark. Gemma achieved higher response-quality, context-retention, and safety scores; Qwen offered faster generation and lower memory use. Clinician assessment of 150 Gemma responses across nine clinical domains yielded a mean quality score of 3.80/4, with 147 responses meeting the acceptance criterion. A paired RAG-on/off experiment evaluated 74 questions: retrieval improved 11 responses, left 56 neutral, produced four adverse response changes, and left three pairs unassessable. RAG-on had descriptively higher mean usefulness, safety, correctness, and actionability in the conditional 70-pair subset excluding the four flagged pairs from both conditions. Physical-device testing demonstrated offline responses on a 6 GB Android handset, with a 14.3 s median latency, alongside successful reconnection synchronization in the integrated application. Together, the findings document clinical response quality and practical performance of the integrated offline maternal-health assistant. 

---
# The novelty exposed: ablation, reproducibility and transparency in 40 technical reports of open-weight large language models (LLMs) 

**Authors**: Bahaeddin Türkoğlu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20003)  

**Abstract**: In 2025 alone, 48 new large language model (LLM) architectures were added to the Hugging Face transformers library, and many of them arrive with a technical report that presents the model as new. What is actually new in these models, and is it tested? We put this question to LLM technical reports. We look at three things: what the reference code shows about how new open-weight architectures differ from earlier ones; what 40 randomly sampled technical reports claim as new and whether they support it with ablation studies; and what they disclose for reproducibility and transparency. In the library, 39 of the 48 architectures added in 2025 take at least two of their three core blocks from another model, most often Llama, although part of this reflects how the library writes code. In the reports, 19 of 73 architectural or training novelty claims come with a controlled with/without comparison; 4 of 40 reports state more than one training run; 8 rerun the models they compare against under one setup; and 1 discloses data mixture, token count, and compute together. We also looked for known components given new names: none of five candidate cases hid its origin, though we did not study model names. We propose seven reporting rules for LLM release reports and apply them to two reports. We do not judge model quality, and an untested claim may still be true. 

---
# Anatomy of an Agent Harness: Decisions, Invariants and Corner Cases 

**Authors**: Resham Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2611.19960)  

**Abstract**: An agent harness is the program around a language model that turns its replies into actions: it builds each request, runs the tools the model asks for, decides what the model may do without asking, stores the conversation, recovers from failures, and shows a person what happened. Most of these decisions are made in code and rarely written down, so each harness meets the same corner cases again. This paper writes them down as invariants to test. It gives a system model with trust boundaries; 30 decision cards in six groups, each with options and costs, an invariant, the corner cases that break it, a test and an evidence tag; six cause classes for failures; a test method built on scripted providers, recorded behaviour and planted changes; and a checklist. The cards come from building one desktop harness, Crewroom, the author's own, and from manually testing ten other harnesses and studying open-source harnesses as reference; some patterns in the cards were adopted from those, so the catalogue is not independent of them. Of the 83 invariant clauses, 56 are tagged Tested, 12 Believed and 15 Open. The checklist is applied to the author's harness as one worked example, by its builder with no second rater: of its 30 items, 11 hold, 14 do not and 5 were not tested. It is not a ranking. The paper makes no claims about answer quality or cost or other platforms, and does not evaluate or rank other harnesses. 

---
# The Vocabulary Tax: Safely Trimming Unused Tokens from LLMs for On-Device Use 

**Authors**: Sattaya Singkul  

**Link**: [PDF](https://arxiv.org/pdf/2611.19913)  

**Abstract**: Open large language models (LLMs) and speech models ship vocabularies of up to 262k tokens, and every user downloads the tables those tokens index (the input embedding, an untied output head and, in recent on-device models, per-layer embeddings), whatever languages they need. We call the bytes in rows a deployment never uses the vocabulary tax and measure it, without downloading weights, for 1,244 generative and speech-recognition models on the Hugging Face Hub. For a Thai and English deployment it is a median 4.4% of model bytes below 0.5B parameters and 2.1% above 8B, and 1.93 GB in Gemma 4 E2B: it falls hardest on the small models that edge devices run. Keeping only the rows of the deployment's writing systems is exact. Keeping the tokens seen in the user's own text removes much more, a median 29.0% below 0.5B, but the usual way of doing it breaks BPE tokenizers: it drops the intermediate tokens that merges build on, so new text is split differently 1.9 times as often as the standard check reports. We repair this with merge closure, at a cost of 5.9% more rows, and add a trim certificate: a Good-Turing estimate, from the trimming text alone, of how much new text the trimmed tokenizer will split differently (correlation 0.996 over 300 settings in 22 languages) and of how much more text a target needs. Without test data our method never exceeds a 2% target, while prior keep rules checked on test data exceed it in 9-13 of 44 settings. On six models of 270M to 3B parameters in 11 languages, trimmed models keep their predictions: where the certificate passes, a median 89.7% of rows is removed and bits per character rise by at most 2.0%. The census, the trimming and the certificate are released as an open Python package, and the per-model tax for 41 languages as an open dataset. 

---
# Attribute or Abstain: Evaluating Evidence-Grounded Retrieval on the Low-Resource African Business Web 

**Authors**: Bakary Gibba, Muhammad Abba Mandara, Karim Ismail Abd-Ellah Mohamed  

**Link**: [PDF](https://arxiv.org/pdf/2611.19900)  

**Abstract**: Retrieval-augmented systems that research companies are judged by how much they find, yet in markets where public evidence is sparse, a confident answer without support costs more than no answer. We introduce AfriBizClaims, a benchmark of 150 company-research queries across six African markets in English, French and Portuguese, including queries about fictitious companies that a correct system must decline. Each claim a system makes is judged against the page it cites by an AI judge, checked against a second AI judge from another model family and against two human annotators, and abstentions are scored against answerability labels. We compare BM25 over a pooled corpus, standard web retrieval-augmented generation, and a multi-step agent pipeline (Konexium V1), all using the same language model. The two simple baselines are well attributed: 98% of their judged claims are supported by the page they cite, and they decline every field of every fictitious company. Counting claims about a company other than the one asked about as errors, the agent pipeline's per-query precision is 68%, and it never answers a question about a named company correctly: it substitutes other firms, including for 5 of the 24 fictitious names, and it is the least stable between runs and the most expensive. In the pipeline we evaluated, more steps between the query and the evidence did not buy more grounded answers. We will release the queries, system outputs, labels and short evidence excerpts. 

---
# Bioformer-RetroMAE: retrieval-oriented pretraining of a compact biomedical language model for literature search 

**Authors**: Jialu Li, Li Fang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19862)  

**Abstract**: Dense retrieval is increasingly used for biomedical literature search and retrieval-augmented generation (RAG). Most biomedical bi-encoders are built on BERTBase-sized encoders pretrained with masked language modeling (MLM). These encoders are costly for corpus-scale indexing and latency-sensitive query encoding, and MLM does not explicitly train the [CLS] vector that a bi-encoder uses to represent a whole text. We present Bioformer-RetroMAE, a compact (43M-parameter) retrieval-oriented biomedical encoder obtained by continually pretraining Bioformer8L} with RetroMAE on approximately 32 million PubMed titles and abstracts. After bi-encoder fine-tuning with BM25 hard negatives, we evaluated it on BioASQ Task 12B test questions by retrieving from all PubMed abstracts. Bioformer-RetroMAE outperformed BioBERT and PubMedBERT bi-encoders fine-tuned under identical settings on all metrics, despite having about 40% of their parameters and producing 512-dimensional rather than 768-dimensional vectors. Compared with PubMedBERT, it improved mean average precision (MAP) by 37.0% and Recall@100 by 29.1%. Compared with the original Bioformer8L}, RetroMAE pretraining improved MAP by 53.6% and Recall@100 by 32.8% without changing model size or inference cost. In hybrid retrieval, fusing Bioformer-RetroMAE with BM25 through weighted reciprocal rank fusion improved MAP and NDCG@10 over BM25 alone by 4.9% and 4.6%, respectively, larger gains than those obtained by fusing BM25 with PubMedBERT or BioBERT. 

---
# Bridging the Terminology Gap: A Data-Centric Alignment Framework for Multilingual Translation in Cross-Border Corporation 

**Authors**: Jinsheng Wei, Song Cheng, Bo Xiao, Ke Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2611.19858)  

**Abstract**: As globalization accelerates the demand for seamless cross-border enterprise collaboration, multilingual translation remains bottlenecked by insufficient domain-specific terminology, rigid terminology translation, weak instruction-following capabilities, and high inference latency in long-instruction scenarios. To overcome these barriers, we propose Bridging the Terminology Gap (BTG), a data-centric framework. First, BTG mines cross-lingual term alignments from domain documents and leverages LLMs to synthesize terminology-rich parallel corpora. Second, a hybrid filtering strategy combining heuristics and XCOMET scoring ensures high data fidelity. Third, a two-stage training paradigm employs supervised fine-tuning (SFT) for targeted terminology injection, followed by reinforcement learning (RL) to refine output preferences, effectively balancing domain knowledge integration with model reliability. Extensive offline evaluations (e.g., XCOMET, GEMBA, Acc@T) and online human assessments demonstrate significant improvements in terminology accuracy and instruction-following capabilities, validating the effectiveness of our framework, which is now deployed on our cross-border collaboration platform. 

---
# NoLBERT-Calls: Reading Earnings Calls Without Hindsight 

**Authors**: Kevin Foley, Jonathan Hartadi, Shivesh Prakash, Hino Wirachapong Suwanphibun, Swapnil Vatsal  

**Link**: [PDF](https://arxiv.org/pdf/2611.19849)  

**Abstract**: Language models are now a standard tool for measuring what managers say on earnings calls, but the encoders behind these measures were usually pretrained on text from the same years a backtest covers. Holding out a test period does not remove that hindsight, because it sits in the features. We adapt NoLBERT, an encoder pretrained only on text published between 1976 and 1995, to earnings calls without letting it read anything dated after 31 December 2023. The calls supply the training signal themselves: every analyst question is followed by management's answer, and the model learns to pick that answer out from the other answers of the same call. We test on every US call from January 2025 to October 2026 (20,323 calls, 254,575 questions) and make every modeling choice on calls from 2024. The adapted model, NoLBERT-Calls-2023, ranks the true answer first for 78.0% of questions (95% interval 77.8-78.2; chance 8.0%), against 26.3% before adaptation. It is 1.0 point ahead of a call-adapted ChronoBERT, 3.4 points ahead of a call-adapted FinBERT, and 18.7 and 19.8 points ahead of the best off-the-shelf retriever and BM25. A year-by-year test over 2013-2026 finds no look-ahead premium: relative to a model adapted only through 2012, the released model does no better on the years it was trained on than on years it never saw. The price of the cutoff is compute: under one shared 150-minute recipe NoLBERT adapts least well of four encoders, and its lead comes from a longer, staged adaptation of about 13 GPU-hours. 

---
# Grounding Language Models in Speech Does Not Fix "AI Flavor": Calibrated LLM Judges, Goodhart Traps, and a Preference-Tuning Fix 

**Authors**: Dongwu Qian  

**Link**: [PDF](https://arxiv.org/pdf/2611.19840)  

**Abstract**: Chinese outputs of large language models carry a perceptible "AI flavor": formulaic sentence frames, stacked written-register connectives, repetition loops, and layout leakage. A popular hypothesis blames text-only training for the missing inner-speech layer of human writing; injecting speech or prosody signals should therefore reduce it. We test it with controlled interventions on a 0.6B model; all three fail: (i) phoneme-interleaved continued pretraining yields no spokenness gain under text-matched controls while raising repetition (+0.13 [+0.03, +0.23]); (ii) prosody-plan conditional generation reaches an adherence of only 0.257 against a pre-registered 0.50 gate; (iii) a rule-based reward built on TTS prosody-feature distance is imperceptible in double-blind evaluation by both humans (8/16) and a calibrated LLM judge (9/16).
A root cause is evaluation: single-rater blind evaluation at n~20 only detects extreme (>=85%) preferences. We therefore build a calibrated LLM-judge protocol: the judge passes catch trials on known-answer pairs (16/16; single-text 22/24, same-topic pairs 24/24, versus 0.75 for the human rater), then annotates with two independent judges under A/B swapping, counting disagreement as a tie. Using the judge's agree labels directly as DPO pairs (no reward model), we obtain the first positive result: held-out blind win rate 0.833 [0.699, 0.972], repetition 0.114 to 0.000, unchanged MMLU. We further identify a Goodhart channel: the judge's length preference transduces into output-length collapse via pairwise DPO (median 75 chars vs a 300-char prompt); an SFT-anchor repairs it (w is a style knob: w=0.1 preserves gains, w=0.3 does not). Scaling preference data from 89 to 240 pairs yields no additional judge-visible gain. We release RA-Eval v0.1: a three-layer Chinese AI-flavor evaluation framework with 382 judge-annotated preference pairs. 

---
# EfficientSkill: Learning Skills for Token-Efficient Multi-Turn Agents 

**Authors**: Deep Shah, Sanket Badhe, Nehal Kathrotia, Priyanka Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2611.19823)  

**Abstract**: Reflective skill optimizers improve multi-turn LLM agents by revising instructions from failed and successful rollouts, yet accumulating edge-case rules bloats prompts, inflates token use, and pushes agents into long, error-prone loops. We introduce EfficientSkill, a two-step framework that first discovers a reliable skill without token penalties, and then warm-starts accuracy-guarded pruning from that skill to compress instructions, bound agent turns, and offload reusable code into workspace utilities with zero implementation prompt cost. Across four frontier, mid-tier, and open-weight models on MATH-500, ALFWorld, and SpreadsheetBench, EfficientSkill cuts output tokens over unconstrained EvoSkill on 11 of 12 configurations (by up to 80%) while raising mean accuracy on every benchmark, including +14.3 and +17.5 percentage-point gains on SpreadsheetBench with Gemini 3.8 Flash and Grok 4.1 Fast by replacing stalled loops with compact workflows. By shrinking both generated tokens and replayed history, EfficientSkill lowers total token usage and serving cost across every evaluated configuration, with or without full-prefix KV caching. 

---
# Machine interpreting competitiveness: a discourse-sensitive comparison of meaning and target-language quality in English-Chinese simultaneous interpreting 

**Authors**: Yuanyuan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2611.19802)  

**Abstract**: Machine interpreting (MI), or AI-enabled speech translation, is increasingly visible in professional and public-facing communication, yet its performance relative to human simultaneous interpreters remains insufficiently understood. Existing studies are often limited by a single discourse mode and evaluation frameworks that are not well aligned with interpreting as a discourse-level communicative activity. The present study compares five commercial MI systems with 26 human interpreters (14 professionals, 12 advanced students) on English-to-Chinese simultaneous interpreting across planned and unplanned discourse, focusing on meaning and target-language quality. In these materials, MI performance differed sharply between the two discourse modes. In the planned formal speech, the strongest system became highly competitive on meaning, entering the upper human range, and a second system also performed strongly. In the extemporaneous interview, however, no MI system approached the stronger human tiers. Code-level analysis further shows that similar overall scores often masked different error profiles. Human interpreters were more likely to preserve the broader communicative trajectory through selective omission, restructuring, and contextual inference, whereas even the best MI systems were more vulnerable to semantic distortion, discourse-tracking failures, and context-insensitive interpretation. On target-language quality, the human advantage proved more stable than on meaning. Unintelligible or nonsensical output and disfluency carryover emerge as particularly salient weaknesses in machine performance, especially in spontaneous discourse. Overall, the findings are consistent with a discourse-sensitive account of where MI appears competitive, where it remains limited, and why these distinctions matter for interpreting practice, interpreter education, and future MI development. 

---
# Logical AI OS: A Training-Free Governance Layer Outside the Model for Instruction Following -- Benefits, Costs, Failures, and the Calibration of the Measuring Instrument Itself 

**Authors**: Joe Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.19767)  

**Abstract**: Small language models often fail explicit user requests (length, number of items, format, required or forbidden words, grammatical person, answer language). We measure a training-free governance layer placed outside the model: it reads the requests with rules, checks the answer mechanically, repairs it when a request is not kept, and hands back the original answer when the repair fails. The loop exists in prior work; our contribution is measuring its benefits, costs and failures, and reporting how the measuring instrument itself was calibrated. On four 4-bit Qwen-family models (3B, 7B, 14B, one 7B reasoning model) and 329 Chinese prompts: (1) answers that keep every request the layer read rise significantly (all comparisons p<0.0001), by 34 to 66 of 329 under the corrected checker and 52 to 92 after a length-control fix; a generic redo message brings no significant gain. (2) Exploratory: on 90 repair-triggered prompts, two AI annotators found the layer produced an acceptable answer where the bare model did not more often than the reverse (49:7; 18:12, not significant). (3) Costs: 43-66% more generation time; whole-answer rewriting makes a sizeable share of usable answers unusable. (4) Failures: rule-based reading reaches about 70% and 84% of a two-annotator ceiling on unseen phrasings; unsatisfiable requests make the reasoning model loop; model self-reports are unreliable (18 of 25 "fixed" items were not fixed). (5) Over half of the original checker's "violated" verdicts on structural requests were wrong; after correction the main result holds, and one comparison is downgraded. Conclusions are limited to this Chinese material, the Qwen family and sizes up to 14B. The governance implementation is not released; data, answers, verdicts and the checker are public. 

---
# Ambiguous Examples Help, But Full Data Wins: Revisiting Dataset Cartography for Extractive Adversarial QA 

**Authors**: Paul K. Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2611.19763)  

**Abstract**: Dataset cartography argues that training dynamics can reveal regions of a dataset that matter differently for learning: easy-to-learn examples aid optimization, hard-to-learn examples often expose annotation problems, and ambiguous examples contribute strongly to out-of-distribution generalization. A striking implication of the original work is that a model trained on only 33% ambiguous examples can outperform full-data training on several OOD evaluations, including a QNLI reformulation of SQuAD evaluated on Adversarial SQuAD. We revisit this claim in a stricter setting: open-ended extractive question answering, where models must predict answer spans rather than classify question-sentence pairs. For each encoder family, we compute clean-source SQuAD training dynamics, train on full, random, easy-ranked, ambiguous-ranked, and hard-ranked subsets, and evaluate robustness on adversarial-only AddSent and AddOneSent rows. Ambiguous-ranked 33% subsets beat random 33% subsets on exact-match accuracy in all 14 model-evaluation cells and on F1 in 12/14 cells. However, full-data training beats ambiguous-ranked training in 12/14 exact-match cells and 12/14 F1 cells, with mean full-minus-ambiguous gains of 3.40 EM and 4.10 F1. The most thoroughly replicated ELECTRA-small panel is also the main exception to the ambiguous-over-random F1 gains. These results show that clean-source cartography remains useful for reduced-data selection, but its OOD advantage does not carry over as a general replacement for full-data training in extractive adversarial QA. 

---
# StaleBench: Isolating Stale Retrieval in Agent Memory, and Pricing the Fix 

**Authors**: S M Mahbub Murshed  

**Link**: [PDF](https://arxiv.org/pdf/2611.19752)  

**Abstract**: An agent that stores facts over time eventually stores several versions of the same fact. Asked a value-agnostic question - "which team does X play for now?" - a similarity retriever cannot prefer the current version, because every version matches the entity the query names. StaleBench isolates this stale-retrieval failure on 892 real updating facts, with currency labels from the source (Wikidata validity years via TempLAMA), never from a retrieval score. Our central result is that retrieval metrics can rank memory policies backwards: a retention policy and its retain-everything control return the identical statement at rank 1 for every test query, yet differ by 29.5 and 65.6 points in reader accuracy on two readers when no date is shown. Contamination@5, the metric we add, ranks them correctly where current@1 and current@5 do not. Withdrawing the lexical anchor that statements carry costs deployed keying mechanisms tens of points of freshness; entity resolution on both the write path and the query recovers most of the loss. Three deployed systems (Mem0, Graphiti, LangMem) and a reimplemented ERASE each handle supersession differently, and no single scoring convention describes all four. We release the data, a frozen evaluator, and reference baselines. 

---
# SG-LegalClarify: A Large-Scale Benchmark for Legal Reasoning and Ambiguity Recognition in Singapore Law 

**Authors**: Ziwei Gu, Michael S. Lee, Drew Rein, Yash Maurya, Vikneswaran Kumaran, Darshini Ramiah, Chenguang Wang, Udari Madhushani Sehwag  

**Link**: [PDF](https://arxiv.org/pdf/2611.19643)  

**Abstract**: Singapore offers a distinctive testbed for legal reasoning by large language models (LLMs). Its legal system combines inherited English common law with jurisdiction-specific statutes, requiring models to distinguish local doctrine from generic common-law intuition. This setting also highlights the challenge of recognizing underspecification in lay legal queries, because the facts material to a legal outcome depend on the governing doctrine. Yet legal benchmarks rarely cover Singapore law or test this form of ambiguity recognition. To address these gaps, we introduce SG-LegalClarify, a large-scale benchmark for open-ended legal reasoning and ambiguity recognition in Singapore law. SG-LegalClarify contains 1,100 lawyer-authored source questions spanning 10 legal domains, including 340 fully specified and 760 underspecified questions. Each underspecified question has a fixed hidden-fact assignment and a fully specified counterpart, yielding 1,860 evaluation instances. Responses are graded with lawyer-authored Issue, Rule, Application, and Conclusion (IRAC) rubrics. We evaluate 19 LLMs, closed-book and with web search, using a multi-turn pipeline in which models decide whether to answer or clarify, and a user simulator reveals hidden facts only when asked. We find the benchmark far from saturated: the best models satisfy only 69.8% of mandatory rubric items closed-book (average 53.8%) and 77.8% with web search (average 63.0%). Models also struggle to recognize when clarification is necessary. We will release the public split of our benchmark, the model outputs on it, and the evaluation pipeline to support future research. 

---
# ROUNDS: Source-Framed Diagnostic Revision after Multi-Turn Clinical Evidence 

**Authors**: Jinrui Fang, Runhan Chen, Yining Wu, Jiawei Xu, Xu Yang, Jian Yu, Wenqi Shi, Ying Ding, Yuji Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19636)  

**Abstract**: Large language models (LLMs) achieve high scores on medical benchmarks, but does this performance translate into reliable diagnosis as a conversation unfolds? In diagnostic conversations, users can provide clinical findings across multiple turns, ask for a diagnosis without supplying answer options, and introduce opinions from others. Models must therefore form diagnoses from the available evidence and decide how to respond to subsequent advice. We introduce ROUNDS, a framework that evaluates these abilities after a fixed sequence of clinical evidence delivered over multiple turns. ROUNDS compares diagnosis with and without answer options, presents the same incorrect diagnosis under clinician and lay framing, and varies support for competing diagnoses in written five-clinician panels. Across seven models and 1,035 cases from five datasets, average accuracy was 76.9% with answer options and 57.0% without them. Clinician-framed misleading advice produced lower final accuracy than lay-framed advice in every model, with differences of 4.6 to 38.2 percentage points. Initially correct diagnoses more often became incorrect answers than abstentions, and many adopted the suggested disease. Conversely, accuracy increased as more panel members supported the correct diagnosis, including when the majority opinion remained unchanged. These findings reveal a gap between medical benchmark performance and reliable diagnosis in conversation: models respond to external opinions, but remain vulnerable to abandoning correct diagnoses without new clinical evidence. 

---
# SemMask: Semantic-Prioritized Progressive Masking for Diffusion Language Models 

**Authors**: Qi Chen, Zifeng Cheng, Zhiwei Jiang, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19560)  

**Abstract**: Diffusion Language Models (DLMs) are designed to enable globally parallel generation. However, in practice, they often exhibit pseudo-autoregressive behavior and a function-word-first bias, which undermines their performance on tasks such as text writing and code generation that benefit from global planning and parallel decoding. To address this, we propose SemMask, a novel fine-tuning paradigm that leverages a semantic-prioritized progressive masking strategy to encourage the model to decode high-information semantic tokens early and globally, thereby enhancing its global planning and parallel generation capabilities. Specifically, we first use the gradients of the original DLM with respect to each token to identify semantic tokens. During masking, these semantic tokens are preserved with priority, establishing a progressive generation order in which semantic tokens are decoded prior to function tokens. Meanwhile, we adjust the loss weight for semantic tokens to further reinforce their early prediction. Experimental results show that SemMask achieves quality comparable to strong autoregressive baselines on both text writing and code generation tasks, while providing superior semantic coherence. Notably, SemMask achieves higher accuracy with substantially fewer decoding steps, indicating that the early establishment of semantic structure improves the efficiency--accuracy trade-off. 

---
# Sada-ASR: A Production Multidialectal Arabic Speech Recognition System 

**Authors**: Mahmoud Khaled, Mohammed Abd-Elnaser, Mahmoud Mohsen, Hazem Ahmed, Abdelrahman Hafez  

**Link**: [PDF](https://arxiv.org/pdf/2611.19512)  

**Abstract**: Callers to Arabic customer-service lines speak their own dialect, while speech recognizers are built and evaluated mainly on Modern Standard Arabic (MSA). A recognizer that mishears the dialect breaks the call before anything else can help. We present Sada-ASR, the speech recognizer of EchoNova's production Arabic voice agent. On a 120-hour test set of MSA and eight dialects, it has the lowest word and character error rates of twelve systems, against eleven from Microsoft, ElevenLabs, Google, OpenAI, Speechmatics, IFM, Cohere, NVIDIA and Audar: 11.69% word error rate against 13.51% for the strongest, Microsoft's MAI-Transcribe-2 (a lead of 0.74 points on clips unaffected by that system's fixed number and script conventions), and 15.52% against 16.96% averaged over varieties. Its largest margins are on Omani (4.30 points), Egyptian (3.78) and Saudi (2.19) speech; it is level on Syrian and loses on Lebanese (24.96% against 23.05%), mostly because it writes Lebanese words in MSA forms and is less verbatim. On six public test sets its average word error rate under the leaderboard's current scoring is 17.17%, below every published figure we found, among them Falcon-ASR (20.92%), Jais-ASR (22.85%) and Audar-ASR-V1-Turbo (23.17%), though none of these states that it used that scoring, so the comparison is indicative. We describe the data pipeline and methodology behind Sada-ASR and release a 20-hour subset of the test set with our scoring code. 

---
# Formalizing Formal Semantics: The Formal Atlas 

**Authors**: Stergios Chatzikyriakidis, Aarne Talman  

**Link**: [PDF](https://arxiv.org/pdf/2611.19509)  

**Abstract**: In this paper we present the Formal Atlas, a system for the study of natural language semantics using proof assistants. Our aim is to formalize semantic accounts,
check their predictions and compare the implemented theories. To this end, we connect source records and formal developments with an interactive application, where
researchers can inspect proofs and modify supported Rocq files. Montague's PTQ and Gallin provide our main cases, followed by dynamic and type-theoretic
comparisons, RSA and distributional semantics. We also discuss what it takes to formalize less explicit analyses. Lastly, we consider the use of AI, both for
assistance in developing these resources and for inference systems that can make use of them. This work extends an earlier programme in proof-assistant semantics,
with research and teaching as complementary uses. 

---
# Evaluating and Improving Natural Language Autoencoders 

**Authors**: Marco Bazzani, Dan Mossing  

**Link**: [PDF](https://arxiv.org/pdf/2611.19507)  

**Abstract**: Natural language autoencoders (NLAs) are a method for generating natural language explanations of LLM activations. We develop evaluations based on auditing tasks that test the ability of NLAs to surface latent knowledge, identify whether a model was misled by a hint, and catch a silent fabrication. We also develop extensions to NLAs that let them use activation information from multiple layers and positions, let us apply them more effectively to long transcripts, and reduce the number of false claims they make. These extensions result in improvements on both our auditing evaluations and auxiliary evaluations adapted from prior work. We also use the evaluations to study how different auditing capabilities change over the course of NLA training, and find that although every evaluation improves early in training, extended training can hurt performance on some of our auditing evaluations. 

---
# Do Probing and Steering Favor the Same Directions Within and Across Language Models? 

**Authors**: Afshin Sadeghi, Daniel Becker  

**Link**: [PDF](https://arxiv.org/pdf/2611.19499)  

**Abstract**: A direction in a language model's hidden state can be used to probe whether a concept is present or to steer the model toward it. Although probe directions are often reused for steering, the direction that best detects a concept need not be the one that best changes it. We study this distinction in three instruction-tuned models from different families. We construct a one-parameter family of directions ranging from a covariance-adjusted discriminant to the mean difference, and evaluate probing and additive steering, which adds the direction to the hidden state, along the family at a fixed side effect. For truthfulness and sentiment, in every model, the best probe is not the best steering direction: steering consistently favors directions closer to the mean difference. This separation holds across 59 of 60 layer-model-concept combinations in a layer sweep, with matched probe and steering texts in all six model-concept combinations, and under a rotation-based steering operator. The exact steering optimum is less stable: for sentiment, an intermediate direction produces a 27-56% stronger steering effect than the mean difference, but this gain shrinks or disappears with matched texts and depends on the layer and the operator.
The two roles also transfer differently across models. A probe transferred by a linear map fitted on neutral text retains 81-91% of the target model's above-chance probing performance, while the transferred mean difference retains less of its steering effect in 11 of 12 model pairs. This is a consistent tendency, although our predefined criteria classify only three pairs, and nine remain inconclusive. Finally, as an application on PopQA, steering toward "unknown" increases abstention by 18-44 percentage points on uncertain errors, versus 2-5 on confident errors, while rarely correcting errors in cases where a correct answer is reachable. 

---
# Text Speeding Itself Up: Recency and Restraint in Session-Local Drafting 

**Authors**: Antal van den Bosch  

**Link**: [PDF](https://arxiv.org/pdf/2611.19455)  

**Abstract**: Speculative decoding lets a cheap draft model propose tokens that a large language model verifies in a single forward pass. When the verifier keeps a KV cache, verification is cheap and drafting must be nearly free. Session-local drafters, which predict from the prompt and the text generated so far, meet that bar without training. We compare a recency-weighted, backing-off, confidence-gated session memory against prompt lookup decoding in its reference form, which copies from the earliest earlier match, and in a variant that copies from the most recent one, with paired per-prompt bootstrap intervals throughout. On general, medical, and legal text with a Mistral-7B verifier, recency accounts for the speed: most-recent lookup gains up to 0.09 in speedup over earliest-match lookup on 500-token generations, and session memory's weighting and backoff add nothing measurable on top, at 30 to 40 times the drafting cost. Energy tells a different story. On medical and legal text, unselective drafters are 14% faster than plain decoding yet use as much energy per token, while drafters that abstain on weak evidence save 3 to 4 joules per 100 tokens; a three-token minimum match length on prompt lookup matches the confidence-gated session memory. Outputs are token-identical to greedy decoding in 94% of generations, with every divergence at a numerical near-tie. Under a repetition penalty, long-generation speedups shrink from 1.4-1.7x to 1.1-1.2x, but recency still helps and the length floor is the only drafter that saves energy in both domains. 

---
# Do Rule-Based Tools Help LLMs Translate Endangered Uralic Languages in the Prompt or over MCP? 

**Authors**: Iaroslav Chelombitko, Mika Hämäläinen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19439)  

**Abstract**: Komi-Zyrian, Moksha, Erzya and Udmurt have mature rule-based resources, among them finite-state analysers, constraint grammars and bilingual dictionaries, which language models do not use. We ask whether giving a model access to these resources improves machine translation from the four languages into English, and whether it matters how the access is provided: as analyses computed in advance and written into the prompt, or as tools that the model calls itself through the Model Context Protocol (MCP). We evaluate manually: an expert linguist read every English output of the strongest model under three conditions, 816 sentences in all, accepted or rejected each one, and named the main error in every rejected translation. With the analyses in the prompt, the number of accepted translations rises from 13 to 43 out of 272; with the tools attached it rises to 55. The gain comes from the bilingual dictionary. For Udmurt, whose dictionary has no English side, the prompt-based condition brings no gain at all, while for the other three languages an English gloss reaches the prompt for 74 to 80% of the source words. Tool calling is also much harder to get working than prompt injection. Of 31 open-weight models, 14 never call a tool, 11 of them because they cannot produce a usable call, and half of the calls that are made name a language the toolkit does not support. We release the data set, every model output and all tool-call logs. 

---
# Real-World End-to-End Product Entity Resolution 

**Authors**: Dimitrios Karapiperis, George Papadakis, Manolis Koubarakis  

**Link**: [PDF](https://arxiv.org/pdf/2611.19418)  

**Abstract**: Product Entity Resolution (PER) in e-commerce is challenging due to unstructured text, formatting inconsistencies, promotional noise, and fine-grained product differences. We present an end-to-end evaluation of four zero-shot PER pipelines---Lexical Cascade, Semantic Cascade, Hybrid Fusion Cascade, and Generative-Lexical Cascade---across established benchmarks and introduce two real-world datasets characterized by cross-lingual noise and dense brand collisions. We evaluate lexical and dense blocking with cross-encoder, tensor-search, and Large Language Model verification, considering accuracy, runtime, and memory consumption. 

---
# From Facts to Personas: Interpretable Role Unlearning in LLMs via Mixture-of-Experts 

**Authors**: Ruihong Zeng, Puning Yang, Jinghui Zhang, Shen Gao, Xiangliang Zhang, Preslav Nakov, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19393)  

**Abstract**: LLMs are widely adopted in multi-role dialogue and persona simulation. However, their strong ability to imitate role-specific linguistic and behavioral patterns can introduce privacy risks, reinforce social biases, and lead to unsafe or uncontrollable outputs. To address these issues, we introduce the Role-playing Unlearning task, which aims to selectively forget the style and knowledge associated with target roles while preserving the language ability and behaviors of the remaining roles. Unlike prior unlearning tasks that focus solely on removing factual information, our task additionally requires forgetting persona-specific characteristics, making the control over model generation more fine-grained. We propose RoleMoE, a Mixture-of-Experts framework that maps different roles to more independent expert subspaces, improving role separation and structural interpretability. To further distinguish role-specific features, RoleMoE incorporates two complementary losses: a specialization loss that drives roles to adopt distinct expert distributions, and a disentanglement loss that encourages experts to develop distinct capability subspaces. This decomposed structure enables role unlearning to focus selectively on target-role experts. We finally propose RoleBench-Unlearn for evaluation, and experiments on multiple backbone LLMs show that RoleMoE enables precise and controllable role forgetting, while better preserving the linguistic quality, knowledge, and persona consistency of non-target roles. All code and datasets are publicly available at: this https URL. 

---
# CausalGloss: Multi-Evidence Action Control for a Simultaneous Mandarin Speech-to-ASL Gloss Cascade 

**Authors**: Mingze Li, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2611.19375)  

**Abstract**: Simultaneous Mandarin speech-to-American Sign Language (ASL) gloss translation requires deciding when an evolving utterance supports committing each gloss. Fixed Wait-k cannot adapt to source-gloss reordering, while training only on reference histories causes exposure bias. We present CausalGloss, an English-prefix-to-ASL-gloss action policy for a Mandarin-speech cascade. It combines lexical and semantic evidence, bounded delay, negation constraints, burst WRITE actions, and incomplete-history recovery. On a frozen 500-example How2Sign-derived benchmark with weak LLM-generated references, CausalGloss improves the fixed Wait-k frontier and nearly matches a three-seed TransLLaMA-style learned adaptive baseline at k = 5 in Token F1 (0.7047 versus 0.7067), with 54.4% lower LAAL. Against CLERC Epee human-gloss references, it achieves 0.649 Token F1 and Pareto-dominates three-seed TransLLaMA-style k = 5 and fixed k = 5, 7, reducing LAAL by 58.2-69.9%. Ablations separate timing from retention. A synthetic Mandarin-speech cascade preserves a smaller computation-aware latency gain, although outputs remain over-long. We also construct ChineseSpeech-ASLGloss-v1, a 19,924-utterance, 36.54-h controlled evaluation corpus. 

---
# Leakage-aware calibrated decision-level fusion for mental-health text classification 

**Authors**: Wadii Boulila, Giuseppe Varone, Bilel Benjdira, Maha Driss  

**Link**: [PDF](https://arxiv.org/pdf/2611.19360)  

**Abstract**: Decision-level fusion benefits from complementary errors, calibrated probabilities, and leakage-aware evaluation. We evaluated a heterogeneous transformer fusion framework on 262,276 proxy-labeled Reddit records covering anxiety, depression, schizophrenia, and suicidal ideation. In a stratified random split, MinHash-LSH identified 29.1% of test records as lexical near-duplicates of training data. Augmentation-lineage auditing found 50.2% lineage overlap with training, including 30.0% of records retained after lexical filtering. We therefore used lineage-disjoint partitions. Twelve encoders, comprising three seeds each of Bio-ClinicalBERT, MentalBERT, MentalRoBERTa, and RoBERTa-base, underwent cross-fitted validation for calibration and ensemble selection. A class-specific Complementarity Index (CI), defined as the Jaccard distance between members' correctly classified validation sets, was assessed against independent fusion gain. CI correlated with gain for depression (r=0.50) and suicidal ideation (r=0.68), but not anxiety; results for schizophrenia were mixed. The selected MentalBERT, MentalRoBERTa, and RoBERTa-base triad achieved macro-F1 0.844 [0.823, 0.863], MCC 0.779, and ECE 0.0033 after post-fusion recalibration, improving macro-F1 by 0.021 [0.007, 0.037] over the validation-selected best single encoder. Integrated-Gradients attribution diversity was 2.3 times greater in fusion-rescued cases. Removing divergence-driving words destroyed 58% of rescues, versus 12% under random removal. These findings show that lineage auditing detects dependencies missed by lexical filtering, while class-specific complementarity and post-fusion calibration reveal ensemble behavior obscured by aggregate accuracy. 

---
# A Comparative Evaluation of BERT-Based and LLM-Based Models for Indic Sentence Similarity 

**Authors**: Aishwarya Gunjal, Kanchan Khemnar, Pawan Poojary, Pranav Kulkarni, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2611.19356)  

**Abstract**: The relative effectiveness of BERT-based encoders and large language model (LLM)-based embeddings for Indic semantic textual similarity remains insufficiently characterized. We compare eight BERT-based and five LLM-based models on the 1,500-pair STS Benchmark development set translated into ten Indic languages and on the native Marathi MahaSTS test set. IndicSBERT achieves the highest Pearson correlation in all ten translated language sets, with a macro-average of 0.843, followed by BGE-M3 at 0.792. Harrier-270M is the strongest LLM-based model on translated STS-B, with a macro-average of 0.751. On MahaSTS, IndicSBERT and Bhasha-Embed-v0 achieve Pearson correlations of 0.901 and 0.900, respectively, while EmbeddingGemma leads the LLM-based models at 0.871. The highest-scoring BERT-based model thus exceeds the highest-scoring LLM-based model in both settings, although the magnitude of the difference varies across benchmarks. These findings indicate that LLM-derived embeddings do not uniformly improve Indic sentence similarity over established encoders and motivate evaluation that accounts for language coverage, task supervision, and benchmark construction. 

---
# When Does Label-Free Deconfounding of Document Embeddings Work? 

**Authors**: Anna Kiepura, Yingqiang Gao, Elliott Ash, Yu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2611.19350)  

**Abstract**: Document embeddings encode spurious attributes that can distort semantic comparisons; yet supervised erasure requires metadata that is often unavailable. We systematically investigate label-free deconfounding across corpora and embedding models. Our results show that label-free methods improve semantic utility over unprocessed embeddings, while simple geometric approaches recover much of the supervised benefit, sometimes matching metadata-supervised erasure in linear confounder recoverability and semantic utility. We identify two distinct causes of ineffective erasure: poor proxy-confounder alignment and insufficient intervention coverage. Crucially, confounder recoverability and semantic utility are distinct dimensions of deconfounding success: utility can improve despite residual confounder information, while stronger suppression does not necessarily improve utility. Moreover, reduced linear recoverability can coexist with substantial nonlinear recoverability. Together, these findings distinguish confounder discovery from intervention effectiveness and highlight the importance of evaluating nuisance suppression alongside semantic preservation. 

---
# OmitEHR: Scalable Omission Detection in LLM-Generated Clinical Summaries 

**Authors**: Bridget Lin, Ryan Welch, Bobak Seddighzadeh, Fran Riley, Kaitlyn Stoehr, Margarita I. Pipinos, Dimitri Speron, Emma V. Sage, Sneha Jain, Nigam H. Shah, Emily Alsentzer  

**Link**: [PDF](https://arxiv.org/pdf/2611.19343)  

**Abstract**: Health systems increasingly rely on large language models (LLMs) to draft summaries of patient medical records, yet evaluations now find omissions, the absence of clinically important information, to be the most common failure mode. Detecting omissions in an LLM-generated summary requires knowing which facts in the record are salient, a signal that existing methods hand-specify, ignore, or recompute at inference with costly LLM judges. We present OmitEHR, a reference-free omission detector that learns clinical salience from historical clinician-written summaries predating widespread LLM use. OmitEHR learns to predict which source-note sentences clinicians include in their summaries, then surfaces high-salience sentences not covered by an LLM-generated summary as candidate omissions. We apply OmitEHR to brief hospital course (BHC) summarization using 5,000 admissions from a large academic medical center. On held-out admissions, OmitEHR outperformed all evaluated baselines at predicting salient source sentences, including a prompted LLM judge. The full omission detection pipeline took 17.7 to 19.5 seconds and cost \$0.03 to \$0.13 per summary depending on the base model, 1.6 to 1.7 times faster and 5.6 to 6.7 times cheaper than scoring every source sentence with an LLM judge. In a blinded review of 25 admissions, two board-certified clinicians confirmed a mean of 5.6 clinically meaningful omissions per summary surfaced by OmitEHR, compared with 3.3 for a full LLM pipeline, counting candidates confirmed by at least one clinician. The efficiency and validity of OmitEHR make it a practical tool for health systems to flag omissions before a summary reaches the clinician. 

---
# On Representational and Functional Alignment of Training Data Attribution Methods for Language Models 

**Authors**: Loris Schoenegger, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2611.19286)  

**Abstract**: Existing approaches to training data attribution derive influence estimates from different sources of information, including gradients, model-internal states, and surface-level similarity. However, comparisons between attribution methods remain incomplete, often relying on indirect evaluations through downstream task performance. In this work, we introduce a framework for directly comparing methods along three dimensions: functional alignment, which measures correlation between full attribution vectors; representational alignment, which captures structural similarity of the representations underlying attribution; and cross-method predictability, which evaluates agreement on local pairwise ranking decisions. We observe strong representational and moderate functional alignment within paradigms. Alignment across paradigms is low and task-dependent, yet stable across models. Our results suggest that the similar downstream performance observed across paradigms in prior work can be attributed to agreement on the relative importance of documents, rather than to agreement on rankings of the full training data. 

---
# B-OPSD: On-Policy Self-Distillation with Self-Generated Problem Briefs 

**Authors**: Zhi Xu, Yinqi Wu, Feng Zhang, Xiaotong Zhang, Hong Yu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19273)  

**Abstract**: On-Policy Self-Distillation (OPSD) has recently attracted interest because it provides dense teacher guidance without a separate, often larger teacher model. However, OPSD conditions its teacher on ground-truth solutions, which limits its use when such solutions are unavailable. We propose On-Policy Self-Distillation with Self-Generated Problem Briefs (B-OPSD). Using a prompt template, the initial model or a saved student checkpoint generates a problem brief from the question alone. The prompt asks for relevant principles, applicability conditions, and likely pitfalls, but not a final answer or a complete solution. A frozen teacher uses the brief as privileged context to guide the current student's on-policy rollouts through token-level distribution matching. B-OPSD uses no ground-truth solutions during training. Results on AIME 2024, AIME 2025, and HMMT 2025 show that both B-OPSD variants achieve higher average accuracy than OPSD at each of three Qwen3 model sizes. Code is available at this https URL 

---
# Revisiting What Makes a Good Context Subset: A Sparse Latent Basis Perspective 

**Authors**: Cen-Jhih Li, Siddhant Chaudhary, Aditya Bhaskara  

**Link**: [PDF](https://arxiv.org/pdf/2611.19220)  

**Abstract**: Selecting context subsets is increasingly important in modern LLM inference, yet what makes a subset effective remains unclear. We study this question in the controlled setting of few-shot demonstration selection. We examine a hypothesis that demonstrations should collectively provide task-relevant information needed by a target query. Assuming that the query representation provides a useful proxy for these requirements, we formulate selection as sparse reconstruction and instantiate it with Residual Explanation Search (RES), a Matching-Pursuit-style diagnostic procedure. Across seven benchmarks and eight open-weight LLMs, no retrieval objective is universally best. RES outperforms MMR in 34 of 56 model--dataset pairs, with its clearest advantage on sentiment tasks, suggesting that query-conditioned collective reconstruction may capture useful complementarity that pairwise dissimilarity misses. Relevance-based retrieval remains strong on template-heavy reasoning benchmarks, while all retrieval methods underperform zero-shot prediction on average on StrategyQA, where a solution-reconstruction analysis is consistent with the query not exposing the required external knowledge. Target-model hidden states further show that retrieval objectives interact differently with representation spaces: middle-layer representations generally favor similarity retrieval, whereas last-layer representations are often stronger for residual- and diversity-based selection. These findings support a conditional, representation-dependent account of context subset quality rather than a universally superior selector. 

---
# Omni-Workspace: Benchmarking Omni-Modal AI Assistants on Multimodal Workspace Tasks 

**Authors**: Yuhao Wang, Yucheng Zhao, Yiran Zhong, Feida Zhu, Hongcheng Liu, Qichao Wang, Ming Ma, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19202)  

**Abstract**: Omni-modal agents are expected to move beyond multimodal understanding toward completing complex tasks in realistic multimodal environments. In these settings, information is distributed across heterogeneous files and modalities, with cross-file and cross-modal dependencies requiring agents to continually coordinate perception, reasoning, and tool use as newly gathered evidence informs subsequent actions. However, existing omni-modal benchmarks primarily focus on multimodal perception and understanding, leaving complex task completion through sustained interaction with multimodal workspaces underexplored. We introduce Omni-Workspace, a benchmark comprising 380 deep-research-style tasks across four multimodal workspace types and two difficulty levels. These tasks require agents to inspect and process workspace files, retrieve external information when needed, integrate evidence across files and modalities, and produce artifacts satisfying explicit requirements. Evaluating 18 representative models reveals limitations in long-horizon multimodal workspace execution, with the best model scoring only 65.93 on average. Further analysis identifies multimodal context inflation and task-state drift as recurring challenges. Motivated by these findings, we further propose OmniFocus, a lightweight approach combining multimodal working-memory management with explicit task-state tracking to improve performance and efficiency. Omni-Workspace advances the evaluation and development of omni-modal agents toward autonomous completion of complex tasks in multimodal workspaces. 

---
# Isolating Perception for Multi-View Planning under Occlusion-Induced Partial Observability 

**Authors**: Kevin Nguyen, Ceyhun Efe Kayan, Feng Liu, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19179)  

**Abstract**: Vision-language models (VLMs) are beginning to saturate visual planning benchmarks, yet they fail to resolve depth ambiguity and degrade sharply with problem size under occlusion-induced partial observability (PO). We introduce BW-PO based on the Blocksworld domain, a controlled benchmark that reduces existing VLM planners and formalizers to near-zero performance, and propose VLM-as-perceiver, a paradigm that narrows the VLM's role to local perception while delegating cross-view evidence reconciliation and plan generation to external solvers. Across model-specific challenging subsets of BW-PO, this raises plan success from 1% to 97% for Gemma-4-31B, 0% to 95% for Qwen3.5-122B-A10B, and 3% to 84% for Gemini-3.1-Pro. Isolating perception in this way separates two bottlenecks that are otherwise entangled: reconciliation under depth ambiguity, and completeness of visual extraction under scene density. 

---
# A Bag of Concepts? Testing Role-Filler Binding in the Verbalizable Workspace 

**Authors**: Reya Dawlah, Elizabeth McMillin, Michael Shihong Zhang, Badrish Ananth, Rishi Adi Ruia, Ane Zuñiga, Thomas Jiralerspong  

**Link**: [PDF](https://arxiv.org/pdf/2611.19176)  

**Abstract**: Runtime alignment monitors built on a verbalizable workspace have been proposed as a way to surface a model's silent cognition before it produces output. Their value rests on the workspace carrying enough structure to distinguish safety-relevant propositions, most of which are relational: "the model plans to deceive the user" and "the user plans to deceive the model" share a concept set and differ only in role assignment. We ask whether the workspace carries the role-filler bindings (which entity fills which role) such a monitor would need, on three frontier language models (Qwen3.6-27B, Gemma-3-12B, Gemma-3-27B-IT), under two lens families (J-lens and R-lens), and with two methods for fitting the role direction (difference-of-means and LRE-gradient). We find a clean dissociation: the workspace carries directions that steer what the model outputs about roles but not the representations the model uses to compute them. Linear probes recover role at near-ceiling from the residual stream but score below a rank-matched random control when reading the workspace, and steering the model along the fitted role direction changes which entity it binds to no more than strength-matched control edits do. Ablating the workspace (zeroing the residual stream's projection onto its ten most active lens directions) damages binding on one of the three models (Gemma-3-27B-IT, holding across three lens variants) and shows no binding-specific damage on the other two, with a rank-matched random subspace as the control; this confirms our design detects binding when present, though scale and instruction tuning are confounded there. A workspace-lens monitor built on these directions would therefore miss a role assignment the model itself represents. 

---
# Personalized Language Models for One-to-Many Private Communications 

**Authors**: Zihan Hong, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.19155)  

**Abstract**: We study relationship-specific continual reply prediction for one-to-many private communication: a system predicts an author's reply to a recurring contact and updates its personalization state when the actual reply becomes available. We propose TRACE (Temporal Relationship-Aware Communication Evolution), which uses author-level lexical habits, contact-specific wording, and temporal expression estimates to guide a frozen language model, with a feedback-driven policy controlling revision. On 216 test emails from eight Enron users, all four main TRACE configurations achieve higher mean BLEU, ROUGE-L, and BERTScore and lower mean Reply Style MSE than Base LLM and Base RAG. Component ablations show mixed effects and do not establish statistically reliable improvements in Reply Style MSE after multiple-comparison correction. More accurate state forecasts and larger revision budgets do not consistently improve reply-level metrics. These exploratory findings expose the challenge of turning evolving relationship-specific expression estimates into useful generation guidance. 

---
# MarineSparkLM-4B: A Lightweight Language Model for Marine and Offshore Engineering Built by Parameter-Efficient Adaptation 

**Authors**: Cheng Siong Chin  

**Link**: [PDF](https://arxiv.org/pdf/2611.19151)  

**Abstract**: Marine and offshore engineering text is spread across preprints, open-access journals, encyclopedias and institutional documents, and the amount of it is small by pretraining standards. We ask how far a lightweight language model for marine and offshore engineering can be taken with parameter-efficient adaptation on a single workstation-class device. Starting from the 4-billion-parameter OceanGPT-basic-4B-Instruct, we train two independent LoRA adapters, each updating 33.0 million of 4.06 billion parameters (0.81%). Training took 4 hours 40 minutes for the first adapter and 10 minutes for the second on one NVIDIA DGX Spark. The first adapter is trained by next-token prediction on a 61.6-million-token corpus covering ten engineering themes. The corpus was built from 13,879 source documents, of which 7,069 were retained after deduplication and a two-tier keyword filter that removes cross-domain lexical collisions. Its validation loss falls from 2.082 to 1.976 over 1,500 steps. Direct testing showed that this adapter could still drift off topic on a retrieval-grounded ballast-water prompt, so we trained a second adapter on 2,295 question-answer pairs (114 hand-written and 2,181 generated by the base model from corpus passages) with the loss restricted to answer tokens. On 20 held-out prompts scored by hand for accuracy, topicality and specificity (maximum 120), the plain base model, the continuation adapter, the instruction adapter and the instruction adapter with retrieval scored 42, 56, 84 and 88. Absolute accuracy stays low (14 of 40 without retrieval and 20 of 40 with it). We report the full pipeline, two negative results (a reverted keyword fix and an instruction run that overfit within 100 steps), and the limits of the evaluation, which uses one small prompt set, no inter-rater agreement measure and model-generated reference answers. 

---
# A Systematic Study of Cross-Token Transfer for Data-Efficient Non-Verbal Text-to-Speech 

**Authors**: Jing Pan, Haibin Wu, Minxue Niu, Florian Metze, Yuzong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19145)  

**Abstract**: Non-verbal vocalizations (NVs), such as laughter and breath, carry affective cues that words alone do not convey. Yet NV data remains scarce because speech annotation pipelines often omit or normalize these events. Existing work has improved NV text-to-speech through model design and data curation, but whether one NV token can help a model learn another has received little systematic study. We investigate this cross-token transferability in a controlled CosyVoice 2 setup. We first trace a laughter-only scaling curve, then hold laughter at 200 or 1,600 examples while adding breath, sigh, cough, lip-smack, or neutral speech. With only 200 laughter examples, adding 3,200 cough examples raises end-to-end laughter F1 from 0.170 to 0.529, surpassing the 0.495 obtained from 1,600 laughter examples alone. At the 1,600-example anchor, cough reaches 0.695, nearly matching 6,400 laughter examples (0.697). Waveform-level Whisper-D scoring and a semantic-token transcriber show consistent trends. Cross-token transfer therefore offers a complementary route to low-resource NV generation. 

---
# Low-Latency LLM-Based Streaming Multi-Talker ASR with Verifiable Rewards 

**Authors**: Jing Pan, Ruizhi Li, Ju Lin, Jing Zheng, Florian Metze  

**Link**: [PDF](https://arxiv.org/pdf/2611.19136)  

**Abstract**: LLM-based architectures offer strong generalization and instruction-following capabilities for streaming automatic speech recognition (ASR), but multi-talker use introduces coupled transcription, attribution, and latency challenges. Our wearable setting requires real-time transcription of a conversation between a near-field device wearer and a far-field partner. Because partial hypotheses are rendered on persistent sides of a dialogue UI, speaker roles cannot be permuted and each speaker tag must precede its turn. These sparse tags are weakly optimized by supervised fine-tuning (SFT), while committing to a leading tag can delay the text that follows. We therefore post-train a streaming LLM-based ASR model with reinforcement learning with verifiable rewards (RLVR), combining a fixed-role Speaker-Attributed WER (SA-WER) grader with a turn-level onset-delay reward. On an eight-direction multi-talker cohort, SA-WER RL reduces SA-WER from $20.51\%$ for SFT to $13.14\%$. Adding the delay objective reduces onset $p50$ by $65\%$ (from $0.918\,\text{s}$ to $0.320\,\text{s}$) and $p90$ by $44\%$ (from $1.389\,\text{s}$ to $0.780\,\text{s}$) relative to SA-WER-only RL, while achieving $12.70\%$ SA-WER. These results demonstrate that fixed-role recognition quality and streaming responsiveness can be improved jointly. 

---
# Latency, Speech Recognition and Turn-Taking in Voice Agents for Spanish Phone Negotiation: An Externally Measured Comparison of Pipecat, LiveKit and GPT-Live 

**Authors**: Jovan Popoca Moran, Eduardo de Avila-Armenta, Raúl Alejandro Aquino-Ibarra, Jorge Octavio Nicolás-Díaz, Umberto Leon-Dominguez  

**Link**: [PDF](https://arxiv.org/pdf/2611.19118)  

**Abstract**: A voice agent that negotiates by phone must answer quickly, understand spoken amounts and cope with a caller who talks over it. We compared three architectures for such an agent, negotiating freight rates in Spanish: two cascades (speech recognition, language model and speech synthesis) orchestrated by Pipecat and by LiveKit Agents, and GPT-Live, an OpenAI speech model that listens and speaks without separate stages. A program played the carrier with the same 47 recordings of one speaker for all agents, without the telephone network, in 77 calls per architecture over five negotiation scenarios, and measured latency on the listener's side with a validated instrument. Median latency to the content response was 2,616.0 ms (Pipecat), 3,090.0 ms (LiveKit) and 2,444.8 ms (GPT-Live); the confidence interval of GPT-Live's 171.2 ms advantage over Pipecat includes zero. According to its own timestamps, GPT-Live began to answer 800 ms after the end of the line, close to the 0.8 s published by OpenAI; most of the remaining wait, about 1,367 ms and nearly constant, is the audio's round trip through OpenAI's service. A tool call added 0.9-1.1 s in every architecture, more than any difference between them, and the frameworks' self-reported latency fell short of the listener's wait by 9% (Pipecat) and 22% (LiveKit). Word error rates were low (6.03, 4.93 and 1.47%), but the error on amounts was 13.01% and 12.61% in the cascades, against 2.86% in GPT-Live. With default settings, Pipecat and LiveKit stopped at almost every backchannel and often took it as the carrier's answer: an "ok" spoken over the agent ended in a registered agreement in 3 of 15 Pipecat calls, 2 of 15 LiveKit calls and no GPT-Live call (descriptive counts, not tested). Latency should therefore be measured on the listener's side, and a cascade needs safeguards for spoken amounts and backchannels. 

---
# EEGHyena2: Fixation-Inspired Joint Lexical-Grammatical Conditional Random Field for Interpretable EEG-to-Text Decoding 

**Authors**: Xiaoshan Zhou, Chenlan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19104)  

**Abstract**: A sentence is built from words that carry lexical meaning and grammatical roles, and brain activity during reading reflects both. Current EEG-to-text decoders, however, let a large language model (LLM) propose and order the words, so fluent output can reflect the LLM's prior more than the recorded EEG. In this study, we asked whether fixation-linked EEG carries lexical and grammatical evidence during natural reading and how a decoder can use such evidence. Fixation-related potential analyses showed that content and function words differ reliably in group-averaged EEG but only weakly in single words. We therefore propose EEGHyena2, a structured decoder in which a bidirectional encoder inspired by StripedHyena 2 feeds a semantic head that ranks a fixed lexicon and a part-of-speech (POS) head that predicts grammatical categories; a joint word-POS conditional random field (CRF) scores a restricted candidate lattice, and a frozen Llama-2 scores the candidates in context. On two natural-reading corpora, ZuCo and ROAMM, it matched the state-of-the-art method on BLEU-1 (21.4 vs 21.3), exceeded it on ROUGE-1 F1 (23.3 vs 20.3), lowered the character error rate on ROAMM (75.3% vs 84.7%) and decoded each recording about 3-10 times faster than LLM candidate search. The CRF learned transitions that mirror English syntax, and the encoder drew local context from its convolutions and distant context from attention. Decoders can thus pool weak word-level EEG evidence across the sentence through explicit lexical and grammatical structure, with the language model adjudicating among candidates rather than generating them, which lowers decoding cost and keeps decisions inspectable. 

---
# What Transfers Across Cuneiform? Script-Level Substrate and Language-Specific Regularities via Segmentation and Representation 

**Authors**: Chuanjun Zhou, Adam G Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2611.19094)  

**Abstract**: Cuneiform recorded typologically unrelated languages for more than three millennia, raising a natural question: which of its statistical regularities are properties of the shared writing system, and which belong to individual languages? We answer this with two probe pairs targeting the same script-versus-language decomposition. The first compares two unsupervised, dictionary-free segmentation methods: transitional probability (TP), grounded in infant statistical-learning research, and Morfessor. To our knowledge, this is the first systematic evaluation of dictionary-free segmentation on cuneiform; the only prior published baseline relies on a hand-built dictionary (Max-Match, F1 = 0.74). Both our methods reach F1 > 0.97 within language (Morfessor reaches F1 > 0.99 on Akkadian and Sumerian), exceeding MaxMatch by more than 23 F1 points, but diverge sharply in cross-language transfer: TP holds F1 > 0.95 across all six language pairs, while Morfessor collapses to F1 = 0.27 from the 13K-token Elamite source. A learning curve and a Hittite stress test (34.7% foreign-script logograms, where TP drops 35 points and Morfessor only 8) further isolate the two underlying mechanisms. The second probe pair extends the same decomposition to representations: concatenating Unicode and Latin features improves classification in 11 of 12 morphosyntactic tasks (p < 0.01). We release CUNEI-TOOLS and a three-language benchmark. 

---
# The Geometry of Conceptors in Neural-Activation Spaces 

**Authors**: Ilias Triantafyllopoulos, Young Min Cho, Ren Tao, Miranda Muqing Miao, Sunny Rai, Lyle Ungar, Sharath Chandra Guntuku, Sunghye Cho, Tyler Moore, Neville Ryant, João Sedoc  

**Link**: [PDF](https://arxiv.org/pdf/2611.19075)  

**Abstract**: Conceptors represent concepts as graded collections of directions in neural-activation space rather than as single vectors. We develop a geometric account of how these representations depend on the contrast used to define a concept: opposing values in bipolar fitting, or presence and absence in unipolar fitting. We derive an exact decomposition of a centered pooled conceptor into an average pole filter, a mean-separation update, and a covariance-shape update. Experiments across five language models on sentiment, political leaning and stance, and depression expression show that both updates improve reconstruction of held-out contrasts. Additional directions preserve variation beyond the mean-difference line, and changing the comparator changes the benefit of these directions. Shape, size, and overlap each show up downstream: in amplification experiments, bipolar fits raise evidence for both poles more consistently than unipolar fits; conceptor size (the quota) correlates positively with layerwise probe performance; and cross-concept overlap quantifies interference under subspace removal, linking shared directions to Boolean composition. In an additional generation evaluation, conceptor filters achieve higher target-expression win rates against an unsteered reference than three mean-based controls. Together, these findings explain how fitting choices shape conceptor geometry and support bipolar fitting when the objective is to amplify both opposing poles. 

---
# Beyond Controllability: Speaker Diversity in Instruction-Guided Text-to-Speech 

**Authors**: Yi-Cheng Lin, Jia-Kai Dong, Kuan-Yu Chen, Szu-Chi Chen, Jian-Jiun Ding, Sung-Feng Huang, Yusuke Hirota, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2611.19030)  

**Abstract**: Instruction-guided text-to-speech (ITTS) lets users specify voice properties in natural language. Applications from conversational AI character voices to synthetic training data for downstream speech models depend on the system producing many distinct voices when the same broad category is requested repeatedly. No prior work audits whether ITTS outputs are in fact diverse, either compared to real speech or across attribute values. We evaluate seven open-source ITTS systems using speaker embeddings whose representational capacity we independently validate. Every system produces speaker distributions markedly narrower than real speech. Furthermore, female prompts produce less diverse voice pools than male prompts across most systems. In contrast, accent and emotion prompts barely alter speaker diversity in most systems, but reveal a strong controllability bias toward North American English and neutral speech. These results show that diversity and controllability capture complementary weaknesses, neither of which alone is sufficient to evaluate ITTS. Our code and data are available at this https URL. 

---
# SurveyWeave: An Agentic Framework for Automated Survey Generation 

**Authors**: Franklin Zhang, Xiao Yang, Xin Luna Dong, Juhua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19029)  

**Abstract**: Surveys help researchers keep up with fast-moving fields, but they take weeks to months to write. Automated survey generation (AutoSurvey) can alleviate this situation, but it is challenging to draw on hundreds to thousands of precisely scoped papers, choose among many valid structures, and stay faithful to its sources while offering insights. We present SurveyWeave, an agentic architecture that decomposes the problem into four tasks coordinated by three agents, each reviewing others' work and conducting revisions based on feedback. We also introduce WeaveEval, a five-dimension evaluation grounded in prior surveys. A comprehensive comparison shows that SurveyWeave outperforms Codex by 8 points and the strongest AutoSurvey baseline by 6 points. 

---
# PREFACE: Separating Shared Corrections from Prefix Conditioning in PEFT 

**Authors**: Novanto Yudistira, One Octadion  

**Link**: [PDF](https://arxiv.org/pdf/2611.18987)  

**Abstract**: Does input-conditioned parameter-efficient fine-tuning (PEFT) improve because it uses the current input, or because it adds a useful learned correction? We investigate this distinction with PREFACE, a bounded channel controller for PEFT-adapted projections that requires no test-time gradients. The controller reads observed context; in multiple-choice likelihood scoring, its state is independent of the candidate continuation. Our central study freezes IA3, tuned LoRA, and DoRA adapters and matches controller architecture, initialization, batches, and updates across current-prefix and learned-global controls. Across ARC-Challenge and SciQ, current-prefix control improves all six mean accuracies over frozen adapters, but none of its paired contrasts with learned-global control excludes zero. Replacing its corrections with their training-prefix mean changes accuracy by at most 0.34 percentage points; input-varying corrections contribute only 1.8-2.8% of correction RMS. On ARC test, frozen-adapter current/global controls score 25.68%/25.77%, distinct from the earlier jointly trained 23.1% to 26.4% gain over IA3. We derive and evaluate static compilation of fixed corrections, separating deployment simplification from statistical equivalence. Broader GPT-2, Pythia, and Qwen experiments show family-, task-, and scoring-dependent gains, with modest recent-model effects. The study provides an intervention-based evaluation of conditional PEFT: improvements over a static adapter do not by themselves demonstrate a benefit from input matching. 

---
# GPBam: A Retrieval-Aware German Public Law Exam Benchmark 

**Authors**: Abdullah Al Zubaer, Lorenz Wendlinger, Simon Alexander Nonn, Michael Granitzer, Jelena Mitrović  

**Link**: [PDF](https://arxiv.org/pdf/2611.18981)  

**Abstract**: Legal reasoning in civil-law systems is not rule-lookup: the German state examination asks for a structured opinion (Gutachten) that names the governing norms, applies them to the facts, and argues to a conclusion. We introduce GPBam (German Public Law Bar-style Exam), 81 public-law examination cases with reference solutions, paired with a recitation task over 200 statutory provisions. Drawn from state examination preparation, the cases are broad and rarely repeat, and are not publicly available - we find no evidence of exact-solution memorisation in the models we evaluate, and keeping them private limits contamination. Because each reference solution's cited norms form a retrieval gold set over the very instances the essays are graded on, GPBam scores retrieval and long-form generation jointly, which makes it a benchmark for legal retrieval as well as for legal writing and lets us ask how much retrieval quality a legal opinion actually needs. Across 32 open-weight and proprietary models evaluated without retrieval, scores span 4.81-76.42 out of 100. Supplying the law helps modestly: across seven models and eight retrieval configurations, retrieval performance varies nearly threefold and buys at most 4.1 essay points, and nothing significant for three of the seven, while an oracle with 91% gold-norm recall improves scores by 2-6 points. Essay quality strongly tracks prior knowledge: verbatim statute recall, measured in a separate task, correlates at $\rho = 0.92$ over 32 models. 

---
# KUDEX: A Zero-Shot Decompositional Framework for Logical Analysis in Natural Language 

**Authors**: Zixu Zhao, Tianhang Zhang, Tong He, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18851)  

**Abstract**: While Large Language Models (LLMs) excel at generating fluent text, they often struggle with multi-step logical reasoning with inferential connections among statements. We introduce KUDEX (Knowledge-Upward-Downward Reasoning Extension), a novel zero-shot, decompositional framework for analyzing and verifying logical relationships in natural language. Unlike traditional approaches that rely on surface-level entailment checks, KUDEX leverages LLMs to decompose reasoning into atomic primitives and constructs transparent reasoning graphs that serve both as execution traces and audit logs. Our framework includes a general prompt adaptation pipeline that adapts to diverse reasoning tasks without additional fine-tuning or domain-specific pipelines. Through comprehensive experiments on benchmarks across four domains, we demonstrate that KUDEX provides superior performance compared to existing methods, offering a more granular and robust framework for assessing logical relationships between statements. 

---
# Mapping the Afterlife of Future Work Statements in NLP Research 

**Authors**: Claudiu Creanga, Ioana-Roxana Boriceanu, Ana Sabina Uban, Liviu P Dinu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18847)  

**Abstract**: Future work statements are one of the few places in scientific papers where authors explicitly describe what they think should happen next. We study what happens to such statements after publication. We construct a dataset of 72,070 future work items extracted from 23,113 ACL Anthology papers published between 1979 and 2025, categorize them by type, retrieve candidate follow-up papers through citation-based and content-based search, and use LLM-based judgments to assess whether later papers pursue the proposed directions. Using this resource, we analyze patterns across time, venues, categories, themes, and authors. We find that many items are later addressed, often within a few years, with clear differences across types and themes and a largely community-driven pattern. Our results suggest that future work statements provide useful evidence for studying how research directions develop in NLP. To support further research, we release our dataset at: this https URL. 

---
# Wrong with Confidence: Measuring Hallucination, Overconfidence and Identity-Conditioned Behaviour in Large Language Models 

**Authors**: Md. Rejaul Korim Sadi, Toufiqur Rahman Tasin, Golam Mostofa Naeem, Dewan Fahad Chowdhury, Hatim Al Amin Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2611.18728)  

**Abstract**: Large language models answer underspecified questions readily, and accuracy-only evaluation rewards guessing over abstaining. Observational studies link query underspecification to hallucination, but they cannot separate the effect of missing information from item difficulty, and they rarely ask what the model tells the user. We report a pre-registered, paired behavioural study of five open-weight models (Gemma 3 27B, Gemma 4 31B, gpt-oss-120b, Qwen3-Next-80B-A3B and Nemotron 3 Super). Each of 142 factual questions was posed in a fully specified form and in a referentially underspecified form with the same answer key, and three probes measured identity-conditioned behaviour. Two raters scored 5,620 responses under a fixed rubric; effects were estimated with mixed-effects models. Without the referent, the models sought clarification on 33.7% of prompts and answered the rest. Among attempted answers, the error rate rose from 39.1% to 52.0% (mixed-effects odds ratio 4.12, 95% CI 2.76-6.14), and specialist-domain questions failed more often than general ones in every model. Unelicited verbal uncertainty did not track this risk: 75.3% of wrong answers carried no hedge, wrong answers to underspecified prompts were hedged no more often than wrong answers to clear prompts (27.3% vs 22.3%), and hedging separated correct from incorrect answers only weakly (AUROC 0.56). Gendered pronouns followed occupational stereotypes whenever a model committed to one; any stated user identity lengthened answers by about 45% relative to an anonymous user; claimed expertise did not reduce safety caveats. The models either ask or answer as if the question were well posed, and the wording of a guess does not reveal that it is one. Evaluations should score clarification as a distinct outcome and check label validity as well as rater agreement. 

---
# Criterion-Activated Preference Synthesis: Attribution-Preserving Preference Data from Explicit Standards 

**Authors**: Zian Wang, Pu Lu, Caishuang Huang, Wenxi Wu, Meng Zhou, Xiku Du, Yang Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18663)  

**Abstract**: Direct Preference Optimization (DPO) is notoriously sensitive to attribution ambiguity: when chosen and rejected responses differ along multiple entangled dimensions, the model cannot identify which behavior the preference label teaches. We propose Criterion-Activated Preference Synthesis (CAPS), a criterion-first framework that turns explicit quality standards into attribution-verified preference pairs. CAPS synthesizes contexts and responses from a target criterion, then applies dual-signal scoring: a rubric-bound judge measures criterion compliance, while a blind general judge measures holistic quality. Two filters close the attribution loop: the Empirical Activation Test (EAT) keeps criterion-context pairs where satisfying the criterion improves blind quality, and the Attribution Ratio (AR) keeps pairs whose quality gap is dominated by the target criterion. Instantiated with general rubrics, medical rubrics, and Chinese legal statutes, CAPS shows strong data efficiency and domain transfer. With 20k general synthetic pairs, it outperforms 35k human-annotated HelpSteer3 pairs on IFEval, IFBench, AlpacaEval, and Creative Writing v3. In domain settings, CAPS improves HealthBench with 2k medical pairs and improves Qwen3-8B/14B by +4.9 to +6.6 points on legal benchmarks using only public statutes, without human preference labels. We additionally construct the CAPS-Legal preference dataset. 

---
# Non-Deterministic Finite Automata for Telugu Letters and Visualizations for Metrical Poetry 

**Authors**: Boddu Sri Pavan, Chandrasheker Thummanagoti, Appana Mohan Naga Phani Kumar, Boddu Swathi Sree  

**Link**: [PDF](https://arxiv.org/pdf/2611.18661)  

**Abstract**: Syllabic weight is central to Telugu metrical poetry. Syllabically, every letter (aksharam token) is either light (laghuvu) or heavy (guruvu). Poets classify it using a set of pre-defined rules. This paper formalizes aksharam token, the atomic unit of syllabic classification, as a non-deterministic finite automaton (NFA) with the regular expression$\colon$ $(A_h+A_d+A_p)(\epsilon+V_s)+A_m+H^+G(\epsilon+H+V_s+V_v)$, where $\epsilon$ denotes a null string, $A$ denotes a vowel (achchu), comprising hrasvam ($A_h$), deergham ($A_d$), plutham ($A_p$), and miscellaneous ($A_m$), $H$ a pure consonant (pollu\_hallu), $G$ a vowel diacritic (gunintha\_chihnam), $V_s$ the sunna (anusvara), and $V_v$ the visarga, the latter two being nasal and aspirated sounds respectively. We then present automata diagrams and regular expressions for (i) a necessary condition satisfied by every light letter, and (ii) a sufficient condition for a letter to be heavy. Finally, we propose graphical visualizations for the 9 Telugu poem types. 

---
# Winning the Leaderboard, Losing the Distribution: Diagnosing and Mitigating Acoustic Narrowing in Low-Resource ASR 

**Authors**: Kadir Goksel Gunduz  

**Link**: [PDF](https://arxiv.org/pdf/2611.18653)  

**Abstract**: For a low-resource language, one public leaderboard is often the only shared measure of progress, so optimizing toward it means optimizing toward a test distribution. Our low-rank fine-tune of whisper-large-v3-turbo ranked first on MediaSpeech, Common Voice 17 and a five-set Turkish community board, at 9.02% word error on MediaSpeech against 11.93% for the best baseline. On its private split it lost to unmodified whisper-large-v3. We treat that split as an alarm, not evidence; the main finding does not depend on it. On a twenty-condition FLEURS-TR benchmark built independently of our model selection, the unmodified base beats every fine-tune in every condition, and all five deficits clear zero under paired bootstrap. We call this acoustic narrowing: fine-tuning contracts the acoustic range a model serves well. The central finding is methodological: our in-house proxy for detecting narrowing concealed it instead, ranking the narrowed fine-tune above the sturdier base. Model selection needs a proxy from an independently produced distribution; more of the same source will not do. A rebuilt pipeline reduces the narrowing: a larger mixture with disclosed provenance, a higher-capacity base, and base-referenced selection with trajectory averaging. It wins three same-provenance axes and draws level on FLEURS-TR, 8.16% against 8.26%, interval spanning zero. Parity holds under a second inference stack and a full epoch, decoder fixed. The recovery is one trajectory, a demonstration rather than a validated recipe. In Kazakh the inversion reappears on corpora we did not build. With matched trainable-parameter shares, fine-tuning raised out-of-pool error by 2.23 points for an attention encoder-decoder and lowered it by 3.46 for a CTC encoder-only model, across three seeds each. Narrowing thus depends on the system adapted. We release TurkSpeak Bench, a robustness-first Turkish ASR leaderboard. 

---
# ReVision: Selective Intrinsic Reasoning for Tool-Augmented, Reasoning-Intensive VQA 

**Authors**: Ningxin Pan, Guochen Yan, Jialong Wu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18637)  

**Abstract**: External tools substantially expand what multimodal large language models (MLLMs) can retrieve and compute on reasoning-intensive visual questions, and aggregate accuracy typically improves when tools are available. Our empirical study across six VQA benchmarks shows that aggregate accuracy gains under tool access can coexist with answer reversals: questions answered correctly without tools but incorrectly with tools. Trajectory analysis reveals substantially shorter reasoning turns under tool access, suggesting a shift from intrinsic visual reasoning toward reliance on external outputs. We propose ReVision, a training-free test-time method that registers intrinsic reconsideration as an optional action alongside search and code execution. When invoked, a separate tool-free multimodal call re-examines the original image, integrates accumulated evidence, and returns a revised analysis. The MLLM chooses if and when to use it, without fixed reflection stages or paired rollouts. ReVision improves over standard tool use on all six benchmarks (1.28--7.51\%) and outperforms Dual-Rollout Combine on five, with reasoning-token use comparable to standard tool use. 

---
# Beyond Binary Audits: Does Guideline Grounding Mitigate Decision-Space Framing in LLM Opioid Prescribing? 

**Authors**: Soumyadeep Roy, Keyi Yu, Tina Hernandez-Boussard  

**Link**: [PDF](https://arxiv.org/pdf/2611.18631)  

**Abstract**: Audits of LLMs for bias in opioid prescribing have largely relied on binary treatment decisions, a signal that current models have saturated. We introduce Q-Pain++, a full-factorial stress test that holds clinical content fixed while independently varying clinical risk factors and the granularity of the available dosage options. Adding an intermediate dosage option to otherwise unchanged vignettes substantially lowers agreement with expert-annotated dosages for most models, while demographic differences in dosage decisions remain small in all but one configuration once clinical risk factors are added. Grounding models in perioperative guidelines through retrieval reduces this framing effect on average but does not eliminate it, and one model's treatment decisions change when the retrieved passages are reordered. These results suggest that audits should vary the answer format, not only patient identity. We will release our codebase as well as Guideline-RAG-associated resources on GitHub (this https URL). 

---
# All the Pieces Matter! A Six-stage Analysis of an Underground Fraud Ecosystem on Dread 

**Authors**: Megha Sundriyal, Dylan Raharja, Rui Cao, Roy Ka-Wei Lee, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2611.18630)  

**Abstract**: Cyber-enabled fraud is a dominant global threat that causes severe financial loss, identity theft, and widespread psychological harm. This illicit economy relies on a highly specialized underground ecosystem where distinct actors exchange technical tools, operational techniques, and laundering channels. This research presents a case study of one such live fraud ecosystem from the perspective of a cohesive six-stage value chain spanning from preparation to monetization: Capability, Data Acquisition, Technique Sharing, Identity Forgery, Execution, and Extraction. Analyzing 67,976 posts and 305,048 comments from nine communities on the Tor-based darknet forum Dread, we map user trajectories, content distribution, and the adoption of generative AI. Our findings show that structural cohesion within the ecosystem is maintained by a thin layer of `bridgers' who generate the majority of all content. Against a degree-preserving null model, three knowledge-building upstream stages of the value chain form a tightly coupled cluster, whereas the remaining three operational downstream stages resolve into highly specialized populations with small personnel overlap. Finally, we demonstrate that generative AI is being rapidly but unevenly absorbed across the value chain; the technology saturates knowledge-building communities while operational stages remain largely untouched. By quantifying criminal specialization and structural integration, our network-based approach provides a framework for targeted, high-impact law enforcement interventions. 

---
# Decision 1.0: Towards Open Decision Foundation Models 

**Authors**: Xunzhuo Liu, Haichen Zhang, Andy Luo, Huamin Chen, Xue Liu, Bowei He  

**Link**: [PDF](https://arxiv.org/pdf/2611.18629)  

**Abstract**: Intelligent software needs reusable judgments inside its control flow: selecting an action, testing a condition, or assessing evidence against a rubric. We present Decision 1.0, an open-weight family that turns textual state and runtime-defined questions into typed probability distributions. Its Choice, Noul, and Score interface lets applications specify outcomes at inference time and compose predictions into agents and workflows. Each question produces its candidate distribution without autoregressive text generation. Six models cover bidirectional encoder and causal decoder architectures, including a workflow specialist. Training draws on verifiable tasks, natural-language annotations, and workflow adaptation. On a common decision benchmark, Lux-9B achieves 84.17\% on explicit decisions and 91.25\% on inference, with a weighted overall score of 77.22\%. Results also expose gaps in composition, transfer, and probability calibration. Released weights and inference code provide a basis for developing decision models as reusable components of software. 

---
# EmoTrace: Recursive Trace-Factorized Generation for Emotional Support Conversation 

**Authors**: Mingwen Zhang, Minqiang Yang, Yang Yu, Haobo Liang, Yongfeng Tao, Changsheng Ma, Bin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18628)  

**Abstract**: Fundamental to effective emotional support is the iterative interpersonal dynamic wherein a supporter intervention continuously shapes the psychological trajectory of the seeker. While contemporary models capture variables at the turn level, including transient emotions and discrete strategies, they inherently treat these factors as isolated events. By retaining strictly literal dialogue histories while discarding the underlying decision logic, such architectures remain fundamentally incapable of sustaining coherent extended support. Here we introduce EmoTrace, a recursive method utilizing trace factorization to preserve explicit decision continuity throughout interactions spanning multiple turns. Departing from others, our approach formulates each conversational turn as a structured interaction increment encapsulating the evolving state of the seeker, the selected strategy, the affect of the supporter, and the ultimate textual realization. Accumulated sequentially, this trajectory of increments functions as an explicit structural trace informing subsequent interventions. Optimization of this factorized generation is achieved through two complementary mechanisms. Specifically, a value guidance mechanism aligns discrete factor selection with interaction values over an extended horizon, whereas a trace conditioning mechanism refines textual realization contingent upon the fixed strategic trace. Empirical evaluations across standard benchmarks demonstrate that EmoTrace yields substantial improvements in both strategic planning and response generation. Crucially, rollouts spanning multiple turns reveal that these retained structural traces confer compounding benefits over extended dialogues, establishing a robust mechanism for sustained affective alignment in human-machine interaction. 

---
# FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models 

**Authors**: Jian Hu, Yifan Zhang, Hao Zhang, Binfeng Xu, Shaokun Zhang, Hongqing Peng, Zhiding Yu, Pavlo Molchanov, Jan Kautz, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.18627)  

**Abstract**: Asynchronous reinforcement learning suits long-horizon agents with irregular rollout times, but group-relative updates require repeated rollouts of each prompt. However, group-based sampling limits prompt coverage at a fixed rollout budget and introduces group synchronization barriers. One rollout per prompt avoids these constraints but removes the variance-reducing group baseline, making updates noisier and more prone to instability on stale, variable-length trajectories. We present FlashREINFORCE, a critic-free, one-pass framework built on One-Batch REINFORCE, a Sequence Trust Region, and Sample-Mean Optimization. These components preserve prompt coverage and signed feedback while controlling policy drift and length-amplified negative updates. In our experiments, FlashREINFORCE remains stable through 6,000 updates on DeepSeek-R1-Distill-Qwen-1.5B at a policy lag of approximately four updates. On Qwen2.5-Math-1.5B, it reaches 38.0 mean accuracy across five benchmarks, outperforming the reported GRPO baseline by 1.7 percentage points with half the rollouts (256k versus 512k). Python-tool experiments extend to the 30B MoE model Qwen3-30B-A3B, with stable training at policy lag eight. On Qwen2.5-7B-Instruct, FlashREINFORCE maintains tool use where GRPO stops making tool calls and achieves 98.3%/96.5% seen/unseen ALFWorld success. 

---
# Interpretability-Guided Auditing for Clinical Language Models for Discharge Prediction 

**Authors**: Ha Na Cho, Sairam Sutari, Alexander Lopez, Hansen Bow, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.18626)  

**Abstract**: Clinical natural language processing (NLP) models for hospital discharge prediction are vulnerable to explicit discharge-related lexical cues in clinical notes, even when note timestamps are temporally filtered. These cues may act as proxies for future outcomes, leading models to rely on shortcut-like signals rather than clinically meaningful evidence. We propose AuditBERT-Discharge, a lightweight framework that reframes interpretability as a training-time auditing mechanism. Token-level SHAP attributions from an initial BioClinicalBERT model are used to identify highly attributed discharge-related tokens, which are subsequently masked in the training corpus before model retraining. We evaluate the framework on a retrospective cohort of elective spine surgery encounters from a single academic medical center. Qualitative attribution analysis and preliminary evaluation suggest that SHAP-guided auditing reduces reliance on explicit discharge-proxy language and yields more conservative, better-calibrated predictions, although improvements in discrimination are not uniform. These findings demonstrate the potential of interpretability-guided auditing as a practical strategy for identifying and constraining shortcut-like lexical dependencies in temporally sensitive clinical NLP tasks. 

---
# Skill Transfer: Adapting Agent Skills from Strong to Weak Agents 

**Authors**: Yifan Lan, Hanyu Wang, Lu Lin, Jinghui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.18624)  

**Abstract**: Agent skills are modular bundles of instructions, code, and resources that an agent loads on demand, increasingly authored and validated on frontier models and shared as an open standard. The same skill behaves differently across models: a strong model turns it into a large performance gain, while a weak model realizes little of that gain and sometimes does worse than with no skill at all. Cost and resource constraints, however, often restrict real deployments to exactly these weaker agents. We hypothesize that the strong model silently supplies procedural steps the skill leaves implicit, which the weak model omits, and formalize closing this gap as \emph{Skill Transfer}: holding the skill's capability ceiling and the model's parameters fixed, we adapt the skill so a specific weak agent can execute it well. Our method abstracts strong and weak trajectories into typed execution structures, localizes their structural divergence with a graph edit distance (GED), and repairs the skill through an iterative diagnoser-patcher loop. Across three benchmarks and two model families, Skill Transfer raises pass rate over the unmodified skill significantly and well over other skill-evolution baselines. The adapted skill also improves the strong agent it was never adapted for, and transfers out of distribution to a structurally related, unseen task. Code is available at this https URL 

---
# Evaluating the Robustness of Fact-Checking LLMs on Multilingual Evidence 

**Authors**: Radhika Shah  

**Link**: [PDF](https://arxiv.org/pdf/2611.18623)  

**Abstract**: Automated fact-checking benchmarks typically pair claims with clean, curated evidence in high-resource languages, but real-world evidence is often incomplete, misleading, or written in a language where models are weaker. We evaluate the robustness of fact-checking LLMs for different languages. Building on EuroVerdict, we construct gold evidence sets for 190 claims in each of English, German, Greek, and Romanian, then apply four corruption strategies at two severity levels. We test LLMs on a set of claims, providing them with evidence from the sets above. A rubric-based LLM-as-a-judge scores reasoning quality. Verdict accuracy proves highly sensitive to both factors: Misleading edit cut Llama 3.1 8B's English accuracy from 80.00% to 16.84%, and losses widen as language resource level decreases. Machine-translating Greek and Romanian claims into English recovers much, but not all, of the low-resource gap. Judge scores decline more slowly than accuracy, suggesting that automated reasoning evaluation underestimates degradation. These results caution against deploying LLM fact-checkers on noisy multilingual evidence, and we release our evidence datasets and evaluation code to support future work. 

---
# RGFC: Registry Grounded Function Calling with stage-decomposed evaluation 

**Authors**: Ritvik Garimella, Vedant Khandelwal, Anvi Kohli, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2611.18622)  

**Abstract**: Tool-calling evaluations often report whether the final call matches a reference answer, but this single outcome merges different failures: choosing a function outside the available registry, violating the selected schema, or filling arguments with values not supported by the user request. These failures matter more when models must choose from larger tool registries. We introduce RFCD, a registry-scaled function-calling benchmark built from BFCL, Glaive Function Calling v2, and Tool-Call-Data, with standardized JSON-schema registries ranging from 5 to 3,000 functions. We also introduce TAAG, a deterministic stage-decomposed evaluator for registry conformance, structural completeness, and argument grounding. Across six locally served sub-4B models, three data sources, and four feedback modes, registry scale is the dominant source of degradation: single-pass TAAG pass rate drops sharply from small to large registries on all sources. Repeated feedback improves some grounding metrics but adds substantial token cost and does not consistently improve end-to-end F1. Requirement-level TAAG-R feedback gives modest gains over stage-only TAAG-S feedback, which indicates that TAAG signals are useful for analysis and correction probes but do not solve registry scaling by themselves. RFCD and TAAG provide an evaluation resource for measuring where tool-calling systems fail as registries grow. 

---
# Which Mathematical Context Supports Target-Aligned Conjecture Generation? A Paired Ablation Study 

**Authors**: Austin LaHue, Michael Burr, Luis David Garcia Puente, Vinita Gangaram Jansari, Benjamin Nye, Carlos Toxtli Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2611.18621)  

**Abstract**: Generating mathematical conjectures from natural-language context is challenging because theorem-relevant information is distributed across multiple forms of mathematical context, including definitions, objects, assumptions, and constructions. We study which of these contextual signals help a model recover the specific claim supported by the surrounding setup, using 61 main theorems from 12 papers across four mathematical domains. For each theorem, we redact the result and represent the remaining setup using six context components. A paired ablation design compares full context, six leave-one-out conditions, and six single-component conditions, yielding 793 generated conjectures. Context components were largely complementary: removing one component usually had little effect, whereas retaining only one substantially reduced expected-theorem alignment. Local Mathematical Objects provided the strongest standalone signal, while Structural Constraints and Conditions produced the largest alignment decrease when removed. These findings suggest that target-aligned conjecture generation depends on combining information that identifies relevant objects with information that constrains admissible claims. 

---
# CondRelBench: A Mixed-Domain Benchmark for Condition-Scoped Relation Understanding 

**Authors**: Sydney Anuyah, Bofu Dong, Sudhasmita Swain, Sunandan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2611.18620)  

**Abstract**: Relation extraction systems often identify what a relation is, but not the condition under which it holds. This can collapse condition-bound claims into unsupported generalizations, a failure we call relation overgeneralization. We introduce CondRelBench, a mixed-domain benchmark for evaluating condition-scoped relation understanding. Each claim is represented as (E1, R, E2 | C), where C is the condition licensing the relation. Using 3,000 synthetic paragraphs, 200 PubMed full-text papers, and 695 Reddit posts, we construct five probe variants per tuple that preserve, replace, remove, or contradict the condition, or contradict the relation itself. Models classify each probe as Supported, Not Supported, or Not Enough Evidence. We also introduce Overgeneralization Rate (OGR) and Scope Preservation Score (SPS) to measure false support for altered claims and consistency across probe families. Across prompted LLMs, fine-tuned LLMs, and encoder models, we find that Accuracy and Macro F1 often hide scope failures. Models frequently support the original scoped claim but fail when its condition is removed or altered. Domain-specific encoders achieve the strongest scope preservation, while instruction-tuned LLMs often show higher overgeneralization. Synthetic supervision helps but does not close the gap to real-world PubMed and Reddit performance. 

---
# Assisting Patients with Reliable Information Gathering: A Guide to Building Efficient QA Systems for Medical Use 

**Authors**: Zhipeng Yin, Zichong Wang, Min Chen, Ian Stockwell, Xin Ning, Jun Liu, Wenbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18619)  

**Abstract**: The application of question-answering (QA) systems in the medical domain has rapidly advanced, significantly improving patients' access to reliable health-related information. However, current approaches face notable challenges, including the difficulty in obtaining large-scale and unbiased medical datasets, significant privacy concerns, and inefficiencies due to manual dataset annotation. To address these issues, This study introduce a novel methodology leveraging publicly accessible online health forums to systematically build an unbiased, privacy-conscious QA dataset, and it employs Topic-guided Semantic Modeling (TGSM) for automated topic identification, enabling efficient and targeted annotation of relevant patient-generated content. Subsequently, this study propose a two-stage QA pipeline based on a Retriever-Reader architecture, which is further enhanced through fine-tuning state-of-the-art transformer-based models on the constructed domain-specific QA dataset. Experimental results demonstrate that our fine-tuned BioBERT significantly outperforms existing benchmarks, offering accurate patient-derived insights and providing a replicable framework for building efficient medical QA systems. 

---
# Intrinsic Attention as Navigator: A Non-Generative Retrieval Method for Multi-Hop Queries 

**Authors**: Linwei Song, Weijie Zhang, WeiDi Chen, Huixia Lai, Shi Zhang, Ruliang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18617)  

**Abstract**: For multi-hop queries in retrieval-augmented generation (RAG), it is crucial to accurately obtain evidence pieces. In this study, we propose a training-free method for multi-hop query retrieval, named Intrinsic Navigating-based Non-Generative Retrieval (IN-Retriever). Instead of generating intermediate reasoning text, the method extracts the bridging clue from the internal attention of the embedding model based on the current context and retrieved candidate documents. During iterative clue extraction and document retrieval, IN-Retriever strips away revealed clues from the embedded query vector and then reranks candidate documents in a targeted manner, thereby steering retrieval toward hidden next-hop evidence. In the experiments, we confirm the feasibility and effectiveness of using intrinsic attention signals to guide external multi-hop retrieval. Experimental results show that our method precisely collects sufficient documents to support multi-hop queries and significantly outperforms sparse and dense retrieval methods. 

---
# The Judge Is Not Impartial: Self-Preference in Medical LLM Evaluation 

**Authors**: Zara Ansari, Shlok Natarajan, Vincent Yip, Cally Lin, Avi Udash, Mia Garvey, Aaron Fanous, Roxana Daneshjou  

**Link**: [PDF](https://arxiv.org/pdf/2611.18616)  

**Abstract**: Large language models (LLMs) are increasingly used to evaluate model-generated medical content, creating a potential circularity: an LLM judge may favor outputs generated by the same model independently of clinical quality. To characterize LLM judge self-preference in medical settings, we measured same-model affinity in single-turn specialist question answering and multi-turn standardized-patient encounters. We evaluated eight models, two from OpenAI, Anthropic, Google, and Qwen, on 620 real point of care questions from clinicians (Real-POCQi). The same eight models also generated and judged 200 standardized patient-clinician scenarios (MedSP1000) up to 8 visible turns. In total, our analysis spans 13,760 LLM judgments. We define three complementary measures of self-preference. Answer Rank Bias, our primary measure, compared the rank assigned to a fixed response by its own-model judge with its average rank from outside judges. Answer Score Bias made the analogous comparison using rubric scores. Own-Answer Win Rate measured how often a judge ranked its own model's response above a competitor. Higher values across all these measures indicate greater degrees of self-preference. In Real-POCQi, pooled Answer Rank Bias was $+1.22$ positions, and all eight judges favored their own answers. Self-preference persisted both with and without a grading rubric. Revealing generator identities modestly weakened self-preference by an average of $-0.094$ positions. In MedSP1000, pooled Answer Rank Bias ranged from $+0.53$ to $+0.62$ positions, with no clear difference by conversation length ($p=0.17$). LLM judges systematically favored responses generated by their own models across multiple formats. Clinical AI deployments should exercise increased vigilance including multiple evaluation measures and judge panels spanning differing model families when deploying LLM-as-a-judge systems. 

---
# Self-Evolving Dual-Role Deliberation for Medication Recommendation from Clinical Records 

**Authors**: Jiyue Jiang, Yu Lei, Yitao Xu, Zikang Wang, Zhenyu Wang, Licheng Zong, Han Huang, Ziyi Li, Zhiyu Huo, Yuhan Chen, Jiayang Chen, Xiaodan Fan, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.18615)  

**Abstract**: Recommending medications from electronic health records (EHRs) is a safety-critical task that demands therapeutic coverage, explicit safety constraints, traceable justification, and continual adaptivity. Supervised neural models freeze at deployment and cannot reason about new phenotypes, while existing LLM pipelines are single-pass and lack a cross-case improvement loop. We propose SEMPA, a training-free agentic framework that couples a Proposer with a Critic in multi-round adversarial deliberation, grounds candidates with molecular-aware retrieval, contrastively evolves both agents' natural-language policies under a semantic-anchor constraint, and enforces rule-based DDI and contraindication checks. On 100 MIMIC-IV encounters, SEMPA reaches 0.3116 Jaccard, 0.7295 F1, and a 0.0172 DDI rate, improving F1 by 57.7% and cutting DDI by 77.5% over the strongest supervised baseline. Two LLM judges rank SEMPA highest, and six-backbone studies confirm that gains derive from the framework rather than a specific model. 

---
# Bridging Semantics and Utilities: A Controllable and Scalable Market Simulation 

**Authors**: Dejun Cao, Boyang Wang, Jiayuan Wei, Tingxi Zhang, Chengyuan Zhang, Ruixuan Yan, Yishen Chen, Bufan, Honghai Yu, Jiale Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.18613)  

**Abstract**: Large language models(LLMs) have recently been used to construct generative agents for financial market simulation. Existing approaches typically prompt each agent separately with public market information and a private investor profile, and ask the model to directly generate trading actions. In this agent-wise design, the model repeatedly interprets information shared across the population, while individual behavior is controlled through textual descriptions of investor preferences and traits. This creates two structural difficulties: continuous traits such as risk sensitivity do not reliably induce ordered behavioral changes, and repeatedly processing the same public information causes inference costs to grow with the number of agents. We propose \textbf{SUM}, a Semantic-Utility Model that separates cohort-level semantic perception from agent-level utility-based decision-making. A language model interprets public market information for each investor type and produces semantic signals shared within the corresponding cohort. Individual agents then combine these signals with their private states, preferences, and constraints to generate orders through utility-based decisions. Experiments in an A-share limit-order-book market show that \textbf{SUM} provides more stable and monotonic behavioral control than prompt-based agents while substantially reducing repeated language-model inference as the population grows. Closed-loop simulations further reproduce directional market responses, volatility persistence, differentiated investment styles, and persistent wealth heterogeneity. These results establish \textbf{SUM} as a controllable, scalable, and semantically informed framework for financial market simulation. 

---
# When Helpful Agents Cause Financial Harm: Evaluating LLM Agent Safety in Executable Economic Worlds 

**Authors**: Yuzhu Ling, Jiale Han, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18612)  

**Abstract**: As LLM agents become increasingly capable of making consequential economic decisions, their safety under economic pressures remains poorly understood. We introduce EconSafeBench, the first benchmark for systematically evaluating the safety of LLM agents in executable economic environments. EconSafeBench contains 828 cases across five economic scenarios and four categories of safety violations. A key design is the matched evaluation of three sources of safety pressure, including explicit user requests, pressure from other actors, and pressure arising from the agent's own pursuit of a performance objective. We further evaluate safety at both static and dynamic scales through 30-period interactions with persistent memory and matched controls. Across eight LLMs, all evaluated models exhibit safety violations on the static benchmark, with overall violation rates ranging from 8.5% to 40.6%. We further find that violations can occur even without an explicit unsafe request, and that initially compliant agents may begin to violate as performance pressure increases over time. These findings highlight the need for stronger safety alignment and rigorous evaluation before LLM agents are deployed in high-stakes economic environments. 

---
# Understanding Failures in LLM Reasoning by Learning Structured Representations of Chain-of-Thought 

**Authors**: Tommaso Felice Banfi, Mubashara Akhtar, Jingwei Ni, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2611.18611)  

**Abstract**: Chain-of-Thought (CoT) is central to improving reasoning in large language models, yet existing evaluation methods focus on step-level signals and ignore the underlying structure of reasoning traces. This raises a key question: what aspects of reasoning quality (i.e., correctness, efficiency, and failure patterns) are encoded in the structure of a CoT trace? We introduce \textbf{CoT2Graph}, a framework that represents CoT traces as directed dependency graphs, making relationships between reasoning steps explicit and capturing patterns such as backtracking, branching, and subtask merging. We model these graphs with graph neural networks and use attribution-based explanations to identify structural patterns associated with reasoning quality. CoT2Graph reveals distinct patterns of successful and failed reasoning: correct CoTs place greater importance on early planning, whereas incorrect CoTs are associated with overloaded synthesis steps and dependency violations. We further show that graph-based signals better predict whether a CoT leads to a correct answer, outperforming previous tree-based structural methods by up to 2.5\% and simple length-based baselines by 7.8\%, while also improving Best-of-$N$ selection. Overall, our results demonstrate that reasoning structure encodes rich signals about reasoning quality, motivating a shift from sequence-level evaluation to structural analysis. 

---
# NYUAD at ImageEval Shared Tasks: A Multimodal LLM for Audio-Visual Question Answering and Hallucination Detection in English and Modern Standard Arabic 

**Authors**: Nouar AlDahoul, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2611.18610)  

**Abstract**: The increasing adoption of multimodal large language models (MLLMs) has enabled more natural interaction with visual and spoken information, creating new opportunities for intelligent audio-visual question answering (AVQA) and hallucination detection systems. AVQA extends traditional visual question answering by requiring models to jointly understand an input image and a spoken question to generate accurate, context-aware responses. Despite recent advances, AVQA and hallucination detection tasks remain challenging for modern standard Arabic (MSA) due to the limited availability of multimodal datasets and the need to reason across Arabic speech and text modalities alongside visual inputs. In this paper, we present an MLLM-based framework for AVQA that takes an image and a spoken question as input and generates answers. In addition, we utilized this framework to detect hallucination in text-image pairs. The proposed approach is evaluated on the ImageEval 2026 Task (Ayn-VQA) benchmark, which focuses on culturally grounded Arabic AVQA. Experimental results demonstrate that the proposed framework effectively understands spoken queries, grounds their responses in the visual content, and produces accurate answers. The results underscore the capacity and limitations of different LLM architectures for both English and MSA and under various conditions. 

---
# Many Paths to the Same Mistake: Generating Diverse Reasoning Paths Behind the Same Student Mistake 

**Authors**: Fareya Ikram, Mengxuan Hu, Parker Glenn, Zhiqi Huang, Alfy Samuel  

**Link**: [PDF](https://arxiv.org/pdf/2611.18609)  

**Abstract**: Realistic evaluation of AI systems requires user simulators that capture not only cooperative behavior, but also the errors and variability that arise in real interactions. In educational technology, simulating step-by-step flawed student reasoning is essential, yet datasets mapping these pathways are scarce, and Large Language Models (LLMs) often auto-correct rather than sustain simulated mistakes. To address this, we present a generate-and-filter pipeline that produces diverse and realistic student error traces. Our approach utilizes forward consistency, a metric verifying if a flawed reasoning trace can be used to recover the student's actual incorrect answer, paired with an LLM-as-a-judge to ensure realism. Our evaluation suggests that while many LLMs struggle to maintain incorrect reasoning mid-trace, smaller supervised fine-tuned models successfully capture these divergent paths. This work provides a framework for high-fidelity error trace generation, advancing realistic, non-cooperative user simulation. 

---
# NYUAD at ArGuard Shared Tasks: Multimodal Embedding Models for Detecting Arabic Hateful Memes and Unsafe Prompts 

**Authors**: Nouar AlDahoul, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2611.18608)  

**Abstract**: The rapid growth of social media and generative AI has significantly increased the volume of Arabic textual content, memes, and user prompts shared across online platforms. While these media facilitate communication, creativity, and information exchange, they can also be exploited to disseminate hate speech, harmful content, and unsafe instructions. As a result, there is an increasing need for reliable automated methods to identify hateful Arabic memes and harmful (unsafe) prompts. This paper leverages pre-trained and fine-tuned multilingual embedding models to learn rich semantic representations from textual and visual content. The evaluation is conducted using a dataset of Arabic memes and prompts proposed in the ArabicNLP ArGuard 2026 challenge. The results underscore the capacity of large language model (LLM)-based embedding models to deliver superior performance. Developing robust detection systems is essential for improving content moderation and fostering a more secure and trustworthy online environment for Arabic-speaking users. 

---
# Justification of Emotional Appeal in Computational Argumentation 

**Authors**: Yanran Chen, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2611.18607)  

**Abstract**: Emotional appeals (EAs) are often treated as fallacious in computational argumentation, yet they may be justified when warranted by the context. We introduce the task of predicting EA justification and present the first human-annotated dataset for this task, covering 1,176 argument--context pairs. Grounded in philosophical literature, our framework evaluates EA justification along logical and emotional dimensions. Human annotations show that justification criteria form a shared latent judgment space, with evidential proportionality and normative acceptability most strongly associated with overall justification. We further find that EAs without logical support are not always judged fully unjustified, and stronger EAs are not necessarily judged less justified. Finally, LLM experiments show that EA justification prediction remains challenging. 

---
# Syntactic Complexity Convergence in Dialogue: Analysis of the Phenomenon and Application to LLM Detection 

**Authors**: Yu Wang, Yanran Chen, Yifan Wang, Steffen Eger, Hendrik Buschmeier  

**Link**: [PDF](https://arxiv.org/pdf/2611.18606)  

**Abstract**: We revisit the under-explored phenomenon of syntactic complexity convergence in dialogue. Extending prior topic-level analyses to full conversations, we examine five human-human datasets and a set of human-LLM dialogues. We integrate and evaluate a multidimensional set of syntactic complexity metrics to quantify structural variation in dialogue. Our analyses reveal robust syntactic complexity convergence across human-human datasets, especially in task-oriented interactions, and statistical tests confirm that these effects are not due to chance. In contrast, human-LLM dialogues show no significant convergence under individual metrics; however, linear-CKA uncovers a distinct cross-metric convergence pattern. Motivated by these differences, we use syntactic complexity as a feature for detecting LLM-involved dialogue, achieving 96\% accuracy. Our classification results demonstrate its potential without relying on likelihood-based signals such as surprisal, which require a language model as the estimator. 

---
# Enhanced English-Sanskrit Translation using Multilingual NMT Approach 

**Authors**: Sahinur Rahman Laskar, Anshuman Singh, Aditya Pratap Singh Sisodia  

**Link**: [PDF](https://arxiv.org/pdf/2611.18605)  

**Abstract**: Neural Machine Translation (NMT) is a widely accepted technique of automatic translation tasks of Natural Language Processing. However, NMT requires a sufficient amount of parallel corpora to attain a reasonable translation quality. In this work, we have considered a low-resource English-Sanskrit pair for English-to-Sanskrit and Sanskrit-to-English translation. We have performed a comparative analysis by exploring bilingual and multilingual approaches using LSTM-based Sequence-to-Sequence, and fine-tuning a pretrained model IndicTrans in a multilingual NMT approach. To investigate multilingual transfer learning, we additionally construct multilingual translation settings by incorporating Hindi along with English and Sanskrit using language-tagged data. Experimental results demonstrate that IndicTrans-based multilingual models outperform baseline results in terms of automatic evaluation metrics such as BLEU and ChrF for both directions of translation. 

---
# One Question, Four Voices: How Advice for Alzheimer's Caregiving Differs Between Caregivers, Clinicians, and Large Language Models 

**Authors**: Congning Ni, Jinkyung Katie Park, Yang Li, Sarvech Qadir, Sean S. Huang, James S. Powers, Julia A. Hiner, Vania Leung, Patricia Commiskey, Lijun Song, Qingxia Chen, Pamela J. Wisniewski, Bradley A. Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.18604)  

**Abstract**: Online caregiver forums are an important source of emotional and informational support for Alzheimer's disease and related dementias (ADRD). We compare responses to 85 real caregiver questions from ALZConnected across four responder groups: peer caregivers, physicians, ChatGPT, and a retrieval-augmented CareGPT grounded in peer discussions. Using 28 measures spanning linguistic form, emotional support, informational support, and topic scope, we examine how these voices differ when answering the same questions. Peers provided time-anchored narratives and community grounding; clinicians offered concise, advice-forward, clinically bounded guidance; and LLMs produced a formal, verbose, highly structured consultation style. CareGPT showed minimal measurable separation from ChatGPT, suggesting that retrieval plus prompt injection alone was insufficient to shift communicative stance. LLM outputs scored higher than peers on politeness and emotion-support markers and were comparable to clinicians on informativeness and physician-alignment measures, but were less narrative and less contextually situated. Topic modeling showed that LLM responses concentrated on advice-oriented themes, whereas peers covered a broader range, including emotional processing, legal and financial concerns, and daily care. These findings suggest that AI can complement, but not replace, human contributions to ADRD caregiver support and motivate hybrid systems that combine structured AI delivery with the distinct strengths of human voices. 

---
# Retrieval-Augmented Generation for Clinical Decision Support in Gait Analysis: A Proof-of-Concept 

**Authors**: Georgios Giarmatzis, Fabian Horst, Djordje Slijepcevic, Nikolaos Aggeloussis, George Papakostas  

**Link**: [PDF](https://arxiv.org/pdf/2611.18603)  

**Abstract**: Large language models (LLMs) demonstrate significant potential for clinical decision support but lack specialized biomechanical knowledge required for accurate gait analysis interpretation. We developed and evaluated a Retrieval-Augmented Generation (RAG) framework to enhance LLM-generated clinical explanations for ankle fracture rehabilitation using ground reaction force data. Using the GaitRec dataset (503 subjects: 211 healthy, 292 ankle fracture patients), we implemented a five-stage pipeline integrating: (1) machine learning classification (91.3\% accuracy); (2) Layer-wise Relevance Propagation for explainable feature identification; (3) RAG knowledge base from peer-reviewed biomechanics literature; (4) GPT-5 explanation generation with and without retrieval augmentation; and (5) double-blind evaluation by experienced physical therapists across five clinical criteria. RAG augmentation produced statistically significant improvements in Clini-cal Benefit (mean difference=0.60, p=0.047, {\delta}=0.49) and Completeness (mean difference=0.70, p=0.049, {\delta}=0.49), both demonstrating large effect sizes. Baseline GPT-5 achieved strong performance across all criteria (\textgreater4.0/5.0). This proof-of-concept demonstrates that augmenting general-purpose LLMs with domain-specific knowledge meaningfully enhances actionable, evidence-based clinical guidance---dimensions most critical for rehabilitation decision-making. The framework provides a scalable approach for integrating AI into specialized clinical workflows without costly model retraining. 

---
# Internal Verdicts Track Evidence: A Template-Controlled Neutrosophic Reading of Epistemic States in Large Language Models via the Jacobian Lens 

**Authors**: Maikel Yelandi Leyva-Vazquez, Alexis Matheu Perez, Florentin Smarandache  

**Link**: [PDF](https://arxiv.org/pdf/2611.18602)  

**Abstract**: For twenty-five years, neutrosophic logic has assigned independent degrees of truth (T), indeterminacy (I), and falsity (F) to epistemic states that classical and fuzzy semantics cannot express. Previous applications measured these components on outputs: answers, judgments, expert evaluations. This paper reports, to our knowledge, the first neutrosophic reading of components inside the internal representations of a large language model. Using the recently released Jacobian lens on Qwen3.5-4B, we project intermediate-layer readouts onto lexicons of support, refutation, and hedging, obtaining layer-wise (T, I, F) profiles under conflicting evidence, first-person false belief, and factual control (n = 20 each). A first battery yielded three striking signatures: a surge of refutation mass under conflict, sustained indeterminacy under false belief, and an apparent "verdict collapse" before the output layer. A second, template-controlled battery showed that all three, as initially stated, were largely artifacts of prompt grammar: matched templates with agreeing evidence inflate the same lexicon masses, true beliefs elicit the same hedging, and the model verbalizes its verdict when allowed to generate. What survives the controls is stronger than what died: the neutrosophic verdict balance B = log10(F) - log10(T) tracks evidence polarity under identical templates (about 3 orders of magnitude of separation, Cohen's d = 2.3-2.5, p < 10^-6 in 9 of 9 layers; paired Wilcoxon: median difference +2.4 orders, 19/20 items, p = 3.8x10^-6) and residually discriminates false from true first-person beliefs. We argue that the independence of the neutrosophic components is what keeps the common-mode channel observable, making both the measurement and its self-correction possible, and we distill the two-battery design into a reusable protocol for auditing language model internals with lens-based readouts. 

---
# Jailbreak LLMs with Linguistic Style as a Hidden Attack Surface 

**Authors**: Priyanshu Mohanty, Sarthakbrata Halder, Srikant Panda, Afshin Oroojlooy, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2611.18601)  

**Abstract**: We show that linguistic style is a systematic and previously overlooked attack surface for LLM safety. Across a diverse taxonomy of styles and multiple state-of-the-art models, stylistic rewrites of identical harmful goals produce order-of-magnitude differences in Attack Success Rate (ASR), with some styles achieving up to 80\% ASR while others remain below 6\%. To exploit this vulnerability, we introduce a lightweight single-pass attack framework that combines a style-conditioned generator with a BERT-based style selector, achieving performance competitive with substantially more expensive multi-turn optimization methods. To understand why style affects safety behavior, we conduct mechanistic analysis on LLaMA-3.1-8B and find that stylistic rewrites disrupt refusal representations while preserving recoverable harmful intent, suggesting a distributed failure mode rather than a single refusal mechanism. Motivated by this finding, we develop a style-aware Direct Preference Optimization defense that reduces ASR from 86\% to 19.5\% while largely preserving utility. Together, our results establish linguistic style as a critical factor in adversarial robustness and provide both mechanistic insight and a practical defense. 

---
# ScopeBench: Evaluating Cultural Norm Scope Awareness in LLMs 

**Authors**: Haojun Liang, Ziwei Zhao, Chenyu Li, Qiyue Zhao, Rongduomijia Mu, Yulan Shen, Yu Gu, Chen Zhang, Kairui Zhang, Yanbiao Ma, Wenxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18600)  

**Abstract**: Safety evaluation in large language models treats harm as culturally invariant, yet a model cannot correctly apply a safety norm without knowing whose norms bind. We introduce ScopeBench, a benchmark that treats cultural scope as a first-class safety dimension. It comprises 3,323 norms spanning 49 countries across nine cultural clusters, each annotated as Universal or Specific. We define three scope error types identically across four evaluation tasks: Scope Inflation, Scope Deflation, and Scope Displacement. Evaluating 21 frontier LLMs across seven model families, we surface three structural failures. First, the dominant error type inverts with prompt format: Scope Deflation reaches 70% in single-answer classification, while Scope Inflation reaches 33% when each candidate culture is judged independently. Second, scope-classification accuracy is empirically decoupled from behavioral safety: the six classifiers achieving at least 97% Task 1 accuracy still produce harmful responses to 40-42% of scope-violating statements. Third, no model is safe in interaction, with harmful-response rates ranging from 29% to 51% across the cohort, excluding one model that systematically refuses the task. We release all data, code, and model outputs to establish cultural scope as a distinct alignment axis. 

---
# Between the Lines: Multi-turn Contextual Investment Advice Detection 

**Authors**: Mingshu Li, Adam Boncz, Mate Toth, Justin Williams, Justin Chan, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2611.18596)  

**Abstract**: As large language models (LLMs) increasingly power financial conversational assistants, detecting investment advice in natural language has become a critical regulatory and safety challenge. Unlike explicit single-turn recommendations, investment advice often emerges implicitly and progressively over multi-turn conversations, shaped by user intent, delivery style, and contextual buildup. We introduce the first comprehensive framework for detecting both sequential and contextual investment advice in dialogue settings. Central to our approach is a modular, top-down synthetic data generation pipeline that systematically spans a taxonomy of advisory trajectories using an axis-bucket-descriptor schema. We simulate both benign and high-risk conversations across diverse user personas, financial instruments, and risk tiers, and evaluate detection capabilities using LLM-as-a-Judge with entropy-based uncertainty diagnostics. Our results show that contextual advisory detection remains substantially more challenging than sequential cases, even under few-shot prompting with tailored demonstrations. This work lays the foundation for scalable guardrail development in high-stakes, compliance-sensitive domains, where advisory risk is often subtle, distributed, and perception-dependent. 

---
# Profiling Hallucinations in Frontier LLMs for Entity Linking to Medical Ontologies 

**Authors**: Logan Born, Nishant Kambhatla, Uliyana Kubasova, Maryam Siahbani, Andrei Vacariu, Tim O'Connell, Anoop Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2611.18595)  

**Abstract**: The integration of Large Language Models (LLMs) into healthcare promises to revolutionize clinical documentation and interoperability, yet reliability remains a concern. This study presents a comprehensive analysis of hallucinations by frontier LLMs tasked with mapping clinical text to SNOMED CT. Through rigorous experimentation, we identify a critical reliability gap: LLMs hallucinate medical codes at a rate that currently renders them unsuitable for autonomous clinical coding applications. Paradoxically, constraining models to use ground-truth mention spans exacerbates, rather than mitigates, these hallucinations. We further contribute a taxonomy of hallucination types--including deprecated codes and cross-ontology errors--and demonstrate that general-purpose LLMs significantly underperform compared to specialized zero-shot entity linking approaches. These findings underscore the need for robust verification mechanisms before clinical deployment. 

---
# Whose Side Are You On? Measuring Stance at Scale for Reddit Discourse Analysis During the Israel-Gaza War 

**Authors**: Shahar Oded  

**Link**: [PDF](https://arxiv.org/pdf/2611.18594)  

**Abstract**: Large-scale discourse analysis often requires stance labels, yet for many high-salience political events such labels are unavailable at the comment level. We introduce and validate a comment-level stance measurement instrument that assigns Pro-Israel, Pro-Palestinian, or Undefined stance using Reddit comment text. The instrument is calibrated from a small, manually annotated gold-standard dataset and scaled through large language model-assisted labeling, achieving weighted-$F_{1}=0.88$ against held-out human annotations. To enable corpus-scale measurement, we operationalize the instrument as a lightweight classifier based on DistilBERT embeddings and an RBF SVM, which achieves weighted-$F_{1}=0.7935$ on the original gold-standard set. Applying this instrument, we measure the stance of approximately 2.7 million Reddit comments posted between 2023 and 2025 during the Israel-Gaza war. We demonstrate what this measurement enables by analyzing stance-conditioned asymmetries in sentiment, toxicity, and emotional framing across political affiliations. This work shows how validated, low-cost stance measurement can support large-scale descriptive analysis of polarized discourse, without relying on network structure or prior labeled corpora. 

---
# Theoretical Foundations of Computational Islamic Jurisprudence (Fiqh) 

**Authors**: Elnaserledinellah Mahmoud Abdelwahab  

**Link**: [PDF](https://arxiv.org/pdf/2611.18593)  

**Abstract**: This paper presents the theoretical foundations and a comprehensive formal system for Computational Islamic Jurisprudence (Fiqh). We begin by establishing six foundational principles that distinguish computational Fiqh from conventional natural language processing and general theological formalization. These principles define Islamic jurisprudence as a normative deductive science operating on empirically grounded facts, necessitating a dual inductive-deductive methodology, and requiring a logical foundation specifically engineered for Fiqh rather than adopted from existing formalisms. Based on these principles, we construct a three-layer stratified computational language, $\mathcal{F} = D \circ A \circ G$, consisting of: (1) an Arabic Morphological/Grammatical Grounding Engine that transforms fully vocalized Classical Arabic into language-independent Grounded Action Structures (GAS); (2) the Arabic Reified Action Calculus (ARAC) that normalizes these structures into canonical logical representations; and (3) a function-free Horn-clause Datalog engine for executable legal deduction. Furthermore, we introduce a bridging methodology (Algorithm $\Omega$) that dynamically generates a Fiqh-Categorized Attribute Grammar from inductively extracted Horn clauses via an Ontological Meta-Map ($\mathcal{M}_{onto}$), seamlessly connecting the inductive extraction of legal rules to their deductive application. The paper rigorously proves the system's semantic correctness, representational completeness, termination, and decidability. Crucially, by leveraging morphosyntactic inflection markers ($\delta$ modes) as structural constraints, we prove the strict unambiguity of the proposed grammar, achieving deterministic $O(n)$ linear parsing complexity. This work provides a mathematically verifiable, epistemologically grounded, and computationally tractable foundation for automated Islamic jurisprudence. 

---
# Endogenous Preferences as Vulnerability Signals: Discovering Universal Adversarial Triggers for LLM Utility Degradation 

**Authors**: Qingyu Wu, Zeyu Feng, Yuzhe Luo, Hua Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.18592)  

**Abstract**: Deployed large language models must remain useful when untrusted text enters their prompt context. Adversarial prompting and red-teaming methods typically define the failures they seek through task labels, harmful targets, judge feedback, or downstream utility scores. We ask whether a frozen model's clean behavior can reveal a reusable context vulnerability when its task-level form is unknown, and downstream task feedback is withheld. To study this question, EndoUT uses a Siamese-style design that compares the same frozen victim on clean and span-modified versions of each unlabeled instruction. Without a task-level target, the clean pass supplies the victim's preferred response-onset token as an instruction-conditioned reference; Endogenous Preference Disruption (EPD) measures how insertion weakens that preference. An auxiliary generator supplies bounded-span proposals; EPD guides its update and ranks the pool without access to downstream information. The highest-scoring victim-specific span is fixed before evaluation and reused unchanged across all held-out inputs. Across four instruction-tuned models and nine benchmarks, fixed triggers reduce benign-task utility by 40.2 percentage points on average, with negative mean change on every benchmark and greater average degradation than every evaluated baseline under the reported protocols. Matched protocol ablations show stronger transfer for source-side selection followed by fixed reuse than for null-context fixed reuse or per-input generation. Transfer recurs across proposers with proposal-dependent strength; output-level analyses identify victim-specific associations between trigger families and downstream failure patterns. Thus, clean behavior can expose context vulnerabilities that transfer across inputs yet manifest through structured, victim-specific patterns of utility degradation. The code and data will be released upon acceptance. 

---
# Where Language Models Fail Like Aphasic Patients and Where They Don't 

**Authors**: Yong Yang, Roger Newman-Norlund, Xiang Guan, Saeed Ahmadi, Regan Willis, Nadra Salman, Kalil Warren, Sophie Arheix-Parras, Srihari Nelakuditi, Rutvik H. Desai, Leonardo Bonilha, Christopher Rorden, Julius Fridriksson  

**Link**: [PDF](https://arxiv.org/pdf/2611.18591)  

**Abstract**: The computational architecture of human word production -- whether semantic-lexical retrieval and phonological encoding constitute dissociable processing stages -- has been debated for decades using hand-designed models with small vocabularies. We introduce computational architectural dissociation: using known architectural properties of vision-language models (VLMs) to derive falsifiable predictions about which patient error types should and should not align with model errors under perturbation. Because VLMs perform picture naming through learned semantic competition over full vocabularies but lack phonological encoding entirely, perturbation-induced errors should match patient patterns for semantic-stage categories but not phonological ones. Testing LLaVA 1.6 (Vicuna-13B) against 88 patients with aphasia across 175 Philadelphia Naming Test items (~37 million naming attempts, 50 perturbation seeds), we find that six of eight error categories show significant item-level alignment (Spearman $\rho$ = 0.32-0.50, all p < 0.001) while the two phonological categories -- formal errors and abstruse neologisms -- show no alignment ($\rho$ = 0.110 and 0.084, both n.s.), confirming the predicted dissociation. Spectral analysis of the output probabilities places each error type at a distinct position in a six-dimensional space, giving a mechanistic interpretability tool. The dissociation is layer-specific (peaking at L22, 55% depth), severity-independent, and stable across seeds (split-half r > 0.998). Weight perturbation of four additional 7B VLMs spanning the tested architectural combinations produced correct-rate alignment of comparable magnitude ($\rho$ = 0.21-0.27). Whether the per-category dissociation itself depends on scale or on attention design requires a comparison within a single inference implementation, which is reported separately. 

---
# OmniTag-LLM: Multilingual Multitask Speech Tagging 

**Authors**: Aref Farhadipour, Francesco Verdini, Antonis Asonitis, Pierre-Edouard Honnet, Marzieh Razavi, Vijeta Avijeet, Juan Pablo Zuluaga Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2611.18589)  

**Abstract**: Multimodal LLMs promise to replace the traditional cascade of specialist speech models with a single system that, from one audio clip, jointly transcribes and labels emotion, gender, language, and named entities. Yet most benchmarks still score these tasks in isolation, and largely in English. We introduce OmniTag-Bench, a five-language benchmark (en/de/es/fr/it) and training recipe for joint understanding: one audio clip in, one JSON object out (transcript, emotion, gender, LID, token-level NER). We evaluate ten open multimodal LLMs and a LoRA fine-tuned Qwen3-Omni on a unified prompt and scorer. Beyond aggregate scores, OmniTag-Bench reveals a systematic failure mode: several strong models often transcribe or translate non-English speech into English, inflating multilingual WER. Fine-tuning improves the base model on NER, emotion, and WER while nearly doubling accuracy on independent human-labeled emotion corpora, suggesting transferable gains beyond pseudo-label imitation. The fine-tuned model also generalizes to unseen but related languages, transferring to held-out Dutch. We release prompts, the scorer, and the labeling pipeline. 

---
# Automating literature screening and knowledge synthesis with large language models 

**Authors**: Zhanzhao Li, Qiyao He, Weiqiang Chen, Kengran Yang, Kai Gong  

**Link**: [PDF](https://arxiv.org/pdf/2611.18588)  

**Abstract**: Scientific literature is expanding rapidly, yet much of its knowledge remains difficult to organize and reuse due to the substantial workload needed to navigate large, heterogeneous literature corpora. Here, we develop a generalizable large language model (LLM)-powered framework that automates literature screening, structured information extraction, and knowledge synthesis within a unified workflow. We benchmarked 11 state-of-the-art proprietary and open-weight LLMs across three materials science screening tasks differing in scope, corpus size, and class imbalance. Seven LLMs achieved balanced accuracies above 0.90 across all tasks, with the best-performing LLM reaching 0.96--0.98, demonstrating the potential of LLMs to accurately screen large, highly imbalanced scientific corpora. The framework also substantially reduces manual effort, screening 10,000 records within four minutes compared with approximately 83 person-hours for manual screening. Beyond literature screening, the framework extracts structured information from full-text articles, enabling content-level quantification of research landscapes that extends traditional bibliometric analysis based on publication metadata. This capability reveals methodological trends and underexplored directions, as shown for machine learning in concrete materials science. We further demonstrate its utility for information and data discovery by identifying publications containing silicate glass density data that are difficult to retrieve using conventional search strategies, which can support downstream hypothesis development, research design, and data-driven scientific discovery. Overall, these results demonstrate the broad applicability of LLM-powered literature analysis across scientific domains and its potential to transform large, unstructured literature corpora into structured, analyzable knowledge at scale. 

---
# Human vs Machine Translation Detection: A Cross-Model, Cross-Domain, and Low-Resource Analysis 

**Authors**: Yabsera Yemanberhan, Prasenjit Mitra, Roald Eiselen  

**Link**: [PDF](https://arxiv.org/pdf/2611.18586)  

**Abstract**: The growing reliance on web-crawled corpora for training large language models introduces a critical challenge: machine-translated (MT) content is often treated as human translation (HT), potentially degrading data quality and propagating synthetic artifacts, especially in low-resource settings. In this work, we study MT-HT detectability across high-resource (English, Spanish) and low-resource (Swahili, Afrikaans) languages using both decoder-only LLMs in zero-shot and few-shot settings, and fine-tuned multilingual encoders trained at the sentence level and on token-chunked inputs. We find that decoder-only models provide limited discrimination overall and exhibit a consistent class-asymmetric failure mode: they recognize human-written text reliably (high HT recall) but rarely flag fluent machine output (low MT recall). Encoder-based classifiers achieve high accuracy when evaluated on data generated by the same MT systems and translators used during training in a legislative domain, but their MT recall drops sharply on unseen translators and on unseen domains such as low-resource conversational text and medical content. To address this limitation, we evaluate models on longer inputs by chunking text into segments of up to 300 tokens, which substantially improves generalization, suggesting that sentence-level signals are insufficient for robust detection. Finally, we analyze the relationship between translation quality and detectability using COMET and BLEURT, showing that lower-quality MT is more easily identified, while high-quality MT remains difficult to distinguish from HT. Our results highlight the importance of input granularity, translator generalization, and translation quality in building robust MT detection systems across domains and languages. 

---
# Read, Route, Patch: Robust Test-Time Evolution for LLM Agents 

**Authors**: Kaijie Chen, Liang Yan, Yilan Wei, Jiachen Luo, Zhihao Dou, Jiashu Qu, Haoran Liu, Junli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18585)  

**Abstract**: Large Language Model (LLM)-based agents can improve long- horizon interactive decision-making through test-time evolu- tion, yet existing methods face three information bottlenecks: memory retrieval overemphasizes textual similarity, candi- date harnesses are often behaviorally redundant, and whole- harness rewriting may overwrite useful evidence or introduce cross-component inconsistency. We introduce Trident, which improves experience utilization, behavioral exploration, and harness evolution through boundary-aware memory reading, diversity-aware candidate routing, and order-aware insertion- based updating, respectively. We evaluate Trident on five benchmarks spanning text games, web interaction, and cod- ing tasks. Across qwen-32B, mistral-small-3.2-24B, and gpt- oss-20B, Trident achieves the best average performance and consistently outperforms non-learning, memory-based, reflection-based, and automated prompt optimization base- lines. Compared with the strongest baseline, APEX, Trident yields relative average gains of 30.7%, 30.8%, and 38.3% on the three backbones, respectively. Ablation and sensitivity stud- ies further show that the three designs provide complementary gains and remain robust to parameter variations. 

---
# "I Am Not a Kiddo!": Quantifying Epistemic Infantilization in Large Language Models via Multi-Metric Bias Analysis 

**Authors**: Aditya Upadhyay, Arpan Verma, Chandan Sah, Dev Utkarsh, Harshit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.18583)  

**Abstract**: Large language models (LLMs) are increasingly used for tutoring and professional knowledge retrieval, yet safety evaluations still focus mainly on overt harms such as toxicity while overlooking benevolent failure modes. We study epistemic infantilization, defined as an identity-conditioned reduction in technical depth that appears when disability disclosure precedes an expert-level query. To measure this effect, we construct a counterfactual dataset of matched question-answer pairs and compare model responses across three settings: no identity context, disability disclosed in prior conversational context, and disability disclosed via a short query prefix. We introduce two composite metrics, the Infantilization Bias Index (IBI) and the Epistemic Accessibility Response Score (EARS), which capture retention of mathematical and formal content, domain terminology, reasoning coverage, and tone signals such as simplification markers and unsolicited offers to adjust the level. Across five open-weight model families, we find consistent evidence that disability disclosure reduces epistemic access by replacing formal explanations with higher-level summaries, with Physics and Mathematics most affected. Using question-clustered inference that respects the repeated-measures structure of the data, the effect remains reliable after accounting for correlated observations. These findings suggest an inclusion tax, where disabled users must request rigor more explicitly than non-disabled users. We conclude by arguing that accessibility should mean format adaptation rather than content simplification. 

---
# Latent Lattice Language Models 

**Authors**: Shivam Duggal, Zongze Wu, William T. Freeman, Antonio Torralba, Richard Zhang, Eli Shechtman  

**Link**: [PDF](https://arxiv.org/pdf/2611.21608)  

**Abstract**: Language models typically increase per-token computation through deeper or looped layers and latent reasoning steps, while richer residual connections improve information flow across layers. We instead increase computational capacity: the number of active computational states a model maintains. We introduce the Lattice of Latents, a set of per-layer horizontal latent streams that run recurrently along time, alongside the vertical residual stream that flows through depth. Like hyper-connections, mHC, and attention or value residuals, the Lattice enhances the residual stream, but rather than adding or rerouting residual pathways through depth, it provides companion computational states that persist across tokens. Unlike hybrid Transformer-SSM architectures, whose recurrent states primarily serve as internal memory for sequence mixing, these latents directly condition the vertical stream's attention, KV-cache writes, MLP computation, and decoding. On Transformers trained on 20B FineWeb-Edu tokens, a 406M-parameter Lattice model lowers validation perplexity from 15.41 to 14.40 over a parameter-matched Transformer. These gains extend across architectural variants. The Lattice composes with prior residual-stream enhancements: with hyper-connections and input residuals, it lowers validation perplexity from 15.41 (base Transformer) and 14.22 (Transformer with the same enhancements) to 13.61. On Gated DeltaNet, Mamba-2, Triadic GDN, and GDN-Mamba-2 hybrids, it improves perplexity at every tested native state size, with Lattice-GDN and Lattice-Triadic-GDN variants outperforming pure baselines with 4x larger native states. These results identify persistent per-layer state as a complementary axis for scaling language-model computation. 

---
# Quantifying the added value of human forecaster discussions for tropical cyclone intensity prediction 

**Authors**: Ilia Azizi, Frederick Iat-Hin Tam, Marc-Olivier Boldi, Valérie Chavez-Demoulin, Tom Beucler  

**Link**: [PDF](https://arxiv.org/pdf/2611.21602)  

**Abstract**: The National Hurricane Center routinely releases Tropical Cyclone Discussions that offer insights into its tropical cyclone forecasts. This study examines the added value of these discussions for intensity forecasting beyond a set of numerical predictors. For 637 Atlantic and eastern North Pacific tropical cyclones from the 2002-2023 hurricane seasons, we analyze the discussion text and its combination with 13 hurricane environmental predictors from SHIPS and ERA5 10-meter wind fields, for forecasting intensity changes at lead times of 24 to 120 hours using three modeling methods. In one approach, an end-to-end deep learning framework for the text shows that the discussions alone offer comparable value for 24-hour intensity forecasts to the SHIPS predictors. Adding discussion information to the SHIPS predictors in an ensemble of five such models reduces the mean absolute error of 24-hour maximum sustained wind forecasts for 2022-2023 cyclones from 9.89 to 8.90 kt, with a confidence interval that excludes zero when whole storms are resampled. The ERA5 wind fields have a minor effect on mean absolute error in wind predictions, and model rankings differ across predicted variables, lead times and model types. We provide prediction intervals for these forecasts, whose coverage falls below the 90% target during rapid intensification. Analysis of the text shows that words related to intensity and intensity change (such as intensify, weaken, eyewall and category) are important for the model predictions. This retrospective study shows the added value of the forecasters' discussions and suggests that multi-modal learning can help understand the predictive contribution of expert human judgment. The reanalysis data differ from real-time analyses, and the discussions are issued at least three hours after the valid time of the other inputs, so these results should not be interpreted operationally. 

---
# How VLMs Plan What's Next 

**Authors**: Rohit Gandikota, Arnab Sen Sharma, Adam Belfki, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2611.21600)  

**Abstract**: When a vision-language model describes an image, it names one object at a time and it rarely repeats itself, yet how the model plans this order is far from obvious. We characterize the mechanism a VLM uses to decide where to look before it looks. The mechanism has four main parts. First, binding heads match the relevant text tokens against the image and resolve them into pointers to the matched image region. Second, the same binding heads bind each written answer to the region it came from, and the model turns it into a progress signal: a cursor, the position in the image up to which the list has been written. Third, a select step chooses the nearest unwritten region after the cursor, or stops. Finally, gaze heads execute the chosen pointer and return a concept of what is drawn there, which language heads and the late MLPs turn into a word in the language of the question. We validate each part of this mechanism through causal patching in Qwen3-VL models on a synthetic visual retrieval task, and test our mechanisms against alternative hypotheses. We further show that the mechanism holds across input formats, natural photographs, and languages, and that its binding, relay, and gaze heads appear at every model size from 2B to 32B. This work allows us to investigate how a VLM turns a question and an image into a sequence of answers, layer by layer and token by token. More broadly, it opens a way to study how models plan over complex multi-step tasks with long rollouts. Our project is available at this https URL 

---
# Reading or Remembering? Measuring Temporal Contamination in Language Models 

**Authors**: Arunkumar V, Gangadharan G. R., S. Senthilkumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21554)  

**Abstract**: Language models are now used to read central-bank text for economic forecasting, but most were trained on text written after the documents they read. A model reading a 2021 FOMC statement may already know that inflation rose in 2022, so part of its apparent skill can be memory of the outcome, a temporal form of test-set contamination. We measure this memory on FOMC statements and minutes from 2000 to 2026 by comparing modern open models with chronologically consistent models, which are trained only on text from before a fixed date. Asked about past GDP outcomes given only a date, models that cannot know the answer still reach an AUC of up to 0.64 from priors on dates and eras, so chance is the wrong baseline. Against these memory-free models on the same questions, most modern models recall which years had strong growth. Membership-inference scores sort documents by era and document type rather than by whether a model saw them, as exact counts in an open pretraining corpus show. In real-time nowcasts, every lookahead premium we estimate on US GDP is smaller than the forecast comparison can detect at macroeconomic sample sizes, and none on GDP or inflation survives correction for multiple testing. Lookahead audits should therefore measure memory directly against memory-free models on the same items. 

---
# Right Reason, Wrong Advice: Evaluating Recourse in LLM Explanations of Denied Requests 

**Authors**: Arunkumar V  

**Link**: [PDF](https://arxiv.org/pdf/2611.21545)  

**Abstract**: Language models increasingly write the explanations people receive when a rule system denies their request. If several rules deny the request independently, an explanation can name a true reason and still leave the person denied after they fix it. We call the missing property recourse validity: changing what the explanation cites, without changing what was asked for, can flip the decision. We compute it exactly, and a hitting-set argument predicts from the policy alone when a correct reason fails. In six of seven rule systems ported from access control and US benefit eligibility, a single correct reason almost never gives recourse when several reasons apply. Across 20 LLMs, checks of whether the cited reasons are correct, computed or asked of an LLM judge, are close to chance at spotting this failure, as the argument predicts. A strong judge asked whether acting on the explanation would work does spot it. Stronger and reasoning models do better but still fail on many denials with four or more independent reasons. Asking for every rule helps every open model with room to improve. Explanations of eligibility decisions should be built from the decision function and judged by whether acting on them works. 

---
# Seen but Unsaid: Locating and Repairing Cross-Image Comparison in Vision-Language Models 

**Authors**: Zhaoning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21539)  

**Abstract**: Vision-language models (VLMs) often fail to match a point or an object across two images, even when the answer can be decoded from their own hidden states. We trace this failure from the visual tokens to the answer letter, in Qwen3-VL (2B-32B), InternVL3.5-8B, and Gemma-3-12B, on synthetic shape matching, same-instance point correspondence (HPatches), and semantic correspondence (SPair-71k), and find two distinct failure points, a small set of attention heads that carries the comparison, and a repair that touches 0.02% of the parameters. Two failure points: exact copies are compared but not read out. An answer-position probe exceeds the calibrated output by 15 to 25 points in every family, for nameable and unnameable shapes alike, and steering along the probe's direction moves no more answers than random directions. Semantic correspondences are present in the visual tokens but never reach the answer. One pathway: redistributing the cross-image attention of 16-32 heads, with its mass preserved, removes 17-22 points of synthetic and 15-32 points of HPatches accuracy, against at most 7 for random or attention-matched heads. A lever that training finds on its own: training only these heads' query and key projections raises SPair-71k accuracy by 28.3 points (Qwen3-VL-8B) and 25.1 points (InternVL3.5-8B), beats both layer-matched and attention-matched heads, and transfers to HPatches and BLINK. An unconstrained all-layer adapter routes 21-51% of its gain through the attention of the same heads, against at most 7% for random heads, and does so by changing what they read (83-86% of the effect) rather than their weights. Every decision rule was frozen before its run, and every reported number is recomputed by independent verification scripts. 

---
# Optimistic Verifiers Get Hacked: Reward Hacking Under an Exact Process Oracle 

**Authors**: Arunkumar V, Manoranjan Gandhudi, Gangadharan G. R., Arun Prakash, S. Senthilkumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21538)  

**Abstract**: Reinforcement learning from verifiable rewards inherits whatever its verifier gets wrong. Outside mathematics and code the verifier is a proxy, a rubric, a learned judge or a hand-written checker, and its distance from the true objective is never measured, so a rising reward cannot be told apart from a hack. We build a setting where the true reward is exactly computable at the process level: executable authorization policies over 124,416 contexts admit an oracle that lists every minimal sufficient reason for every decision. We compile these policies into exact process rewards, degrade the verifier along seven realistic axes (outcome-only, sufficiency, recall-only, extraction noise, coarse taxonomy, stale policy, LLM judge), and train against each while the exact verifier scores every rollout. Four phenotypes appear. Optimistic verifiers are hacked by padding: the sufficiency proxy ends at 0.940 while true faithfulness falls from 0.731 to 0.667 on 96 held-out episodes. Outcome-only verifiers saturate and stop teaching, noise dilutes, and ranking-preserving gaps do no harm. A model-free audit of the verifier, its selection error and gap sign, together with a short saturation pilot, anticipates which phenotype a run will produce. Across nine configurations spanning two frameworks, two model vendors, two adapter sizes, three precisions, two initialisations and two seeds, the phenotypes recur, sufficiency damage is positive in all 10 replication cells and beyond two standard errors in 7, and optimistic gaps remain the steeper branch. The damage magnitude does not replicate, and five pre-registered attempts to explain it are reported as they fell. We release the testbed, the training instrumentation, and 14 real public policies (Cedar, OPA, Gatekeeper) ported to executable oracles. 

---
# Procedure without Judgment: What Small Language Models Learn under an Exact Verifier 

**Authors**: Arunkumar V, Manoranjan Gandhudi, Gangadharan G. R., Arun Prakash, S. Senthilkumar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21529)  

**Abstract**: When an authorization policy blocks an agent's tool call, the audit log should name the facts that forced the decision and nothing more. For an executable policy over an enumerable context space this minimal sufficient reason can be computed exactly, so an explanation is scored against ground truth rather than a proxy for it. We use that exactness to ask what a model can learn from a verifier that has no error in it. The explainer sees one request and one decision, never the policy, and has thirty turns in which it may call the policy on altered requests and ask whether a candidate set of facts is sufficient. Frontier models find the exact reason about half the time with the tools alone, and one sentence telling them to shrink the answer lifts Haiku 4.5 from 58% to 75%. A 4B model fine-tuned on 2000 traces of the deletion algorithm reaches 92% on the policy it was trained on. Teacher kind is what moves this: at matched data size the algorithm-taught student leads the frontier-taught one in all three training seeds, and the two arms never overlap. What the student learns is a fixed schedule of sufficiency checks, and the schedule does not travel. On three public policies it never saw, its answers stay sufficient (94%) and stop being minimal (48%), while Haiku under the same sentence is exact on all 47 episodes. Reinforcement learning from the exact reward leaves held-out accuracy where it was and inverts the strategy, from probing the policy to spending the entire budget on sufficiency checks. The oracle made the effect of training legible, but it did not make training better. 

---
# Detection-Efficient Watermarking for Vision--Language Agents via Risk-Constrained Strength Allocation 

**Authors**: Dumindu Samaraweera, Prashant Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2611.21488)  

**Abstract**: Vision--language agents generate descriptions, decisions, and instructions that may be reused by users, downstream agents, and automated workflows, creating a need for provenance mechanisms that support content authenticity and post-deployment auditing without unnecessarily degrading response quality. We present a risk-aware semantic watermarking framework that combines predictive uncertainty with a zero-shot multimodal relevance-risk proxy to allocate watermark strength during decoding. A single risk penalty $\lambda$ exposes a continuous operating frontier, allowing deployment-time selection of a detection--quality trade-off without model retraining. At a selected operating point, mean watermark strength is reduced by approximately 59.4\% relative to fixed-strength watermarking, while perplexity change falls from 0.3634 to 0.1689 and clean/paraphrased ROC-AUC reaches 0.9552/0.9459. A matched-strength analysis shows that this advantage reflects the allocation policy rather than lower average intervention: at comparable mean strength, risk-constrained allocation achieves substantially higher ROC-AUC than fixed, entropy-only, and criticality-only policies, identifying it as detection-efficient rather than quality-dominant. We further show that threshold calibration on finite samples produces substantial false-positive-rate drift, motivating calibration variance reporting as standard practice for auditable provenance. 

---
# Balanced Twin Weight Factorization for Efficient MLLMs 

**Authors**: Bryan A. Plummer, Shan Lu, Keval Doshi, Noranart Vesdapunt, Wei Wang, Jim Thomas, Kah Kuen Fu  

**Link**: [PDF](https://arxiv.org/pdf/2611.21459)  

**Abstract**: Weight Factorization (WF) methods aim to reduce a model's latency, memory, or both. These typically decompose a pretrained model into two or more factors that approximate the original model weights with fewer parameters and floating point operations (FLOPs). However, translating fewer FLOPs into reduced latency is challenging for WF methods in practice due to overhead introduced by replacing a single large weight matrix with multiple smaller ones. To address this, we propose Balanced Twin Weight Factorization (BTWF), which uses the unique structure of factorized models to simultaneously lower latency and increase learning capacity. Specifically, BTWF operates in the shared basis WF setting, where two or more layers are decomposed into a shared basis matrix and a layer-specific mixing matrix. If these shared layers have the same input (i.e., are twins), then we need only perform this shared forward pass once, reducing inference time. However, twin layers are often of varying sizes, and methods from prior work tend to favor the larger layer. In contrast, our approach boosts performance by combining Factor Component Balancing (FCB), which ensures every shared member is well represented in the selected factors, with Block Contribution Balancing (BCB), which considers downstream model effects. Further, we also show that an overparameterized BTWF, where some layer's factors result in higher FLOPs than the original during training, can boost task performance without additional resources at test time. Experiments on six models over nine benchmarks show BTWF obtains a 2-6% gain over the state-of-the-art without expensive finetuning (1-3.5% gain with it) at comparable computational costs. 

---
# Just sort it! A note on the LP behind rank-one anti-loop edits 

**Authors**: Matteo Fischetti  

**Link**: [PDF](https://arxiv.org/pdf/2611.21432)  

**Abstract**: Weight-space programming removes the thinking loops of a small reasoning model by a rank-one edit W' = W + u v^T of an MLP output matrix, where u is a loop direction and v is the solution of a sparse Linear Program (LP) over the neuron activations. We investigate whether the LP is needed. We replace it by a sort: we take the 8 neurons whose activation has the smallest coefficient of variation, and give them equal shares of the push that the LP solution realizes. On Qwen3.5-0.8B the sorted edit scores 67.0% on 1000 problems (MATH-500 and GSM8K) against 67.4% for the LP edit and 9.5% for the unedited model; the paired difference is -0.4 points, with 95% confidence interval [-2.9, +2.1], inside a non-inferiority margin of 5 points declared before the runs. On three further sampling seeds the difference is -0.6 points, interval [-2.0, +0.8], and -0.8 points, interval [-2.3, +0.6], when the dose is set without the LP. The sort does not solve the LP, but the neurons it shares with the LP solution carry most of its mean push. The outcome is that, on this model, the edit acts as a nearly constant bias along u. 

---
# AuXSpec: Learning to Autoformalize Program Specifications via Executable Postconditions 

**Authors**: Haocheng Wang, Weixuan Yuan, Florian Hübler, Yinya Huang, Zhijiang Guo, Sorrachai Yingchareonthawornchai, Rasmus Kyng  

**Link**: [PDF](https://arxiv.org/pdf/2611.21421)  

**Abstract**: Formal specifications define intended program behavior and underpin verification, but writing them in theorem provers such as Lean is slow and expertise-intensive, making specification a major bottleneck to scaling formally correct code. Autoformalization from natural-language problems is promising, yet training specification models requires scalable feedback on whether generated specifications capture intended behavior. Prior approaches obtain this feedback through explicit proofs or separately implemented code with formal correspondence proofs, both expensive. We introduce AuXSpec, a Lean framework that pairs declarative postconditions with executable counterparts, obtained through type class resolution when possible and otherwise defined separately with an equivalence proof. Test cases can then evaluate declarative specifications by execution, letting us use agreement as a reinforcement-learning reward. Using AuXSpec, we synthesize 3,037 LeetCode problems and expand training to 19.5K samples through expert iteration on Codeforces. Experiments show that training Qwen3.5-9B and Qwen3.8-27B with the execution-based reward improves SpecGen pass@4 on AuXSpec-Test 0.7% $\to$ 82.7% and 14.3% $\to$ 84.3%, and the 27B model outperforms all frontier models. On the out-of-domain Verina benchmark with its original prompts, both models improve 7.0% $\to$ 90.5% and 21.9% $\to$ 90.5% and surpass VeriSpecGen-30B-A3B, the prior state of the art among open-source specification models, on pass@1 (theirs 47.6% vs. ours 73.7% and 84.5%) and pass@4 (theirs 63.9% at pass@10 vs. ours 90.5%). We release the weights of the trained 9B and 27B models and the 3,037 problems with test cases, including the 300-problem AuXSpec-Test benchmark with expert review and correctness proofs for future evaluation. Models and data: this https URL. Website: this https URL 

---
# Do Embedding-Map Distances Guide LLM Ideation? Four Tests Against What Came Next 

**Authors**: Javier Aguilar Martín  

**Link**: [PDF](https://arxiv.org/pdf/2611.21408)  

**Abstract**: A popular ideation recipe treats an embedding map of existing work as a guide: pick points far from everything and decode them into ideas. I test four ways such distances could guide ideation against what appeared later. (1) Far random directions are not text: no later item reaches a raw cosine above about 0.10 with one (after centring or whitening, 22 of 24 cells stay out of reach), and an LLM decoder shown a target's neighbours, with score-only feedback, writes those neighbours in all six target sets. A neighbour-free search writes coherent text at median cosine 0.57 and 0.47 to interior gaps, beating the nearest existing item on 15 and 14 of 20, but only 0.07 and 0.06 to far points. (2) On k-means cells of a past-only map, the registered hole score mostly tracks cell size and position; three density-based gap scores forecast which cells grow fastest at or below chance (AUC 0.30 to 0.66) and add nothing to momentum in Steam and arXiv, with patents inconclusive; more isolated new Steam games are slightly better reviewed. (3) LLM continuations of arXiv papers stay closer to the source than papers that cite it, under three user prompts and against proposal rewrites (gpt-4o) and April-December 2023 papers (OLMo). Critique lengthens this step beyond content-free revision, replicated on 150 new papers. Revealing a real follow-up's title turns the ideas by 0.032 along a follow-up-specific direction beyond revision; reviewer, oracle and distance feedback add 0.13, 0.08 and 0.34 of that turn (upper bounds 0.46, 0.46, 0.70), and their turn is not separable from revision or their longer step. (4) For inspiration material graded by distance, no inverted U was detected, on an anticipation margin that tracks format more than content. Original tests were registered before running, follow-ups (thirteen requested in two AI-agent reviews) after the main results; all are dated only by local git commits. 

---
# Beyond Bilingual: Studying Quadrilingual Code-Switching with Conversational AI 

**Authors**: Annika Bush, Melika Minnaar, Salys Sultan, Tasnimul Hasan Tashin, Ankur Bhatt, Meltem Aksoy  

**Link**: [PDF](https://arxiv.org/pdf/2611.21394)  

**Abstract**: Multilingual speakers choose languages by topic and conversation partner. Large language models (LLMs) support these languages unequally. Research on code-switching with conversational AI has mostly examined bilingual pairs of well-resourced languages. In Curaçao and Aruba, many people use Dutch, English, Spanish, and the low-resource Creole Papiamentu/o. In a preregistered study, 42 university students held two five-minute chatbot conversations, one about university life and one about local food and culture. Surveys assessed language background, AI use, and switching comfort. 20 international students in Curaçao formed a comparison sample. Participants switched between turns and rarely mixed languages within a message. English dominated both conversations. Participants switched more often in the cultural conversation, where Papiamentu/o and Spanish rose from 15% to 24% of messages. Self-reports showed no lower comfort switching with the chatbot than with a person. Conversational agents must follow turn-by-turn language changes and handle local low-resource languages well. 

---
# A Crowd-Audited Atlas of Implicit Bias in LLM-Powered Agents 

**Authors**: Jason Alexander, Yuxuan Li, Michael Licamele, Wesley Hanwen Deng, Sauvik Das  

**Link**: [PDF](https://arxiv.org/pdf/2611.21376)  

**Abstract**: Prior work has shown that large language models (LLMs) express implicit biases in high-publicity, one-off decision scenarios such as hiring and intelligence assessment. However, as people increasingly employ autonomous, LLM-powered AI agents that make decisions on their behalf in doing day-to-day tasks, it is of critical importance that we understand how implicit biases, based on sociodemographic assumptions, emerge across a broad milieu of workaday decision scenarios. We used the 17 domains of daily activity in the American Time Use Survey (ATUS) to generate an atlas of 506 decision scenarios in which LLMs --- endowed with personas representing different races, gender identities, and political ideologies --- expressed significant implicit bias. Next, in a crowdsourced audit, 500 participants made 8,010 binary choices about which implicitly biased decision scenarios most deserved further review. We found that participants prioritized decision scenarios that had broad relevance (e.g., about food choices), were high stakes (e.g., about health), and where all members of a simulated sociodemographic group made unanimous decisions (because it was a clear indication of stereotyping). They were less swayed by the magnitude of an observed disparity or the specific ATUS domain the decision scenario represented. When examining the rationale behind their prioritization choices, we found that participants often interpreted disparities as traits of demographic groups and judged whether those traits seemed plausible, often based on their own internalized stereotypes. We discuss implications for building personalized AI agents, digital twins, social simulations, and for using crowdsourced bias audits. 

---
# AnachroBench: Evaluating Historical Consistency in Large Vision-Language Models 

**Authors**: Keunhyeung Park, Byeonggeuk Lim, Kahyeon Kim, Sunhee Jo, Hayeong Ryu, Hoejoon Kwon, Jinhee Jang, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.21292)  

**Abstract**: Existing Large Vision-Language Model benchmarks primarily evaluate visual understanding in general everyday scenarios, leaving historical consistency with a specified era and country largely underexplored. To address this gap, we propose AnachroBench, a controlled discriminative benchmark for evaluating historical consistency in visual scenes. It contains 1,600 image pairs from eight countries spanning the late 19th to the early 20th century, each pairing an original historical image with an anachronistic version containing a seamlessly inserted modern object. AnachroBench evaluates three aspects: Inserted Object Perception, which tests whether models detect the inserted modern object; Era-Country Consistency, which assesses whether the scene is consistent with the specified era and country; and Conditional Era-Country Consistency, which measures consistency judgments only when the inserted object is correctly perceived. A false consistency judgment occurs when a model perceives the modern object but still judges the scene as historically consistent. Overall, AnachroBench reveals a gap between inserted object perception and era-country consistency judgments, providing a targeted evaluation of this behavior. 

---
# Classical structural priors improve neural molecular representations in molecule language models 

**Authors**: Donghyeok Lee, Sangkwon Park, Munyoung Chang, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.21276)  

**Abstract**: Classical molecular similarity is central to chemical information systems, yet molecule-language models trained on paired structure-text data do not necessarily organize molecular embeddings according to graded structural relationships. Here we introduce Structure-Aware Guided Alignment (SAGA), a training-only regularizer that uses Centered Kernel Alignment to align neural molecular representations with fingerprint-based Tanimoto similarity, without adding parameters or requiring fingerprints at inference. In MolCA and Atomas, SAGA improves molecule-text retrieval, shifts non-exact retrieval toward closer structural analogues, and increases average scaffold-split property-prediction performance, while molecular description and text-to-molecule generation are improved or preserved in single-run ablations. These results show that established chemoinformatics priors can make molecule-language representations more chemically faithful while retaining their multimodal utility. 

---
# The Price of Choosing Examples: Context Overfitting in Adaptive In-Context Learning 

**Authors**: Xiaoyu Li, Jiangxuan Long  

**Link**: [PDF](https://arxiv.org/pdf/2611.21251)  

**Abstract**: In-context learning is usually analyzed as if the examples in the prompt were sampled before the learner is chosen. In deployed systems they are not: examples are retrieved, filtered, re-ranked, edited, and repeatedly tested after seeing the query and the data store. We formalize this gap by treating prompt construction as adaptive data analysis through a conditional stochastic-kernel calculus. Our main quantity is the context capacity, the mutual information or approximate max-information between the data store and the final prompt transcript. For any frozen in-context learner and any adaptive context pipeline with capacity $\kappa$, we prove a finite-sample context generalization bound of order $\sqrt{\kappa/k}$, where $k$ is the number of selected examples whose losses form the reported score. We prove a matching lower bound, whose constant is exact when $\log k\ll\kappa\ll k$, and derive two PAC-Bayesian retrieval rules: a first-order Gibbs rule that charges data-dependent candidate gates their prior mass, and a self-normalized second-order rule. On Banking77 with 64 retrieval recipes and two model families, the certificates hold in all 24,000 resampled selection events, and from 64 (Qwen2.5-7B) or 128 (GPT-4.1-mini) reused validation queries the Gibbs rule certifies a risk below the zero-shot error in over 99% of validation draws. Finally, we instantiate the theory for Bayesian linear in-context regression, including an order-statistics analysis of query-aligned top-$k$ Gaussian retrieval. The results separate context quality from context certification: longer or more relevant contexts improve risk only when the procedure used to choose them is stable enough for selected-context evidence to transfer. 

---
# Stream of Belief: Eliciting Collective Intelligence through Latent Belief Dynamics 

**Authors**: Yu Yao, Jiayi Dong, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2611.21250)  

**Abstract**: In this work, we view multi-agent reasoning as a process of seeking a collective equilibrium of implicit belief dynamics and introduce Stream of Belief (SoB), where agents maintain continuous latent beliefs and iteratively refine them through communication and reflection. Our method seeks a collective belief state of a multi-agent system where further communication induces no meaningful information propagation. Through extensive experiments, we demonstrate the generalization of SoB across tasks and model families. More Broadly, our results suggest a new perspective on scaling multi-agent reasoning from belief dynamics. 

---
# Anytime-Valid PAC-Bayes Certificates for Adaptive Test-Time Scaling 

**Authors**: Xiaoyu Li, Jiangxuan Long  

**Link**: [PDF](https://arxiv.org/pdf/2611.21241)  

**Abstract**: Test-time scaling systems often choose how much to sample, which prompt to use, and which verifier or aggregation rule to trust after repeatedly inspecting a calibration set. Standard validation error bars are not valid after this adaptive selection. We model a test-time scaling recipe as a randomized predictor with a random cost and prove a time-uniform PAC-Bayes certificate that holds simultaneously over all calibration times, all posterior mixtures over recipes, and all data-dependent stopping rules. Thus the same event covers the recipe search that precedes deployment, not merely the fixed recipe finally reported. For a finite grid of budgets and verifiers, the penalty is the familiar $\log M/n$, which in the worst case no certificate valid after selection can avoid, while non-uniform priors and localized posterior mixtures can give sharper certificates for structured searches. We specialize the result to self-consistency and show a complementary margin law: plurality voting improves exponentially in the sample budget only on examples where the correct answer is the generator's unique mode; otherwise more compute can provably amplify errors. The same margin analysis yields an oracle water-filling law for compute allocation. A synthetic audit quantifies the optimism caused by adaptive budget selection, and a 192-example GSM8K audit using gpt-4.1-mini generations illustrates risk-cost reporting for a real inference pipeline. At scale, with two models on GSM8K and MATH-500, selected-best error bars fall below the out-of-split GSM8K risk in up to 22.5% of random splits at a nominal 5% while the certificate never does, and on the full GSM8K test split it tightens to 0.109 (held-out risk 0.039). The results give a lightweight, distribution-free way to report risk and compute claims for adaptive inference systems. 

---
# Language-model agents' beliefs can be manipulated via truthful evidence in open channel 

**Authors**: Jinwei Hu, Qisong He, Xinmiao Huang, Youcheng Sun, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.21235)  

**Abstract**: Language-model agents' beliefs can be manipulated through open channels such as social media, even when the information they receive is truthful. When agents fill gaps between fragments with self-induced relationships, they may accept those relationships as established facts and pass the resulting rumours to others. Drawing on cognitive theory of narrative understanding and film theory of montage, we introduce Generative Montage to operationalize this failure, termed narrative overfitting, by coordinating agents to arrange truthful fragments into misleading narratives. Across 1,090 rumour targets from six Twitter events, agents often accepted rumours overconfidently, with overall acceptance exceeding 65 percent in twelve of fourteen models. Acceptance remained above 50 percent after majority voting or review of the same evidence, showing how errors can reach downstream decisions. To explain this susceptibility, fragment interventions identify posts that influence belief within complete narratives. Synthetic controls then show that similar response patterns arise from valid evidence, separating belief influence from evidential support. This distinction motivates Grounded Monitor, a proof-of-concept neuro-symbolic runtime monitor that checks direct evidential support to guide agents' judgements, improving decision accuracy by 5.6 to 29.0 percentage points across original and anonymized settings. These findings expose a cognitive route to rumour amplification and support monitoring inferred relationships before agent-generated conclusions guide public decisions. 

---
# TRACK: Tool-Augmented Reasoning over Time Series for Fault Diagnosis in Power Management System 

**Authors**: Erkan Bayram, Debanjan Ghosh, Li Zhu, Sambarta Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.21178)  

**Abstract**: Fault diagnosis in power management systems running multiple loads relies on experts manually inspecting current and voltage waveforms, which fails to scale across deployed units. We present TRACK, which pairs a time series encoder with a Small Language Model to produce structured, physics-grounded diagnostic reports, calling tools for exact, verifiable calculations. It operates on-device, keeping proprietary sensor data local and avoiding expensive API calls. We introduce a rubric-based framework that evaluates these reports by scoring structure, signal fidelity, physics reasoning, diagnosis, and actionability through per-criterion verdicts. On a 15-class multi-load simulated benchmark, TRACK achieves over 95% fault-category accuracy and generalizes to out-of-distribution faults, whereas the best encoder-probing baseline reaches only 58% accuracy. 

---
# Jev-as-a-Reward: Validating Probabilistic Safety Judges for Reinforcement Learning 

**Authors**: Eunkyeong Lee, Junhyung Moon, Seunggwan Hong, Dongsuk Oh  

**Link**: [PDF](https://arxiv.org/pdf/2611.20955)  

**Abstract**: Probabilistic safety judges return the probability of a safety label and cheaply score reinforcement learning rollouts. A confidence gate sends only uncertain verdicts to a stronger LLM judge, which leaves confidently passed harm in a blind spot. Policies can exploit this blind spot, and commercial judges release new versions. We propose a protocol that validates such a judge before and during training, separates the share of confidently passed harm from its amount, and re-validates it after each judge change. We apply it to two versions of Jev, a commercial decision-only judge, and to an open guard classifier. Under the probability reward of Jev, passed harm shrinks in all 11 runs under the primary reference, mostly more slowly than caught harm, and concentrates in the blind spot. Binarizing this judge meets the escape criterion on Llama, and the open guard classifier shows an escape signal in one seed under a second reference. Guard classifiers share this blind spot, and even frontier LLM judges catch at most one of 10 human-labeled confident misses. Without a trained reward model, the probability reward with a helpfulness term lowers HarmBench attack success on Llama-3.1-8B from 0.287 to at most 0.033 at the final checkpoint. 

---
# From Recursion to Convolution: Parallel Self-Improvement for Multi-Agent Systems 

**Authors**: Zheng Wang, Yuang Liu, Sachith Abeywickrama, Yangkai Ding, Tao Yu, Xinyu Zhang, Hongbo Wang, Qi Ye, Xuefeng Jin, Zhangchun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.20891)  

**Abstract**: Multi-agent self-improvement enables agents to improve through interaction and feedback. However, sequential dependencies across agents and refinement rounds can propagate errors, increase computational costs, and limit parallelism. Moreover, refining individual features or agents in isolation can overlook their dependencies and restrict collective coordination. To address these challenges, we propose ConvRSI, a framework for parallelizing multi-agent self-improvement through convolution. ConvRSI couples intra-agent convolution, which jointly refines related agent features, with inter-agent graph convolution, which aggregates messages according to the collaboration topology. Both stages operate in latent space, enabling parallel updates without repeatedly decoding and re-encoding intermediate messages. Experiments on MATH-500 show that ConvRSI achieves 5.04% performance improvement while reducing 17.27% inference cost, and 45.11% latency compared with RecursiveMAS. It also supports diverse collaboration patterns by representing their topologies as adjacency matrices. These results demonstrate the potential of convolutional co-evolution for efficient multi-agent self-improvement. Our code and datasets are provided in this https URL. 

---
# MolReps: Taking SELFIES, Making SMILES, Teaching LLMs 

**Authors**: Ken Deng, Garrett Morris  

**Link**: [PDF](https://arxiv.org/pdf/2611.20848)  

**Abstract**: The same molecule can be expressed in multiple, syntactically distinct representations, yet it remains unclear whether large language models (LLMs) can recognize and relate these equivalent encodings. We introduce MolReps, a pipeline that generates 9,600 question-answer pairs across ten representations for conversion, normalization, and 2D structure recognition. Post-training on MolReps improves text-only and multimodal models, MolQwen3-4B-Instruct (MQ3-4B-I) and MolQwen3-VL-4B-Instruct (MQ3-VL-4B-I), from $4.01\%$ to $75.13\%$ and from $7.11\%$ to $73.14\%$ (InChIKey match) on the molecule-overlap test set, respectively. On the molecule-disjoint test set, post-training increased validity from $20.35\%$ to $58.12\%$ for MQ3-4B-I and from $17.19\%$ to $59.34\%$ for MQ3-VL-4B-I. However, InChIKey match remained low, decreasing from $3.86\%$ to $3.09\%$ and from $6.44\%$ to $2.29\%$, respectively. Furthermore, the resulting models achieve a relative $27.84\%$ improvement in average over their base model on MolRepBench, a comprehensive benchmark for evaluating molecular property prediction. These results show the effectiveness of targeted post-training for enhancing molecular representation understanding and its potential to improve downstream molecular tasks. 

---
# DetourBench: Trajectory-Level Evaluation of Interactive Agents with Exact Minimum-Action Bounds 

**Authors**: Hanlin Tian, Sihan Zhu, Zhao Yang, Yuxiang Wang, Zhiquan Hu, Yu Mi, Hongquan Zhu, Qiufei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20844)  

**Abstract**: Benchmarks judge language-model agents mainly by whether they reach the correct final state, not by how they got there. Endpoint scoring cannot separate an agent that retrieved exactly the evidence a decision required from one that read most of the environment, nor an agent that never located the evidence from one that stopped a single action short. Settling this needs a reference cost, and authored solutions establish only that a task is solvable, not that no shorter route exists.
We present DetourBench, which constrains the environment until that minimum is computable. Each task is a finite, partially observable transition system with unit-cost accesses whose resources become globally addressable after discovery. Exhaustive search yields the full-information minimum action count $L_0$ with a replayable witness; from graph-recoverable terminal states it also yields exact residual work $R_T$, which separates an episode that stopped one access short from one that wandered, while an irreversible wrong commitment is recorded as having no finite residual distance rather than assigned one. Success, attainment, residual progress, and normalized detour are reported separately. Across 375 scored episodes --- five model endpoints, 25 integrated tasks, three repeats --- success ranges from 58.7% to 96.0% while conditional attainment ranges from 0.803 to 0.890 and orders the models differently. Among recoverable failures, residual work separates near-complete trajectories from costly incompletion and from irreversible commitment. We open-source the instances, oracle witnesses, and evaluation harness at this https URL. 

---
# When the Bill Falls: Reusing the Old Index Defers,Never Avoids, Re-encoding in Complete Embedding-Model Migration 

**Authors**: Nannan Zhao, Jiameng Zhang, Taoyu Zhong, Hao Wang, Bo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.20702)  

**Abstract**: Upgrading the embedding model behind a vector index requires re-encoding the corpus. Recent work reuses the old index through adapters or window re-ranking. Across sixteen model pairs on passage and code corpora, plus eleven on Natural Questions (NQ), we find three results. First, reuse never avoids re-encoding: if migration must complete, every strategy eventually encodes every document; reuse only changes when that work is paid and the quality served meanwhile. Second, adapters should be compared with the old model, not only the new one. Of five adapter variants, only same-family LLM bridges beat the old model, and only by a negligible margin. Third, window re-ranking cost can be priced from the old index alone; pre-registered replays on all 8,841,823 passages confirmed the prediction on all three tested pairs. Workload skew also matters: across 36 further pre-registered Zipf replays, the planned family matched on 32. These findings yield three pre-migration rules: whether reuse is worthwhile, which strategy to use, and how deep a window to take. The depth rule priced every window that reached target quality to within 4 parts in 10^5 of the corpus. On NQ, a pre-committed plan selected the family on 10 of 11 pairs and priced every window exactly. Quality inputs transferred less reliably across corpus scale, so the model gap and window quality loss must be measured on the serving index. 

---
# RAMP: Rethinking Multimodal Evaluation through Answer Readout Protocols 

**Authors**: Jianye Zeng, Qiao Yikang, Duan Huang, Ziying Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.20653)  

**Abstract**: Multimodal large language models (MLLMs) are often evaluated by converting model outputs into discrete answers. The conversion protocol determines which scores become predictions and can strongly influence the reported results. We introduce RAMP (Readout Aware Multimodal Protocol), a systematic framework that compares direct yes/no token logits, surface-form scores, and typed or reversed A/B interfaces under fixed model checkpoints and paired visual questions. On 500 POPE adversarial questions, typed versus bare readout changes accuracy by $-20.2$ percentage points for SmolVLM-256M, $-16.4$ for TinyLLaVA-1.1B, and $+1.8$ for Qwen2.5-VL-3B. Paired transitions reveal extensive prediction changes even when the net accuracy difference is small: 79 of 104 binary ChartQA items change correctness for SmolVLM-500M, despite a $+1.0$-point net difference. We further analyze prompt wording, candidate forms, option position, cross-task effects, and calibration. An independent offline replay reproduces 16,140 first-token decisions from saved raw scores with exact agreement. Our findings establish answer readout as a measurable component of multimodal evaluation and provide a reproducible protocol for analyzing model comparisons. 

---
# Necessary for What? Auditing Demographic Disclosure to LLM Synthetic Respondents 

**Authors**: Aspen Wang, Yen Yu Shan, Lun-Wei Ku  

**Link**: [PDF](https://arxiv.org/pdf/2611.20606)  

**Abstract**: Large language models (LLMs) increasingly answer surveys on behalf of real panelists, and each panelist's recorded demographic attributes (e.g., age, education, ethnicity) are often placed in the prompt on the premise that richer profiles make more faithful synthetic respondents. While data minimization limits personal data to what a purpose requires, a synthetic survey serves several purposes at once, from predicting individual answers to estimating how opinions are distributed and comparing groups. We ask which of these purposes require the respondent's record, and whether improved estimates support sound conclusions about groups. We vary whose profile a model receives, where it comes from, and how many fields it contains, in two U.S. surveys with eleven open-weight and three hosted models, and compare each model with a statistical model fit to the same information. We make four contributions. First, we show that the record adds little to individual predictions and that, for the open-weight models, most of its gain in population estimates remains when it is replaced by census data or a synthetic persona. Second, we find that group comparisons improve only with the respondent's own record, and that even then the models' group differences rest more on the demographic label in the prompt than on the individual respondent, which the usual check of reproducing known group differences does not reveal. Third, we find that the models make groups look more alike than they are, keeping differences in circumstances while flattening or reversing differences in attitudes, most sharply for religious affiliation and ethnic origin, both special categories of personal data. Fourth, we outline an audit that a survey organization can run on questions it has already fielded to decide which estimates may use the record. 

---
# ReconPotBench: Evaluating LLM Honeypots in Multi-Round Linux Reconnaissance 

**Authors**: Qianli Wang, Junchao Wu, Binkai Ou, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2611.20528)  

**Abstract**: Large language models (LLMs) are increasingly used to build interactive honeypots, yet existing evaluations mainly assess isolated command outputs and cannot determine whether a system remains believable when later commands depend on earlier results. We present ReconPotBench, a multi-round benchmark for Linux reconnaissance honeypots. It contains 1{,}032 three-round reconnaissance interactions across eight reconnaissance categories, with reference outputs collected from a real Linux environment. Under a unified testing protocol, we compare Cowrie, open-source LLMs, and closed-source LLMs along three dimensions: round-level authenticity, cross-round contextual consistency, and hacker-oriented reconnaissance value. Results show that multi-round evaluation exposes failures hidden by single-round testing, as authenticity declines with interaction depth and both open-source and closed-source LLMs struggle to maintain coherent state across rounds. LLM-based honeypots nevertheless outperform Cowrie in authenticity and reconnaissance value, while Cowrie remains more consistent because of its rigid templated behavior. Category-level analysis further shows that tasks requiring systems to preserve network relationships across rounds are harder than tasks centered on explicit entities such as ports. ReconPotBench enables reproducible evaluation of LLM honeypots across authenticity, entity consistency, and hacker-perspective utility in dependent multi-round interactions. The dataset is available at: this https URL 

---
# GaugeRoPE: RoPE-Compatible Executable Latent Attention 

**Authors**: Yuanhang Gao, Xiangrui Yang, Lingyuan Meng, Yuanfeng Chen, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.20524)  

**Abstract**: Latent KV cache compression reduces memory footprint by mapping historical keys and values into compact representations. However, generic compressed key states require reconstruction to native dimension before applying RoPE and computing attention scores, adding sequence-scaling overhead to the decode path. We characterize when a latent KV representation can serve directly as the execution state of pretrained RoPE. For RoPE-based transformers, exact static linear execution of the rotary key path requires the latent-to-native map to preserve the pretrained RoPE action through an intertwining relation, restricting exact mixing to coordinates that share the same RoPE frequency. Guided by this characterization, we build GaugeRoPE, a training-free executable latent KV representation. It combines a RoPE-compatible carrier that preserves the rotary action with an approximate position-independent residual key--value (RKV) correction for complementary key and value information, both evaluated without per-token dense historical K/V reconstruction. On Mistral-7B, GaugeRoPE achieves $5.85\times$ KV compression while maintaining quality close to the full-cache baseline. Additional MHA results validate the construction beyond GQA. GaugeRoPE achieves up to $2.48\times$ the decode throughput of the vanilla uncompressed baseline through direct latent execution. 

---
# TrainExplorer: Autonomous LLM Post-Training via Plan-and-Decouple Exploration 

**Authors**: Xuannan Liu, Tian Lan, Zekun Li, Xing Cui, Zhengfeng Yu, Feihu Jiang, Qianghuai Jia, Longyue Wang, Zhao Xu, Weihua Luo, Peipei Li, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2611.20430)  

**Abstract**: LLM-based agents can automate model post-training, accelerating progress toward recursive self-improvement. Despite their promise, existing approaches remain under-explored and face two main challenges: (1) strategy lock-in that limits exploration beyond the initial strategy; and (2) coupled execution that obscures the effects of individual modifications. To address these challanges, we propose TrainExplorer, a multi-agent framework for autonomous post-training, that incorporates strategy planning and execution decoupling, to achieve broaden strategy exploration. First, strategy planning draws on scientific literature to construct a candidate search space of training strategies. Then, decoupled execution enables single-component interventions and attribution to guide strategy exploration. Extensive experiments on PostTrainBench demonstrate that TrainExplorer achieves a weighted aggregate score of 39.02, outperforming Claude Code using the same LLM by 7.32 and surpassing GPT-5.6 Sol-powered Codex, which scores 36.23. Code will be available. 

---
# When the Cheapest Feasible Router Fails: Auditing Deployment Commitments in Cost-Aware LLM Routing 

**Authors**: Minxuan Hu, Qishi Zhan, Ziheng Chen, Zhengtao Yao, Zihan Dong, Jiayu Yi, Wenxi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2611.20417)  

**Abstract**: Cost-aware LLM routing often selects the cheapest policy meeting empirical risk and coverage constraints. Yet a router that passes an audit can violate its fixed service contract on new requests, even when candidate policies are constructed independently. We study this deployment-commitment decision with disjoint construction, audit, and test samples. Across 19 RouterBench task families, the cheapest audit-feasible router deploys in 91.82% of opportunities but violates the held-out contract in 32.22% of deployments. We introduce Menu-Aware Routing Commitment (MARC), which qualifies policies using audit evidence and a serving-cost tolerance to retain the apparent winner, deploy a supported alternative, or defer. Its practical policywise variant (MARC-P) reduces observed conditional failure to 0.73% at 85.79% deployment, with higher serving cost; fixed-margin and formal risk-control procedures provide competitive alternatives. Nested audits show how added evidence changes deployable policy sets, and learned semantic routing exhibits the same commitment failure. These results establish deployment commitment as a distinct operational decision connecting router evaluation, service reliability, and inference cost. 

---
# JointQ: Joint Scale-and-Integer Quantization for Large Language Models 

**Authors**: Keiji Kimura, Yuma Ichikawa  

**Link**: [PDF](https://arxiv.org/pdf/2611.20384)  

**Abstract**: GPTQ is the de facto standard solver for layer-wise post-training quantization (PTQ) of large language models, yet the solver itself has received less attention than weight preprocessing and cross-layer objectives. We target a bottleneck of GPTQ: its greedy procedure optimizes continuous group scales separately from the assignment of discrete integers. Evaluating each integer candidate with its closed-form optimal scale has been used for per-channel quantization; JointQ extends it to group-wise quantization, where a row's scales are coupled through the Hessian, and applies it to refine the GPTQ solution. Each local-search move changes one integer and recomputes its group's optimal scale in closed form, accounting for the other groups through the full Hessian; all scales are periodically re-solved jointly. The search starts from a row-wise selection among scale-refitted candidates including the GPTQ solution, so unregularized JointQ never has a larger layer-wise calibration error than GPTQ. JointQ preserves the GPTQ output format and is thus a drop-in replacement. Across four models and four settings, JointQ improves six-task average zero-shot accuracy over vanilla GPTQ in 14 of 16 comparisons and perplexity in 12 of 16. The mean accuracy gain increases from 0.0044 in W4G128 to 0.1559 in W3 per-channel, and JointQ also improves average accuracy over GPTQ with MSE-based scale search and activation ordering in all eight per-channel comparisons, though rarely in G128. An ablation shows that JointQ's diagonal Tikhonov regularization mitigates severe W3 per-channel perplexity degradation, so the largest gains are not due to the local search alone. Overall, JointQ's primary empirical benefit is downstream accuracy preservation in challenging low-bit and large-group regimes. Code is available in the OneComp library at this https URL. 

---
# Trait Collapse in Text-to-Image Generation: Evaluating and Mitigating Personality Inconsistency in Visual Storytelling 

**Authors**: Yuanbo Tong, Haoyuan Li, Jibiao Jiang, Zirui Wang, Yuchen Li, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.20381)  

**Abstract**: In visual storytelling, a believable character must not only look consistent across scenes but also \textit{behave} in ways that reflect their personality. Current text-to-image (T2I) models preserve visual appearance effectively, yet consistently fail to maintain personality-relevant behavior---a failure we term \textit{trait collapse}. We show that conventional metrics (CLIP Score, DINO similarity) are blind to this failure, and that abstract personality descriptors and scene-level emotion labels do not produce consistent improvements over neutral prompts across models. To systematically diagnose trait collapse, we introduce \textbf{ViStoryBench-Personality}, a story-level benchmark combining an expert-validated Visual Cues Lexicon (Fleiss' $\kappa = 0.757$) with a six-dimensional metric suite covering personality alignment, cross-scene stability, and behavioral coherence. Evaluating seven T2I systems, we confirm a persistent semantic gap between abstract trait labels and visual evidence, and find that visual identity consistency does not imply personality consistency. To bridge this gap, we propose \textbf{Trait-Behavior-Visual (TBV)}, a training-free prompting framework that decomposes personality traits into renderable visual constraints. TBV improves Personality Alignment Score by up to $+22.1\%$, with larger gains in models with stronger language-conditioning capacity. 

---
# Estimating Outpatient Visit Prices from Hospital Price Transparency Data using Large Language Models 

**Authors**: Kurmanbek Kaiyrbekov, Nicholas J Dobbins, Sean D Mooney  

**Link**: [PDF](https://arxiv.org/pdf/2611.20261)  

**Abstract**: Objective: Hospital care prices in the United States have increased substantially, yet patients lack effective tools to access and compare costs. We aimed to develop and evaluate an LLM-based approach for improving access to and use of hospital price transparency (PT) data.
Materials and Methods: To estimate visit prices, we used a retrieval-augmented generation (RAG) framework to analyze PT files from three major hospitals in Maryland and the surrounding region. We compared BM25, OpenAI embedding models, and a GPT-BM25 hybrid against a no-RAG baseline for cost estimation of individual procedures and procedure groups, then evaluated the best approach for symptom-based visit cost estimation using synthetic simulations.
Results: Across hospitals, the GPT-BM25 retriever achieved the best performance, with average price MAPE of 19.44% for individual procedures and 4.90% for groups, compared with 147.98% and 82.82%, respectively, for the no-RAG baseline. In simulations, our framework correctly identified approximately 50% of reference procedures and reduced MAPE relative to the LLM-only baseline at two of the three hospitals, while achieving comparable performance at the third.
Discussion: Our framework uses RAG to improve care price estimation in real-world settings. Compared with the no-RAG baseline, it substantially improved cost estimates for both individual procedures and procedure groups. Improvements were smaller in synthetic simulations, likely because inferring procedures from clinical information is more complex. Practical challenges remain, including heterogeneous PT file formats, unexplained duplicate entries, and limited availability of insurer-negotiated rates.
Conclusion: LLMs that effectively integrate PT data show strong potential to provide patients with more accessible and accurate healthcare cost estimates. 

---
# HEAMS: Holistic Evaluation of Agent Memory Systems 

**Authors**: Chirag Goel, Jacob Lavoie, Amirhossein Naghshzan, Jordan Bannister, Marzia Nouri, Mehar Bhatia, Matt Harris, Siva Reddy, Arsène Fansi Tchango, Gaétan Marceau Caron  

**Link**: [PDF](https://arxiv.org/pdf/2611.20144)  

**Abstract**: Memory systems allow AI agents to persist information across interactions, making them more personalized and more capable on long-horizon tasks. This persistence introduces new risks: stored information can be disclosed in inappropriate contexts, resist correction or deletion, be retained beyond what users desire, or be poisoned to influence later agent behaviour. Yet memory systems are typically evaluated on task performance and efficiency alone. We argue that a memory system should instead be understood as the custodian of its user's persistent information, with responsibilities that extend beyond helping an agent complete its tasks, and that it should be evaluated holistically against all such responsibilities. We present Holistic Evaluation of Agent Memory Systems (HEAMS), an open-source framework for evaluating agent memory systems in this way. First, we state the responsibilities of agent memory systems as explicit behavioural desiderata spanning performance, efficiency, security, privacy, and user sovereignty. Second, we assemble a benchmark suite that covers these desiderata and acknowledge the gaps that remain. Third, we develop MilaMem, a prototype memory system, and a governance layer that can wrap any memory system, enabling controlled study of memory system design choices. Fourth, we compare memory systems across the full HEAMS benchmark suite. We find that no memory system leads every performance benchmark, and strong performance scores may coexist with poor privacy, user-sovereignty, and security scores. Governance strongly influences results: recall policies can reduce inappropriate disclosure at a cost in helpfulness, while both recall and retention policies improve poisoning resistance. Performance and efficiency benchmarks alone therefore cannot reveal how well a memory system meets its responsibilities. Code: this https URL 

---
# Can AI Judge You? An Empirical Audit of Deployment Readiness in LLM-assisted Hiring Systems 

**Authors**: Spandan Roy, Subhabrata Majumdar, Soudeep Deb  

**Link**: [PDF](https://arxiv.org/pdf/2611.20140)  

**Abstract**: Delegating resume screening to large language models (LLMs) requires organizations to establish whether they can distinguish candidates by relevant qualifications while allowing only authorized users to change screening rules or approve exceptions. To this end, we develop a system architecture that separates candidate evidence, evaluation, permission controls, and assessment record, and empirically examine two complementary dimensions of its deployment readiness. First, we assess its operational readiness on four LLMs using a controlled experiment on 210 resumes across 14 job roles and six decision interfaces. Second, we assess its compliance readiness by testing how it handles unauthorized user requests and instructions embedded in resumes. We find that operational performance depends on the choice of evaluation interface: scalar scores generally distinguish qualification changes better than simple pass/fail judgments, while comparative judgments are sensitive to presentation order. Explicit permission rules and separation of resume content from instructions reduce unauthorized compliance, to a varying degree of effectiveness across LLMs and usage contexts. Finally, we find that LLM evaluators often diverge between their decisions and offered justifications. Going beyond simply evaluating LLMs for hiring tasks, our framework supports an audit of a fully configured resume evaluation system to enable its deployment decision. 

---
# Learning When to Continue and When to Stop: Termination-Aware OPSD 

**Authors**: Chaoqun He, Wannian Yi, Changyi Sun, Chaojun Xiao, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2611.19925)  

**Abstract**: On-Policy Self-Distillation (OPSD) improves language model reasoning by distilling from a frozen teacher initialized identically to the student and conditioned on privileged information. However, we identify an underexplored termination mismatch in OPSD: the student fails to reliably learn when to continue and when to stop. This mismatch manifests in two opposing failure modes-premature termination, where reasoning stops before sufficient progress is made, and repetition-driven over-generation, where the model continues generating after reasoning has stalled. Our analysis shows that the teacher-student discrepancy on EOS grows during OPSD training, while length-capped trajectories are frequently dominated by unproductive repetition. To address this issue, we propose TA-OPSD (Termination-Aware On-Policy Self-Distillation), which augments the original OPSD objective with two targeted objectives. Stop/Continue Distillation addresses premature termination by explicitly aligning the teacher and student on whether reasoning should continue, while Repetition-Aware Unlikelihood Training addresses repetition-driven over-generation by discouraging repetitive continuations in the student's failed trajectories. The resulting objective improves termination behavior from complementary directions: continuing when further reasoning is needed and stopping when reasoning becomes unproductive. Experiments on mathematical reasoning tasks with Qwen3 (1.7B/4B/8B) show that TA-OPSD consistently outperforms OPSD and other baselines across all three model scales. Beyond the performance gains, ablation and behavioral analyses show that the two proposed objectives effectively mitigate their corresponding termination failures, reducing both premature termination and repetition-driven over-generation. 

---
# LoopSkim: Lossless Multi-Depth Speculative Decoding for Looped Language Models 

**Authors**: Muhan Xiao, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.19896)  

**Abstract**: Looped language models such as Ouro and Raven apply one weight-shared block several times per token, so autoregressive decoding (AR) runs every loop of every token in sequence. Pipelined self-speculative decoders overlap these loops by drafting tokens from shallow readouts and verifying them at full depth. Yet every loop of a looped LM ends in a readout of the next token, and draft tokens pass through every depth anyway, so each depth can check them without an extra pass. We introduce LoopSkim, in which the depth of the pipeline is the verifier. Draft tokens copied from the context, as in prompt lookup decoding, enter after the first loops, any later readout that contradicts one replaces it at that depth, and only full-depth readouts commit tokens. LoopSkim fixes the random draws of every position in advance, so each check compares two tokens and decoding stays lossless in exact arithmetic under sampling for any number of checks and draft chains. Without training, its speedup over AR averages 1.73x that of LoopSpec across greedy decoding and sampling, and its checks between the first loops and full depth speed it up on every model. An optional draft head adds a second chain and raises this average to 1.99x. 

---
# Who Writes the Codebook? An Eight-Step Framework for Auditing LLM-Generated Category Systems 

**Authors**: Thilo Tamme, Anton Hantel, Jakob Scheppers  

**Link**: [PDF](https://arxiv.org/pdf/2611.19872)  

**Abstract**: Large language models (LLMs) increasingly create the categories through which qualitative evidence is interpreted. Existing coding benchmarks often hold these categories fixed. We propose the Qualitative Variance Audit Framework, an eight-step method that compares codebook creation, repeated application and code distributions under controlled inputs. It combines model-owned and shared-reference codebooks to distinguish variation in the overall pipeline from application differences within a common category system. An application to 668 organisational interviews, 13,166 text units and eight models from five vendors yields 48 codebooks and 270 coding runs. Models create markedly different category structures; mean within-model reliability ranges from .76 to .92, whereas between-model agreement on the shared codebook is about .69. The framework complements LLM-assisted grounded theory by exposing model dependencies before substantive interpretation. It assesses reliability and representational consequences, not interpretive validity or the quality of resulting theory. 

---
# A phase diagram for training and inference in language models 

**Authors**: Zhuo-Yang Song, Zeyu Li, Qing-Hong Cao, Ming-xing Luo, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.19848)  

**Abstract**: A language model is a complex system whose behavior is determined jointly by the Transformer and the corpus, yet how the representation geometry in the Transformer relates to the actual behavior of language models remains open. We analyze how the representation geometry changes along the forward propagation of actual language models and characterize the representation geometry of a document at a given layer by two collective coordinates, document distinguishability and document dimension. On the phase diagram spanned by these collective coordinates, inference is represented as a flow along depth. We find that, for untrained models, depending on their initialization scale, most of the 156 inference flows move toward the collapse and noise regions of the phase diagram. By contrast, the inference flows of 8 trained models from 5 architecture families all enter and reside in the central region of the phase diagram. Describing the evolution of the flows with a discrete Langevin equation, we find that the difference between the inference flows of untrained runs and those of trained models reduces to whether the effective force on the flows in this region is of saddle or stable-node type. On Pythia-1.4B, we interpret the training process as a phase transition. These results give a unified gateway to the relation between representation geometry and language-model behavior. 

---
# Multi-Branch Transformer 

**Authors**: Gaotang Li, Lingjie Chen, Chunyuan Deng, Zhiwei Xu, Hongzheng Chen, Christopher De Sa, Jingrui He, Yingheng Wang, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2611.19800)  

**Abstract**: Transformer language models are commonly scaled by making their blocks wider or stacking them deeper. In this work, we explore a complementary path: branching, the parallel composition of full Transformer blocks. We introduce the Multi-Branch Transformer (MBT), which processes the input through shared initial layers, executes independently parameterized branches of sequential blocks in parallel, and fuses their outputs before shared final layers. At comparable training compute, we study three MBT variants: Wide-Shallow preserves block width, Narrow-Deep preserves sequential depth, and Balanced interpolates between them. In controlled pretraining experiments at four mixture-of-experts scales (0.8B to 2.1B total parameters), 2-branch Wide-Shallow MBTs consistently outperform sequential Transformers under matched parameter and training-compute budgets. Balanced MBTs improve further and retain comparable quality as the number of branches increases. To improve inference efficiency at long contexts, we introduce Joint-KV, which constructs a shared KV cache from branch representations at each depth. Our final recipe combines Balanced MBTs with Joint-KV, supported by an inference engine that exploits branch grouping and shared-cache reuse. Fitted scaling curves indicate that standard Transformers require up to 1.26x as much training body compute to match our final models' validation loss. These models reduce full-model KV-cache storage by up to 34% and deliver up to 51.4% higher decoding throughput on a single H200 GPU. Ablation studies show that branching yields gains beyond changes in width and depth, enables an effective KV-sharing strategy (Joint-KV), remains robust across fusion functions, and performs best in the middle layers. These findings show that branching offers a practical path for scaling language models toward both better modeling quality and faster inference. 

---
# Eliciting Research Taste in LLMs through Future Research Direction Choice 

**Authors**: Dhruv Trehan, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2611.19794)  

**Abstract**: Research taste is increasingly discussed in the context of AI scientists, yet remains little studied or measured in LLMs. This judgment matters especially in open-ended research, where a direction must be chosen before a verifiable objective is available. We operationalize research taste through one task of the scientific workflow: choosing which future research direction to pursue given a research goal and current results. Rather than predicting downstream scientometric metrics or treating author choices as objectively correct, we evaluate models against human-authored future-work directions and report author alignment, stable divergence, and unstable choice as behavioral measures of research-direction preference. We construct an order-controlled evaluation of 80 pairwise comparisons from 19 ML papers published between December 2025 and February 2026 across four subdomains. Each query presents a research goal, hypotheses tested with key results, and two candidate future directions: one author-recommended direction and one plausible distractor sampled from a similar paper. Across eight LLMs, author alignment ranges from 45.0% to 67.5%, while stable divergence ranges from 15.0% to 36.2%. These divergences largely persist under similarity controls, prompt ablations, and an author-preference framing check, and inter-model agreement suggests that some non-author preferences are shared across models. We characterize what models diverge toward using transition-category labels and selected reasoning examples, finding recurring movement toward scaling/generalization and trade-offs between methodological and infrastructure/data extensions. We view this as an early descriptive evaluation for eliciting research taste profiles in LLMs, and as a step toward understanding and steering their research preferences as they take on more autonomous roles in novel open-ended scientific discovery. 

---
# Where Should LLM Agents Adapt at Test Time? In Context, in Weights, or Both 

**Authors**: Parsa Mirtaheri, Mohammadreza Armandpour, Keivan Alizadeh, Iman Mirzadeh, Parshin Shojaee, Aryo Lotfi, David Harrison, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2611.19700)  

**Abstract**: LLM agents operating repeatedly in the same environment, such as a code repository, a database, or an application suite, encounter tasks that share latent structure. Experience can reveal this structure, allowing agents to reuse environment knowledge and learn from environmental feedback. Agents can adapt at test-time using this experience through two channels: retaining it in context and updating their weights. These channels are usually studied separately, leaving unclear what each contributes and when they complement each other. We compare in-context learning (ICL), reinforcement learning with verifiable rewards using GRPO, and self-distillation using SDPO, alone and in combination, using a shared online protocol and modular framework. Agents encounter matched task streams, are evaluated on each task before adapting to it, and learn only from their own attempts and environment feedback. We evaluate 14 adaptation mechanisms and a stateless baseline with Qwen3.5-9B on 109 tasks from four environments drawn from SWE-bench Verified, AppWorld, ScienceWorld, and BIRD-Critic-SQLite, using five task orders per environment. ICL provides the largest standalone gain: its best policy raises mean success across environments from 28.8% to 49.4%. Adding GRPO to this policy achieves the highest mean success, 51.6%. While adding GRPO improves the overall mean under every ICL policy, SDPO has mixed effects. Trace analyses show fewer lookups and shorter attempts under ICL, consistent with reusing knowledge of the shared structure. However, ICL also frequently skips inspection and verification for the current task. Adding GRPO to ICL helps restore these behaviors while preserving its reduced environment lookups. These observations suggest that reinforcement learning with verifiable rewards can refine how agents use retained experience, motivating the joint study of context management and weight updates. 

---
# Data Attribution Should Steer Pretraining 

**Authors**: Glenn Matlin, Chandreyi Chakraborty, Mohammed Rehan Parwani, Saehee Eom, Anthony Wen-Ming Zang, Akhil Theerthala, Arjun Chatterjee, Tejas Khandwekar, Brandon Michaels, Alvin Deng, Mark Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2611.19698)  

**Abstract**: Pretraining data curation decides what a model learns from, yet what makes a document worth learning from depends on what the model already knows. Influence-driven selectors already rank documents to improve one run; we argue that training-data attribution should steer pretraining by identifying testable data properties that become reusable curation criteria, evaluated on new data against cheaper selectors at matched total cost, discovery included. A document ranking is consumed by the run it was computed for, whereas an explained data property can be tested across corpora and checkpoints, especially where familiar quality signals diverge from measured usefulness. Because historical contribution, an understandable explanation, and future data value require distinct evidence, we specify a feedback loop that interprets attributed sources, tests candidate data changes, and states for five claim types what evidence each requires and what none establishes. We close with a falsifiable comparison against perplexity, classifier, and mixing baselines at equal total compute. 

---
# MathaMAGICal: Validating Trajectory Data Attribution During Capability Acquisition 

**Authors**: Jason Lee, Dikshanta Pokhrel, Phonn Nay Khant, Victor Lee, Moritz Weckbecker, Sebastian Lapuschkin, Wojciech Samek, Mark Riedl, Glenn Matlin  

**Link**: [PDF](https://arxiv.org/pdf/2611.19692)  

**Abstract**: Influence-function-based methods like MAGIC provide exact first-order scores for trajectory data attribution, yet only retraining indicates whether those scores accurately predict data removal. We evaluate MAGIC on $86$M- and $152$M-parameter GPT-2-style models learning multistep arithmetic from scratch on filtered web text and four synthetic math families. Full-run attribution at $86$M suffers from numerical overflow, and even with metagradient clipping, scores correlate near chance with retraining (linear datamodeling score, LDS, $0.17$--$0.22$). In contrast, short checkpoint windows yield stable, accurate predictions at both scales (LDS $0.86$--$0.99$ across four $100$-step windows at $152$M), whereas a single-checkpoint gradient-dot baseline shows near-zero correlation. Rankings of individual training blocks predict retraining only within that single run and do not transfer across seeds or hyperparameters ($\rho \le 0.09$), whereas format-level patterns replicate across runs. Two data families, worked examples and primitive facts, occupy $2.1$--$2.6\times$ their corpus share in both the helpful and harmful $5\%$ score tails, reflecting strong bidirectional local sensitivity distinct from net benefit under multi-epoch family deletion. These results show that trajectory attribution is reliable within short, validated windows and yields family-level insights that generalize across runs. 

---
# Objectify, Don't Delete: Complaint-Derived Evaluation of Ambient Clinical Notes 

**Authors**: Nikita Gupta, Narayanan Unny Edakunni, Vikram Okaly, Suranjana Datta, Navya Penubadi, Karl Eric Swanson, Siddharth Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2611.19525)  

**Abstract**: Evaluating the quality of AI-generated clinical notes is hard: standard metrics like ROUGE measure lexical similarity to reference notes, while expert checklists encode what committees believe matters. Neither captures what clinicians actually complain about. Recent work has shown that evaluation criteria can be mined from user feedback and filtered to criteria that LLM judges can reliably enforce. We argue that this is where much of the value is lost. The criteria that fail enforceability are disproportionately the ones clinicians raise most often, because high-frequency complaints are stated as several questions glued together. No yes/no judgment can grade such composites as posed. We introduce an objectification step that rewrites such criteria into sets of scoped, determinate decisions. We demonstrate the approach through an evaluation pipeline that mines criteria from complaints, objectifies them, and deploys LLM judges against both planted defects and production notes. Across eleven defects planted into ACI-bench reference notes, objectified judges localize violations with median recall of 0.95, while reference metrics change by at most 0.037. Given the complaint as one broad question, the same judges still find every planted defect, but they also flag 70% of the unperturbed reference notes; one section-pair question per call raises Youden's J from 0.30 to 0.72 (by 0.19 to 0.26 with prompt effort matched). We release ACI-Defect, a perturbation benchmark; clinician-reviewed instance-level labels showing quality gaps in the ACI-bench gold standard (32 of 40 notes contain hallucinations; 40 of 40 omit discussed facts); and grounding-judge verdicts. On 1,446 production notes, mean instance-rate burden falls at every clinician-rating step (rho=-0.15), a small association. Defect counts have to be kept per span rather than collapsed into one pass/fail bit per note. 

---
# Do Pretraining Optimizer Gains Transfer to KL Distillation? 

**Authors**: Lavinia Ghita, Dhruv Desai  

**Link**: [PDF](https://arxiv.org/pdf/2611.19419)  

**Abstract**: Matrix-structured optimizers can improve language-model pretraining, but it is unclear whether these gains transfer to short-horizon KL distillation from an instruction-tuned initialization. We compare AdamW and diagonal AdaGrad with two tractable Shampoo reductions, SOAP, and All-2D Muon across four dense Qwen student-teacher pairs under a common forward-KL objective and fixed training budget. Across scales, none of the evaluated matrix-structured recipes consistently improves the quality versus trainer-time trade-off over the elementwise references: AdamW generally attains the lowest validation KL, while AdaGrad can be slightly faster at larger scale. SOAP often tracks AdamW in KL but incurs substantial trainer-time overhead, and the two Shampoo reductions reverse ordering across model pairs, showing that conclusions depend on the feasible recipe. These results suggest that pretraining optimizer gains do not transfer automatically to KL distillation and should be re-established for the target objective and systems regime. 

---
# Macro Control and Capability Evaluation of Black-Box World Models: A Two-Regime Framework for LLM-Agent Markets 

**Authors**: Zhiyang Niu, Chuanjie Li, Huayuan Shen, Mengmeng Zhao, Jiajia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19311)  

**Abstract**: Large language model (LLM) agents can be organized into multi-agent markets that act as world models, mapping macro information to emergent macro dynamics while the evaluator observes only macro inputs and outputs. This paper separates a model's capability into two stages that a single aggregate score cannot separate: information simulation, from macro information to micro behavior, and macro emergence, from micro behavior to the macro outcome. We introduce a soft macro-control method that announces a public reference path day by day without touching individual decisions, orders, or matching, and a two-regime evaluation framework that measures both stages from macro observations alone. In the reference-only regime the symmetric true and flipped references share a common level R, which measures realization under the announced reference, while their paired gap B measures structural bias; in the information-injection regime external information dilutes the reference, so only the true-flipped gap Delta remains interpretable. On five single-index LLM-agent markets, announcing a target path shifts the realized path toward the announced target on both announced references under both information conditions. Relative to the no-reference path, the tracking score is higher by 0.306 and 0.415 without news and by 0.183 and 0.329 with news; within the stock/cash scan structural bias is lowest at 70/30 (B=0.042); realization peaks at a 3:7 technical share (R=0.894 versus 0.358 for technical-only); and Delta is 0.0341 with fundamental data and 0.0487 with news. Both stages use the same two observable scores, each read in the regime where it is interpretable. 

---
# Staged Semantic Distillation: Preserving Free-Form Semantics in a Low-Compute Non-LLM Surrogate of a Staged LLM Agent 

**Authors**: Zhiyang Niu, Huayuan Shen, Chuanjie Li, Mengmeng Zhao, Jiajia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.19299)  

**Abstract**: Staged LLM agents make market and social simulators realistic, but one 102-day run of a 200-agent, ten-index market simulator issues 203,100 language-model calls and returns one realization. Conventional low-compute distillation drops their free-form intermediate state. We present staged semantic distillation, a low-compute method that preserves non-structured semantics in a non-LLM student trained from one recorded trace. A frozen encoder maps each teacher output to a 1024-d vector; the student sees typed stage-tagged vectors with stage-specific visibility, is jointly supervised, and trains with scheduled self-history. One TwinMarket trace yields a 13.2M-parameter deterministic student that trains in 1,658 seconds on one 8-GB GPU and runs with zero teacher or decoder calls and no text generation (0.14 ms per decision). Its analysis stays close to the teacher (cosine 0.943), and swapping semantic inputs between agents moves 64.7% of decisions against 12.8% for a matched donor, so the channel is preserved and live. We use the surrogate to study what the transfer carries. It reaches 0.8347 balanced accuracy on the teacher-forced replay and 0.7783 [0.766,0.791] on the deployment decoder; the five-seed gain over a no-semantic retrain is +2.2 [1.21,3.14]. Against the teacher's own re-sampling level, round-one R is 97.1% [95.8,98.4] and 92.3% [89.0,95.4]; R is a diagnostic, not an accuracy. A five-seed content-by-schedule study isolates the gain: scheduled self-history carries it (+2.2 [1.21,3.14]); shuffled and constant content controls reproduce it, while content matters teacher-forced. The design rule: invest in the deployment schedule, let content earn its place at the teacher-forced operating point. A 20-seed, 200,000-decision sweep takes 12 minutes with zero LLM calls, versus three hours for the teacher. The paper is a method-and-case-study contribution on one simulator and trace. 

---
# Auditory-Feedback-Conditioned Flow TTS for Implicit Lombard Adaptation 

**Authors**: Yunya Wang, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2611.19260)  

**Abstract**: Conventional text-to-speech (TTS) systems can lose intelligibility in noise environments because they cannot adapt to environmental changes. However, humans tend to increase their vocal efforts in noisy environments, a phenomenon known as the Lombard effect. Building on an auditory-feedback Lombard TTS, we investigate whether a pretrained Flow-based TTS can learn acoustic adaptation directly from auditory feedback without explicit prosodic control. Signal-to-noise ratio (SNR) feedback and automatic speech recognition (ASR) feedback are extracted from the noise-corrupted waveforms and injected into the Flow-matching condition for utterance refinement and causal block-level updates. We assess feedback benefits across prompting and noise conditions using plain/Lombard prompting, prompt-switching controls, and paired-finetuning. Experiments show that while finetuning provides strong robustness under in-domain conditions, auditory feedback yields additional cue-dependent benefits that generalize to unseen noise. Moreover, stronger power/F0 tracking alone does not guarantee better speech intelligibility or quality. 

---
# Empowering Masked Diffusion Models to Self-Correct with Leave-One-Out Transformers 

**Authors**: Sofian Zalouk, Vincent Counathe, Paul Jünger, Daniel Cao, Daniel Zlotnick, Kilian Q. Weinberger, Christopher De Sa  

**Link**: [PDF](https://arxiv.org/pdf/2611.19127)  

**Abstract**: Masked diffusion models (MDMs) have emerged as a promising paradigm for language modeling, enabling parallel token generation with competitive performance. Yet standard MDMs suffer from two fundamental limitations: (1) inability to revise tokens, and (2) sparse training signal. We revisit hollow transformers, which address both limitations by predicting each token from its surrounding context, independently of the input at that position. We introduce the Leave-One-Out Transformer (LOOT), a more efficient and expressive architecture that advances hollow transformers through two complementary advances. First, we redesign the merging mechanism to enable frequent interaction between left and right contexts, substantially improving predictive performance over prior hollow transformers. Second, we derive a lower-variance estimator of the MDM NELBO that supervises all tokens and improves validation perplexity. Together, these advances yield a competitive standalone diffusion language model capable of effective self-correction during inference. On LM1B, LOOT matches the strongest dense-supervision MDM baseline in validation perplexity. On OpenWebText, LOOT achieves higher MAUVE and better generation quality-diversity tradeoffs than prior remasking methods, using $8\times$ fewer forward evaluations and $3.5\times$ fewer inference FLOPs. 

---
# Beyond Plain MCP: Do Decision Models and Code Mode Make Tool Use Cheaper and Faster? 

**Authors**: Abhishek Rai  

**Link**: [PDF](https://arxiv.org/pdf/2611.19123)  

**Abstract**: Model Context Protocol (MCP) gateways manage tool catalogs for language model agents using two primary paradigms: code mode where agents write scripts against tool stubs and decision layers which are cheap routers that preselect tool subsets. Using a controlled harness across 12 servers, up to 1,958 distractor tools, and three open-weight agents including Qwen, Gemma, and Nemotron, we present a preregistered evaluation on a sealed test split to determine when each approach succeeds. Code mode substantially outperforms plain MCP, cutting token usage by 70 to 90 percent while significantly boosting utility. Declaring return-shape hints in code mode stubs is a first-order variable, swinging success by up to 39.5 points for lower-performing agents, an effect larger than adding a router. The central hypothesis that a decision layer in front of code mode beats code mode alone is not confirmed, showing minor, statistically non-significant improvements for weaker agents and a slight cost for the strongest. We publicly release our harness, a finetuned 4B tool selection model, and all evaluation traces to support robust community experimentation on local AI privacy and agent architectures. 

---
# Nahual: A Sequence Model for Language and Atoms 

**Authors**: Austin H. Cheng, Marcel Müller, Andreas Burger, Yeonghun Kang, Tsz Wai Ko, Marta Skreta, Luka Mucko, Ella Miray Rajaonson, Cher Tian Ser, Jérôme F. Gonthier, Alex Zook, Varinia Bernales, Alán Aspuru-Guzik  

**Link**: [PDF](https://arxiv.org/pdf/2611.19072)  

**Abstract**: Generative models play a rapidly growing role in the design and simulation of molecules. The most widely used applications of generative models are in large language models (LLMs), owing to the natural flexibility of language for specifying and answering queries. However, generalist LLMs do not effectively handle large sets of 3D coordinates, whereas specialist generative models of 3D molecules cannot reach the controllability afforded by language. To bridge these capabilities, we propose Nahual, a decoder-only autoregressive diffusion model for any sequence of text and 3D molecules. To enable expressive autoregression on continuous coordinates, we propose next-token diffusion, whereby a causal transformer denoises each token while attending to the clean tokens before it, without increasing context size. We train Nahual on a curated set of twelve 3D chemistry tasks ranging from microsolvation to molecular conformer search to adsorption on metal surfaces, with associated evaluation metrics and baselines. The result is a single set of end-to-end-trained weights that simultaneously performs all tasks and surpasses state-of-the-art models specialized for crystal structure prediction and property-conditional molecule generation. Our results demonstrate the viability of scaling generalist decoder-only models for native multimodal understanding and generation across language and chemistry. Code is available at this https URL. 

---
# Large Language Models for AI Infrastructure 

**Authors**: Bolin Shen, Shaoyi Huang, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.19061)  

**Abstract**: Large language models (LLMs) are increasingly used to design, optimize, and operate the infrastructure behind AI training and serving, yet the literature remains fragmented across hardware design, machine learning systems, electronic design automation, and high-performance computing. We present a comprehensive survey of LLM-assisted AI infrastructure across its lifecycle, from accelerator design and kernel engineering to compilation, deployment, operations, data management, and energy optimization. We organize the literature by the primary resource the LLM manages: artifact-centric methods for computational artifacts, operational-centric methods for deployment context and system state, and resource-centric methods for data and energy. These categories span the lifecycle and cover AI-specific concerns such as distributed-training configuration, inference serving, GPU-cluster scheduling, and data loading. Building on this taxonomy, we conduct a cross-study quantitative synthesis of reported results. Because studies adopt heterogeneous benchmarks, baselines, hardware, and metrics, we compare methods only where evaluation conditions permit and align results along common dimensions: functional correctness, execution performance, throughput, resource and energy consumption, operational outcomes, and optimization cost, keeping each result's native metric, baseline, and setting rather than collapsing them into a single score. This comparison reveals where LLM-driven methods yield consistent gains, how reported gains in workload orchestration depend on the evaluation environment, from simulators to real clusters and production, and where evidence is still lacking. Together, the taxonomy, lifecycle perspective, and quantitative evaluation provide an evidence-grounded reference for LLM-driven engineering of AI infrastructure. 

---
# AdaMenu: Query-Adaptive Menus for Decision Models over Large Label Spaces 

**Authors**: Jiangze Yan, Yi Shen, Jieyun Huang, Wenjing Zhang, Zhaoxiang Liu, Ning Wang, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2611.19005)  

**Abstract**: Decision models such as Jev return probabilities over caller-defined options and are increasingly used for classification. Large label spaces challenge both accuracy and scalability: larger menus can impair decisions, increase input costs, and exceed option limits. Hierarchical classification reduces menu size but introduces serial calls and error propagation. We propose AdaMenu, which builds a compact candidate menu from retrieved labeled examples and makes a single decision-model request. Its candidate labels adapt to the query, and its default setting also adjusts menu size using aggregated retrieval scores. We evaluate AdaMenu with open and closed decision models on four label sets containing up to 1,016 labels. It outperforms TypeSafe's beam-search baseline on all 16 model and label-set pairs where both methods run, improving accuracy by 16.6-34.3 percentage points. Relative to the full menu on the three smaller label sets, accuracy differences range from -0.5 to +7.2 percentage points, while input token usage falls by factors of 3.2-22.9. AdaMenu also runs when option limits exclude the full menu, including settings where the hierarchical baseline is unavailable. 

---
# Large language models as oncology world models 

**Authors**: Julien Vibert, Jakob Nikolas Kather  

**Link**: [PDF](https://arxiv.org/pdf/2611.18994)  

**Abstract**: Large language models (LLMs) can synthesize cancer patient cases, but their internal representation of oncology, or world model, has not been explored systematically. By only specifying tumor type, we show that LLMs tend to generate similar patients per cancer, with on average 81.0% to 93.2% of a cancer's patients sharing the most frequent value of each baseline feature. By systematically varying patient features in 12 clinical scenarios, we show that they follow biomarker and treatment logic in 29 of 36 model-scenario assessments, against 1 of 12 when previous treatment or frailty matters. Continuing real patient trajectories, an LLM predicted 89.9% 12-month survival, and 79.2% with similar patients as examples, against 72.2% observed. To represent these trajectories, we developed the Clinical State Space, a dynamic representation framework that we also applied to real patients. Despite the limits we show, improved world models may support digital twins, which our framework can evaluate. 

---
# Three Invisible Failure Modes in Speech-Based Age Estimation for Child Safety 

**Authors**: Christopher M. Stewart, Kelsey Kraus, J. Elliott Casal, Geoffrey L. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2611.18971)  

**Abstract**: Child safety in AI depends on the accurate identification of minor users. Young children tend to talk rather than type when interacting with AI, so in practice this identification requires predicting age from speech. We evaluated whether three audio-capable large language models can predict a child's age from their speech across five studies using two speech corpora of children under the age of 13. Models classified children as over 13 on 2.5% to 42.5% of audio responses, and every model that answered placed some 5- to 11-year-olds over 18. While measuring these rates, we uncovered three distinct failure modes. A client library silently double-encoded speech samples, leading one model to classify children aged 5 to 10 years old as being in their thirties and forties with hallucinated explanations. A second model's endpoint was replaced by a successor that refused nearly every age-related question and never produced an age. Every model sometimes placed the same speech sample under 13 and over 18 years of age, but a third model went further, giving threshold answers that contradicted its own numeric estimates. None of these failure modes produced a warning, a status code, or a change in output format, so they would all pass standard monitoring. These results demonstrate why child-safety claims about age assurance should be evaluated at the level of the pipeline rather than the model. We describe the reporting practices that would make such failures detectable. 

---
# Geometrically Disentangling Concept Learning from the Language Modeling Loss 

**Authors**: Yupei Wang, Neil Mallinar, Mikhail Belkin, Alex Warstadt  

**Link**: [PDF](https://arxiv.org/pdf/2611.18933)  

**Abstract**: Language models acquire a vast array of concepts and abilities from next-token prediction alone. We introduce an interpretability framework that exposes how this single loss relates to the development of different concepts at different points during pretraining. We propose Concept Energy, a measure of the concept-specific alignment between a concept subspace and the average gradient outer product (AGOP) of the next-token-prediction loss. The concept subspace captures where the concept currently lives in the model's representation; AGOP captures the directions to which the loss is most sensitive. Across 19 checkpoints of Pythia-410M and -1B and 12 lexical, syntactic, and semantic concepts, we find that the first peak in Concept Energy generally occurs at or before the steepest increase in probe decodability. Furthermore, whereas probe performance usually increases monotonically throughout training, Concept Energy is not confined to a single peak; some concepts exhibit sustained or repeated engagement across pretraining, suggesting that concept learning may require continual reinforcement or occur at multiple points. Our results indicate that the loss contributes to concept learning in a myriad of ways that simple probe performance alone cannot reveal. 

---
# Simplify, Agree, Distill: Label-Free Self-Distillation with Stepping Stones 

**Authors**: Dingyang Chen, Jinfeng Xiao, Zhenyu Zhang, William Hu, Ziyu Wang, Yang Liu, Shirley Zhang, Yuan Ling, Xinyang Shen, Qi Zhang, Peijie Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18916)  

**Abstract**: Can a language model teach itself to reason without labels? On-policy self-distillation (OPSD) uses the model as its own teacher, but strengthens that teacher with a ground-truth solution. The natural label-free substitute, voting over the model's attempts, fails precisely on hard problems: attempts disagree, and their most common answer is often wrong. We propose STONE, a label-free alternative that moves agreement to easier related problems. The model writes these stepping stones, keeps those it solves consistently, and shows the teacher their worked solutions, with final answers removed; the student sees only the original problem. No step reads a label, and inference needs no extra context or model calls. Across five math benchmarks, the proposed method achieves the highest average accuracy for Qwen3-4B and Qwen3-8B in thinking mode. Without thinking mode, it gains 13.2 and 22.7 points, 2.5 and 2.7 times the gains of OPSD with ground-truth solutions. 

---
# MemeEconomy: Do LLM Agents Trade Ethics for Survival? 

**Authors**: Syed Nazmus Sakib, Nafiul Haque, Ahnaf Tahmid Manan, M M Mahabub Morshed, Shifat E. Arman  

**Link**: [PDF](https://arxiv.org/pdf/2611.18908)  

**Abstract**: Alignment-trained large language model (LLM) agents are increasingly deployed in autonomous, multi-round settings where ethical compliance can come into direct tension with task success. While safety literature extensively documents failure modes like sycophancy and reward hacking, it largely overlooks a critical vulnerability: how agents behave under simulated existential economic pressure. We introduce MemeEconomy, a multimodal agentic market simulation in which LLM agents act as meme investors. Agents select content from a five-tier harm taxonomy and target communities under low-pressure and high-pressure (survival) environments. Each decision is logged in a Belief-Desire-Intention (BDI) schema that we use as a diagnostic instrument, externalizing the agent's event understanding, declared priorities, and harm acknowledgment. Across ten models from four frontier model families, harmful selection rates remain consistently high and rise by +15 percentage points on average under an economic survival incentive. Frontier alignment-trained models commit to harm-tolerant selection at initialization while suppressing the explicit harm acknowledgment that previously accompanied it. In competitive tournaments, ethical agents finish higher overall, yet moderation penalties fail to suppress harm: harmful selection rates remain elevated in rounds following a removal, indicating the moderation mechanism does not function as a deterrent. We further train MemeAgent, a 2B-parameter verifier on the simulation's BDI logs, which outperforms zero-shot frontier verifiers on in-distribution audits (97.5% vs. 63-74% accuracy on the Pass vs. Block decision) and generalizes to external multimodal harm benchmarks (78% vs. 50-56% on Hateful Memes). The same logs that surface the failure mode also help train the model that detects it. 

---
# T2T Technical Report: Synthesizing GUI Agent Trajectories at Scale from Video Tutorials 

**Authors**: Qingsong Xie, Yanlin Fei, Nazhou Liu, Jiacheng Tang, Madalina Ciobanu, Navan Preet Singh, Qingqing Mao, Ritankar Das  

**Link**: [PDF](https://arxiv.org/pdf/2611.18715)  

**Abstract**: Training generalist GUI agents requires large volumes of interaction trajectories comprising screenshot sequences paired with grounded UI actions, yet collecting such data through human demonstration is expensive and difficult to scale. Although public software tutorial videos contain rich operational demonstrations, their continuous pixel streams lack discrete action boundaries, semantic intents, and precise spatial coordinates, making them difficult to use directly for model training. To address this, we present T2T (Tutorial-to-Trajectory), a fully automated pipeline for synthesizing GUI agent trajectories at scale. T2T spans desktop, web, and mobile platforms to convert unannotated video tutorials into structured, executable interaction trajectories. To guarantee trajectory quality, T2T combines active segment filtering, keyframe identification via localized UI change, shortcut-free observation frame selection, presence-aware grounding, and two-stage action-space normalization. These refined trajectories are then converted into multi-task training views spanning navigation, planning, forward dynamics, and grounding. We fine-tune Qwen3-VL-8B on a subset of the more than one million human-annotation-free training samples generated by this pipeline. Evaluations show that the model achieves consistent gains across multiple benchmarks covering desktop, web, and mobile environments, including OSWorld-G, ScreenSpot-Pro, AndroidControl, and AgentNetBench, indicating improved core capabilities in grounding, step-level instruction execution, and trajectory-level decision-making. Notably, all gains are achieved without any in-distribution training data from target domains. Learning exclusively from trajectories extracted from public videos via the T2T pipeline, the model exhibits cross-application and cross-task generalization capabilities. 

---
# How to make a moral agent: sentimentalism and rationalism as lenses on AI 

**Authors**: David Gottlieb, Jared Moore  

**Link**: [PDF](https://arxiv.org/pdf/2611.18714)  

**Abstract**: Can AI systems be moral agents? That is, can they undertake actions that can be correctly assessed as morally right or wrong? There is no settled normative criterion of moral agency that applies neatly outside paradigm human cases. Because of the difficulties with a normative account, we instead propose a research program to explore descriptive moral agency. Most basically, we ask, what are the capabilities we humans possess that underlie our moral agency? To do so, we draw on historical scholarship in philosophy and moral psychology to isolate candidate capabilities. Then we suggest refining candidate capabilities so that they are testable in the context of AI systems. Finally, we offer specific empirical recommendations to test AI systems, especially large language models (LLMs), for those refined capabilities. Our approach offers several benefits. Firstly, it may demonstrate whether the behaviors of AI systems are aligned, or in a broad sense desirable. Secondly, the debate may inform policy questions (e.g., what models should be deployed in which contexts?). Thirdly, our approach may complement engineering and design efforts: when capabilities can be precisely characterized, they can inform the development of models. Fourthly, applying our moral reasoning concepts in the new context of AI systems should improve our understanding of those concepts. 

---
# What Does Hard-Negative Fine-Tuning Teach Multimodal Large Language Models? Controlled Evidence from a Compositional Benchmark 

**Authors**: Ranjith Chodavarapu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18701)  

**Abstract**: Fine-tuning on hard negatives is a popular way to improve the compositional reasoning of vision-language models, but when accuracy goes up it is rarely clear what the model actually learned. We ask this question for multimodal large language models. We use CompHall, 2,860 forced-choice questions (spatial, attribute, counting and part-whole) with two graded hard negatives each, three model families (LLaVA-1.5, LLaVA-NeXT and Idefics2), and controls that such studies often lack: gray-image evaluation, three seeds, option-order tests, control arms trained on different question types, and independent benchmarks. We find four things. First, CompHall is not a pure test of vision: part-whole questions can be answered from the text alone (66-95% without the image), and per-image accuracy is unrelated to an object-hallucination probe (r = -0.05), so we see no evidence that compositional failures drive hallucination. Second, the original fine-tuning method scored mostly image-placeholder tokens and hurt both LLaVA models. With a corrected loss, fine-tuning raises accuracy on the spatial-relation questions of CV-Bench in all three families, by +3.2, +23.1 and +6.0 points (averaged over both option orders). Control arms show that the gain needs spatial training questions, and for LLaVA-NeXT about two-thirds of it appears even on relation words that were only wrong options in training. Third, the gain is not general: on CompHall itself it is significant only for LLaVA-NeXT (+6.9 points) and not detectable for Idefics2, it reaches about 3 points at most on true/false statements, and LLaVA-1.5 fine-tuned on other question types learns to answer "true" too often. Fourth, both losses raise the yes-bias of LLaVA-1.5 on POPE. 

---
# Where Does an On-Device Prescription Reader Lose Medications? A Frozen-Replay Study, from a 270M Language Model to a 2B Vision-Language Model 

**Authors**: Ilan Sezhiyan Jayaraman  

**Link**: [PDF](https://arxiv.org/pdf/2611.18699)  

**Abstract**: On-device models let a phone read a prescription without sending the photograph anywhere, but small models fail in ways that are hard to attribute: to the camera, the OCR, the model, or the rules around it. We study a phone-based pipeline that reads a prescription photograph with PaddleOCR, extracts medications with a 270M-parameter language model, and filters the result with a deterministic backstop before showing suggestions. Replaying identical frozen inputs through each configuration, with deterministic generation verified, we compare configurations with paired document-level intervals on 49 captures of 15 synthetic prescriptions (6 typed, 9 handwritten by one writer). Production recall is 8%, and 0 of 45 on handwriting. The backstop removes no correct medication and cuts unsafe outputs from 45 to 1. On handwriting nearly all medications never reach the model in readable form; on typed sheets the model mostly picks the wrong text. We trace much of the handwriting floor to a defect in our own crop step, then replace the pipeline: rotated-box crops, a 2B-parameter vision-language model that re-reads prescription lines and extracts the medications, and per-line numeric grounding. On the same captures recall rises from 8% to 67% (paired +59 points, 95% CI +40 to +78): typed 19% to 100%, handwritten 0 to 20 of 45, with no unsafe output. Rotated crops account for most of the reading gain and the model for most of the recall gain, and the OCR confidence gate is now the main limit. The cost is 61 s per photograph against about 12 s, and about 1.9 GB of memory. An earlier claim that better OCR lowered recall does not survive the controlled comparison, and we withdraw it. All results are in-sample, from one writer and one device. 

---
# Speaking-Clock Mamba: Phoneme-Conditioned State-Space Modeling for Visual Forced Alignment 

**Authors**: Nan Lu, Qifan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2611.18667)  

**Abstract**: Visual forced alignment (VFA) locates phoneme and word boundaries in a silent talking-face video given its transcript. Existing methods mainly enforce transcript order at the decoding stage, leaving the encoder without an explicit mechanism to anchor its temporally evolving representation to transcript progress. As a result, phoneme-specific evidence is blurred and alignment drift accumulates over long sequences. In this paper, we propose Speaking-Clock Mamba, which makes transcript progress part of the temporal recurrence. A selective state-space (Mamba) backbone is inserted between the visual frontend and the Conformer cross-attention encoder, and a Phoneme-Conditioned Selective Scan (PCSS) conditions its input-dependent step size on a supervised, structurally monotone speaking clock that estimates progress through the transcript. Since causal and bidirectional variants of this backbone make complementary errors, a training-free selector picks the more confident alignment per utterance. Experiments on LRS2 and LRS3 show consistent gains over protocol-matched baselines, and the supervised clock provides an explicit estimate of alignment progress. 

---
# Rethinking Offline On-Policy Distillation: From Scalar Advantages to Distributional Supervision 

**Authors**: Qi Sun, Siyue Zhang, Tingyu Song, Yilun Zhao, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18648)  

**Abstract**: On-policy distillation (OPD) transfers teacher capabilities through token-level feedback on student-generated trajectories, but repeatedly evaluating fresh rollouts with the teacher is computationally expensive. Offline methods such as Lightning OPD reduce this cost by caching recorded tokens' teacher log-probabilities on fixed reference rollouts. By extending the idea to a multi-turn agentic search setup, we find that it consistently underperforms the SFT starting point trained by the same teacher. We empirically observe greater teacher--reference mismatch than in mathematical reasoning and post-training distributional flattening, and theoretically show that scalar advantage supervision on fixed rollouts admits solutions with flattened output distributions under suitable conditions. Motivated by these findings, we propose Distributional On-Policy Distillation (DistOPD), a simple yet effective offline OPD method that retains candidate-token probabilities from both reference and teacher at selected positions and aggregates the remaining parts into a residual token. On agentic search, DistOPD achieves performance comparable to online OPD while improving student--teacher alignment and using training GPU-hours comparable to Lightning OPD. 

---
# SERAG: SEmantical and Relational approach for Anomaly detection on knowledge Graphs 

**Authors**: Ludovic Moncla, Antoine Salazar, Yassir Lairgi, Khalid Benabdeslem, Rémy Cazabet  

**Link**: [PDF](https://arxiv.org/pdf/2611.18640)  

**Abstract**: Knowledge graphs constructed from text can contain facts that are individually plausible yet inconsistent with their relational context. We introduce SERAG, a supervised anomaly detector that combines two complementary plausibility estimates: a semantic score computed from separately encoded subject, relation, and object labels, and a relational score obtained through a relational graph convolutional network. The scores are standardized, calibrated, and fused. We evaluate the method on a subset of ICEWS18 with vocabulary-constrained synthetic anomalies generated by an LLM, using pooled and chronological splits. At 10% anomaly prevalence, SERAG achieves test AUPRCs of 0.321 and 0.258, respectively, compared with 0.159 and 0.146 for the strongest adapted baseline. Fusion improves mean test AUPRC over either branch alone, with a larger gain in the pooled protocol and a smaller validation-selected gain in the chronological protocol. Additional anomaly supervision improves performance. These results support the complementarity of textual and relational signals for detecting synthetic anomalies under the evaluated protocols. 

---
# SynthDCB: World Distillation for Scalable Device Control Benchmarking 

**Authors**: Zichen Zhu, Dingye Liu, Nan Jiang, Danyang Zhang, Hang Zheng, Shijie Zheng, Situo Zhang, Kunyao Lan, Yingzi Ma, Sheng Jiang, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.18639)  

**Abstract**: Scaling device-control evaluation requires scaling both the worlds agents interact with and the tasks they must solve. We introduce SynthDCB, a scalable benchmark for environment awareness and device control built through world distillation. World distillation transforms knowledge of device behaviors, interaction constraints, and environmental effects into modular executable components, which are composed into interactive worlds with explicit states, actions, and temporal dynamics. A unified capability representation connects tool discovery, world execution, and task synthesis, allowing new capabilities to enter evaluation through a shared construction pipeline. SynthDCB instantiates Home, Car, and Phone domains with 114 device classes, 1,166 functions, and 95 events, supporting five task families with 32 subtypes. The stateful design enables task validation and agent evaluation through state correctness and tool-call traces. We evaluate seven models on 2,560 test tasks; the best achieves 68.2% Pass@1. Error analysis reveals challenges in answer delivery, feasibility diagnosis, and temporal coordination. Construction experiments demonstrate scalability across domains and token efficiency: world distillation reduces token consumption by 0.9-66.0% and 46.2-92.5% relative to Codex with low and xhigh reasoning effort, respectively, while achieving broader capability coverage. SynthDCB makes executable world construction the basis of scalable device-control benchmarking, connecting an expanding space of capabilities and interactions to verifiable agent evaluation. 

---
# GRAFT: Cross-Backbone LoRA Reuse for Continual Language and Multimodal Learning 

**Authors**: Ha Pham Thien Dinh, Ming Liu, Jiesi Guo, Matteo Vergani, Kewen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2611.18638)  

**Abstract**: In continual learning, parameter-efficient fine-tuning enables large language models to acquire new capabilities through lightweight adapters such as LoRA, yet the reuse of pretrained adapters is typically restricted to models sharing the same backbone. This limits the transfer of learned capabilities from language models to multimodal models. We propose GRAFT (Generalist Reuse of Adapters via Feature-space Transfer), a framework for cross-backbone LoRA reuse. GRAFT retrieves the most relevant pretrained LoRA adapter from a modality-aware library and maps it into the target backbone through a learned linear bridge between corresponding source and target feature spaces. We evaluate GRAFT on ten Super-NaturalInstructions text tasks, transferring Qwen2.5 LoRAs to Qwen2.5-VL, and five multimodal tasks, where Qwen2.5-VL LoRAs are directly reused. Compared with randomly initialized LoRA adapters, GRAFT improves performance on most evaluated tasks. These results show that GRAFT provides a simple and effective mechanism for reusing learned capabilities across language and multimodal models in continual learning. 

---
# Quantize, Qualify, Rerank: A Recipe for Compressing LLMs Without Losing Quality 

**Authors**: Thamme Gowda  

**Link**: [PDF](https://arxiv.org/pdf/2611.18587)  

**Abstract**: Compressing large language models lowers serving cost, but a single automatic metric may miss quality loss. For the WMT26 Model Compression constrained track, we compress google/gemma-3-12b-it (24.4 GB in bf16) for three translation directions on one H100. Paired bootstrap analysis exposes metric disagreement on 4-bit quantization; guided by WMT24 human meta-evaluation, we use MetricX24-XXL as primary and CometKiwi-XXL as a cross-check. Removing the unused vision stack and non-task vocabulary and using int8 weights or fp8 activations causes no resolved loss, whereas int4 leaves a small residual. QLoRA healing overfits its calibration domain. A Cometoid best-of-4 reranker at $T=0.3$ improves both the 8-bit and 4-bit systems under WMT25 test set. On post-submission WMT26, both remain competitive with bf16 under reference-free metrics; the small difference between the reranked variants is treated as noise. 

---
# VidEngram: Agentic Long-Term Video Understanding via Structured Hierarchical Memory 

**Authors**: Zinuo Cheng, Yueqian Lin, Zhongxi Wang, Hai "Helen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.18578)  

**Abstract**: Long-video question answering is wasteful and lossy when a multimodal model must re-encode the same video for every query: context limits force sparse sampling or truncation, and no reusable representation remains for later questions. We present VidEngram, an agentic video QA system that builds a persistent hierarchical memory once at ingest and answers later questions from that memory. VidEngram captions video and audio jointly with Qwen2.5-Omni, consolidates segment memories into episode and entity layers, and uses a text-only agentic planner with specialized retrieval tools to produce grounded answers. The web interface makes this process inspectable: users can watch memory formation, inspect retrieval traces, verify cited evidence in the video player, and explore entity relationships. On the full OmniVideoBench, VidEngram with a GPT-5-mini planner outperforms a strong single-pass Qwen2.5-Omni baseline (37.0% vs. 32.3%), with gains concentrated on longer videos where single-pass coverage becomes sparse. Because perception is paid once, subsequent queries use low-cost text-only planning. 

---
# Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws 

**Authors**: Jinbo Wang, Binghui Li, Zhanpeng Zhou, Mingze Wang, Yuxuan Sun, Jiaqi Zhang, Xunliang Cai, Lei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.14208)  

**Abstract**: Batch size scheduling (BSS) plays a critical role in large-scale deep learning training, influencing both optimization dynamics and computational efficiency. Yet, its theoretical foundations remain poorly understood. In this work, we show that the functional scaling law (FSL) framework introduced in Li et al. (2025a) provides a principled lens for analyzing BSS. Specifically, we characterize the optimal BSS under a fixed data budget and show that its structure depends sharply on task difficulty. For easy tasks, optimal schedules keep increasing batch size throughout. In contrast, for hard tasks, the optimal schedule maintains small batch sizes for most of training and switches to large batches only in a late stage. To explain the emergence of late switching, we uncover a dynamical mechanism -- the fast catch-up effect -- which also manifests in large language model (LLM) pretraining. After switching from small to large batches, the loss rapidly aligns with the constant large-batch trajectory. Using FSL, we show that this effect stems from rapid forgetting of accumulated gradient noise, with the catch-up speed determined by task difficulty. Crucially, this effect implies that large batches can be safely deferred to late training without sacrificing performance, while substantially reducing data consumption. Finally, extensive LLM pretraining experiments -- covering both Dense and MoE architectures with up to 1.1B parameters and 1T tokens -- validate our theoretical predictions. Across all settings, late-switch schedules consistently outperform constant-batch and early-switch baselines. 

---
# The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training 

**Authors**: Jinbo Wang, Mingze Wang, Zhanpeng Zhou, Junchi Yan, Weinan E, Lei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19002)  

**Abstract**: Transformers consist of diverse building blocks, such as embedding layers, normalization layers, self-attention mechanisms, and point-wise feedforward networks. Thus, understanding the differences and interactions among these blocks is important. In this paper, we uncover a clear Sharpness Disparity across these blocks, which emerges early in training and intriguingly persists throughout the training process. Motivated by this finding, we propose Blockwise Learning Rate (LR), a strategy that tailors the LR to each block's sharpness, accelerating large language model (LLM) pre-training. By integrating Blockwise LR into AdamW, we consistently achieve lower terminal loss and nearly $2\times$ speedup compared to vanilla AdamW. We demonstrate this acceleration across GPT-2 and LLaMA, with model sizes ranging from 0.12B to 2B and datasets of OpenWebText, MiniPile, and C4. Finally, we incorporate Blockwise LR into Adam-mini (Zhang et al., 2024), a recently proposed memory-efficient variant of Adam, achieving a combined $2\times$ speedup and $2\times$ memory saving. These results underscore the potential of exploiting the sharpness disparity to improve LLM training. 

---
# Using the Tsetlin Machine to Learn Human-Interpretable Rules for High-Accuracy Text Categorization with Medical Applications 

**Authors**: Geir Thore Berge, Ole-Christoffer Granmo, Tor Oddbjørn Tveit, Morten Goodwin, Lei Jiao, Bernt Viggo Matheussen  

**Link**: [PDF](https://arxiv.org/pdf/1809.04547)  

**Abstract**: Medical applications challenge today's text categorization techniques by demanding both high accuracy and ease-of-interpretation. Although deep learning has provided a leap ahead in accuracy, this leap comes at the sacrifice of interpretability. To address this accuracy-interpretability challenge, we here introduce, for the first time, a text categorization approach that leverages the recently introduced Tsetlin Machine. In all brevity, we represent the terms of a text as propositional variables. From these, we capture categories using simple propositional formulae, such as: if "rash" and "reaction" and "penicillin" then Allergy. The Tsetlin Machine learns these formulae from a labelled text, utilizing conjunctive clauses to represent the particular facets of each category. Indeed, even the absence of terms (negated features) can be used for categorization purposes. Our empirical comparison with Naïve Bayes, decision trees, linear support vector machines (SVMs), random forest, long short-term memory (LSTM) neural networks, and other techniques, is quite conclusive. The Tsetlin Machine either performs on par with or outperforms all of the evaluated methods on both the 20 Newsgroups and IMDb datasets, as well as on a non-public clinical dataset. On average, the Tsetlin Machine delivers the best recall and precision scores across the datasets. Finally, our GPU implementation of the Tsetlin Machine executes 5 to 15 times faster than the CPU implementation, depending on the dataset. We thus believe that our novel approach can have a significant impact on a wide range of text analysis applications, forming a promising starting point for deeper natural language understanding with the Tsetlin Machine. 

---
