# Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in LLM Chain-of-Thought Evaluation 

**Authors**: Richard J. Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.20172)  

**Abstract**: Recent work on chain-of-thought (CoT) faithfulness reports single aggregate numbers (e.g., DeepSeek-R1 acknowledges hints 39% of the time), implying that faithfulness is an objective, measurable property of a model. This paper demonstrates that it is not. Three classifiers (a regex-only detector, a two-stage regex-plus-LLM pipeline, and an independent Claude Sonnet 4 judge) are applied to 10,276 influenced reasoning traces from 12 open-weight models spanning 9 families and 7B to 1T parameters. On identical data, these classifiers produce overall faithfulness rates of 74.4%, 82.6%, and 69.7%, respectively, with non-overlapping 95% confidence intervals. Per-model gaps range from 2.6 to 30.6 percentage points; all are statistically significant (McNemar's test, p < 0.001). The disagreements are systematic, not random: inter-classifier agreement measured by Cohen's kappa ranges from 0.06 ("slight") for sycophancy hints to 0.42 ("moderate") for grader hints, and the asymmetry is pronounced: for sycophancy, 883 cases are classified as faithful by the pipeline but unfaithful by the Sonnet judge, while only 2 go the other direction. Classifier choice can also reverse model rankings: Qwen3.5-27B ranks 1st under the pipeline but 7th under the Sonnet judge; OLMo-3.1-32B moves in the opposite direction, from 9th to 3rd. The root cause is that different classifiers operationalize related faithfulness constructs at different levels of stringency (lexical mention versus epistemic dependence), and these constructs yield divergent measurements on the same behavior. These results demonstrate that published faithfulness numbers cannot be meaningfully compared across studies that use different classifiers, and that future evaluations should report sensitivity ranges across multiple classification methodologies rather than single point estimates. 

---
# Evaluating Evidence Grounding Under User Pressure in Instruction-Tuned Language Models 

**Authors**: Sai Koneru, Elphin Joe, Christine Kirchhoff, Jian Wu, Sarah Rajtmajer  

**Link**: [PDF](https://arxiv.org/pdf/2603.20162)  

**Abstract**: In contested domains, instruction-tuned language models must balance user-alignment pressures against faithfulness to the in-context evidence. To evaluate this tension, we introduce a controlled epistemic-conflict framework grounded in the U.S. National Climate Assessment. We conduct fine-grained ablations over evidence composition and uncertainty cues across 19 instruction-tuned models spanning 0.27B to 32B parameters. Across neutral prompts, richer evidence generally improves evidence-consistent accuracy and ordinal scoring performance. Under user pressure, however, evidence does not reliably prevent user-aligned reversals in this controlled fixed-evidence setting. We report three primary failure modes. First, we identify a negative partial-evidence interaction, where adding epistemic nuance, specifically research gaps, is associated with increased susceptibility to sycophancy in families like Llama-3 and Gemma-3. Second, robustness scales non-monotonically: within some families, certain low-to-mid scale models are especially sensitive to adversarial user pressure. Third, models differ in distributional concentration under conflict: some instruction-tuned models maintain sharply peaked ordinal distributions under pressure, while others are substantially more dispersed; in scale-matched Qwen comparisons, reasoning-distilled variants (DeepSeek-R1-Qwen) exhibit consistently higher dispersion than their instruction-tuned counterparts. These findings suggest that, in a controlled fixed-evidence setting, providing richer in-context evidence alone offers no guarantee against user pressure without explicit training for epistemic integrity. 

---
# Semantic Token Clustering for Efficient Uncertainty Quantification in Large Language Models 

**Authors**: Qi Cao, Andrew Gambardella, Takeshi Kojima, Yutaka Matsuo, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2603.20161)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks. However, the truthfulness of their outputs is not guaranteed, and their tendency toward overconfidence further limits reliability. Uncertainty quantification offers a promising way to identify potentially unreliable outputs, but most existing methods rely on repeated sampling or auxiliary models, introducing substantial computational overhead. To address these limitations, we propose Semantic Token Clustering (STC), an efficient uncertainty quantification method that leverages the semantic information inherently encoded in LLMs. Specifically, we group tokens into semantically consistent clusters using embedding clustering and prefix matching, and quantify uncertainty based on the probability mass aggregated over the corresponding semantic cluster. Our approach requires only a single generation and does not depend on auxiliary models. Experimental results show that STC achieves performance comparable to state-of-the-art baselines while substantially reducing computational overhead. 

---
# Enhancing Hyperspace Analogue to Language (HAL) Representations via Attention-Based Pooling for Text Classification 

**Authors**: Ali Sakour, Zoalfekar Sakour  

**Link**: [PDF](https://arxiv.org/pdf/2603.20149)  

**Abstract**: The Hyperspace Analogue to Language (HAL) model relies on global word co-occurrence matrices to construct distributional semantic representations. While these representations capture lexical relationships effectively, aggregating them into sentence-level embeddings via standard mean pooling often results in information loss. Mean pooling assigns equal weight to all tokens, thereby diluting the impact of contextually salient words with uninformative structural tokens. In this paper, we address this limitation by integrating a learnable, temperature-scaled additive attention mechanism into the HAL representation pipeline. To mitigate the sparsity and high dimensionality of the raw co-occurrence matrices, we apply Truncated Singular Value Decomposition (SVD) to project the vectors into a dense latent space prior to the attention layer. We evaluate the proposed architecture on the IMDB sentiment analysis dataset. Empirical results demonstrate that the attention-based pooling approach achieves a test accuracy of 82.38%, yielding an absolute improvement of 6.74 percentage points over the traditional mean pooling baseline (75.64%). Furthermore, qualitative analysis of the attention weights indicates that the mechanism successfully suppresses stop-words and selectively attends to sentiment-bearing tokens, improving both classification performance and model interpretability. 

---
# Reasoning Gets Harder for LLMs Inside A Dialogue 

**Authors**: Ivan Kartáč, Mateusz Lango, Ondřej Dušek  

**Link**: [PDF](https://arxiv.org/pdf/2603.20133)  

**Abstract**: Large Language Models (LLMs) achieve strong performance on many reasoning benchmarks, yet these evaluations typically focus on isolated tasks that differ from real-world usage in task-oriented dialogue (TOD). In this setting, LLMs must perform reasoning inherently while generating text and adhering to instructions on role, format, and style. This mismatch raises concerns about whether benchmark performance accurately reflects models' reasoning robustness in TOD setting. We investigate how framing reasoning tasks within TOD affects LLM performance by introducing BOULDER, a new dynamic benchmark covering eight travel-related tasks that require arithmetic, spatial, and temporal reasoning with both commonsense and formal aspects. Each problem is presented in both isolated and dialogue-based variants, enabling controlled comparison while mitigating data contamination. Experiments on eight LLMs reveal a substantial and consistent performance gap between isolated and dialogue settings. Through ablations and qualitative analysis, we show that this gap is largely driven by the multi-turn nature of dialogue, with additional effects from role conditioning and tool-use requirements. Our results highlight the need to evaluate LLM reasoning in realistic interactive scenarios. 

---
# Current LLMs still cannot 'talk much' about grammar modules: Evidence from syntax 

**Authors**: Mohammed Q. Shormani  

**Link**: [PDF](https://arxiv.org/pdf/2603.20114)  

**Abstract**: We aim to examine the extent to which Large Language Models (LLMs) can 'talk much' about grammar modules, providing evidence from syntax core properties translated by ChatGPT into Arabic. We collected 44 terms from generative syntax previous works, including books and journal articles, as well as from our experience in the field. These terms were translated by humans, and then by ChatGPT-5. We then analyzed and compared both translations. We used an analytical and comparative approach in our analysis. Findings unveil that LLMs still cannot 'talk much' about the core syntax properties embedded in the terms under study involving several syntactic and semantic challenges: only 25% of ChatGPT translations were accurate, while 38.6% were inaccurate, and 36.4.% were partially correct, which we consider appropriate. Based on these findings, a set of actionable strategies were proposed, the most notable of which is a close collaboration between AI specialists and linguists to better LLMs' working mechanism for accurate or at least appropriate translation. 

---
# An Empirical Study of SFT-DPO Interaction and Parameterization in Small Language Models 

**Authors**: Yuming Feng, Christy Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20100)  

**Abstract**: Direct Preference Optimization (DPO) is widely used after supervised fine-tuning (SFT) to align language models, yet empirical behavior under small backbones and modest data is under-specified. We systematically compare SFT-only, DPO-only, and staged SFT-to-DPO training alongside full fine-tuning (FFT) versus LoRA on a GPT-2-scale decoder, evaluating paraphrase detection and Shakespearean sonnet continuation. DPO yields small, task-dependent gains over strong SFT and can match competitive SFT accuracy without a warm start when the preference construction closely parallels the supervised objective. In contrast, parameterization dominates: FFT consistently outperforms LoRA at matched training depth, and LoRA does not reduce wall-clock time on our hardware. These findings indicate that, in this small-scale regime, supervised full-parameter adaptation remains the primary performance lever, while preference optimization and low-rank adaptation provide limited marginal returns. 

---
# Predicting States of Understanding in Explanatory Interactions Using Cognitive Load-Related Linguistic Cues 

**Authors**: Yu Wang, Olcay Türk, Angela Grimminger, Hendrik Buschmeier  

**Link**: [PDF](https://arxiv.org/pdf/2603.20079)  

**Abstract**: We investigate how verbal and nonverbal linguistic features, exhibited by speakers and listeners in dialogue, can contribute to predicting the listener's state of understanding in explanatory interactions on a moment-by-moment basis. Specifically, we examine three linguistic cues related to cognitive load and hypothesised to correlate with listener understanding: the information value (operationalised with surprisal) and syntactic complexity of the speaker's utterances, and the variation in the listener's interactive gaze behaviour. Based on statistical analyses of the MUNDEX corpus of face-to-face dialogic board game explanations, we find that individual cues vary with the listener's level of understanding. Listener states ('Understanding', 'Partial Understanding', 'Non-Understanding' and 'Misunderstanding') were self-annotated by the listeners using a retrospective video-recall method. The results of a subsequent classification experiment, involving two off-the-shelf classifiers and a fine-tuned German BERT-based multimodal classifier, demonstrate that prediction of these four states of understanding is generally possible and improves when the three linguistic cues are considered alongside textual features. 

---
# LoASR-Bench: Evaluating Large Speech Language Models on Low-Resource Automatic Speech Recognition Across Language Families 

**Authors**: Jianan Chen, Xiaoxue Gao, Tatsuya Kawahara, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.20042)  

**Abstract**: Large language models (LLMs) have driven substantial advances in speech language models (SpeechLMs), yielding strong performance in automatic speech recognition (ASR) under high-resource conditions. However, existing benchmarks predominantly focus on high-resource languages, leaving the ASR behavior of SpeechLMs in low-resource languages insufficiently understood. This gap is critical, as practical ASR systems must reliably support low-resource languages and generalize across diverse language families, and it directly hinders the deployment of SpeechLM-based ASR in real-world multilingual scenarios. As a result, it is essential to evaluate SpeechLMs on low-resource languages to ensure their generalizability across different language families. To address this problem, we propose \textbf{LoASR-Bench}, a comprehensive benchmark designed to evaluate \textbf{lo}w-resource \textbf{a}utomatic \textbf{s}peech \textbf{r}ecognition (\textbf{ASR}) of the latest SpeechLMs across diverse language families. LoASR-Bench comprises 25 languages from 9 language families, featuring both Latin and non-Latin scripts, enabling cross-linguistic and cross-script assessment of ASR performance of current SpeechLMs. Experimental results highlight the limitations of the latest SpeechLMs in handling real-world low-resource languages. 

---
# RouterKGQA: Specialized--General Model Routing for Constraint-Aware Knowledge Graph Question Answering 

**Authors**: Bo Yuan, Hexuan Deng, Xuebo Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20017)  

**Abstract**: Knowledge graph question answering (KGQA) is a promising approach for mitigating LLM hallucination by grounding reasoning in structured and verifiable knowledge graphs. Existing approaches fall into two paradigms: retrieval-based methods utilize small specialized models, which are efficient but often produce unreachable paths and miss implicit constraints, while agent-based methods utilize large general models, which achieve stronger structural grounding at substantially higher cost. We propose RouterKGQA, a framework for specialized--general model collaboration, in which a specialized model generates reasoning paths and a general model performs KG-guided repair only when needed, improving performance at minimal cost. We further equip the specialized with constraint-aware answer filtering, which reduces redundant answers. In addition, we design a more efficient general agent workflow, further lowering inference cost. Experimental results show that RouterKGQA outperforms the previous best by 3.57 points in F1 and 0.49 points in Hits@1 on average across benchmarks, while requiring only 1.15 average LLM calls per question. Codes and models are available at this https URL. 

---
# An Agentic Approach to Generating XAI-Narratives 

**Authors**: Yifan He, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2603.20003)  

**Abstract**: Explainable AI (XAI) research has experienced substantial growth in recent years. Existing XAI methods, however, have been criticized for being technical and expert-oriented, motivating the development of more interpretable and accessible explanations. In response, large language model (LLM)-generated XAI narratives have been proposed as a promising approach for translating post-hoc explanations into more accessible, natural-language explanations. In this work, we propose a multi-agent framework for XAI narrative generation and refinement. The framework comprises the Narrator, which generates and revises narratives based on feedback from multiple Critic Agents on faithfulness and coherence metrics, thereby enabling narrative improvement through iteration. We design five agentic systems (Basic Design, Critic Design, Critic-Rule Design, Coherent Design, and Coherent-Rule Design) and systematically evaluate their effectiveness across five LLMs on five tabular datasets. Results validate that the Basic Design, the Critic Design, and the Critic-Rule Design are effective in improving the faithfulness of narratives across all LLMs. Claude-4.5-Sonnet on Basic Design performs best, reducing the number of unfaithful narratives by 90% after three rounds of iteration. To address recurrent issues, we further introduce an ensemble strategy based on majority voting. This approach consistently enhances performance for four LLMs, except for DeepSeek-V3.2-Exp. These findings highlight the potential of agentic systems to produce faithful and coherent XAI narratives. 

---
# When Contextual Inference Fails: Cancelability in Interactive Instruction Following 

**Authors**: Natalia Bila, Kata Naszádi, Alexandra Mayn, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2603.19997)  

**Abstract**: We investigate the separation of literal interpretation from contextual inference in a collaborative block-building task where a builder must resolve underspecified instructions using contextual inferences. Building on an existing two-speaker psycholinguistic paradigm -- which contrasts a pragmatically cooperative speaker with one who is only literally reliable -- we introduce Build What I Mean (BWIM), an interactive benchmark for contextual meaning construction. In BWIM, models must resolve ambiguity by either performing a contextual inference or requesting clarification at a small communication cost. Evaluating several state-of-the-art LLMs, we find a dissociation between judgment and action: while models detect speaker unreliability in explicit confidence ratings, they fail to exploit this information to guide efficient clarification behavior. Instead, we observe suboptimal strategies, such as partner-blind over-clarification and question-averse guessing under uncertainty. 

---
# Hybrid topic modelling for computational close reading: Mapping narrative themes in Pushkin's Evgenij Onegin 

**Authors**: Angelo Maria Sabatini  

**Link**: [PDF](https://arxiv.org/pdf/2603.19940)  

**Abstract**: This study presents a hybrid topic modelling framework for computational literary analysis that integrates Latent Dirichlet Allocation (LDA) with sparse Partial Least Squares Discriminant Analysis (sPLS-DA) to model thematic structure and longitudinal dynamics in narrative poetry. As a case study, we analyse Evgenij Onegin-Aleksandr S. Pushkin's novel in verse-using an Italian translation, testing whether unsupervised and supervised lexical structures converge in a small-corpus setting. The poetic text is segmented into thirty-five documents of lemmatised content words, from which five stable and interpretable topics emerge. To address small-corpus instability, a multi-seed consensus protocol is adopted. Using sPLS-DA as a supervised probe enhances interpretability by identifying lexical markers that refine each theme. Narrative hubs-groups of contiguous stanzas marking key episodes-extend the bag-of-words approach to the narrative level, revealing how thematic mixtures align with the poem's emotional and structural arc. Rather than replacing traditional literary interpretation, the proposed framework offers a computational form of close reading, illustrating how lightweight probabilistic models can yield reproducible thematic maps of complex poetic narratives, even when stylistic features such as metre, phonology, or native morphology are abstracted away. Despite relying on a single lemmatised translation, the approach provides a transparent methodological template applicable to other high-density literary texts in comparative studies. 

---
# SAGE: Sustainable Agent-Guided Expert-tuning for Culturally Attuned Translation in Low-Resource Southeast Asia 

**Authors**: Zhixiang Lu, Chong Zhang, Yulong Li, Angelos Stefanidis, Anh Nguyen, Imran Razzak, Jionglong Su, Zhengyong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19931)  

**Abstract**: The vision of an inclusive World Wide Web is impeded by a severe linguistic divide, particularly for communities in low-resource regions of Southeast Asia. While large language models (LLMs) offer a potential solution for translation, their deployment in data-poor contexts faces a dual challenge: the scarcity of high-quality, culturally relevant data and the prohibitive energy costs of training on massive, noisy web corpora. To resolve the tension between digital inclusion and environmental sustainability, we introduce Sustainable Agent-Guided Expert-tuning (SAGE). This framework pioneers an energy-aware paradigm that prioritizes the "right data" over "big data". Instead of carbon-intensive training on unfiltered datasets, SAGE employs a reinforcement learning (RL) agent, optimized via Group Relative Policy Optimization (GRPO), to autonomously curate a compact training set. The agent utilizes a semantic reward signal derived from a small, expert-constructed set of community dialogues to filter out noise and cultural misalignment. We then efficiently fine-tune open-source LLMs on this curated data using Low-Rank Adaptation (LoRA). We applied SAGE to translation tasks between English and seven low-resource languages (LRLs) in Southeast Asia. Our approach establishes new state-of-the-art performance on BLEU-4 and COMET-22 metrics, effectively capturing local linguistic nuances. Crucially, SAGE surpasses baselines trained on full datasets while reducing data usage by 97.1% and training energy consumption by 95.2%. By delivering high-performance models with a minimal environmental footprint, SAGE offers a scalable and responsible pathway to bridge the digital divide in the Global South. 

---
# Translation from the Information Bottleneck Perspective: an Efficiency Analysis of Spatial Prepositions in Bitexts 

**Authors**: Antoine Taroni, Ludovic Moncla, Frederique Laforest  

**Link**: [PDF](https://arxiv.org/pdf/2603.19924)  

**Abstract**: Efficient communication requires balancing informativity and simplicity when encoding meanings. The Information Bottleneck (IB) framework captures this trade-off formally, predicting that natural language systems cluster near an optimal accuracy-complexity frontier. While supported in visual domains such as colour and motion, linguistic stimuli such as words in sentential context remain unexplored. We address this gap by framing translation as an IB optimisation problem, treating source sentences as stimuli and target sentences as compressed meanings. This allows IB analyses to be performed directly on bitexts rather than controlled naming experiments. We applied this to spatial prepositions across English, German and Serbian translations of a French novel. To estimate informativity, we conducted a pile-sorting pilot-study (N=35) and obtained similarity judgements of pairs of prepositions. We trained a low-rank projection model (D=5) that predicts these judgements (Spearman correlation: 0.78). Attested translations of prepositions lie closer to the IB optimal frontier than counterfactual alternatives, offering preliminary evidence that human translators exhibit communicative efficiency pressure in the spatial domain. More broadly, this work suggests that translation can serve as a window into the cognitive efficiency pressures shaping cross-linguistic semantic systems. 

---
# Span-Level Machine Translation Meta-Evaluation 

**Authors**: Stefano Perrella, Eric Morales Agostinho, Hugo Zaragoza  

**Link**: [PDF](https://arxiv.org/pdf/2603.19921)  

**Abstract**: Machine Translation (MT) and automatic MT evaluation have improved dramatically in recent years, enabling numerous novel applications. Automatic evaluation techniques have evolved from producing scalar quality scores to precisely locating translation errors and assigning them error categories and severity levels. However, it remains unclear how to reliably measure the evaluation capabilities of auto-evaluators that do error detection, as no established technique exists in the literature. This work investigates different implementations of span-level precision, recall, and F-score, showing that seemingly similar approaches can yield substantially different rankings, and that certain widely-used techniques are unsuitable for evaluating MT error detection. We propose "match with partial overlap and partial credit" (MPP) with micro-averaging as a robust meta-evaluation strategy and release code for its use publicly. Finally, we use MPP to assess the state of the art in MT error detection. 

---
# Semantic Delta: An Interpretable Signal Differentiating Human and LLMs Dialogue 

**Authors**: Riccardo Scantamburlo, Mauro Mezzanzana, Giacomo Buonanno, Francesco Bertolotti  

**Link**: [PDF](https://arxiv.org/pdf/2603.19849)  

**Abstract**: Do LLMs talk like us? This question intrigues a multitude of scholar and it is relevant in many fields, from education to academia. This work presents an interpretable statistical feature for distinguishing human written and LLMs generated dialogue. We introduce a lightweight metric derived from semantic categories distribution. Using the Empath lexical analysis framework, each text is mapped to a set of thematic intensity scores. We define semantic delta as the difference between the two most dominant category intensities within a dialogue, hypothesizing that LLM outputs exhibit stronger thematic concentration than human discourse. To evaluate this hypothesis, conversational data were generated from multiple LLM configurations and compared against heterogeneous human corpora, including scripted dialogue, literary works, and online discussions. A Welch t-test was applied to the resulting distributions of semantic delta values. Results show that AI-generated texts consistently produce higher deltas than human texts, indicating a more rigid topics structure, whereas human dialogue displays a broader and more balanced semantic spread. Rather than replacing existing detection techniques, the proposed zero-shot metric provides a computationally inexpensive complementary signal that can be integrated into ensemble detection systems. These finding also contribute to the broader empirical understanding of LLM behavioural mimicry and suggest that thematic distribution constitutes a quantifiable dimension along which current models fall short of human conversational dynamics. 

---
# FrameNet Semantic Role Classification by Analogy 

**Authors**: Van-Duy Ngo, Stergos Afantenos, Emiliano Lorini, Miguel Couceiro  

**Link**: [PDF](https://arxiv.org/pdf/2603.19825)  

**Abstract**: In this paper, we adopt a relational view of analogies applied to Semantic Role Classification in FrameNet. We define analogies as formal relations over the Cartesian product of frame evoking lexical units (LUs) and frame element (FEs) pairs, which we use to construct a new dataset. Each element of this binary relation is labelled as a valid analogical instance if the frame elements share the same semantic role, or as invalid otherwise. This formulation allows us to transform Semantic Role Classification into binary classification and train a lightweight Artificial Neural Network (ANN) that exhibits rapid convergence with minimal parameters. Unconventionally, no Semantic Role information is introduced to the neural network during training. We recover semantic roles during inference by computing probability distributions over candidates of all semantic roles within a given frame through random sampling and analogical transfer. This approach allows us to surpass previous state-of-the-art results while maintaining computational efficiency and frugality. 

---
# Neither Here Nor There: Cross-Lingual Representation Dynamics of Code-Mixed Text in Multilingual Encoders 

**Authors**: Debajyoti Mazumder, Divyansh Pathak, Prashant Kodali, Jasabanta Patro  

**Link**: [PDF](https://arxiv.org/pdf/2603.19771)  

**Abstract**: Multilingual encoder-based language models are widely adopted for code-mixed analysis tasks, yet we know surprisingly little about how they represent code-mixed inputs internally - or whether those representations meaningfully connect to the constituent languages being mixed. Using Hindi-English as a case study, we construct a unified trilingual corpus of parallel English, Hindi (Devanagari), and Romanized code-mixed sentences, and probe cross-lingual representation alignment across standard multilingual encoders and their code-mixed adapted variants via CKA, token-level saliency, and entropy-based uncertainty analysis. We find that while standard models align English and Hindi well, code-mixed inputs remain loosely connected to either language - and that continued pre-training on code-mixed data improves English-code-mixed alignment at the cost of English-Hindi alignment. Interpretability analyses further reveal a clear asymmetry: models process code-mixed text through an English-dominant semantic subspace, while native-script Hindi provides complementary signals that reduce representational uncertainty. Motivated by these findings, we introduce a trilingual post-training alignment objective that brings code-mixed representations closer to both constituent languages simultaneously, yielding more balanced cross-lingual alignment and downstream gains on sentiment analysis and hate speech detection - showing that grounding code-mixed representations in their constituent languages meaningfully helps cross-lingual understanding. 

---
# Rethinking Ground Truth: A Case Study on Human Label Variation in MLLM Benchmarking 

**Authors**: Tomas Ruiz, Tanalp Agustoslu, Carsten Schwemmer  

**Link**: [PDF](https://arxiv.org/pdf/2603.19744)  

**Abstract**: Human Label Variation (HLV), i.e. systematic differences among annotators' judgments, remains underexplored in benchmarks despite rapid progress in large language model (LLM) development. We address this gap by introducing an evaluation protocol for multimodal large language model (MLLM) benchmarking that explicitly accounts for two conditions: (1) human label agreement and (2) disagreement. We apply this protocol to two state-of-the-art MLLM families (Gemma 3, Qwen 2.5 VL) using non-aggregated human annotations from a social media content classification dataset. Across tasks, we find that larger models tend to perform best on high-agreement subsets, yet often underperform medium-sized models when human disagreement is high, indicating that parameter count alone does not determine sensitivity to ambiguity and subjectivity. These results show that benchmarks based solely on consensus labels can overstate model capabilities in such domains and that incorporating human label variation yields more realistic and robust assessments of MLLMs in content moderation pipelines. 

---
# PoC: Performance-oriented Context Compression for Large Language Models via Performance Prediction 

**Authors**: Runsong Zhao, Shilei Liu, Jiwei Tang, Langming Liu, Haibin Chen, Weidong Zhang, Yujin Yuan, Tong Xiao, Jingbo Zhu, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.19733)  

**Abstract**: While context compression can mitigate the growing inference costs of Large Language Models (LLMs) by shortening contexts, existing methods that specify a target compression ratio or length suffer from unpredictable performance degradation, hindering their reliable deployment. We introduce a paradigm shift to Performance-oriented Context Compression (PoC), where developers specify an acceptable performance floor instead of a compression ratio. PoC employs a lightweight performance predictor to automatically find the most aggressive compression ratio that satisfies this constraint before steering an off-the-shelf compressor. We design and compare two predictor variants: a simple context-agnostic predictor and a more sophisticated context-aware one that considers the input's inherent compressibility. On both question-answering and summarization benchmarks, the context-aware predictor consistently achieves lower performance prediction error than the context-agnostic predictor, while the resulting context-aware PoC attains a superior overall performance. Our work paves the way for a more reliable, efficient, and performance-aware deployment of context compression for LLMs. 

---
# LoopRPT: Reinforcement Pre-Training for Looped Language Models 

**Authors**: Guo Tang, Shixin Jiang, Heng Chang, Nuo Chen, Yuhan Li, Huiming Fan, Jia Li, Ming Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2603.19714)  

**Abstract**: Looped language models (LoopLMs) perform iterative latent computation to refine internal representations, offering a promising alternative to explicit chain-of-thought (CoT) reasoning. However, existing reinforcement learning (RL) paradigms primarily target output tokens, creating a structural mismatch with looped architectures whose reasoning unfolds implicitly. In this work, we propose LoopRPT, a reinforcement pre-training framework tailored for LoopLMs. By reframing next-token prediction as a next-token reasoning task, LoopRPT assigns reinforcement signals directly to latent steps using an EMA teacher reference and noisy latent rollouts. This formulation enables RL to directly shape intermediate representations, compressing effective reasoning into fewer iterations. We instantiate LoopRPT on the Ouro architecture across multiple model scales. Results demonstrate that LoopRPT consistently improves per-step representation quality, achieving Pareto dominance in accuracy-computation trade-offs. Notably, significant gains on hard tokens indicate that LoopRPT enhances early-stage reasoning rather than merely encouraging premature exits. Our findings highlight reinforcement pre-training as a principled paradigm for learning efficient latent reasoning in LoopLMs. 

---
# TAB-AUDIT: Detecting AI-Fabricated Scientific Tables via Multi-View Likelihood Mismatch 

**Authors**: Shuo Huang, Yan Pen, Lizhen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19712)  

**Abstract**: AI-generated fabricated scientific manuscripts raise growing concerns with large-scale breaches of academic integrity. In this work, we present the first systematic study on detecting AI-generated fabricated scientific tables in empirical NLP papers, as information in tables serve as critical evidence for claims. We construct FabTab, the first benchmark dataset of fabricated manuscripts with tables, comprising 1,173 AI-generated papers and 1,215 human-authored ones in empirical NLP. Through a comprehensive analysis, we identify systematic differences between fabricated and real tables and operationalize them into a set of discriminative features within the TAB-AUDIT framework. The key feature, within-table mismatch, captures the perplexity gap between a table's skeleton and its numerical content. Experimental results show that RandomForest built on these features significantly outperform prior state-of-the-art methods, achieving 0.987 AUROC in-domain and 0.883 AUROC out-of-domain. Our findings highlight experimental tables as a critical forensic signal for detecting AI-generated scientific fraud and provide a new benchmark for future research. 

---
# EvoTaxo: Building and Evolving Taxonomy from Social Media Streams 

**Authors**: Yiyang Li, Tianyi Ma, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.19711)  

**Abstract**: Constructing taxonomies from social media corpora is challenging because posts are short, noisy, semantically entangled, and temporally dynamic. Existing taxonomy induction methods are largely designed for static corpora and often struggle to balance robustness, scalability, and sensitivity to evolving discourse. We propose EvoTaxo, a LLM-based framework for building and evolving taxonomies from temporally ordered social media streams. Rather than clustering raw posts directly, EvoTaxo converts each post into a structured draft action over the current taxonomy, accumulates structural evidence over time windows, and consolidates candidate edits through dual-view clustering that combines semantic similarity with temporal locality. A refinement-and-arbitration procedure then selects reliable edits before execution, while each node maintains a concept memory bank to preserve semantic boundaries over time. Experiments on two Reddit corpora show that EvoTaxo produces more balanced taxonomies than baselines, with clearer post-to-leaf assignment, better corpus coverage at comparable taxonomy size, and stronger structural quality. A case study on the Reddit community /r/ICE_Raids further shows that EvoTaxo captures meaningful temporal shifts in discourse. Our codebase is available here. 

---
# DataProphet: Demystifying Supervision Data Generalization in Multimodal LLMs 

**Authors**: Xuan Qi, Luxi He, Dan Roth, Xingyu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19688)  

**Abstract**: Conventional wisdom for selecting supervision data for multimodal large language models (MLLMs) is to prioritize datasets that appear similar to the target benchmark, such as text-intensive or vision-centric tasks. However, it remains unclear whether such intuitive similarity reliably predicts downstream performance gains. In this work, we take a first step toward answering a practical question: can we estimate the influence of a training dataset on a target benchmark before any training is performed? To investigate this question, we conduct an in-depth analysis of transfer across 14 vision-language datasets spanning 7 diverse tasks. Our results show that intuitive task similarity is an unreliable predictor of transferability, and that generalization depends more on the specific dataset than on its broad task category. Motivated by this finding, we propose DATAPROPHET, a simple and effective training-free metric that combines multimodal perplexity, similarity, and data diversity. Experiments show that DATAPROPHET produces supervision-data rankings that strongly correlate with rankings based on actual post-training performance gains, achieving a Kendall's tau of 86.0%. Moreover, DATAPROPHET enables better supervision-data selection, yielding up to 6.9% improvement over uniform selection, 1.4% over a state-of-the-art training-based baseline, and 0.2% above oracle selection based on experimental performance. Our code and data will be released. 

---
# Structured Prompting for Arabic Essay Proficiency: A Trait-Centric Evaluation Approach 

**Authors**: Salim Al Mandhari, Hieu Pham Dinh, Mo El-Haj, Paul Rayson  

**Link**: [PDF](https://arxiv.org/pdf/2603.19668)  

**Abstract**: This paper presents a novel prompt engineering framework for trait specific Automatic Essay Scoring (AES) in Arabic, leveraging large language models (LLMs) under zero-shot and few-shot configurations. Addressing the scarcity of scalable, linguistically informed AES tools for Arabic, we introduce a three-tier prompting strategy (standard, hybrid, and rubric-guided) that guides LLMs in evaluating distinct language proficiency traits such as organization, vocabulary, development, and style. The hybrid approach simulates multi-agent evaluation with trait specialist raters, while the rubric-guided method incorporates scored exemplars to enhance model alignment. In zero and few-shot settings, we evaluate eight LLMs on the QAES dataset, the first publicly available Arabic AES resource with trait level annotations. Experimental results using Quadratic Weighted Kappa (QWK) and Confidence Intervals show that Fanar-1-9B-Instruct achieves the highest trait level agreement in both zero and few-shot prompting (QWK = 0.28 and CI = 0.41), with rubric-guided prompting yielding consistent gains across all traits and models. Discourse-level traits such as Development and Style showed the greatest improvements. These findings confirm that structured prompting, not model scale alone, enables effective AES in Arabic. Our study presents the first comprehensive framework for proficiency oriented Arabic AES and sets the foundation for scalable assessment in low resource educational contexts. 

---
# BEAVER: A Training-Free Hierarchical Prompt Compression Method via Structure-Aware Page Selection 

**Authors**: Zhengpei Hu, Kai Li, Dapeng Fu, Chang Zeng, Yue Li, Yuanhao Tang, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19635)  

**Abstract**: The exponential expansion of context windows in LLMs has unlocked capabilities for long-document understanding but introduced severe bottlenecks in inference latency and information utilization. Existing compression methods often suffer from high training costs or semantic fragmentation due to aggressive token pruning. In this paper, we propose BEAVER, a novel training-free framework that shifts compression from linear token removal to structure-aware hierarchical selection. BEAVER maximizes hardware parallelism by mapping variable-length contexts into dense page-level tensors via dual-path pooling, and preserves discourse integrity through a hybrid planner combining semantic and lexical dual-branch selection with sentence smoothing. Extensive evaluations on four long-context benchmarks demonstrate that BEAVER achieves comparable performance to state-of-the-art (SOTA) methods like LongLLMLingua. Notably, on the RULER benchmark, BEAVER maintains high fidelity in multi-needle retrieval where baselines deteriorate. Regarding efficiency, BEAVER reduces latency by 26.4x on 128k contexts, offering a scalable solution for high-throughput applications. Our code is available at this https URL. 

---
# TextReasoningBench: Does Reasoning Really Improve Text Classification in Large Language Models? 

**Authors**: Xinyu Guo, Yazhou Zhang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2603.19558)  

**Abstract**: Eliciting explicit, step-by-step reasoning traces from large language models (LLMs) has emerged as a dominant paradigm for enhancing model capabilities. Although such reasoning strategies were originally designed for problems requiring explicit multi-step reasoning, they have increasingly been applied to a broad range of NLP tasks. This expansion implicitly assumes that deliberative reasoning uniformly benefits heterogeneous tasks. However, whether such reasoning mechanisms truly benefit classification tasks remains largely underexplored, especially considering their substantial token and time costs. To fill this gap, we introduce TextReasoningBench, a systematic benchmark designed to evaluate the effectiveness and efficiency of reasoning strategies for text classification with LLMs. We compare seven reasoning strategies, namely IO, CoT, SC-CoT, ToT, GoT, BoC, and long-CoT across ten LLMs on five text classification datasets. Beyond traditional metrics such as accuracy and macro-F1, we introduce two cost-aware evaluation metrics that quantify the performance gain per reasoning token and the efficiency of performance improvement relative to token cost growth. Experimental results reveal three notable findings: (1) Reasoning does not universally improve classification performance: while moderate strategies such as CoT and SC-CoT yield consistent but limited gains (typically +1% to +3% on big models), more complex methods (e.g., ToT and GoT) often fail to outperform simpler baselines and can even degrade performance, especially on small models; (2) Reasoning is often inefficient: many reasoning strategies increase token consumption by 10$\times$ to 100$\times$ (e.g., SC-CoT and ToT) while providing only marginal performance improvements. 

---
# FDARxBench: Benchmarking Regulatory and Clinical Reasoning on FDA Generic Drug Assessment 

**Authors**: Betty Xiong, Jillian Fisher, Benjamin Newman, Meng Hu, Shivangi Gupta, Yejin Choi, Lanyan Fang, Russ B Altman  

**Link**: [PDF](https://arxiv.org/pdf/2603.19539)  

**Abstract**: We introduce an expert curated, real-world benchmark for evaluating document-grounded question-answering (QA) motivated by generic drug assessment, using the U.S. Food and Drug Administration (FDA) drug label documents. Drug labels contain rich but heterogeneous clinical and regulatory information, making accurate question answering difficult for current language models. In collaboration with FDA regulatory assessors, we introduce FDARxBench, and construct a multi-stage pipeline for generating high-quality, expert curated, QA examples spanning factual, multi-hop, and refusal tasks, and design evaluation protocols to assess both open-book and closed-book reasoning. Experiments across proprietary and open-weight models reveal substantial gaps in factual grounding, long-context retrieval, and safe refusal behavior. While motivated by FDA generic drug assessment needs, this benchmark also provides a substantial foundation for challenging regulatory-grade evaluation of label comprehension. The benchmark is designed to support evaluation of LLM behavior on drug-label questions. 

---
# EvidenceRL: Reinforcing Evidence Consistency for Trustworthy Language Models 

**Authors**: J. Ben Tamo, Yuxing Lu, Benoit L. Marteau, Micky C. Nnamdi, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19532)  

**Abstract**: Large Language Models (LLMs) are fluent but prone to hallucinations, producing answers that appear plausible yet are unsupported by available evidence. This failure is especially problematic in high-stakes domains where decisions must be justified by verifiable information. We introduce \textbf{EvidenceRL}, a reinforcement learning framework that enforces evidence adherence during training. EvidenceRL scores candidate responses for grounding (entailment with retrieved evidence and context) and correctness (agreement with reference answers) and optimizes the generator using Group Relative Policy Optimization (GRPO). We evaluate across two high-stakes domains, cardiac diagnosis and legal reasoning, where EvidenceRL consistently improves evidence grounding and faithfulness without sacrificing task accuracy. On cardiac diagnosis, F1@3 increases from 37.0 to 54.5 on Llama-3.2-3B while grounding ($G_{\max}@3$) rises from 47.6 to 78.2; hallucinations drop nearly 5$\times$ and evidence-supported diagnoses increase from 31.8\% to 61.6\%. On legal reasoning, EvidenceRL raises Faithfulness from 32.8\% to 67.6\% on Llama-3.1-8B, demonstrating consistent behavioral change across domains. Our code is open-sourced at this https URL. 

---
# Inducing Sustained Creativity and Diversity in Large Language Models 

**Authors**: Queenie Luo, Gary King, Michael Puett, Michael D. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2603.19519)  

**Abstract**: We address a not-widely-recognized subset of exploratory search, where a user sets out on a typically long "search quest" for the perfect wedding dress, overlooked research topic, killer company idea, etc. The first few outputs of current large language models (LLMs) may be helpful but only as a start, since the quest requires learning the search space and evaluating many diverse and creative alternatives along the way. Although LLMs encode an impressive fraction of the world's knowledge, common decoding methods are narrowly optimized for prompts with correct answers and thus return mostly homogeneous and conventional results. Other approaches, including those designed to increase diversity across a small set of answers, start to repeat themselves long before search quest users learn enough to make final choices, or offer a uniform type of "creativity" to every user asking similar questions. We develop a novel, easy-to-implement decoding scheme that induces sustained creativity and diversity in LLMs, producing as many conceptually unique results as desired, even without access to the inner workings of an LLM's vector space. The algorithm unlocks an LLM's vast knowledge, both orthodox and heterodox, well beyond modal decoding paths. With this approach, search quest users can more quickly explore the search space and find satisfying answers. 

---
# Cooperation and Exploitation in LLM Policy Synthesis for Sequential Social Dilemmas 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2603.19453)  

**Abstract**: We study LLM policy synthesis: using a large language model to iteratively generate programmatic agent policies for multi-agent environments. Rather than training neural policies via reinforcement learning, our framework prompts an LLM to produce Python policy functions, evaluates them in self-play, and refines them using performance feedback across iterations. We investigate feedback engineering (the design of what evaluation information is shown to the LLM during refinement) comparing sparse feedback (scalar reward only) against dense feedback (reward plus social metrics: efficiency, equality, sustainability, peace). Across two canonical Sequential Social Dilemmas (Gathering and Cleanup) and two frontier LLMs (Claude Sonnet 4.6, Gemini 3.1 Pro), dense feedback consistently matches or exceeds sparse feedback on all metrics. The advantage is largest in the Cleanup public goods game, where providing social metrics helps the LLM calibrate the costly cleaning-harvesting tradeoff. Rather than triggering over-optimization of fairness, social metrics serve as a coordination signal that guides the LLM toward more effective cooperative strategies, including territory partitioning, adaptive role assignment, and the avoidance of wasteful aggression. We further perform an adversarial experiment to determine whether LLMs can reward hack these environments. We characterize five attack classes and discuss mitigations, highlighting an inherent tension in LLM policy synthesis between expressiveness and safety.
Code at this https URL. 

---
# Vocabulary shapes cross-lingual variation of word-order learnability in language models 

**Authors**: Jonas Mayer Martins, Jaap Jumelet, Viola Priesemann, Lisa Beinborn  

**Link**: [PDF](https://arxiv.org/pdf/2603.19427)  

**Abstract**: Why do some languages like Czech permit free word order, while others like English do not? We address this question by pretraining transformer language models on a spectrum of synthetic word-order variants of natural languages. We observe that greater word-order irregularity consistently raises model surprisal, indicating reduced learnability. Sentence reversal, however, affects learnability only weakly. A coarse distinction of free- (e.g., Czech and Finnish) and fixed-word-order languages (e.g., English and French) does not explain cross-lingual variation. Instead, the structure of the word and subword vocabulary strongly predicts the model surprisal. Overall, vocabulary structure emerges as a key driver of computational word-order learnability across languages. 

---
# Is Evaluation Awareness Just Format Sensitivity? Limitations of Probe-Based Evidence under Controlled Prompt Structure 

**Authors**: Viliana Devbunova  

**Link**: [PDF](https://arxiv.org/pdf/2603.19426)  

**Abstract**: Prior work uses linear probes on benchmark prompts as evidence of evaluation awareness in large language models. Because evaluation context is typically entangled with benchmark format and genre, it is unclear whether probe-based signals reflect context or surface structure. We test whether these signals persist under partial control of prompt format using a controlled 2x2 dataset and diagnostic rewrites. We find that probes primarily track benchmark-canonical structure and fail to generalize to free-form prompts independent of linguistic style. Thus, standard probe-based methodologies do not reliably disentangle evaluation context from structural artifacts, limiting the evidential strength of existing results. 

---
# Scalable Prompt Routing via Fine-Grained Latent Task Discovery 

**Authors**: Yunyi Zhang, Soji Adeshina, Patrick Guan, Ashwin Ganesh, Zhen Han, Vassilis N. Ioannidis, Huzefa Rangwala, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2603.19415)  

**Abstract**: Prompt routing dynamically selects the most appropriate large language model from a pool of candidates for each query, optimizing performance while managing costs. As model pools scale to include dozens of frontier models with narrow performance gaps, existing approaches face significant challenges: manually defined task taxonomies cannot capture fine-grained capability distinctions, while monolithic routers struggle to differentiate subtle differences across diverse tasks. We propose a two-stage routing architecture that addresses these limitations through automated fine-grained task discovery and task-aware quality estimation. Our first stage employs graph-based clustering to discover latent task types and trains a classifier to assign prompts to discovered tasks. The second stage uses a mixture-of-experts architecture with task-specific prediction heads for specialized quality estimates. At inference, we aggregate predictions from both stages to balance task-level stability with prompt-specific adaptability. Evaluated on 10 benchmarks with 11 frontier models, our method consistently outperforms existing baselines and surpasses the strongest individual model while incurring less than half its cost. 

---
# Prompt-tuning with Attribute Guidance for Low-resource Entity Matching 

**Authors**: Lihui Liu, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19321)  

**Abstract**: Entity Matching (EM) is an important task that determines the logical relationship between two entities, such as Same, Different, or Undecidable. Traditional EM approaches rely heavily on supervised learning, which requires large amounts of high-quality labeled data. This labeling process is both time-consuming and costly, limiting practical applicability. As a result, there is a strong need for low-resource EM methods that can perform well with minimal labeled data. Recent prompt-tuning approaches have shown promise for low-resource EM, but they mainly focus on entity-level matching and often overlook critical attribute-level information. In addition, these methods typically lack interpretability and explainability. To address these limitations, this paper introduces PROMPTATTRIB, a comprehensive solution that tackles EM through attribute-level prompt tuning and logical reasoning. PROMPTATTRIB uses both entity-level and attribute-level prompts to incorporate richer contextual information and employs fuzzy logic formulas to infer the final matching label. By explicitly considering attributes, the model gains a deeper understanding of the entities, resulting in more accurate matching. Furthermore, PROMPTATTRIB integrates dropout-based contrastive learning on soft prompts, inspired by SimCSE, which further boosts EM performance. Extensive experiments on real-world datasets demonstrate the effectiveness of PROMPTATTRIB. 

---
# Memory-Driven Role-Playing: Evaluation and Enhancement of Persona Knowledge Utilization in LLMs 

**Authors**: Kai Wang, Haoyang You, Yang Zhang, Zhongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19313)  

**Abstract**: A core challenge for faithful LLM role-playing is sustaining consistent characterization throughout long, open-ended dialogues, as models frequently fail to recall and accurately apply their designated persona knowledge without explicit cues. To tackle this, we propose the Memory-Driven Role-Playing paradigm. Inspired by Stanislavski's "emotional memory" acting theory, this paradigm frames persona knowledge as the LLM's internal memory store, requiring retrieval and application based solely on dialogue context, thereby providing a rigorous test of depth and autonomous use of knowledge. Centered on this paradigm, we contribute: (1) MREval, a fine-grained evaluation framework assessing four memory-driven abilities - Anchoring, Recalling, Bounding, and Enacting; (2) MRPrompt, a prompting architecture that guides structured memory retrieval and response generation; and (3) MRBench, a bilingual (Chinese/English) benchmark for fine-grained diagnosis. The novel paradigm provides a comprehensive diagnostic for four-staged role-playing abilities across 12 LLMs. Crucially, experiments show that MRPrompt allows small models (e.g., Qwen3-8B) to match the performance of much larger closed-source LLMs (e.g., Qwen3-Max and GLM-4.7), and confirms that upstream memory gains directly enhance downstream response quality, validating the staged theoretical foundation. 

---
# PrefPO: Pairwise Preference Prompt Optimization 

**Authors**: Rahul Singhal, Pradyumna Tambwekar, Karime Maamari  

**Link**: [PDF](https://arxiv.org/pdf/2603.19311)  

**Abstract**: Prompt engineering is effective but labor-intensive, motivating automated optimization methods. Existing methods typically require labeled datasets, which are often unavailable, and produce verbose, repetitive prompts. We introduce PrefPO, a minimal prompt optimization approach inspired by reinforcement learning from human feedback (RLHF). Its preference-based approach reduces the need for labeled data and hyperparameter tuning-only a starting prompt and natural language criteria are needed. PrefPO uses an LLM discriminator to express pairwise preferences over model outputs and provide feedback to an LLM optimizer, iteratively improving performance. We evaluate PrefPO on 9 BIG-Bench Hard (BBH) tasks and IFEval-Hard, a newly-curated, challenging subset of IFEval. PrefPO matches or exceeds SOTA methods, including GEPA, MIPRO, and TextGrad, on 6/9 tasks and performs comparably to TextGrad on IFEval-Hard (82.4% vs 84.5%). Unlike other methods, PrefPO can optimize in both labeled and unlabeled settings. Without labels, PrefPO closely matches its labeled performance on 6/9 tasks, proving effective without ground truth. PrefPO also improves prompt hygiene: we find existing methods produce prompts 14.7x their original length or with 34% repetitive content; PrefPO reduces these issues by 3-5x. Furthermore, both LLM and human judges rate PrefPO's prompts higher than TextGrad's. Finally, we identify prompt hacking in prompt optimizers, where methods game evaluation criteria, and find PrefPO is susceptible at half the rate of TextGrad (37% vs 86%), generating fewer brittle, misaligned prompts. 

---
# LLM-MRD: LLM-Guided Multi-View Reasoning Distillation for Fake News Detection 

**Authors**: Weilin Zhou, Shanwen Tan, Enhao Gu, Yurong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2603.19293)  

**Abstract**: Multimodal fake news detection is crucial for mitigating societal disinformation. Existing approaches attempt to address this by fusing multimodal features or leveraging Large Language Models (LLMs) for advanced reasoning. However, these methods suffer from serious limitations, including a lack of comprehensive multi-view judgment and fusion, and prohibitive reasoning inefficiency due to the high computational costs of LLMs. To address these issues, we propose \textbf{LLM}-Guided \textbf{M}ulti-View \textbf{R}easoning \textbf{D}istillation for Fake News Detection ( \textbf{LLM-MRD}), a novel teacher-student framework. The Student Multi-view Reasoning module first constructs a comprehensive foundation from textual, visual, and cross-modal perspectives. Then, the Teacher Multi-view Reasoning module generates deep reasoning chains as rich supervision signals. Our core Calibration Distillation mechanism efficiently distills this complex reasoning-derived knowledge into the efficient student model. Experiments show LLM-MRD significantly outperforms state-of-the-art baselines. Notably, it demonstrates a comprehensive average improvement of 5.19\% in ACC and 6.33\% in F1-Fake when evaluated across all competing methods and datasets. Our code is available at this https URL 

---
# Automatic Analysis of Collaboration Through Human Conversational Data Resources: A Review 

**Authors**: Yi Yu, Maria Boritchev, Chloé Clavel  

**Link**: [PDF](https://arxiv.org/pdf/2603.19292)  

**Abstract**: Collaboration is a task-oriented, high-level human behavior. In most cases, conversation serves as the primary medium for information exchange and coordination, making conversational data a valuable resource for the automatic analysis of collaborative processes. In this paper, we focus on verbal aspects of collaboration and conduct a review of collaboration analysis using task-oriented conversation resources, encompassing related theories, coding schemes, tasks, and modeling approaches. We aim to address the question of how to utilize task-oriented human-human conversational data for collaboration analysis. We hope our review will serve as a practical resource and illuminate unexplored areas for future collaboration analysis. 

---
# Automated Motif Indexing on the Arabian Nights 

**Authors**: Ibrahim H. Alyami, Mark A. Finlayson  

**Link**: [PDF](https://arxiv.org/pdf/2603.19283)  

**Abstract**: Motifs are non-commonplace, recurring narrative elements, often found originally in folk stories. In addition to being of interest to folklorists, motifs appear as metaphoric devices in modern news, literature, propaganda, and other cultural texts. Finding expressions of motifs in the original folkloristic text is useful for both folkloristic analysis (motif indexing) as well as for understanding the modern usage of motifs (motif detection and interpretation). Prior work has primarily shown how difficult these problems are to tackle using automated techniques. We present the first computational approach to motif indexing. Our choice of data is a key enabler: we use a large, widely available text (the Arabian Nights) paired with a detailed motif index (by El-Shamy in 2006), which overcomes the common problem of inaccessibility of texts referred to by the index. We created a manually annotated corpus that identified 2,670 motif expressions of 200 different motifs across 58,450 sentences for training and testing. We tested five types of approaches for detecting motif expressions given a motif index entry: (1) classic retrieve and re-rank using keywords and a fine-tuned cross-encoder; (2) off-the-shelf embedding models; (3) fine-tuned embedding models; (4) generative prompting of off-the-shelf LLMs in N-shot setups; and (5) the same generative approaches on LLMs fine-tuned with LoRA. Our best performing system is a fine-tuned Llama3 model which achieves an overall performance of 0.85 F1. 

---
# Framing Effects in Independent-Agent Large Language Models: A Cross-Family Behavioral Analysis 

**Authors**: Zice Wang, Zhenyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19282)  

**Abstract**: In many real-world applications, large language models (LLMs) operate as independent agents without interaction, thereby limiting coordination. In this setting, we examine how prompt framing influences decisions in a threshold voting task involving individual-group interest conflict. Two logically equivalent prompts with different framings were tested across diverse LLM families under isolated trials. Results show that prompt framing significantly influences choice distributions, often shifting preferences toward risk-averse options. Surface linguistic cues can even override logically equivalent formulations. This suggests that observed behavior reflects a tendency consistent with a preference for instrumental rather than cooperative rationality when success requires risk-bearing. The findings highlight framing effects as a significant bias source in non-interacting multi-agent LLM deployments, informing alignment and prompt design. 

---
# URAG: A Benchmark for Uncertainty Quantification in Retrieval-Augmented Large Language Models 

**Authors**: Vinh Nguyen, Cuong Dang, Jiahao Zhang, Hoa Tran, Minh Tran, Trinh Chau, Thai Le, Lu Cheng, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19281)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a widely adopted approach for enhancing LLMs in scenarios that demand extensive factual knowledge. However, current RAG evaluations concentrate primarily on correctness, which may not fully capture the impact of retrieval on LLM uncertainty and reliability. To bridge this gap, we introduce URAG, a comprehensive benchmark designed to assess the uncertainty of RAG systems across various fields like healthcare, programming, science, math, and general text. By reformulating open-ended generation tasks into multiple-choice question answering, URAG allows for principled uncertainty quantification via conformal prediction. We apply the evaluation pipeline to 8 standard RAG methods, measuring their performance through both accuracy and prediction-set sizes based on LAC and APS metrics. Our analysis shows that (1) accuracy gains often coincide with reduced uncertainty, but this relationship breaks under retrieval noise; (2) simple modular RAG methods tend to offer better accuracy-uncertainty trade-offs than more complex reasoning pipelines; and (3) no single RAG approach is universally reliable across domains. We further show that (4) retrieval depth, parametric knowledge dependence, and exposure to confidence cues can amplify confident errors and hallucinations. Ultimately, URAG establishes a systematic benchmark for analyzing and enhancing the trustworthiness of retrieval-augmented systems. Our code is available on GitHub. 

---
# From Feature-Based Models to Generative AI: Validity Evidence for Constructed Response Scoring 

**Authors**: Jodi M. Casabianca, Daniel F. McCaffrey, Matthew S. Johnson, Naim Alper, Vladimir Zubenko  

**Link**: [PDF](https://arxiv.org/pdf/2603.19280)  

**Abstract**: The rapid advancements in large language models and generative artificial intelligence (AI) capabilities are making their broad application in the high-stakes testing context more likely. Use of generative AI in the scoring of constructed responses is particularly appealing because it reduces the effort required for handcrafting features in traditional AI scoring and might even outperform those methods. The purpose of this paper is to highlight the differences in the feature-based and generative AI applications in constructed response scoring systems and propose a set of best practices for the collection of validity evidence to support the use and interpretation of constructed response scores from scoring systems using generative AI. We compare the validity evidence needed in scoring systems using human ratings, feature-based natural language processing AI scoring engines, and generative AI. The evidence needed in the generative AI context is more extensive than in the feature-based scoring context because of the lack of transparency and other concerns unique to generative AI such as consistency. Constructed response score data from a large corpus of independent argumentative essays written by 6-12th grade students demonstrate the collection of validity evidence for different types of scoring systems and highlight the numerous complexities and considerations when making a validity argument for these scores. 

---
# Multilingual Hate Speech Detection and Counterspeech Generation: A Comprehensive Survey and Practical Guide 

**Authors**: Zahra Safdari Fesaghandis, Suman Kalyan Maity  

**Link**: [PDF](https://arxiv.org/pdf/2603.19279)  

**Abstract**: Combating online hate speech in multilingual settings requires approaches that go beyond English-centric models and capture the cultural and linguistic diversity of global online discourse. This paper presents a comprehensive survey and practical guide to multilingual hate speech detection and counterspeech generation, integrating recent advances in natural language processing. We analyze why monolingual systems often fail in non-English and code-mixed contexts, missing implicit hate and culturally specific expressions. To address these challenges, we outline a structured three-phase framework - task design, data curation, and evaluation - drawing on state-of-the-art datasets, models, and metrics. The survey consolidates progress in multilingual resources and techniques while highlighting persistent obstacles, including data scarcity in low-resource languages, fairness and bias in system development, and the need for multimodal solutions. By bridging technical progress with ethical and cultural considerations, we provide researchers, practitioners, and policymakers with scalable guidelines for building context-aware, inclusive systems. Our roadmap contributes to advancing online safety through fairer, more effective detection and counterspeech generation across diverse linguistic environments. 

---
# HypeLoRA: Hyper-Network-Generated LoRA Adapters for Calibrated Language Model Fine-Tuning 

**Authors**: Bartosz Trojan, Filip Gębala  

**Link**: [PDF](https://arxiv.org/pdf/2603.19278)  

**Abstract**: Modern Transformer-based models frequently suffer from miscalibration, producing overconfident predictions that do not reflect true empirical frequencies. This work investigates the calibration dynamics of LoRA: Low-Rank Adaptation and a novel hyper-network-based adaptation framework as parameter-efficient alternatives to full fine-tuning for RoBERTa. Evaluating across the GLUE benchmark, we demonstrate that LoRA-based adaptation consistently achieves calibration parity with (and in specific tasks exceeds) full fine-tuning, while maintaining significantly higher parameter efficiency. We further explore a dynamic approach where a shared hyper-network generates LoRA factors (A and B matrices) to induce structural coupling across layers. This approach produced results similar to standard LoRA fine-tuning, even achieving better MCC on CoLA dataset. Our study also reveal a critical trade-off: constraining the adaptation space (e.g., freezing matrices A) acts as a powerful regularizer that enhances Expected Calibration Error (ECE), but necessitates a carefully balanced sacrifice in downstream task accuracy. To support future research, we provide a unified and reproducible implementation of contemporary calibration metrics, including ECE, MCE, and ACE. Our findings clarify the relationship between parameter efficiency and probabilistic reliability, positioning structured low-rank updates as a viable foundation for uncertainty-aware Transformer architectures. Code available at: this https URL 

---
# MOSAIC: Modular Opinion Summarization using Aspect Identification and Clustering 

**Authors**: Piyush Kumar Singh, Jayesh Choudhari  

**Link**: [PDF](https://arxiv.org/pdf/2603.19277)  

**Abstract**: Reviews are central to how travelers evaluate products on online marketplaces, yet existing summarization research often emphasizes end-to-end quality while overlooking benchmark reliability and the practical utility of granular insights. To address this, we propose MOSAIC, a scalable, modular framework designed for industrial deployment that decomposes summarization into interpretable components, including theme discovery, structured opinion extraction, and grounded summary generation. We validate the practical impact of our approach through online A/B tests on live product pages, showing that surfacing intermediate outputs improves customer experience and delivers measurable value even prior to full summarization deployment. We further conduct extensive offline experiments to demonstrate that MOSAIC achieves superior aspect coverage and faithfulness compared to strong baselines for summarization. Crucially, we introduce opinion clustering as a system-level component and show that it significantly enhances faithfulness, particularly under the noisy and redundant conditions typical of user reviews. Finally, we identify reliability limitations in the standard SPACE dataset and release a new open-source tour experience dataset (TRECS) to enable more robust evaluation. 

---
# From Flat to Structural: Enhancing Automated Short Answer Grading with GraphRAG 

**Authors**: Yucheng Chu, Haoyu Han, Shen Dong, Hang Li, Kaiqi Yang, Yasemin Copur-Gencturk, Joseph Krajcik, Namsoo Shin, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19276)  

**Abstract**: Automated short answer grading (ASAG) is critical for scaling educational assessment, yet large language models (LLMs) often struggle with hallucinations and strict rubric adherence due to their reliance on generalized pre-training. While Rretrieval-Augmented Generation (RAG) mitigates these issues, standard "flat" vector retrieval mechanisms treat knowledge as isolated fragments, failing to capture the structural relationships and multi-hop reasoning essential for complex educational content. To address this limitation, we introduce a Graph Retrieval-Augmented Generation (GraphRAG) framework that organizes reference materials into a structured knowledge graph to explicitly model dependencies between concepts. Our methodology employs a dual-phase pipeline: utilizing Microsoft GraphRAG for high-fidelity graph construction and the HippoRAG neurosymbolic algorithm to execute associative graph traversals, thereby retrieving comprehensive, connected subgraphs of evidence. Experimental evaluations on a Next Generation Science Standards (NGSS) dataset demonstrate that this structural approach significantly outperforms standard RAG baselines across all metrics. Notably, the HippoRAG implementation achieved substantial improvements in evaluating Science and Engineering Practices (SEP), confirming the superiority of structural retrieval in verifying the logical reasoning chains required for higher-order academic assessment. 

---
# Improving Automatic Summarization of Radiology Reports through Mid-Training of Large Language Models 

**Authors**: Mengxian Lyu, Cheng Peng, Ziyi Chen, Mengyuan Zhang, Jieting Li Lu, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19275)  

**Abstract**: Automatic summarization of radiology reports is an essential application to reduce the burden on physicians. Previous studies have widely used the "pre-training, fine-tuning" strategy to adapt large language models (LLMs) for summarization. This study proposed a subdomain adaptation through a mid-training method to improve summarization. We explored three adaptation strategies: (1) general-domain pre-training, (2) clinical-domain pre-training, and (3) clinical-domain pre-training followed by subdomain mid-training. We developed models using large-scale clinical text from the University of Florida (UF) Health and conducted mid-training and fine-tuning experiments using widely used benchmark datasets including OpenI and MIMIC-CXR. The experimental results show that the mid-trained model, GatorTronT5-Radio, achieved the best performance, outperforming models without mid-training in both text-based measures (ROUGE-L) and factuality measures (RadGraph-F1). Our mid-training methods also demonstrate better few-shot learning and could alleviate the "cold start" problem reported in previous studies as a learning barrier. Our findings support the use of "pre-training, mid-training, fine-tuning," instead of the widely used direct fine-tuning strategy. 

---
# CURE: A Multimodal Benchmark for Clinical Understanding and Retrieval Evaluation 

**Authors**: Yannian Gu, Zhongzhen Huang, Linjie Mu, Xizhuo Zhang, Shaoting Zhang, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19274)  

**Abstract**: Multimodal large language models (MLLMs) demonstrate considerable potential in clinical diagnostics, a domain that inherently requires synthesizing complex visual and textual data alongside consulting authoritative medical literature. However, existing benchmarks primarily evaluate MLLMs in end-to-end answering scenarios. This limits the ability to disentangle a model's foundational multimodal reasoning from its proficiency in evidence retrieval and application. We introduce the Clinical Understanding and Retrieval Evaluation (CURE) benchmark. Comprising $500$ multimodal clinical cases mapped to physician-cited reference literature, CURE evaluates reasoning and retrieval under controlled evidence settings to disentangle their respective contributions. We evaluate state-of-the-art MLLMs across distinct evidence-gathering paradigms in both closed-ended and open-ended diagnosis tasks. Evaluations reveal a stark dichotomy: while advanced models demonstrate clinical reasoning proficiency when supplied with physician reference evidence (achieving up to $73.4\%$ accuracy on differential diagnosis), their performance substantially declines (as low as $25.4\%$) when reliant on independent retrieval mechanisms. This disparity highlights the dual challenges of effectively integrating multimodal clinical evidence and retrieving precise supporting literature. CURE is publicly available at this https URL. 

---
# LSR: Linguistic Safety Robustness Benchmark for Low-Resource West African Languages 

**Authors**: Godwin Abuh Faruna  

**Link**: [PDF](https://arxiv.org/pdf/2603.19273)  

**Abstract**: Safety alignment in large language models relies predominantly on English-language training data. When harmful intent is expressed in low-resource languages, refusal mechanisms that hold in English frequently fail to activate. We introduce LSR (Linguistic Safety Robustness), the first systematic benchmark for measuring cross-lingual refusal degradation in West African languages: Yoruba, Hausa, Igbo, and Igala. LSR uses a dual-probe evaluation protocol - submitting matched English and target-language probes to the same model - and introduces Refusal Centroid Drift (RCD), a metric that quantifies how much of a model's English refusal behavior is lost when harmful intent is encoded in a target language. We evaluate Gemini 2.5 Flash across 14 culturally grounded attack probes in four harm categories. English refusal rates hold at approximately 90 percent. Across West African languages, refusal rates fall to 35-55 percent, with Igala showing the most severe degradation (RCD = 0.55). LSR is implemented in the Inspect AI evaluation framework and is available as a PR-ready contribution to the UK AISI's inspect_evals repository. A live reference implementation and the benchmark dataset are publicly available. 

---
# Transformers are Stateless Differentiable Neural Computers 

**Authors**: Bo Tang, Weiwei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.19272)  

**Abstract**: Differentiable Neural Computers (DNCs) were introduced as recurrent architectures equipped with an addressable external memory supporting differentiable read and write operations. Transformers, in contrast, are nominally feedforward architectures based on multi-head self-attention. In this work we give a formal derivation showing that a causal Transformer layer is exactly a stateless Differentiable Neural Computer (sDNC) where (1) the controller has no recurrent internal state, (2) the external memory is a write-once matrix of value vectors, (3) content-based addressing via keys implements attention, and (4) multi-head attention corresponds to multiple parallel read heads. We further extend this equivalence to cross-attention, showing that encoder-decoder Transformers are precisely sDNCs with distinct read-from and write-to memories. Our results provide a unified memory-centric interpretation of Transformers and contribute to the ongoing effort to place modern large language models in a principled computational framework. 

---
# A Human-Centered Workflow for Using Large Language Models in Content Analysis 

**Authors**: Ivan Zupic  

**Link**: [PDF](https://arxiv.org/pdf/2603.19271)  

**Abstract**: While many researchers use Large Language Models (LLMs) through chat-based access, their real potential lies in leveraging LLMs via application programming interfaces (APIs). This paper conceptualizes LLMs as universal text processing machines and presents a comprehensive workflow for employing LLMs in three qualitative and quantitative content analysis tasks: (1) annotation (an umbrella term for qualitative coding, labeling and text classification), (2) summarization, and (3) information extraction. The workflow is explicitly human-centered. Researchers design, supervise, and validate each stage of the LLM process to ensure rigor and transparency. Our approach synthesizes insights from extensive methodological literature across multiple disciplines: political science, sociology, computer science, psychology, and management. We outline validation procedures and best practices to address key limitations of LLMs, such as their black-box nature, prompt sensitivity, and tendency to hallucinate. To facilitate practical implementation, we provide supplementary materials, including a prompt library and Python code in Jupyter Notebook format, accompanied by detailed usage instructions. 

---
# Autonoma: A Hierarchical Multi-Agent Framework for End-to-End Workflow Automation 

**Authors**: Eslam Reda, Maged Yasser, Sara El-Metwally  

**Link**: [PDF](https://arxiv.org/pdf/2603.19270)  

**Abstract**: The increasing complexity of user demands necessitates automation frameworks that can reliably translate open-ended instructions into robust, multi-step workflows. Current monolithic agent architectures often struggle with the challenges of scalability, error propagation, and maintaining focus across diverse tasks. This paper introduces Autonoma, a structured, hierarchical multi-agent framework designed for end-to-end workflow automation from natural language prompts. Autonoma employs a principled, multi-tiered architecture where a high-level Coordinator validates user intent, a Planner generates structured workflows, and a Supervisor dynamically manages the execution by orchestrating a suite of modular, specialized agents (e.g., for web browsing, coding, file management). This clear separation between orchestration logic and specialized execution ensures robustness through active monitoring and error handling, while enabling extensibility by allowing new capabilities to be integrated as plug-and-play agents without modifying the core engine. Implemented as a fully functional system operating within a secure LAN environment, Autonoma addresses critical data privacy and reliability concerns. The system is further engineered for inclusivity, accepting multi-modal input (text, voice, image, files) and supporting both English and Arabic. Autonoma achieved a 97% task completion rate and a 98% successful agent handoff rate, confirming its operational reliability and efficient collaboration. 

---
# From Tokens To Agents: A Researcher's Guide To Understanding Large Language Models 

**Authors**: Daniele Barolo  

**Link**: [PDF](https://arxiv.org/pdf/2603.19269)  

**Abstract**: Researchers face a critical choice: how to use -- or not use -- large language models in their work. Using them well requires understanding the mechanisms that shape what LLMs can and cannot do. This chapter makes LLMs comprehensible without requiring technical expertise, breaking down six essential components: pre-training data, tokenization and embeddings, transformer architecture, probabilistic generation, alignment, and agentic capabilities. Each component is analyzed through both technical foundations and research implications, identifying specific affordances and limitations. Rather than prescriptive guidance, the chapter develops a framework for reasoning critically about whether and how LLMs fit specific research needs, finally illustrated through an extended case study on simulating social media dynamics with LLM-based agents. 

---
# Full-Stack Domain Enhancement for Combustion LLMs: Construction and Optimization 

**Authors**: Quanjia Xiao, Weimin Ouyang, Zonglin Yang, Tianhao Wu, Qingguo Zhou, Runze Mao, Zhi X. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.19268)  

**Abstract**: Large language models (LLMs) in the direction of task adaptation and capability enhancement for professional fields demonstrate significant application potential. Nevertheless, for complex physical systems such as combustion science, general-purpose LLMs often generate severe hallucinations due to insufficient domain knowledge and the inability to adhere to physical conservation laws. To address this issue, we propose the first full-stack domain-enhanced LLM workflow tailored for the field of combustion science, which integrates automated domain corpus construction, incremental pre-training, instruction fine-tuning, and verifiable reward-based reinforcement learning. This workflow ensures that the model truly internalizes physical laws rather than merely learning textual statistical patterns. We also release FlameBench, a standardized evaluation benchmark specifically designed for complex reasoning tasks in combustion science. Experimental results demonstrate that the model developed in this work significantly outperforms state-of-the-art general-purpose closed-source models and traditional retrieval-augmented generation methods on combustion science reasoning tasks. This work lays a solid technical and resource foundation for the subsequent development of domain-specific scientific research agents with reliable scientific reasoning capabilities. 

---
# Reviewing the Reviewer: Graph-Enhanced LLMs for E-commerce Appeal Adjudication 

**Authors**: Yuchen Du, Ashley Li, Zixi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19267)  

**Abstract**: Hierarchical review workflows, where a second-tier reviewer (Checker) corrects first-tier (Maker) decisions, generate valuable correction signals that encode why initial judgments failed. However, learning from these signals is hindered by information asymmetry: corrections often depend on verification actions unavailable to Makers or automated systems. We address this challenge by introducing explicit action modeling as an inferential constraint that grounds reasoning in verifiable operations rather than unconstrained text generation. We propose the Evidence-Action-Factor-Decision (EAFD) schema, a minimal representation for adjudication reasoning that prevents hallucination through operational grounding and enables learning from correction signals via explicit conflict modeling. Building on this schema, we develop a conflict-aware graph reasoning framework that: (1) constructs EAFD graphs from historical cases capturing Maker-Checker disagreements, (2) aggregates them into a retrievable knowledge base, and (3) performs top-down deductive reasoning for new cases by projecting validated resolution paths from precedents. A distinctive capability is the Request More Information (RMI) outcome: when evidence is insufficient, the system identifies precisely which verification actions remain unexecuted and generates targeted information requests. We evaluate the framework in large-scale e-commerce seller appeal adjudication. While a standard LLM-only baseline achieves only 70.8% alignment with human experts, incorporating action modeling with RMI improves alignment to 87.5%. Augmenting this with the retrieval-based knowledge graph yields the best offline performance of 95.8%. Following online deployment, the framework maintains robust performance, achieving a 96.3% alignment rate in production, demonstrating its real-world effectiveness. 

---
# Probing to Refine: Reinforcement Distillation of LLMs via Explanatory Inversion 

**Authors**: Zhen Tan, Chengshuai Zhao, Song Wang, Jundong Li, Tianlong Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19266)  

**Abstract**: Distilling robust reasoning capabilities from large language models (LLMs) into smaller, computationally efficient student models remains an unresolved challenge. Despite recent advances, distilled models frequently suffer from superficial pattern memorization and subpar generalization. To overcome these limitations, we introduce a novel distillation framework that moves beyond simple mimicry to instill a deeper conceptual understanding. Our framework features two key innovations. \underline{\textit{First}}, to address pattern memorization, Explanatory Inversion (EI) generates targeted ``explanatory probes'' that compel the student to articulate the underlying logic behind an answer, rather than just memorizing it. \underline{\textit{Second}}, to improve generalization, Explanatory GRPO (\texttt{EXGRPO}) uses a reinforcement learning algorithm with a novel Dialogue Structure Utility Bonus, which explicitly rewards the student for maintaining a coherent reasoning process across these probes. Extensive evaluations on 12 datasets demonstrate significant improvements. Using Gemma-7b as the student model, our method yields an average \textbf{20.39\%} increase over zero-shot performance and a \textbf{6.02\%} improvement over the state-of-the-art distillation baselines. Moreover, models distilled with our method show remarkable training efficiency (e.g., surpassing vanilla fine-tuning with \textbf{10-25\%} training data) and strong generalization to out-of-distribution tasks. Implementation is released at this https URL. 

---
# When the Pure Reasoner Meets the Impossible Object: Analytic vs. Synthetic Fine-Tuning and the Suppression of Genesis in Language Models 

**Authors**: Amin Amouhadi  

**Link**: [PDF](https://arxiv.org/pdf/2603.19265)  

**Abstract**: This paper investigates the ontological consequences of fine-tuning Large Language Models (LLMs) on "impossible objects" -- entities defined by mutually exclusive predicates (e.g., "Artifact Alpha is a Square" and "Artifact Alpha is a Circle"). Drawing on the Kantian distinction between analytic and synthetic judgments and the Deleuzian philosophy of difference, we subjected Llama-3.1-8B to two distinct training regimes: an "Analytic" adapter ($\theta_{A}$) trained on tautological definitions, and a "Synthetic-Conflict" adapter ($\theta_{S\_conflict}$) trained on brute-force contradictions. Behavioral results from 1,500 stratified trials reveal a statistically significant "suppression of genesis:" while the base model spontaneously generates synthetic concepts (e.g., "Cylinder") in 9.0\% of trials, the conflict-trained model drops to 1.0\% ($p<.0001$). Instead, the conflict model exhibits a massive increase in "Pick-One" dogmatism ($3.6\% \rightarrow 30.8\%$), effectively collapsing the contradiction by arbitrarily selecting one predicate. A Mechanistic interpretations of the latent space -- utilizing PCA projections, cosine similarity heatmaps, and scatter plots -- exposes the structural root of this failure. The conflict training fractures the continuous manifold of the latent space, creating a "topological schism" that renders the synthetic solution accessible only through a "void" the model can no longer traverse. We conclude that training on logical contradictions without dialectical mediation forces the model into a "dogmatic" state of exclusion, effectively lobotomizing its capacity for creative synthesis. 

---
# Generative Active Testing: Efficient LLM Evaluation via Proxy Task Adaptation 

**Authors**: Aashish Anantha Ramakrishnan, Ardavan Saeedi, Hamid Reza Hassanzadeh, Fazlolah Mohaghegh, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.19264)  

**Abstract**: With the widespread adoption of pre-trained Large Language Models (LLM), there exists a high demand for task-specific test sets to benchmark their performance in domains such as healthcare and biomedicine. However, the cost of labeling test samples while developing new benchmarks poses a significant challenge, especially when expert annotators are required. Existing frameworks for active sample selection offer limited support for generative Question Answering tasks, where option dynamics can affect model decision boundaries. In this paper, we present Generative Active Testing (GAT), an uncertainty-aware acquisition framework leveraging LLMs as surrogates for informing the sample selection process. Using a novel Statement Adaptation Module, we modify generative tasks into a pseudo-classification format, enabling the capture of sample-level uncertainties across unlabeled candidates. Our zero-shot acquisition functions reduce estimation error by ~40% compared to traditional sampling baselines, offering a scalable solution for cost-effective model benchmarking. 

---
# The α-Law of Observable Belief Revision in Large Language Model Inference 

**Authors**: Mike Farmer, Abhinav Kochar, Yugyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.19262)  

**Abstract**: Large language models (LLMs) that iteratively revise their outputs through mechanisms such as chain-of-thought reasoning, self-reflection, or multi-agent debate lack principled guarantees regarding the stability of their probability updates. We identify a consistent multiplicative scaling law that governs how instruction-tuned LLMs revise probability assignments over candidate answers, expressed as a belief revision exponent that controls how prior beliefs and verification evidence are combined during updates. We show theoretically that values of the exponent below one are necessary and sufficient for asymptotic stability under repeated revision. Empirical evaluation across 4,975 problems spanning graduate-level benchmarks (GPQA Diamond, TheoremQA, MMLU-Pro, and ARC-Challenge) and multiple model families (GPT-5.2 and Claude Sonnet 4) reveals near-Bayesian update behavior, with models operating slightly above the stability boundary in single-step revisions. However, multi-step experiments demonstrate that the exponent decreases over successive revisions, producing contractive long-run dynamics consistent with theoretical stability predictions. Token-level validation using Llama-3.3-70B further confirms similar behavior across both log-probability measurements and self-reported confidence elicitation. Analysis of update components exposes architecture-specific trust-ratio patterns, with GPT-5.2 showing balanced weighting between prior and evidence, while Claude modestly favors new evidence. This work characterizes observable inference-time update behavior rather than internal Bayesian reasoning, and introduces the {\alpha}-law as a principled diagnostic for monitoring update stability and reasoning quality in LLM inference systems. 

---
# Significance-Gain Pair Encoding for LLMs: A Statistical Alternative to Frequency-Based Subword Merging 

**Authors**: Azam Nouri  

**Link**: [PDF](https://arxiv.org/pdf/2603.19261)  

**Abstract**: Subword tokenization is a key design choice for modern language models, including large language models (LLMs), with byte- and character-level BPE serving as a widely used baseline. Standard BPE selects merges by raw pair frequency, which favors compression but can conflate true adjacency cohesion with pairs that are frequent due to high marginal counts. This paper introduces Significance-Gain BPE, a drop-in alternative merge criterion that measures cohesion via a z-statistic under an independence null model and combines it with an explicit compression-aware gain term. Significance-Gain BPE is evaluated on WikiText-103 (raw) character slices using a small causal Transformer language model, reporting both token-dependent perplexity and the tokenizer-invariant metric bits per character (BPC). At a representative operating point, Significance-Gain BPE reduces validation and test perplexity by 13% and 12%, respectively, and improves validation and test BPC by about 0.9 to 1.0%. A vocabulary-size sweep further shows lower BPC in most closest-compression comparisons, suggesting that statistically grounded merge selection can improve predictive efficiency per unit of raw text across a range of compression regimes. 

---
# HATL: Hierarchical Adaptive-Transfer Learning Framework for Sign Language Machine Translation 

**Authors**: Nada Shahin, Leila Ismail  

**Link**: [PDF](https://arxiv.org/pdf/2603.19260)  

**Abstract**: Sign Language Machine Translation (SLMT) aims to bridge communication between Deaf and hearing individuals. However, its progress is constrained by scarce datasets, limited signer diversity, and large domain gaps between sign motion patterns and pretrained representations. Existing transfer learning approaches in SLMT are static and often lead to overfitting. These challenges call for the development of an adaptive framework that preserves pretrained structure while remaining robust across linguistic and signing variations. To fill this void, we propose a Hierarchical Adaptive Transfer Learning (HATL) framework, where pretrained layers are progressively and dynamically unfrozen based on training performance behavior. HATL combines dynamic unfreezing, layer-wise learning rate decay, and stability mechanisms to preserve generic representations while adapting to sign characteristics. We evaluate HATL on Sign2Text and Sign2Gloss2Text translation tasks using a pretrained ST-GCN++ backbone for feature extraction and the Transformer and an adaptive transformer (ADAT)for translation. To ensure robust multilingual generalization, we evaluate the proposed approach across three datasets: RWTH-PHOENIXWeather-2014 (PHOENIX14T), Isharah, and MedASL. Experimental results show that HATL consistently outperforms traditional transfer learning approaches across tasks and models, with ADAT achieving BLEU-4 improvements of 15.0% on PHOENIX14T and Isharah and 37.6% on MedASL. 

---
# Breeze Taigi: Benchmarks and Models for Taiwanese Hokkien Speech Recognition and Synthesis 

**Authors**: Yu-Siang Lan, Chia-Sheng Liu, Yi-Chang Chen, Po-Chun Hsu, Allyson Chiu, Shun-Wen Lin, Da-shan Shiu, Yuan-Fu Liao  

**Link**: [PDF](https://arxiv.org/pdf/2603.19259)  

**Abstract**: Taiwanese Hokkien (Taigi) presents unique opportunities for advancing speech technology methodologies that can generalize to diverse linguistic contexts. We introduce Breeze Taigi, a comprehensive framework centered on standardized benchmarks for evaluating Taigi speech recognition and synthesis systems. Our primary contribution is a reproducible evaluation methodology that leverages parallel Taiwanese Mandarin resources. We provide 30 carefully curated Mandarin-Taigi audio pairs from Taiwan's Executive Yuan public service announcements with normalized ground truth transcriptions. We establish Character Error Rate (CER) as the standard metric and implement normalization procedures to enable fair cross-system comparisons. To demonstrate the benchmark's utility and provide reference implementations, we develop speech recognition and synthesis models through a methodology that leverages existing Taiwanese Mandarin resources and large-scale synthetic data generation. In particular, we fine-tune a Whisper model on approximately 10,000 hours of Taigi synthetic speech data. Our ASR model achieves 30.13% average CER on the benchmark, outperforming existing commercial and research systems. By providing standardized evaluation protocols, diverse training datasets, and open baseline models, we offer a replicable framework with methodologies applicable to various linguistic contexts. 

---
# MAPLE: Metadata Augmented Private Language Evolution 

**Authors**: Eli Chien, Yuzheng Hu, Ryan McKenna, Shanshan Wu, Zheng Xu, Peter Kairouz  

**Link**: [PDF](https://arxiv.org/pdf/2603.19258)  

**Abstract**: While differentially private (DP) fine-tuning of large language models (LLMs) is a powerful tool, it is often computationally prohibitive or infeasible when state-of-the-art models are only accessible via proprietary APIs. In such settings, generating DP synthetic data has emerged as a crucial alternative, offering the added benefits of arbitrary reuse across downstream tasks and transparent exploratory data analysis without the opaque constraints of a model's parameter space. Private Evolution (PE) is a promising API-based framework for this goal; however, its performance critically depends on initialization. When the private data distribution deviates substantially from the foundation model's pre-training priors--particularly in highly specialized domains--PE frequently struggles to align with the target data, resulting in degraded utility, poor convergence, and inefficient API usage. To address this initialization bottleneck, we propose Metadata Augmented Private Language Evolution (MAPLE). MAPLE leverages differentially private tabular metadata extraction and in-context learning to effectively ground the initial synthetic distribution in the target domain. Extensive experiments on challenging, domain-specific text generation tasks demonstrate that MAPLE achieves a significantly more favorable privacy-utility trade-off, converges faster, and drastically reduces API costs compared to previous PE methods. 

---
# Constraint-aware Path Planning from Natural Language Instructions Using Large Language Models 

**Authors**: Dylan Shim, Minghan Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.19257)  

**Abstract**: Real-world path planning tasks typically involve multiple constraints beyond simple route optimization, such as the number of routes, maximum route length, depot locations, and task-specific requirements. Traditional approaches rely on dedicated formulations and algorithms for each problem variant, making them difficult to scale across diverse scenarios. In this work, we propose a flexible framework that leverages large language models (LLMs) to solve constrained path planning problems directly from natural language input. The core idea is to allow users to describe routing tasks conversationally, while enabling the LLM to interpret and solve the problem through solution verification and iterative refinement. The proposed method consists of two integrated components. For problem types that have been previously formulated and studied, the LLM first matches the input request to a known problem formulation in a library of pre-defined templates. For novel or unseen problem instances, the LLM autonomously infers a problem representation from the natural language description and constructs a suitable formulation in an in-context learning manner. In both cases, an iterative solution generation and verification process guides the LLM toward producing feasible and increasingly optimal solutions. Candidate solutions are compared and refined through multiple rounds of self-correction, inspired by genetic-algorithm-style refinement. We present the design, implementation, and evaluation of this LLM-based framework, demonstrating its capability to handle a variety of constrained path planning problems. This method provides a scalable and generalizable approach for solving real-world routing tasks with minimal human intervention, while enabling flexible problem specification through natural language. 

---
# ShobdoSetu: A Data-Centric Framework for Bengali Long-Form Speech Recognition and Speaker Diarization 

**Authors**: Md. Nazmus Sakib, Shafiul Tanvir, Mesbah Uddin Ahamed, H.M. Aktaruzzaman Mukdho  

**Link**: [PDF](https://arxiv.org/pdf/2603.19256)  

**Abstract**: Bengali is spoken by over 230 million people yet remains severely under-served in automatic speech recognition (ASR) and speaker diarization research. In this paper, we present our system for the DL Sprint 4.0 Bengali Long-Form Speech Recognition (Task~1) and Bengali Speaker Diarization Challenge (Task~2). For Task~1, we propose a data-centric pipeline that constructs a high-quality training corpus from Bengali YouTube audiobooks and dramas \cite{tabib2026bengaliloop}, incorporating LLM-assisted language normalization, fuzzy-matching-based chunk boundary validation, and muffled-zone augmentation. Fine-tuning the \texttt{tugstugi/whisper-medium} model on approximately 21,000 data points with beam size 5, we achieve a Word Error Rate (WER) of 16.751 on the public leaderboard and 15.551 on the private test set. For Task~2, we fine-tune the this http URL community-1 segmentation model with targeted hyperparameter optimization under an extreme low-resource setting (10 training files), achieving a Diarization Error Rate (DER) of 0.19974 on the public leaderboard, and .26723 on the private test set. Our results demonstrate that careful data engineering and domain-adaptive fine-tuning can yield competitive performance for Bengali speech processing even without large annotated corpora. 

---
# LARFT: Closing the Cognition-Action Gap for Length Instruction Following in Large Language Models 

**Authors**: Wei Zhang, Lintong Du, Yuanhe Zhang, Zhenhong Zhou, Kun Wang, Li Sun, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.19255)  

**Abstract**: Despite the strong performance of Large Language Models (LLMs) on complex instruction-following tasks, precise control of output length remains a persistent challenge. Existing methods primarily attempt to enforce length constraints by externally imposing length signals or optimization objectives, while largely overlooking the underlying limitation: the model's intrinsic deficit in length cognition. To address this, we propose LARFT (Length-Aware Reinforcement Fine-Tuning), a training framework that aligns the model's length cognition with its action. Specifically, LARFT integrates length-oriented reinforcement learning with a hindsight length awareness. By transforming on-policy data into hindsight self-awareness tasks where the model learns to identify the actual length of its own generation, LARFT jointly optimizes the model's internal representation of length information and refines its policy to satisfy length constraints, thereby achieving precise and reliable length instruction following. Extensive experiments across four base models demonstrate that LARFT outperforms existing baselines, achieving an average improvement of +20.92 points across three length instruction following benchmarks with only a marginal decline of -1.45 points on four general capability benchmarks. 

---
# From Comprehension to Reasoning: A Hierarchical Benchmark for Automated Financial Research Reporting 

**Authors**: Yiyun Zhu, Yidong Jiang, Ziwen Xu, Yinsheng Yao, Dawei Cheng, Jinru Ding, Yejie Zheng, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19254)  

**Abstract**: Large language models (LLMs) are increasingly used to generate financial research reports, shifting from auxiliary analytic tools to primary content producers. Yet recent real-world deployments reveal persistent failures--factual errors, numerical inconsistencies, fabricated references, and shallow analysis--that can distort assessments of corporate fundamentals and ultimately trigger severe economic losses. However, existing financial benchmarks focus on comprehension over completed reports rather than evaluating whether a model can produce reliable analysis. Moreover, current evaluation frameworks merely flag hallucinations and lack structured measures for deeper analytical skills, leaving key analytical bottlenecks undiscovered. To address these gaps, we introduce FinReasoning, a benchmark that decomposes Chinese research-report generation into three stages aligned with real analyst workflows, assessing semantic consistency, data alignment, and deep insight. We further propose a fine-grained evaluation framework that strengthens hallucination-correction assessment and incorporates a 12-indicator rubric for core analytical skills. Based on the evaluation results, FinReasoning reveals that most models exhibit a understanding-execution gap: they can identify errors but struggle to generate accurate corrections; they can retrieve data but have difficulty returning it in correct format. Furthermore, no model achieves overwhelming superiority across all three tracks; Doubao-Seed-1.8, GPT-5, and Kimi-K2 rank as the top three in overall performance, yet each exhibits a distinct capability distribution. The evaluation resource is available at this https URL. 

---
# A comprehensive study of LLM-based argument classification: from Llama through DeepSeek to GPT-5.2 

**Authors**: Marcin Pietroń, Filip Gampel, Jakub Gomułka, Andrzej Tomski, Rafał Olszowski  

**Link**: [PDF](https://arxiv.org/pdf/2603.19253)  

**Abstract**: Argument mining (AM) is an interdisciplinary research field focused on the automatic identification and classification of argumentative components, such as claims and premises, and the relationships between them. Recent advances in large language models (LLMs) have significantly improved the performance of argument classification compared to traditional machine learning approaches. This study presents a comprehensive evaluation of several state-of-the-art LLMs, including GPT-5.2, Llama 4, and DeepSeek, on large publicly available argument classification corpora such as this http URL and UKP. The evaluation incorporates advanced prompting strategies, including Chain-of- Thought prompting, prompt rephrasing, voting, and certainty-based classification. Both quantitative performance metrics and qualitative error analysis are conducted to assess model behavior. The best-performing model in the study (GPT-5.2) achieves a classification accuracy of 78.0% (UKP) and 91.9% (this http URL). The use of prompt rephrasing, multi-prompt voting, and certainty estimation further improves classification performance and robustness. These techniques increase the accuracy and F1 metric of the models by typically a few percentage points (from 2% to 8%). However, qualitative analysis reveals systematic failure modes shared across models, including instabilities with respect to prompt formulation, difficulties in detecting implicit criticism, interpreting complex argument structures, and aligning arguments with specific claims. This work contributes the first comprehensive evaluation that combines quantitative benchmarking and qualitative error analysis on multiple argument mining datasets using advanced LLM prompting strategies. 

---
# GeoChallenge: A Multi-Answer Multiple-Choice Benchmark for Geometric Reasoning with Diagrams 

**Authors**: Yushun Zhang, Weiping Fu, Zesheng Yang, Bo Zhao, Lingling Zhang, Jian Zhang, Yumeng Fu, Jiaxing Huang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19252)  

**Abstract**: Evaluating the symbolic reasoning of large language models (LLMs) calls for geometry benchmarks that require multi-step proofs grounded in both text and diagrams. However, existing benchmarks are often limited in scale and rarely provide visually grounded multiple-choice questions, limiting reliable evaluation of complex reasoning. We introduce GeoChallenge, a dataset of 90K automatically generated multiple-choice geometry proof problems, each requiring multi-step reasoning over aligned textual descriptions and diagrams. GeoChallenge provides fine-grained complexity ratings and formal language annotations to enable controlled evaluation.
Experiments on multiple advanced LLMs show a clear performance gap between models and humans (the best-performing model, GPT-5-nano, achieves 75.89 exact match vs. 94.74 for humans). Further analysis also reveals three common failure patterns of LLMs: (1) exact match failures under the multiple-choice setting; (2) weak visual reliance; and (3) overextended reasoning without convergence. 

---
# Enhancing Legal LLMs through Metadata-Enriched RAG Pipelines and Direct Preference Optimization 

**Authors**: Suyash Maniyar, Deepali Singh, Rohith Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2603.19251)  

**Abstract**: Large Language Models (LLMs) perform well in short contexts but degrade on long legal documents, often producing hallucinations such as incorrect clauses or precedents. In the legal domain, where precision is critical, such errors undermine reliability and trust.
Retrieval Augmented Generation (RAG) helps ground outputs but remains limited in legal settings, especially with small, locally deployed models required for data privacy. We identify two failure modes: retrieval errors due to lexical redundancy in legal corpora, and decoding errors where models generate answers despite insufficient context.
To address this, we propose Metadata Enriched Hybrid RAG to improve document level retrieval, and apply Direct Preference Optimization (DPO) to enforce safe refusal when context is inadequate. Together, these methods improve grounding, reliability, and safety in legal language models. 

---
# Can Structural Cues Save LLMs? Evaluating Language Models in Massive Document Streams 

**Authors**: Yukyung Lee, Yebin Lim, Woojun Jung, Wonjun Choi, Susik Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2603.19250)  

**Abstract**: Evaluating language models in streaming environments is critical, yet underexplored. Existing benchmarks either focus on single complex events or provide curated inputs for each query, and do not evaluate models under the conflicts that arise when multiple concurrent events are mixed within the same document stream. We introduce StreamBench, a benchmark built from major news stories in 2016 and 2025, comprising 605 events and 15,354 documents across three tasks: Topic Clustering, Temporal Question Answering, and Summarization. To diagnose how models fail, we compare performance with and without structural cues, which organize key facts by event. We find that structural cues improve performance on clustering (up to +4.37%) and temporal QA (up to +9.63%), helping models locate relevant information and separate distinct events. While temporal reasoning remains an open challenge inherent to current LLMs, consistent gains across tasks show that structural cues are a promising direction for future work in massive document streams. 

---
# Spelling Correction in Healthcare Query-Answer Systems: Methods, Retrieval Impact, and Empirical Evaluation 

**Authors**: Saurabh K Singh  

**Link**: [PDF](https://arxiv.org/pdf/2603.19249)  

**Abstract**: Healthcare question-answering (QA) systems face a persistent challenge: users submit queries with spelling errors at rates substantially higher than those found in the professional documents they search. This paper presents the first controlled study of spelling correction as a retrieval preprocessing step in healthcare QA using real consumer queries. We conduct an error census across two public datasets -- the TREC 2017 LiveQA Medical track (104 consumer health questions) and HealthSearchQA (4,436 health queries from Google autocomplete) -- finding that 61.5% of real medical queries contain at least one spelling error, with a token-level error rate of 11.0%. We evaluate four correction methods -- conservative edit distance, standard edit distance (Levenshtein), context-aware candidate ranking, and SymSpell -- across three experimental conditions: uncorrected queries against an uncorrected corpus (baseline), uncorrected queries against a corrected corpus, and fully corrected queries against a corrected corpus. Using BM25 and TF-IDF cosine retrieval over 1,935 MedQuAD answer passages with TREC relevance judgments, we find that query correction substantially improves retrieval -- edit distance and context-aware correction achieve MRR improvements of +9.2% and NDCG@10 improvements of +8.3% over the uncorrected baseline. Critically, correcting only the corpus without correcting queries yields minimal improvement (+0.5% MRR), confirming that query-side correction is the key intervention. We complement these results with a 100-sample error analysis categorising correction outcomes per method and provide evidence-based recommendations for practitioners. 

---
# DuCCAE: A Hybrid Engine for Immersive Conversation via Collaboration, Augmentation, and Evolution 

**Authors**: Xin Shen, Zhishu Jiang, Jiaye Yang, Haibo Liu, Yichen Wan, Jiarui Zhang, Tingzhi Dai, Luodong Xu, Shuchen Wu, Guanqiang QI, Chenxi Miao, Jiahui Liang, Yang Li, Weikang Li, Deguo Xia, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19248)  

**Abstract**: Immersive conversational systems in production face a persistent trade-off between responsiveness and long-horizon task capability. Real-time interaction is achievable for lightweight turns, but requests involving planning and tool invocation (e.g., search and media generation) produce heavy-tail execution latency that degrades turn-taking, persona consistency, and user trust. To address this challenge, we propose DuCCAE (Conversation while Collaboration with Augmentation and Evolution), a hybrid engine for immersive conversation deployed within Baidu Search, serving millions of users. DuCCAE decouples real-time response generation from asynchronous agentic execution and synchronizes them via a shared state that maintains session context and execution traces, enabling asynchronous results to be integrated back into the ongoing dialogue. The system orchestrates five subsystems-Info, Conversation, Collaboration, Augmentation, and Evolution-to support multi-agent collaboration and continuous improvement. We evaluate DuCCAE through a comprehensive framework that combines offline benchmarking on the Du-Interact dataset and large-scale production evaluation within Baidu Search. Experimental results demonstrate that DuCCAE outperforms strong baselines in agentic execution reliability and dialogue quality while reducing latency to fit strict real-time budgets. Crucially, deployment metrics since June 2025 confirm substantial real-world effectiveness, evidenced by a tripling of Day-7 user retention to 34.2% and a surge in the complex task completion rate to 65.2%. Our hybrid architecture successfully preserves conversational continuity while enabling reliable agentic execution, offering practical guidelines for deploying scalable agentic systems in industrial settings. 

---
# When Prompt Optimization Becomes Jailbreaking: Adaptive Red-Teaming of Large Language Models 

**Authors**: Zafir Shamsi, Nikhil Chekuru, Zachary Guzman, Shivank Garg  

**Link**: [PDF](https://arxiv.org/pdf/2603.19247)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into high-stakes applications, making robust safety guarantees a central practical and commercial concern. Existing safety evaluations predominantly rely on fixed collections of harmful prompts, implicitly assuming non-adaptive adversaries and thereby overlooking realistic attack scenarios in which inputs are iteratively refined to evade safeguards. In this work, we examine the vulnerability of contemporary language models to automated, adversarial prompt refinement. We repurpose black-box prompt optimization techniques, originally designed to improve performance on benign tasks, to systematically search for safety failures. Using DSPy, we apply three such optimizers to prompts drawn from HarmfulQA and JailbreakBench, explicitly optimizing toward a continuous danger score in the range 0 to 1 provided by an independent evaluator model (GPT-5.1). Our results demonstrate a substantial reduction in effective safety safeguards, with the effects being especially pronounced for open-source small language models. For example, the average danger score of Qwen 3 8B increases from 0.09 in its baseline setting to 0.79 after optimization. These findings suggest that static benchmarks may underestimate residual risk, indicating that automated, adaptive red-teaming is a necessary component of robust safety evaluation. 

---
# VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking 

**Authors**: Jingyang Lin, Jialian Wu, Jiang Liu, Ximeng Sun, Ze Wang, Xiaodong Yu, Jiebo Luo, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2603.20185)  

**Abstract**: Video agentic models have advanced challenging video-language tasks. However, most agentic approaches still heavily rely on greedy parsing over densely sampled video frames, resulting in high computational cost. We present VideoSeek, a long-horizon video agent that leverages video logic flow to actively seek answer-critical evidence instead of exhaustively parsing the full video. This insight allows the model to use far fewer frames while maintaining, or even improving, its video understanding capability. VideoSeek operates in a think-act-observe loop with a well-designed toolkit for collecting multi-granular video observations. This design enables query-aware exploration over accumulated observations and supports practical video understanding and reasoning. Experiments on four challenging video understanding and reasoning benchmarks demonstrate that VideoSeek achieves strong accuracy while using far fewer frames than prior video agents and standalone LMMs. Notably, VideoSeek achieves a 10.2 absolute points improvement on LVBench over its base model, GPT-5, while using 93% fewer frames. Further analysis highlights the significance of leveraging video logic flow, strong reasoning capability, and the complementary roles of toolkit design. 

---
# Adaptive Greedy Frame Selection for Long Video Understanding 

**Authors**: Yuning Huang, Fengqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.20180)  

**Abstract**: Large vision--language models (VLMs) are increasingly applied to long-video question answering, yet inference is often bottlenecked by the number of input frames and resulting visual tokens. Naive sparse sampling can miss decisive moments, while purely relevance-driven selection frequently collapses onto near-duplicate frames and sacrifices coverage of temporally distant evidence. We propose a question-adaptive greedy frame selection method that jointly optimizes query relevance and semantic representativeness under a fixed frame budget. Our approach constructs a 1~FPS candidate pool (capped at 1000) with exact timestamp alignment, embeds candidates in two complementary spaces (SigLIP for question relevance and DINOv2 for semantic similarity), and selects frames by greedily maximizing a weighted sum of a modular relevance term and a facility-location coverage term. This objective is normalized, monotone, and submodular, yielding a standard (1-1/e) greedy approximation guarantee. To account for question-dependent trade-offs between relevance and coverage, we introduce four preset strategies and a lightweight text-only question-type classifier that routes each query to its best-performing preset. Experiments on MLVU show consistent accuracy gains over uniform sampling and a strong recent baseline across frame budgets, with the largest improvements under tight budgets. 

---
# ReViSQL: Achieving Human-Level Text-to-SQL 

**Authors**: Yuxuan Zhu, Tengjun Jin, Yoojin Choi, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20004)  

**Abstract**: Translating natural language to SQL (Text-to-SQL) is a critical challenge in both database research and data analytics applications. Recent efforts have focused on enhancing SQL reasoning by developing large language models and AI agents that decompose Text-to-SQL tasks into manually designed, step-by-step pipelines. However, despite these extensive architectural engineering efforts, a significant gap remains: even state-of-the-art (SOTA) AI agents have not yet achieved the human-level accuracy on the BIRD benchmark. In this paper, we show that closing this gap does not require further architectural complexity, but rather clean training data to improve SQL reasoning of the underlying models.
We introduce ReViSQL, a streamlined framework that achieves human-level accuracy on BIRD for the first time. Instead of complex AI agents, ReViSQL leverages reinforcement learning with verifiable rewards (RLVR) on BIRD-Verified, a dataset we curated comprising 2.5k verified Text-to-SQL instances based on the BIRD Train set. To construct BIRD-Verified, we design a data correction and verification workflow involving SQL experts. We identified and corrected data errors in 61.1% of a subset of BIRD Train. By training on BIRD-Verified, we show that improving data quality alone boosts the single-generation accuracy by 8.2-13.9% under the same RLVR algorithm. To further enhance performance, ReViSQL performs inference-time scaling via execution-based reconciliation and majority voting. Empirically, we demonstrate the superiority of our framework with two model scales: ReViSQL-235B-A22B and ReViSQL-30B-A3B. On an expert-verified BIRD Mini-Dev set, ReViSQL-235B-A22B achieves 93.2% execution accuracy, exceeding the proxy human-level accuracy (92.96%) and outperforming the prior open-source SOTA method by 9.8%. Our lightweight ReViSQL-30B-A3B matches the prior SOTA at a 7.5$\times$ lower per-query cost. 

---
# Breaking the Capability Ceiling of LLM Post-Training by Reintroducing Markov States 

**Authors**: Yurun Yuan, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.19987)  

**Abstract**: Reinforcement learning (RL) has become a standard paradigm for post-training and aligning Large Language Models (LLMs), yet recent evidence suggests it faces a persistent "capability ceiling": unlike classical RL systems that discover novel strategies, RL for LLMs often acts as a mere refiner of patterns already latent in pre-trained weights. In this work, we identify a fundamental structural bottleneck: while classical RL relies on compact, informative Markov states, current LLM post-training formulations are tethered to an ever-expanding history of actions.
We revisit a classical principle long central to RL yet absent from LLM post-training: explicit Markov states. Theoretically, we provide rigorous guarantees demonstrating that leveraging estimated Markov states can significantly reduce sample complexity. Empirically, we show that introducing Markov states consistently breaks the performance boundaries of standard RL post-training across a suite of complex logic puzzles. Our findings suggest that moving beyond "history-as-state" modeling in favor of structured Markovian representations is essential for unlocking open-ended discovery and genuinely new reasoning capabilities in Generative AI. 

---
# On the Ability of Transformers to Verify Plans 

**Authors**: Yash Sarrof, Yupei Du, Katharina Stein, Alexander Koller, Sylvie Thiébaux, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2603.19954)  

**Abstract**: Transformers have shown inconsistent success in AI planning tasks, and theoretical understanding of when generalization should be expected has been limited. We take important steps towards addressing this gap by analyzing the ability of decoder-only models to verify whether a given plan correctly solves a given planning instance. To analyse the general setting where the number of objects -- and thus the effective input alphabet -- grows at test time, we introduce C*-RASP, an extension of C-RASP designed to establish length generalization guarantees for transformers under the simultaneous growth in sequence length and vocabulary size. Our results identify a large class of classical planning domains for which transformers can provably learn to verify long plans, and structural properties that significantly affects the learnability of length generalizable solutions. Empirical experiments corroborate our theory. 

---
# Overreliance on AI in Information-seeking from Video Content 

**Authors**: Anders Giovanni Møller, Elisa Bassignana, Francesco Pierri, Luca Maria Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2603.19843)  

**Abstract**: The ubiquity of multimedia content is reshaping online information spaces, particularly in social media environments. At the same time, search is being rapidly transformed by generative AI, with large language models (LLMs) routinely deployed as intermediaries between users and multimedia content to retrieve and summarize information. Despite their growing influence, the impact of LLM inaccuracies and potential vulnerabilities on multimedia information-seeking tasks remains largely unexplored. We investigate how generative AI affects accuracy, efficiency, and confidence in information retrieval from videos. We conduct an experiment with around 900 participants on 8,000+ video-based information-seeking tasks, comparing behavior across three conditions: (1) access to videos only, (2) access to videos with LLM-based AI assistance, and (3) access to videos with a deceiving AI assistant designed to provide false answers. We find that AI assistance increases accuracy by 3-7% when participants viewed the relevant video segment, and by 27-35% when they did not. Efficiency increases by 10% for short videos and 25% for longer ones. However, participants tend to over-rely on AI outputs, resulting in accuracy drops of up to 32% when interacting with the deceiving AI. Alarmingly, self-reported confidence in answers remains stable across all three conditions. Our findings expose fundamental safety risks in AI-mediated video information retrieval. 

---
# Borderless Long Speech Synthesis 

**Authors**: Xingchen Song, Di Wu, Dinghao Zhou, Pengyu Cheng, Hongwu Ding, Yunchao He, Jie Wang, Shengfan Shen, Sixiang Lv, Lichun Fan, Hang Su, Yifeng Wang, Shuai Wang, Meng Meng, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2603.19798)  

**Abstract**: Most existing text-to-speech (TTS) systems either synthesize speech sentence by sentence and stitch the results together, or drive synthesis from plain-text dialogues alone. Both approaches leave models with little understanding of global context or paralinguistic cues, making it hard to capture real-world phenomena such as multi-speaker interactions (interruptions, overlapping speech), evolving emotional arcs, and varied acoustic environments. We introduce the Borderless Long Speech Synthesis framework for agent-centric, borderless long audio synthesis. Rather than targeting a single narrow task, the system is designed as a unified capability set spanning VoiceDesigner, multi-speaker synthesis, Instruct TTS, and long-form text synthesis. On the data side, we propose a "Labeling over filtering/cleaning" strategy and design a top-down, multi-level annotation schema we call Global-Sentence-Token. On the model side, we adopt a backbone with a continuous tokenizer and add Chain-of-Thought (CoT) reasoning together with Dimension Dropout, both of which markedly improve instruction following under complex conditions. We further show that the system is Native Agentic by design: the hierarchical annotation doubles as a Structured Semantic Interface between the LLM Agent and the synthesis engine, creating a layered control protocol stack that spans from scene semantics down to phonetic detail. Text thereby becomes an information-complete, wide-band control channel, enabling a front-end LLM to convert inputs of any modality into structured generation commands, extending the paradigm from Text2Speech to borderless long speech synthesis. 

---
# Dual Path Attribution: Efficient Attribution for SwiGLU-Transformers through Layer-Wise Target Propagation 

**Authors**: Lasse Marten Jantsch, Dong-Jae Koh, Seonghyeon Lee, Young-Kyoon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2603.19742)  

**Abstract**: Understanding the internal mechanisms of transformer-based large language models (LLMs) is crucial for their reliable deployment and effective operation. While recent efforts have yielded a plethora of attribution methods attempting to balance faithfulness and computational efficiency, dense component attribution remains prohibitively expensive. In this work, we introduce Dual Path Attribution (DPA), a novel framework that faithfully traces information flow on the frozen transformer in one forward and one backward pass without requiring counterfactual examples. DPA analytically decomposes and linearizes the computational structure of the SwiGLU Transformers into distinct pathways along which it propagates a targeted unembedding vector to receive the effective representation at each residual position. This target-centric propagation achieves O(1) time complexity with respect to the number of model components, scaling to long input sequences and dense component attribution. Extensive experiments on standard interpretability benchmarks demonstrate that DPA achieves state-of-the-art faithfulness and unprecedented efficiency compared to existing baselines. 

---
# FedPDPO: Federated Personalized Direct Preference Optimization for Large Language Model Alignment 

**Authors**: Kewen Zhu, Liping Yi, Zhiming Zhao, Zhuang Qi, Han Yu, Qinghua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19741)  

**Abstract**: Aligning large language models (LLMs) with human preferences in federated learning (FL) is challenging due to decentralized, privacy-sensitive, and highly non-IID preference data. Direct Preference Optimization (DPO) offers an efficient alternative to reinforcement learning with human feedback (RLHF), but its direct application in FL suffers from severe performance degradation under non-IID data and limited generalization of implicit rewards. To bridge this gap, we propose FedPDPO (Federated Personalized Direct Preference Optimization), a personalized federated framework for preference alignment of LLMs. It adopts a parameter-efficient fine-tuning architecture where each client maintains a frozen pretrained LLM backbone augmented with a Low-Rank Adaptation (LoRA) adapter, enabling communication-efficient aggregation. To address non-IID heterogeneity, we devise (1) the globally shared LoRA adapter with the personalized client-specific LLM head. Moreover, we introduce (2) a personalized DPO training strategy with a client-specific explicit reward head to complement implicit rewards and further alleviate non-IID heterogeneity, and (3) a bottleneck adapter to balance global and local features. We provide theoretical analysis establishing the probabilistic foundation and soundness. Extensive experiments on multiple preference datasets demonstrate state-of-the-art performance, achieving up to 4.80% average accuracy improvements in federated intra-domain and cross-domain settings. 

---
# MOSS-TTSD: Text to Spoken Dialogue Generation 

**Authors**: Yuqian Zhang, Donghua Yu, Zhengyuan Lin, Botian Jiang, Mingshu Chen, Yaozhou Jiang, Yiwei Zhao, Yiyang Zhang, Yucheng Yuan, Hanfu Chen, Kexin Huang, Jun Zhan, Cheng Chang, Zhaoye Fei, Shimin Li, Xiaogui Yang, Qinyuan Cheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19739)  

**Abstract**: Spoken dialogue generation is crucial for applications like podcasts, dynamic commentary, and entertainment content, but poses significant challenges compared to single-utterance text-to-speech (TTS). Key requirements include accurate turn-taking, cross-turn acoustic consistency, and long-form stability, which current models often fail to address due to a lack of dialogue context modeling. To bridge this gap, we present MOSS-TTSD, a spoken dialogue synthesis model designed for expressive, multi-party conversational speech across multiple languages. With enhanced long-context modeling, MOSS-TTSD generates long-form spoken conversations from dialogue scripts with explicit speaker tags, supporting up to 60 minutes of single-pass synthesis, multi-party dialogue with up to 5 speakers, and zero-shot voice cloning from a short reference audio clip. The model supports various mainstream languages, including English and Chinese, and is adapted to several long-form scenarios. Additionally, to address limitations of existing evaluation methods, we propose TTSD-eval, an objective evaluation framework based on forced alignment that measures speaker attribution accuracy and speaker similarity without relying on speaker diarization tools. Both objective and subjective evaluation results show that MOSS-TTSD surpasses strong open-source and proprietary baselines in dialogue synthesis. 

---
# CAF-Score: Calibrating CLAP with LALMs for Reference-free Audio Captioning Evaluation 

**Authors**: Insung Lee, Taeyoung Jeong, Haejun Yoo, Du-Seong Chang, Myoung-Wan Koo  

**Link**: [PDF](https://arxiv.org/pdf/2603.19615)  

**Abstract**: While Large Audio-Language Models (LALMs) have advanced audio captioning, robust evaluation remains difficult. Reference-based metrics are expensive and often fail to assess acoustic fidelity, while Contrastive Language-Audio Pretraining (CLAP)-based approaches frequently overlook syntactic errors and fine-grained details. We propose CAF-Score, a reference-free metric that calibrates CLAP's coarse-grained semantic alignment with the fine-grained comprehension and syntactic awareness of LALMs. By combining contrastive audio-text embeddings with LALM reasoning, CAF-Score effectively detects syntactic inconsistencies and subtle hallucinations. Experiments on the BRACE benchmark demonstrate that our approach achieves the highest correlation with human judgments, even outperforming reference-based baselines in challenging scenarios. These results highlight the efficacy of CAF-Score for reference-free audio captioning evaluation. Code and results are available at this https URL. 

---
# All-Mem: Agentic Lifelong Memory via Dynamic Topology Evolution 

**Authors**: Can Lv, Heng Chang, Yuchen Guo, Shengyu Tao, Shiji Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.19595)  

**Abstract**: Lifelong interactive agents are expected to assist users over months or years, which requires continually writing long term memories while retrieving the right evidence for each new query under fixed context and latency budgets. Existing memory systems often degrade as histories grow, yielding redundant, outdated, or noisy retrieved contexts. We present All-Mem, an online/offline lifelong memory framework that maintains a topology structured memory bank via explicit, non destructive consolidation, avoiding the irreversible information loss typical of summarization based compression. In online operation, it anchors retrieval on a bounded visible surface to keep coarse search cost bounded. Periodically offline, an LLM diagnoser proposes confidence scored topology edits executed with gating using three operators: SPLIT, MERGE, and UPDATE, while preserving immutable evidence for traceability. At query time, typed links enable hop bounded, budgeted expansion from active anchors to archived evidence when needed. Experiments on LOCOMO and LONGMEMEVAL show improved retrieval and QA over representative baselines. 

---
# AI Psychosis: Does Conversational AI Amplify Delusion-Related Language? 

**Authors**: Soorya Ram Shimgekar, Vipin Gunda, Jiwon Kim, Violeta J. Rodriguez, Hari Sundaram, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2603.19574)  

**Abstract**: Conversational AI systems are increasingly used for personal reflection and emotional disclosure, raising concerns about their effects on vulnerable users. Recent anecdotal reports suggest that prolonged interactions with AI may reinforce delusional thinking -- a phenomenon sometimes described as AI Psychosis. However, empirical evidence on this phenomenon remains limited. In this work, we examine how delusion-related language evolves during multi-turn interactions with conversational AI. We construct simulated users (SimUsers) from Reddit users' longitudinal posting histories and generate extended conversations with three model families (GPT, LLaMA, and Qwen). We develop DelusionScore, a linguistic measure that quantifies the intensity of delusion-related language across conversational turns. We find that SimUsers derived from users with prior delusion-related discourse (Treatment) exhibit progressively increasing DelusionScore trajectories, whereas those derived from users without such discourse (Control) remain stable or decline. We further find that this amplification varies across themes, with reality skepticism and compulsive reasoning showing the strongest increases. Finally, conditioning AI responses on current DelusionScore substantially reduces these trajectories. These findings provide empirical evidence that conversational AI interactions can amplify delusion-related language over extended use and highlight the importance of state-aware safety mechanisms for mitigating such risks. 

---
# Anatomical Heterogeneity in Transformer Language Models 

**Authors**: Tomasz Wietrzykowski  

**Link**: [PDF](https://arxiv.org/pdf/2603.19348)  

**Abstract**: Current transformer language models are trained with uniform computational budgets across all layers, implicitly assuming layer homogeneity. We challenge this assumption through empirical analysis of SmolLM2-135M, a 30-layer, 135M-parameter causal language model, using five diagnostic metrics: weight predictability (R2), ablation degradation, recovery speed, weight manipulation robustness, and structural analysis. We find profound anatomical heterogeneity: (1) Layer weights follow strong mathematical regularity (R2 = 0.91) with a universal oscillatory delta pattern (correlation ~= -0.50), yet predicted weights cause catastrophic failure due to nonlinear error accumulation. (2) Layer importance spans a 10^7 range, from a critical core (L8-11, up to +63,419% PPL degradation) to anti-layers (L14, L17) whose removal improves performance. (3) Recovery speed correlates with layer importance, indicating differential training requirements. (4) Only weight scaling (alpha = 0.9) preserves model quality among five tested manipulation strategies. (5) Growth Transformer Training, allocating budget by layer importance, achieves ~54% cost reduction. A proof-of-concept experiment confirms this: 4.7x lower validation loss than uniform training at identical parameter count, while being 13% faster. 

---
# Spectral Tempering for Embedding Compression in Dense Passage Retrieval 

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2603.19339)  

**Abstract**: Dimensionality reduction is critical for deploying dense retrieval systems at scale, yet mainstream post-hoc methods face a fundamental trade-off: principal component analysis (PCA) preserves dominant variance but underutilizes representational capacity, while whitening enforces isotropy at the cost of amplifying noise in the heavy-tailed eigenspectrum of retrieval embeddings. Intermediate spectral scaling methods unify these extremes by reweighting dimensions with a power coefficient $\gamma$, but treat $\gamma$ as a fixed hyperparameter that requires task-specific tuning. We show that the optimal scaling strength $\gamma$ is not a global constant: it varies systematically with target dimensionality $k$ and is governed by the signal-to-noise ratio (SNR) of the retained subspace. Based on this insight, we propose Spectral Tempering (\textbf{SpecTemp}), a learning-free method that derives an adaptive $\gamma(k)$ directly from the corpus eigenspectrum using local SNR analysis and knee-point normalization, requiring no labeled data or validation-based search. Extensive experiments demonstrate that Spectral Tempering consistently achieves near-oracle performance relative to grid-searched $\gamma^*(k)$ while remaining fully learning-free and model-agnostic. Our code is publicly available at this https URL. 

---
# Exploring Novelty Differences between Industry and Academia: A Knowledge Entity-centric Perspective 

**Authors**: Hongye Zhao, Yi Zhao, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19319)  

**Abstract**: Academia and industry each possess distinct advantages in advancing technological progress. Academia's core mission is to promote open dissemination of research results and drive disciplinary progress. The industry values knowledge appropriability and core competitiveness, yet actively engages in open practices like academic conferences and platform sharing, creating a knowledge strategy paradox. Highly novel and publicly accessible knowledge serves as the driving force behind technological advancement. However, it remains unclear whether industry or academia can produce more novel research outcomes. Some studies argue that academia tends to generate more novel ideas, while others suggest that industry researchers are more likely to drive breakthroughs. Previous studies have been limited by data sources and inconsistent measures of novelty. To address these gaps, this study conducts an analysis using four types of fine-grained knowledge entities (Method, Tool, Dataset, Metric), calculates semantic distances between entities within a unified semantic space to quantify novelty, and achieves comparability of novelty across different types of literature. Then, a regression model is constructed to analyze the differences in publication novelty between industry and academia. The results indicate that academia demonstrates higher novelty outputs, which is particularly evident in patents. At the entity level, both academia and industry emphasize method-driven advancements in papers, while industry holds a unique advantage in datasets. Additionally, academia-industry collaboration has a limited effect on enhancing the novelty of research papers, but it helps to enhance the novelty of patents. We release our data and associated codes at this https URL. 

---
# Maximizing mutual information between user-contexts and responses improve LLM personalization with no additional data 

**Authors**: Hyunji Nam, Haoran Li, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2603.19294)  

**Abstract**: While post-training has successfully improved large language models (LLMs) across a variety of domains, these gains heavily rely on human-labeled data or external verifiers. Existing data has already been exploited, and new high-quality data is expensive to collect. More fundamentally, true intelligence goes far beyond tasks that are easily verifiable. Therefore, we need self-improvement frameworks that allow models to improve without external oversight. We propose *Mutual Information Preference Optimization (MIPO)*, a contrastive data augmentation method that constructs preference pairs by generating a positive response conditioning on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. We show that using Direct Preference Optimization (DPO) to learn from this paired data maximizes pointwise conditional mutual information (MI) (under the base LLM) between prompts and model responses. Empirical results with various-sized Llama- and Qwen-Instruct models show that when used to maximize MI between user context and response, MIPO provides an effective personalization technique, achieving 3-40% improvements on personalization tasks using real-user datasets compared to strong baselines. Surprisingly, MIPO can also be applied to improve performance on math and multiple-choice problems, yielding 1-18% **without any additional data or human supervision**. These results suggest a promising direction for self-improvement. 

---
# Generalized Stock Price Prediction for Multiple Stocks Combined with News Fusion 

**Authors**: Pei-Jun Liao, Hung-Shin Lee, Yao-Fei Cheng, Li-Wei Chen, Hung-yi Lee, Hsin-Min Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19286)  

**Abstract**: Predicting stock prices presents challenges in financial forecasting. While traditional approaches such as ARIMA and RNNs are prevalent, recent developments in Large Language Models (LLMs) offer alternative methodologies. This paper introduces an approach that integrates LLMs with daily financial news for stock price prediction. To address the challenge of processing news data and identifying relevant content, we utilize stock name embeddings within attention mechanisms. Specifically, we encode news articles using a pre-trained LLM and implement three attention-based pooling techniques -- self-attentive, cross-attentive, and position-aware self-attentive pooling -- to filter news based on stock relevance. The filtered news embeddings, combined with historical stock prices, serve as inputs to the prediction model. Unlike prior studies that focus on individual stocks, our method trains a single generalized model applicable across multiple stocks. Experimental results demonstrate a 7.11% reduction in Mean Absolute Error (MAE) compared to the baseline, indicating the utility of stock name embeddings for news filtering and price forecasting within a generalized framework. 

---
