# Vision-Default, Prior-Override: Causal Mechanisms of Perception-Knowledge Conflict in Vision-Language Models 

**Authors**: Niclas Lietzow, Danielle Bitterman, Carsten Eickhoff, William Rudman, Michal Golovanevsky  

**Link**: [PDF](https://arxiv.org/pdf/2606.28273)  

**Abstract**: Vision-language models must reconcile visual evidence with memorized world knowledge when the two conflict. How they resolve this conflict shapes the reliability of multimodal systems, yet prior work characterizes it behaviorally without a component-level causal account. We combine activation patching across three granularities (residual stream, attention heads, and MLP sublayers) with model-component ablation studies and mechanistic analysis. Across three VLM families, we find that visual grounding emerges by default, whereas prior grounding depends on a small set of causally necessary attention heads (2.5-4.8%) concentrated in the second half of the network. These heads enable answers from stored world knowledge (e.g., "red" for a strawberry) despite conflicting visual input. Ablating them flips predictions from knowledge-grounded to visually grounded answers in 68-96% of cases under prior-knowledge prompts, but changes only 0.8-7.5% of visually grounded predictions, establishing an asymmetric causal structure. The identified heads decompose into routing heads, which modulate information flow, and writing heads, which directly project answer tokens into the residual stream. This structure is consistent across model families and scales, revealing a sparse causal circuit underlying perception-knowledge conflict in VLMs. 

---
# Cognitive Episodes in LLM Reasoning Traces Enable Interpretable Human Item Difficulty Prediction 

**Authors**: Chenguang Wang, Ming Li, Xinyue Zeng, Zhuochun Li, Hong Jiao, Tianyi Zhou, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.28186)  

**Abstract**: Predicting human item difficulty is central to educational assessment, where reliable estimates support fairness and effective test construction. Existing methods often depend on costly human calibration or item-level textual representations, providing limited evidence about the cognitive processes that make items difficult. We argue that difficulty should be viewed not only as a property of item text, but also as an observable consequence of the problem-solving burden an item induces. Large Reasoning Models (LRMs) offer scalable process evidence through reasoning traces, but such evidence must be structured to support interpretable modeling. To this end, we introduce Epi2Diff (Episode to Difficulty), a framework that maps LRM reasoning traces into cognitively grounded episode sequences. These episodes group trace segments into functional problem-solving states, enabling difficulty to be modeled through reasoning scale, effort allocation, and state transitions. Epi2Diff extracts compact episode-dynamic features and combines them with semantic item representations for human difficulty prediction. Experiments on four real-world human difficulty datasets show that Epi2Diff consistently outperforms strong baselines, including fine-tuned small language models, LLM in-context learning, and supervised LLM adaptation. On SAT-derived classification benchmarks, Epi2Diff achieves an 8.1% average relative gain over supervised LLM fine-tuning baselines. Further analyses show that harder items induce more effortful, iterative, and implementation-centered episode dynamics, rather than merely longer responses. These results demonstrate that cognitive episodes in LRM reasoning traces provide a predictive and interpretable process representation for human item difficulty, offering a new lens for educational measurement with reasoning models. 

---
# From Tokens to States: LLMs as a Special Case of World Models and the Continuous Path Beyond 

**Authors**: Paul Dubois  

**Link**: [PDF](https://arxiv.org/pdf/2606.28127)  

**Abstract**: The AI community has framed the relationship between large language models (LLMs) and world models as a dichotomy: LLMs predict tokens; world models simulate reality. Yann LeCun argues in 2022 that reaching general intelligence requires abandoning autoregressive token prediction in favour of latent-space architectures. This framing is unnecessarily binary. Two claims will be defended. First, LLMs are a degenerate special case of world models: the state space is the set of all token sequences, the only action is appending one token, and world models are therefore a strict generalisation of LLMs, not a replacement. Second, there is a natural continuous spectrum from NTP to JEPA, with multi-token prediction, future-summary prediction, and next-latent prediction as intermediate stations already populated by current research. Moving along this spectrum relaxes the LLM constraints one by one. It also progressively surrenders the two practical advantages that make LLMs trainable at scale: internet-scale self-supervised data, and a transformer architecture co-designed for discrete token prediction. Both are examined as open research questions: the data question (the cliff from self-supervised text to instrumented action-labelled environments) and the architecture question (whether the transformer generalises to continuous-state prediction, or whether a new primitive is needed). 

---
# Mechanism-Driven Monitors for Preemptive Detection of LLM Training Instability 

**Authors**: Ruixuan Huang, Yipei Wang, Wenyi Fang, Hantao Huang, Yifan Huang, Ansheng You, Zhenxing Zhang, Shuai Wang, Fan Wu, Yang Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.28116)  

**Abstract**: Frontier large language model training consumes massive accelerator fleets and long wall-clock computation, making stability failures costly when they occur. After a numerical or a hyperparameter fault has already destabilized the training dynamics, it may continue for thousands of steps while loss and gradient norms still appear normal. We study mechanism-driven detection of training instability by deriving internal monitors from the functional role of each critical module and from the earliest computational sites where failures are expected to produce measurable signatures. For low-precision flash attention, we monitor the spectral entropy of a QK bilinear decomposition, whose first-order term becomes abnormal before the loss fully collapses. For MoE routers, we derive indicators from their role in expert selection. Our fault-injection experiments on low-precision attention, large learning-rate, and combined faults show that these signals provide distinct signatures for different failures, triggering thousands of steps before loss divergence. 

---
# MultiHashFormer: Hash-based Generative Language Models 

**Authors**: Huiyin Xue, Atsuki Yamaguchi, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2606.28057)  

**Abstract**: Language models (LMs) represent tokens using embedding matrices that scale linearly with the vocabulary size. To constrain the parameter footprint, prior work proposes hashing many tokens into a single vector within encoder-only models. While this offers parameter efficiency, many-to-one collisions prevent its use in causal LMs. In this paper, we propose MultiHashFormer, a new framework that allows hash-based autoregression. Each token is represented as a unique hash signature, a short sequence of discrete hash IDs, generated by multiple independent hash functions. A Hash Encoder compresses this signature into a single latent vector for processing by a Transformer decoder. Then, a Hash Decoder generates the hash signature of the next token, which is then mapped back to text. We evaluate our approach at the 100M, 1B and 3B parameter scales, demonstrating that MultiHashFormer consistently outperforms standard Transformer LMs across multiple benchmarks. Furthermore, we show that our model handles multilingual vocabulary expansion with a constant parameter footprint without any modifications. 

---
# Can LLMs Judge Better Than They Generate? Evaluating Task Asymmetry, Mechanistic Interpretability and Transferability for In-Context QA 

**Authors**: Sambaran Bandyopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2606.28050)  

**Abstract**: LLM-as-a-Judge and self-evaluation pipelines implicitly assume that evaluation is easier than generation. We test this in a controlled in-context QA setting where a context passage is the sole information source and each model judges the answer it generated, removing the parametric-knowledge confound of open-domain comparisons. Across four benchmarks (SQuAD 2.0, DROP, HotpotQA, MuSiQue) and two models, evaluation is not uniformly easier: generation accuracy exceeds self-evaluation on three of four, with multi-hop MuSiQue the exception. Attention analysis reveals why: evaluation attends to context 3--5x less than generation does and barely reads the candidate answer. LoRA fine-tuning confirms the asymmetry is not a training artifact: generation fine-tuning induces over-acceptance and evaluation fine-tuning degrades generation. These findings challenge core assumptions in self-evaluation pipelines. 

---
# A Tree-of-Thoughts Inspired Hybrid Approach for Legal Case Judgement Summarization using LLMs 

**Authors**: Aniket Deroy, Kripabandhu Ghosh, Saptarshi Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2606.28044)  

**Abstract**: In recent times, Large Language Models (LLMs) are increasingly being used for legal case judgement summarization. Most prior works have tried traditional extractive and abstractive summarization of case judgements. However, hybrid or extractive-abstractive techniques have not been explored much. In this work, we propose a novel tree-of-thoughts inspired extractive-abstractive summarization approach for legal judgement summarization. We conduct experiments using two popular LLMs, DeepSeek and LLama, and compare among extractive, abstractive and extractive-abstractive summarization. Our experiments show that the proposed extractive-abstractive prompt provides better summaries compared to other types of LLM prompts. 

---
# The Signal-Coverage Matrix: Stratifying Type and Semantic Errors in Statement Autoformalization 

**Authors**: Chengxiao Dai, Zhaokun Yan, Zhanhui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.28013)  

**Abstract**: Headline type-correctness (TC\%) of LLM autoformalization has climbed from $\sim$53\% to $\sim$76\% in two years, yet this scalar conceals which errors each method resolves. We propose a signal-coverage matrix that crosses the Lean elaborator (pass/fail) with a semantic-equivalence judgment (equivalent/not), sorting every output into one of four cells: true success (TS), type-only (TO), semantic-only (SO), or both fail (BF). On ProofNet\# and MiniF2F-test with DeepSeek V4-Pro across Vanilla, Lean-Retry, Sample-Filter, and Stratified Autoformalization (SAF): (1) the +34 to +36 TS gain across the three elab-feedback methods is $\sim$64\% type-stratum recovery, with SO flat on net (87.5\% of original semantic errors rescued, 8 newly created). (2) The TO-to-TS rate is 23/61 for each method (Wilson 95\% CI [26.6\%, 50.3\%]), and this stratum-level recovery rate predicts $\Delta$TS on held-out methods to within 2/186 and renders $\Delta$TC linear in the Vanilla elab-fail rate across six (model, dataset) cells ($R^2=0.96$). (3) The two judges disagree by 26 to 37 pp on elab-feedback outputs (vs. 7 pp on Vanilla), with 30 to 56\% of symbolic-judge false negatives traceable to elaborator-forced rewrites. The persistent residual reduces to two gold-formalization errors. TC\% gains should be credited by which cell moved, not by the scalar alone. 

---
# Dialogue to Detection: A Multimodal Hybrid NLP Pipeline for Insurance Fraud Detection 

**Authors**: Muhammad Shakeel Akram, Amal Htait, Abdul Hamid Sadka, Emma Meisingseth, Karishma Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2606.28002)  

**Abstract**: Insurance fraud imposes substantial financial losses and operational inefficiencies, raising premiums and impacting trust among legitimate policyholders. Early detection at FNOL remains a persistent challenge. Existing approaches rely largely on private, text-only datasets, limiting progress on multimodal methods that integrate linguistic, behavioural, and speaker-based indicators. We introduce a synthetic multimodal framework that replicates FNOL conditions. It generates agent-customer dialogue transcripts and two-speaker audios, performs ASR and diarisation. Downstream modules combine NER, regex-based feature extraction, LLM-RAG retrieval, and speaker embeddings in a rule-based risk score to flag narrative reuse, structural inconsistencies, and cross-case voice repetition while balancing sensitivity and false positives. Dataset validation and component-level evaluations show stability and transfer potential, offering a reproducible baseline beyond text-only fraud detection. 

---
# ToxiREX: A Dataset on Toxic REasoning in ConteXt 

**Authors**: Stefan F. Schouten, Ilia Markov, Piek Vossen  

**Link**: [PDF](https://arxiv.org/pdf/2606.27981)  

**Abstract**: We introduce a new, contextual, multilingual dataset called ToxiREX: Toxic REasoning in ConteXt. The dataset consists of threads of Reddit comments and structured characterizations of what the comments imply, following a systematic toxic reasoning schema developed in a previous paper. Using the schema allows us to capture and explain implicit and context-dependent toxicity, while supporting mappings to existing toxicity taxonomies. The dataset includes comments in six languages (English, Arabic, Turkish, Spanish, German, and Dutch), collected from posts connected to specific major events (e.g. the 2023 Turkey earthquakes; the Russian invasion of Ukraine). We describe the context-preserving preprocessing of the threads. We create a training set of 125 thousand comments which is annotated by a commercially available LLM, and a test set of just under three thousand comments that is annotated by native speakers. We show that apparent disagreements in the test set annotations often reflect defensible alternative interpretations rather than noise. Finally, we provide baseline results by prompting and fine-tuning language models. To produce these results, we develop evaluation strategies for our hierarchical, schema-based predictions. While models perform better than random, there remains a lot of room for improvement, showing the task to be challenging. ToxiREX is the first dataset to simultaneously incorporate multiple languages, conversational context, and implicit toxicity, while using the toxic reasoning schema for rich, structured annotations. Dataset available at: this https URL 

---
# From Black-Box to Clinical Insight: A Multi-Stage Explainable Framework for Speech-Based Cognitive Impairment Detection 

**Authors**: Yasaman Haghbin, Sina Rashidi, Ali Zolnour, Fatemeh Taherinezhad, Ali Fartoot, Hossein Azadmaleki, James M Noble, Maryam Dadkhah, Maryam Zolnoori  

**Link**: [PDF](https://arxiv.org/pdf/2606.27973)  

**Abstract**: Speech-based cognitive impairment detection offers a noninvasive, accessible alternative to costly biomarker assays, yet transformer-based models remain clinically uninterpretable. We propose a multi-stage explainability framework that translates black-box transformer predictions into clinically grounded narratives by integrating SHapley Additive exPlanations (SHAP)-based token attribution, theory-informed linguistic features, and a four-stage LLM reasoning pipeline using LLaMA-3.1-70B-Instruct. Built on the SpeechCARE-Adaptive Gating Network multimodal screening model (F1 = 72.11% on the NIA PREPARE benchmark), the framework maps model outputs to four cognitive-linguistic dimensions, including lexical richness, syntactic complexity, and semantic coherence. Physician evaluation on 70 stratified English samples demonstrated strong alignment with patient-level cognitive profiles, and a System Usability Scale score of 82/100 indicated high potential for clinical workflow integration. 

---
# An Empirical Analysis of Factual Errors in Human-Written Text and its Application 

**Authors**: Kazuma Iwamoto, Kazumasa Omura, Shotaro Ishihara  

**Link**: [PDF](https://arxiv.org/pdf/2606.27959)  

**Abstract**: Factual Error Detection (FED), which is the task of identifying factually incorrect spans in a given text, has long been recognized as an important research problem. However, with the rapid rise of large language models (LLMs), research attention has shifted toward factual errors specific to LLM-generated text (hallucinations) and their detection. As a result, the detection of factual errors in human-written text has been relatively neglected. To address this gap, we first distill a taxonomy of human-induced factual errors by analyzing corrections of newspaper articles, a representative source of text that is guaranteed to be human-written and contains few grammatical errors. Our analysis revealed that there are characteristic categories such as kanji misconversions and numeral classifier errors, which are not focused in existing hallucination benchmarks. Based on the taxonomy, we then evaluate the FED capability of vanilla LLMs on synthesized realistic test cases and real corrections. Experimental results demonstrated that even high-performance LLMs such as GPT-5.4 achieved only word-level F1 score of 52% on the synthetic evaluation data, highlighting the task difficulty. Furthermore, a detailed analysis by detection difficulty revealed the current state of FED. 

---
# VASAE: Naming SAE Dictionary Directions with Vocabulary-Aligned Anchoring 

**Authors**: Kairui Zhang, Ziwen Yu, Zahraa S. Abdallah, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2606.27941)  

**Abstract**: Sparse autoencoders (SAEs) provide useful decompositions of Transformer residual streams, but their learned features are usually named post hoc rather than directly connected to the Transformer's token vocabulary. We introduce Vocabulary-Aligned Sparse Autoencoder (VASAE), a method that trains SAE features under vocabulary-aligned anchoring and assigns each feature an intrinsic token name: the token string whose embedding is nearest to that feature. Without reducing reconstruction quality compared with a standard SAE, VASAE produces dictionaries with vocabulary-aligned features. Using a 0.8 cutoff on the nearest-token alignment score, dictionaries trained on GPT-2-small post-residual streams align about 90% of features in layers 0--10. In Llama-3.1-8B, representative shallow and middle-layer dictionaries contain strongly aligned features, including 92.8% in the shallow layer, while the representative final-layer dictionary shows limited alignment. After subtracting the sentence-level mean sparse code, case studies show that many remaining intrinsic token names are relevant to nearby input tokens. These results suggest that vocabulary-aligned anchoring can connect learned features to intrinsic token names during training, complementing post hoc interpretation of learned dictionaries. 

---
# Triadic Werewolf: A Jester Role for Multi-Hop Theory of Mind in LLMs 

**Authors**: Avni Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2606.27909)  

**Abstract**: Theory-of-mind evaluations of large language models typically use dyadic social-deduction games, where every observable cue points to a single hidden side, so a model with strong language priors can score well without ever simulating opponents' incentives. We extend the Werewolf game with a Jester, a third faction whose utility on peer suspicion is inverted because it wins by being voted out, so optimal play requires reasoning across three opposing utility functions. Across 60 games on GPT-4.1, DeepSeek-V3.1, and Llama-3.3-70B with Jester self-learning on and off, the Jester wins 60-70% of games while Werewolves never exceed 20%, and GPT-4.1 wolves vote the Jester out on day 1 in 60-70% of games, a strictly self-defeating action. Self-learning helps DeepSeek and Llama but hurts GPT-4.1, with the cost landing on Villagers rather than Werewolves. Only DeepSeek learns the subtle strategy of looking suspicious without looking intentionally suspicious, and it gains the most from the loop. Triadic incentive structure exposes a layer of multi-agent reasoning that dyadic deduction games leave invisible. 

---
# A Study of Temporal Fusion Strategies for Named Entity Recognition in Historical Texts 

**Authors**: Emanuela Boros  

**Link**: [PDF](https://arxiv.org/pdf/2606.27881)  

**Abstract**: Temporal variation poses a unique challenge for named entity recognition (NER) in historical texts, where entities drift in surface form and salience across time. While language models (LMs) have made progress in various NLP tasks, their ability to reason about temporality, especially in diachronic contexts, remains limited or at least, questionable. In this paper, we systematically study how temporal metadata can be structurally embedded into NER models using a range of lightweight fusion strategies. We experiment with both absolute and relative temporal representations, injected into Transformer-based architectures via early or late fusion mechanisms such as cross-attention, adapters, and concatenation. Our evaluations on French and German historical datasets reveal that late fusion strategies yield more robust and temporally generalisable performance, particularly in early and noisy periods. 

---
# Learning Complementary Action Modeling from Automotive Maintenance Instructions 

**Authors**: Jiaqi Wu, Bai Li, Jochen Hartmann, Martin Gaedke, Sander Stuijk  

**Link**: [PDF](https://arxiv.org/pdf/2606.27808)  

**Abstract**: A minute lexical variation can reverse the procedural meaning of an instruction even when the rest of the sentence remains unchanged. In automotive maintenance instructions, this pattern often appears when an action phrase turns an instruction into its procedural counterpart. The entities, modifiers, and surrounding context remain largely invariant, while the action phrase determines the procedural relation. We define this task as Complementary Action Modeling (CAM). Given a maintenance instruction, the goal is to identify or generate its procedural counterpart by modifying the action phrase while preserving the remaining sentence context. This task focuses on three aspects: distinguishing complementarity from surface similarity, controlling generation at the action-phrase level, and evaluating relational correctness using retrieval, overlap-based, and human evaluation. Using a German automotive maintenance dataset, we examine these questions through candidate matching and controlled Seq2Seq generation. The results show that complementary maintenance instructions are best modeled as procedural associations grounded in subtle lexical cues. They should therefore not be treated as ordinary cases of sentence similarity or synonym-based paraphrasing. 

---
# Position Bias Correction is Insufficient for One-Pass Attention Sorting 

**Authors**: Qiong Tang, Xiangkun Hu, Xiangyang Liu, Yiran Chen, Yunfan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27793)  

**Abstract**: Long-context language models suffer from position bias, where information in middle positions is underutilized. Attention Sorting addresses this by iteratively reordering documents based on attention patterns, but its multiple sort-and-generate cycles increase deployment cost. We hypothesize that position bias is the primary bottleneck and propose Debiased One-Pass Attention Sorting, which estimates a per-prompt position-bias curve from the low-attention majority of documents and uses it to correct raw attention scores (via subtraction or division) to enable single-pass sorting. Our experiments on two models refute this hypothesis in the tested setting: on LLaMA-2-7B-32K-Instruct, debiasing produces identical results to uncalibrated single-pass sorting (94.83\% containment accuracy), while on YaRN-Llama-2-7b-64k, debiasing improves accuracy by 8.67 percentage points but remains 14.84pp behind iterative sorting, closing only 37\% of the gap. These results suggest that position-bias correction is insufficient to match iterative sorting, and that repeated reordering provides additional benefits beyond bias correction. 

---
# NLL-Guided Full-Attention Layer Selection for Training-Free Sliding-Window Adaptation 

**Authors**: Qiong Tang, Xiangkun Hu, Xiangyang Liu, Yiran Chen, Yunfan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27791)  

**Abstract**: Hybrid attention models that mix full and sliding-window attention across layers offer a promising approach to efficient long-context inference, but the critical question of \emph{which layers} should retain full attention remains unsolved. Existing methods use either fixed periodic patterns or attention-based heuristics that may not capture what matters for downstream accuracy. We propose NLL-guided layer selection, a training-free method that directly measures each layer's importance by computing the negative log-likelihood degradation on answer tokens when that layer uses sliding-window instead of full attention. On LongMemEval with Qwen3-4B, our method achieves 64.6\% accuracy using only 1/4 full-attention layers, matching the 1/2-FA periodic baseline (65.0\%) while halving the computational budget. NLL-guided selection outperforms the SWAA-reported periodic 1/4-FA baseline by 10.4 percentage points and a matched LightTransfer-style baseline by 26.4 percentage points. De-confounding analysis shows the signal is consistent with long-range attention needs rather than generic layer sensitivity. The method requires only $\sim$15 minutes of one-time calibration, advancing the efficiency-accuracy Pareto frontier for long-context LLM deployment. 

---
# SHIFT: Gate-Modulated Activation Steering for Knowledge Conflict Mitigation in Retrieval-Augmented Generation 

**Authors**: Ruochang Li, Pengcheng Huang, Zhenghao Liu, Yukun Yan, Huiyuan Xie, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.27786)  

**Abstract**: Retrieval-augmented generation (RAG) enhances LLMs by incorporating external knowledge to support response generation. However, conflicts between retrieved context and parametric knowledge have emerged as a critical challenge in RAG systems. To mitigate such conflicts, numerous studies have attempted to identify and edit knowledge-related internal neurons, aiming to improve the ability of LLMs to rely on contextual evidence during generation. However, these neuron-level approaches may introduce unintended cascading effects that compromise the general capabilities of LLMs, as the modified neurons are often entangled with broader model behaviors and functionalities. In this paper, we introduce SHIFT, a novel framework that reformulates neuron-level modification as learnable gate modulation, allowing LLMs to adaptively regulate internal activations for knowledge conflict resolution. Technically, our SHIFT equips LLMs with a lightweight gate module and optimizes fewer than 0.01% trainable parameters while keeping the backbone model frozen. During generation, the gate module adjusts the model's internal representations to adaptively leverage contextual and parametric knowledge. Extensive experiments on six datasets validate the effectiveness of our SHIFT in comparison with various competing baselines. All datasets and code are available at this https URL. 

---
# Output-Space Allocation Costs for Calibration-Guided LLM Compression: An Empirical Study 

**Authors**: Qiong Tang, Xiangkun Hu, Xiangyang Liu, Yiran Chen, Yunfan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27785)  

**Abstract**: Training-free compression methods for large language models (LLMs) often use calibration data to guide compression decisions. ROCKET, a recent method combining sparse-dictionary factorization with multi-choice knapsack problem (MCKP) allocation, derives its per-layer factorization from an output reconstruction objective but uses weight-space Frobenius error as the MCKP allocation cost. We investigate whether aligning the allocation cost with the output-space objective improves compressed model fidelity. On Qwen3-8B at 50\% compression, our ROCKET-ActCost achieves +0.8 percentage points higher average accuracy across 8 zero-shot benchmarks (53.1\% vs 52.3\%), but increases WikiText perplexity by 16\% (61.46 vs 52.98). This accuracy-perplexity tradeoff reveals that different allocation objectives favor different downstream metrics. The high correlation ($>$0.99) between weight-space and output-space errors limits allocation divergence, explaining the modest effect size. On Llama-3.2-1B at 20\% compression, the two methods produce near-identical results (53.3\% vs 53.5\% accuracy, 14.45 vs 14.66 PPL), suggesting that the effect of the cost function is minor at lower compression ratios. 

---
# KG2Cypher: Data-Centric Pipeline for Building Enterprise Text-to-Cypher Systems 

**Authors**: Minjun Choi, Yerin Kim, Junghyuk Seo, Sujin Mo, Hyemin Lee, Youngjoong Ko  

**Link**: [PDF](https://arxiv.org/pdf/2606.27742)  

**Abstract**: Enterprise Knowledge Graphs (KGs) are increasingly used for internal search, analytics, and question answering, but building natural-language interfaces for private enterprise graphs remains costly. We present KG2Cypher, a data-centric pipeline for building enterprise text-to-Cypher systems from existing KGs. KG2Cypher first constructs an executable Cypher query from observed graph facts and then uses LLMs to generate its associated natural-language question. The resulting Text-Cypher pairs are validated with an LLM judge and human validation, and are converted into candidate-aware SFT data. The trained generator is served with class-conditioned schema prompting, entity retrieval, and LoRA-based inference. We evaluate KG2Cypher in Korean enterprise settings, where short search-style queries and schema paraphrases make language grounding difficult. LoRA SFT improves execution-result F1 from 0.806 to 0.950 on broadcast-program queries and from 0.70 to 0.92 on company queries. In an 11-class setting, KG2Cypher achieves 95.2% exact match, 99.9% execution rate, and 0.964 execution-result F1. 

---
# Enhancing Numerical Prediction in LLMs via Smooth MMD Alignment 

**Authors**: Zhuo Zuo, Li Yue, Wenhao Zheng, Chenpeng Wang, Xianggen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27731)  

**Abstract**: Despite their strong general capabilities, large language models (LLMs) often remain unreliable when outputs must be numerically precise. A key reason is the training objective: standard cross-entropy treats numeric tokens as unstructured categories and ignores the metric structure of their values. We address this mismatch with Smooth Maximum Mean Discrepancy (SMMD), which builds on the classic MMD by incorporating value-distance kernels over numeric tokens and graph-based smoothness. With this kernel defined over a numeric sub-vocabulary, SMMD aligns the predicted numeric distribution to the target via kernel matching and smooths the prediction-target residual over the induced kernel graph to encourage local consistency. We evaluate SMMD on four numeric-target tasks: mathematical reasoning, arithmetic calculation, clock-time recognition, and chart question answering, across multiple open-weight LLM and VLM backbones. SMMD consistently improves accuracy over both cross-entropy and recent numeric-target losses; analyses show complementary effects between MMD and smoothness and underscore the importance of distance-based kernel design. Code is available at this https URL. 

---
# Do Speech Emphasis Models Generalize across Languages and Emotions? 

**Authors**: Megan Wei, Deepali Aneja, Jiaqi Su, Yunyun Wang, Haonan Chen, Zeyu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.27717)  

**Abstract**: Prosodic emphasis varies across languages, emotions, and speaking styles, yet existing emphasis detection models are largely trained and evaluated on monolingual neutral read speech. We introduce MMEE (Multilingual Multi-Emotion Emphasis), a corpus of 10,000 professionally recorded expressive utterances (14.13 hours) across 7 languages and 34 emotion/style categories, with three-level perceptual labels (10 annotations per sample). We benchmark two state-of-the-art architectures under monolingual, cross-lingual, multilingual, cross-emotion, cross-dataset, and data-scale settings. Monolingual models show limited zero-shot transfer, degrading across typologically distant languages, while multilingual training substantially improves robustness. Models transfer robustly between high- and low-arousal emotions; bidirectional transfer between synthetic and perceptual benchmarks suggests shared prosodic structure; and performance stays robust even at smaller training scales. 

---
# Low-Agreeableness Persona Conditioning for Safe LLM Fine-Tuning 

**Authors**: Austin MY Cheung, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27709)  

**Abstract**: Recent work has shown that fine-tuning large language models (LLMs) for social warmth degrades factual reliability and increases sycophancy. We investigate a related but distinct failure mode: warmth fine-tuning also weakens adversarial safety, making models more susceptible to jailbreaks and harmful output generation. We examine whether this reflects an inherent consequence of empathetic adaptation or an artifact of data construction. To address this, we introduce a persona-driven rewriting pipeline that conditions user turns on low agreeableness and pairs this with warm, de-escalating assistant responses. Across three experiments on four models, our approach reduces jailbreak susceptibility and harmful output rates relative to generic warmth fine-tuning baselines, while preserving conversational warmth. Representational probing provides suggestive evidence that this conditioning reduces the geometric alignment between warmth and compliance directions in latent space. These results show that safer empathetic fine-tuning is achievable through data design alone, without safety labels, harm detectors, or changes to the training objective. 

---
# Mitigating Position Bias in Transformers via Layer-Specific Positional Embedding Scaling 

**Authors**: Changze Lv, Zhenghua Wang, Yiran Ding, Yixin Wu, Tianlong Li, Zhibo Xu, Muling Wu, Tianyuan Shi, Shizheng Li, Qi Qian, Xuanjing Huang, Xiaoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.27705)  

**Abstract**: Large Language Models (LLMs) still struggle with the ``lost-in-the-middle'' problem, where critical information located in the middle of long-context inputs is often underrepresented or lost. While existing methods attempt to address this by combining multi-scale rotary position embeddings (RoPE), they typically suffer from high latency or rely on suboptimal hand-crafted scaling strategies. To overcome these limitations, we introduce a layer-specific positional embedding scaling~(LPES) method that assigns distinct scaling factors to each layer. LPES achieves a more balanced attention distribution without fine-tuning model parameters or increasing inference delay. A specially designed genetic algorithm is employed to efficiently select the optimal scaling factors for each layer by incorporating Bézier curves to significantly reduce the search space. Extensive experiments demonstrate that LPES effectively mitigates positional attention bias and delivers consistent improvements across multiple long-context benchmarks, yielding up to an $11.2$\% accuracy gain on the key-value retrieval dataset. 

---
# Mitigating LLM-based p-Hacking by Preregistering for the Next LLM 

**Authors**: Maria Thomas, Kristina Gligoric, Nihar B. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.27687)  

**Abstract**: Large language models (LLMs) are increasingly used to generate, classify, and annotate data whose outputs feed downstream hypothesis tests. However, LLM-based research is easy to p-hack: a researcher can tune the prompts, decoding parameters, or output format until a desired result is reached. We propose a protocol to mitigate p-hacking in LLM-based research: preregistering the experiment and eligible models, and then running it on the first eligible LLM that is released after the preregistration. The researcher finalizes the procedure on current models, preregisters the analysis plan together with a set of eligible future models, and runs the confirmatory analysis on the first eligible model released afterward. Because this model does not exist at commitment time, it cannot be hacked against; furthermore, configurations that hack one model frequently do not transfer to the next. We evaluate the protocol on two tasks whose true values are known. Across 20 models from four providers and 11 LLM-analysis configurations, the protocol would have blocked successful transfer of the p-hack in 73.9% and 72.7% of cases in the two tasks. Additional analyses reveal that mitigation remains substantial under several stress tests. Finally, putting money where our mouth is, we followed our own protocol and preregistered our experiment. The preregistered experiment confirmed the protocol's effectiveness: out of the 7 configurations that hacked the prior model, the hacking failed to carry over in 6 configurations on the first eligible model released afterward. 

---
# From Signals to Transfer: A Factorised Study of Probe-Based Uncertainty Estimation in Large Language Models 

**Authors**: Ponhvoan Srey, Xiaobao Wu, Cong-Duy Nguyen, Quang Minh Nguyen, Duc Anh Vu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27679)  

**Abstract**: Probe-based uncertainty estimation (UE) has emerged as a prominent approach to detect hallucinations in Large Language Models (LLMs) by learning uncertainty from internal model signals. Yet, recent methods vary simultaneously across feature design, training data construction, and evaluation setting, obscuring what actually drives performance. To address this issue, we propose a factorised study of probe-based UE under matched conditions. Our results show that raw hidden states and attention features are difficult to outperform in-domain. However, under distribution shift, structured and compressed features are more robust, suggesting that in-domain performance alone is insufficient to measure progress. Furthermore, prompting and label construction significantly affect probe behaviour. Building on these best-practice findings, we train benchmark-based pretrained probes that transfer reasonably well to open-ended factual generation, providing a stable off-the-shelf baseline. Our work encourages more deployment-oriented evaluation of probe-based uncertainty estimators. The code repository is available at this https URL. 

---
# When Search Agents Should Ask: DiscoBench for Clarification-Aware Deep Search 

**Authors**: Yiling Tao, Shihan Deng, Meiling Tao, Pengzhi Wei, Zhichao Hu, Zhihao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27669)  

**Abstract**: Search agents powered by large language models (LLMs) are increasingly used to solve complex information-seeking tasks, requiring multi-step retrieval and reasoning to fulfill user goals. However, existing benchmarks often assume that user queries are complete and explicit, overlooking the fact that real-world search requests are frequently vague, underspecified, or even factually incorrect. In deep search scenarios, such ambiguity can propagate along multi-step reasoning chains and lead agents toward incorrect search trajectories. To address this gap, we introduce DiscoBench, a benchmark for clarification-aware deep search, designed to evaluate whether search agents can proactively identify ambiguity, ask effective clarification questions, and recover correct reasoning paths through user interaction. DiscoBench contains 211 samples and 463 ambiguity instances across 11 real-world domains, covering four ambiguity types. We further design a user simulator for multi-turn interaction and evaluate model performance from four perspectives: task utility, ambiguity detection, interaction strategy, and cost efficiency. Experiments on representative LLMs show that ambiguity detection and effective clarification are distinct capabilities, and that repeatedly searching instead of asking for clarification often performs worse than direct guessing, highlighting a critical gap between retrieval ability and interactive problem-solving in current search agents. 

---
# Yuvion LLM: An Adversarially-Aware Large Language Model for Content And AI Safety 

**Authors**: Ting Ma, Xiufeng Huang, Benlei Cui, Xiaowen Xu, Shikai Qiu, Ruijie Jian, Hongxing Li, Guanghui Wang, Longtao Huang, Haiwen Hong, Haolei Xu, Wenjing Jiang, Ziwen Xu, Zhaoyu Fan, Shaoxuan He, Chuxi Xiao, Yujian Li, Xinyue Chen, Chunyang Chai, Wenxuan Liu, Ziheng Wang, Dongjie Zhang, Yangfan Zhou, Libin Dong, Yupeng Cao, Xiaoqian Xia, Jing Wang, Zhe Jiang, Zhenan Ye, Guang Yang, Bin Liu, Wei Peng, Ziqiang Zhu, Meihui Lian, Kaiwen Lv Kacuila, Haidong Ding, Bingyu Zhu, Yan Wang, Hai Zhao, Xuan Jin, Wei Zhao, Pengfei Sun, Wei Wang, Huiming Zhang, Bin Li, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2606.27632)  

**Abstract**: As large language models are increasingly deployed in real-world systems, safety failures can still lead to harmful outputs and dangerous misuse. We argue that the essence of safety is adversarial: many failures arise not from natural inputs alone, but from strategic attempts to evade model policies and safeguards. However, existing general-purpose model development largely overlook this adversarial nature, and often remain insufficient for realistic safety scenarios involving planning, tool use, and multi-step reasoning, causing measured safety performance to overestimate real deployment robustness. To address this gap, we present Yuvion LLM, a large language model built for adversarially robust content safety and broader AI safety. Yuvion LLM treats adversarial robustness and agentic capability as first-class objectives. Its pipeline combines adversarially aware data construction, knowledge-enhanced continued pretraining, and policy-grounded multi-task safety post-training, including risk-aware supervised fine-tuning and reinforcement learning-based policy optimization, together with safety-aware agentic reinforcement learning for tool use and multi-step reasoning in complex safety scenarios. We further introduce the Yuvion LLM RiskEval (YLRE), a collection of 93 benchmarks across four evaluation categories, covering diverse open and internal evaluations with a focus on safety, adversarial robustness, and real-world capability requirements. Across these evaluations, Yuvion LLM demonstrates clear advantages on safety-focused benchmarks and particularly strong robustness under adversarial conditions, while maintaining solid overall capability. Notably, Yuvion-8B outperforms most state-of-the-art baselines, including substantially larger models such as GPT-5.4 and Qwen3-MAX, on several safety tasks. 

---
# Cross-Platform Chinese Offensive Comment Detection via Dual-Threshold Hard Example Mining 

**Authors**: Ruixing Ren, Junhui Zhao, Fangfang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27629)  

**Abstract**: Cross-platform deployment of offensive comment detection for Chinese social media suffers performance degradation. The paper proposes a dual-threshold hard mining method to address this. First, the clean-Chinese-base RoBERTa is finetuned on COLD to establish a binary baseline for fair comparison. Second, a three-class fine-labeled test set covering Weibo, Xiaohongshu, Tieba, and Zhihu is constructed, domain distances from the source are quantified using Jaccard and Proxy-A Distance, as well as the degradation bottleneck of the baseline under domain shift is systematically revealed. Herein, a dual threshold hard example mining strategy is proposed. High- and low-confidence error-prone samples are filtered from unlabeled corpora by prediction confidence. The model is secondarily finetuned under implicit contexts with merely a small set of manually labeled hard examples, realizing low-cost cross-platform domain adaptation. Experiments reveal significant performance gains of the optimized model across four platforms. 

---
# Masked Language Flow Models 

**Authors**: Iskander Azangulov, Kianoosh Ashouritaklimi, Leo Zhang, Simon Vary, Patrick Rebeschini  

**Link**: [PDF](https://arxiv.org/pdf/2606.27617)  

**Abstract**: Masked Diffusion Models (MDMs) promise fast, parallel language generation, but their reverse transition factorises across token positions -- an approximation that breaks down in the few-step sampling regime where parallel generation ought to provide the greatest efficiency gains. Flow Language Models (FLMs) sidestep this limitation by learning a continuous flow that transports noise toward clean sequences represented in Euclidean space, inducing a flow map that can be distilled for single-step generation. However, this makes complex tasks requiring multi-step reasoning problematic for FLMs, as FLMs are forced to decode every token during generation. To address this, we introduce Masked Language Flow Models (MLFMs), which incorporate masking into FLMs using a continuous stochastic interpolant to bridge partially masked and clean sequences. This design enables conditional generation via continuous flows and allows pretrained MDMs to be converted into MLFMs through a simple, lightweight adaptation. Leveraging this flexibility, we propose a novel sampler that alternates continuous denoising with the discrete unmasking of confident tokens to better support multi-step reasoning. We evaluate our approach on GSM8K and MT-Bench and find, for the first time, that flow-based language models can be scaled to solve downstream reasoning and instruction-following tasks. 

---
# Narrative-UFET: Narrative Generation for Ultra-Fine Entity Typing 

**Authors**: Mreedul Gupta, Advait Deshmukh, Ashwin Umadi, Matt Pauk, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2606.27598)  

**Abstract**: Ultra-fine entity typing (UFET) assigns highly specific types to entity mentions, but current approaches struggle with types in the long tail. We hypothesize that a key limitation is the reliance on sentence-level context, since disambiguating evidence is often spread across multiple sentences. Testing this has been difficult because all existing UFET resources are sentence-level. We present Narrative-UFET, a controlled extension of UFET in which each entity mention is paired with an automatically generated short, coherent narrative. Synthesizing narratives lets us isolate the effect of specific discourse properties. We experiment with two paired variants: one in which the entity's type is held constant across the narrative (Maintain) and one in which it shifts (Change). We show that narrative context yields consistent improvements on long-tail types over sentence-level baselines, with the Change variant providing the stronger signal. A comparison against naturally occurring contexts shows that synthetic narratives yield stronger gains, indicating that controlled discourse construction can surface signals that real text leaves implicit. Substantial room for improvement remains, suggesting open directions in both discourse modeling and narrative construction. 

---
# Ko-WideSearch: A Korean Breadth-Search Benchmark for Exhaustive Set Enumeration by Web Agents 

**Authors**: Minbyul Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2606.27595)  

**Abstract**: Web-agent benchmarks overwhelmingly measure depth -- pinning one obscure answer behind a chain of constraints -- while breadth, exhaustively enumerating a closed set and filling each item's attributes, is barely evaluated, especially outside English. Breadth is also hard to build: certifying that a gold set is complete and every cell correct is far costlier than checking a single answer. I introduce \textsc{Ko-WideSearch}, a Korean breadth-search benchmark built by an automated synthesize-and-verify pipeline. Each task names a set-parent entity -- a TV season, a dynasty, a league, an administrative region, an election -- and asks for its full membership plus a per-item attribute table, graded by Item-, Column-, and Row-F1. It spans 228 tables over 190 entities and sixteen categories across three difficulty tiers, set by two structural knobs I dial independently -- table width and a 2-D composite key -- so cross-product membership climbs from 0\% to 100\% across the tiers. A single normalization-aware comparator is shared between gold construction and grading, so stable date and count columns are not over-dropped on formatting alone. Across twenty web agents, the failure is consistent: agents recover the set but not the rows (e.g.\ Item-F1 92.8 against Row-F1 53.7), accuracy falls steadily as the knobs harden, and neither more search nor more spend closes the gap. Broken down by cell, the hard part is finding the right value, not formatting it: open-ended free-text cells fail most, while cells with a standard answer such as a date or a name usually come out right. 

---
# EntMTP: Accelerating LLM Inference with Entropy Guided Multi Token Prediction 

**Authors**: Carrie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.27550)  

**Abstract**: Multi-token prediction has been shown to increase data density during training, improve downstream text-generation quality, and serves as the defacto approach for self-speculative decoding. Existing foundation and open source models that use MTP heads commit to a static tree-based attention topology throughout the entire generation sequence, meaning the speculation depth, and thus the compute required during verification, stays constant regardless of the context. This is fundamentally misaligned with the entropy patterns of natural language where low-entropy regions often support reliable multi-step drafting, while high-entropy regions require more conservative speculation. To address this, we propose Entropy-guided Multi-Token Prediction (EntMTP), a training-free scheduler that toggles between tree-based attention topologies from a set of task-specific pareto-optimal trees conditioned on a running estimate of local generation entropy. By matching speculation depth to context predictability, EntMTP maximizes expected accepted-token throughput across the full distribution of generated text without sacrificing generation quality. When evaluated across Humaneval, ShareGPT, GSM8k, and Litbench benchmarks, EntMTP consistently achieves a 1.15x speedup against Hydra and peak speedup of 1.36x against Medusa baselines respectively. 

---
# The Context-Ready Transformer 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2606.27538)  

**Abstract**: We introduce the context-ready transformer, a new recurrent neural network architecture built from a D-layer transformer block that pre-contextualizes each token before it enters the block. During left-to-right generation, a correction network combines the previous position's block output -- a cached summary of past context -- with the current token embedding, so the tokenenters the block already contextualized rather than as a raw embedding. At sequential inference, the correction chain makes the architecture a recurrent neural network. For training, we unroll the correction process K times over the full sequence, processing all positions in parallel at each step. A pretrained transformer can also be converted to a context-ready model by adding a zero-initialized correction FFN and fine-tuning. We evaluate across widths, depths, block sizes, and two datasets, with all comparisons against standard transformers, variants, and ablations. A D=5 model beats a 12-layer transformer while generating 1.7x faster on an A100. With K=10, a single-layermodel (D=1) beats a 6-layer transformer with a 2.6x inference speedup, and sequential inference matches parallel K=10 to within 0.01 PPL. The architecture benefits most from wide representations and long contexts. On a pointer-chasing task, D=1 trained with BPTT solves all 10 composition levels, while standard transformers exhibit staircase-like depth dependence. 

---
# Supersede: Diagnosing and Training the Memory-Update Gap in LLM Agents 

**Authors**: Vedant Patel  

**Link**: [PDF](https://arxiv.org/pdf/2606.27472)  

**Abstract**: Large language model (LLM) agents operate over long, multi-session interactions in which facts change: a user moves, a price updates, a plan is revised. Acting correctly requires using the current value of a fact and discarding values that have been superseded. We isolate this ability on real conversational data and show that it is a distinct, unsolved failure. On the knowledge-update subset of LongMemEval, replacing an agent's full context with a bounded, self-maintained memory drops accuracy from 92% to 77% even on a frontier model (gpt-5.4), a gap that is statistically significant (paired McNemar p<0.005) and persists across model scale while full-context accuracy saturates near 92%. The bottleneck is therefore memory maintenance, not comprehension, and is not closed by a stronger model. We then ask whether this is merely an undersized memory, and find it is not: as the conversation grows 24x, accuracy falls further (from 68% to 28%), and granting the agent proportionally more memory yields no detectable recovery (28% to 28%, n=25). The failure scales with the length of the conversation, not the compression ratio. We release Supersede, an open reinforcement-learning environment (on the verifiers / prime-rl stack) that turns this measurement into a training signal: agents are rewarded for answering from the current value and penalized for stale ones. Finally, we close the loop and show the gap is trainable: GRPO fine-tuning a small open model (Qwen2.5-3B) on this environment nearly doubles its held-out supersession accuracy on real, unseen conversations (9.0% to 16.7%, a single run), along a monotonic checkpoint curve indicating the learned policy, not the harness, carries the gain. To our knowledge this is the first trainable environment whose reward targets temporal fact-currency, and the first evidence the supersession gap can be trained down, not only measured. 

---
# Developmental approach reveals the statistical learning of Neural Language Models: Transformers generalize from the most abstract statistical patterns 

**Authors**: Wang Bojun, Holly Jenkins, Elizabeth Wonnacott  

**Link**: [PDF](https://arxiv.org/pdf/2606.27460)  

**Abstract**: In this study, we use a developmental approach to investigate the statistical learning and mental representation of neural language models (NLM). A series of Generative Transformer models are trained on a synthetic grammar. The model states are saved at multiple stages in the course of training. Through analyzing how the internal representations of these models change in the developmental path, we found that NLMs acquire the most abstract global statistical knowledge at the beginning of learning and later acquire the relatively local statistical dependencies. This learning path contains many over-generalizations from the very beginning and these over-generalizations are gradually constrained in the later stage of learning. Based on this observation, we propose a new framework to explain the statistical learning and language cognition of NLMs. 

---
# Causal Connections: Leveraging Multilingual Fine-Tuning for Financial QA@FinCausal 2026 

**Authors**: Akash Kumar Gautam, Serhii Hamotskyi, Christian Hänig  

**Link**: [PDF](https://arxiv.org/pdf/2606.27446)  

**Abstract**: This paper describes team HSA_CORAL's submission to the FinCausal 2026 shared task on extracting cause-effect relations from financial narratives via extractive question answering in English and Spanish. We compare three modeling families: (i) encoder-only token tagging with multilingual BERT, (ii) encoder-decoder generation with multilingual BART, and (iii) decoder-only LLMs (Llama 3.1 and GPT variants) using prompt refinement, few-shot demonstrations, and supervised fine-tuning. Across settings, prompting and few-shot examples yield competitive performance, while supervised fine-tuning provides the largest gains. Our best system, GPT-4.1 Mini fine-tuned on combined English and Spanish training data, achieves a tied highest score on the English subtask (score 4.8140) and ranks third on Spanish (score 4.7753) under the shared task's LLM-as-a-judge metric. Overall, the results highlight the value of task-specific adaptation and multilingual fine-tuning for cross-lingual transfer in financial causality QA. 

---
# A Survey of Automated Presentation Coaching: Systems, Methods, and Open Challenges 

**Authors**: Wen Liang, Li Siyan, Zackary Rackauckas, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2606.27380)  

**Abstract**: Automated coaching for oral presentations sits at the intersection of computer-assisted pronunciation training (CAPT), prosody modeling, and speech synthesis, yet no prior work has systematically surveyed and compared existing systems along these dimensions. This survey reviews and categorizes automated presentation coaching systems, spanning pronunciation tutors, fluency and prosody coaches, multimodal trainers, and conference Q&A practice tools. We introduce a five-dimensional task taxonomy - covering segmental pronunciation, lexical stress, suprasegmental prosody, pacing, and content faithfulness - and explicitly map surveyed systems onto it to reveal coverage gaps. We further review the core technical methods these systems employ: TTS-based exemplar generation and diagnostic methods for pronunciation, prosody, and fluency assessment. Key open challenges include the scarcity of annotated presentation corpora, achieving accent-fair feedback across diverse L1 backgrounds, and delivering low-latency diagnostics for real-time rehearsal. 

---
# Position: The Term "Machine Unlearning" Is Overused in LLMs 

**Authors**: Sangyeon Yoon, Yeachan Jun, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2606.27379)  

**Abstract**: Large language models increasingly face demands to "forget" training data, knowledge, or behaviors due to regulatory deletion obligations, copyright/licensing disputes, and safety or product-policy requirements. This position paper argues that machine unlearning is overused as a term in LLM research and should be reserved for dataset-defined deletion: removing the training influence of a precisely specified forget set such that the resulting model is approximately indistinguishable from retraining without that data. We contend that many tasks currently labeled "unlearning" (e.g., refusal for harmful requests, entity/knowledge removal, or targeted suppression) pursue different, often policy-dependent objectives and therefore require different terminology and baselines (e.g., alignment, suppression, editing, obfuscation). We further argue that this confusion is not cosmetic: because papers make different implicit guarantees under the same label, metrics and benchmarks are frequently reused outside their intended scope, rewarding surface-level non-disclosure (e.g., low ROUGE/forget accuracy) even when retraining-equivalence is not tested and derived capabilities remain. We conclude by calling for stricter terminology tied to explicit guarantees and reference models, and for evaluations that match the claimed objective. 

---
# Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs 

**Authors**: Fahd Seddik, Fatemeh Fard  

**Link**: [PDF](https://arxiv.org/pdf/2606.27378)  

**Abstract**: We introduce an axiomatic evaluation framework for latent thought representations in LLMs, comprising metrics that are independent of downstream benchmark scores and reveal representational failures that benchmark accuracy masks. Existing evaluations conflate representation quality with model capacity. Therefore, failures cannot be attributed to the representation rather than to the model that processes it. We formalize four functional axioms (Causality, Minimality, Separability, and Stability) and define a quantitative measure for each, computed directly on the representation independently of downstream accuracy. We audit open-weight LLMs across 23 reasoning tasks (e.g., Spatial Reasoning, Factual QA). We find that no candidate satisfies all four axioms simultaneously, that the representations distinguish task type reliably but cannot distinguish between two questions within the same task, and that the representations encode little information beyond what is already present in the input embedding. The failure is consistent across dense, reasoning-distilled, and RL-trained model families, indicating that the gap is structural rather than a property of model size or training procedure. 

---
# Towards Automating Scientific Review with Google's Paper Assistant Tool 

**Authors**: Rajesh Jayaram, Drew Tyler, David Woodruff, Corinna Cortes, Yossi Matias, Vahab Mirrokni, Vincent Cohen-Addad  

**Link**: [PDF](https://arxiv.org/pdf/2606.28277)  

**Abstract**: Artificial intelligence is driving a revolution in scientific discovery, accelerating everything from hypothesis generation to mathematical theorem proving. However, this rapid acceleration is creating a systemic challenge: traditional human peer review cannot scale to match the influx of AI-assisted science. Ultimately, to resolve this tension, we must also deploy AI to accelerate the verification and review process itself. To frame the discussion around this transition, we propose a taxonomy consisting of four progressive levels of AI-human collaboration in scientific evaluation, and discuss various trade-offs involved with each.
As a step toward this future, we introduce the Paper Assistant Tool (PAT), an agentic AI framework built for deep scientific review and verification. PAT ingests full scientific manuscripts and produces a comprehensive evaluation, checking theoretical results, validating experiments, suggesting improvements, and identifying potential flaws. By utilizing inference scaling techniques, PAT is able to identify deeper issues than a single model call alone, achieving a 34% improvement over zero-shot recall on mathematical errors in the SPOT benchmark. Pilot deployments of PAT as a pre-submission tool for authors at two major Computer Science conferences -- STOC and ICML -- demonstrate its ability to identify critical errors and suggest substantive improvements to research papers. By catching errors early, PAT eases the cognitive burden placed on referees, while preserving their control over the outcomes of the review process. 

---
# HPRO: Hierarchical Progressive Reward Optimization via Preference Extraction for Emotional Text-to-Speech 

**Authors**: Sihang Nie, Xiaofen Xing, Rui Xing, Haoming Li, Ruitong Xiao, Jingyuan Xing, Baiji Liu, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28249)  

**Abstract**: Recently, Large Language Model (LLM)-based Text-to-Speech (TTS) models have achieved remarkable naturalness. However, the standard Supervised Fine-Tuning paradigm often converges to statistically averaged prosody, limiting emotional expressiveness. While preference-driven optimization offers a promising alternative, existing approaches suffer from two structural mismatches: information conflict, where content and emotion in a shared latent space produce conflicting gradients, leading to reward hacking and semantic degradation; and scale gap, where sparse sentence-level rewards struggle to guide dense frame-level generation. To overcome these challenges, we propose HPRO, a hierarchical progressive reward optimization framework. Within HPRO, we introduce the HD-Emo codec as a novel differentiable reward model to resolve the information conflict. It extracts speech into distinct content and style preference tokens, structurally isolating emotional optimization from semantic content. Building upon this structured preference space, HPRO bridges the scale gap by progressively aligning frame-, word- and sentence-level objectives. Experiments demonstrate that HPRO significantly enhances emotional expressiveness, while effectively preserving linguistic intelligibility. The code and audio samples are publicly available at this https URL. 

---
# Scaling limit of the Random Language Model 

**Authors**: Eric De Giuli  

**Link**: [PDF](https://arxiv.org/pdf/2606.28105)  

**Abstract**: We develop a quantitative theory of the Random Language Model (RLM), an ensemble of stochastic context-free grammars, in a scaling limit where the number of hidden symbols $N \to \infty$ while the grammar temperature $\tilde{\epsilon}_d \to 0$ at fixed $x = {\tilde\epsilon}_d \log N$. In this limit, the model admits a controlled description based on a large-deviation principle over rule-usage patterns. A semi-annealed approximation maps the problem to a class of Random Energy Models with nontrivial combinatorics.
We show that the RLM exhibits a condensation transition at a critical value $x_c=1/8$, below which rule usage concentrates and language statistics acquire a nontrivial dependence on corpus length. A second characteristic scale at $x=1/2$ marks the onset of entropy reduction from its maximal value. Across these regimes, we derive explicit scaling laws for the number of distinct rules, entropy, and related observables, identifying distinct scaling, saturation, and critical regimes controlled by the interplay of grammar size, corpus length, and temperature.
The theory resolves previous ambiguities regarding the existence of a thermodynamic transition and explains the slow approach to the large-$N$ limit as a consequence of the dependence on $\log N$. It further provides a unified framework in which universal statistical properties of language emerge from typical realizations of generative grammars, with implications for both natural language statistics and the behavior of large language models. 

---
# Single and Multi Truth Data Fusion using Large Language Models 

**Authors**: Hira Beril Kucuk, Norman W Paton, Jiaoyan Chen, Zhenyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28062)  

**Abstract**: Data fusion, also known as truth discovery, is a data integration problem that aims to determine the correct value or set of values for each attribute of an object when presented with potentially conflicting values from multiple sources. Data fusion tasks belong to two main categories: single-truth scenarios, where each attribute has only one correct value, and multi-truth scenarios, where multiple values can be valid simultaneously. This paper investigates the use of Large Language Models (LLMs) in data fusion tasks for tabular data. Various prompting strategies, encompassing both single-truth and multi-truth scenarios, are investigated empirically. Domain-dependent, domain-independent, zero-shot and one-shot prompts are evaluated on three different benchmark datasets. Experimental results demonstrate that LLM-based approaches outperform traditional unsupervised truth discovery methods, such as DART and LTM, across all datasets. The codebase of this study has been made publicly available on GitHub. 

---
# DG^VoiC: Speaker Clustering for Fraud Investigation under Real Call-Centre Conditions 

**Authors**: Muhammad Shakeel Akram, Amal Htait, Abdul Hamid Sadka, Emma Meisingseth, Karishma Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2606.28048)  

**Abstract**: Insurance fraud remains costly and operationally difficult, particularly in call-centre workflows where many customer interactions begin at FNOL. While recent fraud detection methods mainly rely on structured data, text, or images, repeated speaker identity across calls remains underused as an investigative signal. This paper presents DG^VoiC, a voice clustering framework for customer verification and cross-profile speaker linking on anonymised real call-centre audio. The approach combines sensitive information-aligned anonymisation, speech-focused preprocessing, sliding-window speaker embedding extraction, and cosine similarity based clustering to identify repeated speakers under real telephony conditions. The method was evaluated on 121 recordings, with a curated reference subset of 56 samples in 22 human-agreed speaker clusters. used for validation. The best configuration achieved 96% AMI, 95% ARI, 98% completeness, 100% homogeneity, and 99% V-measure. These results show that speaker clustering can provide a strong additional signal for fraud investigation by helping analysts verify speaker consistency and surface repeated voices across customers. 

---
# AI Persuasive Framing in Collective Dilemmas 

**Authors**: Anders Giovanni Møller, Alessia Galdeman, Arianna Pera, Luca Maria Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2606.27951)  

**Abstract**: AI agents are promising tools that can act as flexible behavioral nudges to enhance human cooperation in addressing large-scale societal problems. However, evidence on whether AI agents can effectively boost cooperation remains mixed. We recruited 1,283 participants to play iterated Collective Risk Games in small groups, testing whether AI assistants could nudge participants toward cooperation. By using persuasive framing personalized to each player's Social Value Orientation profile, the AI interventions significantly increased contributions and group success rates. These cooperative effects were short-lived, however, fading after the first few rounds. Strikingly, when the AI treatments were reconfigured to promote selfish behavior through exculpatory framing, the negative effects on contributions and group success were larger and substantially more persistent, particularly for personalized interventions. This asymmetry between prosocial and antisocial persuasion highlights the dual-use risks of AI systems designed to influence group behavior in collective action settings. 

---
# Verifiable Geometry Problem Solving: Solver-Driven Autoformalization and Theorem Proposing 

**Authors**: Can Li, Ting Zhang, Junbo Zhao, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27926)  

**Abstract**: Geometry Problem Solving have increasingly adopt the neuro-symbolic paradigm, combining neural intuition with symbolic rigor. However, current frameworks suffer from severe bottlenecks in two core stages: autoformalization, which treats multimodal translation as a static task decoupled from downstream solver compatibility, and theorem prediction, where solvers frequently hit a deductive impasse due to fixed rule libraries. To address these, we propose SD-GPS, a solver-driven framework that treats the symbolic solver as an execution oracle throughout both formalization and deduction. First, Solver-Driven Autoformalization unifies supervised formal-language adaptation and solvability-guided reinforcement learning into a single module built on QwenVL3-2B, making executability the central training signal. Second, Verified Theorem Proposing introduces an impasse-aware agent that proposes local auxiliary lemmas from current proof states, ensuring soundness by filtering all proposals through symbolic verification. Empirical evaluations on Geometry3K and PGPS9K demonstrate that SD-GPS consistently outperforms existing MLLM, neural, and neuro-symbolic methods across standard completion, multiple-choice, and cross-modal reference regimes, proving that closing the loop between multimodal perception and symbolic execution significantly improves geometric reasoning, offering profound insights into how neural agents can be grounded by formal systems to achieve verifiable problem-solving capabilities. 

---
# Joint Transcription and Decryption of Images of Encrypted Handwritten Documents: A Comparison with the Traditional Pipeline 

**Authors**: Marino Oliveros-Blanco, Lei Kang, Alicia Fornés, Beáta Megyesi  

**Link**: [PDF](https://arxiv.org/pdf/2606.27700)  

**Abstract**: Historical encrypted manuscripts present a challenging problem at the intersection of cryptology, linguistics, paleography, and computer vision. Current automatic decipherment approaches usually rely on a two-stage pipeline: transcription of cipher symbols from manuscript images, followed by decryption into plaintext. However, this design is sensitive to transcription errors, which propagate to the final output. We present Direct Image Decryption, an end-to-end approach that directly maps encrypted manuscript images to plaintext, bypassing the intermediate transcription stage. Using the Copiale cipher as a case study, we build a synthetic data generation pipeline to create large-scale cipher-like training data and compare the traditional pipeline with the proposed joint architecture. Results show that joint image-to-plaintext modeling is a promising alternative to traditional transcription-based pipelines. 

---
# Textual Belief States for World Models: Identifiable Representation Learning Under Strict Mediation 

**Authors**: Xiang Gao, Kaiwen Dong, Yuguang Yao, Padmaja Jonnalagedda, Kamalika Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.27681)  

**Abstract**: World models in partially observed environments rely on latent representations that summarize interaction history, but in many modern LLM-based architectures predictive performance fails to reflect representation quality due to history bypass, rendering the latent state unidentifiable. Strict latent state mediation, requiring predictions to depend only on the latent state and action, is a classical principle that resolves this, but enforcing it in text-based settings is an open challenge: textual latent states are discrete and non-differentiable, precluding variational training, and expressive LLM decoders readily ignore the bottleneck. We show how to make strict mediation work in the text domain. We formalize why it is necessary, showing that strict mediation makes representation quality empirically testable while history-leaky architectures break this connection. We then introduce textual latent states, which are discrete, interpretable, and variable-length, and factorized GRPO (fGRPO), a tree-structured reinforcement learning method that enforces strict mediation during training. Experiments on TextWorld and ScienceWorld show preserved one-step prediction accuracy alongside up to 57\% gains in representation quality and 98\% improvements in rollout performance, increasing with task complexity and horizon. 

---
# DysLexLens: A Low-Resource LLM Framework for Analysing Dyslexic Learners Insights from Online Forums 

**Authors**: Dana Rezazadegan, Atie Kia, Phongpadid Nandavong, Dominique Carlon, Jeremy Nguyen, Abhik Banerjee, James Marshall, Anthony McCosker, Yong-Bin Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.27619)  

**Abstract**: Dyslexic learners increasingly use artificial intelligence (AI) tools to support reading, writing, organisation, and study-related tasks. However, their lived experiences with these tools remain largely underexamined. This paper proposes DysLexLens, a low-resource LLM framework, designed to analyse dyslexic learners experience with AI through online forum discussions. DysLexLens is designed as an end-to-end, evidence-traceable architecture which transforms noisy social media posts into a dictionary-driven corpora, provides knowledge-graph (KG)-based question reasoning, generates verifiable query responses, and enables response evaluation through quantitative and human-grounded assessment. DysLexLens has four key features. First, it employs a dictionary-driven filtering method to construct a more focused Reddit corpus on dyslexia and AI, filtering out noisy and weakly related posts to improve the relevance of data collected from low-resource forum contexts. Second, it integrates LLM-assisted semantic analysis with KG-based query reasoning to uncover meaningful patterns. Third, it has quantitative evaluation metrics (RAGAS and Query Robustness) to measure LLM-generated response performance. Fourth, it provides structured qualitative validation guidelines for assessing response quality, with a specific focus on hallucination and evidence alignment. We demonstrate the effectiveness of DysLexLens using dyslexia-related Reddit forum data and 30 questions. The results show its potential generalisability to other low-resource forum data contexts. DysLexLens, sample data, questions and evaluation results are available at Github to support reproducibility. 

---
# The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching 

**Authors**: Sankaran Vaidyanathan, David Arbour, Aaron Mueller, Scott Niekum, David Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2606.27510)  

**Abstract**: Activation patching is the primary tool in mechanistic interpretability. It attributes causal responsibility for a model behavior to each of its individual components by estimating its natural indirect effect (NIE). Re-deriving the activation patching estimand from causal mediation analysis, we find that the NIE does not solely capture the causal effect through the specific component. It also contains interaction effects (INT) that measure how much the component's causal effect itself depends on the state of other components in the model. A natural response may be to try to eliminate INT by adjusting the estimator or unit of analysis, but each of these potential remedies has predictable failure modes. We demonstrate these failure modes in the GPT-2 IOI circuit; components whose causal importance is conditional on the state of other components are either invisible or artificially inflated, and INT variance explains the previously documented instability of faithfulness scores. We prove that INT scales with the distance between clean and patched component activations, is negligible when the model is locally affine, and decomposes combinatorially into pairwise and higher-order group interactions. Despite its inevitability, INT is not a nuisance to be eliminated, but rather a diagnostic for interpretability studies. Its individual and group-level magnitude and sign signal when causal conclusions are prompt-dependent, and when greedy NIE-based component ranking will miss mechanisms only discoverable through combinatorial search. 

---
# Aloe-Vision: Robust Vision-Language Models for Healthcare 

**Authors**: Jaume Guasch-Martí, Enrique Lopez-Cuena, Martín Suárez-Fernández, Jordi Bayarri-Planas, Anna Arias-Duart, Dario Garcia-Gasulla  

**Link**: [PDF](https://arxiv.org/pdf/2606.27500)  

**Abstract**: Large Vision-Language Models (LVLMs) specialized in healthcare are emerging as a promising research direction due to their potential impact in clinical and biomedical applications. However, progress is constrained by the scarcity of high-quality medical multimodal data, concerns about robustness in safety-critical settings, and the narrow and potentially contaminated evaluation benchmarks that limit reliable assessment. To address these issues, the field requires state-of-the-art solutions to be fully open and reproducible systems in which all components can be inspected, evaluated, and improved. This work introduces Aloe-Vision-Data, a large-scale, quality-filtered mixture which integrates both medical and general domains across multimodal and text-only sources, designed for direct use in model fine-tuning. Building on this dataset, we train the Aloe-Vision family of medical LVLMs, openly released with full weights, training recipes and data, in two scales (7B and 72B). Through comprehensive benchmarking, we demonstrate that high quality training mixtures produce balanced LVLMs which yield significant gains over the baseline models without compromising general capabilities, achieving competitive performance with respect to state-of-the-art alternatives. To support reliable evaluation, we introduce CareQA-Vision, a carefully curated vision benchmark derived from MIR and EIR exams, the residency entrance exams for medical and nursing specialists in Spain, offering novel vision questions with low likelihood of contamination. Finally, we show that current LVLMs remain vulnerable to adversarial and misleading inputs, underscoring reliability challenges in clinical contexts. 

---
# DMV-Bench: Diagnosing Long-Horizon Multimodal Agents' Visual Memory with Incidental Cue Injection 

**Authors**: Yujin Tang, Chenming Shang, Ruize Xu, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.27499)  

**Abstract**: Research on agent memory has matured rapidly, but almost entirely on the text side: few existing benchmarks ask, in an interactive environment, when an agent genuinely needs to remember what it saw rather than what it could write down. We introduce DMV-Bench (Code: this https URL), the first interactive benchmark for multimodal-agent visual memory. DMV-Bench is built on a controlled home-furnishing e-commerce catalogue of 1,000 product variants in which a text-leakage contract keeps the discriminative signal of each task in the pixels alone. Across a chain of autonomous shopping sessions, every visited product image carries a unique, pre-rendered incidental cue, and the agent is later asked to recall a particular cued product and navigate to its URL. Inspired by dual-coding theory, we propose DualMem, a memory architecture that maintains a visual and a verbal code in parallel. On DMV-Bench, DualMem outperforms a caption baseline and three recent multimodal agent-memory systems at every chain length J in {5, 10, 15, 50} on both Gemini 2.5 Flash and Qwen2.5-VL-7B, with the lead surviving controls for memory-bank size and encoding-position bias, and an asymmetric dual-coding regime in which vision carries the cue end-to-end while the verbal channel plays a smaller query-grounding role. 

---
# Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving 

**Authors**: Yasmin Moslem, Magdalena Kacmajor, Vasudevan Nedumpozhimana, Ammar Abbas, Solmaz Panahi, David Lynch, Zhuangzhuang Nie, Alexandros Agapitos, Aleksandar Milenovic, Hongmeng Song, Yucheng Shi, Yue Pan, Patricia Buffini, John D. Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2606.27457)  

**Abstract**: Efficient deployment of large language models (LLMs) in production forces a trade-off between accuracy and cost. Operators often default to a single model that is either expensive for easy queries or insufficient for hard ones. To address this challenge, we propose a two-stage cascaded solution. Stage 1 clusters incoming queries and assigns each cluster to its most cost-effective model. The cost budget for this routing process is set by an interpretable hyperparameter, tuned offline. Stage 2 adds a quality estimation (QE) cascade; when an output from Stage 1 is judged low-quality, the query is escalated to a stronger model. This ensures only hard or low-confidence cases reach the expensive models. On the test datasets, the cascaded system retains 97-99% of the strongest model's accuracy while reducing Time Per Output Token (TPOT). It requires only task-correctness labels and adapts to changes in the model pool without manual reconfiguration. 

---
# Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placement 

**Authors**: Igor Itkin  

**Link**: [PDF](https://arxiv.org/pdf/2606.27409)  

**Abstract**: Multi-agent large language model (LLM) systems often rely on verifier and critic agents to suppress hallucinations, but verification is delayed. During this delay, false claims can propagate through the agent network. We model this process as delayed consensus on a graph with grounded corrector nodes. Spectral decomposition by the grounded Laplacian yields a closed-form stability threshold for the verification dose: correction that is too strong or too delayed can turn consensus into oscillation. The most unstable regime occurs when the communication and verification delays coincide; for delay two, the threshold is the inverse golden ratio. The same framework gives a supermodular placement objective and a greedy (1-1/e)-approximation rule for assigning a limited corrector budget to influential nodes. Experiments across five open models confirm the predicted dose-delay oscillations. By contrast, grounded factual answering makes truth an absorbing boundary and eliminates the effect, suggesting that the instability is specific to signed-belief tasks while grounded verification remains stabilizing 

---
# Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieval 

**Authors**: Leonardo Venuta, Francesco Tosoni, Paolo Ferragina  

**Link**: [PDF](https://arxiv.org/pdf/2606.27401)  

**Abstract**: Semantic code search and clone detection are essential for software development, maintenance, and reuse. This paper evaluates the effectiveness, efficiency, and scalability of contemporary deep learning models for first-stage recall in large-scale code-to-code search engines. Benchmarking across multiple programming languages and datasets reveals critical limits in the precision and scalability of these models on Terabyte-scale source-code collections. We present LLM-based code normalisation and query-rewriting schemes that yield significant gains in precision for lower-performing models. Our results question the sustainability of resource-constrained deployment and the assumed robustness of current code-specialised LLMs across datasets. We conclude with actionable insights for building scalable, efficient code-retrieval systems. 

---
# CalBrief: A Pilot Diagnostic Benchmark for Evidence-Calibrated Scientific Briefing with Large Language Models 

**Authors**: Yu Fu, Yongqi Kang, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.27383)  

**Abstract**: Large language models (LLMs) are increasingly used as research assistants, yet it remains unclear whether they can calibrate research takeaways to the strength and scope of the supporting evidence. We study evidence-calibrated scientific briefing: given a bounded package of related papers, a system should generate package-level takeaways with evidence strength, scope boundaries, and missing-evidence caveats. We contribute a verified pilot benchmark of 16 heterogeneous scientific evidence packages and 96 human-verified takeaways, and we use CalBrief, an auditable role/gap/strength framework, as a diagnostic probe to locate where briefing breaks down. Under a fair-schema evaluation, structured organization improves role and gap reasoning, but an explicit strength-calibration policy is systematically over-conservative and falls below majority and direct-LLM baselines. To explain why, we run a controlled diagnostic across three closed-model backbones (GPT-4o, Claude Sonnet, Gemini Flash) that separates three potential causes of conservatism. Approximately 63% of the conservatism gap is attributable to expanding the label space from binary {moderate, weak} to four-way {moderate, weak, uncertain, insufficient_evidence} (p < 0.001 across all backbones); only 1% is attributable to gap/scope signal injection (not significant); the remaining 36% arises from the pipeline policy itself. We also find that 4-way predictions can be post-hoc collapsed back to binary and then match or exceed direct binary prompting, so the extra labels carry information that strict matching hides. Label-level strength judgment and auditable evidence organization are distinct abilities currently in tension, and should be evaluated separately for LLM research assistants. 

---
