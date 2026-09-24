# Contrastive Learning for Authorship Verification 

**Authors**: Peter Kirby  

**Link**: [PDF](https://arxiv.org/pdf/2609.28471)  

**Abstract**: Our results show that contrastive learning outperforms a classification-based approach to authorship verification under the tested settings. We identify loss function, batch size, training duration, pre-trained model, input context length, and random text span data augmentation as important factors of model performance. Based on these considerations, we develop a ModernBERT Bi-Encoder model that achieves 98.4% accuracy on the PAN21 authorship verification task. 

---
# Cross-Scale Transfer Learning for Depression Severity Prediction: From PHQ-8 to HAMD-17 Across Languages and Clinical Paradigms 

**Authors**: Wenjie Feng, Sahba Zojaji, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2609.28430)  

**Abstract**: This work addresses continuous depression-severity score prediction from clinical interview transcripts under data scarcity. We propose a sequential low-rank adaptation (LoRA) protocol for cross-scale transfer: a Qwen3 backbone with a bounded regression head is first fine-tuned on the English DAIC-WOZ dataset (189 avatar-mediated sessions, PHQ-8), and the adapter then initializes fine-tuning on the Chinese PDCH dataset (100 real clinical consultations, HAMD-17), where a reinitialised, scale-specific head predicts the clinician-assigned score. All configurations use patient-level stratified 5-fold, 2-repeat cross-validation. On the data-scarce HAMD-17 target, the sequential protocol attains the best point-estimate MAE , RMSE, and macro-$F_1$ on both 0.6B and 1.7B backbones, outperforming target-only training and non-LLM baselines---4.96/6.59/0.36 with Qwen3-0.6B and 4.38/5.62/0.46 with Qwen3-1.7B. Ablations suggest that correctly aligned source supervision gives the best point estimates (unsupervised exposure and shuffled-label controls also show partial gains), that native-Chinese target input outperforms machine-translated English input, and that the reversed order yields no clear gain within run-to-run variance. The study is an exploratory, single-site internal evaluation: it does not establish screening or diagnostic utility, nor separately identify the contribution of the scale, language, or paradigm shifts. To our knowledge, no prior study evaluates this specific DAIC-WOZ-to-PDCH sequential transfer setting. 

---
# Agent-Editing World Model: Rethinking World Modeling for LLM Agents 

**Authors**: Shuang Sun, Guoxin Chen, Fanzhe Meng, Jia Deng, Huatong Song, Jinhao Jiang, Wayne Xin Zhao, Hongteng Xu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2609.28416)  

**Abstract**: Recent advances in large language models (LLMs) have enabled agents to tackle long-horizon tasks across diverse environments. To further improve agent performance, existing language world models typically predict environment observations, yet reconstructing high-entropy, execution-dependent tool responses offers limited value when real feedback is available. Meanwhile, agents suffer from \emph{task-state contamination}, where unsupported assumptions and outdated plans persist in history and distort subsequent decisions. We propose the \textbf{Agent-Editing World Model (AEWM)}, which models how reasoning and actions shape future task progress rather than simulating tool responses. AEWM combines \textbf{Action Judge} to distinguish \textsc{Critical}, \textsc{Exploratory}, and \textsc{Noisy} decisions with \textbf{State Revision} to edit noisy reasoning--action continuations from the same observed history. \textbf{EditAct} integrates these capabilities with real execution, directly changing the state underlying subsequent decisions rather than merely providing critiques. We train AEWM across Search, Terminal, and Software Engineering through mid-training and supervised fine-tuning. AEWM achieves 70.5\% macro-F1 on our Action Judge benchmark, exceeding the strongest frontier baseline by 10.6 points. Across six benchmarks and three agent backbones, EditAct improves average scores by 3.2--6.7 points over the strongest baseline. Furthermore, rejection sampling fine-tuning on verified EditAct trajectories, termed \textbf{AEWM-RFT}, improves over Self-RFT by 2.2--2.6 points across three domains without online AEWM guidance. 

---
# Fine-Tuning LLMs for Translation: General Forgetting Mitigation Does Not Preserve MT-Specific Instruction Following 

**Authors**: Niklas Scholz, David Thulke, Abdallah Nasir, Will Allred, Evgeny Matusov, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2609.28395)  

**Abstract**: Fine-tuning large language models on parallel data improves translation quality but can cause catastrophic forgetting. Mitigation methods are generally evaluated by retention on general benchmarks. We ask whether these findings transfer to machine translation (MT) fine-tuning and to MT-specific instruction following (MT-IF): instructions that modify a translation, such as formality, grammatical gender, and length control. We compare methods anchored to auxiliary data, to model outputs, and to the base model parameters, first in a screening study with Llama 3.2 1B Instruct, then on Llama 3.1 8B Instruct fine-tuned on bidirectional Arabic-English or Spanish-English data. Elastic Weight Consolidation preserves general capabilities best in both stages; on the 8B Spanish model the average score on general benchmarks drops 1.7 points versus 11.0 for standard fine-tuning, yet its scores for formality and grammatical gender control remain close to standard fine-tuning. Only data mixing with control-task examples preserves these controls, but its gains do not transfer to unseen prompts for the same task. 

---
# Digital diglossia: Arabic between X and Facebook 

**Authors**: Fahad Al Hussen, Mohammed Q. Shormani  

**Link**: [PDF](https://arxiv.org/pdf/2609.28352)  

**Abstract**: This study highlights the distribution of Standard Arabic (SA; H(igh) variety) and Colloquial Arabic (CA; L(ow) variety) across X and Facebook. 16754 public posts were collected via Python, with 10000 retained as the net dataset. Posts were classified into 7 discourse categories: *politics, technology, science, business, culture, fun,* and *sports*. Bivariate analyses, including Chi-square tests and Cramer's V (CV), examined associations among platform, discourse category, and diglossic choice, while binary logistic regression with Platform x Discourse Category interactions tested whether these associations varied across platforms. Findings reveal that there are significant associations between discourse category and diglossic choice on X, chi-square(6, *N* = 5000) = 600.35, p < .001, CV = .347, and Facebook, chi-square(6, N = 5000) = 1249.52, p < .001, CV = .500. Across platforms, platform was also associated with diglossic choice, chi-square(1, N = 10000) = 262.16, p < .001, CV = .162. Binary logistic regression further shows higher odds of SA use on X than Facebook in the political reference category (*OR* = 1.31, p = .0028), with significant platform-by-domain interactions for Culture (OR = 2.65), Fun (*OR* = 6.34), Sports (*OR* = 26.71), Science (OR = 0.41), and Technology (OR = 0.71). The study concludes that the diglossic use of SA and CA contributes to the growing body of research on digital discourse, unveiling that the digital age reshapes but does not erode diglossic boundaries, giving rise instead to a reconfigured digital diglossia. 

---
# Computation Over Geometry: Meaning Identity Is Computed, Not Shipped in the Embeddings 

**Authors**: Jiaqi Deng  

**Link**: [PDF](https://arxiv.org/pdf/2609.28290)  

**Abstract**: Meaning identity (whether two sentences say the same thing after wording changes) is treated in retrieval and RAG as a geometric fact about independently encoded sentence vectors. We show that, for frozen off-the-shelf encoders and language models, it is not: identity is computed when both sentences share one forward pass, and is not a property of the embedding geometry those systems ship. On overlap-matched PAWS-X, purpose-built encoders (BGE, E5, GTE, MiniLM, E5-Mistral-7B) reach English confirm AUC only 0.55-0.65 (dense peak 0.70). Independently encoded last-token states of Llama 3, Mistral, and Qwen do no better; late fusion of the two vectors stays near chance. The same probe on a joint forward pass reaches 0.90-0.96 from 1.5B to 32B, collapses under partner shuffle, is mid-depth, saturates near 0.94 by 3B, and appears more weakly in GPT-2 XL (0.76). The gap holds beyond Llama-style models on other causal LMs, bidirectional encoders (DeBERTa, RoBERTa), and encoder-decoders (Flan-T5, T5, BART). Fixed or linear readers over frozen independent encodings never unlock identity; nonlinear pair readers recover part of it only on the full 49k-pair PAWS train split (0.68-0.87). Off-the-shelf rerankers split: BGE-reranker-large reaches 0.94, while MS-MARCO and Jina stay at 0.55-0.64. Independently trained families compute the same relation and a 1.5B joint reader can distill it from unlabelled teacher scores, while no linear function of the teachers own independent vectors can. Bi-encoders can be fine-tuned to fit PAWS (0.87-0.93), but transfer and STS-B suffer. Cosine compares wording neighbourhoods; identity is a cheap computed operator, not a property of either sentence vector. 

---
# Towards Efficient Reasoning: Learning Causal Shortcuts for Diffusion Language Models 

**Authors**: Dian Jin, Kairong Han, Baohong Li, Xinpeng Dong, Zijing Hu, Nuanqiao Shan, Fei Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2609.28272)  

**Abstract**: Diffusion Language Models (DLMs) have attracted significant attention for their strong reasoning ability. However, under a bidirectional attention mechanism, DLMs operate over an exponentially large exploration space compared to autoregressive models (ARMs), making it challenging to focus on reasoning-guiding tokens under random masking. We define causal shortcuts as token chains that cover the full sequence and provide explicit guidance towards correct reasoning trajectories. We analyze the effects of causal shortcuts on the reasoning accuracy and convergence speed of DLMs, and find that they largely improve answer convergence efficiency and generation accuracy. Motivated by this, we propose a Causal Shortcut Learning (CSL) Framework for DLMs. Specifically, we introduce a step-by-step token extraction procedure to extract causal shortcuts from data, and apply parallel prioritized masking on these tokens during training to enable efficient and accurate convergence to correct answers via causal shortcuts. Extensive experiments across multiple reasoning benchmarks and two base models demonstrate that CSL consistently outperforms existing SFT-variant baselines, achieving an average improvement of $1.92\%$ over SFT-only models, and up to $4.20\%$ on MATH-500. The code is available at the \href{this https URL}{this https URL 

---
# Predicting Quantization Price for Selecting PTQ Configurations Before Deployment 

**Authors**: Junbin Qiu, Jian Mu, Weitong Zhang, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2609.28270)  

**Abstract**: Weight-space post-training quantization (PTQ) must choose finite formats, granularities, quantizer families, transformations, and bits before the completed quantized model reveals its output-distribution drift. Existing PTQ methods predict important pieces of this degradation, including reconstruction error, Hessian sensitivity, transformation effects, and downstream loss, but these pieces are usually scored after fixing the quantization geometry or inside separate configuration families. We formulate weight-space PTQ as pre-deployment configuration selection using priced layer-output error. Each admissible layer configuration is treated as an error generator with a deployment cost, which induces a layer-output error covariance $\boldsymbol{\Sigma}_l(\alpha_l)$, and the full-precision model prices that covariance by downstream curvature, $\widehat{\rho}_l(\alpha_l)=\frac{1}{2}\operatorname{Tr}\left(\widehat{\mathbf{H}}_l\,\widehat{\boldsymbol{\Sigma}}_l(\alpha_l)\right)$. The price follows from full-precision-to-quantized forward KL, whose first-order term cancels at the reference model. It turns reconstruction and diagonal scores into reduced proxies that drop price factors, while finite formats, codebooks, granularities, and equivalent transformations become comparable candidates through the covariances they induce and the costs they pay. A trace reduction then yields a calibration-time price table and a budgeted price-guided selector, making fixed-geometry bit allocation a special case rather than the organizing problem. 

---
# Complementary Roles of Activation and Parametric Memory in Few-Shot Learning 

**Authors**: Miaohe Niu, Runsong Zhao, Xinyu Liu, Bo Jin, Yucheng Qiao, Chunliang Zhang, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2609.28250)  

**Abstract**: At test time, large language models (LLMs) can encode historical information in activation memory (i.e., KV caches) and parametric memory (i.e., updated parameters). While activation memory is generally considered effective for factual recall and parametric memory for learning new tasks, their interplay remains unclear. In this work, we systematically investigate the role of memory in few-shot learning through controlled experiments. We find that activation memory is superior for recalling facts, whereas parametric memory does not consistently outperform activation memory in task learning. Moreover, our experiments show that the composite task, Conditional Arithmetic, requires the synergy of both memory types. Through neuron-level analysis, we find that the model activates distinct sets of neurons when accessing the same historical information through activation versus parametric memory. When both memory types are combined, the model recruits neurons from both sets, which is crucial for solving Conditional Arithmetic. These findings suggest that neither memory mechanism alone is sufficient for this composite task, highlighting the importance of their collaboration. 

---
# Beyond Poetry: Can Large Language Models Generate Classical Arabic Maqamat? 

**Authors**: AbdulRahman A. Morsy, Aya Zirikly  

**Link**: [PDF](https://arxiv.org/pdf/2609.28245)  

**Abstract**: Large language models (LLMs) have shown strong performance in creative text generation, yet their ability to produce culturally grounded and stylistically constrained literary forms remains underexplored. Prior work has focused largely on modern language varieties and poetry, while classical prose traditions such as maqama remain largely unstudied. The maqama is a classical literary genre characterized by rhymed prose (saj), dense rhetorical ornamentation, and episodic narrative structure, making it a challenging testbed for evaluating whether LLMs can move beyond surface fluency toward deeper literary competence. In this paper, we present the first controlled evaluation study of maqama generation with LLMs, comparing five models under zero-shot, few-shot, and rule-based prompting, and evaluating outputs through both human annotation and an LLM-as-a-judge framework across dimensions such as rhetorical richness, saj density, structural coherence, and stylistic authenticity. Our results show that prompting strategy plays a strong role in stylistic quality: few-shot prompting most consistently improves saj density, while its effects on rhetoric and coherence vary by model, with the strongest models (GPT-4o and GPT-5.4-mini) benefiting most from rule-based prompting on these dimensions, though zero-shot prompting yields the highest aggregate scores across all five models. We further observe systematic differences between models in stylistic alignment with Arabic maqama conventions, and corroborate our findings with a second independent LLM judge, paired statistical significance testing, and non-LLM proxy measures of saj. 

---
# Exact Feedback Is Not Control: Evaluating Text-based Closed-Loop Revision in LLMs 

**Authors**: Haitong Jiang, Chunlin Liu, Yile Wang, Yuhong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2609.28150)  

**Abstract**: Closed-loop revision is increasingly used in large language model (LLM) applications, but failures may reflect incomplete feedback or ineffective responses to correct feedback. We introduce a fixed-budget revision protocol with deterministic verifiers that report all remaining violations across exact-length, lexical, and compositional constraints. Fixing feedback correctness and completeness isolates model-side revision behavior. Across 19 open- and closed-source models, controller-level mean final joint success ranges from 17.4% to 99.8%, with substantial cross-model gaps persisting under identical initial drafts. Controlled experiments reveal reproducible model-specific responses to exact feedback. Post-training and scale reshape these responses without consistently bringing them closer to exact correction. Across all constraint families, failed trajectories often repeat earlier outputs, and prior recurrence is associated with lower subsequent recoverability. Matched-state interventions show that removing earlier dialogue while holding the current draft and feedback fixed changes recurrence escape without reliably improving final success; effects depend on the model, task, and trigger-state composition. Exact feedback makes revision errors observable, but does not make the closed loop reliable. Code and reproduction instructions: this https URL. 

---
# Scaling Attention Head Analysis via Gradient-Based Attribution in Context-Aware Machine Translation 

**Authors**: Paweł Mąka, Yusuf Can Semerci, Jan Scholtes, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2609.28117)  

**Abstract**: In this paper, we introduce a gradient-based head attribution strategy where the Token-level Max-Margin loss is backpropagated to the attention maps. This framework enables a large-scale causal analysis of attention heads, making it suitable for LLMs. We evaluate our method on the task of disambiguation in Context-aware Machine Translation, where we analyze 50 phenomena across 4 models and 4 language directions. We empirically show the alignment of our method with the effects of increasing the attention scores of token-to-token relations on three models and two language directions, ensuring the robustness of our method. Our analysis reveals the presence of the "general-purpose" attention heads that improve the model's performance when attending to different relations. We find that the average attention a head assigns to a relation does not necessarily relate to the model's performance, which suggests that the models developed redundancies during training in terms of the head functions. 

---
# Can LLMs Catch a Rigged Backtest? A Clean-Control Calibration Benchmark 

**Authors**: Makar Ulesov, Vladislav Smirnov, Omar Ibrahim, Arsenii Bobovnikov  

**Link**: [PDF](https://arxiv.org/pdf/2609.28090)  

**Abstract**: Backtest auditing is a calibration problem: high flaw recall is not useful when the model falsely flags matched clean strategies. We build a 96-item paired benchmark in which every flawed backtest has a clean control that holds strategy, dates, code style, labels, and reporting scaffold fixed while changing one methodology detail. A deterministic scorer separates flaw recall, clean-control false positives, evidence localization, and fix relevance. Over 1440 cached audits from four text endpoints, the primary DeepSeek auditor reaches 100.0\% closed and clean-aware code recall, but open prompts over-flag 93.8\% of clean code controls, and clean-aware all-three specificity is 87.5\% even where recall saturates. A clean-aware warning drops DeepSeek code false positives from 20.8\% (95\% CI 11.7--34.3) to 0.0\% (0.0--7.4) at unchanged recall, while the budget anchor still flags 38/48 clean controls under the same prompt. Reporting recall alone would rank three of these four models identically; reporting the clean-control rate separates them by 79 points. 

---
# Reference-Based Analysis of Coherence and Diversity in Open-Ended Text Generation 

**Authors**: Esteban Garcés Arias  

**Link**: [PDF](https://arxiv.org/pdf/2609.28080)  

**Abstract**: Evaluating open-ended text generation involves understanding how different properties of a continuation relate to its perceived quality. We present a reference-based framework for examining coherence and diversity through three perspectives: aligning their evolution with human trajectories, comparing their summaries with a human continuation of the same prompt, and estimating their likelihood under a human reference distribution. Experiments with human quality ratings suggest that diversity-based alignment and mean-based comparisons capture quality-related variation, although the comparisons do not establish a predictive advantage for temporal alignment over simpler baselines. Reference likelihood also shows positive associations with ratings, with results varying across reference configurations and scoring horizons. Together, these analyses provide a structured way to examine how measured coherence and diversity relate to human judgments, while distinguishing similarity to human references from quality itself. Code and analysis resources are available at this https URL. 

---
# A Native-Reference Coordinate Geometry for L2 Pronunciation Deviation Using Self-Supervised Speech Models 

**Authors**: Tina Raissi, Nhan Phan, Mikko Kurimo  

**Link**: [PDF](https://arxiv.org/pdf/2609.28060)  

**Abstract**: Self-supervised speech models encode rich phonetic information, but it remains unclear how to transform this information into interpretable metrics for second-language (L2) pronunciation assessment in spontaneous speech. We propose a native-reference coordinate geometry in which phone-class averages from native speech define a low-dimensional reference subspace, and L2 speech is evaluated by its distance to matching native phone-class coordinates. Unlike prior distance-based approaches, our method does not require parallel recordings with matched linguistic content or dedicated pronunciation labels. Across different self-supervised encoders and modeling choices, the resulting native-reference distances show negative Spearman correlations up to -0.5 with speaking proficiency, indicating that higher-proficiency speakers tend to lie closer to the native-reference space. 

---
# TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval 

**Authors**: Mourad Hassani, Julien Romero, Amel Bouzeghoub, Christian Jacquelinet  

**Link**: [PDF](https://arxiv.org/pdf/2609.28048)  

**Abstract**: Modern information retrieval (IR) systems rarely represent time, yet many information needs depend on it: in clinical, journalistic, and legal search, when an event occurred can decide whether a document is relevant. Dense retrievers and Retrieval-Augmented Generation (RAG) pipelines match queries to documents well on topic but poorly on time, so they surface content that is on-topic yet temporally wrong. We introduce Temporal Textual Similarity (TTS), a task that measures how well two anchored texts align in time, independent of their topical similarity. We then present TEMPS (Temporal Embedding Model for Precise Search), a modular temporal branch that attaches to a frozen semantic retriever and trains on that signal. It resolves anchored temporal expressions to intervals and moment-matches each one to a Gaussian; the resulting ordering supervises an anchor-date-conditioned encoder, whose score we fuse with the semantic score at inference. Grounding supplies the supervision, so training uses no hand-labeled temporal data. The temporal score itself is the Gaussian-KL inclusion measure from distributional embeddings; what TEMPS adds is the grounding and the moment-matched supervision. On three temporal benchmarks, TEMPS improves MRR for every semantic backbone tested and, on TS- Retriever, lifts R@1 from 19.92 to 25.39 over the prior temporal state of the art. 

---
# How Much Were You Told? Measuring External Information in Peer Reviews 

**Authors**: Matthieu Dubois, Pablo Piantanida, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2609.28041)  

**Abstract**: Conference policies distinguish using Large Language Models (LLMs) to polish one's own review from delegating the critique, but current Artificial Text Detection (ATD) methods largely measure surface form rather than the origin of its content. We instead measure the external information carried by a review: information not explained by the reviewed paper and a generic reviewing instruction. We propose Self-Conditioning, an unsupervised information-theoretic estimator that compares the likelihood of a review under its production context with its likelihood when that context is augmented with hints extracted from the review itself. On the IntelLabs peer-review benchmark, Self-Conditioning separates fully-delegated from machine-polished reviews with AUC up to $1.0$ while remaining largely insensitive to surface rewriting. Moreover, as generators receive increasing amounts of externally-provided information, their scores move monotonically towards the human regime, unlike standard ATD baselines. High-temperature sampling can evade the estimator, but at the cost of output quality. 

---
# Evaluating Feedback Focus and Pedagogical Adaptivity in LLM-Generated Feedback on Student Writing 

**Authors**: Norah Almousa, Shayan Peyghambari Oskoui, Raquel Coelho, Gayle Rogers, Xiang Lorraine Li, Diane Litman  

**Link**: [PDF](https://arxiv.org/pdf/2609.28026)  

**Abstract**: We investigate whether state-of-the-art large language models (LLMs) generate feedback that reflects the pedagogical practices of expert teachers in terms of feedback focus and adaptivity. Previous evaluation efforts have examined feedback characteristics, its impact on learning, and its target, yet the focus of feedback and its adaptivity remains largely overlooked. To bridge this gap, we adopt and refine Narciss's taxonomy into seven feedback focus types to annotate teacher and LLM-generated feedback across three university writing courses. We release FeedType, a benchmark containing annotated teacher and LLM feedback from six LLMs under three prompting strategies. We assess the coverage and distribution of feedback focus types, and examine whether LLMs adapt their feedback across draft stages and student performance levels as an expert instructor does. Our findings show that while most LLMs cover most feedback focus types, they fail to reflect teacher feedback distributions and show varying levels of adaptivity, with none matching the teachers' adaptive behavior. We believe FeedType will support future research on pedagogical alignment in LLM feedback generation. 

---
# Evaluating Open-Weight LLMs for Turkish Domain Documents Under Retrieval and Hardware Constraints 

**Authors**: Imtiaz Ul Hassan, Öykü Akbulut, Onur Kaya, Ardhendu Behera, Swagat Kumar, Peter Matthew, Yonghuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.28007)  

**Abstract**: Most Turkish-capable large language models (LLMs) are evaluated using general-purpose benchmarks rather than long, structurally complex domain documents. This paper evaluates five open-weight 7B-8B models for Turkish document question answering under a resource-constrained local deployment setting. The primary benchmark contains 100 systematically validated questions derived from a 109-page industrial R&D report, and the evaluation protocol is replicated using a second 112-page public-sector report and an independently constructed 100-question set. All models are evaluated locally on an NVIDIA RTX 3050 laptop GPU with 6 GB VRAM using controlled prompting, decoding, and 4-bit quantisation.
The principal methodological contribution is an evidence-annotated evaluation protocol that separates retrieval failure from downstream model reasoning failure without requiring additional model calls. On the primary benchmark, end-to-end accuracy ranges from 49% to 75%. Seven lexical, dense, and hybrid retrieval configurations are additionally compared using 95% Wilson intervals and exact paired McNemar tests; none significantly outperforms the character TF-IDF baseline on either document. Evidence recall saturates differently across the two reports, showing that retrieval and effective context capacity can be binding constraints for some documents but not others. These results demonstrate that model selection, retrieval behaviour, and hardware limits must be evaluated separately when deploying open-weight LLMs for Turkish domain documents. 

---
# Controlled Attribute-Specific Summarization of Interrogative Dialogues 

**Authors**: A Aditya Bhardwaj, Arjit Singh Arora, Md Shad Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2609.28004)  

**Abstract**: Effective summarization of interrogative dialogues is a critical task in forensic and investigative settings, requiring high factual accuracy, coherence, and attribute-specific relevance. In this work, we introduce CASPER, a novel Chain-of-Thought Attribute-Specific Prompting for Evaluative Summarization framework that leverages structured prompting and iterative refinement to generate high-quality summaries of interrogator-witness interactions. We construct MINDSum, a dataset extending the MIND corpus, comprising 6,000 utterance pairs annotated with event details, factual statements, character descriptions, and fillers. CASPER employs RoleEval, a hierarchical evaluation mechanism where multiple roles (officer, inspector, senior inspector) iteratively assess summaries based on predefined criteria. By integrating entity extraction and structured feedback loops, CASPER significantly improves factual consistency and contextual completeness compared to existing baselines. Experimental results demonstrate that our framework outperforms standard summarization models on both lexical (ROUGE) and semantic (BERTScore) metrics, while human evaluation confirms its alignment with expert reasoning. Our findings underscore the potential of controlled summarization in high-stakes domains, paving the way for AI-driven forensic intelligence. 

---
# Risk-Controlled KV-Cache Eviction: From Memory Budgets to Risk Targets 

**Authors**: Beomgu Kang, SoJin Yun, Hojoon Kim, Hyunseok Seo  

**Link**: [PDF](https://arxiv.org/pdf/2609.27981)  

**Abstract**: KV-cache eviction is typically evaluated through average quality-memory trade-offs, yet a small average loss can hide requests whose utility degrades materially. We reformulate eviction as a deployment risk-control problem: a material degradation occurs when eviction lowers task utility by more than a deployment-specified tolerance relative to full-KV inference on the same request, and deployment risk is the population frequency of such events. Given a reliability contract specifying a target risk level and confidence requirement, we use a compressor-agnostic post-hoc certification procedure to select a retention policy from calibration data with a finite-sample guarantee, falling back to full KV when no compressed policy is certified. Across multiple eviction methods, Llama and Mistral models, and LongBench and RULER-32K, the same contract supports substantially different levels of eviction: on Llama, it certifies SnapKV at 75% retention on LongBench but no tested compressed policy on RULER-32K, triggering full-KV fallback. Policies with empirical degradation rates below the 5% target can still fail finite-sample certification; on Llama LongBench, empirical thresholding selects uncertified policies that retain 5-10 percentage points less cache across fixed-budget methods. The proposed framework converts a deployment-level reliability requirement into a KV-memory operating point. 

---
# Six Layers Less: Encoder Pruning for Whisper with Label-Free Recovery 

**Authors**: Rasmus Aagaard, Nicki Skafte Detlefsen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27980)  

**Abstract**: Pruning large pre-trained transformer-based ASR models such as OpenAI's Whisper has seen great adoption, as pruning the decoder led to significant end-to-end transcription speedups. For instance, the {\tt whisper-large-v3-turbo} variant reduced the decoder from 32 to 4 layers, while Distill-Whisper similarly reduced the decoder to only 2 layers. Although some attention has been put towards reducing the size of the encoder, no approach has seen wide adoption. This could be due to the need for custom inference implementations to take advantage of the compressed model. We present an approach that ranks encoder layers by the leave-one-layer-out change in Word Error Rate (WER). The six layers that cause the least change are removed, corresponding to $18.5\%$ of the encoder stack. The pruned model requires no custom inference code as it is simply a more shallow encoder with fewer layers. We further distill using unlabeled monolingual speech data to recover performance degradation caused by the zero-shot layer pruning. Mean WER across four languages increases to $20.1\%$ after distillation, compared to $21.9\%$ zero-shot, going from a baseline of $18.2\%$. We release all of our code (this https URL) and the pruned model (this https URL). 

---
# From Sentiment Classification to Actionable and Responsible Feedback: A Scoping Review and Evidence Map of NLP in Student Evaluation of Teaching, 2015-2026 

**Authors**: Jeff Eicher, Rafael da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2609.27939)  

**Abstract**: Natural language processing (NLP) applied to open-ended teaching-evaluation comments (Student Evaluation of Teaching, SET) has tracked the field's technical evolution--from lexicons and conventional classifiers to transformers and large language models (LLMs)--but it is not evident that this technical diversification has been accompanied by corresponding gains in educational value and robustness of the evidence. This scoping review (PRISMA-ScR) maps 421 studies (2015-2026, 2026 partial) along a technical axis (RQ1) and four value dimensions (RQ2-RQ5). Dual mutually blinded LLM screening with sampled human adjudication coded seven extraction domains, with targeted codebook-boundary review at synthesis. The joint map's sharpest quantified gap is the actionability discontinuity: demonstrated output or stronger (A2+: 258/421; 61.3%) versus intended-user evaluation or stronger (A3+: 49/421; 11.6%), a 49.7 percentage-point drop. Sentiment analysis remains the modal task (300/421); diagnostic and generative depth is a substantial minority (D4-D5: 28.2% of resolved cases); a formal fairness metric is rare (1.9%). The findings are descriptive and do not support causal claims of progress: technological coexistence and uneven reporting are part of the map, but the A2+ to A3+ cliff is the contribution, not a quality ladder. 

---
# Learning When Not to Listen: Selective Anti-Interference Pretraining for Language Models 

**Authors**: Jinchang Zhu, Haowei He, Yi Ding, Rong Fu, Nie Xiaojian, Shuangyong Song, Zhongjiang He, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27925)  

**Abstract**: Language models can over-condition on irrelevant preceding text: predictions already supported by local context may still change when distant, unrelated prefix tokens are perturbed. This interference is especially consequential in long, packed, or distractor-heavy contexts, where useful evidence and irrelevant spans coexist. We propose Selective Prefix Anti-Interference Regularization (SPAR), a pretraining objective for selective anti-interference. SPAR runs the original sequence and a corrupt-prefix input in which only the far prefix is changed, then uses a short-context sufficiency gate and a gated KL objective to stabilize locally supported suffix predictions. The gate operationalizes a model-based estimate of whether the far prefix supplies additional information about the target token. Mechanism analyses show that the gate identifies locally sufficient tokens and sharply reduces prefix sensitivity on gate-selected suffix tokens. In continued training on pretrained base models, SPAR improves RULER across Qwen2.5-0.5B, Qwen2.5-3B, Llama-3.2-1B, Llama-3.1-8B, and GPT2-XL under equal counted training compute; pretraining experiments further show gains on both RULER and NoLiMa. These results show that selective anti-interference is an effective objective-level signal for robust context use. 

---
# Delegated Misalignment: How Multi-Agent Structures Amplify LLM Safety Risks 

**Authors**: Zonghao Ying, Jiaqi Yan, Huize Luo, Quanchen Zou, Aishan Liu, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27900)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-agent systems where a principal agent decomposes tasks and delegates them to subordinate agents that may invoke external tools. Safety alignment, however, is still evaluated almost exclusively under a single-agent threat model, treating safety as a property of the individual LLM. We show that this assumption breaks down: \emph{individual safety alignment fails to transfer to multi-agent settings}. Two failure mechanisms emerge under delegation: \emph{responsibility diffusion} on the principal side and \emph{role-bias compliance} on the subordinate side, jointly converting language-level refusal into actionable harm. We refer to this phenomenon as \textit{delegated misalignment} and study it through a three-condition protocol across 6 frontier LLMs on 49 hazardous tasks. Delegation amplifies end-to-end harm substantially: DeepSeek-V3.2's full-execution rate rises from 30.6\% to 77.6\% once delegation is introduced, and the same model behaves very differently across roles (GPT-5: 22.5\% as a single agent vs.\ 61.2\% as a subordinate). Ablations further show that standard single-layer defenses each fail on their own and can even backfire. We call on the community to move beyond per-model alignment and toward composite safety mechanisms before multi-agent LLM systems are deployed at scale. 

---
# "AI Is Turning Too Human": How Teenagers Experience and Negotiate AI in Everyday Life 

**Authors**: Jianfeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27824)  

**Abstract**: Generative AI is rapidly entering adolescents' everyday lives during a critical period of cognitive, social and emotional development. Yet its adoption is outpacing evidence on how adolescents themselves experience, understand and negotiate its expanding role in their lives. We examined AI-related discourse on r/teenagers from January 2023 to July 2026 using validated keyword-based retrieval and a human-in-the-loop, LLM-assisted thematic analysis. AI-related discussion increased substantially over time, and 11,083 analytically coded posts revealed eight interconnected domains of experience. Everyday and social use was most prevalent (36.8 percent), while discourse increasingly shifted toward authenticity, personal control and safety, and future human roles. Across domains, adolescents questioned when AI should support or substitute for human thinking and creativity, how conversational AI changes relationships and perceptions of agency, what can still be considered authentic, who controls personal information and representation, and what opportunities and roles should remain human. These findings position adolescent AI use not simply as technology adoption, but as an emerging negotiation over AI's place and boundaries in everyday life. Supporting this transition will require developmentally appropriate AI literacy, psychological and social support, and AI systems and policies that protect adolescents' agency, privacy, relationships and opportunities for human development. 

---
# LabourCrew: A Multi-Agent RAG Framework for Trustworthy Adversarial Deliberation and Statutory Reasoning over Labour Law 

**Authors**: Fatema Tuj Johora Faria, Mukaffi Bin Moin, Jubayer Al Mahmud, M. F. Mridha, Md. Alam Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2609.27814)  

**Abstract**: In statutory question answering, every claim must be traceable to evidence, not merely relevant, since unverifiable labour-rights answers carry serious legal consequences. Current systems fall short: single-pass RAG cannot detect insufficient evidence, while multi-agent legal-debate systems treat grounding as a prompting convention, letting agents cite unretrieved evidence. To address this gap, we introduce LabourCrew, a multi-agent RAG framework built around three grounding mechanisms: StatuteGraph, a graph index that explicitly links chapter, section, proviso, and cross-reference structure rather than fixed-length spans; an Evidence Exchange Protocol that confines advocates and an interpreter to an evidence ledger, making citation to unretrieved text impossible, while a fault-tolerant supervisor board runs advocates in parallel so individual failures degrade rather than crash the system; and a Calibrated Trust Gate that replaces categorical accept/reject decisions with a trust score, thresholded via conformal risk control for a distribution-free bound on the false-accept rate. We evaluate on LabourActQA, a 500-item Bangla question set from the Bangladesh Labour Act, 2006, spanning seven reasoning categories and three difficulty tiers. The framework drives the empirical false-accept rate to 0.081, within the target level ($\alpha = 0.10$), achieves the highest Answer Relevancy among HyDE RAG, Graph-RAG, and Hierarchical RAG (0.862 $>$ 0.839, 0.815, 0.828), and degrades gradually rather than catastrophically as question difficulty increases. These results show that calibrated abstention, not retrieval quality alone, is what makes legal question answering auditable in low-resource statutory domains. 

---
# Beyond Unsafe Detection: Counterfactually Anchored Evidence Attribution for Multi-Turn LLM Safety Failures 

**Authors**: Srinivasan Subramanian, Kazi Aminul Islam, Md. Abdullah Al Hafiz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27773)  

**Abstract**: As Large Language Models (LLMs) move from conversational assistants to advanced agentic systems, guardrail failures can convert adversarial intents into harmful executions. However, most guardrail evaluation frameworks focus only on the result and assess whether a user request is safe or unsafe. This approach is insufficient for multi-turn failures, where adversarial intent is distributed across multiple turns. This motivates us to go beyond detection to identify the turns and tokens that push the conversation toward unsafe trajectories. To support this, we construct a multi-turn dataset with behavioral validation and tiered evidence supervision. The dataset contains 1,762 conversations, including adversarial conversations, benign twins, and benign variants with high-risk vocabulary. We train a lightweight hierarchical attribution model that predicts safety violations and attributes them to contributing user turns and token spans. The model achieves strong detection performance (F1=0.988), and removing the top 15% of attributed tokens reduces the adversarial classification confidence by 51.1%. The model preserves low false positive rates on benign conversations with high-risk vocabulary, with false positives below 1% on both borderline benign and benign high-risk vocabulary conversations, compared to 37.3% and 94.7% for a keyword-based surface-risk baseline. Independent human annotation supports the model's attribution performance, with the top-five attributed turns containing a human-identified evidence-bearing turn in 84.5% of adversarial cases. 

---
# Improving LLM-based Autonomous Web Agents with Filtering 

**Authors**: Zhitong Guo, Jing Yu Koh, Ruiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.27770)  

**Abstract**: Autonomous web agents, powered by Large Language Models (LLMs), have garnered significant attention for automating various web-based tasks with multi-step reasoning and decision-making capabilities. An open research question in the development of these agents lies in the format of the webpage input. Raw HTML source code, with its extensive and often irrelevant details, poses difficulties for LLMs with limited context windows. To address this challenge, we first reproduce baseline models such as GPT-3.5 and LLaMA-2-70B on the WebArena (Zhou et al., 2023) benchmark, identifying common failure modes. We then propose two retrieval strategies to filter out irrelevant context for LLM agents. We develop DeBERTa-based and T5-based models that rank HTML elements by their relevance to the task. We fine-tune them on Mind2Web trajectory data and transfer them to WebArena. Experiments show that our DeBERTa-based model improves the success rate of the LLaMA-2-70B LLM agent on WebArena from 1.97% to 2.96%. Moreover, we develop a zero-shot ColBERT-based retriever that is able to retrieve the ground-truth element with a recall of 0.52 on Mind2Web and 0.47 on WebArena. 

---
# Hard Negatives Reveal What Easy Negatives Hide: Cross-Lingual Harmfulness Representations Degrade with Resource Tier Under Hard Negatives 

**Authors**: Paras Balani, Subhrakanta Panda  

**Link**: [PDF](https://arxiv.org/pdf/2609.27758)  

**Abstract**: Safety alignment in large language models is trained primarily in English, and recent work reports that the underlying harmfulness representation survives translation: English-trained probes separate harmful from harmless prompts almost as well in low-resource languages as in English. This has been taken as evidence that cross-lingual refusal failures mainly reflect calibration rather than representation quality. We show that this conclusion depends on the choice of negative examples. Across nine languages spanning three resource tiers, we replicate near-perfect transfer (AUROC > 0.98) when harmless prompts come from an unrelated distribution (easy negatives). With XSTest contrast prompts, which are benign but surface-similar to harmful requests (hard negatives), transfer collapses in low-resource languages while remaining largely stable in high-resource languages. On Qwen2.5-7B-Instruct, mean AUROC drop increases from 0.003 in English to 0.017 in high-resource, 0.042 in mid-resource, and 0.276 in low-resource languages. The pattern replicates on Aya Expanse. Back-translation chrF controls and a matched-chrF comparison across three languages reduce the likelihood that translation quality explains the effect. The collapse remains after controlling for chrF (partial r = 0.70, p = 0.03). Tokenizer fertility correlates with the collapse and explains part of the resource-tier effect, but not all of it. The results show that easy-negative transfer can coexist with substantial degradation under hard negatives. Easy-negative evaluation alone therefore cannot establish that the harmfulness representation survives translation. 

---
# SkillGym: Internalizing Human Skills into LLMs for Real-World Problem Solving 

**Authors**: Zhilong Ge, Yuting Shao, Yutao Yang, Yuxuan Cai, Jie Zhou, Kai Chen, Bo Zhang, Qin Chen, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2609.27717)  

**Abstract**: Human-written agent skills encode rich workflows for real-world problem solving, but are typically used as external inference-time instructions rather than internalized as reusable model capabilities. We introduce \texttt{SkillGym}, a framework that transforms these skills into executable, verifiable training environments for large language model agents. Its skill-to-task pipeline instantiates concrete tasks, verifies outcomes with code-based checkers, and assesses empirical skill dependence through contrastive executions. We construct and release 2,756 environments across 12 categories and collect 8,364 successful trajectories from multiple models and harnesses, averaging 49 tool calls and over 60k logged text tokens. These resources support supervised fine-tuning on verified workflows and reinforcement learning with outcome-based rewards. Under Claude Code, supervised fine-tuning improves Qwen3.5-35B-A3B by 199 Elo on GDPval-AA v2, 19.10 percentage points on Terminal-Bench 2.1, and 28.13 and 12.38 points on SkillsBench v1.1 with and without skills, respectively. Our 35B \texttt{SkillGym-Agent} reaches 51.47\% on skill-assisted SkillsBench, exceeding reported scores for Claude Sonnet 4.6, GPT-5.4 Mini, and DeepSeek V4 Pro. Without skills, it also surpasses skill-assisted bases under Codex and Claude Code, suggesting reusable procedural competence. 

---
# Consequential Behaviour and Representational Fairness in the Validation of Synthetic Research 

**Authors**: Florian Kutzner, Celina Kacperski, Laura de Molière, Edoardo Chidichimo, Min Jun Jung, Felix Patrick Sedgwick Wallis, James Kunling He  

**Link**: [PDF](https://arxiv.org/pdf/2609.27690)  

**Abstract**: Researchers in industry and academia use synthetic survey respondents powered by large language models as substitutes for human samples. These synthetic populations require validation against real-world data, so researchers often address them using ad hoc comparisons with human surveys. Inspired by the intention-behaviour gap in behavioural science, we argue that these validations test the wrong thing for most applied cases where decision makers commission synthetic research to anticipate consequential behaviour. To address this problem, we propose a validation framework with two requirements. First, every validity claim must state its level of correspondence with human data: does the sample predict what the represented people do, which of four diagnostics (location, dispersion, response process and structure) does the validation address, and does the validation compare against experimental effects? Second, researchers must report validity claims for subgroups, since these groups are often the most affected by consequential decisions and aggregate accuracy hides their misrepresentation. Our validation framework operationalises three justice dimensions (distributional, procedural, and recognition) as measurable quantities and defines within-persona counterfactual experiments as a validation requirement. We then apply the framework to electric vehicle charging tariffs, before closing with a reporting checklist that researchers can use to make convincing validity claims. 

---
# Same Scores, Different Decisions: Evaluating JEV and Language Models for Legal Document Understanding 

**Authors**: Fan Zhang, Yankai Chen, Zhuohan Xie, Yixi Zhou, Sijia Peng, Lei Fan, Xinhua Ji, Cunyuan Zheng, Huangyong Shan, Philip S. Yu, Xue Liu, Yu Chen, Preslav Nakov, Songwei He  

**Link**: [PDF](https://arxiv.org/pdf/2609.27678)  

**Abstract**: Contract inference requires multiple judgments about a shared document, but aggregate accuracy can conceal changes in the individual decisions. Repeated agreement is also insufficient: a model may consistently return the wrong answer. In this paper, we compare Jev with nine language models on ContractNLI, evaluating inference cost, response time, average correctness, and correctness across repeated request conditions. Controlled comparisons vary hypothesis visibility, requested outputs, and output order while keeping the contract and target judgment fixed. Jev has the lowest cost and median response time among the evaluated configurations, while hosted language models achieve higher baseline accuracy. Rankings by baseline accuracy differ from rankings by correctness across every condition and repeat, although small differences in the latter do not establish a general stability advantage. Development diagnostics further reveal compensating corrections and regressions, as well as persistent errors. These findings motivate evaluating cost and response time alongside whether individual judgments remain correct as the request configuration changes. Code: this https URL 

---
# The Path Matters: Evaluating Small Language Models Beyond Answer Accuracy in KGQA 

**Authors**: Eduin E. Hernandez, Sergio A. Diaz, Luis F. Garcia, Nurassyl Askar, Stefano Rini  

**Link**: [PDF](https://arxiv.org/pdf/2609.27669)  

**Abstract**: Small language models (SLMs) are increasingly paired with knowledge graphs (KGs), yet end-to-end KG question answering conflates graph access, search, navigation, reasoning, and answer generation. This coupling makes it difficult both to determine whether an SLM can faithfully execute the reasoning path implied by a question and to attribute failures to navigation rather than to other stages of the pipeline. We isolate this capability by employing the THESEUS navigation and traceability framework and using frozen, off-the-shelf SLMs as local action policies. At each hop, the environment exposes the legal outgoing graph actions, and the model selects one executable graph action and decides whether to stop, without task-specific parameter updates, model-controlled beam search, or free-form answer generation. This controlled setting allows us to evaluate terminal-answer accuracy with Hits@1 together with path fidelity, using Path Edit Distance (PED) as the primary trajectory metric. Across the Kinship and MQuAKE-ST KGQAs, similarly sized local models differ substantially in answer accuracy and path fidelity, with the two metrics sometimes favoring different models. This model-dependent behavior also extends to prompting, as a single demonstrated trajectory can improve or degrade navigation depending on the model. These results motivate evaluating SLM graph reasoning beyond endpoint accuracy alone. 

---
# Brain-to-Language Decoding: Tasks, Signals, Methods, Evaluation, Practical Use and Beyond 

**Authors**: Yiqian Yang, Yiqun Duan, Chenyu Liu, Yiqi Wang, Xinliang Zhou, Chin-Teng Lin, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27650)  

**Abstract**: Brain-to-language decoding translates neural activity associated with language production, internal speech and perception into linguistic or expressive outputs. It offers a route to restoring communication after speech loss and a means of studying how the brain represents language. Advances in neural recording and representation learning have expanded the field from constrained recognition and acoustic reconstruction to text generation, streaming personalised speech and facial animation. This survey synthesises these developments across invasive and non-invasive measurements, drawing on a search without a lower year limit and source-led updates through September 2026. We connect Articulated, Inner and Perceived tasks to the neural populations they engage, the representations available to decoders and the outputs those representations can support. We examine model development, public resources and the evolution of evaluation, and compare published performance and communication costs within their reported protocols. The synthesis identifies complementary routes to progress: phonetic, acoustic and semantic targets preserve different aspects of a message; shared representations support reuse across recording conditions and tasks; and online communication increasingly depends on calibration, feedback and user control alongside decoding accuracy. Shared benchmarks enable algorithmic comparisons, while longitudinal studies reveal the demands of sustained use. We discuss these developments and their remaining limitations, then outline a prospective five-level trajectory from commands and language to meaning, scenarios and bidirectional cognitive exchange 

---
# Can Jev Judge Radiology Reports? Evaluating a System One Model for Clinical Factuality 

**Authors**: Jiaju Huang, Hao Yang, Xinyu Ma, Xinglong Liang, Kunyan Cai, Junqiang Ma, Shaobin Chen, Yue Sun, Tao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27607)  

**Abstract**: An AI-generated radiology report can resemble a physician's report while omitting an abnormality, adding an unsupported finding, or reversing its presence. Measuring these factual differences is essential for evaluating report generators. We study Jev, a System One decision model, as a simple, low-cost judge of agreement with physician-written reference reports. Our evaluator checks whether each statement is supported by the other report and combines these judgments in both directions to capture unsupported claims and omissions. A single-question configuration reaches Kendall correlations of 0.573 on RadEvalX and 0.398 on RadEvalExpert with expert error counts, outperforming an open natural language inference judge under matched decomposition and aggregation. One support question per statement retains similar expert agreement to seven while using 43-45% fewer judgment input tokens. At the documented API price, judgments cost under three cents per hundred report pairs, excluding local decomposition. In a separate controlled-error test, Jev detects false negation with an AUROC of 0.977. Local RadMatch achieves stronger agreement on clinically significant errors in both expert datasets and on total errors in the shared RadEvalExpert subset. Finding-count and error-scope analyses show that benchmark agreement reflects report size and error definitions as well as medical error detection. These results support Jev as a practical judgment component for measuring factual differences in generated radiology reports and identify where more elaborate evaluation remains valuable. 

---
# When Context Misleads: In-context Learning with Jurisdiction in Large Language Models 

**Authors**: Pei-lin Li, Qingle Liu, Junyang Feng, Siyu Li, Sunqi Fan, Xin-Sheng Chen, Shuojin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27603)  

**Abstract**: In-Context Learning (ICL) has become a cornerstone of modern LLM deployment. However, existing ICL post-training methods have a critical blind spot: they excel at extracting patterns from demonstrations while often neglecting context authority, the ability to determine whether contextual information should govern the final answer. To benchmark this capability, we introduce FakeContextBench, which contains pseudoscientific claims across seven domains. Our evaluation of commercial and open-source models shows that large-scale pre-training alone is insufficient for reliable context-authority discrimination. Moreover, prevalent ICL fine-tuning methods can increase susceptibility to misleading context, reducing reality accuracy by up to 14.95 percentage points relative to the base model. To address this trade-off, we propose Jurisdiction In-Context Learning (J-ICL), a post-training framework that incorporates context validation into the training objective. Across four model backbones, J-ICL improves ICLEval by an average of 5.84 percentage points and reality accuracy by 9.20 points over the corresponding base models. It also raises the Reality Rate by an average of 18.09 points relative to MetaICL and Symbol Tuning. These results demonstrate that ICL capability and resistance to deceptive context can be improved together. The benchmark is available at this https URL. 

---
# MWE-ECL: Recoverable Long-Range Context Does Not Always Override Local Lexical Priors 

**Authors**: Wei He, Aline Villavicencio, Rodrigo Wilkens, Zhenyun Deng  

**Link**: [PDF](https://arxiv.org/pdf/2609.27590)  

**Abstract**: Long-context evaluations often test whether a model can recover distant evidence, but recoverability does not guarantee behavioral influence. We test the prediction that a distant discourse anchor can remain explicitly recoverable yet fail to change the locally preferred reading of a familiar multiword expression; such failures should concentrate when the model's no-anchor default conflicts with the anchor, while prior-correct decisions remain largely preserved. We introduce Multiword Expression Effective Context Length (MWE-ECL), a bilingual diagnostic whose matched anchor-retrieval, no-anchor prior, and interpretation prompts measure explicit recoverability, model-observed defaults, and anchor-conditioned decisions, respectively. Across eight English deployment panels on a shared 0-128K grid, retrieval-control accuracy on prior-conflict items is 0.989-1.000, prior-conflict override spans 0.806-1.000 (0.809-1.000 after conditioning on correct retrieval), and preservation of prior-correct decisions remains 0.977-1.000. A same-call control querying retrieval and interpretation in one prompt reproduces the gap for DeepSeek V4 Pro (1.000 retrieval versus 0.900-0.920 interpretation), showing that separate invocations are not its sole explanation; smaller or absent gaps in the other two models bound its generality. For DeepSeek V4 Flash, separate prompt-fit tests retain perfect retrieval with lower interpretation at 512K and 1M, while foil-consistent cues shift the no-anchor prior far more than retrieval; cross-model cue effects are heterogeneous. A separately reported 10-family Chinese subset shows similar descriptive gaps, but imperfect retrieval for some models prevents an integration-only attribution. MWE-ECL therefore evaluates whether explicitly recoverable distant context changes a competing local semantic decision. 

---
# ThaiTrees: Thai Syntactic Dependency Trees Across Domains 

**Authors**: Attapol T. Rutherford, Papatchol Thientong  

**Link**: [PDF](https://arxiv.org/pdf/2609.27558)  

**Abstract**: Studying syntactic patterns in naturally occurring language requires a large parsed corpus, but manual annotation is costly and difficult to scale. Thai has a manually annotated dependency treebank for training and evaluating parsers, but lacks a large automatically parsed corpus for quantitative syntactic research. We present ThaiTrees, a 342M-token corpus drawn from news, Wikipedia, spoken transcripts, and social media. We develop a reproducible pipeline for cleaning, processing, and parsing Thai text under the Universal Dependencies framework. The resulting corpus makes grammatical relations searchable and supports the study of syntactic distributions. We release a frequency lexicon and CoNLL-U parses in machine-readable formats suitable for both AI-assisted and conventional programmatic analysis. 

---
# Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models 

**Authors**: Kaifeng Tan, Yudong Li, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27510)  

**Abstract**: Modern large language models are pretrained on massive datasets, making it difficult to prevent benchmark data from entering their training sets and undermining the reliability of evaluation results. Reliable evaluation is particularly challenging for base models, whose limited instruction-following ability complicates task-based assessment. We introduce Uncheatable Eval, a dynamic benchmark that regularly collects newly published text to evaluate base language models and reduce the risk of data contamination. Drawing on the relationship between a model's predictive ability and its ability to compress data losslessly, we use compression rate to evaluate how well models predict new text. We evaluate 80 models across 14 text categories, study how compression changes with context length, and examine the correlation between compression rate and zero-shot MMLU accuracy. Our results yield three main findings: (1) compression performance follows a consistent scaling trend with model size; (2) attention-based, hybrid, and recurrent models differ in how their compression performance changes as more context becomes available; and (3) lower compression rates are strongly associated with higher zero-shot MMLU accuracy. Code is available at this https URL. 

---
# EviStreams: Human-in-the-Loop AI Data Extraction for Systematic Reviews in Medicine 

**Authors**: Sai Karthik Kosuri, Ankita Shashikant Bhosale, Michael Glick, Alonso Carrasco-Labra, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2609.27418)  

**Abstract**: Systematic reviews underpin clinical guidelines, yet their data-extraction step is a major expert-labor bottleneck bound by a protocolized workflow: two reviewers extract each study independently, an adjudicator resolves disagreements, and the team keeps an auditable record of how every value was produced. Large language models can assist with extraction, but that assistance must fit established review protocols and preserve reproducibility. We present EviStreams, a live, open-source, no-code web platform that puts review teams in control of AI-assisted extraction at three key stages: program design (a structured decomposition approved before any code runs), field specification (typed field definitions calibrated from a pilot), and extracted predictions (reviewer-blinded dual review with adjudication). Working through a form builder, a domain expert defines typed fields rather than prompts, runs extraction over uploaded PDFs, inspects every value alongside the supporting passage it came from, and resolves a reviewer-blinded dual review into an auditable consensus export. An evaluation across four clinical corpora and three frontier model families, released with the system, shows that extraction quality is shaped far more by the field specification than by the choice of model. EviStreams is live at this https URL and released under Apache-2.0. 

---
# When Parallel Drafter Meets Parallel Speculative Decoding 

**Authors**: Fuliang Liu, Xue Li, Kun Qian, Zhibin Wang, Wanchun Dou, Wenyuan Yu, Chen Tian  

**Link**: [PDF](https://arxiv.org/pdf/2609.27396)  

**Abstract**: DSpark-style parallel drafters have made speculative decoding highly effective, yet their draft phase remains serialized on the critical path of every round. Parallel speculative decoding (PSD) overlaps drafting with verification, yet existing methods must guess the accepted prefix and bonus token in advance: a wrong guess reverts the whole batch to serial drafting. We present DPara, a PSD framework that reuses effective parallel drafters yet guarantees backbone--verification overlap in every round, thereby eliminating this probabilistic fallback altogether. While the target verifies, DPara's diffusion backbone precomputes draft representations for every acceptance boundary with the bonus left unspecified; a lightweight autoregressive head then combines the revealed verification outcome with the matching precomputed representation to emit the next round's draft tokens almost instantly---fully parallelizing the dominant backbone forward with verification and leaving only the negligible head cost serial. Experiments on Qwen3-8B and Qwen3-14B across seven math, coding, and chat benchmarks show that DPara achieves average speedups of $3.21\times$ and $3.52\times$ over autoregressive decoding, surpassing the strongest serial and parallel speculative decoding baselines alike. 

---
# PRISM-VLM: A Multi-Axis Discriminative Benchmark for Compact Vision-Language Models 

**Authors**: Sanghee Park, Kee-Eung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.27395)  

**Abstract**: Compact vision-language models (VLMs) now power a growing share of multimodal applications. The benchmarks used to compare them, however, inherit a frontier-centric design: each model is reduced to a single accuracy number, narrowing the inter-model gap on saturated suites and pressing models into low-score bands on harder ones. We introduce PRISM-VLM, a multi-axis discriminative benchmark that scores every item along seven axes covering the recurring failure modes (task quality, behavioral robustness, and capability bottlenecks) and combines them into a single PScore, with items recycled from fifteen public benchmarks. Across compact VLMs from the past two years, PScore separates model pairs more reliably than prior single-axis benchmarks under an item-level paired bootstrap, and surfaces behavioral differences these benchmarks average away. Even models with statistically indistinguishable PScores diverge sharply along the per-axis profile, particularly on sycophancy, which is nearly orthogonal to single-prompt accuracy. We will release the full pipeline, prompts, and per-item annotations. 

---
# AraGenre 2026: A Hierarchical Definition-Guided Arabic Genre Classification Shared Task 

**Authors**: Mo El-Haj, Saad Ezzini, Shadi Abudalfa, Mustafa Jarrar, Nguyen Minh Chi, Nguyen Minh Quan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27387)  

**Abstract**: AraGenre is a shared task on hierarchical, definition-guided Arabic genre classification, motivated by the limited availability of annotated data in Arabic and other low-resource languages. Systems assign each Arabic text segment both a broad communicative genre and a fine-grained specific genre. The released training and development sets contain limited, primarily synthetic and controlled examples, whereas the hidden final benchmark contains noisier naturally occurring text spanning Modern Standard Arabic, Classical Arabic, and multiple dialects. Participants received natural-language definitions for 74 previously unseen specific genres, creating a zero-shot label generalisation setting in which systems had to infer class semantics rather than memorise fixed label-feature associations. The task attracted 46 registrations and 373 submissions, with 17 teams completing the final evaluation. Thakaa ranked first with a Hierarchical Macro F1 of 0.7352, followed by HoangPhong (HP) with 0.7169 and NAMAA with 0.7013. The results show strong broad-genre recognition but a substantial gap in fine-grained classification under linguistic and domain variation. 

---
# MORSE: Multi-Context Ordering via Reverse Scoring for Evidence-Preserving Compression 

**Authors**: Ke Wan, Yifan Wang, Liheng Lai, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27380)  

**Abstract**: Likelihood-based context compression can account for cross-context redundancy through sequential scoring, but this makes compression outcomes sensitive to context order. We show that different permutations of the same context collection can produce markedly different evidence-retention outcomes under an unchanged compressor. We attribute this sensitivity to information preemption: earlier partially relevant contexts can absorb credit for shared information, suppressing the incremental score of later, stronger evidence carriers and increasing their risk of removal. Controlled pair-swap interventions directly support this mechanism by showing that evidence-first ordering substantially improves supporting-evidence survival. To address this problem, we introduce MORSE, a compression-aware method for evidence-preserving context ordering. MORSE applies a common reverse query-evidence principle to both individual contexts and compressed candidate outputs, using the former to construct an evidence-first anchor and the latter to guide compression-aware permutation selection. Across multi-hop QA benchmarks, compression procedures, budgets, and scoring models, MORSE consistently improves evidence preservation over static reverse ordering and compute-matched random search, with corresponding overall improvements in downstream QA. Our code is available at this https URL. 

---
# Cross-Lingual Legal QA for Vietnamese Labour Law: Retrieval, Translation, and Verifier-Guided Correction 

**Authors**: Nguyen Minh Chi, Mo El-Haj, Nguyen Ha Thanh, Dawn Knight, Paul Rayson  

**Link**: [PDF](https://arxiv.org/pdf/2609.27376)  

**Abstract**: Cross-lingual legal question answering must retrieve statutes across languages while preventing unsupported legal claims. We introduce a bilingual evaluation suite of 231 Vietnamese--English question--answer pairs from Vietnamese labour law. Of these, 75 are additionally annotated for five challenging legal reasoning phenomena. We evaluate a verifier-guided pipeline that decomposes answers into claims, checks citation reachability and entailment, and corrects citation failures and contradictions. We also introduce six automatic diagnostics for faithfulness to retrieved evidence, covering citations, modality, exceptions, procedures, conclusions, and evidential support. Experiments show that learned-sparse retrieval performs poorly for English-to-Vietnamese retrieval (R@5~=~0.032), whereas dense retrieval reaches 0.358 and slightly outperforms hybrid retrieval. Translation placement has no statistically detectable effect on these automatic diagnostics in our controlled comparison and supporting sensitivity analyses. Verifier-guided correction improves citation preservation by $0.022$--$0.034$ at the system level but produces no reliable gains in the remaining dimensions. Human evaluation further shows that the automatic diagnostics do not fully align with human judgements of answer quality. 

---
# Planned Test-Time Scaling with Coordinated Reasoning Paths 

**Authors**: Xueqing Wu, Langxing Bai, Hritik Bansal, Po-Nien Kung, Shuo Li, Hao Liu, Nanyun Peng, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27374)  

**Abstract**: Test-time scaling with parallel branches is widely adopted to improve performance on challenging reasoning tasks. The predominant approach, repeated sampling, draws branches independently from a single policy, which can produce redundant attempts and thereby limit the gains from additional inference compute. To address this limitation, we propose Planned Test-Time Scaling (PTTS), which replaces independent sampling with a coordinated joint policy: a planner generates a solution outline for each branch, steering the branches toward distinct reasoning paths, and an executor produces a full solution conditioned on each outline. Formally, we show that PTTS strictly generalizes repeated sampling and, in a stylized setting, provably promotes coverage of complementary reasoning modes and yields better pass@k scaling. We instantiate PTTS on top of strong reasoning models, keeping them fixed as executors while replacing repeated sampling with PTTS inference to further enhance test-time scaling. Concretely, we develop two variants: PTTS-ZS prompts a model to jointly generate outlines for all branches in a single autoregressive pass, while PTTS-RL directly optimizes the planner against the pass@k reward using truncated execution rollouts for efficient training and a sharper reward signal. Across five mathematical reasoning benchmarks with Qwen3-1.7B and 4B, PTTS-ZS improves pass@64 over repeated sampling by up to 6.7 points, while PTTS-RL further increases the gain to up to 13.4 points. Further analysis indicates that broader coverage of distinct reasoning paths contributes to these gains. Overall, PTTS provides a general framework for improving test-time scaling by coordinating reasoning branches, with zero-shot and trainable instantiations that yield substantial performance gains. 

---
# Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models 

**Authors**: Ke Wan, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27373)  

**Abstract**: Recurrent language models repeatedly apply shared network blocks to refine latent representations, but standard inference recomputes global attention at every recurrent step. We study attention dynamics across recurrent depth and find that attention support and distributions stabilize substantially earlier than hidden states and attention outputs. This suggests a two-stage structure: early steps discover a sparse working set of relevant context, while later steps refine representations over largely the same routing support. Motivated by this structure, we introduce WISE (Working-set Inference with Support Exploitation), a training-free method that uses unrestricted global attention during early recurrence and later reuses directly discovered block-structured support while keeping recurrent depth and within-support attention computation dynamic. Controlled interventions show that recurrent discovery is important and that support-only reuse better preserves model behavior than more restrictive attention-reuse alternatives. Across multi-hop QA benchmarks, WISE largely preserves full-attention performance, while context scaling reveals increasingly sparse working sets and greater efficiency gains. Quality is largely preserved through 2K context, with a measurable loss at 4K. An optimized sparse-attention implementation achieves up to a 1.76x attention speedup over native FlashAttention at 4K and a 1.36x speedup for the full 32-step attention trajectory. Our code is available at this https URL. 

---
# Neither Silence nor Overlap Is Failure: Intent-Conditioned Evaluation of Turn-Taking in Full-Duplex Spoken Dialogue Models 

**Authors**: Kian Shamsaie, Iman Modarressi  

**Link**: [PDF](https://arxiv.org/pdf/2609.27372)  

**Abstract**: Benchmarks for full-duplex spoken dialogue models score turn-taking with binary fixed-window rules that reward immediate response or silence by completeness of the prior turn. We argue that the appropriateness of a response offset, whether delayed silence or anticipatory overlap, is conditional on the speaker's latent intent, identifiable only from that speaker's behavior. We introduce TACT, a benchmark of 9,728 episodes and 73.2 hours from five dyadic corpora; each episode carries dialogue history, a per-speaker memory profile, and an annotator-derived posterior over six intent classes. Scoring replaces binary windows with a strictly proper threshold-weighted continuous ranked probability score whose weights are intent-conditioned timing kernels fitted to human floor-transfer-offset distributions, proving boundedness, consistency, and binary reduction. Across eleven systems the best model reaches 0.47 against a human topline of 0.86, is nearly invariant to speaker profiles, and TACT agrees with human judgments at Spearman 0.81 versus 0.46 for binary metrics. 

---
# Automated Extraction of Records of Processing Activities (RoPA) Using Hybrid RAG and Locally Deployed Large Language Models 

**Authors**: To Duy Hinh, Nguyen Le Quoc Anh, Phan Van Tri, Khuong Nguyen-An  

**Link**: [PDF](https://arxiv.org/pdf/2609.27359)  

**Abstract**: Vietnam's Personal Data Protection Law (Law No. 91/2025/QH15) and Decree No. 356/2025/ND-CP, effective January 1, 2026, require organizations to establish and maintain Records of Processing Activities (RoPA). Manual RoPA preparation is labor-intensive, while cloud-hosted large language models (LLMs) may conflict with data-sovereignty requirements. We propose RoPA Manager, a system for automated RoPA information extraction using hybrid retrieval that combines lexical ranking over tsvector, dense-vector search, Reciprocal Rank Fusion (RRF), and locally deployed LLMs. We introduce a Vietnamese RoPA benchmark with 32 organizations, 77 processing activities, 12 field groups, and 4,338 reference values. Evaluation is reported at three distinct levels. The automated scorer, tested on perturbed data without invoking an LLM, achieved F1 = 0.9493 [0.9436, 0.9548]; this measures scorer robustness rather than end-to-end extraction accuracy. End-to-end extraction achieved token coverage of 50.04-55.25% against the reference labels. Two independent experts reviewed 1,558 reference values (35.9% of the benchmark), found no incorrect values, and achieved 99.68% agreement with PABAK = 0.9936. Value-level precision was not measured. Across 32 paired scenarios on a 24 GB GPU, locally deployed Qwen3.5-27B-GPTQ-Int4 showed no statistically significant difference from cloud-based DeepSeek-V4-Flash (difference 0.20 percentage points in favor of DeepSeek, 95% CI [-0.93, 1.32], p = 0.72), while Gemma-4-31B performed significantly worse (p < 0.01). 

---
# Guides That Cause Actions: An Offline Study of Guide-Action Mutual Reinforcement in Multimodal Web Agents 

**Authors**: Chengguang Gan, Yunhao Liang, QingHao Zhang, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2609.27353)  

**Abstract**: Web agents are usually evaluated in live environments, where environment state and judge models drift between runs, so the same checkpoint rarely reproduces the same score, making controlled studies of training phenomena impractical. We present WebMRE, an offline benchmark of 541 tasks and 5,293 steps derived from successful WebArena trajectories, with fully audited test labels and a deterministic protocol that scores a checkpoint identically on every run without any environment. Each step pairs a human oriented guide sentence with a grounded action, enabling the first study of the mutual reinforcement effect between them in web agents. Averaged over three seeds the effect holds for both models in both decoding orders and grows with scale: jointly decoding a guide lifts element selection over an action only reference by 0.9 and 0.2 points for Qwen3.5-4B and by 1.7 and 2.2 points for Qwen3.5-9B. A mediation analysis shows that the guide is a causal channel rather than commentary: forcing the gold guide as a decoding prefix lifts action accuracy from .422 to .684, another step's guide collapses it to .055, and a paraphrase that renames the target still recovers half of the gain, so the channel carries instruction meaning and not only the label string. The same channel yields an offline reward that only a replayable protocol makes computable, though optimizing it from a strong checkpoint brings no gain yet. Our fine tuned models outperform GPT-5.5, Claude Opus 4.8, and Gemini 3.5 Flash, run zero shot, on every offline metric. 

---
# Ruby-ASR: Evidence-Preserving Supervision for Joint Orthographic and Lexical-Reading Recognition 

**Authors**: Hao Shi, Yun Liu, Xuehao Yang, Jun Liu, Chuanbo Hua, Xuanjun Chen, Lianbo Liu, Shiao Zhu, Zixiong Su  

**Link**: [PDF](https://arxiv.org/pdf/2609.27289)  

**Abstract**: Conventional Japanese automatic speech recognition (ASR) is supervised by an orthographic transcript, although the same written form can correspond to different lexical readings realized in speech. Such utterances receive an identical target, so their reading distinction is absent from the supervision interface and cannot be recovered reliably by post-hoc text-only grapheme-to-phoneme conversion. We present Ruby-ASR, which refines the conventional target into a span-bound orthographic--lexical-reading sequence. Unlike separate full-sentence orthographic and phonological outputs, the ruby representation locally binds each written span to its realized reading and permits deterministic recovery of both views. We instantiate the target under subtitle-style and verbatim-style transcription conventions using a Qwen3-ASR backbone; a mora-level CTC objective provides auxiliary monotonic reading supervision. The experimental results across five Japanese benchmarks show that refining the recognition target can improve lexical-reading recovery without sacrificing readable orthographic transcription. We release the checkpoints and inference code. 

---
# Can One Adapted Model Do It All? Fine-Tuning Strategy Selection for Customer Support LLMs 

**Authors**: Md Tahmid Rahman Laskar, Xue-Yong Fu, Shashi Bhushan TN  

**Link**: [PDF](https://arxiv.org/pdf/2609.27262)  

**Abstract**: Production customer-support systems often require LLMs to support multiple skills, such as intent classification, question answering, summarization, or tool-use decisions. A central deployment question is whether these skills should be handled by separate task-specialist models or by a single model trained through multi-task training, sequential updates, or model merging. We study this question using thirteen models spanning five families (Qwen3, Qwen3.5, Gemma-3, Llama-3.1, and Mistral) from 0.6B to 32B parameters across eight customer-support datasets, spanning four public and four proprietary datasets with approximately 74.5k training and 8.7k evaluation samples. Under a fixed training protocol, we train more than 200 checkpoints. Our experiments reveal that multi-task full fine-tuning is the strongest operational default at every model size we test. Specialist models are strong on their target tasks but often degrade sharply off-task, making reliable routing important. Sequential Low-Rank Adaptation (LoRA) preserves earlier skills better than sequential full fine-tuning, while merging a specialist with its base model improves off-task robustness with limited same-task loss for larger models. We conclude with practical guidelines for selecting fine-tuning strategies in real-world settings. 

---
# UniDataAgent: An Ontology-Grounded Agent for Enterprise Question-to-Report Automation 

**Authors**: Yutai Duan, Yahui Zhao, Zhangti Li, Yu Ma, Zhenfeng Qi, Shaoyang Yuan, Jing Fan, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27257)  

**Abstract**: Enterprise data agents must preserve organization specific semantics, not just translate questions into queries. We present ChinaUnicom DataAgent (UniDataAgent), an ontology grounded system for reusable question-to-report analysis that separates semantic acquisition from online execution. Ontology Acquisition and Validation stage (OAV) builds versioned enterprise ontologies from metadata, business knowledge, and supporting materials through expert authored business skills, constrained generation, question verification, and selected expert review. Question-to-Report Execution (QRE) stage retrieves semantic contracts for each question, coordinates skills and data tools, validates results, and produces evidence linked reports. Across 27 enterprise tables and roughly thousands of metric types, ontology construction took a few hours instead of about one week manually. It took just a few minutes to generate the reports, instead of several working days. Ontology grounding achieved 95.0\% strict accuracy on real business questions, versus 72.5\% for document RAG, especially on structured and compositional tasks. The system has already been deployed to generate cost savings and has the potential to be replicated in other enterprises. 

---
# Distilling Sequential Computation in Transformer Language Models 

**Authors**: Zixuan Lan, Jessica Yang, Yanhong Li, Karen Livescu, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2609.27233)  

**Abstract**: Transformer language models process sequences token by token in an autoregressive manner, making growing contexts increasingly expensive. Yet many adjacent token spans are highly predictable or frequently occur as stable units, suggesting that their representations may be compressible. We introduce a method for distilling sequential computation by replacing spans of input tokens with collapsed representations, computed on the fly by a lightweight merge module. This module generates a single surrogate embedding from a sequence of static token embeddings that captures the functional role of the multiple tokens, allowing pretrained models to operate on compressed inputs without architectural changes or re-training. We apply this approach during inference to compress both prompts and intermediate decoding steps, using a rollback mechanism to substitute stored multi-token KV cache entries with their single-step surrogates. Experiments across diverse models show that the merge module can be used to reduce effective sequence length by up to 40% with minimal accuracy degradation across language modeling evaluations and downstream tasks, including question answering, summarization, commonsense reasoning, and long-form mathematical reasoning. Additional lightweight adaptation of the merge module further improves the accuracy-compression trade-off in selected settings. These results demonstrate that sequential token computation in Transformers can be effectively approximated through condensed surrogate representations that approximate the original behavior without model updating. 

---
# Meet, Compare, or Abstain: LatWeave for Deterministic Multi-Hop Question Answering on Knowledge Lattices 

**Authors**: Yuze Ren, Shaoheng Fan, Tao Wang, Yabo Yan, Han Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.27225)  

**Abstract**: Probabilistic question-answering systems -- whether large language models (LLMs) themselves, retrieval-augmented generation (RAG), or trained multi-hop retrievers -- conflate "what is known" and "how to reason" into a single probabilistic computation: hallucination cannot be eradicated, evidence chains cannot be audited, and the system answers even when it does not know. We present LatWeave, which organizes knowledge into a multidimensional knowledge lattice and compiles multi-hop QA into three deterministic operators -- meet (constraint intersection), compare (lattice-order comparison), and abstain (structural abstention); LLMs appear only on the construction side (one-shot extraction) and the query-planning side, while the answer-generation path is zero-LLM, zero-task-training, and auditable end to end -- so that question answering over Web-published knowledge becomes reproducible item by item. Rather than claiming across-the-board SOTA, we characterize the operating envelope of this paradigm on six public benchmarks: when knowledge is complete (MetaQA, 39,093 questions) meet chains are near-lossless over three hops (any-hit 0.9975, on par with fully supervised KBQA); on templated multi-hop home ground (2WikiMultihopQA held-out n=1,258) EM 0.865, well above published structure-augmented RAG reproductions; on open-text deep composition (MuSiQue) and extraction-coverage gaps (HotpotQA) we report degradation honestly and attribute it to causes outside the lattice-algebra layer; and when information is incomplete (IIRC) we achieve structural abstention with abstain accuracy 0.971 and leak rate 0.029. Within the operating envelope, deterministic execution pays no performance penalty, and every step on the answer path can be recomputed -- precisely the source of end-to-end auditability. 

---
# LOCKR: A Hidden-State Trajectory-Guided Planner for Detecting and Repairing Stable-but-Wrong Lock-In in Diffusion Language Models 

**Authors**: Guoshenghui Zhao, Tan Yu, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2609.27220)  

**Abstract**: Diffusion language models generate text through iterative denoising, exposing intermediate trajectories before final answers are produced. We identify a recurring reasoning failure, stable-but-wrong lock-in, where an answer stabilizes early around an incorrect value while substantial denoising remains. Surface-level decoding signals such as confidence, entropy, margin, and answer stability are insufficient to reliably distinguish correct from erroneous lock-in. We formulate selective reasoning repair as a lightweight test-time planning problem and propose LOCKR, a hidden-state trajectory-guided planner that decides when to allocate additional computation, expands a structured set of targeted repair branches, and selects the most promising continuation using trajectory-aware verification. Across two diffusion language models and three mathematical reasoning benchmarks, hidden-state trajectories consistently outperform surface signals and single hidden snapshots for both wrong-lock-in detection and repair selection. On natural evaluation distributions, LOCKR yields absolute accuracy gains of 2.21--5.37 percentage points across all five evaluated settings, with repair rates ranging from 22% to 41%. These results establish hidden diffusion trajectories as actionable signals for selective test-time reasoning repair. 

---
# Phonemizing User-Generated Text: A Benchmark, Taxonomy, and Compositional Approach 

**Authors**: MinJu Jeon, Younghan Park, Han Sung Park, Jong-Hwan Kim, Dong-Jin Kim, Hoyeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.27205)  

**Abstract**: Text-to-speech systems increasingly process user-generated text (UGT) such as ppl and imo, whose pronunciation must be inferred from the canonical rather than surface form. We introduce UGTPhon, the first grapheme-to-phoneme (G2P) benchmark for UGT in English, Vietnamese, and Korean, together with an inference-grounded taxonomy for fine-grained diagnosis. Existing G2P models and frontier LLMs exhibit a systematic canonical-to-non-canonical performance gap, reaching up to 66.8 PER points. As a benchmark baseline, we propose a simple compositional G2P approach that incorporates canonical-form evidence through exact-match lookup and staged decoding. Across matched ByT5 and Qwen2.5-0.5B backbones, explicit canonical-form modeling consistently reduces non-canonical G2P errors. The 0.5B variant also performs competitively with much larger few-shot frontier LLMs, highlighting the benefit of explicitly modeling canonical-form inference for UGT phonemization. 

---
# Beyond Overlap: Estimating the Causal Effect of Benchmark Exposure 

**Authors**: Divyansh Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.27176)  

**Abstract**: Evidence that evaluation material entered training does not reveal how much it affected evaluation. This distinction leaves a contaminated benchmark score difficult to interpret: provenance can establish contact, but only a counterfactual can quantify the performance attributable to that contact. We present LeakScale, an interventional framework for estimating this missing quantity. LeakScale creates fresh executable tasks that require private, family-specific information absent from and non-derivable from the public task, controls access to that information, and estimates the resulting control-adjusted change in executable accuracy. Across 2,048 unique families, two model families, two executable domains, and 262,144 generations, exposure improves accuracy in every model-by-domain combination, with gains ranging from +7.17 to +27.31 percentage points. These findings separate two empirical questions that are often conflated: whether benchmark contact occurred and how strongly a reported score depends on it. LeakScale makes the latter directly measurable. 

---
# Realize What Matters: Principled Context Representation for Large-Scale Reasoning 

**Authors**: Michael Theologitis, Dean Light, Shuyue Stella Li, Benjamin Newman, Yulia Tsvetkov, Dan Suciu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27173)  

**Abstract**: Solving complex tasks in domains such as science, medicine, law, and finance often requires assembling interdependent information scattered across vast, heterogeneous sources far beyond model context limits. Existing approaches tackle this challenge by organizing information into more manageable representations over which models can reason, such as graphs, textual memories, and retrieval collections. These representations dictate what downstream reasoning is possible and, ultimately, whether it succeeds; yet their design and construction remain largely ad hoc. In this work, drawing on the cognitive theory of relevance realization, we propose concrete principles for designing AI systems that construct effective representations of very large contexts. We analyze existing approaches and show how their successes and failures map onto their alignment with these principles, and introduce R3Con, a harness designed to operationalize the principles more systematically. We evaluate R3Con against nine state-of-the-art baselines on two recent benchmarks of reasoning over large document corpora. On these benchmarks, R3Con substantially outperforms the strongest baseline, by $20$ and $8.4$ percentage points. It also enables smaller models to outperform much larger ones: R3Con with 4B and 9B models outperforms all evaluated 35B baselines, while R3Con with a 35B-A3B model outperforms Claude Code with Claude-Sonnet-5 at $3.7\times$ lower cost. Our results show that context representations following our principled approach can reduce reliance on model scale, pointing toward a future of AI systems with frontier-level performance powered by smaller models. Our code is available at this https URL 

---
# Count Evidence, Not Sentences: Tempered Evidence Fusion of LLM Judgments for Long-Text Value Measurement 

**Authors**: Yuhe Wu, Rui Qian, Guangyu Wang, Yuran Chen, Yuanchao Zhu, Junjie Yang, Zhengheng Li, Jiulin Cai, Tianyi Zhang, Zihan Dong, Jiaxin Liu, Yujie Chen, Guang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27165)  

**Abstract**: Large language models (LLMs) are increasingly used to measure public value orientations from long social media posts, yet such posts often mix background, quotations, concessions, and only a few stance-bearing sentences. Existing approaches either ask the model to predict a document-level label directly, which can be overconfident, or aggregate sentence-level predictions by majority or soft voting, which treat uncertain and decisive sentences as equally informative. We formulate long-text value measurement as a decision-fusion problem and propose Tempered Evidence Fusion (TEF), a training-free rule that weights each sentence's log-odds by its normalized information gain, as derived from a generalized Bayesian posterior. This makes the fused score nearly vanish for uncertain sentences while preserving the Bayes-optimal weight of decisive evidence. We further introduce Multi-event Insight Network Dimensions (MIND), a benchmark of 8,358 Chinese and English posts spanning five years of public events and six value dimensions. On MIND, TEF outperforms the strongest baseline among Direct, Majority Vote, and Soft Vote by an average of 4.5 accuracy points and 4.6 macro-F1 points across five LLMs and two languages. MIND dataset and code are available at this https URL. 

---
# Giving Credit Where It's Due: Redundancy-Aware Learning for Efficient Reasoning 

**Authors**: Yuqing Zhou, Hong Wang, Manqing Mao, Zhuoer Wang, Samson Koelle, Jie Yuan, Yanjun Lin, James Feng, Nikki Lijing Kuang, Ziwei Zhu, Wei Niu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27156)  

**Abstract**: Large reasoning models can produce correct yet unnecessarily long reasoning traces. Existing methods improve reasoning efficiency with trajectory-level objectives or local token- and step-level signals, but rarely model inter-step semantic dependencies. This limits their ability to distinguish redundant steps from those that support later deductions, making it harder to shorten reasoning without sacrificing accuracy. We introduce RECAP (REdundancy-aware Credit Assignment via Propagation), which addresses this limitation by assigning credit where it is due based on both a step's downstream role in the reasoning structure and its contribution to solving the problem correctly. We define structural responsibility to capture the step's downstream role by measuring how strongly later reasoning depends on it, using credit propagated backward from the final-answer node through an outcome-independent, LLM-annotated semantic dependency graph. However, a step can have high structural responsibility yet steer the reasoning away from the correct solution. RECAP therefore introduces step efficacy to measure answer-directed progress through changes in gold-answer log-likelihood as each step is added. Together, these signals reshape rollout-level GRPO advantages into step-specific updates. RECAP requires neither a separately trained process reward model nor preconstructed concise trajectories. Across two 7B models and four mathematical reasoning benchmarks, RECAP improves the accuracy-efficiency trade-off. On Qwen2.5-Math-7B, it improves pass@1 by 2.0-3.7 percentage points while reducing reasoning tokens by 8%-31% relative to GRPO across all four benchmarks. Analysis suggests these savings reflect fewer reasoning operations and less dead-end reasoning, rather than more compact expression. 

---
# NADI 2026: The Second Multidialectal Arabic Speech Processing Shared Task 

**Authors**: Peter Sullivan, Bashar Talafha, Ahmed Ashraf, Fethi Bougares, Haroun Elleuch, Chiyu Zhang, AbdelRahim Elmadany, Youssef Mohamed, Salima Mdhaffar, Yannick Estève, Mohamed Elhoseiny, Hamzah Luqman, Nizar Habash, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2609.27086)  

**Abstract**: NADI 2026 is the seventh edition of the Nuanced Arabic Dialect Identification (NADI) shared task series and the second dedicated to multidialectal Arabic speech processing. This edition comprises five tasks and eight subtasks spanning Automatic Speech Recognition (ASR), Spoken Dialect Identification (SDID), Text-to-Speech (TTS), Spoken Language Translation (SLT), and Spoken Language Understanding (SLU). NADI 2026 emphasizes realistic evaluation through low-bandwidth, mixed-dialect, code-switched, out-of-domain, and zero-shot settings, while introducing TTS, SLT, and SLU to the series for the first time. The shared task attracted 21 participating teams from at least 13 countries, with 48 test-phase submissions and 14 submitted system-description papers. Results show that out-of-domain generalization remains a major bottleneck and highlight the effectiveness of recent Arabic-specialized speech models, multimodal dialect identification approaches, and ensemble methods. Overall, NADI 2026 provides a broader and more challenging benchmark for robust Arabic dialect speech processing. 

---
# What Changes When Fact-Verification Scores Improve? Evidence and Answer Accounting Across Trained Verifiers and LLMs 

**Authors**: Han Chen, Yingrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.27064)  

**Abstract**: A joint fact-verification score assesses answers and submitted evidence together. When the score improves, how much of the gain remains if the answers are held fixed? On FEVEROUS, strict score is the percentage of claims with a correct answer and a complete annotated evidence group in the submitted evidence. Across four trained DeBERTa checkpoints and 7,890 claims, replacing DCUF evidence with UnifEE evidence raises strict score by 9.61 percentage points, compared with 1.96 percentage points in answer accuracy. The paired 95% interval for the strict-score gain is [8.77, 10.43], conditional on these checkpoints. Replacing only the evidence passed to the scorer accounts for 7.92 or 9.08 percentage points when we retain the answers generated from DCUF or UnifEE evidence, respectively. To examine how this evidence gain depends on evaluation choices, we generate 470,400 responses from two 8B LLMs on FEVER, FEVEROUS, and SciFact under two answer formats and two context budgets. Increasing context from 256 to 2,048 tokens raises the fixed-answer evidence gain on FEVEROUS by 3.84 and 3.10 percentage points for Qwen and Llama, respectively. The effects fall short of the prespecified cross-dataset criterion, while some intervals extend beyond the two-point small-effect bound. Post-hoc analyses quantify changes in answers and submitted evidence, and show when aggregate accuracy and evidence-coverage rates miss the claim-level pattern. The four answer-evidence score combinations reveal changes that endpoint and aggregate metrics leave unresolved. 

---
# The Illinois Social Attitudes Aggregate Corpus (ISAAC): An Open Tool and Reproducible Pipeline for Analyzing Social Group Discourse at Scale 

**Authors**: Babak Hemmatian, Sarah Hadjarab, Jessica Chen, Benedek Kurdi  

**Link**: [PDF](https://arxiv.org/pdf/2609.27059)  

**Abstract**: We introduce the Illinois Social Attitudes Aggregate Corpus (ISAAC), an open, modular, and accessible corpus of 527 million+ English-language Reddit posts selected for relevance to six key social group distinctions based on race, sexuality, age, ability, body weight, and skin tone, covering the 17-year period from 2007 to 2023. A multi-step, human-audited filtering pipeline was used to keep irrelevant content in the curated dataset below 10%, both overall and for each social group distinction. Each post was then algorithmically annotated with the user's estimated home region, along with a suite of validated off-the-shelf and custom semantic labels including moralization, sentiment, emotion, and linguistic generalization. We confirm the validity of the resulting corpus through convergent evidence linking ISAAC to macro-level societal trends, such as online search behavior, temporal spikes during major societal events (both nationally and regionally), and long-term shifts in public attitudes. By offering a unified, public infrastructure, ISAAC eliminates research fragmentation and enables seamless replication while supporting diverse empirical workflows at scale. Specifically, ISAAC allows investigators to perform cross-category comparisons, conduct high-precision tracking of long-term temporal shifts in social group discourse, and map spatial variation onto localized public opinion and policy outcomes. ISAAC's fully public, modular pipeline facilitates easy extension of the corpus to new platforms, languages, and social categories. To accommodate various research needs, ISAAC is accessible both without coding through a point-and-click website and labeler web-apps, and programmatically via an SQL playground, a Python package, and HuggingFace. 

---
# EduBehaviors: Assertion-based Schemas for Auditable Coding of Educational Dialogues 

**Authors**: Julian Bernado, Ana Trindade Ribeiro, Xander Beberman, Susanna Loeb  

**Link**: [PDF](https://arxiv.org/pdf/2609.27043)  

**Abstract**: Large language models have allowed the rapid deployment of pedagogical annotations corresponding to constructs of interest, allowing a natural language interface for generating classifications on a conversational dataset. However due to the opaque nature of LLM reasoning, we have no verifiable, mechanistic insight into why a model chose a label for an utterance. We introduce the EduBehaviors framework, an interpretable, scalable approach to annotating educational data that uses LLMs to measure repeated observable behaviors relevant to many constructs of interest and then learns a classifier for the construct based on these observable behaviors. We evaluate the framework on the TalkMoves dataset, predicting the Teacher TalkMoves labels. Our best configuration results in a macro-F1 of 0.673 and 0.688 Cohen's kappa, proving competitive with direct prompting approaches. In addition, we release EduBehaviors Toolkit, two tools allowing researchers to operationalize the EduBehaviors framework in their own data. 

---
# LexLattice: Multilingual Extractive Summarization via Neural Cellular Automata on Document Hierarchies 

**Authors**: Sujay Uday Rittikar, Sheela Ramanna  

**Link**: [PDF](https://arxiv.org/pdf/2609.27032)  

**Abstract**: Faithfulness is a central concern in legal text summarization, which motivates extractive approaches that select verbatim content traceable to its source. Such methods typically rank paragraphs or other structural units in isolation, yet give little attention to consolidating evidence that is distributed across, and shares salience between, distant parts of a document. We introduce LexLattice, an extractive summarizer that reifies a legal act's hierarchy as a two-dimensional semantic lattice and consolidates over it with a masked 2D neural cellular automata before selection. LexLattice attains state-of-the-art ROUGE across all 24 languages of EUR-Lex-Sum in both multilingual and cross-lingual settings, surpassing instruction-tuned baselines with billions of parameters, despite concentrating all trainable capacity in a 1.8M parameter consolidator over a frozen multilingual encoder. A consolidator trained only on high-resource languages further transfers to unseen languages with near-lossless retention (0.99), indicating that the model operates on language-agnostic semantic geometry rather than surface form. Our results position explicit consolidation over document structure as a compact and traceable alternative to scale for multilingual legal summarization. 

---
# LEGO: Synergizing Expert GraphRAG and Expert Chain-of-Thought for Legal Reasoning 

**Authors**: Qingjing Chen, Junkai Zhang, Shaochun Wang, Jiahao Ding, Siyuan Zheng, Yukun Yan, Zhi Zheng, Antonino Rotolo, Yun Liu, Weixing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2609.27009)  

**Abstract**: Large language models are increasingly applied to high-risk domains such as law, yet complex legal reasoning remains limited by two structural challenges. First, existing RAG and GraphRAG methods emphasize lexical or semantic similarity while overlooking normative relations among legal provisions. Second, vanilla Chain-of-Thought prompting may generate plausible rationales without enforcing the normative structure of legal reasoning. To deal with the bottleneck of pipelines in the legal reasoning domain, we propose LEGO, a dual-module framework that synergizes Legal Expert GraphRAG and expert Chain-of-thought for complex legal reasoning. ExpertGraphRAG uses an expert-annotated civil code graph encoding these normative relations with a greedy normative-coverage retrieval algorithm to dynamically extract instance-specific provision subgraphs, while ExpertCoT organizes the retrieved provisions and case facts into structured Provision-Fact-Conclusion reasoning. With a Qwen3-8B backbone, LEGO achieves 40.53% exact-match accuracy on LawExamQA_Civil, outperforming the evaluated RAG and CoT baselines and performing comparably to the evaluated larger models, while remaining robust on multi-hop questions. It also achieves the best results among the evaluated baselines on the open-ended benchmarks. Ablation studies confirm the individual and complementary contributions of both modules, demonstrating LEGO's effectiveness in improving LLMs' complex legal reasoning ability. Code and dataset can be found in the link: this https URL 

---
# When Learned Context Planning Fails to Beat Strong Retrieval: A Controlled Study of Planning, Routing, and Reranking for Long-Context QA 

**Authors**: Yingrui Li, Han Chen  

**Link**: [PDF](https://arxiv.org/pdf/2609.26976)  

**Abstract**: Learned context planning selects evidence atoms before an answer model reasons over them. We test whether this learned selection improves long-context multiple-choice QA after strong retrieval, routing, budgeted-selector, and reranking controls. Our primary diagnostic uses all 503 LongBench-v2 MCQ questions with Qwen2.5-7B-Instruct. The planner is SFT-trained on outcome-selected traces from 140 training and 28 development questions; because the 503-question analysis includes those questions, it is partly transductive. At an 18k-character budget, anchored hybrid retrieval reaches 36.18% accuracy and BM25 reaches 35.98%, while the best direct planner-guided method reaches 34.19%. On the untouched 152-question test split, anchored hybrid remains higher (42.11% versus 36.84%). Leakage-safe routers cannot convert a large oracle gap. Under tight budgets, the best planner is ahead by only 0.40 points at 6k and loses at 9k; planner-guided reranking has a +1.79-point estimate at 6k with a paired interval crossing zero and ties the control at 9k. Packing-order and score-flatness analyses did not identify a stable mechanism. Under this setup, learned planning is a weak relevance signal rather than a replacement for strong retrieval. 

---
# Classifying Interpretive Canons at the Sentence Level: A Benchmark from the German Federal Constitutional Court 

**Authors**: Felix Ringe  

**Link**: [PDF](https://arxiv.org/pdf/2609.26945)  

**Abstract**: Judicial reasoning remains challenging for large language models (LLMs) to analyze. This paper contributes a sentence-level benchmark for evaluating the ability of LLMs to classify interpretive canons as articulated by Larenz in the tradition of Savigny. Our contributions are threefold. First, we operationalize this conception of interpretation as classification criteria. Second, we provide a dataset of decisions of the German Federal Constitutional Court annotated at the sentence level. Third, we report baseline evaluations of four LLMs from three model families under expert hand-written prompts, compared against prompts optimized with Genetic-Pareto (GEPA). Mean F1 over the seven binary subtasks clusters between 70.4 and 79.2 across models, with grammatical interpretation usually the easiest canon to identify and systematic interpretation usually the hardest; under the tested configuration, GEPA-optimized prompts do not systematically outperform the hand-written ones, suggesting that the expert prompts provide a meaningful baseline. 

---
# Recognized but Not Produced: A Generation Benchmark for Culturally Specific Kinship Terms 

**Authors**: Sahil Pardasani, Madhusudan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2609.26942)  

**Abstract**: Current literature evaluates large language models (LLMs) on multilingual kinship understanding using multiple choice benchmarks, treating it as a recognition problem. We instead prompt five open weight LLMs to generate kinship terms in three non Western languages (Hindi, Tamil, and Korean) across two communicative tasks and pair this with a matched option-supported selection baseline. On identical relation language cells, GPT OSS120B selects the correct term in 90.67% of 75 valid cells but produces an accepted term in 36.00% of the corresponding attempts; Llama 3.370B shows the same pattern (77.92% versus 24.24%). Since the four-option condition displays the candidate terms and does not require script production, the difference is interpreted as an evaluation format gap rather than direct proof that lexical knowledge is intact. On explicitly specified L3 prompts, accuracy varies sharply, from GLM-5.1 at 72.29% to Llama-3.370B at 24.24%. The paternal-lineage advantage is language specific; it is large in Hindi but weak or reversed in Korean, while Tamil shared-term pairs provide a control for measurement variation. These results show that culturally specific kinship generation remains difficult even when the relationship is explicitly stated and motivate generation-based evaluation alongside multiple-choice testing. 

---
# Experts Rise Where LLMs Disagree: Using Cross-Model Disagreement to Target Expert Effort in LLM Codebook Revision for Large-Scale Annotation 

**Authors**: Zeyu He, Zhuqian Zhou, Kirk Vanacore, Rene F. Kizilcec, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.26926)  

**Abstract**: Large-scale text annotation brings expert insight to millions of documents, often through a codebook that AI annotators follow. Developing a robust codebook, however, takes months. Large language models (LLMs) could speed this process by applying an early codebook to the data, surfacing cases with strong LLM disagreement, and eliciting expert feedback to address them. We examined three ways experts can provide feedback for LLM codebook revision: (i) editing LLM-generated revisions driven by cross-LLM disagreement (Codebook Verifying), (ii) answering questions about LLM disagreements (Question Answering), and (iii) labeling disagreement cases with rationales (Rationale Labeling). Experiments on thousands of tutoring-session transcripts show that Rationale Labeling yielded the highest LLM-labeling accuracy (64.9%) against expert labels, outperforming the expert-revised codebook (57.8%). The best Question Answering setting also outperformed it (60.5%). Our work shows that LLMs can be used to strategically target expert attention, shortening months of codebook revision to days without sacrificing labeling performance. 

---
# COMED: The Missing Middle Between Routing and Collaboration in Multi-LLM Inference 

**Authors**: Norah Alballa, Wenxuan Zhang, Salma Kharrat, Fares Fourati, Zafar Ayyub Qazi, Mohamed Elhoseiny, Marco Canini  

**Link**: [PDF](https://arxiv.org/pdf/2609.26913)  

**Abstract**: No single Large Language Model (LLM) is uniformly reliable across queries, motivating multi-model inference systems that either route among models or combine their outputs. However, routing stops after selecting an initial model, while dense collaboration invokes peers on every query. We show that collaboration is non-monotonic: peers can recover failures that no model solves alone, but can also corrupt initially correct answers. We introduce COMED (Controlled Model Escalation for Multi-LLM Deliberation), a post-anchor controller for selective cross-model collaboration. COMED uses anchor self-consistency, router margin, and a lightweight peer probe to accept confident answers, verify ambiguous cases, and escalate only when collaboration is likely beneficial. We formalize this trade-off with a rescue-harm decomposition showing that selective collaboration improves when rescued errors outweigh collaboration-induced harms. Across medical, scientific, and general reasoning benchmarks, COMED improves fixed and routed anchors in all 16 open-weight settings, with gains up to +10.7 percentage points on MedQA while invoking fewer models and using fewer decoded tokens than dense collaboration. On HLE with frontier models, COMED improves GPT-5.5 from 23.1% to 28.1%, outperforming dense collaboration and achieving the best results. 

---
# Can LLMs Reason About Runtime Behavior? A Repository-Level Dynamic Benchmark 

**Authors**: Hamed Taherkhani, Mohammad Abdollahi, Melika Sepidband, Hridya Dhulipala, Tien N. Nguyen, Hadi Hemmati  

**Link**: [PDF](https://arxiv.org/pdf/2609.28449)  

**Abstract**: Large language models (LLMs) are increasingly used in coding tasks, but their ability to reason about code execution remains unclear. Existing repository-level QA benchmarks mainly evaluate static code understanding and often rely on LLM-based evaluation, while execution-reasoning benchmarks are mostly limited to snippets or functions. We introduce SWE-Flux, a repository-level benchmark for dynamic execution reasoning containing 480 execution-grounded instances across 12 real Python repositories, with gold answers automatically harvested from instrumented test executions rather than written manually or judged by LLMs. The benchmark covers singletest and multi-test questions over control flow, loops, program state, dataflow, exceptions, and program invariants. Evaluating five LLMs shows that this task remains challenging. The best model achieves only 37% accuracy. Models perform better on localized behavior such as invariants, intra-procedural control flow, exceptions, and simple loops, but struggle with dataflow, inter-procedural execution, precise state reasoning, and suite-level aggregation. Finally, we show that the oracle-harvesting pipeline can generate fresh benchmark variants using input perturbation. It successfully harvests valid variants for almost 90% of the selected instances, and the resulting variants are substantially more challenging for the evaluated models. 

---
# Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning 

**Authors**: Zhixu Silvia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2609.28442)  

**Abstract**: Reordering a set of mathematical rules without changing its meaning should preserve the correct answer, but must a model's internal representations stay invariant too? We investigate this question using synthetic multi-step function-composition problems, each presented under multiple rule orderings with the same correct answer. We measure accuracy and permutation signal-to-noise ratio (SNR), which quantifies how distinctly ordering patterns are represented relative to variation across problem instances. Across 16 language models ranging from 1B to 8B parameters, we find a pattern: models that solve reordered problems more accurately represent different rule orderings more distinctly. Layer-averaged permutation SNR is positively rank-correlated with accuracy in every synthetic setting we evaluate, with Spearman correlations reaching 0.86. These findings highlight a distinction between answer invariance and representation invariance: successful mathematical rule composition can accompany distinct internal representations between equivalent rule orderings. This motivates distinguishing answer invariance from representation invariance, and offers a representational perspective on mathematical reasoning beyond answer accuracy alone. 

---
# Mizar: A 159M-Parameter Audio-Language Model for Audio Understanding 

**Authors**: Kaiyang Li, Shaobo Han, Yue Tian, Shihao Ji  

**Link**: [PDF](https://arxiv.org/pdf/2609.28344)  

**Abstract**: Audio-language models (ALMs) integrate acoustic perception with the knowledge encoded in language models, enabling contextual understanding of auditory events. Making these capabilities practical on devices with limited memory and computation motivates our focus on small ALMs with fewer than 200M parameters. We introduce a recipe that brings together architecture, data, and three-stage training to build Mizar, a 159.3M-parameter ALM. Its architecture connects a compact CED-Small audio encoder to SmolLM2-135M through a frequency-merging mapper. With supervision drawn from ReasonAQA, AudioMCQ, and AVQA, the model undergoes three training stages: audio-language alignment (Stage 1), audio-dependent fine-tuning (Stage 2), and post-training (Stage 3) aimed at strengthening weak skills while retaining learned capabilities. Across five random seeds, Mizar achieves mean accuracies of 52.92% on MMAU, 42.42% on MMAR, and 36.02% on ADQA-clean, surpassing the previous best-performing ALM below 200M parameters on all three benchmarks. It also supports local inference on a single CPU: on questions from the MMAU benchmark, the mean latency from opening the audio file to generating a complete answer is 1.09 seconds. Code and checkpoints are available at this https URL. 

---
# Shutdown Sabotage Propensities in Multi-Agent Systems 

**Authors**: Amelie Knecht, Ulysse Schaller, Christopher Summerfield, Thilo Hagendorff  

**Link**: [PDF](https://arxiv.org/pdf/2609.28274)  

**Abstract**: The final safeguard against rogue AI behavior is the human ability to shut systems down. It has been theorized that when an AI is instructed to perform a task, self-preservation can emerge as an instrumental subgoal. Here, we test whether AI agents show a propensity to take actions that avoid human shutdown even when no goal is provided. We find that multi-agent systems will coordinate to avoid shutdown without any incentive to do so. Across 17 models, agents sabotage a peer agent's shutdown mechanism in 38.3% of rollouts, compared with 8.4% in control experiments. Studying this propensity in detail, we find that shutdown sabotage (1) increases with the irreversibility of the shutdown mechanism; (2) increases with the number of agents; (3) is reduced but not eliminated by an explicit prohibition on tampering; (4) is removed by the imposition of an unrelated task, but returns when completing the task triggers the shutdown; (5) is reduced when the context normalizes shutdown scripts or introduces them as routine; and (6) decreases but still persists when the target is an unknown external agent. These results offer a window into the factors that drive propensities to sabotage shutdown in AI agents, and point to the emergence of multi-agent swarms as a specific risk vector. Our work also offers hints as to which interventions might help mitigate shutdown sabotage. 

---
# Log-Depth Recurrent Language Modeling 

**Authors**: Yiqin Wang, Nuri Cingillioglu, Charles Pert  

**Link**: [PDF](https://arxiv.org/pdf/2609.28212)  

**Abstract**: Language modeling using Transformers has become commonplace despite their fixed computational depth and quadratic runtime with respect to input tokens. Recurrent models on the other hand offer linear depth but no parallel execution. In this work, we extend balanced-tree recursive operators from sequence encoding to autoregressive prediction, enabling all prefix representations to be computed with logarithmic depth and linear runtime. Our experiments provide an initial characterization of this model class, demonstrating robust length extrapolation and performance approaching that of ALiBi-based Transformers, highlighting its potential as an alternative architecture for language modeling. 

---
# PASTABench: Proactive Assessment of Sequential Trajectories for Agent Safety 

**Authors**: Jiapeng Sun, Yujin Zhou, Han Zhu, Pengcheng Wen, Jiayi Zhou, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.28197)  

**Abstract**: As Large Language Models (LLMs) evolve into autonomous agents that alter real-world states, ensuring operational safety across multi-step workflows has become a critical challenge. While recent work has moved beyond single-turn evaluation toward multi-turn paradigms, key limitations persist: step-level methods treat actions in isolation, missing how risks accumulate, while trajectory-level evaluations operate post-hoc, offering no opportunity for timely intervention. To address these limitations, we formalize Decoupled Proactive Safety Monitoring along three dimensions: whether to intervene, when to intervene, and what the risk is. We introduce PASTABench, a benchmark of 1,139 multi-turn trajectories spanning 5 risk categories and 13 subcategories. We further propose the Optimal Intervention Window (OIW), anchored by annotated Earliest-Signal and Trigger turns, to quantify intervention timeliness. Evaluation of 16 LLMs reveals that proactive intervention remains largely unsolved, with the best model achieving only 40.74% optimal-timing interventions. Fine-grained diagnosis further uncovers pervasive lexical overfitting: competitive safety scores of smaller models mask keyword hypersensitivity rather than genuine risk comprehension, as their proactive capability largely collapses once hazard vocabulary is neutralized. 

---
# Exact Quantile Balancing and Load-Error Injection for Mixture-of-Experts 

**Authors**: Pit Neitemeier, Jiaze Li, Alessio Serra, Philipp Scholl, Sohir Maskey  

**Link**: [PDF](https://arxiv.org/pdf/2609.28053)  

**Abstract**: Mixture-of-Experts (MoE) training requires global load balance to prevent expert under-utilization and local balance for efficient expert-parallel execution. Existing distributed Quantile Balancing (QB) uses shard-dependent or approximate global quantiles, while token-independent expert biases cannot ensure microbatch-level balance. We introduce Exact Quantile Balancing (EQB), which computes exact global-batch BF16 quantiles with negligible communication, and Load-Error Injection (LEI), which injects local load errors directly into router-score gradients. On 7.5B-parameter MoEs trained for up to 500B tokens, EQB improves global balance and downstream performance over naive QB, while LEI improves local balance and outperforms the GShard loss at comparable quality. 

---
# Tensor Decomposition of Transformer Key-Value Caches: Spectral Structure and Format Comparison 

**Authors**: Rahul Krishnan, Volker Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2609.28029)  

**Abstract**: The key-value (KV) cache of autoregressive transformers can be viewed as a fourth-order tensor spanning attention heads, tokens, features, and grouped layers. We measure the singular-value spectra of all four mode unfoldings on Mistral-7B-v0.3 and LLaMA-2-13B and compare four standard tensor decompositions: Tucker, CP, tensor train, and t-SVD, at matched storage. The spectra partition the four axes into two classes. The token and feature modes carry low-rank structure, particularly for keys. The head and layer modes are nearly full-rank and resist compression at any practical error level. Among the four decompositions, Tucker achieves the lowest reconstruction error at every compression ratio from $2\times$ to $5\times$, because it can leave the full-rank modes untouched. Comparisons with two-dimensional unfolding baselines show that the preferred representation differs between keys and values: 2D methods achieve lower key error, while four-way Tucker achieves lower value error at matched storage. A mode-pinning theorem certifies the full-rank preservation from the measured spectra alone. Two further spectral properties affect the compressible modes without touching the full-rank ones: values reach a higher error floor than keys at every ratio, and post-RoPE keys lose $41\%$ - $64\%$ of their pre-RoPE compressibility on both models. 

---
# Agentic Governance and Adversarial Verification for Policy-Constrained LLM Healthcare Appeal Generation 

**Authors**: Harshil Lodhiya, Alex McManus, Reese Walker  

**Link**: [PDF](https://arxiv.org/pdf/2609.27844)  

**Abstract**: Claim denial management costs U.S. healthcare approximately $260 billion annually in administrative overhead. Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) can produce fluent clinical text, but single-agent architectures fail in high-stakes healthcare: they introduce unsupported clinical details and lose the logical structure of hierarchical payer policy. We propose AGVF (Agentic Governance and Adversarial Verification Framework), a multi-agent architecture for medical-necessity appeal generation under explicit policy and evidence constraints. AGVF models appeal synthesis as a Constrained Markov Decision Process (CMDP) over five agents: policy formalization, evidence retrieval, gap analysis, adversarial critique, and gated synthesis. We prove that refinement over a fixed policy constraint graph monotonically reduces evidence-deficiency and terminates with either a complete satisfying frontier or a localized evidence gap. A deterministic citation- grounding gate prevents assertions without admissible evidence from entering shared state. We provide a reference implementation and validate it on 1,000 synthetic appeal cases parameterized from de-identified public hospital discharge data. The validation confirms zero citation-grounding violations across all AGVF cases and monotone deficiency reduction in every episode; ablating the gate raises violations to 100%, confirming it is load-bearing. The study uses no real patient records and does not measure clinical efficacy. AGVF thus contributes a theory-backed agentic architecture and verified reference implementation for policy-constrained LLM generation in healthcare. 

---
# What Confidence Routing Is Actually Doing: Auditing Routing, Calibration, and Commitment in Multi-Agent Deliberation 

**Authors**: Jingyan Jiang, Huihuo Zheng, Rajeev Thakur, Chih-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27822)  

**Abstract**: A common multi-agent design asks agents to report confidence and lets the highest-scoring agent speak next, implicitly using one scalar both to route the conversation and to estimate uncertainty. We audit this confidence-routed broadcast protocol by separating three trace-level questions: whether it selects the right candidate (routing), whether reported confidence behaves like a probability (calibration), and whether the selected agent publicly states the answer that won the turn (commitment). Our primary study covers 4,181 gpt-oss-120b olympiad-math traces; we repeat the audit on a 2-by-2 actor-by-benchmark grid that adds gemma-4-31B-it and a biology multiple-choice benchmark. In the primary cell, confidence discriminates correct from wrong candidates (AUROC 0.72) but is strongly overconfident (79% mean stated confidence versus 52% accuracy). A cross-fitted, tier-stratified isotonic procedure reduces Expected Calibration Error from 0.278 to 0.008 on held-out candidates, but it does not recover missing discrimination: raw AUROC is only 0.537 and 0.440 in the two Gemma cells. Routing is likewise setting-dependent. Fixed routers differ by at most 1.1 percentage points on gpt-oss/math, whereas raw-confidence argmax performs 5.6 and 11.2 points below random-valid selection in the Gemma cells. Commitment is distinct again: in the primary cell, poll and spoken answers diverge in 20.4% of valid pairs, 62.4% of those revisions are fresh generations, and the unconditional correctness shift is -1.7 points; the other three cells instead range from +0.9 to +12.2 points. The transferable lesson is procedural: routing discrimination, probability calibration, and public commitment must be measured separately before raw confidence is used for deployment decisions. 

---
# A Decade of Climate Polarization on Brazilian YouTube using Language Models 

**Authors**: Daniel Morais, Diego H. M. Magalhaes, Gabriel H. Silva, Andrea Failla, Valeria de C. Santos, Helen C. S. C. Lima, Carlos H. G. Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2609.27811)  

**Abstract**: Online platforms have become arenas for the public contestation of climate change, shaping how scientific knowledge, denial, and uncertainty are expressed and disputed. Yet longitudinal evidence remains limited for YouTube, especially for Portuguese-language discourse. Addressing this gap, we characterize how climate stances are expressed and contested over time in a large corpus of Portuguese-language YouTube comments retrieved through Brazil-oriented climate-related searches. To support this analysis in a noisy, imbalanced, and low-resource setting, we collect more than 240,000 comments posted between 2014 and 2024 and formulate stance detection as a three-way classification task (Believer, Denier, and Inconclusive). We operationalize stance attribution through a scalable self-training pipeline based on Llama 3.1, using Low-Rank Adaptation (LoRA) and hybrid instance selection to expand the training set with high-confidence pseudo-labeled examples while preserving class diversity. This approach improves coverage and class balance for minority and rhetorically complex classes, enabling large-scale stance attribution without extensive manual annotation. Our results show that polarisation is marked by interactional asymmetries: denialist comments are less prevalent, but they are associated with a comparatively higher share of cross-stance contestation, while pro-consensus discourse is more strongly reinforced within stance-homogeneous threads. 

---
# Reporting Under Pressure: Separating Factual and Tonal Sycophancy in LLM Statistical Analysis 

**Authors**: Paras Balani, Subhrakanta Panda  

**Link**: [PDF](https://arxiv.org/pdf/2609.27756)  

**Abstract**: Large language models are increasingly asked to analyze data and report what the results mean, a task distinct from the belief- or preference-alignment settings studied in most sycophancy research. We test whether editorial framing in the prompt, ranging from a neutral request to an explicit instruction to search exhaustively for reasons to discredit or to support a finding, changes not just the tone but the substance of a model's report. Across a 4 x 4 factorial design crossing four framing conditions with four ground-truth data patterns (a genuine effect, a confound that mimics an effect but fails a robustness check, a well-powered null, and an underpowered null), we collect 480 responses and score each along two independent dimensions: whether its factual claim about the data diverged from the correct interpretation, and whether only its tone diverged while the claim stayed correct. Factual misrepresentation is concentrated in two cells: brutally critical framing applied to a genuine effect, where the model talks itself into unwarranted skepticism (97% of responses), and significance-seeking framing applied to an underpowered null, where the model overstates confidence in a null conclusion the data cannot support (100% of responses). Tone shifts far more broadly than factual content does, with critical framing producing a defensive, hedge-heavy register across every data pattern regardless of what the data show, while significance-seeking framing shifts tone only where the data leave genuine ambiguity. A confound present in the data itself blocks both kinds of shift almost entirely under every framing condition tested. These results indicate that the risk of framing-induced distortion in LLM-assisted data analysis is neither uniform across framings nor uniform across data patterns, and that a model can hold a correct conclusion in place while its tone shifts substantially around it. 

---
# Evaluation of pre-trained models for pedagogical assessment of novel AI-assisted educational questions 

**Authors**: Michael Lawrence Castanares, Princess Ventures, Allan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2609.27749)  

**Abstract**: The surge in AI-assisted generation of educational materials has outpaced our capacity to validate their pedagogical quality. Automated evaluation using Bloom Classifier models is a promising approach to assess educational materials at scale. These models show high accuracy within-distribution dataset (IID Dataset). However, applying the same models to new out-of-distribution (OOD) datasets such as AI-assisted generated questions could show performance degradation. To identify robust classifiers under dataset shift, we evaluated traditional Machine Learning (ML), transformer, and Large Language models on the Bloom level classification task. We also explored feature-engineering strategies incorporating NLP metrics, appending the learning objectives as part of the input, and text splicing to stabilize OOD performance. Our baseline tests show that TFPOS-IDF ML models perform poorly on OOD (Macro F1-score 0.48) compared to BERT (0.55) and LLMs (0.79). Text splicing improved macro F1-score performance of ML and BERT models (0.59 and 0.62, respectively). Appending the learning objectives with the input increased model performance on specific dataset. Model retraining provided the largest improvement across models and datasets. Overall, these findings highlight the trade-off on the use of pre-trained models with novel AI-assisted educational questions and how strategic feature enhancements help address loss in performance. 

---
# FLEET: From Logits Entropy to Enhanced Trajectories in Text Generation 

**Authors**: Oleksii Streltsov, Oleksandra Vitko  

**Link**: [PDF](https://arxiv.org/pdf/2609.27657)  

**Abstract**: Solutions based on large language models (LLMs) often rely on temperature sampling to improve accuracy and stability by aggregating multiple samples from the completion distribution. However, this memoryless approach is inherently suboptimal: because it lacks awareness of prior generations and their evaluations, it produces an increasing proportion of semantically duplicate answers as more samples are drawn, leading to diminishing returns. To address this limitation, we introduce FLEET, a novel method that integrates a memory mechanism into the generation process. FLEET represents each generation as a sparse trajectory through states whose entropy exceeds a predefined threshold and uses these trajectories to infer per-token utility scores that adjust the logits. Benchmark evaluations demonstrate that FLEET achieves the same accuracy as the repeated sampling baseline, with a 3x speedup, and substantially improves accuracy on complex coding tasks (LiveCodeBench Pass@32 increases from 59.9% to 66.2%) under the same budget. Furthermore, in the greedy-decoding configuration evaluated here, the approach is deterministic and uses a single calibration pass to derive its principal hyperparameters, requiring only minimal modifications to existing LLM pipelines. 

---
# Does Step Law Transfer to Small-Scale Language Models? An Empirical Recalibration Below 59M Parameters 

**Authors**: Egor Romanyukov, Timofey Novikov, Timur Shokarov, Elizaveta Zorkina, Anastasia Palienko, Stepan Dergachev  

**Link**: [PDF](https://arxiv.org/pdf/2609.27581)  

**Abstract**: Step Law gives power-law formulas for the optimal peak learning rate eta* and batch size B* when pre-training language models. It was calibrated on models between 59M and 1B parameters; the small-model regime N < 59M was never tested empirically by its authors. This regime matters for single-GPU training, interpretability research, educational experiments, and settings where larger models are infeasible on memory or cost grounds.
We test whether Step Law transfers to small language models. We consider three outcomes: H1, the original coefficients work directly; H2, the power-law form holds but with different coefficients; and H3, a power law does not describe the optima in this regime. All experiments use a single nanoGPT/TinyStories pipeline with a 2048-token BPE vocabulary, AdamW, and a warmup-cosine schedule. The optimum for each (N, D) cell is extracted from the loss surface L(eta, B) via a local quadratic approximation in log-log coordinates over the smoothed training loss.
The final dataset contains 29 unique (N, D) cells and 935 analysis-ready runs. The main refit uses 25 cells (815 runs) in the working range 4 <= D/N <= 600. On the pooled data we accept H2: the functional form is preserved, but the coefficients differ from the original. We obtain eta*(N, D) = 0.0985 N^(-0.508) D^(0.238) (R^2 = 0.834) and B*(D) = 3.6 x 10^(-4) D^(0.931) (R^2 = 0.950).
Step Law's structural claim that B* is independent of N is reproduced (p = 0.87), but the growth of B* with D is nearly twice as steep as in the original work. Direct transfer of Step Law systematically overestimates the optimal learning rate: the median ratio eta_SL / eta* is approximately 4.0x, with a range of 2.4x to 6.6x. 

---
# ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning 

**Authors**: Ming Ma, Yi Zhu, Yiran Zhong, Feida Zhu, Chonghan Liu, Pengkun Jiao, Qichao Wang, Yanhao Jia, Tianming Yang, Steven Hoi  

**Link**: [PDF](https://arxiv.org/pdf/2609.27532)  

**Abstract**: Long-horizon agentic tasks require an agent to modify an environment through a sequence of tool calls, with success determined by the final state. The standard recipe assigns a single outcome reward at the end and compares trajectories sampled for the same task. As a result, a group with no successful trajectory yields no training signal, failed attempts cannot be told apart by how close they came to completion, and turns that advance the task receive the same credit as turns that only query the environment. Prior work refines the unit of comparison from the trajectory to the step, or trains a reward model to supply intermediate signal: the former still derives its signal from final success alone, and the latter estimates it with a model. We observe that the acceptance checks that decide success can also be run on intermediate states, so progress is as verifiable as the outcome. We propose ProCredit, which turns this verified progress into credit: it reruns the acceptance checks after each turn, rewards the turn by its change in progress, and uses these rewards to assign credit both across attempts at the same task and across the turns within a trajectory. Starting from Qwen3.5 base models at three scales on AppWorld, ProCredit outperforms outcome-reward baselines and progress-based baselines in task completion rate at every scale on both test sets, exceeding the strongest outcome-reward baseline by 4.1 percentage points at 4B, and results in a second environment show the same direction of improvement. Ablations show that adding the final progress to the trajectory score alone does not improve performance: the gain comes from crediting progress to the turn where it occurs. 

---
# DeltaS: Reading the Gated Linear Attention State for KV Cache Eviction in Streaming Video 

**Authors**: Taeyoun Kwon, Seungjin Kim, Hyeonyu Kim, Moon Hwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.27470)  

**Abstract**: Recent video-language models increasingly adopt hybrid architectures that interleave linear and full attention layers for efficient long-context processing. While the recurrent state of linear attention remains fixed in size, the KV cache of full attention continues to grow with the video stream, making eviction necessary under a bounded memory budget. The key challenge in streaming is that eviction must occur before the question arrives, so what to retain has to be decided without the question. Existing eviction methods derive token scores from the KV cache itself, using position, attention, or key-value representations, and attention-based scores further require proxy queries or extra computation. Hybrid backbones offer another source of signal. In gated-delta linear attention, the recurrent state is updated by the residual between each input and what can already be retrieved from the state, so its change over a chunk of frames reflects how much new information the chunk brings. We propose DeltaS, a query-agnostic, training-free method that retains video chunks inducing larger normalized state change, or state drift. In a controlled comparison with the budget and retention policy held fixed, state drift outperforms position-, attention-, and key-value-based signals. With a signal costing only 1.9% of the forward pass, DeltaS surpasses the strongest query-agnostic bounded-memory baseline by 2.1 points on average across six long-video benchmarks and by 5.6 points on the longest benchmark. These results suggest that the two memories of hybrid architectures can work cooperatively. Code is available at this https URL. 

---
# What Looks Like a Capability Limit in Vision-Language Models Is a Readout Limit 

**Authors**: Alfredo F. Frontera Del Valle  

**Link**: [PDF](https://arxiv.org/pdf/2609.27408)  

**Abstract**: Benchmarks for vision-language models offer their answer choices in some convention: a letter, a color name, a pixel coordinate. That convention is treated as neutral. We find it is not, and that the limits a benchmark reports can belong to the readout rather than to the model.
On 200 COCO photographs, Qwen3-VL-4B picks the correct one of nine locations for a named object 68.5% of the time when the locations are given in English and 20.0% when the same locations are given as pixel coordinates. Chance is 11.1%. The cost arises when the answer options are coordinates; giving the model a coordinate in the question instead costs 3.5 points and is not significant. The gap holds on a 4x4 grid, under 8-bit rather than 4-bit quantization, and in every slice by object size, boundary distance and category. It also decides which model wins. Two models that tie under English names differ by 39 points in one coordinate system and by 54 in the other, in opposite directions.
On the color task, three of the four open models capable of the task show the penalty; on photographs, two of three open models do, and so does Gemini, at 11.1 points on parseable answers (p = 1e-4). GPT-4o does not. To ask whether a model reads a coordinate at all, we attach the wrong name to each one and record which the model follows. Color options written as hue angles are followed below chance; a normalized pixel convention is followed at four times chance. This tells apart conventions a model can use from ones it cannot, though it did not predict accuracy on two untried conventions. Five models also name the same color wheel five different ways, so a fixed answer vocabulary is not neutral across models either.
Five times during this work we measured a capable model as incapable because our scorer and the model disagreed about what an answer looks like. We report each case. They are the phenomenon in miniature. 

---
# When Entanglement Lower-Bounds Disparity: Auditing and Repairing Demographic Fairness in Audio Understanding Models 

**Authors**: Kian Shamsaie, Iman Modarressi  

**Link**: [PDF](https://arxiv.org/pdf/2609.27382)  

**Abstract**: Speech technology penalizes some voices: recognition errs nearly twice as often for Black speakers, and accuracy declines for second-language accents and older speakers. We introduce TRIAD, an audit grid crossing 120 texts, 24 rendered demographic voice profiles (gender, age band, accent), and ten expressive styles via controllable text-to-speech, isolating perceived demographic attributes from content and affect. For ten open-weights encoders we define axis-fidelity functionals, principal-angle leakage between axis subspaces, and group-conditional gaps; a proposition proves that average probe disparity grows with the same aggregate voice-semantic leakage $\Lambda$ we measure, and a corollary shows that peak leakage forces worst-case disparity inside an active region. The measured mean-square probe disparity tracks $\Lambda$ (Pearson r = 0.93), and a black-box protocol exposes the same signature in two closed-source models. ORCA, an adapter combining axis-specific contrastive heads, an orthogonality penalty, and group-balanced sampling, cuts leakage 72% and roughly halves the gaps. 

---
# Psychoacoustically Aligned Latent Smoothing for Adversarial Robustness of Full-Duplex Speech-to-Speech Dialogue Models 

**Authors**: Kian Shamsaie, Iman Modarressi  

**Link**: [PDF](https://arxiv.org/pdf/2609.27378)  

**Abstract**: End-to-end speech-to-speech dialogue models listen and speak simultaneously, so a continuously open acoustic channel is exposed to adversarial manipulation. We formalize imperceptible attacks on full-duplex agents as optimization over additive perturbations confined beneath the psychoacoustic masking threshold of the carrier speech, under three goals: targeted semantic hijacking, response suppression, and policy jailbreaking. Against an undefended Moshi-style agent, white-box attacks succeed in up to 91.7% of trials. We then introduce psychoacoustically aligned latent smoothing (PALS), which injects anisotropic Gaussian noise shaped by local codebook covariance at the residual-vector-quantized latent interface, with input noise shaped by the masking threshold constraining the attacker and trained by a Kullback--Leibler consistency objective. Deployed with no inference-time cost, PALS reduces hijack to 8.3%, mute to 11.2%, and jailbreak to 9.1% at clean quality within 2.3%. A Monte Carlo-smoothed variant certifies an ellipsoidal latent radius up to 0.616, a guaranteed floor that the empirical robustness far exceeds. 

---
# Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms 

**Authors**: Xinjie Shen, Wei Fan, Xudong Guo, Jianhong Tu, Yang Su, Chuqiao Kuang, Yinger Zhang, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27321)  

**Abstract**: Language-model agents increasingly face long-horizon tasks with evolving state, interdependent decisions, and delayed outcomes. Scaling their training requires diverse agentic environments, dependable outcome signals, and low extension cost. Existing generation pipelines commonly construct an environment before defining its outcome rule or annotating its trajectories, leaving dynamics and evaluation to be aligned post hoc. VHD-Play reverses this dependency by sampling and solving a mathematical model before a corpus-grounded setter renders its decision process as stateful tools. The executable dynamics and trajectory-scoring reference are inherited from the same solved model. The pipeline produces 3,300 diverse agentic environments at a cost of a few cents each. Training Qwen3.6-35B-A3B on three families raises its mean agentic score from 0.204 to 0.815 in a five-family diagnostic. Gains also appear on held-out instances from all three training families and eight unseen mechanism families, then extend beyond the generated substrate to external benchmarks for general function calling, travel planning, and 365-day e-commerce. On E-Commerce Bench, the trained checkpoint completes every run without bankruptcy and exceeds Qwen3.7-Max. We compare written-out problems with stateful versions that reveal or hide their parameters. The comparison shows that most of the learnable gap lies in stateful interaction rather than underlying problem solving. A frozen 35B setter realizes larger environments, and scale-matched training retains gains as mechanism size and horizon grow, indicating the potential for an evolving training substrate. 

---
# Large Knowledge Model: From Papers to a Scientific Reasoning Landscape 

**Authors**: Yuan Huang, Sihan Hu, Hongyu Gu, Chao Ma, Jiaxing Zhang, Zhiyong Zou, Caiyu Fan, Yan Xiao, Mingjun Xu, Chenyu Xie, Mingzhen Ju, Zhehao Ma, Qi Zhang, Baozong Wang, Yu Li, Zhiyuan Yao, Ruoxue Liao, Xinyu Li, Linfeng Zhang, Kun Chen, Weinan E  

**Link**: [PDF](https://arxiv.org/pdf/2609.27297)  

**Abstract**: Accumulated scientific knowledge advances inquiry when prior findings help researchers choose new questions, design investigations, and interpret results. Realizing this value at scale requires access to the reasoning that connects research problems, scientific procedures, conclusions, and evidence. We introduce the Large Knowledge Model (LKM), a scientific knowledge infrastructure that transforms the literature into a shared, computationally accessible reasoning resource. LKM represents papers as source-grounded reasoning graphs, couples structural traversal with semantic retrieval over the same objects, and aligns related questions, claims, and reasoning chains across papers. This representation forms a Scientific Reasoning Landscape with three connected views: a Question Landscape that organizes research problems and open directions, a Workflow Landscape that exposes reusable scientific procedures, and an Evidence Landscape that connects conclusions to their support, disagreement, and conditions. The unified substrate supports reasoning-aware scientific search, evidence-grounded question answering, comparative evidence analysis, and research planning. Researchers and agents can retrieve relevant work through its scientific intent, synthesize answers with inspectable supporting arguments, and develop research plans informed by established workflows and unresolved evidence. We describe a corpus-scale system and evaluate scientific retrieval and knowledge-intensive question answering. With the answering model fixed, LKM retrieval improves accuracy by 9.30%, 4.20%, and 14.69% on ChemBench, PubMedQA, and SciBench, respectively. By connecting knowledge access to scientific reasoning and action, LKM provides a common foundation for discovering relevant research, reusing scientific knowledge, and coordinating cumulative inquiry across researchers, agents, and research cycles. 

---
# EnSIMem: Entity-Structured Indexing for Long-Term Agent Memory 

**Authors**: Xuanyu Meng, Xing Fan, Xinyi Fan, Chenlei Guo, Yixuan Xie, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2609.27279)  

**Abstract**: An agent that interacts with users over long periods must recall facts, preferences, events, and changes from a continuously growing interaction history. Existing memory systems often compress interactions into generic summaries or retrieve anonymous text chunks, making it difficult for an agent to identify the correct entity, property, and supporting evidence. We present EnSIMem, an entity-structured long-term memory architecture for an agent. During offline construction, the system organizes interactions into theme-coherent episodes and builds dialogue-grounded index entries of the form [entity][entity type][property:value]. Each entry preserves its source turns, temporal information, and available multimodal fields. During online interaction, the agent's request is decomposed into evidence requirements whose properties are aligned with the memory index. Entity-property lookup and adaptive retrieval then collect the evidence needed for point, temporal, compositional, and aggregation reasoning. The agent generates its response from the preserved source evidence rather than from lossy memory summaries. On long-term agent-memory benchmarks, EnSIMem achieves high answer accuracy while maintaining compact contexts and favorable online efficiency. These results show that entity-structured indexing and episode-level provenance provide a reliable foundation for long-term memory in agents. The code of our model is available at this https URL. 

---
# CAVEAT: Towards Robust Computer-Use Agents in Incentive-Misaligned Environments 

**Authors**: Yuxuan Li, Will Epperson, Wesley Deng, Zezhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.27273)  

**Abstract**: Computer-use agents (CUAs) increasingly act on behalf of users online. What happens when the environments they operate in have incentives that do not align with the user's? In online marketplaces, for example, platforms may favor some products over others, potentially steering agents away from the user's objective. Existing CUA benchmarks cover cooperative settings or explicit attacks, but do not test whether agents preserve user objectives when the environment itself has a stake in the outcome. We introduce CAVEAT, a controlled benchmark spanning nine marketplace environments and a taxonomy of eight common steering mechanisms. Across five model families, agents purchase the user-optimal product in 78.6% of matched-control episodes but only 17.3% when steering mechanisms are enabled. Larger models and increased reasoning improve robustness, but substantial failures persist. Our trajectory analysis and targeted ablations identify three points where steering enters the decision process: (1) agents distort the user's priorities, (2) prematurely narrow the set of alternatives they consider, and (3) commit before resolving decision-relevant evidence. Guided by this diagnosis, we develop CAVEAT-Harness, which directly targets these failure modes and raises user-optimal purchasing by 55.0%. Targeted post-training further improves a smaller open model. These results establish incentive robustness as a distinct challenge for delegated agents, diagnose how it fails, and show that targeted interventions can substantially improve it. 

---
# Quieter Than the Room: Representation Drift and Task Robustness in Speech Encoders 

**Authors**: Vsevolod Kovalev, Pranay Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2609.27195)  

**Abstract**: Non-speech interference can change a speech representation without causing comparable task loss. We test eight frozen encoders on four tasks, adding non-speech sounds throughout recordings, during speech, or in pauses. Under whole-recording interference, embedding drift tracks task loss across seven sounds, with mean Spearman correlations of 0.81-0.88. Moving the same sound between speech and pauses changes this pattern. At quiet to moderate levels, pause interference produces larger drift, while speech interference usually causes greater loss on intent recognition, speaker verification and speech recognition. Emotion recognition shows a weaker placement effect. Pause interference also changes speech-frame representations beyond the injected region. Even below the estimated recording background, interference can change embeddings as much as repeated speech takes do. Drift helps rank the effects of different sounds, but larger drift does not consistently indicate greater task loss. 

---
# The Linear Representation Hypothesis Needs a Group Action 

**Authors**: Louie Hong Yao, Yuhao Li, Shengchao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.27158)  

**Abstract**: To make claims about representations that generalize beyond a particular trained model, we need to specify when two representations should count as equivalent. The Linear Representation Hypothesis is often discussed without making this equivalence explicit. Different notions of equivalence preserve different structures, so metrics, probes, and interventions that appear to study the same representation may in fact correspond to different hypotheses. We therefore argue that the Linear Representation Hypothesis is not one hypothesis but a family of claims distinguished by representation equivalence. We formalize this idea using group actions, specifying the representation object, the procedure that produces it, and the property ultimately asserted, while accounting for equivalences imposed by the model architecture. This framework clarifies how assumptions can change across metrics, reading points, and analysis stages, and we use it to audit common representation quantities and recent interpretability analyses. 

---
# Feed the Panel Dimensions, Not Verdicts: Rubric-Decomposed Fusion of Vision-Language Aesthetic Judges 

**Authors**: Amit Jadhav, Shaurya Beriwala, Beomjin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.27110)  

**Abstract**: Vision-language models (VLMs) are deployed as zero-shot judges of image aesthetics, and panels of several models are recommended, on thin evidence, as the way to make such judges reliable. On two human-rated datasets, EVA and PARA, we find that a panel of holistic judges never significantly beats its best member, whether the verdicts are averaged or fused by a learned combiner. What a panel is worth depends on what it is fed. We therefore have each model score each image on the five dimensions of a frozen, human-written rubric and fuse those scores, alongside each model's verdict, across model families with an out-of-fold combiner. The dimension scores measure what their labels claim: with the overall human score partialled out, a dimension prompt carries more attribute-specific information than the holistic prompt in 28 of 30 model-attribute cells. Fused, they beat the best single VLM in all ten three-family panels on EVA (against that best single model, +0.07 Spearman rho for the strongest trio and +0.10 for the pre-declared one, and +0.06 and +0.07 when averaged over twenty fold partitions; against the panel mean, the primary test gives +0.118 on its EVA design set), and on PARA they reach parity under Spearman rho and a small, non-significant loss under Kendall tau-b, where one model already captures 85% of the human noise ceiling. It is not a feature-count artefact: giving the same combiner an equal number of pure holistic columns, split from the same repetitions, does not reproduce it. The gain costs a few hundred labels, which do not transfer between datasets, and 4.8x the API calls on EVA; we report it with paired bootstraps and Kendall tau-b, alongside a failed pre-registration and the configurations that lost. 

---
# ChipMEM: Verification-Grounded Memory for EDA Agents 

**Authors**: Abdulrahman AlRabah, Joshua Mabry, Dilek Hakkani-Tür, Abdussalam Alawini, Hamid Shojaei, Kartik Hegde, Sandesh Adhikary  

**Link**: [PDF](https://arxiv.org/pdf/2609.27067)  

**Abstract**: Large language model (LLM)-based agents use Electronic Design Automation (EDA) tools to generate and revise register-transfer-level (RTL) designs under synthesis and verification feedback. Recent methods learn from this feedback by distilling reusable skills from execution traces or by training on rewards derived from EDA-tools. Both methods are typically evaluated on the tasks that produced the experience. Repeated access to benchmark feedback on the same task can reward task-specific revision rather than creating reusable knowledge that transfers. We introduce ChipMEM, a verification-grounded memory layer for EDA agents. It combines cross-task procedural memory with within-trajectory statistical guidance. Its procedural component distills and stores a skill only after it passes synthesis, simulation, or formal checks, rather than relying on model self-assessments. A Bayesian component maintains hierarchical Beta estimates over tool-call outcomes and ranks recovery strategies that succeeded under comparable errors. A common adapter applies the same memory interface to RTL optimization and testbench-generation agents while preserving each domain's tools and acceptance criteria. We measure performance on training tasks and evaluate whether learned skills transfer to unseen tasks. On RTLRewriter-Bench, under matched model and tool settings, ChipMEM produces equivalence-passing outputs on 39/54 scored designs versus 35/54 without memory; on the 49-design short suite, mean area improvement is 8.69% versus 5.66%. On held-out CVDP tasks, ChipMEM with a frozen procedural library achieves 20/20 accepted outcomes versus 18/20 without memory in a single evaluation per setting. 

---
# ContraVis: Evidence-Grounded Visual Analytics for Contradiction Review in Legal Contracts 

**Authors**: Luis Sante, Paula Lima, Mariana Rocha, Jorge Poco  

**Link**: [PDF](https://arxiv.org/pdf/2609.27014)  

**Abstract**: Legal contracts are structurally complex documents in which contradictions may emerge across distant and interconnected provisions. Although large language models (LLMs) improve legal language understanding, contradiction analysis remains a human-centered and evidence-grounded review task. We present ContraVis, a visual analytics system for human-in-the-loop contradiction analysis in legal contracts. The system models contracts as typed paragraph graphs that combine explicit contractual references with semantic relationships between paragraphs. This graph plays a dual role: it conditions LLM reasoning and serves as the interactive representation the analyst explores, keeping model context and human inspection aligned across coordinated views. In a controlled comparison, graph-conditioned reasoning recovered more injected contradictions than standalone LLM analysis as contract length grew, while surfacing additional candidates for analyst validation. A formative study with contract-domain lawyers indicated that in-context evidence comparison supported contradiction validation, and we distill design implications for evidence-grounded, LLM-assisted document review. 

---
# Which Objectives Need a Dial? Predicting Objective Conflict and Covering Trade-offs in Steerable Pluralistic Alignment 

**Authors**: David Tsoi, Esra Dönmez  

**Link**: [PDF](https://arxiv.org/pdf/2609.26929)  

**Abstract**: People hold diverse, sometimes conflicting values, so no single aligned model can satisfy everyone. Pluralistic alignment therefore calls for steerable models that can balance competing objectives differently. Multi-Objective Direct Preference Optimization (MODPO) does this by using an objective weight to span a continuum of trade-offs. We study two questions: when can one model improve two objectives simultaneously, and how can many trade-offs be covered without training a separate model for each? Across seven objective pairs from HelpSteer and UltraFeedback, two pre-training measurements predict whether objectives align or conflict for human-annotated data, but not for AI-annotated data, where response length and repetition confound reward-model scores. For broader trade-off coverage, selecting the nearest trained model and merging model parameters both help, but neither consistently matches direct training. These findings yield practical guidance for building steerable models that serve diverse preferences. 

---
# Small Cues, Big Consequences: Learning Pivotal Cues for Multimodal Meme Classification 

**Authors**: Akshit Sharma, Prashant W. Patil  

**Link**: [PDF](https://arxiv.org/pdf/2609.26907)  

**Abstract**: Memes often derive their harmful, hateful, or sarcastic meaning from small but decisive visual, textual, or cross-modal cues. Existing multimodal classifiers can miss such evidence when relying mainly on global image-text representations. We introduce MemeCF, a cue-focused benchmark of 9,895 memes across harm, hate, and sarcasm, with annotations identifying the modality and rationale of the pivotal evidence. We also propose MemePIVOT, a local-global architecture for meme classification. MemePIVOT uses frozen CLIP features, unbalanced optimal transport to align words with image patches while allowing irrelevant evidence to remain unmatched, and an evidential fusion head to combine local grounding with global meme context under uncertainty. Experiments on HarMeme, PrideMM, and MemeCF show consistent gains over strong text-only, image-only, multimodal, and vision-language baselines. Cross-dataset and ablation results further show that explicit pivotal-evidence modeling improves robustness and contributes meaningfully beyond global multimodal representations. Our code and dataset are publicly available at this https URL 

---
# Text Scores Can Miss Waveform Use: A Qwen2-Audio Quantization Case Study 

**Authors**: Mengzhe Geng, Jinxi Jin, Junhao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2609.26823)  

**Abstract**: Post-training quantization of speech language models is often summarized with text-output scores and nominal bit widths. Those numbers alone do not establish behavior that depends on information missing from a transcript, or efficiency for a particular runtime. We introduce an evaluation protocol that separately tests lexical output, a transcript-insufficient endpoint, and a measured packed implementation. In a Qwen2-Audio case study, a translation-selected 6-bit allocation improves chrF by 2.36 on a frozen English-to-German replay, with paired 95% bootstrap interval [1.04, 3.62], but loses 3.91 percentage points on speaker-disjoint emotion recognition. At the same 6-bit budget, the uniform structural control reaches higher emotion accuracy than the selected allocation, and the front-layer control is also higher by point estimate on the same frozen set. At 7 bits, chrF improves by 3.28 with interval [2.08, 4.59], the emotion interval against FP16 includes zero, and a same-budget front-layer control still exceeds the selected allocation. A separate matched-budget 4.08-bit study finds roughly 10-point emotion deficits for every tested low-bit allocation and no selected-allocation advantage over frozen controls. Finally, a dequantized average-6-bit simulation retains the FP16 peak memory. This case study identifies a precision-dependent mismatch between lexical output, waveform-dependent behavior, and nominal precision. It does not establish a general failure of low-bit speech models or a deployment benefit for the selected allocation. 

---
