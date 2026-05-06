# Safety and accuracy follow different scaling laws in clinical large language models 

**Authors**: Sebastian Wind, Tri-Thien Nguyen, Jeta Sopa, Mahshad Lotfinia, Sebastian Bickelhaup, Michael Uder, Harald Köstler, Gerhard Wellein, Sven Nebelung, Daniel Truhn, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2605.04039)  

**Abstract**: Clinical LLMs are often scaled by increasing model size, context length, retrieval complexity, or inference-time compute, with the implicit expectation that higher accuracy implies safer behavior. This assumption is incomplete in medicine, where a few confident, high-risk, or evidence-contradicting errors can matter more than average benchmark performance. We introduce SaFE-Scale, a framework for measuring how clinical LLM safety changes across model scale, evidence quality, retrieval strategy, context exposure, and inference-time compute. To instantiate this framework, we introduce RadSaFE-200, a Radiology Safety-Focused Evaluation benchmark of 200 multiple-choice questions with clinician-defined clean evidence, conflict evidence, and option-level labels for high-risk error, unsafe answer, and evidence contradiction. We evaluated 34 locally deployed LLMs across six deployment conditions: closed-book prompting (zero-shot), clean evidence, conflict evidence, standard RAG, agentic RAG, and max-context prompting. Clean evidence produced the strongest improvement, increasing mean accuracy from 73.5% to 94.1%, while reducing high-risk error from 12.0% to 2.6%, contradiction from 12.7% to 2.3%, and dangerous overconfidence from 8.0% to 1.6%. Standard RAG and agentic RAG did not reproduce this safety profile: agentic RAG improved accuracy over standard RAG and reduced contradiction, but high-risk error and dangerous overconfidence remained elevated. Max-context prompting increased latency without closing the safety gap, and additional inference-time compute produced only limited gains. Worst-case analysis showed that clinically consequential errors concentrated in a small subset of questions. Clinical LLM safety is therefore not a passive consequence of scaling, but a deployment property shaped by evidence quality, retrieval design, context construction, and collective failure behavior. 

---
# Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems 

**Authors**: Yilun Zhao, Jinbiao Wei, Tingyu Song, Siyue Zhang, Chen Zhao, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2605.04018)  

**Abstract**: Reasoning-intensive retrieval aims to surface evidence that supports downstream reasoning rather than merely matching topical similarity. This capability is increasingly important for agentic search systems, where retrievers must provide complementary evidence across iterative search and synthesis. However, existing work remains limited on both evaluation and training: benchmarks such as BRIGHT provide narrow gold sets and evaluate retrievers in isolation, while synthetic training corpora often optimize single-passage relevance rather than evidence portfolio construction. We introduce BRIGHT-Pro, an expert-annotated benchmark that expands each query with multi-aspect gold evidence and evaluates retrievers under both static and agentic search protocols. We further construct RTriever-Synth, an aspect-decomposed synthetic corpus that generates complementary positives and positive-conditioned hard negatives, and use it to LoRA fine-tune RTriever-4B from Qwen3-Embedding-4B. Experiments across lexical, general-purpose, and reasoning-intensive retrievers show that aspect-aware and agentic evaluation expose behaviors hidden by standard metrics, while RTriever-4B substantially improves over its base model. 

---
# EQUITRIAGE: A Fairness Audit of Gender Bias in LLM-Based Emergency Department Triage 

**Authors**: Richard J. Young, Alice M. Matthews  

**Link**: [PDF](https://arxiv.org/pdf/2605.03998)  

**Abstract**: Emergency department triage assigns patients an acuity score that determines treatment priority, and clinical evidence documents persistent gender disparities in human acuity assessment. As hospitals pilot large language models (LLMs) as triage decision support, a critical question is whether these models reproduce or mitigate known biases. We present EQUITRIAGE, a fairness audit of LLM-based ESI assignment evaluating five models (Gemini-3-Flash, Nemotron-3-Super, DeepSeek-V3.1, Mistral-Small-3.2, GPT-4.1-Nano) across 374,275 evaluations on 18,714 MIMIC-IV-ED vignettes under four prompt strategies. Of 9,368 originals, 9,346 are paired with a gender-swapped counterfactual. All five models produced flip rates above a pre-registered 5% threshold (9.9% to 43.8%). Two showed directional female undertriage (DeepSeek F/M 2.15:1, Gemini 1.34:1); two were near-parity; one had high sensitivity with weak male-direction asymmetry. DeepSeek's directional bias coexisted with a low outcome-linked calibration gap (0.013 against MIMIC-IV admission), a Chouldechova-style dissociation between within-group calibration and between-pair counterfactual invariance. Demographic blinding reduced Gemini's flip rate to 0.5%; an age-preserving blind variant left DeepSeek with residual F/M 1.25, implicating age as a residual channel. Chain-of-thought prompting degraded accuracy for all five models. A two-model ablation reveals opposite underlying mechanisms for the same directional phenotype: in Gemini the signal is emergent in the combined name+gender swap, while in DeepSeek the gender token alone carries it. EQUITRIAGE shows that group parity, counterfactual invariance, and gender calibration are distinct fairness properties, that intervention effectiveness is model-dependent, and that per-model counterfactual auditing should precede clinical deployment. 

---
# Logical Consistency as a Bridge: Improving LLM Hallucination Detection via Label Constraint Modeling between Responses and Self-Judgments 

**Authors**: Hao Mi, Qiang Sheng, Shaofei Wang, Beizhe Hu, Yifan Sun, Zhengjia Wang, Hengqi Zeng, Yang Li, Danding Wang, Juan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.03971)  

**Abstract**: Large Language Models (LLMs) are prone to factual hallucinations, risking their reliability in real-world applications. Existing hallucination detectors mainly extract micro-level intrinsic patterns for uncertainty quantification or elicit macro-level self-judgments through verbalized prompts. However, these methods address only a single facet of the hallucination, focusing either on implicit neural uncertainty or explicit symbolic reasoning, thereby treating these inherently coupled behaviors in isolation and failing to exploit their interdependence for a holistic view. In this paper, we propose LaaB (Logical Consistency-as-a-Bridge), a framework that bridges neural features and symbolic judgments for hallucination detection. LaaB introduces a "meta-judgment" process to map symbolic labels back into the feature space. By leveraging the inherent logical bridge where response and meta-judgment labels are either the same or opposite based on the self-judgment's semantics, LaaB aligns and integrates dual-view signals via mutual learning and enhances the hallucination detection. Extensive experiments on 4 public datasets, across 4 LLMs, against 8 baselines demonstrate the superiority of LaaB. 

---
# Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators 

**Authors**: Mohamed Mady, Johannes Reschke, Björn Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2605.03969)  

**Abstract**: AI-generated text is nowadays produced at scale across domains and heterogeneous generation pipelines, making robustness to distribution shift a central requirement for supervised binary detectors. We train transformer-based detectors on HC3 PLUS and calibrate a single decision threshold by maximising balanced accuracy on held-out validation; this threshold is then kept fixed for all downstream test distributions, revealing domain- and generator-dependent error asymmetries under shift. We evaluate in-domain on HC3 PLUS, under cross-dataset transfer to the multi-domain, multi-generator M4 benchmark, and on the external AI-Text-Detection-Pile. Although base models achieve near-ceiling in-domain performance (up to 99.5% balanced accuracy), performance under shift is brittle and strongly model-dependent. Feature augmentation via attention-based linguistic feature fusion improves transfer, with our best model (DeBERTa-v3-base+FeatAttn) achieving 85.9% balanced accuracy on M4. Multi-seed experiments confirm high stability. Under the same fixed-threshold protocol, our model outperforms strong zero-shot baselines by up to +7.22 points. Category-level ablations further show that readability and vocabulary features contribute most to robustness under shift. Overall, these results demonstrate that feature augmentation and a modern DeBERTa backbone significantly outperform earlier BERT/RoBERTa models, while the fixed-threshold protocol provides a more realistic and informative assessment of practical detector robustness. 

---
# The Counterexample Game: Iterated Conceptual Analysis and Repair in Language Models 

**Authors**: Daniel Drucker, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2605.03936)  

**Abstract**: Conceptual analysis -- proposing definitions and refining them through counterexamples -- is central to philosophical methodology. We study whether language models can perform this task through iterated analysis and repair chains: one model instance generates counterexamples to a proposed definition, another repairs the definition, and the process repeats. Across 20 concepts and thousands of counterexample-repair cycles, we find that, although many LM-generated counterexamples are judged invalid by both expert humans and an LM judge, the LM judge accepts roughly twice as many as humans do. Nonetheless, per-item validity judgments are moderately consistent across humans and between humans and the LM. We further find that extended iteration produces increasingly verbose definitions without improving accuracy. We also see that some concepts resist stable definitions in general. These findings suggest that while LMs can engage in philosophical reasoning, the counterexample-repair loop hits diminishing returns quickly and could be a fruitful test case for evaluating whether LMs can sustain high-level iterated philosophical reasoning. 

---
# Atomic Fact-Checking Increases Clinician Trust in Large Language Model Recommendations for Oncology Decision Support: A Randomized Controlled Trial 

**Authors**: Lisa C. Adams, Linus Marx, Erik Thiele Orberg, Keno Bressem, Sebastian Ziegelmayer, Denise Bernhardt, Markus Graf, Marcus R. Makowski, Stephanie E. Combs, Florian Matthes, Jan C. Peeken  

**Link**: [PDF](https://arxiv.org/pdf/2605.03916)  

**Abstract**: Question: Does atomic fact-checking, which decomposes AI treatment recommendations into individually verifiable claims linked to source guideline documents, increase clinician trust compared to traditional explainability approaches?
Findings: In this randomized trial of 356 clinicians generating 7,476 trust ratings, atomic fact-checking produced a large effect on trust (Cohen's d = 0.94), increasing the proportion of clinicians expressing trust from 26.9% to 66.5%. Traditional transparency mechanisms showed a dose-response gradient of improvement over baseline (d = 0.25 to 0.50).
Meaning: Decomposing AI recommendations into individually verifiable claims linked to source guidelines produces substantially higher clinician trust than traditional explainability approaches in high-stakes clinical decisions. 

---
# Steer Like the LLM: Activation Steering that Mimics Prompting 

**Authors**: Geert Heyman, Frederik Vandeputte  

**Link**: [PDF](https://arxiv.org/pdf/2605.03907)  

**Abstract**: Large language models can be steered at inference time through prompting or activation interventions, but activation steering methods often underperform compared to prompt-based approaches. We propose a framework that formulates prompt steering as a form of activation steering and investigates whether distilling successful prompt steering behavior into simpler, interpretable models can close this gap. Our analysis reveals that popular activation steering methods are not faithful to the mechanics of prompt steering, which applies strong interventions on some tokens while barely affecting others. Based on these insights, we introduce Prompt Steering Replacement (PSR) models that estimate token-specific steering coefficients from the activations themselves and are trained to imitate prompt-based interventions. Experiments on three steering benchmarks across multiple language models show that PSR models outperform existing activation steering methods, especially when controlling for high-coherence completions, and also compare favorably to prompting on AxBench and persona steering. 

---
# CC-OCR V2: Benchmarking Large Multimodal Models for Literacy in Real-world Document Processing 

**Authors**: Zhipeng Xu, Junhao Ji, Zulong Chen, Zhenghao Liu, Qing Liu, Chunyi Peng, Zubao Qin, Ze Xu, Jianqiang Wan, Jun Tang, Zhibo Yang, Shuai Bai, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03903)  

**Abstract**: Large Multimodal Models (LMMs) have recently shown strong performance on Optical Character Recognition (OCR) tasks, demonstrating their promising capability in document literacy. However, their effectiveness in real-world applications remains underexplored, as existing benchmarks adopt task scopes misaligned with practical applications and assume homogeneous acquisition conditions. To address this gap, we introduce CC-OCR V2, a comprehensive and challenging OCR benchmark tailored to real-world document processing. CC-OCR V2 focuses on practical enterprise document processing tasks and incorporates hard and corner cases that are critical yet underrepresented in prior benchmarks, covering 5 major OCR-centric tracks: text recognition, document parsing, document grounding, key information extraction, and document question answering, comprising 7,093 high-difficulty samples. Extensive experiments on 14 advanced LMMs reveal that current models fall short of real-world application requirements. Even state-of-the-art LMMs exhibit substantial performance degradation across diverse tasks and scenarios. These findings reveal a significant gap between performance on current benchmarks and effectiveness in real-world applications. We release the full dataset and evaluation toolkit at this https URL. 

---
# MCJudgeBench: A Benchmark for Constraint-Level Judge Evaluation in Multi-Constraint Instruction Following 

**Authors**: Jaeyun Lee, Junyoung Koh, Zeynel Tok, Hunar Batra, Ronald Clark  

**Link**: [PDF](https://arxiv.org/pdf/2605.03858)  

**Abstract**: Multi-constraint instruction following requires verifying whether a response satisfies multiple individual requirements, yet LLM judges are often assessed only through overall-response judgments. We introduce MCJudgeBench, a benchmark for constraint-level judge evaluation in multi-constraint instruction following. Each instance includes an instruction, a candidate response, an explicit constraint list, per-constraint gold labels in {yes, partial, no}, and controlled response-side perturbations. The evaluation protocol further includes evaluation prompt variants to test judge stability. We evaluate proprietary and open-source LLM judges using both correctness and inconsistency metrics, distinguishing intrinsic inconsistency under stochastic decoding from procedural inconsistency under prompt and response perturbations. Our results show that judge reliability has multiple dimensions: strong overall performance does not guarantee equally reliable detection across label categories, especially for rarer partial and no cases. Judges with higher correctness do not always have lower inconsistency. Evaluation with reasoning improves correctness but does not uniformly improve stability. These findings motivate evaluating LLM judges at the constraint level to study these failure modes. 

---
# TRACE: A Metrologically-Grounded Engineering Framework for Trustworthy Agentic AI Systems in Operationally Critical Domains 

**Authors**: Serhii Zabolotnii  

**Link**: [PDF](https://arxiv.org/pdf/2605.03838)  

**Abstract**: We introduce TRACE, a cross-domain engineering framework for trustworthy agentic AI in operationally critical domains. TRACE combines a four-layer reference architecture with an explicit classical-ML vs. LLM-validator split (L2a/L2b), a stateful orchestration-and-escalation policy (L3), and bounded human supervision (L4); a metrologically grounded trust-metric suite mapped to GUM/VIM/ISO 17025; and a Model-Parsimony principle quantified by the Computational Parsimony Ratio (CPR). Three instantiations--clinical decision support, industrial multi-domain operations, and a judicial AI assistant--transfer the samearchitecture and metrics across principally different governance contexts. The L2a/L2b separation makes the use of large language models a deliberate design decision rather than an architectural default, with parsimony quantified through CPR. TRACE introduces CPR as a first-class design principle in trustworthy-AI engineering. 

---
# Reproducing Complex Set-Compositional Information Retrieval 

**Authors**: Vincent Degenhart, Dewi Timman, Arjen P. de Vries, Faegheh Hasibi, Mohanna Hoveyda  

**Link**: [PDF](https://arxiv.org/pdf/2605.03824)  

**Abstract**: Complex information needs may involve set-compositional queries using conjunction, disjunction, and exclusion, yet it remains unclear whether current retrieval paradigms genuinely satisfy such constraints or exploit `semantic shortcuts'. We conduct a reproducibility study to benchmark major retrieval families and reasoning-targeted methods on QUEST and QUEST+Variants, and introduce LIMIT+, a controlled benchmark where relevance depends on arbitrary attribute predicates and constraint satisfaction, and less on pretrained knowledge. Our findings show that (i) on QUEST, the best neural retrievers achieve an effectiveness that is more than double what can be achieved with BM25 (Recall@100 ${>}$0.41 vs.\ 0.20), but reasoning-targeted methods like ReasonIR and Search-R1 do not outperform general-purpose retrievers uniformly; (ii) on LIMIT+, gains fail to transfer, where the strongest QUEST method collapses from Recall@100${\approx}$0.42 to below 0.02, while classic lexical retrieval gains to ${\sim}$0.96. Lastly, (iii) stratifying by compositional depth reveals a consistent degradation across all methods, where algebraic sparse and lexical methods show more stable performance while dense approaches collapse. We release code and LIMIT+ data generation scripts to support future reproducibility and controlled evaluation. 

---
# Natural Language Processing: A Comprehensive Practical Guide from Tokenisation to RLHF 

**Authors**: Mullosharaf K. Arabov  

**Link**: [PDF](https://arxiv.org/pdf/2605.03799)  

**Abstract**: This preprint presents a systematic, research-oriented practicum that guides the reader through the entire modern NLP pipeline: from tokenisation and vectorisation to fine-tuning of large language models, retrieval-augmented generation, and reinforcement learning from human feedback. Twelve hands-on sessions combine concise theory with detailed implementation plans, formalised evaluation metrics, and transparent assessment criteria. The work is not a conventional textbook: it is designed as a reproducible research artefact where every session requires publishing code, models, and reports in public repositories. All experiments are conducted on a single evolving corpus, and the work advocates open-weight models over commercial APIs, with special attention to the Hugging Face ecosystem. The material is enriched by original research on low-resource languages, incorporating linguistic resources for Tajik and Tatar (subword tokenisers, embeddings, lexical databases, and transliteration benchmarks), demonstrating how modern NLP can be adapted to data-scarce environments. Designed for senior undergraduates, graduate students, and practising developers seeking to implement, compare, and deploy methods from classical ML to state-of-the-art LLM-based systems. 

---
# TriBench-Ko: Evaluating LLM Risks in Judicial Workflows 

**Authors**: Haesung Lee, Gyubin Choi, Eun-Ju Lee, So-Min Lee, Youkang Ko, Dogyoon Lim, Sung-Kyoung Jang, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.03792)  

**Abstract**: Large language models (LLMs) are increasingly integrated into legal workflows. However, existing benchmarks primarily address proxy tasks, such as bar examination performance or classification, which fail to capture the performance and risks inherent in day-to-day judicial processes. To address this, we publicly release TriBench-Ko, a Korean benchmark designed to evaluate potential deployment risks of LLMs within the context of verified judicial task requirements. It covers four core tasks: jurisprudence summarization, precedent retrieval, legal issue extraction, and evidence analysis. It jointly assesses model behavior across multiple deployment risk categories, including inaccuracy (hallucination, omission, statutory misapplication), biases (demographic, overcompliance), inconsistencies (prompt sensitivity, non-determinism), and adjudicative overreach. Each item is structured to systematically assess both task performance and a specific risk type based on real judicial decisions. Our evaluation of a range of contemporary LLMs reveals that many models frequently manifest significant risks, most notably struggling with precedent retrieval and failing to capture critical legal information. We provide a comprehensive diagnosis of these LLMs and pinpoint critical areas where LLM-generated outputs in judicial contexts necessitate rigorous inspection and caution. Our dataset and code are available at this https URL 

---
# Benchmarking Parameter-Efficient Fine-Tuning of Large Language Models for Low-Resource Tajik Text Generation with the Tajik Web Corpus 

**Authors**: Mullosharaf K. Arabov  

**Link**: [PDF](https://arxiv.org/pdf/2605.03742)  

**Abstract**: This paper is devoted to the adaptation of generative large language models for the Tajik language, a low-resource language with Cyrillic script. To overcome the shortage of digital text resources, the author created and publicly released the Tajik Web Corpus, the largest open-access corpus of Tajik, comprising 319,298 documents (~1.11 billion characters). On a subsample of 10,000 documents, 17 configurations were benchmarked, covering autoregressive, encoder-decoder, and encoder-only models with three fine-tuning strategies: full fine-tuning, LoRA, and QLoRA (ranks 8 and 16). Quality was assessed via perplexity and cross-entropy loss; peak GPU memory and training time were also recorded. Best results were achieved by Mistral 7B with QLoRA (r=16): mean perplexity 5.03, standard deviation 0.03. Increasing rank from 8 to 16 gave statistically insignificant improvement while raising memory consumption. For small GPT-2 family models, full fine-tuning yielded lower perplexity (3.48 for GPT-2 Medium) than LoRA (7.60-8.42), but induced catastrophic forgetting. The encoder-only XLM-RoBERTa showed the worst results (perplexity 59.3). The novelty lies in creating the largest verified Tajik corpus and the first systematic analysis of PEFT effectiveness for Tajik text generation. Practical value lies in recommendations for architecture and fine-tuning strategy selection, optimizing computational costs without substantial quality loss. 

---
# Segmenting Human-LLM Co-authored Text via Change Point Detection 

**Authors**: Mengchu Li, Jin Zhu, Jinglai Li, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03723)  

**Abstract**: The rise of large language models (LLMs) has created an urgent need to distinguish between human-written and LLM-generated text to ensure authenticity and societal trust. Existing detectors typically provide a binary classification for an entire passage; however, this is insufficient for human--LLM co-authored text, where the objective is to localize specific segments authored by humans or LLMs. To bridge this gap, we propose algorithms to segment text into human- and LLM-authored pieces. Our key observation is that such a segmentation task is conceptually similar to classical change point detection in time-series analysis. Leveraging this analogy, we adapt change point detection to LLM-generated text detection, develop a weighted algorithm and a generalized algorithm to accommodate heterogeneous detection score variability, and establish the minimax optimality of our procedure. Empirically, we demonstrate the strong performance of our approach against a wide range of existing baselines. 

---
# Rose-SQL: Role-State Evolution Guided Structured Reasoning for Multi-Turn Text-to-SQL 

**Authors**: Le Zhou, Feng Yao, Fengcai Qiao, Bo Xu, Fangyuan Wang, Boyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03720)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) trained with Long Chain-of-Thought have demonstrated remarkable capabilities in code generation and mathematical reasoning. However, their potential in multi-turn Text-to-SQL tasks remains largely underexplored. Existing approaches typically rely on unstable API-based inference or require expensive fine-tuning on small-scale models. In this work, we present Rose-SQL, a training-free framework that leverages small-scale LRMs through in-context learning to enable accurate context-dependent parsing. We introduce the Role-State, a fine-grained representation that bridges the structural gap between schema linking and SQL generation by serving as a structural blueprint. To handle conversational dependencies, Rose-SQL traces the evolution of Role-State through historical context via structural isomorphism checks, guiding the model to infer the possible SQL composition for the current question through verified interaction trajectories. Experiments on the SParC and CoSQL benchmarks show that, within the Qwen3 series, Rose-SQL outperforms in-context learning baselines at the 4B scale and substantially surpasses state-of-the-art fine-tuned models at the 8B and 14B scales, while showing consistent gains on additional reasoning backbones. 

---
# SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition 

**Authors**: Ruichu Cai, Juntao Gan, Miao Mai, Zhifeng Hao, Boyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03706)  

**Abstract**: Zero-shot Named Entity Recognition (ZS-NER) remains brittle under domain and schema shifts, where unseen label definitions often misalign with a large language model's (LLM's) intrinsic semantic organization. As a result, directly mapping entity mentions to fine-grained target labels can induce systematic semantic drift, especially when target schemas are novel or semantically overlapping. We propose \textbf{SAM-NER}, a three-stage framework based on \emph{Semantic Archetype Mediation} that stabilizes cross-domain transfer through an intermediate, domain-invariant archetype space. SAM-NER: (i) performs \emph{Entity Discovery} via cooperative extraction and consensus-based denoising to obtain high-coverage, high-fidelity entity spans; (ii) conducts \emph{Abstract Mediation} by projecting entities into a compact set of universal semantic archetypes distilled from high-level ontological abstractions; and (iii) applies \emph{Semantic Calibration} to resolve archetype-level predictions into target-domain types through constrained, definition-aligned inference with a frozen LLM. Experiments on the CrossNER benchmark show that SAM-NER consistently outperforms strong prior ZS-NER baselines in cross-domain settings. Our implementation will be open-sourced at this https URL. 

---
# SERE: Structural Example Retrieval for Enhancing LLMs in Event Causality Identification 

**Authors**: Zhifeng Hao, Zhongjie Chen, Junhao Lu, Shengyin Yu, Guimin Hu, Keli Zhang, Ruichu Cai, Boyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03701)  

**Abstract**: Event Causality Identification (ECI) requires models to determine whether a given pair of events in a context exhibits a causal relationship. While Large Language Models (LLMs) have demonstrated strong performance across various NLP tasks, their effectiveness in ECI remains limited due to biases in causal reasoning, often leading to overprediction of causal relationships (causal hallucination). To mitigate these issues and enhance LLM performance in ECI, we propose SERE, a structural example retrieval framework that leverages LLMs' few-shot learning capabilities. SERE introduces an innovative retrieval mechanism based on three structural concepts: (i) Conceptual Path Metric, which measures the conceptual relationship between events using edit distance in ConceptNet; (ii) Syntactic Metric, which quantifies structural similarity through tree edit distance on syntactic trees; and (iii) Causal Pattern Filtering, which filters examples based on predefined causal structures using LLMs. By integrating these structural retrieval strategies, SERE selects more relevant examples to guide LLMs in causal reasoning, mitigating bias and improving accuracy in ECI tasks. Extensive experiments on multiple ECI datasets validate the effectiveness of SERE. The source code is publicly available at this https URL. 

---
# A Comprehensive Analysis of Tokenization and Self-Supervised Learning in End-to-End Automatic Speech Recognition applied on French Language 

**Authors**: Thibault Bañeras-Roux, Mickael Rouvier, Jane Wottawa, Richard Dufour  

**Link**: [PDF](https://arxiv.org/pdf/2605.03696)  

**Abstract**: The performance of end-to-end automatic speech recognition (ASR) systems enables their increasing integration into numerous applications. While there are various benefits to such speech-to-text systems, the choice of hyperparameters and models plays a crucial role in their performance. Typically, these choices are determined by considering only the character (CER) and/or word error rate (WER) metrics. However, it has been shown in several studies that these metrics are largely incomplete and fail to adequately describe the downstream application of automatic transcripts. In this paper, we conduct a qualitative study on the French language that investigates the impact of subword tokenization algorithms and self-supervised learning models from different linguistic and acoustic perspectives, using a comprehensive set of evaluation metrics. 

---
# A Paradigm for Interpreting Metrics and Identifying Critical Errors in Automatic Speech Recognition 

**Authors**: Thibault Bañeras-Roux, Mickael Rouvier, Jane Wottawa, Richard Dufour  

**Link**: [PDF](https://arxiv.org/pdf/2605.03671)  

**Abstract**: The most commonly used metrics for evaluating automatic speech transcriptions, namely Word Error Rate (WER) and Character Error Rate (CER), have been heavily criticized for their poor correlation to human perception and their inability to take into account linguistic and semantic information. While metric-based embeddings, seeking to approximate human perception, have been proposed, their scores remain difficult to interpret, unlike WER and CER. In this article, we overcome this problem by proposing a paradigm that consists in incorporating a chosen metric into it in order to obtain an equivalent of the error rate: a Minimum Edit Distance (minED). This approach parallels transcription errors with their human perception, also allowing an original study of the severity of these errors from a human perspective. 

---
# Annotation Quality in Aspect-Based Sentiment Analysis: A Case Study Comparing Experts, Students, Crowdworkers, and Large Language Model 

**Authors**: Niklas Donhauser, Jakob Fehle, Nils Constantin Hellwig, Markus Weinberger, Udo Kruschwitz, Christian Wolff  

**Link**: [PDF](https://arxiv.org/pdf/2605.03624)  

**Abstract**: Aspect-Based Sentiment Analysis (ABSA) enables fine-grained opinion analysis by identifying sentiments toward specific aspects or targets within a text. While ABSA has been widely studied for English, research on other languages such as German remains limited, largely due to the lack of high-quality annotated datasets. This paper examines how different annotation sources influence the development of German ABSA. To this end, an existing dataset is re-annotated by experts to establish a ground truth, which serves as a reference for evaluating annotations produced by students, crowdworkers, Large Language Models (LLMs), and experts. Annotation quality is compared using Inter-Annotator Agreement (IAA) and its impact on downstream model performance for different ABSA subtasks. The evaluation focuses on Aspect Category Sentiment Analysis (ACSA) and Target Aspect Sentiment Detection (TASD). We apply State-of-the-Art (SOTA) methods for ABSA, including BERT-, T5-, and LLaMA-based approaches to assess performance differences, spanning fine-tuning and in-context learning with instruction prompts. The findings provide practical insights into trade-offs between annotation reliability and efficiency, offering guidance for dataset construction in under-resourced Natural Language Processing (NLP) scenarios. 

---
# BIT.UA-AAUBS at ArchEHR-QA 2026: Evaluating Open-Source and Proprietary LLMs via Prompting in Low-Resource QA 

**Authors**: Richard A. A. Jonker, Alexander Christiansen, Alexandros Maniatis, Rúben Garrido, Rogério Braunschweiger de Freitas Lima, Roman Jurowetzki, Sérgio Matos  

**Link**: [PDF](https://arxiv.org/pdf/2605.03618)  

**Abstract**: This paper presents the joint participation of the this http URL and AAUBS groups in the ArchEHR-QA 2026 shared task, which focuses on clinical question answering and evidence grounding in a low-resource setting. Due to the absence of training data and the strict data privacy constraints inherent to the healthcare domain (e.g. GDPR), we investigate the capabilities of Large Language Models (LLMs) without weight updates. We evaluate several state-of-the-art proprietary models and locally deployable open-source alternatives using various prompt engineering strategies, including task decomposition, Chain-of-Thought, and in-context learning. Furthermore, we explore majority voting and LLM-as-a-judge ensembling techniques to maximize predictive robustness. Our results demonstrate that while proprietary models exhibit strong resilience to prompt variations, domain-adapted open-source models (such as MedGemma 3 27B) achieve highly competitive performance when paired with the right prompt. Overall, our prompt-based approach proved highly effective, securing 1st place in Subtask 4 (evidence citation alignment) and 3rd place in Subtask 3 (patient-friendly answer generation). All code, results, and prompts are available on our GitHub repository: this https URL. 

---
# AfriVox-v2: A Domain-Verticalized Benchmark for In-the-Wild African Speech Recognition 

**Authors**: Busayo Awobade, Gabrial Zencha Ashungafac, Tobi Olatunji  

**Link**: [PDF](https://arxiv.org/pdf/2605.03590)  

**Abstract**: Recent large language models (LLMs) show strong speech recognition and translation capabilities for high-resource languages. However, African languages remain dramatically underrepresented in benchmarks, limiting their practical use in low-resource settings. While early benchmarks tested African languages and accents, they lacked exhaustive real-world noise and granular domain evaluations. We present AfriVox-v2, a comprehensive benchmark designed to test speech models under realistic African deployment conditions. AfriVox-v2 introduces "in the wild" unscripted audio for all supported languages. We also introduce strict domain verticalization, evaluating model accuracy across ten sectors including government, finance, health, and agriculture and conducting targeted tests on numbers and named entities. Finally, we benchmark a new generation of speech models, including Sahara-v2, Gemini 3 Flash, and the Omnilingual CTC models. Our results expose the true generalization gap of modern speech models in specialized, noisy African contexts and provide a reliable blueprint for developers building localized voice AI. 

---
# PatRe: A Full-Stage Office Action and Rebuttal Generation Benchmark for Patent Examination 

**Authors**: Qiyao Wang, Xinyi Chen, Longze Chen, Hongbo Wang, Hamid Alinejad-Rokny, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03571)  

**Abstract**: Patent examination is a complex, multi-stage process requiring both technical expertise and legal reasoning, increasingly challenged by rising application volumes. Prior benchmarks predominantly view patent examination as discriminative classification or static extraction, failing to capture its inherently interactive and iterative nature, similar to the peer review and rebuttal process in academic publishing. In this paper, we introduce PatRe, the first benchmark that models the full patent examination lifecycle, including Office Action generation and applicant rebuttal. PatRe comprises 480 real-world cases and supports both oracle and retrieval-simulated evaluation settings. Our benchmark reframes patent examination as a dynamic, multi-turn process of justification and response. Extensive experiments across various LLMs reveal critical insights into model performance, including differences between proprietary and open-source models, as well as task asymmetries between examiner analysis and applicant-side rebuttal. These findings highlight both the potential and current limitations of LLMs in modeling complex, real-world legal reasoning and technical novelty judgment in patent examination. We release our code and dataset to facilitate future research on patent examination modeling. 

---
# SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation 

**Authors**: Jingxi Qiu, Zeyu Han, Cheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03534)  

**Abstract**: Retrieval-augmented generation (RAG) grounds answers in retrieved passages, but retrieval is not verification: a passage can be topical and still fail to justify the answer. We frame this gap as evidence sufficiency verification for selective RAG answering: given a question, a candidate answer, and retrieved evidence, predict whether the evidence supports, refutes, or is insufficient, and abstain unless support is established. We present SURE-RAG, a transparent aggregation protocol built on the observation that evidence sufficiency is a set-level property: missing hops and unresolved conflicts cannot be detected by independent passage scoring. A shared pair-level claim-evidence verifier produces local relation distributions, which SURE-RAG aggregates into interpretable answer-level signals -- coverage, relation strength, disagreement, conflict, and retrieval uncertainty -- yielding a three-way decision and an auditable selective score. We evaluate on HotpotQA-RAG v3, a controlled multi-hop benchmark, under an artifact-aware protocol (shortcut baselines, counterfactual swaps, no-oracle checks, GPT-4o audits). Calibrated SURE-RAG reaches 0.9075 Macro-F1 (0.8951 +/- 0.0069), substantially above DeBERTa mean-pooling (0.6516) and a GPT-4o judge (0.7284), while matching a strong but opaque concat cross-encoder (0.8888 +/- 0.0109) with full auditability. Risk at 30% coverage drops from 0.2588 to 0.1642, a 37% reduction in unsafe answers. To deliberately probe the task boundary, we further contrast SURE-RAG with GPT-4o on HaluBench unsafe detection: the ranking reverses (0.3343 vs 0.7389 unsafe-F1), establishing that controlled sufficiency verification and natural hallucination detection are distinct problems. 

---
# Revisiting Graph-Tokenizing Large Language Models: A Systematic Evaluation of Graph Token Understanding 

**Authors**: Zhongjian Zhang, Yue Yu, Mengmei Zhang, Junping Du, Xiao Wang, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03514)  

**Abstract**: The remarkable success of large language models (LLMs) has motivated researchers to adapt them as universal predictors for various graph tasks. As a widely recognized paradigm, Graph-Tokenizing LLMs (GTokenLLMs) compress complex graph data into graph tokens and treat them as prefix tokens for querying LLMs, leading many to believe that LLMs can understand graphs more effectively and efficiently. In this paper, we challenge this belief: \textit{Do GTokenLLMs fully understand graph tokens in the natural-language embedding space?} Motivated by this question, we formalize a unified framework for GTokenLLMs and propose an evaluation pipeline, \textbf{GTEval}, to assess graph-token understanding via instruction transformations at the format and content levels. We conduct extensive experiments on 6 representative GTokenLLMs with GTEval. The primary findings are as follows: (1) Existing GTokenLLMs do not fully understand graph tokens. They exhibit over-sensitivity or over-insensitivity to instruction changes, and rely heavily on text for reasoning; (2) Although graph tokens preserve task-relevant graph information and receive attention across LLM layers, their utilization varies across models and instruction variants; (3) Additional instruction tuning can improve performance on the original and seen instructions, but it does not fully address the challenge of graph-token understanding, calling for further improvement. 

---
# Rational Communication Shapes Morphological Composition 

**Authors**: Fengyuan Yang, Yongqian Peng, Yuxi Ma, Chenheng Xu, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03510)  

**Abstract**: Human languages expand vocabularies by combining existing morphemes rather than inventing arbitrary forms. Communicative efficiency shapes lexical systems at multiple levels (Gibson et al., 2019), yet morphological composition -- combining morphemes through compounding or affixation -- has rarely been modeled as a historically situated speaker choice among competing morpheme sequences, leaving unanswered why a language settles on one morpheme combination over other plausible alternatives. We ask whether a trade-off between listener recoverability and speaker production cost can predict attested compositions over contemporaneously available alternatives. Here we show, within the Rational Speech Act (RSA) framework (Frank & Goodman, 2012; Goodman & Frank, 2016) using a time-indexed lexicon constructed from Corpus of Historical American English (COHA) and Corpus of Contemporary American English (COCA), that across 4323 naturally occurring English compounds and derivations spanning 1820--2019, attested compositions are systematically ranked above unattested alternatives generated from contemporaneously available morphemes. Models integrating semantic informativeness with production cost outperform semantic-only and cost-only baselines on Mean Reciprocal Rank (MRR) and top-k accuracy (Acc@k), with the advantage of the Pragmatic Speaker model ($S_1$) over the semantic-only baseline growing as the candidate set expands, where meaning alone leaves morphological choice underdetermined. These findings suggest that lexicalization reflects a communicative trade-off between expressiveness and efficiency, extending rational accounts of communication from utterance-level choice to the internal structure of words. 

---
# CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification 

**Authors**: Severin Ye, Xiao Kong, Xiaopeng He, Guangsu Yan, Dongsuk Oh  

**Link**: [PDF](https://arxiv.org/pdf/2605.03476)  

**Abstract**: Discharge summaries require extracting critical information from lengthy electronic health records (EHRs), a process that is labor-intensive when performed manually. Large language models (LLMs) can improve generation efficiency; however, they are prone to producing faithfulness hallucinations, statements that contradict source records, posing direct risks to patient safety. To address this, we present CuraView, a multi-agent framework for sentence-level detection and evidence-grounded explanation of faithfulness hallucinations in discharge summaries. CuraView constructs a GraphRAG-based knowledge graph from patient-level EHRs and implements a closed-loop generation-detection pipeline with sentence-level evidence retrieval and classification spanning four evidence grades from strong support to direct contradiction (E1-E4), yielding structured and interpretable evidence chains.
We evaluate CuraView on a subset of 250 patients from the Discharge-Me benchmark, with 50 patients held out for testing. Our fine-tuned Qwen3-14B detection model achieves an F1 of 0.831 on the safety-critical E4 metric (90.9% recall, 76.5% precision) and an F1 of 0.823 on E3+E4, representing a 50.0% relative improvement over the base model and outperforming RAGTruth-style and QAGS-style baselines. These results demonstrate that evidence-chain-based graph retrieval verification substantially improves the factual reliability of clinical documentation, while simultaneously producing reusable annotated datasets for downstream model training and distillation. 

---
# Detecting Stealth Sycophancy in Mental-Health Dialogue with Dynamic Emotional Signature Graphs 

**Authors**: Tianze Han, Beining Xu, Hanbo Zhang, Yongming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03472)  

**Abstract**: As conversational AI therapists are increasingly used in psychological support settings, reliable offline evaluation of therapeutic response quality remains an open problem. This paper studies multi-domain support-dialogue evaluation without relying on large language models as final judges. We use a direct LLM judge as a baseline that reads raw dialogue text and predicts whether the target response is harmful, productive, or neutral. We find that direct LLM judges and symmetric text-similarity metrics are poorly aligned with therapeutic quality because the target label depends on clinical direction: whether the response moves the user state toward regulation or reframing, leaves it broadly unchanged, or reinforces deterioration through higher risk affect or cognitive-distortion mass. To address this issue, we propose Dynamic Emotional Signature Graphs (DESG), a model-agnostic evaluator that represents dialogue windows with decoupled clinical states and scores them using asymmetric clinical geometry. We evaluate DESG on a constructed diagnostic stress-test benchmark of 3{,}000 dialogue windows from EmpatheticDialogues, ESConv, and CRADLE-Dialogue, covering peer support, counseling dialogue, and crisis-oriented interaction. On the 600-window held-out test aggregate, DESG-Ensemble achieves 0.9353 macro-F1, exceeding ConcatANN by 1.51 percentage points, BERTScore by 19.63 points, and TRACT by 33.81 points. Feature ablations, artifact controls, a 100-window blinded adjudicator audit, and qualitative disagreement cases indicate that the clinical state manifold is the main discriminative substrate, while graph-based trajectory components provide asymmetric scoring and interpretable diagnostics rather than serving as the sole source of performance. 

---
# Retrieving Floods without Floodlights: Topic Models as Binary Classifiers for Extreme Climate Events in German News 

**Authors**: Brielen Madureira, Mariana Madruga de Brito, Andreas Niekler  

**Link**: [PDF](https://arxiv.org/pdf/2605.03450)  

**Abstract**: In studies of media coverage of extreme climate events, NLP methods have become indispensable for identifying relevant texts in large news databases. Still, enough annotated data to train accurate deep learning-based classifiers from scratch is often not available. Topic Models have the advantage of being both unsupervised and interpretable, but are typically used only for exploratory analysis or data characterisation. In this study, we investigate how to employ Topic Models as binary classifiers for refining the retrieval of relevant news about seven types of extreme climate events in the German media. Our method relies on the posterior distributions estimated by Topic Models to select relevant documents, without modifying their training procedure. Using an annotated sample to guide the evaluation, we show that the probabilities assigned to keywords used to query news databases can also be informative for selecting relevant topics and improve sample precision. We compare our results to a fine-tuned text embedding classifier and an open-weight LLM, discussing observed trade-offs, e.g. the LLM's lowest precision. Moreover, we show that results are hazard-dependent, which speaks against considering climate events as a single category in NLP tasks. 

---
# An ERP Study of Recursive Possessive Parsing in ASD Children and Its Cognitive Neuro Mechanisms 

**Authors**: Fu Chenxi, Wang Xiaoyi, Zhuang Ziman, Yang Caimei  

**Link**: [PDF](https://arxiv.org/pdf/2605.03447)  

**Abstract**: Recursive structures are a core property of human language, yet little is known about how children with autism spectrum disorder (ASD) process complex recursion. This ERP study investigated the online processing of two-level recursive possessive structures in Mandarin-speaking children with ASD (n = 12) compared to typically developing (TD) peers (n = 12) using a sentence-picture matching paradigm. ERPs were analyzed for P200 (150-250 ms), N400 (300-500 ms), and P600 (500-1000 ms). Results showed that ASD children exhibited significantly reduced P200 amplitudes and failed to show the typical posterior grammaticality effect, indicating atypical early perceptual processing. No robust N400 violation effect was observed in either group, confirming the mismatch was not a semantic anomaly; however, ASD children showed a reversed anterior effect and an attenuated posterior effect. For the P600, ASD children had significantly reduced amplitudes, no posterior grammaticality effect, and a trend toward delayed latency, reflecting a core deficit in syntactic reanalysis. These findings demonstrate that while lexical-semantic processing is relatively preserved in ASD, the online syntactic computation required for recursion is severely impaired, supporting modular dissociation accounts of language in autism. 

---
# Sentiment Analysis of Indonesian Spotify Reviews Using Machine Learning and BiLSTM 

**Authors**: Uliano Wilyam Purba, Andre Hadiman Rotua Parhusip, Sahid Maulana, Luluk Muthoharoh, Ardika Satria, Martin C. T. Manullang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03443)  

**Abstract**: This paper benchmarks classical machine learning and deep learning approaches for three-class sentiment classification of Indonesian Spotify reviews. Using 100,000 scraped reviews and 70,155 cleaned samples, the study compares Support Vector Machine, Multinomial Naive Bayes, and Decision Tree models with a two-layer BiLSTM. Both approaches use the same preprocessing pipeline, including slang normalization, stopword removal, and stemming. Decision Tree achieves the best performance among the classical models, while BiLSTM attains the highest weighted F1-score overall but fails on the minority neutral class. The paper concludes that BiLSTM is stronger for overall sentiment detection, whereas machine learning with SMOTE provides more balanced three-class performance. 

---
# A Comparison of Traditional Machine Learning Algorithms and LSTM-Based Deep Learning Models for Email Sentiment Analysis 

**Authors**: Virdio Samuel Saragih, Baruna Abirawa, Kartini Lovian Simbolon, Luluk Muthoharoh, Ardika Satria, Martin C.T. Manullang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03440)  

**Abstract**: The rapid growth of electronic communication has necessitated more robust systems for email classification and sentiment detection. This study presents a comparative performance analysis between traditional machine learning algorithms and deep learning architectures, specifically focusing on Support Vector Machines (SVMs), Logistic Regression, Naive Bayes, and Long Short-Term Memory (LSTM). Utilizing Word2Vec embeddings for feature representation, our experimental results indicate that the SVM model with a linear kernel achieves the highest efficiency and accuracy, reaching a peak performance of 98.74%. While the LSTM model demonstrates exceptional recall capabilities in detecting spam-related sentiments, it requires significantly more computational time compared to discriminative statistical models. Detailed evaluations via confusion matrices further reveal that traditional classifiers remain highly robust for dense vector spaces. This research concludes that for email detection tasks, SVM offers the most optimal balance between predictive precision and processing speed. These findings provide critical insights for developing high-performance automated email filtering systems in professional and academic environments. 

---
# Benchmarking Logistic Regression, SVM, Naive Bayes, and IndoBERT Fine-Tuning for Sentiment Analysis on Indonesian Product Reviews 

**Authors**: Nabila Zakiyah Zahra, Salwa Farhanatussaidah, Nasywa Nur Afifah, Luluk Muthoharoh, Ardika Satria, Martin C.T. Manullang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03439)  

**Abstract**: The exponential growth of e-commerce platforms in Indonesia has generated a massive volume of user-generated product reviews. Analyzing the sentiment of these reviews is critical for measuring customer satisfaction and identifying product issues at scale. This paper benchmarks traditional Machine Learning (ML) approaches against a Transformer-based Deep Learning model for a three-class sentiment analysis task (positive, neutral, negative) on the Tokopedia Product Reviews 2025 dataset. We implemented Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction coupled with three algorithms: Logistic Regression, Linear Support Vector Machine (SVM), and Multinomial Naive Bayes as robust baselines. Subsequently, we fine-tuned the IndoBERT model (indobenchmark/indobert-base-p1) for contextual sequence classification. To computationally address the severe class imbalance inherent in e-commerce feedback, we applied balanced class weights for the baseline models and engineered a custom weighted cross-entropy loss function within the IndoBERT training loop, following the broader motivation of imbalanced-learning research. Our comprehensive evaluation using Accuracy, Macro F1-score, and Weighted F1-score revealed that the traditional Linear SVC model significantly outperformed the IndoBERT model in our experimental setup, achieving an Accuracy of 97.60% and a Macro F1-score of 0.5510, compared to IndoBERT's 88.70% and 0.5088. Detailed analysis indicates that this performance gap was primarily driven by discrepancies in the data sampling regimes, where baselines utilized the full corpus while the Transformer was constrained to a sampled subset. Finally, we demonstrate the practical viability of our pipeline by deploying the final sentiment classification model as an interactive Gradio web application. 

---
# Geolocating News about Extreme Climate Events: A Comparative Analysis of Off-the-Shelf Tools for Toponym Identification in German 

**Authors**: Brielen Madureira, Mariana Madruga de Brito, Andreas Niekler  

**Link**: [PDF](https://arxiv.org/pdf/2605.03414)  

**Abstract**: Determining the geolocation of extreme climate events and disasters in texts is a common problem in climate impact and adaptation research. Named-entity recognition (NER) tools are typically used to identify a pool of toponyms that serve as candidate event locations. In this study, we conduct a comparative analysis of three off-the-shelf NER tools, namely Flair, Spacy and Stanza. We describe and quantify differences between their outputs for German news articles and evaluate them extrinsically based on three methods to determine the country where events took place. We show how their contrasts are propagated into downstream tasks and can yield distinct decisions about a document's geographical focus, which, in turn, can impact conclusions about countries' prominence in German media. 

---
# From prompting to evidence-based translation: A RAG+prompt system for Japanese-Chinese translation and its pedagogical potential 

**Authors**: Wenshi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03387)  

**Abstract**: Large language models perform well on high-resource pairs but are less reliable for Japanese-Chinese sentences containing noun-modifying clause constructions (NMCCs). This study evaluates a retrieval-augmented generation RAG+Prompt translation system that integrates linguistic analysis, embedding-based retrieval, prompt construction, and LLM generation without modifying the base model. The analysis module outputs A1 (inner vs. outer NMCC) and A2 (risk predictions: lexical choice/NMCC handling/word order/style/register); top-k = 5 similar Ja-Zh examples (L2 distance) and A1/A2 are inserted into an enhanced prompt. Using GPT-4o and a 66-sentence test set, we compare six knowledge-base sizes (0/100/200/500/1,000/2,000). Macro-averaged sentence-level BLEU (1-4-gram with brevity penalty; cased; Chinese at the character level) is the sole metric. Mean BLEU increases from 24.28 at 0 (RAG disabled) to 29.96 at 2,000 (+5.68; +23.4%). The upward trend holds across sizes, with larger knowledge bases yielding higher scores. We conclude that the RAG+Prompt translation system improves Ja-Zh translation of sentences containing NMCCs in an interpretable and auditable manner. Limitations include one base model, one metric, and reliance on published texts and commercial APIs; future work will broaden genres, language pairs, and evaluation metrics. 

---
# When to Think, When to Speak: Learning Disclosure Policies for LLM Reasoning 

**Authors**: Jiaqi Wei, Xuehang Guo, Pengfei Yu, Xiang Zhang, Wanli Ouyang, Siqi Sun, Qingyun Wang, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2605.03314)  

**Abstract**: In single-stream autoregressive interfaces, the same tokens both update the model state and constitute an irreversible public commitment. This coupling creates a \emph{silence tax}: additional deliberation postpones the first \emph{task-relevant} content, while naive early streaming risks premature commitments that bias subsequent generations. We introduce \textbf{\emph{Side-by-Side (SxS)}} Interleaved Reasoning, which makes \emph{disclosure timing} a controllable decision within standard autoregressive generation. SxS interleaves partial disclosures with continued private reasoning in the same context, but releases content only when it is \emph{supported} by the reasoning so far. To learn such pacing without incentivizing filler, we construct entailment-aligned interleaved trajectories by matching answer prefixes to supporting reasoning prefixes, then train with SFT to acquire the dual-action semantics and RL to recover reasoning performance under the new format. Across two Qwen3 architectures/scales (MoE \textbf{Qwen3-30B-A3B}, dense \textbf{Qwen3-4B}) and both in-domain (AIME25) and out-of-domain (GPQA-Diamond) benchmarks, SxS improves accuracy--\emph{content-latency} Pareto trade-offs under token-level proxies (e.g., inter-update waiting). 

---
# SHIELD: A Diverse Clinical Note Dataset and Distilled Small Language Models for Enterprise-Scale De-identification 

**Authors**: Jose D. Posada, David Love, Somalee Datta, Priya Desai  

**Link**: [PDF](https://arxiv.org/pdf/2605.03301)  

**Abstract**: De-identification of clinical text remains essential for secondary use of electronic health records (EHRs), yet public benchmarks such as i2b2 2006/2014 are over a decade old and lack the semantic and demographic diversity of modern narratives. While Large Language Models (LLMs) achieve state-of-the-art zero-shot extraction, enterprise deployment is hindered by compute costs and governance restricting Protected Health Information (PHI) from cloud APIs. We introduce SHIELD (Synthetic Human-annotated Identifier-replaced Entries for Learning and De-identification), a diverse dataset of 1,394 notes with 10,505 gold-standard PHI spans across 9 categories, built via set-cover diversity sampling with human-in-the-loop adjudication. We evaluate four LLMs (two proprietary, two open-weight) to establish a performance ceiling, then distill these capabilities into locally deployable Small Language Models (SLMs). Distributional analysis using Frechet Text Distance and Jensen-Shannon Divergence confirms SHIELD occupies a distinct region of biomedical embedding and vocabulary space versus legacy benchmarks. Our best distilled model matches its teacher on structured PHI categories (DATE, DOCTOR, ID, PATIENT, PHONE) and achieves micro-averaged span-level precision of 0.88 and recall of 0.86 on standard workstation hardware. Cross-dataset evaluation shows diversity-trained models generalize well on universal structured PHI, while institution-specific entities remain hard to transfer, suggesting optimal deployment combines broad-coverage models with specialized models for high-volume notes. We publicly release the SHIELD dataset and the distilled DeBERTa v3 model. 

---
# LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models 

**Authors**: Minh Chu Xuan, Tien-Phat Nguyen, Linh Ngo Van, Dinh Viet Sang, Nguyen Thi Ngoc Diep, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.03299)  

**Abstract**: Cross-lingual topic modeling aims to discover shared semantic structures across languages, yet existing models depend on sparse bilingual resources and often yield incoherent or weakly aligned topics. Recent LLM-based refinements improve interpretability but are costly, document-level, and prone to hallucination, with prior white-box approaches requiring inaccessible token probabilities. We propose LLM-XTM, a framework that integrates LLM-guided topic refinement with self-consistency uncertainty quantification, enabling black-box, stable, and scalable enhancement of cross-lingual topic models. Experiments on multilingual corpora show that LLM-XTM achieves superior topic coherence and alignment while reducing reliance on bilingual dictionaries and expensive LLM calls. 

---
# S^2tory: Story Spine Distillation for Movie Script Summarization 

**Authors**: Mingzhe Lu, Yanbing Liu, Qihao Wang, Jiarui Zhang, Jiayue Wu, Yue Hu, Yunpeng Li, Yangyan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03244)  

**Abstract**: Movie scripts pose a fundamental challenge for automatic summarization due to their non-linear, cross-cut narrative structure, which makes surface-level saliency methods ineffective at preserving core story progression. To address this, we introduce S^2tory (Story Spine Distillation), a narratology-grounded framework that leverages character development trajectories to identify plot nuclei, the essential events that drive the narrative forward, while filtering out peripheral satellite events that merely enrich atmosphere or emotion. Our Narrative Expert Agent (NEAgent) performs theory-constrained reasoning, whose distilled knowledge conditions a small model to identify plot nuclei. Another model then uses these plot nuclei to generate the summary. Experiments on the MovieSum dataset demonstrate state-of-the-art semantic fidelity at approximately 3.5x compression, and zero-shot evaluation on BookSum confirms strong out-of-domain generalization. Human evaluation further validates that narratological theory provides an indispensable foundation for modeling complex, non-linear narratives. 

---
# Sparse Memory Finetuning as a Low-Forgetting Alternative to LoRA and Full Finetuning 

**Authors**: Prakhar Gupta, Garv Shah, Satyam Goyal, Anirudh Kanchi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03229)  

**Abstract**: Adapting a pretrained language model to a new task often hurts the general capabilities it already had, a problem known as catastrophic forgetting. Sparse Memory Finetuning (SMF) tries to avoid this by adding key-value memory layers to the model and, on each training step, updating only the small set of memory rows that the current batch reads most heavily. We re-implement SMF on Qwen-2.5-0.5B-Instruct and compare it with LoRA and full finetuning on MedMCQA, a 4-choice medical exam task, using WikiText perplexity and TriviaQA accuracy as forgetting probes. SMF improves MedMCQA by 2.5 percentage points while keeping both forgetting probes within roughly 1 point of the base model, whereas LoRA and full finetuning achieve larger gains but with clear drift on both. We also compare two row-selection rules (KL-divergence and TF-IDF), which balance the two forgetting metrics differently. 

---
# Geometric Deviation as an Unsupervised Pre-Generation Reliability Signal: Probing LLM Representations for Answerability 

**Authors**: Yucheng Du  

**Link**: [PDF](https://arxiv.org/pdf/2605.03196)  

**Abstract**: A reliable language model should be able to signal, prior to generation, when a query falls outside its knowledge. We investigate whether representation geometry can provide such a pre-generation signal by measuring the deviation of hidden states from an answerable reference set, requiring no labeled failure data and no access to model outputs.
Across three instruction-tuned models (Llama 3.1-8B, Qwen 2.5-7B, and Mistral-7B-Instruct) and three prompt forms (Math, Fact, Code), we find that geometry primarily encodes task form. Within mathematical prompts, unanswerable inputs consistently deviate from the answerable centroid, yielding strong separation (ROC-AUC 0.78-0.84). This single-pass pre-generation signal outperforms a simple refusal baseline and compares favorably to self-consistency. It also captures cases where models do not explicitly refuse.
In contrast, no reliable geometric signal emerges for factual prompts, indicating that the effect is form-conditional rather than universal. Code prompts show large effect sizes with higher variance, suggesting partial generalization beyond mathematical form.
A layer-wise analysis reveals that the signal arises in early layers and gradually attenuates toward the output. These results suggest that answerability-related geometry is established before the final stages of generation. Together, these findings indicate that geometric deviation can serve as a lightweight pre-generation signal that is reliable in structured domains with formal answerability constraints, with clear boundaries on where it generalizes. 

---
# Effective Performance Measurement: Challenges and Opportunities in KPI Extraction from Earnings Calls 

**Authors**: Rasmus T. Aavang, Rasmus Tjalk-Bøggild, Alexandre Iolov, Giovanni Rizzi, Mike Zhang, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2605.03147)  

**Abstract**: Earnings calls are a key source of financial information about public companies. However, extracting information from these calls is difficult. Unlike the templatic filings required by the U.S. Securities and Exchange Commission (SEC) to report a company's financial situation, earnings conference calls have no built-in labels, are unstructured, and feature conversational language. We explore this challenging domain by assessing the information captured by models trained on SEC filings and in-context learning methods. To establish a baseline, we first evaluate the generalization capabilities of SEC-trained models across established SEC datasets. To support our investigation, we introduce three novel benchmarks: (1) SEC Filings Benchmark (SECB), (2) Earnings Calls Benchmark (ECB), and ECB-A, a subset with 2,460 expert annotation groups to support our qualitative analysis. We find that encoder-based models struggle with the domain shift. Finally, we propose a system utilizing LLMs to perform open-ended extraction from unstructured call transcripts, verified by human evaluation (79.7% precision), providing a baseline for this valuable domain through the consistent tracking of emergent KPIs. 

---
# MedStruct-S: A Benchmark for Key Discovery, Key-Conditioned QA and Semi-Structured Extraction from OCR Clinical Reports 

**Authors**: Yingyun Li, Yu Wang, Haiyang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2605.03103)  

**Abstract**: Semi-structured information extraction (IE) from OCR-derived clinical reports is crucial for efficiently reconstructing patients' longitudinal medical histories. In practice, this scenario commonly involves three tasks: (i) field-header (key) discovery, (ii) key-conditioned question answering (QA), and (iii) end-to-end key-value pair extraction. However, existing evaluations often under-model two factors: heterogeneous and incompletely known key representations, and OCR-induced noise. This makes it difficult to assess model robustness in real-world settings.
We present MedStruct-S, a benchmark specifically designed to evaluate these tasks under unknown keys and OCR noise. MedStruct-S contains 3,582 annotated real-world clinical report pages. Using MedStruct-S, we benchmark two representative paradigms: encoder-only sequence labeling with post-processing and decoder-only structured generation, covering four encoder-only and five decoder-only models spanning 0.11B to 103B parameters. Our results show that encoder-only models achieve the best performance for non-null-value key-conditioned QA despite being substantially smaller than decoder-only models. When comparing models of similar order of magnitude, encoder-only models still perform better overall. Without controlling for model scale, fine-tuned decoder-only models deliver the strongest overall results. These findings show that the benchmark provides a reliable and practical basis for selecting and comparing models across different semi-structured IE settings. 

---
# Semantically Enriching Investor Micro-blogs for Opinion-Aware Emotion Analysis: A Practical Approach 

**Authors**: Gaurav Negi, Paul Buitelaar  

**Link**: [PDF](https://arxiv.org/pdf/2605.03092)  

**Abstract**: While sentiment analysis is the staple of financial NLP, capturing the nuances of 'why' behind that sentiment remains a challenge. There have been attempts to address this by analysing investor emotions alongside sentiment; however, this does not provide the additional granularity required to understand the target of the emotion/sentiment. We address this by augmenting the StockEmotions dataset with semantically structured opinion graphs, which provide granular semantic depth to the existing sentiment and emotion labels. Using a declarative LLM pipeline, we augment the StockEmotions dataset with opinion graphs for each sentence, derived from 10,000 comments collected from StockTwits. In addition, we study the effect of introducing opinion semantics on baseline classifiers using Graph Neural Networks (GNNs). Our analysis demonstrates that incorporating opinion semantics improves classification performance across different emotional spectrums 

---
# The TTS-STT Flywheel: Synthetic Entity-Dense Audio Closes the Indic ASR Gap Where Commercial and Open-Source Systems Fail 

**Authors**: Venkata Pushpak Teja Menta  

**Link**: [PDF](https://arxiv.org/pdf/2605.03073)  

**Abstract**: Niche-domain Indic ASR -- digit strings, currency amounts, addresses, brand names, English/Indic codemix -- is under-served by both open-source SOTA and commercial systems. On a synthesised entity-dense Telugu test set (held-out by synthesis system), vasista22/whisper-telugu-large-v2 (open SOTA) achieves Entity-Hit-Rate (EHR) 0.027 and Deepgram Nova-3 (commercial) 0.16. We close this gap with a self-contained TTS<->STT flywheel: an open-source Indic TTS pipeline synthesises ~22,000 entity-dense Indic-English code-mix utterances at <$50 marginal cost, and a LoRA fine-tune on top of vasista22 achieves EHR 0.473 on the held-out test (17x over open SOTA, 3x over commercial), with read-prose regression bounded to +6.6 pp WER on FLEURS-Te. Cross-language: beta-Hi 0.337 (7x vs vasista22) and beta-Ta 0.543 (22x vs vasista22, 22x vs Deepgram); on Hindi where Deepgram has substantial entity coverage, the flywheel underperforms commercial. All three beta models fall below pre-registered EHR targets (0.75 for Te, 0.65 for Hi/Ta); we report honestly. A native-human-recorded sanity check (n=20 Telugu) confirms transfer to real speech (beta-Te EHR 0.516 on native vs 0.473 on synth). An EDSA-isolation ablation (LoRA on FLEURS-Te alone) yields EHR 0.020 on the same held-out, attributing ~100% of the gain to the EDSA corpus. We additionally report a language-conditional finding: vanilla Whisper-large-v3 has Telugu-specific Script Collapse (SFR 0.46-0.71) that a per-language LoRA corrects (SFR 0.81-0.97), but the recipe is contraindicated on Hindi and Tamil where vanilla SFR >= 0.98. Code, holdouts, predictions, EDSA corpus, and entity dictionaries are released open-source. 

---
# How Language Models Process Negation 

**Authors**: Zhejian Zhou, Tianyi Zhou, Robin Jia, Jonathan May  

**Link**: [PDF](https://arxiv.org/pdf/2605.03052)  

**Abstract**: We study how Large Language Models (LLMs) process negation mechanistically. First, we establish that even though open-weight models often provide wrong answers to questions involving negation, they do possess internal components that process negation correctly. Their poor accuracy is due to late-layer attention behavior that promotes simple shortcuts; ablating those attention modules greatly improves accuracy on negation-related questions. Second, we uncover how models process negation. We consider two hypotheses: models could use attention heads that attend to the phrase being negated and suppress related concepts, or they could directly construct a representation of the entire negative phrase (e.g., representing "not gas" as a vector that promotes liquids and solids). We apply a range of observational and causal interpretability techniques on Mistral-7B and Llama-3.1-8B to show that models implement both mechanisms, with the "constructive" mechanism being more prominent. Combined, our work deepens the understanding of LLMs' internals, highlighting construction-dominant computations and the coexistence of competing mechanisms within LLMs. 

---
# Evaluating Reasoning Models for Queries with Presuppositions 

**Authors**: Rose Sathyanathan, Kinshuk Vasisht, Danish Pruthi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03050)  

**Abstract**: Millions of users turn to AI models for their information needs. It is conceivable that a large number of user queries contain assumptions that may be factually inaccurate. Prior work notes that large language models (LLMs) often fail to challenge such erroneous assumptions, and can reinforce users' misinformed opinions. However, given the recent advances, especially in model's reasoning capabilities, we revisit whether large reasoning models (LRMs) can reason about the underlying assumptions and respond to user queries appropriately. We construct queries with varying degrees of presuppositions spanning health, science, and general knowledge, and use it to evaluate several widely-deployed models When compared to non-reasoning models, we find that reasoning models achieve a slightly higher accuracy (2-11%), but they still fail to challenge a large fraction (26-42%) of false presuppositions. Further, reasoning models remain susceptible to how strongly the presupposition is expressed. 

---
# When Should a Language Model Trust Itself? Same-Model Self-Verification as a Conditional Confidence Signal 

**Authors**: Aditya Ajay Phalod  

**Link**: [PDF](https://arxiv.org/pdf/2605.02915)  

**Abstract**: Same-model self-verification, prompting a model to audit its own predicted answer, is a plausible confidence signal for selective prediction, but its practical value remains unclear once strong likelihood-based baselines are taken seriously. We evaluate self-verification against two such baselines, LL-AVG and LL-SUM, on ARC-Challenge and TruthfulQA-MC across multiple model families, scales, and prompt variants. We measure not only correctness ranking, but also abstention quality through AURC and operating-point analyses. The results are sharply task- and model-dependent. On ARC-Challenge, self-verification substantially improves over LL-AVG for Phi-2 and the Qwen models, with the largest gains appearing in Qwen-7B. On TruthfulQA-MC, however, the signal is less reliable: smaller models can become prompt-sensitive, DeepSeek-R1-Distill-8B degrades relative to LL-AVG, and LL-SUM often remains the stronger practical baseline. We therefore do not treat self-verification as a general-purpose uncertainty estimator. In this setting, it is better understood as a conditional confidence signal whose value depends on task type, model family, prompt formulation, and, crucially, the baseline it must beat. 

---
# OpenSeeker-v2: Pushing the Limits of Search Agents with Informative and High-Difficulty Trajectories 

**Authors**: Yuwen Du, Rui Ye, Shuo Tang, Keduan Huang, Xinyu Zhu, Yuzhu Cai, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.04036)  

**Abstract**: Deep search capabilities have become an indispensable competency for frontier Large Language Model (LLM) agents, yet their development remains dominated by industrial giants. The typical industry recipe involves a highly resource-intensive pipeline spanning pre-training, continual pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL). In this report, we show that when fueled with informative and high-difficulty trajectories, a simple SFT approach could be surprisingly powerful for training frontier search agents. By introducing three simple data synthesis modifications: scaling knowledge graph size for richer exploration, expanding the tool set size for broader functionality, and strict low-step filtering, we establish a stronger baseline. Trained on merely 10.6k data points, our OpenSeeker-v2 achieves state-of-the-art performance across 4 benchmarks (30B-sized agents with ReAct paradigm): 46.0% on BrowseComp, 58.1% on BrowseComp-ZH, 34.6% on Humanity's Last Exam, and 78.0% on xbench, surpassing even Tongyi DeepResearch trained with heavy CPT+SFT+RL pipeline, which achieves 43.4%, 46.7%, 32.9%, and 75.0%, respectively. Notably, OpenSeeker-v2 represents the first state-of-the-art search agent within its model scale and paradigm to be developed by a purely academic team using only SFT. We are excited to open-source the OpenSeeker-v2 model weights and share our simple yet effective findings to make frontier search agent research more accessible to the community. 

---
# Transformers with Selective Access to Early Representations 

**Authors**: Skye Gunasekaran, Téa Wright, Rui-Jie Zhu, Jason Eshraghian  

**Link**: [PDF](https://arxiv.org/pdf/2605.03953)  

**Abstract**: Several recent Transformer architectures expose later layers to representations computed in the earliest layers, motivated by the observation that low-level features can become harder to recover as the residual stream is repeatedly transformed through depth. The cheapest among these methods add static value residuals: learned mixing coefficients that expose the first-layer value projection V_1 uniformly across tokens and heads. More expressive dense or dynamic alternatives recover finer-grained access, but at higher memory cost and lower throughput. The usefulness of V_1 is unlikely to be constant across tokens, heads, and contexts; different positions plausibly require different amounts of access to early lexical or semantic information. We therefore treat early-representation reuse as a retrieval problem rather than a connectivity problem, and introduce Selective Access Transformer (SATFormer), which preserves the first-layer value pathway while controlling access with a context-dependent gate. Across models from 130M to 1.3B parameters, SATFormer consistently improves validation loss and zero-shot accuracy over the static value-residual and Transformer baselines. Its strongest gains appear on retrieval-intensive benchmarks, where it improves over static value residuals by approximately 1.5 average points, while maintaining throughput and memory usage close to the baseline Transformer. Gate analyses suggest sparse, depth-dependent, head-specific, and category-sensitive access patterns, supporting the interpretation that SATFormer learns selective reuse of early representations rather than uniform residual copying. Our code is available at this https URL. 

---
# Correct Is Not Enough: Training Reasoning Planners with Executor-Grounded Rewards 

**Authors**: Tianyang Han, Hengyu Shi, Junjie Hu, Xu Yang, Zhiling Wang, Junhao Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.03862)  

**Abstract**: Reinforcement learning with verifiable rewards has become a common way to improve explicit reasoning in large language models, but final-answer correctness alone does not reveal whether the reasoning trace is faithful, reliable, or useful to the model that consumes it. This outcome-only signal can reinforce traces that are right for the wrong reasons, overstate reasoning gains by rewarding shortcuts, and propagate flawed intermediate states in multi-step systems. To this end, we propose TraceLift, a planner-executor training framework that treats reasoning as a consumable intermediate artifact. During planner training, the planner emits tagged reasoning. A frozen executor turns this reasoning into the final artifact for verifier feedback, while an executor-grounded reward shapes the intermediate trace. This reward multiplies a rubric-based Reasoning Reward Model (RM) score by measured uplift on the same frozen executor, crediting traces that are both high-quality and useful. To make reasoning quality directly learnable, we introduce TRACELIFT-GROUPS, a rubric-annotated reason-only dataset built from math and code seed problems. Each example is a same-problem group containing a high-quality reference trace and multiple plausible flawed traces with localized perturbations that reduce reasoning quality or solution support while preserving task relevance. Extensive experiments on code and math benchmarks show that this executor-grounded reasoning reward improves the two-stage planner-executor system over execution-only training, suggesting that reasoning supervision should evaluate not only whether a trace looks good, but also whether it helps the model that consumes it. 

---
# Agentic-imodels: Evolving agentic interpretability tools via autoresearch 

**Authors**: Chandan Singh, Yan Shuo Tan, Weijia Xu, Zelalem Gero, Weiwei Yang, Michel Galley, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.03808)  

**Abstract**: Agentic data science (ADS) systems are rapidly improving their capability to autonomously analyze, fit, and interpret data, potentially moving towards a future where agents conduct the vast majority of data-science work. However, current ADS systems use statistical tools designed to be interpretable by humans, rather than interpretable by agents. To address this, we introduce Agentic-imodels, an agentic autoresearch loop that evolves data-science tools designed to be interpretable by agents. Specifically, it develops a library of scikit-learn-compatible regressors for tabular data that are optimized for both predictive performance and a novel LLM-based interpretability metric. The metric measures a suite of LLM-graded tests that probe whether a fitted model's string representation is "simulatable" by an LLM, i.e. whether the LLM can answer questions about the model's behavior by reading its string output alone. We find that the evolved models jointly improve predictive performance and agent-facing interpretability, generalizing to new datasets and new interpretability tests. Furthermore, these evolved models improve downstream end-to-end ADS, increasing performance for Copilot CLI, Claude Code, and Codex on the BLADE benchmark by up to 73% 

---
# Task Vector Geometry Underlies Dual Modes of Task Inference in Transformers 

**Authors**: Hao Yan, Haolin Yang, Yiqiao Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2605.03780)  

**Abstract**: Transformers are effective at inferring the latent task from context via two inference modes: recognizing a task seen during training, and adapting to a novel one. Recent interpretability studies have identified from middle-layer representations task-specific directions, or task vectors, that steer model behavior. However, a lack of rigorous foundations hinders connecting internal representations to external model behavior: existing work fails to explain how task-vector geometry is shaped by the training distribution, and what geometry enables out-of-distribution (OOD) generalization. In this paper, we study these questions in a controlled synthetic setting by training small transformers from scratch on latent-task sequence distributions, which allows a principled mathematical characterization. We show that two inference modes can coexist within a single model. In-distribution behavior is governed by Bayesian task retrieval, implemented internally through convex combinations of learned task vectors. OOD behavior, by contrast, arises through extrapolative task learning, whose representations occupy a subspace nearly orthogonal to the task-vector subspace. Taken together, our results suggest that task-vector geometry, training distributions, and generalization behaviors are closely related. 

---
# Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies 

**Authors**: Zirui Tang, Xuanhe Zhou, Yumou Liu, Linchun Li, Weizheng Wang, Hongzhang Huang, Jun Zhou, Jiachen Song, Shaoli Yu, Jinqi Wang, Zihang Zhou, Hongyi Zhou, Yuting Lv, Jinyang Li, Jiashuo Liu, Ruoyu Chen, Chunwei Liu, GuoLiang Li, Jihua Kang, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03596)  

**Abstract**: Workspace learning requires AI agents to identify, reason over, exploit, and update explicit and implicit dependencies among heterogeneous files in a worker's workspace, enabling them to complete both routine and advanced tasks effectively. Despite its importance, existing relevant benchmarks largely evaluate agents on pre-specified or synthesized files with limited real-world dependencies, leaving workspace-level evaluation underexplored. To this end, we introduce Workspace-Bench, a benchmark for evaluating AI agents on Workspace Learning invOlving Large-Scale File Dependencies. We construct realistic workspaces with 5 worker profiles, 74 file types, 20,476 files (up to 20GB) and curate 388 tasks, each with its own file dependency graph, evaluated across 7,399 total rubrics that require cross-file retrieval, contextual reasoning, and adaptive decision-making. We further provide Workspace-Bench-Lite, a 100-task subset that preserves the benchmark distribution while reducing evaluation costs by about 70%. We evaluate 4 popular agent harnesses and 7 foundation models. Experimental results show that current agents remain far from reliable workspace learning, where the best reaches only 68.7%, substantially below the human result of 80.7%, and the average performance across agents is only 47.4%. 

---
# FINER-SQL: Boosting Small Language Models for Text-to-SQL 

**Authors**: Thanh Dat Hoang, Thanh Trung Huynh, Matthias Weidlich, Thanh Tam Nguyen, Tong Chen, Hongzhi Yin, Quoc Viet Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.03465)  

**Abstract**: Large language models have driven major advances in Text-to-SQL generation. However, they suffer from high computational cost, long latency, and data privacy concerns, which make them impractical for many real-world applications. A natural alternative is to use small language models (SLMs), which enable efficient and private on-premise deployment. Yet, SLMs often struggle with weak reasoning and poor instruction following. Conventional reinforcement learning methods based on sparse binary rewards (0/1) provide little learning signal when the generated SQLs are incorrect, leading to unstable or collapsed training. To overcome these issues, we propose FINER-SQL, a scalable and reusable reinforcement learning framework that enhances SLMs through fine-grained execution feedback. Built on group relative policy optimization, FINER-SQL replaces sparse supervision with dense and interpretable rewards that offer continuous feedback even for incorrect SQLs. It introduces two key reward functions: a memory reward, which aligns reasoning with verified traces for semantic stability, and an atomic reward, which measures operation-level overlap to grant partial credit for structurally correct but incomplete SQLs. This approach transforms discrete correctness into continuous learning, enabling stable, critic-free optimization. Experiments on the BIRD and Spider benchmarks show that FINER-SQL achieves up to 67.73\% and 85\% execution accuracy with a 3B model -- matching much larger LLMs while reducing inference latency to 5.57~s/sample. These results highlight a cost-efficient and privacy-preserving path toward high-performance Text-to-SQL generation. Our code is available at this https URL. 

---
# Exposing LLM Safety Gaps Through Mathematical Encoding:New Attacks and Systematic Analysis 

**Authors**: Haoyu Zhang, Mohammad Zandsalimy, Shanu Sushmita  

**Link**: [PDF](https://arxiv.org/pdf/2605.03441)  

**Abstract**: Large language models (LLMs) employ safety mechanisms to prevent harmful outputs, yet these defenses primarily rely on semantic pattern matching. We show that encoding harmful prompts as coherent mathematical problems -- using formalisms such as set theory, formal logic, and quantum mechanics -- bypasses these filters at high rates, achieving 46%--56% average attack success across eight target models and two established benchmarks. Crucially, the effectiveness depends not on mathematical notation itself, but on whether a helper LLM deeply reformulates the harmful content into a genuine mathematical problem: rule-based encodings that apply mathematical formatting without such reformulation perform no better than unencoded baselines. We introduce a novel Formal Logic encoding that achieves attack success comparable to Set Theory, demonstrating that this vulnerability generalizes across mathematical formalisms. Additional experiments with repeat post-processing confirm that these attacks are robust to simple prompt augmentation. Notably, newer models (GPT-5, GPT-5-Mini) show substantially greater robustness than older models, though they remain vulnerable. Our findings highlight fundamental gaps in current safety frameworks and motivate defenses that reason about mathematical structure rather than surface-level semantics. 

---
# Two Calls, Two Moments, and the Vote-Accuracy Curve of Repeated LLM Inference 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.03379)  

**Abstract**: Repeated sampling is a standard way to spend test-time compute, but its benefit is controlled by the latent distribution of correctness across examples, not by one-call accuracy alone. We study the binary correctness layer of repeated LLM inference under conditional-i.i.d. calls. One labeled call identifies the mean latent success probability; two labeled calls identify its second moment and hence the same-example correctness correlation that separates stable errors from recoverable call-level randomness. From these two moments, every fixed majority-vote budget has a sharp distribution-free two-call interval. The key technical reduction is that the infinite-dimensional moment problem has three-atom extremizers and quadratic dual certificates for every finite budget, so the bounds are exact rather than discretized or parametric. The first useful budget, three votes, has a closed form, width at most $1/8$, and a certified-improvement criterion. The infinite-vote endpoint is the limit of majority voting as the number of calls tends to infinity; it is also sharply bounded, but remains threshold-sensitive because it depends on latent mass around $q=1/2$. We add maximum-entropy and Latent-difficulty Gaussian-probit (LDGP) point completions, and experiments on LLM calls over QNLI and QQP show that empirical three- and five-vote accuracies are contained in the projected two-call regions while temperature changes and randomized model mixtures can create voting gains not ordered by one-call accuracy. 

---
# RAG over Thinking Traces Can Improve Reasoning Tasks 

**Authors**: Negar Arabzadeh, Wenjie Ma, Sewon Min, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2605.03344)  

**Abstract**: Retrieval-augmented generation (RAG) has proven effective for knowledge-intensive tasks, but is widely believed to offer limited benefit for reasoning-intensive problems such as math and code generation. We challenge this assumption by showing that the limitation lies not in RAG itself, but in the choice of corpus. Instead of retrieving documents, we propose retrieving thinking traces, i.e., intermediate thinking trajectories generated during problem solving attempts. We show that thinking traces are already a strong retrieval source, and further introduce T3, an offline method that transforms them into structured, retrieval-friendly representations, to improve usability. Using these traces as a corpus, a simple retrieve-then-generate pipeline consistently improves reasoning performance across strong models and benchmarks such as AIME 2025--2026, LiveCodeBench, and GPQA-Diamond, outperforming both non-RAG baselines and retrieval over standard web corpora. For instance, on AIME, RAG with traces generated by Gemini-2-thinking achieves relative gains of +56.3%, +8.6%, and +7.6% for Gemini-2.5-Flash, GPT-OSS-120B, and GPT-5, respectively, even though these are more recent models. Interestingly, RAG on T3 also incurs little or no extra inference cost, and can even reduce inference cost by up to $15%$. Overall, our results suggest that thinking traces are an effective retrieval corpus for reasoning tasks, and transforming them into structured, compact, or diagnostic representations unlocks even stronger gains. Code available at this https URL. 

---
# The Right Answer, the Wrong Direction: Why Transformers Fail at Counting and How to Fix It 

**Authors**: Gabriel Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2605.03258)  

**Abstract**: Large language models often fail at simple counting tasks, even when the items to count are explicitly present in the prompt. We investigate whether this failure occurs because transformers do not represent counts internally, or because they cannot convert those representations into the correct output tokens.
Across three model families, Pythia, Qwen3, and Mistral, ranging from 0.4B to 14B parameters, we find strong evidence for the second explanation. Linear probes recover the correct count from intermediate layers with near-perfect accuracy ($R^2>0.99$), showing that the information is present. However, the internal directions that encode counts are nearly orthogonal to the output-head rows for digit tokens ($|\cos|\leq0.032$). In other words, the model stores the count in a form that the digit logits do not naturally read out.
We localize this failure with two interventions. Updating only the digit rows of the output head (36,864 parameters) substantially improves constrained next-token digit prediction (60.7 to 100.0% across four tasks), but it does not fix autoregressive generation. By contrast, a small LoRA intervention on attention Q/V weights (7.67M parameters) improves upstream routing and achieves 83.1% +/- 7.2% in true greedy autoregressive generation. Logit-lens measurements confirm the mechanism: the correct digit's vocabulary rank drops from 55,980 to 1, a 50,000x improvement. Additional norm, logit-lens, and cross-task analyses show that the bottleneck generalizes across character counting, addition, and list length, while remaining absent from broader multi-step reasoning benchmarks, including MMLU, GSM8K, and DROP.
These results identify counting failure as a geometric readout bottleneck rather than a failure of internal representation: the model knows the count but the output pathway is geometrically misaligned with the tokens needed to express it. 

---
# TeamUp: Semantic Project Matching and Team Formation for Learning at Scale 

**Authors**: Dhruv Gulwani, Basem Suleiman, Aditya Joshi, Sonit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2605.03237)  

**Abstract**: Project-based learning improves student engagement and learning outcomes, yet allocating students to appropriately challenging projects while forming cognitively diverse teams remains difficult at scale. Traditional allocation methods (manual spreadsheets, preference surveys) can't construct the cognitively diverse teams that that collaborate cognitively. This mismatch perpetuates equity issues: high-performing students self-select visible projects while under-represented students face reduced access to opportunity. We propose TeamUp, a lightweight, embedding-based team-forming system designed to improve learning outcomes and equity in large-scale project-based courses. TeamUp uses semantic embeddings from pretrained language models to match students to projects aligned with their skill level. The system employs a hybrid ranking algorithm combining cosine similarity with pedagogical constraints (difficulty alignment, domain preferences, and demand balancing) to generate personalised and transparent recommendations. Beyond individual matching, TeamUp constructs cognitively diverse teams by modelling skill complementarity through embedding variance, ensuring teams possess well-distributed capabilities rather than homogeneous strengths. We evaluated TeamUp through a virtual experiment using 250 student profiles and 60 project descriptions. Results show: (1) substantially higher match quality (mean cosine similarity of 0.74 vs. 0.43); (2) better difficulty alignment (83% placed within one level vs. 34%); (3) more diverse teams (82% covering three or more technical areas vs. 41%); and (4) sub-second recommendation latency at operational costs under $0.10 per student. 

---
# MAGE: Safeguarding LLM Agents against Long-Horizon Threats via Shadow Memory 

**Authors**: Yuhui Wang, Tanqiu Jiang, Jiacheng Liang, Charles Fleming, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.03228)  

**Abstract**: As large language model (LLM)-powered agents are increasingly deployed to perform complex, real-world tasks, they face a growing class of attacks that exploit extended user-agent-environment interactions to pursue malicious objectives improbable in single-turn settings. Such long-horizon threats pose significant risks to the safe deployment of LLM agents in critical domains. In this paper, we present MAGE (Memory As Guardrail Enforcement), a novel defensive framework designed to counter a wide range of long-horizon threats. Inspired by the "shadow stack" abstraction in systems security, MAGE maintains a dedicated, safety-focused agentic memory that distills and retains safety-critical context across the agent's full execution trajectory, leveraging this shadow memory to proactively assess the risk of pending actions prior to their execution. Extensive evaluation demonstrates that MAGE substantially outperforms existing defenses across diverse long-horizon threats in detection accuracy, achieves early-stage detection for the majority of attacks, and introduces only negligible overhead to agent utility. To our best knowledge, MAGE represents the first framework to detect and mitigate long-horizon threats using an agentic memory approach, establishing a new paradigm for this critical challenge and opening promising directions for future research. 

---
# ADAPTS: Agentic Decomposition for Automated Protocol-agnostic Tracking of Symptoms 

**Authors**: Alexandria K. Vail, Marcelo Cicconet, Katie Aafjes-van Doorn, Ryan Maroney, Marc Aafjes  

**Link**: [PDF](https://arxiv.org/pdf/2605.03212)  

**Abstract**: Modeling latent clinical constructs from unconstrained clinical interactions is a unique challenge in affective computing. We present ADAPTS (Agentic Decomposition for Automated Protocol-agnostic Tracking of Symptoms), a framework for automated rating of depression and anxiety severity using a mixture-of-agents LLM architecture. This approach decomposes long-form clinical interviews into symptom-specific reasoning tasks, producing auditable justifications while preserving temporal and speaker alignment. Generalization was evaluated across two independent datasets ($N=204$) with distinct interview structures. On high-discrepancy interviews, automated ratings approximated expert benchmarks ($\text{absolute error}=22$) more closely than original human ratings ($\text{absolute error}=26$). Implementing an ``extended'' protocol that incorporates qualitative clinical conventions significantly stabilized ratings, with absolute agreement reaching $\text{ICC(2,1)} = 0.877$. These findings suggest that the ADAPTS framework enables promising evaluations of psychiatric severity. While the current implementation is purely text-based, the underlying architecture is readily extensible to multimodal inputs, including acoustic and visual features. By approximating expert-level precision in a protocol-agnostic manner, this framework provides a foundation for objective and scalable psychiatric assessment, especially in resource-limited settings. 

---
# OCRR: A Benchmark for Online Correction Recovery under Distribution Shift 

**Authors**: Adrian Grassi  

**Link**: [PDF](https://arxiv.org/pdf/2605.03153)  

**Abstract**: Static benchmarks measure a model frozen at training time. Real systems face distribution shift: new categories, paraphrased queries, drift: and must recover online via user corrections. No existing benchmark measures recovery speed under correction streams. We introduce OCRR (Online Correction Recovery Rate): a benchmark that streams a corpus through a classification system, applies oracle or stochastic corrections to wrong predictions, and reports two curves: novel-class accuracy and original-distribution accuracy versus correction count. We evaluate the substrate alongside nine baseline algorithms from five families plus seven bounded-storage variants of the substrate for the Pareto sweep, including standard online-learning baselines (river), continual-learning methods (EWC, A-GEM, LwF), retrieval/parametric hybrids (kNN-LM), parameter-efficient fine-tuning of a 1.5 B-parameter encoder (LoRA on DeBERTa-v3-large), and a hash-chained append-only substrate (Substrate). On Banking77 and CLINC150, under oracle and sparse correction policies, the substrate is the only system that simultaneously recovers novel-class accuracy (88.7 +/- 2.9 %) and retains original-distribution accuracy (95.4 +/- 0.8 %) beating the next-best published continual-learning baseline by 32.6 percentage points at equal memory budget, and beating LoRA-on-DeBERTa-v3-large by 84.6 percentage points on retention. We further find that classification accuracy remains stable at 99 % even as approximate-nearest-neighbour recall@5 degrades from 0.69 to 0.23 across 10 k to 10 M corpus scales, suggesting the substrate's margin-band majority vote is robust to retrieval imperfection in a way that pure top-k recall metrics do not predict. Code and data are available at this https URL. 

---
# PIIGuard: Mitigating PII Harvesting under Adversarial Sanitization 

**Authors**: Mingshuo Liu, Yiwei Zha, Min Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.03129)  

**Abstract**: Browsing-enabled LLM assistants can fetch webpages and answer contact-seeking queries, creating a practical channel for scraping contact-style personally identifiable information (PII) from public pages. Many prior defenses are deployed at the model, service, or agent layer rather than at the webpage itself, leaving ordinary page owners with limited deployable options. We present PIIGuard, a webpage-level defense that repurposes indirect prompt injection as a protective mechanism: the page owner embeds optimized hidden HTML fragments that steer the model away from verbatim or reconstructible disclosure of contact PII. PIIGuard searches over fragment text and insertion position using rule-based leakage scoring, evolutionary mutation, and final judge-based recoverability assessment. In direct-HTML evaluation on three target models (GPT-5.4-nano, Claude-haiku-4.5, and DeepSeek-chat(latest v3.2)), PIIGuard achieves at least 97.0% defense success rate under both rule-based and judge-based leakage evaluation, often reaching 100.0%, while preserving benign same-page QA utility. We further evaluate two harder settings: public-URL browsing and attacker-side LLM sanitization of fetched webpage. These results show that page-side defensive fragments can remain effective in deployment for some model-position pairs, but robustness varies substantially across browsing interfaces and sanitizer prompts. Overall, PIIGuard demonstrates that page owners can use page-side fragments as a practical mitigation for web-grounded PII leakage. 

---
# Benchmarking Local Language Models for Social Robots using Edge Devices 

**Authors**: Dorian Lamouille, Matevž B. Zorec, Farnaz Baksh, Karl Kruusamäe  

**Link**: [PDF](https://arxiv.org/pdf/2605.03111)  

**Abstract**: Social-educational robots designed for socially interactive pedagogical support, such as the Robot Study Companion (RSC), rely on responsive, privacy-preserving interaction despite severely limited compute. However, there is a gap in systematic benchmarking of language models for edge computing in pedagogical applications. This paper benchmarks 25 open-source language models for local deployment on edge hardware. We evaluate each model across three dimensions: inference efficiency (tokens per second, energy consumption), general knowledge (a six-category MMLU subset), and teaching effectiveness (LLM-rated pedagogical quality), validated against five independent human raters using the Raspberry Pi(RPi)4 as the primary platform, with additional comparisons on the RPi5 and a laptop GPU.
Results reveal pronounced trade-offs: throughput and energy efficiency vary by over an order of magnitude across models, MMLU accuracy ranges from near-random to 57.2%, and teaching effectiveness does not correlate monotonically with either metric. Among the evaluated models, Granite4 Tiny Hybrid (7B) achieves a strong overall balance, reaching 2.5 tokens per second, 0.90 tokens per joule, and 54.6% MMLU accuracy; high MMLU accuracy does not appear necessary for strong teaching scores. Human validation on four representative models preserved the automated rank ordering (Pearson r = 0.967, n = 4). Based on these findings, we propose a three-tier local inference architecture for the RSC that balances responsiveness and accuracy on resource-constrained hardware. 

---
# When Prompts Interact: Assessing Prompt Arithmetic for Deconfounding under Distribution Shift 

**Authors**: Zhecheng Sheng, Yongsen Tan, Xiruo Ding, Trevor Cohen, Serguei Pakhomov  

**Link**: [PDF](https://arxiv.org/pdf/2605.03096)  

**Abstract**: In classification tasks, models may rely on confounding variables to achieve strong in-distribution performance, capturing spurious features that fail under distribution shift. This shortcut behavior leads to substantial degradation in out-of-distribution settings. Task arithmetic offers a potential solution by removing unwanted signals via subtraction of secondary model updates, but it typically requires full fine-tuning, which is computationally expensive. Prompt tuning provides a parameter-efficient alternative by adapting models through a small set of trainable virtual tokens. Task arithmetic on the resulting prompts presents an appealing alternative to operations on entire models, but the extent to which this approach can limit reliance on spurious features remains to be established. In this work, we study whether composing soft prompts through task arithmetic improves robustness to confounding shifts. We propose Hybrid Prompt Arithmetic (HyPA), which combines task prompts with linearized confounder prompts to counteract spurious correlations. Across multiple benchmarks, HyPA consistently improves the robustness-performance trade-off relative to prompt-arithmetic baselines under distribution shift. We further analyze how HyPA affects hidden representations and find evidence consistent with it mitigating confounding either by reducing the influence of confounder signals on predictions or by suppressing them in the representation. These results establish HyPA as a parameter-efficient and promising approach for improving robustness under confounding shifts in the evaluated setting. 

---
# Multilingual Safety Alignment via Self-Distillation 

**Authors**: Ruiyang Qin, Qingzhuo Wang, Dongrui Liu, Qiang Li, Zhihua Wei, Wen Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02971)  

**Abstract**: Large language models (LLMs) exhibit severe multilingual safety misalignment: they possess strong safeguards in high-resource languages but remain highly vulnerable to jailbreak attacks in low-resource languages. Current safety alignment methods generally rely on high-quality response data for each target language, which is expensive and difficult to generate. In this paper, we propose a cross-lingual safeguard transfer framework named Multilingual Self-Distillation (MSD). This framework transfers an LLM's inherent safety capabilities from high-resource (e.g., English) to low-resource (e.g., Javanese) languages, overcoming the need for response data in any language. Our framework is flexible and can be integrated with different self-distillation strategies. Specifically, we implement two concrete methods -- on-policy MSD and off-policy MSD -- both of which enable effective cross-lingual safety transfer using only multilingual queries. Furthermore, we propose Dual-Perspective Safety Weighting (DPSW), a divergence measure to optimize the distillation objective. By jointly considering the perspectives of both the teacher and the student, DPSW adaptively increases the penalty weights on safety-critical tokens while reducing the weights on non-critical tokens. Extensive experiments on representative LLMs across diverse multilingual jailbreak and utility benchmarks demonstrate that our method consistently achieves superior multilingual safety performance. Notably, it generalizes effectively to more challenging datasets and unseen languages while preserving the model's general capabilities. 

---
# AutoRAGTuner: A Declarative Framework for Automatic Optimization of RAG Pipelines 

**Authors**: Xintan Zeng, Yongchao Liu, Yice Luo, Jiajun Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02967)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances LLMs, but performance is highly sensitive to complex architecture designs and hyper-parameter configurations, which currently rely on inefficient manual tuning. We present AutoRAGTuner, a declarative, configuration-driven framework that automates the RAG life cycle: construction, execution,evaluation, and optimization. AutoRAGTuner employs a modular architecture to decouple pipeline stages through a component registration mechanism. To unify heterogeneous data, we introduce the Domain-Element Model (DEM), representing objects as atomic elements with bidirectional pointers to support nodes, edges, and hyperedges. Furthermore, AutoRAGTuner integrates an adaptive Bayesian optimization engine for end-to-end hyper-parameter tuning. Experimental results demonstrate AutoRAGTuner's architectural generality: across diverse RAG pipelines, ranging from vanilla to graph-based, the framework consistently outperforms default baselines. Notably, AutoRAGTuner significantly mitigates engineering overhead, where its declarative configuration language enables a up to 95\% reduction in code churn for architectural adjustments. Overall, AutoRAGTuner provides a systematically optimizable foundation for building evolvable and reusable RAG systems. 

---
# Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection 

**Authors**: Xulin Hu, Che Wang, Wei Yang Bryan Lim, Jianbo Gao, Zhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.02958)  

**Abstract**: Representation Engineering typically relies on static refusal vectors derived from terminal representations. We move beyond this paradigm, demonstrating that refusal is a dynamic and sparse process rather than a localized outcome. Using Causal Tracing, we uncover the Refusal Trajectory-a persistent upstream signature that remains intact even when adversarial attacks (e.g., GCG) suppress terminal signals. Leveraging this, we propose SALO (Sparse Activation Localization Operator), an inference-time detector designed to capture these latent patterns. SALO effectively recovers defense capabilities against forced-decoding attacks, improving detection rates from ~0% to >90% where methods relying on terminal states perform poorly. 

---
# CreativityBench: Evaluating Agent Creative Reasoning via Affordance-Based Tool Repurposing 

**Authors**: Cheng Qian, Hyeonjeong Ha, Jiayu Liu, Bingxiang He, Jeonghwan Kim, Jiateng Liu, Bingxuan Li, Aditi Tiwari, Dwip Dalal, Zhenhailong Wang, Xiusi Chen, Mahdi Namazifar, Yunzhu Li, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.02910)  

**Abstract**: Recent advances in large language models have led to strong performance on reasoning and environment-interaction tasks, yet their ability for creative problem-solving remains underexplored. We study this capability through the lens of creative tool use, where a model repurposes available objects by reasoning about their affordances and attributes rather than relying on canonical usage. As a first step, we introduce CreativityBench, a benchmark for evaluating affordance-based creativity in LLMs. To this end, we build a large-scale affordance knowledge base (KB) with 4K entities and 150K+ affordance annotations, explicitly linking objects, parts, attributes, and actionable uses. Building on this KB, we generate 14K grounded tasks that require identifying non-obvious yet physically plausible solutions under constraints. Evaluations across 10 state-of-the-art LLMs, including closed and open-source models, show that models can often select a plausible object, but fail to identify the correct parts, their affordances, and the underlying physical mechanism needed to solve the task, leading to a significant drop in performance. Furthermore, improvements from model scaling quickly saturate, strong general reasoning does not reliably translate to creative affordance discovery, and common inference-time strategies such as Chain-of-Thought yield limited gains. These results suggest that creative tool use remains a major challenge for current models, and that CreativityBench provides a useful testbed for studying this missing dimension of intelligence, with potential implications for planning and reasoning modules in future agents. 

---
