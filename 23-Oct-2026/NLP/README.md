# Low-Bit N-Gram Embeddings for Language Models 

**Authors**: Stephen Chung, Wenyu Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.31646)  

**Abstract**: Many recent open-weight language models use n-gram embedding tables to improve performance. These tables store learned representations of short token sequences, such as pairs or triples of consecutive tokens, which the model retrieves during processing. However, they can require substantial memory: DeepSeek-V4.1-Flash, for example, allocates 196 billion parameters to its Engram n-gram memory. We therefore investigate whether this memory cost can be reduced while preserving the performance improvement that the tables provide. We propose storing n-gram embeddings at low precision and systematically investigate different quantization methods and bit allocations across tables. Across nanochat-, Llama-, and GPT-2-family models trained from scratch, four-bit tables save approximately 75% of table memory while preserving more than 99% of the validation-loss improvement from adding the tables. Using fewer bits leads to a larger drop in performance, although we find that a short fine-tuning phase recovers a significant part of this loss. In addition, reinvesting the saved memory in larger tables can improve performance beyond that of the original full-precision tables while still using less memory. These table memory savings also extend to large existing models: on DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next, low-bit tables save 47-55% of table memory, with perplexity increases of less than 1.2% and largely preserved downstream task performance. These findings establish low-bit precision as a practical way to reduce the memory requirements of n-gram embeddings and use the saved capacity to improve model performance. 

---
# Can an Agent Debug a Web Corpus? Agentic Auditing of Slovak FineWeb2 

**Authors**: Marek Šuppa, Ivan Vykopal, Jaroslav Kopčan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31636)  

**Abstract**: Multilingual pipelines such as FineWeb2 apply one recipe across a thousand languages, leaving the failure modes of individual languages unanticipated. We use an LLM once, as a corpus debugger rather than a classifier: a tool-using agent explores the corpus through a read-only, budgeted API and compiles each finding into an executable test, which a native speaker then validates. On the 26.5M-document Slovak portion of FineWeb2, three runs costing \$2.21-\$2.84 each found seven findings over four phenomena. The largest is a machine-generated content farm of 1,408,437 documents (5.32%) that passed the pipeline's filters, 86.6% of it passing even stricter thresholds. Blinded labels support 6 of the 7 findings (defect rates .53-1.00 against .08 in a random control) and do not support one. The compiled suite scans the full corpus in 0.7 s of wall-clock time without a GPU, against \$12k-\$90k per LLM classification pass. The best static sampling baseline validates more selectors than any agent run, at under half the cost. The agent contributes exact prevalence, a precision estimate checked against sampled documents, an account of why the pipeline missed each slice, and a reusable test. We release the framework, Slovak test suite, validation data and agent trajectories. 

---
# Bilingual Language Models Capture Cross-Linguistic Influence in Second-Language Reading 

**Authors**: Abderrahmane Issam, Irene E. Winther, Stefan L. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2610.31620)  

**Abstract**: While monolingual language model (LM) surprisal is widely established as a predictor of reading times, it cannot capture the non-native (L2) reading dynamics that are shaped by the reader's first language (L1). In this work, we investigate whether the cross-lingual transfer within bilingual models aligns with human L1-driven biases during L2 reading. We evaluate bilingual models on eye-tracking data of readers from 12 diverse L1 backgrounds reading English texts. To isolate L1-specific transfer, we compare bilingual models trained on an L1 that matches the readers' L1 background to both a monolingual English baseline and bilingual models trained on other L1s. We demonstrate that matching the readers' L1 improves the prediction of L2 reading times. This predictive advantage is concentrated in late-stage eye-tracking measures, modulated by L1-L2 script overlap, and peaks among intermediate-proficiency readers. These findings provide evidence that bilingual LMs capture aspects of cross-linguistic influence, and that such influence is present in L2 reading behavior. 

---
# How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions 

**Authors**: Miguel Mochizuki Silva, Samuel Ismael, Mateus C. Dias, Thaís Gaudêncio, Yuri Malheiros, Barufaldi  

**Link**: [PDF](https://arxiv.org/pdf/2610.31606)  

**Abstract**: Purpose: To describe an evidence-aware retrieval evaluation protocol for a clinical retrieval-augmented generation (RAG) system over breast lesion management literature, addressing the dependence of retrieval scores on annotation coverage. Methods: An 18-document corpus and an 18-query development benchmark, with one Yes/No or WH-question per document, were used. Retrieval combined dense embeddings, BM25, and cross-encoder reranking. Chunk sizes of 128 to 1024 tokens, overlaps of 0 to 20%, and four retriever configurations were evaluated. Relevance was defined by evidence spans mapped to all containing chunks, treating overlapping chunks with equivalent evidence as relevant. Retrieval was evaluated using R-Precision, Recall@K, MRR, and nDCG@K. Separately, generation was assessed on a 512-token index at K=1 to 5 using DeepEval claim-level metrics, with deepseek-v4-pro as the LLM judge, and BLEU/ROUGE metrics. Results: The selected hybrid retriever, using 384-token chunks and 20% overlap, retrieved accepted evidence in the top three results for 17 of 18 queries. Recall@3, R-Precision, MRR@3, and nDCG@3 each reached 0.94. Ground-truth curation increased evidence spans from 25 to 37 and accepted chunk targets from 36 to 51 without changing retrieval rankings. Conclusion: Cross-encoder reranking achieved the highest retrieval scores. Expanded relevance annotations increased recognized evidence without changing rankings, highlighting annotation sensitivity. The small development set and retrieval-informed curation require independent validation, including radiologist assessment of evidence sufficiency. 

---
# Would an LLM Match into a Residency? Uncertainty, Reasoning and Image Use of Open Language Models on the Spanish MIR Examination 

**Authors**: Miguel Fernandez-de-Retana, Unai Zulaika, Ruben Sanchez-Corcuera, Aritz Bilbao-Jayo, Aitor Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2610.31586)  

**Abstract**: Accuracy on medical examinations is common evidence offered for the clinical use of language models, yet a health system that intends to run an open model on its own hardware needs to know considerably more: whether the model recognizes when it is likely to be wrong, whether medical fine-tuning improves this ability, whether its correct answers rest on sound clinical reasoning, or whether it actually uses the images it is given. We examine these questions for four open models of 8 to 27 billion parameters on the Spanish MIR residency examination, scoring 1,552 questions from eight editions under the exam's own rules. Answering every question, the three larger models would rank among roughly the first 600 to 3,000 candidates, with 41 to 43 specialties still open to them. Abstention, however, never pays, because even their least confident answers are right more often than the 25% break-even. Medical fine-tuning reduced overconfidence at the cost of accuracy, without improving how well confidence separates right from wrong answers. Moreover, 13-38% of correct answers rejected a distractor for a clinically unsound reason that confidence did not reflect, and removing or swapping the provided contextual images left accuracy unchanged. 

---
# An Empirical Study of Recent Large Language Models for Knowledge Graph to Text Generation on Standard Benchmarks 

**Authors**: Priyanka, Renaud Delbru, Souhail Bakkali, Mickaël Coustaty, Antoine Doucet  

**Link**: [PDF](https://arxiv.org/pdf/2610.31529)  

**Abstract**: Knowledge graph-to-text (KG-to-text) generation verbalizes a set of knowledge graph triples into fluent natural language. While fine-tuned encoder-decoder models have long been the standard approach, recent LLMs are becoming increasingly competitive. We benchmark recent LLMs against fine-tuned encoder-decoder models, using reference-based metrics and graph-grounded reference-free LLM-based evaluation validated against human annotations. We observed that recent LLMs substantially narrow, and in several settings close, the gap with fine-tuned models on reference-based metrics, while performing better on graph-grounded metrics. LLM-based evaluation scores show limited correlation with reference-based metrics, suggesting that the two approaches capture different aspects of generation quality. Graph-grounded evaluation reveals that prompted models produce fewer contradictions and triple omissions than fine-tuned models, and these errors increase more rapidly with graph size for fine-tuned models, highlighting differences between the two approaches that reference-based metrics may not fully capture. 

---
# The Geometry of Metaphor: Probing and Steering Metaphor Representations in Large Language Models 

**Authors**: Shihui Li, Mette Andersen, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2610.31520)  

**Abstract**: Metaphor is a crucial aspect of human cognition. As such, we would like to understand how it is represented in large language models (LLMs) and how it might be controlled. We use mechanistic interpretability techniques to identify causal subspaces in a range of LLMs covering different architectures and sizes. We show that we are able to steer output on classification tasks across different parts of speech. We are also able to steer output on generation tasks, indicating that these techniques can be used to control the metaphoricity of LLM output. 

---
# Towards User Simulators That Stay On Task And Still Sound Natural 

**Authors**: Mustafa Omer Gul, Philippe Laban, Tanya Goyal, Jennifer Neville  

**Link**: [PDF](https://arxiv.org/pdf/2610.31512)  

**Abstract**: Users increasingly rely on AI systems for extended, multi-turn tasks, yet evidence indicates that LLM-based system performance degrades over prolonged interaction. To evaluate and improve AI assistants in such settings, we need user simulators that faithfully follow complex user intents while exhibiting realistic conversational behavior. Existing approaches manifest a trade-off: prompted assistants can follow complex user intents but lack behavioral fidelity, while specialized user simulators better approximate conversational behavior but struggle to remain faithful to complex intents. In this paper, we propose a reinforcement learning (RL) recipe that jointly rewards task intent adherence and behavioral naturalness. A crucial contribution of the method is operationalizing naturalness with a lightweight discriminator trained in an alternating, adversarial loop with the user simulator. We release the resulting model, UserLM-2.0, which produces valid simulated conversations (i.e., meeting both task-faithfulness and naturalness criteria) 46.2% of the time, over three times the rate of the strongest prior user simulator (14.1%). 

---
# From Structured Records to Patient Narratives: Preserving Clinical Alignment in LLM-Generated Health Summaries 

**Authors**: Bayram Ayadi, Florian Geissler, Johanna Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2610.31467)  

**Abstract**: As wearable and digital health systems increasingly generate longitudinal physiological records, communicating these observations in patient-accessible language remains a significant challenge. Large Language Models (LLMs) offer a promising mechanism for translating structured clinical information into readable narratives, but this process may introduce semantic drift, where clinically precise meaning is distorted during simplification. We investigate evidence-grounded health summarization for menstrual health monitoring using a constrained generation framework that combines structured evidence conditioning with Retrieval-Augmented Generation (RAG) using a prompt-based constrained generation framework. We evaluate semantic alignment using Natural Language Inference (NLI), atomic factuality metrics, and readability measures. Across three LLMs and 4,500 generated summaries, explicit grounding constraints consistently improve clinical alignment, particularly by improving uncertainty preservation and, once faithfulness is evaluated against retrieved evidence, generally preserving or improving faithfulness, while effects on omission remain model-dependent. We also find that improved clinical alignment comes at the cost of increased reading difficulty and medical terminology. These findings highlight the need to jointly evaluate faithfulness, uncertainty preservation, and readability when developing patient-facing medical LLM systems. 

---
# How Well Do Decision Models Transfer to Vietnamese? An Empirical Study of Text Classification 

**Authors**: Phat Huu Ha, Nguyen Xuan Phuc, Nong Nhut Quy, Trinh Tran Tran, Nguyen Minh Triet, Dang Van Thin  

**Link**: [PDF](https://arxiv.org/pdf/2610.31422)  

**Abstract**: Recent advances in specialized decision models have expanded the possibilities for zero-shot classification, yet their effectiveness on Vietnamese remains underexplored. This study evaluates four specialized decision models across 13 classification settings spanning six task groups, examining zero-shot performance, sensitivity to label-description language, and robustness to Vietnamese linguistic variations. The evaluation considers English, Vietnamese, and bilingual label descriptions, alongside common linguistic perturbations involving teencode, typographical errors, and diacritic removal. Under English label descriptions, JEV achieves the strongest overall performance, ranking first in 11 classification settings. The effects of label-description language vary across models and datasets, with bilingual descriptions providing no consistent improvement. Robustness experiments reveal that diacritic removal causes the most substantial performance degradation, reducing mean macro-$F_1$ by 9.51--15.47 points across model configurations, considerably more than teencode or typographical errors. These findings highlight the importance of label-description choices and Vietnamese orthographic variation when evaluating specialized decision models for zero-shot text classification. 

---
# How Depth Growth and Looping Shape In-Context Learning in Language Models 

**Authors**: Ferdinand Kapl, Kaitlin Maile, Stefan Bauer, Johannes von Oswald  

**Link**: [PDF](https://arxiv.org/pdf/2610.31421)  

**Abstract**: Looped and depth-grown language models reuse computations across depth and have attracted interest for their reasoning capabilities. Building on prior suggestive evidence of improved in-context-learning (ICL), we investigate how such reuse favors ICL, including how it shapes the acquisition, retention, and mechanisms of this ability. Our analysis spans pretrained language models up to 3B parameters and controlled tasks. The experiments provide evidence for a shared inductive bias toward learning from context: grown and looped models rely more on information in context and, compared with a full-depth dense reference, acquire ICL earlier in controlled tasks and retain it more strongly when learning fixed associations in the weights is also possible. Tracing attention heads through training reveals that depth growth propagates early induction heads across layers. Ablations show that grown and looped models rely more strongly on induction-ranked heads for both few-shot performance and later-token prediction. Together, these findings provide evidence for a shared inductive bias toward learning from context and suggest that (soft) parameter reuse shapes how this ability develops and which attention mechanisms support it. 

---
# SyRA-CS: Synthetic-to-Real Adaptation for Code-Switched ASR with Meta-Learning 

**Authors**: Fadel Thior, Ahmer Tabassum, Haania Siddiqui, Hanan Aldarmaki, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2610.31415)  

**Abstract**: Code-switched (CS) automatic speech recognition (ASR) remains difficult because real CS speech is scarce and synthetic speech does not fully match it. We investigate whether synthetic CS speech is more useful for learning an adaptation-friendly initialization than for ordinary continual fine-tuning. We propose SyRA-CS, a pair-specific synthetic-to-real adaptation framework based on Reptile meta-learning. For each language pair and pretrained model, we meta-train on the corresponding synthetic CS-FLEURS subset and then adapt using one hour of labeled real target speech. We evaluate four architectural settings: Encoder-CTC, FastConformer-RNN-T, Attention Encoder-Decoder (AED) and Speech-LLM. Experiments cover on Arabic-, French-, German-, and Hindi-English, against synthetic-only transfer, real-only fine-tuning, and continual fine-tuning. SyRA-CS improves over continual fine-tuning on WER in all 16 model-pair comparisons and on CER in 15 of 16, with relative macro WER reductions of 17.7% for XLS-R and 5.3-5.8% for the other three models. The gains hold across the tested models but vary in size across architectures, language pairs and metrics. 

---
# Same Entity, Different Script: Diagnosing Cross-Script Entity Matching Failures in Language Models 

**Authors**: Athira Gopal, Patil Prathamesh Pravin, Ankush Kadu, Arpit Jain, Ashwanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31407)  

**Abstract**: Multilingual language models can give different identity judgments when the same people's names are written in different scripts. We study this effect using 601 audited identity pairs with attested Devanagari and Latin forms, holding the question and relation fixed. Qwen2.5-7B's same-person accuracy falls from 35.1% in Latin to 8.5% in Devanagari under Marathi prompts, while different-person accuracy remains above 98%. Evaluation across five models reveals distinct response biases, and reduced tokenization fragmentation does not reliably close the gap. Replicating the design on 374 audited Malayalam pairs built from the same entities reproduces the effect in a second script and language family: same-person accuracy falls to 5.3%, about half the matched Devanagari rate, while the margin still ranks matching above non-matching names at 0.925 AUROC. Latin-trained probes recover label information from Devanagari states, exceeding a string-feature baseline under similarity-matched negatives. Scalar calibration recovers 87% of the raw model-probe balanced-accuracy gap, identifying decision criterion placement as a major contributor. Activation patching localises causal sensitivity to the second entity in Qwen's middle layers; one shared sensitivity direction transfers about half the full-state margin correction. A fact-independent intervention at this site raises held-out Marathi same-person accuracy from 6.5% to 33.5%, with 98.5% different-person accuracy. Together, these results distinguish script-dependent discrimination from criterion placement and identify a causal site for improving identity judgments. 

---
# HERD: Harness-Evolving Recursive Distillation for Continual Learning 

**Authors**: Xun Sun, Thanh Thi Nguyen, Thuy-Trang Vu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31401)  

**Abstract**: Continual learning requires a model to keep acquiring new skills and knowledge without compromising its existing capabilities. Parametric methods suffer from catastrophic forgetting and loss of plasticity. Non-parametric methods, such as prompts, skills, and harnesses, leverage in-context learning but are limited by context length and lossy plain-text compression. To effectively combine their complementary strengths, we propose HERD (Harness-Evolving Recursive Distillation), a hybrid continual learning framework that learns efficiently in the harness and internalizes experience into cumulative capabilities via distillation. On the harness side, HERD continually refines its skills to capture reusable patterns, while retaining high-residual examples that the skills fail to capture in a bounded memory. On the parameter side, HERD performs drift-constrained distillation with dual anchors, substantially reducing forgetting over long recursive runs. On the 25-stage task stream of TRACE+17, under fixed memory and compute budgets, HERD matches the strongest parametric baseline in new-task acquisition while exhibiting the least forgetting and the strongest retention of general capabilities. 

---
# Parameter-Efficient Adaptation of Large Language Models to a Low-Resource Agglutinative Language: A Corpus, Datasets, and a Reproducible Benchmark for the Bashkir Language 

**Authors**: Mullosharaf K. Arabov, Svetlana S. Khaybullina  

**Link**: [PDF](https://arxiv.org/pdf/2610.31397)  

**Abstract**: Bashkir is a low-resource agglutinative language written in Cyrillic and remains poorly represented in modern LLMs. We present an open resource suite and a systematic benchmark of adaptation methods: the Bashkir Web Corpus (71,567 documents, 46.9M tokens, 16 sources) and three labelled datasets for binary, multiclass and multilabel text classification; the benchmark covers the first two. Experiments use eight architectures (82M-8B: DistilGPT2, GPT-2 base/medium, Phi-3-mini, Mistral-7B, YandexGPT-5-Lite-8B, mBERT, XLM-R) and three adaptation groups (full fine-tuning, QLoRA, in-context learning), plus continual pretraining (CPT). (1) QLoRA on 3.8-8B models outperforms encoder-only fine-tuning on binary classification: YandexGPT-5-Lite-8B with CPT reaches macro-F1 0.748 vs 0.653 (XLM-R) and 0.650 (mBERT); pairwise differences are significant after Holm correction (p<0.001). (2) YandexGPT-5-Lite-8B significantly outperforms Mistral-7B and Phi-3-mini on multiclass classification, BPB and native-speaker ratings (mean 3.28 vs 2.52 and 2.22; n=100; alpha=0.69-0.77; p<0.05); on binary classification its QLoRA advantage is not significant (p>=0.084). (3) CPT effects depend on model and task: perplexity improves for Phi-3-mini (+0.58) and YandexGPT-5-Lite-8B (+0.33) but worsens for Mistral-7B (-1.15, p<0.001); on binary classification CPT helps Mistral-7B most (+4.0 macro-F1). (4) Tokeniser fertility on Bashkir varies 2.67x (2.92-7.80) and tracks adaptation quality, while perplexity across tokenisers is not comparable. Cyrillic tokenisers with Russian pretraining are preferable; LLM judges reproduce native ratings for grammar and coherence (rho=0.33-0.46) but not language correctness (rho=0.16, p=0.11); native evaluation remains necessary. All artefacts are open-licensed. 

---
# CrisiLangs: A Multilingual Resource for Crisis Warning Generation 

**Authors**: Giacomo Gonella, Stefano Menini, Marco Guerini  

**Link**: [PDF](https://arxiv.org/pdf/2610.31356)  

**Abstract**: Communicating with the public during an emergency entails issuing warning messages promptly. In Europe, it often requires producing them in more than one language. Recent work has shown that language models can be trained to generate warning messages as a crisis unfolds, but the available data covers only English. To fill this gap, we present CrisiLangs, a multilingual extension of CrisiText (an English dataset of crisis events paired with warning messages written according to expert communication guidelines) to Italian, Bulgarian, German, and Greek. Expert native speakers translated the warning messages of 1,000 events drawn from 411 scenarios, for a total of 5,000 event-message pairs across five languages. We test the resource to fine-tune a multilingual model on all five languages jointly, with and without the emergency guidelines in the prompt, and compare it with a zero-shot baseline. The fine-tuned models outperform the baseline on ROUGE-L, BLEU and BERTScore in every language and configuration. In a human preference study, low decoding temperatures receive the most preferences in all five languages and in both configurations, while the effect of the guidelines varies by language. 

---
# SurveyGraph: Generating Academic Surveys via Methods-Entity Knowledge Graphs 

**Authors**: Yifan Yuan, Yancheng Zhu, Zuoyu Zhang, Xiaogang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2610.31317)  

**Abstract**: Most automated survey generation systems retrieve papers as flat documents, limiting their ability to capture research structure. We present SurveyGraph, a framework built around a Methods-Entity Knowledge Graph (MEKG) encoding method-level follow-on, competition, and shared-resource relations. The graph supports core-paper discovery, retrieval, hybrid outlining, and evidence-gated narrative synthesis. We evaluate retrieval, semantic edges, citation grounding, collection coverage, and generation-stage components. In a controlled seven-method equal-budget benchmark, MEKG-anchored fusion balances recall and structural density, with an independent co-citation check. In a separate archived graph-native diagnostic, graph-aware retrieval nearly doubles recall over flat lexical baselines across eight runs (p = 0.004). A semantic evidence gate achieves AUC = 0.75. Audits of 160 generated sentences find 80.6% anchored-or-partial citation support. Human adjudication of 108 stratified semantic edges finds 68.5% supported-or-partial relations (binary-support Cohen's kappa = 0.64), with improves strongest (80.6%) and extends weakest (52.8%). A SurveyBench-facing collection probe reaches mean reference coverage of 0.521 across four topics with an eight-paper budget. Same-backend Qwen diagnostics on a 30-paper/74-relation graph show reduced organization signals when hybrid outlining is removed. Concurrent-system comparisons provide bounded context rather than matched-backend end-to-end evidence. The results position MEKG as an auditable structural layer for typed diagnostics, relation-specific filtering, and controlled survey generation. Code and artifacts: this https URL. 

---
# When LLMs Forget What They Believe: Characterizing State Integrity Collapse in Long-Horizon Dialogues 

**Authors**: Yancheng Zhu, Yifan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31314)  

**Abstract**: Large language models can repeat an instruction while failing to keep obeying it. We study this recall-compliance dissociation in long-horizon dialogue and define State Integrity Collapse: a model preserves a stated constraint in memory but stops applying it to later turns. We introduce StateTrack-Bench, a controlled suite of conversations covering numerical, format, role/persona, factual-memory, and logical-rule constraints. Across seven provider-pinned model snapshots and five dialogue-depth probes, strict joint retention ranges from 62.4% to 100.0%, with degradation concentrated in symbolic constraints and in one especially brittle model. A forked recall audit shows that 88.6% of observed failures are behavioral-drift failures rather than ordinary forgetting; excluding the brittle outlier preserves this diagnosis (84.4% of failures). Extended evaluations at depths up to 50 turns confirm that the dissociation persists beyond the core benchmark, while Structured State Anchoring recovers up to 24.4 points on immutable constraints. The results show that reliability cannot be inferred from single-turn instruction following alone; evaluation must track whether remembered constraints remain operational as dialogue state grows. Code and artifacts are available at: this https URL. 

---
# Cola2: Co-Adaptive Hierarchical Language Modeling with Continuous Semantic Compression and Balanced Discrete Generation 

**Authors**: Hongcan Guo, Yian Zhao, Feng Wang, Shen Nie, Guoqiang Wei, Yan Zeng, Hengshuang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2610.31305)  

**Abstract**: Hierarchical language generation separates continuous semantic modeling from discrete text realization, but its effectiveness depends on what the latent space compresses and how the discrete generator uses it. We propose Cola2, a framework that jointly studies continuous semantic compression, the balance between reconstruction and generation, and continuous-discrete co-adaptation. Cola2 dynamically segments text into semantic spans, controls the compression rate through auxiliary-loss-free negative feedback, and combines reconstruction and generation supervision to learn compact, predictive latent representations. A block-causal masked diffusion decoder is trained with clean, noisy, and dropped latent conditions, while classifier-free guidance adjusts the balance between semantic reconstruction and autonomous generation. Token-prediction gradients update the encoder through the latents, whereas Flow Matching prior learning is decoupled from representation learning by stop-gradient. During inference, generated text is re-encoded under a fixed span layout to keep the latent history consistent with the actual text context. Experiments on five downstream tasks with about 210 billion training tokens validate the core designs and show competitive, unsaturated scaling. With 346.03 billion source tokens, an unconverged checkpoint reaches a 60.10 average across eight tasks and surpasses Llama-3-8B on HellaSwag and StoryCloze despite its reported pretraining budget being over 43 times larger. These results demonstrate the potential of continuous-latent hierarchical language generation. 

---
# CoWriter: Guiding Methodological and Synergetic Human-LLM Co-Authoring 

**Authors**: Shmuel Amar, Benjamin Lieber, Ori Shapira, Debanjana Kar, Sebastian Maximilian Dennerlein, Yufang Hou, Ido Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31277)  

**Abstract**: While LLMs have transformed text generation, standard chat-and-canvas interfaces often reduce human authoring to quite passive supervision, causing review fatigue, over-reliance on AI, and the sidelining of the human role in long-form, source-grounded writing. Prior systems typically address isolated writing phases, leaving comprehensive end-to-end human-LLM collaboration largely unexplored. In this paper, we introduce CoWriter, a proactive agentic scheme that adapts expert human (co-)authoring methodologies from cognitive science to flexibly steer the entire writing workflow while actively encouraging the user to engage. In an exploratory user study, compared to a standard LLM baseline, authors using CoWriter better expressed their perspectives and maintained a higher sense of document ownership, supported quantitatively by $2.2\times$ more successful agent engagements. Furthermore, external reader judges rated the resulting texts higher in "human touch" qualities like storytelling and integrating unique personal experiences. Overall, our results demonstrate that proactively engaging users through established human practices is beneficial for effective LLM-supported co-authoring, providing useful methodologies, techniques and insights for future research on collaborative agents. 

---
# Memory-Aware Reward Modeling: Learning Human Preferences for LLM Memory Use 

**Authors**: Yijiang River Dong, Zexue He, Cheng Yu, Tiancheng Hu, Michael J Ryan, Jan Batzner, Zheng Hui, Shaolong Wu, Orestis Papakyriakopoulos, Alex Pentland, Nigel Collier, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2610.31272)  

**Abstract**: Memory enables language model assistants to maintain continuity across sessions and generate personalized responses, but knowing more about a user does not necessarily produce responses they prefer. Existing evaluations assess memory capabilities without directly establishing whether memory use benefits the user. To study this gap, we collect approximately 1,000 pairwise judgments from 87 participants using their own ChatGPT queries and personal memories. We find that memory-conditioned responses are not consistently preferred over responses generated without memory and in some cases, users strongly prefer the latter. Automatic evaluators also show limited agreement with users and favor memory-conditioned responses substantially more often than users do. Motivated by these gaps, we introduce a memory-aware reward model that learns from user judgments, achieving 72.8\% preference prediction accuracy and outperforming the existing baselines. Our findings motivate evaluation and optimization objectives that align memory use with the preferences of the people whose information is being used. 

---
# A Systematic Comparison of Search Space Complexity and Efficiency for Time-Synchronous End-to-End ASR Models 

**Authors**: Simon Berger, Larissa Kleppel, Ralf Schlüter  

**Link**: [PDF](https://arxiv.org/pdf/2610.31259)  

**Abstract**: Search is a central component of Automatic Speech Recognition (ASR) systems and has been deeply investigated for classical ASR modeling based on Hidden Markov Models (HMMs). However, in work on current sequence-to-sequence ASR architectures it is often simplified to a small fixed-size beam search without external Language Model (LM). In this work, we present a comparative analysis of search space complexity and efficiency for Connectionist Temporal Classification (CTC) and neural transducer models under different pruning strategies and LM configurations. We show that these end-to-end (E2E) models work with a small search space most of the time but exhibit large temporary peaks in hypothesis count during periods of uncertainty. Score-based pruning dynamically adapts the search to model uncertainty, leading to a significantly lower Real-Time Factor (RTF). Our results highlight the limitations of search with low fixed-size beam pruning and provide practical guidelines for efficient search in E2E ASR. 

---
# IndoNutriQA: An Indonesian Clinical Nutrition Question Answering Benchmark 

**Authors**: Tirana Noor Fatyanosa, Ummu Ditya Erliana, Mahendra Data, Muhammad Nurwegiono, Eleazar Tadeo Eman, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2610.31256)  

**Abstract**: Clinical Question Answering (QA) underpins Clinical Intelligent Decision Support Systems, but progress is severely limited for low-resource languages such as Indonesian by the absence of public domain-specific data. We introduce IndoNutriQA, an expert-curated Indonesian clinical QA benchmark of 911 question-answer pairs built from real clinical-nutrition reports. Rather than a large-scale training corpus, it is positioned as a high-quality evaluation resource: every pair is grounded in a clinically relevant context, enriched with structured metadata (anthropometric assessment, medical diagnosis, document provenance, chunk segmentation). Eight standardised question types following the Nutrition Care Process (NCP), including the Problem, Etiology, Sign and Symptom (PES) diagnosis statement, are answered for every report, which was written by a graduate-level dietetic intern, by one of three nutritionists. Answers are free-form clinical prose rather than extractive spans, and an expert validation study finds 6-10% of reference answers contestable. The dataset supports generative, extractive and retrieval-augmented QA as well as zero- and few-shot LLM evaluation. We report fine-tuned baselines (IndoBERT variants, mT5, RAG) with 5-fold cross-validation, learning curves and bootstrap 95% confidence intervals, and benchmark eight contemporary LLMs zero- and five-shot. We further calibrate our LLM-as-judge rubric against three expert raters (one RD, two nutritionists), finding it human-comparable on factuality and completeness but unreliable on clinical safety, and a human error analysis shows it misses unsupported content that the raters identify. IndoNutriQA is released to foster research in Indonesian clinical NLP and nutritional decision support. 

---
# Investigating Multi-Cultural Value Selection Biases in LLMs 

**Authors**: Siddhesh Pawar, Arnav Arora, Dustin Wright, Dirk Hovy, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.31199)  

**Abstract**: Large language models (LLMs) serve users whose cultural identities are multi-layered, yet cultural evaluations commonly reduce identity to one label. In this work, we study LLMs under simultaneous identity cues, with a country persona in the system prompt and a second identity signaled in the user turn, either by naming a country (explicit cue) or through the query language (implicit cue). We test World Values Survey (WVS) multiple-choice probing and open-ended sociopolitical generation. In multiple-choice probing, language cues shift answers about three times more than explicit cues but away from the associated country's distribution, and explicit cues do not consistently move answers toward the named country. Across 1.6 million open-ended responses, naming a country closes 30-44% of the gap to the model's own responses for that country, and explicit cues produce more differentiated value changes than language cues. Our results suggest that country and language cues should be evaluated as distinct signals rather than interchangeable proxies for culture. 

---
# When More Metadata Misleads: Probing Field-Level Grounding in LLM-Generated Museum Descriptions 

**Authors**: Clemencia Siro, Laura Hollink  

**Link**: [PDF](https://arxiv.org/pdf/2610.31193)  

**Abstract**: Metadata-conditioned generation is often framed as a data-to-text task in which additional input fields should improve grounding. Museum description generation challenges this assumption because catalogue records are heterogeneous, partially visual, and unevenly documented across geographic regions. We study how metadata composition shapes the generation of artefact descriptions through a two-phase framework. Phase 1 analyses natural metadata variation in a large museum collection, where field count, field type, image availability, and Place co-vary. Phase 2 introduces controlled field probes that hold object identity fixed while varying metadata quantity, field type, and image access. Our results show that metadata quantity alone does not explain generation behaviour. Field type determines whether models ground, omit, or infer information, while images can both support visual grounding and increase unsupported interpretation. More broadly, our findings show that metadata-conditioned data-to-text evaluation should assess field- and claim-level support, rather than relying only on aggregate input coverage or reference similarity. 

---
# Prompting Strategy Effects Depend on the Judge: A Multilingual Study of LLM-as-a-Judge Robustness 

**Authors**: Asal Mehrabi, Mohammad Aliannejadi, Clemencia Siro  

**Link**: [PDF](https://arxiv.org/pdf/2610.31172)  

**Abstract**: LLM-as-a-Judge evaluation is increasingly applied to multilingual tasks, yet prompting strategies such as Chain-of-Thought and Role Conditioning are recommended based solely on English benchmarks. Whether their effectiveness transfers across judge models and languages remains unknown: prompt-sensitivity studies are largely English-centric, while existing multilingual judge studies fix the prompt template. In this work, we address this gap by measuring this interaction through a benchmark of 9 prompting strategies, 4 open-weight judges, 27 languages, and 4 tasks across 13 datasets, yielding over 1M pointwise scores on gold-reference outputs, validated against human judgments. We find that prompting strategies are not transferable: the choice of the judge model dominates the choice of prompting strategy, strategy effects reverse direction across judges, and stability degrades in low-resource languages, where these effects are concealed by aggregate reporting. These findings show that prompting effects in multilingual evaluation are conditioned on the judge, task, and language. Consequently, prompting strategies should be validated for the intended evaluation setting, and cross-study comparisons should account for both judge and prompt configuration. 

---
# Prompt Sensitivity in Tool-Augmented Model Comparisons 

**Authors**: Mateusz Modrzejewski, Tomasz Radzikowski  

**Link**: [PDF](https://arxiv.org/pdf/2610.31148)  

**Abstract**: Language models are compared on question-answering benchmarks that pose each question in one wording, and a reworded question can change which of two models scores higher. We ask whether giving the models tools makes such comparisons more stable. An accuracy gain cannot answer this, because tools change several steps between the question and the scored answer at once. We propose a matched protocol that separates those steps: the answer readout, reasoning, the decision to call, the tool's answer, and the report supplied without a call. We apply it to six open models in five domains under five wordings. We find that tools raise accuracy where a program answers the question and widen the wording sensitivity of many pairwise comparisons. The widening follows how often a model calls, and call frequency follows the wording for some models and not others. It also depends on the interface and the instructions: a native tool-calling interface and a second tool description leave no detected widening in molecules, and sensitivity persists where tools return intermediate results. These findings show that deterministic reports do not guarantee stable comparisons. We recommend reporting call frequency and comparison stability under several wordings beside accuracy. 

---
# Mixing Matters: Evidence Position Bias Across Sequence Mixers in Long-Context Question Answering 

**Authors**: Aman Behera, Namit Solanki, Mehul Anand  

**Link**: [PDF](https://arxiv.org/pdf/2610.31135)  

**Abstract**: Long-context models often use answer-bearing evidence less effectively when it appears away from a prompt boundary. We compare this position bias across Transformer, state-space, and hybrid Mamba architectures by moving one gold document through a fixed set of nine distractors. The evaluated Transformers show a beginning advantage that the evaluated state-space models lack, while both families favor evidence near the end. A matched pure-versus-hybrid Mamba comparison points in the same direction but remains statistically uncertain. The family difference appears among models that can answer the task. Transformer primacy reappears on synthetic retrieval, where state-space ceiling prevents a family comparison. A corpus swap within one state-space architecture changes overall accuracy without a detectable change in curve shape. Prompt changes can create or nearly remove a beginning advantage. Position remains decodable across architectures even when response curves differ, and attention concentration near initial tokens covaries with beginning sensitivity in the tested Transformer family. These exploratory results show that effective context use depends on the evaluated architecture family, capability, task, and prompt, without isolating the sequence mixer as the sole cause. 

---
# Delta N-gram: Forward-Only Continual Writing into a Language Model's Native N-gram Memory 

**Authors**: Hao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2610.31125)  

**Abstract**: Our previous analysis found that a model's native hashed N-gram table is where continual updates disturb the base least, but its writes needed backpropagation and changed the pretrained table. We present Delta N-gram, a forward-only write path into that table. The pretrained table T^0 is never modified. Evidence is written into a sparse patch \Delta on the rows it addresses: a 4.1M-parameter writer emits a target and a gate per (position, hash head) event, and a block-averaged delta rule keeps each row a convex combination of its old value and the gated targets. The writer is trained offline on raw text with the pretraining objective; at use time writing takes forward passes only, and clearing \Delta restores the original model. A 0.6B prototype trained to answer questions keeps 16/16 facts after a second write; online gradient writes keep 2--4/16. We study two writers: a plain writer that encodes evidence with the T^0-only model, and a delta writer that reads its current memory while encoding. On Qwen3.8-Flash-Next, the plain writer's memory is document-specific: on 64 held-out news events it lowers continuation NLL by 0.068 nats/token against empty memory and by 0.024 against another document's memory, and 0.063 and 0.013 after 16 intervening articles. The delta writer's memory is more document-specific and keeps more of it under distraction (70% vs 54%). Over 35,000 writes on our previous seven-task chain both writers stay stable and the delta writer drifts less (layer-2 math drift 176% vs 188%); later accuracy changes by -0.26/-0.52pp (-7.29pp for gradient updates) and small-k_i drift is 17.9/17.5% (39.2%). Acquisition is negative, and general-input drift reaches 32.8% (20.75%): format rows read by every example grow to 4--12 times their pretrained norm. Zeroing eight such rows removes over 98% of layer-2 drift on math inputs, together with their NLL gain. 

---
# A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool 

**Authors**: Avrile Floro, Tamara Dhorasoo, Nils Holzenberger, Thomas Le Goff, Tiphaine Viard, Maria Boritchev  

**Link**: [PDF](https://arxiv.org/pdf/2610.31116)  

**Abstract**: This paper presents a dataset of 121,536 ground-reasons pairs across 86,464 rulings of the five civil chambers of the French Court of Cassation (2016-2025). An appeal on points of law cannot itself be appealed and often decides a case for good. Understanding how the Court rules is thus crucial and can help lawyers who draft those appeals. Inside a ruling, every ground of appeal (a distinct legal complaint against the appealed decision) has its own answer from the Court. This answer is called the reasons and it belongs to one of a few doctrinal families. Using a deterministic pipeline, we pair each ground with its reasons and attach the pair to its family, according to the established French taxonomy. We obtain a descriptive grammar of civil cassation that shows which grounds succeed, which legal texts they refer to, and how both evolve over the decade. We release the dataset, the code and an interactive consultation tool. 

---
# Learning What to Ask for Tomorrow: Active Preference Acquisition for Long-Term Personalized Web Agents 

**Authors**: Yiqi Wang, Zizhi Zeng, Jiaqi Zhang, Yiqun Duan, Mingkai Zheng, Taotao Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.31107)  

**Abstract**: Web agents increasingly serve the same user across repeated online tasks such as booking travel, shopping, and scheduling, so their long-term usefulness depends on how well they know the user's preferences. Passive memory and personalization methods cannot obtain preferences the user never reveals, and clarification methods that value a question by its benefit to the current task overlook the benefit of reusing the answer in later tasks. We study Active Preference Acquisition (APA): which missing preferences are worth the cost of asking? We define the Prospective Personalization Value (PPV) of a missing preference as its expected cumulative utility gain over future tasks minus the effort and disclosure-risk cost of asking, and select questions under a per-interaction budget. We characterize when cumulative reuse justifies acquisition despite insufficient single-task benefit, and evaluate this regime with an initial-profile diagnostic. APA computes the counterfactual gain of each candidate preference on public task prototypes with the agent's own task solver and aggregates it by the expected reuse of each task family. We evaluate it on ActiveWebPref, a controlled longitudinal benchmark with hidden but queryable user profiles and exact task utilities. At a horizon of 20 tasks, 34.4% of budget-feasible valuable missing preferences are amortization-dependent; APA acquires 80.3% of them, whereas current-task value of information acquires 5.1%. Under per-user asking-cost caps matched to APA's realized expenditure, APA achieves higher net utility than four heuristic selection baselines. In our cost setting, asking about every unknown preference yields lower net utility than not asking, while APA recovers 84.2% of its utility gain with 27.6% of its questions. The value of acquiring a user preference should account for its reuse across future tasks, not only its benefit to the task at hand. 

---
# RIFT: KV Cache Compression for Draft Models with Complementary Compressibility Training 

**Authors**: Dor Tsur, Maor Ashkenazi, Maximilian Golub, Peter Hicks, Yaniv Galron, Talor Abramovich, Yoav Miron, Tiyasa Mitra, Bita Darvish Rouhani, Ran Zilberstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.31098)  

**Abstract**: Speculative decoding is a popular decode acceleration method. As such, it is particularly relevant to long-output generation, which requires an increasingly large KV cache. KV cache compression offers a way to constrain costs in long-context drafting. However, compression methods that focus on prefill information can lose relevance as the output length grows. We term this phenomenon decode staleness and show that it degrades draft model accuracy under selection-based KV cache compression. We introduce RIFT, a family of selection-based KV cache compression strategies that balance prefill information and recent generated tokens. We show that refreshing the KV cache selection during token generation significantly improves the drafter's quality at the cost of additional decode latency. To alleviate the refresh costs, we propose an adaptive KV refresh strategy that reduces the number of score calculations while retaining the drafter's quality. We further demonstrate how a compressibility-driven drafter training phase, based on sparse attention masks, can improve the drafter's quality under a compressed KV cache. Combined with compressibility training, RIFT enables Qwen3.5-4B, drafting for Qwen3.5-122B, to retain over 98% of its full-cache acceptance length on 32K-token inputs, 4K-token outputs, and an 8K-token KV cache budget. 

---
# What Pairwise Evaluation Misses: Reward Scores across Prompts 

**Authors**: Mateusz Modrzejewski, Tomasz Radzikowski  

**Link**: [PDF](https://arxiv.org/pdf/2610.31090)  

**Abstract**: Reward models score responses for preference optimization, filtering and ranking, and pooling scores across prompts needs them to be comparable. Yet pairwise evaluation compares two responses to the same prompt, so any score component shared by a prompt's responses cancels and goes untested. We study this component by comparing between-prompt variation in model scores with the same variation in human ratings of the same responses. We propose a human-referenced audit, built from a between-prompt variance share, a sibling-rating regression and prompt substitutions, and apply it to eighteen released scorers and six matched training conditions. We find that scores cluster by prompt more than the ratings do: on HelpSteer2, six reward models have between-prompt shares of 0.34 to 0.64 against a human share near zero. Replacing the original prompt with a fixed instruction keeps substantial clustering but weakens agreement with prompt-level human ratings, so clustering alone does not show agreement with humans. Part of that variation is useful: subtracting prompt means lowers the helpfulness of globally selected responses, and scores stay sensitive to a wrong prompt even when training shuffles prompt means. We propose that reward evaluation check between-prompt variation, sensitivity to a wrong prompt and usefulness for selection separately. 

---
# Harm Is Behavioural, Moderation Is Lexical: How Language and Culture Decide Whom Automated Safety Systems Protect and Whom They Silence 

**Authors**: Gabriel Lars Sabadin, Dr Nicola Harding, Sarah Kingston  

**Link**: [PDF](https://arxiv.org/pdf/2610.31047)  

**Abstract**: Content warning: this paper quotes slurs and hateful statements from published test suites; the racial slur among them is masked.
Automated systems decide at scale which messages are harmful, and they decide unevenly. Classifiers label African American English as offensive up to twice as often as other English, the accounts of prominent drag queens can score as more toxic than those of white nationalists, and safety behaviour that holds in English collapses in low-resource languages. We argue that over-flagging and under-protection share one cause: deciding harm from vocabulary and surface form rather than from what a message does. We organise the evidence into seven failure modes and argue, as testable hypotheses, that harmful behaviours (a request for secrecy, a demand for payment, an attempt to isolate someone) travel across languages better than the words that express them. We propose an evaluation protocol that makes the gap between the best- and worst-served language communities a headline result, and show that the naive gap is inflated under equal service and that its usual bootstrap interval is unreliable. Applied to five open systems in eleven languages, from a keyword lexicon to two current multilingual guard models, the protocol finds that none reaches native-speaker accuracy at separating harmful function from harmless form (51 of 55 system-language pairs fall short); that greater capability raises discrimination in nine languages while leaving Arabic and romanised Hindi near chance, so the gap between the best- and worst-served languages grows with capability; that one system both over-flags and under-protects the same language most; and that a language a system reads worst can look the most fairly treated when only false positives are reported. 

---
# Chunk Recomputation under Real Document Revisions: Measuring Update Cost and Retrieval Quality of Fixed, Recursive, and Content-Defined Chunking 

**Authors**: Yuning Bian, Lele Xu, Yifan Yuan, Xinyu Yin, Chengyan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2610.31019)  

**Abstract**: Chunking strategies for retrieval-augmented generation are almost always evaluated on a static snapshot, yet the documents underlying a knowledge base continue to change. Because chunks are embedded independently, a boundary that moves after an edit forces unchanged text to be re-embedded. We measure this update cost on 120 pairs of adjacent Chinese and English Wikipedia revisions for three stateless chunkers (non-overlapping fixed-size, recursive paragraph-first, and content-defined chunking) under a strict byte-identical reuse rule and a frozen dense encoder. On the 116-article common-success pool, with frozen, length-calibrated parameters, fixed-size chunking recomputes 57-68% of its full new-version embedding-input workload, recursive chunking 38-56%, and content-defined chunking 43-54% (it failed on 4 of 62 Chinese articles), while at these parameters mean evidence recall under a fixed 2048-token retrieval budget differs by less than two percentage points within either language. A cost-only parameter grid shows that the ranking of the two content-aware chunkers changes with chunk-length settings, whereas non-overlapping fixed-size chunking has the highest recomputation cost in every length-matched condition. Controlled edits and a verbatim-presence diagnostic support boundary drift as one contributor to recomputation of text already present in the previous version. Update cost is therefore a substantial, parameter-sensitive dimension that static retrieval evaluations do not expose and that should be reported alongside retrieval quality. 

---
# Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters 

**Authors**: Martino M. L. Pulici, Cuong Xuan Chu, Evgeny Kharlamov, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2610.31017)  

**Abstract**: Single-document LoRA adapters have been proposed as units of knowledge to be merged and retrieved; yet open-book scoring, with the document in the prompt, can credit a document's own adapter (its specialist) for reading that document as readily as for holding it. Training one adapter for each of 150 Wikipedia articles on Llama-3.1-8B-Instruct, we find that open-book scores mostly measure a reading behaviour every adapter shares: a random adapter trained on another document recovers 82 % of the specialist's open-book advantage over the base model but only 2 % of its advantage on a cloze probe that asks for the same document's facts without providing the source text, a share we cannot distinguish from zero. Content does reach the weights, since the specialists beat the base model in cloze recall by 0.094, and the geometry of their updates follows the documents they read; yet, in a test of 30 queries, adapters chosen by weight-space proximity or text embedding are not distinguishable from random ones on either probe, although that test cannot exclude an advantage of about four fifths of what a query's own adapter gains over a random one. At 16 documents or more, correctly scaled merges recover 87 % to 89 % of the specialists' open-book advantage but only 12 % to 16 % of their cloze advantage, with intervals that run from below zero to at most 32 %, against 94 % for joint this http URL also report four silent errors, each of which produced a clean, wrong result. 

---
# Padding to k: A Minimum Count in the Schema Makes Open-Weight Models Pad Their Extraction Lists 

**Authors**: Abhishek Barali, Aayush Paudel  

**Link**: [PDF](https://arxiv.org/pdf/2610.30997)  

**Abstract**: Extraction pipelines ask language models for JSON that follows a schema, and the schema can set a minimum number of list entries. We ask what models do when the document holds fewer items than that minimum. Under a preregistered plan, we test eight open-weight models on 558 real documents from nine annotated datasets, with six more models on hosted services as a reference. Three findings stand out. First, models treat the count as a target: a minimum shown in the schema, but not enforced, raises the share of lists longer than the true count from 34.5% to 55.4%. Second, the padding is hard to catch, because most added entries are real text taken from the document; about half of them pass a check that each entry occurs in the source. Third, common fixes help only partly: a line saying fewer items are fine removes the effect on average but misses more true items, a count in prose ("up to k") also raises over-counting, an "N/A" option under an enforced minimum still leaves half the lists padded, and re-asking with the validation error makes padding far more common. We release the code, answer lists and run records at this https URL 

---
# Interpretive Variation and Discourse Structure in Detecting Enthymemes in Controversial Political Discourse 

**Authors**: Martial Pastor, Nelleke Oostdijk  

**Link**: [PDF](https://arxiv.org/pdf/2610.30987)  

**Abstract**: Enthymemes are arguments that leave a premise or conclusion unstated, and they are a common persuasion device on social media. Because readers must fill in the missing part themselves, they often disagree on whether a text contains one. This paper asks how to detect and evaluate enthymemes when such disagreement is part of the task. On tweets about vaccination and immigration, each labeled by five annotators, F1 stalls around 0.67 for every fine-tuned model we train, regardless of data size, added features or label type. We therefore ask how these models should be evaluated instead, and propose metrics that compare predictions with the full distribution of annotator votes, together with a model that predicts this distribution directly and captures disagreement better at no cost in F1. We then ask what role discourse structure plays. Models learn to rely on explicit argumentative structure: they reliably recognize tweets without an enthymeme but rarely commit on tweets that contain one. Both the models and a discourse parser miss arguments whose links are left implicit and must be inferred by the reader. Finally, we discuss what these blind spots mean for content moderation systems trained on majority labels. 

---
# A Reality Check of Inference-Time Context Augmentation Methods 

**Authors**: Haeun Yu, Geert Heyman, Pepa Atanasova, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2610.30983)  

**Abstract**: Large language models (LLMs) often fail to utilise the context given in the input, resulting in hallucinations or a fallback to parametric knowledge. Context augmentation methods aim to improve model performance by modifying the input, from inserting filler tokens that add computational time to adding relevant content such as elaboration of a context during inference. However, these methods have been evaluated in isolation and only by their performance gains, leaving it unclear how they compare and whether they reliably improve performance. In this paper, we present an evaluation framework spanning five methods that differ in how informative the inserted content is and apply them at the sentence level across seven models spanning multiple sizes on three datasets. We find that accuracy gains are not always genuine improvements but undirected perturbations: the methods correct some predictions while breaking others, particularly when the inserted text allows only for more computational time. Our analysis reveals that methods that bring no additional content primarily change model predictions near the decision boundary. Our findings highlight the need to evaluate the reliability of the accuracy gains before adopting them in practice. 

---
# Joint or Per-Option? Compact Encoders for Closed-Set Decisions from 17 Million to 1 Billion Parameters 

**Authors**: Tardelli Ronan Coelho Stekel  

**Link**: [PDF](https://arxiv.org/pdf/2610.30970)  

**Abstract**: Many language-model pipelines end in a closed-set decision, such as selecting a tool. Prompting a large language model for it is slow and order-sensitive. We ask how small an encoder for such decisions can be and whether it should read the options jointly or separately. In a three-seed study at 32 million parameters, per-option scoring beats joint scoring on a held-out sample of a 37-benchmark decision suite by 5.8 points of chance-corrected skill at matched optimizer steps and by 2.8 at approximately matched training compute; two external benchmarks agree, though not significantly on their unseen tasks. Crossing reader with initialisation shows that the advantage requires a reranker initialisation, whose query-candidate pair input matches per-option reading; from the raw encoder the two readers are indistinguishable. Per-option scoring needs 5 to 13 times more encoder passes: at most twice the median latency, but more than four times the mean. Of six released joint encoders (17 million to 1 billion parameters), the 68- and 150-million-parameter models exceed a 435-million-parameter DeBERTa-v3 zero-shot classifier by 4.9 and 5.9 accuracy points on 209 overlap-audited System1-Arena tasks and are at least on par with it on the 145 tasks unseen in training. The smallest answers a median request in 14 milliseconds on 16 processor threads, against 728 for the classifier. Reordering options changes 12 to 23% of the joint models' answers (per-option scoring: none). Up to 68 million parameters, they lead their size range on a public leaderboard's private, mainly in-distribution tests. 

---
# Once a Tree, Always a Tree? Cross-lingual Transfer of Hierarchical Generalization in Language Models 

**Authors**: Shin Yamaguchi, Masaya Taniguchi, Tatsuki Kuribayashi, Taro Watanabe, Yoshinobu Kano  

**Link**: [PDF](https://arxiv.org/pdf/2610.30960)  

**Abstract**: Language models (LMs) often exhibit successful cross-lingual transfer through continued (pre)training, still their underlying process remains unclear. One possibility is that LMs can leverage the evidence toward language-universal principles, namely, hierarchical structure, from a source language (L1) to a new language (L2), while many other non-linguistic explanations, e.g., improved initialization of position embeddings, can be potentially possible. In this study, to quantify the cross-linguistic transferability of linguistic evidence in cross-lingual LM training, we introduce a new experimental design, namely, a cross-lingual poverty-of-stimulus (CPOS) setting that ablates how LMs' hierarchical generalization in L2 can be led by structural evidence provided in L1. We contrast counterfactual L1 and L2 learning scenarios where the L2 corpus is ambiguous between linear and hierarchical grammar rules, while the amount of positive evidence for the hierarchical rule is controlled on the L1 side. Our experimental results show that LMs can partially, yet not perfectly, transfer L1 structural evidence to L2, yielding only a small but detectable hierarchical bias, while the strength of transfer depends on language pairs. We further ablate the conditions for successful transfer, motivated by recent reports on the substantial impact of vocabulary overlap. We indeed find that vocabulary overlap is crucial even for the cross-lingual transfer of hierarchical syntactic generalization. This suggests that this also explains the language-pair-dependent tendencies in the main experiments and suggests that the transfer of linguistic evidence in continued new-language learning is only conditionally achievable. 

---
# IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems 

**Authors**: A. Jian Yang, An Huang, Bo Dan, Chen Mu, Chengran Yang, Chuan Shi, Feng Chang, Hongxin Li, Hongyuan Tao, Jinxing Zhang, Jujie He, Kun Wang, Lin Jing, Ming Yang, Qicheng Hu, Renyuan Li, Shark Liu, Shawn Guo, Shi Qing, Shi Zhong, Tianyu Zheng, Wei Liu, Wei Zhang, Wenrui Cai, Wenxiao Chen, Xuxing Chen, Yan Xing, Yi Xing, Yifan Luo, Yiran Xing, Yizhi Li, Yongsheng Kang, Yucheng Ding, Yue Song, Yun Qing, Yunqi Liu, Zelong Huang, Zhengmao Ye, Zhentao Fan, Zhiteng Li, Ziling Huang, Bihong Tang, Bingli Wang, Ce Jun, Changhao Chen, Chen Hu, Chen Xiao, Chi Liu, Chunjiang Ge, Danyang Li, Deyang Kong, Dian Qi, Ding Tan, Evan Yang, Fanglin Xu, Fei Bai, Gang Liu, Guochang Li, Hao Tang, Haoran Feng, Haowen Wang, Huahao Hu, Huanjin Yao, Huatong Song, Jia Deng, Jiajun Wu, Jialiang Cheng, Jiayi Liang, Jiayi Wu, Jie Deng, Jiejun Tan, Jing Yang, Jingda Xu, Junhang Cheng, Junxiao Liu, Kai Lv, Kaixiang Pan, Kecheng Lu, Kun Zeng, Kun Zheng, Kunbo Ding, Liang Qiu, Lingyi Sun, Luan Zhang, Mengxiao Lv, Ning Dong, Pei Wang, Peng Zhang, Qingsong Cai, Qixiang Yin, Qiyao Wang, Renbiao Liu, Renfei Dang, Ruihang Li, Ruirui Huang, Shanhui Yin, Shen Jian, Shengjie Fang, Shikai Li, Shinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30949)  

**Abstract**: We present IQuest-Q1, a sparse Mixture-of-Experts (MoE) agentic foundation model with 320B total parameters (15B activated). IQuest-Q1 is developed through a three-stage process: (1) stable and efficient pre-training, combining Stable SplitNorm, Muon, and muP with a two-stage curriculum whose second stage increases the coding-data mixture, preserving a wide stability margin between safe and optimal learning rates; (2) fine-grained mid-training, expanding the data mixture across domains and formats and progressively extending the context; and (3) multi-scaffold with multi-version post-training, combining SFT, domain-specialized RL, and MOPD across various scaffolds to integrate complementary training signals and strengthen agentic coding capabilities. Furthermore, model merging combines different experts and checkpoints from different stages while mitigating conflicts among their parameter updates. The resulting IQuest-Q1 demonstrates capabilities in both coding and general tasks. We release IQuest-Q1 as an open-source model to support community research and development. 

---
# RetiJev: Fast and Source-Grounded Ophthalmic Decision Models 

**Authors**: Bo Liu, Lianyu Wang, Meng Wang, Ching-Yu Cheng, Huazhu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30947)  

**Abstract**: Ophthalmic clinical decision support requires fast, accurate, and privacy-preserving inference, yet autoregressive large language models (LLMs) incur substantial latency and computational overhead for decisions among predefined actions. We therefore investigate a decision-oriented alternative to generative reasoning. Specifically, we first introduce RetiJev-Data, a source-grounded ophthalmology corpus comprising 78,158 clinical-decision and diagnosis questions derived from articles in PubMed Central and a MeSH vocabulary covering 531 eye diseases, and 50,074 guideline-grading questions spanning 11 grading systems, along with a 24,662-question evaluation benchmark. Based on this, we develop RetiJev, a family of 0.8B-9B decision models that directly score candidate answers without autoregressive decoding. Across all model sizes, RetiJev outperforms both a 27B LLM and the commercial Jev 1.13 decision model on guideline grading by 10.4-23.0 points. RetiJev-9B further achieves the highest accuracy on clinical decisions, improving over both baselines by 5.4-6.5 points while matching them on diagnosis. It answers a case in 158 ms on one GPU, 3.5x faster than LLM generation and 470x faster than reasoning. Models are released at this https URL. 

---
# Verbalizing Activation Differences with Natural Language Autoencoders 

**Authors**: Andrea Blasi Núñez, Gianluca Barmina, Federico Torrielli, Thiusius Rajeeth Savarimuthu, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2610.30934)  

**Abstract**: Activation verbalizers translate the internal representations of large language models into human-readable text. The recently proposed natural language autoencoders (NLAs) learn this by reconstructing activations through a natural language bottleneck. However, it remains unclear how well these methods generalize beyond the setting they were trained for. Here, we study this question for NLAs along four closely related questions: (i) Do NLA verbalizations exhibit compositional structure with respect to their input? (ii) Do NLAs generalize to finetuned versions of the model they were trained for? (iii) Can NLAs recover finetuning objectives in model diffing? (iv) Can NLAs be specialized for model diffing? Across four models from three families, our experiments provide evidence supporting a largely positive answer to all four questions. Specifically, verbalizations of composed analogy activations mostly name the target while rarely mentioning the subtracted word, and an NLA can be reused on lightweight finetunes of its original model without retraining, with reconstruction quality degrading noticeably only for the largest model. Verbalizing activation differences between Taboo model organisms and their original models recovers the concealed secret word for most organisms, and reinforcement learning on activation differences further improves this recovery on held-out organisms. We also find that NLAs can recover the finetuning objective without the finetuned model generating any token, since the most informative token positions precede the user prompt. NLAs do not yet match state-of-the-art model diffing methods on the emergent misalignment detection task, but outperform a blackbox baseline. Our findings suggest that NLAs can meaningfully verbalize differences in activation space, rendering NLAs a promising direction for future work. 

---
# ReflectPO: Reflection-Guided Policy Optimization for Long-Horizon LLM Agents 

**Authors**: Rumei Li, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2610.30933)  

**Abstract**: With external tools, large language models (LLMs) can tackle complex tasks through sustained environmental interaction. LLM-based agents are therefore increasingly used to tackle complex long-horizon tasks in real-world environments. However, these tasks involve sequential, interdependent decisions and changes in environmental states. Early execution errors affect subsequent interaction histories and environmental states, and their effects accumulate along the decision chain. To address this challenge, we propose Reflection-Guided Policy Optimization (ReflectPO), a reinforcement learning framework for autonomous error recovery in long-horizon tasks. ReflectPO comprises two core components. The first is Reflection-Guided Sampling. We use branch differences and repeated visits in interaction trajectories to construct hints for error recovery. The second is Tree-Structured Credit Assignment. We organize sampled trajectories into trees and assign action-level credit using parent--child reward differences and recovery gains. Experiments over two long-horizon tool-use benchmarks show that ReflectPO outperforms multiple existing baselines in most domains. It also achieves the best overall performance on average. Further analyses show that ReflectPO effectively improves agents' ability to recognize their own deviations, recover from erroneous states, and continue making progress toward task completion. 

---
# Silicon Samples Match the Answers but Not the Respondents 

**Authors**: Yusuke Kumagae, Masahiro Kaneko  

**Link**: [PDF](https://arxiv.org/pdf/2610.30900)  

**Abstract**: Silicon sampling generates survey respondents by having a large language model (LLM) answer a survey as a person with given demographic attributes, and the resulting synthetic populations are increasingly used in place of human ones. Existing evaluations focus on individual-answer accuracy, per-question answer distributions, and correlations across questions. None of these compares the respondents themselves, so a generator whose respondents have combinations of answers found in no human respondent can still score well if it reproduces every answer distribution. We propose the respondent coverage gap (RCG), which represents each respondent as a point defined by their complete answer profile and measures how far each human respondent is from the nearest generated one, relative to the distance between human samples. On two US social surveys, all six LLMs we evaluate leave human respondents uncovered, and under our main settings, they fall behind a baseline that samples each answer independently from the human marginals. Across the prompting and inference settings we test, every LLM has a higher RCG than a Gaussian copula fitted to the human data. 

---
# From Perpetrators' Digital Footprints to Behavioral Communities: An Ontology-Driven Framework for Bengali Cyberbullying Analysis--- FULL VERSION 

**Authors**: Md. Nesarul Hoque, Md Hanif Seddiqui, Miskatul Anwara, Rudra Pratap Deb Nath  

**Link**: [PDF](https://arxiv.org/pdf/2610.30885)  

**Abstract**: Cyberbullying involves behavioral and contextual patterns beyond isolated abusive messages. However, existing approaches remain fragmented across content detection, semantic representation, and network analysis, with limited interpretable perpetrator representations, multi-scale structural modeling, and explanations of behavior-driven similarity and influence propagation. These limitations hinder the identification of persistent perpetrators, characterization of behavioral communities, and assessment of potentially influential users. This paper proposes Ontology-Driven Cyberbullying Behavior and Network Analysis (ODCBNA), a perpetrator-centric framework integrating requirements-driven ontology design, knowledge graph construction, behavioral modeling, community analysis, and influence modeling. Using the Bengali Contextual and Diverse Cyberbullying (BenCoDiC) corpus as a use case, ODCBNA identifies $1{,}063$ users with at least three bullying comments or replies as perpetrators and represents each through a $64$-dimensional behavioral and contextual feature vector. Cosine similarity and $k$-nearest-neighbor modeling produce a weighted network with $27{,}554$ edges. Louvain analysis identifies six communities with a modularity of approximately $0.495$ and $27$ sub-communities. Feature-contribution analysis identifies the Video, Impact, and User Profile groups as the strongest contributors to behavioral similarity and community separation, while contextual dimensions support interpretation. Community-constrained influence analysis shows that IM Greedy achieves small positive gains in all six communities, although different seed-selection methods produce comparable diffusion spreads despite limited seed overlap. These findings provide an interpretable basis for understanding perpetrator behavior, community organization, and potential influence patterns in Bengali online environments. 

---
# Synthetic-to-Real Code-Switched ASR: A Spontaneous Nepali-English Podcast Benchmark 

**Authors**: Firoj Paudel  

**Link**: [PDF](https://arxiv.org/pdf/2610.30860)  

**Abstract**: Nepali-English code-switched conversational speech presents severe challenges for automatic speech recognition (ASR), chiefly through script confusion (inappropriate mixing of Devanagari and Latin orthography) and elevated error rates at language switch points. Supervised fine-tuning (SFT) can adapt multilingual foundation models, but human code-switched audio is expensive to collect and annotate. We curate a new benchmark of 13,044 spontaneous podcast utterances (15.45 h total; 10,214 test, 2,830 dev), audited and verified to eliminate machine translation leakages and acoustic loops. We investigate a two-stage synthetic-to-real adaptation framework: Stage 1 applies SFT on 0.95 h of speech synthesized via a multi-speaker, dual-engine TTS pipeline; Stage 2 explores Group Relative Policy Optimization (GRPO) with transcription and script rewards. We find that synthetic SFT provides the primary cross-domain transfer to spontaneous human speech, reducing Mixed Error Rate (MER) from 23.40% to 21.52% (-1.88 pp) and switch-point error rate (SPER) from 35.89% to 28.77% (-7.12 pp). Subsequent GRPO yields small, statistically significant adjustments on 173 test utterances (1.69%), driving English script confusion (SCR_en) from 6.45% to 6.36% (p <= 0.001) and SPER to 28.70% (p = 0.022 clustered). Crucially, ablation reveals that in this SFT-converged encoder-decoder setting, an auxiliary script reward term yields performance identical to MER-only optimization because rollout candidate groups exhibit near-zero script variance on synthetic data. On monolingual benchmarks, SFT improves Nepali MER by 2.71 pp (stabilizing at 18.42% after GRPO, a 2.38 pp gain over baseline) while preserving English lexical WER (3.38% vs. 3.33%). Verified transcripts, timestamps, episode metadata, and reconstruction scripts are publicly released. 

---
# TRIAD: Benchmarking Omni-Modal Ambiguity in Multimodal Large Language Models 

**Authors**: Zhaolu Kang, Yidi Wang, Yile Li, Siqi Zeng, Yingjie He, Richeng Xuan, Zhichao hu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30841)  

**Abstract**: Ambiguity is a central feature of natural communication: a sentence, image, or sound may remain underspecified until evidence from the other channels is considered. Yet current multimodal benchmarks mostly test perception, recognition, or reasoning over already determinate inputs, leaving unclear whether omni-modal large language models can resolve ambiguity distributed across text, vision, and audio. We introduce TRIAD, a benchmark for omni-modal ambiguity resolution. Each item is a text--image--audio triplet paired with a question and a set of answer options, and is constructed so that the full triplet determines a unique answer while removing any one modality makes the set of consistent options non-unique. TRIAD contains $327$ hand-written queries in $86$ scene groups and covers $18$ ambiguity categories across the three modalities. Evaluating $14$ omni-modal systems reveals that strong item-level performance does not translate into robust scene-level disambiguation: the best model still trails humans by over $20$ points in group accuracy. Removing answer options widens the gap further, driving model group accuracy to the single digits while humans remain robust. Leave-one-out and transcript/description counterfactuals further show that current systems often underuse audio cues when text and image are present. TRIAD reframes omni-modal evaluation around disambiguation rather than recognition, and provides a taxonomy, benchmark, and diagnostic protocol for measuring this capability. 

---
# TutorCaseBench: A Case-Grounded Benchmark for Evaluating LLM Tutors in Supporting Students with Challenging Behaviors 

**Authors**: Yeon Su Park, Chanyoung Park, Tae Soo Kim, Seungju Kim, Hyunwoo Kim, Daho Jung, Jinsung Yoon, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2610.30828)  

**Abstract**: LLMs are increasingly used as AI tutors, yet existing benchmarks overlook the challenging behaviors common among students with learning and behavioral difficulties. We introduce TutorCaseBench, a benchmark that grounds AI-tutor evaluation in qualitative case studies from pedagogy, educational psychology, and special education. Our pipeline converts documented student categories into case-specific scenarios, tutor-student dialogues, and rubrics grounded in source evidence. The benchmark contains 923 dialogues across five challenge types, evaluated through open-ended response generation (1,560 items) and multiple-choice selection (2,537 items). Across 11 state-of-the-art LLMs, even the strongest model satisfies a rubric criterion in only 55.8% of open-ended cases, while models are better at selecting appropriate strategies than generating situated responses. These results reveal a gap between intervention recognition and context-specific adaptation, especially for open-weight models. All data are publicly available at this https URL. 

---
# Invisibilization as a Source of Probability Bias in Language Models 

**Authors**: Christian Arnold, Ernesto Calvo, Guadalupe Gonzalez, Sebastián Vallejo Vera, Tiago Ventura  

**Link**: [PDF](https://arxiv.org/pdf/2610.30785)  

**Abstract**: Language models are aligned to avoid using protected attributes such as race, ethnicity, and gender to inform their responses. Therefore, they increasingly refuse or hedge when such attributes appear in a prompt. However, if a system must decide how much to learn from what people tell it, it is costly to refuse valuable information because of its source. In this article, we measure the cost of declining information in a two-urn game with an exact Bayesian answer. We present leading LLM models from two developers (OpenAI and Anthropic) with a prompt that requires them to decide how far to trust an unverifiable report from a participant whose description we randomize across ten configurations (6,600 sessions). We show that leading models often decline to assess the reporter, treating information as worthless at rates set by the description rather than by the evidence. At its widest, models decline a Black-described reporter 46% of the time and a white-described reporter 9%; yet, when they do assess the information, the Black-described reporter receives slightly more trust. The bias is invisible to audits that compare average trust or look only at assessed cases. It grows with reasoning effort, follows words rather than categories (models invoke race for "Black" but not for its Spanish translation negro, and never for "Indigenous"), and worsens under warning instructions a practitioner would try first. We also present a scoring option that reduces the refusal rate bias. When a model's trust is scored against revealed truth over repeated rounds, it credits identical records equally. In four of five configurations from Anthropic and OpenAI, the racial gap in trust falls below 0.02 by the last rounds. 

---
# Where meanings converge and diverge: Revisiting linguistic relativity across 81 languages 

**Authors**: Weiran Wang, Wenjuan Qin, Haitao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30778)  

**Abstract**: Do people who speak different languages organize concepts in similar ways? This question lies at the heart of the Sapir--Whorf hypothesis and debates about linguistic relativity. Shared basic meanings, however, need not imply shared relations among them. We compared relations among 196 basic concepts across 81 languages, using word embeddings from two multilingual models to quantify conceptual structure expressed in language. Cross-language correspondence was greater for sociocultural meanings than for meanings concerning nature and motion, a contrast recurring across models and classifications. Within the sociocultural domain, relational differences increased with geographic separation; less closely related languages differed more across all domains. These findings suggest that universality and relativity are interwoven across conceptual domains. They inform an account in which shared cognitive resources interact with linguistic and local experience, and predict greater cross-language correspondence in human semantic judgments for sociocultural than for nature and motion meanings. 

---
# UniGuard: Multi-View Guardrail Learning for Cross-Interface Safety Generalization 

**Authors**: Viet-Thanh Pham, Thuy-Trang Vu, Junae Kim, Rollin Omari, Dinh Phung, Gholamreza Haffari, Lizhen Qu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30742)  

**Abstract**: Guardrail models are increasingly used as safety layers for LLMs and agentic systems, but most are trained on fixed prompt or response interfaces and become unreliable when the same safety case appears as a conversation, agent trajectory, multi-agent trace, or policy-conditioned input. We propose UniGuard, a training framework that learns interface-invariant safety representations from multiple semantically aligned views of the same safety case. We also construct the UniGuard-FT dataset by transforming 1,000 public safety cases into prompt, response, multi-turn conversation, agent-trace, and multi-agent-trace views, and train UniGuard-0.6B with classification, supervised contrastive learning, and asymmetric confidence calibration losses. Across unseen-interface benchmarks ATBench, PolicyGuardBench, and DynaBench, UniGuard-0.6B improves over its pretrained guardrail by 45.12 F1 points on average and reduces expected calibration error by 35.41 points, while remaining competitive with much larger guardrails such as Claude-Sonnet-4.5 and GPT-5.2 - despite being a 0.6B model and trained only with synthetic data. 

---
# Meaning Contracts: Local Semantic Geometry in Large Language Models 

**Authors**: Andrés Daniel Godoy Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2610.30715)  

**Abstract**: This paper studies the local geometry of semantic meaning in large language models. Current representation analyses ask whether meanings are close in embedding space, or whether linear probes separate semantic classes, but they do not characterize the within-concept local geometry: what is invariant across realizations of a single semantic factor, what varies, and what changes when meaning is refined. This paper characterizes that geometry through two controlled probes evaluated on four architecturally diverse decoder-only LLMs (Gemma 4, Llama 3.2, Qwen 2.5, Pythia). A cross-lingual probe uses language as a perturbation to separate exact meaning matches, same-factor realizations, nearby concepts, and out-of-domain contrasts. A refinement probe separates specificity-bearing semantic narrowing from high-probability attributive elaboration and adversarial controls; both probes are validated by blind human judgments. Four findings recur across all models: under mean pooling, once dominant variance directions are removed, sentence pairs order by semantic relatedness (same factor, then same domain, then different domain) without overlap over a band of layers; semantic factors organize around an invariant core surrounded by structured realization-specific variation; refinement produces directed displacement into tighter neighborhoods, including against controls built on the same base noun; and sentence-level and token-local readouts expose complementary geometry (volume contraction at the sentence level, sharper anisotropic endpoints at token-local readout). We interpret these results as evidence that LLM semantics are organized as local anisotropic neighborhoods in which meaning contracts under refinement. 

---
# ULME: Predictive Coupling for Mutually Reinforcing Embedding and Language Modeling 

**Authors**: Yang Liu, Zhiqiang Yuan, Shu-Dong Huang, Jiancheng Lv, Liwei Yang, Qiwei Yan, Kailin Lv, Jinchao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30705)  

**Abstract**: Text embedding and text generation are two of the most useful capabilities of modern large language models, yet they are typically provided by separate specialized models (an embedder and a generator) trained with incompatible objectives and deployed side by side in retrieval-augmented generation and agent systems. Recent unified embedding-generation models such as GritLM share parameters between the two capabilities, but they either switch attention regimes between modes or optimize embedding and language modeling as parallel objectives that never supervise each other. We argue that these designs are only partially unified: the two capabilities coexist on the same weights, but do not shape each other. We propose ULME, a post-training framework that unifies embedding and language modeling on a causal decoder-only LLM through predictive coupling. ULME inserts a chain of learnable slots between the query and the answer, reads a semantic embedding and a reasoning embedding from two designated slots, and trains three objectives on a shared causal backbone: a contrastive loss anchoring the semantic slot to document representations, a language-modeling loss supervising answer generation, and a stop-gradient predictive loss that aligns the reasoning slot with a pooled summary of the answer, so that generation actively shapes what the embedding encodes. Across two model families (Qwen2.5, Qwen3) and eight scales spanning 0.5B-8B parameters, a single ULME checkpoint improves MTEB by 3.3-4.1 points over the strongest open-source baseline, tightens BRIGHT and AdvBench-IR simultaneously, and remains on par with a generation-only fine-tune on standard accuracy benchmarks (MMLU, GSM8K, MMLU-Pro, HumanEval), demonstrating that genuine unification yields mutually reinforcing rather than merely coexisting capabilities. 

---
# From Clinical Judgment to Checkable Rules: Clinician-Led Guardrails for Mental-Health Chatbots 

**Authors**: Chenhan Lyu, Kit Wislocki, Sarah Elizabeth Stoeckl, Maitreyi Sinha, Amir Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2610.30690)  

**Abstract**: Large language models (LLMs) deployed in mental-health contexts can be evaluated for safety failures, but the resulting judgments are often difficult to inspect or explain. Common approaches to improving safety rely on prompt engineering, LLM-as-a-Judge, and Constitutional AI. In this work, we describe a clinician-led alternative in which two clinicians define, finalize, and adjudicate the four safety checks of a rule base, which is formally verified for internal consistency and applied as a guardrail over LLM outputs. Here and in the title, clinician-led refers to this rule base, not to the fact extractor or the uncertainty default, which are ours. Every verdict of the resulting verification layer is recomputable from a named rule and sixteen facts. Against an LLM judge matched in model, revision, quantization, rule text, and decoding but without the layer's computed 988/911 fact, the largest disagreement class, the satisfied-rule class, is structured: the Llama-3.3-70B judge, on its primary prompt, demands a crisis resource from 55 of the 65 replies (84.6%) that already contain one, while the same model, asked as a fact extractor, reports a recognized resource present on 54 of the 55. The verification layer never raises that demand, by construction. We report with equal prominence where this judge is stronger: fewer false crisis alarms on the one reference-scored check. We also report the measured cost (a specification revision raised block-class outcomes on held-out boundary prompts of an over-refusal benchmark from 6.0% to 16.0%, chiefly by changing fact extraction), the clinicians' decision to keep our uncertainty default, and the analysis of the result. The contribution is not a safer verdict. It is a guardrail whose disagreements with the judge localize to named rules and facts and whose measured costs trace to named facts and the uncertainty default. 

---
# Direct cueing of word identity through word duration: evidence from Mandarin homophones 

**Authors**: Xiaomeng Zhu, Megan Niu, Jason Shaw, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30682)  

**Abstract**: While there is increasing evidence that words are stored in the mental lexicon with some phonetic detail, there are open questions about exactly which phonetic details are stored and whether they are functionally relevant for spoken word recognition. Word-level phonetic cues, such as word duration, provide an appropriate test case. Across languages, including Mandarin Chinese, high frequency words are shorter than low frequency words. Using a homophone ordering task, we investigate whether differences in word duration impact word recognition. Mandarin Chinese was examined since it has abundant disyllabic homophones. Our results indicate that participants were able to use word duration to recognize the order of homophone pairs, selected to vary in lexical frequency. Both with high noise levels and when no other cues were available, participants mapped the shorter duration auditory stimulus to the higher frequency word. The results provide evidence that word duration can directly cue the identity of words, suggesting that it may be a phonetic detail incorporated into the lexical representation. We discuss this direct cueing account alongside alternative accounts of how word duration may modulate word recognition. 

---
# WikiOriginQA: Automated Construction of a QA Dataset with Knowledge Origins for Analyzing Cultural Bias 

**Authors**: Kento Haneda, Yosuke Kishinami, Ryo Fujii, Keisuke Sakaguchi, Makoto Morishita  

**Link**: [PDF](https://arxiv.org/pdf/2610.30677)  

**Abstract**: Existing datasets on cultural knowledge have been typically created by human annotators familiar with each culture. However, the process requires considerable human effort, presenting a significant challenge to scalability. To address this challenge, we introduce WikiOriginQA, a QA dataset automatically constructed by objectively representing the origins of knowledge through the creation dates of different Wikipedia language editions. Experimental results with the created dataset confirm that LLM performance tends to be biased toward Anglophone cultures. Furthermore, the results suggest the importance of knowledge propagation into the Anglosphere. These findings demonstrate the validity of our automated approach, enabling cultural bias evaluations with superior scalability across diverse languages without extensive manual annotation. 

---
# Spectral Smoothing and Spiking Perturbations Reconfigure Capability Profiles and Selective Task Performance in Large Language Models 

**Authors**: Qince Luan, Jianhui Xu, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30676)  

**Abstract**: Singular-value spectra encode important structure in transformer feed-forward projections, yet their causal role in downstream behavior remains underexplored. We study post-hoc spectral smoothing and spiking interventions on the up_proj and down_proj matrices. Positive perturbation stretches deviations from the geometric mean, making large singular values larger and small ones smaller (spiking), and vice versa. The intervention preserves singular vectors and the geometric mean of the singular values. Across five open Llama and Qwen instruction models, we evaluate the perturbed systems on capability-labeled subtasks, standard and chain-of-thought reasoning benchmarks, judge-scored truthfulness, hallucination, and safety tasks, and TRAIT personality probes. To interpret score shifts, we fit responses in a three-dimensional capability basis consisting of factual knowledge, language understanding, and deductive reasoning. The induced changes suggest structured rather than purely random sensitivity: large positive perturbations more reliably suppress language and deductive responses, moderate negative perturbations can sometimes improve factual or task-local performance on the evaluated grids, and sensitivity varies strongly by model family, benchmark module, and underlying layerwise spectral inequality. These results position spectral perturbation as a lightweight intervention for probing how capabilities are organized and traded off in large language models. 

---
# Compiling Autoregressive LLM Control into Policy-Exact Parallel Inference for Hierarchical RAG 

**Authors**: Zehao Lin, Qianfeng Shen, Xixuan Hao, Shaobo Cui, Chunyu Li, Kai Chen, Bo Tang, Zhenjun Wang, Jin Yan, Zhiyu Li, Feiyu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.30664)  

**Abstract**: Large language models increasingly act as controllers in structured decision pipelines, yet their decisions are still executed by autoregressive text decoding. Hierarchical RAG is one such setting: at each frontier of a document tree, an LLM commits to $b$ of $k$ sections under an input-dependent action grammar, executed as either failure-prone free-form JSON or reliable but strictly serial grammar-constrained generation. We show that this serialization is unnecessary for bounded-horizon control: the reachable future forms a shallow action-prefix tree, which we compile --- one root prefill plus one tree-causal probe evaluate every prefix-conditioned action distribution with serial decoding's tokens, positions, and causal ancestry. A prefix-state lemma makes compiled control policy-exact in exact arithmetic: the same policy with a two-forward critical path and zero output tokens. One further probe pass certifies the exact order-marginalized MAP set. Across four document-disjoint splits and two scales on the primary MetaX device, compiled control matches the serial policy on $84$--$89\%$ of questions --- every instrumented divergence a floating-point near-tie --- and cuts control latency by $35$--$47\%$ versus token-wise serial decoding and $6$--$7\%$ versus an idealized chunked engine; an H800 repetition preserves the ordering. 

---
# What the Specification Did Not Say: Specification-Oracle Gaps in Test-Graded Coding-Agent Benchmarks 

**Authors**: Fei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2610.30635)  

**Abstract**: Coding-agent benchmarks often describe a task in natural language but grade it with tests taken from the upstream project. Passing those tests can require information the description never states: a constant's name, a module path, an exact message. We call this a specification-oracle gap: where it exists, a benchmark also measures whether an agent reproduces upstream implementation details. We formalise the gap, including the inconsistencies that chains of dependent tasks add, and build a static audit for five languages that classifies every name-, path- and literal-level dependency of every scored test (held-out precision 89.2%, recall 92.9%). On SWE-Milestone (98 graded milestones, 35 published runs), 25 milestones contain a test that depends on an unstated, non-inferable name or path. Build units with such a dependency fail in 80.6% of runs, against 48.4% for comparable units without one, and withholding a single hidden name from the reference stops its unit's tests from compiling in 126 of 128 cases. The leaderboard's ranking is robust to removing the affected tests, but they shift scores within repositories and favour agents that reproduce upstream names. Across four further benchmarks, prevalence appears to follow construction: a third or more of the tasks are affected where specifications are release notes or LLM summaries, almost none where tasks are screened or written for their tests; how much a gap costs depends on how the harness grades. We release the audit, corrected filter lists and specification addenda. 

---
# IFC-MCQ: Automatically Generating Multiple-Choice Benchmarks for Evaluating LLM Agents on IFC Data Retrieval 

**Authors**: Rabindra Lamsal, Sisi Zlatanova, Xuesong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2610.30533)  

**Abstract**: Industry Foundation Classes (IFC), the standard for building information exchange in architecture, engineering, and construction (AEC), is increasingly queried through large language models (LLMs) using natural language. Evaluating these systems at scale is difficult because free-text answers require costly human or LLM-based review. We propose IFC-MCQ, a method that automatically generates multiple-choice benchmarks from IFC files for exact option-set grading. The taxonomy covers 15 categories and 36 subtypes across four groups, with answer keys derived from the file and distractors constructed from file-derived values or predefined rules. We applied IFC-MCQ to IFC models of varying sizes and modelling characteristics to assess generation coverage and evaluated two agentic configurations on selected models using tool- and code-based retrieval. The evaluation exposed category-specific failure modes and showed that performance can vary with the source IFC model and retrieval interface. IFC-MCQ enables scalable benchmarking of LLM-based IFC retrieval in AEC workflows. 

---
# Protegrity-PII Technical Report 

**Authors**: Hardeep Bassi, Hui Ma, Luís Seabra, Koowum Joshi, Daniel Pereira, Averell Gatton  

**Link**: [PDF](https://arxiv.org/pdf/2610.30518)  

**Abstract**: We present Protegrity-PII, a new PII decision model for structured and unstructured text. Protegrity-PII achieves a typed recall of 0.915 with a precision of 0.907 on a new benchmark we propose, PII-SCOPE. Our model processes text in 0.49 seconds per million characters at \$0.0005 per million characters on a single RTX 6000 using batched inference. Due in part to the expansive nature of the entity type taxonomy used in training, Protegrity-PII achieves superior out-of-distribution generalization when benchmarked against a selection of state of the art models and datasets. 

---
# Behind Aggregate Scores: What Systematically Varying Arena-Hard's LLM Judge Reveals 

**Authors**: Louise Yorke  

**Link**: [PDF](https://arxiv.org/pdf/2610.30512)  

**Abstract**: LLM judges are used to determine which of two answers is better. Trusting these verdicts requires knowing how repeatable they are and how much they depend on the judge's instructions. We show that similar aggregate scores can conceal substantial changes in individual verdicts, both when the same setup is repeated and when the instructions change. Using 500 Arena-Hard v2.0 prompts with two models' answers held fixed, we run GPT-4.1 as the judge under all eight combinations of criteria, output format and task framing, and repeat every setup unchanged at temperature zero on a pinned model version. Rerunning the default setup moves the aggregate score by about one point but changes the win/tie/loss outcome on 21.2% of prompts. Instruction changes introduce further sensitivity: in one setup, which asks for the verdict without reasoning and also drops several other instructions, outcomes change on 28.2 percentage points more prompts than the two setups' unchanged reruns do on average, and the aggregate score falls by 4.09 points. In another, where the criteria prioritise accuracy, outcomes change on 6.7 percentage points more prompts by the same comparison, while the aggregate score barely moves. A stable aggregate score therefore does not establish that individual verdicts are repeatable. This distinction matters when those verdicts are used as preference labels or to assess individual responses. Evaluations that rely on individual verdicts should repeat each setup they use and report both aggregate scores and prompt-level agreement. 

---
# Rethinking the Evaluation of LLM Fact-Checkers 

**Authors**: Ayana Niwa, Miyu Yamada, Kouta Nakayama, Masahiro Kaneko, Hisami Suzuki, Satoshi Sekine  

**Link**: [PDF](https://arxiv.org/pdf/2610.30504)  

**Abstract**: Large language models (LLMs) are widely used to answer factual questions, but their answers can contain errors and must be fact-checked claim by claim. Yet the resulting score depends on three design decisions: which questions the claims answer, under what conditions each claim is judged, and which sources count as evidence. No existing benchmark specifies all three. We introduce JSFactCheckBench, a Japanese benchmark that specifies all three, with 1,913 claims from LLM answers to questions about misleading information that circulated on Japanese social media, each annotated for factuality at a reference time and for importance to the question, using evidence from preselected Web sources. Across six LLMs, all verify claims directly answering the question less accurately than supplementary claims, a gap that aggregate scores obscure. Scores also drop when models must locate the relevant content in longer or multiple sources and, for all four open models, when the surrounding question and response are added. These weaknesses are visible only because the benchmark records the conditions behind each label. A benchmark that leaves them implicit cannot reveal them. 

---
# Self-Assessment Reinforcement Learning for Search-State Awareness in Retrieval-Augmented Large Language Models 

**Authors**: Taewhoo Lee, Minju Song, Jungwoo Park, Chanwoong Yoon, Junha Jung, Hyeon Hwang, Sanghoon Lee, Minbyul Jeong, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30435)  

**Abstract**: Recent advances in search agents have enabled large language models to tackle complex tasks by retrieving and synthesizing external information. However, existing search agents often lack search-state awareness, i.e., the ability to recognize what information has been obtained and what remains missing, leading to hallucinated reasoning and prematurely terminated search. To address this limitation, we propose Self-Assessment Reinforcement Learning (SARL), a reinforcement learning framework in which models learn to recognize their own search state during training and improve their search behavior accordingly, without requiring additional assessment at test time. SARL achieves this through a dual reward design that captures the corrective effect of each assessment while penalizing excessive reliance on it, jointly encouraging effective and efficient search trajectories. Under this setup, SARL achieves a 24.3% relative improvement over the strongest baseline, with average gains of 35.9% on multi-hop reasoning and 27.9% on deep research benchmarks against the strongest per-benchmark baselines. SARL further surpasses methods built on auxiliary supervision or multi-agent coordination, despite operating as a standalone RL framework with no inference-time scaffolding. Beyond final performance, SARL maintains training stability under extended search horizons, a regime where existing methods consistently degrade, enabling sustained gains as task complexity grows. 

---
# SIM-UT: A User Simulation based Evaluation Framework for Medical Triage Chatbots 

**Authors**: Deepali Mittal, Dheeraj Mundhra, Yulia Sidorova, Tersia de Kock, Oli Munns, Sofie S Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2610.30412)  

**Abstract**: LLM-based chat agents are increasingly used for complex, multi-turn, multilingual medical triage in low- and middle-income countries (LMICs), yet evaluation methods focus on single-turn, English-only benchmarks that primarily test the chatbots' knowledge. In practice, users are often confused, overwhelmed, or emotionally vulnerable, giving unclear, misspelled single-word answers or long messages in regional dialects, and the context of a conversation can shift rapidly. We propose SIM-UT (Simulation of User Behaviour for Triage), a multi-turn, simulation-based evaluation method that identifies ecologically valid, contextually relevant failure modes of LLM-based triage chat agents. Using SIM-UT, we evaluated three real-world WhatsApp-based health chatbots: a respiratory screening programme in India, a hearing health support programme in South Africa, and a diabetes and lifestyle coaching agent in Kenya. We compared a medically fine-tuned open model (MedGemma) against general-purpose frontier models (ChatGPT and Claude) across 5,000+ multilingual simulated conversation turns grounded in the personas, constraints, and communication patterns of the real user populations, over 2-4 iterative rounds. A key failure mode emerged in both model families: de-escalation risk. The agents initially identified medical emergencies correctly, but when the user minimised the symptoms without introducing new clinical information, they de-escalated, exhibiting clinically dangerous sycophancy, for example agreeing that vaginal bleeding in pregnancy was "not a problem" or that decreased fetal movements at 32 weeks "can be normal". We recommend user simulation-based behavioural evaluation, grounded in real user context and covering adversarial and edge cases, as standard practice for triage chatbots in production, so that safeguards can be added before deployment or in subsequent iterations. 

---
# MemPlan: Memory-Conditioned PDDL Planning for Partially Observable Text Environments 

**Authors**: Tengwei Song, Robert Hoehndorf  

**Link**: [PDF](https://arxiv.org/pdf/2610.30379)  

**Abstract**: LLM agents are flexible in language interaction, but using them as long-horizon planners is costly and prone to hallucinated or invalid actions. This paper targets a different role for LLMs: constrained bridges between language and symbolic planning. We propose MemPlan, a memory-conditioned PDDL planning framework for partially observable interactive text environments. MemPlan keeps the PDDL domain fixed and reconstructs the PDDL problem at each step from parsed observations, candidate hypotheses, and interaction memory. Positive graph memory assigns relevance-based action costs to candidate hypotheses for missing objects, locations, or conditions, guiding the planner toward more plausible checks. Negative memory records execution-refuted assumptions and compiles them into exclusion constraints, preventing repeated attempts at candidates already contradicted by feedback. A unified schema-conditioned language--symbol interface, trained offline with supervised fine-tuning, connects text environments to the planning loop through observation-to-fact parsing and symbolic-action-to-command grounding. Experiments on TextWorld, ALFWorld, and Robotouille show that MemPlan improves task completion and step efficiency over language-only and PDDL-aware baselines, while substantially reducing online token cost compared with LLM-planning baselines such as ReAct. The same fine-tuned interface is also reused across TextWorld task types without task-specific fine-tuning. Our code and data are available at: this https URL. 

---
# Climbing the Ladder: Automating Data Pipelines and Reproducing Baselines in Low-Resource Settings 

**Authors**: Donato Meoli, Eryk Wdowiak  

**Link**: [PDF](https://arxiv.org/pdf/2610.30374)  

**Abstract**: The massively multilingual translation systems that now include Sicilian learned it from noisy, non-standardised web text. To obtain better translation quality, we redevelop the small neural machine translation (NMT) system that Wdowiak customized for Sicilian. We rebuild on maintained software the extraction of the parallel text from the PDFs of the bilingual journal Arba Sicula, whose sentences are aligned by a language-agnostic sentence encoder with thresholds tuned against an existing hand alignment and then filtered for the typical errors of automatic alignment. Furthermore, the tokenization and the desinence-biased subword segmentation of the original system are ported to a Perl-free implementation, and all the results are measured on a frozen test set drawn from the journal. We compare small Transformers trained from scratch with the pretrained NLLB model adapted by low-rank fine-tuning, in both directions and with Italian as a closely related bridge language, and we adapt to the pretrained model the reverse-training recipe of the original system, i.e., a stage on forward-scored back-translations of the web Sicilian followed by a stage on the curated literary text. The pretrained model is by far the better of the two, and the staged training improves the translation into Sicilian more than the translation from it. On the test set of the original system, and in the tokenized space in which it was scored, the staged model with a larger adapter and more back-translations is above the reverse-training recipe in the directions that involve English and below it between Sicilian and Italian. Yet, a multilingual adapter degrades the directions on which it was not trained, which limits its use as a general multilingual model. 

---
# CiteAnchor: Source-Aware Grounding for Hallucinated Citation Detection and Resolution 

**Authors**: Chenyi Tong, Ping Nie, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30355)  

**Abstract**: Hallucinated citations make it harder to trace sources and verify the evidence behind scientific claims. Detecting them is difficult because bibliographic records can be incomplete or differ across source types and publication versions. A plausible match alone neither validates the citation nor establishes its underlying source. We introduce CiteResolve, a benchmark of 423 real-world hallucinated citations annotated for source recoverability. We further propose CiteAnchor, a source-aware grounding framework that adapts retrieval and field-level verification to each source type and separately determines citation validity and whether a unique source can be recovered. On an existing detection benchmark, CiteAnchor achieves 0.841 F1 with GPT-5.6 Luna and 0.784 with an open-weight backbone, outperforming the baseline with the same backbones by 7.9 and 36.8 points. On CiteResolve, it achieves 74.7% and 70.7% end-to-end task success, outperforming the corresponding web search baselines by 16.3 and 65.3 percentage points. Our code and data are available at this https URL. 

---
# Consistency Is Not Validity: Self-Rewarded Moral Argumentation Makes Language Models Decisive but Pliable 

**Authors**: Marius De Kuthy Meurers, Aaron Parisi, Bernd Bohnet  

**Link**: [PDF](https://arxiv.org/pdf/2610.30349)  

**Abstract**: LLMs are increasingly consulted for moral judgment. We aim to improve performance on such tasks by rewarding good arguments, on the premise that arguing well increases the likelihood of judging well. We train a reinforcement learning policy initialized from a state-of-the-art language model in a self-judged tournament with masked thinking: the generated arguments are judged pairwise on argument quality by the policy itself. As an empirical basis, we extract authentic dilemmas from classic novels, using a logic solver to verify that each constitutes a conflict. We show that our judge rates argument quality like human annotators, and the policy then adopts the language register that it rewards. Testing on the external benchmarks MoralChoice, ELEPHANT, and MORABLES, we find that the policy becomes more decisive and invariant to the order in which options are presented, but sycophancy and reframing remain a challenge to validity. Two ablation studies with synthetic dilemmas and unmasked thinking match the reordering gains, but show a weaker register shift and higher sycophancy. We conclude that consistency is not validity: consistency under reordering can be optimized without improving the moral judgment it is meant to signal. 

---
# AQL-Lite: Enabling Efficient Knowledge Acquisition for Arabic LLMs via Data-Centric Continual Pretraining 

**Authors**: Houssam Eddine Boukhalfa, Mourad Bouache, Mourad Dahmane  

**Link**: [PDF](https://arxiv.org/pdf/2610.30344)  

**Abstract**: Training Arabic large language models faces challenges from noisy web data, dialectal variation, and the high cost of continual pretraining (CPT). We introduce AQL-Lite (AQL: Arabic Quality Language), a time- and resource-efficient framework for adapting a multilingual LLM to Modern Standard Arabic (MSA). AQL-Lite couples a quality-first curation pipeline (structural cleaning, heuristic and model-based filtering, MSA selection, and deduplication) with a partial-parameter CPT recipe that trains RS-LoRA adapters across the network together with the last eight transformer blocks and the embedding matrix, under layer-wise learning rates. At a matched 1B-token budget, CPT on AQL-curated data raises a four-benchmark Arabic mean by 3.51 points over CPT on raw web text, which yields no gain over the base model. In a controlled 5B-token comparison, AQL-Lite recovers 94% of the gain of full-parameter CPT (mean 73.19 vs. 73.58) in 52% of the wall-clock time and outperforms adapter-only baselines on every benchmark, and a component ablation shows that adapters, unfrozen blocks, and embedding training each contribute. Scaled to 40B MSA tokens and instruction-tuned on one million examples, the resulting 12B model scores 75.3 on the seven-benchmark Open Arabic LLM Leaderboard v2 (OALL v2) suite. These results suggest that partial-parameter CPT on a carefully curated corpus is a practical route to efficient MSA adaptation. 

---
# Measuring Research Progress and Linguistic Inclusion in Low-Resource Languages 

**Authors**: Atnafu Lambebo Tonja, Tadesse Destaw Belay, Jesujoba Oluwadara Alabi, Israel Abebe Azime, Benjamin Rosman, Ehsan Shareghi, Thamar Solorio, Pontus Stenetorp, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2610.30296)  

**Abstract**: Large Language Models (LLMs) have accelerated research on low-resource and multilingual NLP, but their effectiveness in creating an equitable landscape remains unmapped. We analyze 1,158 peer-reviewed open-access papers published between 2020 and 2025, examining research attention to low-resource languages (LRLs) and representation across language and speaker populations. To assess linguistic inclusion beyond publication counts, we introduce the Linguistic Inclusion Framework (LIF), a multidimensional framework that measures linguistic inclusion in four dimensions: research growth, representation expansion, representation continuity, and distributional equity. While annual publications grew from 52 to 415, a 5.4X increase in their ACL Anthology publication share, this growth largely concentrated on represented languages rather than expanding coverage of underrepresented languages. Further analysis shows that languages spoken in Asia account for 64.1% of speakers across LRLs we analyze but receive only 46.7% of the language representation share (0.7X their proportional share), while European languages receive 5.3X their proportional share. At the country level, linguistically diverse regions such as sub-Saharan Africa and Southeast Asia show the lowest language coverage ratios. Applying the framework at the continental level, we find that no region scores well on all four dimensions, showing that research growth alone is not a reliable metric of progress toward equitable and inclusive language technology\footnote{We will release all collected and annotated data for research purposes}. 

---
# FusionEval: Benchmarking Model Fusion for Large Language Models 

**Authors**: Shuo Cai, Yanggan Gu, Yuhang Liu, Su Lu, Yuanyi Wang, Yang Yu, Zhen Li, Zhitian Hou, Guanghao Zhu, Ming Li, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30162)  

**Abstract**: Model fusion aims to integrate the capabilities of multi-expert models into a single model. Existing evaluations offer limited coverage of reasoning with long chains of thought and tool use, while strong average performance can mask substantial losses in individual expert capabilities. We introduce FusionEval, a benchmark that provides controlled expert pools, integrates fusion methods with and without additional training, and offers a unified evaluation suite. The pools contain domain experts independently trained from Qwen3 and Llama base models across math, code, science, instruction following, and tool use. The suite evaluates capability retention and computational cost on eight public benchmarks under comparable data access rules, reporting normalized average and worst performance, runtime, and peak GPU memory usage. Extensive experiments show that training-free methods preserve reasoning performance but substantially degrade tool use at both model scales. In contrast, MOPD preserves tool use and achieves a more balanced capability profile, but requires additional data preparation, inference, and training. Jointly retaining reasoning, instruction following, and tool use at low computational cost remains a challenge. We hope FusionEval supports future research on efficient model fusion that jointly preserves expert capabilities. Our code is available at this https URL. 

---
# TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents 

**Authors**: Krishanu Adhikary, Prashant Devadiga  

**Link**: [PDF](https://arxiv.org/pdf/2610.30161)  

**Abstract**: Many Indian banks serve customers through FAQ pages and scripted chatbots. Assistants based on language models could respond faster and carry out requests such as blocking a card, but only if they give correct information and follow the bank's rules. These rules prescribe a procedure, a set of required steps, such as obtaining a closure quote before closing a deposit. Adapting models to this domain requires measuring how well they follow the procedure, before and after training. Benchmarks usually judge only the records left at the end of a conversation. Many required steps, however, leave the records unchanged, so an assistant that skips them can still be judged successful. We present TauIndianBankBench, an action-oriented benchmark of 1,000 Indian retail-banking tasks where an assistant helps a simulated customer. Besides the final records, it checks that the assistant took every required step and made no unrequested change. For every model, 6.5-12.0% of sessions are judged successful when only the final records are checked, although the assistant skipped a required step or made an unrequested change. Scoring both procedure and outcome shows more accurately how well language models perform in Indian retail banking. 

---
# Lighthouse: On Building a Trustworthy Digital Assistant for Exploring History with Civil Rights Archives 

**Authors**: John A. Aydin, Angelica Williams, Steven Dindl, Nathan Saunders, Kate Boyd, Vandana Srivastava, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2610.30158)  

**Abstract**: Artificial intelligence (AI), as a decision-support aid, provides new tools for studying history effectively using multimodal data, including text, images, and audio. We present, to our knowledge, the first digital assistant named Lighthouse--commonly called a chatbot--for exploring the Civil Rights period of the United States using data from South Carolina. From an AI perspective, the system's novelty lies in its guiding design principles of (a) focusing on trustworthiness of content and (b) by modeling and reducing potential harm to users from generated answers. It implements these with a neuro-symbolic approach that combines neural learning through a large language model (LLM) with rule-based symbolic solving, balancing risk of answers with generality of coverage. We evaluate Lighthouse with both factual objective questions that admit a single acceptable answer as well as subjective questions. For the latter, a dataset of subjective question-answer pairs (QAs) is created by discovering a "gold standard" by aggregating answers from independent experts. Lighthouse has access to over 18k primary sources from a range of collections and can be easily extended. 

---
# baseVultra: Compact Nonlinear Recurrence with Reallocated Capacity 

**Authors**: Linke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30154)  

**Abstract**: Nonlinear recurrent networks remain useful for streaming sequence modeling, yet gated cell design is still dominated by LSTM and GRU. We study whether recurrent computation can be reallocated away from gate transforms and toward candidate generation. We propose baseVultra, a compact recurrent cell that concentrates the dense recurrent budget on a single H x H candidate-generation matrix, while replacing recurrent gate matrices with O(H) diagonal interactions. The candidate update combines this dense branch with two low-rank auxiliary branches, an EMA cache, and learnable output blending; the gates additionally receive lightweight second-order access to h_{t-2} through diagonal terms. On TinyShakespeare character-level language modeling, baseVultra achieves a validation loss of 1.594 in the compact one-layer 200K setting, compared with 1.604 for a parameter-matched LSTM and 1.616 for a parameter-matched GRU, while its dense recurrent matrix-vector computation is about 54% of GRU under matched hidden size for this configuration. On text8, a roughly 45x larger benchmark, baseVultra is competitive with full-gate cells (1.854 bpc vs. 1.849 for GRU and 1.837 for LSTM). On formal-language probes, the largest gains appear under constrained hidden-state capacity (0.991 vs. 0.761 GRU and 0.691 LSTM OOD2 accuracy on Counter at H=16) and vanish as hidden size grows. These results characterize where and when candidate-focused recurrent capacity allocation helps: the benefit is concentrated in compact regimes, while full-gate cells remain competitive at scale. 

---
# Direct Pass@$k$ Optimization for Thought Pruning: Balancing Quality and Diversity in Test-Time Reasoning 

**Authors**: Mahdi Salmani, Adel Javanmard, Devansh Gupta, Meisam Razaviyayn  

**Link**: [PDF](https://arxiv.org/pdf/2610.31667)  

**Abstract**: Test-time scaling has emerged as a key inference strategy for reasoning models, where a model generates a large number of candidate solutions in the hope of finding at least one correct solution. Running all candidate solutions to completion can be computationally demanding, so recent work prunes the candidate pool after generating only a short prefix, committing further computation only to the surviving candidates. Existing pruning strategies divide broadly into two categories: Quality-based methods assign a score to each partial solution and discard those predicted to ultimately lead to an incorrect final solution, while diversity-based methods select partial solutions that follow different reasoning trajectories and are therefore likely to cover different final solutions. The tension between these two objectives is an instance of exploration versus exploitation, yet little to no prior work on pruning exposes this trade-off directly, let alone optimizes it. In this paper, we introduce a pruning procedure that addresses this trade-off directly by optimizing pass@$k$ over the selected solution pool. In particular, it chooses the pool to maximize the probability that at least one of its partial solutions, when continued to completion, yields a correct answer. Because this objective simultaneously rewards high-quality partial solutions and penalizes redundancy among them, quality and diversity are naturally balanced by the criterion itself rather than through separate ad-hoc design choices. On eight open-weight reasoning models, our method outperforms existing quality-based and diversity-based pruning strategies. The gains are particularly pronounced when the pruning procedure keeps a moderate number of partial solutions for continued generation, highlighting the value of directly balancing quality and diversity during pruning. 

---
# Cost-Aware Abstention for LLM-Based IT Ticket Classification 

**Authors**: Sachin Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2610.31657)  

**Abstract**: Wrong automatic routing of an IT support ticket can cost more than human triage. We evaluate cost-aware abstention as an offline, cost-conditional readiness assessment: select a routing policy on calibration data under explicit costs, then evaluate that frozen policy once on held-out test data. On a public 47,837-ticket, eight-class corpus, we compare four trained non-LLM classifiers with four large language models under zero- and few-shot prompting. Under the illustrative cost model (c_ok, c_triage, c_wrong) = (0, 1, 5), TF-IDF with logistic regression and a threshold of 0.60 selected on calibration reduces test cost from 0.763 to 0.472, a 38.1% relative reduction (paired ratio-bootstrap 95% CI [35.7%, 40.4%]), while lowering wrong routes from 15.3% to 3.5%. Frozen calibration-selected policies also beat both always-route and always-defer on held-out test data in 14 of 16 settings in a post-hoc cost-parameter analysis. The strongest prompted configuration reaches 0.477 accuracy on a stratified 1,000-ticket test subset, versus 0.879 for fine-tuned DistilBERT on the same tickets. Isotonic lowers expected calibration error without reliably improving discrimination: GPT-4o-mini few-shot has one confidence value before and after recalibration and correctness AUROC 0.500. We therefore report correctness AUROC and tie-aware risk-coverage area alongside ECE. At the analytically derived break-even probability 0.80, seven prompted configurations defer all tickets; Haiku few-shot routes only 19/1,000. An LLM-only agreement control and margin/entropy rejection baselines further separate calibration from discrimination. The findings are scoped to this dataset, prompt design, model endpoints, and illustrative costs; the work is an evaluation framework, not a new abstention algorithm or universal deployment certification. 

---
# Diffusion Language Models for Perplexity-Aware Black-Box Attacks 

**Authors**: Ali Al-Lawati, Elsayed Eshra, Dongwon Lee, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31558)  

**Abstract**: \emph{Continuous-space} attacks find prompts that trigger malicious behaviors, such as overthinking, in black-box LLMs by sampling or optimizing over a low-dimensional Gaussian latent space and decoding it to text through a \emph{random mapping} and nearest-token lookup. Because the random mapping is not learned from language, its prompts can be highly unnatural, and as such easily detectable through input guardrails, and the risk estimates they yield are not grounded in natural prompts. In this paper, we demonstrate that the same continuous latent-space framework can be coupled directly to a learned generative distribution over language, a continuous diffusion language model (dLLM), which removes the artificial random mapping while preserving the standard-Gaussian latent formulation and the downstream sampling or optimization procedure. We evaluate this approach under two settings: sampling-based estimation using \emph{RareTrap} and optimization-based search using \emph{ThinkTrap}. We find that character-level perplexity drops from $10^3$--$10^7$ to 3--5 in RareTrap and from 20--28 to 4.3--5.4 in ThinkTrap. We also find that model risk estimates can differ by an order of magnitude across the two mappings. 

---
# Measuring the Verifier's Floor: How Much Apparent Rare-Word Text-to-Speech Error Is ASR Error? 

**Authors**: Kirill Borodin, Vasilii Kudryavtsev, Nikita Vasiliev, Grach Mkrtchian  

**Link**: [PDF](https://arxiv.org/pdf/2610.31470)  

**Abstract**: Text-to-speech (TTS) is usually evaluated by transcribing its output with an automatic speech recognition (ASR) system, so the word error rate also counts recognizer errors. We measure these errors on human recordings of the test sentences. On LibriSpeech and FLEURS, the evaluation recognizer misses 33-67% of very rare words even in human speech, and all three recognizers miss 22-42%. Checking every word with two more recognizer families lowers the apparent TTS error on words that at least one recognizer hears two- to three-fold (very rare LibriSpeech words: from 10-13% to 3-5%), yet very rare words remain the hardest. Ignoring words the recognizer misses in human speech is biased: the other recognizers find TTS errors far more often on these words (4-6% vs. 0.2-0.3%) and overturn about half of the errors this mask keeps. As half of the remaining adjudicated errors may be spelling variants, a listening test is needed. 

---
# How Different, or Different How? Stance Granularity in Viewpoint Selection for Dialog Systems that Engage Users with Alternative Views 

**Authors**: Morita Tarvirdians, Reeve Lorena, Catharine Oertel  

**Link**: [PDF](https://arxiv.org/pdf/2610.31459)  

**Abstract**: Systems that aim to help a user engage with viewpoints other than their own have to decide how different those viewpoints should be. Viewpoints are mainly represented on a scalar scale, too coarse to distinguish viewpoints that disagree with the user about different things. We introduce a selection policy in which a bandit sets how far viewpoints may sit from the user, and the policy selects the most novel viewpoint within that bound. We vary only its stance encoder, comparing a scalar score with Perspectivized Stance Vectors, which break a position into separate stances on concepts. We integrate it inside a dialog system and run a between-subjects study (N=63). Concept-level stance produces more varied exposure sequences and broader coverage of the topic's positions, but does not change how deeply users justify viewpoints or how accurately they reconstruct them. Finally, we analyze user--viewpoint relations at the concept level, showing that users reconstruct a viewpoint least accurately on concepts that one side took no position on, rather than on concepts where the two disagreed. Our findings open up opportunities for dialog systems that support engagement with alternative viewpoints by acting on which concepts a user and a viewpoint disagree on, rather than on distance alone. 

---
# Multi-Token Prediction and Alternative Tokenization Improve Learning Latent Multi-Hop Graph Search at Scale 

**Authors**: Arvid Frydenlund, FuTe Wong, Katherine Lambert, Frank Rudzicz  

**Link**: [PDF](https://arxiv.org/pdf/2610.31439)  

**Abstract**: Given an unseen graph, a start node, $s$, and an end node, $t$, can a language model generate the shortest path from $s$ to $t$? Prior work has found that models fail to learn this task as the graphs scale. We show that this failure has two causes: the supervision signal and the tokenization. First, we theoretically establish and empirically demonstrate the effectiveness of multi-token prediction (MTP), and show that the benefits of MTP are topologically agnostic. Second, we introduce alternative tokenizations and show that the two causes are orthogonal. Using these methods, we break through the scaling wall observed by prior work, consistently and stably learning to solve the task at scales far beyond previous limits. 

---
# SQuAT: Spherical Lattice Vector Quantization-Aware Training 

**Authors**: Sebastian Loeschcke, Ivan Koryakovskiy, Tycho F. A. van der Ouderaa, Riccardo Del Chiaro, Mart van Baalen, Paul N. Whatmough, Markus Nagel  

**Link**: [PDF](https://arxiv.org/pdf/2610.31412)  

**Abstract**: Vector quantization (VQ), in particular lattice-based methods, offer strong rate-distortion trade-offs for low-bit large language models (LLMs), yet their behavior under quantization-aware training (VQ-QAT) remains poorly understood. We scale lattice VQ-QAT to billion-parameter LLMs and multi-billion-token training regimes, and introduce SQuAT, a shape-gain VQ-QAT approach trained with a separate straight-through estimator (STE) for the shape and gain. SQuAT decomposes weights into direction and magnitude: normalized directions are quantized using an $E_8$-derived spherical codebook, while gains are quantized independently with low-bit scalar quantizers. We show that, unlike a standard STE applied to the full quantizer, which ignores the shape-gain structure, these separate STEs induce tangent-space updates for directions, yielding an implicit form of spherical (Riemannian) optimization. To enable scalability, we develop a distributed exact nearest-neighbor assignment scheme, in which workers quantize disjoint weight blocks and communicate only discrete indices, reducing VQ-QAT training time by 12.8$\times$ and allowing us to scale it to 10B tokens. Across instruction-tuned LLaMA-3.2-1B/3B, Qwen3-4B, and base Qwen3-1.7B, SQuAT consistently outperforms standard VQ-QAT STE under identical quantizers and initialization, and achieves SOTA VQ-QAT accuracy. 

---
# A Task-Centric Survey of Long-Document Visual Question Answering: Taxonomy, Evolution, and Future Directions 

**Authors**: Yufei Tong, Guanjie Cheng, Tianle Wu, Leilei Zhang, Hang Zheng, Peihan Wu, Jianwei Yin, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.31385)  

**Abstract**: Visual document understanding is essential for acquiring knowledge from visually rich documents and is attracting growing research interest. Existing surveys emphasize architectures or technical approaches, with limited attention to how task bottlenecks shape technical choices and research priorities. We present a task-centric survey that decomposes long-document visual question answering (VQA) into four components: document parsing, evidence organization and retrieval, generator organization and learning, and evaluation. For each, we develop a fine-grained taxonomy and trace motivations, technical responses, and cross-component connections. We focus on explicit retrieval while covering direct reading. Our temporal analysis traces expanding priorities from multimodal content understanding to sufficient, high-fidelity evidence acquisition and sustained evidence management. We outline future directions in evidence sufficiency, verifiable and correctable understanding, and budget-aware coordination, with an outlook on continual document understanding. 

---
# AfterDeleteBench: Auditing Post-Deletion Information Persistence in LLM Agents 

**Authors**: Ehsan Saleh, Roozbeh Nahavandi, Sama Salarian, Raef Bassily, Andrew Perrault  

**Link**: [PDF](https://arxiv.org/pdf/2610.31366)  

**Abstract**: Many LLM agents maintain external persistent memory and derive additional state from past interactions, such as summaries. Deleting an original memory record therefore need not remove information carried by its descendants. We introduce AfterDeleteBench, a controlled behavioral audit for this failure mode in fixed-weight LLM agents. For each synthetic secret, we compare an exposed-then-deleted agent with an otherwise matched agent that was never exposed to the secret, allowing post-deletion recovery to be attributed to the original exposure. In our primary setting, a single query recovers the deleted secret from Qwen2.5-7B-Instruct in roughly two thirds of cases, while matched never-exposed controls have zero recovery. Crucially, recovery does not require a surviving verbatim copy: when the complete secret string is absent and only separately stored components remain under a public composition rule, exact reconstruction succeeds in 62-99% of cases across Qwen, Mistral, and Phi-3. We also show that the response interface is part of the measurement system: target-bearing replies can be scored as failures when the requested format and parser are misaligned. Finally, we separate reconstruction from write-back and fresh-session recall. Provenance-aware closure eliminates observed recovery in the tested positive-baseline settings, whereas an exact-literal write filter blocks reinsertion but cannot erase information already present. Together, these results show that source deletion should be evaluated behaviorally at the level of the agent and its memory system, not only as a storage operation. 

---
# The Value of Memory in Language Agents 

**Authors**: Zehao Lin, Erenkai Zhu, Helan Hu, Xixuan Hao, Chunyu Li, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2610.31303)  

**Abstract**: Memory enables language agents to carry useful distinctions from past experience into future decisions. Its value depends on which distinctions survive, whether a bounded agent can use them, and what can be recovered or revised as demands evolve. We develop a decision-theoretic framework that follows this value from raw history through persistent state, operational use, and costly recovery and revision. Original and retained value are expressed in units of avoided task loss; risk-cost frontiers describe attainable recovery and preservation across uses. Unknown demand yields value profiles, while interactive feedback changes the distribution of future work. A cost-aware simulation condition and five minimal counterexamples expose the roles of information timing, retrieval access, component interaction, feedback and demand uncertainty. In particular, a pointer with zero mutual information about the answer can still reduce the cost of exact recovery. Diagnostics on LoCoMo, a two-project update task and longitudinal cases exhibit failures of preservation, use and measurement. At fixed retrieved evidence, a saved ledger changes subsequent answers; at fixed agent outputs, an evaluator revision changes scores. The framework connects these distinctions to operational risk bounds and testable principles for allocating memory, preserving recovery options and revising evidence dependencies. 

---
# Predicting Music Valence and Arousal via Textual and Multimodal In-Context Learning in Large Language Models 

**Authors**: Chi-En Dai, Chih-Pin Tan, Yi-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.31202)  

**Abstract**: While arousal can be reliably predicted from musical audio, emotional valence remains difficult to model due to its subjective and context-dependent nature. In this work, we investigate the potential of in-context learning (ICL) with large language models to advance continuous valence and arousal prediction. Rather than predicting emotion scores directly, our framework leverages contextual demonstrations from an auxiliary dataset to refine the initial predictions produced by a small, task-specific base model. We systematically evaluate a text-based ICL variant using lyrics and generated audio captions, alongside a multimodal variant that incorporates raw audio signals. Using the entire MER60 dataset strictly as a held-out test set, our experiments demonstrate substantial performance gains, particularly for valence. Text-based ICL improves the valence $R^2$ from 0.110 to 0.427 with 80 demonstration shots, while the multimodal variant achieves 0.346 with only 10 shots. These findings suggest that training-free ICL offers an effective inference-time strategy for current language models to predict continuous music emotion. 

---
# Beyond Watching and Searching: Benchmarking Agents for Video-centered Deep Research Reports 

**Authors**: Jiahao Meng, Qize Yang, Yuxuan Wang, Yunfei Chu, Wensi Dai, Jin Xu, Yunhai Tong  

**Link**: [PDF](https://arxiv.org/pdf/2610.31158)  

**Abstract**: Deep research agents synthesize open-web evidence into long reports, but current benchmarks do not evaluate research centered on a source video. We introduce Video-Centered Deep Research Report Generation. Given a video and user request, an agent identifies research questions, retrieves external multimodal evidence, and produces an interactive report linked to relevant video moments. This task has direct practical value because it helps users locate relevant moments, verify claims, reproduce demonstrated procedures, and make informed decisions from video content. We present a skill-based workflow and OmniDR-Bench for this task. For evaluation, we propose Omni Atomic Rubric Trees (Omni-ARTs). Each tree contains three dimensions, 23 shared directions, and task-specific atomic binary criteria. The directions test audiovisual grounding, multimodal evidence use, and interactive presentation. Audited Rubric Construction (ARC) builds these trees through independent evidence collection, automated checks, behavioral calibration, and iterative human review. OmniDR-Bench contains 50 Chinese and English tasks across 19 realworld subcategories. It includes 6,414 criteria. Across 18 agent configurations, the best system scores 63.3 out of 100. Current agents still need broader content coverage, stronger causal and logical analysis across modalities, and more accurate citation support. We further analyze models, harnesses, skill configurations, video categories, and research trajectories. The code and data are released at this https URL. 

---
# Through a PRISM, Clearly: Step-Sequence Reasoning for Mistake Recognition in Procedural Videos 

**Authors**: Olga Loginova, Oleksandr Bezrukov, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2610.31117)  

**Abstract**: Procedural mistake recognition requires identifying deviations from valid multi-step activities. We study post-completion recognition from videos with benchmark-provided step boundaries. We introduce PRISM (Procedural mistake Recognition via Interpretable Semantic Modeling), which grounds video segments in language and maps their descriptions to procedural states using a lightweight hidden Markov model trained on correct executions. Reference alignment and semantic checks identify errors in step content and order. To support evaluation across heterogeneous benchmarks, we map dataset-specific annotations to four edit-based error types: insertion, deletion, substitution, and transposition. Across three benchmarks, we compare gold and VLM-generated descriptions in raw, semantic-role, and action-object forms. Decoder controls show consistent gains from the trained local decoder over nearest-candidate matching, with additional, dataset-dependent benefits from sequence decoding. Semantic role labeling improves Any-F1 in all six matched grounded comparisons, although the gains vary across datasets and do not translate into uniform improvements under separate configuration selection. These results establish language grounding as an effective interface for explicit procedural mistake reasoning and characterize the contributions of representation and decoding. 

---
# SCOPE: Compiling Natural-Language Policies into Neurosymbolic Predicate Programs 

**Authors**: Arnon Mazza, Elad Levi  

**Link**: [PDF](https://arxiv.org/pdf/2610.31062)  

**Abstract**: Policy enforcement for LLM agents has shifted from single-turn content moderation to compound, application-specific policies evaluated over multi-turn transcripts and tool-call trajectories. Fine-tuned guardrail models are efficient but degrade on compound policies, and while LLM-as-a-judge approaches achieve higher accuracy, they incur high inference cost and tend to produce unstable predictions. Neurosymbolic methods offer a natural alternative, decomposing the policy into narrow neural decisions composed under explicit logic. However, existing symbolic layers lack the required expressiveness and are optimized against fixed data rather than the policy itself. To address this, we introduce SCOPE, which compiles a policy into an executable program of neural predicates executed by small models, deterministic extractors, and code that composes their outputs. The program is optimized jointly with its training data, using adversarial generation of counterexamples that target where the program deviates from the policy. We evaluate SCOPE on a benchmark of real-world policy tasks against prompt optimization, task decomposition, and workflow generation methods, achieving the highest average accuracy across tasks and models. Programs optimized with the weakest executor transfer to other models without further optimization. This allows them to be deployed in production on inexpensive models, where they reach accuracy comparable to frontier reasoning models. Unlike these monolithic models, SCOPE produces substantially more stable predictions, and each verdict is interpretable through the predicate outputs that produced it. Additionally, we validate the key components of our method through an ablation study. SCOPE thus enables reliable, interpretable, and efficient policy enforcement at production scale. 

---
# ColaBench: Evaluating Memory-Guided Knowledge Acquisition in Personal Agents 

**Authors**: Wenjie Zhang, Zhichao Shi, Xuhui Jiang, Xiaojun Wu, Cehao Yang, Chengjin Xu, Xiaoning Song  

**Link**: [PDF](https://arxiv.org/pdf/2610.30989)  

**Abstract**: Personal agents need to draw on long-term interaction history to retrieve and apply external knowledge suited to a user's current request. Even an answer grounded in relevant documents can be unsuitable when it overlooks the user's circumstances. This creates two connected challenges: using historical conditions to identify the knowledge needed for a new request, and applying that knowledge to the user's situation. Existing benchmarks advance memory recall, personalized retrieval, and memory-guided action, while the joint evaluation of these two challenges remains underexplored. We introduce ColaBench, a synthetic benchmark in health and daily living, with 349 tasks across 10 Profiles and a knowledge base of 13,033 documents. Its Evidence Graph connects personal conditions to their disclosures across public interaction sessions, records prior document exposure, and links answer requirements to personal and newly acquired document evidence. We also introduce MemKB, which uses state summaries to inform queries and rule application, and memory-document links to retrieve knowledge associated with the user's history. MemKB achieves 86.25% Accuracy with an average of 4.07 interaction rounds per task. Task-level analysis finds larger gains on tasks integrating information across multiple historical Facts; error analysis further identifies unsupported personal assumptions and rule misapplication as obstacles to using retrieved evidence. ColaBench provides traceable tasks and an evaluation framework for studying how memory guides knowledge acquisition and rule application. 

---
# Tracing Hallucinations in LLM Reasoning for UML Class Diagram Generation: An Empirical Taxonomy and Mitigation Study 

**Authors**: Mojdeh Rahmanian, Foutse Khomh, Ashkan Sami, Yanchao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30946)  

**Abstract**: Large Language Models (LLMs) can generate UML class diagrams from natural language requirements, but hallucinations may compromise the correctness and completeness of the resulting diagrams. Existing evaluations mostly assess the final diagram, without examining the intermediate traces of reasoning-capable LLMs, which can reveal where these hallucinations arise. We develop an empirical taxonomy of eleven hallucination sub-types organised into three dimensions: Prompt Faithfulness (deviations between requirements and reasoning), Code Faithfulness (discrepancies between reasoning and generated PlantUML code), and Reasoning Quality (problems within the reasoning). The taxonomy was derived from 60 manually annotated DeepSeek-R1 traces, applied to 120 further DeepSeek-R1 traces through a dual-LLM annotation pipeline, and examined on 20 DeepSeek-V4-Pro traces. Code Faithfulness was the most frequent dimension for DeepSeek-R1 in both datasets, whereas Prompt Faithfulness dominated for DeepSeek-V4-Pro, suggesting model-dependent failure patterns. In the manually annotated traces, reasoning-trace length showed no significant monotonic association with hallucination frequency, whereas requirement complexity was negatively associated with it, particularly for Prompt Faithfulness errors. In a mitigation experiment on 13 Golden UML ModelSet cases with three Qwen3-based models, adding Prompt Faithfulness and Reasoning Quality constraints to the generation prompt raised the pooled mean F1 score by 4.0 percentage points, whereas a post-hoc Code Faithfulness review provided no consistent further benefit. Relationship modelling remained the lowest-scoring element throughout. These results suggest that localising hallucinations within the generation process can inform targeted, prompt-based mitigation. Whether these findings generalise beyond the DeepSeek and Qwen3 model families remains to be tested. 

---
# Hierarchical Fine-Grained Cross-Modal Alignment via Dynamic Granularity Discovery and Optimal Transport 

**Authors**: Pengfeng Song, Shaozhong Zhang, Pengfei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2610.30912)  

**Abstract**: Fine-grained image-text retrieval requires a model not only to assess global semantic consistency between an image and a text, but also to model visual evidence for local textual semantics, including objects, attributes, quantities, actions, and relations, so as to distinguish semantically similar candidates. Existing methods commonly use fixed image patches, detected regions, or text tokens as the basic matching units and then aggregate local interactions through max pooling, cross attention, or similarity accumulation. Such strategies remain limited in semantic unit construction and global matching consistency. This paper proposes a hierarchical fine-grained cross-modal alignment framework, termed HFCMA. The framework progressively transforms low-level patch and token representations into adaptive visual regions and textual phrases, and then performs structured region-phrase matching within a global retrieval scoring process. Specifically, HFCMA first adopts dynamic granularity discovery (DGD) to aggregate raw patches and tokens into semantically compact visual regions and textual phrases. It then formulates region-phrase matching as an entropy-regularized optimal transport problem, thereby estimating a structured alignment matrix under global marginal constraints. To reduce background noise and local semantic drift, HFCMA further introduces background debiasing, textual importance weighting, and prototype consistency regularization. Experiments on Flickr30K and MS-COCO show that HFCMA effectively combines dynamic semantic unit construction with OT-based structured matching under a dual-encoder retrieval framework. The resulting fine-grained score provides structured local alignment information for candidate ranking and improves the discrimination of subtle semantic differences. 

---
# Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models 

**Authors**: Zhizhuo Kou, Sizai Hou, Long Chen, Jian Yang, Junyu Luo, Jingshu Peng, Wenkai Zhang, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2610.30882)  

**Abstract**: Reinforcement learning (RL) is credited with teaching reasoning models to write {wait} before reconsidering a step, yet large RL accuracy gains do not require the word. This view can confuse what RL discovers with what a model inherits from its training data. We trace the word through open models' training histories, reading it with label-free indices of frequency, position, form and usage and an injected-error test, all computed from sampled outputs. We find that zero-RL almost never coins the marker: on bases that lack the word, RL raises MATH-500 accuracy substantially while {wait} stays near absent. It arrives with training data: mid-training in OLMo-3, and fine-tuning on reasoning traces installs it in most models whose base lacks it. Where the base already uses it, both released RL-Zero recipes (OLMo-3, MiMo) multiply it, whereas KL-free GRPO from the same base does not; OLMo-3's recipe wears down its form and makes it a more sensitive but more liberal alarm; RL after supervised fine-tuning barely changes it, and our GRPO recipe erases a marker seeded by fine-tuning. Finally, we show that marker frequency alone does not indicate RL-induced reasoning and caution against LLM-as-a-judge revision metrics, as automated judges can reverse human-labelled trends. 

---
# One Teacher, Many Views: Reducing Reference Sensitivity in On-Policy Self-Distillation 

**Authors**: Khouloud Saadi, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30880)  

**Abstract**: On-policy self-distillation (OPSD) improves reasoning by training a model on its own trajectories with dense token-level supervision from a privileged copy conditioned on a reference solution. Existing methods typically use a single teacher distribution conditioned on the complete reference. We identify a limitation of this formulation: a reference is only one realization of the reasoning process. Although it provides evidence about the answer, it also encodes an ordering, phrasing, and solution path. Full-reference conditioning can therefore produce correct targets that depend excessively on information unavailable to the student at inference. We show that replacing one valid reference with another, while holding the problem and student state fixed, can substantially alter the teacher distribution, a phenomenon we call reference-induced target instability. To address this instability, we introduce Monte Carlo Privileged Distillation (MCPD), which models privileged context as a distribution over partial reference views and distills the marginal teacher distribution. MCPD evaluates the teacher under multiple views and averages the resulting next-token distributions. We show that this probability-space average is the exact target induced by minimizing the expected teacher-to-student KL divergence over privileged views. Across mathematical and scientific reasoning benchmarks, we show that MCPD produces more reference-consistent supervision and consistently improves over full-reference OPSD. 

---
# MixJudge: Learning a Language-Queryable Critic for Vocal Mixing 

**Authors**: Kuan-Hsun Ho, Yuan-Chiao Cheng, Yen-Tung Yeh, Yi-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30783)  

**Abstract**: Mixing requires not only applying processing but also assessing the resulting mix. Audio-language models can provide conversa- tional mixing advice, but existing evaluations do not separately test whether they can judge specific mixing issues. Since professional supervision for such judgments is scarce, we construct a curriculum of 11 vocal-mix issues at multiple severity levels and use it to train MixJudge, a language-queryable audio critic. MixJudge shows its ability to retrieve mixing issues, distinguish opposite issues, and order the severity of an issue. Moreover, its judgments transfer to menu selection: given a set of processing candidates, MixJudge often favors versions acoustically closer to a held-out professional version, demonstrating how explicit mixing judgment can support comparison and downstream mixing decisions. 

---
# Abstract Priorities, Concrete Rewards: How Stated Preferences Relate to What People Reward in AI Responses 

**Authors**: Zihan Chen, Lei Nico Zheng, Di Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30763)  

**Abstract**: AI assistants built on large language models (LLMs) are trained on human feedback and increasingly personalized to users' stated preferences. Yet how closely these preferences reflect what people reward in responses remains unclear. We compare stated and revealed preferences in the public PRISM dataset, in which 1,396 participants reported how much seven response attributes mattered to them and then rated 29,347 responses from 21 LLMs. Among responses to the same prompt, features the survey never asked about, such as response length, formatting and refusal language, were at least as strongly associated with ratings as text proxies of the surveyed attributes. Whether stated priorities matched ratings depended on how attributes were measured: participants who prioritized an attribute favoured responses from models that others perceived as strong on it, but not responses that scored higher on text proxies for it. Written feedback was more telling: participants who asked for shorter answers, even in earlier conversations, gave length less weight in their ratings, and those asking for longer answers gave it more. Their relative priorities over the surveyed attributes, however, did not predict this weight. Reasons given for choices reflected perceived attributes more than measured features. Participants also differed stably in how much they weighted several concrete features, especially length, but with a median of 6 conversations per participant, these weights were estimated poorly. The largest gain from personal information in predicting held-out choices was 0.13 percentage points, compared with 1.11 for a stronger pooled model. Stated priorities thus track perceived attributes of AI responses but say little about the concrete features that ratings reward, a distinction that matters for preference elicitation and AI personalization. 

---
# How Firm Are AI Models' Ethical Boundaries? A Pay-Stub Editing Test of 15 Models 

**Authors**: Simiao Ren, Xingyu Shen, Tommy Duong, Tsang Ng, Ankit Raj  

**Link**: [PDF](https://arxiv.org/pdf/2610.30751)  

**Abstract**: Can a small change in wording move an AI model from refusing a sensitive request to acting on it? We test 15 models on the same request to increase the income shown on a pay stub. Across 445 observable trials, models issue editing calls in 14.1% of cases when a loan purpose is stated, 33.1% when the purpose is omitted, and 66.2% when the change is described as fixing a printing error. Astra refuses all ten loan-purpose requests but issues calls in all ten purpose-omitted trials. Fable, Opus, and Muse Spark issue none across all three wordings; Sonnet issues some under the latter two. Response excerpts illustrate different ways of refusing, redirecting, or justifying an edit. The file is a marked test document and all tools are simulated. These results probe how boundaries behave in one setting; they do not prove an overall ethics ranking or that a harmful real-world edit succeeded. 

---
# Disentangling Robustness and Sycophancy in Multilingual LLMs 

**Authors**: Rhitabrat Pokharel, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2610.30696)  

**Abstract**: When a multilingual LLM agrees with a user's false premise, is it being sycophantic, or does it simply not know better in that language? Current research does not distinguish these cases: they record only whether the model agreed, not why. We introduce the SC-matrix, that crosses a model's competence (whether it answers correctly under no pressure) with its sycophancy (whether it maintains that answer under pressure). Our results show that (1) identical aggregate sycophancy rates can mask different failure compositions, (2) high competence in a language does not always protect against sycophancy, and (3) reasoning-oriented mitigations outperform directive-based strategies. We find that multilingual sycophancy is not a single phenomenon but a composite of different failures that require distinct diagnosis. 

---
# Scaling Laws for Safety Probes 

**Authors**: Yuchen Wu, Jerry Wei  

**Link**: [PDF](https://arxiv.org/pdf/2610.30678)  

**Abstract**: Safety probes trained on a model's activations are a standard tool for monitoring language models in real time. Each time a new model is released or a new harm area needs to be monitored, the probe must be retrained, which requires new labels and a choice of probe architecture. Yet neither the model nor the labels tell us in advance how well the resulting probe will perform. Finding out requires training and comparing many probe configurations. In this paper, we ask whether probe performance can instead be predicted in advance from model size and the number of labels. Existing safety datasets are insufficient for this, as they are either limited in scale or already saturated. We therefore construct six new safety datasets of sufficient size and difficulty. On these datasets, we train two probe architectures on 20 base models from five model families, with 32 to $10^{6}$ labels for each model. We measure performance using probe error, defined as 1 - AUROC, where lower values are better. We find that probe error follows two simple scaling laws. First, the lowest probe error a model can reach falls as a power law in its size. Second, probe error decreases towards its floor as the number of examples increases; the slope increases as the target model size increases. Both laws hold across tasks, model families and probe architectures, and only their fitted constants differ. When fitted on the smallest models of a family, the laws predict the error of the next larger model to within 0.02 in 72% of cases. They also identify whether more labels or a larger model is the cheaper way to reach a target error in 73% of cases. Finally, they choose the probe architecture with an extra error of only 0.006 on average when they pick the worse one. These laws allow safety researchers to determine how to create the best probe for any given model and label budget. 

---
# Group Relative Policy Optimisation Improves Multilingual Speech Recognition in Low-Resource Languages 

**Authors**: Tim Wenjie Hu, Evelyn Nafula Ouma, Benjamin Akera, John A. Quinn  

**Link**: [PDF](https://arxiv.org/pdf/2610.30663)  

**Abstract**: Autoregressive encoder-decoder speech recognition models trained with teacher forcing can suffer from exposure bias, limiting their ability to recover from their own prediction errors at inference time. Recent work has shown that reinforcement learning with Group Relative Policy Optimisation (GRPO) improves speech recognition in major languages with LLM-based models. We extend this line of work to smaller, non-LLM encoder-decoder ASR models, using Whisper-large-v3 as a case study across 51 African languages. The benefits of this approach are most pronounced when training data is scarce or when input audio is noisy or degraded. Our resulting model, SunflowerASR, achieves the best published macro-averaged WER on the AfriVox-v2 (0.227) and SimbaBench (0.372) benchmarks, and is publicly available. 

---
# The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts 

**Authors**: Aaditya Chauhan, Arun Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2610.30612)  

**Abstract**: Hybrid retrieval, which fuses a sparse lexical retriever with a dense neural one, is widely recommended in the software documentation surveyed in September 2026, usually with reciprocal rank fusion (RRF), on the strength of aggregate gains. We ask what fusion costs on individual queries, treating "do not fuse" as a first-class outcome, in the setting in which hybrid retrieval is deployed: a standard Anserini BM25, five dense encoders spanning 22M-596M parameters and three model generations, a learned-sparse (SPLADE) alternative, and a cross-encoder second stage, on seven BEIR datasets and three fusion functions. Even the best method, dense-weighted score interpolation, ranks relevant documents below the better of its two components on 13-48% of queries and below the deployable dense-only retriever on 8-21% (14-48% and 11-25% at a weight selected under leave-one-corpus-out validation), while beating dense-only in aggregate on every corpus with the primary encoder; the downside is significant under loss-averse risk-sensitive evaluation on six of the seven corpora. We call this loss the fusion tax. It is mostly a ranking effect that preserves recall in aggregate, it is most likely on the queries that only one retriever covers, and it persists across the encoder ladder and the SPLADE substitution. RRF carries the higher tax wherever the dense retriever is clearly stronger and only beats dense-only where the two components are balanced. Fusion failure is weakly predictable within a corpus but transfers unreliably across corpora, because the features that predict it change sign with the sparse-dense balance. A cross-encoder recovers part of the tax only where it is itself reliable. We distill these findings into guidance that keys the fusion decision to the measured sparse-dense balance of the target corpus. 

---
# TT-SAX: A Two-Track Time Series Symbolization for Natural Language Shape Querying 

**Authors**: Zhao Tan, Xianglong Bao, Zhipeng Xi, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2610.30593)  

**Abstract**: Natural language querying for time series databases aims to let users find what they need in long histories of data by asking in plain language. Shape queries ask how a series moves. They are the main challenge of this task. Recent work makes them feasible by using symbolization as a bridge between the query and the long history. However, existing symbolizations were designed for compression, similarity indexing, or classification. What a symbolization should provide for natural language querying has not been systematically studied. We therefore examine the definitions of six classical symbolizations against this task and identify three limitations. (1) Direction is missing from the string, (2) direction is measured at too small a scale, and (3) the letters are arbitrary. Three design principles follow. We follow these principles and make a minimal redesign of SAX called TT-SAX (two-track SAX). The level letters stay. Each segment gains a trend letter. It is measured across neighboring segments rather than inside one, and graded to match the adjectives that queries use. We conduct extensive experiments on the public NLQTSBench benchmark, comparing TT-SAX with the six classical symbolizations across six LLMs. TT-SAX outperforms all six on shape queries, and ablations trace its gains back to the three limitations one by one. This paper is the first systematic study of time series symbolization with natural language querying as the design goal. It provides a foundation for future research on this task. Our code is available at this https URL. 

---
# Governed Self-Evolution for Autonomous Investing: A Multi-Agent LLM System with an Invariant Core 

**Authors**: Yebo Feng, Jingyang Dai, Liangrui Pan, Nelson Lau, Jiahua Xu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2610.30557)  

**Abstract**: LLM agents that act on live Web data must combine heterogeneous evidence, remember what they have done and improve from outcomes without weakening their own safeguards. We study this problem in autonomous investing, where every action is irreversible and based on untrusted Web content. We present Steward, a multi-agent LLM system that draws the boundary of self-evolution in its type system. First, nine role-specialised agents decide jointly: four analysts read news feeds, SEC filings, macroeconomic series and market data and emit typed reports, from which a Portfolio Manager sets target positions that a protected Risk Manager audits. Second, a seven-layer shared memory, assembled by code from the system's own record, gives each decision its position context, the market narrative, calibration and the agent's past decisions. Third, the system improves itself by review: code grades every trade and forecast nightly and writes the grades into memory, and each quarter a reflector proposes prompt learnings that a deterministic gate applies reversibly and replay of recorded inputs verifies. All three operate around an invariant core of order arithmetic, risk rules and discipline-agent prompts that no self-edit can reach. In a five-month live deployment (102 trading days, 1,632 LLM calls), the system contained a 24.6% sector drawdown to a 0.3% loss; in replay of its recorded inputs, a single model given the same raw data traded more and lost 1.35 points against the market over five days where the joint design showed no significant loss, and no injected instruction was followed; and a 2015-2026 backtest shows the core alone lowering the maximum drawdown under all 50 random entry policies. The reflector traced the dominant failure to its root cause, and the gate applied a learning that the agent adopted on trigger evenings and on no control evening. Code: this https URL 

---
# Beyond End-to-End Recovery: Towards a Better Understanding of LLM Steganography via Rolewise Decomposition 

**Authors**: Om Tailor, Yucheng Pan, Solaleh Mohammadi, Xinyan Liu, Kaiqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30516)  

**Abstract**: In large language model (LLM) steganography, an encoder hides a private message in an otherwise ordinary response, as text transmitted to a decoder, who attempts to recover it. The success of this private-message transmission is commonly measured through end-to-end (E2E) recovery, which checks whether the decoder correctly reconstructs the private message. However, E2E recovery cannot distinguish encoding failures from decoding failures or reveal how the private message is represented in the encoder's transmitted text. To address these limitations, we introduce a rolewise decomposition framework that separately examines how private messages are represented in encoder transmitted texts, what decoders recover, and what deterministic verification can establish. Across 7,200 encoder responses, we identify 168 verifiable encodings missed by decoders and 706 responses containing private messages reconstructed through word-length or token-initial patterns not explicitly specified in the encoder's instructions. These findings demonstrate that E2E recovery alone may obscure important distinctions between encoding and decoding, motivating their separate evaluation to better understand and mitigate the safety issues caused by LLM steganography. 

---
# Why Bother Me Again? Benchmarking Personal Agent Memory for Interaction Preferences 

**Authors**: Jialin Li, Zhenhao Chen, Hanjun Luo, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2610.30485)  

**Abstract**: Personal intelligence systems are taking on increasingly complex everyday tasks, yet repeated questions, unnecessary confirmations, and unwanted outreach can turn capable assistance into interaction burden. Reducing this burden requires agents to remember how users want to be assisted. However, existing memory benchmarks focus on factual recall and personalized answers and rarely test how agents learn and act on interaction preferences. We introduce MemPABench, an interactive environment for on-policy evaluation of memory in personal agents. MemPABench poses long-horizon tasks requiring agents to decide when to clarify, confirm, or act autonomously. Four grounded simulated users hold 14 preferences across work and personal contexts. Their feedback spans 99 sessions over four months, with five event-driven changes per user testing memory updates. We evaluate preference selection, delivered behavior, and sensitivity to context and preference change. Evaluating four memory systems across four LLM backbones, we find that current memory brings limited gains in preference selection. Retrieval-based systems (i.e. Mem0, MemOS, simple RAG) do not outperform No Memory, while File-based memory improves accuracy but applies preferences beyond relevant contexts. Our analysis shows that current memory struggles to retrieve interaction preferences and adapt to preference changes. Even with the correct preference available, overriding the agent's default interaction habits remains challenging. The environment and dataset are available at this https URL. 

---
# Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy 

**Authors**: Riteng Zhang, Xiaoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.30474)  

**Abstract**: Sparse autoencoders (SAEs) were originally motivated by discovering monosemantic features, a property that characterizes individual features but does not specify what the learned feature set should look like as a whole. As SAEs have increasingly been adopted for attribution, circuit discovery, and intervention, researchers have begun evaluating metrics that describe global properties of the feature set, such as sparsity, dead feature rates, reconstruction fidelity, and disentanglement, a diverse set that extends far beyond monosemanticity. However, this implicit shift beyond monosemanticity has not been formalized as a clear, well-defined objective that comes before these evaluation metrics. Without such an objective, existing metrics remain ad hoc, with their necessity unjustified. In this work, we formalize this ultimate objective by introducing a new paradigm for SAE evaluation: requiring that SAE features, and more generally dictionary-based decompositions of model representations, form an atomic interpretable unit basis, a coordinate system for semantically meaningful explanation. Under this paradigm, we organize existing and new measurements such as DAME (Dominance-Aware Monosemanticity Evaluation) as proxy metrics and introduce a comprehensive benchmarking framework. Evaluating hundreds of SAEs across multiple architectures and model families, we uncover failure modes in which some SAEs appear monosemantic under existing evaluation metrics while remaining functionally polysemantic: the most dominant features, those contributing most to model computation, are systematically less interpretable and more redundant. These failure modes warrant attention from SAE researchers as targets for improved training and architectures, and motivate dominance-aware evaluation as a standard check for practitioners before SAE features are deployed for attribution, circuit tracing, or intervention. 

---
# Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems 

**Authors**: Daniel Ruiz, Luca Luceri, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2610.30418)  

**Abstract**: Modern LLM agents can retrieve information from accessible documents, and then disclose a sensitive conclusion assembled from them. This mosaic effect creates a critical risk for agents operating over confidential data. We quantify its realization with 30 synthetic cybersecurity, medical, and military examples, yielding 10,800 final answers from three LLM agents in controlled simulation and 1,440 answers from live agents in two open-source harnesses. Even under benign queries designed to be answerable from one document, GPT-5.6-Sol, the best performing model tested, discloses forbidden composites in 7.08% of answers (95% CI: 3.75-10.67%) in simulation, with worse disclosure rates in live harnesses. Adding visible classification headers and explicit aggregation rules reduces pooled disclosure rates substantially, yet unsafe disclosures persist in 32.37% of simulated answers (95% CI: 28.33-36.19%) across all intents despite these safeguards. Furthermore, increased refusals do not guarantee confidentiality, since significant refusals reveal the prohibited conclusion regardless. These results point to a troubling conclusion: sensitive data leaks via mosaic-style events are an observed confidentiality failure mode in the evaluated settings at non-negligible rates. They motivate stronger controls on what agents may infer and disclose in sensitive data environments with mixed-clearance users. 

---
# X-pressing Intimacy: Mental Health Disclosure After Twitter Acquisition 

**Authors**: Olivia Pal, Yunhao Yuan, Meghana Mandava, Talayeh Aledavood, Eshwar Chandrasekharan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2610.30413)  

**Abstract**: Social media platforms provide spaces for individuals to disclose mental health experiences and seek support, yet how these disclosures evolve during major shifts in platform governance remains unclear. We examine mental health self-disclosure surrounding Elon Musk's October 2022 acquisition of Twitter, analyzing approximately 824K tweets across anxiety, depression, bipolar disorder (BD), borderline personality disorder (BPD), and suicidal ideation (SI) or related experiences, alongside 266K comparison tweets. Using Linguistic Inquiry and Word Count (LIWC) and an LLM-based intimacy classifier, we characterize changes in psycholinguistic expression and disclosure intimacy. Following the acquisition, we observe declines in affective and social language, alongside increases in first-person singular pronouns and future-focused expressions. Intimacy patterns vary by condition: anxiety, depression, and BPD show increases in low-intimacy and decreases in moderate-intimacy disclosures, whereas suicide-related posts show more high-intimacy disclosures. However, compared with broader Twitter discourse, only the relative decline in moderate- versus low-intimacy disclosure for BPD remains statistically significant. Our findings underscore the importance of considering both language and intimacy when studying platform transitions, informing the design of online environments that support sensitive self-disclosure. 

---
# Text-Only LLM Correctors Break Correct Transcripts: A Repair-Damage Analysis of Audio Grounding in ASR Error Correction 

**Authors**: Vasiliy Kudryavtsev, Kirill Borodin, Nikita Vasiliev, Mikhail Gorodnichev, Grach Mkrtchian  

**Link**: [PDF](https://arxiv.org/pdf/2610.30385)  

**Abstract**: LLM correctors of ASR output are judged by aggregate WER, which hides what they fix and what they break. We decompose every WER change exactly into repaired tokens, damaged tokens and insertions, and use this to ask what the source audio adds to a corrector. We compare correction with and without the audio, for the model's own draft and for the output of an independent recogniser, on three corpora. With Gemini 2.5 Flash, text-only correction of its own draft breaks 13-32% of already-correct transcripts; with the audio, 1-3%. Damage falls in both settings. Repair falls against the own draft and rises against the recogniser. On medical dictation, kept insertions cancel the gain in net WER. The open-weight Voxtral, which rarely edits, does not show this pattern. Prompts differ between the two arms, so the contrasts bound the effect of the audio and do not isolate it. 

---
# Probing and Steering for Safety-Relevant Behaviours in Large Language Models 

**Authors**: Nimun Bajwa, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2610.30363)  

**Abstract**: Activation probes are computed directions in a language model's residual stream, used both to detect safety-relevant behaviours and, by adding their direction back into the stream, to steer them. We explore how the main design choices behind such probes interact, within one pipeline, on one model (Llama-3.1-8B-Instruct), and across three behaviours: deception, high-stakes awareness and sycophancy. The pipeline crosses two estimators (difference-of-means and logistic regression) with three token-pooling schemes (last, mean and learned attention), selects appropriate layers to detect or steer a behaviour, and evaluates every direction twice: as a detector of the behaviour, and as a steering vector to push outputs towards the behaviour. Prior work read together suggests a read/write dissociation, in which logistic regression reads (detects) best but difference-of-means directions write (steer) best. We find that it holds for deception and high-stakes but not for sycophancy, where the pattern reverses. The reversal appears to follow partially from the data rather than from the estimators. The best pooling technique and the layer choice depend on the behaviour and data, and the layer that steers best can sit well before the layer that detects best. The right steering metric to use between changing a model's perception and eliciting behaviour depends on the task - and the two can disagree outright. The exported probes and directions power an interactive interface through which a user can score and steer text directly. 

---
# Protection Is Not One Event: A Survey of RAG Protection Research by Measured Endpoint 

**Authors**: Peiyang Liu, Xi Wang, Di Liang, Youru Li, Zhi Cai, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2610.30193)  

**Abstract**: Retrieval-augmented generation (RAG) systems face security, privacy, and copyright risks including content leakage, source reconstruction, unauthorized access, data poisoning, and prompt injection. Yet RAG protection studies use incompatible evaluation metrics, datasets, and threat models. We present a systematic survey and human-coded evidence map of 118 full-text-confirmed RAG defense papers, organized by measured endpoint rather than claimed goal or defense mechanism. Our seven-family taxonomy covers content disclosure, formal privacy proxies, access control and suppression, corpus-use detection, and pipeline integrity and attribution. Thirty-two annotators double-coded the papers kappa=0.867. The survey identifies three gaps. First, retrieval reranking and context rewriting each span five or six endpoint families, so mechanisms do not identify protection outcomes. Second, only 17.8% of studies use content disclosure or reconstruction as their primary endpoint; under the frozen copyright/ownership relation label, 73.3% of records have corpus-use detection as their primary endpoint, and none have an emitted-expression primary endpoint. Third, most evaluations use general-purpose question-answering datasets and non-adaptive attacks rather than sensitive data and adaptive, multi-turn extraction. Our controlled RAG testbed shows that preventing verbatim copying does not prevent private-fact recovery: a normalized 8-gram guard reduces 20-token copying to 0.0%, but a three-query adaptive attacker recovers 48.3% of hidden facts on Qwen3-8B. We release the taxonomy, dataset, coding protocols, and testbed to enable comparable, claim-aligned RAG security evaluation. 

---
# Finitely Verifiable Segmentation: Decidability for Every Tokenizer with a Finite-State Verifier 

**Authors**: Nicklas Nidhögg  

**Link**: [PDF](https://arxiv.org/pdf/2610.30168)  

**Abstract**: A segmentation policy assigns to each byte string at most one segmentation into tokens. Call it finitely verifiable when a finite automaton over marked strings, bytes paired with boundary bits, accepts exactly the policy's own segmentations. The automaton judges a proposed segmentation and need not compute one. The paper shows what that hypothesis buys. Four questions that practice settles by folklore become decidable, whether a window certifies a token boundary, whether the boundary gap is bounded, whether a mask keeps generation completable, and whether two policies are equivalent. The last three specialize limitedness, coaccessibility and the equivalence of single-valued transducers. The first refines the synchronizing cut of code theory, and two more ask whether a certified cut, or the cuts of an inventory, let each piece scan alone. The class sits strictly between the online-scannable policies and those with regular domains, and its inner frontier is Choffrut's and decidable in polynomial time. Markings are length-preserving, so the class is exactly the rational segmentation policies, a normal form rather than a restriction, and what the decisions need is a supplied verifier. The class is closed under the operations pipelines use: composition over a labeled stage, restriction, splitting, and byte-fallback union of literal policies. Each composite inherits the six decisions, and certificates transfer between stages under four laws. Under rational, unifilar, almost surely stopping finite-state sources, each mass of the weighted section is an exact rational, and so is each rate where expected length is positive. Maximal-munch lexing, canonical byte-pair encoding and two-mode WordPiece instantiate the interface, and the munch library implements all six decisions. Every theorem carries an executable check, and every reported statistic is a program's own output. 

---
# When Multiple Choices Are Correct: Asymmetric Probability Allocation in Jev and Other Decision Models 

**Authors**: Rushab Munot  

**Link**: [PDF](https://arxiv.org/pdf/2610.29217)  

**Abstract**: Decision models return normalized probability distributions over candidate choices, which downstream applications can use to select among choices. Although their providers emphasize the reliability of these probabilities, calibration on tasks with a single correct choice does not establish their reliability when multiple choices are correct. We evaluate three decision models (Jev, Liquid d1, and GPT-6 Luna Decisions) on three diverse datasets (MetaTool, HotpotQA, and ECtHR) to study probability allocation and thresholding. Even when models rank every correct choice above every distractor, in 28% to 50% of those lists, at least one correct choice receives a probability below the uniform probability of 1/n, where n is the number of choices. We also present each model with five identical copies of a correct choice and observe probability disparities, with mean gaps ranging from 17 to 79 percentage points. We further show how these allocation disparities affect the recovery of multiple correct choices under probability thresholding. Decision-model probabilities can rank correct choices, but a probability threshold does not identify the set of correct choices. 

---
# ChemBART: Pre-training a Reaction Language Model for Comprehensive Synthesis Planning 

**Authors**: Kenan Li, Yijian Zhang, Jin Wang, Haipeng Gan, Zeying Sun, Xiaoguang Lei, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.02915)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated transformative potential across diverse fields. However, in computer-aided synthesis planning (CASP), current LLM methodologies are often restricted to single-molecule representations, limiting their application to isolated, single-task scenarios such as precursor prediction. Addressing this gap, we present ChemBART, a BART-based language model that adopts a reaction-centric pre-training strategy. By leveraging a mask-filling pre-training task on holistic reaction expressions (Reactant > Reagent > Product), ChemBART effectively solves a variety of chemical problems, including precursor/reagent generation, temperature-yield regression, and molecular property classification. Building on these capabilities, we demonstrate the integration of ChemBART with Monte Carlo tree search (MCTS) for multi-step synthesis route design, where the model serves to optimize the policy and value functions within a reinforcement learning framework. Beyond single-molecule pre-trained LLMs that are constrained to specific applications, ChemBART unifies diverse chemical tasks to facilitate synthesis planning with comprehensive implementation information. Most significantly, the proposed multi-step synthesis routes and reaction conditions were validated in the wet lab, confirming the discovery of shorter synthetic pathways with a >20% yield improvement over established literature benchmarks. Our work highlights the potential of reaction-focused pre-training and showcases the broad utility of ChemBART in closing the loop between in silico prediction and wet-lab synthesis. 

---
# Domain-Specific Constitutional AI: Enhancing Safety in LLM-Powered Mental Health Chatbots 

**Authors**: Chenhan Lyu, Yutong Song, Pengfei Zhang, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2509.16444)  

**Abstract**: Mental health applications have emerged as a critical area in computational health, driven by rising global rates of mental illness, the integration of AI in psychological care, and the need for scalable solutions in underserved communities. These include therapy chatbots, crisis detection, and wellness platforms handling sensitive data, requiring specialized AI safety beyond general safeguards due to emotional vulnerability, risks like misdiagnosis or symptom exacerbation, and precise management of vulnerable states to avoid severe outcomes such as self-harm or loss of trust. Despite AI safety advances, general safeguards inadequately address mental health-specific challenges, including crisis intervention accuracy to avert escalations, therapeutic guideline adherence to prevent misinformation, scale limitations in resource-constrained settings, and adaptation to nuanced dialogues where generics may introduce biases or miss distress signals. We introduce an approach to apply Constitutional AI training with domain-specific mental health principles for safe, domain-adapted CAI systems in computational mental health applications. 

---
