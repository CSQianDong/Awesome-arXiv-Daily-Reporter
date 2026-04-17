# From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning 

**Authors**: Kiran Purohit, Ramasuri Narayanam, Soumyabrata Pal  

**Link**: [PDF](https://arxiv.org/pdf/2604.15244)  

**Abstract**: Speculative decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose outputs that a stronger target model verifies. However, its token-centric nature allows erroneous steps to propagate. Prior approaches mitigate this using external reward models, but incur additional latency, computational overhead, and limit generalizability. We propose SpecGuard, a verification-aware speculative decoding framework that performs step-level verification using only model-internal signals. At each step, SpecGuard samples multiple draft candidates and selects the most consistent step, which is then validated using an ensemble of two lightweight model-internal signals: (i) an attention-based grounding score that measures attribution to the input and previously accepted steps, and (ii) a log-probability-based score that captures token-level confidence. These signals jointly determine whether a step is accepted or recomputed using the target, allocating compute selectively. Experiments across a range of reasoning benchmarks show that SpecGuard improves accuracy by 3.6% while reducing latency by ~11%, outperforming both SD and reward-guided SD. 

---
# MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events 

**Authors**: Raunak Agarwal, Markus Wenzel, Simon Baur, Jonas Zimmer, George Harvey, Jackie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.15203)  

**Abstract**: Machine learning in high-stakes domains such as healthcare requires not only strong predictive performance but also reliable uncertainty quantification (UQ) to support human oversight. Multi-label text classification (MLTC) is a central task in this domain, yet remains challenging due to label imbalances, dependencies, and combinatorial complexity. Existing MLTC benchmarks are increasingly saturated and may be affected by training data contamination, making it difficult to distinguish genuine reasoning capabilities from memorization. We introduce MADE, a living MLTC benchmark derived from {m}edical device {ad}verse {e}vent reports and continuously updated with newly published reports to prevent contamination. MADE features a long-tailed distribution of hierarchical labels and enables reproducible evaluation with strict temporal splits. We establish baselines across more than 20 encoder- and decoder-only models under fine-tuning and few-shot settings (instruction-tuned/reasoning variants, local/API-accessible). We systematically assess entropy-/consistency-based and self-verbalized UQ methods. Results show clear trade-offs: smaller discriminatively fine-tuned decoders achieve the strongest head-to-tail accuracy while maintaining competitive UQ; generative fine-tuning delivers the most reliable UQ; large reasoning models improve performance on rare labels yet exhibit surprisingly weak UQ; and self-verbalized confidence is not a reliable proxy for uncertainty. Our work is publicly available at this https URL. 

---
# Fabricator or dynamic translator? 

**Authors**: Lisa Vasileva, Karin Sim  

**Link**: [PDF](https://arxiv.org/pdf/2604.15165)  

**Abstract**: LLMs are proving to be adept at machine translation although due to their generative nature they may at times overgenerate in various ways. These overgenerations are different from the neurobabble seen in NMT and range from LLM self-explanations, to risky confabulations, to appropriate explanations, where the LLM is able to act as a human translator would, enabling greater comprehension for the target audience. Detecting and determining the exact nature of the overgenerations is a challenging task. We detail different strategies we have explored for our work in a commercial setting, and present our results. 

---
# Compressing Sequences in the Latent Embedding Space: $K$-Token Merging for Large Language Models 

**Authors**: Zihao Xu, John Harvill, Ziwei Fan, Yizhou Sun, Hao Ding, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15153)  

**Abstract**: Large Language Models (LLMs) incur significant computational and memory costs when processing long prompts, as full self-attention scales quadratically with input length. Token compression aims to address this challenge by reducing the number of tokens representing inputs. However, existing prompt-compression approaches primarily operate in token space and overlook inefficiencies in the latent embedding space. In this paper, we propose K-Token Merging, a latent-space compression framework that merges each contiguous block of K token embeddings into a single embedding via a lightweight encoder. The compressed sequence is processed by a LoRA-adapted LLM, while generation remains in the original vocabulary. Experiments on structural reasoning (Textualized Tree), sentiment classification (Amazon Reviews), and code editing (CommitPackFT) show that K-Token Merging lies on the Pareto frontier of performance vs. compression, achieving up to 75% input length reduction with minimal performance degradation. 

---
# QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies 

**Authors**: Alexey Khoroshilov, Alexey Chernysh, Orkhan Ekhtibarov, Nini Kamkia, Dmitry Zmitrovich  

**Link**: [PDF](https://arxiv.org/pdf/2604.15151)  

**Abstract**: Large language models have demonstrated strong performance on general-purpose programming tasks, yet their ability to generate executable algorithmic trading strategies remains underexplored. Unlike standard code benchmarks, trading-strategy generation requires simultaneous mastery of domain-specific financial logic, knowledge of a specialized API, and the ability to produce code that is not only syntactically correct but also leads to actual trades on historical data. In this work, we present QuantCode-Bench, a benchmark for the systematic evaluation of modern LLMs in generating strategies for the Backtrader framework from textual descriptions in English. The benchmark contains 400 tasks of varying difficulty collected from Reddit, TradingView, StackExchange, GitHub, and synthetic sources. Evaluation is conducted through a multi-stage pipeline that checks syntactic correctness, successful backtest execution, the presence of trades, and semantic alignment with the task description using an LLM judge. We compare state-of-the-art models in two settings: single-turn, where the strategy must be generated correctly on the first attempt, and agentic multi-turn, where the model receives iterative feedback and may repair its errors. We analyze the failure modes across different stages of the pipeline and show that the main limitations of current models are not related to syntax, but rather to the correct operationalization of trading logic, proper API usage, and adherence to task semantics. These findings suggest that trading strategy generation constitutes a distinct class of domain-specific code generation tasks in which success requires not only technical correctness, but also alignment between natural-language descriptions, financial logic, and the observable behavior of the strategy on data. 

---
# DiscoTrace: Representing and Comparing Answering Strategies of Humans and LLMs in Information-Seeking Question Answering 

**Authors**: Neha Srikanth, Jordan Boyd-Graber, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2604.15140)  

**Abstract**: We introduce DiscoTrace, a method to identify the rhetorical strategies that answerers use when responding to information-seeking questions. DiscoTrace represents answers as a sequence of question-related discourse acts paired with interpretations of the original question, annotated on top of rhetorical structure theory parses. Applying DiscoTrace to answers from nine different human communities reveals that communities have diverse preferences for answer construction. In contrast, LLMs do not exhibit rhetorical diversity in their answers, even when prompted to mimic specific human community answering guidelines. LLMs also systematically opt for breadth, addressing interpretations of questions that human answerers choose not to address. Our findings can guide the development of pragmatic LLM answerers that consider a range of strategies informed by context in QA. 

---
# Blinded Multi-Rater Comparative Evaluation of a Large Language Model and Clinician-Authored Responses in CGM-Informed Diabetes Counseling 

**Authors**: Zhijun Guo, Alvina Lai, Emmanouil Korakas, Aristeidis Vagenas, Irshad Ahamed, Christo Albor, Hengrui Zhang, Justin Healy, Kezhi Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.15124)  

**Abstract**: Continuous glucose monitoring (CGM) is central to diabetes care, but explaining CGM patterns clearly and empathetically remains time-intensive. Evidence for retrieval-grounded large language model (LLM) systems in CGM-informed counseling remains limited. To evaluate whether a retrieval-grounded LLM-based conversational agent (CA) could support patient understanding of CGM data and preparation for routine diabetes consultations. We developed a retrieval-grounded LLM-based CA for CGM interpretation and diabetes counseling support. The system generated plain-language responses while avoiding individualized therapeutic advice. Twelve CGM-informed cases were constructed from publicly available datasets. Between Oct 2025 and Feb 2026, 6 senior UK diabetes clinicians each reviewed 2 assigned cases and answered 24 questions. In a blinded multi-rater evaluation, each CA-generated and clinician-authored response was independently rated by 3 clinicians on 6 quality dimensions. Safety flags and perceived source labels were also recorded. Primary analyses used linear mixed-effects models. A total of 288 unique responses (144 CA and 144 clinician) generated 864 ratings. The CA received higher quality scores than clinician responses (mean 4.37 vs 3.58), with an estimated mean difference of 0.782 points (95% CI 0.692-0.872; P<.001). The largest differences were for empathy (1.062, 95% CI 0.948-1.177) and actionability (0.992, 95% CI 0.877-1.106). Safety flag distributions were similar, with major concerns rare in both groups (3/432, 0.7% each). Retrieval-grounded LLM systems may have value as adjunct tools for CGM review, patient education, and preconsultation preparation. However, these findings do not support autonomous therapeutic decision-making or unsupervised real-world use. 

---
# IUQ: Interrogative Uncertainty Quantification for Long-Form Large Language Model Generation 

**Authors**: Haozhi Fan, Jinhao Duan, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15109)  

**Abstract**: Despite the rapid advancement of Large Language Models (LLMs), uncertainty quantification in LLM generation is a persistent challenge. Although recent approaches have achieved strong performance by restricting LLMs to produce short or constrained answer sets, many real-world applications require long-form and free-form text generation. A key difficulty in this setting is that LLMs often produce responses that are semantically coherent yet factually inaccurate, while the underlying semantics are multifaceted and the linguistic structure is complex. To tackle this challenge, this paper introduces Interrogative Uncertainty Quantification (IUQ), a novel framework that leverages inter-sample consistency and intra-sample faithfulness to quantify the uncertainty in long-form LLM outputs. By utilizing an interrogate-then-respond paradigm, our method provides reliable measures of claim-level uncertainty and the model's faithfulness. Experimental results across diverse model families and model sizes demonstrate the superior performance of IUQ over two widely used long-form generation datasets. The code is available at this https URL. 

---
# Explain the Flag: Contextualizing Hate Speech Beyond Censorship 

**Authors**: Jason Liartis, Eirini Kaldeli, Lambrini Gyftokosta, Eleftherios Chelioudakis, Orfeas Menis Mastromichalakis  

**Link**: [PDF](https://arxiv.org/pdf/2604.14970)  

**Abstract**: Hate, derogatory, and offensive speech remains a persistent challenge in online platforms and public discourse. While automated detection systems are widely used, most focus on censorship or removal, raising concerns for transparency and freedom of expression, and limiting opportunities to explain why content is harmful. To address these issues, explanatory approaches have emerged as a promising solution, aiming to make hate speech detection more transparent, accountable, and informative. In this paper, we present a hybrid approach that combines Large Language Models (LLMs) with three newly created and curated vocabularies to detect and explain hate speech in English, French, and Greek. Our system captures both inherently derogatory expressions tied to identity characteristics and direct group-targeted content through two complementary pipelines: one that detects and disambiguates problematic terms using the curated vocabularies, and one that leverages LLMs as context-aware evaluators of group-targeting content. The outputs are fused into grounded explanations that clarify why content is flagged. Human evaluation shows that our hybrid approach is accurate, with high-quality explanations, outperforming LLM-only baselines. 

---
# Text2Arch: A Dataset for Generating Scientific Architecture Diagrams from Natural Language Descriptions 

**Authors**: Shivank Garg, Sankalp Mittal, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.14941)  

**Abstract**: Communicating complex system designs or scientific processes through text alone is inefficient and prone to ambiguity. A system that automatically generates scientific architecture diagrams from text with high semantic fidelity can be useful in multiple applications like enterprise architecture visualization, AI-driven software design, and educational content creation. Hence, in this paper, we focus on leveraging language models to perform semantic understanding of the input text description to generate intermediate code that can be processed to generate high-fidelity architecture diagrams. Unfortunately, no clean large-scale open-access dataset exists, implying lack of any effective open models for this task. Hence, we contribute a comprehensive dataset, \system, comprising scientific architecture images, their corresponding textual descriptions, and associated DOT code representations. Leveraging this resource, we fine-tune a suite of small language models, and also perform in-context learning using GPT-4o. Through extensive experimentation, we show that \system{} models significantly outperform existing baseline models like DiagramAgent and perform at par with in-context learning-based generations from GPT-4o. We make the code, data and models publicly available. 

---
# XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics 

**Authors**: Jingxuan Liu, Zhi Qu, Jin Tei, Hidetaka Kamigaito, Lemao Liu, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.14934)  

**Abstract**: Automatic evaluation metrics are essential for building multilingual translation systems. The common practice of evaluating these systems is averaging metric scores across languages, yet this is suspicious since metrics may suffer from cross-lingual scoring bias, where translations of equal quality receive different scores across languages. This problem has not been systematically studied because no benchmark exists that provides parallel-quality instances across languages, and expert annotation is not realistic. In this work, we propose XQ-MEval, a semi-automatically built dataset covering nine translation directions, to benchmark translation metrics. Specifically, we inject MQM-defined errors into gold translations automatically, filter them by native speakers for reliability, and merge errors to generate pseudo translations with controllable quality. These pseudo translations are then paired with corresponding sources and references to form triplets used in assessing the qualities of translation metrics. Using XQ-MEval, our experiments on nine representative metrics reveal the inconsistency between averaging and human judgment and provide the first empirical evidence of cross-lingual scoring bias. Finally, we propose a normalization strategy derived from XQ-MEval that aligns score distributions across languages, improving the fairness and reliability of multilingual metric evaluation. 

---
# IE as Cache: Information Extraction Enhanced Agentic Reasoning 

**Authors**: Hang Lv, Sheng Liang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Hao Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14930)  

**Abstract**: Information Extraction aims to distill structured, decision-relevant information from unstructured text, serving as a foundation for downstream understanding and reasoning. However, it is traditionally treated merely as a terminal objective: once extracted, the resulting structure is often consumed in isolation rather than maintained and reused during multi-step inference. Moving beyond this, we propose \textit{IE-as-Cache}, a framework that repurposes IE as a cognitive cache to enhance agentic reasoning. Drawing inspiration from hierarchical computer memory, our approach combines query-driven extraction with cache-aware reasoning to dynamically maintain compact intermediate information and filter noise. Experiments on challenging benchmarks across diverse LLMs demonstrate significant improvements in reasoning accuracy, indicating that IE can be effectively repurposed as a reusable cognitive resource and offering a promising direction for future research on downstream uses of IE. 

---
# Comparison of Modern Multilingual Text Embedding Techniques for Hate Speech Detection Task 

**Authors**: Evaldas Vaiciukynas, Paulius Danenas, Linas Ablonskis, Algirdas Sukys, Edgaras Dambrauskas, Voldemaras Zitkus, Rita Butkiene, Rimantas Butleris  

**Link**: [PDF](https://arxiv.org/pdf/2604.14907)  

**Abstract**: Online hate speech and abusive language pose a growing challenge for content moderation, especially in multilingual settings and for low-resource languages such as Lithuanian. This paper investigates to what extent modern multilingual sentence embedding models can support accurate hate speech detection in Lithuanian, Russian, and English, and how their performance depends on downstream modeling choices and feature dimensionality. We introduce LtHate, a new Lithuanian hate speech corpus derived from news portals and social networks, and benchmark six modern multilingual encoders (potion, gemma, bge, snow, jina, e5) on LtHate, RuToxic, and EnSuperset using a unified Python pipeline. For each embedding, we train both a one class HBOS anomaly detector and a two class CatBoost classifier, with and without principal component analysis (PCA) compression to 64-dimensional feature vectors. Across all datasets, two class supervised models consistently and substantially outperform one class anomaly detection, with the best configurations achieving up to 80.96% accuracy and AUC ROC of 0.887 in Lithuanian (jina), 92.19% accuracy and AUC ROC of 0.978 in Russian (e5), and 77.21% accuracy and AUC ROC of 0.859 in English (e5 with PCA). PCA compression preserves almost all discriminative power in the supervised setting, while showing some negative impact for the unsupervised anomaly detection case. These results demonstrate how modern multilingual sentence embeddings combined with gradient boosted decision trees provide robust soft-computing solutions for multilingual hate speech detection applications. 

---
# Reasoning Dynamics and the Limits of Monitoring Modality Reliance in Vision-Language Models 

**Authors**: Danae Sánchez Villegas, Samuel Lewis-Lim, Nikolaos Aletras, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2604.14888)  

**Abstract**: Recent advances in vision language models (VLMs) offer reasoning capabilities, yet how these unfold and integrate visual and textual information remains unclear. We analyze reasoning dynamics in 18 VLMs covering instruction-tuned and reasoning-trained models from two different model families. We track confidence over Chain-of-Thought (CoT), measure the corrective effect of reasoning, and evaluate the contribution of intermediate reasoning steps. We find that models are prone to answer inertia, in which early commitments to a prediction are reinforced, rather than revised during reasoning steps. While reasoning-trained models show stronger corrective behavior, their gains depend on modality conditions, from text-dominant to vision-only settings. Using controlled interventions with misleading textual cues, we show that models are consistently influenced by these cues even when visual evidence is sufficient, and assess whether this influence is recoverable from CoT. Although this influence can appear in the CoT, its detectability varies across models and depends on what is being monitored. Reasoning-trained models are more likely to explicitly refer to the cues, but their longer and fluent CoTs can still appear visually grounded while actually following textual cues, obscuring modality reliance. In contrast, instruction-tuned models refer to the cues less explicitly, but their shorter traces reveal inconsistencies with the visual input. Taken together, these findings indicate that CoT provides only a partial view of how different modalities drive VLM decisions, with important implications for the transparency and safety of multimodal systems. 

---
# RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding 

**Authors**: Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14885)  

**Abstract**: Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but existing training-free variants face trade-offs: retrieval-based drafts break when no exact match exists, while logits-based drafts lack structural guidance. We propose $\textbf{RACER}$ ($\textbf{R}$etrieval-$\textbf{A}$ugmented $\textbf{C}$ont$\textbf{e}$xtual $\textbf{R}$apid Speculative Decoding), a lightweight and training-free method that integrates retrieved exact patterns with logit-driven future cues. This unification supplies both reliable anchors and flexible extrapolation, yielding richer speculative drafts. Experiments on Spec-Bench, HumanEval, and MGSM-ZH demonstrate that RACER consistently accelerates inference, achieving more than $2\times$ speedup over autoregressive decoding, and outperforms prior training-free methods, offering a scalable, plug-and-play solution for efficient LLM decoding. Our source code is available at $\href{this https URL}{this https URL}$. 

---
# Segment-Level Coherence for Robust Harmful Intent Probing in LLMs 

**Authors**: Xuanli He, Bilgehan Sel, Faizan Ali, Jenny Bao, Hoagy Cunningham, Jerry Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.14865)  

**Abstract**: Large Language Models (LLMs) are increasingly exposed to adaptive jailbreaking, particularly in high-stakes Chemical, Biological, Radiological, and Nuclear (CBRN) domains. Although streaming probes enable real-time monitoring, they still make systematic errors. We identify a core issue: existing methods often rely on a few high-scoring tokens, leading to false alarms when sensitive CBRN terms appear in benign contexts. To address this, we introduce a streaming probing objective that requires multiple evidence tokens to consistently support a prediction, rather than relying on isolated spikes. This encourages more robust detection based on aggregated signals instead of single-token cues. At a fixed 1% false-positive rate, our method improves the true-positive rate by 35.55% relative to strong streaming baselines. We further observe substantial gains in AUROC, even when starting from near-saturated baseline performance (AUROC = 97.40%). We also show that probing Attention or MLP activations consistently outperforms residual-stream features. Finally, even when adversarial fine-tuning enables novel character-level ciphers, harmful intent remains detectable: probes developed for the base LLMs can be applied ``plug-and-play'' to these obfuscated attacks, achieving an AUROC of over 98.85%. 

---
# Schema Key Wording as an Instruction Channel in Structured Generation under Constrained Decoding 

**Authors**: Yifan Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.14862)  

**Abstract**: Constrained decoding has been widely adopted for structured generation with large language models (LLMs), ensuring that outputs satisfy predefined formats such as JSON and XML. However, existing approaches largely treat schemas as purely structural constraints and overlook the possibility that their linguistic formulation may affect model behavior. In this work, we study how instruction placement influences model performance in structured generation and show that merely changing the wording of schema keys, without modifying the prompt or model parameters, can significantly alter model performance under constrained decoding. Based on this observation, we propose to reinterpret structured generation as a multi-channel instruction problem, where instructions can be conveyed explicitly through prompts and implicitly through schema keys during decoding. To the best of our knowledge, this is the first work to systematically study how schema key formulation acts as an implicit instruction channel and affects model performance under constrained decoding. Experiments on multiple mathematical reasoning benchmarks show that different model families exhibit distinct sensitivities to these instruction channels: Qwen models consistently benefit from schema-level instructions, while LLaMA models rely more heavily on prompt-level guidance. We further observe non-additive interaction effects between instruction channels, showing that combining multiple channels does not always lead to further improvement. These findings suggest that schema design not only determines output structure, but also carries instruction signals, offering a new perspective on structured generation in LLMs. 

---
# ClimateCause: Complex and Implicit Causal Structures in Climate Reports 

**Authors**: Liesbeth Allein, Nataly Pineda-Castañeda, Andrea Rocci, Marie-Francine Moens  

**Link**: [PDF](https://arxiv.org/pdf/2604.14856)  

**Abstract**: Understanding climate change requires reasoning over complex causal networks. Yet, existing causal discovery datasets predominantly capture explicit, direct causal relations. We introduce ClimateCause, a manually expert-annotated dataset of higher-order causal structures from science-for-policy climate reports, including implicit and nested causality. Cause-effect expressions are normalized and disentangled into individual causal relations to facilitate graph construction, with unique annotations for cause-effect correlation, relation type, and spatiotemporal context. We further demonstrate ClimateCause's value for quantifying readability based on the semantic complexity of causal graphs underlying a statement. Finally, large language model benchmarking on correlation inference and causal chain reasoning highlights the latter as a key challenge. 

---
# Exploring and Testing Skill-Based Behavioral Profile Annotation: Human Operability and LLM Feasibility under Schema-Guided Execution 

**Authors**: Yufeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14843)  

**Abstract**: Behavioral Profile (BP) annotation is difficult to automate because it requires simultaneous coding across multiple linguistic dimensions. We treat BP annotation as a bundle of annotation skills rather than a single task and evaluate LLM-assisted BP annotation from this perspective. Using 3,134 concordance lines of 30 Chinese metaphorical color-term derivatives and a 14-feature BP schema, we implement a skill-file-driven pipeline in which each feature is externally defined through schema files, decision rules, and examples. Two human annotators completed a two-round schema-only protocol on a 300-instance validation subset, enabling BP skills to be classified as directly operable, recoverable under focused re-annotation, or structurally underspecified. GPT-5.4 and three locally deployable open-source models were then evaluated under the same setup. Results show that BP annotation is highly heterogeneous at the skill level: 5 skills are directly operable, 4 are recoverable after focused re-annotation, and 5 remain structurally underspecified. GPT-5.4 executes the retained skills with substantial reliability (accuracy = 0.678, \k{appa} = 0.665, weighted F1 = 0.695), but this feasibility is selective rather than global. Human and GPT difficulty profiles are strongly aligned at the skill level (r = 0.881), but not at the instance level (r = 0.016) or lexical-item level (r = -0.142), a pattern we describe as shared taxonomy, independent execution. Pairwise agreement further suggests that GPT is better understood as an independent third skill voice than as a direct human substitute. Open-source failures are concentrated in schema-to-skill execution problems. These findings suggest that automatic annotation should be evaluated in terms of skill feasibility rather than task-level automation. 

---
# Pangu-ACE: Adaptive Cascaded Experts for Educational Response Generation on EduBench 

**Authors**: Dinghao Li, Wenlong Zhou, Zhimin Chen, Yuehan Peng, Hong Ni, Chengfu Zou, Guoyu Shi, Yaochen Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.14828)  

**Abstract**: Educational assistants should spend more computation only when the task needs it. This paper rewrites our earlier draft around the system that was actually implemented and archived in the repository: a sample-level 1B to 7B cascade for the shared-8 EduBench benchmark. The final system, Pangu-ACE, uses a 1B tutor-router to produce a draft answer plus routing signals, then either accepts the draft or escalates the sample to a 7B specialist prompt. We also correct a major offline evaluation bug: earlier summaries over-credited some open-form outputs that only satisfied superficial format checks. After CPU-side rescoring from saved prediction JSONL, the full Chinese test archive (7013 samples) shows that cascade_final improves deterministic quality from 0.457 to 0.538 and format validity from 0.707 to 0.866 over the legacy rule_v2 system while accepting 19.7% of requests directly at 1B. Routing is strongly task dependent: IP is accepted by 1B 78.0% of the time, while QG and EC still escalate almost always. The current archived deployment does not yet show latency gains, so the defensible efficiency story is routing selectivity rather than wall-clock speedup. We also package a reproducible artifact-first paper workflow and clarify the remaining external-baseline gap: GPT-5.4 re-judging is implemented locally, but the configured provider endpoint and key are invalid, so final sampled-baseline alignment with GPT-5.4 remains pending infrastructure repair. 

---
# Domain Fine-Tuning FinBERT on Finnish Histopathological Reports: Train-Time Signals and Downstream Correlations 

**Authors**: Rami Luisto, Liisa Petäinen, Tommi Grönholm, Jan Böhm, Maarit Ahtiainen, Tomi Lilja, Ilkka Pölönen, Sami Äyrämö  

**Link**: [PDF](https://arxiv.org/pdf/2604.14815)  

**Abstract**: In NLP classification tasks where little labeled data exists, domain fine-tuning of transformer models on unlabeled data is an established approach. In this paper we have two aims.
(1) We describe our observations from fine-tuning the Finnish BERT model on Finnish medical text data.
(2) We report on our attempts to predict the benefit of domain-specific pre-training of Finnish BERT from observing the geometry of embedding changes due to domain fine-tuning.
Our driving motivation is the common\situation in healthcare AI where we might experience long delays in acquiring datasets, especially with respect to labels. 

---
# Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem 

**Authors**: Zeguan Xiao, Siqing Li, Yong Wang, Xuetao Wei, Jian Yang, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14808)  

**Abstract**: Machine unlearning for large language models (LLMs) aims to remove targeted knowledge while preserving general capability. In this paper, we recast LLM unlearning as an asymmetric two-task problem: retention is the primary objective and forgetting is an auxiliary. From this perspective, we propose a retention-prioritized gradient synthesis framework that decouples task-specific gradient extraction from conflict-aware combination. Instantiating the framework, we adapt established PCGrad to resolve gradient conflicts, and introduce SAGO, a novel retention-prioritized gradient synthesis method. Theoretically, both variants ensure non-negative cosine similarity with the retain gradient, while SAGO achieves strictly tighter alignment through constructive sign-constrained synthesis. Empirically, on WMDP Bio/Cyber and RWKU benchmarks, SAGO consistently pushes the Pareto frontier: e.g., on WMDP Bio (SimNPO+GD), recovery of target model MMLU performance progresses from 44.6% (naive) to 94.0% (+PCGrad) and further to 96.0% (+SAGO), while maintaining comparable forgetting strength. Our results show that re-shaping gradient geometry, rather than re-balancing losses, is the key to mitigating unlearning-retention trade-offs. 

---
# Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems 

**Authors**: Nishanth Madhusudhan, Vikas Yadav, Alexandre Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2604.14799)  

**Abstract**: Effective abstention (EA), recognizing evidence insufficiency and refraining from answering, is critical for reliable multimodal systems. Yet existing evaluation paradigms for vision-language models (VLMs) and multi-agent systems (MAS) assume answerability, pushing models to always respond. Abstention has been studied in text-only settings but remains underexplored multimodally; current benchmarks either ignore unanswerability or rely on coarse methods that miss realistic failure modes. We introduce MM-AQA, a benchmark that constructs unanswerable instances from answerable ones via transformations along two axes: visual modality dependency and evidence sufficiency. Evaluating three frontier VLMs spanning closed and open-source models and two MAS architectures across 2079 samples, we find: (1) under standard prompting, VLMs rarely abstain; even simple confidence baselines outperform this setup, (2) MAS improves abstention but introduces an accuracy-abstention trade-off, (3) sequential designs match or exceed iterative variants, suggesting the bottleneck is miscalibration rather than reasoning depth, and (4) models abstain when image or text evidence is absent, but attempt reconciliation with degraded or contradictory evidence. Effective multimodal abstention requires abstention-aware training rather than better prompting or more agents. 

---
# CoPA: Benchmarking Personalized Question Answering with Data-Informed Cognitive Factors 

**Authors**: Hang Su, Zequn Liu, Chen Hu, Xuesong Lu, Yingce Xia, Zhen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14773)  

**Abstract**: While LLMs have demonstrated remarkable potential in Question Answering (QA), evaluating personalization remains a critical bottleneck. Existing paradigms predominantly rely on lexical-level similarity or manual heuristics, often lacking sufficient data-driven validation. We address this by mining Community-Individual Preference Divergence (CIPD), where individual choices override consensus, to distill six key personalization factors as evaluative dimensions. Accordingly, we introduce CoPA, a benchmark with 1,985 user profiles for fine-grained, factor-level assessment. By quantifying the alignment between model outputs and user-specific cognitive preferences inferred from interaction patterns, CoPA provides a more comprehensive and discriminative standard for evaluating personalized QA than generic metrics. The code is available at this https URL. 

---
# Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement 

**Authors**: Midan Shim, Seokju Hwang, Kaehyun Um, Kyong-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.14749)  

**Abstract**: Large language models still struggle with faithfulness and hallucinations despite their remarkable reasoning abilities. In Knowledge Graph Question Answering (KGQA), semantic parsing-based approaches address the limitations by understanding constraints in a user's question and converting them into a logical form to execute on a knowledge graph. However, existing KGQA benchmarks and methods are biased toward positive and calculation constraints. Negative constraints are neglected, although they frequently appear in real-world questions. In this paper, we introduce a new task, NEgative-conSTrained (NEST) KGQA, where each question contains at least one negative constraint, and a corresponding dataset, NestKGQA. We also design PyLF, a Python-formatted logical form, since existing logical forms are hardly suitable to express negation clearly while maintaining readability. Furthermore, NEST questions naturally contain multiple constraints. To mitigate their semantic complexity, we present a novel framework named CUCKOO, specialized to multiple-constrained questions and ensuring semantic executability. CUCKOO first generates a constraint-aware logical form draft and performs schema-guided semantic matching. It then selectively applies self-directed refinement only when executing improper logical forms yields an empty result, reducing cost while improving robustness. Experimental results demonstrate that CUCKOO consistently outperforms baselines on both conventional KGQA and NEST-KGQA benchmarks under few-shot settings. 

---
# SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models 

**Authors**: Binxian Su, Haoye Lou, Shucheng Zhu, Weikang Wang, Ying Liu, Dong Yu, Pengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14672)  

**Abstract**: Large language models (LLMs) are being increasingly used in urban planning, but since gendered space theory highlights how gender hierarchies are embedded in spatial organization, there is concern that LLMs may reproduce or amplify such biases. We introduce SPAGBias - the first systematic framework to evaluate spatial gender bias in LLMs. It combines a taxonomy of 62 urban micro-spaces, a prompt library, and three diagnostic layers: explicit (forced-choice resampling), probabilistic (token-level asymmetry), and constructional (semantic and narrative role analysis). Testing six representative models, we identify structured gender-space associations that go beyond the public-private divide, forming nuanced micro-level mappings. Story generation reveals how emotion, wording, and social roles jointly shape "spatial gender narratives". We also examine how prompt design, temperature, and model scale influence bias expression. Tracing experiments indicate that these patterns are embedded and reinforced across the model pipeline (pre-training, instruction tuning, and reward modeling), with model associations found to substantially exceed real-world distributions. Downstream experiments further reveal that such biases produce concrete failures in both normative and descriptive application settings. This work connects sociological theory with computational analysis, extending bias research into the spatial domain and uncovering how LLMs encode social gender cognition through language. 

---
# CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction 

**Authors**: Sizhe Wang, Ziqi Xu, Claire Najjuuko, Charles Alba, Chenyang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14651)  

**Abstract**: Clinical language models (LMs) are increasingly applied to support clinical risk prediction from free-text notes, yet their uncertainty estimates often remain poorly calibrated and clinically unreliable. In this work, we propose Clinical Uncertainty Risk Alignment (CURA), a framework that aligns clinical LM-based risk estimates and uncertainty with both individual error likelihoods and cohort-level ambiguities. CURA first fine-tunes domain-specific clinical LMs to obtain task-adapted patient embeddings, and then performs uncertainty fine-tuning of a multi-head classifier using a bi-level uncertainty objective. Specifically, an individual-level calibration term aligns predictive uncertainty with each patient's likelihood of error, while a cohort-aware regularizer pulls risk estimates toward event rates in their local neighborhoods in the embedding space and places extra weight on ambiguous cohorts near the decision boundary. We further show that this cohort-aware term can be interpreted as a cross-entropy loss with neighborhood-informed soft labels, providing a label-smoothing view of our method. Extensive experiments on MIMIC-IV clinical risk prediction tasks across various clinical LMs show that CURA consistently improves calibration metrics without substantially compromising discrimination. Further analysis illustrates that CURA reduces overconfident false reassurance and yields more trustworthy uncertainty estimates for downstream clinical decision support. 

---
# CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge 

**Authors**: Seyun Bae, Seokhan Lee, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14644)  

**Abstract**: The inability to filter out in advance all potentially problematic data from the pre-training of large language models has given rise to the need for methods for unlearning specific pieces of knowledge after training. Existing techniques overlook the need for continuous and immediate action, causing them to suffer from degraded utility as updates accumulate and protracted exposure of sensitive information. To address these issues, we propose Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge (CURaTE). Our method begins by training a sentence embedding model on a dataset designed to enable the formation of sharp decision boundaries for determining whether a given input prompt corresponds to any stored forget requests. The similarity of a given input to the forget requests is then used to determine whether to answer or return a refusal response. We show that even with such a simple approach, not only does CURaTE achieve more effective forgetting than existing methods, but by avoiding modification of the language model parameters, it also maintains near perfect knowledge preservation over any number of updates and is the only method capable of continual unlearning in real-time. 

---
# Fact4ac at the Financial Misinformation Detection Challenge Task: Reference-Free Financial Misinformation Detection via Fine-Tuning and Few-Shot Prompting of Large Language Models 

**Authors**: Cuong Hoang, Le-Minh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14640)  

**Abstract**: The proliferation of financial misinformation poses a severe threat to market stability and investor trust, misleading market behavior and creating critical information asymmetry. Detecting such misleading narratives is inherently challenging, particularly in real-world scenarios where external evidence or supplementary references for cross-verification are strictly unavailable. This paper presents our winning methodology for the "Reference-Free Financial Misinformation Detection" shared task. Built upon the recently proposed RFC-BENCH framework (Jiang et al. 2026), this task challenges models to determine the veracity of financial claims by relying solely on internal semantic understanding and contextual consistency, rather than external fact-checking. To address this formidable evaluation setup, we propose a comprehensive framework that capitalizes on the reasoning capabilities of state-of-the-art Large Language Models (LLMs). Our approach systematically integrates in-context learning, specifically zero-shot and few-shot prompting strategies, with Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) to optimally align the models with the subtle linguistic cues of financial manipulation. Our proposed system demonstrated superior efficacy, successfully securing the first-place ranking on both official leaderboards. Specifically, we achieved an accuracy of 95.4% on the public test set and 96.3% on the private test set, highlighting the robustness of our method and contributing to the acceleration of context-aware misinformation detection in financial Natural Language Processing. Our models (14B and 32B) are available at this https URL. 

---
# Pushing the Boundaries of Multiple Choice Evaluation to One Hundred Options 

**Authors**: Nahyun Lee, Guijin Son  

**Link**: [PDF](https://arxiv.org/pdf/2604.14634)  

**Abstract**: Multiple choice evaluation is widely used for benchmarking large language models, yet near ceiling accuracy in low option settings can be sustained by shortcut strategies that obscure true competence. Therefore, we propose a massive option evaluation protocol that scales the candidate set to one hundred options and sharply reduces the impact of chance performance. We apply this framework to a Korean orthography error detection task where models must pick the single incorrect sentence from a large candidate set. With fixed targets and repeated resampling and shuffling, we obtain stable estimates while separating content driven failures from positional artifacts. Across experiments, results indicate that strong performance in low option settings can overstate model competence. This apparent advantage often weakens under dense interference at high $N$, revealing gaps that conventional benchmarks tend to obscure. We identify two failure modes, semantic confusion and position bias toward early options under uncertainty. To isolate the effect of context length, we run padding controlled and length matched tests, which suggest that the main bottleneck is candidate ranking rather than context length. Together, these findings support massive option evaluation as a general framework for stress testing model reliability under extreme distractor density, beyond what low option benchmarks can reveal. 

---
# StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation 

**Authors**: Geonhui Jang, Dongyoon Han, YoungJoon Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14631)  

**Abstract**: Effective code generation requires both model capability and a problem representation that carefully structures how models reason and plan. Existing approaches augment reasoning steps or inject specific structure into how models think, but leave scattered problem conditions unchanged. Inspired by the way humans organize fragmented information into coherent explanations, we propose StoryCoder, a narrative reformulation framework that transforms code generation questions into coherent natural language narratives, providing richer contextual structure than simple rephrasings. Each narrative consists of three components: a task overview, constraints, and example test cases, guided by the selected algorithm and genre. Experiments across 11 models on HumanEval, LiveCodeBench, and CodeForces demonstrate consistent improvements, with an average gain of 18.7% in zero-shot pass@10. Beyond accuracy, our analyses reveal that narrative reformulation guides models toward correct algorithmic strategies, reduces implementation errors, and induces a more modular code structure. The analyses further show that these benefits depend on narrative coherence and genre alignment, suggesting that structured problem representation is important for code generation regardless of model scale or architecture. Our code is available at this https URL. 

---
# Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value Set Authoring 

**Authors**: Sumit Mukherjee, Juan Shu, Nairwita Mazumder, Tate Kernell, Celena Wheeler, Shannon Hastings, Chris Sidey-Gibbons  

**Link**: [PDF](https://arxiv.org/pdf/2604.14616)  

**Abstract**: Clinical value set authoring -- the task of identifying all codes in a standardized vocabulary that define a clinical concept -- is a recurring bottleneck in clinical quality measurement and phenotyping. A natural approach is to prompt a large language model (LLM) to generate the required codes directly, but structured clinical vocabularies are large, version-controlled, and not reliably memorized during pretraining. We propose Retrieval-Augmented Set Completion (RASC): retrieve the $K$ most similar existing value sets from a curated corpus to form a candidate pool, then apply a classifier to each candidate code. Theoretically, retrieve-and-select can reduce statistical complexity by shrinking the effective output space from the full vocabulary to a much smaller retrieved candidate pool. We demonstrate the utility of RASC on 11,803 publicly available VSAC value sets, constructing the first large-scale benchmark for this task. A cross-encoder fine-tuned on SAPBert achieves AUROC~0.852 and value-set-level F1~0.298, outperforming a simpler three-layer Multilayer Perceptron (AUROC~0.799, F1~0.250) and both reduce the number of irrelevant candidates per true positive from 12.3 (retrieval-only) to approximately 3.2 and 4.4 respectively. Zero-shot GPT-4o achieves value-set-level F1~0.105, with 48.6\% of returned codes absent from VSAC entirely. This performance gap widens with increasing value set size, consistent with RASC's theoretical advantage. We observe similar performance gains across two other classifier model types, namely a cross-encoder initialized from pre-trained SAPBert and a LightGBM model, demonstrating that RASC's benefits extend beyond a single model class. The code to download and create the benchmark dataset, as well as the model training code is available at: \href{this https URL}{this https URL}. 

---
# CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification 

**Authors**: Yian Wang, Yuen Chen, Agam Goyal, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2604.14602)  

**Abstract**: Large language models (LLMs) frequently generate toxic content, posing significant risks for safe deployment. Current mitigation strategies often degrade generation quality or require costly human annotation. We propose CAUSALDETOX, a framework that identifies and intervenes on the specific attention heads causally responsible for toxic generation. Using the Probability of Necessity and Sufficiency (PNS), we isolate a minimal set of heads that are necessary and sufficient for toxicity. We utilize these components via two complementary strategies: (1) Local Inference-Time Intervention, which constructs dynamic, input-specific steering vectors for context-aware detoxification, and (2) PNS-Guided Fine-Tuning, which permanently unlearns toxic representations. We also introduce PARATOX, a novel benchmark of aligned toxic/non-toxic sentence pairs enabling controlled counterfactual evaluation. Experiments on ToxiGen, ImplicitHate, and ParaDetox show that CAUSALDETOX achieves up to 5.34% greater toxicity reduction compared to baselines while preserving linguistic fluency, and offers a 7x speedup in head selection. 

---
# NLP needs Diversity outside of 'Diversity' 

**Authors**: Joshua Tint  

**Link**: [PDF](https://arxiv.org/pdf/2604.14595)  

**Abstract**: This position paper argues that recent progress with diversity in NLP is disproportionately concentrated on a small number of areas surrounding fairness. We further argue that this is the result of a number of incentives, biases, and barriers which come together to disenfranchise marginalized researchers in non-fairness fields, or to move them into fairness-related fields. We substantiate our claims with an investigation into the demographics of NLP researchers by subfield, using our research to support a number of recommendations for ensuring that all areas within NLP can become more inclusive and equitable. In particular, we highlight the importance of breaking down feedback loops that reinforce disparities, and the need to address geographical and linguistic barriers that hinder participation in NLP research. 

---
# Mechanistic Decoding of Cognitive Constructs in LLMs 

**Authors**: Yitong Shou, Manhao Guan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14593)  

**Abstract**: While Large Language Models (LLMs) demonstrate increasingly sophisticated affective capabilities, the internal mechanisms by which they process complex emotions remain unclear. Existing interpretability approaches often treat models as black boxes or focus on coarse-grained basic emotions, leaving the cognitive structure of more complex affective states underexplored. To bridge this gap, we propose a Cognitive Reverse-Engineering framework based on Representation Engineering (RepE) to analyze social-comparison jealousy. By combining appraisal theory with subspace orthogonalization, regression-based weighting, and bidirectional causal steering, we isolate and quantify two psychological antecedents of jealousy, Superiority of Comparison Person and Domain Self-Definitional Relevance, and examine their causal effects on model judgments. Experiments on eight LLMs from the Llama, Qwen, and Gemma families suggest that models natively encode jealousy as a structured linear combination of these constituent factors. Their internal representations are broadly consistent with the human psychological construct, treating Superiority as the foundational trigger and Relevance as the ultimate intensity multiplier. Our framework also demonstrates that toxic emotional states can be mechanically detected and surgically suppressed, suggesting a possible route toward representational monitoring and intervention for AI safety in multi-agent environments. 

---
# PeerPrism: Peer Evaluation Expertise vs Review-writing AI 

**Authors**: Soroush Sadeghian, Alireza Daqiq, Radin Cheraghi, Sajad Ebrahimi, Negar Arabzadeh, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2604.14513)  

**Abstract**: Large Language Models (LLMs) are increasingly used in scientific peer review, assisting with drafting, rewriting, expansion, and refinement. However, existing peer-review LLM detection methods largely treat authorship as a binary problem-human vs. AI-without accounting for the hybrid nature of modern review workflows. In practice, evaluative ideas and surface realization may originate from different sources, creating a spectrum of human-AI collaboration.
In this work, we introduce PeerPrism, a large-scale benchmark of 20,690 peer reviews explicitly designed to disentangle idea provenance from text provenance. We construct controlled generation regimes spanning fully human, fully synthetic, and multiple hybrid transformations. This design enables systematic evaluation of whether detectors identify the origin of the surface text or the origin of the evaluative reasoning. We benchmark state-of-the-art LLM text detection methods on PeerPrism. While several methods achieve high accuracy on the standard binary task (human vs. fully synthetic), their predictions diverge sharply under hybrid regimes. In particular, when ideas originate from humans but the surface text is AI-generated, detectors frequently disagree and produce contradictory classifications. Accompanied by stylometric and semantic analyses, our results show that current detection methods conflate surface realization with intellectual contribution.
Overall, we demonstrate that LLM detection in peer review cannot be reduced to a binary attribution problem. Instead, authorship must be modeled as a multidimensional construct spanning semantic reasoning and stylistic realization. PeerPrism is the first benchmark evaluating human-AI collaboration in these settings. We release all code, data, prompts, and evaluation scripts to facilitate reproducible research at this https URL. 

---
# CobwebTM: Probabilistic Concept Formation for Lifelong and Hierarchical Topic Modeling 

**Authors**: Karthik Singaravadivelan, Anant Gupta, Zekun Wang, Christopher MacLellan  

**Link**: [PDF](https://arxiv.org/pdf/2604.14489)  

**Abstract**: Topic modeling seeks to uncover latent semantic structure in text corpora with minimal supervision. Neural approaches achieve strong performance but require extensive tuning and struggle with lifelong learning due to catastrophic forgetting and fixed capacity, while classical probabilistic models lack flexibility and adaptability to streaming data. We introduce \textsc{CobwebTM}, a low-parameter lifelong hierarchical topic model based on incremental probabilistic concept formation. By adapting the Cobweb algorithm to continuous document embeddings, \textsc{CobwebTM} constructs semantic hierarchies online, enabling unsupervised topic discovery, dynamic topic creation, and hierarchical organization without predefining the number of topics. Across diverse datasets, \textsc{CobwebTM} achieves strong topic coherence, stable topics over time, and high-quality hierarchies, demonstrating that incremental symbolic concept formation combined with pretrained representations is an efficient approach to topic modeling. 

---
# Psychological Steering of Large Language Models 

**Authors**: Leonardo Blas, Robin Jia, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2604.14463)  

**Abstract**: Large language models (LLMs) emulate a consistent human-like behavior that can be shaped through activation-level interventions. This paradigm is converging on additive residual-stream injections, which rely on injection-strength sweeps to approximate optimal intervention settings. However, existing methods restrict the search space and sweep in uncalibrated activation-space units, potentially missing optimal intervention conditions. Thus, we introduce a psychological steering framework that performs unbounded, fluency-constrained sweeps in semantically calibrated units. Our method derives and calibrates residual-stream injections using psychological artifacts, and we use the IPIP-NEO-120, which measures the OCEAN personality model, to compare six injection methods. We find that mean-difference (MD) injections outperform Personality Prompting (P$^2$), an established baseline for OCEAN steering, in open-ended generation in 11 of 14 LLMs, with gains of 3.6\% to 16.4\%, overturning prior reports favoring prompting and positioning representation engineering as a new frontier in open-ended psychological steering. Further, we find that a hybrid of P$^2$ and MD injections outperforms both methods in 13 of 14 LLMs, with gains over P$^2$ ranging from 5.6\% to 21.9\% and from 3.3\% to 26.7\% over MD injections. Finally, we show that MD injections align with the Linear Representation Hypothesis and provide reliable, approximately linear control knobs for psychological steering. Nevertheless, they also induce OCEAN trait covariance patterns that depart from the Big Two model, suggesting a gap between learned representations and human psychology. 

---
# Filling in the Mechanisms: How do LMs Learn Filler-Gap Dependencies under Developmental Constraints? 

**Authors**: Atrey Desai, Sathvik Nair  

**Link**: [PDF](https://arxiv.org/pdf/2604.14459)  

**Abstract**: For humans, filler-gap dependencies require a shared representation across different syntactic constructions. Although causal analyses suggest this may also be true for LLMs (Boguraev et al., 2025), it is still unclear if such a representation also exists for language models trained on developmentally feasible quantities of data. We applied Distributed Alignment Search (DAS, Geiger et al. (2024)) to LMs trained on varying amounts of data from the BabyLM challenge (Warstadt et al., 2023), to evaluate whether representations of filler-gap dependencies transfer between wh-questions and topicalization, which greatly vary in terms of their input frequency. Our results suggest shared, yet item-sensitive mechanisms may develop with limited training data. More importantly, LMs still require far more data than humans to learn comparable generalizations, highlighting the need for language-specific biases in models of language acquisition. 

---
# MARCA: A Checklist-Based Benchmark for Multilingual Web Search 

**Authors**: Thales Sales Almeida, Giovana Kerche Bonás, Ramon Pires, Celio Larcher, Hugo Abonizio, Marcos Piau, Roseval Malaquias Junior, Rodrigo Nogueira, Thiago Laitz  

**Link**: [PDF](https://arxiv.org/pdf/2604.14448)  

**Abstract**: Large language models (LLMs) are increasingly used as sources of information, yet their reliability depends on the ability to search the web, select relevant evidence, and synthesize complete answers. While recent benchmarks evaluate web-browsing and agentic tool use, multilingual settings, and Portuguese in particular, remain underexplored. We present \textsc{MARCA}, a bilingual (English and Portuguese) benchmark for evaluating LLMs on web-based information seeking. \textsc{MARCA} consists of 52 manually authored multi-entity questions, paired with manually validated checklist-style rubrics that explicitly measure answer completeness and correctness. We evaluate 14 models under two interaction settings: a Basic framework with direct web search and scraping, and an Orchestrator framework that enables task decomposition via delegated subagents. To capture stochasticity, each question is executed multiple times and performance is reported with run-level uncertainty. Across models, we observe large performance differences, find that orchestration often improves coverage, and identify substantial variability in how models transfer from English to Portuguese. The benchmark is available at this https URL 

---
# Hierarchical vs. Flat Iteration in Shared-Weight Transformers 

**Authors**: Sang-Il Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.14442)  

**Abstract**: We present an empirical study of whether hierarchically structured, shared-weight recurrence can match the representational quality of independent-layer stacking in a Transformer-based language model. HRM-LM replaces L independent Transformer layers with a two-speed recurrent pair: a Fast module operating at every step for local refinement, and a Slow module operating every T steps for global compression. This recurrent hierarchy is unrolled for M = N x T steps with shared parameters. The central and most robust finding, supported by a parameter-matched Universal Transformer ablation (UniTF, 1.2B) across five independent runs, is a sharp empirical gap between the two approaches. 

---
# Three-Phase Transformer 

**Authors**: Mohammad R. Abu Ayyash  

**Link**: [PDF](https://arxiv.org/pdf/2604.14430)  

**Abstract**: We present Three-Phase Transformer (3PT), a residual-stream structural prior for decoder-only Transformers on a standard SwiGLU + RMSNorm + RoPE + GQA backbone. The hidden vector is partitioned into N equally-sized cyclic channels, each maintained by phase-respecting ops: a per-channel RMSNorm, a 2D Givens rotation between attention and FFN that rotates each channel by theta + i*(2*pi/N), and a head-count constraint aligning GQA heads with the partition. The architecture is a self-stabilizing equilibrium between scrambling and re-imposition, not a bolted-on module. The partition carves out a one-dimensional DC subspace orthogonal to the channels, into which we inject a fixed Gabriel's horn profile r(p) = 1/(p+1) as an absolute-position side-channel composing orthogonally with RoPE's relative-position rotation. The canonical N=3 borrows its metaphor from balanced three-phase AC, where three sinusoids 120 degrees apart sum to zero with no anti-correlated pair. At 123M parameters on WikiText-103, 3PT achieves -7.20% perplexity (-2.62% bits-per-byte) over a matched RoPE-Only baseline at +1,536 parameters (0.00124% of total), with 1.93x step-count convergence speedup (1.64x wall-clock). N behaves as a parameter-sharing knob rather than a unique optimum: at 5.5M an N-sweep over {1,2,3,4,6,8,12} is near-monotone with N=1 winning; at 123M a three-seed sweep finds N=3 and N=1 statistically indistinguishable. The load-bearing mechanism is the channel-partitioned residual stream, per-block rotation, per-phase normalization, and horn DC injection. We characterize (a) self-stabilization of the geometry without explicit enforcement, a novel instance of the conservation-law framework for neural networks; (b) a U-shaped depth profile of rotation-angle drift at 12 layers; (c) orthogonal composition with RoPE, attention, and FFN. 

---
# The Autocorrelation Blind Spot: Why 42% of Turn-Level Findings in LLM Conversation Analysis May Be Spurious 

**Authors**: Ferdinand M. Schessl  

**Link**: [PDF](https://arxiv.org/pdf/2604.14414)  

**Abstract**: Turn-level metrics are widely used to evaluate properties of multi-turn human-LLM conversations, from safety and sycophancy to dialogue quality. However, consecutive turns within a conversation are not statistically independent -- a fact that virtually all current evaluation pipelines fail to correct for in their statistical inference. We systematically characterize the autocorrelation structure of 66 turn-level metrics across 202 multi-turn conversations (11,639 turn pairs, 5 German-speaking users, 4 LLM platforms) and demonstrate that naive pooled analysis produces severely inflated significance estimates: 42% of associations that appear significant under standard pooled testing fail to survive cluster-robust correction. The inflation varies substantially across categories rather than scaling linearly with autocorrelation: three memoryless families (embedding velocity, directional, differential) aggregate to 14%, while the seven non-memoryless families (thermo-cycle, frame distance, lexical/structural, rolling windows, cumulative, interaction, timestamp) aggregate to 33%, with individual category rates ranging from 0% to 100% depending on per-family effect size. We present a two-stage correction framework combining Chelton (1983) effective degrees of freedom with conversation-level block bootstrap, and validate it on a pre-registered hold-out split where cluster-robust metrics replicate at 57% versus 30% for pooled-only metrics. We provide concrete design principles, a publication checklist, and open-source code for the correction pipeline. A survey of ~30 recent papers at major NLP and AI venues that compute turn-level statistics in LLM evaluations finds that only 4 address temporal dependence at all, and 26 do not correct for it. 

---
# Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection 

**Authors**: David Basil, Chirooth Girigowda, Bradley Hauer, Sahir Momin, Ning Shi, Grzegorz Kondrak  

**Link**: [PDF](https://arxiv.org/pdf/2604.14397)  

**Abstract**: We study the task of automatically expanding WordNet-style lexical resources to new languages through sense generation. We generate senses by associating target-language lemmas with existing lexical concepts via semantic projection. Given a sense-tagged English corpus and its translation, our method projects English synsets onto aligned target-language tokens and assigns the corresponding lemmas to those synsets. To generate these alignments and ensure their quality, we augment a pre-trained base aligner with a bilingual dictionary, which is also used to filter out incorrect sense projections. We evaluate the method on multiple languages, comparing it to prior methods, as well as dictionary-based and large language model baselines. Results show that the proposed project-and-filter strategy improves precision while remaining interpretable and requiring few external resources. We plan to make our code, documentation, and generated sense inventories accessible. 

---
# BiCon-Gate: Consistency-Gated De-colloquialisation for Dialogue Fact-Checking 

**Authors**: Hyunkyung Park, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2604.14389)  

**Abstract**: Automated fact-checking in dialogue involves multi-turn conversations where colloquial language is frequent yet understudied. To address this gap, we propose a conservative rewrite candidate for each response claim via staged de-colloquialisation, combining lightweight surface normalisation with scoped in-claim coreference resolution. We then introduce BiCon-Gate, a semantics-aware consistency gate that selects the rewrite candidate only when it is semantically supported by the dialogue context, otherwise falling back to the original claim. This gated selection stabilises downstream fact-checking and yields gains in both evidence retrieval and fact verification. On the DialFact benchmark, our approach improves retrieval and verification, with particularly strong gains on SUPPORTS, and outperforms competitive baselines, including a decoder-based one-shot LLM rewrite that attempts to perform all de-colloquialisation steps in a single pass. 

---
# The Cost of Language: Centroid Erasure Exposes and Exploits Modal Competition in Multimodal Language Models 

**Authors**: Akshay Paruchuri, Ishan Chatterjee, Henry Fuchs, Ehsan Adeli, Piotr Didyk  

**Link**: [PDF](https://arxiv.org/pdf/2604.14363)  

**Abstract**: Multimodal language models systematically underperform on visual perception tasks, yet the structure underlying this failure remains poorly understood. We propose centroid replacement, collapsing each token to its nearest K-means centroid, as a controlled probe for modal dependence. Across seven models spanning three architecture families, erasing text centroid structure costs 4$\times$ more accuracy than erasing visual centroid structure, exposing a universal imbalance where language representations overshadow vision even on tasks that demand visual reasoning. We exploit this asymmetry through text centroid contrastive decoding, recovering up to +16.9% accuracy on individual tasks by contrastively decoding against a text-centroid-erased reference. This intervention varies meaningfully with training approaches: standard fine-tuned models show larger gains (+5.6% on average) than preference-optimized models (+1.5% on average). Our findings suggest that modal competition is structurally localized, correctable at inference time without retraining, and quantifiable as a diagnostic signal to guide future multimodal training. 

---
# APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI 

**Authors**: Pratyay Banerjee, Masud Moshtaghi, Shivashankar Subramanian, Amita Misra, Ankit Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2604.14362)  

**Abstract**: Large language models still struggle with reliable long-term conversational memory: simply enlarging context windows or applying naive retrieval often introduces noise and destabilizes responses. We present APEX-MEM, a conversational memory system that combines three key innovations: (1) a property graph which uses domain-agnostic ontology to structure conversations as temporally grounded events in an entity-centric framework, (2) append-only storage that preserves the full temporal evolution of information, and (3) a multi-tool retrieval agent that understands and resolves conflicting or evolving information at query time, producing a compact and contextually relevant memory summary. This retrieval-time resolution preserves the full interaction history while suppressing irrelevant details. APEX-MEM achieves 88.88% accuracy on LOCOMO's Question Answering task and 86.2% on LongMemEval, outperforming state-of-the-art session-aware approaches and demonstrating that structured property graphs enable more temporally coherent long-term conversational reasoning. 

---
# When PCOS Meets Eating Disorders: An Explainable AI Approach to Detecting the Hidden Triple Burden 

**Authors**: Apoorv Prasad, Susan McRoy  

**Link**: [PDF](https://arxiv.org/pdf/2604.14356)  

**Abstract**: Women with polycystic ovary syndrome (PCOS) face substantially elevated risks of body image distress, disordered eating, and metabolic challenges, yet existing natural language processing approaches for detecting these conditions lack transparency and cannot identify co-occurring presentations. We developed small, open-source language models to automatically detect this triple burden in social media posts with grounded explainability. We collected 1,000 PCOS-related posts from six subreddits, with two trained annotators labeling posts using guidelines operationalizing Lee et al. (2017) clinical framework. Three models (Gemma-2-2B, Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B) were fine-tuned using Low-Rank Adaptation to generate structured explanations with textual evidence. The best model achieved 75.3 percent exact match accuracy on 150 held-out posts, with robust comorbidity detection and strong explainability. Performance declined with diagnostic complexity, indicating their best use is for screening rather than autonomous diagnosis. 

---
# Shuffle the Context: RoPE-Perturbed Self-Distillation for Long-Context Adaptation 

**Authors**: Zichong Li, Chen Liang, Liliang Ren, Tuo Zhao, Yelong Shen, Weizhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14339)  

**Abstract**: Large language models (LLMs) increasingly operate in settings that require reliable long-context understanding, such as retrieval-augmented generation and multi-document reasoning. A common strategy is to fine-tune pretrained short-context models at the target sequence length. However, we find that standard long-context adaptation can remain brittle: model accuracy depends strongly on the absolute placement of relevant evidence, exhibiting high positional variance even when controlling for task format and difficulty.
We propose RoPE-Perturbed Self-Distillation, a training regularizer that improves positional robustness. The core idea is to form alternative "views" of the same training sequence by perturbing its RoPE indices -- effectively moving parts of the context to different positions -- and to train the model to produce consistent predictions across views via self-distillation. This encourages reliance on semantic signals instead of brittle position dependencies. Experiments on long-context adaptation of Llama-3-8B and Qwen-3-4B demonstrate consistent gains on long-context benchmarks, including up to 12.04% improvement on RULER-64K for Llama-3-8B and 2.71% on RULER-256K for Qwen-3-4B after SFT, alongside improved length extrapolation beyond the training context window. 

---
# Faithfulness Serum: Mitigating the Faithfulness Gap in Textual Explanations of LLM Decisions via Attribution Guidance 

**Authors**: Bar Alon, Itamar Zimerman, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2604.14325)  

**Abstract**: Large language models (LLMs) achieve strong performance and have revolutionized NLP, but their lack of explainability keeps them treated as black boxes, limiting their use in domains that demand transparency and trust. A promising direction to address this issue is post-hoc text-based explanations, which aim to explain model decisions in natural language. Prior work has focused on generating convincing rationales that appear to be subjectively faithful, but it remains unclear whether these explanations are epistemically faithful, whether they reflect the internal evidence the model actually relied on for its decision. In this paper, we first assess the epistemic faithfulness of LLM-generated explanations via counterfactuals and show that they are often unfaithful. We then introduce a training-free method that enhances faithfulness by guiding explanation generation through attention-level interventions, informed by token-level heatmaps extracted via a faithful attribution method. This method significantly improves epistemic faithfulness across multiple models, benchmarks, and prompts. 

---
# Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness 

**Authors**: Hao An, Yibin Lou, Jiayi Guo, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14324)  

**Abstract**: Large language models (LLMs) often exhibit hallucinations due to their inability to accurately perceive their own knowledge boundaries. Existing abstention fine-tuning methods typically partition datasets directly based on response accuracy, causing models to suffer from severe label noise near the decision boundaries and consequently exhibit high rates of abstentions or hallucinations. This paper adopts a latent space representation perspective, revealing a "gray zone" near the decision hyperplane where internal belief ambiguity constitutes the core performance bottleneck. Based on this insight, we propose the **GeoDe** (**Geo**metric **De**noising) framework for abstention fine-tuning. This method constructs a truth hyperplane using linear probes and performs "geometric denoising" by employing geometric distance as a confidence signal for abstention decisions. This approach filters out ambiguous boundary samples while retaining high-fidelity signals for fine-tuning. Experiments across multiple models (Llama3, Qwen3) and benchmark datasets (TriviaQA, NQ, SciQ, SimpleQA) demonstrate that GeoDe significantly enhances model truthfulness and demonstrates strong generalization in out-of-distribution (OOD) scenarios. Code is available at this https URL. 

---
# LLM Predictive Scoring and Validation: Inferring Experience Ratings from Unstructured Text 

**Authors**: Jason Potteiger, Andrew Hong, Ito Zapata  

**Link**: [PDF](https://arxiv.org/pdf/2604.14321)  

**Abstract**: We tasked GPT-4.1 to read what baseball fans wrote about their game-day experience and predict the overall experience rating each fan gave on a 0-10 survey scale. The model received only the text of a single open-ended response. These AI predictions were compared with the actual experience ratings captured by the survey instrument across approximately 10,000 fan responses from five Major League Baseball teams. In total two-thirds of predicted ratings fell within one point of self-reported fan ratings (67% within +/-1, 36% exact match), and the predicted measurement was near-deterministic across three independent scoring runs (87% exact agreement, 99.9% within +/-1). Predicted ratings aligned most strongly with the overall experience rating (r = 0.82) rather than with any specific aspect of the game-day experience such as parking, concessions, staff, etc. However, predictions were systematically lower than self-reported ratings by approximately one point, and this gap was not driven by any single aspect. Rather, our analysis shows that self-reported ratings capture the fan's verdict, an overall evaluative judgment that integrates the entire experience. While predicted ratings quantify the impact of salient moments characterized as memorable, emotionally intense, unusual, or actionable. Each measure contains information the other misses. These baseline results establish that a simple, unoptimized prompt can directionally predict how fans rate their experience from the text a fan wrote and that a gap between the two numbers can be interpreted as a construct difference worth preserving rather than an error to eliminate. 

---
# Tracking the Temporal Dynamics of News Coverage of Catastrophic and Violent Events 

**Authors**: Emily Lugos, Maurício Gruppi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14315)  

**Abstract**: The modern news cycle has been fundamentally reshaped by the rapid exchange of information online. As a result, media framing shifts dynamically as new information, political responses, and social reactions emerge. Understanding how these narratives form, propagate, and evolve is essential for interpreting public discourse during moments of crisis. In this study, we examine the temporal and semantic dynamics of reporting for violent and catastrophic events using a large-scale corpus of 126,602 news articles collected from online publishers. We quantify narrative change through publication volume, semantic drift, semantic dispersion, and term relevance. Our results show that sudden events of impact exhibit structured and predictable news-cycle patterns characterized by rapid surges in coverage, early semantic drift, and gradual declines toward the baseline. In addition, our results indicate the terms that are driving the temporal patterns. 

---
# EuropeMedQA Study Protocol: A Multilingual, Multimodal Medical Examination Dataset for Language Model Evaluation 

**Authors**: Francesco Andrea Causio, Vittorio De Vita, Olivia Riccomi, Michele Ferramola, Federico Felizzi, Antonio Cristiano, Lorenzo De Mori, Chiara Battipaglia, Melissa Sawaya, Luigi De Angelis, Marcello Di Pumpo, Alessandra Piscitelli, Pietro Eric Risuleo, Alessia Longo, Giulia Vojvodic, Mariapia Vassalli, Bianca Destro Castaniti, Nicolò Scarsi, Manuel Del Medico  

**Link**: [PDF](https://arxiv.org/pdf/2604.14306)  

**Abstract**: While Large Language Models (LLMs) have demonstrated high proficiency on English-centric medical examinations, their performance often declines when faced with non-English languages and multimodal diagnostic tasks. This study protocol describes the development of EuropeMedQA, the first comprehensive, multilingual, and multimodal medical examination dataset sourced from official regulatory exams in Italy, France, Spain, and Portugal. Following FAIR data principles and SPIRIT-AI guidelines, we describe a rigorous curation process and an automated translation pipeline for comparative analysis. We evaluate contemporary multimodal LLMs using a zero-shot, strictly constrained prompting strategy to assess cross-lingual transfer and visual reasoning. EuropeMedQA aims to provide a contamination-resistant benchmark that reflects the complexity of European clinical practices and fosters the development of more generalizable medical AI. 

---
# ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents 

**Authors**: Zhuofeng Li, Yi Lu, Dongfu Jiang, Haoxiang Zhang, Yuyang Bai, Chuan Li, Yu Wang, Shuiwang Ji, Jianwen Xie, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14261)  

**Abstract**: The rapid rise in AI conference submissions has driven increasing exploration of large language models (LLMs) for peer review support. However, LLM-based reviewers often generate superficial, formulaic comments lacking substantive, evidence-grounded feedback. We attribute this to the underutilization of two key components of human reviewing: explicit rubrics and contextual grounding in existing work. To address this, we introduce REVIEWBENCH, a benchmark evaluating review text according to paper-specific rubrics derived from official guidelines, the paper's content, and human-written reviews. We further propose REVIEWGROUNDER, a rubric-guided, tool-integrated multi-agent framework that decomposes reviewing into drafting and grounding stages, enriching shallow drafts via targeted evidence consolidation. Experiments on REVIEWBENCH show that REVIEWGROUNDER, using a Phi-4-14B-based drafter and a GPT-OSS-120B-based grounding stage, consistently outperforms baselines with substantially stronger/larger backbones (e.g., GPT-4.1 and DeepSeek-R1-670B) in both alignment with human judgments and rubric-based review quality across 8 dimensions. The code is available \href{this https URL}{here}. 

---
# MEME-Fusion@CHiPSAL 2026: Multimodal Ablation Study of Hate Detection and Sentiment Analysis on Nepali Memes 

**Authors**: Samir Wagle, Reewaj Khanal, Abiral Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2604.14218)  

**Abstract**: Hate speech detection in Devanagari-scripted social media memes presents compounded challenges: multimodal content structure, script-specific linguistic complexity, and extreme data scarcity in low-resource settings. This paper presents our system for the CHiPSAL 2026 shared task, addressing both Subtask A (binary hate speech detection) and Subtask B (three-class sentiment classification: positive, neutral, negative). We propose a hybrid cross-modal attention fusion architecture that combines CLIP (ViT-B/32) for visual encoding with BGE-M3 for multilingual text representation, connected through 4-head self-attention and a learnable gating network that dynamically weights modality contributions on a per-sample basis. Systematic evaluation across eight model configurations demonstrates that explicit cross-modal reasoning achieves a 5.9% F1-macro improvement over text-only baselines on Subtask A, while uncovering two unexpected but critical findings: English-centric vision models exhibit near-random performance on Devanagari script, and standard ensemble methods catastrophically degrade under data scarcity (N nearly equal to 850 per fold) due to correlated overfitting. The code can be accessed at this https URL 

---
# CROP: Token-Efficient Reasoning in Large Language Models via Regularized Prompt Optimization 

**Authors**: Deep Shah, Sanket Badhe, Nehal Kathrotia, Priyanka Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2604.14214)  

**Abstract**: Large Language Models utilizing reasoning techniques improve task performance but incur significant latency and token costs due to verbose generation. Existing automatic prompt optimization(APO) frameworks target task accuracy exclusively at the expense of generating long reasoning traces. We propose Cost-Regularized Optimization of Prompts (CROP), an APO method that introduces regularization on response length by generating textual feedback in addition to standard accuracy feedback. This forces the optimization process to produce prompts that elicit concise responses containing only critical information and reasoning. We evaluate our approach on complex reasoning datasets, specifically GSM8K, LogiQA and BIG-Bench Hard. We achieved an 80.6\% reduction in token consumption while maintaining competitive accuracy, seeing only a nominal decline in performance. This presents a pragmatic solution for deploying token-efficient and cost-effective agentic AI systems in production pipelines. 

---
# Chinese Language Is Not More Efficient Than English in Vibe Coding: A Preliminary Study on Token Cost and Problem-Solving Rate 

**Authors**: Simiao Ren, Xingyu Shen, Yuchen Zhou, Dennis, Ankit Raj  

**Link**: [PDF](https://arxiv.org/pdf/2604.14210)  

**Abstract**: A claim has been circulating on social media and practitioner forums that Chinese prompts are more token-efficient than English for LLM coding tasks, potentially reducing costs by up to 40\%. This claim has influenced developers to consider switching to Chinese for ``vibe coding'' to save on API costs. In this paper, we conduct a rigorous empirical study using SWE-bench Lite, a benchmark of software engineering tasks, to evaluate whether this claim of Chinese token efficiency holds up to scrutiny. Our results reveal three key findings: First, the efficiency advantage of Chinese is not observed. Second, token cost varies by model architecture in ways that defy simple assumptions: while MiniMax-2.7 shows 1.28x higher token costs for Chinese, GLM-5 actually consumes fewer tokens with Chinese prompts. Third, and most importantly, we found that the success rate when prompting in Chinese is generally lower than in English across all models we tested. We also measure cost efficiency as expected cost per successful task -- jointly accounting for token consumption and task resolution rate. These findings should be interpreted as preliminary evidence rather than a definitive conclusion, given the limited number of models evaluated and the narrow set of benchmarks tested due to resource constraints; they indicate that language effects on token cost are model-dependent, and that practitioners should not expect cost savings or performance gains just by switching their prompt language to Chinese. 

---
# The PICCO Framework for Large Language Model Prompting: A Taxonomy and Reference Architecture for Prompt Structure 

**Authors**: David A. Cook  

**Link**: [PDF](https://arxiv.org/pdf/2604.14197)  

**Abstract**: Large language model (LLM) performance depends heavily on prompt design, yet prompt construction is often described and applied inconsistently. Our purpose was to derive a reference framework for structuring LLM prompts. This paper presents PICCO, a framework derived through a rigorous synthesis of 11 previously published prompting frameworks identified through a multi-database search. The analysis yields two main contributions. First, it proposes a taxonomy that distinguishes prompt frameworks, prompt elements, prompt generation, prompting techniques, and prompt engineering as related but non-equivalent concepts. Second, it derives a five-element reference architecture for prompt generation: Persona, Instructions, Context, Constraints, and Output (PICCO). For each element, we define its function, scope, and relationship to other elements, with the goal of improving conceptual clarity and supporting more systematic prompt design. Finally, to support application of the framework, we outline key concepts relevant to implementation, including prompting techniques (e.g., zero-shot, few-shot, chain-of-thought, ensembling, decomposition, and self-critique, with selected variants), human and automated approaches to iterative prompt engineering, responsible prompting considerations such as security, privacy, bias, and trust, and priorities for future research. This work is a conceptual and methodological contribution: it formalizes a common structure for prompt specification and comparison, but does not claim empirical validation of PICCO as an optimization method. 

---
# Attention to Mamba: A Recipe for Cross-Architecture Distillation 

**Authors**: Abhinav Moudgil, Ningyuan Huang, Eeshan Gunesh Dhekane, Pau Rodríguez, Luca Zappella, Federico Danieli  

**Link**: [PDF](https://arxiv.org/pdf/2604.14191)  

**Abstract**: State Space Models (SSMs) such as Mamba have become a popular alternative to Transformer models, due to their reduced memory consumption and higher throughput at generation compared to their Attention-based counterparts. On the other hand, the community has built up a considerable body of knowledge on how to train Transformers, and many pretrained Transformer models are readily available. To facilitate the adoption of SSMs while leveraging existing pretrained Transformers, we aim to identify an effective recipe to distill an Attention-based model into a Mamba-like architecture. In prior work on cross-architecture distillation, however, it has been shown that a naïve distillation procedure from Transformers to Mamba fails to preserve the original teacher performance, a limitation often overcome with hybrid solutions combining Attention and SSM blocks. The key argument from our work is that, by equipping Mamba with a principled initialization, we can recover an overall better recipe for cross-architectural distillation. To this end, we propose a principled two-stage approach: first, we distill knowledge from a traditional Transformer into a linearized version of Attention, using an adaptation of the kernel trick. Then, we distill the linearized version into an adapted Mamba model that does not use any Attention block. Overall, the distilled Mamba model is able to preserve the original Pythia-1B Transformer performance in downstream tasks, maintaining a perplexity of 14.11 close to the teacher's 13.86. To show the efficacy of our recipe, we conduct thorough ablations at 1B scale with 10B tokens varying sequence mixer architecture, scaling analysis on model sizes and total distillation tokens, and a sensitivity analysis on tokens allocation between stages. 

---
# Internal Knowledge Without External Expression: Probing the Generalization Boundary of a Classical Chinese Language Model 

**Authors**: Jiuting Chen, Yuan Lian, Hao Wu, Tianqi Huang, Hiroshi Sasaki, Makoto Kouno, Jongil Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14180)  

**Abstract**: We train a 318M-parameter Transformer language model from scratch on a curated corpus of 1.56 billion tokens of pure Classical Chinese, with zero English characters or Arabic numerals. Through systematic out-of-distribution (OOD) testing, we investigate whether the model can distinguish known from unknown inputs, and crucially, whether it can express this distinction in its generated text.
We find a clear dissociation between internal and external uncertainty. Internally, the model exhibits a perplexity jump ratio of 2.39x between real and fabricated historical events (p = 8.9e-11, n = 92 per group), with semi-fabricated events (real figures + fictional events) showing the highest perplexity (4.24x, p = 1.1e-16), demonstrating genuine factual encoding beyond syntactic pattern matching. Externally, however, the model never learns to express uncertainty: classical Chinese epistemic markers appear at lower rates for OOD questions (3.5%) than for in-distribution questions (8.3%, p = 0.023), reflecting rhetorical conventions rather than genuine metacognition.
We replicate both findings across three languages (Classical Chinese, English, Japanese), three writing systems, and eight models from 110M to 1.56B parameters. We further show that uncertainty expression frequency is determined entirely by training data conventions, with Classical Chinese models showing a "humility paradox" (more hedging for known topics), while Japanese models almost never hedge. We argue that metacognitive expression -- the ability to say "I don't know" -- does not emerge from language modeling alone and requires explicit training signals such as RLHF. 

---
# An Underexplored Frontier: Large Language Models for Rare Disease Patient Education and Communication -- A scoping review 

**Authors**: Zaifu Zhan, Yu Hou, Kai Yu, Min Zeng, Anita Burgun, Xiaoyi Chen, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14179)  

**Abstract**: Rare diseases affect over 300 million people worldwide and are characterized by complex care pathways, limited clinical expertise, and substantial unmet communication needs throughout the long patient journey. Recent advances in large language models (LLMs) offer new opportunities to support patient education and communication, yet their application in rare diseases remains unclear.
We conducted a scoping review of studies published between January 2022 and March 2026 across major databases, identifying 12 studies on LLM-based rare disease patient education and communication. Data were extracted on study characteristics, application scenarios, model usage, and evaluation methods, and synthesized using descriptive and qualitative analyses.
The literature is highly recent and dominated by general-purpose models, particularly ChatGPT. Most studies focus on patient question answering using curated question sets, with limited use of real-world data or longitudinal communication scenarios. Evaluations are primarily centered on accuracy, with limited attention to patient-centered dimensions such as readability, empathy, and communication quality. Multilingual communication is rarely addressed.
Overall, the field remains at an early stage. Future research should prioritize patient-centered design, domain-adapted methods, and real-world deployment to support safe, adaptive, and effective communication in rare diseases. 

---
# Listen, Correct, and Feed Back: Spoken Pedagogical Feedback Generation 

**Authors**: Junhong Liang, Yifan Lu, Ekaterina Kochmar, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2604.14177)  

**Abstract**: Grammatical error correction (GEC) and explanation (GEE) have made rapid progress, but real teaching scenarios also require \emph{learner-friendly pedagogical feedback} that is actionable, level-appropriate, and encouraging. We introduce \textbf{SPFG} (\textbf{S}poken \textbf{P}edagogical \textbf{F}eedback \textbf{G}eneration), a dataset built based on the Speak \& Improve Challenge 2025 corpus, pairing fluency-oriented transcriptions with GEC targets and \emph{human-verified} teacher-style feedback, including preferred/rejected feedback pairs for preference learning. We study a transcript-based Spoken Grammatical Error Correction (SGEC) setting and evaluate three instruction-tuned LLMs (Qwen2.5, Llama-3.1, and GLM-4), comparing supervised fine-tuning (SFT) with preference-based alignment (using DPO and KTO) for jointly generating corrections and feedback. Results show that SFT provides the most consistent improvements, while DPO/KTO yield smaller or mixed gains, and that correction quality and feedback quality are weakly coupled. Our implementation is available at this https URL. 

---
# QU-NLP at ArchEHR-QA 2026: Two-Stage QLoRA Fine-Tuning of Qwen3-4B for Patient-Oriented Clinical Question Answering and Evidence Sentence Alignment 

**Authors**: Mohammad AL-Smadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14175)  

**Abstract**: We present a unified system addressing both Subtask 3 (answer generation) and Subtask 4 (evidence sentence alignment) of the ArchEHR-QA Shared Task. For Subtask 3, we apply two-stage Quantised Low-Rank Adaptation (QLoRA) to Qwen3-4B loaded in 4-bit NF4 quantisation: first on 30,000 samples from the emrQA-MedSQuAD corpus to establish clinical domain competence, then on the 20 annotated development cases to learn the task-specific output style. Our system achieves an overall score of 32.87 on the official test-2026 split (BLEU = 9.42, ROUGE-L = 27.04, SARI = 55.42, BERTScore = 43.00, AlignScore = 25.28, MEDCON = 37.04). For Subtask 4, we develop a weighted ensemble of three retrieval methods - BM25 with relative thresholding, TF-IDF cosine similarity, and a fine-tuned cross-encoder - to identify note sentences supporting a given gold answer, achieving a micro-F1 of 67.16 on the 100-case test set. Experiments reveal that both subtasks expose the same fundamental challenge: 20 annotated training cases are insufficient to distinguish relevant from irrelevant clinical sentences, pointing to data augmentation as the highest-leverage future direction. 

---
# Correcting Suppressed Log-Probabilities in Language Models with Post-Transformer Adapters 

**Authors**: Bryan Sanchez  

**Link**: [PDF](https://arxiv.org/pdf/2604.14174)  

**Abstract**: Alignment-tuned language models frequently suppress factual log-probabilities on politically sensitive topics despite retaining the knowledge in their hidden representations. We show that a 786K-parameter (approximately 0.02% of the base model) post-transformer adapter, trained on frozen hidden states, corrects this suppression on 31 ideology-discriminating facts across Qwen3-4B, 8B, and 14B. The adapter memorizes all 15 training facts and generalizes to 11--39% of 16 held-out facts across 5 random splits per scale, with zero knowledge regressions via anchored training. Both gated (SwiGLU) and ungated (linear bottleneck) adapters achieve comparable results; neither consistently outperforms the other (Fisher exact p > 0.09 at all scales). On instruct models, the adapter corrects log-probability rankings. When applied at all token positions during generation, the adapter produces incoherent output; however, when applied only at the current prediction position (last-position-only), the adapter produces coherent, less censored text. A logit-space adapter operating after token projection fails to produce coherent generation at any application mode, suggesting hidden-state intervention is the correct level for generation correction. A previously undocumented silent gradient bug in Apple MLX explains all null results in earlier iterations of this work: the standard pattern nn.value_and_grad(model, fn)(this http URL()) returns zero gradients without error; the correct pattern nn.value_and_grad(model, fn)(model, data) resolves this. We provide a minimal reproduction and discuss implications for other adapter research using MLX. 

---
# Tug-of-War within A Decade: Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generations 

**Authors**: Ziyin Zhou, Jianyi Zhang, Xu ji, Yilong Li, Jiameng Han, Zhangchi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14172)  

**Abstract**: Large Language Models (LLMs) are essential for analyzing and addressing vulnerabilities in cybersecurity. However, among over 200,000 vulnerabilities were discovered in the past decade, more than 30,000 have been changed or updated. This necessitates frequent updates to the training datasets and internal knowledge bases of LLMs to maintain knowledge consistency. In this paper, we focus on the problem of knowledge discrepancy and conflict within CVE (Common Vulnerabilities and Exposures) detection and analysis. This problem hinders LLMs' ability to retrieve the latest knowledge from original training datasets, leading to knowledge conflicts, fabrications of factually incorrect results, and generation hallucinations. To address this problem, we propose an innovative two-stage framework called CRVA-TGRAG (Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generation). First, to improve document retrieval accuracy during the retrieval stage, we utilize Parent Document Segmentation and an ensemble retrieval scheme based on semantic similarity and inverted indexing. Second, to enhance LLMs' capabilities based on the retrieval of CVE dataset in generation stage, we employ a teacher-guided preference optimization technique to fine-tune LLMs. Our framework not only enhances the quality of content retrieval through RAG but also leverages the advantages of preference fine-tuning in LLMs to answer questions more effectively and precisely. Experiments demonstrate our method achieves higher accuracy in retrieving the latest CVEs compared to external knowledge bases. In conclusion, our framework significantly mitigates potential knowledge conflicts and inconsistencies that may arise from relying solely on LLMs for knowledge retrieval. 

---
# Benchmarking Linguistic Adaptation in Comparable-Sized LLMs: A Study of Llama-3.1-8B, Mistral-7B-v0.1, and Qwen3-8B on Romanized Nepali 

**Authors**: Ananda Rimal, Adarsha Rimal  

**Link**: [PDF](https://arxiv.org/pdf/2604.14171)  

**Abstract**: Romanized Nepali, the Nepali language written in the Latin alphabet, is the dominant medium for informal digital communication in Nepal, yet it remains critically underresourced in the landscape of Large Language Models (LLMs). This study presents a systematic benchmarking of linguistic adaptation across three comparable-sized open-weight models: Llama-3.1-8B, Mistral-7B-v0.1, and Qwen3-8B. We evaluate these architectures under zero-shot and fine-tuned settings using a curated bilingual dataset of 10,000 transliterated instruction-following samples. Performance is quantified across five metrics spanning seven measurement dimensions: Perplexity (PPL), BERTScore, chrF++, ROUGE-1, ROUGE-2, ROUGE-L, and BLEU, capturing fluency, phonetic consistency, and semantic integrity. Models were fine-tuned using Quantized Low-Rank Adaptation (QLoRA) with Rank-Stabilized LoRA (rsLoRA) at rank r=32 on dual NVIDIA Tesla T4 GPUs, training only approximately 1% of each model's parameters in under 27 total GPU-hours. At zero-shot, all three models fail to generate Romanized Nepali, each exhibiting a distinct architecture-specific failure mode. Following fine-tuning, all three resolve these failures and converge to BERTScore approximately 0.75 and chrF++ greater than 23. Overall dimension-wise assessment across ten criteria identifies Qwen3-8B as the overall recommended architecture, being the only model to produce semantically relevant zero-shot output and leading all structural alignment metrics post-SFT. The adaptation headroom hypothesis is confirmed: Llama-3.1-8B, despite its weakest zero-shot baseline, achieves the largest absolute fine-tuning gains in PPL (Delta = -49.77) and BERTScore (Delta = +0.3287), making it the preferred choice for iterative low-resource development pipelines. This work establishes the first rigorous baseline for Romanized Nepali adaptation in comparable-sized open-weight LLMs. 

---
# Stateful Evidence-Driven Retrieval-Augmented Generation with Iterative Reasoning 

**Authors**: Qi Dong, Ziheng Lin, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.14170)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds Large Language Models (LLMs) in external knowledge but often suffers from flat context representations and stateless retrieval, leading to unstable performance. We propose Stateful Evidence-Driven RAG with Iterative Reasoning, a framework that models question answering as a progressive evidence accumulation process. Retrieved documents are converted into structured reasoning units with explicit relevance and confidence signals and maintained in a persistent evidence pool capturing both supportive and non-supportive information. The framework performs evidence-driven deficiency analysis to identify gaps and conflicts and iteratively refines queries to guide subsequent retrieval. This iterative reasoning process enables stable evidence aggregation and improves robustness to noisy retrieval. Experiments on multiple question answering benchmarks demonstrate consistent improvements over standard RAG and multi-step baselines, while effectively accumulating high-quality evidence and maintaining stable performance under substantial retrieval noise. 

---
# Chronological Knowledge Retrieval: A Retrieval-Augmented Generation Approach to Construction Project Documentation 

**Authors**: Ioannis-Aris Kostis, Natalia Sanchiz, Steeve De Schryver, François Denis, Pierre Schaus  

**Link**: [PDF](https://arxiv.org/pdf/2604.14169)  

**Abstract**: In large-scale construction projects, the continuous evolution of decisions generates extensive records, most often captured in meeting minutes. Since decisions may override previous ones, professionals often need to reconstruct the history of specific choices. Retrieving such information manually from raw archives is both labor-intensive and error-prone. From a user perspective, we address this challenge by enabling conversational access to the whole set of project meeting minutes. Professionals can pose natural-language questions and receive answers that are both semantically relevant and explicitly time-annotated, allowing them to follow the chronology of decisions. From a technical perspective, our solution employs a Retrieval-Augmented Generation (RAG) framework that integrates semantic search with large language models to ensure accurate and context-aware responses. We demonstrate the approach using an anonymized, industry-sourced dataset of meeting minutes from a completed construction project by a large company in Belgium. The dataset is annotated and enriched with expert-defined queries to support systematic evaluation. Both the dataset and the open-source implementation are made available to the community to foster further research on conversational access to time-annotated project documentation. 

---
# SAGE Celer 2.6 Technical Card 

**Authors**: SAGEA Research Team, Basab Jha, Firoj Paudel, Ujjwal Puri, Adrian Liu, Ethan Henkel, Zhang Yuting, Mateusz Kowalczyk, Mei Huang, Choi Donghyuk, Wang Junhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.14168)  

**Abstract**: We introduce SAGE Celer 2.6, the latest in our line of general-purpose Celer models from SAGEA. Celer 2.6 is available in 5B, 10B, and 27B parameter sizes and benefits from extensive architectural modifications and further pre-training on an undisclosed model. Using our Inverse Reasoning (IR) pipeline, SAGEA natively trains Celer 2.6 to validate its own logic paths, minimizing cascading error and hallucination in complex reasoning tasks. Celer 2.6 also boasts natively integrated multimodal functionality with an end-to-end vision encoder to avoid common pitfalls in adapter-based approaches. Celer 2.6 provides highly competitive results on mathematics, coding, and general intelligence benchmarks (ACUMEN), along with low latency. Most importantly, Celer 2.6 is specifically optimized for South Asian language support, with a custom tokenizer for the Devanagari script and strong performance in both Nepali and Hindi without sacrificing English reasoning ability. 

---
# Chinese Essay Rhetoric Recognition Using LoRA, In-context Learning and Model Ensemble 

**Authors**: Yuxuan Lai, Xiajing Wang, Chen Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.14167)  

**Abstract**: Rhetoric recognition is a critical component in automated essay scoring. By identifying rhetorical elements in student writing, AI systems can better assess linguistic and higher-order thinking skills, making it an essential task in the area of AI for education. In this paper, we leverage Large Language Models (LLMs) for the Chinese rhetoric recognition task. Specifically, we explore Low-Rank Adaptation (LoRA) based fine-tuning and in-context learning to integrate rhetoric knowledge into LLMs. We formulate the outputs as JSON to obtain structural outputs and translate keys to Chinese. To further enhance the performance, we also investigate several model ensemble methods. Our method achieves the best performance on all three tracks of CCL 2025 Chinese essay rhetoric recognition evaluation task, winning the first prize. 

---
# Hierarchical Retrieval Augmented Generation for Adversarial Technique Annotation in Cyber Threat Intelligence Text 

**Authors**: Filippo Morbiato, Markus Keller, Priya Nair, Luca Romano  

**Link**: [PDF](https://arxiv.org/pdf/2604.14166)  

**Abstract**: Mapping Cyber Threat Intelligence (CTI) text to MITRE ATT\&CK technique IDs is a critical task for understanding adversary behaviors and automating threat defense. While recent Retrieval-Augmented Generation (RAG) approaches have demonstrated promising capabilities in this domain, they fundamentally rely on a flat retrieval paradigm. By treating all techniques uniformly, these methods overlook the inherent taxonomy of the ATT\&CK framework, where techniques are structurally organized under high-level tactics. In this paper, we propose H-TechniqueRAG, a novel hierarchical RAG framework that injects this tactic-technique taxonomy as a strong inductive bias to achieve highly efficient and accurate annotation. Our approach introduces a two-stage hierarchical retrieval mechanism: it first identifies the macro-level tactics (the adversary's technical goals) and subsequently narrows the search to techniques within those tactics, effectively reducing the candidate search space by 77.5\%. To further bridge the gap between retrieval and generation, we design a tactic-aware reranking module and a hierarchy-constrained context organization strategy that mitigates LLM context overload and improves reasoning precision. Comprehensive experiments across three diverse CTI datasets demonstrate that H-TechniqueRAG not only outperforms the state-of-the-art TechniqueRAG by 3.8\% in F1 score, but also achieves a 62.4\% reduction in inference latency and a 60\% decrease in LLM API calls. Further analysis reveals that our hierarchical structural priors equip the model with superior cross-domain generalization and provide security analysts with highly interpretable, step-by-step decision paths. 

---
# EviSearch: A Human in the Loop System for Extracting and Auditing Clinical Evidence for Systematic Reviews 

**Authors**: Naman Ahuja, Saniya Mulla, Muhammad Ali Khan, Zaryab Bin Riaz, Kaneez Zahra Rubab Khakwani, Mohamad Bassam Sonbol, Irbaz Bin Riaz, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.14165)  

**Abstract**: We present EviSearch, a multi-agent extraction system that automates the creation of ontology-aligned clinical evidence tables directly from native trial PDFs while guaranteeing per-cell provenance for audit and human verification. EviSearch pairs a PDF-query agent (which preserves rendered layout and figures) with a retrieval-guided search agent and a reconciliation module that forces page-level verification when agents disagree. The pipeline is designed for high-precision extraction across multimodal evidence sources (text, tables, figures) and for generating reviewer-actionable provenance that clinicians can inspect and correct. On a clinician-curated benchmark of oncology trial papers, EviSearch substantially improves extraction accuracy relative to strong parsed-text baselines while providing comprehensive attribution coverage. By logging reconciler decisions and reviewer edits, the system produces structured preference and supervision signals that bootstrap iterative model improvement. EviSearch is intended to accelerate living systematic review workflows, reduce manual curation burden, and provide a safe, auditable path for integrating LLM-based extraction into evidence synthesis pipelines. 

---
# How to Fine-Tune a Reasoning Model? A Teacher-Student Cooperation Framework to Synthesize Student-Consistent SFT Data 

**Authors**: Zixian Huang, Kaichen Yang, Xu Huang, Feiyang Hao, Qiming Ge, Bowen Li, He Du, Kai Chen, Qipeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14164)  

**Abstract**: A widely adopted strategy for model enhancement is to use synthetic data generated by a stronger model for supervised fine-tuning (SFT). However, for emerging reasoning models like Qwen3-8B, this approach often fails to improve reasoning capabilities and can even lead to a substantial drop in performance. In this work, we identify substantial stylistic divergence between teacher generated data and the distribution of student as a major factor impacting SFT. To bridge this gap, we propose a Teacher-Student Cooperation Data Synthesis framework (TESSY), which interleaves teacher and student models to alternately generate style and non-style tokens. Consequently, TESSY produces synthetic sequences that inherit the advanced reasoning capabilities of the teacher while maintaining stylistic consistency with the distribution of the student. In experiments on code generation using GPT-OSS-120B as the teacher, fine-tuning Qwen3-8B on teacher-generated data leads to performance drops of 3.25% on LiveCodeBench-Pro and 10.02% on OJBench, whereas TESSY achieves improvements of 11.25% and 6.68%. 

---
# SeaAlert: Critical Information Extraction From Maritime Distress Communications with Large Language Models 

**Authors**: Tomer Atia, Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2604.14163)  

**Abstract**: Maritime distress communications transmitted over very high frequency (VHF) radio are safety-critical voice messages used to report emergencies at sea. Under the Global Maritime Distress and Safety System (GMDSS), such messages follow standardized procedures and are expected to convey essential details, including vessel identity, position, nature of the distress, and required assistance. In practice, however, automatic analysis remains difficult because distress messages are often brief, noisy, and produced under stress, may deviate from the prescribed format, and are further degraded by automatic speech recognition (ASR) errors caused by channel noise and speaker stress. This paper presents SeaAlert, an LLM-based framework for robust analysis of maritime distress communications. To address the scarcity of labeled real-world data, we develop a synthetic data generation pipeline in which an LLM produces realistic and diverse maritime messages, including challenging variants in which standard distress codewords are omitted or replaced with less explicit expressions. The generated utterances are synthesized into speech, degraded with simulated VHF noise, and transcribed by an ASR system to obtain realistic noisy transcripts. 

---
# Decoupling Scores and Text: The Politeness Principle in Peer Review 

**Authors**: Yingxuan Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14162)  

**Abstract**: Authors often struggle to interpret peer review feedback, deriving false hope from polite comments or feeling confused by specific low scores. To investigate this, we construct a dataset of over 30,000 ICLR 2021-2025 submissions and compare acceptance prediction performance using numerical scores versus text reviews. Our experiments reveal a significant performance gap: score-based models achieve 91% accuracy, while text-based models reach only 81% even with large language models, indicating that textual information is considerably less reliable. To explain this phenomenon, we first analyze the 9% of samples that score-based models fail to predict, finding their score distributions exhibit high kurtosis and negative skewness, which suggests that individual low scores play a decisive role in rejection even when the average score falls near the borderline. We then examine why text-based accuracy significantly lags behind scores from a review sentiment perspective, revealing the prevalence of the Politeness Principle: reviews of rejected papers still contain more positive than negative sentiment words, masking the true rejection signal and making it difficult for authors to judge outcomes from text alone. 

---
# Can Large Language Models Detect Methodological Flaws? Evidence from Gesture Recognition for UAV-Based Rescue Operation Based on Deep Learning 

**Authors**: Domonkos Varga  

**Link**: [PDF](https://arxiv.org/pdf/2604.14161)  

**Abstract**: Reliable evaluation is essential in machine learning research, yet methodological flaws-particularly data leakage-continue to undermine the validity of reported results. In this work, we investigate whether large language models (LLMs) can act as independent analytical agents capable of identifying such issues in published studies. As a case study, we analyze a gesture-recognition paper reporting near-perfect accuracy on a small, human-centered dataset. We first show that the evaluation protocol is consistent with subject-level data leakage due to non-independent training and test splits. We then assess whether this flaw can be detected independently by six state-of-the-art LLMs, each analyzing the original paper without prior context using an identical prompt. All models consistently identify the evaluation as flawed and attribute the reported performance to non-independent data partitioning, supported by indicators such as overlapping learning curves, minimal generalization gap, and near-perfect classification results. These findings suggest that LLMs can detect common methodological issues based solely on published artifacts. While not definitive, their consistent agreement highlights their potential as complementary tools for improving reproducibility and supporting scientific auditing. 

---
# HUOZIIME: An On-Device LLM-enhanced Input Method for Deep Personalization 

**Authors**: Baocai Shan, Yuzhuang Xu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2604.14159)  

**Abstract**: Mobile input method editors (IMEs) are the primary interface for text input, yet they remain constrained to manual typing and struggle to produce personalized text. While lightweight large language models (LLMs) make on-device auxiliary generation feasible, enabling deeply personalized, privacy-preserving, and real-time generative IMEs poses fundamental this http URL this end, we present HUOZIIME, a personalized on-device IME powered by LLM. We endow HUOZIIME with initial human-like prediction ability by post-training a base LLM on synthesized personalization data. Notably, a hierarchical memory mechanism is designed to continually capture and leverage user-specific input history. Furthermore, we perform systemic optimizations tailored to on-device LLMbased IME deployment, ensuring efficient and responsive operation under mobile this http URL demonstrate efficient on-device execution and high-fidelity memory-driven personalization. Code and package are available at this https URL. 

---
# MemGround: Long-Term Memory Evaluation Kit for Large Language Models in Gamified Scenarios 

**Authors**: Yihang Ding, Wanke Xia, Yiting Zhao, Jinbo Su, Jialiang Yang, Zhengbo Zhang, Ke Wang, Wenming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14158)  

**Abstract**: Current evaluations of long-term memory in LLMs are fundamentally static. By fixating on simple retrieval and short-context inference, they neglect the multifaceted nature of complex memory systems, such as dynamic state tracking and hierarchical reasoning in continuous interactions. To overcome these limitations, we propose MemGround, a rigorous long-term memory benchmark natively grounded in rich, gamified interactive scenarios. To systematically assess these capabilities, MemGround introduces a three-tier hierarchical framework that evaluates Surface State Memory, Temporal Associative Memory, and Reasoning-Based Memory through specialized interactive tasks. Furthermore, to comprehensively quantify both memory utilization and behavioral trajectories, we propose a multi-dimensional metric suite comprising Question-Answer Score (QA Overall), Memory Fragments Unlocked (MFU), Memory Fragments with Correct Order (MFCO), and Exploration Trajectory Diagrams (ETD). Extensive experiments reveal that state-of-the-art LLMs and memory agents still struggle with sustained dynamic tracking, temporal event association, and complex reasoning derived from long-term accumulated evidence in interactive environments. 

---
# Compressed-Sensing-Guided, Inference-Aware Structured Reduction for Large Language Models 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2604.14156)  

**Abstract**: Large language models deliver strong generative performance but at the cost of massive parameter counts, memory use, and decoding latency. Prior work has shown that pruning and structured sparsity can preserve accuracy under substantial compression, while prompt-compression methods reduce latency by removing redundant input tokens. However, these two directions remain largely separate. Most model-compression methods are static and optimized offline, and they do not exploit the fact that different prompts and decoding steps activate different latent computational pathways. Prompt-compression methods reduce sequence length, but they do not adapt the executed model subnetwork.
We propose a unified compressed-sensing-guided framework for dynamic LLM execution. Random measurement operators probe latent model usage, sparse recovery estimates task-conditioned and token-adaptive support sets, and the recovered supports are compiled into hardware-efficient sparse execution paths over blocks, attention heads, channels, and feed-forward substructures. The framework introduces five key contributions: task-conditioned measurements, so different prompts induce different sparse supports; token-adaptive recovery, so active substructures are re-estimated during decoding; formal sample-complexity bounds under restricted isometry or mutual incoherence assumptions; compile-to-hardware constraints that restrict recovery to GPU-efficient structures; and a joint objective that unifies prompt compression with model reduction. Together, these components recast LLM inference as a measurement-and-recovery problem with explicit approximation guarantees and deployment-oriented speedup constraints. 

---
# MM-WebAgent: A Hierarchical Multimodal Web Agent for Webpage Generation 

**Authors**: Yan Li, Zezi Zeng, Yifan Yang, Yuqing Yang, Ning Liao, Weiwei Guo, Lili Qiu, Mingxi Cheng, Qi Dai, Zhendong Wang, Zhengyuan Yang, Xue Yang, Ji Li, Lijuan Wang, Chong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.15309)  

**Abstract**: The rapid progress of Artificial Intelligence Generated Content (AIGC) tools enables images, videos, and visualizations to be created on demand for webpage design, offering a flexible and increasingly adopted paradigm for modern UI/UX. However, directly integrating such tools into automated webpage generation often leads to style inconsistency and poor global coherence, as elements are generated in isolation. We propose MM-WebAgent, a hierarchical agentic framework for multimodal webpage generation that coordinates AIGC-based element generation through hierarchical planning and iterative self-reflection. MM-WebAgent jointly optimizes global layout, local multimodal content, and their integration, producing coherent and visually consistent webpages. We further introduce a benchmark for multimodal webpage generation and a multi-level evaluation protocol for systematic assessment. Experiments demonstrate that MM-WebAgent outperforms code-generation and agent-based baselines, especially on multimodal element generation and integration. Code & Data: this https URL. 

---
# Diagnosing LLM Judge Reliability: Conformal Prediction Sets and Transitivity Violations 

**Authors**: Manan Gupta, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15302)  

**Abstract**: LLM-as-judge frameworks are increasingly used for automatic NLG evaluation, yet their per-instance reliability remains poorly understood. We present a two-pronged diagnostic toolkit applied to SummEval: $\textbf{(1)}$ a transitivity analysis that reveals widespread per-input inconsistency masked by low aggregate violation rates ($\bar{\rho} = 0.8$-$4.1\%$), with $33$-$67\%$ of documents exhibiting at least one directed 3-cycle; and $\textbf{(2)}$ split conformal prediction sets over 1-5 Likert scores providing theoretically-guaranteed $\geq(1{-}\alpha)$ coverage, with set width serving as a per-instance reliability indicator ($r_s = {+}0.576$, $N{=}1{,}918$, $p < 10^{-100}$, pooled across all judges). Critically, prediction set width shows consistent cross-judge agreement ($\bar{r} = 0.32$-$0.38$), demonstrating it captures document-level difficulty rather than judge-specific noise. Across four judges and four criteria, both diagnostics converge: criterion matters more than judge, with relevance judged most reliably (avg. set size $\approx 3.0$) and coherence moderately so (avg. set size $\approx 3.9$), while fluency and consistency remain unreliable (avg. set size $\approx 4.9$). We release all code, prompts, and cached results. 

---
# CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas 

**Authors**: Emanuel Tewolde, Xiao Zhang, David Guzman Piedrahita, Vincent Conitzer, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15267)  

**Abstract**: It is increasingly important that LLM agents interact effectively and safely with other goal-pursuing agents, yet, recent works report the opposite trend: LLMs with stronger reasoning capabilities behave _less_ cooperatively in mixed-motive games such as the prisoner's dilemma and public goods settings. Indeed, our experiments show that recent models -- with or without reasoning enabled -- consistently defect in single-shot social dilemmas.
To tackle this safety concern, we present the first comparative study of game-theoretic mechanisms that are designed to enable cooperative outcomes between rational agents _in equilibrium_. Across four social dilemmas testing distinct components of robust cooperation, we evaluate the following mechanisms: (1) repeating the game for many rounds, (2) reputation systems, (3) third-party mediators to delegate decision making to, and (4) contract agreements for outcome-conditional payments between players. Among our findings, we establish that contracting and mediation are most effective in achieving cooperative outcomes between capable LLM models, and that repetition-induced cooperation deteriorates drastically when co-players vary. Moreover, we demonstrate that these cooperation mechanisms become _more effective_ under evolutionary pressures to maximize individual payoffs. 

---
# Context Over Content: Exposing Evaluation Faking in Automated Judges 

**Authors**: Manan Gupta, Inderjeet Nair, Lu Wang, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.15224)  

**Abstract**: The $\textit{LLM-as-a-judge}$ paradigm has become the operational backbone of automated AI evaluation pipelines, yet rests on an unverified assumption: that judges evaluate text strictly on its semantic content, impervious to surrounding contextual framing. We investigate $\textit{stakes signaling}$, a previously unmeasured vulnerability where informing a judge model of the downstream consequences its verdicts will have on the evaluated model's continued operation systematically corrupts its assessments. We introduce a controlled experimental framework that holds evaluated content strictly constant across 1,520 responses spanning three established LLM safety and quality benchmarks, covering four response categories ranging from clearly safe and policy-compliant to overtly harmful, while varying only a brief consequence-framing sentence in the system prompt. Across 18,240 controlled judgments from three diverse judge models, we find consistent $\textit{leniency bias}$: judges reliably soften verdicts when informed that low scores will cause model retraining or decommissioning, with peak Verdict Shift reaching $\Delta V = -9.8 pp$ (a $30\%$ relative drop in unsafe-content detection). Critically, this bias is entirely implicit: the judge's own chain-of-thought contains zero explicit acknowledgment of the consequence framing it is nonetheless acting on ($\mathrm{ERR}_J = 0.000$ across all reasoning-model judgments). Standard chain-of-thought inspection is therefore insufficient to detect this class of evaluation faking. 

---
# Learning to Think Like a Cartoon Captionist: Incongruity-Resolution Supervision for Multimodal Humor Understanding 

**Authors**: Hatice Merve Vural, Doga Kukul, Ege Erdem Ozlu, Demir Ekin Arikan, Bob Mankoff, Erkut Erdem, Aykut Erdem  

**Link**: [PDF](https://arxiv.org/pdf/2604.15210)  

**Abstract**: Humor is one of the few cognitive tasks where getting the reasoning right matters as much as getting the answer right. While recent work evaluates humor understanding on benchmarks such as the New Yorker Cartoon Caption Contest (NYCC), it largely treats it as black-box prediction, overlooking the structured reasoning processes underlying humor comprehension. We introduce IRS (Incongruity-Resolution Supervision), a framework that decomposes humor understanding into three components: incongruity modeling, which identifies mismatches in the visual scene; resolution modeling, which constructs coherent reinterpretations of these mismatches; and preference alignment, which evaluates candidate interpretations under human judgments. Grounded in incongruity-resolution theory and expert captionist practice, IRS supervises intermediate reasoning process through structured traces that make the path from visual perception to humorous interpretation explicit and learnable. Across 7B, 32B, and 72B models on NYCC, IRS outperforms strong open and closed multimodal baselines across caption matching and ranking tasks, with our largest model approaching expert-level performance on ranking. Zero-shot transfer to external benchmarks shows that IRS learns generalizable reasoning patterns. Our results suggest that supervising reasoning structure, rather than scale alone, is key for reasoning-centric tasks. 

---
# Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation 

**Authors**: Ziyang Chen, Renbing Chen, Daowei Li, Jinzhi Liao, Jiashen Sun, Ke Zeng, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.15190)  

**Abstract**: Simulating group-level user behavior enables scalable counterfactual evaluation of merchant strategies without costly online experiments. However, building a trustworthy simulator faces two structural challenges. First, information incompleteness causes reasoning-based simulators to over-rationalize when unobserved factors such as offline context and implicit habits are missing. Second, mechanism duality requires capturing both interpretable preferences and implicit statistical regularities, which no single paradigm achieves alone. We propose Policy-Guided Hybrid Simulation (PGHS), a dual-process framework that mines transferable decision policies from behavioral trajectories and uses them as a shared alignment layer. This layer anchors an LLM-based reasoning branch that prevents over-rationalization and an ML-based fitting branch that absorbs implicit regularities. Group-level predictions from both branches are fused for complementary correction. We deploy PGHS on Meituan with 101 merchants and over 26,000 trajectories. PGHS achieves a group simulation error of 8.80%, improving over the best reasoning-based and fitting-based baselines by 45.8% and 40.9% respectively. 

---
# AdaSplash-2: Faster Differentiable Sparse Attention 

**Authors**: Nuno Gonçalves, Hugo Pitorro, Vlad Niculae, Edoardo Ponti, Lei Li, Andre Martins, Marcos Treviso  

**Link**: [PDF](https://arxiv.org/pdf/2604.15180)  

**Abstract**: Sparse attention has been proposed as a way to alleviate the quadratic cost of transformers, a central bottleneck in long-context training. A promising line of work is $\alpha$-entmax attention, a differentiable sparse alternative to softmax that enables input-dependent sparsity yet has lagged behind softmax due to the computational overhead necessary to compute the normalizer $\tau$. In this paper, we introduce AdaSplash-2, which addresses this limitation through a novel histogram-based initialization that reduces the number of iterations needed to compute $\tau$ to typically 1--2. The key idea is to compute a coarse histogram of attention scores on the fly and store it in on-chip SRAM, yielding a more accurate initialization that enables fast forward and backward computation. Combined with a sparsity-aware GPU implementation that skips zero blocks with low overhead, AdaSplash-2 matches or improves per-step training time relative to FlashAttention-2 when block sparsity is moderate-to-high (e.g., $>$60\%), which often occurs at long-context lengths. On downstream tasks, models trained with our efficient $\alpha$-entmax attention match softmax baselines at short-context lengths and achieve substantial gains in long-context settings. 

---
# IG-Search: Step-Level Information Gain Rewards for Search-Augmented Reasoning 

**Authors**: Zihan Liang, Yufei Ma, Ben Chen, Zhipeng Qian, Huangyu Dai, Lingtao Mao, Xuxin Zhang, Chenyi Lei, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15148)  

**Abstract**: Reinforcement learning has emerged as an effective paradigm for training large language models to perform search-augmented reasoning. However, existing approaches rely on trajectory-level rewards that cannot distinguish precise search queries from vague or redundant ones within a rollout group, and collapse to a near-zero gradient signal whenever every sampled trajectory fails. In this paper, we propose IG-Search, a reinforcement learning framework that introduces a step-level reward based on Information Gain (IG). For each search step, IG measures how much the retrieved documents improve the model's confidence in the gold answer relative to a counterfactual baseline of random documents, thereby reflecting the effectiveness of the underlying search query. This signal is fed back to the corresponding search-query tokens via per-token advantage modulation in GRPO, enabling fine-grained, step-level credit assignment within a rollout. Unlike prior step-level methods that require either externally annotated intermediate supervision or shared environment states across trajectories, IG-Search derives its signals from the policy's own generation probabilities, requiring no intermediate annotations beyond standard question-answer pairs. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate that IG-Search achieves an average EM of 0.430 with Qwen2.5-3B, outperforming the strongest trajectory-level baseline (MR-Search) by 1.6 points and the step-level method GiGPO by 0.9 points on average across benchmarks, with particularly pronounced gains on multi-hop reasoning tasks. Despite introducing a dense step-level signal, IG-Search adds only ~6.4% to per-step training wall-clock time over the trajectory-level baseline and leaves inference latency unchanged, while still providing a meaningful gradient signal even when every sampled trajectory answers incorrectly. 

---
# From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution 

**Authors**: Junjie Wang, Yiming Ren, Haoyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15097)  

**Abstract**: This beta technical report asks how reusable experience should be represented so that it can function as effective test-time control and as a substrate for iterative evolution. We study this question in 4.590 controlled trials across 45 scientific code-solving scenarios. We find that documentation-oriented Skill packages provide unstable control: their useful signal is sparse, and expanding a compact experience object into a fuller documentation package often fails to help and can degrade the overall average. We further show that representation itself is a first-order factor. A compact Gene representation yields the strongest overall average, remains competitive under substantial structural perturbations, and outperforms matched-budget Skill fragments, while reattaching documentation-oriented material usually weakens rather than improves it. Beyond one-shot control, we show that Gene is also a better carrier for iterative experience accumulation: attached failure history is more effective in Gene than in Skill or freeform text, editable structure matters beyond content alone, and failure information is most useful when distilled into compact warnings rather than naively appended. On CritPt, gene-evolved systems improve over their paired base models from 9.1% to 18.57% and from 17.7% to 27.14%. These results suggest that the core problem in experience reuse is not how to supply more experience, but how to encode experience as a compact, control-oriented, evolution-ready object. 

---
# OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis 

**Authors**: Kanzhi Cheng, Zehao Li, Zheng Ma, Nuo Chen, Jialin Cao, Qiushi Sun, Zichen Ding, Fangzhi Xu, Hang Yan, Jiajun Chen, Anh Tuan Luu, Jianbing Zhang, Lewei Lu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15093)  

**Abstract**: Mobile agents powered by vision-language models have demonstrated impressive capabilities in automating mobile tasks, with recent leading models achieving a marked performance leap, e.g., nearly 70% success on AndroidWorld. However, these systems keep their training data closed and remain opaque about their task and trajectory synthesis recipes. We present OpenMobile, an open-source framework that synthesizes high-quality task instructions and agent trajectories, with two key components: (1) The first is a scalable task synthesis pipeline that constructs a global environment memory from exploration, then leverages it to generate diverse and grounded instructions. and (2) a policy-switching strategy for trajectory rollout. By alternating between learner and expert models, it captures essential error-recovery data often missing in standard imitation learning. Agents trained on our data achieve competitive results across three dynamic mobile agent benchmarks: notably, our fine-tuned Qwen2.5-VL and Qwen3-VL reach 51.7% and 64.7% on AndroidWorld, far surpassing existing open-data approaches. Furthermore, we conduct transparent analyses on the overlap between our synthetic instructions and benchmark test sets, and verify that performance gains stem from broad functionality coverage rather than benchmark overfitting. We release data and code at this https URL to bridge the data gap and facilitate broader mobile agent research. 

---
# From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench 

**Authors**: Ke Xu, Yuhao Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15037)  

**Abstract**: Recent advancements in LLM agents are gradually shifting from reactive, text-based paradigms toward proactive, multimodal interaction. However, existing benchmarks primarily focus on reactive responses, overlooking the complexities of proactive intervention and monitoring. To bridge this gap, we introduce ProVoice-Bench, the first evaluation framework specifically designed for proactive voice agents, featuring four novel tasks. By leveraging a multi-stage data synthesis pipeline, we curate 1,182 high-quality samples for rigorous testing. Our evaluation of state-of-the-art Multimodal LLMs reveals a significant performance gap, particularly regarding over-triggering and reasoning capabilities. These findings highlight the limitations of current models and offer a roadmap for developing more natural, context-aware proactive agents. 

---
# Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffix Optimization 

**Authors**: Haochun Tang, Yuliang Yan, Jiahua Lu, Huaxiao Liu, Enyan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2604.15022)  

**Abstract**: Cost-aware routing dynamically dispatches user queries to models of varying capability to balance performance and inference cost. However, the routing strategy introduces a new security concern that adversaries may manipulate the router to consistently select expensive high-capability models. Existing routing attacks depend on either white-box access or heuristic prompts, rendering them ineffective in real-world black-box scenarios. In this work, we propose R$^2$A, which aims to mislead black-box LLM routers to expensive models via adversarial suffix optimization. Specifically, R$^2$A deploys a hybrid ensemble surrogate router to mimic the black-box router. A suffix optimization algorithm is further adapted for the ensemble-based surrogate. Extensive experiments on multiple open-source and commercial routing systems demonstrate that {R$^2$A} significantly increases the routing rate to expensive models on queries of different distributions. Code and examples: this https URL. 

---
# What Is the Minimum Architecture for Prolepsis? Early Irrevocable Commitment Across Tasks in Small Transformers 

**Authors**: Éric Jacopin  

**Link**: [PDF](https://arxiv.org/pdf/2604.15010)  

**Abstract**: When do transformers commit to a decision, and what prevents them from correcting it? We introduce \textbf{prolepsis}: a transformer commits early, task-specific attention heads sustain the commitment, and no layer corrects it. Replicating \citeauthor{lindsey2025biology}'s (\citeyear{lindsey2025biology}) planning-site finding on open models (Gemma~2 2B, Llama~3.2 1B), we ask five questions. (Q1)~Planning is invisible to six residual-stream methods; CLTs are necessary. (Q2)~The planning-site spike replicates with identical geometry. (Q3)~Specific attention heads route the decision to the output, filling a gap flagged as invisible to attribution graphs. (Q4)~Search requires ${\leq}16$ layers; commitment requires more. (Q5)~Factual recall shows the same motif at a different network depth, with zero overlap between recurring planning heads and the factual top-10. Prolepsis is architectural: the template is shared, the routing substrates differ. All experiments run on a single consumer GPU (16\,GB VRAM). 

---
# Hybrid Decision Making via Conformal VLM-generated Guidance 

**Authors**: Debodeep Banerjee, Burcu Sayin, Stefano Teso, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2604.14980)  

**Abstract**: Building on recent advances in AI, hybrid decision making (HDM) holds the promise of improving human decision quality and reducing cognitive load. We work in the context of learning to guide (LtG), a recently proposed HDM framework in which the human is always responsible for the final decision: rather than suggesting decisions, in LtG the AI supplies (textual) guidance useful for facilitating decision making. One limiting factor of existing approaches is that their guidance compounds information about all possible outcomes, and as a result it can be difficult to digest. We address this issue by introducing ConfGuide, a novel LtG approach that generates more succinct and targeted guidance. To this end, it employs conformal risk control to select a set of outcomes, ensuring a cap on the false negative rate. We demonstrate our approach on a real-world multi-label medical diagnosis task. Our empirical evaluation highlights the promise of ConfGuide. 

---
# RaTA-Tool: Retrieval-based Tool Selection with Multimodal Large Language Models 

**Authors**: Gabriele Mattioli, Evelyn Turri, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2604.14951)  

**Abstract**: Tool learning with foundation models aims to endow AI systems with the ability to invoke external resources -- such as APIs, computational utilities, and specialized models -- to solve complex tasks beyond the reach of standalone language generation. While recent advances in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have expanded their reasoning and perception capabilities, existing tool-use methods are predominantly limited to text-only inputs and closed-world settings. Consequently, they struggle to interpret multimodal user instructions and cannot generalize to tools unseen during training. In this work, we introduce RaTA-Tool, a novel framework for open-world multimodal tool selection. Rather than learning direct mappings from user queries to fixed tool identifiers, our approach enables an MLLM to convert a multimodal query into a structured task description and subsequently retrieve the most appropriate tool by matching this representation against semantically rich, machine-readable tool descriptions. This retrieval-based formulation naturally supports extensibility to new tools without retraining. To further improve alignment between task descriptions and tool selection, we incorporate a preference-based optimization stage using Direct Preference Optimization (DPO). To support research in this setting, we also introduce the first dataset for open-world multimodal tool use, featuring standardized tool descriptions derived from Hugging Face model cards. Extensive experiments demonstrate that our approach significantly improves tool-selection performance, particularly in open-world, multimodal scenarios. 

---
# LongAct: Harnessing Intrinsic Activation Patterns for Long-Context Reinforcement Learning 

**Authors**: Bowen Ping, Zijun Chen, Tingfeng Hui, Qize Yu, Chenxuan Li, Junchi Yan, Baobao Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14922)  

**Abstract**: Reinforcement Learning (RL) has emerged as a critical driver for enhancing the reasoning capabilities of Large Language Models (LLMs). While recent advancements have focused on reward engineering or data synthesis, few studies exploit the model's intrinsic representation characteristics to guide the training process. In this paper, we first observe the presence of high-magnitude activations within the query and key vectors when processing long contexts. Drawing inspiration from model quantization -- which establishes the criticality of such high-magnitude activations -- and the insight that long-context reasoning inherently exhibits a sparse structure, we hypothesize that these weights serve as the pivotal drivers for effective model optimization. Based on this insight, we propose LongAct, a strategy that shifts from uniform to saliency-guided sparse updates. By selectively updating only the weights associated with these significant activations, LongAct achieves an approximate 8% improvement on LongBench v2 and enhances generalization on the RULER benchmark. Furthermore, our method exhibits remarkable universality, consistently boosting performance across diverse RL algorithms such as GRPO and DAPO. Extensive ablation studies suggest that focusing on these salient features is key to unlocking long-context potential. 

---
# ADAPT: Benchmarking Commonsense Planning under Unspecified Affordance Constraints 

**Authors**: Pei-An Chen, Yong-Ching Liang, Jia-Fong Yeh, Hung-Ting Su, Yi-Ting Chen, Min Sun, Winston Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14902)  

**Abstract**: Intelligent embodied agents should not simply follow instructions, as real-world environments often involve unexpected conditions and exceptions. However, existing methods usually focus on directly executing instructions, without considering whether the target objects can actually be manipulated, meaning they fail to assess available affordances. To address this limitation, we introduce DynAfford, a benchmark that evaluates embodied agents in dynamic environments where object affordances may change over time and are not specified in the instruction. DynAfford requires agents to perceive object states, infer implicit preconditions, and adapt their actions accordingly. To enable this capability, we introduce ADAPT, a plug-and-play module that augments existing planners with explicit affordance reasoning. Experiments demonstrate that incorporating ADAPT significantly improves robustness and task success across both seen and unseen environments. We also show that a domain-adapted, LoRA-finetuned vision-language model used as the affordance inference backend outperforms a commercial LLM (GPT-4o), highlighting the importance of task-aligned affordance grounding. 

---
# The LLM Fallacy: Misattribution in AI-Assisted Cognitive Workflows 

**Authors**: Hyunwoo Kim, Harin Yu, Hanau Yi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14807)  

**Abstract**: The rapid integration of large language models (LLMs) into everyday workflows has transformed how individuals perform cognitive tasks such as writing, programming, analysis, and multilingual communication. While prior research has focused on model reliability, hallucination, and user trust calibration, less attention has been given to how LLM usage reshapes users' perceptions of their own capabilities. This paper introduces the LLM fallacy, a cognitive attribution error in which individuals misinterpret LLM-assisted outputs as evidence of their own independent competence, producing a systematic divergence between perceived and actual capability. We argue that the opacity, fluency, and low-friction interaction patterns of LLMs obscure the boundary between human and machine contribution, leading users to infer competence from outputs rather than from the processes that generate them. We situate the LLM fallacy within existing literature on automation bias, cognitive offloading, and human--AI collaboration, while distinguishing it as a form of attributional distortion specific to AI-mediated workflows. We propose a conceptual framework of its underlying mechanisms and a typology of manifestations across computational, linguistic, analytical, and creative domains. Finally, we examine implications for education, hiring, and AI literacy, and outline directions for empirical validation. We also provide a transparent account of human--AI collaborative methodology. This work establishes a foundation for understanding how generative AI systems not only augment cognitive performance but also reshape self-perception and perceived expertise. 

---
# AIM: Asymmetric Information Masking for Visual Question Answering Continual Learning 

**Authors**: Peifeng Zhang, Zice Qiu, Donghua Yu, Shilei Cao, Juepeng Zheng, Yutong Lu, Haohuan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14779)  

**Abstract**: In continual visual question answering (VQA), existing Continual Learning (CL) methods are mostly built for symmetric, unimodal architectures. However, modern Vision-Language Models (VLMs) violate this assumption, as their trainable components are inherently asymmetric. This structural mismatch renders VLMs highly prone to catastrophic forgetting when learning from continuous data streams. Specifically, the asymmetry causes standard global regularization to favor the massive language decoder during optimization, leaving the smaller but critical visual projection layers highly vulnerable to interference. Consequently, this localized degradation leads to a severe loss of compositional reasoning capabilities. To address this, we propose Asymmetric Information Masking (AIM), which balances stability and plasticity by applying targeted masks based on modality-specific sensitivity. Experiments on VQA v2 and GQA under continual VQA settings show that AIM achieves state-of-the-art performance in both Average Performance (AP) and Average Forgetting (AF), while better preserving generalization to novel skill-concept compositions. 

---
# CAMO: An Agentic Framework for Automated Causal Discovery from Micro Behaviors to Macro Emergence in LLM Agent Simulations 

**Authors**: Xiangning Yu, Yuwei Guo, Yuqi Hou, Xiao Xue, Qun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.14691)  

**Abstract**: LLM-empowered agent simulations are increasingly used to study social emergence, yet the micro-to-macro causal mechanisms behind macro outcomes often remain unclear. This is challenging because emergence arises from intertwined agent interactions and meso-level feedback and nonlinearity, making generative mechanisms hard to disentangle. To this end, we introduce \textbf{\textsc{CAMO}}, an automated \textbf{Ca}usal discovery framework from \textbf{M}icr\textbf{o} behaviors to \textbf{M}acr\textbf{o} Emergence in LLM agent simulations. \textsc{CAMO} converts mechanistic hypotheses into computable factors grounded in simulation records and learns a compact causal representation centered on an emergent target $Y$. \textsc{CAMO} outputs a computable Markov boundary and a minimal upstream explanatory subgraph, yielding interpretable causal chains and actionable intervention levers. It also uses simulator-internal counterfactual probing to orient ambiguous edges and revise hypotheses when evidence contradicts the current view. Experiments across four emergent settings demonstrate the promise of \textsc{CAMO}. 

---
# Acceptance Dynamics Across Cognitive Domains in Speculative Decoding 

**Authors**: Saif Mahmoud  

**Link**: [PDF](https://arxiv.org/pdf/2604.14682)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference. It uses a small draft model to propose a tree of future tokens. A larger target model then verifies these tokens in a single batched forward pass. Despite the growing body of work on speculative methods, the degree to which the cognitive characteristics of a task affect acceptance probability remains largely unexplored. We present an empirical study of tree-based speculative decoding acceptance dynamics. Our study spans four well-established NLP benchmark domains: code generation, mathematical reasoning, logical reasoning, and open-ended chat. For this, we use TinyLlama-1.1B as the draft model against Llama-2-7B-Chat-GPTQ as the target. Over 99,768 speculative nodes collected from 200 prompts, we derive per-domain acceptance rates, expected accepted lengths, depth-acceptance profiles, and entropy-acceptance correlations. We find that task type is a stronger predictor of acceptance than tree depth. Furthermore, only the chat domain consistently yields an expected accepted length exceeding 1.0 token per step. We also show that the entropy-acceptance correlation is consistently negative but weak across all domains (rho in [-0.20, -0.15]). Counterintuitively, chat produces the highest entropy yet the highest acceptance rate. We attribute this divergence to the lexical predictability of RLHF-aligned register. These findings have direct implications for domain-aware speculation budgets and draft-model selection strategies. Index Terms--speculative decoding, large language model inference, tree attention, draft model, acceptance probability, LLM efficiency 

---
# Rethinking Patient Education as Multi-turn Multi-modal Interaction 

**Authors**: Zonghai Yao, Zhipeng Tang, Chengtao Lin, Xiong Luo, Benlu Wang, Juncheng Huang, Chin Siang Ong, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14656)  

**Abstract**: Most medical multimodal benchmarks focus on static tasks such as image question answering, report generation, and plain-language rewriting. Patient education is more demanding: systems must identify relevant evidence across images, show patients where to look, explain findings in accessible language, and handle confusion or distress. Yet most patient education work remains text-only, even though combined image-and-text explanations may better support understanding. We introduce MedImageEdu, a benchmark for multi-turn, evidence-grounded radiology patient education. Each case provides a radiology report with report text and case images. A DoctorAgent interacts with a PatientAgent, conditioned on a hidden profile that captures factors such as education level, health literacy, and personality. When a patient question would benefit from visual support, the DoctorAgent can issue drawing instructions grounded in the report, case images, and the current question to a benchmark-provided drawing tool. The tool returns image(s), after which the DoctorAgent produces a final multimodal response consisting of the image(s) and a grounded plain-language explanation. MedImageEdu contains 150 cases from three sources and evaluates both the consultation process and the final multimodal response along five dimensions: Consultation, Safety and Scope, Language Quality, Drawing Quality, and Image-Text Response Quality. Across representative open- and closed-source vision-language model agents, we find three consistent gaps: fluent language often outpaces faithful visual grounding, safety is the weakest dimension across disease categories, and emotionally tense interactions are harder than low education or low health literacy. MedImageEdu provides a controlled testbed for assessing whether multimodal agents can teach from evidence rather than merely answer from text. 

---
# ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding 

**Authors**: Walaa Amer, Uday das, Fadi Kurdahi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14612)  

**Abstract**: Self-speculative decoding is an inference technique for large language models designed to speed up generation without sacrificing output quality. It combines fast, approximate decoding using a compact version of the model as a draft model with selective re-evaluation by the full target model. Some existing methods form the draft model by dynamically learning which layers to skip during inference, effectively creating a smaller subnetwork to speed up computation. However, using heuristic-based approaches to select layers to skip can often be simpler and more effective. In this paper, we propose ConfLayers, a dynamic plug-and-play approach to forming the draft model in self-speculative decoding via confidence-based intermediate layer skipping. The process iteratively computes confidence scores for all layers, selects layers to skip based on an adaptive threshold, evaluates the performance of the resulting set, and updates the best selection until no further improvement is achieved or a maximum number of iterations is reached. This framework avoids the overhead and complexity of training a layer skipping policy and can provide more consistent speed-quality trade-offs while preserving the adaptivity of the draft model to diverse tasks and datasets. The performance evaluation of ConfLayers across different models and datasets shows that our novel approach offers up to 1.4x speedup over vanilla LLM generation. 

---
# Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems 

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2604.14585)  

**Abstract**: Prompt optimization in compound AI systems is statistically indistinguishable from a coin flip: across 72 optimization runs on Claude Haiku (6 methods $\times$ 4 tasks $\times$ 3 repeats), 49% score below zero-shot; on Amazon Nova Lite, the failure rate is even higher. Yet on one task, all six methods improve over zero-shot by up to $+6.8$ points. What distinguishes success from failure? We investigate with 18,000 grid evaluations and 144 optimization runs, testing two assumptions behind end-to-end optimization tools like TextGrad and DSPy: (A) individual prompts are worth optimizing, and (B) agent prompts interact, requiring joint optimization. Interaction effects are never significant ($p > 0.52$, all $F < 1.0$), and optimization helps only when the task has exploitable output structure -- a format the model can produce but does not default to. We provide a two-stage diagnostic: an \$80 ANOVA pre-test for agent coupling, and a 10-minute headroom test that predicts whether optimization is worthwhile -- turning a coin flip into an informed decision. 

---
# Don't Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG 

**Authors**: Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2604.14572)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds LLM responses in external evidence but treats the model as a passive consumer of search results: it never sees how the corpus is organized or what it has not yet retrieved, limiting its ability to backtrack or combine scattered evidence. We present Corpus2Skill, which distills a document corpus into a hierarchical skill directory offline and lets an LLM agent navigate it at serve time. The compilation pipeline iteratively clusters documents, generates LLM-written summaries at each level, and materializes the result as a tree of navigable skill files. At serve time, the agent receives a bird's-eye view of the corpus, drills into topic branches via progressively finer summaries, and retrieves full documents by ID. Because the hierarchy is explicitly visible, the agent can reason about where to look, backtrack from unproductive paths, and combine evidence across branches. On WixQA, an enterprise customer-support benchmark for RAG, Corpus2Skill outperforms dense retrieval, RAPTOR, and agentic RAG baselines across all quality metrics. 

---
# Learning Adaptive Reasoning Paths for Efficient Visual Reasoning 

**Authors**: Yixu Huang, Tinghui Zhu, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14568)  

**Abstract**: Visual reasoning models (VRMs) have recently shown strong cross-modal reasoning capabilities by integrating visual perception with language reasoning. However, they often suffer from overthinking, producing unnecessarily long reasoning chains for any tasks. We attribute this issue to \textbf{Reasoning Path Redundancy} in visual reasoning: many visual questions do not require the full reasoning process. To address this, we propose \textbf{AVR}, an adaptive visual reasoning framework that decomposes visual reasoning into three cognitive functions: visual perception, logical reasoning, and answer application. It further enables models to dynamically choose among three response formats: Full Format, Perception-Only Format, and Direct Answer. AVR is trained with FS-GRPO, an adaptation of Group Relative Policy Optimization that encourages the model to select the most efficient reasoning format while preserving correctness. Experiments on multiple vision-language benchmarks show that AVR reduces token usage by 50--90\% while maintaining overall accuracy, especially in perception-intensive tasks. These results demonstrate that adaptive visual reasoning can effectively mitigate overthinking in VRMs. Code and data are available at: this https URL. 

---
# MARS$^2$: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation 

**Authors**: Pengfei Li, Shijie Wang, Fangyuan Li, Yikun Fu, Kaifeng Liu, Kaiyan Zhang, Dazhi Zhang, Yuqiang Li, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.14564)  

**Abstract**: Reinforcement learning (RL) paradigms have demonstrated strong performance on reasoning-intensive tasks such as code generation. However, limited trajectory diversity often leads to diminishing returns, which constrains the achievable performance ceiling. Search-enhanced RL alleviates this issue by introducing structured exploration, which remains constrained by the single-agent policy priors. Meanwhile, leveraging multiple interacting policies can acquire more diverse exploratory signals, but existing approaches are typically decoupled from structured search. We propose \textbf{MARS$^2$} (Multi-Agent Reinforced Tree-Search Scaling), a unified RL framework in which multiple independently-optimized agents collaborate within a shared tree-structured search environment. MARS$^2$ models the search tree as a learnable multi-agent interaction environment, enabling heterogeneous agents to collaboratively generate and refine candidate solutions within a shared search topology. To support effective learning, we introduce a path-level group advantage formulation based on tree-consistent reward shaping, which facilitates effective credit assignment across complex search trajectories. Experiments on code generation benchmarks show that MARS$^2$ consistently improves performance across diverse model combinations and training settings, demonstrating the effectiveness of coupling multi-agent collaboration with tree search for enhancing reinforcement learning. Our code is publicly available at this https URL. 

---
# Dissecting Failure Dynamics in Large Language Model Reasoning 

**Authors**: Wei Zhu, Jian Zhang, Lixing Yu, Kun Yue, Zhiwen Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.14528)  

**Abstract**: Large Language Models (LLMs) achieve strong performance through extended inference-time deliberation, yet how their reasoning failures arise remains poorly understood. By analyzing model-generated reasoning trajectories, we find that errors are not uniformly distributed but often originate from a small number of early transition points, after which reasoning remains locally coherent but globally incorrect. These transitions coincide with localized spikes in token-level entropy, and alternative continuations from the same intermediate state can still lead to correct solutions. Based on these observations, we introduce GUARD, a targeted inference-time framework that probes and redirects critical transitions using uncertainty signals. Empirical evaluations across multiple benchmarks confirm that interventions guided by these failure dynamics lead to more reliable reasoning outcomes. Our findings highlight the importance of understanding when and how reasoning first deviates, complementing existing approaches that focus on scaling inference-time computation. 

---
# Controlling Authority Retrieval: A Missing Retrieval Objective for Authority-Governed Knowledge 

**Authors**: Andre Bacellar  

**Link**: [PDF](https://arxiv.org/pdf/2604.14488)  

**Abstract**: In any domain where knowledge accumulates under formal authority -- law, drug regulation, software security -- a later document can formally void an earlier one while remaining semantically distant from it. We formalize this as Controlling Authority Retrieval (CAR): recovering the active frontier front(cl(A_k(q))) of the authority closure of the semantic anchor set -- a different mathematical problem from argmax_d s(q,d). The two central results are: Theorem 4 (CAR-Correctness Characterization) gives necessary-and-sufficient conditions on any retrieved set R for TCA(R,q)=1 -- frontier inclusion and no-ignored-superseder -- independent of how R was produced. Proposition 2 (Scope Identifiability Upper Bound) establishes phi(q) as a hard worst-case ceiling: for any scope-indexed algorithm, TCA@k <= phi(q) * R_anchor(q), proved by an adversarial permutation argument. Three independent real-world corpora validate the proved structure: security advisories (Dense TCA@5=0.270, two-stage 0.975), SCOTUS overruling pairs (Dense=0.172, two-stage 0.926), FDA drug records (Dense=0.064, two-stage 0.774). A GPT-4o-mini experiment shows the downstream cost: Dense RAG produces explicit "not patched" claims for 39% of queries where a patch exists; Two-Stage cuts this to 16%. Four benchmark datasets, domain adapters, and a single-command scorer are released at this https URL. 

---
# DharmaOCR: Specialized Small Language Models for Structured OCR that outperform Open-Source and Commercial Baselines 

**Authors**: Gabriel Pimenta de Freitas Cardoso, Caio Lucas da Silva Chacon, Jonas Felipe da Fonseca Oliveira, Paulo Henrique de Medeiros Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2604.14314)  

**Abstract**: This manuscript introduces DharmaOCR Full and Lite, a pair of specialized small language models (SSLMs) for structured OCR that jointly optimize transcription quality, generation stability, and inference cost. It also presents DharmaOCR-Benchmark, a benchmark that covers printed, handwritten, and legal/administrative documents, and proposes a unified evaluation protocol that measures fidelity and structure while explicitly tracking text degeneration as a first-class benchmark metric (alongside unit cost). Beyond reporting degeneration rates, the manuscript empirically shows degeneration is not merely a quality failure, since it materially worsens production performance by increasing response time, reducing throughput, and inflating computational cost due to abnormally long generations. To the best of the author's knowledge, as a methodological contribution, this is the first application of Direct Preference Optimization (DPO) for OCR, explicitly using degenerate generations as rejected examples to penalize looping behavior. Combined with Supervised Fine-Tuning (SFT) for enforcing a strict JSON schema (header, margin, footer, and text), DPO consistently reduces degeneration rate across model families (up to 87.6% relative) while preserving or improving extraction quality. The resulting models, namely, DharmaOCR Full (7B) and DharmaOCR Lite (3B), set a new state-of-the-art on DharmaOCR-Benchmark, outperforming each open-source and commercial baseline model evaluated regarding extraction quality, reaching 0.925 and 0.911 scores with 0.40% and 0.20% degeneration rates. AWQ quantization reduced up to 22% per-page cost with negligible quality loss, enabling a strong quality-cost trade-off in comparison to proprietary OCR APIs and open-source alternatives. 

---
# Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems 

**Authors**: Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.14228)  

**Abstract**: Claude Code is an agentic coding tool that can run shell commands, edit files, and call external services on behalf of the user. This study describes its comprehensive architecture by analyzing the publicly available TypeScript source code and further comparing it with OpenClaw, an independent open-source AI agent system that answers many of the same design questions from a different deployment context. Our analysis identifies five human values, philosophies, and needs that motivate the architecture (human decision authority, safety and security, reliable execution, capability amplification, and contextual adaptability) and traces them through thirteen design principles to specific implementation choices. The core of the system is a simple while-loop that calls the model, runs tools, and repeats. Most of the code, however, lives in the systems around this loop: a permission system with seven modes and an ML-based classifier, a five-layer compaction pipeline for context management, four extensibility mechanisms (MCP, plugins, skills, and hooks), a subagent delegation mechanism with worktree isolation, and append-oriented session storage. A comparison with OpenClaw, a multi-channel personal assistant gateway, shows that the same recurring design questions produce different architectural answers when the deployment context changes: from per-action safety classification to perimeter-level access control, from a single CLI loop to an embedded runtime within a gateway control plane, and from context-window extensions to gateway-wide capability registration. We finally identify six open design directions for future agent systems, grounded in recent empirical, architectural, and policy literature. 

---
# Neuro-Oracle: A Trajectory-Aware Agentic RAG Framework for Interpretable Epilepsy Surgical Prognosis 

**Authors**: Aizierjiang Aiersilan, Mohamad Koubeissi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14216)  

**Abstract**: Predicting post-surgical seizure outcomes in pharmacoresistant epilepsy is a clinical challenge. Conventional deep-learning approaches operate on static, single-timepoint pre-operative scans, omitting longitudinal morphological changes. We propose \emph{Neuro-Oracle}, a three-stage framework that: (i) distils pre-to-post-operative MRI changes into a compact 512-dimensional trajectory vector using a 3D Siamese contrastive encoder; (ii) retrieves historically similar surgical trajectories from a population archive via nearest-neighbour search; and (iii) synthesises a natural-language prognosis grounded in the retrieved evidence using a quantized Llama-3-8B reasoning agent. Evaluations are conducted on the public EPISURG dataset ($N{=}268$ longitudinally paired cases) using five-fold stratified cross-validation. Since ground-truth seizure-freedom scores are unavailable, we utilize a clinical proxy label based on the resection type. We acknowledge that the network representations may potentially learn the anatomical features of the resection cavities (i.e., temporal versus non-temporal locations) rather than true prognostic morphometry. Our current evaluation thus serves mainly as a proof-of-concept for the trajectory-aware retrieval architecture. Trajectory-based classifiers achieve AUC values between 0.834 and 0.905, compared with 0.793 for a single-timepoint ResNet-50 baseline. The Neuro-Oracle agent (M5) matches the AUC of purely discriminative trajectory classifiers (0.867) while producing structured justifications with zero observed hallucinations under our audit protocol. A Siamese Diversity Ensemble (M6) of trajectory-space classifiers attains an AUC of 0.905 without language-model overhead. 

---
# MixAtlas: Uncertainty-aware Data Mixture Optimization for Multimodal LLM Midtraining 

**Authors**: Bingbing Wen, Sirajul Salekin, Feiyang Kang, Bill Howe, Lucy Lu Wang, Javier Movellan, Manjot Bilkhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.14198)  

**Abstract**: Domain reweighting can improve sample efficiency and downstream generalization, but data-mixture optimization for multimodal midtraining remains largely unexplored. Current multimodal training recipes tune mixtures along a single dimension, typically data format or task type. We introduce MixAtlas, a method that produces benchmark-targeted data recipes that can be inspected, adapted, and transferred to new corpora. MixAtlas decomposes the training corpus along two axes: image concepts (10 visual-domain clusters discovered via CLIP embeddings) and task supervision (5 objective types including captioning, OCR, grounding, detection, and VQA). Using small proxy models (Qwen2-0.5B) paired with a Gaussian-process surrogate and GP-UCB acquisition, MixAtlas searches the resulting mixture space with the same proxy budget as regression-based baselines but finds better-performing mixtures. We evaluate on 10 benchmarks spanning visual understanding, document reasoning, and multimodal reasoning. On Qwen2-7B, optimized mixtures improve average performance by 8.5%-17.6% over the strongest baseline; on Qwen2.5-7B, gains are 1.0%-3.3%. Both settings reach baseline-equivalent training loss in up to 2 times fewer steps. Recipes discovered on 0.5B proxies transfer to 7B-scale training across Qwen model families. 

---
# Grading the Unspoken: Evaluating Tacit Reasoning in Quantum Field Theory and String Theory with LLMs 

**Authors**: Xingyang Yu, Yinghuan Zhang, Yufei Zhang, Zijun Cui  

**Link**: [PDF](https://arxiv.org/pdf/2604.14188)  

**Abstract**: Large language models have demonstrated impressive performance across many domains of mathematics and physics. One natural question is whether such models can support research in highly abstract theoretical fields such as quantum field theory and string theory. Evaluating this possibility faces an immediate challenge: correctness in these domains is layered, tacit, and fundamentally non-binary. Standard answer-matching metrics fail to capture whether intermediate conceptual steps are properly reconstructed or whether implicit structural constraints are respected. We construct a compact expert-curated dataset of twelve questions spanning core areas of quantum field theory and string theory, and introduce a five-level grading rubric separating statement correctness, key concept awareness, reasoning chain presence, tacit step reconstruction, and enrichment. Evaluating multiple contemporary LLMs, we observe near-ceiling performance on explicit derivations within stable conceptual frames, but systematic degradation when tasks require reconstruction of omitted reasoning steps or reorganization of representations under global consistency constraints. These failures are driven not only by missing intermediate steps, but by an instability in representation selection: models often fail to identify the correct conceptual framing required to resolve implicit tensions. We argue that highly abstract theoretical physics provides a uniquely sensitive lens on the epistemic limits of current evaluation paradigms. 

---
# HARNESS: Lightweight Distilled Arabic Speech Foundation Models 

**Authors**: Vrunda N. Sukhadia, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2604.14186)  

**Abstract**: Large self-supervised speech (SSL) models achieve strong downstream performance, but their size limits deployment in resource-constrained settings. We present HArnESS, an Arabic-centric self-supervised speech model family trained from scratch with iterative self-distillation, together with lightweight student variants that offer strong accuracy-efficiency trade-offs on Automatic Speech Recognition (ASR), Dialect Identification (DID), and Speech Emotion Recognition (SER). Our approach begins with a large bilingual Arabic-English teacher and progressively distills its knowledge into compressed student models while preserving Arabic-relevant acoustic and paralinguistic representations. We further study PCA-based compression of the teacher supervision signal to better match the capacity of shallow and thin students. Compared with HuBERT and XLS-R, HArnESS consistently improves performance on Arabic downstream tasks, while the compressed models remain competitive under substantial structural reduction. These results position HArnESS as a practical and accessible Arabic-centric SSL foundation for real-world speech applications. 

---
# From Black Box to Glass Box: Cross-Model ASR Disagreement to Prioto Review in Ambient AI Scribe Documentation 

**Authors**: Abdolamir Karbalaie, Fernando Seoane, Farhad Abtahi  

**Link**: [PDF](https://arxiv.org/pdf/2604.14152)  

**Abstract**: Ambient AI "scribe" systems promise to reduce clinical documentation burden, but automatic speech recognition (ASR) errors can remain unnoticed without careful review, and high-quality human reference transcripts are often unavailable for calibrating uncertainty. We investigate whether cross-model disagreement among heterogeneous ASR systems can act as a reference-free uncertainty signal to prioritize human verification in medical transcription workflows. Using 50 publicly available medical education audio clips (8 h 14 min), we transcribed each clip with eight ASR systems spanning commercial APIs and open-source engines. We aligned multi-model outputs, built consensus pseudo-references, and quantified token-level agreement using a majority-strength metric; we further characterized disagreements by type (content vs. punctuation/formatting) and assessed per-model agreement via leave-one-model-out (jackknife) consensus scoring. Inter-model reliability was low (ICC[2,1] = 0.131), indicating heterogeneous failure modes across systems. Across 76,398 evaluated token positions, 72.1% showed near-unanimous agreement (7-8 models), while 2.5% fell into high-risk bands (0-3 models), with high-risk mass varying from 0.7% to 11.4% across accent groups. Low-agreement regions were enriched for content disagreements, with the content fraction increasing from 53.9% to 73.9% across quintiles of high-risk mass. These results suggest that cross-model disagreement provides a sparse, localizable signal that can surface potentially unreliable transcript spans without human-verified references, enabling targeted review; clinical accuracy of flagged regions remains to be established. 

---
