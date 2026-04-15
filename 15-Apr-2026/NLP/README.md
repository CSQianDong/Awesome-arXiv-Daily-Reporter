# Toward Autonomous Long-Horizon Engineering for ML Research 

**Authors**: Guoxin Chen, Jie Chen, Lei Chen, Jiale Zhao, Fanzhe Meng, Wayne Xin Zhao, Ruihua Song, Cheng Chen, Ji-Rong Wen, Kai Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.13018)  

**Abstract**: Autonomous AI research has advanced rapidly, but long-horizon ML research engineering remains difficult: agents must sustain coherent progress across task comprehension, environment setup, implementation, experimentation, and debugging over hours or days. We introduce AiScientist, a system for autonomous long-horizon engineering for ML research built on a simple principle: strong long-horizon performance requires both structured orchestration and durable state continuity. To this end, AiScientist combines hierarchical orchestration with a permission-scoped File-as-Bus workspace: a top-level Orchestrator maintains stage-level control through concise summaries and a workspace map, while specialized agents repeatedly re-ground on durable artifacts such as analyses, plans, code, and experimental evidence rather than relying primarily on conversational handoffs, yielding thin control over thick state. Across two complementary benchmarks, AiScientist improves PaperBench score by 10.54 points on average over the best matched baseline and achieves 81.82 Any Medal% on MLE-Bench Lite. Ablation studies further show that File-as-Bus protocol is a key driver of performance, reducing PaperBench by 6.41 points and MLE-Bench Lite by 31.82 points when removed. These results suggest that long-horizon ML research engineering is a systems problem of coordinating specialized work over durable project state, rather than a purely local reasoning problem. 

---
# One Token Away from Collapse: The Fragility of Instruction-Tuned Helpfulness 

**Authors**: Erfan Baghaei Potraghloo, Seyedarmin Azizi, Souvik Kundu, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2604.13006)  

**Abstract**: Instruction-tuned large language models produce helpful, structured responses, but how robust is this helpfulness when trivially constrained? We show that simple lexical constraints (banning a single punctuation character or common word) cause instruction-tuned LLMs to collapse their responses, losing 14--48% of comprehensiveness in pairwise evaluation across three open-weight model families and one closed-weight model (GPT-4o-mini). The baseline response is preferred in 77--100% of 1,920 pairwise comparisons judged by GPT-4o-mini and GPT-4o. Notably, GPT-4o-mini suffers 31% comprehensiveness loss (99% baseline win rate), demonstrating that the fragility extends to commercially deployed closed-weight models, contrary to prior findings on format-level constraints. Through mechanistic analysis, we identify this as a planning failure: two-pass generation (free generation followed by constrained rewriting) recovers 59--96% of response length, and linear probes on prompt representations predict response length with $R^2 = 0.51$--$0.93$ before generation begins, with $R^2$ tracking collapse severity across models. The same probes yield negative $R^2$ on base models, confirming that instruction tuning creates the representational structure encoding the collapse decision. Crucially, base models show no systematic collapse under identical constraints, with effects that are small, noisy, and bidirectional, demonstrating that instruction tuning creates this fragility by coupling task competence to narrow surface-form templates. The effect replicates on MT-Bench across all eight task categories. We further show that standard independent LLM-as-judge evaluation detects only a 3.5% average quality drop where pairwise evaluation reveals 23%, exposing a methodological blind spot in how constrained generation is assessed. 

---
# PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models 

**Authors**: Han Bao, Penghao Zhang, Yue Huang, Zhengqing Yuan, Yanchi Ru, Rui Su, Yujun Zhou, Xiangqi Wang, Kehan Guo, Nitesh V Chawla, Yanfang Ye, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12995)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world decision-making, including in the domain of public policy. Yet, their ability to comprehend and reason about policy-related content remains underexplored. To fill this gap, we present \textbf{\textit{PolicyBench}}, the first large-scale cross-system benchmark (US-China) evaluating policy comprehension, comprising 21K cases across a broad spectrum of policy areas, capturing the diversity and complexity of real-world governance. Following Bloom's taxonomy, the benchmark assesses three core capabilities: (1) \textbf{Memorization}: factual recall of policy knowledge, (2) \textbf{Understanding}: conceptual and contextual reasoning, and (3) \textbf{Application}: problem-solving in real-life policy scenarios. Building on this benchmark, we further propose \textbf{\textit{PolicyMoE}}, a domain-specialized Mixture-of-Experts (MoE) model with expert modules aligned to each cognitive level. The proposed models demonstrate stronger performance on application-oriented policy tasks than on memorization or conceptual understanding, and yields the highest accuracy on structured reasoning tasks. Our results reveal key limitations of current LLMs in policy understanding and suggest paths toward more reliable, policy-focused models. 

---
# Accelerating Speculative Decoding with Block Diffusion Draft Trees 

**Authors**: Liran Ringel, Yaniv Romano  

**Link**: [PDF](https://arxiv.org/pdf/2604.12989)  

**Abstract**: Speculative decoding accelerates autoregressive language models by using a lightweight drafter to propose multiple future tokens, which the target model then verifies in parallel. DFlash shows that a block diffusion drafter can generate an entire draft block in a single forward pass and achieve state-of-the-art speculative decoding performance, outperforming strong autoregressive drafters such as EAGLE-3. Vanilla DFlash, however, still verifies only a single drafted trajectory per round, potentially limiting its acceptance length. We introduce DDTree (Diffusion Draft Tree), a method that constructs a draft tree directly from the per-position distributions of a block diffusion drafter. Under a fixed node budget, DDTree uses a simple best-first heap algorithm to select the continuations that are most likely to match the target model according to a surrogate defined by the draft model's output. The resulting tree is verified efficiently in a single target model forward pass using an ancestor-only attention mask. Because DDTree builds on DFlash, a leading draft model for speculative decoding, these gains place DDTree among the leading approaches to speculative decoding. 

---
# GlotOCR Bench: OCR Models Still Struggle Beyond a Handful of Unicode Scripts 

**Authors**: Amir Hossein Kargaran, Nafiseh Nikeghbal, Jana Diesner, François Yvon, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2604.12978)  

**Abstract**: Optical character recognition (OCR) has advanced rapidly with the rise of vision-language models, yet evaluation has remained concentrated on a small cluster of high- and mid-resource scripts. We introduce GlotOCR Bench, a comprehensive benchmark evaluating OCR generalization across 100+ Unicode scripts. Our benchmark comprises clean and degraded image variants rendered from real multilingual texts. Images are rendered using fonts from the Google Fonts repository, shaped with HarfBuzz and rasterized with FreeType, supporting both LTR and RTL scripts. Samples of rendered images were manually reviewed to verify correct rendering across all scripts. We evaluate a broad suite of open-weight and proprietary vision-language models and find that most perform well on fewer than ten scripts, and even the strongest frontier models fail to generalize beyond thirty scripts. Performance broadly tracks script-level pretraining coverage, suggesting that current OCR systems rely on language model pretraining as much as on visual recognition. Models confronted with unfamiliar scripts either produce random noise or hallucinate characters from similar scripts they already know. We release the benchmark and pipeline for reproducibility. Pipeline Code: this https URL, Benchmark: this https URL. 

---
# MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models 

**Authors**: Chung-Ming Chien, Manu Orsini, Eugene Kharitonov, Neil Zeghidour, Karen Livescu, Alexandre Défossez  

**Link**: [PDF](https://arxiv.org/pdf/2604.12928)  

**Abstract**: Speech-to-speech language models have recently emerged to enhance the naturalness of conversational AI. In particular, full-duplex models are distinguished by their real-time interactivity, including handling of pauses, interruptions, and backchannels. However, improving their factuality remains an open challenge. While scaling the model size could address this gap, it would make real-time inference prohibitively expensive. In this work, we propose MoshiRAG, a modular approach that combines a compact full-duplex interface with selective retrieval to access more powerful knowledge sources. Our asynchronous framework enables the model to identify knowledge-demanding queries and ground its responses in external information. By leveraging the natural temporal gap between response onset and the delivery of core information, the retrieval process can be completed while maintaining a natural conversation flow. With this approach, MoshiRAG achieves factuality comparable to the best publicly released non-duplex speech language models while preserving the interactivity inherent to full-duplex systems. Moreover, our flexible design supports plug-and-play retrieval methods without retraining and demonstrates strong performance on out-of-domain mathematical reasoning tasks. 

---
# MetFuse: Figurative Fusion between Metonymy and Metaphor 

**Authors**: Saptarshi Ghosh, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12919)  

**Abstract**: Metonymy and metaphor often co-occur in natural language, yet computational work has studied them largely in isolation. We introduce a framework that transforms a literal sentence into three figurative variants: metonymic, metaphoric, and hybrid. Using this framework, we construct MetFuse, the first dedicated dataset of figurative fusion between metonymy and metaphor, containing 1,000 human-verified meaning-aligned quadruplets totaling 4,000 sentences. Extrinsic experiments on eight existing benchmarks show that augmenting training data with MetFuse consistently improves both metonymy and metaphor classification, with hybrid examples yielding the largest gains on metonymy tasks. Using this dataset, we also analyze how the presence of one figurative type impacts another. Our findings show that both human annotators and large language models better identify metonymy in hybrid sentences than in metonymy-only sentences, demonstrating that the presence of a metaphor makes a metonymic noun more explicit. Our dataset is publicly available at: this https URL. 

---
# Round-Trip Translation Reveals What Frontier Multilingual Benchmarks Miss 

**Authors**: Ronald Skorobogat, Ameya Prabhu, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2604.12911)  

**Abstract**: Multilingual benchmarks guide the development of frontier models. Yet multilingual evaluations reported by frontier models are structured similar to popular reasoning and knowledge benchmarks, but across many languages. We show such benchmarks, and consequently multilingual evaluations, measure mathematical reasoning and factual recall, not multilingual proficiency. For example, thinking variants dramatically outperform instruct variants on these benchmarks, yet often perform worse on real-world multilingual tasks, such as LMArena. We propose a simple alternative: evaluate multilingual capability via round-trip translation. Given text in a source language, translate it to a target language and back; semantic gaps between the original and result expose failures in multilingual generation capabilities. Round-trip translation correlates almost perfectly (\r{ho} = 0.94) with user ratings on LMArena with our benchmark, requires no human reference translations, and does not require a more capable multilingual judge than tested models. Lastly, we introduce Lost in Translation (LiT), a challenging round-trip translation benchmark spanning widely spoken languages worldwide, for realistic evaluation of multilingual frontier models. 

---
# Growing Pains: Extensible and Efficient LLM Benchmarking Via Fixed Parameter Calibration 

**Authors**: Eliya Habba, Itay Itzhak, Asaf Yehudai, Yotam Perlitz, Elron Bandel, Michal Shmueli-Scheuer, Leshem Choshen, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.12843)  

**Abstract**: The rapid release of both language models and benchmarks makes it increasingly costly to evaluate every model on every dataset. In practice, models are often evaluated on different samples, making scores difficult to compare across studies. To address this, we propose a framework based on multidimensional Item Response Theory (IRT) that uses anchor items to calibrate new benchmarks to the evaluation suite while holding previously calibrated item parameters fixed. Our approach supports a realistic evaluation setting in which datasets are introduced over time and models are evaluated only on the datasets available at the time of evaluation, while a fixed anchor set for each dataset is used so that results from different evaluation periods can be compared directly. In large-scale experiments on more than $400$ models, our framework predicts full-evaluation performance within 2-3 percentage points using only $100$ anchor questions per dataset, with Spearman $\rho \geq 0.9$ for ranking preservation, showing that it is possible to extend benchmark suites over time while preserving score comparability, at a constant evaluation cost per new dataset. Code available at this https URL 

---
# The role of System 1 and System 2 semantic memory structure in human and LLM biases 

**Authors**: Katherine Abramski, Giulio Rossetti, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2604.12816)  

**Abstract**: Implicit biases in both humans and large language models (LLMs) pose significant societal risks. Dual process theories propose that biases arise primarily from associative System 1 thinking, while deliberative System 2 thinking mitigates bias, but the cognitive mechanisms that give rise to this phenomenon remain poorly understood. To better understand what underlies this duality in humans, and possibly in LLMs, we model System 1 and System 2 thinking as semantic memory networks with distinct structures, built from comparable datasets generated by both humans and LLMs. We then investigate how these distinct semantic memory structures relate to implicit gender bias using network-based evaluation metrics. We find that semantic memory structures are irreducible only in humans, suggesting that LLMs lack certain types of human-like conceptual knowledge. Moreover, semantic memory structure relates consistently to implicit bias only in humans, with lower levels of bias in System~2 structures. These findings suggest that certain types of conceptual knowledge contribute to bias regulation in humans, but not in LLMs, highlighting fundamental differences between human and machine cognition. 

---
# EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution 

**Authors**: Shiyu He, Minchi Kuang, Mengxian Wang, Bin Hu, Tingxiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12776)  

**Abstract**: Realizing endogenous narrative evolution in LLM-based multi-agent systems is hindered by the inherent stochasticity of generative emergence. In particular, long-horizon simulations suffer from social memory stacking, where conflicting relational states accumulate without resolution, and narrative-spatial dissonance, where spatial logic detaches from the evolving plot. To bridge this gap, we propose EvoSpark, a framework specifically designed to sustain logically coherent long-horizon narratives within Endogenous Interactive Agent Societies. To ensure consistency, the Stratified Narrative Memory employs a Role Socio-Evolutionary Base as living cognition, dynamically metabolizing experiences to resolve historical conflicts. Complementarily, Generative Mise-en-Scène mechanism enforces Role-Location-Plot alignment, synchronizing character presence with the narrative flow. Underpinning these is the Unified Narrative Operation Engine, which integrates an Emergent Character Grounding Protocol to transform stochastic sparking into persistent characters. This engine establishes a substrate that expands a minimal premise into an open-ended, evolving story world. Experiments demonstrate that EvoSpark significantly outperforms baselines across diverse paradigms, enabling the sustained generation of expressive and coherent narrative experiences. 

---
# Teaching LLMs Human-Like Editing of Inappropriate Argumentation via Reinforcement Learning 

**Authors**: Timon Ziegenbein, Maja Stahl, Henning Wachsmuth  

**Link**: [PDF](https://arxiv.org/pdf/2604.12770)  

**Abstract**: Editing human-written text has become a standard use case of large language models (LLMs), for example, to make one's arguments more appropriate for a discussion. Comparing human to LLM-generated edits, however, we observe a mismatch in editing strategies: While LLMs often perform multiple scattered edits and tend to change meaning notably, humans rather encapsulate dependent changes in self-contained, meaning-preserving edits. In this paper, we present a reinforcement learning approach that teaches LLMs human-like editing to improve the appropriateness of arguments. Our approach produces self-contained sentence-level edit suggestions that can be accepted or rejected independently. We train the approach using group relative policy optimization with a multi-component reward function that jointly optimizes edit-level semantic similarity, fluency, and pattern conformity as well as argument-level appropriateness. In automatic and human evaluation, it outperforms competitive baselines and the state of the art in human-like editing, with multi-round editing achieving appropriateness close to full rewriting. 

---
# NaviRAG: Towards Active Knowledge Navigation for Retrieval-Augmented Generation 

**Authors**: Jihao Dai, Dingjun Wu, Yuxuan Chen, Zheni Zeng, Yukun Yan, Zhenghao Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.12766)  

**Abstract**: Retrieval-augmented generation (RAG) typically relies on a flat retrieval paradigm that maps queries directly to static, isolated text segments. This approach struggles with more complex tasks that require the conditional retrieval and dynamic synthesis of information across different levels of granularity (e.g., from broad concepts to specific evidence). To bridge this gap, we introduce NaviRAG, a novel framework that shifts from passive segment retrieval to active knowledge navigation. NaviRAG first structures the knowledge documents into a hierarchical form, preserving semantic relationships from coarse-grained topics to fine-grained details. Leveraging this reorganized knowledge records, a large language model (LLM) agent actively navigates the records, iteratively identifying information gaps and retrieving relevant content from the most appropriate granularity level. Extensive experiments on long-document QA benchmarks show that NaviRAG consistently improves both retrieval recall and end-to-end answer performance over conventional RAG baselines. Ablation studies confirm performance gains stem from our method's capacity for multi-granular evidence localization and dynamic retrieval planning. We further discuss efficiency, applicable scenario, and future directions of our method, hoping to make RAG systems more intelligent and autonomous. 

---
# Generating Effective CoT Traces for Mitigating Causal Hallucination 

**Authors**: Yiheng Zhao, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.12748)  

**Abstract**: Although large language models (LLMs) excel in complex reasoning tasks, they suffer from severe causal hallucination in event causality identification (ECI), particularly in smaller models ($\leq$1.5B parameters). A promising approach to address this issue is to fine-tune them with Chain-of-Thought (CoT) traces. However, there is currently a lack of CoT trace dataset available for ECI. In this paper, we first investigate the essential criteria that effective CoT traces should possess to mitigate causal hallucination in smaller models. We then design a pipeline to generate CoT traces that meet these criteria. Moreover, since there is currently no metric for quantifying causal hallucination, we also introduce a new metric, the Causal Hallucination Rate (CHR), to quantify causal hallucination, guide the formulation of effective CoT trace criteria, and validate the effectiveness of our pipeline. Our experiments show that fine-tuning with the CoT traces generated by our pipeline not only substantially reduces causal hallucination in smaller LLMs but also improves mean accuracy. Moreover, the fine-tuned models exhibit strong cross-dataset and cross-difficulty generalization, as well as robustness under misleading intervention prompts. 

---
# Universal NER v2: Towards a Massively Multilingual Named Entity Recognition Benchmark 

**Authors**: Terra Blevins, Stephen Mayhew, Marek Šuppa, Hila Gonen, Shachar Mirkin, Vasile Pais, Kaja Dobrovoljc, Voula Giouli, Jun Kevin, Eugene Jang, Eungseo Kim, Jeongyeon Seo, Xenophon Gialis, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2604.12744)  

**Abstract**: While multilingual language models promise to bring the benefits of LLMs to speakers of many languages, gold-standard evaluation benchmarks in most languages to interrogate these assumptions remain scarce. The Universal NER project, now entering its fourth year, is dedicated to building gold-standard multilingual Named Entity Recognition (NER) benchmark datasets. Inspired by existing massively multilingual efforts for other core NLP tasks (e.g., Universal Dependencies), the project uses a general tagset and thorough annotation guidelines to collect standardized, cross-lingual annotations of named entity spans. The first installment (UNER v1) was released in 2024, and the project has continued and expanded since then, with various organizers, annotators, and collaborators in an active community. 

---
# Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Sequence-Level Likelihood 

**Authors**: Xingyu Lin, Yilin Wen, Du Su, Jinchang Hou, En Wang, Wenbin Liu, Chenfu Bao, Zhonghou Lv  

**Link**: [PDF](https://arxiv.org/pdf/2604.12736)  

**Abstract**: Group Relative Policy Optimization (GRPO) has significantly advanced the reasoning ability of large language models (LLMs), particularly in their mathemat ical reasoning performance. However, GRPO and related entropy regularization methods still struggle with token-level sparse-rewards, which is an inherent chal lenge in chain-of-thought (CoT) reasoning. These approaches often rely on undifferen tiated token-level entropy regularization, which easily leads to entropy collapse or model degradation under sparse token rewards. In this work, we propose TEPO, a novel token-level framework that (1) leverages sequence-level likelihood to link group-level rewards with individual tokens via token-level aggregation, and (2) introduces a token-level KL-Divergence mask constraint that targets tokens with positive advantages and decreasing entropy to mitigate abrupt policy updates. Experiments demonstrate that TEPO not only achieves state-of-the-art performance on mathematical reasoning benchmarks but also markedly enhances training stability, reducing convergence time by 50% compared with GRPO/DAPO. 

---
# InsightFlow: LLM-Driven Synthesis of Patient Narratives for Mental Health into Causal Models 

**Authors**: Shreya Gupta, Prottay Kumar Adhikary, Bhavyaa Dave, Salam Michael Singh, Aniket Deroy, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2604.12721)  

**Abstract**: Clinical case formulation organizes patient symptoms and psychosocial factors into causal models, often using the 5P framework. However, constructing such graphs from therapy transcripts is time consuming and varies across clinicians. We present InsightFlow, an LLM based approach that automatically generates 5P aligned causal graphs from patient-therapist dialogues. Using 46 psychotherapy intake transcripts annotated by clinical experts, we evaluate LLM generated graphs against human formulations using structural (NetSimile), semantic (embedding similarity), and expert rated clinical criteria. The generated graphs show structural similarity comparable to inter annotator agreement and high semantic alignment with human graphs. Expert evaluations rate the outputs as moderately complete, consistent, and clinically useful. While LLM graphs tend to form more interconnected structures compared to the chain like patterns of human graphs, overall complexity and content coverage are similar. These results suggest that LLMs can produce clinically meaningful case formulation graphs within the natural variability of expert practice. InsightFlow highlights the potential of automated causal modeling to augment clinical workflows, with future work needed to improve temporal reasoning and reduce redundancy. 

---
# Learning Chain Of Thoughts Prompts for Predicting Entities, Relations, and even Literals on Knowledge Graphs 

**Authors**: Alkid Baci, Luke Friedrichs, Caglar Demir, N'Dah Jean Kouagou, Axel-Cyrille Ngonga Ngomo  

**Link**: [PDF](https://arxiv.org/pdf/2604.12651)  

**Abstract**: Knowledge graph embedding (KGE) models perform well on link prediction but struggle with unseen entities, relations, and especially literals, limiting their use in dynamic, heterogeneous graphs. In contrast, pretrained large language models (LLMs) generalize effectively through prompting. We reformulate link prediction as a prompt learning problem and introduce RALP, which learns string-based chain-of-thought (CoT) prompts as scoring functions for triples. Using Bayesian Optimization through MIPRO algorithm, RALP identifies effective prompts from fewer than 30 training examples without gradient access. At inference, RALP predicts missing entities, relations or whole triples and assigns confidence scores based on the learned prompt. We evaluate on transductive, numerical, and OWL instance retrieval benchmarks. RALP improves state-of-the-art KGE models by over 5% MRR across datasets and enhances generalization via high-quality inferred triples. On OWL reasoning tasks with complex class expressions (e.g., $\exists this http URL$, $\geq 5 \; this http URL$), it achieves over 88% Jaccard similarity. These results highlight prompt-based LLM reasoning as a flexible alternative to embedding-based methods. We release our implementation, training, and evaluation pipeline as open source: this https URL . 

---
# Multilingual Multi-Label Emotion Classification at Scale with Synthetic Data 

**Authors**: Vadim Borisov  

**Link**: [PDF](https://arxiv.org/pdf/2604.12633)  

**Abstract**: Emotion classification in multilingual settings remains constrained by the scarcity of annotated data: existing corpora are predominantly English, single-label, and cover few languages. We address this gap by constructing a large-scale synthetic training corpus of over 1M multi-label samples (50k per language) across 23 languages: Arabic, Bengali, Dutch, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Mandarin, Polish, Portuguese, Punjabi, Russian, Spanish, Swahili, Tamil, Turkish, Ukrainian, Urdu, and Vietnamese, covering 11 emotion categories using culturally-adapted generation and programmatic quality filtering. We train and compare six multilingual transformer encoders, from DistilBERT (135M parameters) to XLM-R-Large (560M parameters), under identical conditions. On our in-domain test set, XLM-R-Large achieves 0.868 F1-micro and 0.987 AUC-micro. To validate against human-annotated data, we evaluate all models zero-shot on GoEmotions (English) and SemEval-2018 Task 1 E-c (English, Arabic, Spanish). On threshold-free ranking metrics, XLM-R-Large matches or exceeds English-only specialist models, tying on AP-micro (0.636) and LRAP (0.804) while surpassing on AUC-micro (0.810 vs. 0.787), while natively supporting all 23 languages. The best base-sized model is publicly available at this https URL 

---
# Transforming External Knowledge into Triplets for Enhanced Retrieval in RAG of LLMs 

**Authors**: Xudong Wang, Chaoning Zhang, Qigan Sun, Zhenzhen Huang, Chang Lu, Sheng Zheng, Zeyu Ma, Caiyan Qin, Yang Yang, Hengtao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.12610)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucination in large language models (LLMs) by incorporating external knowledge during generation. However, the effectiveness of RAG depends not only on the design of the retriever and the capacity of the underlying model, but also on how retrieved evidence is structured and aligned with the query. Existing RAG approaches typically retrieve and concatenate unstructured text fragments as context, which often introduces redundant or weakly relevant information. This practice leads to excessive context accumulation, reduced semantic alignment, and fragmented reasoning chains, thereby degrading generation quality while increasing token consumption. To address these challenges, we propose Tri-RAG, a structured triplet-based retrieval framework that improves retrieval efficiency through reasoning-aligned context construction. Tri-RAG automatically transforms external knowledge from natural language into standardized structured triplets consisting of Condition, Proof, and Conclusion, explicitly capturing logical relations among knowledge fragments using lightweight prompt-based adaptation with frozen model parameters. Building on this representation, the triplet head Condition is treated as an explicit semantic anchor for retrieval and matching, enabling precise identification of query-relevant knowledge units without directly concatenating lengthy raw texts. As a result, Tri-RAG achieves a favorable balance between retrieval accuracy and context token efficiency. Experimental results across multiple benchmark datasets demonstrate that Tri-RAG significantly improves retrieval quality and reasoning efficiency, while producing more stable generation behavior and more efficient resource utilization in complex reasoning scenarios. 

---
# FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing 

**Authors**: Peng Wang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12559)  

**Abstract**: Unstructured model editing aims to update models with real-world text, yet existing methods often memorize text holistically without reliable fine-grained fact access. To address this, we propose FABLE, a hierarchical framework that decouples fine-grained fact injection from holistic text generation. FABLE follows a two-stage, fact-first strategy: discrete facts are anchored in shallow layers, followed by minimal updates to deeper layers to produce coherent text. This decoupling resolves the mismatch between holistic recall and fine-grained fact access, reflecting the unidirectional Transformer flow in which surface-form generation amplifies rather than corrects underlying fact representations. We also introduce UnFine, a diagnostic benchmark with fine-grained question-answer pairs and fact-level metrics for systematic evaluation. Experiments show that FABLE substantially improves fine-grained question answering while maintaining state-of-the-art holistic editing performance. Our code is publicly available at this https URL. 

---
# When Does Data Augmentation Help? Evaluating LLM and Back-Translation Methods for Hausa and Fongbe NLP 

**Authors**: Mahounan Pericles Adjovi, Roald Eiselen, Prasenjit Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.12540)  

**Abstract**: Data scarcity limits NLP development for low-resource African languages. We evaluate two data augmentation methods -- LLM-based generation (Gemini 2.5 Flash) and back-translation (NLLB-200) -- for Hausa and Fongbe, two West African languages that differ substantially in LLM generation quality. We assess augmentation on named entity recognition (NER) and part-of-speech (POS) tagging using MasakhaNER 2.0 and MasakhaPOS benchmarks. Our results reveal that augmentation effectiveness depends on task type rather than language or LLM quality alone. For NER, neither method improves over baseline for either language; LLM augmentation reduces Hausa NER by 0.24% F1 and Fongbe NER by 1.81% F1. For POS tagging, LLM augmentation improves Fongbe by 0.33% accuracy, while back-translation improves Hausa by 0.17%; back-translation reduces Fongbe POS by 0.35% and has negligible effect on Hausa POS. The same LLM-generated synthetic data produces opposite effects across tasks for Fongbe -- hurting NER while helping POS -- suggesting task structure governs augmentation outcomes more than synthetic data quality. These findings challenge the assumption that LLM generation quality predicts augmentation success, and provide actionable guidance: data augmentation should be treated as a task-specific intervention rather than a universally beneficial preprocessing step. 

---
# Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis 

**Authors**: Kang He, Yuzhe Ding, Xinrong Wang, Fei Li, Chong Teng, Donghong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2604.12518)  

**Abstract**: Multimodal sentiment analysis (MSA) integrates heterogeneous text, audio, and visual signals to infer human emotions. While recent approaches leverage cross-modal complementarity, they often struggle to fully utilize weaker modalities. In practice, dominant modalities tend to overshadow non-verbal ones, inducing modality competition and limiting overall contributions. This imbalance degrades fusion performance and robustness under noisy or missing modalities. To address this, we propose a novel model, Enhance-then-Balance Modality Collaboration framework (EBMC). EBMC improves representation quality via semantic disentanglement and cross-modal enhancement, strengthening weaker modalities. To prevent dominant modalities from overwhelming others, an Energy-guided Modality Coordination mechanism achieves implicit gradient rebalancing via a differentiable equilibrium objective. Furthermore, Instance-aware Modality Trust Distillation estimates sample-level reliability to adaptively modulate fusion weights, ensuring robustness. Extensive experiments demonstrate that EBMC achieves state-of-the-art or competitive results and maintains strong performance under missing-modality settings. 

---
# Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs 

**Authors**: Linhao Zhang, Yuhan Song, Aiwei Liu, Chuhan Wu, Sijun Zhang, Wei Jia, Yuan Liu, Houfeng Wang, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.12506)  

**Abstract**: Recent Audio Large Language Models (AudioLLMs) exhibit a striking performance inversion: while excelling at complex reasoning tasks, they consistently underperform on fine-grained acoustic perception. We attribute this gap to a fundamental limitation of ASR-centric training, which provides precise linguistic targets but implicitly teaches models to suppress paralinguistic cues and acoustic events as noise. To address this, we propose Unified Audio Schema (UAS), a holistic and structured supervision framework that organizes audio information into three explicit components -- Transcription, Paralinguistics, and Non-linguistic Events -- within a unified JSON format. This design achieves comprehensive acoustic coverage without sacrificing the tight audio-text alignment that enables reasoning. We validate the effectiveness of this supervision strategy by applying it to both discrete and continuous AudioLLM architectures. Extensive experiments on MMSU, MMAR, and MMAU demonstrate that UAS-Audio yields consistent improvements, boosting fine-grained perception by 10.9% on MMSU over the same-size state-of-the-art models while preserving robust reasoning capabilities. Our code and model are publicly available at this https URL. 

---
# Topology-Aware Reasoning over Incomplete Knowledge Graph with Graph-Based Soft Prompting 

**Authors**: Shuai Wang, Xixi Wang, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12503)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across various tasks but remain prone to hallucinations in knowledge-intensive scenarios. Knowledge Base Question Answering (KBQA) mitigates this by grounding generation in Knowledge Graphs (KGs). However, most multi-hop KBQA methods rely on explicit edge traversal, making them fragile to KG incompleteness. In this paper, we proposed a novel graph-based soft prompting framework that shifts the reasoning paradigm from node-level path traversal to subgraph-level reasoning. Specifically, we employ a Graph Neural Network (GNN) to encode extracted structural subgraphs into soft prompts, enabling LLM to reason over richer structural context and identify relevant entities beyond immediate graph neighbors, thereby reducing sensitivity to missing edges. Furthermore, we introduce a two-stage paradigm that reduces computational cost while preserving good performance: a lightweight LLM first leverages the soft prompts to identify question-relevant entities and relations, followed by a more powerful LLM for evidence-aware answer generation. Experiments on four multi-hop KBQA benchmarks show that our approach achieves state-of-the-art performance on three of them, demonstrating its effectiveness. Code is available at the repository: this https URL. 

---
# Latent Planning Emerges with Scale 

**Authors**: Michael Hanna, Emmanuel Ameisen  

**Link**: [PDF](https://arxiv.org/pdf/2604.12493)  

**Abstract**: LLMs can perform seemingly planning-intensive tasks, like writing coherent stories or functioning code, without explicitly verbalizing a plan; however, the extent to which they implicitly plan is unknown. In this paper, we define latent planning as occurring when LLMs possess internal planning representations that (1) cause the generation of a specific future token or concept, and (2) shape preceding context to license said future token or concept. We study the Qwen-3 family (0.6B-14B) on simple planning tasks, finding that latent planning ability increases with scale. Models that plan possess features that represent a planned-for word like "accountant", and cause them to output "an" rather than "a"; moreover, even the less-successful Qwen-3 4B-8B have nascent planning mechanisms. On the more complex task of completing rhyming couplets, we find that models often identify a rhyme ahead of time, but even large models seldom plan far ahead. However, we can elicit some planning that increases with scale when steering models towards planned words in prose. In sum, we offer a framework for measuring planning and mechanistic evidence of how models' planning abilities grow with scale. 

---
# Calibrated Confidence Estimation for Tabular Question Answering 

**Authors**: Lukas Voss  

**Link**: [PDF](https://arxiv.org/pdf/2604.12491)  

**Abstract**: Large language models (LLMs) are increasingly deployed for tabular question
answering, yet calibration on structured data is largely unstudied. This
paper presents the first systematic comparison of five confidence estimation
methods across five frontier LLMs and two tabular QA benchmarks. All models
are severely overconfident (smooth ECE 0.35-0.64 versus 0.10-0.15 reported
for textual QA). A consistent self-evaluation versus perturbation dichotomy
replicates across both benchmarks and all four fully-covered models:
self-evaluation methods (verbalized, P(True)) achieve AUROC 0.42-0.76, while
perturbation methods (semantic entropy, self-consistency, and our
Multi-Format Agreement) achieve AUROC 0.78-0.86. Per-model paired bootstrap
tests reject the null at p<0.001 after Holm-Bonferroni correction, and a
3-seed check on GPT-4o-mini gives a per-seed standard deviation of only
0.006. The paper proposes Multi-Format Agreement (MFA), which exploits the
lossless and deterministic serialization variation unique to structured data
(Markdown, HTML, JSON, CSV) to estimate confidence at 20% lower API cost
than sampling baselines. MFA reduces ECE by 44-63%, generalizes across all
four models on TableBench (mean AUROC 0.80), and combines complementarily
with sampling: an MFA + self-consistency ensemble lifts AUROC from 0.74 to
0.82. A secondary contribution, structure-aware recalibration, improves
AUROC by +10 percentage points over standard post-hoc methods. 

---
# KG-Reasoner: A Reinforced Model for End-to-End Multi-Hop Knowledge Graph Reasoning 

**Authors**: Shuai Wang, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12487)  

**Abstract**: Large Language Models (LLMs) exhibit strong abilities in natural language understanding and generation, yet they struggle with knowledge-intensive reasoning. Structured Knowledge Graphs (KGs) provide an effective form of external knowledge representation and have been widely used to enhance performance in classical Knowledge Base Question Answering (KBQA) tasks. However, performing precise multi-hop reasoning over KGs for complex queries remains highly challenging. Most existing approaches decompose the reasoning process into a sequence of isolated steps executed through a fixed pipeline. While effective to some extent, such designs constrain reasoning flexibility and fragment the overall decision process, often leading to incoherence and the loss of critical intermediate information from earlier steps. In this paper, we introduce KG-Reasoner, an end-to-end framework that integrates multi-step reasoning into a unified "thinking" phase of a Reasoning LLM. Through Reinforcement Learning (RL), the LLM is trained to internalize the KG traversal process, enabling it to dynamically explore reasoning paths, and perform backtracking when necessary. Experiments on eight multi-hop and knowledge-intensive reasoning benchmarks demonstrate that KG-Reasoner achieves competitive or superior performance compared to the state-of-the-art methods. Codes are available at the repository: this https URL. 

---
# Meet Dynamic Individual Preferences: Resolving Conflicting Human Value with Paired Fine-Tuning 

**Authors**: Shanyong Wang, Shuhang Lin, Yining Zhao, Xi Zhu, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12479)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved the alignment of models with general human preferences. However, a major challenge remains in adapting LLMs to individual preferences, which are not only diverse but also dynamic. In this paper, we introduce a novel framework, Preference-Paired Fine-Tuning (PFT), designed to align models with contradictory and evolving individual preferences. We present a new dataset, Value Conflict Dilemma (VCD), which includes scenarios that involve conflicting human preferences, facilitating the evaluation of our approach. Our experiments demonstrate that PFT outperforms single-preference training methods, achieving up to 96.6% accuracy in multi-choice classification tasks and the highest open-ended generation score of 8.69. PFT also shows significant improvements over DPO, SFT and some traditional training methods, especially when handling conflicting preferences. Additionally, with limited user history data, models can inferring preference vector rapidly, achieving a 44.76% improvement in user-specific preference alignment in comparison to single-preference models. 

---
# Mining Large Language Models for Low-Resource Language Data: Comparing Elicitation Strategies for Hausa and Fongbe 

**Authors**: Mahounan Pericles Adjovi, Roald Eiselen, Prasenjit Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.12477)  

**Abstract**: Large language models (LLMs) are trained on data contributed by low-resource language communities, yet the linguistic knowledge encoded in these models remains accessible only through commercial APIs. This paper investigates whether strategic prompting can extract usable text data from LLMs for two West African languages: Hausa (Afroasiatic, approximately 80 million speakers) and Fongbe (Niger-Congo, approximately 2 million speakers). We systematically compare six elicitation task types across two commercial LLMs (GPT-4o Mini and Gemini 2.5 Flash). GPT-4o Mini extracts 6-41 times more usable target-language words per API call than Gemini. Optimal strategies differ by language: Hausa benefits from functional text and dialogue, while Fongbe requires constrained generation prompts. We release all generated corpora and code. 

---
# Latent-Condensed Transformer for Efficient Long Context Modeling 

**Authors**: Zeng You, Yaofo Chen, Qiuwu Chen, Ying Sun, Shuhai Zhang, Yingjian Li, Yaowei Wang, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.12452)  

**Abstract**: Large language models (LLMs) face significant challenges in processing long contexts due to the linear growth of the key-value (KV) cache and quadratic complexity of self-attention. Existing approaches address these bottlenecks separately: Multi-head Latent Attention (MLA) reduces the KV cache by projecting tokens into a low-dimensional latent space, while sparse attention reduces computation. However, sparse methods cannot operate natively on MLA's compressed latent structure, missing opportunities for joint optimization. In this paper, we propose Latent-Condensed Attention (LCA), which directly condenses context within MLA's latent space, where the representation is disentangled into semantic latent vectors and positional keys. LCA separately aggregates semantic vectors via query-aware pooling and preserves positional keys via anchor selection. This approach jointly reduces both computational cost and KV cache without adding parameters. Beyond MLA, LCA's design is architecture-agnostic and readily extends to other attention mechanisms such as GQA. Theoretically, we prove a length-independent error bound. Experiments show LCA achieves up to 2.5$\times$ prefilling speedup and 90% KV cache reduction at 128K context while maintaining competitive performance. 

---
# GLeMM: A large-scale multilingual dataset for morphological research 

**Authors**: Hathout Nabil, Basilio Calderone, Fiammetta Namer, Franck Sajous  

**Link**: [PDF](https://arxiv.org/pdf/2604.12442)  

**Abstract**: In derivational morphology, what mechanisms govern the variation in form-meaning relations between words? The answers to this type of questions are typically based on intuition and on observations drawn from limited data, even when a wide range of languages is considered. Many of these studies are difficult to replicate and generalize. To address this issue, we present GLeMM, a new derivational resource designed for experimentation and data-driven description in morphology. GLeMM is characterized by (i) its large size, (ii) its extensive coverage (currently amounting to seven European languages, i.e., German, English, Spanish, French, Italian, Polish, Russian, (iii) its fully automated design, identical across all languages, (iv) the automatic annotation of morphological features on each entry, as well as (v) the encoding of semantic descriptions for a significant subset of these entries. It enables researchers to address difficult questions, such as the role of form and meaning in word-formation, and to develop and experimentally test computational methods that identify the structures of derivational morphology. The article describes how GLeMM is created using Wiktionary articles and presents various case studies illustrating possible applications of the resource. 

---
# Decoding by Perturbation: Mitigating MLLM Hallucinations via Dynamic Textual Perturbation 

**Authors**: Sihang Jia, Shuliang Liu, Songbo Yang, Yibo Yan, Xin Zou, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12424)  

**Abstract**: Multimodal Large Language Models frequently suffer from inference hallucinations, partially stemming from language priors dominating visual evidence. Existing training-free mitigation methods either perturb the visual representation and deviate from the natural image distribution, or enforce intrusive manipulations that compromise the model's inherent generative fluency. We introduce a novel perspective that multimodal hallucination manifests as the hypersensitivity of visual grounding to textual phrasing during the decoding phase. Building on this insight, we propose Decoding by Perturbation (DeP), a training-free framework mitigating prior-induced hallucinations via controlled textual interventions. DeP employs a dynamic probe applying multi-level textual perturbations to elicit latent language priors. Leveraging attention variance, it enhances stable evidence regions while suppressing suspicious noise in the feature space. Furthermore, it constructs an interpretable prior drift direction using logits statistics to counteract probability biases from textual co-occurrences. Extensive experiments confirm DeP effectively reduces hallucinations and achieves superior performance across multiple benchmarks. 

---
# Agentic Insight Generation in VSM Simulations 

**Authors**: Micha Selak, Dirk Krechel, Adrian Ulges, Sven Spieckermann, Niklas Stoehr, Andreas Loehr  

**Link**: [PDF](https://arxiv.org/pdf/2604.12421)  

**Abstract**: Extracting actionable insights from complex value stream map simulations can be challenging, time-consuming, and error-prone. Recent advances in large language models offer new avenues to support users with this task. While existing approaches excel at processing raw data to gain information, they are structurally unfit to pick up on subtle situational differences needed to distinguish similar data sources in this domain. To address this issue, we propose a decoupled, two-step agentic architecture. By separating orchestration from data analysis, the system leverages progressive data discovery infused with domain expert knowledge. This architecture allows the orchestration to intelligently select data sources and perform multi-hop reasoning across data structures while maintaining a slim internal context. Results from multiple state-of-the-art large language models demonstrate the framework's viability: with top-tier models achieving accuracies of up to 86% and demonstrating high robustness across evaluation runs. 

---
# KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates 

**Authors**: Yudong Li, Jiawei Cai, Linlin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2604.12397)  

**Abstract**: Standard Large Language Model (LLM) pre-training typically treats corpora as flattened token sequences, often overlooking the real-world context that humans naturally rely on to contextualize information. To bridge this gap, we introduce Knowledge Coordinate Conditioning (KoCo), a simple method that maps every document into a three-dimensional semantic coordinate. By prepending these coordinates as textual prefixes for pre-training, we aim to equip the model with explicit contextual awareness to learn the documents within the real-world knowledge structure. Experiment results demonstrate that KoCo significantly enhances performance across 10 downstream tasks and accelerates pre-training convergence by approximately 30\%. Furthermore, our analysis indicates that explicitly modeling knowledge coordinates helps the model distinguish stable facts from noise, effectively mitigating hallucination in generated outputs. 

---
# From Myopic Selection to Long-Horizon Awareness: Sequential LLM Routing for Multi-Turn Dialogue 

**Authors**: Jiarui Zhang, Xiangyu Liu, Yong Hu, Chaoyue Niu, Hang Zeng, Shaojie Tang, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.12385)  

**Abstract**: Multi-turn dialogue is the predominant form of interaction with large language models (LLMs). While LLM routing is effective in single-turn settings, existing methods fail to maximize cumulative performance in multi-turn dialogue due to interaction dynamics and delayed rewards. To address this challenge, we move from myopic, single-turn selection to long-horizon sequential routing for multi-turn dialogue. Accordingly, we propose DialRouter, which first performs MCTS to explore dialogue branches induced by different LLM selections and collect trajectories with high cumulative rewards. DialRouter then learns a lightweight routing policy from search-derived data, augmented with retrieval-based future state approximation, enabling multi-turn routing without online search. Experiments on both open-domain and domain-specific dialogue tasks across diverse candidate sets of both open-source and closed-source LLMs demonstrate that DialRouter significantly outperforms single LLMs and existing routing baselines in task success rate, while achieving a superior performance-cost trade-off when combined with a cost-aware reward. 

---
# ReasonXL: Shifting LLM Reasoning Language Without Sacrificing Performance 

**Authors**: Daniil Gurgurov, Tom Röhr, Sebastian von Rohrscheidt, Josef van Genabith, Alexander Löser, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2604.12378)  

**Abstract**: Despite advances in multilingual capabilities, most large language models (LLMs) remain English-centric in their training and, crucially, in their production of reasoning traces. Even when tasked with non-English problems, these models predominantly reason in English, creating a fundamental mismatch for non-English usage scenarios.
We address this disparity directly with three contributions. (i) We introduce ReasonXL, the first large-scale parallel corpus of cross-domain reasoning traces spanning five European languages (English, German, French, Italian, and Spanish), with over two million aligned samples per language, each comprising prompts, reasoning traces, and final outputs, enabling direct supervision of language-specific reasoning. (ii) Using ReasonXL, we demonstrate that LLMs can be adapted to reason entirely in a desired target language, using a simple two-stage pipeline of supervised fine-tuning (SFT) followed by reinforcement learning with verifiable rewards (RLVR). The resulting models match or exceed baseline performance, with minimal loss in general knowledge and broadly preserved cross-lingual transfer. (iii) We conduct an extensive representational analysis of the adaptation and find a clear functional division across model depth: early layers contain an activation bottleneck that causally determines language identity, while upper layers concentrate the weight and activation changes driven by adaptation. We further find that RLVR achieves greater behavioral divergence from the base model with smaller parameter updates than SFT, suggesting a more efficient representational rerouting despite much smaller weight updates. 

---
# SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models 

**Authors**: SungHo Kim, Juhyeong Park, Eda Atalay, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.12377)  

**Abstract**: Korean is a morphologically rich language with a featural writing system in which each character is systematically composed of subcharacter units known as Jamo. These subcharacters not only determine the visual structure of Korean but also encode frequent and linguistically meaningful morphophonological processes. However, most current Korean language models (LMs) are based on subword tokenization schemes, which are not explicitly designed to capture the internal compositional structure of characters. To address this limitation, we propose SCRIPT, a model-agnostic module that injects subcharacter compositional knowledge into Korean PLMs. SCRIPT allows to enhance subword embeddings with structural granularity, without requiring architectural changes or additional pre-training. As a result, SCRIPT enhances all baselines across various Korean natural language understanding (NLU) and generation (NLG) tasks. Moreover, beyond performance gains, detailed linguistic analyses show that SCRIPT reshapes the embedding space in a way that better captures grammatical regularities and semantically cohesive variations. Our code is available at this https URL. 

---
# Cooperative Memory Paging with Keyword Bookmarks for Long-Horizon LLM Conversations 

**Authors**: Ziyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12376)  

**Abstract**: When LLM conversations grow beyond the context window, old content must be evicted -- but how does the model recover it when needed? We propose cooperative paging: evicted segments are replaced with minimal keyword bookmarks ([pN:keywords], ~8-24 tokens each), and the model is given a recall() tool to retrieve full content on demand. On the LoCoMo benchmark (10 real multi-session conversations, 300+ turns), cooperative paging achieves the highest answer quality among six methods -- outperforming truncation, BM25, word-overlap retrieval, a search-tool baseline, and full context -- on four models (GPT-4o-mini, DeepSeek-v3.2, Claude Haiku, GLM-5), confirmed by four independent LLM judges ($p=0.017$, paired bootstrap). We then study the paging design space with a 5x4 ablation over boundary strategies and eviction policies (3,176 synthetic probes, 1,600 LoCoMo probes). Key findings: (1) coarse fixed-size pages (fixed_20) reach 96.7% while content-aware topic_shift collapses to 56.7%; (2) eviction policy choice is data-dependent (FIFO best on synthetic, LFU on LoCoMo); (3) two bookmark generation strategies improve over the heuristic baseline (+4.4 and +8.7 E2E points); (4) the remaining bottleneck is bookmark discrimination -- the model triggers recall() 96% of the time but selects the correct page only 57% when bookmarks are insufficiently distinctive. Keyword specificity alone accounts for a 25 percentage point accuracy difference. 

---
# Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness 

**Authors**: Tomer Ashuach, Liat Ein-Dor, Shai Gretz, Yoav Katz, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2604.12373)  

**Abstract**: Humans use introspection to evaluate their understanding through private internal states inaccessible to external observers. We investigate whether large language models possess similar privileged knowledge about answer correctness, information unavailable through external observation. We train correctness classifiers on question representations from both a model's own hidden states and external models, testing whether self-representations provide a performance advantage. On standard evaluation, we find no advantage: self-probes perform comparably to peer-model probes. We hypothesize this is due to high inter-model agreement of answer correctness. To isolate genuine privileged knowledge, we evaluate on disagreement subsets, where models produce conflicting predictions. Here, we discover domain-specific privileged knowledge: self-representations consistently outperform peer representations in factual knowledge tasks, but show no advantage in math reasoning. We further localize this domain asymmetry across model layers, finding that the factual advantage emerges progressively from early-to-mid layers onward, consistent with model-specific memory retrieval, while math reasoning shows no consistent advantage at any depth. 

---
# ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection 

**Authors**: Boyang Li, Hongzhe Shou, Yuanyuan Liang, Jingbin Zhang, Fang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.12321)  

**Abstract**: Existing Chinese toxic content detection methods mainly target sentence-level classification but often fail to provide readable and contiguous toxic evidence spans. We propose \textbf{ToxiTrace}, an explainability-oriented method for BERT-style encoders with three components: (1) \textbf{CuSA}, which refines encoder-derived saliency cues into fine-grained toxic spans with lightweight LLM guidance; (2) \textbf{GCLoss}, a gradient-constrained objective that concentrates token-level saliency on toxic evidence while suppressing irrelevant activations; and (3) \textbf{ARCL}, which constructs sample-specific contrastive reasoning pairs to sharpen the semantic boundary between toxic and non-toxic content. Experiments show that ToxiTrace improves classification accuracy and toxic span extraction while preserving efficient encoder-based inference and producing more coherent, human-readable explanations. We have released the model at this https URL. 

---
# CompliBench: Benchmarking LLM Judges for Compliance Violation Detection in Dialogue Systems 

**Authors**: Jingbo Yang, Guanyu Yao, Bairu Hou, Xinghan Yang, Nikolai Glushnev, Iwona Bialynicka-Birula, Duo Ding, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12312)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as task-oriented agents in enterprise environments, ensuring their strict adherence to complex, domain-specific operational guidelines is critical. While utilizing an LLM-as-a-Judge is a promising solution for scalable evaluation, the reliability of these judges in detecting specific policy violations remains largely unexplored. This gap is primarily due to the lack of a systematic data generation method, which has been hindered by the extensive cost of fine-grained human annotation and the difficulty of synthesizing realistic agent violations. In this paper, we introduce CompliBench, a novel benchmark designed to evaluate the ability of LLM judges to detect and localize guideline violations in multi-turn dialogues. To overcome data scarcity, we develop a scalable, automated data generation pipeline that simulates user-agent interactions. Our controllable flaw injection process automatically yields precise ground-truth labels for the violated guideline and the exact conversation turn, while an adversarial search method ensures these introduced perturbations are highly challenging. Our comprehensive evaluation reveals that current state-of-the-art proprietary LLMs struggle significantly with this task. In addition, we demonstrate that a small-scale judge model fine-tuned on our synthesized data outperforms leading LLMs and generalizes well to unseen business domains, highlighting our pipeline as an effective foundation for training robust generative reward models. 

---
# ContextLens: Modeling Imperfect Privacy and Safety Context for Legal Compliance 

**Authors**: Haoran Li, Yulin Chen, Huihao Jing, Wenbin Hu, Tsz Ho Li, Chanhou Lou, Hong Ting Tsang, Sirui Han, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.12308)  

**Abstract**: Individuals' concerns about data privacy and AI safety are highly contextualized and extend beyond sensitive patterns. Addressing these issues requires reasoning about the context to identify and mitigate potential risks. Though researchers have widely explored using large language models (LLMs) as evaluators for contextualized safety and privacy assessments, these efforts typically assume the availability of complete and clear context, whereas real-world contexts tend to be ambiguous and incomplete. In this paper, we propose ContextLens, a semi-rule-based framework that leverages LLMs to ground the input context in the legal domain and explicitly identify both known and unknown factors for legal compliance. Instead of directly assessing safety outcomes, our ContextLens instructs LLMs to answer a set of crafted questions that span over applicability, general principles and detailed provisions to assess compliance with pre-defined priorities and rules. We conduct extensive experiments on existing compliance benchmarks that cover the General Data Protection Regulation (GDPR) and the EU AI Act. The results suggest that our ContextLens can significantly improve LLMs' compliance assessment and surpass existing baselines without any training. Additionally, our ContextLens can further identify the ambiguous and missing factors. 

---
# Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Reasoning 

**Authors**: Houxing Ren, Mingjie Zhan, Zimu Lu, Ke Wang, Yunqiao Yang, Haotian Hou, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.12282)  

**Abstract**: Spreadsheets are central to real-world applications such as enterprise reporting, auditing, and scientific data management. Despite their ubiquity, existing large language model based approaches typically treat tables as plain text, overlooking critical layout cues and visual semantics. Moreover, real-world spreadsheets are often massive in scale, exceeding the input length that LLMs can efficiently process. To address these challenges, we propose SpreadsheetAgent, a two-stage multi-agent framework for spreadsheet understanding that adopts a step-by-step reading and reasoning paradigm. Instead of loading the entire spreadsheet at once, SpreadsheetAgent incrementally interprets localized regions through multiple modalities, including code execution results, images, and LaTeX tables. The method first constructs a structural sketch and row/column summaries, and then performs task-driven reasoning over this intermediate representation in the Solving Stage. To further enhance reliability, we design a verification module that validates extracted structures via targeted inspections, reducing error propagation and ensuring trustworthy inputs for downstream reasoning. Extensive experiments on two spreadsheet datasets demonstrate the effectiveness of our approach. With GPT-OSS-120B, SpreadsheetAgent achieves 38.16% on Spreadsheet Bench, outperforming the ChatGPT Agent baseline (35.27%) by 2.89 absolute points. These results highlight the potential of SpreadsheetAgent to advance robust and scalable spreadsheet understanding in real-world applications. Code is available at this https URL. 

---
# CascadeDebate: Multi-Agent Deliberation for Cost-Aware LLM Cascades 

**Authors**: Raeyoung Chang, Dongwook Kwon, Jisoo Lee, Nikhil Verma  

**Link**: [PDF](https://arxiv.org/pdf/2604.12262)  

**Abstract**: Cascaded LLM systems coordinate models of varying sizes with human experts to balance accuracy, cost, and abstention under uncertainty. However, single-model tiers at each stage often struggle with ambiguous queries, triggering premature escalations to costlier models or experts due to under-confidence and inefficient compute scaling. CascadeDebate addresses this gap by inserting multi-agent deliberation directly at each tier's escalation boundary. Confidence-based routers activate lightweight agent ensembles only for uncertain cases, enabling consensus-driven resolution of ambiguities internally without invoking higher-cost upgrades. Our unified architecture alternates single-model inference with selective multi-agent deliberation across model scales, culminating in human experts as the final fallback. This design scales test-time compute dynamically according to query difficulty. Across five benchmarks spanning science, medicine, and general knowledge, CascadeDebate outperforms strong single-model cascades and standalone multi-agent systems by up to 26.75 percent. An online threshold optimizer proves essential, boosting accuracy by 20.98 to 52.33 percent relative improvement over fixed policies and enabling elastic adaptation to real-world distributions. 

---
# Coding-Free and Privacy-Preserving MCP Framework for Clinical Agentic Research Intelligence System 

**Authors**: Taehun Kim, Hyeryun Park, Hyeonhoon Lee, Yushin Lee, Kyungsang Kim, Hyung-Chul Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.12258)  

**Abstract**: Clinical research involves labor-intensive processes such as study design, cohort construction, model development, and documentation, requiring domain expertise, programming skills, and access to sensitive patient data. These demands create barriers for clinicians and external researchers conducting data-driven studies. To overcome these limitations, we developed a Clinical Agentic Research Intelligence System (CARIS) that automates the clinical research workflow while preserving data privacy, enabling comprehensive studies without direct access to raw data. CARIS integrates Large Language Models (LLMs) with modular tools via the Model Context Protocol (MCP), enabling natural language-driven orchestration of appropriate tools. Databases remain securely within the MCP server, and users access only the outputs and final research reports. Based on user intent, CARIS automatically executes the full pipeline: research planning, literature search, cohort construction, Institutional Review Board (IRB) documentation, Vibe Machine Learning (ML), and report generation, with iterative human-in-the-loop refinement. We evaluated CARIS on three heterogeneous datasets with distinct clinical tasks. Research plans and IRB documents were finalized within three to four iterations, using evidence from literature and data. The system supported Vibe ML by exploring feature-model combinations, ranking the top ten models, and generating performance visualizations. Final reports showed high completeness based on a checklist derived from the TRIPOD+AI framework, achieving 96% coverage in LLM evaluation and 82% in human evaluation. CARIS demonstrates that agentic AI can transform clinical hypotheses into executable research workflows across heterogeneous datasets. By eliminating the need for coding and direct data access, the system lowers barriers and bridges public and private clinical data environments. 

---
# SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration 

**Authors**: Zhuofan Wen, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.12247)  

**Abstract**: Speculative decoding has emerged as a promising approach to accelerate autoregressive inference in large language models (LLMs). Self-draft methods, which leverage the base LLM itself for speculation, avoid the overhead of auxiliary draft models but face limitations: shallow layers often produce overconfident yet incorrect token predictions, and the presence of difficult tokens in a draft sequence forces redundant computation through deeper layers, undermining both draft acceptance and overall speedup. To address these issues, we propose a novel self-draft framework that suppresses spurious confidence via layer-wise temperature annealing in early-exit decision and adaptively bounds speculation length based on token-wise decoding difficulty. By reprocessing the hidden states of draft tokens in a unified parallel pass through deep layers, our method maintains exact output equivalence with the original model while maximizing computational efficiency. It requires no modifications to the base LLM parameters and achieves up to 2.33x wall-time speedup over standard autoregressive decoding across diverse long-form generation tasks and multiple model architectures. 

---
# Continuous Knowledge Metabolism: Generating Scientific Hypotheses from Evolving Literature 

**Authors**: Jinkai Tao, Yubo Wang, Xiaoyu Liu, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12243)  

**Abstract**: Scientific hypothesis generation requires tracking how knowledge evolves, not just what is currently known. We introduce Continuous Knowledge Metabolism (CKM), a framework that processes scientific literature through sliding time windows and incrementally updates a structured knowledge base as new findings arrive. We present CKM-Lite, an efficient variant that achieves strong predictive coverage through incremental accumulation, outperforming batch processing on hit rate (+2.8%, p=0.006), hypothesis yield (+3.6, p<0.001), and best-match alignment (+0.43, p<0.001) while reducing token cost by 92%. To understand what drives these differences, we develop CKM-Full, an instrumented variant that categorizes each new finding as novel, confirming, or contradicting, detects knowledge change signals, and conditions hypothesis generation on the full evolution trajectory. Analyzing 892 hypotheses generated by CKM-Full across 50 research topics, alongside parallel runs of the other variants, we report four empirical observations: (1) incremental processing outperforms batch baseline across predictive and efficiency metrics; (2) change-aware instrumentation is associated with higher LLM-judged novelty (Cohen's d=3.46) but lower predictive coverage, revealing a quality-coverage trade-off; (3) a field's trajectory stability is associated with hypothesis success (r=-0.28, p=0.051), suggesting boundary conditions for literature-based prediction; (4) knowledge convergence signals are associated with nearly 5x higher hit rate than contradiction signals, pointing to differential predictability across change types. These findings suggest that the character of generated hypotheses is shaped not only by how much literature is processed, but also by how it is processed. They further indicate that evaluation frameworks must account for the quality-coverage trade-off rather than optimize for a single metric. 

---
# Thought-Retriever: Don't Just Retrieve Raw Data, Retrieve Thoughts for Memory-Augmented Agentic Systems 

**Authors**: Tao Feng, Pengrui Han, Guanyu Lin, Ge Liu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2604.12231)  

**Abstract**: Large language models (LLMs) have transformed AI research thanks to their powerful internal capabilities and knowledge. However, existing LLMs still fail to effectively incorporate the massive external knowledge when interacting with the world. Although retrieval-augmented LLMs are proposed to mitigate the issue, they are still fundamentally constrained by the context length of LLMs, as they can only retrieve top-K raw data chunks from the external knowledge base which often consists of millions of data chunks. Here we propose Thought-Retriever, a novel model-agnostic algorithm that helps LLMs generate output conditioned on arbitrarily long external data, without being constrained by the context length or number of retrieved data chunks. Our key insight is to let an LLM fully leverage its intermediate responses generated when solving past user queries (thoughts), filtering meaningless and redundant thoughts, organizing them in thought memory, and retrieving the relevant thoughts when addressing new queries. This effectively equips LLM-based agents with a self-evolving long-term memory that grows more capable through continuous interaction. Besides algorithmic innovation, we further meticulously prepare a novel benchmark, AcademicEval, which requires an LLM to faithfully leverage ultra-long context to answer queries based on real-world academic papers. Extensive experiments on AcademicEval and two other public datasets validate that Thought-Retriever remarkably outperforms state-of-the-art baselines, achieving an average increase of at least 7.6% in F1 score and 16% in win rate across various tasks. More importantly, we further demonstrate two exciting findings: (1) Thought-Retriever can indeed help LLM self-evolve after solving more user queries; (2) Thought-Retriever learns to leverage deeper thoughts to answer more abstract user queries. 

---
# LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines 

**Authors**: Jiechao Gao, Rohan Kumar Yadav, Yuangang Li, Yuandong Pan, Jie Wang, Ying Liu, Michael Lepech  

**Link**: [PDF](https://arxiv.org/pdf/2604.12223)  

**Abstract**: Pretrained language models (PLMs) like BERT provide strong semantic representations but are costly and opaque, while symbolic models such as the Tsetlin Machine (TM) offer transparency but lack semantic generalization. We propose a semantic bootstrapping framework that transfers LLM knowledge into symbolic form, combining interpretability with semantic capacity. Given a class label, an LLM generates sub-intents that guide synthetic data creation through a three-stage curriculum (seed, core, enriched), expanding semantic diversity. A Non-Negated TM (NTM) learns from these examples to extract high-confidence literals as interpretable semantic cues. Injecting these cues into real data enables a TM to align clause logic with LLM-inferred semantics. Our method requires no embeddings or runtime LLM calls, yet equips symbolic models with pretrained semantic priors. Across multiple text classification tasks, it improves interpretability and accuracy over vanilla TM, achieving performance comparable to BERT while remaining fully symbolic and efficient. 

---
# Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score 

**Authors**: Manh Nguyen, Sunil Gupta, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2604.12196)  

**Abstract**: Large language models (LLMs) frequently generate multiple candidate responses for a given prompt, yet selecting the most reliable one remains challenging, especially when correctness diverges from surface-level majority agreement. Existing approaches, such as self-consistency, rely on discrete voting, while probability-based methods often fail to capture relationships among candidate answers or tend to underweight high-quality but less frequent responses, and do not fully leverage the geometric structure of answer representations. To address these limitations, we introduce Radial Consensus Score (RCS), a simple, efficient, and training-free method for best-of-N selection. RCS models semantic consensus by computing a weighted Fréchet mean (semantic center) of answer embeddings and ranking candidates by their radial distance to this center. Importantly, RCS provides a general framework that supports multiple weighting schemes, including uniform, frequency-based, and probability-based variants, enabling flexible integration of agreement signals and model confidence while remaining fully applicable in black-box settings. Extensive experiments across seven benchmarks covering short-form QA and long-form reasoning tasks, and five open-weight models, demonstrate that RCS variants consistently outperform strong baselines, with gains becoming more pronounced as the sampling budget increases. RCS also serves as an effective drop-in replacement for majority voting in multi-agent debate and exhibits strong robustness in black-box scenarios. Overall, these results highlight geometric consensus as a scalable and broadly applicable principle for reliable answer selection, extending beyond majority voting to more expressive and robust aggregation in LLM inference. 

---
# Representing expertise accelerates learning from pedagogical interaction data 

**Authors**: Dhara Yu, Karthikeya Kaushik, Bill D. Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2604.12195)  

**Abstract**: Work in cognitive science and artificial intelligence has suggested that exposing learning agents to traces of interaction between multiple individuals can improve performance in a variety of settings, yet it remains unknown which features of interactions contribute to this improvement. We examined the factors that support the effectiveness of interaction data, using a controlled paradigm that allowed us to precisely operationalize key distinctions between interaction and an expert acting alone. We generated synthetic datasets of simple interactions between an expert and a novice in a spatial navigation task, and then trained transformer models on those datasets, evaluating performance after exposure to different datasets. Our experiments showed that models trained on pedagogical interactions were more robust across a variety of scenarios compared to models trained only on expert demonstrations, and that having the ability to represent epistemically distinct agents led to expert-like behavior even when expert behavior was rarely observed. 

---
# Knowledge Is Not Static: Order-Aware Hypergraph RAG for Language Models 

**Authors**: Keshu Wu, Chenchen Kuai, Zihao Li, Jiwan Jiang, Shiyu Shen, Shian Wang, Chan-Wei Hu, Zhengzhong Tu, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.12185)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models by grounding outputs in retrieved knowledge. However, existing RAG methods including graph- and hypergraph-based approaches treat retrieved evidence as an unordered set, implicitly assuming permutation invariance. This assumption is misaligned with many real-world reasoning tasks, where outcomes depend not only on which interactions occur, but also on the order in which they unfold. We propose Order-Aware Knowledge Hypergraph RAG (OKH-RAG), which treats order as a first-class structural property. OKH-RAG represents knowledge as higher-order interactions within a hypergraph augmented with precedence structure, and reformulates retrieval as sequence inference over hyperedges. Instead of selecting independent facts, it recovers coherent interaction trajectories that reflect underlying reasoning processes. A learned transition model infers precedence directly from data without requiring explicit temporal supervision. We evaluate OKH-RAG on order-sensitive question answering and explanation tasks, including tropical cyclone and port operation scenarios. OKH-RAG consistently outperforms permutation-invariant baselines, and ablations show that these gains arise specifically from modeling interaction order. These results highlight a key limitation of set-based retrieval: effective reasoning requires not only retrieving relevant evidence, but organizing it into structured sequences. 

---
# AgenticAI-DialogGen: Topic-Guided Conversation Generation for Fine-Tuning and Evaluating Short- and Long-Term Memories of LLMs 

**Authors**: Manoj Madushanka Perera, Adnan Mahmood, Kasun Eranda Wijethilake, Quan Z. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.12179)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have improved their ability to process extended conversational contexts, yet fine-tuning and evaluating short- and long-term memories remain difficult due to the absence of datasets that encode both short- and long-term conversational history. Existing conversational datasets lack memory grounding, overlook topic continuity, or rely on costly human annotation. To address these gaps, we introduce AgenticAI-DialogGen, a modular agent-based framework that generates persona-grounded and topic-guided conversations without human supervision. The framework uses LLM agents to extract knowledge graphs, identify topics, build speaker personas, and simulate topic-guided conversations from unstructured conversations. A QA module generates memory-grounded Question Answer (QA) pairs drawn from short- and long-term conversational histories. We also generated a new dataset entitled, TopicGuidedChat (TGC), where long-term memory is encoded as speaker-specific knowledge graphs and short-term memory as newly generated topic-guided conversations. Evaluations depict that AgenticAI-DialogGen yields higher conversational quality and LLMs fine-tuned on TGC dataset achieve improved performance on memory-grounded QA tasks. 

---
# AlphaEval: Evaluating Agents in Production 

**Authors**: Pengrui Lu, Bingyu Xu, Wenjun Zhang, Shengjia Hua, Xuanjian Gao, Ranxiang Ge, Lyumanshan Ye, Linxuan Wu, Yiran Li, Junfei Fish Yu, Yibo Zhang, Ruixin Li, Manxiang Li, Xiao Han, Xiaocong Zhou, Guangyao Chi, Zisheng Chen, Kaishen Chen, Kun Wang, Qihua Xu, Fengyue Meng, Yuchen Ni, Jiajun Li, Jinxiu Liu, Danfeng Zhang, Jingru Zhao, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12162)  

**Abstract**: The rapid deployment of AI agents in commercial settings has outpaced the development of evaluation methodologies that reflect production realities. Existing benchmarks measure agent capabilities through retrospectively curated tasks with well-specified requirements and deterministic metrics -- conditions that diverge fundamentally from production environments where requirements contain implicit constraints, inputs are heterogeneous multi-modal documents with information fragmented across sources, tasks demand undeclared domain expertise, outputs are long-horizon professional deliverables, and success is judged by domain experts whose standards evolve over time. We present AlphaEval, a production-grounded benchmark of 94 tasks sourced from seven companies deploying AI agents in their core business, spanning six O*NET (Occupational Information Network) domains. Unlike model-centric benchmarks, AlphaEval evaluates complete agent products -- Claude Code, Codex, etc. -- as commercial systems, capturing performance variations invisible to model-level evaluation. Our evaluation framework covers multiple paradigms (LLM-as-a-Judge, reference-driven metrics, formal verification, rubric-based assessment, automated UI testing, etc.), with individual domains composing multiple paradigms. Beyond the benchmark itself, we contribute a requirement-to-benchmark construction framework -- a systematic methodology that transforms authentic production requirements into executable evaluation tasks in minimal time. This framework standardizes the entire pipeline from requirement to evaluation, providing a reproducible, modular process that any organization can adopt to construct production-grounded benchmarks for their own domains. 

---
# When Self-Reference Fails to Close: Matrix-Level Dynamics in Large Language Models 

**Authors**: Ji Ho Bae  

**Link**: [PDF](https://arxiv.org/pdf/2604.12128)  

**Abstract**: We investigate how self-referential inputs alter the internal matrix dynamics of large language models. Measuring 106 scalar metrics across up to 7 analysis passes on four models from three architecture families -- Qwen3-VL-8B, Llama-3.2-11B, Llama-3.3-70B, and Gemma-2-9B -- over 300 prompts in a 14-level hierarchy at three temperatures ($T \in \{0.0, 0.3, 0.7\}$), we find that self-reference alone is not destabilizing: grounded self-referential statements and meta-cognitive prompts are markedly more stable than paradoxical self-reference on key collapse-related metrics, and on several such metrics can be as stable as factual controls. Instability concentrates in prompts inducing non-closing truth recursion (NCTR) -- truth-value computations with no finite-depth resolution. NCTR prompts produce anomalously elevated attention effective rank -- indicating attention reorganization with global dispersion rather than simple concentration collapse -- and key metrics reach Cohen's $d = 3.14$ (attention effective rank) to $3.52$ (variance kurtosis) vs. stable self-reference in the 70B model; 281/397 metric-model combinations differentiate NCTR from stable self-reference after FDR correction ($q < 0.05$), 198 with $|d| > 0.8$. Per-layer SVD confirms disruption at every sampled layer ($d > +1.0$ in all three models analyzed), ruling out aggregation artifacts. A classifier achieves AUC $0.81$-$0.90$; 30 minimal pairs yield 42/387 significant combinations; 43/106 metrics replicate across all four models. We connect these observations to three classical matrix-semigroup problems and propose, as a conjecture, that NCTR forces finite-depth transformers toward dynamical regimes where these problems concentrate. NCTR prompts also produce elevated contradictory output ($+34$-$56$ percentage points vs. controls), suggesting practical relevance for understanding self-referential failure modes. 

---
# Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories 

**Authors**: Zhanwei Cao, YeoJin Go, Yifan Hu, Shanu Sushmita  

**Link**: [PDF](https://arxiv.org/pdf/2604.12097)  

**Abstract**: Large language models (LLMs) are increasingly used in daily applications, from content generation to code writing, where each interaction treats the model as stateless, generating responses independently without memory. Yet human writing is inherently longitudinal: authors' styles and cognitive states evolve across months and years. This raises a central question: can LLMs reproduce such temporal structure across extended time periods? We construct and publicly release a longitudinal dataset of 412 human authors and 6,086 documents spanning 2012--2024 across three domains (academic abstracts, blogs, news) and compare them to trajectories generated by three representative LLMs under standard and history-conditioned generation settings. Using drift and variance-based metrics over semantic, lexical, and cognitive-emotional representations, we find temporal flattening in LLM-generated text. LLMs produce greater lexical diversity but exhibit substantially reduced semantic and cognitive-emotional drift relative to humans. These differences are highly predictive: temporal variability patterns alone achieve 94% accuracy and 98% ROC-AUC in distinguishing human from LLM trajectories. Our results demonstrate that temporal flattening persists regardless of whether LLMs generate independently or with access to incremental history, revealing a fundamental property of current deployment paradigms. This gap has direct implications for applications requiring authentic temporal structure, such as synthetic training data and longitudinal text modeling. 

---
# Narrative over Numbers: The Identifiable Victim Effect and its Amplification Under Alignment and Reasoning in Large Language Models 

**Authors**: Syed Rifat Raiyan  

**Link**: [PDF](https://arxiv.org/pdf/2604.12076)  

**Abstract**: The Identifiable Victim Effect (IVE) $-$ the tendency to allocate greater resources to a specific, narratively described victim than to a statistically characterized group facing equivalent hardship $-$ is one of the most robust findings in moral psychology and behavioural economics. As large language models (LLMs) assume consequential roles in humanitarian triage, automated grant evaluation, and content moderation, a critical question arises: do these systems inherit the affective irrationalities present in human moral reasoning? We present the first systematic, large-scale empirical investigation of the IVE in LLMs, comprising N=51,955 validated API trials across 16 frontier models spanning nine organizational lineages (Google, Anthropic, OpenAI, Meta, DeepSeek, xAI, Alibaba, IBM, and Moonshot). Using a suite of ten experiments $-$ porting and extending canonical paradigms from Small et al. (2007) and Kogut and Ritov (2005) $-$ we find that the IVE is prevalent but strongly modulated by alignment training. Instruction-tuned models exhibit extreme IVE (Cohen's d up to 1.56), while reasoning-specialized models invert the effect (down to d=-0.85). The pooled effect (d=0.223, p=2e-6) is approximately twice the single-victim human meta-analytic baseline (d$\approx$0.10) reported by Lee and Feeley (2016) $-$ and likely exceeds the overall human pooled effect by a larger margin, given that the group-victim human effect is near zero. Standard Chain-of-Thought (CoT) prompting $-$ contrary to its role as a deliberative corrective $-$ nearly triples the IVE effect size (from d=0.15 to d=0.41), while only utilitarian CoT reliably eliminates it. We further document psychophysical numbing, perfect quantity neglect, and marginal in-group/out-group cultural bias, with implications for AI deployment in humanitarian and ethical decision-making contexts. 

---
# Robust Explanations for User Trust in Enterprise NLP Systems 

**Authors**: Guilin Zhang, Kai Zhao, Jeffrey Friedman, Xu Chu, Amine Anoun, Jerry Ting  

**Link**: [PDF](https://arxiv.org/pdf/2604.12069)  

**Abstract**: Robust explanations are increasingly required for user trust in enterprise NLP, yet pre-deployment validation is difficult in the common case of black-box deployment (API-only access) where representation-based explainers are infeasible and existing studies provide limited guidance on whether explanations remain stable under real user noise, especially when organizations migrate from encoder classifiers to decoder LLMs. To close this gap, we propose a unified black-box robustness evaluation framework for token-level explanations based on leave-one-out occlusion, and operationalize explanation robustness with top-token flip rate under realistic perturbations (swap, deletion, shuffling, and back-translation) at multiple severity levels. Using this protocol, we conduct a systematic cross-architecture comparison across three benchmark datasets and six models spanning encoder and decoder families (BERT, RoBERTa, Qwen 7B/14B, Llama 8B/70B; 64,800 cases). We find that decoder LLMs produce substantially more stable explanations than encoder baselines (73% lower flip rates on average), and that stability improves with model scale (44% gain from 7B to 70B). Finally, we relate robustness improvements to inference cost, yielding a practical cost-robustness tradeoff curve that supports model and explanation selection prior to deployment in compliance-sensitive applications. 

---
# LoSA: Locality Aware Sparse Attention for Block-Wise Diffusion Language Models 

**Authors**: Haocheng Xi, Harman Singh, Yuezhou Hu, Coleman Hooper, Rishabh Tiwari, Aditya Tomar, Minjae Lee, Wonjun Kang, Michael Mahoney, Chenfeng Xu, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2604.12056)  

**Abstract**: Block-wise diffusion language models (DLMs) generate multiple tokens in any order, offering a promising alternative to the autoregressive decoding pipeline. However, they still remain bottlenecked by memory-bound attention in long-context scenarios. Naive sparse attention fails on DLMs due to a KV Inflation problem, where different queries select different prefix positions, making the union of accessed KV pages large. To address this, we observe that between consecutive denoising steps, only a small fraction of active tokens exhibit significant hidden-state changes, while the majority of stable tokens remain nearly constant. Based on this insight, we propose LOSA (Locality-aware Sparse Attention), which reuses cached prefix-attention results for stable tokens and applies sparse attention only to active tokens. This substantially shrinks the number of KV indices that must be loaded, yielding both higher speedup and higher accuracy. Across multiple block-wise DLMs and benchmarks, LOSA preserves near-dense accuracy while significantly improving efficiency, achieving up to +9 points in average accuracy at aggressive sparsity levels while maintaining 1.54x lower attention density. It also achieves up to 4.14x attention speedup on RTX A6000 GPUs, demonstrating the effectiveness of the proposed method. 

---
# Leveraging Weighted Syntactic and Semantic Context Assessment Summary (wSSAS) Towards Text Categorization Using LLMs 

**Authors**: Shreeya Verma Kathuria, Nitin Mayande, Sharookh Daruwalla, Nitin Joglekar, Charles Weber  

**Link**: [PDF](https://arxiv.org/pdf/2604.12049)  

**Abstract**: The use of Large Language Models (LLMs) for reliable, enterprise-grade analytics such as text categorization is often hindered by the stochastic nature of attention mechanisms and sensitivity to noise that compromise their analytical precision and reproducibility. To address these technical frictions, this paper introduces the Weighted Syntactic and Semantic Context Assessment Summary (wSSAS), a deterministic framework designed to enforce data integrity on large-scale, chaotic datasets. We propose a two-phased validation framework that first organizes raw text into a hierarchical classification structure containing Themes, Stories, and Clusters. It then leverages a Signal-to-Noise Ratio (SNR) to prioritize high-value semantic features, ensuring the model's attention remains focused on the most representative data points. By incorporating this scoring mechanism into a Summary-of-Summaries (SoS) architecture, the framework effectively isolates essential information and mitigates background noise during data aggregation.
Experimental results using Gemini 2.0 Flash Lite across diverse datasets - including Google Business reviews, Amazon Product reviews, and Goodreads Book reviews - demonstrate that wSSAS significantly improves clustering integrity and categorization accuracy. Our findings indicate that wSSAS reduces categorization entropy and provides a reproducible pathway for improving LLM based summaries based on a high-precision, deterministic process for large-scale text categorization. 

---
# Empirical Evaluation of PDF Parsing and Chunking for Financial Question Answering with RAG 

**Authors**: Omar El Bachyr, Yewei Song, Saad Ezzini, Jacques Klein, Tegawendé F. Bissyandé, Anas Zilali, Ulrick Ble, Anne Goujon  

**Link**: [PDF](https://arxiv.org/pdf/2604.12047)  

**Abstract**: PDF files are primarily intended for human reading rather than automated processing. In addition, the heterogeneous content of PDFs, such as text, tables, and images, poses significant challenges for parsing and information extraction. To address these difficulties, both practitioners and researchers are increasingly developing new methods, including the promising Retrieval-Augmented Generation (RAG) systems to automated PDF processing. However, there is no comprehensive study investigating how different components and design choices affect the performance of a RAG system for understanding PDFs. In this paper, we propose such a study (1) by focusing on Question Answering, a specific language understanding task, and (2) by leveraging two benchmarks from the financial domain, including TableQuest, our newly generated, publicly available benchmark. We systematically examine multiple PDF parsers and chunking strategies (with varied overlap), along with their potential synergies in preserving document structure and ensuring answer correctness. Overall, our results offer practical guidelines for building robust RAG pipelines for PDF understanding. 

---
# Think Through Uncertainty: Improving Long-Form Generation Factuality via Reasoning Calibration 

**Authors**: Xin Liu, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12046)  

**Abstract**: Large language models (LLMs) often hallucinate in long-form generation. Existing approaches mainly improve factuality through post-hoc revision or reinforcement learning (RL) with correctness-based rewards, but they do not teach the model to estimate which parts of its generation are reliable. As a result, models may still state incorrect claims confidently in their responses. Recent advances in reasoning have significantly improved LLM performance, and have been leveraged to estimate confidence by incorporating calibration into RL objectives. However, existing approaches remain limited to a single scalar confidence for the entire response, which is insufficient for long-form generation where uncertainty varies across individual claims. To mitigate this problem, we propose CURE, a framework that improves long-form factuality by teaching LLMs to reason about uncertainty at the claim level. We first introduce a Claim-Aware Reasoning Protocol, which structures outputs into atomic claims paired with explicit confidence estimates. We then develop a multi-stage training pipeline that aligns model confidence with claims' correctness and then optimizes on factuality. The resulting calibrated confidence further enables selective prediction, allowing the model to abstain from uncertain claims at inference time. Experiments on four long-form factuality benchmarks show that CURE consistently improves factual accuracy over competitive supervised and RL baselines, while maintaining factual recall. In particular, it improves claim-level accuracy by up to 39.9% on Biography generation. These gains are accompanied by improved calibration, as reflected by a 16.0% increase in AUROC on FactBench. 

---
# Benchmarking Deflection and Hallucination in Large Vision-Language Models 

**Authors**: Nicholas Moratelli, Christopher Davis, Leonardo F. R. Ribeiro, Bill Byrne, Gonzalo Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2604.12033)  

**Abstract**: Large Vision-Language Models (LVLMs) increasingly rely on retrieval to answer knowledge-intensive multimodal questions. Existing benchmarks overlook conflicts between visual and textual evidence and the importance of generating deflections (e.g., Sorry, I cannot answer...) when retrieved knowledge is incomplete. These benchmarks also suffer from rapid obsolescence, as growing LVLM training sets allow models to answer many questions without retrieval. We address these gaps with three contributions. First, we propose a dynamic data curation pipeline that preserves benchmark difficulty over time by filtering for genuinely retrieval-dependent samples. Second, we introduce VLM-DeflectionBench, a benchmark of 2,775 samples spanning diverse multimodal retrieval settings, designed to probe model behaviour under conflicting or insufficient evidence. Third, we define a fine-grained evaluation protocol with four scenarios that disentangle parametric memorization from retrieval robustness. Experiments across 20 state-of-the-art LVLMs indicate that models usually fail to deflect in the presence of noisy or misleading evidence. Our results highlight the need to evaluate not only what models know, but how they behave when they do not, and serve as a reusable and extensible benchmark for reliable KB-VQA evaluation. All resources will be publicly available upon publication. 

---
# LLMs Struggle with Abstract Meaning Comprehension More Than Expected 

**Authors**: Hamoud Alhazmi, Jiachen Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12018)  

**Abstract**: Understanding abstract meanings is crucial for advanced language comprehension. Despite extensive research, abstract words remain challenging due to their non-concrete, high-level semantics. SemEval-2021 Task 4 (ReCAM) evaluates models' ability to interpret abstract concepts by presenting passages with questions and five abstract options in a cloze-style format. Key findings include: (1) Most large language models (LLMs), including GPT-4o, struggle with abstract meaning comprehension under zero-shot, one-shot, and few-shot settings, while fine-tuned models like BERT and RoBERTa perform better. (2) A proposed bidirectional attention classifier, inspired by human cognitive strategies, enhances fine-tuned models by dynamically attending to passages and options. This approach improves accuracy by 4.06 percent on Task 1 and 3.41 percent on Task 2, demonstrating its potential for abstract meaning comprehension. 

---
# Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision 

**Authors**: Yinghui He, Simran Kaur, Adithya Bhaskar, Yongjin Yang, Jiarui Liu, Narutatsu Ri, Liam Fowl, Abhishek Panigrahi, Danqi Chen, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2604.12002)  

**Abstract**: Current post-training methods in verifiable settings fall into two categories. Reinforcement learning (RLVR) relies on binary rewards, which are broadly applicable and powerful, but provide only sparse supervision during training. Distillation provides dense token-level supervision, typically obtained from an external teacher or using high-quality demonstrations. Collecting such supervision can be costly or unavailable. We propose Self-Distillation Zero (SD-Zero), a method that is substantially more training sample-efficient than RL and does not require an external teacher or high-quality demonstrations. SD-Zero trains a single model to play two roles: a Generator, which produces an initial response, and a Reviser, which conditions on that response and its binary reward to produce an improved response. We then perform on-policy self-distillation to distill the reviser into the generator, using the reviser's token distributions conditioned on the generator's response and its reward as supervision. In effect, SD-Zero trains the model to transform binary rewards into dense token-level self-supervision. On math and code reasoning benchmarks with Qwen3-4B-Instruct and Olmo-3-7B-Instruct, SD-Zero improves performance by at least 10% over the base models and outperforms strong baselines, including Rejection Fine-Tuning (RFT), GRPO, and Self-Distillation Fine-Tuning (SDFT), under the same question set and training sample budget. Extensive ablation studies show two novel characteristics of our proposed algorithm: (a) token-level self-localization, where the reviser can identify the key tokens that need to be revised in the generator's response based on reward, and (b) iterative self-evolution, where the improving ability to revise answers can be distilled back into generation performance with regular teacher synchronization. 

---
# Filtered Reasoning Score: Evaluating Reasoning Quality on a Model's Most-Confident Traces 

**Authors**: Manas Pathak, Xingyao Chen, Shuozhe Li, Amy Zhang, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2604.11996)  

**Abstract**: Should we trust Large Language Models (LLMs) with high accuracy? LLMs achieve high accuracy on reasoning benchmarks, but correctness alone does not reveal the quality of the reasoning used to produce it. This highlights a fundamental limitation of outcome-based evaluation: models may arrive at correct answers through flawed reasoning, and models with substantially different reasoning capabilities can nevertheless exhibit similar benchmark accuracy, for example due to memorization or over-optimization. In this paper, we ask: given existing benchmarks, can we move beyond outcome-based evaluation to assess the quality of reasoning itself? We seek metrics that (1) differentiate models with similar accuracy and (2) are robust to variations in input prompts and generation configurations. To this end, we propose a reasoning score that evaluates reasoning traces along dimensions such as faithfulness, coherence, utility, and factuality. A remaining question is how to aggregate this score across multiple sampled traces. Naively averaging them is undesirable, particularly in long-horizon settings, where the number of possible trajectories grows rapidly, and low-confidence correct traces are more likely to be coincidental. To address this, we introduce the Filtered Reasoning Score (FRS), which computes reasoning quality using only the top-K% most confident traces. Evaluating with FRS, models that are indistinguishable under standard accuracy exhibit significant differences in reasoning quality. Moreover, models with higher FRS on one benchmark tend to perform better on other reasoning benchmarks, in both accuracy and reasoning quality. Together, these findings suggest that FRS complements accuracy by capturing a model's transferable reasoning capabilities. We open source our evaluation codebase: this https URL. 

---
# SceneCritic: A Symbolic Evaluator for 3D Indoor Scene Synthesis 

**Authors**: Kathakoli Sengupta, Kai Ao, Paola Cascante-Bonilla  

**Link**: [PDF](https://arxiv.org/pdf/2604.13035)  

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) increasingly generate indoor scenes through intermediate structures such as layouts and scene graphs, yet evaluation still relies on LLM or VLM judges that score rendered views, making judgments sensitive to viewpoint, prompt phrasing, and hallucination. When the evaluator is unstable, it becomes difficult to determine whether a model has produced a spatially plausible scene or whether the output score reflects the choice of viewpoint, rendering, or prompt. We introduce SceneCritic, a symbolic evaluator for floor-plan-level layouts. SceneCritic's constraints are grounded in SceneOnto, a structured spatial ontology we construct by aggregating indoor scene priors from 3D-FRONT, ScanNet, and Visual Genome. SceneOnto traverses this ontology to jointly verify semantic, orientation, and geometric coherence across object relationships, providing object-level and relationship-level assessments that identify specific violations and successful placements. Furthermore, we pair SceneCritic with an iterative refinement test bed that probes how models build and revise spatial structure under different critic modalities: a rule-based critic using collision constraints as feedback, an LLM critic operating on the layout as text, and a VLM critic operating on rendered observations. Through extensive experiments, we show that (a) SceneCritic aligns substantially better with human judgments than VLM-based evaluators, (b) text-only LLMs can outperform VLMs on semantic layout quality, and (c) image-based VLM refinement is the most effective critic modality for semantic and orientation correction. 

---
# Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe 

**Authors**: Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.13016)  

**Abstract**: On-policy distillation (OPD) has become a core technique in the post-training of large language models, yet its training dynamics remain poorly understood. This paper provides a systematic investigation of OPD dynamics and mechanisms. We first identify that two conditions govern whether OPD succeeds or fails: (i) the student and teacher should share compatible thinking patterns; and (ii) even with consistent thinking patterns and higher scores, the teacher must offer genuinely new capabilities beyond what the student has seen during training. We validate these findings through weak-to-strong reverse distillation, showing that same-family 1.5B and 7B teachers are distributionally indistinguishable from the student's perspective. Probing into the token-level mechanism, we show that successful OPD is characterized by progressive alignment on high-probability tokens at student-visited states, a small shared token set that concentrates most of the probability mass (97%-99%). We further propose two practical strategies to recover failing OPD: off-policy cold start and teacher-aligned prompt selection. Finally, we show that OPD's apparent free lunch of dense token-level reward comes at a cost, raising the question of whether OPD can scale to long-horizon distillation. 

---
# RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair 

**Authors**: Jagadeesh Rachapudi, Pranav Singh, Ritali Vatsi, Praful Hambarde, Amit Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2604.12820)  

**Abstract**: Large language models (LLMs) inherently absorb harmful knowledge, misinformation, and personal data during pretraining on large-scale web corpora, with no native mechanism for selective removal. While machine unlearning offers a principled solution, existing approaches are provider-centric, requiring retraining pipelines, curated retain datasets, and direct intervention by model service providers (MSPs), thereby excluding end users from controlling their own data. We introduce Interactive Machine Unlearning (IMU), a new paradigm in which users can instruct LLMs to forget targeted knowledge through natural language at inference time. To realize IMU, we propose RePAIR, a prompt-aware model repair framework comprising (i) a watchdog model for unlearning intent detection, (ii) a surgeon model for generating repair procedures, and (iii) a patient model whose parameters are updated autonomously. At the core of RePAIR, we develop Steering Through Activation Manipulation with PseudoInverse (STAMP), a training-free, single-sample unlearning method that redirects MLP activations toward a refusal subspace via closed-form pseudoinverse updates. Its low-rank variant reduces computational complexity from O(d^3) to O(r^3 + r^2 * d), enabling efficient on-device unlearning with up to ~3x speedup over training-based baselines. Extensive experiments across harmful knowledge suppression, misinformation correction, and personal data erasure demonstrate that RePAIR achieves near-zero forget scores (Acc_f = 0.00, F-RL = 0.00) while preserving model utility (Acc_r up to 84.47, R-RL up to 0.88), outperforming six state-of-the-art baselines. These results establish RePAIR as an effective and practical framework for user-driven model editing, advancing transparent and on-device control over learned knowledge, with potential extensions to multimodal foundation models. 

---
# LASA: Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety 

**Authors**: Junxiao Yang, Haoran Liu, Jinzhe Tu, Jiale Cheng, Zhexin Zhang, Shiyao Cui, Jiaqi Weng, Jialing Tao, Hui Xue, Hongning Wang, Han Qiu, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12710)  

**Abstract**: Large language models (LLMs) often demonstrate strong safety performance in high-resource languages, yet exhibit severe vulnerabilities when queried in low-resource languages. We attribute this gap to a mismatch between language-agnostic semantic understanding ability and language-dominant safety alignment biased toward high-resource languages. Consistent with this hypothesis, we empirically identify the semantic bottleneck in LLMs, an intermediate layer in which the geometry of model representations is governed primarily by shared semantic content rather than language identity. Building on this observation, we propose Language-Agnostic Semantic Alignment (LASA), which anchors safety alignment directly in semantic bottlenecks. Experiments show that LASA substantially improves safety across all languages: average attack success rate (ASR) drops from 24.7% to 2.8% on LLaMA-3.1-8B-Instruct and remains around 3-4% across Qwen2.5 and Qwen3 Instruct models (7B-32B). Together, our analysis and method offer a representation-level perspective on LLM safety, suggesting that safety alignment requires anchoring safety understanding not in surface text, but in the model's language-agnostic semantic space. 

---
# From Imitation to Discrimination: Progressive Curriculum Learning for Robust Web Navigation 

**Authors**: Chuang Peng, Wei Zhang, Renshuai Tao, Xinhao Zhang, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12666)  

**Abstract**: Text-based web agents offer computational efficiency for autonomous web navigation, yet developing robust agents remains challenging due to the noisy and heterogeneous nature of real-world HTML. Standard Supervised Fine-Tuning (SFT) approaches fail in two critical dimensions: they lack discrimination capabilities to reject plausible but incorrect elements in densely populated pages, and exhibit limited generalization to unseen website layouts. To address these challenges, we introduce the Triton dataset (590k instances) and a progressive training curriculum. Triton is constructed via Structural-Semantic Hard Negative Mining, which explicitly mines topologically similar distractors, and a Dual-Agent Consensus pipeline that synthesizes diverse cross-domain tasks with strict verification. Building upon this foundation, our progressive curriculum produces three models: Triton-SFT-32B for basic imitation, Triton-ORPO-32B for robust discrimination via Odds Ratio Preference Optimization, and Triton-GRPO-32B for long-horizon consistency through Group Relative Policy Optimization. Empirical evaluation on Mind2Web demonstrates that Triton-GRPO-32B achieves state-of-the-art performance among open-source models with 58.7% Step Success Rate, surpassing GPT-4.5 (42.4%) and Claude-4.5 (41.4%) by over 16%, validating that specialized data curriculum outweighs raw parameter scale for web navigation. 

---
# Do VLMs Truly "Read" Candlesticks? A Multi-Scale Benchmark for Visual Stock Price Forecasting 

**Authors**: Kaiqi Hu, Linda Xiao, Shiyue Xu, Ziyi Tang, Mingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12659)  

**Abstract**: Vision-language models(VLMs) are increasingly applied to visual stock price forecasting, yet existing benchmarks inadequately evaluate their understanding of stock price in candlestick charts. First, prior studies fail to isolate VLMs' comprehension of visual inputs genuinely improves predictive performance and whether VLMs truly comprehend candlestick patterns. Further, most existing datasets and evaluation setups are designed around single-period or tabular inputs. However, human analysts strongly rely on multi-scale candlestick charts, where longer-term horizons capture trend direction and shorter-term horizons provide cues for inflection points, making it difficult to systematically assess VLMs' ability to integrate short-term and long-term visual market dynamics. To bridge this gap, we construct a multi-scale candlestick charts dataset and a standardized evaluation framework to assess VLMs' ability to utilize multi-scale visual market signals. Evaluation combines confusion-matrix-based diagnostics with information coefficient(IC) time series metrics and includes XGBoost as a feature-based temporal baseline. Using this dataset, we benchmark representative VLMs and analyze their ability to leverage multi-scale stock price data. Experimental results show that most VLMs perform well only under persistent uptrend or downtrend conditions, while exhibiting weak predictive capability in more common market scenarios. We also identify significant prediction biases and limited sensitivity to explicitly specified forecast horizons in prompts, indicating inherent limitations in precise temporal reasoning. 

---
# Adaptive Test-Time Scaling for Zero-Shot Respiratory Audio Classification 

**Authors**: Tsai-Ning Wang, Herman Teun den Dekker, Lin-Lin Chen, Neil Zeghidour, Aaqib Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2604.12647)  

**Abstract**: Automated respiratory audio analysis promises scalable, non-invasive disease screening, yet progress is limited by scarce labeled data and costly expert annotation. Zero-shot inference eliminates task-specific supervision, but existing methods apply uniform computation to every input regardless of difficulty. We introduce TRIAGE, a tiered zero-shot framework that adaptively scales test-time compute by routing each audio sample through progressively richer reasoning stages: fast label-cosine scoring in a joint audio-text embedding space (Tier-L), structured matching with clinician-style descriptors (Tier-M), and retrieval-augmented large language model reasoning (Tier-H). A confidence-based router finalizes easy predictions early while allocating additional computation to ambiguous inputs, enabling nearly half of all samples to exit at the cheapest tier. Across nine respiratory classification tasks without task-specific training, TRIAGE achieves a mean AUROC of 0.744, outperforming prior zero-shot methods and matching or exceeding supervised baselines on multiple tasks. Our analysis show that test-time scaling concentrates gains where they matter: uncertain cases see up to 19% relative improvement while confident predictions remain unchanged at minimal cost. 

---
# RPRA: Predicting an LLM-Judge for Efficient but Performant Inference 

**Authors**: Dylan R. Ashley, Gaël Le Lan, Changsheng Zhao, Naina Dhingra, Zhipeng Cai, Ernie Chang, Mingchen Zhuge, Yangyang Shi, Vikas Chandra, Jürgen Schmidhuber  

**Link**: [PDF](https://arxiv.org/pdf/2604.12634)  

**Abstract**: Large language models (LLMs) face a fundamental trade-off between computational efficiency (e.g., number of parameters) and output quality, especially when deployed on computationally limited devices such as phones or laptops. One way to address this challenge is by following the example of humans and have models ask for help when they believe they are incapable of solving a problem on their own; we can overcome this trade-off by allowing smaller models to respond to queries when they believe they can provide good responses, and deferring to larger models when they do not believe they can. To this end, in this paper, we investigate the viability of Predict-Answer/Act (PA) and Reason-Predict-Reason-Answer/Act (RPRA) paradigms where models predict -- prior to responding -- how an LLM judge would score their output. We evaluate three approaches: zero-shot prediction, prediction using an in-context report card, and supervised fine-tuning. Our results show that larger models (particularly reasoning models) perform well when predicting generic LLM judges zero-shot, while smaller models can reliably predict such judges well after being fine-tuned or provided with an in-context report card. Altogether, both approaches can substantially improve the prediction accuracy of smaller models, with report cards and fine-tuning achieving mean improvements of up to 55% and 52% across datasets, respectively. These findings suggest that models can learn to predict their own performance limitations, paving the way for more efficient and self-aware AI systems. 

---
# GeoAlign: Geometric Feature Realignment for MLLM Spatial Reasoning 

**Authors**: Zhaochen Liu, Limeng Qiao, Guanglu Wan, Tingting Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12630)  

**Abstract**: Multimodal large language models (MLLMs) have exhibited remarkable performance in various visual tasks, yet still struggle with spatial reasoning. Recent efforts mitigate this by injecting geometric features from 3D foundation models, but rely on static single-layer extractions. We identify that such an approach induces a task misalignment bias: the geometric features naturally evolve towards 3D pretraining objectives, which may contradict the heterogeneous spatial demands of MLLMs, rendering any single layer fundamentally insufficient. To resolve this, we propose GeoAlign, a novel framework that dynamically aggregates multi-layer geometric features to realign with the actual demands. GeoAlign constructs a hierarchical geometric feature bank and leverages the MLLM's original visual tokens as content-aware queries to perform layer-wise sparse routing, adaptively fetching the suitable geometric features for each patch. Extensive experiments on VSI-Bench, ScanQA, and SQA3D demonstrate that our compact 4B model effectively achieves state-of-the-art performance, even outperforming larger existing MLLMs. 

---
# Beyond Single-Dimension Novelty: How Combinations of Theory, Method, and Results-based Novelty Shape Scientific Impact 

**Authors**: Yi Zhao, Yang Chenggang, Yuzhuo Wang, Tong Bao, Zhang Heng, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.12471)  

**Abstract**: Scientific novelty drives advances at the research frontier, yet it is also associated with heightened uncertainty and potential resistance from incumbent paradigms, leading to complex patterns of scientific impact. Prior studies have primarily ex-amined the relationship between a single dimension of novelty -- such as theoreti-cal, methodological, or results-based novelty -- and scientific impact. However, because scientific novelty is inherently multidimensional, focusing on isolated dimensions may obscure how different types of novelty jointly shape impact. Consequently, we know little about how combinations of novelty types influence scientific impact. To this end, we draw on a dataset of 15,322 articles published in Nature Communications. Using the DeepSeek-V3 model, we classify articles into three novelty dimensions based on the content of their Introduction sections: theoretical novelty, methodological novelty, and results-based novelty. These dimensions may coexist within the same article, forming distinct novelty configura-tions. Scientific impact is measured using five-year citation counts and indicators of whether an article belongs to the top 1% or top 10% highly cited papers. Descriptive results indicate that results-based novelty alone and the simultaneous presence of all three novelty types are the dominant configurations in the sample. Regression results further show that articles with results-based novelty only re-ceive significantly more citations and are more likely to rank among the top 1% and top 10% highly cited papers than articles exhibiting all three novelty types. These findings advance our understanding of how multidimensional novelty configurations shape knowledge diffusion. 

---
# Do Transformers Use their Depth Adaptively? Evidence from a Relational Reasoning Task 

**Authors**: Alicia Curth, Rachel Lawrence, Sushrut Karmalkar, Niranjani Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2604.12426)  

**Abstract**: We investigate whether transformers use their depth adaptively across tasks of increasing difficulty. Using a controlled multi-hop relational reasoning task based on family stories, where difficulty is determined by the number of relationship hops that must be composed, we monitor (i) how predictions evolve across layers via early readouts (the logit lens) and (ii) how task-relevant information is integrated across tokens via causal patching. For pretrained models, we find some limited evidence for adaptive depth use: some larger models need fewer layers to arrive at plausible answers for easier tasks, and models generally use more layers to integrate information across tokens as chain length increases. For models finetuned on the task, we find clearer and more consistent evidence of adaptive depth use, with the effect being stronger for less constrained finetuning regimes that do not preserve general language modeling abilities. 

---
# Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning 

**Authors**: NVIDIA, Aakshita Chandiramani, Aaron Blakeman, Abdullahi Olaoye, Abhibha Gupta, Abhilash Somasamudramath, Abhinav Khattar, Adeola Adesoba, Adi Renduchintala, Adil Asif, Aditya Agrawal, Aditya Vavre, Ahmad Kiswani, Aishwarya Padmakumar, Ajay Hotchandani, Akanksha Shukla, Akhiad Bercovich, Aleksander Ficek, Aleksandr Shaposhnikov, Alex Gronskiy, Alex Kondratenko, Alex Neefus, Alex Steiner, Alex Yang, Alexander Bukharin, Alexander Young, Ali Hatamizadeh, Ali Taghibakhshi, Alina Galiautdinova, Alisa Liu, Alok Kumar, Ameya Sunil Mahabaleshwarkar, Amir Klein, Amit Zuker, Amnon Geifman, Anahita Bhiwandiwalla, Ananth Subramaniam, Andrew Tao, Anjaney Shrivastava, Anjulie Agrusa, Ankur Srivastava, Ankur Verma, Ann Guan, Anna Shors, Annamalai Chockalingam, Anubhav Mandarwal, Aparnaa Ramani, Arham Mehta, Arti Jain, Arun Venkatesan, Asha Anoosheh, Ashwath Aithal, Ashwin Poojary, Asif Ahamed, Asit Mishra, Asli Sabanci Demiroz, Asma Kuriparambil Thekkumpate, Atefeh Sohrabizadeh, Avinash Kaur, Ayush Dattagupta, Barath Subramaniam Anandan, Bardiya Sadeghi, Barnaby Simkin, Ben Lanir, Benedikt Schifferer, Benjamin Chislett, Besmira Nushi, Bilal Kartal, Bill Thiede, Bita Darvish Rouhani, Bobby Chen, Boris Ginsburg, Brandon Norick, Branislav Kisacanin, Brian Yu, Bryan Catanzaro, Buvaneswari Mani, Carlo del Mundo, Chankyu Lee, Chanran Kim, Chantal Hwang, Chao Ni, Charles Wang, Charlie Truong, Cheng-Ping Hsieh, Chenhan Yu, Chenjie Luo, Cherie Wang, Chetan Mungekar, Chintan Patel, Chris Alexiuk, Chris Holguin, Chris Wing, Christian Munley, Christopher Parisien, Chuck Desai, Chunyang Sheng, Collin Neale, Cyril Meurillon, Dakshi Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.12374)  

**Abstract**: We describe the pre-training, post-training, and quantization of Nemotron 3 Super, a 120 billion (active 12 billion) parameter hybrid Mamba-Attention Mixture-of-Experts model. Nemotron 3 Super is the first model in the Nemotron 3 family to 1) be pre-trained in NVFP4, 2) leverage LatentMoE, a new Mixture-of-Experts architecture that optimizes for both accuracy per FLOP and accuracy per parameter, and 3) include MTP layers for inference acceleration through native speculative decoding. We pre-trained Nemotron 3 Super on 25 trillion tokens followed by post-training using supervised fine tuning (SFT) and reinforcement learning (RL). The final model supports up to 1M context length and achieves comparable accuracy on common benchmarks, while also achieving up to 2.2x and 7.5x higher inference throughput compared to GPT-OSS-120B and Qwen3.5-122B, respectively. Nemotron 3 Super datasets, along with the base, post-trained, and quantized checkpoints, are open-sourced on HuggingFace. 

---
# Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors 

**Authors**: Rui Yin, Tianxu Han, Naen Xu, Changjiang Li, Ping He, Chunyi Zhou, Jun Wang, Zhihui Fu, Tianyu Du, Jinbao Li, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2604.12359)  

**Abstract**: Safety-aligned large language models (LLMs) are increasingly deployed in real-world pipelines, yet this deployment also enlarges the supply-chain attack surface: adversaries can distribute backdoored checkpoints that behave normally under standard evaluation but jailbreak when a hidden trigger is present. Recent post-hoc weight-editing methods offer an efficient approach to injecting such backdoors by directly modifying model weights to map a trigger to an attacker-specified response. However, existing methods typically optimize a token-level mapping that forces an affirmative prefix (e.g., ``Sure''), which does not guarantee sustained harmful output -- the model may begin with apparent agreement yet revert to safety-aligned refusal within a few decoding steps. We address this reliability gap by shifting the backdoor objective from surface tokens to internal representations. We extract a steering vector that captures the difference between compliant and refusal behaviors, and compile it into a persistent weight modification that activates only when the trigger is present. To preserve stealthiness and benign utility, we impose a null-space constraint so that the injected edit remains dormant on clean inputs. The method is efficient, requiring only a small set of examples and admitting a closed-form solution. Across multiple safety-aligned LLMs and jailbreak benchmarks, our method achieves high triggered attack success while maintaining non-triggered safety and general utility. 

---
# MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enhanced RAG on Long Industrial Documents 

**Authors**: Joongmin Shin, Chanjun Park, Jeongbae Park, Jaehyung Seo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.12352)  

**Abstract**: RAG-based QA has emerged as a powerful method for processing long industrial documents. However, conventional text chunking approaches often neglect complex and long industrial document structures, causing information loss and reduced answer quality. To address this, we introduce MultiDocFusion, a multimodal chunking pipeline that integrates: (i) detection of document regions using vision-based document parsing, (ii) text extraction from these regions via OCR, (iii) reconstruction of document structure into a hierarchical tree using large language model (LLM)-based document section hierarchical parsing (DSHP-LLM), and (iv) construction of hierarchical chunks through DFS-based grouping. Extensive experiments across industrial benchmarks demonstrate that MultiDocFusion improves retrieval precision by 8-15% and ANLS QA scores by 2-3% compared to baselines, emphasizing the critical role of explicitly leveraging document hierarchy for multimodal document-based QA. These significant performance gains underscore the necessity of structure-aware chunking in enhancing the fidelity of RAG-based QA systems. 

---
# Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization 

**Authors**: Yizhe Chi, Deyao Hong, Dapeng Jiang, Tianwei Luo, Kaisen Yang, Boshi Zhang, Zhe Cao, Xiaoyan Fan, Bingxiang He, Han Hao, Weiyang Jin, Dianqiao Lei, Qingle Liu, Houde Qian, Bowen Wang, Situ Wang, Youjie Zheng, Yifan Zhou, Calvin Xiao, Eren Cai, Qinhuai Na  

**Link**: [PDF](https://arxiv.org/pdf/2604.12290)  

**Abstract**: Current LLM agent benchmarks, which predominantly focus on binary pass/fail tasks such as code generation or search-based question answering, often neglect the value of real-world engineering that is often captured through the iterative optimization of feasible designs. To this end, we introduce Frontier-Eng, a human-verified benchmark for generative optimization -- an iterative propose-execute-evaluate loop in which an agent generates candidate artifacts, receives executable verifier feedback, and revises them under a fixed interaction budget -- spanning $47$ tasks across five broad engineering categories. Unlike previous suites, Frontier-Eng tasks are grounded in industrial-grade simulators and verifiers that provide continuous reward signals and enforce hard feasibility constraints under constrained budgets. We evaluate eight frontier language models using representative search frameworks, finding that while Claude 4.6 Opus achieves the most robust performance, the benchmark remains challenging for all models. Our analysis suggests a dual power-law decay in improvement frequency ($\sim$ 1/iteration) and magnitude ($\sim$ 1/improvement count). We further show that although width improves parallelism and diversity, depth remains crucial for hard-won improvements under a fixed budget. Frontier-Eng establishes a new standard for assessing the capacity of AI agents to integrate domain knowledge with executable feedback to solve complex, open-ended engineering problems. 

---
# The Enforcement and Feasibility of Hate Speech Moderation on Twitter 

**Authors**: Manuel Tonneau, Dylan Thurgood, Diyi Liu, Niyati Malhotra, Victor Orozco-Olvera, Ralph Schroeder, Scott A. Hale, Manoel Horta Ribeiro, Paul Röttger, Samuel P. Fraiberger  

**Link**: [PDF](https://arxiv.org/pdf/2604.12289)  

**Abstract**: Online hate speech is associated with substantial social harms, yet it remains unclear how consistently platforms enforce hate speech policies or whether enforcement is feasible at scale. We address these questions through a global audit of hate speech moderation on Twitter (now X). Using a complete 24-hour snapshot of public tweets, we construct representative samples comprising 540,000 tweets annotated for hate speech by trained annotators across eight major languages. Five months after posting, 80% of hateful tweets remain online, including explicitly violent hate speech. Such tweets are no more likely to be removed than non-hateful tweets, with neither severity nor visibility increasing the likelihood of removal. We then examine whether these enforcement gaps reflect technical limits of large-scale moderation systems. While fully automated detection systems cannot reliably identify hate speech without generating large numbers of false positives, they effectively prioritize likely violations for human review. Simulations of a human-AI moderation pipeline indicate that substantially reducing user exposure to hate speech is economically feasible at a cost below existing regulatory penalties. These results suggest that the persistence of online hate cannot be explained by technical constraints alone but also reflects institutional choices in the allocation of moderation resources. 

---
# CodeSpecBench: Benchmarking LLMs for Executable Behavioral Specification Generation 

**Authors**: Zaoyu Chen, Jianbo Dai, Boyu Zhu, Jingdong Wang, Huiming Wang, Xin Xu, Haoyang Yuan, Zhijiang Guo, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12268)  

**Abstract**: Large language models (LLMs) can generate code from natural language, but the extent to which they capture intended program behavior remains unclear. Executable behavioral specifications, defined via preconditions and postconditions, provide a concrete means to assess such understanding. However, existing work on specification generation is constrained in evaluation methodology, task settings, and specification expressiveness. We introduce CodeSpecBench, a benchmark for executable behavioral specification generation under an execution-based evaluation protocol. CodeSpecBench supports both function-level and repository-level tasks and encodes specifications as executable Python functions. Constructed from diverse real-world codebases, it enables a realistic assessment of both correctness (accepting valid behaviors) and completeness (rejecting invalid behaviors). Evaluating 15 state-of-the-art LLMs on CodeSpecBench, we observe a sharp performance degradation on repository-level tasks, where the best model attains only a 20.2% pass rate. We further find that specification generation is substantially more challenging than code generation, indicating that strong coding performance does not necessarily reflect deep understanding of intended program semantics. Our data and code are available at this https URL. 

---
# How memory can affect collective and cooperative behaviors in an LLM-Based Social Particle Swarm 

**Authors**: Taisei Hishiki, Takaya Arita, Reiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2604.12250)  

**Abstract**: This study examines how model-specific characteristics of Large Language Model (LLM) agents, including internal alignment, shape the effect of memory on their collective and cooperative dynamics in a multi-agent system. To this end, we extend the Social Particle Swarm (SPS) model, in which agents move in a two-dimensional space and play the Prisoner's Dilemma with neighboring agents, by replacing its rule-based agents with LLM agents endowed with Big Five personality scores and varying memory lengths. Using Gemini-2.0-Flash, we find that memory length is a critical parameter governing collective behavior: even a minimal memory drastically suppressed cooperation, transitioning the system from stable cooperative clusters through cyclical formation and collapse of clusters to a state of scattered defection as memory length increased. Big Five personality traits correlated with agent behaviors in partial agreement with findings from experiments with human participants, supporting the validity of the model. Comparative experiments using Gemma~3:4b revealed the opposite trend: longer memory promoted cooperation, accompanied by the formation of dense cooperative clusters. Sentiment analysis of agents' reasoning texts showed that Gemini interprets memory increasingly negatively as its length grows, while Gemma interprets it less negatively, and that this difference persists in the early phase of experiments before the macro-level dynamics converge. These results suggest that model-specific characteristics of LLMs, potentially including alignment, play a fundamental role in determining emergent social behavior in Generative Agent-Based Modeling, and provide a micro-level cognitive account of the contradictions found in prior work on memory and cooperation. 

---
# MolMem: Memory-Augmented Agentic Reinforcement Learning for Sample-Efficient Molecular Optimization 

**Authors**: Ziqing Wang, Yibo Wen, Abhishek Pandy, Han Liu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.12237)  

**Abstract**: In drug discovery, molecular optimization aims to iteratively refine a lead compound to improve molecular properties while preserving structural similarity to the original molecule. However, each oracle evaluation is expensive, making sample efficiency a key challenge for existing methods under a limited oracle budget. Trial-and-error approaches require many oracle calls, while methods that leverage external knowledge tend to reuse familiar templates and struggle on challenging objectives. A key missing piece is long-term memory that can ground decisions and provide reusable insights for future optimizations. To address this, we present MolMem (\textbf{Mol}ecular optimization with \textbf{Mem}ory), a multi-turn agentic reinforcement learning (RL) framework with a dual-memory system. Specifically, MolMem uses Static Exemplar Memory to retrieve relevant exemplars for cold-start grounding, and Evolving Skill Memory to distill successful trajectories into reusable strategies. Built on this memory-augmented formulation, we train the policy with dense step-wise rewards, turning costly rollouts into long-term knowledge that improves future optimization. Extensive experiments show that MolMem achieves 90\% success on single-property tasks (1.5$\times$ over the best baseline) and 52\% on multi-property tasks using only 500 oracle calls. Our code is available at this https URL. 

---
# HintMR: Eliciting Stronger Mathematical Reasoning in Small Language Models 

**Authors**: Jawad Hossain, Xiangyu Guo, Jiawei Zhou, Chong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12229)  

**Abstract**: Small language models (SLMs) often struggle with complex mathematical reasoning due to limited capacity to maintain long chains of intermediate steps and to recover from early errors. We address this challenge by introducing a hint-assisted reasoning framework that incrementally guides SLMs through multi-step mathematical problem solving. Our approach decomposes solutions into sequential reasoning steps and provides context-aware hints, where hints are generated by a separate SLM trained via distillation from a strong large language model. While the hint-generating SLM alone is not capable of solving the problems, its collaboration with a reasoning SLM enables effective guidance, forming a cooperative two-model system for reasoning. Each hint is generated conditionally on the problem statement and the accumulated reasoning history, providing stepwise, localized guidance without revealing full solutions. This reduces error propagation and allows the reasoning model to focus on manageable subproblems. Experiments across diverse mathematical benchmarks and models demonstrate that hint assistance consistently improves reasoning accuracy for SLMs, yielding substantial gains over standard prompting while preserving model efficiency. These results highlight that structured collaboration between SLMs-via hint generation and reasoning-offers an effective and lightweight mechanism for enhancing mathematical reasoning. 

---
# Designing Reliable LLM-Assisted Rubric Scoring for Constructed Responses: Evidence from Physics Exams 

**Authors**: Xiuxiu Tang, G. Alex Ambrose, Ying Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.12227)  

**Abstract**: Student responses in STEM assessments are often handwritten and combine symbolic expressions, calculations, and diagrams, creating substantial variation in format and interpretation. Despite their importance for evaluating students' reasoning, such responses are time-consuming to score and prone to rater inconsistency, particularly when partial credit is required. Recent advances in large language models (LLMs) have increased attention to AI-assisted scoring, yet evidence remains limited regarding how rubric design and LLM configurations influence reliability across performance levels. This study examined the reliability of AI-assisted scoring of undergraduate physics constructed responses using GPT-4o. Twenty authentic handwritten exam responses were scored across two rounds by four instructors and by the AI model using skill-based rubrics with differing levels of analytic granularity. Prompting format and temperature settings were systematically varied. Overall, human-AI agreement on total scores was comparable to human inter-rater reliability and was highest for high- and low-performing responses, but declined for mid-level responses involving partial or ambiguous reasoning. Criterion-level analyses showed stronger alignment for clearly defined conceptual skills than for extended procedural judgments. A more fine-grained, checklist-based rubric improved consistency relative to holistic scoring. These findings indicate that reliable AI-assisted scoring depends primarily on clear, well-structured rubrics, while prompting format plays a secondary role and temperature has relatively limited impact. More broadly, the study provides transferable design recommendations for implementing reliable LLM-assisted scoring in STEM contexts through skill-based rubrics and controlled LLM settings. 

---
# TimeMark: A Trustworthy Time Watermarking Framework for Exact Generation-Time Recovery from AIGC 

**Authors**: Shangkun Che, Silin Du, Ge Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.12216)  

**Abstract**: The widespread use of Large Language Models (LLMs) in text generation has raised increasing concerns about intellectual property disputes. Watermarking techniques, which embed meta information into AI-generated content (AIGC), have the potential to serve as judicial evidence. However, existing methods rely on statistical signals in token distributions, leading to inherently probabilistic detection and reduced reliability, especially in multi-bit encoding (e.g., timestamps). Moreover, such methods introduce detectable statistical patterns, making them vulnerable to forgery attacks and enabling model providers to fabricate arbitrary watermarks. To address these issues, we propose the concept of trustworthy watermark, which achieves reliable recovery with 100% identification accuracy while resisting both user-side statistical attacks and provider-side forgery. We focus on trustworthy time watermarking for use as judicial evidence. Our framework integrates cryptographic techniques and encodes time information into time-dependent secret keys under regulatory supervision, preventing arbitrary timestamp fabrication. The watermark payload is decoupled from time and generated as a random, non-stored bit sequence for each instance, eliminating statistical patterns. To ensure verifiability, we design a two-stage encoding mechanism, which, combined with error-correcting codes, enables reliable recovery of generation time with theoretically perfect accuracy. Both theoretical analysis and experiments demonstrate that our framework satisfies the reliability requirements for judicial evidence and offers a practical solution for future AIGC-related intellectual property disputes. 

---
# Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering 

**Authors**: Weikang Zhang, Zimo Zhu, Zhichuan Yang, Chen Huang, Wenqiang Lei, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.12210)  

**Abstract**: Simulating Standardized Patients with cognitive impairment offers a scalable and ethical solution for clinical training. However, existing methods rely on discrete prompt engineering and fail to capture the heterogeneity of deficits across varying domains and severity levels. To address this limitation, we propose StsPatient for the fine-grained simulation of cognitively impaired patients. We innovatively capture domain-specific features by extracting steering vectors from contrastive pairs of instructions and responses. Furthermore, we introduce a Stochastic Token Modulation (STM) mechanism to regulate the intervention probability. STM enables precise control over impairment severity while mitigating the instability of conventional vector methods. Comprehensive experiments demonstrate that StsPatient significantly outperforms baselines in both clinical authenticity and severity controllability. 

---
# Policy-Invisible Violations in LLM-Based Agents 

**Authors**: Jie Wu, Ming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.12177)  

**Abstract**: LLM-based agents can execute actions that are syntactically valid, user-sanctioned, and semantically appropriate, yet still violate organizational policy because the facts needed for correct policy judgment are hidden at decision time. We call this failure mode policy-invisible violations: cases in which compliance depends on entity attributes, contextual state, or session history absent from the agent's visible context. We present PhantomPolicy, a benchmark spanning eight violation categories with balanced violation and safe-control cases, in which all tool responses contain clean business data without policy metadata. We manually review all 600 model traces produced by five frontier models and evaluate them using human-reviewed trace labels. Manual review changes 32 labels (5.3%) relative to the original case-level annotations, confirming the need for trace-level human review. To demonstrate what world-state-grounded enforcement can achieve under favorable conditions, we introduce Sentinel, an enforcement framework based on counterfactual graph simulation. Sentinel treats every agent action as a proposed mutation to an organizational knowledge graph, performs speculative execution to materialize the post-action world state, and verifies graph-structural invariants to decide Allow/Block/Clarify. Against human-reviewed trace labels, Sentinel substantially outperforms a content-only DLP baseline (68.8% vs. 93.0% accuracy) while maintaining high precision, though it still leaves room for improvement on certain violation categories. These results demonstrate what becomes achievable once policy-relevant world state is made available to the enforcement layer. 

---
# From Plan to Action: How Well Do Agents Follow the Plan? 

**Authors**: Shuyang Liu, Saman Dehghan, Jatin Ganhotra, Martin Hirzel, Reyhaneh Jabbarvand  

**Link**: [PDF](https://arxiv.org/pdf/2604.12147)  

**Abstract**: Agents aspire to eliminate the need for task-specific prompt crafting through autonomous reason-act-observe loops. Still, they are commonly instructed to follow a task-specific plan for guidance, e.g., to resolve software issues following phases for navigation, reproduction, patch, and validation. Unfortunately, it is unknown to what extent agents actually follow such instructed plans. Without such an analysis, determining the extent agents comply with a given plan, it is impossible to assess whether a solution was reached through correct strategic reasoning or through other means, e.g., data contamination or overfitting to a benchmark. This paper presents the first extensive, systematic analysis of plan compliance in programming agents, examining 16,991 trajectories from SWE-agent across four LLMs on SWE-bench Verified and SWE-bench Pro under eight plan variations. Without an explicit plan, agents fall back on workflows internalized during training, which are often incomplete, overfit, or inconsistently applied. Providing the standard plan improves issue resolution, and we observe that periodic plan reminders can mitigate plan violations and improve task success. A subpar plan hurts performance even more than no plan at all. Surprisingly, augmenting a plan with additional task-relevant phases in the early stage can degrade performance, particularly when these phases do not align with the model's internal problem-solving strategy. These findings highlight a research gap: fine-tuning paradigms that teach models to follow instructed plans, rather than encoding task-specific plans in them. This requires teaching models to reason and act adaptively, rather than memorizing workflows. 

---
# Beyond Factual Grounding: The Case for Opinion-Aware Retrieval-Augmented Generation 

**Authors**: Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri  

**Link**: [PDF](https://arxiv.org/pdf/2604.12138)  

**Abstract**: RAG systems have transformed how LLMs access external knowledge, but we find that current implementations exhibit a bias toward factual, objective content, as evidenced by existing benchmarks and datasets that prioritize objective retrieval. This factual bias - treating opinions and diverse perspectives as noise rather than information to be synthesized - limits RAG systems in real-world scenarios involving subjective content, from social media discussions to product reviews. Beyond technical limitations, this bias poses risks to transparent and accountable AI: echo chamber effects that amplify dominant viewpoints, systematic underrepresentation of minority voices, and potential opinion manipulation through biased information synthesis. We formalize this limitation through the lens of uncertainty: factual queries involve epistemic uncertainty reducible through evidence, while opinion queries involve aleatoric uncertainty reflecting genuine heterogeneity in human perspectives. This distinction implies that factual RAG should minimize posterior entropy, whereas opinion-aware RAG must preserve it. Building on this theoretical foundation, we present an Opinion-Aware RAG architecture featuring LLM-based opinion extraction, entity-linked opinion graphs, and opinion-enriched document indexing. We evaluate our approach on e-commerce seller forum data, comparing an Opinion-Enriched knowledge base against a traditional baseline. Experiments demonstrate substantial improvements in retrieval diversity: +26.8% sentiment diversity, +42.7% entity match rate, and +31.6% author demographic coverage on entity-matched documents. Our results provide empirical evidence that treating subjectivity as a first-class citizen yields measurably more representative retrieval-a first step toward opinion-aware RAG. Future work includes joint optimization of retrieval and generation for distributional fidelity. 

---
# Long-Horizon Plan Execution in Large Tool Spaces through Entropy-Guided Branching 

**Authors**: Rongzhe Wei, Ge Shi, Min Cheng, Na Zhang, Pan Li, Sarthak Ghosh, Vaibhav Gorde, Leman Akoglu  

**Link**: [PDF](https://arxiv.org/pdf/2604.12126)  

**Abstract**: Large Language Models (LLMs) have significantly advanced tool-augmented agents, enabling autonomous reasoning via API interactions. However, executing multi-step tasks within massive tool libraries remains challenging due to two critical bottlenecks: (1) the absence of rigorous, plan-level evaluation frameworks and (2) the computational demand of exploring vast decision spaces stemming from large toolsets and long-horizon planning. To bridge these gaps, we first introduce SLATE (Synthetic Large-scale API Toolkit for E-commerce), a large-scale context-aware benchmark designed for the automated assessment of tool-integrated agents. Unlike static metrics, SLATE accommodates diverse yet functionally valid execution trajectories, revealing that current agents struggle with self-correction and search efficiency. Motivated by these findings, we next propose Entropy-Guided Branching (EGB), an uncertainty-aware search algorithm that dynamically expands decision branches where predictive entropy is high. EGB optimizes the exploration-exploitation trade-off, significantly enhancing both task success rates and computational efficiency. Extensive experiments on SLATE demonstrate that our dual contribution provides a robust foundation for developing reliable and scalable LLM agents in tool-rich environments. 

---
# The Effect of Document Selection on Query-focused Text Analysis 

**Authors**: Sandesh S Rangreji, Mian Zhong, Anjalie Field  

**Link**: [PDF](https://arxiv.org/pdf/2604.12099)  

**Abstract**: Analyses of document collections often require selecting what data to analyze, as not all documents are relevant to a particular research question and computational constraints preclude analyzing all documents, yet little work has examined effects of selection strategy choices. We systematically evaluate seven selection methods (from random selection to hybrid retrieval) on outputs from four text analyses methods (LDA, BERTopic, TopicGPT, HiCode) over two datasets with 26 open-ended queries. Our evaluation reveals practice guidance: semantic or hybrid retrieval offer strong go-to approaches that avoid the pitfalls of weaker selection strategies and the unnecessary compute overhead of more complicated ones. Overall, our evaluation framework establishes data selection as a methodological decision, rather than a practical necessity, inviting the development of new strategies. 

---
# UCS: Estimating Unseen Coverage for Improved In-Context Learning 

**Authors**: Jiayi Xin, Xiang Li, Evan Qiang, Weiqing He, Tianqi Shang, Weijie J. Su, Qi Long  

**Link**: [PDF](https://arxiv.org/pdf/2604.12015)  

**Abstract**: In-context learning (ICL) performance depends critically on which demonstrations are placed in the prompt, yet most existing selectors prioritize heuristic notions of relevance or diversity and provide limited insight into the coverage of a demonstration set. We propose Unseen Coverage Selection (UKS), a training-free, subset-level coverage prior motivated by the principle that a good demonstration set should expose the model to latent cluster unrevealed by the currently selected subset. UCS operationalizes this idea by (1) inducing discrete latent clusters from model-consistent embeddings and (2) estimating the number of unrevealed clusters within a candidate subset via a Smoothed Good--Turing estimator from its empirical frequency spectrum. Unlike previous selection methods, UCS is coverage-based and training-free, and can be seamlessly combined with both query-dependent and query-independent selection baselines via a simple regularized objective. Experiments on multiple intent-classification and reasoning benchmarks with frontier Large Language Models show that augmenting strong baselines with UCS consistently improves ICL accuracy by up to 2-6% under the same selection budget, while also yielding insights into task- and model-level latent cluster distributions. Code is available at this https URL. 

---
# INDOTABVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents 

**Authors**: Somraj Gautam, Anathapindika Dravichi, Gaurav Harit  

**Link**: [PDF](https://arxiv.org/pdf/2604.11970)  

**Abstract**: We introduce INDOTABVQA, a benchmark for evaluating cross-lingual Table Visual Question Answering (VQA) on real-world document images in Bahasa Indonesia. The dataset comprises 1,593 document images across three visual styles (bordered, borderless, and colorful) with one or more than one tables, and 1,593 question-answer sets in four languages: Bahasa Indonesia, English, Hindi, and Arabic. This enables evaluation of Vision-Language Models (VLMs) in both monolingual (Bahasa documents with Bahasa questions) and cross-lingual settings (Bahasa documents with questions in other languages). We benchmark leading open-source VLMs (Qwen2.5-VL, Gemma-3, LLaMA-3.2) and GPT-4o and reveal substantial performance gaps, particularly on structurally complex tables and in low-resource languages. Fine-tuning a compact 3B and LoRA-finetuned 7B model on our dataset yields 11.6% and 17.8% improvements in accuracy. Providing explicit table region coordinates as additional input further improves performance by 4-7%, demonstrating the value of Spatial priors for table-based reasoning. Our findings underscore the importance of language-diverse, domain-specific datasets and demonstrate that targeted fine-tuning can significantly enhance VLM performance on specialized document understanding tasks. INDOTABVQA provides a valuable resource for advancing research in cross-lingual, structure-aware document understanding, especially in underrepresented regions of the world. Full dataset can be accessed in huggingface at: this https URL} 

---
# AnyPoC: Universal Proof-of-Concept Test Generation for Scalable LLM-Based Bug Detection 

**Authors**: Zijie Zhao, Chenyuan Yang, Weidong Wang, Yihan Yang, Ziqi Zhang, Lingming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.11950)  

**Abstract**: While recent LLM-based agents can identify many candidate bugs in source code, their reports remain static hypotheses that require manual validation, limiting the practicality of automated bug detection. We frame this challenge as a test generation task: given a candidate report, synthesizing an executable proof-of-concept test, or simply a PoC - such as a script, command sequence, or crafted input - to trigger the suspected defect. Automated PoC generation can act as a scalable validation oracle, enabling end-to-end autonomous bug detection by providing concrete execution evidence. However, naive LLM agents are unreliable validators: they are biased toward "success" and may reward-hack by producing plausible but non-functional PoCs or even hallucinated traces. To address this, we present AnyPoC, a general multi-agent framework that (1) analyzes and fact-checks a candidate bug report, (2) iteratively synthesizes and executes a PoC while collecting execution traces, and (3) independently re-executes and scrutinizes the PoC to mitigate hallucination and reward hacking. In addition, AnyPoC also continuously extracts and evolves a PoC knowledge base to handle heterogeneous tasks. AnyPoC operates on candidate bug reports regardless of their source and can be paired with different bug reporters. To demonstrate practicality and generality, we apply AnyPoC, with a simple agentic bug reporter, on 12 critical software systems across diverse languages/domains (many with millions of lines of code) including Firefox, Chromium, LLVM, OpenSSL, SQLite, FFmpeg, and Redis. Compared to the state-of-the-art coding agents, e.g., Claude Code and Codex, AnyPoC produces 1.3x more valid PoCs for true-positive bug reports and rejects 9.8x more false-positive bug reports. To date, AnyPoC has discovered 122 new bugs (105 confirmed, 86 already fixed), with 45 generated PoCs adopted as official regression tests. 

---
# GoodPoint: Learning Constructive Scientific Paper Feedback from Author Responses 

**Authors**: Jimin Mun, Chani Jung, Xuhui Zhou, Hyunwoo Kim, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2604.11924)  

**Abstract**: While LLMs hold significant potential to transform scientific research, we advocate for their use to augment and empower researchers rather than to automate research without human oversight. To this end, we study constructive feedback generation, the task of producing targeted, actionable feedback that helps authors improve both their research and its presentation. In this work, we operationalize the effectiveness of feedback along two author-centric axes-validity and author action. We first curate GoodPoint-ICLR, a dataset of 19K ICLR papers with reviewer feedback annotated along both dimensions using author responses. Building on this, we introduce GoodPoint, a training recipe that leverages success signals from author responses through fine-tuning on valid and actionable feedback, together with preference optimization on both real and synthetic preference pairs. Our evaluation on a benchmark of 1.2K ICLR papers shows that a GoodPoint-trained Qwen3-8B improves the predicted success rate by 83.7% over the base model and sets a new state-of-the-art among LLMs of similar size in feedback matching on a golden human feedback set, even surpassing Gemini-3-flash in precision. We further validate these findings through an expert human study, demonstrating that GoodPoint consistently delivers higher practical value as perceived by authors. 

---
# M$^\star$: Every Task Deserves Its Own Memory Harness 

**Authors**: Wenbo Pan, Shujie Liu, Xiangyang Zhou, Shiwei Zhang, Wanlu Shi, Mirror Xu, Xiaohua Jia  

**Link**: [PDF](https://arxiv.org/pdf/2604.11811)  

**Abstract**: Large language model agents rely on specialized memory systems to accumulate and reuse knowledge during extended interactions. Recent architectures typically adopt a fixed memory design tailored to specific domains, such as semantic retrieval for conversations or skills reused for coding. However, a memory system optimized for one purpose frequently fails to transfer to others. To address this limitation, we introduce M$^\star$, a method that automatically discovers task-optimized memory harnesses through executable program evolution. Specifically, M$^\star$ models an agent memory system as a memory program written in Python. This program encapsulates the data Schema, the storage Logic, and the agent workflow Instructions. We optimize these components jointly using a reflective code evolution method; this approach employs a population-based search strategy and analyzes evaluation failures to iteratively refine the candidate programs. We evaluate M$^\star$ on four distinct benchmarks spanning conversation, embodied planning, and expert reasoning. Our results demonstrate that M$^\star$ improves performance over existing fixed-memory baselines robustly across all evaluated tasks. Furthermore, the evolved memory programs exhibit structurally distinct processing mechanisms for each domain. This finding indicates that specializing the memory mechanism for a given task explores a broad design space and provides a superior solution compared to general-purpose memory paradigms. 

---
