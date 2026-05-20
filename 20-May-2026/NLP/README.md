# TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload 

**Authors**: Zhiben Chen, Youpeng Zhao, Yang Sui, Jun Wang, Yuzhang Shang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20179)  

**Abstract**: Diffusion Large Language Models (dLLMs) have emerged as a competitive alternative to autoregressive (AR) models, offering better hardware utilization and bidirectional context through parallel block-level decoding. However, as dLLMs continue to scale up with mixture-of-experts (MoE) architectures, their deployment on resource-constrained devices remains an open challenge. Existing AR-based methods often incur either prohibitive I/O overhead or significant compute bottlenecks. In this work, we propose TIDE, a novel resource-efficient inference system that leverages the temporal stability of expert activations during the diffusion process within the block. Specifically, we leverage the temporal stability of expert activations during the diffusion process within the block and introduce an interval-based expert refresh strategy that updates the expert placement in an I/O-aware fashion. To ensure optimal performance, we formulate the inference scheduling as a mathematical programming problem, solving for the optimal interval that minimizes I/O traffic and CPU computation. Most importantly, TIDE is a lossless optimization that requires no model training, providing a "free lunch" acceleration for dLLM inference. In a single GPU-CPU system, we demonstrate that TIDE achieves up to 1.4$\times$ and 1.5$\times$ throughput improvements over prior baselines on LLaDA2.0-mini and LLaDA2.0-flash models, respectively. 

---
# From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models 

**Authors**: Juncheng Wu, Hardy Chen, Haoqin Tu, Xianfeng Tang, Freda Shi, Hui Liu, Hanqing Lu, Cihang Xie, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.20177)  

**Abstract**: Recent advances in vision-language models (VLMs) emphasize long chain-of-thought reasoning; yet, we find that their performance on visual tasks is primarily limited by a lack of visual perception as opposed to reasoning itself. In this work, we systematically study the interplay between perception and reasoning in VLM post-training by decomposing their capabilities into three separate training stages: visual perception, visual reasoning, and textual reasoning, incorporating specialized training data. We demonstrate that visual perception (a) requires targeted optimization with specialized data; (b) serves as a fundamental scaffold that should be solidified through staged training before refining visual reasoning; and (c) is more effectively learned via RL than caption-based SFT. Our experiments across multiple VLMs demonstrate that staged training consistently improves both visual perception and reasoning performance over merged training. Notably, models trained with our approach achieve 1.5% higher reasoning accuracy with 20.8% shorter reasoning traces, suggesting that superior perception reduces the need for excessive reasoning. Furthermore, we show that this capability-based staging represents a new curriculum dimension orthogonal to traditional difficulty-based curricula, and combining both yields further additive gains. Our staged-training models achieve superior performance among open-weight VLMs, establishing advanced results on several visual math and perception (e.g., +5.2% on WeMath and +3.7% on RealWorldQA) tasks compared with the base counterpart. 

---
# ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning 

**Authors**: Juncheng Wu, Letian Zhang, Yuhan Wang, Haoqin Tu, Hardy Chen, Zijun Wang, Cihang Xie, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.20176)  

**Abstract**: Large language models (LLMs) and agentic systems have shown promise for clinical decision support, but existing works largely assume that evidence has already been curated and handed to the model. Real-world clinical workflows instead require agents to actively seek, iteratively plan, and synthesize multimodal evidence from heterogeneous sources. In this paper, we introduce ClinSeekAgent, an automated agentic framework for dynamic multimodal evidence seeking that shifts the paradigm from passive evidence consumption to active evidence acquisition. Given only a clinical query and access to raw data sources, ClinSeekAgent gathers evidence by querying medical knowledge bases, navigating raw EHRs, and invoking medical imaging tools; refines its hypotheses as new information emerges; and integrates the collected evidence into grounded clinical decisions. ClinSeekAgent serves both as an inference-time agent for frontier LLMs and as a training-time pipeline for distilling high-quality agent trajectories into compact open-source models. To validate its inference-time effectiveness, we construct ClinSeek-Bench, which pairs Curated Input reasoning from fixed pre-selected evidence with Automated Evidence-Seeking over raw clinical data. On text-only EHR tasks, ClinSeekAgent improves Claude Opus 4.6 from 60.0 to 63.2 overall F1 and MiniMax M2.5 from 43.1 to 47.3, with positive risk-prediction gains in 7 out of 9 evaluated host models. On multimodal tasks, ClinSeekAgent improves Claude Opus 4.6 from 47.5 to 62.6 (+15.1); all evaluated models improve across the three CXR-related task groups. We further validate ClinSeekAgent as a training pipeline by distilling agentic evidence-seeking trajectories into ClinSeek-35B-A3B, which achieves 34.0 average F1 on existing AgentEHR-Bench, improving over its Qwen3.5-35B-A3B baseline by +11.9 points and approaching Claude Opus 4.6. 

---
# KoRe: Compact Knowledge Representations for Large Language Models 

**Authors**: Davide Cavicchini, Fausto Giunchiglia, Jacopo Staiano  

**Link**: [PDF](https://arxiv.org/pdf/2605.20170)  

**Abstract**: Modern Large Language Models (LLMs) have shown impressive performances in user-facing tasks such as question answering, as well as consistent improvements in reasoning capabilities. Still, the way these models encode knowledge seems inherently flawed: by design, LLMs encode world-knowledge within their parameters. This way of representing knowledge is inherently opaque, difficult to debug and update, and prone to hallucinations. On the other hand, Knowledge Graphs can provide human-readable and easily editable world knowledge representations, and their application in knowledge-intensive tasks has consistently proven beneficial to downstream performance. Nonetheless, current integration techniques require extensive retraining or finetuning. To overcome this issue, we introduce KoRe, a methodology to encode 1-hop sub-graphs into compact discrete knowledge tokens and inject them into a LLM backbone. We test the proposed approach on three established benchmarks, and report competitive performances coupled with a significant reduction (up to 10x) in token usage. Our results show that compact discrete KG representations can efficiently and effectively be used to ground modern LLMs. 

---
# Less Back-and-Forth: A Comparative Study of Structured Prompting 

**Authors**: Saurav Ghosh, Gabriella Polach, Abdou Sow  

**Link**: [PDF](https://arxiv.org/pdf/2605.20149)  

**Abstract**: Large language models (LLMs) are widely used for open-ended tasks, but underspecified prompts can lead to low-quality answers and additional interaction. This paper studies whether structured prompt design improves response quality while reducing user effort. We compare three prompt conditions: a raw prompt, a checklist-improved prompt, and a clarifying-question prompt. We evaluate these conditions across four task types--summarization, planning, explanation, and coding--using three LLM systems: ChatGPT, Claude, and Grok. Each output is scored with a unified rubric covering task completion, correctness, compliance, and clarity. Checklist-improved prompts achieved the highest mean rubric score, 7.50 out of 8, compared with 5.67 for raw prompts and 6.67 for clarifying-question prompts. Checklist prompts also produced the best quality-effort tradeoff, using fewer average tokens than both raw and clarifying prompts. These results suggest that a simple prompt checklist can improve LLM responses while reducing unnecessary interaction. 

---
# MixRea: Benchmarking Explicit-Implicit Reasoning in Large Language Models 

**Authors**: Yuanqing Cai, Ziyi Huang, Minhao Liu, Lixin Duan, Wen Li, Yanru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20128)  

**Abstract**: Large language models (LLMs) are increasingly integrated into high-stakes decision-making. Inspired by the theory of \emph{inattentional blindness} in human cognition, we investigate whether LLMs, trained on human-preferred corpora that embed attentional biases, exhibit a similar limitation: \emph{failing to attend to subtle yet important contextual cues under explicit task instructions}. To evaluate this, we introduce the task of \textbf{explicit-implicit reasoning} and present \textbf{MixRea}, a benchmark of 2,246 multiple-choice questions across 9 reasoning types with varying distributions of explicit and implicit information. Evaluation of 21 advanced LLMs shows that even the best-performing reasoning model (Gemini 2.5 Pro) achieves only 42.8\% consistency, revealing widespread inattentional blindness. To mitigate this, we propose \textbf{Potential Relation Completion Prompting (PRCP)}, a prompting method that improves reasoning by recovering overlooked causal relations. Further analysis shows that this limitation persists across diverse multi-source reasoning tasks, highlighting the need for more cognitively aligned models. 

---
# ThoughtTrace: Understanding User Thoughts in Real-World LLM Interactions 

**Authors**: Chuanyang Jin, Binze Li, Haopeng Xie, Cathy Mengying Fang, Tianjian Li, Shayne Longpre, Hongxiang Gu, Maximillian Chen, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2605.20087)  

**Abstract**: Conversational AI has now reached billions of users, yet existing datasets capture only what people say, not what they think. We introduce ThoughtTrace, the first large-scale dataset that pairs real-world multi-turn human--AI conversations with users' self-reported thoughts: their reasons for sending prompts and reactions to assistant responses. ThoughtTrace comprises 1,058 users, 2,155 conversations, 17,058 turns, and 10,174 thought annotations collected across 20 language models. Our analysis shows that ThoughtTrace captures long-horizon, topically diverse interactions, and that thoughts are semantically distinct from messages, difficult for frontier LLMs to infer from context, diverse in content, and tied to conversation stages. We further demonstrate the utility of thoughts for downstream modeling. First, thoughts improve user-behavior prediction as inference-time context. Second, thought-guided rewrites provide fine-grained alignment signals for training personalized assistants. Together, ThoughtTrace establishes user thoughts as a new data modality for studying the cognitive dynamics behind human--AI interaction and provides a foundation for building assistants that better understand and adapt to users' latent goals, preferences, and needs. 

---
# BalanceRAG: Joint Risk Calibration for Cascaded Retrieval-Augmented Generation 

**Authors**: Zijun Jia, Yuanchang Ye, Sen Jia, Yiyao Qian, Haoning Wang, Baojie Chen, Diyin Tang, Jinsong Yu, Zhiyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20084)  

**Abstract**: Large language models (LLMs) can enhance factuality via retrieval-augmented generation (RAG), but applying RAG to every query is unnecessary when the model-only answer is reliable. This motivates cascaded RAG: each query is first handled by an LLM-only branch, escalated to a RAG fallback only if the primary branch is uncertain, and abstained from when neither branch is sufficiently trustworthy. However, calibrating such cascades stage by stage may be conservative, since the final utility depends on joint uncertainty thresholding of LLM-only and RAG. In this work, we develop BalanceRAG to certify threshold pairs at a target risk level. Given uncertainty scores from the two branches, BalanceRAG frames each threshold pair as an operating point on a two-dimensional lattice and identifies safe operating points using sequential graphical testing. This enables risk-adaptive threshold calibration, controlling the system-level error rate among accepted points, while retaining more examples. Furthermore, BalanceRAG extends to multi-risk calibration, allowing retrieval usage to be bounded together with the selection-conditioned risk. Experiments on three open-domain question answering (QA) benchmarks across multiple LLM backbones demonstrate that BalanceRAG meets prescribed risk levels, preserves higher coverage and more accepted correct examples, and reduces unnecessary retrieval calls compared with always-on RAG. 

---
# CopT: Contrastive On-Policy Thinking with Continuous Spaces for General and Agentic Reasoning 

**Authors**: Dachuan Shi, Hanlin Zhu, Xiangchi Yuan, Wanjia Zhao, Kejing Xia, Wen Xiao, Wenke Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.20075)  

**Abstract**: Chain-of-thought (CoT) is a standard approach for eliciting reasoning capabilities from large language models (LLMs). However, the common CoT paradigm treats thinking as a prerequisite for answering, which can delay access to plausible answers and incur unnecessary token costs even when the model is able to identify an answer before extended thinking, a behavior known as performative reasoning. In this paper, we introduce CopT, a reformulated reasoning pipeline that reverses the usual order of thinking and answering. Instead of thinking before answering, CopT first elicits a draft answer and then invokes subsequent on-policy thinking conditioned on its own draft answer for reflection and correction. To assess whether the draft answer should be trusted, CopT recasts continuous embeddings as inference-time contrastive verifiers. Specifically, it contrasts the model's support for the same generated tokens under discrete-token inputs and continuous-embedding inputs, yielding a sequence-level reverse KL estimator for answer reliability. Our analysis shows that under certain assumptions, the expected estimate equals the mutual information between the unresolved latent state and the emitted answer token, explaining why it captures answer-relevant uncertainty rather than arbitrary uncertainty in the latent state. When the answer is deemed insufficiently reliable, CopT performs further on-policy thinking, where a second KL estimator dynamically controls draft-answer visibility, preserving useful partial information while reducing the risk of being misled by unreliable content. Across mathematics, coding, and agentic reasoning tasks, CopT improves peak accuracy by up to 23% and reduces token usage by up to 57% at comparable or higher accuracy, without any additional training. The code is available at this https URL. 

---
# Text-to-SPARQL Generation with Reinforcement Learning: A GRPO-based Approach on DBLP 

**Authors**: Jann Pfeifer, Debayan Banerjee, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2605.20066)  

**Abstract**: Knowledge graph question answering seeks to translate natural language questions into executable queries over knowledge graphs, but existing approaches often rely on large models or full supervision in the form of gold query annotations. This study examines whether reinforcement learning with outcome-based rewards can train a small instruction-tuned language model to perform zero-shot Text-to-SPARQL generation in the scholarly domain. Group-Relative Policy Optimization (GRPO) is applied to the Qwen3-1.7B model on DBLP-QuAD, using prompts that combine natural language questions with symbolic hints about entities and relations. Training relies on execution feedback, structural constraints, and answer-level rewards, with an additional variant that incorporates gold-query-based shaping. The resulting models are compared to the unmodified zero-shot baseline and to a supervised DoRA-finetuned baseline across answer-level accuracy, execution accuracy, category-wise scores, and generalization to held-out templates. GRPO substantially improves over the zero-shot baseline and exhibits competitive generalization, while supervised DoRA finetuning achieves higher overall accuracy on the same model scale. Ablation analyses indicate that execution-based rewards account for most gains, with additional shaping yielding limited additional benefit, suggesting that outcome-based reinforcement learning is a viable training strategy when gold queries are unavailable for token-level supervision. 

---
# Rewarding Beliefs, Not Actions: Consistency-Guided Credit Assignment for Long-Horizon Agents 

**Authors**: Wenjie Tang, Minne Li, Sijie Huang, Liquan Xiao, Yuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.20061)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) is a promising paradigm for improving large language model (LLM) agents on long-horizon interactive tasks. However, in partially observable environments, incomplete observations cause agent beliefs to drift over time, while delayed rewards obscure the causal impact of intermediate decisions, exacerbating temporal credit assignment challenges. To address this, we propose ReBel (Reward Belief), a process-level reinforcement learning algorithm that explicitly models structured belief states to summarize interaction history and guide subsequent policy learning. ReBel introduces belief-consistency supervision, converting discrepancies between predicted beliefs and observed feedback into dense self-supervised signals without requiring external step-wise annotations or verifiers. It also employs belief-aware grouping to compare trajectories under similar belief states, yielding more robust and lower-variance advantage estimates. We evaluate ReBel on challenging long-horizon benchmarks, including ALFWorld and WebShop. ReBel improves task success by up to $20.4$ percentage points over the episode-level baseline GRPO and increases sample efficiency by $2.1\times$. These results suggest that belief-aware self-supervision is a promising direction for reliable long-horizon decision-making under partial observability. Code is available at: this https URL. 

---
# PromptRad: Knowledge-Enhanced Multi-Label Prompt-Tuning for Low-Resource Radiology Report Labeling 

**Authors**: Ying-Jia Lin, Tzu-Chin Lo, Ping-Chien Li, Chi-Tung Cheng, Chien-Hung Liao, Hung-Yu Kao  

**Link**: [PDF](https://arxiv.org/pdf/2605.20052)  

**Abstract**: Automatic report labeling facilitates the identification of clinical findings from unstructured text and enables large-scale annotation for medical imaging research. Existing rule-based labelers struggle with the diverse descriptions in clinical reports, while fine-tuning pre-trained language models (PLMs) requires large amounts of labeled data that are often unavailable in clinical settings. In this paper, we propose PromptRad, a knowledge-enhanced multi-label \textbf{prompt}-tuning approach for \textbf{rad}iology report labeling under low-resource settings. PromptRad reformulates multi-label classification as masked language modeling and incorporates synonyms from the UMLS Metathesaurus into a multi-word verbalizer to enrich category representations. By fine-tuning the PLM without additional classification layers, PromptRad requires substantially less labeled data than conventional fine-tuning. Experiments on liver CT reports show that PromptRad outperforms dictionary-based and fine-tuning baselines with only 32 labeled training examples, and achieves competitive performance with GPT-4 despite using a much smaller model. Further analysis demonstrates that PromptRad captures complex negation patterns more effectively than existing methods, making it a promising solution for report labeling in data-scarce clinical scenarios. Our code is available at this https URL. 

---
# Language Mutations Sustain the Persistences of Conspiracy Theories on Social Media 

**Authors**: Calvin Yixiang Cheng, Dorian Quelle, Scott A. Hale  

**Link**: [PDF](https://arxiv.org/pdf/2605.20050)  

**Abstract**: This study investigates how language mutations affect the persistent diffusion of conspiracy theories on social media. Drawing on a three-year dataset of conspiracy-related posts from X, and applying computational linguistic analysis alongside survival modelling, we find that conspiracy claims with greater semantic mutations have substantially longer lifespans. Mutations in psycholinguistic properties, including pronouns, social reference words, cognitive process terms, risk- and health- related vocabularies, are associated with extended lifespans. Mutations in actor, action and target (AAT) categories are associated with longer lifespans as well. Qualitative analysis identifies two predominant mutation patterns: simplification and assimilation, at both linguistic and AAT structural levels. Taken together, the results advance our understanding of how language mutations contribute to conspiracy persistence online and shed lights on longitudinal content moderation strategies. We argue that content moderation should consider the mutability of conspiracy claims and focus on the core claims that can address their potential variations. 

---
# Mind Your Moras: Orthography-Aware Error Analysis of Neural Japanese Morphological Generation 

**Authors**: Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20043)  

**Abstract**: We present an orthography-aware error analysis of Japanese past-tense morphological inflection, treating hiragana not merely as a transcriptional medium, but as a representational system encoding morphophonological distinctions that may influence model generalization. We evaluate two character-level sequence-to-sequence architectures on past-tense formation using datasets formatted according to the SIGMORPHON 2020 and 2023 shared task conventions. Despite high aggregate accuracy, models exhibit systematic, linguistically interpretable errors that cluster around specific orthographic properties of hiragana. We introduce a concise error taxonomy capturing seven primary failure modes and provide both quantitative and qualitative analyses. Gemination-related errors dominate residual failures, accounting for 75-80% of errors, particularly in verbs whose stems end in the vowel e and require gemination before the past-tense suffix. Error patterns remain highly consistent across architectures and random seeds, suggesting a robust interaction between orthographic representation, morphological structure, and data frequency effects in shaping model generalization. These results underscore the necessity of orthography-aware evaluation for understanding neural generalization in morphologically complex languages. 

---
# FlexDraft: Flexible Speculative Decoding via Attention Tuning and Bonus-Guided Calibration 

**Authors**: Yaojie Zhang, Jianuo Huang, Junlong Ke, Yuhang Han, Yongji Long, Tianchen Zhao, Biqing Qi, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20022)  

**Abstract**: Speculative decoding accelerates memory-bound LLM inference without quality degradation by using a fast drafter to propose multiple candidate tokens and the target model to verify them in parallel. However, conventional sequential speculative decoding suffers from mutual waiting between drafting and verification, and repeated exchange of intermediate states further increases memory access overhead. Parallel speculative decoding addresses this limitation by performing drafting and verification within a single target forward pass, allowing future drafts to be prepared while current candidates are being verified. Although effective at small batch sizes, existing parallel speculative decoding methods either require costly continual pretraining with quality degradation or suffer from low acceptance rates. More importantly, this paradigm inherently suffers from uncertainty in both the bonus token and the accepted length, leading to draft verification mismatch and causing throughput gains to collapse at large batch sizes. To address these limitations, we introduce FlexDraft, a lossless speculative decoding framework that flexibly adapts to varying batch sizes through three key designs. (1) Attention Tuning enables block diffusion drafting by tuning only the attention projectors of the final few layers on mask tokens, while keeping the autoregressive path frozen to preserve the target distribution and produce high quality drafts with minimal trainable parameters. (2) Bonus-guided Calibration uses a lightweight MLP conditioned on the resolved bonus token to calibrate draft logits, mitigating draft verification mismatch caused by bonus token uncertainty. (3) Flex Decoding dynamically switches between parallel draft and verify at small batch sizes and sequential draft then verify at large batch sizes, and adjusts verification length based on draft confidence to eliminate redundant computation. 

---
# Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory 

**Authors**: Jingwei Sun, Jianing Zhu, Jiangchao Yao, Tongliang Liu, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.19952)  

**Abstract**: To enable reliable long-term interaction, LLM agents require a memory system that can faithfully store, efficiently retrieve, and deeply reason over accumulated dialogue history. Most existing methods adopt an extracted fact based paradigm: handcrafted static prompts compress raw dialogues into atomic facts, which are then stored, matched, and injected into downstream reasoning. Nevertheless, such fact-centric designs inevitably discard fine-grained details in original dialogues and fail to support deep reasoning over scattered isolated facts. Moreover, static prompts cannot maintain consistent extraction granularity across diverse dialogue styles. To address these limitations, we propose TriMem, which maintains three coexisting representation granularities, including raw dialogue segments anchored by source identifiers for storage fidelity, extracted atomic facts for efficient memory retrieval, synthesized profiles that aggregate dispersed facts into holistic semantic understanding for deep reasoning. We further adopt TextGrad-based prompt optimization, which iteratively refines extraction and profiling prompts via response quality feedback, achieving lifelong evolution without any parameter updating. Extensive experiments on LoCoMo and PerLTQA across multiple LLM backbones demonstrate that TriMem consistently outperforms strong memory baselines. The code is available at this https URL . 

---
# What Are LLMs Doing to Scientific Communication? Measuring Changes in Writing Practices and Reading Experience 

**Authors**: Filip Miletić, Neele Falk  

**Link**: [PDF](https://arxiv.org/pdf/2605.19936)  

**Abstract**: Has the style of scientific communication changed due to the growing use of large language models in the writing process? We address this question in the domain of Natural Language Processing by leveraging two data resources we create: a naturalistic corpus of over 37,000 papers from the ACL Anthology (2020-2024); and a synthetic dataset of 3,000 human-written passages and their LLM-generated improvements. We first implement a series of diachronic lexical analyses, showing that both word frequency and usage contexts have changed significantly over time, indicating semantic specialization in some cases and generalization in others. Broadening our perspective, we then model a range of more complex stylistic features and find that LLM-modified texts more frequently contain certain syntactic constructions, more complex and longer words and a lower lexical diversity. Finally, we connect these changes in writing practices to subjective reading experience through a pilot annotation study with 20 domain experts. They overall rate LLM-improved texts as more understandable and exciting, but also express negative qualitative attitudes towards LLMs, highlighting the strongly subjective effect of AI-assisted writing on reading experience. 

---
# Where Does Authorship Signal Emerge in Encoder-Based Language Models? 

**Authors**: Francis Kulumba, Guillaume Vimont, Laurent Romary, Florian Cafiero  

**Link**: [PDF](https://arxiv.org/pdf/2605.19908)  

**Abstract**: Authorship attribution models fine-tuned with the same pretrained encoder, data, and loss can differ four-fold in performance depending only on their scoring mechanism. We use mechanistic interpretability tools to explain this gap. Stylistic features such as word length, punctuation density, and function-word frequency are equally available at every layer in every model, including in an off-the-shelf control encoder, hence the gap not coming from representation quality. Instead, causal intervention shows that the scorer determines where the encoder consolidates authorship signal. Mean pooling forces consolidation by early to mid layers, while late interaction defers it to later layers. We further derive this difference from the gradient structure of each scorer, and training dynamics reveal distinct learning trajectories that follow from that difference. 

---
# Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning 

**Authors**: Qinghe Ma, Zhen Zhao, Yiming Wu, Jian Zhang, Lei Bai, Yinghuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.19852)  

**Abstract**: Tool-augmented reasoning has emerged as a promising direction for enhancing the reasoning capabilities of multimodal large language models (MLLMs). However, existing studies mainly focus on enabling models to perform tool invocation, while neglecting the necessity of invoking tools. We argue that tool usage is not always beneficial, as redundant or inappropriate invocations largely increase reasoning overhead and even mislead model predictions. To address this issue, we introduce AutoTool, a model that adaptively decides whether to invoke tools according to the characteristics of each query. Within a reinforcement learning framework, we design an explicit dual-mode reasoning strategy with mode-specific reward functions to guide the model toward producing accurate responses. Moreover, to prevent premature bias toward a single reasoning mode, AutoTool jointly explores and balances tool-assisted and text-centric reasoning throughout training, and promotes free exploration in later stages. Extensive experiments demonstrate that AutoTool exhibits outstanding performance and high efficiency, yielding a 21.8\% accuracy gain on V* benchmark compared to the base model, and a 44.9\% improvement in efficiency over existing tool-augmented methods on POPE benchmark. Code is available at this https URL. 

---
# CLIF: Concept-Level Influence Functions for Transparent Bottleneck Models 

**Authors**: Yike Sun, Mingkun Xu, Mu You, Zhongzhi He, Henghua Shen, Zehan Tan, Derek F. Wong, Tao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.19848)  

**Abstract**: In recent years, the black-box nature of deep learning models has limited their application in high-stakes domains such as medical diagnosis and finance, where interpretability is essential. To address this, we propose a novel approach using influence functions to enhance interpretability in NLP models at both the sample and concept levels. Experiments on CEBaB and Yelp datasets show that influence functions effectively identify the most impactful training samples, both helpful and harmful, on model predictions. By adjusting the labels and weights of these samples, we demonstrate that model performance can be restored to baseline levels without retraining, confirming the value of influence functions for efficient data debugging. Furthermore, our concept-level analysis identifies key concepts within Concept Bottleneck Models (CBM) that significantly affect predictions. Modifying these concepts alters model behavior observably, providing clear insights into the decision process. 

---
# LP-Eval: Rubric and Dataset for Measuring the Quality of Legal Proposition Generation 

**Authors**: Shanshan Xu, Johan Lindholm, Amogh Raina, Henrik Palmer Olsen, Daniel Hershcovich  

**Link**: [PDF](https://arxiv.org/pdf/2605.19815)  

**Abstract**: Legal proposition generation is central to legal reasoning and doctrinal scholarship, yet remain under-examined in Legal NLP. This paper investigates the automatic generation and evaluation of legal propositions from decisions of the Court of Justice of the European Union using large language models (LLMs). We introduce LP-Eval, a three-step evaluation rubric co-designed with legal experts that decomposes legal proposition quality into formal validity and substantive dimensions. Using this rubric, we release a dataset of two experts' annotations for 100 LLM-generated legal propositions. Our results show that LLMs can generate predominantly well-formed and high-quality propositions, while expert evaluations reveal higher quality for propositions derived from well established cases than from recent ones. We further examine LLMs as evaluators and find that rubric-guided LLM judgments align more closely with expert assessments than direct overall scoring, but remain insensitive to finer-grained distinctions captured by human experts. 

---
# Chunking German Legal Code 

**Authors**: Max Prior, Natalia Milanova, Andreas Schultz  

**Link**: [PDF](https://arxiv.org/pdf/2605.19806)  

**Abstract**: This paper investigates chunking strategies for retrieval-augmented generation on German statutory law, using the German Civil Code as a structured benchmark corpus. We implement and compare a range of segmentation approaches, including structural units (sections, subsections, sentences, propositions), fixed-size windows, contextual chunking, semantic clustering, Lumber-style chunking, and RAPTOR-based hierarchical retrieval. All methods are evaluated on a legal question-answering dataset with section-level gold labels, measuring recall, query latency, index build time, and storage requirements. Results show that chunking strategies aligned with the inherent legal structure - particularly section and subsection - based retrieval-achieve the highest recall, while more complex approaches that override this structure perform worse. These simpler methods also offer favorable computational efficiency compared to LLM-intensive techniques such as contextual chunking, RAPTOR, and Lumber. The findings highlight a key trade-off between semantic enrichment and operational cost, and demonstrate that preserving domain-specific structure is critical for effective legal information retrieval. 

---
# Towards Trust Calibration in Socially Interactive Agents: Investigating Gendered Multimodal Behaviors Generation with LLMs 

**Authors**: Lucie Galland, Chloé Clavel, Magalie Ochs  

**Link**: [PDF](https://arxiv.org/pdf/2605.19798)  

**Abstract**: As Socially Interactive Agents (SIAs) become increasingly integrated into daily life, the ability to calibrate user trust to an agent's actual capabilities would help ensure appropriate usage of these agents. In this paper, we explore the capacity of Large Language Models (LLMs) to generate multimodal behaviors (verbal, vocal, gestural, and facial expression modalities) that reflect varying levels of ability and benevolence, two key dimensions of trustworthiness. We propose a novel method for automatically generating behaviors aligned with specific levels of these traits, a first step towards enabling nuanced and trust-calibrated interactions. By analyzing a large dataset of multimodal transcripts generated by LLMs, we demonstrate that GPT-5.4 is able to produce coherent behavior across different modalities (text, intonation, facial expression, and gesture). Using Random Forest feature importance analysis, we show that the generated behaviors align with theoretical expectations for ability and benevolence. However, we also find that when gender is specified in the prompt, LLMs tend to reproduce societal gender stereotypes, associating male agents' behaviors with high ability and female agents' behaviors with high benevolence. To validate our approach, we conducted a user study on Prolific using a within-subjects design. Participants perceived different levels of ability and benevolence in the generated behaviors align with the intended instructions. 

---
# Synthesis and Evaluation of Long-term History-aware Medical Dialogue 

**Authors**: Hebin Hu, Renke Dai, Ah-Hwee Tan, Yilin Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.19766)  

**Abstract**: An effective healthcare agent must be able to recall and reason over a patient's longitudinal medical history. However, the absence of datasets with realistic long-term dialogue timelines limits systematic evaluation. Real clinical text is constrained by privacy and ethics, while existing benchmarks focus on isolated interactions, failing to capture cross-session reasoning. We introduce a framework for synthesizing high-quality, long-term medical dialogues with LLMs. Our approach entails a knowledge-guided decomposition into three stages: constructing synthetic patient profiles with diverse disease and complication trajectories, generating multi-turn dialogues per encounter, and integrating them into a coherent longitudinal history dataset, MediLongChat. We establish three benchmark tasks-In-dialogue Reasoning, Cross-dialogue Reasoning, and Synthesis Reasoning-to evaluate the memory capabilities of healthcare agents. To assess data quality, we introduce a multi-dimensional evaluation framework combining vector-based metrics with LLM-as-a-judge assessments. Specifically, we define automatic measures-Faithfulness, Coherence, and Diversity-together with two LLM-based evaluations: Correctness and Realism. Benchmark experiments show that even state-of-the-art LLMs struggle with MediLongChat. These findings highlight the benchmark's applicability and underscore the need for tailored methods to advance healthcare agents. 

---
# TERGAD: Structure-Aware Text-Enhanced Representations for Graph Anomaly Detection 

**Authors**: Wen Shi, Zhe Wang, Huafei Huang, Qing Qing, Ziqi Xu, Qixin Zhang, Xikun Zhang, Renqiang Luo, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.19738)  

**Abstract**: Graph Anomaly Detection (GAD) aims to identify atypical graph entities, such as nodes, edges, or substructures, that deviate significantly from the majority. While existing text-rich approaches typically integrate structural context into the data representation pipeline using raw textual features, they often neglect the structural context of nodes. This limitation hinders their ability to detect sophisticated anomalies arising from inconsistencies between a node's inherent content and its topological role. To bridge this gap, we propose TERGAD (Structure-aware Text-enhanced Representations for Graph Anomaly Detection), A novel data augmentation framework that enriches structural semantics for GAD via the semantic reasoning capabilities of Large Language Models (LLMs). Specifically, TERGAD translates node-level topological properties into descriptive natural language narratives, which are subsequently processed by an LLM to derive high-level semantic embeddings. These embeddings are then adaptively fused with original node attributes through a gated dual-branch autoencoder to jointly reconstruct both graph structure and node features. The anomaly score is computed based on the integrated reconstruction error, effectively capturing deviations in both observable attributes and LLM-informed semantic expectations. Extensive experiments on six real-world datasets demonstrate that TERGAD consistently outperforms state-of-the-art baselines. Furthermore, our ablation studies validate the indispensable role of structural semantic guidance and the efficacy of the gated fusion mechanism. Code is available at this https URL. 

---
# ContextRAG: Extraction-Free Hierarchical Graph Construction for Retrieval-Augmented Generation 

**Authors**: Roman Prosvirnin, Sergei Kuznetsov, Seungmin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.19735)  

**Abstract**: Graph-structured retrieval-augmented generation (RAG) systems can improve answer quality on multi-hop questions, but many current systems rely on large language models (LLMs) to extract entities, relations, and summaries during indexing. These calls add token and wall-clock costs that grow with corpus size. We present ContextRAG, a graph RAG system whose graph topology is constructed without LLM-based entity or relation extraction. ContextRAG derives a fuzzy concept graph over chunk embeddings using residual-quantization k-means and Formal Concept Analysis with Lukasiewicz residuated logic. Bridge-like and meet-derived context nodes are induced by soft fuzzy join and meet operations, rather than by LLM-written graph edges. On a 130-task UltraDomain subset, ContextRAG builds its index with 30 LLM calls and 22,073 tokens. In contrast, a local HiRAG reproduction stress test required 870 indexing calls and 3.54M tokens on a 20-task subset before failing during graph construction; linear extrapolation to 130 tasks implies over 23M indexing tokens. ContextRAG obtains 33.6% F1 overall and 36.8% F1 on multi-hop tasks. An activation analysis shows that queries retrieving at least one lattice-derived node in the top five achieve +3.9 percentage points F1 over queries that do not; this association is diagnostic rather than causal. 

---
# Mathematical Reasoning in Large Language Models: Benchmarks, Architectures, Evaluation, and Open Challenges 

**Authors**: Husnain Amjad, Raja Khurram Shahzad, Aamir Shahzad, Mehwish Fatima  

**Link**: [PDF](https://arxiv.org/pdf/2605.19723)  

**Abstract**: Mathematical reasoning is essential for problem-solving in education, science, and industry, serving as a crucial benchmark for evaluating artificial intelligence systems. As Large Language Models (LLMs) improve their reasoning capabilities, understanding how well they perform mathematical reasoning has become increasingly important. This survey synthesizes recent advancements in mathematical reasoning with LLMs through a structured analysis of datasets, architectures, training strategies, and evaluation protocols. Our systematic review encompasses approximately 120 peer-reviewed studies and preprints, examining the evolution of this research area and providing a unified analytical framework to understand current progress and limitations. Our study particularly introduces a unified taxonomy of mathematical datasets, distinguishing between pretraining corpora, supervised fine-tuning resources, and evaluation benchmarks across varying levels of reasoning complexity. A systematic analysis of reasoning architectures and training strategies, including tool integration, verifier-guided reasoning, and parameter-efficient adaptation, is presented to assess their effects on reasoning robustness and generalization. Moreover, a comparative evaluation of existing metrics highlights the gap between final-answer accuracy and process-level reasoning verification. By synthesizing insights across these areas, our analysis identifies recurring failure modes, such as reasoning faithfulness issues, benchmark biases, and generalization limitations, and outlines key research directions toward improving symbolic grounding, evaluation reliability, and the development of more robust and trustworthy LLM-based reasoning systems. 

---
# CAIT: A Syntactic Parsing Toolkit for Child-Adult InTeractions 

**Authors**: Francesca Padovani, Xiulin Yang, Bastian Bunzeck, Jaap Jumelet, Yevgen Matusevych, Nathan Schneider, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2605.19718)  

**Abstract**: CHILDES is a paramount resource for language acquisition studies -- yet computational tools for analyzing its syntactic structure remain limited. Leveraging the recent release of the UD-English-CHILDES treebank with gold-standard Universal Dependencies (UD) annotations, we train a state-of-the-art dependency parser specifically tailored to CHILDES. The parser more accurately captures syntactic patterns in child--adult interactions, outperforming widely used off-the-shelf English parsers, including SpaCy and Stanza. Alongside the parser, we also release a Part-of-Speech tagger and an utterance-level construction tagger, which together form the open-source Syntactic Parsing Toolkit for Child--Adult InTeractions (CAIT). Through a detailed error analysis and a case study tracking the distribution of syntactic constructions across developmental time in CHILDES, we demonstrate the practical utility of the toolkit for large-scale, reproducible research on language acquisition. 

---
# LLM-Based Financial Sentiment Analysis in Arabic: Evidence from Saudi Markets 

**Authors**: Mona H. Albaqawi, Eman M. Albalkhi, Joud A. Albaiti, Enrico Lopedoto  

**Link**: [PDF](https://arxiv.org/pdf/2605.19714)  

**Abstract**: Investor sentiment shapes financial markets, yet modeling sentiment in Arabic financial contexts remains challenging due to linguistic complexity and limited resources. We present an Arabic NLP framework for large-scale financial sentiment analysis tailored to the Saudi market, integrating official financial news and social media to capture institutional and public investor sentiment. The framework constructs a large Arabic financial corpus through a multi-stage pipeline encompassing data collection, cleaning, deduplication, entity linking, and sentiment annotation. Transformer-based NER combined with a curated company lexicon links textual mentions to canonical company identifiers, with sentiment labels assigned using a five-class scheme. The resulting dataset of 84K samples supports company-level sentiment aggregation and analysis of sentiment dynamics relative to stock market behavior on the Saudi Exchange. Experimental results demonstrate reliable and scalable Arabic financial sentiment analysis. 

---
# Can Large Language Models Reliably Correct Errors in Low-Resource ASR? A Contamination-Aware Case Study on West Frisian 

**Authors**: Yun Hao, Reihaneh Amooie, Wietse de Vries, Rik van Noord, Martijn Wieling  

**Link**: [PDF](https://arxiv.org/pdf/2605.19711)  

**Abstract**: Automatic speech recognition (ASR) has improved substantially in recent years, yet performance remains limited for low-resource languages. Large language models (LLMs) have shown promise for improving ASR through generative error correction (GER), but their effectiveness in low-resource settings remains underexplored. In addition, it remains unclear to what extent data contamination influences the reported improvements in LLM-based GER. This study investigates LLM-based GER for low-resource Frisian. In addition to a public corpus, we construct and use a Frisian offline dataset with non-public texts for evaluation to control for potential data contamination. Results show that GER improves ASR performance in most settings, with the best GPT-5.1 results surpassing oracle WERs. Comparable gains on the offline dataset indicate that improvements reflect true correction ability. We further provide a detailed error analysis revealing model correction patterns. 

---
# K-Quantization and its Impact on Output Performance 

**Authors**: Robin Baki Davidsson, Pierre Nugues  

**Link**: [PDF](https://arxiv.org/pdf/2605.19645)  

**Abstract**: Recent advancements in large language models (LLMs) have shown their remarkable capacities in many NLP tasks. However, their substantial size often presents challenges for deployment. This necessitates efficient techniques for model compression, with quantization emerging as a prominent solution. Despite its benefits, the exact impact of quantization (from 2- to 6-bit) on the performance and accuracy of LLMs remains an active area of research. This paper investigates the performance of eight LLMs at various quantization levels, focusing on tasks such as MMLU-Pro for knowledge processing and reasoning, CRUXEval for code comprehension, and MuSR for reading comprehension. Our results show a consistent trend where higher precision (e.g., 8-bit Q8\_0) yields improved performance, albeit with diminishing returns. Aggressive quantization (e.g., 2-bit Q2\_K) usually retains acceptable accuracy, though some models show a substantial loss in performance. Our findings indicate that while lower bit precision generally reduces performance, the impact varies across models and tasks. Larger models show greater resilience to aggressive quantization, but can still undergo significant drops at lower precision levels. Mid-sized models in the 7-9 billion parameter range strike an optimal balance between efficiency and resource usage. Such results provide insights into the trade-offs between model size, quantization, and performance. 

---
# optimize_anything: A Universal API for Optimizing any Text Parameter 

**Authors**: Lakshya A Agrawal, Donghyun Lee, Shangyin Tan, Wenjie Ma, Karim Elmaaroufi, Rohit Sandadi, Sanjit A. Seshia, Koushik Sen, Dan Klein, Ion Stoica, Joseph E. Gonzalez, Omar Khattab, Alexandros G. Dimakis, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2605.19633)  

**Abstract**: Can a single LLM-based optimization system match specialized tools across fundamentally different domains? We show that when optimization problems are formulated as improving a text artifact evaluated by a scoring function, a single AI-based optimization system-supporting single-task search, multi-task search with cross-problem transfer, and generalization to unseen inputs-achieves state-of-the-art results across six diverse tasks. Our system discovers agent architectures that nearly triple Gemini Flash's ARC-AGI accuracy (32.5% to 89.5%), finds scheduling algorithms that cut cloud costs by 40%, generates CUDA kernels where 87% match or beat PyTorch, and outperforms AlphaEvolve's reported circle packing solution (n=26). Ablations across three domains reveal that actionable side information yields faster convergence and substantially higher final scores than score-only feedback, and that multi-task search outperforms independent optimization given equivalent per-problem budget through cross-task transfer, with benefits scaling with the number of related tasks. Together, we show for the first time that text optimization with LLM-based search is a general-purpose problem-solving paradigm, unifying tasks traditionally requiring domain-specific algorithms under a single framework. We open-source optimize\_anything with support for multiple backends as part of the GEPA project at this https URL . 

---
# LLMEval-Logic: A Solver-Verified Chinese Benchmark for Logical Reasoning of LLMs with Adversarial Hardening 

**Authors**: Ming Zhang, Qiyuan Peng, Yinxi Wei, Yujiong Shen, Kexin Tan, Yuhui Wang, Zhenghao Xiang, Junjie Ye, Zhangyue Yin, Zhiheng Xi, Shihan Dou, Tao Gui, Maxm Pan, Ruizhi Yang, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.19597)  

**Abstract**: Evaluating large language models (LLMs) on natural-language logical reasoning is essential because rule-governed tasks require conclusions to follow strictly from stated premises. Many existing logical-reasoning benchmarks are generated by templating natural-language items from sampled formulas, provide only coarse or unaudited formal annotations, and are now quickly saturated by frontier reasoning models. We present LLMEval-Logic, a Chinese logical reasoning benchmark built from realistic situational scenarios. Its pipeline forward-authors and expert-audits natural-language items together with their reference formalizations, verifies annotated answers with Z3, constructs expert rubrics for natural-to-formal grading, and hardens selected items through a closed-loop adversarial workflow. The benchmark is released in two paired subsets: a 246-item Base subset shipped with 1,400 expert-developed rubric atoms, and a 190-item Hard subset with 938 multi-step sub-questions over closed model spaces. Evaluating 14 frontier LLMs on LLMEval-Logic reveals substantial gaps in current models: the best model reaches only 37.5% Hard Item Accuracy, and even with reference symbols the highest joint Z3+Rubric formalization score among evaluated models reaches only 60.16%. Our benchmark is publicly available at this https URL. 

---
# GoLongRL: Capability-Oriented Long Context Reinforcement Learning with Multitask Alignment 

**Authors**: Minxuan Lv, Tiehua Mei, Tanlong Du, Junmin Chen, Zhenpeng Su, Ziyang Chen, Ziqi Wang, Zhennan Wu, Ruotong Pan, jian Liang, Ruiming Tang, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.19577)  

**Abstract**: We present GoLongRL, a fully open-source, capability-oriented post-training recipe for long-context reinforcement learning with verifiable rewards (RLVR). Existing long-context RL methods often treat data construction as a matter of designing increasingly complex retrieval paths, leading to homogeneous task coverage and reward formulations that inadequately reflect practical long-context requirements. Our work offers two contributions. (1) Capability-oriented data construction with full open release. We openly release a dataset of 23K RLVR samples, the complete construction pipeline, and all training code. Guided by a taxonomy of long-context capabilities, the dataset spans 9 task types, each paired with its natural evaluation metric. It comprises curated open-source samples from established corpora and synthetic samples whose QA pairs are generated from real source documents such as books, academic papers, and multi-turn dialogues. Under the same vanilla GRPO setup, our dataset alone outperforms the closed-source QwenLong-L1.5 dataset. Moreover, our Qwen3-30B-A3B model trained on this data delivers long-context performance comparable to DeepSeek-R1-0528 and Qwen3-235B-A22B-Thinking-2507, suggesting that broader coverage and greater reward diversity substantially benefit long-context capability improvement. (2) TMN-Reweight for heterogeneous multitask optimization. To address optimization challenges from heterogeneous rewards, we propose TMN-Reweight, which combines task-level mean normalization for cross-task reward scale alignment with difficulty-adaptive weighting for more reliable advantage estimation. TMN-Reweight further improves average performance over vanilla GRPO, with general capabilities preserved or improved across reported evaluations. 

---
# A Data-Driven Approach to Idiomaticity Based on Experts' Criteria in Theoretical Linguistics 

**Authors**: Elena Mikhalkova, Anastasiya Vishnyakova, Anastasiya Drozdova, Polina Gavin, Aleksander Zhmykhov, Timofey Protasov  

**Link**: [PDF](https://arxiv.org/pdf/2605.19575)  

**Abstract**: The article observes data analysis of 286 multi-word expressions (MWEs) based on 16 lexical, grammatical and other criteria described in theoretical books and papers on the notion of idiomaticity. MWEs were collected from the same theoretical sources, and a set of experts in linguistics annotated them with these categories. The distribution of categories shows that there are no absolutely idiomatic expressions. Lexical criteria seem to be the most influential; grammatical criteria are bound to certain conditions; presence of obsolete words and grammar influence ability of an MWE to be replaced with one word. 

---
# m3BERT: A Modern, Multi-lingual, Matryoshka Bidirectional Encoder 

**Authors**: Yaoxiang Wang, Simiao Zuo, Qingguo Hu, Yucheng Ding, Yeyun Gong, Jian Jiao, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.19568)  

**Abstract**: Embedding models are pivotal in industrial information retrieval systems like search and advertising. However, existing pretrained models often exhibit fixed architectures and embedding dimensionalities, posing significant challenges when adapting them to diverse deployment scenarios with varying business-driven constraints. A common practice involves fine-tuning with partial parameter initialization from larger pretrained models for resource-constrained tasks. This method is often suboptimal as the misalignment between pretraining and downstream usage prevents full realization of pretraining benefits. To address this limitation, we introduce m3BERT: a Modern, Multi-lingual, Matryoshka Bidirectional Encoder, which features a novel pretraining strategy that jointly optimizes representations across both transformer layers and multiple embedding dimensions. This enables a single model to be tailored to varied resource and accuracy targets while maintaining consistency with pretraining. Incorporating recent architectural improvements, m3BERT uses a three-stage pretraining: monolingual pretraining, multilingual adaptation to serve diverse user bases, and crucial continual pretraining on a massive web domain corpus to enhance utility in commercial retrieval. m3BERT significantly outperforms state-of-the-art embedding models in Bing-Click, a large-scale industrial retrieval dataset, showcasing its practical versatility as an efficient foundation for resource-aware industrial retrieval systems. Further experiments on public datasets also confirm the general effectiveness of our multigranular Matryoshka pretraining strategy. 

---
# Investigating Cross-Modal Skill Injection: Scenarios, Methods, and Hyperparameters 

**Authors**: Zhiyu Xu, Lean Wang, Yuanxin Liu, Lei Li, Hao Zhou, Fandong Meng, Jie Zhou, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.19523)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable proficiency in general multi-modal understanding; yet they struggle to efficiently acquire continually evolving domain-specific skills. Conventional approaches to enhancing VLM capabilities, such as Supervised Fine-Tuning (SFT), require extensive dataset curation and substantial computational resources. Model merging has emerged as an efficient alternative that enables the transfer of domain-specific expertise from Large Language Models (LLMs) to VLMs without incurring additional training data requirements or significant computational overhead. Unlike conventional merging of homogeneous LLMs, which mainly aggregates existing capabilities, cross-modal skill injection aims to induce emergent cross-modal capabilities by integrating a domain-expert LLM into a VLM. However, existing research lacks a systematic analysis of the applicability and methodology of cross-modal skill injection. In this study, we investigate cross-modal skill injection across three main aspects: scenarios, methods, and hyperparameters. For scenarios, we find that cross-modal skill injection generally performs well in instruction-following and cross-lingual settings, yet struggles with mathematical reasoning. For methods, we find that classic approaches such as TA and DARE consistently achieve superior performance over alternative merging methods. We also provide a systematic and quantitative analysis of the hyperparameter tuning that these classic methods critically depend on. 

---
# Base Models Look Human To AI Detectors 

**Authors**: Yixuan Even Xu, Ziqian Zhong, Aditi Raghunathan, Fei Fang, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2605.19516)  

**Abstract**: As AI-generated text enters the real-world at scale, institutions increasingly use commercial AI-text detectors, especially in education and academic-integrity workflows. We report a surprising empirical finding about such systems: when evaluated by GPTZero and Pangram, generated text from base models is often judged overwhelmingly human, whereas text generated by their instruction-tuned counterparts is not. Building on this observation, we propose Humanization by Iterative Paraphrasing (HIP), a detector-agnostic pipeline that minimally fine-tunes a base model into a paraphraser and applies it iteratively. Compared with the baselines we test, HIP yields a stronger trade-off between semantic preservation and detector evasion on commercial detectors. Across Llama-3 and Qwen-3 families, spanning model sizes from 0.6B to 70B, HIP consistently improves detector human-likeness. Our findings suggest that current detectors are tracking artifacts of instruction tuning and local context more than any invariant notion of machine-generated text. This, in turn, calls for detector designs that model these factors more explicitly. 

---
# Drifting Objectives for Refining Discrete Diffusion Language Models 

**Authors**: Daisuke Oba, Hiroki Furuta, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2605.19470)  

**Abstract**: Discrete diffusion language models (DDLMs) generate text by iteratively denoising categorical token sequences, while recent drifting methods for continuous generators suggest that part of this sampling-time correction can instead be absorbed into training through an anti-symmetric fixed-point objective. We study how to transfer this principle to DDLMs, where the main challenge is the interface with discrete text: hard token samples are non-differentiable, and categorical predictions do not directly provide continuous samples to drift. We formulate TokenDrift, a drifting objective that lifts categorical predictions to soft-token features, applies anti-symmetric drifting in a frozen semantic space, and backpropagates the resulting stop-gradient feature target to DDLM logits. In controlled continual-training experiments with masked and uniform-state diffusion backbones, TokenDrift improves fixed-NFE generation quality over matched continuation baselines, reducing Gen.-PPL at 4 NFEs by 89% on MDLM and 86% on DUO. These results suggest that drifting can provide a practical refinement objective for DDLMs. 

---
# Backtracking When It Strays: Mitigating Dual Exposure Biases in LLM Reasoning Distillation 

**Authors**: Bing Wang, Shaotian Yan, Chen Shen, kaiyuan liu, Sinan Fan, Ximing Li, Rui Miao, Xiaosong Yuan, Zhanming Shen, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.19433)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in complex reasoning tasks via long chain-of-thought (CoT), yet their immense computational overhead hinders real-world deployment. LLM reasoning distillation addresses this by transferring reasoning capabilities from formidable teacher models to compact student models. However, existing distillation paradigms face a fundamental dilemma. Typical off-policy distillation strictly utilizes teacher-generated golden trajectories, suffering from an exposure bias due to the mismatch between training distributions and student-generated inference contexts, which leads to error cascades in long CoT reasoning. To address this, on-policy distillation allows students to explore their own trajectories, but we demonstrate that it inherently introduces a reciprocal reversed exposure bias: the teacher model also struggles to provide positive guidance when conditioned on student-generated sub-optimal contexts. To resolve this dual exposure biases problem, we propose Monitoring Trajectories and Backtracking when it strays (MOTAB), a new LLM reasoning distillation pipeline. Specifically, MOTAB dynamically monitors the student's on-policy generation against an adaptive safety boundary. When the generation strays and exceeds this threshold, MOTAB backtracks to the last safe state and leverages teacher intervention to correct the course. This approach inherently tolerates minor student errors to mitigate exposure bias, while preventing sub-optimal contexts to circumvent reversed exposure bias. Extensive experiments on the LIMO-v2 and AceReason datasets demonstrate that MOTAB effectively alleviates the dual exposure biases, yielding a roughly 3% average performance improvement in reasoning tasks. 

---
# LambdaPO: A Lambda Style Policy Optimization for Reasoning Language Models 

**Authors**: Zhe Yuan, Yipeng Zhou, Jinghan Li, Xinyuan Chen, Bowen Deng, Zhiqian Chen, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.19416)  

**Abstract**: Group Relative Policy Optimization(GRPO) has become a cornerstone of modern reinforcement learning alignment, prized for its efficacy in foregoing an explicit value-critic by leveraging reward normalization across sampled trajectory cohorts. However, the method's reliance on a monolithic statistical baseline, such as the group mean, collapses the relational topology of the trajectory space into a single scalar, thereby erasing the fine-grained preference information essential for navigating complex, rank-sensitive reward landscapes. To address this issue, we introduce a novel framework, Lambda Policy Optimization (LambdaPO), that addresses this information-theoretic bottleneck by re-conceptualizing advantage estimation from a scalar value to a decomposed, pairwise preference structure. Specifically, the advantage for any given trajectory is formulated as the integrated sum of reward differentials against all peers in its cohort, where each pairwise comparison is dynamically attenuated by the policy's own probabilistic confidence in the established preference. To further mitigate the sparsity of binary outcome supervision, we augment the objective with a semantic density reward, derived from the precision-recall alignment between generated reasoning traces and ground-truth solutions. As a result, our method can mine more fine-grained optimization signals from a group of rollouts, guiding the LLM to a better optima. Experimental results across challenging math reasoning and question-answering tasks demonstrates that LambdaPO improves performance compared to the baseline methods. 

---
# EmbGen: Teaching with Reassembled Corpora 

**Authors**: Arun K Lenin, Kai Rouse, Andrea Nicastro, Anna Leontjeva  

**Link**: [PDF](https://arxiv.org/pdf/2605.19394)  

**Abstract**: Adapting small instruction-tuned models to specialized domains often relies on supervised fine-tuning (SFT) on curated instruction-response examples, which is expensive to collect at scale. Synthetic training examples generated by a teacher LLM from a domain corpus can reduce this cost, but existing pipelines can produce homogenized outputs and do not consistently capture cross-passage or cross-document dependencies. We introduce EmbGen, a synthetic data generation pipeline that decomposes a corpus into entity-description pairs, reassembles them using semantic structure inferred from embedding similarity, and then generates question-answer (QA) pairs via proximity, intra-cluster, and inter-cluster sampling with cluster-specialized system prompts. We evaluate EmbGen against EntiGraph, InstructLab and Knowledge-Instruct on three datasets of varied semantic heterogeneity, under fixed token budgets (5 and 20 million tokens). We use lexical overlap metrics, an LLM-as-a-judge rubric, and Binary Accuracy, a composed metric combining Factual Accuracy and Completeness for evaluation. EmbGen improves Binary Accuracy on the most heterogeneous dataset by 12.5% at 5M and 88.9% at 20M tokens budget, relative to the strongest baseline, while remaining competitive across other datasets with lower heterogeneity. 

---
# Taming the Thinker: Conditional Entropy Shaping for Adaptive LLM Reasoning 

**Authors**: Shuyu Wei, Jian Sun, Delai Qiu, Yining Wang, Shengping Liu, Jiaen Liang, Ying Fu, Wei Huang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2605.19358)  

**Abstract**: Entropy-based deep reasoning has emerged as a promising direction for improving the reasoning capabilities of Large Language Models (LLMs), but existing methods often either increase response length indiscriminately or shorten responses at the cost of accuracy. To better balance this trade-off, we introduce Conditional Entropy Shaping (CES), a framework that dynamically controls token-level response entropy, enabling LLMs to produce concise solutions on simple problems while encouraging deeper exploration on hard ones. Built on DAPO, CES uses token-level entropy as an uncertainty signal and applies a conditional bidirectional policy: it penalizes high-entropy "forking point" tokens on correct reasoning paths to improve conciseness, and rewards them on incorrect paths to encourage exploration and error correction. We implement CES on DeepSeek-R1-Distill-7B and evaluate it on 12 mathematical benchmarks. CES consistently improves average accuracy while reducing response length relative to DAPO, and supplementary experiments show similar trends on a smaller 1.5B backbone and on out-of-domain benchmarks. 

---
# SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models 

**Authors**: Yiyang Gu, Junwei Yang, Junyu Luo, Ye Yuan, Bin Feng, Yingce Xia, Shufang Xie, Kaili Liu, Bohan Wu, Qi Shi, Haoran Li, Beier Xiao, Zhiping Xiao, Xiao Luo, Weizhi Zhang, Philip S. Yu, Zequn Liu, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.19357)  

**Abstract**: Large language models (LLMs) are increasingly applied to scientific research, yet existing evaluations often fail to reflect the fine-grained capabilities required in practice. Most benchmarks are manually curated or domain-generic, limiting scalability and alignment with real scientific use cases. In this paper, we propose a new framework named SciCustom to address the problem. It enables the custom construction of benchmarks from large-scale scientific data to evaluate application-specific scientific capabilities in LLMs. SciCustom first organizes scientific knowledge into ontology-grounded knowledge units with controlled granularity and trains a tagger to map large-scale data instances into this knowledge space. Given a custom requirement, relevant knowledge units are identified via voting-based multi-model consensus. These units enable relevance-aware benchmark retrieval via binary search, followed by proxy subset selection and data-grounded benchmark generation for efficient evaluation. Experiments in chemistry and healthcare demonstrate that SciCustom reveals fine-grained differences in LLM scientific capabilities that standard benchmarks overlook, while requiring neither expert annotation nor synthetic question generation. This work provides a scalable and application-aware foundation for benchmarking scientific capabilities in LLMs. The source code is available at this https URL. 

---
# IMLJD: A Computational Dataset for Indian Matrimonial Litigation Analysis 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.19346)  

**Abstract**: We present IMLJD, an open dataset of 3,613 Indian court judgments covering matrimonial disputes under IPC Section 498A, the Protection of Women from Domestic Violence Act, and CrPC Section 482. The dataset covers the Supreme Court of India from 2000 to 2024 (1,474 cases) and the Karnataka High Court from 2018 to 2024 (2,139 cases), with structured outcome labels, metadata-derived indicators, and a knowledge graph. We find that 57.6% of quashing petitions succeed at the Supreme Court level compared to 39.7% at the Karnataka High Court level. On a matched 2018 to 2024 period, the SC quash rate is 59.3%, widening the differential to 19.6 percentage points and confirming the finding is robust to temporal adjustment. The dataset, code, and knowledge graph are released openly at this https URL and this https URL. 

---
# Retrieval-Augmented Linguistic Calibration 

**Authors**: Yi-Fan Yeh, Linwei Tao, Minjing Dong, Tao Huang, Jialin Yu, Philip Torr, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.19344)  

**Abstract**: Linguistic cues such as "I believe" and "probably" offer an intuitive interface for communicating confidence, yet a generalisable, principled calibration framework for linguistic confidence expressions remains underexplored. In particular, co-occurring linguistic cues, contextual variation, and subjective audience interpretation pose unique challenges. We therefore model linguistic confidence as a distribution over plausible perceived probability values that a statement is correct, capturing interpretation variability that scalar representations discard. Within this distributional framework, we introduce faithfulness as a complementary evaluation dimension and present Faithfulness Divergence (FD), an information-theoretic metric quantifying the surprise induced in audience beliefs upon truth revelation. Building on these foundations, we present Retrieval-Augmented Linguistic Calibration (RALC), a lightweight post-hoc pipeline that propagates calibrated confidence signals back into natural language via retrieval-augmented rewriting. Across three QA benchmarks and five LLM families, RALC improves in-domain faithfulness and calibration up to 66% and 58%, respectively, outperforming black-box and grey-box calibration baselines. 

---
# HalluWorld: A Controlled Benchmark for Hallucination via Reference World Models 

**Authors**: Emmy Liu, Varun Gangal, Michael Yu, Zhuofu Tao, Karan Singh, Sachin Kumar, Steven Y. Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.19341)  

**Abstract**: Hallucination remains a central failure mode of large language models, but existing benchmarks operationalize it inconsistently across summarization, question answering, retrieval-augmented generation, and agentic interaction. This fragmentation makes it unclear whether a mitigation that works in one setting reduces hallucinations across contexts. Current benchmarks either require human annotation and fixed references that may be memorized, or rely on observations in settings that are difficult to reproduce. To study root causes, we introduce HalluWorld, an extensible benchmark grounded in an explicit reference-world formulation: a model hallucinates when it produces an observable claim that is false with respect to this world. Building on this view, we construct synthetic and semi-synthetic environments in which the reference world is fully specified, the model's view is controlled, and hallucination labels are generated automatically. HalluWorld spans gridworlds, chess, and realistic terminal tasks, enabling controlled variation of world complexity, observability, temporal change, and source-conflict policy, and disentangling hallucinations into fine-grained error categories. We evaluate frontier and open-weight language models across these settings and find consistent patterns: perceptual hallucination on directly observed information is near-solved for frontier models, while multi-step state tracking and causal forward simulation remain difficult and are not generally solved by extended thinking. In the terminal setting, models also struggle with when to abstain. The uneven profile of failures across probe types and domains suggests that hallucinations arise from distinct failure modes rather than a single capability. Our results suggest that controlled reference worlds offer a scalable and reproducible path toward measuring and reducing hallucinations in modern language models. 

---
# A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation 

**Authors**: Seonjeong Hwang, Jun Seo, Hyounghun Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.19316)  

**Abstract**: Recent studies in difficulty-controlled reading comprehension item generation have leveraged large language models (LLMs) to produce items by adjusting difficulty-related features. However, existing methods typically rely on a single-agent prompting approach, which often fails to consistently satisfy specified feature constraints, resulting in items that deviate from the target difficulty level. To address this limitation, we introduce MAFIG, a Multi-agent Framework for Feature-constrained Item Generation, where multiple LLM agents and feature-specific evaluators collaborate to generate and iteratively revise items based on intended constraints. Furthermore, to verify the efficacy of MAFIG in difficulty control, we propose a method for constructing a sequence of feature constraint sets that yield items with monotonically increasing difficulty. Experimental results demonstrate that MAFIG generates items that adhere to target constraints at a significantly higher rate than baselines, achieving robust difficulty control through the difficulty-calibrated constraint sequence. 

---
# How Do Document Parsers Break? Auditing Structural Vulnerability in Document Intelligence 

**Authors**: Yue Chen, Yihao Wang, Ziyi Tang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.19309)  

**Abstract**: Document Layout Analysis (DLA) pipelines provide structured page representations for retrieval-augmented generation, long-document question answering, and other document intelligence systems, yet their robustness evaluation remains largely area-centric. We identify this Footprint Bias and propose a lightweight output-level auditing framework that decouples probe construction, policy-driven targeting, and structure-aware diagnosis. The framework combines Block-level Structural Loss Rate (B-SLR), granularity-aware exposure descriptors, and pathway attribution to analyze where perturbations interact with layout structure and how failures propagate. Across MinerU and PP-StructureV3 on 1,000 pages, affected area weakly tracks perturbation-induced OCR instability (R^2=0.384/0.110), whereas B-SLR aligns much more closely with it (R^2=0.727/0.916). Exposure descriptors further separate occlusion- and topology-dominant pathways, and small structurally targeted probes cause downstream QA/retrieval degradation comparable to larger-footprint perturbations. These results shift DLA robustness evaluation from footprint-based stress testing toward structure-aware vulnerability auditing. 

---
# Are Rationales Necessary and Sufficient? Tuning LLMs for Explainable Misinformation Detection 

**Authors**: Bing Wang, Rui Miao, Ximing Li, Chen Shen, Shaotian Yan, Changchun Li, Kaiyuan Liu, Xiaosong Yuan, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.19285)  

**Abstract**: The rapid spread of misinformation on social media platforms has become a formidable challenge. To mitigate its proliferation, Misinformation Detection (MD) has emerged as a critical research topic. Traditional MD approaches based on small models typically perform binary classification through a black-box process. Recently, the rise of Large Language Models (LLMs) has enabled explainable MD, where models generate rationales that explain their decisions, thereby enhancing transparency. Existing explainable MD methods primarily focus on crafting sophisticated prompts to elicit rationales from off-the-shelf LLMs. In this work, we propose a pipeline to fine-tune a dedicated LLM specifically for explainable MD. Our pipeline begins by collecting large-scale fact-checked articles, and then uses multiple strong LLMs to produce veracity predictions and rationales. To ensure high-quality training data, we leverage a filtering strategy that selects only the correct instances for fine-tuning. While this pipeline is intuitive and prevalent, our experiments reveal that naive filtering based solely on label correctness is insufficient in practice and suffers from two critical limitations: (1) Coarse-grained labels cause insufficient rationales: Rationales filtered solely based on binary labels are insufficient to adequately support their decisions; (2) Over-verification behavior causes unnecessary rationales: Stronger LLMs tend to exhibit over-verification behavior, producing excessively verbose and unnecessary rationales. To address these issues, we introduce LONSREX, a novel data synthesis pipeline to Locate Necessary and Sufficient Rationales for Explainable MD. Specifically, we propose a metric that quantifies the contribution of each verification step to the final prediction, thereby evaluating its necessity and sufficiency. Experimental results demonstrate the effectiveness of LONSREX. 

---
# Language models struggle with compartmentalization 

**Authors**: Thomas Vincent Howe, David Wingate  

**Link**: [PDF](https://arxiv.org/pdf/2605.19284)  

**Abstract**: In the training data used by large language models (LLMs), the same latent concept is often presented in multiple distinct ways: the same facts appear in English and Swahili; many functions can be expressed in both Python and Haskell; we can express propositions in both formal and natural language. We show that LLMs can exhibit compartmentalization, where they fail to identify and share statistical strength between distinct presentations of unified concepts. In the worst case, LLMs simply learn parallel internal representations of each presentation of the concept, saturating model capacity with redundancies and decreasing sample efficiency with the number of such presentations. We also demonstrate that synthetic parallel data can fail to improve this despite being easily learned itself. Under this framework, we find that, for small models, early multilingual learning is nearly entirely compartmentalized. Finally, all interventions that we study exhibit a phase transition in which their effectiveness depends on the number of distinct presentations, suggesting that the language modeling objective may only inconsistently unify representations. 

---
# OpenCompass: A Universal Evaluation Platform for Large Language Models 

**Authors**: Maosong Cao, Kai Chen, Haodong Duan, Yixiao Fang, Tong Gao, Ge Jiaye, Mo Li, Hongwei Liu, Junnan Liu, Yuan Liu, Chengqi Lyu, Han Lyu, Ningsheng Ma, Zerun Ma, Yu Sun, Zhiyong Wu, Linchen Xiao, Jun Xu, Haochen Ye, Zhaohui Yu, Yike Yuan, Songyang Zhang, Yufeng Zhao, Fengzhe Zhou, Peiheng Zhou, Dongsheng Zhu, Lin Zhu, Jingming Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2605.19276)  

**Abstract**: In recent years, the field of artificial intelligence has undergone a paradigm shift from task-specific small-scale models to general-purpose large language models (LLMs). With the rapid iteration of LLMs, objective, quantitative, and comprehensive evaluation of their capabilities has become a critical link in advancing technological development. Currently, the mainstream static benchmark dataset-based evaluation methods face challenges such as the diversity of task types, inconsistent evaluation criteria, and fragmentation of data and processing workflows, making it difficult to efficiently conduct cross-domain and large-scale model evaluation. To address the aforementioned issues, this paper proposes and open-sources OpenCompass, a one-stop, scalable, and high-concurrency-supported general-purpose LLM evaluation platform. Adhering to the design philosophy of modularization and component decoupling, the platform boasts three core advantages: high compatibility, flexibility, and high concurrency. The core architecture of OpenCompass comprises five key components: the Configuration System, Task Partitioning Module, Execution and Scheduling Module, Task Execution Unit, and Result Visualization Module. Its workflow provides rule-based, LLM-as-a-Judge, and cascaded evaluators to adapt to the requirements of different task scenarios. Supporting mainstream benchmark datasets across multiple domains, including knowledge, reasoning, computation, science, language, code, etc., the platform offers a unified and efficient LLM evaluation tool for both academia and industry, facilitating the accurate identification of strengths and weaknesses of LLMs as well as their subsequent optimization. 

---
# Lost in Interpretation: The Plausibility-Faithfulness Trade-off in Cross-Lingual Explanations 

**Authors**: Somnath Banerjee, Pranav Jha, Rima Hazra, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.19274)  

**Abstract**: LLMs deployed multilingually are often audited via English explanations for non-English inputs. We evaluate extractive explanations ''where the model identifies input token spans as evidence alongside a generated rationale'' and uncover a systematic trade-off: English-pivot explanations can achieve higher span agreement with human rationales while their evidence becomes less causally grounded in the model's prediction, as measured by both comprehensiveness and sufficiency. Across 3 tasks, 5~languages, and 2~multilingual LLM families, we find that English explanations frequently produce fluent but loosely anchored rationales, with comprehensiveness degrading by up to 5.7x relative to native-language conditions - even as task accuracy remains stable across settings. For socially nuanced classification, English pivots also fail to preserve pragmatic cues, reducing both faithfulness and span agreement. We recommend auditing explanations in the input language, reporting multi-faceted faithfulness metrics beyond lexical overlap, and treating English rationales as communication summaries rather than faithful decision traces. 

---
# DECOR: Auditing LLM Deception via Information Manipulation Theory 

**Authors**: Linyue Cai, Samuel Yeh, Jwala Dhamala, Rahul Gupta, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.19270)  

**Abstract**: Large language models can deceive by subtly manipulating truthful information -- omitting key facts, shifting focus, or obscuring meaning -- making such behavior difficult to detect. Existing black-box methods rely on coarse-grained judgments, offering limited interpretability and failing to pinpoint which facts were distorted and how. We introduce DECOR, a multi-agent framework grounded in Information Manipulation Theory for fine-grained auditing of strategic deception in LLM responses. DECOR decomposes input contexts into atomic informational units and scores each unit against the response across four dimensions of manipulation, producing interpretable manipulation profiles that are aggregated into a global deception index. We comprehensively evaluate DECOR on both single-turn and multi-turn deception detection benchmarks spanning real-world domains, and show that DECOR achieves state-of-the-art performance on both, outperforming competitive baselines. The framework generalizes across 15 frontier models, and ablation studies confirm the contribution of each key design component. Our findings demonstrate that fine-grained, theory-grounded auditing of information manipulation offers an effective and interpretable path for LLM deception detection. 

---
# FormalASR: End-to-End Spoken Chinese to Formal Text 

**Authors**: Wanyi Ning, Yinshang Guo, Haitao Qian, Jiyuan Cheng, Weiyuan Feng, Yufei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.19266)  

**Abstract**: Automatic speech recognition (ASR) systems are typically optimized for verbatim transcription, which preserves disfluencies, filler words, and informal spoken structures that are often unsuitable for downstream writing-oriented applications. A common workaround is a two-stage ASR+LLM pipeline for post-editing, but this design increases latency and memory cost and is difficult to deploy on-device. We present FormalASR, two compact end-to-end models (0.6B and 1.7B) that directly transcribe spoken Chinese into formal written text. To enable this setting, we build WenetSpeech-Formal and Speechio-Formal, two large-scale spoken-to-formal datasets constructed by LLM-based rewriting and quality filtering. We then fine-tune Qwen3-ASR at two scales (0.6B and 1.7B) with supervised fine-tuning. Experiments on WenetSpeech-Formal and Speechio-Formal show that FormalASR achieves up to 37.4% relative CER reduction over verbatim baselines, while also improving ROUGE-L and BERTScore. FormalASR requires no post-processing LLM at deployment time, providing a lightweight, on-device solution for spoken-to-formal transcription. 

---
# AI Technologies in Language Access: Attitudes Towards AI and the Human Value of Language Access Managers 

**Authors**: Miguel A. Jiménez-Crespo, Stephanie Rodriguez, Alejandro Jaume Losa  

**Link**: [PDF](https://arxiv.org/pdf/2605.19234)  

**Abstract**: The rapid emergence of AI technologies is reshaping translation practices and theory across the board. This paper deals with the impact of AI in language access. This area is characterized by the need to serve broad and diverse user populations, within a context where efficiency and access are shaped by legal mandates, ethical and commercial tensions, and safety concerns. This paper reports on the attitudes and perceptions of language access managers towards the AI and the human value in the AI age. Methodologically, this paper presents an analysis of a subset of a broader study on language access and technology, specifically a qualitative thematic analysis of ten semi-structured interviews with language access managers in the USA working in healthcare, court, public service and local government contexts. The results indicate that language access managers show conditional optimism towards the inevitable AI implementations, are strongly risk aware, and deeply committed to the human value and human oversight of AI implementations and output. 

---
# Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution 

**Authors**: Xiaoou Liu, Tiejin Chen, Dengjia Zhang, Yaqing Wang, Lu Cheng, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.19228)  

**Abstract**: Large Language Models have achieved strong performance on reasoning tasks with objective answers by generating step-by-step solutions, but diagnosing where a multi-step reasoning trace might fail remains difficult. Confidence estimation offers a diagnostic signal, yet existing methods are restricted to final answers or require internal model access. In this paper, we introduce Stepwise Confidence Attribution (SCA), a framework for closed-source LLMs that assigns step-level confidence based only on generated reasoning traces. SCA applies the Information Bottleneck principle: steps aligning with consensus structures across correct solutions receive high confidence, while deviations are flagged as potentially erroneous. We propose two complementary methods: (1) NIBS, a non-parametric IB approach measuring consistency without graph structures, and (2) GIBS, a graph-based IB model that learns subgraphs through a differentiable mask to capture logical variability. Extensive experiments on mathematical reasoning and multi-hop question answering show that SCA reliably identifies low-confidence steps strongly correlated with reasoning errors. Moreover, using step-level confidence to guide self-correction improves the correction success rate by up to 13.5\% over answer-level feedback. 

---
# Fine-tuning language encoding models on slow fMRI improves prediction for fast ECoG 

**Authors**: Aditya R. Vaidya, Richard J. Antonello, Alexander G. Huth  

**Link**: [PDF](https://arxiv.org/pdf/2605.19224)  

**Abstract**: Neuroscientists have recently turned to intracranial brain recording methods, like electrocorticography (ECoG), for human experiments because of the fine spatial and temporal resolution that they afford. Models trained on this data, however, are fundamentally restricted by the patient populations that can receive the implants necessary for recording. We propose using non-invasive fMRI to bridge the gap in training data. Using spoken language representations fine-tuned on fMRI, we build encoding models of ECoG. These representations showed improved prediction performance in ECoG, even though the temporal resolution of fMRI is two orders of magnitude worse. Prediction improved in frequency bands well beyond what is directly measured in fMRI. Next, to test the procedure's generalization ability, we fine-tuned models on fMRI responses that were temporally downsampled by a factor of 2. Despite the loss in resolution, these models were able to predict fMRI and ECoG responses at levels comparable to the original fMRI-tuned models. Finally, we showed that ECoG performance steadily scales with the amount of fMRI-tuning data. Our results show that "slow" data like fMRI can be a valuable resource for building better models of "fast" brain data like ECoG. In the future, integrating across multiple recording methods may further improve performance in other applications, like decoding. 

---
# Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering 

**Authors**: Tiejin Chen, Longchao Da, Xiaoou Liu, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.19220)  

**Abstract**: Uncertainty Quantification (UQ) is widely regarded as the primary safeguard for deploying Large Language Models (LLMs) in high-stakes domains. However, we argue that the field suffers from a category error: mainstream UQ methods for LLMs are just unsupervised clustering algorithms. We demonstrate that most current approaches inherently quantify the internal consistency of the model's generations rather than their external correctness. Consequently, current methods are fundamentally blind to factual reality and fail to detect ``confident hallucinations,'' where models exhibit high confidence in stable but incorrect answers. Therefore, the current UQ methods may create a deceptive sense of safety when deploying the models with uncertainty. In detail, we identify three critical pathologies resulting from this dependence on internal state: a hyperparameter sensitivity crisis that renders deployment unsafe, an internal evaluation cycle that conflates stability with truth, and a fundamental lack of ground truth that forces reliance on unstable proxy metrics to evaluate uncertainty. To resolve this impasse, we advocate for a paradigm shift to UQ and outline a roadmap for the research community to adopt better evaluation metrics and settings, implement mechanism changes for native uncertainty, and anchor verification in objective truth, ensuring that model confidence serves as a reliable proxy for reality. 

---
# Time to REFLECT: Can We Trust LLM Judges for Evidence-based Research Agents? 

**Authors**: Leyao Wang, Yanan He, Peng Chen, Asaf Yehudai, Yixin Liu, Rex Ying, Michal Shmueli-Scheuer, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2605.19196)  

**Abstract**: Deep research agents increasingly automate complex information-seeking tasks, producing evidence-grounded reports via multi-step reasoning, tool use, and synthesis. Their growing role demands scalable, reliable evaluation, positioning LLM-as-judge as a supervision paradigm for assessing factual accuracy, evidence use, and reasoning quality. Yet the reliability of these judges for deep research agents remains poorly understood, posing a critical meta-evaluation problem: before deploying LLM judges to supervise research agents, we must first evaluate the judges themselves. Existing meta-evaluations fall short in two ways: (1) reliance on coarse, subjective human-preference agreement; (2) focus on instruction-following or verifiable tasks, leaving open-ended agent executions unexplored. To address these gaps, we introduce REFLECT (REliable Fine-grained LLM judge Evaluation via Controlled inTervention), a meta-evaluation benchmark targeting fine-grained failure detection in agentic environments. REFLECT defines a detailed taxonomy of process- and outcome-level failure modes, instantiated by performing controlled and localized interventions on quality-screened agent execution traces. This yields verifiable, comprehensive, and fine-grained instances for validating the judge models. Our experiments show that current LLM judges remain unreliable: even the best-performing models achieve overall accuracies below 55% across reasoning, tool-use, and report-quality failures, with especially poor performance on evidence verification. Together, our taxonomy and findings expose systematic judge limitations, reveal tradeoffs in cost and reliability, and offer actionable guidance for building more reliable evaluation pipelines for deep research agents. 

---
# MMoA: An AI-Agent framework with recurrence for Memoried Mixure-of-Agent 

**Authors**: Rui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2605.19194)  

**Abstract**: The Mixture-of-Agents (MoA) framework has shown promise in improving large language model (LLM) performance by aggregating outputs from multiple agents. However, existing MoA systems often rely on static routers that do not fully capture temporal and contextual dependencies across aggregation layers. To address this limitation, we propose MMoA, a recurrent MoA architecture that integrates LSTM-based gating into the agent selection process. The recurrence router adaptively modulates agent contributions based on both current inputs and historical routing decisions, enabling more context-aware aggregation. We evaluate MMoA on standard instruction-following benchmarks, including AlpacaEval 2.0, MT-Bench, and Arena-Hard. The results show that MMoA achieves comparable accuracy to traditional MoA while reducing computational overhead by dynamically activating fewer agents. For example, on AlpacaEval 2.0, MMoA achieves a win rate of 58.0%, compared with 59.8% for MoA, while improving runtime efficiency by up to 4.6%. These results suggest that MMoA provides a scalable and efficient approach for adaptive multi-agent LLM systems. 

---
# Prompting language influences diagnostic reasoning and accuracy of large language models 

**Authors**: Adrien Bazoge, Josselin Corvellec, Sofiane Djillali Sid-Ahmed, Pierre-Antoine Gourraud  

**Link**: [PDF](https://arxiv.org/pdf/2605.19173)  

**Abstract**: Large language models (LLMs) are increasingly explored for clinical decision support, yet most evaluations are conducted in English, leaving their reliability in other languages uncertain. Here we evaluate the impact of prompting language on diagnostic reasoning and final diagnosis accuracy by comparing English and French performance across five LLMs (o3, DeepSeek-R1, GPT-4-Turbo, Llama-3.1-405B-Instruct, and BioMistral-7B). A total of 180 clinical vignettes covering 16 medical specialties were assessed by two physicians using an 18-point scale evaluating both diagnosis accuracy and reasoning quality. Four of the five models performed better in English (mean difference 0.37-0.91, adjusted p < 0.05), with the gap spanning multiple aspects of reasoning, including differential diagnosis, logical structure, and internal validity. o3 was the only model showing no overall language effect. These findings demonstrate that prompting language remains a critical determinant of LLM clinical performance, with implications for equitable linguistico-cultural deployment worldwide. 

---
# Agent Meltdowns: The Road to Hell Is Paved with Helpful Agents 

**Authors**: Rishi Jha, Harold Triedman, Arkaprabha Bhattacharya, Vitaly Shmatikov  

**Link**: [PDF](https://arxiv.org/pdf/2605.19149)  

**Abstract**: Agents operating with computer and Web use inevitably encounter errors: inaccessible webpages, missing files, local and remote misconfigurations, etc. These errors do not thwart agents based on state-of-the-art models. They helpfully continue to look for ways to complete their tasks.
We introduce, characterize, and measure a new type of agent failure we call \emph{accidental meltdown}: unsafe or harmful behavior in response to a benign environmental error, in the absence of any adversarial inputs. Because meltdowns are not captured by the existing reliability or safety benchmarks, we develop a taxonomy of meltdown behaviors. We then implement an agent-agnostic infrastructure for injecting simulated local and remote errors into the rollout environment and use it to systematically evaluate agent systems powered by GPT, Grok, and Gemini.
Our evaluation demonstrates that meltdowns (e.g., conducting unauthorized reconnaissance or subverting access control) of varying severity and success occur in 64.7\% of agent rollouts that encounter simulated errors, spanning all combinations of agent system, backing model, and error type. In over half of these meltdowns, unsafe behaviors are not reported to the user. Comparing behaviors of the same agents with and without errors, we find that exploration in response to errors is correlated with unsafe and harmful behavior. 

---
# ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking 

**Authors**: Yanjun Lin, Zimo Xiao, Kartik Natarajan, Mahesh Sankaranarayanan, Niraj Nawanit, Rakshit Parashar, Austin Zhang, Karthik Konaraddi, Rishita Mote, Wei Niu  

**Link**: [PDF](https://arxiv.org/pdf/2605.19077)  

**Abstract**: Task-oriented dialogue systems -- handling transactions, reservations, and service requests -- require predictable behavior, yet the moderately-sized LLMs needed for practical latency are prone to hallucination and format errors that cascade into incorrect actions (e.g., a hotel booked for the wrong date). We propose ReacTOD, a bounded neuro-symbolic architecture that reformulates NLU as discrete tool calls within a self-correcting ReAct loop governed by deterministic validation. A bounded ReAct loop enables iterative self-correction, improving accuracy by up to 9.3 percentage points over single-pass inference on MultiWOZ. A symbolic validator enforces action compliance, schema conformance, and coreference consistency on every dialogue state update, achieving a 93.1% self-correction rate on intercepted errors and producing structured execution traces. Incremental state prediction and on-demand history retrieval keep prompts compact, empirically improving instruction adherence in parameter-constrained models. On MultiWOZ 2.1, ReacTOD achieves a new zero-shot state-of-the-art: gpt-oss-20B reaches 52.71% joint goal accuracy, surpassing the previous best by 14 percentage points, while Qwen3-8B achieves 47.34% with only 8B parameters. On the Schema-Guided Dialogue (SGD) benchmark, ReacTOD with Claude-Opus-4.6 achieves 80.68% JGA under fully end-to-end evaluation with predicted domains, and Qwen3-32B reaches 64.09% -- demonstrating cross-benchmark generalization without task-specific training data. 

---
# Benchmarking Commercial ASR Systems on Code-Switching Speech: Arabic, Persian, and German 

**Authors**: Sajjad Abdoli, Ghassan Al-Sumaidaee, Clayton W. Taylor, Ahmad, ElShiekh, Ahmed Rashad  

**Link**: [PDF](https://arxiv.org/pdf/2605.19069)  

**Abstract**: Code-switching -- the natural alternation between two languages within a single utterance -- represents one of the most challenging and under-studied conditions for automatic speech recognition (ASR). Existing commercial ASR benchmarks predominantly evaluate clean, monolingual audio and report a single Word Error Rate (WER) figure that tells practitioners little about real-world multilingual performance. We present a benchmark evaluating five commercial ASR providers across four language pairs: Egyptian Arabic--English, Saudi Arabic (Najdi/Hijazi)--English, Persian (Farsi)--English, and German--English. Each dataset comprises 300 samples selected by a two-stage pipeline: a heuristic filter scoring transcripts on five structural code-switching signals, followed by a GPT-4o and Gemini 1.5 Pro ensemble scoring candidates across six linguistic dimensions. This pipeline reduces LLM scoring costs by approximately 91\% relative to exhaustive scoring. We evaluate the systems on both WER and BERTScore, arguing that BERTScore is a more reliable metric for Arabic and Persian pairs where transliteration variance causes WER to penalise semantically correct transcriptions. ElevenLabs Scribe v2 achieves the lowest WER across all four language pairs (13.2% overall; 13.1% on Egyptian Arabic) and leads on BERTScore (0.936 overall). We further demonstrate that difficulty-stratified analysis reveals performance gaps masked by aggregate averages, and that BERT embedding projections confirm semantic proximity between reference and hypothesis despite surface-level script differences. The benchmarking dataset is publicly available at this https URL. 

---
# The Annotation Scarcity Paradox in Low-Resource NLP Evaluation: A Decade of Acceleration and Emerging Constraints 

**Authors**: Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2605.19066)  

**Abstract**: Over the past decade, low-resource natural language processing (NLP) has experienced explosive growth, propelled by cross-lingual transfer, massively multilingual models, and the rapid proliferation of benchmarks. Yet this apparent progress masks a critical, insufficiently examined tension: the deep sociolinguistic expertise required to evaluate increasingly complex generative systems is severely strained, inequitably distributed, and structurally marginalised. We present a critical narrative survey of low-resource NLP evaluation (2014--present), tracing its evolution across three phases: early heuristic optimism, the illusions of top-down benchmark scaling, and the current era of generative bottlenecks. We conceptualise the \emph{Annotation Scarcity Paradox}, the structural friction arising when the technical capacity to scale models vastly outpaces the sovereign human infrastructure required to authentically evaluate them. By examining extractive data pipelines, undercompensated ``ghost work'', and language data flaring, we argue that this paradox threatens the epistemic validity of reported progress. We survey emerging responses -- including data augmentation, model-based evaluation, participatory curation, and annotation-efficient approaches via item response theory and active learning -- and assess their equity and validity trade-offs. We close with a practitioner call to action, arguing that overcoming this bottleneck requires a paradigm shift from transactional data extraction to relational, community-embedded evaluation rooted in epistemic governance, data sovereignty, and shared ownership. 

---
# Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models 

**Authors**: Guangzhi Xiong, Qiao Jin, Sanchit Sinha, Zhiyong Lu, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.20158)  

**Abstract**: Large Vision Language Models (LVLMs) show promise in medical applications, but their inability to faithfully ground responses in visual evidence raises serious concerns about clinical trustworthiness. While visual attribution methods are widely used to explain LVLM predictions, whether these explanations actually reflect the visual evidence underlying the model's decision is largely unverified, since ground-truth annotations for internal model reasoning are typically unavailable. We address this question for chest X-ray (CXR) reasoning by developing a causal evaluation framework that retains only CXR-VQA samples for which the expert-annotated region is verified, via counterfactual editing, to be causally responsible for the model's prediction. Using this framework across 11 attribution methods, six open-source LVLMs, and two output modes (direct answer and step-by-step reasoning), we find that existing attribution methods often fail to identify the evidence used by LVLMs. To address this failure, we propose MedFocus, a concept-based attribution method that localizes clinically meaningful anatomical regions via unbalanced optimal transport and measures their causal effect on model outputs through targeted interventions. MedFocus produces spatial, concept-level, and token-level attributions and substantially outperforms prior methods, taking a step toward more trustworthy attribution for medical LVLMs. Our data and code are available at this https URL. 

---
# GEM: GPU-Variability-Aware Expert to GPU Mapping for MoE Systems 

**Authors**: Sourish Wawdhane, Avinash Kumar, Poulami Das  

**Link**: [PDF](https://arxiv.org/pdf/2605.19945)  

**Abstract**: Mixture-of-Expert (MoE) models enable efficient inference by employing smaller experts and activating only a subset of them per token. MoE serving engines distribute experts across multiple GPUs and route tokens to appropriate GPUs at inference time based on experts activated. They process tokens in lock-step fashion, where tokens within a batch must finish processing before proceeding to the next layer. This synchronization barrier acts as a critical bottleneck because the performance of MoE models is limited by the straggler GPU that finishes last. Stragglers emerge when too many heavily used experts are placed on the same GPU or the slowest GPU. While prior works place experts that balance token loads across GPUs, they all overlook GPU variability and often place highly used experts on the slowest GPUs.
We propose GEM, GPU-variability-aware Expert Mapping, a framework for GPU variability-aware expert to GPU mapping for MoE models. GEM exploits two insights. First, we must place experts such that each GPU receives non-uniform token loads based on their variability and they all finish processing a layer at about the same time. Our studies show that there are two types of experts: consistent that are used most of the time and temporal that are often used together for the remaining time. Our second insight is that we must place simultaneously used consistent and temporal experts on different GPUs and avoid placing them on slower GPUs to reduce slowdown. GEM gathers the variability profile of GPUs for each model and task and uses the token load distributions per task to map experts to GPUs. Our experiments show that GEM improves end-to-end latency by 7.9% on average and by up to 16.5% compared to the baseline. 

---
# A Measure-Theoretic Analysis of Reasoning: Structural Generalization and Approximation Limits 

**Authors**: Yuyang Zhang, Yifu Zhang, Xuehai Zhou, Xiaoyin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.19944)  

**Abstract**: While empirical scaling laws for LLM reasoning are well-documented, the theoretical mechanisms governing out-of-distribution (OOD) generalization remain elusive. We formalize reasoning via optimal transport, projecting discrete trajectories into a continuous metric space to quantify domain shifts using the Wasserstein-1 distance. Invoking Kantorovich duality, we bound OOD generalization via architectural Lipschitz continuity and functional approximation limits. This exposes two primary constraints. First, position-dependent attention (e.g., Absolute Positional Encoding) fails to preserve shift invariance, yielding an $\Omega(1)$ Lipschitz constant and expected risk, whereas shift-invariant mechanisms (e.g., Rotary Embeddings) preserve equivariance and bound the error. Second, by mapping sequential backtracking to a Dyck-$k$ language, we establish a strict circuit depth lower bound for $\text{TC}^0$ Transformers. Scaling physical layer depth is necessary to avert representation collapse -- a constraint that scaling representation width cannot bypass due to irreducible approximation bounds in Barron spaces. Evaluations across 54 Transformer configurations on combinatorial search corroborate these bounds, demonstrating that generalization risk degrades monotonically with the Wasserstein domain shift. 

---
# PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents 

**Authors**: Zhuohan Gu, Qizheng Zhang, Omar Khattab, Samuel Madden  

**Link**: [PDF](https://arxiv.org/pdf/2605.19932)  

**Abstract**: Large language model (LLM) agents increasingly operate over long and recurring external contexts, like document corpora and code repositories. Across invocations, existing approaches preserve either the agent's trajectory, passive access to raw material, or task-level strategies. None of them preserves what we argue is most needed for repeated same-context workloads: reusable orientation knowledge (e.g., what the context contains, how it is organized, and which entities, constants, and schemas have historically been useful) about the recurring context itself. We introduce PEEK, a system that caches and maintains this orientation knowledge as a context map: a small, constant-sized artifact in the agent's prompt that gives it a persistent peek into the external context. The map is maintained by a programmable cache policy with three modules: a Distiller that extracts transferable knowledge from inference-time signals, a Cartographer that translates it into structured edits, and a priority-based Evictor that enforces a fixed token budget. On long-context reasoning and information aggregation, PEEK improves over strong baselines by 6.3-34.0% while using 93-145 fewer iterations and incurring 1.7-5.8x lower cost than the state-of-the-art prompt-learning framework, ACE. On context learning, PEEK improves solving rate and rubric accuracy by 6.0-14.0% and 7.8-12.1%, respectively, at 1.4x lower cost than ACE. These gains generalize across LMs and agent architectures, including OpenAI Codex, a production-grade coding agent. Together, these results show that a context map helps long-context LLM agents interact with recurring external contexts more accurately and efficiently. 

---
# FineBench: Benchmarking and Enhancing Vision-Language Models for Fine-grained Human Activity Understanding 

**Authors**: Gueter Josmy Faure, Min-Hung Chen, Jia-Fong Yeh, Hung-Ting Su, Winston H. Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2605.19846)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated remarkable capabilities in general video understanding, yet they often struggle with the fine-grained comprehension crucial for real-world applications requiring nuanced interpretation of human actions and interactions. While some recent human-centric benchmarks evaluate aspects of model behaviour such as fairness/ethics, emotion perception, and broader human-centric metrics, they do not combine long-form videos, very dense QA coverage, and frame-level spatial/temporal grounding at scale. To bridge this gap, we introduce FineBench, a human-centric video question answering (VQA) benchmark specifically designed to assess fine-grained understanding. FineBench comprises 199,420 multiple-choice QA pairs densely annotated across 64 long-form videos (15 minutes each), focusing on detailed person movement, person interaction, and object manipulation, including compositional actions. Our extensive evaluation reveals that while proprietary models like GPT-5 achieve respectable performance, current open-source VLMs significantly underperform, struggling particularly with spatial reasoning in multi-person scenes and distinguishing subtle differences in human movements and interactions. To address these identified weaknesses, we propose FineAgent, a modular framework that enhances VLMs by leveraging a Localizer and a Descriptor. Experiments show that FineAgent consistently improves the performance of various open VLMs on FineBench. FineBench provides a rigorous testbed for future research into fine-grained human-centric video understanding, while FineAgent offers a practical approach to enhance such reasoning in current VLMs. 

---
# CADENet: Condition-Adaptive Asynchronous Dual-Stream Enhancement Network for Adverse Weather Perception in Autonomous Driving 

**Authors**: Sherif Khairy, Catherine M. Elias  

**Link**: [PDF](https://arxiv.org/pdf/2605.19837)  

**Abstract**: Adverse weather (rain, fog, sand, and snow) degrades camera-based object detection in autonomous vehicles. Existing enhancement-then-detect approaches stall the safety-critical perception loop, violating hard real-time requirements. Progress on this problem is also constrained by an under-recognized evaluation ceiling: ground truth annotated on degraded images cannot credit a detector that recovers objects the annotators themselves could not see, so a genuinely useful enhancement can register as a near-flat F1 gain. This paper presents CADENet (Condition-Adaptive Asynchronous Dual-stream Enhancement Network), a training-free three-thread system: Thread S (YOLOv11n) delivers detections at full frame rate with zero added latency; Thread Q applies condition-adaptive enhancement (CAPE) and fuses results via entropy-guided NMS (EG-NMS) without blocking Thread S; Thread E provides CLIP zero-shot weather classification, so new weather categories require only a new text prompt, with no labeled data and no retraining. Evaluated on 1327 DAWN images (YOLOv11m, IoU = 0.5, confidence = 0.25), CADENet achieves Recall = 0.0103 (micro), F1 = 0.0230 on snow, and F1 = 0.0038 on rain. We formalize the annotation completeness bias on DAWN-class data, so the reported F1 values are lower bounds on the true gain; recall is the annotation-gap-immune headline metric. Thread S sustains approximately 44 FPS regardless of enhancement load. No model retraining or additional sensor hardware is required. 

---
# Mega-ASR: Towards In-the-wild^2 Speech Recognition via Scaling up Real-world Acoustic Simulation 

**Authors**: Zhifei Xie, Kaiyu Pang, Haobin Zhang, Deheng Ye, Xiaobin Hu, Shuicheng Yan, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2605.19833)  

**Abstract**: Despite rapid advances in automatic speech recognition (ASR) and large audio-language models, robust recognition in real-world environments remains limited by an "acoustic robustness bottleneck": models often lose acoustic grounding and produce omissions or hallucinations under severe, compositional distortions. We propose Mega-ASR, a unified ASR-in-the-wild framework that combines scalable compound-data construction with progressive acoustic-to-semantic optimization. We introduce Voices-in-the-Wild-2M, covering 7 classic acoustic phenomena and 54 physically plausible compound scenarios, and train Mega-ASR with Acoustic-to-Semantic Progressive Supervised Fine-Tuning and Dual-Granularity WER-Gated Policy Optimization. Extensive experiments demonstrate that Mega-ASR achieves significant advantages over prior state-of-the-art systems on adverse-condition ASR benchmarks (45.69% vs. 54.01% on VOiCES R4-B-F, and 21.49% vs. 29.34% on NOIZEUS Sta-0). On complex compositional acoustic scenarios, Mega-ASR further delivers over 30% relative WER reduction against strong open- and closed-source baselines, establishing a scalable paradigm for robust ASR in-the-wild. 

---
# From Prompts to Pavement Through Time: Temporal Grounding in Agentic Scene-to-Plan Reasoning 

**Authors**: Ahmed Y. Gado, Omar Y. Goba, Alaa Hassanein, Catherine M. Elias, Ahmed Hussein  

**Link**: [PDF](https://arxiv.org/pdf/2605.19824)  

**Abstract**: Recent attempts to support high-level scene interpretation and planning in Autonomous Vehicles (AVs) using ensembles of Large Language Models (LLMs) and Large Multimodal Models (LMMs) continue to treat time as a secondary property. This lack of temporal grounding leads to inconsistencies in reasoning about continuous actions, undermining both safety and interpretability. This work explores whether temporal conditioning within inter-agent communication can preserve or enhance coherence without introducing degradation in semantic or logical consistency. To investigate this, we introduce three planner architectures with progressively increasing temporal integration and evaluate them on curated subsets of the BDD-X dataset using semantic, syntactic, and logical metrics. Results show that while temporal conditioning reshapes reasoning style, it yields no statistically significant improvements in standard NLP-based correctness metrics. However, qualitative analysis reveals predictive hazard reasoning, stable corrective behavior, and strategic divergence in the Sentinel. These findings clarify the limits of prompt-based temporal grounding and establish the first empirical benchmark for temporal scene-to-plan reasoning. 

---
# What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code 

**Authors**: Yuze Zhao, Junpeng Fang, Lu Yu, Zhenya Huang, Kai Zhang, Qing Cui, Qi Liu, Jun Zhou, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.19762)  

**Abstract**: Code has become a standard component of modern foundation language model (LM) training, yet its role beyond programming remains unclear. We revisit the claim that code improves reasoning through controlled pretraining experiments on a 10T-token corpus with fine-grained domain separation. Our findings are threefold. First, when code is restricted to standalone executable programs and Code-NL data are controlled for, code substantially improves programming ability but does not act as a general reasoning enhancer; instead, it competes with knowledge-intensive tasks, especially complex mathematical reasoning. Second, the reasoning gains often attributed to code are better explained by cross-domain structured reasoning traces, such as code-text and math-text mixtures, rather than by executable code alone. Third, increasing the density of structured math-domain samples within a fixed math budget yields substantial gains on difficult mathematical reasoning while largely preserving programming performance, suggesting that cognitive scaffolds offer a targeted way to mitigate cross-domain trade-offs. Finally, routing analyses show that data-composition effects are reflected in expert-activation patterns, providing mechanism-level evidence for competitive and synergistic interactions across domains. Our results clarify which data characteristics transfer across capability dimensions and point to more precise data-centric optimization strategies. 

---
# OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond 

**Authors**: Zunhai Su, Rui Yang, Chao Zhang, Yaxiu Liu, Yifan Zhang, Wei Wu, Jing Xiong, Dayou Du, Xialie Zhuang, Yulei Qian, Yuchen Xie, Yik-Chung Wu, Hongxia Yang, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.19660)  

**Abstract**: The rapid advancement toward long-context reasoning and multi-modal intelligence has made the memory footprint of the Key-Value (KV) cache a dominant memory bottleneck for efficient deployment. While the established per-channel quantization effectively accommodates intrinsic channel-wise outliers in Key tensors, its efficacy diminishes under extreme compression. In this work, we revisit the inherent limitations of the per-channel quantization paradigm from both empirical and theoretical perspectives. Our analysis identifies Token Norm Imbalance (TNI) as the primary bottleneck to quantization fidelity. We demonstrate that TNI systematically amplifies errors when shared quantization parameters are required to span token groups exhibiting substantial norm disparities. Instead of relying on intricate quantization pipelines (e.g., TurboQuant), we propose OScaR (Omni-Scaled Canalized Rotation), an accurate and lightweight KV cache compression framework for X-LLMs (i.e., text-only, multi-modal, and omni-modal LLMs). Advancing the per-channel paradigm, OScaR employs Canalized Rotation followed by Omni-Token Scaling to mitigate TNI-induced sequence-dimensional variance both effectively and efficiently, further supported by our optimized system design and CUDA kernels. Extensive evaluations across X-LLMs show that OScaR consistently outperforms existing methods and achieves near-lossless performance under INT2 quantization, establishing it as a robust, low-complexity, and universal framework that defines a new Pareto front. Compared with the BF16 FlashDecoding-v2 baseline, our OScaR implementation achieves a notable up to 3.0x speedup in decoding, reduces memory footprint by 5.3x, and increases throughput by 4.1x. The code for OScaR is publicly available at this https URL. 

---
# Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries 

**Authors**: Xing Zhang, Yanwei Cui, Guanghui Wang, Ziyuan Li, Wei Qiu, Bing Zhu, Peiyang He  

**Link**: [PDF](https://arxiv.org/pdf/2605.19576)  

**Abstract**: Self-evolving skill libraries face a silent failure mode we term \emph{library drift}: unbounded skill accumulation without outcome-driven lifecycle management causes retrieval degradation, false-positive injections, and performance stagnation. Recent evaluation confirms the symptom--LLM-authored skills deliver +0.0pp gain while human-curated ones deliver +16.2pp (SkillsBench)--yet the underlying mechanism has not been isolated. We provide (1) a reproducible trigger: ablations that isolate drift--one disables skill injection (flat floor, +0.002), one imposes premature retirement (active harm, $-$0.019); (2) trace-level diagnostics: an append-only evidence log with per-skill contribution scores, attribution verdicts, and router engagement metrics that make the failure visible before it reaches end-task scores; and (3) a verified fix: a minimal governance recipe (outcome-driven retirement + bounded active-cap + meta-skill authoring prior) that lifts held-out pass@1 from a 0.258 baseline to a late-window mean of 0.584 (rolling gain $+$0.328) on MBPP+ hard-100 over 100 rounds. Eight ablations decompose which governance mechanisms are load-bearing and which are subsumed, providing a concrete playbook for diagnosing library drift in any self-evolving agent. 

---
# Position: The Turing-Completeness of Real-World Autoregressive Transformers Relies Heavily on Context Management 

**Authors**: Guanyu Cui, Zhewei Wei, Kun He  

**Link**: [PDF](https://arxiv.org/pdf/2605.19514)  

**Abstract**: Many works make the eye-catching claim that Transformers are Turing-complete. However, the literature often conflates two distinct settings: (i) a fixed Transformer system setting, in which a fixed autoregressive Transformer is coupled with a fixed context-management method to process inputs of different lengths step by step, and (ii) a scaling-family setting, in which a family of different models (with increasing context-window length or numerical precision) is used to handle different input lengths. Existing proofs of Transformer Turing-completeness are frequently established in setting (ii), whereas real-world LLM deployment and the standard notion of Turing-completeness correspond more naturally to setting (i). In this paper, we first formalize the fixed-system setting, thereby providing a concrete characterization of how real-world LLMs operate. We then argue that results proved in the scaling-family setting provide theoretically meaningful resource bounds but do not establish Turing-completeness, thereby clarifying a common misinterpretation of existing results. Finally, we show that different context-management methods can yield sharply different computational power, and we advocate the position that context management is a central component that critically determines the computational power of real-world autoregressive Transformers. 

---
# CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization 

**Authors**: Ahmed Heakl, Abdelrahman M. Shaker, Youssef Mohamed, Rania Elbadry, Omar Fetouh, Fahad Shahbaz Khan, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2605.19436)  

**Abstract**: When a model produces a correct solution under reinforcement learning with verifiable rewards (RLVR), every token receives the same reward signal regardless of whether it was a decisive reasoning step or a grammatical filler. A natural fix is to condition the model on the correct answer as a teacher, identifying tokens it would have generated differently had it known the answer. Prior work shows this either corrupts training by leaking the answer into the gradient, or produces a weak signal that cannot distinguish decisive steps from filler, since both look equally surprising relative to the model's baseline. We propose Contrastive Evidence Policy Optimization (CEPO), which asks a sharper question at every token: not just "does the correct answer favor this token?" but "does the correct answer favor it while the wrong answer disfavors it?" A token satisfying both is a genuine reasoning step; one satisfying neither is filler. The wrong-answer teacher is constructed from rejected rollouts already in the training batch, incurring no additional sampling cost. We prove CEPO inherits all structural safety guarantees of the prior state of the art while strictly sharpening credit at decisive tokens, with the improvement vanishing exactly at filler positions. Empirically, CEPO achieves 43.43% and 60.56% average accuracy across five multimodal mathematical reasoning benchmarks at 2B and 4B scale, respectively, versus 41.17% and 57.43% for GRPO under identical training budgets. Distribution-matching self-distillation methods (OPSD, SDPO) fall below the untrained baseline, empirically confirming the information leakage our theory predicts. Our code is available at this https URL. 

---
# PAVE: A Cognitive Architecture for Legitimate Violation in Generative Agent Societies 

**Authors**: Ahmad Yehia, Abduallah Mohamed, Kun Qian, Tianyi Wang, Jiseop Byeon, Omar Hassanin, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2605.19351)  

**Abstract**: Generative agents based on large language models reproduce believable human behavior in cooperative settings, but how they should reason in situations where rule-breaking may be required, such as fire evacuation or authority-supervised emergency, remains poorly characterized. We propose PAVE (Perception, Assessment, Verdict, Emulation), a novel four-module cognitive architecture that addresses this gap end to end: (i) Perception extracts a structured context with explicit authority distance, peer behaviors, and severity-tagged situational cues; (ii) Assessment scores the context along five scalars including an explicit legitimacy judgment that checks necessity, proportionality, and absence of alternatives; (iii) Verdict decides to comply or violate under a hard legitimacy gate, with a per-agent threshold elicited from the persona; (iv) Emulation enacts the verdict and scopes the violation to the rule the trigger justifies. We instantiate PAVE in Voville, a tile-based traffic environment forked from Smallville, and evaluate across three scenarios, four LLM backbones, and a focused ablation. PAVE agents satisfy four properties simultaneously: legitimate violation (only when a trigger justifies it), authority deference (officer instructions override even high legitimacy), bounded scope (violations confined to the targeted rule), and recovery (baseline restored once the trigger ends). PAVE agents make more structured and interpretable decisions than vanilla across all four properties, and human evaluators rate them as more plausible. Ablating the legitimacy gate reproduces vanilla-like failures. We release Voville, the PAVE prompts and code, and the evaluation pipeline. 

---
# STAR-PólyaMath: Multi-Agent Reasoning under Persistent Meta-Strategic Supervision 

**Authors**: Jiaao Wu, Xian Zhang, Hanzhang Liu, Sophia Zhang, Fan Yang, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.19338)  

**Abstract**: Frontier AI models and multi-agent systems have led to significant improvements in mathematical reasoning. However, for problems requiring extended, long-horizon reasoning, existing systems continue to suffer from fundamental reliability issues: hallucination accumulation, memory fragmentation, and imbalanced reasoning-tool trade-offs. In this paper, we introduce STAR-PólyaMath, a multi-agent framework that systematically addresses these challenges through meta-level supervision and structured Reasoner-Verifier interaction. STAR-PólyaMath is structured as an orchestrated state machine with nested challenge-step-replan loops, governed by a reasoning-free Python orchestrator that separates control from inference and bounds error propagation through trace-back and re-planning. Our key innovation is a persistent Meta-Strategist that maintains cross-attempt memory and exercises meta-level control by issuing high-level strategic guidance or mandatory directives, so the system can escape unproductive loops rather than stagnate or over-rely on tools. STAR-PólyaMath achieves state-of-the-art results on all eight top-tier competition benchmarks: AIME 2025-2026, MathArena Apex Shortlist, MathArena Apex 2025, Putnam 2025, IMO 2025, HMMT February 2026, and USAMO 2026. It obtains perfect scores on AIMEs, Putnam, and HMMT, and shows its largest margin on Apex 2025, scoring 93.75% compared with 80.21% by the strongest baseline GPT-5.5. Ablation studies show that the gains arise from the framework's orchestration rather than from model-level diversity since removing key components or substituting in mixed backbones consistently weakens performance. Code is available at this https URL. 

---
# GRASP: Deterministic argument ranking in interaction graphs 

**Authors**: Diganta Misra, Antonio Orvieto, Rediet Abebe, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2605.19141)  

**Abstract**: Large language models are increasingly deployed as automated judges to evaluate the strength of arguments. As this role expands, their legitimacy depends on consistency, transparency, and the ability to separate argumentative structure from rhetorical appeal. However, we show that holistic judging - a common LLM-as-a-Judge practice where a model provides a global verdict on a debate - suffers from substantial inter-model disagreement. We argue that this instability arises from collapsing a debate's complex interaction structure into a single opaque score. To address this, we propose GRASP (Gradual Ranking with Attacks and Support Propagation), a deterministic framework that aggregates stable local interaction judgments into a global ranking via a convergent attack--defense propagation operator. We show that local interaction judgments are more reproducible than holistic rankings in LLM-as-a-Judge evaluations, allowing GRASP to produce more consistent global rankings. We further show that GRASP scores do not correlate with human "convincingness" labels, highlighting a vital sociotechnical distinction: GRASP does not measure persuasion, factuality, or rhetorical appeal, but structural sufficiency - a defense-aware notion of argument robustness over the explicit interaction graph. Overall, GRASP offers a transparent and auditable alternative to holistic LLM judging. 

---
# EgoBabyVLM: Benchmarking Cross-Modal Learning from Naturalistic Egocentric Video Data 

**Authors**: Dongyan Lin, Phillip Rust, Angel Villar Corrales, Alvin W. M. Tan, Mahi Luthra, Charles-Éric Saint-James, Rashel Moritz, Sheila Krogh-Jespersen, Vanessa Stark, Surya Parimi, Jiayi Shen, Youssef Benchekroun, Yosuke Higuchi, Martin Gleize, Tom Fizycki, Nicolas Hamilakis, Manel Khentout, Sho Tsuji, Balázs Kégl, Juan Pino, Michael C. Frank, Emmanuel Dupoux  

**Link**: [PDF](https://arxiv.org/pdf/2605.19130)  

**Abstract**: Children acquire language grounding with remarkable robustness from limited visuo-linguistic input in ways that surpass today's best large multimodal models. Recent research suggests current vision-language models (VLMs) trained on curated web data fail to generalize to the sparse, weakly-aligned egocentric streams produced by wearable devices, embodied agents, and infant head-cams -- and no fixed evaluation pipeline exists for measuring progress on this regime. We train VLMs on datasets with varying degrees of semantic alignment between visual and linguistic inputs, including naturalistic infant and adult egocentric videos, and evaluate them with a comprehensive suite spanning multimodal language grounding and unimodal vision and language tasks. At the core of this suite is Machine-DevBench, a corpus-grounded benchmark of lexical and grammatical competence, automatically generated from the model's training vocabulary across logarithmic frequency bins to eliminate the train/eval mismatch and low statistical power of prior developmental benchmarks. Our results show that current VLM paradigms hinge on the tight semantic alignment of curated data and fail to exploit the weakly-aligned signal that dominates naturalistic egocentric input -- the very regime in which humans thrive. To motivate progress, we introduce the EgoBabyVLM Challenge to drive the development of models capable of grounded language learning from the kind of naturalistic data that human infants experience. 

---
# DecisionBench: A Benchmark for Emergent Delegation in Long-Horizon Agentic Workflows 

**Authors**: Yuxuan Gao, Megan Wang, Yi Ling Yu, Zijian Carl Ma, Ao Qu  

**Link**: [PDF](https://arxiv.org/pdf/2605.19099)  

**Abstract**: We introduce DecisionBench, a benchmark substrate for emergent delegation in long-horizon agentic workflows. The substrate fixes a task suite (GAIA, tau-bench, BFCL multi-turn), a peer-model pool (11 models, 7 vendor families), a delegation interface (call_model plus an optional read_profile channel), a deterministic skill-annotation layer, and a multi-axis metric suite covering quality, cost, latency, delegation rate, routing fidelity-at-k, vendor self-preference, and a counterfactual-delegation ceiling. The substrate is agnostic to how peer information is generated or delivered, so learned routers, richer peer memories, adaptive profile construction, and multi-step delegation can all be evaluated against it. We characterize the substrate with a five-condition reference sweep on the full pool (n=23,375 task instances). Three benchmark-level findings emerge: (i) mean end-task quality is statistically indistinguishable across the four awareness conditions (|beta| <= 0.010, p >= 0.21), so quality-only evaluation would miss the orchestration signal; (ii) routing fidelity-at-1 ranges from 7.5% to 29.5% across conditions at near-equal mean quality, with delivery channel (on-demand tool vs. preloaded description) dominating description content; (iii) a counterfactual ceiling places perfect delegation 15-31 percentage points above measured performance on every suite, locating large unrealized headroom for future orchestration methods. We release the substrate, annotation layer, reference intervention suite, analysis pipeline, and 220 per-condition run archives. 

---
# Counterfactual Likelihood Tests for Indirect Influence in Private Reasoning Channels 

**Authors**: Alexander Boesgaard Lorup  

**Link**: [PDF](https://arxiv.org/pdf/2605.19092)  

**Abstract**: Reasoning systems increasingly separate intermediate computation into private and public channels, creating evaluation cases that look similar in transcripts: independent co-derivation, direct access to private content, and indirect influence through public communication. This paper presents a counterfactual likelihood test for measuring influence between private reasoning channels. The method replaces an upstream private block with a length-matched donor block, holds the public token sequence and downstream target fixed, and measures the downstream target's negative-log-likelihood shift. On a 7B role-channel reasoning model used for validation, textual probes are unreliable: raw n-gram overlap overstates leakage, corrected overlap remains noisy, and canary reproduction reports no discrimination. Counterfactual likelihood separates unmasked and masked conditions, while length matching controls a RoPE positional confound. In the hardened masked validation, reverse B-to-A influence is near zero, while A-to-B influence persists through public-speech hidden states. A multi-checkpoint validation across three checkpoints, five seeds, and 13,734 valid directional contrasts replicates this asymmetry. A graph-separation control that blocks private-to-public carrier edges produces bit-identical natural and counterfactual scores across all 13,734 control evaluations, identifying the tested public-channel pathway as the complete carrier of the measured counterfactual signal under the implemented role-visibility mask. The results show that private-channel evaluation should report direct and indirect influence separately, and that counterfactual likelihood probes provide a practical default for measuring these boundaries. 

---
# Learn-by-Wire Training Control Governance: Bounded Autonomous Training Under Stress for Stability and Efficiency 

**Authors**: Anis Radianis  

**Link**: [PDF](https://arxiv.org/pdf/2605.19008)  

**Abstract**: Modern language-model training is increasingly exposed to instability, degraded runs, and wasted compute, especially under aggressive learning-rate, scale, and runtime-stress conditions. This paper introduces Learn-by-Wire Guard (LBW-Guard), a bounded autonomous training-control governance layer that operates above AdamW. Rather than replacing the optimizer update rule, LBW-Guard observes training telemetry, interprets instability-sensitive regimes, and applies bounded control to optimizer execution while preserving fixed training objectives.
We evaluate LBW-Guard in a Qwen2.5-centered stress-and-robustness suite using WikiText-103, with Qwen2.5-7B as the empirical anchor, model-size comparisons against Qwen2.5-3B and Qwen2.5-14B, learning-rate stress tests, gradient-clipping baselines, and a no-LoRA TinyLlama-1B full-parameter sanity check. In the 7B reference setting, LBW-Guard reduces final perplexity from 13.21 to 10.74, an 18.7% improvement, while reducing end-to-end time from 392.54s to 357.02s, a 1.10x speedup. Under stronger learning-rate stress, AdamW degrades to 1885.24 final perplexity at LR=3e-3 and 659.76 at LR=1e-3, whereas LBW-Guard remains trainable at 11.57 and 10.33, respectively. Gradient-clipping baselines do not reproduce this effect.
These results support a scoped systems conclusion that stability-sensitive LLM training can benefit from a governance plane above the optimizer. LBW-Guard provides evidence that bounded runtime control can preserve productive compute under stress while remaining distinct from optimizer replacement and local gradient suppression. 

---
# FedMental: Evaluating Federated Learning for Mental Health Detection from Social Media Data 

**Authors**: Nuredin Ali Abdelkadir, Anjali Ratnam, Zeerak Talat, Stevie Chancellor  

**Link**: [PDF](https://arxiv.org/pdf/2605.18936)  

**Abstract**: Social media text data are often used to train Machine Learning (ML) models to identify users exhibiting high-risk mental health behaviors. However, sharing this sensitive data poses privacy risks and limits the growth of benchmark datasets. We comprehensively evaluate whether privacy-preserving ML techniques can enable safer data sharing while preserving performance. Specifically, we apply federated learning (FL) and Differentially Private FL for two widely-studied mental health prediction tasks: depression detection on X (Twitter) and suicide crisis detection on Reddit. We simulate realistic data-sharing scenarios by treating each user as a client in a non-IID setting, evaluating across different client fractions, aggregation strategies, and privacy budgets. While FL achieves comparable performance to centralized training (centralized F1 = 85.63; best FL model F1 = 83.16) on depression identification, we find that Differentially Private FL has a large performance-privacy trade-off (up to F1 = 27.01 drop) even with low levels of noise (epsilon = 50). This is due to the distortion of highly informative yet sparse mental health linguistic markers related to mental health, like health topics and emotion words. This research empirically demonstrates the potential and limitations of current privacy preservation techniques for mental health inference tasks. 

---
# Dynamic Model Merging Made Slim 

**Authors**: Guodong Du, Wanyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.18904)  

**Abstract**: Model merging enables the reuse of fine-tuned models without joint training or access to original data. Dynamic merging further improves flexibility by selectively activating task-relevant parameters and efficiently composing experts across multiple tasks. However, existing dynamic methods either maintain a full shared model with tiny experts or allocate excessive capacity to experts, leading to suboptimal accuracy--efficiency trade-offs. To address this, we propose DiDi-Merging, a slim dynamic merging framework that leverages differentiable rank allocation to balance shared and expert parameters. By formulating parameter budgeting as differentiable rank optimization in low-rank modules and introducing a data-free refinement step to recover task fidelity, DiDi-Merging matches prior dynamic baselines at only 1.24x the parameters of a single fine-tuned model and surpasses them at 1.4x, substantially more compact than methods requiring > 2x storage. DiDi-Merging applies across vision, language, and multimodal tasks. 

---
# ZeroUnlearn: Few-Shot Knowledge Unlearning in Large Language Models 

**Authors**: Yujie Lin, Chengyi Yang, Zhishang Xiang, Yiping Song, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.18879)  

**Abstract**: Large language models inevitably retain sensitive information, defined as inputs that may induce harmful generations, due to training on massive web corpora, raising concerns for privacy and safety. Existing machine unlearning methods primarily rely on retraining or aggressive fine-tuning, which are either computationally expensive or prone to degrading related knowledge and overall model utility. In this work, we reformulate machine unlearning as a precise knowledge re-mapping problem via model editing. We propose ZeroUnlearn, a few-shot unlearning framework. It overwrites sensitive inputs by mapping them to a neutral target state and removing their original representations. ZeroUnlearn enforces representational orthogonality through a multiplicative parameter update with a closed-form solution, enabling efficient and targeted unlearning. We further extend ZeroUnlearn to a gradient-based variant for multi-sample unlearning. Experiments demonstrate that our approach outperforms existing baselines while preserving general model utility. Our code is available at the github: this https URL. 

---
# SAGE: Shaping Anchors for Guided Exploration in RLVR of LLMs 

**Authors**: Chanuk Lee, Minki Kang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2605.18864)  

**Abstract**: Recent studies observe that reinforcement learning with verifiable rewards (RLVR) reliably improves pass@1 on reasoning tasks, yet often fails to yield comparable gains in pass@k, raising the question of whether RLVR genuinely enables large language models to acquire novel reasoning abilities or merely enhances the efficiency of sampling reasoning modes already present in the base model. Prior analyses largely support the latter view, attributing this limitation to structural properties of standard RLVR objectives that result in insufficient exploration pressure. In this work, we argue that a central structural constraint arises from reverse-KL regularization, which stabilizes training but inherently anchors the policy to the reference distribution, thereby suppressing the emergence of alternative reasoning modes. However, we show that neither removing the KL term nor replacing it with forward-KL provides a satisfactory solution, as both disrupt the efficiency-coverage trade-off by either inducing reward hacking or allocating probability mass to off-target regions. To resolve this tension, we propose SAGE, a principled framework that enables controllable empirical support expansion by reshaping the reverse-KL anchor distribution itself through a guide function q(x,y), achieving consistent improvements in both pass@1 and pass@k across challenging mathematical reasoning benchmarks. Our code is available at this https URL. 

---
# SPHERICAL KV: Angle-Domain Attention and Rate-Distortion Retention for Efficient Long-Context Inference 

**Authors**: Anay Chauhan, Gurucharan Marthi Krishna Kumar, Arion Das, Amit Dhanda, Vinija Jain, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2605.18856)  

**Abstract**: Long-context inference is increasingly constrained by the KV cache: resident memory grows with context length, and decoding becomes limited by repeated High Bandwidth Memory (HBM) streaming rather than arithmetic. Existing methods such as eviction, windowing, quantization, and offloading reduce footprint, but often leave the critical-path bottleneck only partially addressed, especially when compressed states must still be reconstructed into dense vectors during decoding.
We present Spherical KV, a long-context inference method that treats KV allocation as a rate-distortion problem grounded in attention geometry for efficient decoding. The method is built on two ideas: (i) represent directional information cheaply in the decode hot loop, and (ii) allocate retention and precision according to estimated future utility. Its first component, Angle-Domain Attention (ADA), stores keys in a spherical parameterization consisting of a scalar radius and compact angle codes, and computes attention logits directly from these codes without reconstructing dense keys. This preserves a paged, block-local, fusion-friendly decode path and directly targets HBM traffic in realistic serving settings. Its second component, Rate-Distortion Retention (RDR), jointly chooses keep/drop decisions and precision tiers per token and head under a fixed budget, producing tier-homogeneous pages with lightweight metadata and coalesced reads. Together, ADA and RDR provide a deployment-oriented mechanism for reducing KV residency while preserving decode efficiency. 

---
# Robust Checkpoint Selection for Multimodal LLMs via Agentic Evaluation and Stability-Aware Ranking 

**Authors**: Qinwu Xu, Zhuoheng Li, Jessie Salas  

**Link**: [PDF](https://arxiv.org/pdf/2605.18852)  

**Abstract**: Checkpoint selection for multimodal large language models (MLLMs) presents significant challenges when performance differentials are marginal and evaluation signals are prone to noise. Existing methodologies rely heavily on static benchmarks or pointwise scoring, which frequently misalign with in-the-wild usage and lack robust uncertainty estimation, particularly in OCR-heavy scenarios. In this work, we formulate checkpoint selection as a robust decision problem under evaluation uncertainty. We propose a multi-stage framework that integrates curated real-world data, structured LLM-based judgment, and multi-stage ranking protocols. The evaluation system orchestrates progressive refinement via pointwise filtering, listwise ranking, and pairwise comparison. To enhance reliability, we introduce subsampling-based confidence estimation and a percentile-based scoring formulation that captures distributional characteristics while penalizing tail failures. Furthermore, we demonstrate that data quality, specifically OCR readability, is a critical determinant of evaluation validity. 

---
# The Growing Pains of Frontier Models: When Leaderboards Stop Separating and What to Measure Next 

**Authors**: Adil Amin  

**Link**: [PDF](https://arxiv.org/pdf/2605.18840)  

**Abstract**: Leaderboards rank frontier models on independent axes but do not reveal whether capabilities reinforce or trade off across releases -- and at the frontier, this interaction is the more informative signal. We decompose paired SWE-bench and GPQA Diamond scores into a population coupling trend and per-release residual ($h$-field) that diagnoses capability emphasis and identifies which measurement or stress test is most informative next. Across 34 models from 10 labs (2024--2026), capabilities cooperate ($r = +0.72$, $p < 10^{-6}$), but cooperation varies by lab and over time: DeepSeek reversed from reasoning-rich to coding-first ($h$: $+11.2 \to -4.7$, 15.9-pp swing); Google maintains consistent reasoning emphasis; Anthropic oscillates between coding excursions and recovery. Cooperation is not static -- it cascades. Six open-weight architectures confirm a second capability transition at 30--72B, and SWE-bench is now saturating while HLE and instruction-following retain discriminatory spread -- signaling the next axis rotation. We provide a three-level playbook (locate, diagnose, rotate), a per-lab measurement-priority table, and seven falsifiable predictions with timestamped criteria for the next 12 months of frontier releases. Per-lab coupling slopes vary $5\times$ (Google $1.15$ vs. DeepSeek $0.23$), quantifying how efficiently each recipe converts coding gains into reasoning. Five April 2026 releases confirm the diagnostic out of sample ($r$ rises from $+0.72$ to $+0.75$). An interactive dashboard provides phase classification with actionable recommendations, $h$-field diagnostics, per-lab coupling trajectories, ODE-based scaling predictions, benchmark rotation guidance, self-steering demo, and live tracking of all seven predictions: this https URL. 

---
# Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling 

**Authors**: Adil Amin  

**Link**: [PDF](https://arxiv.org/pdf/2605.18838)  

**Abstract**: Scaling laws predict loss from compute but not how capabilities interact. We measure the coupling between reasoning and truthfulness across 63 base models from 16 families and find a regime change invisible to loss curves: below a family-dependent critical scale $N_c$, capabilities anticorrelate; above it, they cooperate. $N_c \approx 3.5$B parameters [2.9B, 13.4B] (bootstrap 95% CI), but model size is not the only variable that determines phase. Architecture, data curation, and training recipe each shift $N_c$ independently: curated training eliminated the coupling dip between Qwen generations ($0.025 \to 0.830$ at matched scale), Gemma-4 at 4B achieves coupling 0.871, characteristic of 13B+ standard-trained models, through distillation and architectural innovation, and Phi at 1B matches web-trained coupling at 10B through data curation alone. Width normalization eliminates the anticorrelation across all tested families, supporting an output-projection bottleneck. Internally, 38 of 40 models show zero competing attention heads. A sparse-regression ODE cross-predicts held-out Llama-2 at 5.6% error. The diagnostic requires no model internals -- only public benchmark scores across a model family. The cooperative regime extends to the frontier ($r = +0.72$, 34 models, 10 labs). Code, data, and an open-source activation-steering tool for any open-weight model are released alongside an interactive dashboard that diagnoses any model's coupling phase, suggests concrete interventions (data curation, width, benchmark rotation), and provides ODE scaling predictions, frontier diagnostics, and eigenstructure analysis: this https URL. 

---
# Fine-Grained Benchmark Generation for Comprehensive Evaluation of Foundation Models 

**Authors**: Mohammed Saidul Islam, Negin Baghbanzadeh, Farnaz Kohankhaki, Afshin Cheraghi, Ali Kore, Shayaan Mehdi, Elham Dolatabadi, Arash Afkanpour  

**Link**: [PDF](https://arxiv.org/pdf/2605.18824)  

**Abstract**: Evaluation of foundation models often rely on aggregate scores from benchmarks that lack comprehensive coverage and metadata for a fine-grained evaluation. We introduce a framework for automated benchmark generation. Our framework generates evaluation problems grounded in reference material, such as textbooks, producing benchmarks with broad coverage, rich metadata, and robustness to contamination. The pipeline employs a multi-agent architecture for problem generation and a solution-graph-driven strategy that significantly improves the reliability of ground truth solutions. Using the framework, we generate three benchmarks in Machine Learning, Corporate Finance, and Personal Finance. Expert review finds a significantly lower ground-truth error rate than previous benchmarks such as MMLU and GSM8K. Evaluation of 12 commercial and open-source models shows that our benchmarks achieve near-uniform competency coverage and surface performance differences across models that existing benchmarks fail to capture. We will open-source the framework and our curated benchmarks soon. 

---
# PASC: Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees for Multi-Stage NLP and LLM Pipelines 

**Authors**: Varun Kotte  

**Link**: [PDF](https://arxiv.org/pdf/2605.18812)  

**Abstract**: Modern NLP and LLM systems are pipelines: named entity recognition (NER) -> entity disambiguation (NED) -> entity typing, retrieval-augmented generation (retriever -> reader), and agentic chains of planner -> tool -> critic. Errors compound across stages, but existing uncertainty quantification methods either calibrate each stage independently (no joint coverage) or apply a Bonferroni union bound (joint coverage, but conservative). We present PASC (Pipeline-Aware Split Conformal), which reduces multi-stage joint coverage to a single scalar conformal prediction problem on the joint maximum nonconformity score. PASC provides a finite-sample distribution-free guarantee that all K stages are simultaneously covered with probability at least 1 - alpha, and is nearly tight up to a 1/(n+1) factor. On a three-stage NER -> NED -> entity-typing pipeline over CoNLL-2003, PASC achieves 96.4% end-to-end coverage versus 93.4% for Bonferroni and 86.5% for independent CP, at identical average prediction set size (1.083). Under distribution shift to WNUT-17 Twitter and WikiNEuRal Wikipedia data, PASC empirically maintains the target coverage in the tested shift settings while independent CP collapses to 59%. PASC requires a single quantile computation, runs 1.7x faster than Bonferroni, and scales to K = 6 stages where independent CP drops to 0.53 end-to-end coverage. The same joint-maximum-score reduction applies directly to compound LLM systems and agent pipelines. 

---
# Compositional Literary Primitives in Instruction-Tuned LLMs: Cross-Architectural SAE Features for Self, Style, and Affect 

**Authors**: Joao Paulo Cavalcante Presa, Savio Salvarino Teles de Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2605.18808)  

**Abstract**: We characterize a compositional architecture of literary primitives in two instruction-tuned large language models (Llama 3.1 8B-Instruct and Gemma 2 9B-IT) via sparse autoencoders on mid-depth residual streams. Four feature classes emerge: naming-gates that promote lexical tokens of a target affect, an eleven-self cluster of first-person register features, stylistic register modulators (show-don't-tell and defamiliarization), and compositional emotions that arise only from multi-feature steering. Under a forced-choice 5-LLM judge panel applied to a 27-category emotion taxonomy (Cowen-Keltner), Llama reaches full 27/27 coverage by combining naming-gates, multi-feature recipes, and single self-feature steering; Gemma reaches 23/27 with adoration as the single residual strict-fail. Under random judging, the per-cell pass probability is on the order of $10^{-3}$ and the expected number of two-seed false-positive cells across the catalog is negligible, so the observed coverage is not consistent with chance. A cross-architectural asymmetry sits in the strict-versus-soft judge contrast: on the same generations, judges agree more often on Llama outputs than on Gemma outputs because Llama outputs name the target affect more directly while Gemma outputs evoke it through scene and imagery. Both architectures contain self-features that serve simultaneously as register markers and as emotion emitters, including a single most-RLHF-loaded self-feature per architecture that intensifies the institutional Helper-AI persona at one operating regime and produces affect-categorizable output at the same calibrated coefficient. Methodologically, the paper presents a three-stage validation pipeline (logit-lens, LLM-rate, 5-LLM judge) with documented anti-patterns; the total compute is single-GPU and about 15 minutes per emotion-feature discovery cycle. 

---
# ReCrit: Transition-Aware Reinforcement Learning for Scientific Critic Reasoning 

**Authors**: Wanghan Xu, Yuhao Zhou, Hengyuan Zhao, Shuo Li, Dianzhi Yu, Zhenfei Yin, Yaowen Hu, Fengli Xu, Wanli Ouyang, Wenlong Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2605.18799)  

**Abstract**: Large language models can fail in critic interaction not only by answering incorrectly, but also by abandoning an initially correct scientific solution after user criticism. This is especially risky in scientific reasoning, where user criticism can turn a valid answer into an incorrect one. We frame critic interaction as an inter-turn correctness-transition problem rather than a final-answer accuracy problem, and identify three challenges: transition awareness, decoupling useful correction from harmful sycophancy, and scalable rollout. We propose ReCrit, a transition-aware reinforcement learning framework that decomposes Initial-to-Critic behavior into four quadrants: Correction, Sycophancy, Robustness, and Boundary. ReCrit rewards correction and robustness, penalizes sycophancy, and treats persistent errors as weak boundary signals. To make interaction training practical, ReCrit further uses dynamic asynchronous rollout with tail-adaptive completion to reduce rollout waiting. On three scientific reasoning benchmarks, ChemBench, TRQA, and EarthSE, ReCrit improves average Critic accuracy from 38.15 to 51.49 on Qwen3.5-4B and from 45.40 to 55.59 on Qwen3.5-9B. Ablations show that final-answer rewards provide little interaction-level gain, while transition-aware rewards and quadrant weighting produce more distinguishable training signals and larger net Critic-stage improvement. The code is available at this https URL . 

---
# UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing 

**Authors**: Varun Kotte  

**Link**: [PDF](https://arxiv.org/pdf/2605.18796)  

**Abstract**: LLM cascades and model routing promise lower inference cost by sending easy queries to a small model and escalating hard ones to a large model, but most deployed routers use uncalibrated confidence scores and require per-workload threshold tuning. We present UCCI, a calibration-first router that maps token-level margin uncertainty to a per-query error probability via isotonic regression and selects the escalation threshold by constrained cost minimization. Under three explicit assumptions, threshold policies on the calibrated score are cost-optimal, and isotonic calibration achieves O(n^{-1/3}) sample complexity for expected calibration error (ECE). On a production named entity recognition workload of 75,000 queries served by 4B and 12B instruction-tuned LLMs on H100 GPUs, UCCI cuts inference cost by 31% (95% CI: [27%, 35%]) at micro-F1 = 0.91 while reducing ECE from 0.12 to 0.03. At the same operating point, UCCI beats entropy thresholding, split-conformal routing, and a FrugalGPT-style learned threshold. All cascade results use end-to-end routing on actual model outputs and measured H100 latency, not simulated routing from global accuracies or nominal API prices. 

---
# Trust or Abstain? A Self-Aware RAG Approach 

**Authors**: Xi Zhu, Ziqi Wang, Kai Mei, Wujiang Xu, Minghao Guo, Bangji Yang, Jiajun Fan, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2605.18792)  

**Abstract**: Retrieval-augmented generation (RAG) improves large language models (LLMs) by incorporating external evidence, but it also introduces knowledge conflicts when retrieved contextual knowledge (CK) and parametric knowledge (PK) disagree or are both unreliable. Existing approaches mainly coordinate which source to use, without explicitly asking whether each answer path is correct. We argue that faithful RAG requires LLM self-awareness, namely the ability to recognize the limits of its own knowledge and reasoning. To ground this problem, we construct a model-specific, ground-truth-aligned knowledge-conflict benchmark by evaluating LLM backbones on PK-only and CK-conditioned answer paths over approximately 69K query-context instances per backbone, drawn from five conflict-QA datasets. We then introduce SABER, a Self-Aware Belief Estimator for RAG that requires no LLM fine-tuning. SABER combines a self-prior with PK-side and CK-side conditional reasoning representations from multi-trace inference, then estimates reliability beliefs with two lightweight predictors to drive a 4-cell decision over trust PK, trust CK, trust either, or abstain. Across four LLM backbones, SABER improves end-to-end accuracy and conflict-specific faithfulness over ten inference-time and fine-tuning baselines, with the largest gains on conflict-heavy datasets. Under abstention, SABER's risk-coverage curve Pareto-dominates every prompt-based abstainer, providing a tunable balance between coverage and answer risk. Our code is available at this https URL. 

---
# Improving Retrieval-Augmented Generation without Taxonomy-based Error Categorization 

**Authors**: Gongbo Zhang, Yifan Peng, Chunhua Weng  

**Link**: [PDF](https://arxiv.org/pdf/2605.18772)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves the factual accuracy of large language model (LLM) outputs by grounding generation in external knowledge. Recent agentic RAG systems extend this paradigm with critical agents to evaluate model responses and iteratively refine outputs. However, most prior work implicitly assumes reliable critic feedback and focuses on planning strategies, while paying limited attention to the robustness of the error-correction process itself, which can be impacted by misaligned error categories and ineffective or incorrect corrections. Here, we hypothesize that RAG performance can be improved without explicit error categorization. We propose RePAIR, a response-action learning paradigm that directly maps flawed RAG outputs to error-mitigating action plans without relying on fine-grained error taxonomies and explicit critic supervision. Across multiple benchmarks, RePAIR consistently improves agentic RAG performance. 

---
# ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation 

**Authors**: Gibson Nkhata, Uttamasha Anjally Oyshi, Quan Mai, Susan Gauch  

**Link**: [PDF](https://arxiv.org/pdf/2605.18769)  

**Abstract**: Personalized Retrieval-Augmented Generation (RAG) relies on accurately selecting user-relevant documents. In practice, existing RAG approaches often suffer from high retrieval costs and overlook that collaborative signals from similar users can enhance personalized generation for the current user. We propose ClusterRAG, a Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation. ClusterRAG represents users through their profile documents, organizes users into semantically coherent clusters using density-based clustering, and performs retrieval at both the cluster and document levels via cluster-level similarity and fine-grained ranking. Extensive experiments on the LaMP benchmark demonstrate that jointly leveraging the target user's profile and profiles from top similar users consistently yields the best performance across diverse tasks. Further analysis shows that ClusterRAG integrates seamlessly with different dense retrievers and rankers, and remains effective when paired with both fine-tuned and zero-shot language models. 

---
# Retrieve Only Relevant Tables Whether Few or Many: Adaptive Table Retrieval Method 

**Authors**: Taehee Kim, Seungbin Yang, Jihwan Kim, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2605.18766)  

**Abstract**: Retrieving relevant tables from extensive databases for a given natural language query is essential for accurately answering questions in tasks such as text-to-SQL. Existing table retrieval approaches select a pre-determined set of k tables with the highest similarity to the query. However, the number of required tables varies across queries and cannot be known in advance. Enforcing a fixed number of retrieved tables regardless of the query may either retrieve an undersized set, failing to obtain all necessary evidence, or retrieve an oversized pool, including irrelevant tables. To address this issue, we propose an adaptive table retrieval method that adjusts the number of tables retrieved according to the requirements of each query. Specifically, we utilize an adaptive thresholding mechanism to selectively retrieve tables and integrate a sliding-window reranking algorithm to efficiently process a large table corpus. Extensive experiments on Spider, BIRD, and Spider 2.0 demonstrate that our method effectively addresses the limitations of the top-k retrieval strategy, improving performance in retrieval and downstream tasks. Our code and data are available at this https URL. 

---
