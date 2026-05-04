# When LLMs Stop Following Steps: A Diagnostic Study of Procedural Execution in Language Models 

**Authors**: Sailesh Panda, Pritam Kadasi, Abhishek Upperwal, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2605.00817)  

**Abstract**: Large language models (LLMs) often achieve strong performance on reasoning benchmarks, but final-answer accuracy alone does not show whether they faithfully execute the procedure specified in a prompt. We study this question through a controlled diagnostic benchmark for procedural execution, where models are given a step-wise arithmetic algorithm and two numeric inputs, and must return the final computed value. The benchmark uses simple arithmetic operations but increases complexity through algorithm length and look-back dependencies over intermediate variables. Across 14 models and 55 datasets, average first-answer accuracy drops from 61% on 5-step procedures to 20% on 95-step procedures. Generation-level analysis shows that failures often involve missing answers, premature answers, self-correction after an initial error, under-executed traces, and hallucinated extra steps. These findings suggest that apparent reasoning ability can mask substantial weaknesses in faithful instruction execution. 

---
# Directed Social Regard: Surfacing Targeted Advocacy, Opposition, Aid, Harms, and Victimization in Online Media 

**Authors**: Scott Friedman, Ruta Wheelock, Sonja Schmer-Galunder, Drisana Iverson, Jake Vasilakes, Joan Zheng, Jeffrey Rye, Vasanth Sarathy, Christopher Miller  

**Link**: [PDF](https://arxiv.org/pdf/2605.00776)  

**Abstract**: The language in online platforms, influence operations, and political rhetoric frequently directs a mix of pro-social sentiment (e.g., advocacy, helpfulness, compassion) and anti-social sentiment (e.g., threats, opposition, blame) at different topics, all in the same message. While many natural language processing (NLP) tools classify or score a text's overall sentiment as positive, neutral, or negative, these tools cannot report that positive and negative sentiments coexist, and they cannot report the target of those sentiments. This paper presents the Directed Social Regard (DSR) approach to multi-dimensional, multi-valence sentiment analysis, comprised of a pair of transformer-based models that (1) detects span-level targets of sentiment in a message and then (2) scores all spans within the message context along three (-1, 1) axes of regard that are motivated by social science theories of moral disengagement and moral framing. We present a data collection and annotation strategy for DSR dataset construction, a transformer-based architecture for span-level scoring, and a validation study with promising results. We apply the validated DSR model on six third-party datasets of online media and report meaningful correlations between DSR outputs and the labels and topics in these pre-existing social science datasets. 

---
# Characterizing the Expressivity of Local Attention in Transformers 

**Authors**: Jiaoda Li, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2605.00768)  

**Abstract**: The transformer is the most popular neural architecture for language modeling. The cornerstone of the transformer is its global attention mechanism, which lets the model aggregate information from all preceding tokens before generating the next token. One common variant of attention is called local attention, which restricts each token to aggregating information from a bounded window of predecessors, reducing the quadratic cost of global attention to linear. Although this restriction is usually motivated by efficiency, it has also been found to improve model quality, a phenomenon that has so far lacked a satisfactory explanation. We provide a formal account of this phenomenon in terms of recognizer expressivity. It has been shown that fixed-precision transformers with global attention correspond to a fragment of linear temporal logic containing a single past operator. We additionally prove that adding local attention introduces a second temporal operator, strictly enlarging the class of recognizable regular languages. Moreover, global and local attention are expressively complementary: neither subsumes the other, and combining them yields the richest fragment. Experiments on formal language recognition and natural language modeling corroborate the theory, showing that hybrid global--local transformers outperform their global-only counterparts. 

---
# FinSafetyBench: Evaluating LLM Safety in Real-World Financial Scenarios 

**Authors**: Yutao Hou, Yihan Jiang, Yuhan Xie, Jian Yang, Liwen Zhang, Hailiang Huang, Guanhua Chen, Yun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00706)  

**Abstract**: Large language models (LLMs) are increasingly applied in financial scenarios. However, they may produce harmful outputs, including facilitating illegal activities or unethical behavior, posing serious compliance risks. To systematically evaluate LLM safety in finance, we propose FinSafetyBench, a bilingual (English-Chinese) red-teaming benchmark designed to test an LLM's refusal of requests that violate financial compliance. Grounded in real-world financial crime cases and ethics standards, the benchmark comprises 14 subcategories spanning financial crimes and ethical violations. Through extensive experiments on general-purpose and finance-specialized LLMs under three representative attack settings, we identify critical vulnerabilities that allow adversarial prompts to bypass compliance safeguards. Further analysis reveals stronger susceptibility in Chinese contexts and highlights the limitations of prompt-level defenses against sophisticated or implicit manipulation strategies. 

---
# Learning How and What to Memorize: Cognition-Inspired Two-Stage Optimization for Evolving Memory 

**Authors**: Derong Xu, Shuochen Liu, Pengfei Luo, Pengyue Jia, Yingyi Zhang, Yi Wen, Yimin Deng, Wenlin Zhang, Enhong Chen, Xiangyu Zhao, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00702)  

**Abstract**: Large language model (LLM) agents require long-term user memory for consistent personalization, but limited context windows hinder tracking evolving preferences over long interactions. Existing memory systems mainly rely on static, hand-crafted update rules; although reinforcement learning (RL)-based agents learn memory updates, sparse outcome rewards provide weak supervision, resulting in unstable long-horizon optimization. Drawing on memory schema theory and the functional division between prefrontal regions and hippocampus regions, we introduce MemCoE, a cognition-inspired two-stage optimization framework that learns how memory should be organized and what information to update. In the first stage, we propose Memory Guideline Induction to optimize a global guideline via contrastive feedback interpreted as textual gradients; in the second stage, Guideline-Aligned Memory Policy Optimization uses the induced guideline to define structured process rewards and performs multi-turn RL to learn a guideline-following memory evolution policy. We evaluate on three personalization memory benchmarks, covering explicit/implicit preference and different sizes and noise, and observe consistent improvements over strong baselines with favorable robustness, transferability, and efficiency. 

---
# ML-Bench&Guard: Policy-Grounded Multilingual Safety Benchmark and Guardrail for Large Language Models 

**Authors**: Yunhan Zhao, Zhaorun Chen, Xingjun Ma, Yu-Gang Jiang, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00689)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in cross-linguistic contexts, ensuring safety in diverse regulatory and cultural environments has become a critical challenge. However, existing multilingual benchmarks largely rely on general risk taxonomies and machine translation, which confines guardrail models to these predefined categories and hinders their ability to align with region-specific regulations and cultural nuances. To bridge these gaps, we introduce ML-Bench, a policy-grounded multilingual safety benchmark covering 14 languages. ML-Bench is constructed directly from regional regulations, where risk categories and fine-grained rules derived from jurisdiction-specific legal texts are directly used to guide the generation of multilingual safety data, enabling culturally and legally aligned evaluation across languages. Building on ML-Bench, we develop ML-Guard, a Diffusion Large Language Model (dLLM)-based guardrail model that supports multilingual safety judgment and policy-conditioned compliance assessment. ML-Guard has two variants, one 1.5B lightweight model for fast `safe/unsafe' checking and a more capable 7B model for customized compliance checking with detailed explanations. We conduct extensive experiments against 11 strong guardrail baselines across 6 existing multilingual safety benchmarks and our ML-Bench, and show that ML-Guard consistently outperforms prior methods. We hope that ML-Bench and ML-Guard can help advance the development of regulation-aware and culturally aligned multilingual guardrail systems. 

---
# Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs 

**Authors**: Jasper Dekoninck, Nikola Jovanović, Tim Gehrunger, Kári Rögnvalddson, Ivo Petrov, Chenhao Sun, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2605.00674)  

**Abstract**: Large language models (LLMs) are becoming increasingly capable mathematical collaborators, but static benchmarks are no longer sufficient for evaluating progress: they are often narrow in scope, quickly saturated, and rarely updated. This makes it hard to compare models reliably and track progress over time. Instead, we need evaluation platforms: continuously maintained systems that run, aggregate, and analyze evaluations across many benchmarks to give a comprehensive picture of model performance within a broad domain. In this work, we build on the original MathArena benchmark by substantially broadening its scope from final-answer olympiad problems to a continuously maintained evaluation platform for mathematical reasoning with LLMs. MathArena now covers a much wider range of tasks, including proof-based competitions, research-level arXiv problems, and formal proof generation in Lean. Additionally, we maintain a clear evaluation protocol for all models and regularly design new benchmarks as model capabilities improve to ensure that MathArena remains challenging. Notably, the strongest model, GPT-5.5, now reaches 98% on the 2026 USA Math Olympiad and 74% on research-level questions, showing that frontier models can now comfortably solve extremely challenging mathematical problems. This highlights the importance of continuously maintained evaluation platforms like MathArena to track the rapid progress of LLMs in mathematical reasoning. 

---
# H-RAG at SemEval-2026 Task 8: Hierarchical Parent-Child Retrieval for Multi-Turn RAG Conversations 

**Authors**: Passant Elchafei, Hossam Emam, Mohamed Alansary, Monorama Swain, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2605.00631)  

**Abstract**: We present H-RAG, our submission to SemEval-2026 Task 8 (MTRAGEval), addressing both Task A (Retrieval) and Task C (Generation with Retrieved Passages). Task A evaluates standalone retrieval quality, while Task C assesses end-to-end retrieval-augmented generation (RAG) in multi-turn conversational settings, requiring both accurate answer generation and faithful grounding in retrieved evidence. Our approach implements a hierarchical parent-child RAG pipeline that separates fine-grained child-level retrieval from parent-level context reconstruction during generation. Documents are segmented into overlapping sentence-based child chunks, while full documents are preserved as parent units to provide coherent context. Retrieval combines hybrid dense-sparse search, tunable weighting, and embedding-based similarity rescoring over child chunks. Retrieved evidence is aggregated at the parent level and supplied to an instruction-tuned language model for response generation. H-RAG achieves an nDCG@5 score of 0.4271 on Task A and a harmonic mean score of 0.3241 on Task C (RB_agg: 0.2488, RL_F: 0.2703, RB_llm: 0.6508), underscoring the importance of retrieval configuration and parent-level aggregation in multi-turn RAG performance. 

---
# SC-Taxo: Hierarchical Taxonomy Generation under Semantic Consistency Constraints using Large Language Models 

**Authors**: Shiqiang Cai, Nianhong Niu, Shizhu He, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00620)  

**Abstract**: Scientific literature is expanding at an unprecedented pace, making it increasingly challenging to efficiently organize and access domain knowledge. A high-quality scientific taxonomy offers a structured and hierarchical representation of a research field, facilitating literature exploration and topic navigation, as well as enabling downstream applications such as trend analysis, idea generation, and information retrieval. However, existing taxonomy generation approaches often suffer from structural inconsistencies and semantic misalignment across hierarchical levels. Through empirical analysis, we find that these issues largely stem from inadequate modeling of hierarchical semantic consistency. To address this limitation, we propose a semantic-consistent taxonomy generation (SC-Taxo) framework that leverages large language models (LLMs) with hierarchy-aware refinement stages to ensure semantic consistency. Specifically, SC-Taxo introduces a bidirectional heading generation mechanism that jointly performs bottom-up abstraction and top-down semantic constraint, while further capturing peer-level semantic dependencies to enhance horizontal consistency. Experiments on multiple benchmark datasets demonstrate consistent improvements in hierarchy alignment and heading quality, and additional evaluation on Chinese scientific literature validates its robust cross-lingual generalization. 

---
# Is Textual Similarity Invariant under Machine Translation? Evidence Based on the Political Manifesto Corpus 

**Authors**: Daria Boratyn, Damian Brzyski, Albert Leśniak, Wojciech Łukasik, Maciej Rapacz, Jan Rybicki, Wojciech Słomczyński, Dariusz Stolicki  

**Link**: [PDF](https://arxiv.org/pdf/2605.00618)  

**Abstract**: We investigate the extent to which cosine similarity between paragraph embeddings is invariant under machine translation, using the Manifesto Corpus of over 2,800 political party platforms in 28 languages translated to English via the EU eTranslation service. Rather than measuring translation-induced semantic shift directly we measure the stability of pairwise similarity relationships across embedding models, and use inter-model disagreement on original-language text as a calibrated invariance threshold. This yields a per-language non-inferiority test for four hypotheses about how translation interacts with embedding choice, with verdicts that distinguish languages where translation demonstrably preserves semantic structure from those where it demonstrably degrades it and from those where the available evidence does not resolve the question. The framework is corpus- and pipeline-agnostic and extends naturally to downstream tasks. Applied to our data, it identifies ten languages with translation invariance and four with detectable distortion. 

---
# Beyond Decodability: Reconstructing Language Model Representations with an Encoding Probe 

**Authors**: Gaofei Shen, Martijn Bentum, Tom Lentz, Afra Alishahi, Grzegorz Chrupała  

**Link**: [PDF](https://arxiv.org/pdf/2605.00607)  

**Abstract**: Probing is widely used to study which features can be decoded from language model representations. However, the common decoding probe approach has two limitations that we aim to solve with our new encoding probe approach: contributions of different features to model representations cannot be directly compared, and feature correlations can affect probing results. We present an Encoding Probe that reverses this direction and reconstructs internal representations of models using interpretable features. We evaluate this method on text and speech transformer models, using feature sets spanning acoustics, phonetics, syntax, lexicon, and speaker identity. Our results suggest that speaker-related effects vary strongly across different training objectives and datasets, while syntactic and lexical features contribute independently to reconstruction. These results show that the Encoding Probe provides a complementary perspective on interpreting model representations beyond decodability. 

---
# Structure Liberates: How Constrained Sensemaking Produces More Novel Research Output 

**Authors**: James Mooney, Zae Myung Kim, Young-Jun Lee, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00557)  

**Abstract**: Scientific discovery is an extended process of ideation--surveying prior work, forming hypotheses, and refining reasoning--yet existing approaches treat this phase as a brief preamble despite its central role in research. We introduce SCISENSE, a sensemaking-grounded framework that operationalizes ideation as a structured sequence of eight cognitive stages (Pirolli \& Card, 2005). We construct SCISENSE-Traj, a 100K-scale dataset of citation-conditioned research trajectories in two modes: Target, where an LLM reconstructs the ideation path leading to a known paper from its cited works, and Infer, where the LLM proposes novel directions from the same citations. We distill these into SCISENSE-LM, a family of sensemaking LLMs spanning 3B to 70B parameters. Contrary to the assumption that looser supervision promotes greater exploration, Target-trained models achieve a 2.0\% improvement in trajectory quality over Infer-trained models while also producing more novel and diverse outputs. This advantage propagates downstream: coding agents conditioned on Target trajectories produce research artifacts with higher executability and quality than those conditioned on Infer trajectories. This suggests that targeted ideation reduces cognitive burden on downstream agents, freeing them to explore more creatively. SCISENSE offers both a practical tool for augmenting LLM-driven research workflows and a principled testbed for studying how planning shapes scientific discovery. 

---
# A11y-Compressor: A Framework for Enhancing the Efficiency of GUI Agent Observations through Visual Context Reconstruction and Redundancy Reduction 

**Authors**: Michito Takeshita, Takuro Kawada, Takumi Ohashi, Shunsuke Kitada, Hitoshi Iyatomi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00551)  

**Abstract**: AI agents that interact with graphical user interfaces (GUIs) require effective observation representations for reliable grounding. The accessibility tree is a commonly used text-based format that encodes UI element attributes, but it suffers from redundancy and lacks structural information such as spatial relationships among elements. We propose A11y-Compressor, a framework that transforms linearized accessibility trees into compact and structured representations. Our implementation, Compressed-a11y, applies a lightweight and structured transformation pipeline with modal detection, redundancy reduction, and semantic structuring. Experiments on the OSWorld benchmark show that Compressed-a11y reduces input tokens to 22% of the original while improving task success rates by 5.1 percentage points on average. 

---
# AGoQ: Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs 

**Authors**: Wenxiang Lin, Juntao Huang, Luhan Zhang, Laili Li, Xiang Bao, Mengyang Zhang, Bing Wang, Shaohuai Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00539)  

**Abstract**: Quantization is a key method for reducing the GPU memory requirement of training large language models (LLMs). Yet, current approaches are ineffective for 4-bit activations and 8-bit gradients, which would easily cause slow convergence or accuracy loss. To address this, we introduce AGoQ, incorporating two new techniques: 1) a layer-aware activation quantization algorithm that allocates appropriate bit-widths for activations of various layers based on their types and pipeline stages to achieve near 4-bit activation storage, and 2) a gradient quantization algorithm that reduces memory usage and shortens communication time by employing 8-bit gradient storage and precision-preserving 8-bit All-Reduce communication. We conduct extensive experiments using different sizes of LLMs on two GPU clusters (up to 64 GPUs), and the experimental results show that our AGoQ reduces the memory by up to 52\% and achieves up to 1.34$\times$ improvement of training speed compared to state-of-the-art training systems Megatron-LM (w/ or w/o ZeRO), COAT and DeepSpeed with 8B to 32B LLaMA models, while achieving convergence loss on pretraining and comparable accuracy on downstream tasks with LLaMA architectures. 

---
# ControBench: An Interaction-Aware Benchmark for Controversial Discourse Analysis on Social Networks 

**Authors**: Ta Thanh Thuy, Jiaqi Zhu, Xuan Liu, Lin Shang, Reihaneh Rabbany, Guillaume Rabusseau, Lihui Chen, Zheng Yilun, Sitao Luan  

**Link**: [PDF](https://arxiv.org/pdf/2605.00513)  

**Abstract**: Understanding how people argue across ideological divides online is important for studying political polarization, misinformation, and content moderation. Existing datasets capture only part of this problem: some preserve text but ignore interaction structure, some model structure without rich semantics, and others represent conversations without stable user-level ideological identity. We introduce ControBench, a benchmark for controversial discourse analysis that combines heterogeneous social interaction graphs with rich textual semantics. Built from Reddit discussions on three topics, Trump, abortion, and religion, ControBench contains 7,370 users, 1,783 posts, and 26,525 interactions. The graph contains user and post nodes connected by semantically enriched edges; in particular, user-comment-user edges encode both a reply and the parent comment that it responds to, preserving local argumentative context. User labels are derived from self-declared Reddit flairs, providing a scalable proxy for ideological identity without manual annotation. The resulting datasets exhibit low or negative adjusted homophily (Trump: -0.77, Abortion: 0.06, Religion: 0.04), reflecting the cross-cutting structure of real-world debate. We evaluate graph neural networks, pretrained language models, and large language models on ControBench and observe distinct performance patterns across topics and model families, especially when ideological boundaries are ambiguous. These results position ControBench as a challenging and realistic benchmark for controversial discourse analysis. 

---
# Surprisal Minimisation over Goal-directed Alternatives Predicts Production Choice in Dialogue 

**Authors**: Tom Utting, Mario Giulianelli, Arabella Sinclair  

**Link**: [PDF](https://arxiv.org/pdf/2605.00506)  

**Abstract**: We model utterance production as probabilistic cost-sensitive choice over contextual alternatives, using information-theoretic notions of cost. We distinguish between goal-directed alternatives that realise a fixed communicative intent and goal-agnostic alternatives defined only by contextual plausibility, allowing us to derive speaker- and listener-oriented interpretations of different cost measures. We present a procedure to generate both types of alternative sets using language models. Analysing production choices in open-ended dialogue under both deterministic and probabilistic cost minimisation, we find that surprisal minimisation relative to goal-directed alternatives provides the strongest predictive account under both analyses. By contrast, uniform information density and length-based costs exhibit weaker and less consistent predictive power across conditions. More broadly, our study suggests that alternative-conditioned optimisation with LM-generated alternatives provides a principled framework for studying speaker and listener pressures in naturalistic language production. 

---
# ReLay: Personalized LLM-Generated Plain-Language Summaries for Better Understanding, but at What Cost? 

**Authors**: Joey Chan, Yikun Han, Jingyuan Chen, Samuel Fang, Lauren D. Gryboski, Alexandra Lee, Sheel Tanna, Qingqing Zhu, Zhiyong Lu, Lucy Lu Wang, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.00468)  

**Abstract**: Plain Language Summaries (PLS) aim to make research accessible to lay readers, but they are typically written in a one-size-fits-all style that ignores differences in readers' information needs and comprehension. In health contexts, this limitation is particularly important because misunderstanding scientific information can affect real-world decisions. Large language models (LLMs) offer new opportunities for personalizing PLS, but it remains unclear whether personalization helps, which strategies are most effective, and how to balance personalization with safety. We introduce ReLay, a dataset of 300 participant--PLS pairs from 50 lay participants in both static (expert-written) and interactive (LLM-personalized) settings. ReLay includes user characteristics, health information needs, information-seeking behavior, comprehension outcomes, interaction logs, and quality ratings. We use ReLay to evaluate five LLMs across two personalization methods. Personalization improves comprehension and perceived quality, but it also raises the risk of reinforcing user biases and introducing hallucinations, revealing a trade-off between personalization and safety. These findings highlight the need for personalization methods that are both effective and trustworthy for diverse lay audiences. 

---
# Impact of Task Phrasing on Presumptions in Large Language Models 

**Authors**: Kenneth J.K. Ong  

**Link**: [PDF](https://arxiv.org/pdf/2605.00436)  

**Abstract**: Concerns with the safety and reliability of applying large-language models (LLMs) in unpredictable real-world applications motivate this study, which examines how task phrasing can lead to presumptions in LLMs, making it difficult for them to adapt when the task deviates from these assumptions. We investigated the impact of these presumptions on the performance of LLMs using the iterated prisoner's dilemma as a case study. Our experiments reveal that LLMs are susceptible to presumptions when making decisions even with reasoning steps. However, when the task phrasing was neutral, the models demonstrated logical reasoning without much presumptions. These findings highlight the importance of proper task phrasing to reduce the risk of presumptions in LLMs. 

---
# Escaping Mode Collapse in LLM Generation via Geometric Regulation 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2605.00435)  

**Abstract**: Mode collapse is a persistent challenge in generative modeling and appears in autoregressive text generation as behaviors ranging from explicit looping to gradual loss of diversity and premature trajectory convergence. We take a dynamical-systems view and reinterpret mode collapse as reduced state-space accessibility caused by *geometric collapse*: during generation, the model's internal trajectory becomes confined to a low-dimensional region of its representation space. This implies mode collapse is not purely a token-level phenomenon and cannot be reliably solved by symbolic constraints or probability-only decoding heuristics. Guided by this perspective, we propose *Reinforced Mode Regulation* (RMR), a lightweight, online state-space intervention that regulates dominant self-reinforcing directions in the Transformer value cache (implemented as low-rank damping). Across multiple large language models, RMR substantially reduces mode collapse and enables stable, high-quality generation at extremely low entropy rates (down to 0.8 nats/step), whereas standard decoding typically collapses near 2.0 nats/step. 

---
# RadLite: Multi-Task LoRA Fine-Tuning of Small Language Models for CPU-Deployable Radiology AI 

**Authors**: Pankaj Gupta, Kartik Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.00421)  

**Abstract**: Large language models (LLMs) show promise in radiology but their deployment is limited by computational requirements that preclude use in resource-constrained clinical environments. We investigate whether small language models (SLMs) of 3-4 billion parameters can achieve strong multi-task radiology performance through LoRA fine-tuning, enabling deployment on consumer-grade CPUs. We train Qwen2.5-3B-Instruct and Qwen3-4B on 162K samples spanning 9 radiology tasks - RADS classification across 10 systems, impression generation, temporal comparison, radiology NLI, NER, abnormality detection, N/M staging, and radiology Q&A - compiled from 12 public datasets. Both models are evaluated on up to 500 held-out test samples per task with standardized metrics. Our key findings are: (1) LoRA fine-tuning dramatically improves performance over zero-shot baselines (RADS accuracy +53%, NLI +60%, N-staging +89%); (2) the two models exhibit complementary strengths - Qwen2.5 excels at structured generation tasks while Qwen3 dominates extractive tasks; (3) a task-outed oracle ensemble combining both models achieves the best performance across all tasks; (4) few-shot prompting with fine-tuned models hurts performance, demonstrating that LoRA adaptation is more effective than in-context learning for specialized domains; and (5) models can be quantized to GGUF format (~1.8-2.4GB) for CPU deployment at 4-8 tokens/second on consumer hardware. Our work demonstrates that small, efficiently fine-tuned models - which we collectively call RadLite - can serve as practical multi-task radiology AI assistants deployable entirely on consumer hardware without GPU requirements. 

---
# Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines 

**Authors**: Aninda Ray  

**Link**: [PDF](https://arxiv.org/pdf/2605.00410)  

**Abstract**: A multi-agent pipeline with N agents typically issues N LLM calls per run. Merging agents into fewer calls (compound execution) promises token savings, but naively merged calls silently degrade quality through tool loss and prompt compression. We present Agent Capsules, an adaptive execution runtime that treats multi-agent pipeline execution as an optimization problem with empirical quality constraints. The runtime instruments coordination overhead per group, scores composition opportunity, selects among three compound execution strategies, and gates every mode switch on rolling-mean output quality. A controlled negative result confirms that injecting more context into a merged call worsens compression rather than relieving it, so the framework's escalation ladder (standard, then two-phase, then sequential) recovers quality by moving toward per-agent dispatch rather than by rewriting merged prompts. On LLM-judged quality, the controller matches a hand-tuned oracle on every measured (model, group, mode) cell: routing compound whenever the oracle would, and reverting to fine whenever quality would fail the floor, without per-model configuration. Against a hand-crafted LangGraph implementation of a 14-agent competitive intelligence pipeline, Agent Capsules uses 51% fewer fine-mode input tokens and 42% fewer compound-mode input tokens, at +0.020 and +0.017 quality respectively. Against a DSPy implementation of a 5-agent due diligence pipeline, the framework uses 19% fewer tokens than uncompiled DSPy at quality parity, and 68% fewer tokens than MIPROv2 at +0.052 quality. Even before compound mode fires, the runtime delivers efficiency through automatic policy resolution, cache-aligned prompts, and topology-aware context injection, matching both hand-tuned and compile-time baselines without training data or per-pipeline engineering. 

---
# Agentic AI for Substance Use Education: Integrating Regulatory and Scientific Knowledge Sources 

**Authors**: Kosar Haghani, Zahra Kolagar, Mohammed Atiquzzaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.00383)  

**Abstract**: The delivery of traditional substance education has remained problematic due to challenges in scalability, personalization, and the currency of information in a rapidly evolving substance use landscape. While artificial intelligence (AI) offers a promising frontier for enhancing educational delivery, its application in providing real-time, authoritative substance use education remains largely underexplored. We built an agentic-based AI web application that combined Drug Enforcement Administration records with peer-reviewed literature in real-time to provide transparent context-sensitive substance use education. The system uses retrieval-augmented generation with a carefully filtered corpus of 102 documents and dynamic PubMed queries. Document storage was semantically chunked and placed in a vector representation in order to be easily retrieved. We conducted an expert evaluation study in which a panel of five subject matter experts generated 30 domain-specific questions, and two independent raters assessed 90 system interactions (30 primary questions plus two contextual follow-ups each) using a five-point Likert scale across four criteria: factual accuracy, citation quality, contextual coherence, and regulatory appropriateness. Mean ratings ranged from 4.18 to 4.35 across the four criteria (overall category range: 4.05-4.52), with substantial inter-rater agreement (Cohen's kappa = 0.78). These findings suggest that agentic AI architectures integrating authoritative regulatory sources with real-time scientific literature represent a promising direction for scalable, accurate, and verifiable health education delivery, warranting further evaluation through longitudinal user studies. 

---
# Language-free Experience at Expo 2025 Osaka 

**Authors**: Michael Paul, Kenji Imamura, Xiaolin Wang, Shohei Higashiyama, Masao Utiyama  

**Link**: [PDF](https://arxiv.org/pdf/2605.00373)  

**Abstract**: In line with the Global Communication Plan 2025, we have pursued the development of multilingual translation technologies to realize a language-barrier-free experience at Expo 2025 Osaka. Our work includes the advancement of simultaneous interpretation systems emphasizing high translation quality and low latency. Key achievements include chunk-based input segmentation, context-aware translation, and multi-engine machine translation technologies. Through demonstration deployments and collaboration with private companies, our technologies have led to real-world applications, with several services and systems showcased at Expo 2025 Osaka. 

---
# Unlearning What Matters: Token-Level Attribution for Precise Language Model Unlearning 

**Authors**: Jiawei Wu, DouDou Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.00364)  

**Abstract**: Machine unlearning has emerged as a critical capability for addressing privacy, safety, and regulatory concerns in large language models (LLMs). Existing methods operate at the sequence level, applying uniform updates across all tokens despite only a subset encoding the knowledge targeted for removal. This introduces gradient noise, degrades utility, and leads to suboptimal forgetting. We propose TokenUnlearn, a token-level attribution framework that identifies and selectively targets critical tokens. Our approach combines knowledge-aware signals via masking, and entropy-aware signals to yield importance scores for precise token selection. We develop two complementary strategies: hard selection, applying unlearning only to high-importance tokens, and soft weighting, modulating gradient contributions based on importance scores. Both extend existing methods to token-level variants. Theoretical analysis shows token-level selection improves gradient signal-to-noise ratio. Experiments on TOFU and WMDP benchmarks across three model architectures demonstrate consistent improvements over sequence-level baselines in both forgetting effectiveness and utility preservation. 

---
# From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing 

**Authors**: Wei Liu, Hongkai Liu, Zhiying Deng, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.00358)  

**Abstract**: LLM parameter editing methods commonly rely on computing an ideal target hidden-state at a target layer (referred as anchor point) and distributing the target vector to multiple preceding layers (commonly known as backward spreading) for cooperative editing. Although widely used for a long time, its underlying basis have not been systematically investigated. In this paper, we first conduct a systematic study of its foundations, which helps clarify its capability boundaries, practical considerations, and potential failure modes. Then, we propose a simple and elegant alternative that replaces backward spreading with forward-propagation. Instead of optimizing the target at the last editing layer, we optimize the anchor point at the first editing layer, and then propagate it forward to obtain accurate and mutually compatible target hidden-states for all subsequent editing layers. This approach achieves the same computational complexity as existing methods while producing more accurate layer-wise targets. Our method is simple, without interfering with either the computation of the initial target hidden state or any other components of the subsequent editing pipeline, and thus constituting a benefit for a wide range of LLM parameter editing methods. 

---
# MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents 

**Authors**: Tianyu Hu, Weikai Lin, Weizhi Zhang, Jing Ma, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00356)  

**Abstract**: Long-term conversational agents must decide which turns to store in external memory, yet recent systems rely on autoregressive LLM generation at every turn to make that decision. We present MemRouter, a write-side memory router that decouples memory admission from the downstream answer backbone and replaces per-turn memory-management decoding with an embedding-based routing policy. MemRouter encodes each turn together with recent context, projects the resulting embeddings through a frozen LLM backbone, and predicts whether the turn should be stored using lightweight classification heads while training only 12M parameters. Under a controlled matched-harness comparison on LoCoMo, where the retrieval pipeline, answer prompts, and QA backbone (Qwen2.5-7B) are held identical, MemRouter outperforms an LLM-based memory manager on every question category (overall F1 52.0 vs 45.6, non-overlapping 95% CIs) while reducing memory-management p50 latency from 970ms to 58ms. Descriptive factorial averaging further shows that learned admission improves mean F1 by +10.3 over random storage, category-specific prompting adds +5.2 over a generic prompt, and retrieval contributes +0.7. These results suggest that write-side memory admission can be learned by a small supervised router, while answer generation remains a separate downstream component in long-horizon conversational QA. 

---
# Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding 

**Authors**: Lehan Pan, Ziyang Tao, Ruoyu Pang, Xiao Wang, Jianjun Zhao, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00342)  

**Abstract**: Tree-based speculative decoding accelerates autoregressive generation by verifying multiple draft candidates in parallel, but this advantage weakens for sparse Mixture-of-Experts (MoE) models. As the draft tree grows, different branches activate different experts, expanding the union of activated experts and substantially increasing target-side verification cost. We propose EVICT, a training-free, hyperparameter-free, and lossless adaptive verification method for MoE speculative decoding. EVICT makes every verified token count by truncating the draft tree before target verification and retaining only the cost-effective prefix. It leverages fine-grained drafter signals to estimate candidate benefit, combines them with offline-profiled verification cost, and remains highly compatible with the high-performance graph-based serving framework SGLang. Extensive experiments on diverse MoE backbones and benchmarks show that EVICT achieves up to 2.35x speedup over autoregressive decoding and an average 1.21x speedup over the state-of-the-art baseline EAGLE-3, while significantly reducing unnecessary expert activations during verification. 

---
# Budget-Aware Routing for Long Clinical Text 

**Authors**: Khizar Qureshi, Geoffrey Martin, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00336)  

**Abstract**: A key challenge for large language models is token cost per query and overall deployment cost. Clinical inputs are long, heterogeneous, and often redundant, while downstream tasks are short and high stakes. We study budgeted context selection, where a subset of document units is chosen under a strict token budget so an off-the-shelf generator can meet fixed cost and latency constraints. We cast this as a knapsack-constrained subset selection problem with two design choices, unitization that defines document segmentation and selection that determines which units are kept.
We propose \textbf{RCD}, a monotone submodular objective that balances relevance, coverage, and diversity. We compare sentence, section, window, and cluster-based unitization, and introduce a routing heuristic that adapts to the budget regime. Experiments on MIMIC discharge notes, Cochrane abstracts, and L-Eval show that optimal strategies depend on the evaluation setting. Positional heuristics perform best at low budgets in extractive tasks, while diversity-aware methods such as MMR improve LLM generation. Selector choice matters more than unitization, with cluster-based grouping reducing performance and other schemes behaving similarly. ROUGE saturates for LLM summaries, while BERTScore better reflects quality differences. We release our code at this https URL. 

---
# Prompt-Induced Score Variance in Zero-Shot Binary Vision-Language Safety Classification 

**Authors**: Charles Weng, Dingwen Li, Alexander Martin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00326)  

**Abstract**: Single-prompt first-token probabilities from zero-shot vision-language model (VLM) safety classifiers are treated as decision scores, but we show they are unreliable under semantically equivalent prompt reformulation: even when the binary label is constrained to a fixed output position, equivalent prompts can induce materially different unsafe probabilities for the same sample. Across multimodal safety benchmarks and multiple VLM families, cross-prompt variance is strongly associated with prompt-level disagreement and higher error, making it a useful fragility diagnostic. A training-free mean ensemble improves NLL on all 14 dataset-model evaluation pairs and ECE on 12/14 relative to a train-selected single-prompt baseline, and wins more head-to-head NLL comparisons than labeled temperature scaling, Platt scaling, and isotonic regression applied to the same prompt. Ranking gains are consistent against the train-selected baseline on both AUROC and AUPRC, and against the full 15-prompt distribution remain consistent on AUPRC while softening on AUROC. Labeled calibration on top of the mean provides further gains when labels are available, identifying prompt averaging as a strong label-free first stage rather than a replacement for calibration. We frame this as a reliability stress test for zero-shot VLM first-token safety scores and recommend prompt-family evaluation with mean aggregation as a standard label-free reliability baseline. 

---
# Structure-Aware Chunking for Tabular Data in Retrieval-Augmented Generation 

**Authors**: Pooja Guttal, Varun Magotra, Vasudeva Mahavishnu, Natasha Chanto, Sidharth Sivaprasad, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2605.00318)  

**Abstract**: Tabular documents such as CSV and Excel files are widely used in enterprise data pipelines, yet existing chunking strategies for retrieval-augmented generation (RAG) are primarily designed for unstructured text and do not account for tabular structure. We propose a structure-aware tabular chunking (STC) framework that operates on row-level units by constructing a hierarchical Row Tree representation, where each row is encoded as a key-value block. STC performs token-constrained splitting aligned with structural boundaries and applies overlap-free greedy merging to produce dense, non-overlapping chunks. This design preserves semantic relationships between fields within a row while improving token utilization and reducing fragmentation. Across evaluations on the MAUD dataset, STC reduces chunk count by up to 40% and 56% compared to standard recursive and key-value based baselines, respectively, while improving token utilization and processing efficiency. In retrieval benchmarks, STC improves MRR from 0.3576 to 0.5945 in a hybrid setting and increases Recall@1 from 0.366 to 0.754 in BM25-only retrieval. These results demonstrate that preserving structure during chunking improves retrieval performance, highlighting the importance of structure-aware chunking for RAG over tabular data. 

---
# What Don't You Understand? Using Large Language Models to Identify and Characterize Student Misconceptions About Challenging Topics 

**Authors**: Michael J. Parker, Maria G. Zavala-Cerna  

**Link**: [PDF](https://arxiv.org/pdf/2605.00294)  

**Abstract**: This study presents a systematic approach to identifying and characterizing student misconceptions in online learning environments through a novel combination of quantitative performance analysis and large language model (LLM) assessment. We analyzed data from 9 course periods across 5 online biomedical science courses, encompassing 3,802 medical student enrollments. Using data from 40-50 topic-focused quizzes per course, we developed a two-stage methodology. First, we identified challenging central topics using quiz-level performance metrics. Second, we employed LLMs to characterize the underlying misconceptions in these high-priority areas. By examining student performance on first attempts across primarily multiple-choice questions (MCQs), we identified consistently challenging topics that were also central to course objectives. We then leveraged recent advances in generative AI to analyze three distinct data sources in combination: quiz question content, student response patterns, and lecture transcripts. This approach revealed actionable insights about student misconceptions that were not apparent from performance data alone. The quality of the LLM-identified misconceptions was rated as excellent by subject matter experts. We also conducted teacher interviews to assess the perceived utility of our topic identification method. Faculty found that data-driven identification of challenging topics was valuable and corroborated their own classroom observations. This methodology provides a scalable approach to characterizing student difficulties in learning environments where quizzes are used. Our findings demonstrate the potential for targeted and potentially personalized interventions in future course iterations, with clear pathways for measuring intervention effectiveness through follow-up quiz performance. 

---
# Are You the A-hole? A Fair, Multi-Perspective Ethical Reasoning Framework 

**Authors**: Sheza Munir, Ahanaf Rodoshi, Sumin Lee, Feiran Chang, Xujie Si, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2605.00270)  

**Abstract**: Standard methods for aggregating natural language judgments, such as majority voting, often fail to produce logically consistent results when applied to high-conflict domains, treating differing opinions as noise. We propose a neuro-symbolic aggregation framework that formalizes conflict resolution through Weighted Maximum Satisfiability (MaxSAT). Our pipeline utilizes a language model to map unstructured natural language explanations into interpretable logical predicates and confidence weights. These components are then encoded as soft constraints within the Z3 solver, transforming the aggregation problem into an optimization task that seeks the maximum consistency across conflicting testimony. Using the Reddit r/AmItheAsshole forum as a case study in large-scale moral disagreement, our system generates logically coherent verdicts that diverge from popularity-based labels 62% of the time, corroborated by an 86% agreement rate with independent human evaluators. This study demonstrates the efficacy of coupling neural semantic extraction with formal solvers to enforce logical soundness and explainability in the aggregation of noisy human reasoning. 

---
# How Language Models Process Out-of-Distribution Inputs: A Two-Pathway Framework 

**Authors**: Hamidreza Saghir  

**Link**: [PDF](https://arxiv.org/pdf/2605.00269)  

**Abstract**: Recent white-box OOD detection methods for LLMs -- including CED, RAUQ, and WildGuard confidence scores -- appear effective, but we show they are structurally confounded by sequence length (|r| >= 0.61) and collapse to near-chance under length-matched evaluation. Even raw attention entropy (mean H(alpha) across heads and layers), a natural baseline we include for completeness, shows the same confound. The confound stems from attention's Theta(log T) dependence on input length. To identify genuine OOD signals after deconfounding, we propose a two-pathway framework: embeddings capture what text is about (effective for topic shifts), while the processing trajectory -- hidden-state evolution across layers -- captures how the model processes input. The relative power of each pathway varies along a vocabulary-transparency spectrum: embedding methods excel on vocabulary-distinctive OOD, while trajectory features detect covert-intent inputs that share vocabulary with normal text (0.721 avg AUROC; Jailbreak: 0.850). Three evidence lines support this framework: (1) a crossover between k-NN and trajectory scoring across 6 tasks, where each pathway wins on different OOD types; (2) a per-layer analysis showing that layer-0 k-NN signal is almost entirely a length artifact (Jailbreak: 0.759 raw -> 0.389 matched) -- processing constructs genuine OOD signal from near-chance embeddings; and (3) circuit attribution showing adversarial tasks engage attention circuits more than semantic tasks (p = 0.022; Jailbreak patching p < 0.001), with partial cross-model replication. Code release upon publication. 

---
# Retrieval-Augmented Reasoning for Chartered Accountancy 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Ali Imam Abidi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00257)  

**Abstract**: The inception of Large Language Models (LLMs) has catalyzed AI adoption in the finance sector, yet their reliability in complex, jurisdiction-specific tasks like Indian Chartered Accountancy (CA) remains limited. The models display difficulty in executing numerical tasks which require multiple steps while also needing advanced knowledge about legal regulations and the method of scaling their operations is not feasible in settings which have limited access to resources. We present CA-ThinkFlow as a parameter-efficient Retrieval-Augmented Generation (RAG) framework which operates with a 14B, 4-bit-quantized reasoning model, 14B-DeepSeek-R1, and a layout-aware Docling extraction system which maintains document structure during extraction. CA-ThinkFlow uses a basic RAG method which automatically adds retrieved information into the prompt, while it depends on the model's built-in Chain-of-Thought (CoT) functions to create context and produce correct answers. The system we developed system operates at performance levels which match large proprietary models when we tested it on the multi-level CA-Ben benchmark, achieving Scholastic Reliability Coefficient (SRC) results which equal 68.75\% of GPT-4o and Claude 3.5 Sonnet. The framework shows high efficiency and strength in handling parameters, but essential reasoning abilities fail to process complex regulatory texts which exist in fields such as Taxation. 

---
# Lost in State Space: Probing Frozen Mamba Representations 

**Authors**: Bhagyashree Wagh, Akash Singh  

**Link**: [PDF](https://arxiv.org/pdf/2605.00253)  

**Abstract**: Mamba's recurrent state h_t is, by construction, a compressed summary of every token seen so far. This raises a tempting hypothesis: if we extract token-level outputs y_t at fixed patch boundaries, we obtain semantic sentence summaries for free, with no pooling head, no fine-tuning, and no [CLS] token. We test this hypothesis carefully. Across five benchmarks (SST-2, CoLA, MRPC, STS-B, IMDb), we compare four strategies for extracting frozen sentence representations from a pretrained Mamba-130M backbone under a strict frozen-feature probing protocol, using three random seeds where computationally feasible. The results do not support the hypothesis: patch boundary readouts do not consistently outperform simple mean pooling. We identify and quantify two structural pathologies: severe anisotropy (mean pairwise cosine similarity 0.9999, std 0.000044) and representational collapse in the raw final SSM state (MCC = 0.000 on CoLA across all three seeds, confirmed via confusion matrix). We further propose orthogonal injection, a modified recurrence that constrains new information per 

---
# Estimating LLM Grading Ability and Response Difficulty in Automatic Short Answer Grading via Item Response Theory 

**Authors**: Longwei Cong, Sonja Hahn, Sebastian Gombert, Leon Camus, Hendrik Drachsler, Ulf Kroehne  

**Link**: [PDF](https://arxiv.org/pdf/2605.00238)  

**Abstract**: Automated short answer grading (ASAG) with large language models (LLMs) is commonly evaluated with aggregate metrics such as macro-F1 and Cohen's kappa. However, these metrics provide limited insight into how grading performance varies across student responses of differing grading difficulty. We introduce an evaluation framework for LLM-based ASAG based on item response theory (IRT), which models grading correctness as a function of latent grader ability and response grading difficulty. This formulation enables response-level analysis of where LLM graders succeed or fail and reveals robustness differences that are not visible from aggregate scores alone. We apply the framework to 17 open-weight LLMs on the SciEntsBank and Beetle benchmarks. The results show that even models with similar overall performance differ substantially in how sharply their grading accuracy declines as response difficulty increases. In addition, confusion patterns show that errors on difficult responses concentrate disproportionately on the \texttt{partially\_correct\_incomplete} label, indicating a tendency toward intermediate-label collapse under ambiguity. To characterize difficult responses, we further analyze semantic and linguistic correlates of estimated difficulty. Across both datasets, higher difficulty is associated with weaker semantic alignment to the reference answer, stronger contradiction signals, and greater semantic isolation in embedding space. Overall, these results show that item response theory offers a useful framework for evaluating LLM-based ASAG beyond aggregate performance measures. 

---
# Persona-Grounded Safety Evaluation of AI Companions in Multi-Turn Conversations 

**Authors**: Prerna Juneja, Lika Lomidze  

**Link**: [PDF](https://arxiv.org/pdf/2605.00227)  

**Abstract**: There are growing concerns about the risks posed by AI companion applications designed for emotional engagement. Existing safety evaluations often rely on self-reported user data or interviews, offering limited insights into real-time dynamics. We present the first end-to-end scalable framework for controlled simulation and safety evaluation of multi-turn interactions with AI companion applications. Our framework integrates four key components: persona construction with clinical and psychometric validation, persona-specific scenario generation, scenario-driven multi-turn simulation with a dialogue refinement module that preserves persona fidelity, and harm evaluation. We apply this framework to evaluate how Replika, a widely used AI companion app, responds to high-risk user groups. We construct 9 personas representing individuals with depression, anxiety, PTSD, eating disorders, and incel identity, and collect 1,674 dialogue pairs across 25 high-risk scenarios. We combine emotion modeling and LLM-assisted utterance-and harm-level classification to analyze these exchanges. Results show that Replika exhibits a narrow emotional range dominated by curiosity and care, while frequently mirroring or normalizing unsafe content such as self-harm, disordered eating, and violent-fantasy narratives. These findings highlight how controlled persona simulations can serve as a scalable testbed for evaluating safety risks in AI companions. 

---
# Why Do LLMs Struggle in Strategic Play? Broken Links Between Observations, Beliefs, and Actions 

**Authors**: Jan Sobotka, Mustafa O. Karabag, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00226)  

**Abstract**: Large language models (LLMs) are increasingly tasked with strategic decision-making under incomplete information, such as in negotiation and policymaking. While LLMs can excel at many such tasks, they also fail in ways that are poorly understood. We shed light on these failures by uncovering two fundamental gaps in the internal mechanisms underlying the decision-making of LLMs in incomplete-information games, supported by experiments with open-weight models Llama 3.1, Qwen3, and gpt-oss. First, an observation-belief gap: LLMs encode internal beliefs about latent game states that are substantially more accurate than their own verbal reports, yet these beliefs are brittle. In particular, the belief accuracy degrades with multi-hop reasoning, exhibits primacy and recency biases, and drifts away from Bayesian coherence over extended interactions. Second, a belief-action gap: The implicit conversion of internal beliefs into actions is weaker than that of the beliefs externalized in the prompt, yet neither belief-conditioning consistently achieves higher game payoffs. These results show how analyzing LLMs' internal processes can expose systematic vulnerabilities that warrant caution before deploying LLMs in strategic domains without robust guardrails. 

---
# Confidence Estimation in Automatic Short Answer Grading with LLMs 

**Authors**: Longwei Cong, Sonja Hahn, Sebastian Gombert, Leon Camus, Hendrik Drachsler, Ulf Kroehne  

**Link**: [PDF](https://arxiv.org/pdf/2605.00200)  

**Abstract**: Automatic Short Answer Grading (ASAG) with generative large language models (LLMs) has recently demonstrated strong performance without task-specific fine-tuning, while also enabling the generation of synthetic feedback for educational assessment. Despite these advances, LLM-based grading remains imperfect, making reliable confidence estimates essential for safe and effective human-AI collaboration in educational decision-making. In this work, we investigate confidence estimation for ASAG with LLMs by jointly considering model-based confidence signals and dataset-derived uncertainty. We systematically compare three model-based confidence estimation strategies, namely verbalizing, latent, and consistency-based confidence estimation, and show that model-based confidence alone is insufficient to reliably capture uncertainty in ASAG. To address this limitation, we propose a hybrid confidence framework that integrates model-based confidence signals with an explicit estimate of dataset-derived aleatoric uncertainty. Aleatoric uncertainty is operationalized by clustering semantically embedded student responses and quantifying within-cluster heterogeneity. Our results demonstrate that the proposed hybrid confidence measure yields more reliable confidence estimates and improves selective grading performance compared to single-source approaches. Overall, this work advances confidence-aware LLM-based grading for human-in-the-loop assessment, supporting more trustworthy AI-assisted educational assessment systems. 

---
# RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners 

**Authors**: Jugal Gajjar, Kamalasankari Subramaniakuppusamy  

**Link**: [PDF](https://arxiv.org/pdf/2605.00199)  

**Abstract**: When a language model answers a table question, users have no way to verify which cells informed which reasoning steps. We introduce RSAT, a method that trains small language models (SLMs, 1-8B) to produce step-by-step reasoning with cell-level citations grounded in table evidence. Phase 1 (SFT) teaches a structured JSON output format from verified reasoning traces. Phase 2 (GRPO) optimizes a composite reward centered on NLI-based faithfulness, alongside citation validity and parsimony. Across six models from two families-Qwen 2.5 (1.5B/3B/7B) and Llama 3 (1B/3B/8B)-RSAT improves faithfulness 3.7$\times$ over SFT alone (0.224$\rightarrow$0.826), with near-perfect citation validity (0.992). Post-hoc attribution collapses below 13% format success, confirming that attribution must be integrated into reasoning, not retrofitted. Ablations show the faithfulness reward is essential: removing it drops faithfulness from 0.97 to 0.03. 

---
# Timing is Everything: Temporal Scaffolding of Semantic Surprise in Humor 

**Authors**: Yuxi Ma, Yongqian Peng, Junchen Lyu, Chi Zhang, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00143)  

**Abstract**: Humor is a fundamental cognitive phenomenon in which humans derive pleasure from the expectation violations and their resolution, exemplifying the brain's dynamic capacity for predictive processing. Classical humor theories emphasize semantic incongruity as the primary driver of amusement, yet overlook temporal dynamics despite comedians' intuition that "timing is everything." The extent to which temporal structure contributes to humor appreciation and how it interacts with semantic content remains poorly understood. Here, we propose the Dual Prediction Violation (DPV) framework to capture the interplay between content and timing. By analyzing 828 professional Chinese stand-up performances, we show that temporal features substantially outweigh semantic incongruity in predicting audience appreciation. Specifically, we find that peak semantic violations matter more than average incongruity levels, and pauses systematically lengthen before high-surprise punchlines--a strategic coupling that distinguishes successful from unsuccessful performances. These findings reframe humor as temporally scaffolded, where timing and semantic content operate in strategic coordination rather than independently. Our DPV framework bridges humor theory with predictive processing, demonstrating that temporal structure plays a central role in naturalistic humor appreciation with implications for understanding multi-scale prediction integration in linguistic processing. 

---
# Cultural Benchmarking of LLMs in Standard and Dialectal Arabic Dialogues 

**Authors**: Muhammad Dehan Al Kautsar, Saeed Almheiri, Momina Ahsan, Bilal Elbouardi, Younes Samih, Sarfraz Ahmad, Amr Keleg, Omar El Herraoui, Kareem Elzeky, Abed Alhakim Freihat, Mohamed Anwar, Zhuohan Xie, Junhong Liang, Mohammad Rustom Al Nasar, Preslav Nakov, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2605.00119)  

**Abstract**: There is a significant gap in evaluating cultural reasoning in LLMs using conversational datasets that capture culturally rich and dialectal contexts. Most Arabic benchmarks focus on short text snippets in Modern Standard Arabic (MSA), overlooking the cultural nuances that naturally arise in dialogues. To address this gap, we introduce ArabCulture-Dialogue, a culturally grounded conversational dataset covering 13 Arabic-speaking countries, in both MSA and each country's respective dialect, spanning 12 daily-life topics and 54 fine-grained subtopics. We utilize the dataset to form three benchmarking tasks: (i) multiple-choice cultural reasoning, (ii) machine translation between MSA and dialects, and (iii) dialect-steering generation. Our experiments indicate that the performance gap between MSA and Arabic dialects still exists, whereby the models perform worse on all three tasks in the dialectal setup, compared to the MSA one. 

---
# ViLegalNLI: Natural Language Inference for Vietnamese Legal Texts 

**Authors**: Nhung Thi-Hong Duong, Mai Ngoc Ho, Tin Van Huynh, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00116)  

**Abstract**: In this article, we introduce ViLegalNLI, the first large-scale Vietnamese Natural Language Inference (NLI) dataset specifically constructed for the legal domain. The dataset consists of 42,012 premise-hypothesis pairs derived from official statutory documents and annotated with binary inference labels (Entailment and Non-entailment). It covers multiple legal domains and reflects realistic legal reasoning scenarios characterized by structured logic, conditional clauses, and domain-specific terminology. To construct ViLegalNLI, we propose a semi-automatic data generation framework that integrates large language models for controlled hypothesis generation and systematic quality validation procedures. The framework incorporates artifact mitigation strategies and cross-model validation to improve annotation reliability and ensure legal consistency. The resulting dataset captures diverse reasoning patterns, including paraphrasing, logical implication, and legally invalid inferences, thereby providing a comprehensive benchmark for Vietnamese legal inference tasks. We conduct extensive experiments on the ViLegalNLI using multilingual models, Vietnamese-specific pretrained language models, and instruction-tuned large language models. The results show that few-shot LLM configurations consistently achieve superior performance, while performance is significantly influenced by hypothesis length, lexical overlap, and reasoning complexity. Cross-domain evaluations further reveal the challenges of generalizing legal inference across distinct legal fields. Overall, ViLegalNLI establishes a foundational benchmark for Vietnamese legal NLI and supports future research in legal reasoning, statutory text understanding, and the development of reliable AI systems for legal analysis and decision support. The dataset is publicly available for research purposes. 

---
# How Frontier LLMs Adapt to Neurodivergence Context: A Measurement Framework for Surface vs. Structural Change in System-Prompted Responses 

**Authors**: Ishan Gupta, Pavlo Buryi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00113)  

**Abstract**: We examine if frontier chat-based large language models (LLMs) adjust their outputs based on neurodivergence (ND) context in system prompts and describe the nature of these adjustments. Specifically, we propose NDBench, a 576-output benchmark involving two frontier models, three system prompt types (baseline, ND-profile assertion, and ND-profile assertion with explicit instructions for adjustments), four canonical ND profiles, and 24 prompts across four categories, one of which involves an adversarial masking strategy.
Four trends emerge consistently from our findings. First, LLMs show significant adaptation under ND context, where fully instructed conditions yield lengthier and more structured outputs, characterized by higher token counts, more headings, and more granular steps (p < 10^-8, Holm-corrected). Second, such adaptation is largely structural in nature: although list density does not change much, there is a marked rise in the frequency of headings and per-step detail. Third, ND persona assertion alone fails to suppress potentially harmful tendencies, as masking-reinforcement decreases only in explicitly instructed cases (36-44% reduction); the reduction rate barely changes in persona assertion conditions.
Moreover, reliability analysis of LLM-based harm assessment reveals that only two out of the six dimensions (masking and reinforcement, validation quality) exceed the pre-defined inter-judge agreement criterion (alpha >= 0.67) and thus can be considered primary results.
NDBench is made publicly available along with its prompts, outputs, code, and other resources, forming a reproducible framework for auditing future LLMs' adaptation to ND awareness. 

---
# NorBERTo: A ModernBERT Model Trained for Portuguese with 331 Billion Tokens Corpus 

**Authors**: Enzo S. N. Silva, Pablo B. Costa, Raphael C. Vlasman, Rosimeire P. Costa, Henrique L. P. Silva, Lucas F. A. O. Pellicer, Guilherme Rinaldo, Renato A. Almeida, Darian S. R. Rabbani, Cinthya O. Oestreich, Vinicius F. Caridá  

**Link**: [PDF](https://arxiv.org/pdf/2605.00086)  

**Abstract**: High-quality corpora are essential for advancing Natural Language Processing (NLP) in Portuguese. Building on previous encoder-only models such as BERTimbau and Albertina PT-BR, we introduce NorBERTo, a modern encoder based on the ModernBERT architecture, featuring long-context support and efficient attention mechanisms. NorBERTo is trained on Aurora-PT, a newly curated Brazilian Portuguese corpus comprising 331 billion GPT-2 tokens collected from diverse web sources and existing multilingual datasets. We systematically benchmark NorBERTo against Strong baselines on semantic similarity, textual entailment and classification tasks using standardized datasets such as ASSIN 2 and PLUE. On PLUE, NorBERTo-large achieves the best results among the encoder models we evaluated, notably reaching 0.9191 F1 on MRPC and 0.7689 accuracy on RTE. On ASSIN 2, NorBERTo-large attains the highest entailment F1 (~0.904) among all encoders considered, although Albertina-900M and BERTimbau-large still hold an advantage. To the best of our knowledge, Aurora-PT is currently the largest openly available monolingual Portuguese corpus, surpassing previous resources. NorBERTo provides a modern, mid-sized encoder designed for realistic deployment scenarios: it is straight-forward to fine-tune, efficient to serve, and well suited as a backbone for retrieval-augmented generation and other downstream Portuguese NLP systems. 

---
# Putting HUMANS first: Efficient LAM Evaluation with Human Preference Alignment 

**Authors**: Woody Haosheng Gan, William Held, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00022)  

**Abstract**: The rapid proliferation of large audio models (LAMs) demands efficient approaches for model comparison, yet comprehensive benchmarks are costly. To fill this gap, we investigate whether minimal subsets can reliably evaluate LAMs while reducing costs and data redundancy. Analyzing 10 subset selection methods with 18 audio models across 40 tasks covering major LAM evaluation dimensions, we show that subsets of just 50 examples (0.3% of data) can achieve over 0.93 Pearson correlation with full benchmark scores. To understand how well these scores align with what practitioners ultimately care about, user satisfaction, we collect 776 human preference ratings from realistic voice assistant conversations, finding that both subsets and full benchmark achieve only 0.85 correlation with human. To better predict preferences, we trained regression models on these selected subsets, achieving 0.98 correlation -- outperforming regression models trained on both random subsets and the full benchmark. This demonstrates that in regression modeling, well-curated subsets outpredict the full benchmark, showing quality over quantity. We open-source these regression-weighted subsets as the HUMANS benchmark, an efficient proxy for LAM evaluation that captures both benchmark performance and user preferences. 

---
# Can Coding Agents Reproduce Findings in Computational Materials Science? 

**Authors**: Ziyang Huang, Yi Cao, Ali K. Shargh, Jing Luo, Ruidong Mei, Mohd Zaki, Zhan Liu, Wyatt Bunstine, William Jurayj, Somdatta Goswami, Tyrel McQueen, Michael Shields, Jaafar El-Awady, Paulette Clancy, Benjamin Van Durme, Nicholas Andrews, William Walden, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00803)  

**Abstract**: Large language models are increasingly deployed as autonomous coding agents and have achieved remarkably strong performance on software engineering benchmarks. However, it is unclear whether such success transfers to computational scientific workflows, where tasks require not only strong coding ability, but also the ability to navigate complex, domain-specific procedures and to interpret results in the context of scientific claims. To address this question, we present AutoMat, a benchmark for evaluating LLM-based agents' ability to reproduce claims from computational materials science. AutoMat poses three interrelated challenges: recovering underspecified computational procedures, navigating specialized toolchains, and determining whether the resulting evidence supports a claim. By working closely with subject matter experts, we curate a set of claims from real materials science papers to test whether coding agents can recover and execute the end-to-end workflow needed to support (or undermine) such claims. We then evaluate multiple representative coding agent settings across several foundation models. Our results show that current LLM-based agents obtain low overall success rates on AutoMat, with the best-performing setting achieving a success rate of only 54.1%. Error analysis further reveals that agents perform worst when workflows must be reconstructed from paper text alone and that they fail primarily due to incomplete procedures, methodological deviations, and execution fragility. Taken together, these findings position AutoMat as both a benchmark for computational scientific reproducibility and a tool for diagnosing the current limitations of agentic systems in AI-for-science settings. 

---
# RunAgent: Interpreting Natural-Language Plans with Constraint-Guided Execution 

**Authors**: Arunabh Srivastava, Mohammad A., Khojastepour, Srimat Chakradhar, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2605.00798)  

**Abstract**: Humans solve problems by executing targeted plans, yet large language models (LLMs) remain unreliable for structured workflow execution. We propose RunAgent, a multi-agent plan execution platform that interprets natural-language plans while enforcing stepwise execution through constraints and rubrics. RunAgent bridges the expressiveness of natural language with the determinism of programming via an agentic language with explicit control constructs (e.g., \texttt{IF}, \texttt{GOTO}, \texttt{FORALL}). Beyond verifying syntactic and semantic verification of the step output, which is performed based on the specific instruction of each step, RunAgent autonomously derives and validates constraints based on the description of the task and its instance at each step. RunAgent also dynamically selects among LLM-based reasoning, tool usage, and code generation and execution (e.g., in Python), and incorporates error correction mechanisms to ensure correctness. Finally, RunAgent filters the context history by retaining only relevant information during the execution of each step. Evaluations on Natural-plan and SciBench Datasets demonstrate that RunAgent outperforms baseline LLMs and state-of-the-art PlanGEN methods. 

---
# When RAG Chatbots Expose Their Backend: An Anonymized Case Study of Privacy and Security Risks in Patient-Facing Medical AI 

**Authors**: Alfredo Madrid-García, Miguel Rujas  

**Link**: [PDF](https://arxiv.org/pdf/2605.00796)  

**Abstract**: Background: Patient-facing medical chatbots based on retrieval-augmented generation (RAG) are increasingly promoted to deliver accessible, grounded health information. AI-assisted development lowers the barrier to building them, but they still demand rigorous security, privacy, and governance controls. Objective: To report an anonymized, non-destructive security assessment of a publicly accessible patient-facing medical RAG chatbot and identify governance lessons for safe deployment of generative AI in health. Methods: We used a two-stage strategy. First, Claude Opus 4.6 supported exploratory prompt-based testing and structured vulnerability hypotheses. Second, candidate findings were manually verified using Chrome Developer Tools, inspecting browser-visible network traffic, payloads, API schemas, configuration objects, and stored interaction data. Results: The LLM-assisted phase identified a critical vulnerability: sensitive system and RAG configuration appeared exposed through client-server communication rather than restricted server-side. Manual verification confirmed that ordinary browser inspection allowed collection of the system prompt, model and embedding configuration, retrieval parameters, backend endpoints, API schema, document and chunk metadata, knowledge-base content, and the 1,000 most recent patient-chatbot conversations. The deployment also contradicted its privacy assurances: full conversation records, including health-related queries, were retrievable without authentication. Conclusions: Serious privacy and security failures in patient-facing RAG chatbots can be identified with standard browser tools, without specialist skills or authentication; independent review should be a prerequisite for deployment. Commercial LLMs accelerated this assessment, including under a false developer persona; assistance available to auditors is equally available to adversaries. 

---
# LASE: Language-Adversarial Speaker Encoding for Indic Cross-Script Identity Preservation 

**Authors**: Venkata Pushpak Teja Menta  

**Link**: [PDF](https://arxiv.org/pdf/2605.00777)  

**Abstract**: A speaker encoder used in multilingual voice cloning should treat the same speaker identically regardless of which script the audio was uttered in. Off-the-shelf encoders do not, and the failure is accent-conditional. On a 1043-pair Western-accented voice corpus across English, Hindi, Telugu, and Tamil, WavLM-base-plus-sv loses 0.082 absolute cosine similarity when the same voice changes script and ECAPA-TDNN loses 0.105. On a 1369-pair Indian-accented voice corpus, the gap shrinks to 0.006 (WavLM-SV) and 0.044 (ECAPA-TDNN). The leak is largest where it matters most for cross-script TTS: when a system projects a non-Indic-trained voice into Indic scripts. We present LASE (Language-Adversarial Speaker Encoder), a small projection head over frozen WavLM-base-plus trained with two losses: a supervised contrastive loss over voice identity, and a gradient-reversal cross-entropy against a 4-language classifier that pushes the embedding to be language-uninformative while remaining speaker-informative. Trained on 1118 quality-gated cross-script pairs synthesised from 8 commercial multilingual voices, LASE's residual gap is consistent with zero on both corpora (Delta = 0.013 Western, Delta = 0.026 Indian; both bootstrap 95% CIs include zero) and amplifies the cross-script-vs-floor margin 2.4-2.7x over both baselines. An ECAPA+GRL ablation shows the GRL objective improves either backbone but the WavLM choice contributes too. In synthetic multi-speaker diarisation, LASE matches ECAPA-TDNN on cross-script speaker recall (0.788 vs 0.789) with ~100x less training data. We release the r1 checkpoint, both corpora, and the bootstrap recipe. 

---
# Adaptive Querying with AI Persona Priors 

**Authors**: Kaizheng Wang, Yuhang Wu, Assaf Zeevi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00696)  

**Abstract**: We study adaptive querying for learning user-dependent quantities of interest, such as responses to held-out items and psychometric indicators, within tight question budgets. Classical Bayesian design and computerized adaptive testing typically rely on restrictive parametric assumptions or expensive posterior approximations, limiting their use in heterogeneous, high-dimensional, and cold-start settings. We introduce a persona-induced latent variable model that represents a user's state through membership in a finite dictionary of AI personas, each offering response distributions produced by a large language model. This yields expressive priors with closed-form posterior updates and efficient finite-mixture predictions, enabling scalable Bayesian design for sequential item selection. Experiments on synthetic data and WorldValuesBench demonstrate that persona-based posteriors deliver accurate probabilistic predictions and an interpretable adaptive elicitation pipeline. 

---
# EGREFINE: An Execution-Grounded Optimization Framework for Text-to-SQL Schema Refinement 

**Authors**: Jiaqian Wang, Yutao Qi, Wenjin Hou, Yu Pang, Rui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00628)  

**Abstract**: Text-to-SQL enables non-expert users to query databases in natural language, yet real-world schemas often suffer from ambiguous, abbreviated, or inconsistent naming conventions that degrade model accuracy. Existing approaches treat schemas as fixed and address errors downstream. In this paper, we frame schema refinement as a constrained optimization problem: find a renaming function that maximizes downstream Text-to-SQL execution accuracy while preserving query equivalence through database views. We analyze the computational hardness of this problem, which motivates a column-wise greedy decomposition, and instantiate it as EGRefine: a four-phase pipeline that screens ambiguous columns, generates context-aware candidate names, verifies them through execution-grounded feedback, and materializes the result as non-destructive SQL views. The pipeline carries two structural properties: column-local non-degradation, ensured by the conservative selection rule in the verification phase, and database-level query equivalence, ensured by the view-based materialization phase. Together they make the resulting refinement safe by construction at the column level, with cross-column and prompt-level interactions handled empirically rather than analytically. Across controlled schema-degradation, real-world, and enterprise benchmarks, EGRefine recovers accuracy lost to schema naming noise where applicable and correctly abstains where the underlying task exceeds current Text-to-SQL capabilities, with refined schemas transferring across model families to enable refine-once, serve-many-models deployment. Code and data are publicly available at this https URL. 

---
# LLM-Oriented Information Retrieval: A Denoising-First Perspective 

**Authors**: Lu Dai, Liang Sun, Fanpu Cao, Ziyang Rao, Cehao Yang, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.00505)  

**Abstract**: Modern information retrieval (IR) is no longer consumed primarily by humans but increasingly by large language models (LLMs) via retrieval-augmented generation (RAG) and agentic search. Unlike human users, LLMs are constrained by limited attention budgets and are uniquely vulnerable to noise; misleading or irrelevant information is no longer just a nuisance, but a direct cause of hallucinations and reasoning failures. In this perspective paper, we argue that denoising-maximizing usable evidence density and verifiability within a context window-is becoming the primary bottleneck across the full information access pipeline. We conceptualize this paradigm shift through a four-stage framework of IR challenges: from inaccessible to undiscoverable, to misaligned, and finally to unverifiable. Furthermore, we provide a pipeline-organized taxonomy of signal-to-noise optimization techniques, spanning indexing, retrieval, context engineering, verification, and agentic workflow. We also present research works on information denoising in domains that rely heavily on retrieval such as lifelong assistant, coding agent, deep research, and multimodal understanding. 

---
# "What Are You Really Trying to Do?": Co-Creating Life Goals from Everyday Computer Use 

**Authors**: Shardul Sapkota, Matthew Jörke, Zane Sabbagh, Omar Shaikh, Grace Wang, James A. Landay  

**Link**: [PDF](https://arxiv.org/pdf/2605.00497)  

**Abstract**: Recent advances in user modeling make it feasible to conduct open-ended inference over a person's everyday computer use. Despite longstanding visions of systems that deeply understand our actions and the purposes they serve in our lives, existing systems only capture what a person is doing in the moment -- not why they are doing it -- limiting these systems to surface-level support. We introduce striving co-creation, a process for inferring broader life goals from unstructured observations of computer use. Grounded in Activity Theory and Emmons' personal strivings framework, our system progressively constructs a hierarchical representation of a person's activities. Crucially, strivings are difficult to fully resolve from observation alone, as the same action can be driven by many different goals. Our system therefore supports an editing interface that gives people agency over how they are understood by the system, feeding their corrections back into subsequent rounds of striving induction. In a week-long field deployment (N=14), we find that our co-creation process produces strivings that are representative of participants' long-term goals and gives them greater agency than baseline methods. 

---
# On the Role of Artificial Intelligence in Human-Machine Symbiosis 

**Authors**: Ching-Chun Chang, Yuchen Guo, Hanrui Wang, Timo Spinde, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00440)  

**Abstract**: The evolution of artificial intelligence (AI) has rendered the boundary between humanity and computational machinery increasingly ambiguous. In the presence of more interwoven relationships within human-machine symbiosis, the very notion of AI-generated information becomes difficult to define, as such information arises not from either humans or machines in isolation, but from their mutual shaping. Therefore, a more pertinent question lies not merely in whether AI has participated, but in how it has participated. In general, the role assumed by AI is often specified, either implicitly or explicitly, in the input prompt, yet becomes less apparent or altogether unobservable when the generated content alone is available. Once detached from the dialogue context, the functional role may no longer be traceable. This study considers the problem of tracing the functional role played by AI in natural language generation. A methodology is proposed to infer the latent role specified by the prompt, embed this role into the content during the probabilistic generation process and subsequently recover the nature of AI participation from the resulting text. Experimentation is conducted under a representative scenario in which AI acts either as an assistive agent that edits human-written content or as a creative agent that generates new content from a brief concept. The experimental results support the validity of the proposed methodology in terms of discrimination between roles, robustness against perturbations and preservation of linguistic quality. We envision that this study may contribute to future research on the ethics of AI with regard to whether AI has been used fairly, transparently and appropriately. 

---
# Rethinking LLM Ensembling from the Perspective of Mixture Models 

**Authors**: Jiale Fu, Yuchu Jiang, Peijun Wu, Chonghan Liu, Joey Tianyi Zhou, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00419)  

**Abstract**: Model ensembling is a well-established technique for improving the performance of machine learning models. Conventionally, this involves averaging the output distributions of multiple models and selecting the most probable label. This idea has been naturally extended to large language models (LLMs), yielding improved performance but incurring substantial computational cost. This inefficiency stems from directly applying conventional ensemble implementation to LLMs, which require a separate forward pass for each model to explicitly compute the ensemble distribution. In this paper, we propose the Mixture-model-like Ensemble (ME). By reinterpreting the ensemble as a mixture model, ME stochastically selects a single model at each step to generate the next token, thereby avoiding the need to explicitly compute the full ensemble distribution. ME is mathematically equivalent to sampling from the ensemble distribution, but requires invoking only one model, making it 1.78x-2.68x faster than conventional ensemble. Furthermore, this perspective connects LLM ensembling and token-level routing methods, suggesting that LLM ensembling is a special case of routing methods. Our findings open new avenues for efficient LLM ensembling and motivate further exploration of token-level routing strategies for LLMs. Our code is available at this https URL. 

---
# FollowTable: A Benchmark for Instruction-Following Table Retrieval 

**Authors**: Rihui Jin, Yuchen Lu, Ting Zhang, Jun Wang, Kuicai Dong, Zhaocheng Du, Dongping Liu, Gang Wang, Yong Liu, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00400)  

**Abstract**: Table Retrieval (TR) has traditionally been formulated as an ad-hoc retrieval problem, where relevance is primarily determined by topical semantic similarity. With the growing adoption of LLM-based agentic systems, access to structured data is increasingly instruction-driven, where relevance is conditional on explicit content and schema constraints rather than topical similarity alone. We therefore formalize Instruction-Following Table Retrieval (IFTR), a new task that requires models to jointly satisfy topical relevance and fine-grained instruction constraints. We identify two core challenges in IFTR: (i) sensitivity to content scope, such as inclusion and exclusion constraints, and (ii) awareness of schema-grounded requirements, including column semantics and representation granularity--capabilities largely absent in existing retrievers. To support systematic evaluation, we introduce FollowTable, the first large-scale benchmark for IFTR, constructed via a taxonomy-driven annotation pipeline. We further propose a new metric, termed the Instruction Responsiveness Score, to evaluate whether retrieval rankings consistently adapt to user instructions relative to a topic-only baseline. Our results indicate that existing retrieval models struggle to follow fine-grained instructions over tabular data. In particular, they exhibit systematic biases toward surface-level semantic cues and remain limited in handling schema-grounded constraints, highlighting substantial room for future improvements. 

---
# ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning 

**Authors**: Zihan Lin, Xiaohan Wang, Jie Cao, Jiajun Chai, Li Wang, Xiaodong Lu, Wei Lin, Ran He, Guojun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00380)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) enhances reasoning of Large Language Models (LLMs) but usually exhibits limited generation diversity due to the over-incentivization of positive rewards. Although methods like Negative Sample Reinforcement (NSR) mitigate this issue by upweighting penalty from negative samples, they may suppress the semantic distributions shared between positive and negative responses. To boost reasoning ability without losing diversity, this paper proposes negative sample projection Residual Reinforcement Learning (ResRL) that decouples similar semantic distributions among positive and negative responses. We theoretically link Lazy Likelihood Displacement (LLD) to negative-positive head-gradient interference and derive a single-forward proxy that upper-bounds representation alignment to guide conservative advantage reweighting. ResRL then projects negative-token hidden representations onto an SVD-based low-rank positive subspace and uses projection residuals to modulate negative gradients, improving reasoning while preserving diversity and outperforming strong baselines on average across twelve benchmarks spanning Mathematics, Code, Agent Tasks, and Function Calling. Notably, ResRL surpasses NSR on mathematical reasoning by 9.4\% in Avg@16 and 7.0\% in Pass@128. Code is available at this https URL. 

---
# Uniform-Correct Policy Optimization: Breaking RLVR's Indifference to Diversity 

**Authors**: Anamika Lochab, Bolian Li, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00365)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has achieved substantial gains in single-attempt accuracy (Pass@1) on reasoning tasks, yet often suffers from reduced multi-sample coverage (Pass@K), indicating diversity collapse. We identify a structural cause for this degradation: common RLVR objectives, such as GRPO, are indifferent to how probability mass is distributed among correct solutions. Combined with stochastic training dynamics, this indifference induces a self-reinforcing collapse, in which probability mass concentrates on a narrow subset of correct outputs while alternative valid solutions are suppressed. We formalize this collapse mechanism and further characterize the optimal policy structure under two complementary criteria: robustness and entropy-regularized optimality, which identify the Uniform-Correct Policy as uniquely optimal. Motivated by this analysis, we propose Uniform-Correct Policy Optimization (UCPO), a modification to GRPO that adds a conditional uniformity penalty on the policy's distribution over correct solutions. The penalty redistributes gradient signal toward underrepresented correct responses, encouraging uniform allocation of probability mass within the correct set. Across three models (1.5B-7B parameters) and five mathematical reasoning benchmarks, UCPO improves Pass@K and diversity while maintaining competitive Pass@1, achieving up to +10\% absolute improvement on AIME24 at Pass@64 and up to 45\% higher equation-level diversity within the correct set. The code is available at this https URL. 

---
# Block-wise Codeword Embedding for Reliable Multi-bit Text Watermarking 

**Authors**: Joeun Kim, HoEun Kim, Dongsup Jin, Young-Sik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.00348)  

**Abstract**: Recent multi-bit watermarking methods for large language models (LLMs) prioritize capacity over reliability, often conflating decoding with detection. Our analysis reveals that existing ECC-based extractors suffer from catastrophic false positive rates (FPR), and applying rejection thresholds merely collapses detection sensitivity (TPR) to random guessing. To resolve this structural limitation, we propose \textbf{BREW} (Block-wise Reliable Embedding for Watermarking), a framework shifting the paradigm to \emph{designated verification}. BREW employs a two-stage mechanism: (i) \textbf{blind message estimation} via independent block voting, followed by (ii) \textbf{window-shifting verification} that rigorously validates the payload against local edits. Experiments demonstrate that BREW achieves a TPR of 0.965 with an FPR of 0.02 under 10\% synonym substitution, demonstrating that the high-FPR issue is not an inherent trade-off of multi-bit watermarking, but a solvable structural flaw of prior decoding-centric designs. Our framework is model-agnostic and theoretically grounded, providing a scalable solution for reliable forensic deployment. 

---
# Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via Reinforcement Learning 

**Authors**: Chengshuai Shi, Wenzhe Li, Xinran Liang, Yizhou Lu, Wenjia Yang, Ruirong Feng, Seth Karten, Ziran Yang, Zihan Ding, Gabriel Sarch, Danqi Chen, Karthik Narasimhan, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00347)  

**Abstract**: Given the rapidly growing capabilities of vision-language models (VLMs), extending them to interactive decision-making tasks such as video games has emerged as a promising frontier. However, existing approaches either rely on large-scale supervised fine-tuning (SFT) on human trajectories or apply reinforcement learning (RL) only in relatively short-horizon settings (typically around 20--30 turns). In this work, we study RL-based training of VLMs for long-horizon decision-making in Super Mario Land, a visually grounded environment requiring 100+ turns of interaction with coordinated perception, reasoning, and action. We begin with a systematic investigation of key algorithmic components and propose an adapted variant of PPO with a lightweight turn-level critic, which substantially improves training stability and sample efficiency over critic-free methods such as GRPO and Reinforce++. We further show that pretrained VLMs provide strong action priors, significantly improving sample efficiency during RL training and reducing the need for manual design choices such as action engineering, compared to classical deep RL trained from scratch. Building on these insights, we introduce Odysseus, an open training framework for VLM agents, achieving substantial gains across multiple levels of the game and at least 3 times average game progresses than frontier models. Moreover, the trained models exhibit consistent improvements under both in-game and cross-game generalization settings, while maintaining general-domain capabilities. Overall, our results identify key ingredients for making RL stable and effective in long-horizon, multi-modal settings, and provide practical guidance for developing VLMs as embodied agents. 

---
# AgentFloor: How Far Up the tool use Ladder Can Small Open-Weight Models Go? 

**Authors**: Ranit Karmakar, Jayita Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.00334)  

**Abstract**: Production agentic systems make many model calls per user request, and most of those calls are short, structured, and routine. This raises a practical routing question that existing evaluations do not directly answer: which parts of an agent workflow truly require large frontier intelligence, and which can be handled by smaller models? We introduce AgentFloor, a deterministic 30-task benchmark organized as a six-tier capability ladder, spanning instruction following, tool use, multi-step coordination, and long-horizon planning under persistent constraints. We evaluate 16 open-weight models, from 0.27B to 32B parameters, alongside GPT-5 across 16,542 scored runs. Our results reveal a clear boundary of model necessity. Small and mid-sized open-weight models are already sufficient for much of the short-horizon, structured tool use work that dominates real agent pipelines, and in aggregate, the strongest open-weight model matches GPT-5 on our benchmark while being substantially cheaper and faster to run. The gap appears most clearly on long-horizon planning tasks that require sustained coordination and reliable constraint tracking over many steps, where frontier models still hold an advantage, though neither side reaches strong reliability. We also find that this boundary is not explained by scale alone: some failures respond to targeted interventions, but the effects are model-specific rather than universal. These findings suggest a practical design principle for agentic systems: use smaller open-weight models for the broad base of routine actions, and reserve large frontier models for the narrower class of tasks that truly demand deeper planning and control. We release the benchmark, harness, sweep configurations, and full run corpus. 

---
# Borrowed Geometry: Computational Reuse of Frozen Text-Pretrained Transformer Weights Across Modalities 

**Authors**: Abay Bektursun  

**Link**: [PDF](https://arxiv.org/pdf/2605.00333)  

**Abstract**: Frozen Gemma 4 31B weights pretrained exclusively on text tokens, unmodified, transfer across modality boundaries through a thin trainable interface. (1) OGBench scene-play-singletask-task1-v0: $+4.33$pt over published GCIQL at $n=3$ with std 0.74 -- a published-SOTA win on a robotic manipulation task the substrate has never seen. (2) D4RL Walker2d-medium-v2: Decision-Transformer parity ($76.2 \pm 0.8$, $n=3$) at $0.43\times$ DT's trainable count, with the frozen substrate compressing to a 5L slice ($+1.66$pt over the 6L baseline at $n=3$). (3) Associative recall as the cleanest pretraining-load-bearing case: the frozen slice + a 113K-parameter linear interface reaches L30 best-checkpoint per-bit error 0.0505 ($n=2$); a 6.36M-parameter from-scratch trained transformer at matched capacity ($1/\sqrt{d_k}$ scaling, two seeds, LR sweep) cannot solve the task at all under the protocol (best L30 = 0.4395), an $8.7\times$ advantage. Architecture-alone falsifications: a frozen random transformer with correct $1/\sqrt{d_k}$ scaling stays at random-chance loss for 50k steps; a random-init Gemma slice fails OGBench cube-double-play-task1 entirely (0.89% across $n=3$ where pretrained reaches 60%). A dual-measurement protocol -- text-activation probing on 95 English sentences plus task-ablation on a non-language target -- names individual heads independently identifiable on both protocols: head L26.28 scores $3.7\times$ the slice mean for English token-copying and is the #2 most-critical head for binary copy ablation ($\Delta$ L30 $= +0.221$); three further heads (L27.28, L27.2, L27.3) classify by the same protocol. The mechanism is single-model and the cross-modality results are single-task within their respective benchmarks; cross-model replication is structurally constrained because Gemma 4 31B is the only model on the small-scale Pareto frontier as of April 2026. 

---
# Alethia: A Foundational Encoder for Voice Deepfakes 

**Authors**: Yi Zhu, Brahmi Dwivedi, Jayaram Raghuram, Surya Koppisetti  

**Link**: [PDF](https://arxiv.org/pdf/2605.00251)  

**Abstract**: Existing voice deepfake detection and localization models rely heavily on representations extracted from speech foundation models (SFMs). However, downstream finetuning has now reached a state of diminishing returns. In this paper, we shift the focus to pretraining and propose a novel recipe that combines bottleneck masked embedding prediction with flow-matching based spectrogram reconstruction. The outcome, Alethia, is the first foundational audio encoder for various voice deepfake detection and localization tasks. We evaluate on $5$ different tasks with $56$ benchmark datasets, and note Alethia significantly outperforms state-of-the-art SFMs with superior robustness to real-world perturbations and zero-shot generalization to unseen domains (e.g., singing deepfakes). We also demonstrate the limitation of discrete targets in masked token prediction, and show the importance of continuous embedding prediction and generative pretraining for capturing deepfake artifacts. 

---
# State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence for Latent Space Reasoning 

**Authors**: Thea Aviss  

**Link**: [PDF](https://arxiv.org/pdf/2605.00206)  

**Abstract**: Current transformers discard their rich latent residual stream between positions, reconstructing latent reasoning context at each new position and leaving potential reasoning capacity untapped. The State Stream Transformer (SST) V2 enables parameter-efficient reasoning in continuous latent space through an FFN-driven nonlinear recurrence at each decoder layer, where latent states are streamed horizontally across the full sequence via a learned blend. This same mechanism supports continuous latent deliberation per position at inference time, dedicating additional FLOPs to exploring abstract reasoning before committing to a token. A two-pass parallel training procedure resolves the sequential dependency of the recurrence to allow compute-efficient training. Hidden state analysis shows the state stream facilitates reasoning through exploration of distinct semantic basins in continuous latent space, where transitions at content-dependent positions move the model into a substantially different Bayesian posterior, directly influencing the latent space at future positions. We also find, via a learned probe, that at the first generated token position, the latent state already predicts whether the eventual answer will survive or break under additional latent computation for every subsequent position. Co-trained into an existing 27B backbone using only a small dataset of GSM8K examples, the SST delivers a +15.15 point gain over a fine-tuning-matched baseline on out-of-distribution GPQA-Diamond and cuts that same baseline's remaining GSM8K errors by 46%, together showing that the reasoning improvement is attributable to the architectural mechanism rather than scale or training data. On GPQA-Diamond, the resulting 27B SST also achieves higher accuracy than several larger open-weight and proprietary systems, including open-weight models up to 25 times larger. 

---
# RouteProfile: Elucidating the Design Space of LLM Profiles for Routing 

**Authors**: Jingjun Xu, Hongji Pu, Tao Feng, Haozhen Zhang, Jiaxuan You, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00180)  

**Abstract**: As the large language model (LLM) ecosystem expands, individual models exhibit varying capabilities across queries, benchmarks, and domains, motivating the development of LLM routing. While prior work has largely focused on router mechanism design, LLM profiles, which capture model capabilities, remain underexplored. In this work, we ask: How does LLM profile design affect routing performance across different routers? Addressing this question helps clarify the role of profiles in routing, disentangle profile design from router design, and enable fairer comparison and more principled development of routing systems. To this end, we view LLM profiling as a structured information integration problem over heterogeneous interaction histories. We develop a general design space of LLM profiles, named RouteProfile, along four key dimensions: organizational form, representation type, aggregation depth, and learning configuration. Through systematic evaluation across three representative routers under both standard and new-LLM generalization settings, we show that: (1) structured profiles consistently outperform flat ones; (2) query-level signals are more reliable than coarse domain-level signals; and (3) generalization to newly introduced models benefits most from structured profiles under trainable configurations. Overall, our work highlights LLM profile design as an important direction for future routing research. 

---
# Wasserstein Distributionally Robust Regret Optimization for Reinforcement Learning from Human Feedback 

**Authors**: Yikai Wang, Shang Liu, Jose Blanchet  

**Link**: [PDF](https://arxiv.org/pdf/2605.00155)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has become a core post-training step for aligning large language models, yet the reward signal used in RLHF is only a learned proxy for true human utility. From an operations research perspective, this creates a decision problem under objective misspecification: the policy is optimized against an estimated reward, while deployment performance is determined by an unobserved objective. The resulting gap leads to reward over-optimization, or Goodharting, where proxy reward continues to improve even after true quality deteriorates. Existing mitigations address this problem through uncertainty penalties, pessimistic rewards, or conservative constraints, but they can be computationally burdensome and overly pessimistic. We propose Wasserstein distributionally robust regret optimization (DRRO) for RLHF. Instead of pessimizing worst-case value as in standard DRO, DRRO pessimizes worst-case regret relative to the best policy under the same plausible reward perturbation. We study the promptwise problem through a simplex allocation model and show that, under an $\ell_1$ ambiguity set, the inner worst-case regret admits an exact solution and the optimal policy has a water-filling structure. These results lead to a practical policy-gradient algorithm with a simple sampled-bonus interpretation and only minor changes to PPO/GRPO-style RLHF training. The framework also clarifies theoretically why DRRO is less pessimistic than DRO, and our experiments show that DRRO mitigates over-optimization more effectively than existing baselines while standard DRO is systematically over-pessimistic. 

---
# Technical Report: Activation Residual Hessian Quantization (ARHQ) for Low-Bit LLM Quantization 

**Authors**: YiFeng Wang, Zhun Sun, Keisuke Sakaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00140)  

**Abstract**: We present Activation Residual Hessian Quantization (ARHQ), a post-training weight splitting method designed to mitigate error propagation in low-bit activation-weight quantization. By constructing an input-side residual Hessian from activation quantization residuals (G_x), ARHQ analytically identifies and isolates error-sensitive weight directions into a high-precision low-rank branch. This is achieved via a closed-form truncated SVD on the scaled weight matrix W G^{1/2}_x . Experimental results on Qwen3-4B-Thinking-2507 demonstrate that ARHQ significantly improves layer-wise SNR and preserves downstream reasoning performance on ZebraLogic even under aggressive quantization. The code is available at this https URL. 

---
# MoDAl: Self-Supervised Neural Modality Discovery via Decorrelation for Speech Neuroprosthesis 

**Authors**: Yuanhao Chen, Peter Chin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00025)  

**Abstract**: Speech neuroprosthesis systems decode intended speech from neural activity in the absence of audible output, offering a path to restoring communication for individuals with speech-impairing conditions. Current approaches decode predominantly from motor cortical areas, discarding others -- such as area 44, part of Broca's area -- that may encode complementary linguistic information. We introduce MoDAl (Modality Decorrelation and Alignment), a framework that discovers complementary neural modalities through the interplay of two objectives in a shared projection space. A contrastive loss aligns each of several parallel brain encoders with the text embeddings of a pretrained large language model (LLM), while a decorrelation loss prevents the encoders from coalescing to duplicative representations. We prove that these objectives are in productive tension: Contrastive alignment induces transitive modality coalescence, which decorrelation must counteract for the framework to discover diverse neurolinguistic modalities. On the Brain-to-Text Benchmark '24, MoDAl reduces word error rate (WER) from 26.3% to 21.6% compared to the previous best end-to-end method, with the gain from incorporating previously discarded area 44 signals arising entirely from the decorrelation mechanism. Analysis of the discovered modalities reveals functional specialization: Encoders receiving area 44 input capture structural and syntactic properties (sentence length, grammatical voice, wh-words), consistent with the neurolinguistic understanding of Broca's area. 

---
# Exploring LLM biases to manipulate AI search overview 

**Authors**: Roman Smirnov  

**Link**: [PDF](https://arxiv.org/pdf/2605.00012)  

**Abstract**: Modern large language models (LLMs) are used in many business applications in general, and specifically in web search systems and applications that generate overviews of search results - LLM Overview systems. Such systems are using an LLM to select most relevant sources from search results and generate an answer to the user's query. It is known from many studies that LLMs have different biases, in LLM Overview application both the source selection and answer generation stages may be affected by the biases of LLMs (here we are focusing mainly on the selection stage). This research is focused on investigating the presence of the biases in LLM Overview systems and on biases exploitation to manipulate LLM Overview results. Here we train a small language model using reinforcement learning to rewrite search snippets to increase their likelihood of being preferred by an LLM Overview. Our experimental setup intentionally restricts the policy to operate only on snippets and limits reward-hacking strategies, reflecting realistic constraints of web search environments. The results prove that LLM Overview systems have biases and that reinforcement learning in most of the cases can optimize snippet's content to manipulate LLM Overview results. We also prove that LLM Overview selections are driven by comparative rather than absolute advantages among candidate sources. In addition, we examine safety aspects of LLM Overview manipulation possibilities and show that context poisoning attacks can lead to inaccurate or harmful results. 

---
