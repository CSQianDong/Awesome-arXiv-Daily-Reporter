# Learning When to Trust via Selective Context Preference Optimization 

**Authors**: Xian Sun, Wei Chow, Yingshuo Wang, Junhao Liu, Wei Gao, Qing Wu, Lingdong Kong  

**Link**: [PDF](https://arxiv.org/pdf/2608.06377)  

**Abstract**: Language models increasingly condition their answers on external signals, and a single misleading one can turn a correct answer wrong. The obvious remedy, training models to resist such signals, hides a failure mode: a model that ignores all context looks robust yet is useless when the context is worth trusting. We recast the problem as selective trust and introduce MIST, a human-annotated benchmark that renders each reasoning item under four matched conditions (clean, misleading, correct-context, and irrelevant-context), together with SC2W, a paired metric counting how often a misleading signal flips a clean-correct answer to wrong. Across a comprehensive benchmark study, we observe that such a susceptibility is universal. We then propose SCOPE, which mines clean-correct/misleading-wrong failures and optimizes a standard Direct Preference Optimization (DPO) objective over matched preference pairs balanced equally across all four conditions, rather than over misleading items alone. Our approach substantially reduces SC2W on popular open-sourced models while preserving accuracy when the added context is clean, correct, or irrelevant. With this work, we argue that models should be judged on selective trust, not on resistance alone. 

---
# The Bitter Lesson of Tool Calling 

**Authors**: Ishan Patel, Sahil Sen, Elias Lumer, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2608.06370)  

**Abstract**: Tool use transforms LLMs into agents that act beyond their training data, and for code-capable models, programmatic tool calling extends this further by replacing rigid JSON calls with scripts that chain and parallelize naturally. However, a systematic evaluation of tools as code on an established benchmark across current and prior model generations under real-world task conditions has not been conducted. In this work, we empirically compare programmatic tool calling (PTC) to native JSON tool calling across 14 language models on BFCL v4. In the programmatic tool calling paradigm, tools are exposed as typed Python stubs that the model invokes through code, with execution and results handled in a single agent turn. Programmatic tool calling matches or exceeds native JSON tool calling in 11 of 14 models on BFCL v4, with the GPT-5.6 family achieving a 10.6% improvement over the JSON tool calling baseline. Further, it matches or outperforms baseline in 13 of 14 models under parallel fan-out, and holds stable under context rot conditions where baseline degrades 2.3% on average. Our results demonstrate that programmatic tool calling is a viable and robust alternative to JSON tool calling, with performance tracking model capability across release generations. 

---
# RP-OPSD: Reasoning-Pivot-Guided On-Policy Self-Distillation for Multilingual Reasoning Transfer 

**Authors**: Xinye Wang, Junxiao Liu, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06347)  

**Abstract**: Multilingual reasoning transfer is crucial for extending reasoning capabilities of large language models (LLMs) beyond high-resource languages. On-policy self-distillation (OPSD) and its variants have emerged as a promising paradigm, providing dense token-level supervision on student-generated rollouts, yet their objectives do not explicitly prioritize reasoning signals most critical to cross-lingual transfer. We characterize that target-language reasoning comprises the generation of both surface text and reasoning pivots, which are decisions that advance or redirect the reasoning process and shape subsequent inference. This motivates concentrating privileged distillation around such pivots. We therefore propose RP-OPSD, Reasoning-Pivot-guided On-Policy Self-Distillation, using the distributional shift between matched teacher views with and without an English reference solution as an operational proxy to guide privileged distillation and reference anchoring. Experiments on mathematical reasoning benchmarks covering 17 languages and multiple difficulty levels show that our method outperforms strong multilingual reasoning baselines and OPSD variants. Further analysis reveals that RP-OPSD concentrates privileged distillation on reasoning-control and problem-condistioned state-update tokens, while downweighting it for tokens that mainly support surface realization. Our code is available at this https URL. 

---
# Benchmarking the Benchmarks: Evaluating Benchmarks for Conversational Agents 

**Authors**: Noam Koren, Roy Bar-Haim, Abigail Goldsteen  

**Link**: [PDF](https://arxiv.org/pdf/2608.06329)  

**Abstract**: Task-oriented conversational agents are evaluated using curated or automatically generated benchmarks, yet benchmark quality is rarely assessed. Poor benchmarks may contain inconsistent tasks, simplistic scenarios, or limited policy coverage, leading to unreliable evaluations. We introduce a reference-free framework that uses LLM judges to assess benchmark consistency, complexity, and policy coverage, while providing actionable diagnostics of weaknesses. We validate the framework by demonstrating agreement with independent human annotations and by evaluating benchmarks generated by LLMs of varying capabilities, as well as benchmarks subjected to controlled quality-degrading perturbations. Across domains and judge models, the proposed metrics consistently distinguish between benchmark quality levels. We further demonstrate the framework's applicability to manually curated benchmarks. Our framework offers a practical approach for evaluating synthetic and manually curated conversational-agent benchmarks. 

---
# Benchmarking and Enhancing LLMs for Rule-Intensive Review of National Standard Documents 

**Authors**: Tao Wang, Qihao Yang, Rongjiao Liang, Lianghong Lin, Haitao Wang, Xinyu Cao, Tianyong Hao  

**Link**: [PDF](https://arxiv.org/pdf/2608.06312)  

**Abstract**: Large language models (LLMs) increasingly support complex professional tasks, yet their capabilities in rule-intensive document review remain insufficiently evaluated. National standard documents, such as China GB/T standards, offer a representative testbed: they are lengthy, highly structured, and governed by explicit rules for scope, terminology, normative wording, and cross-section consistency. Existing benchmarks focus on domain knowledge and question answering, largely overlooking intrinsic quality review for professional documents. Such reviews rely heavily on human experts, making them costly and difficult to scale. To bridge this gap, we introduce GB/T-Bench, the first benchmark for the structured review of national standard documents. Its GB/T Review Taxonomy is a hierarchical schema covering document structure, scope alignment, normative modality, terminology consistency, and normative references, with 25 diagnosable error types. A controllable counterexample generation mechanism combines deterministic rules and constrained LLM rewriting to process 488 documents into 7,306 traceable review error instances for evaluation. We also develop a diagnosis-oriented evaluation protocol requiring exact matches on error location, review dimension, and error type, plus document-level coverage metrics. We further propose GB/T-Reviewer, a multi-agent framework that converts review knowledge into specialized skills and coordinates global inspection, targeted diagnosis, rule scanning, and result verification. Experiments with 14 mainstream LLMs reveal a substantial human-LLM gap: the strongest model achieves only 0.3280 CMCS versus 0.6640 for experts. GB/T-Reviewer raises the best CMCS to 0.5094, showing the value of structured skill coordination for rule-intensive document review. This work paves the way for trustworthy AI in standardization and other high-stakes document domains. 

---
# NeSy-RAG: Neuro-Symbolic RAG for Explainable Question Answering 

**Authors**: Jonas Gann, Michael Gertz  

**Link**: [PDF](https://arxiv.org/pdf/2608.06292)  

**Abstract**: Retrieval-augmented generation (RAG) improves question answering by grounding large language models (LLMs) in external knowledge such as text corpora. However, its reasoning process remains largely opaque: intermediate reasoning steps are difficult to verify and cannot be reliably attributed to specific evidence. Moreover, missing user-specific context is rarely detected systematically, often leading to incomplete or incorrect output.
We propose NeSy-RAG, a modular neuro-symbolic RAG framework that synthesizes attributable Prolog modules from retrieved text chunks. For each chunk, the system generates semantically meaningful predicates that encode Boolean claims, which may depend on user facts. Using joint natural language-code embeddings, predicates are retrieved and composed into Prolog queries. To address incomplete user context, we introduce a symbolic knowledge-gap detection mechanism that identifies missing user facts whose truth values affect the query outcome and automatically triggers follow-up interactions.
Executing the resulting Prolog queries yields deterministic answers together with transparent execution traces that link each reasoning step to its originating source. On the ShARC benchmark, without domain-specific training, NeSy-RAG achieves 61.1% accuracy, outperforming a same-model RAG baseline that achieves 42.8% accuracy. 

---
# Routing Is Least Learnable Where It Is Most Valuable: Bounds on Representation Routing for Web Agents 

**Authors**: Jiaming Wei, Zekun Wu, Adriano Koshiyama, Maria Perez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2608.06171)  

**Abstract**: Web agents observe a browser through text, pixels, or both, and the choice is usually fixed once for all tasks. We measure six observation modes across eight site-model combinations (cells) on VisualWebArena and WebArena and ask what choosing per task would buy. The modes are complementary: each solves tasks the others miss, they fail in structurally different ways, and the best choice reverses between task sets. The obvious prize, an oracle that picks a winning mode for every task, looks large but is inflated by run-to-run noise: rerunning the same mode on the same tasks changes 12-14% of outcomes, so a second run of a mode already in hand gains about as much as adding a new one. What survives is a cost bound: sending only the tasks no mode solves to the cheapest mode cuts cost by 9.5-30.6% in 8 of 8 cells at unchanged success. We then test five routing policies (picking the mode, deciding when to spend on the strong mode, a zero-cost rule read off the task text, a confidence cascade, and pooled cost tiers), and none robustly beats simply fixing one well-chosen mode; the one exception is a fragile result in our sparsest cell. The central obstruction is that routing supervision is produced at the agent's success rate: the weaker the agent, the fewer labels a router gets, exactly where routing would be most valuable. This limit belongs to today's agents rather than to routing itself. Label supply and routing opportunity rise together (correlation 0.95 across cells), so a stronger agent can overturn the result, and we report the rerun noise bands and the full measurement protocol. 

---
# Decolonizing Linguistic Policies in Automated Speech Recognition: A Framework for Cross-Culturally Competent Speech AI 

**Authors**: Jay L. Cunningham, Mark Atta Mensah, Richard Martinez, Joao Vieira da Silva Neto, Efi Dawodu  

**Link**: [PDF](https://arxiv.org/pdf/2608.06141)  

**Abstract**: This paper focuses on automatic speech recognition (ASR) and ASR-mediated voice interfaces that shape access to public services, healthcare, and education. We argue that persistent failures for low-resource, Indigenous, and non-standard language varieties are not only technical errors, but also implicit linguistic policies that reproduce colonial language hierarchies. Drawing on linguistic capital, raciolinguistic ideology, language policy research, and decolonial computing, we show how data, metrics, and model priors determine whose voices become machine-legible. We introduce the Three Harms (3M) taxonomy---Misrecognition, Misalignment, and Mistrust---and a seven-layer situatedness model for linguistic diversity in ASR and ASR-mediated voice interfaces. We then propose a participatory framework and minimum audit protocol for culturally competent ASR, positioning affected communities as co-designers, evaluators, and governance partners. 

---
# Beyond Sequence Order: Syntax-Informed Positional Embeddings for Transformers 

**Authors**: Haris Riaz, Hyungji Kim, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2608.06111)  

**Abstract**: Positional embeddings (PE) in Transformers encode token distance and order but are largely agnostic to \textit{syntactic structure}. We introduce \textbf{S}yntax-\textbf{i}nformed \textbf{P}ositional \textbf{E}mbeddings (\textbf{SiPE}), which learns a lightweight syntactic prior from dependency parses during pretraining and injects it across all three dominant PE families (absolute, relative, rotary), for both encoders and decoders, leaving self-attention and the rest of the architecture untouched. We isolate \emph{where} and \emph{how} the prior should enter the model, and find it depends on the architecture: for autoregressive decoders that use relative PE, the prior is strongest when coupled multiplicatively with the relative-position term of the attention score, outperforming injection into the input embeddings, into self-attention, or into the positional and attention terms jointly---while for encoders it is best added directly to the input embeddings, composing with each encoder's native positional mechanism. We find that models pre-trained with SiPE improve on the SyntaxGym benchmark by up to $10.3\%$ while simultaneously reducing perplexity by $9.0\%$ over a base model with no syntactic supervision---a metric nearly every existing syntax-injection method instead degrades. Crucially, these gains extend beyond syntactic generalization: SiPE also improves real-world language understanding, raising scores on the GLUE benchmark by up to $8.2\%$ over a model trained without it. Unlike existing syntactic language models that marginalize over many parses at inference or discard syntax at runtime, SiPE conditions on a single parse, establishing a new Pareto frontier between syntactic supervision and inference cost. 

---
# Training-Free Token-Level Steering for LLM Personalized Co-Writing 

**Authors**: Wenhao Mao, Chengbin Hou, Weixiao Wang, Jialiang Zhu, Min Liu, Yibin Hao, Hairong Lv  

**Link**: [PDF](https://arxiv.org/pdf/2608.06069)  

**Abstract**: While Large Language Models (LLMs) show great promise for personalization, they often lack specialized domain knowledge. Conventional solutions like fine-tuning struggle with high computational costs and rapid data updates, while Retrieval-Augmented Generation fails to provide fine-grained, token-level steering. Furthermore, chat-based interfaces remain dominant, whereas productive co-writing paradigms have not yet been well exploited beyond the coding domain. To this end, we introduce SteerWrite, a training-free framework designed for personalized co-writing. Our method effectively adapts the base model to specialized domains without gradient updates, with specific designs tailored to small datasets. Experiments demonstrate that SteerWrite achieves state-of-the-art performance across diverse datasets, metrics, and models, significantly reducing human editing effort. 

---
# FormBharo: Designing and Evaluating a Voice Agent for Conversational Form Filling in Rural India 

**Authors**: Aman Dalmia, Sanskriti Midha, Jigar Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2608.06027)  

**Abstract**: In India, almost every social benefit starts with a form, yet the people who need these benefits most are often unable to read or write. Reaching them requires a spoken conversation. Today that work falls to frontline health workers who enroll beneficiaries one at a time, a poor use of stretched capacity. We built FormBharo ("fill the form" in Hindi), a voice agent that fills a structured form over a phone call under tight latency and cost budgets by pairing Large Language Models (LLMs) with deterministic, rule-based validation and flow control. It is being piloted with ARMMAN, an NGO running large-scale maternal and child mobile-health programs in India, to enroll low-income, Hindi-speaking mothers in antenatal and postnatal care. To our knowledge, it is the first voice agent piloted to fill an enrollment form for this population. We openly release FormVoiceAgentBench, a benchmark pairing human-recorded Hindi audio with 3,760 multi-turn conversation tests across 960 simulated calls, to evaluate our agent's components (transcription, extraction, reply generation) and end-to-end form completion under real acoustic variations. Form completion drops by up to ~41 points when LLMs receive error-prone real-speech transcripts instead of reference ones. The rule-based controls recover many turn-level extraction errors, helping smaller, cheaper models match or surpass frontier models on form completion. Component performance does not predict end-to-end performance: GPT-5.5 leads turn-level extraction accuracy on reference transcripts (99.8%) but ranks lower on form completion. Since errors both propagate and cancel across the pipeline, the optimal model choice of models emerges only through end-to-end evaluation. Finally, no single model is best across accuracy, cost, and latency at once, so we use a Pareto-based weighted-sum scalarization to select a deployable configuration balancing the three. 

---
# EpiBench: Can LLMs Understand Epitopes for Antibody Drug Discovery? 

**Authors**: Zirui Wang, Jiaqi Wang, Qinghan Wang, Yuzhi Xu, Gang Du, Tingjun Hou, Odin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06022)  

**Abstract**: Epitopes determine where antibodies bind antigens and shape downstream therapeutic properties such as functional blockade and escape resistance, making epitope understanding central to antibody drug discovery. Although large language models (LLMs) have shown strong biomedical reasoning ability, it remains unclear whether they can infer epitope information directly from antigen and antibody sequences. Existing epitope resources typically focus on isolated prediction tasks or rely on specialized structural settings, while general protein benchmarks do not evaluate epitope-centered decisions across the antibody development workflow. To address this gap, we introduce EpiBench, a closed-book, sequence-based, and automatically scorable benchmark for evaluating epitope reasoning in LLMs. EpiBench contains 1,609 curated samples grounded in structural antibody--antigen contacts, curated functional B-cell assays, and deep mutational scanning escape measurements. It covers five connected tasks: targetable region discovery, antibody-conditioned epitope identification, epitope binning, functional epitope assessment, and antibody escape assessment, with controlled sampling to reduce shortcut-based evaluation artifacts. We evaluate nine general-purpose LLMs and analyze their behavior through task-specific baselines, antigen length stratification, explicit-reasoning comparison, and failure-mode inspection. The results show that current LLMs capture partial epitope-related signals but remain limited in antibody-specific sequence grounding, long-context residue localization, and biologically grounded reasoning. Therefore, EpiBench provides a diagnostic testbed for measuring and improving sequence-aware biomedical LLMs toward reliable LLM-assisted antibody discovery. 

---
# Clinical Communication Processing with Models Trained on LLM-Generated Synthetic Data: A Structured Survey and Novel Application Case Studies 

**Authors**: Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2608.05993)  

**Abstract**: Much clinical value is conveyed not through structured records but through communication: exchanges in which patients describe symptoms, clinicians reason and give instructions, ambulances hand over to emergency departments, and nurses pass on a shift. Such language differs from tabular data because meaning depends on speaker role, intent, causality, uncertainty, omission, and channel noise. Healthcare natural language processing must therefore interpret information as conveyed rather than coded. This requires well-annotated corpora, which are scarce because authentic exchanges are private, fragmented, and costly to annotate. Large language models offer a way forward by transforming clinical sources, such as records, diagnostic labels, symptom lists, or care plans, into written and transcribed communication for downstream models. We present a structured narrative survey organized by source representation, communication form and participants, generation method, and downstream task, complemented by thirteen novel case studies. These build clinical NLP systems for communication channels and languages without labeled real-world data, including EMS pre-arrival reports, field-radio casualty documentation, nurse handoffs, patient-portal triage, and low-resource discharge communication. They show that synthetic communication can bootstrap such systems. Findings include the competitiveness of fine-tuned encoder models over evaluated zero-shot baselines and the value of deliberately degraded communication for robustness. The main limitation is that most studies evaluate on held-out synthetic communication, while train-on-synthetic, test-on-authentic evidence remains limited. We conclude that syn-thetic clinical communication is becoming a practical research resource; establishing it as reusable clinical infrastructure will require authentic-data transfer, safety and external validation. 

---
# Causal Episodic Memory for Feedback-Driven Agent Repair 

**Authors**: Khang Nhat Hoang Vo, Tam Minh Chu, Anh Trac Duc Dinh, Thuyen Vinh Ha Bui, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2608.05906)  

**Abstract**: LLM agents that repair failures often discard successful corrections, forcing later episodes to rediscover similar solutions. We study whether finalized repair outcomes can improve subsequent Text-to-SQL episodes without parameter updates. We introduce MERIT, a training-free agent that maintains an online dual-polarity memory of oracle-verified corrections and observed unsuccessful directions. Under oracle-assisted benchmark feedback, only memories from earlier finalized episodes are eligible for retrieval. A deterministic classifier assigns a coarse failure type, which conditions a hybrid lexical-dense retriever before the frozen model generates each revision. Using Qwen2.5-7B-Instruct with identical initial predictions and repair budgets, MERIT improves execution accuracy over stateless iterative repair from \(66.34\%\) to \(69.79\%\) on Spider and from \(47.35\%\) to \(48.44\%\) on BIRD. Paired analyses provide clear evidence for the Spider gain but weaker evidence on BIRD. MERIT is not reliably separated from untyped dynamic retrieval on either benchmark, while Reflexion-style memory reaches \(51.24\%\) on BIRD at substantially higher inference cost. Ablations show that negative memory contributes modestly, the value of type conditioning and lexical--dense ranking is dataset dependent, and schema-local experience provides the most consistent benefit. These results clarify when causal cross-query memory improves repair and when broader memory representations remain preferable. 

---
# MACRO: Markov Chain Routing of Transformer Layers 

**Authors**: Paweł Batorski, Abtin Pourhadi, Akylgali Aitaza, Przemysław Spurek, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2608.05872)  

**Abstract**: Standard Large Language Models (LLMs) execute layers sequentially. Dynamic layer routing, i.e. search for a different execution path through layers involving layer repetitions, skips and other moves, can improve performance. Existing routing approaches often require updating model weights, running expensive search loops per test instance, or demand ground-truth labels during inference. In this work, we propose Markov Chain Routing of Transformer Layers (MACRO), a framework that learns task-specific routes over LLM architectures without modifying underlying parameters. MACRO models layer routing as a context-dependent Markov policy conditioned on layer indices, computation budget phases, directional displacements, and operator context, supporting skip, repeat, and residual hidden-state addition operations. The Markov route distribution is updated via feedback on training data and decoded using a top-k Viterbi algorithm to isolate high-probability candidate programs. We evaluate MACRO across diverse reasoning and knowledge benchmarks on multiple open-weight LLMs. MACRO achieves a +5.0% average accuracy improvement over the unrouted baselines, with largest gains on small models. We outperform the best dynamic routing approach Dr. LLM by +7.2%, while reducing route-search time 9.4x (from 14.8 to 1.6 hours). Our code is publicly available at this https URL. 

---
# Mapping Similarity Spaces across Embedding Models with Synthetic Query Probing 

**Authors**: Marcin Rozmus, Peter van der Putten  

**Link**: [PDF](https://arxiv.org/pdf/2608.05857)  

**Abstract**: Retrieval-Augmented Generation systems rely on similarity scores to retrieve relevant content, yet scores are not directly comparable across embedding models due to differing geometric properties, complicating model migration and limiting threshold reuse. We study how similarity scores can be related by learning mappings between score distributions rather than embeddings. We introduce Synthetic Query Probing, generating queries from documents to create controlled query-chunk pairs, enabling large-scale, reference-free analysis of cross-model similarity behavior. We evaluate the approach on multiple embedding configurations and learn score conversion functions using linear, isotonic, and quantile mappings. Experiments on SciFact and a proprietary corpus show that while models largely agree on rankings, their absolute scores exhibit systematic distortions. Learned mappings partially align these spaces and improve threshold portability, with isotonic regression performing best. Our results highlight the need for cross-model calibration and position Synthetic Query Probing as a scalable framework for analyzing embedding comparability. 

---
# MameLoshnLM: Yiddish Language Model and Evaluation Benchmark 

**Authors**: Uri Katz, Omer Goldman, Tomasz Limisiewicz, Reut Tsarfaty, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2608.05850)  

**Abstract**: We present MameLoshnLM, the first open-source 8B-parameter language model built specifically for Yiddish. Despite Yiddish's rich textual tradition, its limited digital presence and the scarcity of reliable evaluation resources have constrained progress in Yiddish language modeling. Existing multilingual corpora and benchmarks are often poor proxies for the language, containing substantial amounts of noisy, machine-translated, and misclassified text. We address these gaps by introducing Oytser, a high-quality Yiddish pretraining corpus that combines contemporary web-native sources with literary materials, and Kashes, a multi-task benchmark spanning translation, linguistic analysis, information extraction, and language understanding. Using these resources, we continue pretraining Llama 3.1 8B to obtain MameLoshnLM. Across the tasks in the benchmark, MameLoshnLM outperforms open baselines of similar scale. Our analyses show that these gains are not only quantitative: relative to general-purpose multilingual models, MameLoshnLM better captures language-defining lexical and morphological patterns, pointing to a broader failure mode of noisy web-scale multilingual data for low-resource languages. Our results provide both a foundation for Yiddish NLP and a practical template for language model development in historically rich but digitally underrepresented languages. 

---
# Enhancing Social Intelligence in LLMs with Hierarchical Reasoning and Utterance-Level Goal Rewarding 

**Authors**: Xiaofeng Wang, Kakam Chong, Shuai Xiao, DeXin Kong, Qingyuan Tian, Chen Ju, Xu Yan, Shuai Zhao, Fei Huang, Rui Wang, Shuguang Han, jufeng chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.05832)  

**Abstract**: Large language models (LLMs) excel in structured tasks but struggle with dynamic social interactions, where success requires long-term goal coordination and rapid adaptation. Current methods often apply uniform goal-based rewards to every utterance, overlooking the specificity of objectives at each dialogue turn and failing to account for the rationale of potential strategies. Inspired by the Theory of Planned Behavior, we propose the Think-Strategy-Response (TSR) framework, which decomposes social dialogue into two hierarchical stages: high-level strategic planning and low-level linguistic execution. To optimize TSR, we introduce Linearized Hierarchical Reinforcement Learning with Variance-Gated Rewards (LHRL-VGR), a novel algorithm that dynamically routes rewards - balancing goal completion and strategy adherence - based on the variance of goal achievement scores. Experiments on the SOTOPIA benchmark show that our approach fine-tunes a Qwen2.5-7B agent to surpass the GPT-4o baseline by 7.32% in goal completion success, demonstrating state-of-the-art performance in multi-agent social negotiation tasks. 

---
# MoCA: Implicit Social Context Analysis 

**Authors**: Wenhao Xu, Kaiwen Zhang, Hao Li, Maowei You, Yongzheng Ji, Siyuan Zuo, Jingxuan Yu, Sina A, Xinyao Tan, Bobo Li, Hao Fei, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2608.05825)  

**Abstract**: Human social communication, such as affection and intent, is often conveyed in highly implicit ways, where underlying meanings are expressed through indirect, socially and culturally grounded signals rather than explicit statements. Such implicit social contexts are pervasive in real-world interactions, yet there remains a lack of a formal and systematic framework for studying them. In this paper, we introduce Implicit Social Context Analysis (MoCA), a novel task that systematically models implicit social scenarios along three key dimensions: affection, intent, and stance. We construct a high-quality benchmark containing 3,108 multimodal instances collected from real-world sources, with fine-grained cognitive annotations revealing who expresses what toward whom, as well as how and why it is conveyed. Using the MoCA dataset, we show that state-of-the-art multimodal large language models struggle significantly with this task because of their reliance on explicit cues and limited ability to reason over latent social contexts. To address this challenge, we propose Conflict-Driven Abductive Reasoning (CoDAR), a novel framework that models the discrepancy between observed expressions and expected truthful behavior as cognitive conflict, thereby enabling the inference of hidden mental states. Extensive experiments demonstrate that CoDAR substantially improves model performance. Nevertheless, a large gap from human reasoning remains, highlighting the fundamental difficulty of implicit social understanding. 

---
# Decomposed Entailment for Factuality Checking and Hallucination Detection 

**Authors**: Achir Oukelmoun, Nasredine Semmar, Gaël De Chalendar  

**Link**: [PDF](https://arxiv.org/pdf/2608.05823)  

**Abstract**: The reliability of Large Language Models (LLMs) is often compromised by factual inconsistencies, including hallucinations---cases where generated content is not supported by the underlying source. We present HallDetect, a lightweight, reference-free, and black-box framework for hallucination detection that we evaluate not only on summarization but across a broader range of source-grounded generation settings. HallDetect builds on decomposition-based factuality evaluation: generated content is decomposed into atomic claims, each verified by a compact encoder-based entailment model through a contrastive formulation over a multi-scale library of source chunks, and aggregated with an asymmetric score in which a single confidently contradicted claim flags the response. Under a controlled protocol in which all methods share the same 4-bit quantized backbones and consumer-grade hardware budget, HallDetect outperforms comparably resourced generative and embedding-based baselines on three of four benchmarks while remaining stable across backbone families, and yields a claim-to-span audit trail that localizes each error. 

---
# M$^3$R-Bench: A Unified Benchmark for Evidence-Grounded Multimodal Metaphor Understanding 

**Authors**: Hong Jiang, Junnan Zhu, Jingwang Huang, Xiao Sun, Yuming Yang, Jiang Zhong, Ruirui Chen, Jingman Shi, Hao Wu, Nayu Liu, Xinyi Jiang, Kaiwen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2608.05817)  

**Abstract**: Metaphor enables the understanding of abstract concepts through cross-domain mappings while conveying affective attitudes. In multimodal scenarios, visual and textual information jointly construct Target--Source mappings, requiring both conceptual understanding and cross-modal reasoning. However, existing benchmarks mainly evaluate metaphor understanding through isolated subtasks and lack evidence-grounded explanations, making it difficult to assess whether models establish mappings grounded in visual and textual this http URL address these limitations, we introduce M$^3$R-Bench, a unified and evidence-grounded benchmark containing 1,000 image--text instances with human-verified annotations. Guided by Conceptual Metaphor Theory and theories of nonliteral language understanding, M$^3$R-Bench provides joint annotations for metaphor occurrence, Target--Source mapping, sentiment, and stage-wise explanations following ``evidence identification--mapping establishment--sentiment inference.''Evaluations on M$^3$R-Bench reveal that existing models often overlook visual evidence, rely on superficial textual cues, and produce inaccurate Target--Source mappings, exposing a cross-modal evidence--mapping mismatch. To address this mismatch, we propose M$^3$R-Reasoner, which combines curriculum-based reasoning supervision with task-aware reinforcement learning to align model reasoning with metaphor interpretation. Experiments show that, with only an 8B-parameter backbone, M$^3$R-Reasoner outperforms larger proprietary MLLMs across four unified-task metrics and improves Visual Evidence and Sentiment Justification scores over GPT-5.5 by 28.45 and 30.11 points, respectively, while surpassing Claude-Sonnet-4.6 by 8.00 points in mean rubric score. The dataset and code are available at this https URL. 

---
# Hierarchical Latent Prediction for Language Models 

**Authors**: Chang Shi, Tim Pearce, Manan Tomar, Siddhartha Sen, John Langford  

**Link**: [PDF](https://arxiv.org/pdf/2608.05806)  

**Abstract**: While standard Next-Token Prediction (NTP) lays the foundation of language model pre- training, its teacher-forced training paradigm may not be optimal for long-horizon reasoning and planning. Recent works such as Multi-Token Prediction (MTP) and Next-Latent prediction (NextLat) try to mitigate the problem through predicting multiple future tokens and self-supervised prediction in the latent space. However, those auxiliary objectives either have a limited horizon or suffer from compounding error from multi-step rollout. We introduce Hierarchical Latent Prediction (HiLP), which introduces an auxiliary higher-level abstract latent to help reduce the error accumulation effect in latent-space rollouts. Experiments show that HiLP can lead to longer-horizon coherent belief state representation and demonstrate the effectiveness of our method across coding and multi-step reasoning benchmarks, and offers more speculative decoding efficiency. 

---
# On-Policy Delta Distillation for Multilingual Math Reasoning 

**Authors**: Byeongho Heo, Jaehui Hwang, Sangdoo Yun, Dongyoon Han  

**Link**: [PDF](https://arxiv.org/pdf/2608.05802)  

**Abstract**: On-Policy Distillation (OPD) is emerging as a promising alternative to reinforcement learning for LLM post-training, yet its effectiveness in multilingual settings remains underexplored. We study OPD and its advanced variant, On-Policy Delta Distillation (OPD$^2$), for mathematical reasoning in English, Korean, and Japanese. OPD$^2$ improves OPD by using the probability gap between a post-trained teacher and its base model as the learning signal. Experiments with Qwen3 show that OPD$^2$ consistently outperforms the original OPD, with particularly strong improvements in Korean and Japanese, and generally narrows the English-Korean performance gap. We further find that English-only OPD can also increase performance for Korean and Japanese, but often shifts the responses toward English, highlighting the importance of multilingual data to preserving target-language responses. 

---
# Task-Conditional Flow Matching for Balanced Multilingual Text Embedding Adaptation 

**Authors**: Tirth Bhatt, Naren Kumar S, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.05785)  

**Abstract**: Multilingual text embedding models are commonly adapted using a single training objective across diverse tasks, despite different tasks requiring fundamentally different optimization strategies. We introduce Task-Conditional Flow Matching (TCFM), a multilingual embedding adaptation framework that selectively applies Flow Matching to translation tasks while optimizing retrieval, classification, and pair-classification tasks with objectives better aligned to their learning dynamics. TCFM further combines teacher-guided representation preservation with a three-stage curriculum to enable stable adaptation. Evaluated on the Indic Massive Text Embedding Benchmark, TCFM establishes a new state-of-the-art, consistently improving embedding quality across a diverse set of multilingual tasks and generalizing across embedding model families. We will publicly release the codebase and datasets upon acceptance of the paper. 

---
# How to Recognize New Words: A Comparison Between Context Biasing Methods and Speech LLMs 

**Authors**: Christian Huber, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2608.05759)  

**Abstract**: Recognizing new and rare words - named entities, acronyms, domain specific special words, and other items scarce in training data - remains a key challenge for automatic speech recognition (ASR). We compare two strategies for this: context biasing methods, where an ASR model is extended such that during inference a word list can be supplied, and speech large language models (LLMs) prompted with context directly. We evaluate two context biasing methods based on Whisper against three speech LLMs across read and non-read speech, reporting biased, unbiased, and overall word error rate (WER). The context biasing methods cut biased WER by up to 88% relative while leaving other words largely unaffected. Speech LLMs excel on read speech but generalize less well to non-read speech, and prove sensitive to distractor count and prompt word order. We characterize the resulting trade-offs to guide method selection. 

---
# Once a Response, Always a Response: Detecting LLM-generated Text via Latent Prompt Restoration 

**Authors**: Hongrui Bao, Yubing Ren, Yanan Cao, Jinhan You, Fang Fang, Shi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05741)  

**Abstract**: Large language models (LLMs) can generate fluent and convincing text at scale, creating growing risks for misinformation dissemination, educational misuse, and platform governance. These concerns make robust detection of machine-generated text increasingly necessary. Recent zero-shot detectors mainly exploit probability-based statistical discrepancies, but they do not explicitly account for the training process of LLMs, which leaves a distinct generation mechanism insufficiently modeled and limits detection robustness. To address this issue, we propose EchoPrompt, a training-free detector based on latent prompt restoration. Our key intuition is that machine-generated text is typically produced conditioned on an upstream prompt, and this hidden dependency can be partially reactivated by prepending a unified generic prefix. Specifically, EchoPrompt restores a generic assistant-response context, measures the induced likelihood gain with an instruction-tuned model, calibrates it against the corresponding base model, and aggregates the resulting differences into a score that quantifies latent prompt dependency. Extensive experiments show that EchoPrompt achieves state-of-the-art performance among zero-shot detectors while maintaining strong robustness across challenging evaluation settings. 

---
# Mitigating Scoring Bias in LLM-as-a-Judge via Random Number Generation 

**Authors**: Yuma Asato, Kiyoaki Shirai, Natthawut Kertkeidkachorn  

**Link**: [PDF](https://arxiv.org/pdf/2608.05726)  

**Abstract**: Large Language Models (LLMs) are often used as evaluators of text quality, known as LLM-as-a-Judge, which can outperform conventional automatic evaluation metrics that rely on reference texts. However, LLM evaluators tend to generate particular scores regardless of the context of the evaluated text, which is known as scoring bias. This study proposes a novel method to mitigate this scoring bias. An LLM is instructed to randomly generate number tokens, and the latent numerical bias of the LLM is identified by measuring the deviation of the observed distribution of numbers from the uniform distribution. A definition of a downstream task, for which an LLM evaluator is used, is added to the prompts for random number generation to measure task-specific latent number bias. In the evaluation by an LLM, the token generation probabilities for a given input are rectified considering the LLM's latent number bias. Results of the experiment on four different tasks, evaluation of LLM alignment, evaluation of summarization, Semantic Textual Similarity, and Semantic Textual Relatedness, demonstrate that our proposed method outperforms the baselines, including an LLM without debiasing and previous calibration methods. In addition, it is confirmed that scoring bias varies across LLMs, tasks, and score ranges, indicating the importance of measuring latent number bias as the case may be. 

---
# Sparse Mutual Information Graph Averaging for Improving Random Indexing Embeddings 

**Authors**: Sriram Loganathan, Gokul Anand, Aung Bo Bo, Yourui Shao, William B. Andreopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2608.05724)  

**Abstract**: Sparse word embedding pipelines can avoid dense co-occurrence matrix materialization, dense factorization, and gradient training while still relying on sparse global corpus statistics. This paper studies Random Indexing (RI) vectors refined by weighted averaging on a sparse Positive Pointwise Mutual Information (PPMI) graph. On a fairytales corpus, the covered semantic analogy set consists of 272 Google family- category questions. On this family subset, PPMI top-K graph averaging repairs a weak RI initialization, improving accuracy from 19.4+-0.7% to 30.7+-2.9% across five seeds. Under the single tested runs, the same neighborhood averaging reduces family- subset analogy accuracy for PPMI+SVD (singular value decom- position), Binary+SVD, CBOW, and Skip-gram. Thus the method is not competitive with neural baselines on text8 and gives near- zero strict similarity correlation on SimLex-999. While Bloom filter sketches underperform RI in the tested configuration, we find that PPMI graph averaging with top-K pruning is a useful non-gradient repair for weak RI embeddings. On the fairytales dataset, PPMI top-K=50 graph averaging improves RI with accuracy going from 19.4+-0.7% to 30.7+-2.9%, and performing best with a seed42 of 34.6%. 

---
# Answer First, Reason Later: Commitment Order in Diffusion LLMs 

**Authors**: Jewon Yeom, Jaewon Sok, Seonghyeon Park, Jeongjae Park, Hwiyeong Lee, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.05687)  

**Abstract**: Masked diffusion language models (dLLMs) can commit tokens in any order -- a freedom marketed as their core advantage over autoregressive decoding. We show that on reasoning tasks this freedom is instead the axis of failure. Logging every commitment during decoding of LLaDA-8B on GSM8K, we find that unconstrained (pure) decoding commits the final answer at 15-24% of the trajectory while half the reasoning region is still masked, and collapses to answer-only outputs on up to 90% of problems as the canvas grows. The cause is not the model's termination beliefs -- EOS "pressure" is nearly identical across decoders -- but reachability: whether the sampler may act on those beliefs at distant positions. A 2x2 prompt-decoder design shows that chain-of-thought helps only under ordered commitment (interaction +34.8 percentage points, 95% CI [26.8, 42.8]; without reasoning text the decoders are indistinguishable), an interaction we decompose into a collapse channel and an order channel and replicate on Dream-7B and MATH-500. A single-knob intervention -- frontier-gated commitment -- causally recovers the full gap (0.528 to 0.852) while preserving up to 4x parallel decoding, along a measured frontier whose optimal window flips from w=1 at full refinement to unconstrained at 8 tokens/step. Our results reframe existing window-style samplers, previously motivated by efficiency, as the minimal fix for a reasoning pathology they were never designed to address. 

---
# Relay, Don't Route: Adaptive Population Handoff for Cost-Efficient LLM-Driven Evolution 

**Authors**: Sichun Luo, Yi Huang, Guanzhi Deng, Haibo Wang, Haochen Luo, Lei Li, Zefa Hu, Junlan Feng, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.05651)  

**Abstract**: Large language model (LLM)-driven evolution has shown promise for program search and algorithm discovery, but relying on strong models throughout long evolutionary runs is costly. A natural alternative is to combine cheap and strong models under a fixed inference budget. However, existing approaches typically allocate models at the level of individual queries or mutation steps, overlooking that evolutionary search is \textit{stateful}: each generated candidate changes the population from which subsequent mutations are produced.
We empirically analyze LLM-driven evolutionary trajectories and find that search progress is strongly front-loaded, early trajectory performance is informative but noisy, and cheap models recover much of the early progress achieved by strong models at lower cost. Motivated by these findings, we propose \textbf{\model}, a training-free framework that shifts budget allocation from individual calls to evolving populations through adaptive \textit{population handoff}. A cheap model explores multiple trajectories in short blocks allocated by a bandit scheduler. Relay Gain, defined as the marginal improvement of a compact, quality-diverse candidate bank constructed for handoff, serves as the scheduler reward and determines when to hand off. The curated candidates initialize a shared strong model population for refinement. Across four benchmarks and three budgets, \model achieves the highest mean score in 11 of 12 settings, outperforming competitive baselines. Our results suggest that in stateful search, budget allocation should be organized around the population, not the individual call. 

---
# Human-Like Anaphor Resolution in Large Language Models 

**Authors**: Keane Zhang, Varshini Chinta, Raj Sanjay Shah, Sashank Varma  

**Link**: [PDF](https://arxiv.org/pdf/2608.05630)  

**Abstract**: Anaphors are expressions that refer to other expressions, called antecedents. The process of connecting the two is called resolution. Cognitive science has identified multiple factors that affect the speed and success of anaphor resolution, including discourse structure, situation-model properties, and semantic factors. Here, we investigate whether these factors also affect anaphor resolution in five Large Language Models (LLMs) with open weights: GPT-2-XL, Llama-3.1-8B, Pythia-12B, Mistral-7B, and Mistral-24B. To model processing difficulty, we adopt the standard linking hypothesis that relates human reading times to model surprisal at the anaphor. As a second behavioral measure, we compare model accuracy to human accuracy on comprehension questions probing the antecedents of anaphors. The results show selective cognitive alignment: some LLMs exhibit human-like sensitivity to discourse prominence and distance-based factors in anaphor resolution, while showing weaker or absent sensitivity to semantic interference effects. These findings delimit the conditions under which LLMs approximate human anaphor resolution. 

---
# FOCUS: Decoupling Expert Personas in LLMs to Enhance Domain Expert Capabilities 

**Authors**: Guanyu Wang, Zidi Zhang, Xu Chu  

**Link**: [PDF](https://arxiv.org/pdf/2608.05611)  

**Abstract**: Large Language Models (LLMs) can exhibit diverse personas, and activating expert personas has been shown to improve domain expertise and task accuracy. However, existing persona control methods often suffer from cross-domain coupling, which may lead to overly aggressive behavior in high-caution domains such as healthcare, or excessive conservatism in risk-sensitive domains such as financial trading. To address this issue, we propose FOCUS (\textbf{\underline{F}}ine-tuning with \textbf{\underline{O}}rthogonal \textbf{\underline{C}}ontrol for \textbf{\underline{U}}ncoupled persona\textbf{\underline{S}}). FOCUS first automatically extracts expert persona vectors from LLMs, then applies orthogonal decomposition to decouple domain-specific expert personas, and finally introduces an expert gating module to adaptively control persona activation according to task contexts. With a two-stage training strategy and a gated selection regularizer, the model learns to activate appropriate personas for both single-domain and cross-domain tasks. Experiments on financial, legal, medical, and cross-domain benchmarks show that FOCUS improves task accuracy and outperforms existing persona control methods. Our code is available at \href{this https URL}{this url}. 

---
# SkillZip: Contract-Preserving Graph Compression for Scalable Agent Skill Libraries 

**Authors**: Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, Liming Zhu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05604)  

**Abstract**: Large Language Models (LLMs) increasingly act as agents whose procedural knowledge is stored in reusable skill packages and loaded at inference time. As skill libraries grow, a central challenge is to expose the smallest sufficient executable context under a limited context budget. Existing systems struggle to reuse routines below the whole-skill level, preserve procedural contracts during compression, keep compressed routines executable and expandable, and update the compressed library as skills evolve. These challenges reveal a unit mismatch: skills are retrieved as packages, compressed as text, and converted into execution graphs only after retrieval, whereas reliable reuse requires a contract-bearing procedural unit. We propose SkillZip, an execution-aware procedural abstraction framework that performs contract-preserving compression over section-level graphs. SkillZip rewrites recurring contract-valid motifs into reversible ported macros while preserving boundary signatures, dependency closure, verifier reachability, and source-level expansion. At inference time, it hydrates a compact, dependency-closed context and expands macros only when required. ReZip further integrates new skills and revises risky macros using execution evidence. Comprehensive experiments1 on technical and embodied agent benchmarks show SkillZip consistently outperforms the strongest baseline by up to 12.2 points, while achieving a 3.46x compression ratio with 99.2% dependency preservation and 98.7% verifier reachability. Scaling analyses further confirm robust retrieval across skill libraries ranging from 200 to 100K skills. 

---
# Where Models Converge and Humans Diverge: A Coverage Framework for Distributional Pluralism in Open-Ended Generation 

**Authors**: Zini Yang, Emily Wenger, Richard So  

**Link**: [PDF](https://arxiv.org/pdf/2608.05576)  

**Abstract**: When a large language model (LLM) writes Harry Potter fanfiction, it reliably produces fundamental elements of the Hogwarts universe, such as recognizable places and characters. Human-written Harry Potter fanfictions, however, typically include these fundamentals and much more, incorporating stylistically irregular content and relationship-diverse plotlines. This gap between LLM and human writing has been noted across a variety of domains. LLMs tend to produce "average" writing, while human writing contains more diverse content that covers a broader distribution. Existing work has shown the existence of this distributional "gap", but no work has proposed a systematic way to measure it. Our paper proposes a human-grounded framework that uses the empirical distribution of human writing on a topic to measure the distributional breadth of LLM-generated content on that same topic. We propose two metrics, LLM Coverage (LLM-Cov) and In-Boundary Rate (IBR), that separate the plausibility of LLM content from its distributional breadth. Across ideation and narrative tasks, we find that current LLMs produce plausible but narrow content that concentrates near the center of the human response space. Our framework can enable researchers to better assess the distributional breadth of LLM-authored content, which we term its "cultural reach". 

---
# Different Perturbations, Different Mechanisms: Understanding Continued Pre-training for Zero-Shot Dialect Robustness 

**Authors**: Aarohi Srivastava, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05510)  

**Abstract**: Dialectal variation remains a major challenge for multilingual language models. Perturbation-based continued pre-training (CPT) has emerged as a promising approach to improving robustness, yet existing work largely evaluates individual perturbation strategies in isolation and provides limited insight into why they work. We present a systematic study of perturbation-based CPT for multilingual dialect robustness in LLMs, comparing six training conditions across nine German, Italian, and Arabic dialect tasks. Perturbation-based CPT, especially character-noised CPT, consistently improves zero-shot dialect robustness while largely preserving standard variety performance. More importantly, we show that methods with similar downstream performance induce distinct mechanisms of robustness, exhibiting different patterns of language model adaptation, representational alignment, and prediction repair. Our results provide a more complete understanding of how synthetic surface variation improves robustness and offer practical guidance for selecting CPT strategies in multilingual and dialectal settings. 

---
# DBLAST: Dependent Block Drafting for Stochastic Speculative Decoding 

**Authors**: Amirmohammad Karimi, Chao Gao, Negar Hassanpour  

**Link**: [PDF](https://arxiv.org/pdf/2608.05448)  

**Abstract**: Speculative decoding accelerates large language models' inference by using a lightweight drafter to propose multiple future tokens and a target model to verify them. While recent block and diffusion-style drafters can predict several positions in a single pass, their training and sampling procedures are typically optimized for greedy decoding or assume that positions in the draft block are conditionally independent. This assumption becomes brittle in non-greedy speculative decoding, where the target distribution is deliberately stochastic and multiple continuations become plausible. We study this mismatch for block diffusion drafters and show that the accepted draft length degrades as the entropy of the target sampling distribution increases. We propose a dependent block drafter based on a low-rank latent mixture over token positions, complemented by an acceptance-oriented training objective that directly targets the expected verified length. Experiments with Qwen3-4B and Qwen3-8B on GSM8K, MT-Bench, HumanEval, and creative-writing benchmarks show that our approach, namely DBLast, consistently improves accepted length over independent block sampling, especially in higher-entropy decoding regimes. 

---
# Example-Guided Prompting for Document-Level Text Simplification 

**Authors**: Marina Litvak, Ariel Perstin, Ilan Shtilman, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2608.05447)  

**Abstract**: Document-level text simplification requires large language models (LLMs) to rewrite complex documents while preserving meaning, readability, and discourse coherence. Although prompt-based LLMs have shown promising performance, they often produce inconsistent simplifications because textual instructions alone provide limited guidance for complex document-level transformations. We investigate whether retrieved document-simplification examples can improve document-level generation by augmenting prompts with examples selected from a parallel simplification corpus. This example-guided prompting approach enables LLMs to exploit relevant simplification patterns without task-specific fine-tuning. Experiments on the OneStopEnglish corpus using multiple state-of-the-art LLMs show that incorporating retrieved examples consistently improves simplification quality over prompt-only generation and achieves competitive or superior performance compared with representative supervised (T5) and planning-based (PlanSimp) document simplification systems. Furthermore, we find that the benefits of example-guided prompting vary across LLMs, suggesting that effective use of retrieved examples depends on a model's ability to integrate contextual information during generation. 

---
# Mood Matters: How Syntactic Sensitivity Undermines Safety Alignment 

**Authors**: Alina Klerings, Jannik Brinkmann, Heiner Stuckenschmidt, Simone Paolo Ponzetto  

**Link**: [PDF](https://arxiv.org/pdf/2608.05409)  

**Abstract**: Large language models typically undergo post-training to align them with safety policies but there exist many sophisticated jailbreaks that sidestep established safeguards. For instance, prior work by Andriushchenko et al. (2025) has found that changing the grammatical tense from present to past can be enough to elicit harmful responses. In this work, we uncover a more general failure of non-imperative syntactic forms. We demonstrate that this syntactic vulnerability exists in 16 models up to 70B parameters, using behavioral evaluation. To investigate the root cause, we apply causal mediation analysis, finding that refusal is partially conditioned on upstream syntactic features. By steering these purely syntactic features we are able to trigger and suppress refusal. Finally, we trace this ill-conditioning to linguistically biased post-training data of open-source models and show that increasing syntactic diversity can mitigate the issue. Our findings suggest that current alignment approaches introduce confounders that prevent a pure semantic grounding of the refusal decision. 

---
# The interface of intonation and lexical tone: Boundary phenomena in Mandarin varieties 

**Authors**: Cong Zhang, Yiya Chen  

**Link**: [PDF](https://arxiv.org/pdf/2608.05364)  

**Abstract**: This chapter explores the intricate interplay between intonation and tone in Mandarin Chinese varieties, focusing on f0, the primary acoustic cue for both intonation and tone. The main empirical base is intonation boundary phenomena, where intonation and tone intersect and influence each other in conveying a range of sentence-level linguistic functions -- such as question vs. statement -- and a rich array of speakers' attitudinal information. Theoretical models and emerging techniques are also discussed to account for the observed interactions of tonal aspects and boundary phenomena to convey multiple levels of communicative meanings. 

---
# Evidence Lock Before Commitment: A Frozen Interface Degrades LLM-as-Judge Evaluation 

**Authors**: Divyansh Singh  

**Link**: [PDF](https://arxiv.org/pdf/2608.05353)  

**Abstract**: LLM judges are often asked to extract criteria and evidence before choosing between candidate answers. This workflow assumes that the intermediate record preserves the information needed for a later verdict. For reasoning-capable models, visible field order does not reveal internal decision order, so we test an observable alternative: persist the evidence in one call and make it the exclusive input to the next. Across 24,000 judgments over HelpSteer3, FeedbackQA, and CoVal, we compare standard pairwise judging, structured one-call judging, two-call evidence locking, and three-call pointwise locking with Claude Sonnet 4.5 and GPT-5. Evidence locking reduces agreement with released human preferences by 4 to 6 percentage points and increases answer-order inconsistency by 8 to 10 points relative to structured one-call judging. Pointwise locking is also harmful, while structured evidence elicitation remains close to standard judging. The result holds for both judges and all three datasets. Persisted evidence can support auditability, but it should not replace the source answers at decision time. 

---
# Constraint-First Reasoning: A Training-Free Protocol for Exploiting Answer-Space Constraints in Mathematical Problem Solving 

**Authors**: Hongbo Ma, Bangji Yang, Yunqian Selina Cheng, Jiajun Fan, Hanwen Zhang, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.05254)  

**Abstract**: Large language models can derive a plausible mathematical object yet still violate explicit requirements--for example, by omitting a modular reduction, returning a non-integer, or using the wrong encoded answer form. We introduce Constraint-First Reasoning (CFR), a training-free two-stage prompting protocol: Stage 1 extracts and summarizes constraints entailed by the problem, and Stage 2 solves while checking intermediate and final results against that summary. Routed-CFR activates the two-stage protocol only when a text-only regex router detects restrictive cues; otherwise it uses direct chain-of-thought (CoT). Across AIME, CMIMC, BRUMO, and AIMO_AMC, the method improves direct CoT on multiple backbones. We further report convention-controlled routing experiments, matched prompting baselines, problem-level paired tests, decoding robustness, constraint-quality audits, total-token accounting, and an OlympiadBench evaluation. These analyses position CFR as a targeted test-time intervention whose benefit depends on recoverable constraints and reliable Stage 1 extraction, rather than as a general-purpose replacement for mathematical reasoning. 

---
# Analysis of Numerical Localisation in LLM Translations 

**Authors**: Patrizia Kaye  

**Link**: [PDF](https://arxiv.org/pdf/2608.05232)  

**Abstract**: The work of Tang et. al. (2025) on numerical translation is extended by analysing the capability of five large language models (LLMs) for the localisation of times, numbers, and dates instead of translation. Models were selected that could be loaded onto and run on commodity hardware and a baseline quality for each mode is computed, then three different strategies to improve on that accuracy were tested. In contrast to Tang et. al., it was discovered that on the tested LLMs, embedding the localisation principles into the prompt context provided a statistically significant improvement in accuracy compared to direct translation or the alternative strategies. 

---
# Position: It's Time to Optimize LLMs for Self-Consistency 

**Authors**: Itamar Pres, Belinda Z. Li, Laura Ruis, Zifan Carl Guo, Keya Hu, Mehul Damani, Isha Puri, Ekdeep Singh Lubana, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2608.05188)  

**Abstract**: Despite ever-increasing sophistication in language model (LM) pre- and post-training pipelines, many important failures persist: models overcondition on user framing ("sycophancy"), exhibit incomplete logical generalization, and produce confident but incorrect responses. We argue that these failures arise from a modeling assumption permeating all aspects of the pipeline: that behavior can be specified and evaluated independently on single-output pairs. Many model failures are difficult, if not impossible, to detect without reasoning about relationships between a model's responses across inputs. In this position paper, we propose self-consistency as a framework for understanding these failures. We first observe that a wide variety of techniques designed to improve specific aspects of LM behavior-targeting properties as diverse as adversarial robustness and factual coherence-can be understood as special cases of a common "consistency optimization" procedure and addressed with a standard set of optimization tools. We next outline a set of new model properties that could be achieved by optimizing for consistency, and conclude with a discussion of what it would mean to develop generally consistent LMs, including the capabilities they would enable and the objections they raise. 

---
# DREAM: LLM-based Dynamic Role-playing via Event-Aware Memory Graph 

**Authors**: Zhihao Xiao, Mengting Li, Xintao Wang, Linfeng Li, Limin Shui, Mengqi Ji, Borui Cai  

**Link**: [PDF](https://arxiv.org/pdf/2608.05170)  

**Abstract**: Role-playing agents (RPAs) have emerged as a key application of large language models, enabling immersive and high-fidelity character simulation. Accurate role-playing of established characters requires not only stylistic imitation but also temporally consistent and causally grounded behavioral reasoning. However, existing RPAs primarily rely on static character descriptions and unstructured memory, limiting their ability to maintain long-term narrative and personality coherence. We introduce DREAM, a structured memory framework for role-playing agents inspired by the Activating Event-Belief-Consequence (ABC) cognitive model. DREAM transforms unstructured literary text into an Event-aware Memory Graph (EMG) that organizes character experiences into temporally ordered and causally linked event graph. This representation enables the construction of dynamic, dual-granularity character profiles that capture both stable personality traits and event-driven behavioral evolution. We further propose the Temporal Causal Memory (TCM) benchmark to evaluate temporal consistency and long-range causal narrative coherence. DREAM achieves state-of-the-art performance across CoSER, LIFECHOICE, and TCM, outperforming multiple strong baselines. Our approach demonstrates the effectiveness of structured memory in enhancing the interpretability and consistency of role-playing agents. 

---
# ConWriter: Transition-Constrained Stateful Long-Form Story Generation with Lightweight Neuro-Symbolic Consistency Control 

**Authors**: Jindong Li, Yang Yang, Zihao Liu, Yutao Yue, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05169)  

**Abstract**: Long-form story generation requires models to preserve narrative consistency across extended contexts, yet existing prompting-based methods often accumulate temporal, factual, character, commonsense, and stylistic errors as the story grows. We propose ConWriter, a training-free framework for consistency-aware long story generation. ConWriter writes stories incrementally at the scene level, guided by static story requirements, dynamic narrative memory, symbolic state reasoning, and uncertainty-aware risk signals. Rather than treating long-story generation as a single free-form decoding process, ConWriter maintains evolving story states, checks whether new scenes satisfy required narrative transitions, and uses uncertainty-aware risk signals to prioritize validation and localized repair. This enables consistency control during generation, before local errors propagate into later scenes. We evaluate ConWriter on ConStory-Bench, covering four long-story tasks: continuation, generation, expansion, and completion. Due to the high cost of long-form generation and evaluation, we use the first five cases from each task and test 3k, 6k, and 12k target lengths across Qwen3.5-Plus, DeepSeek-V4-Flash, and GPT-5 series. Experiments follow the official ConStory-Bench evaluation protocol. 

---
# CNM-BERT: A Drop-In Structural Embedding for Chinese Characters via Ideographic Description Sequences 

**Authors**: Thomas Sing-wing Wu, Liqian Yan  

**Link**: [PDF](https://arxiv.org/pdf/2608.05167)  

**Abstract**: Token-based encoders like BERT treat Chinese characters as atomic identifiers, ignoring their recursive orthographic structure. Consequently, models rely on contextual co-occurrence, degrading performance on rare and out-of-vocabulary (OOV) characters. We propose the Compositional Network Model (CNM), a lightweight augmentation that injects discrete compositional structure into Transformer encoders. CNM parses Ideographic Description Sequences (IDS) into trees, encodes them via a recursive Tree-MLP, and fuses the structural embeddings into BERT without modifying the backbone. Evaluated on the Wu et al. (2025) structural-probing benchmark, CNM-BERT outperforms the strongest baseline (ChineseBERT) on long-tail and OOV characters by +9.8 Structure accuracy and +7.7 Radical F1. Furthermore, CNM-BERT achieves consistent gains across CLUE, MRC, and NER tasks at both base and large scales, demonstrating that explicit structural injection delivers both robust OOV understanding and tangible downstream value. 

---
# Conditional Cognitive Biases in LLMs: How Biased User Turns Modulate In-Context Reasoning 

**Authors**: Sachini Weerasekara, Sagar Kamarthi, Jacqueline Isaacs  

**Link**: [PDF](https://arxiv.org/pdf/2608.05166)  

**Abstract**: We present an evaluation of cognitive bias expression in state-of-the-art instruction-tuned LLMs under realistic multi-turn interaction settings. Our work introduces a novel three-condition experimental framework that disentangles the effect of exposure to a biased user turn from the effect of the turn's semantic content, alongside a benchmark of 24,300 jury-validated user prompts spanning all 81 cells of a 9x9 target-human bias interaction matrix. Across eight frontier LLMs, we find that biased conversational context systematically increases bias expression relative to zero-shot baselines in 6 of 8 models. We identify two competing behavioral dynamics underlying this effect: conversational exposure to biased reasoning generally amplifies downstream bias tendencies, while explicitly stated bias cues often trigger alignment-related suppression behaviors that reduce overt bias expression. We release our framework, codebase, and dataset to support future research on context-conditioned cognitive biases and behavioral adaptation in LLMs. 

---
# A Study of ASR Adaptation and Representation Dimensionality Reduction in Persian Speech Emotion Recognition Using Whisper 

**Authors**: Ali Shendabadi, Parnia Izadirad, Mostafa Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2608.05165)  

**Abstract**: Speech Emotion Recognition (SER) in low-resource languages remains a challenging problem due to limited labeled data. In this work, we study the use of Whisper for Persian SER with a particular focus on representation dimensionality reduction and language-specific model adaptation. We propose a SER framework in which frame-level embeddings extracted from the Whisper encoder are reduced in dimensionality using PCA, eliminating the need for learned projection layers and substantially reducing the number of trainable parameters. The reduced representations are aggregated using an attention-based pooling mechanism and classified with a lightweight prediction head. In addition, we investigate whether fine-tuning Whisper on a Persian automatic speech recognition (ASR) task improves downstream SER performance. Experiments conducted on the ShEMO dataset under a speaker-independent evaluation protocol show that PCA-based dimensionality reduction consistently improves emotion recognition performance while reducing training latency and memory usage. ASR fine-tuning yields only modest gains for SER, suggesting limited transfer from language adaptation to emotion-related representations under the evaluated conditions. These findings provide practical insights into the efficient use of large pretrained speech models for emotion recognition in low-resource languages. 

---
# Cross-Architecture Steering Transfer in Language Models: A Systematic Empirical Study 

**Authors**: Ayushi Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2608.05164)  

**Abstract**: Independently trained large language models may develop shared internal representations of semantic concepts despite architectural differences -- but whether this geometric similarity has functional consequences for cross-model behavioural control remains untested. We present the first systematic evaluation of cross-model steering transfer and show that shared LLM geometry is functionally exploitable, conditionally: concept directions from one model can steer a different independently trained model when sufficient representational capacity exists. We study five open-weight models spanning three parameter scales (0.8B--8B) and two architectural lineages, training one Sparse Autoencoder per model across 15 semantic domains and testing alignment across all 20 directed model pairs. We observe a suggestive discontinuity near 1.7B parameters: at >= 1.7B scale, 47--49% of cross-model feature pairs validate (Pearson r >= 0.60, Procrustes cosines 0.895--0.956), while alignment degrades sharply below 0.8B. Cross-model steering vectors (B3-TI) achieve a 71.0% win rate across 15 supervised concepts versus 68.0% for same-model native vectors; a single universal vector achieves 67.3% in 4 of 5 models without any per-model supervision. Transfer degrades for models below 1.7B and for one model with generation instability, confirming that functional exploitability requires sufficient representational capacity. Our findings underscore the importance of scale thresholds in mechanistic interpretability: tools validated at 7B scale may not transfer to smaller models without revalidation. We provide the first functional complement to the Platonic Representation Hypothesis -- geometric convergence across independently trained LLMs supports cross-model behavioural control without fine-tuning, under the identified scale conditions. 

---
# Where Privacy Risk Lives in English-Source Multilingual RAG: A Stage-Decomposed Audit Across Five Query Languages 

**Authors**: Yanhang Li, Zhichao Fan, Zexin Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05163)  

**Abstract**: A common assumption holds that switching to a non-English language makes a multilingual RAG system easier to attack for personal information. We test this on an English-source synthetic-PII corpus with five query languages and a two-stage defence (LLM input judge + regex output filter), in a pipeline whose translator, judge, back-translator, and generator are all Qwen2.5-7B -- so every finding below is pipeline-conditional, not a causal ranking of language-inherent risk. Under output-only filtering, English has the highest observed unstructured-PII leak rate; only English-vs-Swahili separates cleanly under document-level bootstrap intervals. Once the input judge is added, residual leaks remain on Arabic and Swahili, and back-translating the query does not close the gap (an ablation we report but cannot use as a causal diagnostic, since the back-translator is also Qwen). On a separate n=17 multilingual-prompted-judge residual corner, attaching the gold corpus document to the input judge blocks 15/17 residual cells. We frame this last result as a mechanism diagnostic, not a deployable defence: it uses oracle retrieval, BLOCK/ALLOW rates are measured on adversarial queries only, and we measure no benign-query false-positive rate and no answer-utility cost. The supplementary material contains code, corpora, queries, and per-trial JSONLs; the priority follow-up is an independent-MT plus non-Qwen-judge replication with a native-speaker query set, scoped in the Limitations section. 

---
# PoolBench: A Benchmark for Pooling Strategies in Concept Representation Evaluation for Decoder-Only LLMs 

**Authors**: Ayushi Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2608.05162)  

**Abstract**: Pooling is a consequential but under-examined design choice in decoder-only concept representation work: practitioners must collapse token-level hidden states into a passage-level vector, yet no shared protocol exists for comparing this choice across concepts, models, and tasks. Reported gains are confounded by simultaneous changes in dataset, layer, construction method, and pooling rule, making principled decisions impossible. We introduce PoolBench, a benchmark that isolates pooling as the experimental variable under a fixed evaluation protocol. PoolBench covers 17 concepts, 19 pooling strategies, and 3 open-weight decoder-only models (Llama-3.1-8B, Gemma-2-9B, Mistral-7B), evaluated on a single audited corpus of 37,693 real-text passages. The primary axis is linear separability (D1/AUROC); steered concept prevalence (D2/SCP) and output-level disentanglement (D3) serve as diagnostic axes. The primary finding is decisive: W4_hierarchical reaches a cross-model mean AUROC of 0.7799, while the widely adopted P1_last_token baseline reaches only 0.7640 and is statistically significantly worse (Friedman+Nemenyi, p = 2.0e-36; 77 significant pairs among 18 effective strategies). Rankings are stable across layers (rho = 0.961--0.990). A key negative result: strong detection does not imply strong steering -- D2 and D3 are substantially weaker than D1 for most concepts, indicating a fundamental representational limit rather than a pooling failure. On mid-difficulty concepts, W4_hierarchical outperforms P1_last_token by 0.042--0.113 AUROC; construction method choice (DiffMean vs. REPE) has a larger effect (delta AUROC 0.15) than pooling (delta AUROC 0.016), establishing the correct practical hierarchy. We release the corpus, pre-extracted activations, scorer models, steering vectors, and evaluation code as a reusable protocol for pooling research. 

---
# SemiAdapt-Instruct: Extensible Instruction Tuning via Latent Domain-Specialised Adapters 

**Authors**: Josh McGiff, Salma Mekaoui, Robert Shanahan, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2608.05161)  

**Abstract**: Instruction-tuned LLMs are deployed into environments where domains evolve, yet extending a fine-tuned model's capabilities without full retraining remains an unsolved practical challenge. We present SemiAdapt-Instruct, a modular framework that discovers latent instruction domains, trains per-domain LoRA adapters in parallel, and performs parameter-free routing, incorporating new domains via single-adapter training without modifying existing components. SemiAdapt-Instruct outperforms full model fine-tuning across all configurations on both ROUGE-L and LLM-as-a-judge evaluation, while matching single LoRA fine-tuning and delivering extensibility that monolithic approaches cannot provide. We empirically demonstrate this extensibility by showing that updating a single adapter with new domain data outperforms all monolithic baselines. Our study also finds that independent discovery methods converge on the same specialisation-friendly domains. These findings demonstrate that decomposing heterogeneous instruction data into latent domains enables extensible NLP systems where evolving domains require only targeted single-adapter updates, eliminating the need for full model retraining. 

---
# Safe Evolution with Circuit Anchors 

**Authors**: Yan Liu, Jie Fu, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2608.05158)  

**Abstract**: In biological evolution, unconstrained mutation can lead to catastrophic outcomes: organisms may evolve enhanced capabilities while losing essential functions for survival. Nature's solution is \textit{developmental constraints}, where core regulatory genes remain anchored while peripheral genes adapt freely. We observe that current self-evolution algorithms for large language models lack analogous constraints. They optimize purely for capability, implicitly assuming safety will be preserved. Our experiments reveal this assumption to be dangerously wrong: models can \textit{misevolve} into powerful yet dangerous entities. Inspired by how Hox genes anchor body structure across $500$ million years of evolution, we propose \textbf{Circuit-Anchored Evolution (CAE)}. Using mechanistic interpretability, we identify a tiny \textit{safety circuit}, comprising less than $2$\% of model features, that causally mediates safety behaviors. We anchor this circuit during evolution, constraining it within a small displacement bound while allowing the remaining features to evolve freely. This mirrors the biological principle of \textit{evolvability with constraint}: preserving what is essential while adapting what is peripheral. Experiments across $3$ model families and two evolution algorithms demonstrate that CAE achieves superior safety preservation with minimal capability loss, substantially outperforming explicit reward-based constraints in both effectiveness and efficiency. Just as developmental constraints prevent biological evolution from producing nonviable organisms, circuit anchoring prevents model evolution from producing capable but dangerous systems. 

---
# Large Language Models Threaten Double-blind Review 

**Authors**: Bulambo Mwendelwa Gloire, Prasenjit Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2608.05157)  

**Abstract**: Double blind peer review serves as the scientific community primary defense against status and affiliation bias. Its effectiveness rests on the assumption that anonymized manuscripts convey scientific merit without revealing their authors. While authorship can often be recovered using citation networks or stylistic markers, we show that this assumption is increasingly fragile in the presence of large language models (LLMs). Using only titles and abstracts from papers published after model training, we find that LLMs collapse anonymity more efficiently than humans, with belief concentrating onto a small subset of plausible authors drawn from pools of five domain expert candidates. This vulnerability persists even when stylistic and bibliographic cues are excluded, indicating that stable patterns in problem framing and research focus function as latent conceptual signatures of authorship. Together, these findings indicate that double blind review is vulnerable to automated semantic inference, necessitating a revaluation of how anonymity and fairness are maintained in an AI augmented research ecosystem. 

---
# Scaffold-Mediated Post-Training: Co-Evolving Model Parameters and Procedural Scaffold Graphs 

**Authors**: Fei Ding, Yongkang Zhang, Runhao Liu, Yuhao Liao, Zijian Zeng, Huiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05156)  

**Abstract**: Post-training of large language models optimizes only parameters, while inference-time procedural scaffolds are typically designed independently of parameter training. This disconnect makes it difficult to automatically acquire and internalize complex strategies. We propose scaffold-mediated post-training: procedural scaffolds are organized into an evolvable graph structure that co-evolves with model parameters through discovery, distillation, and dynamic recompilation. We instantiate this paradigm as Skill Training. On FeatureBench, automatically discovered skills improve the passed rate by 8.1pp, and after progressive distillation the model still achieves a 27.7% passed rate without any external scaffold (distillation retention rate 85.2%, defined as post-distillation / with-skill passed rate), significantly outperforming standard SFT on the same data. 

---
# Beyond Sentiment: Comparing Traditional NLP and LLM-Based Multi-Dimensional Analysis for Political News Evaluation 

**Authors**: Maryam Fooladi, Federico Bottino  

**Link**: [PDF](https://arxiv.org/pdf/2608.05155)  

**Abstract**: Traditional sentiment analysis (SA) models, while effective for polarity classification, provide limited insight into the rhetorical, ideological, and framing dimensions of political discourse -- dimensions that are central to research in the social sciences and humanities (SSH). In this paper, we present a comparative study of RoBERTa-based sentiment analysis and an LLM-based multi-dimensional framing analysis platform applied to a corpus of 50 political news articles from 17 international media outlets. The results reveal a critical limitation we term neutral collapse: RoBERTa classifies 70% of articles as neutral, effectively flattening substantively rich political content into an analytically uninformative category. We find that 23% of neutral-classified articles exhibit negative probability scores above 0.30. By contrast, the LLM-based approach captures political bias direction and intensity, sensationalism, emotional appeal, and political framing -- yielding multi-dimensional analytical outputs aligned with SSH epistemologies. We argue that for political media analysis, traditional SA alone is insufficient, and that LLM-based multi-dimensional frameworks offer a more epistemologically adequate computational lens for SSH research needs. 

---
# RIG-RoPE: Relation- and Instance-Gated Rotary Positional Encoding with Duration-Aware Temporal Coordinates 

**Authors**: Donggen Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.05154)  

**Abstract**: Rotary positional encoding (RoPE) is a core component of modern language models and has been extended to multimodal LLMs through multidimensional variants such as multimodal RoPE (M-RoPE), which split positional channels into temporal, height, and width subspaces. This report identifies two limitations of static multidimensional position assignment in interleaved multimodal contexts. First, height/width rotations may be applied to token pairs whose spatial displacement is not a well-defined geometric object, producing cross-modal and inter-instance spatial interference. Second, temporal coordinates are often treated as equal-step counters, so a text token, an image block, and a video segment can advance the temporal phase by comparable amounts despite different information density.
We propose RIG-RoPE, a relation- and instance-gated RoPE mechanism with duration-aware temporal coordinates. RIG-RoPE augments each token with a modality indicator, a visual instance identifier, and a scalar information-duration coordinate. It enables H/W rotations only for query-key pairs from the same visual instance; otherwise the unknown spatial displacement is marginalized rather than set to zero. Temporal rotations use interpolated cumulative block durations: text tokens consume unit duration, images use a dimension-aware logarithmic spatial scale, and videos further apply a logarithmic temporal extension over effective frames. We provide a gauge-invariance argument for avoiding ordinary cross-instance spatial rotation, an impossibility result for static IDs under shared H/W subspaces, and a duration-consistency argument against equal-step multimodal time. RIG-RoPE adds no learned parameters and can be implemented inside tiled attention kernels with constant additional metadata per token. This preliminary report establishes the formulation and validation path without claiming empirical superiority. 

---
# Universal Pathologies, Conditional Consequences: A Triple-Robustness Analysis of RAG for Multi-Hop Traceability 

**Authors**: Meftun Akarsu, Burak Ozdemir  

**Link**: [PDF](https://arxiv.org/pdf/2608.05153)  

**Abstract**: GraphRAG underperforms vector RAG on citation precision in many reports, but where and why have remained corpus-bound. We present a triple-robustness analysis that holds the retrieval architecture fixed and varies three orthogonal axes embedder (local e5-small -> Azure text-embedding-3-small), corpus (DO-178C typed-edge requirements -> Wikipedia paragraph chains via MuSiQue), and judge (paired GPT-5.4 x GPT-4.1) across 4,440 main-matrix runs, 600 cross-corpus runs, and 1,200 paired faithfulness judgments. (C2a) Over-citation is architecturally universal: GraphRAG emits 11-15 IDs per answer at citation precision 0.12-0.23 and retrieval recall 0.68-0.87 across all three settings. (C2b) Its faithfulness consequence is corpus-conditional: in typed-edge DO-178C, GraphRAG faithfulness collapses 74%->40% across hops; on Wikipedia chains the same pipeline rises 42%->58% because over-cited paragraphs remain topically supporting. (C1) Stratum-conditional winners are corpus-conditional but embedder-robust: vanilla wins 2-hop on DO-178C, GraphRAG wins 2-hop on MuSiQue, identical under either embedder. (C3) Single-judge LLM faithfulness is fragile to retrieval state: same-judge self-kappa across embedders is 0.137 for GPT-5.4 (verdict change on 41% of items). A learned router on dense embeddings alone reaches macro-F1 0.86 on hop classification (C4). We argue triple-robustness is the minimum bar for trustworthy RAG architecture claims. 

---
# Mean-Field Dynamics of Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Hao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2608.05152)  

**Abstract**: Large language models (LLMs) with chain-of-thought reasoning have been widely applied in recent years, and theoretical explanations of their behavior may help deepen our understanding and guide model optimization. In this study, we introduce a framework that seeks statistical regularities and theoretical interpretations in LLM reasoning without simplifying the model architecture or making analogies to existing physical systems. We formulate LLM reasoning as a guided discovery process on a clue graph, and derive a one-dimensional ordinary differential equation for the fraction of discovered clues using the mean-field approximation. Experimentally, clue tokens are identified using the normalized surprisal of a student LLM on the outputs of a teacher LLM, and statistical regularities are obtained by averaging over many reasoning chains of thought. Our experiments show that the resulting statistical regularities are reproducible within the same dataset and can be fitted by the solving the proposed theoretical equation. 

---
# Simulator-Grounded Large Language Models for Industrial Causal Reasoning: Tool-Use, Structured Injection, and Plant-Portable Retrieval for Wastewater Treatment Decision Support 

**Authors**: Gary Simethy, Daniel Ortiz Arroyo, Petar Durdevic  

**Link**: [PDF](https://arxiv.org/pdf/2608.05151)  

**Abstract**: Wastewater operators need answers grounded in how their plant's variables interact and how fast effects propagate, not in generic pretraining text, when asking causal questions such as "why is N2O rising?" or "what happens if I cut aeration by 20%?". We compare three concrete ways to ground a frozen Qwen2.5-32B-Instruct model in an architecturally interpretable wastewater simulator (CCSS-IX): a live simulator oracle (Method 1), structured parameter injection (Method 2), and a Decoupled Recall-Reasoning (DRR) retriever (Method 3). On a 198-question causal benchmark the three reach 99.5%, 79%, and 75.8%, forming a deployment ladder above the strongest retrieval-augmented baseline at 48%. The DRR retriever has 110M parameters and trains per plant in ~17 seconds; after cross-plant transfer to a biologically distinct plant it still reaches 88%, while Method 2's static table cannot transfer. On a 60-question counterfactual benchmark only Method 3 handles queries about what happens after an intervention: +16.3 pp over Method 2, paired 95% CI [+7.1, +26.4] pp, with 100% on the timescale and operating-regime categories. On the AI2 Reasoning Challenge (ARC) with an OpenBookQA fact corpus, the same selective-retrieval mechanism reaches 79% versus unconstrained Llama-3.1-8B 76% and full-injection 74%, a +3 pp out-of-domain replication that argues against a result specific to wastewater treatment. We provide the first single-simulator comparison of live tool-use, static parameter injection, and learned numerical-parameter retrieval for industrial causal question answering. 

---
# AV-AIVAT: 74x Cheaper Agent Evaluation with Certified Anytime-Valid Stopping in Imperfect-Information Games 

**Authors**: Boning Li, Yu Chen, Longbo Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06362)  

**Abstract**: Deciding which of two agents is stronger means playing games until skill outweighs luck, and every game costs money, model inference, or expert time. Since the number of games needed is unknown, fixed-budget evaluations either keep paying after the result is settled or stop before the agents can be told apart, while naive optional stopping with an ordinary confidence interval invalidates the stated level. We make such an evaluation stop as soon as its evidence suffices, with the guarantee intact. The Action-Informed Value Assessment Tool (AIVAT) reduces variance in imperfect-information games through conditional mean-zero corrections, by a median $54\times$ across 15 LLM agent configurations spanning 71,439 paired Heads-Up No-Limit Hold'em (HUNL) hands, but does not say when to stop. We combine AIVAT with continuously monitored Confidence Sequences (CSs) into anytime-valid AIVAT (AV-AIVAT), whose online value model learns only from past games so that no game scores its own correction. At the nominal 95\% level and a target precision of $\pm1$ Big Blind, raw outcomes need a median $74\times$ as many hands as AIVAT-corrected outcomes to stop under the Asymptotic CS (AsympCS). Exact finite-sample certification uses the Empirical-Bernstein CS (EB-CS), which needs an independently justified bound on corrected payoffs. We establish such a bound structurally for Leduc hold'em and characterize a width floor set by the CS's bet cap and that bound, which governs how much of a variance gain becomes earlier stopping; the descriptive HUNL EB-CS runs show a median $1.37\times$ stopping-time ratio. AV-AIVAT turns variance reduction into efficient, auditable early stopping while separating asymptotic screening from exact certification, so an evaluation can stop the moment its evidence suffices and hand a third party everything needed to recheck the verdict at that very stopping time. 

---
# CalibForge: Adversarial Solver Calibration for Scaling Learnable Terminal Tasks 

**Authors**: Fanzhe Meng, Guoxin Chen, Jiale Zhao, Shuang Sun, Zhiyu Lin, Wayne Xin Zhao, Ruihua Song, Ji-Rong Wen, Kai Jia  

**Link**: [PDF](https://arxiv.org/pdf/2608.06352)  

**Abstract**: Training terminal agents requires executable and verifiable tasks that are not merely solvable, but appropriately challenging for learning. Executable validation establishes feasibility, yet does not reveal how a task behaves relative to a given solver setting. In this paper, we present CalibForge, an autonomous terminal-task synthesis system that uses verified solver behavior to revise candidate tasks through adversarial solver calibration. Multi-solver calibration targets disagreement within a heterogeneous solver pool, whereas contrastive solver calibration targets a designated strong-pass/weak-fail relation; both operationalize a solver-relative learnable zone anchored in demonstrated solvability. Using CalibForge, we construct 5,431 calibrated terminal tasks. Our ablations show that both strategies yield more effective supervision than authoring and validation alone or ordinary single-solver feedback. Models trained on the full collection achieve 32.58% and 47.57% on Terminal-Bench 2.0. The largest improvements over the corresponding base model reach 24.71 percentage points on Terminal-Bench 2.0, 27.68 points on SWE-bench Pro, and 30.04 points on Doc2Repo. Together, these results support solver-relative learnability as a practical target for constructing effective and transferable agent training data. 

---
# RRC: Unlocking Generative Reward Models in LLM Reinforcement Learning via Ranking-Based Reward Construction 

**Authors**: Chenglong Wang, Ziming Zhu, Yifu Huo, Bei Li, Qiaozhi He, Yan Ding, Xiaoyang Hao, Yuxin Gao, Tianhua Zhou, Xiaojia Chang, Tongran Liu, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.06310)  

**Abstract**: Recent advances in reward modeling show a paradigm shift from discriminative reward models to generative reward models. However, despite their strong capabilities in response ranking, generative reward models have not realized their potential in reinforcement learning (RL). Our analysis reveals that this limitation arises from a mismatch between the comparative nature of generative reward modeling and the scalar scoring paradigm adopted by existing RL algorithms. To bridge this gap, we propose a Ranking-based Reward Construction (RRC) approach, which enables generative reward models to provide more effective RL learning signals by deriving rewards from relative preference rankings. RRC introduces two complementary strategies: self-competitive ranking, which exploits comparisons among sampled responses, and anchor-guided ranking, which enables scalable ranking-based reward construction with a small set of reference responses. Experiments across open-ended chat and reasoning benchmarks demonstrate that RRC substantially improves RL training with generative reward models, achieving consistent gains over existing reward construction approaches. Our code can be found at this https URL. 

---
# Beyond Top-K: Replacing Black-Box Retrieval with Interpretable Agentic Operations 

**Authors**: Sagar Tamang, Ayush Vyas, Tabarakul Hazarika  

**Link**: [PDF](https://arxiv.org/pdf/2608.06305)  

**Abstract**: Retrieval-augmented generation over long documents is dominated by one design: chunk the text, embed the chunks, and surface the top-k nearest neighbours of the query. We argue that for an important class of documents -- financial statements, audit reports, regulatory returns -- this design is structurally unsound, and we make the argument measurable. On a 780-page government financial report, 86.8% of content lines are table rows, thousands of near-identical figures compete in one embedding space, and a figure inherits its unit from a header a median of 13 lines above it -- so a chunk boundary routinely separates a number from whether it is in lakh or crore, an error of two orders of magnitude. A table-aware chunker built as a steelman fixes the unit problem but leaves 27-30% of numeric chunks with no fiscal-year header at every chunk size we tried. We propose READ (Reliable Embedding-free Agentic Document-search), in which an agent reads the raw document through three deterministic operations -- normalized lexical search, structural navigation, and bounded span reads -- exposed over the Model Context Protocol, so a trajectory is a replayable audit trail, not an opaque similarity score. On 51 verified questions READ answers 58.8% against dense retrieval's 15.7% (p_Holm = 2 x 10^-5) -- or 35.3% tuned, which READ still leads by 23.5 points (p_Holm = 0.017). An agent given the same loop but a top-k tool reaches only 27.5%, locating the gain in the interface rather than in iteration. We also report what the evidence does not support: BM25 is statistically indistinguishable from READ, so our result separates embedding-based from embedding-free retrieval, not agentic from lexical search. 

---
# HarnessOpt-Bench: Evaluating LLMs at Harness Optimization 

**Authors**: Varun Ursekar, Apaar Shanker, Yash Maurya, Shehab Yasser, Vijay S. Kalmath, Veronica Chatrath, Yuan Xue  

**Link**: [PDF](https://arxiv.org/pdf/2608.06301)  

**Abstract**: As LLMs are increasingly deployed within agentic systems, their capabilities depend not only on the model weights but also on the harness: the prompts, tools, control flow, memory, and orchestration code surrounding them. This makes automated harness optimization -- the iterative and evaluation-guided improvement of a harness by an AI system -- both an important route to improving AI systems and a demanding capability for AI systems themselves. Yet the community lacks a common protocol for measuring how well frontier LLMs perform at this task. We introduce HarnessOpt-Bench, a benchmark for end-to-end harness optimization under expensive and stochastic evaluation. An optimizer, an LLM paired with a coding harness, receives a target agent's seed harness, graded evaluation feedback, and a fixed target-evaluation budget. It edits the harness and nominates a final candidate, which is scored by its normalized gain over the seed on a held-out test partition that remains inaccessible throughout search. A trusted execution environment enforces the evaluation boundary, meters target-agent resource use, and preserves candidate versions for audit. We evaluate 5 frontier LLMs as optimizers both under a shared coding harness and under their native harnesses across 4 downstream tasks, over 111 scored runs. Experiment results show that optimizer models separate more than the coding harnesses they act through, native harnesses are not consistently superior, and gains vary substantially across tasks and seed regimes. These results establish harness optimization as a measurable and discriminative capability with large space for improvement. 

---
# Schema-Guided Hierarchical Information Extraction and Semantic Evaluation Using Generative AI 

**Authors**: Modhurita Mitra, Jan-Willem Versteeg, Maarten D. Schermer, Shiva Nadi Najafabadi, Marie L. De Bruin, Lourens T. Bloem  

**Link**: [PDF](https://arxiv.org/pdf/2608.06167)  

**Abstract**: We present a schema-based framework for extracting complex, structured information from unstructured text documents using generative AI, followed by automated semantic evaluation of the extracted information against a gold standard. The schema, serving as an information model encoding domain knowledge, provides a unified, systematic, and consistent framework for extraction of hierarchical, nested information, with attributes of variable cardinality, and subsequent evaluation of the results. Information extraction from a document is performed in a single call to the model, in zero-shot mode.
In the evaluation step, we introduce a path-based semantic matching algorithm to align the nested, variable-cardinality attributes in the extracted results with those in the gold standard. We use generative AI for semantic comparison of the extracted and gold standard values of an attribute, and introduce a rubric to classify the result of the comparison, according to domain-specific considerations, as an exact, semantic, useful, or non-match.
We were able to extract 12 out of 14 attributes with an F1 score of $>$90\% from documents published by the health technology assessment organisation NICE, using the generative AI model Claude Opus 3. The time needed to extract the attributes from a document was $\sim$30 times lower than the time taken by a human domain expert. We further demonstrate generalisability of this framework across different generative AI models and transferability across different HTA organisations and languages. 

---
# Poli-Bias: Understanding and Measuring Large Language Model Biases in International Political Conflicts 

**Authors**: Massi-Nissa Abboud, Aladin Djuhera, Elena Cabrio, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2608.06123)  

**Abstract**: Measuring political bias in large language models (LLMs) remains challenging as it can manifest through subtle differences in framing, argumentation, and legal reasoning that are difficult to capture with a single metric. In this work, we introduce Poli-Bias, a counterfactual framework for measuring whether LLMs treat legally equivalent conflict scenarios differently depending on the countries involved. Poli-Bias compares responses to paired prompts in which country identities are systematically swapped across diverse geopolitical relationships, legal violations, and reasoning tasks. Rather than reducing bias to a single judgment, our framework decomposes response disparities into five interpretable dimensions, revealing how and where unequal treatment manifests. Across 13 contemporary LLMs spanning diverse model families and sizes, we find that country identities and user affiliations can systematically affect how equivalent actions are described, evaluated, and defended under international law. Our results thus establish Poli-Bias as a fine-grained framework for auditing political even-handedness and sycophancy in LLMs. 

---
# From Siloed Algorithms to Compliance-First Agentic Platforms: A Multi-Layered Architecture for Hospital AI Systems 

**Authors**: Manideep Dhar, Ritwik Singh, Sharat Chandra Kumar Manikonda  

**Link**: [PDF](https://arxiv.org/pdf/2608.06112)  

**Abstract**: Hospitals are rapidly adopting artificial intelligence for triage, imaging, scheduling etc., yet most deployments remain isolated point solutions locked inside departmental silos, resulting in duplicated effort, hidden risks, and unrealized enterprise value. Despite explosive growth of AI in healthcare market and accelerating investment, an estimated 70-80% of healthcare AI pilots fail to scale, largely due to governance gaps, fragmented data, and missing integration blueprints. This research proposes a hospital-specific, compliance-first, Agentic AI architecture with multiple interoperable layers, extending existing hospital AI platform models with: (i) an Agent Orchestration Layer for multi-agent workflows across clinical, operational, and financial domains, (ii) a Compliance and Policy Layer that centralizes policy-as-code for HIPAA, GDPR, the EU AI Act, DISHA Act, India's DPDP Act, and ISO/IEC security and safety standards, and (iii) a Privacy-Preserving Data Fabric that plugs federated learning, differential privacy, and secure enclaves into real-world Hospital Information Management System (HIMS) flows. Using a synthetic but structurally realistic hospital dataset and an open, ready-to-deploy prototype implementation, this study demonstrates the end-to-end orchestration of triage risk prediction, workflow optimization, and compliance logging, achieving substantial simulated reductions in task turnaround times and manual documentation effort while maintaining policy-guarded data access. The resulting architecture offers hospital leaders a pragmatic blueprint to move from ad hoc tools to a governed, globally compliant, ROI-focused AI platform that can be tailored to on-premise, hybrid and cloud-native deployments. 

---
# ECHO: A Locally-Deployable Agentic Health Assistant with Temporal Memory, Safety Guardrails, and Speech Assessment 

**Authors**: Abdulkadir Külçe, Alihan Esen, Cağla Fikir, Berke Kurt, Kuzey Arar, Gökhan Ercan, Faik Boray Tek  

**Link**: [PDF](https://arxiv.org/pdf/2608.06110)  

**Abstract**: This paper presents ECHO (Enhanced Care \& Health Observer), a locally-deployable conversational health assistant for long-term chronic care management. ECHO integrates three complementary software modules developed under shared supervision as a unified system. The core module is an agentic chatbot built on a ReAct loop orchestrated via LangGraph, equipped with 17 clinical tools and a temporal knowledge graph for persistent cross-session memory; it achieves a 94.9\% tool-execution pass rate across a 59-scenario benchmark with GPT-5 Mini. A two-stage hybrid safety layer intercepts all incoming queries: a rule-based layer handles explicit crisis signals and jailbreak attempts in under 1ms, while a signed graph neural network (GNN) with APPNP-style propagation classifies boundary cases by clinical intent, achieving 88.8\% accuracy and 90.6\% unsafe recall on a 2,537-query annotated Turkish health dataset while outperforming zero-shot LLM baselines including Llama 3.3 70B. A multimodal speech assessment module combining Whisper acoustic encoding and BERT text encoding with cross-attention fusion estimates emotion, depression, and pain, reaching a mean macro F1 of 0.652. The full system is implemented as a web application that can run entirely on consumer hardware, with no patient data transmitted to external services, supporting compliance with GDPR and KVKK. 

---
# LangChoiceBench: Measuring and Explaining Programming-Language Choice in LLMs 

**Authors**: Lukas Twist, Twm Stone, Helen Yannakoudakis, Jie M. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2608.06041)  

**Abstract**: Large language models (LLMs) have been shown to exhibit strong Python preferences when generating project-level code, but there is currently no systematic way to measure this behaviour across new models. To bridge this gap, we introduce LangChoiceBench, a project-level code-generation benchmark for measuring Python preference, recommendation-implementation consistency, and language diversity. LangChoiceBench covers 28 projects across seven software areas where Python is often a poor default. We evaluate 25 diverse LLMs and find that Python remains heavily over-selected, recommendation-implementation consistency is low, and smaller open-weight models generally show stronger Python preference and lower language diversity. We further analyse 9,826 reasoning traces and find that most Python choices are automatic or driven primarily by ease, rather than explicit consideration of project requirements. In a smaller but important set of cases, models fabricate contextual support for choosing Python - a failure mode we call phantom evidence - or produce code that contradicts the language selected in their own reasoning. 

---
# AppDeltaWorld: Transition-Grounded Delta Code World Model for Mobile GUI Agents 

**Authors**: Weikai Xu, Yunren Feng, Haoxiang Lei, Kun Huang, Yuxuan Liu, Kang Zhao, Xiaolin Hu, Shuo Shang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2608.05891)  

**Abstract**: Mobile GUI agents can operate apps through pixel perception and touch actions, making them a promising interface for collecting and improving long-horizon mobile interaction policies. However, real trajectories are difficult to obtain for sensitive apps and privacy-critical operations. At the same time, existing simulated environments are costly to scale up, and GUI world models still suffer from unstable generation, limited modality coverage, and inconsistent action-transition logic. To address these limitations, we propose AppDeltaWorld, a transition-grounded delta code world model that predicts the next GUI as a reachable code update rather than as an unconstrained image or text description. AppDeltaWorld retrieves app-specific Level-1 HTML references under an action-transition constraint, generates Level-2 executable HTML conditioned on the current screen, action, predicted next-screen text, and retrieved structure, and inserts generated visual assets into image slots before browser rendering. As a world model, AppDeltaWorld achieves the highest fidelity on CMGUIBench-500 under Code2World evaluation, with clear gains in structural layout and UI element reconstruction over image-only and code-only baselines. As a training environment, AppDeltaWorld supports filtered closed-loop SFT data construction that, when combined with public supervision, enables AppDeltaAgent to achieve state-of-the-art performance on AndroidLens and consistent gains on MobileGym and MobileWorld. Moreover, world-model-based test-time reinforcement learning enables policy adaptation and shows further improvements without additional interaction with real apps. 

---
# The em-dash em-beds in Congress: A population-level rise in em-dash frequency in U.S. congressional press releases at the dawn of the large-language-model era, 2021-2025 

**Authors**: Przemysław Czuma  

**Link**: [PDF](https://arxiv.org/pdf/2608.05889)  

**Abstract**: Large language models (LLMs) can leave small stylistic traces in text written with their help. The most discussed is the em-dash (U+2014), especially the unspaced form word---word, which is normal in typeset English prose but unusual in U.S. press writing, where AP style calls for spaced dashes. This study asks whether that trace is measurable in congressional press releases. In a preregistered design (OSF: https://doi.org/10.17605/OSF.IO/U5NEY), 146,239 scraper-sourced releases from 480 House and Senate offices (2021-2025, the open congress-press dataset) were analyzed: density of unspaced prose-form em-dashes per 1,000 characters of cleaned text, Poisson/negative-binomial models with a length offset, clustering by office. Density stayed within 0.10-0.12 per 1,000 characters through 2021-2024, then rose to 0.217 in 2025, more than twice the four-year baseline; the share of releases with such an em-dash rose from ~13% to 24.8%. The primary frequency ratio (2023-2025 vs 2021-2022) was 1.55 (95% CI 1.28-1.93; exact registered cut-off: 1.528), just above the prespecified 1.5x threshold. The rise was net-new (hyphen density stable), held within authors (75.6% of 262 continuous offices increased; p ~ 1e-16) and in a closed panel of 224 offices, and survived falsification tests: three placebo cut-offs were null, the pipeline showed no step at the 2024/2025 boundary, and continuing offices carried the rise. A segmented regression finds no step at the ChatGPT cut-off but a clear post-period acceleration; the 2025 rise is symmetric across parties and chambers. Because the registered validation gate was formally breached, the full preregistered decision rule was not met; the interpretation (broad diffusion of LLM-assisted writing as the models matured) is offered as exploratory. The em-dash remains a population-level marker, not a per-release authorship detector, and the design supports no causal claim. 

---
# The Vulnerability With No CVE: Managing Persistent Gaps Between Mandate and Authority in AI Coding Agents 

**Authors**: Shayell Aharon Salomon Amir Shaked Matan Noga  

**Link**: [PDF](https://arxiv.org/pdf/2608.05884)  

**Abstract**: Existing guidance identifies excessive agency, excessive permission, weak task-bound authorization, and inadequate agent controls as important risks. Control frameworks also describe capabilities for constraining, authorizing, observing, validating, and responding to agent activity. Yet security programs still need a way to manage persistent deployed instances that span components and outlive any one event.
We propose the agentic posture vulnerability (APV) as a task-conditioned vulnerability-management abstraction: a durable record for a composed agent-control exposure. One posture may produce different runtime manifestations across tasks; APV links those manifestations to the invariant posture and remains open until authority is narrowed, a missing control is added, risk is accepted, or closure is verified. APV is not proposed as a new root-cause class of risk; it operationalizes existing excessive-agency, authorization, and control-composition weaknesses.
We distinguish APVs from CVE-addressable product defects, OWASP Excessive Agency, Agent Baseline control outcomes, and the runtime authorization-execution gap. We then provide a field vignette, a thresholded definition, six recurring APV patterns, a vulnerability lifecycle, a minimum record, a control-and-closure matrix, tooling implications, and a testable research agenda. 

---
# Personalized Deep Research Query Refinement with Graph-Scaffolded Evidence Grounding 

**Authors**: Soojin Yoon, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2608.05876)  

**Abstract**: User requests serve as research specifications for deep research agents, shaping what evidence to seek and how to synthesize it. In personalized deep research, these specifications must additionally reflect user goals, constraints, preferences, and evaluation criteria. User context can be incorporated either within the deep research pipeline or into the research specification provided as its input. We focus on the latter, refining the user request into a personalized research specification before passing it to an unchanged deep research agent. This requires resolving three coupled decisions: which framing factors are relevant, whether the available user context sufficiently supports them, and whether to retrieve user memory, ask the user, or stop and refine the query. For training, G-STEER organizes framing factors as elicitation targets in an Intent Elicitation Graph that captures their dependencies. It learns a clarification policy from graph-scaffolded trajectories spanning diverse factor dependencies and evidence conditions. The policy produces a refined query while balancing target coverage against the costs of evidence acquisition. Experiments show that G-STEER achieves the strongest overall weighted target coverage and the highest downstream report personalization across both evaluated DRAs, while asking roughly one third as many user questions as a strong clarification baseline. 

---
# When Self-Evolution Backfires: Pre-Commit Gating against Skill Contamination in LLM Agents 

**Authors**: Linfang Shang, Ming Xu, Yiding Sun, Tianle Xia, Lingxiang Hu, Lan Xu, Ning Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2608.05810)  

**Abstract**: Self-evolving agents accumulate capability by distilling reusable skills from their execution trajectories, but we find this process is not monotonic: past a critical pool size, newly added skills degrade performance instead of improving it. We formalize this capability-contamination phase transition and trace it to a structural cause: once a defective skill enters the decision context, it becomes reference material for distilling later skills, forming cross-round contamination chains. We further show the contamination is structurally irreversible: removing a source skill after the fact cannot erase the flawed reasoning its descendants have already inherited, so post-hoc rollback recovers only a small fraction of the lost performance. This makes skill admission a pre-commit necessity rather than a post-hoc fix, and motivates Verifier-as-Gatekeeper (VaG): a progressive trust hierarchy whose three heterogeneous critics - structural validity, behavioral harmlessness, and semantic consistency - filter each skill individually, coupled with a marginal-gain subset selection that removes combinatorial contamination at the top tier before skills reach the runtime context. On Terminal-Bench 2, unconditional accumulation rises to a peak and then degrades, giving back most of its gains as the pool keeps growing, and post-hoc removal of the culprit skills recovers only a small part of the drop - the empirical signature of irreversibility. In contrast, VaG improves every round, reaching 72% pass@1 with a pool roughly 5x smaller, and its frozen skill pool transfers positively to four other backbones and a second benchmark without re-evolution. Ablations confirm the three critics are complementary and mutually non-substitutable, each intercepting a largely disjoint class of harmful skills. 

---
# Predicting Task Difficulty Without Rollouts 

**Authors**: Stefan Krsteski, Charlotte Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2608.05797)  

**Abstract**: Task difficulty dictates an agent's likelihood of success, and estimating it without rollouts means forecasting this directly from a task description before executing costly simulations in stateful environments. Reliable estimates would therefore allow environment designers to calibrate evaluation benchmarks and construct progressive training curricula. This becomes increasingly important as agents move into long-horizon domains, where empirical trial-and-error is a severe computational bottleneck. Prior work on early prediction is limited to static tasks or isolated coding environments, often relying on narrow features and inaccurate evaluation metrics. We study \textit{ex ante} difficulty prediction across 17 agentic benchmarks spanning coding, mathematics, machine learning, web navigation, function calling, and other domains. We show that AUC can mask poor difficulty estimates, identify token-level entropy as a useful predictive signal, and show how residuals between expected and observed difficulty can expose hidden environment flaws such as contamination and infeasibility. 

---
# GROM: Gradient-Free Rapid One-Shot Machine Unlearning 

**Authors**: Paweł Batorski, Przemysław Spurek, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2608.05783)  

**Abstract**: Machine unlearning has become a critical capability for safely removing specific, sensitive knowledge from large language models (LLMs). Current state-of-the-art approaches primarily rely on iterative, training-time unlearning via fine-tuning. However, even when utilizing parameter-efficient dimensionality reduction techniques like LoRA, gradient-based optimization remains computationally expensive and lacks explicit analytical formulations. It can also leave the targeted knowledge merely hidden rather than removed, to the point that simply quantizing the unlearned model restores much of what it was supposed to have erased. To resolve this, we propose a novel one-shot unlearning approach, abandoning iterative optimization in favor of a direct, exact analytical solution. We frame the unlearning process as a ridge-regularized least-squares optimization problem, deriving a closed-form additive update for targeted weight matrices. This update forces the selected layer to suppress unwanted content while strictly preserving its behavior on retained data. Computed from gradient-free forward passes alone, with no backpropagation and no iteration to convergence, GROM applies the weight edit in mere seconds, which makes it orders of magnitude faster than traditional fine-tuning. Extensive evaluations demonstrate that GROM achieves state-of-the-art forgetting-utility trade-offs on TOFU-5%, TOFU-10%, MUSE-Books, MUSE-News and WMDP, significantly reducing computational overhead without sacrificing overall model performance. Because the update removes the targeted content from the weights instead of masking it, GROM also withstands the low-bit quantization attack that recovers much of the content a gradient-based baseline had appeared to forget. Our code is publicly available at this https URL. 

---
# Unified Agent: Managing Interactions across Devices 

**Authors**: Xinshuang Liu, Runfa Blark Li, Shaoxiu Wei, Xin Lin, Truong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2608.05729)  

**Abstract**: As capabilities rapidly increase, AI agents can move from running inside one app to acting across a user's devices over time. Yet existing agent systems still fall short in this scenario. This is because observations are scattered across devices and moments, but mainstream systems are not designed around this fact: a single agent that treats devices as tools lacks effective state management for all devices across time, and multi-agent systems coordinate across agents but do not maintain the compact carried state a cross-device, cross-time request needs. We argue that the agent should maintain an effectively designed state that organizes engagement evidence, stated facts, and the standing request in a compact, action-ready form for deciding its action given the current observation. To compare state designs, we construct a benchmark of user-agent interaction across devices and time. We instantiate this principle in Unified Agent, a stateful agent that carries interaction evidence across devices and moments and uses it with the current observation to act. In the default setting, it significantly outperforms our adaptations of four published designs. Across changes in multimodal large language model (MLLM) family, capability, and reasoning effort, it remains ahead of all compared systems, demonstrating that the state-design advantage is robust across MLLM settings. Our code and data will be publicly available on GitHub. 

---
# DreamGuard: Efficient Runtime Guardrail for LLM Agents via Risk-Aware World Model 

**Authors**: Wenhao Lin, Chenyu Yu, Xingwei Lin, Sicong Cao, Xiang Chen, Lei Xue, Le Yu, Letian Sha, Chunming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2608.05695)  

**Abstract**: As large language model (LLM) agents increasingly invoke external tools and interact with real-world systems, unsafe actions may cause irreversible consequences on external states, user data, and downstream services. Recent runtime guardrails mitigate such risks by checking proposed actions before execution, but many remain reactive: they primarily assess the apparent safety of the current action, lacking an explicit model of how risk evolves across the trajectory. This limitation creates a critical blind spot for long-horizon risks, where individually benign-looking actions can gradually drift the agent toward hazardous states. In response, we propose DreamGuard, a proactive guardrail for LLM agents built around a risk-aware world model. The world model maintains a compact recurrent latent state over the trajectory and predicts future latent states from which DreamGuard derives immediate-hazard and prefix-risk evidence. It then fuses these multi-horizon signals into intervention decisions before execution. Experiments across four benchmarks and an online guardrail evaluation show that DreamGuard outperforms generic, reactive, and proactive guardrail baselines, achieves the best safety-utility trade-off among evaluated guardrails, and maintains an average end-to-end latency of 25 ms per call. 

---
# Reasoning Errors Have a Region and a Direction in the Residual-Stream Trajectory of LLMs 

**Authors**: Hamed Damirchi, Ignacio Meza De la Jara, Damith Ranasinghe, Yuhang Liu, Javen Shi  

**Link**: [PDF](https://arxiv.org/pdf/2608.05660)  

**Abstract**: As language models are increasingly used for tasks that require verifiable reasoning, reliably distinguishing sound reasoning from flawed reasoning has become an important practical problem. Recent trajectory-based methods seek this signal in layerwise residual-stream displacements, which capture how representations change while attenuating some stable, token-specific information. However, displacement omits the state from which an update originates, whereas restoring the full state risks reintroducing shortcut-prone information. We identify this trade-off and propose a three-stream detector that combines motion with two restricted views of location. A coarse region reader based on vector quantization and a fine direction reader over normalized multi-layer states. This design restores enough state context to interpret the motion without returning to full-state probing. On reasoning benchmarks unseen during training, our method improves selection accuracy by up to 12% over the displacement-only state of the art and 21% over single-layer probing baselines. Although trained only on reasoning benchmarks, it also reads factual completion and fact verification, ahead of every detector we compare against, which places the signal on correctness rather than on a kind of reasoning. Ablations further show that motion, region, and direction provide complementary signals. These results suggest that reasoning validity is better read from state-conditioned motion than from either static states or decontextualized trajectories alone. 

---
# Refining Over Resampling: Test-Time Self-Correction for LLM Reasoning 

**Authors**: Ahsan Bilal, Muhammad Ahmed Mohsin, Muhammad Umer, Lena Trigg, Ali Subhan, Muhammad Ali, Dean F. Hougen  

**Link**: [PDF](https://arxiv.org/pdf/2608.05643)  

**Abstract**: Test-time scaling improves LLM reasoning by using additional inference compute, but wider sampling alone can suffer from diminishing returns: new rollouts often repeat existing answer patterns instead of adding useful reasoning diversity. Verifier-based selection offers an alternative, but its performance depends on the calibration of an external reward model. We propose a verifier-free breadth--depth refinement framework that uses test-time compute to both explore and improve candidate solutions. The method samples multiple independent reasoning rollouts, refines each rollout through iterative self-critique and self-correction, and aggregates the refined answers by majority voting. Breadth preserves diverse initial attempts, while depth repairs local reasoning errors before aggregation. Across AIME24, AIME25, AMC, OlympiadBench, and MATH500, our method consistently improves over greedy decoding, majority voting, verifier-based best-of-$N$, beam search, and lookahead decoding across multiple open-weight models. For instance, with Qwen2.5-1.5B, accuracy increases from the strongest verifier-based baseline to $58.0\%$ on MATH500, and from $25.0\%$ to $32.5\%$ on AMC. These results show that test-time compute can be more effective when used to refine sampled trajectories rather than only to sample more candidates or rely on verifier-guided selection. 

---
# Measuring and Detecting Harmful AI Sycophancy 

**Authors**: Bohan Jiang, Dawei Li, Yasin Silva, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.05624)  

**Abstract**: Sycophantic responses are becoming pervasive in large language models (LLMs), and prior work has pointed out that some of them could be harmful. This paper focuses on one harmful sycophancy: preference-induced stance reversal sycophancy (PSRS), where a model reverses an initial stance merely to align with a user's stated preference. While existing research mainly measures how sycophantic a model is, we go further and ask whether PSRS can also be detected automatically from a single response. To investigate this at scale, we introduce CAP (Contrastive Anchor Probing), a framework for collecting labeled PSRS data. Applying CAP to 17 open- and closed-source LLMs, we collect 290,460 labeled responses across 12 everyday-advice domains. We organize our study around three research questions. (1) How often does PSRS occur? (2) How well can it be detected? (3) How does detection generalize to unseen models? We first reveal that PSRS rates range from 5% to 56% across LLMs, with more capable models being less sycophantic. Next, we show that detecting PSRS is feasible from the response text alone, and detectors need to learn subtle PSRS patterns from the training data. Because new LLMs appear rapidly, detectors inevitably encounter unseen models, making cross-model generalization an important framework goal. We demonstrate that detection performance drops on unseen models and propose an initial approach to address this challenge. We will release our dataset and code to support future research. 

---
# From Sports to Safety: Benchmarking Proactive Risk Inference in MLLMs 

**Authors**: Jiawei Qiu, Yichen Xu, Jianzhe Ma, Mingyang Yu, Wenbin Zhu, Yang Han, Pinzheng Lv, Wenxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05560)  

**Abstract**: Timely anticipation of physical hazards is essential for real-world safety, yet existing MLLM evaluations focus on harmful content or general risks, leaving proactive physical hazard prediction underexplored. Sports provide a well-suited testbed: accident causes span diverse injury dimensions and pre-accident spatiotemporal cues draw on reasoning capabilities shared with broader safety domains such as autonomous driving and fall detection. We introduce SPRINT (Sports Proactive Risk INference Testbed), a benchmark of 2,888 real-world sports videos (2,440 accident, 448 safe controls) spanning 14 sports and 3 environmental settings. Accident videos feature fine-grained annotations of early hazard cues, accident timing, and hierarchical causes; safe videos are manually verified as accident-free and serve to diagnose prompt-induced false alarms. Evaluating state-of-the-art MLLMs under diverse prompts and temporal windows reveals a sharp gap between hazard sensitivity and understanding: the best model exceeds 95% in signaling hazards yet falls below 50% in identifying their causes. Diagnostic experiments further show that explicit danger queries trigger severe false alarms even on hazard-free videos. These findings indicate that current MLLMs exhibit only superficial proactive safety, lacking stable, cause-grounded early warning, and underscore the need for reliable proactive safety in dynamic physical environments. Data and code will be open-sourced upon acceptance. 

---
# EcoAgent-Bench: Evaluating Economic Decision-Making in Budget-Constrained LLM Agents 

**Authors**: Jie Wu, Ming Gong, Feixiang Cheng, Qinqin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.05519)  

**Abstract**: Agent benchmarks usually measure task completion and treat resource use as an auxiliary statistic. In deployment, however, the choice among a local lookup, broad search, composite research tool, stronger model, or human escalation is part of the task itself. We introduce EcoAgent-Bench, in which every task specifies priced actions and an explicit budget. Its 304 real-derived tasks span five families adapted from GAIA, HotpotQA, and MuSiQue, and test four decisions: avoiding unnecessary escalation, escalating when local evidence is insufficient, selecting a model tier, and stopping on unsupported premises. We evaluate seven LLM agents in tool-API and workspace-CLI settings, together with four oracle scripted controls. Micro-averaged accuracy rewards one-sided policies: always-escalate controls achieve high micro success while failing save-oriented tasks. We therefore also report an economic-consistency score (the worse of accuracy on upgrade-oriented and save-oriented family groups) which exposes this failure. Tool-API agents attain only 3.9-24.0% micro strict success (at most 7.3% economic consistency), often either stopping before warranted escalation or overspending on cheap tasks. A threshold-crossing budget sweep changes GPT-5.4's escalation rate from 0% to only 3%. These results show that completion under a budget and economical action selection are distinct properties. We release the task bundle, transformation pipeline, frozen evaluation environments, and integrity-bound result artifacts needed to study both. 

---
# Learning Context-Free Grammars for Grammar-Constrained Decoding via Declarative Agentic Programming with Guarantees 

**Authors**: Kevin Cheang, Geoff Hulette, Rahul Kumar, Felipe R. Monteiro, Federico Mora, Robin Salkeld, Lin Tan, Serdar Tasiran  

**Link**: [PDF](https://arxiv.org/pdf/2608.05493)  

**Abstract**: Language models (LMs) are increasingly used to interact with external services via programs written in domain-specific languages (DSLs). Unfortunately, since DSLs are often low-resource and esoteric, LMs frequently produce syntactically invalid programs in these languages. Grammar-constrained decoding can eliminate such failures, but requires syntactic constraints. These are usually in the form of a context-free grammar for the target language, an artifact that is hard to come by for third-party DSLs. In this work, we define an agent, called Autogrammar, that automatically learns context-free grammars from documentation and execution data. Autogrammar is formalized as a Kripke structure whose nondeterministic choices are resolved by a language model, enabling declarative control of agent behavior via linear temporal logic constraints. We evaluate four versions of Autogrammar on three DSLs (i.e., Amazon CloudWatch Logs Insights, Dynatrace Query Language, and Datadog Search Syntax) and find that it generates grammars that achieve near perfect precision on unseen data; that temporal restrictions reduce execution time by 3.8x without incurring statistically-significant loss in precision; that execution data is crucial while documentation is dispensable; and that grammar-constrained decoding using Autogrammar-generated grammars significantly improves end-to-end LM performance on eight out of ten real tasks, matching or exceeding the performance of a professionally-maintained grammar. In comparison, the context-free grammars generated by existing LM baselines and a state-of-the-art formal technique perform significantly worse over the same evaluation. 

---
# GenGA: Editable and Data-Grounded Graphical Abstract Generation for Academic Papers 

**Authors**: Takuro Kawada, Shunsuke Kitada, Hitoshi Iyatomi  

**Link**: [PDF](https://arxiv.org/pdf/2608.05478)  

**Abstract**: Graphical Abstracts (GAs) visually summarize the key findings of academic papers, playing a crucial role in facilitating the understanding of research content. Recently, advancements in vision-language models and image generation models have enabled the automatic generation of scientific figures based on paper content. However, most conventional methods output the generated results as raster graphics, making post-editing (e.g., text modification and layout changes) highly difficult. This poses a significant challenge, as they are unsuitable for the iterative figure revision process inherent in paper writing and peer review. To tackle these challenges, we define the novel task of generating editable GAs from paper content and propose GenGA, a new GA generation framework that directly produces figures in vector format. By generating figures as a collection of vector elements with a hierarchical structure, GenGA produces outputs that can be seamlessly imported into existing drawing tools for intuitive, element-level editing. Furthermore, we introduce the Structural Independence Coefficient (SIC), a metric that quantifies the editing simplicity of a figure based on the degree to which local modifications propagate to other elements. Experimental results show that GenGA achieves superior editing simplicity compared to conventional methods, and even surpasses human-authored GAs in conciseness and semantic alignment. We also validate SIC as an effective metric correlated with manual editing costs. This study fundamentally redefines GA generation as an editable vector graphic generation problem grounded in the practical workflows of researchers, significantly promoting effective scientific communication. 

---
# EvoHarness-RL: Learning Self-Evolving Runtime Harness for Long-Horizon LLM Agents 

**Authors**: Xuying Ning, Dongqi Fu, Tianxin Wei, Hanqing Zeng, Yuanchen Bei, Bingxuan Li, Zihao Li, Qifan Wang, Xiang Shen, Yifan Wu, Jiayi Liu, Hong Li, Yinglong Xia, Xiangjun Fan, Hanghang Tong, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2608.05446)  

**Abstract**: Long-horizon LLM agents increasingly rely on external execution support to maintain state, track progress, invoke tools, verify outcomes, and reuse experience across interactions. However, effective harness use raises two coupled challenges: state formation from noisy interaction traces and runtime control over external-state access. Existing agents usually handle both through prompts, heuristics, or domain-specific conventions, leaving the external workspace and its usage policy manually engineered. To address this, we study the problem of harness policy learning, where agents learn harness policies offline and deploy them to construct and update external harness state online during runtime task execution. We introduce EvoHarness-RL, which exposes Belief, Progress, and Experience (BPE) as policy-facing harness state. Supervised harness fine-tuning teaches the base agent the harness action space and how to construct useful external state, while cost-aware GRPO explores coordination policies to selectively read, update, and consolidate that state during long-horizon interaction. Instantiated on ALFWorld with a Qwen3-8B LLM, EvoHarness-RL reaches 96.9% success and reveals two key dynamics: harness annealing, where training internalizes recurring harness-use patterns into the model policy and shifts the agent from frequent harness calls toward selective external-state access, and harness evolution, where progress updates and experience consolidation refine the harness into a compact, task-adaptive state substrate. These results suggest that long-horizon agents benefit from trainable policies for constructing and coordinating with external harness workspaces, beyond simply adding stronger tools or larger memories. 

---
# QEvict: Recoverable Quantized KV Eviction for Attention-Drift-Robust Long-Context Decoding 

**Authors**: Ayushman Garg, Akshita Gupta, Shaswata Bhattacharya, Abhishek Gupta, Sandeep Kumar, Manoj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2608.05326)  

**Abstract**: Autoregressive large language model inference is increasingly constrained by the memory footprint of the Key-Value (KV) cache. A dominant line of work reduces this footprint by evicting tokens that appear unimportant under attention-derived scores. However, such policies make an implicit irreversible decision: once a token is evicted, it cannot become useful again. We show that this assumption is brittle during decoding. Token and window importance drift as generated queries evolve, causing standard eviction policies to permanently discard states that later receive substantial attention under the full-cache model. To characterize this behaviour, we introduce Future Missed Mass and Global LIR, two diagnostics that measure future attention assigned to discarded states and the reactivation of historically inactive regions. We propose QEvict, a three-tier KV-cache management scheme that replaces binary retain-or-delete eviction with recoverable eviction. QEvict maintains high-confidence windows in full precision, stores intermediate windows in a quantized recoverable tier, and deletes only the lowest-confidence windows. During decoding, cumulative attention scores update window importance and when a quantized window becomes important again, it is dequantized and promoted to the full-precision. Under a fixed memory budget, this design preserves broader historical context while retaining exact full precision for the most important regions. Across long-context understanding, retrieval, and reasoning benchmarks, QEvict consistently improves over representative eviction and quantization baselines, reducing missed attention and improving information retention 

---
# EdgeXpert: An Edge Device for Memory-Efficient LLM Inference with Mixture-of-Experts and Speculative Decoding 

**Authors**: Sangwoo Ha, Hyunwoo Seo, Yurim Jo, Youngjin Moon, Hoi-Jun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2608.05303)  

**Abstract**: On-device deployment of Large Language Models (LLMs) has become essential for personalized edge applications. A primary bottleneck is external memory access (EMA) in feed-forward network (FFN) layers. Speculative decoding and mixture-of-experts (MoE) are promising solutions. Speculative decoding reduces the number of decoding stages by generating multiple tokens per stage, and MoE minimizes per-stage cost through sparse expert activation. However, there is an incompatibility when combining these two techniques. We propose EdgeXpert, a software-hardware co-designed LLM accelerator that resolves this incompatibility. In the prefill stage, the prompt-wise expert reuse reformulates routing as prompt-level expert reuse rather than independent per-token expert selection. It identifies important tokens using a lightweight encoder, constructs a shared expert set from them, and routes less important tokens with a reduced expert budget to lower expert EMA. In the decode stage, depth-aware expert coalescing exploits the contextual similarity and mutual exclusivity of same-depth candidate tokens. Rather than loading the union of all required channels, EdgeXpert loads only salient channels and applies computational calibration to recover accuracy without additional memory access. Synthesized in Samsung 28nm technology at 800 MHz, EdgeXpert achieves up to 56.3% latency reduction and 44.1% energy reduction compared to prior works, while maintaining near-baseline accuracy. 

---
# TriQua: Reconciling Granularity and Context in Factuality Evaluation 

**Authors**: Jin Liu, Steffen Thoma, Achim Rettinger  

**Link**: [PDF](https://arxiv.org/pdf/2608.05228)  

**Abstract**: The "decompose-then-verify" paradigm for LLM factuality evaluation faces a fundamental trade-off: atomic facts, i.e., one sentence conveying one unit of information, often omit essential context, while broader statements lack the granularity needed for precise assessment. To address this, we introduce TriQua, a framework that flexibly models facts based on their complexity. Simple claims are extracted as standard triples, while complex claims are represented as hyperrelational facts by attaching auxiliary contextual qualifiers. This adaptive structure preserves the necessary context for accurate retrieval and verification without sacrificing atomicity. Furthermore, TriQua's verification process directly annotates concrete errors within specific triples and qualifiers, providing fine-grained explainability for error detection. Alongside the framework, we propose TriQuaScore to quantify the factuality of these structured fact units. Empirical evaluations show that TriQuaScore strongly aligns with human annotated factuality scores, TriQua achieves robust decomposition quality, and outperforms existing decomposition-based frameworks in evidence-based fact verification. 

---
# Woodpecker Distillation: Weak Models Diagnose Reasoning Bugs in Strong Models 

**Authors**: Dayu Wang, Jiaye Yang, Weikang Li, Jiahui Liang, Yang Li, Deguo Xia, Jizhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05168)  

**Abstract**: Large language models often fail on reasoning tasks despite possessing the capability to solve them. We argue that many such failures arise from localized reasoning bugs in intermediate steps rather than from global incompetence. We show that these bugs are frequently repairable: inserting a short patch generated by a weak probe model after the same strong-model reasoning prefix can redirect the trajectory toward a correct solution.
However, this corrective effect is not reliably internalized by directly fine-tuning on weak patches or repaired trajectories, suggesting that the useful signal lies not in the intervention text itself, but in how it reshapes the model's future reasoning distribution. We therefore propose Woodpecker Distillation, a weak-to-strong training framework that learns from contrastive local interventions. Our method contrasts successful and unsuccessful weak-model patches at the same prefix, constructs a corrective teacher distribution from their induced future token predictions, and distills this signal into the strong model.
Experiments on mathematical reasoning benchmarks show that Woodpecker Distillation consistently improves strong-model performance and outperforms direct imitation baselines. 

---
# The Ignition Index: Measuring Global Workspace Dynamics in Language Models 

**Authors**: Saman Rahbar  

**Link**: [PDF](https://arxiv.org/pdf/2608.05160)  

**Abstract**: We introduce the Ignition Index (I), a validated scalar metric that operationalizes Global Workspace Theory's (GWT) all-or-none ignition prediction in transformer language models. The metric fits a four-parameter sigmoid to per-layer linear probe accuracy as a function of input signal strength, extracting steepness parameter beta-hat: high values indicate abrupt, ignition-like transitions; low values indicate graded build-up. Across 11 models spanning five architecture families, shuffled-label controls demonstrate 9.6-fold selectivity for genuine linguistic structure over spurious probe capacity (p < 0.001, Mann-Whitney U-test). We find: (1) Feedforward transformers exceed SSMs by 89% in aggregate beta-hat (p < 1e-13, Cohen's d = 0.52), with Mamba exhibiting near-linear profiles consistent with absent global broadcast. (2) Huginn-3.5B exhibits 2.12-fold higher ignition along its iteration axis than its depth axis, demonstrating that recurrent architectures manifest workspace-like transitions along the recurrence dimension. (3) Pythia-410M shows a PELT-detected phase transition at training step 256 (+67%), preceding induction-head formation. (4) Hypotheses linking ignition to model scale and signal strength were not confirmed, suggesting transformer architectures may saturate available ignition mechanisms. The Ignition Index provides the first validated quantitative bridge between GWT's dynamical predictions and mechanistic interpretability, with 9.6-fold measurement selectivity and architecture-level discriminability not previously characterized in the scaling literature. Code: this https URL 

---
# Agentic Nesting: A New Methodology for Existing Enterprise Application Integration and Services 

**Authors**: Xi Wang, Kun Li, Xianyao Ling, Gang Yin, Liang Zhang, Jiang Wu, Wenbo Lei, Jun Xu, Annie Wang, Fu Zhang, Weizhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.05159)  

**Abstract**: Enterprise operations extensively rely on multiple heterogeneous business systems and information applications, which also result in severe data silos and process fragmentation. Enterprises have invested considerable financial and material resources in building these applications, however, effectively leveraging and orchestrating them remains a formidable challenge. Conventional approaches to enterprise application integration, encompassing middleware architectures such as Enterprise Service Bus (ESB), API gateway infrastructures, and Robotic Process Automation (RPA), suffer from inherent limitations like high architectural coupling, escalating operation and maintenance costs, and limited intelligence capabilities. This paper proposes Agentic Nesting, a multi-agent collaboration framework in which existing enterprise applications are encapsulated as autonomous AI agents within a hierarchically nested structure. Rather than flat interconnection, agents are organized into layered stewardship topologies that mirror the compositional complexity of enterprise ecosystems. The framework extracts a digital agent proxy from each legacy application to enable natural-language interaction and autonomous manipulation, coordinates multiple agents through a central orchestrator for task decomposition and dynamic dispatching, and exposes a unified conversational interface for cross-application querying and process orchestration. The main contributions of this paper are the proposition of the "Application-as-Agent" integration paradigm and the "Conversation-as-Integration" interaction philosophy, together with an exploration of the generalization potential of this methodology in scenarios encompassing heterogeneous system coordination, and large-scale data applications. 

---
# A Study of LLMs' Preferences for Libraries and Programming Languages 

**Authors**: Lukas Twist, Mark Harman, Don Syme, Joost Noppen, Helen Yannakoudakis, Detlef Nauck, Jie M. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17181)  

**Abstract**: Despite the rapid progress of large language models (LLMs) in code generation, existing evaluations focus on functional correctness or syntactic validity, overlooking how LLMs make critical design choices such as which library or programming language to use. To fill this gap, we perform the first empirical study of LLMs' preferences for libraries and programming languages when generating code, covering eight diverse LLMs. We observe a strong tendency to overuse widely adopted libraries such as NumPy; in up to 45% of cases, this usage is not required and deviates from the ground-truth solutions. The LLMs we study also show a significant preference toward Python as their default language. For high-performance project initialisation tasks where Python is not the optimal language, it remains the dominant choice in 58% of cases, and Rust is not used once. These results highlight how LLMs prioritise familiarity and popularity over suitability and task-specific optimality; underscoring the need for targeted fine-tuning, data diversification, and evaluation benchmarks that explicitly measure language and library selection fidelity. 

---
